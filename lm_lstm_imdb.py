'''
Build a simple neural language model using GRU units
'''

import numpy as np
import os
import theano
import theano.tensor as T
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lm_data import IMDB_JMARS

import cPickle as pkl
import numpy
import copy
from costs import iwae_multi_eval
from tqdm import tqdm
import warnings
import time
import cPickle
from collections import OrderedDict

profile = False
seed = 1234
num_iwae_samps = 25
num_iwae_iters = 1
num_iwae_samps_train = 5
numpy.random.seed(seed)
is_train = tensor.scalar('is_train')


def param_init_nflayer(options, params, prefix='nf', nz=None):
    params[_p(prefix, 'u')] = norm_weight(nz, 1, scale=0.01)
    params[_p(prefix, 'w')] = norm_weight(nz, 1, scale=0.01)
    params[_p(prefix, 'b')] = numpy.zeros((1,)).astype('float32')
    return params


def nflayer(tparams, state_below, options, prefix='nf', **kwargs):
    # 1) calculate u_hat to ensure invertibility (appendix A.1 to)
    # 2) calculate the forward transformation of the input f(z) (Eq. 8)
    # 3) calculate u_hat^T psi(z)
    # 4) calculate logdet-jacobian log|1 + u_hat^T psi(z)| to be used in the LL function
    z = state_below
    u = tparams[_p(prefix, 'u')].flatten()
    w = tparams[_p(prefix, 'w')].flatten()
    b = tparams[_p(prefix, 'b')]
    # z is (batch_size, num_latent_units)
    uw = tensor.dot(u, w)
    muw = -1 + tensor.nnet.softplus(uw) # = -1 + T.log(1 + T.exp(uw))
    u_hat = u + (muw - uw) * tensor.transpose(w) / tensor.sum(w ** 2)
    zwb = tensor.dot(z, w) + b[0]
    f_z = z + u_hat.dimshuffle('x', 0) * tensor.tanh(zwb).dimshuffle(0, 'x')
    # tanh(x)dx = 1 - tanh(x)**2
    psi = tensor.dot((1 - tensor.tanh(zwb) ** 2).dimshuffle(0, 'x'), w.dimshuffle('x', 0))
    psi_u = T.dot(psi, u_hat)
    logdet_jacobian = T.log(T.abs_(1 + psi_u))
    return [f_z, logdet_jacobian]


def masked_softmax(x, axis=-1, mask=None):
    if mask is not None:
        x = (mask * x) + (1 - mask) * (-10)
        x = tensor.clip(x, -10., 10.)
    e_x = tensor.exp(x - tensor.max(x, axis=axis, keepdims=True))
    if mask is not None:
        e_x = e_x * mask
    softmax = e_x / (tensor.sum(e_x, axis=axis, keepdims=True) + 1e-6)
    return softmax


def gradient_clipping(grads, tparams, clip_c=1.0):
    g2 = 0.
    for g in grads:
        g2 += (g**2).sum()
    g2 = tensor.sqrt(g2)
    not_finite = tensor.or_(tensor.isnan(g2), tensor.isinf(g2))
    new_grads = []
    for p, g in zip(tparams.values(), grads):
        new_grads.append(tensor.switch(g2 > clip_c, g * (clip_c / g2), g))
    return new_grads, not_finite, tensor.lt(clip_c, g2)


def categorical_crossentropy(t, o):
    '''
    Compute categorical cross-entropy between targets and model output.
    '''
    assert (t.ndim == 2)
    assert (o.ndim == 3)
    o = o.reshape((o.shape[0] * o.shape[1], o.shape[2]))
    t_flat = t.flatten()
    probs = tensor.diag(o.T[t_flat])
    probs = probs.reshape((t.shape[0], t.shape[1]))
    return -tensor.log(probs + 1e-6)


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def chunk(sequence, n):
    """ Yield successive n-sized chunks from sequence. """
    for i in range(0, len(sequence), n):
        yield sequence[i:i + n]


C = - 0.5 * np.log(2 * np.pi)


def log_prob_gaussian(x, mean, log_var):
    return C - log_var / 2 - (x - mean) ** 2 / (2 * T.exp(log_var))


def gaussian_kld(mu_left, logvar_left, mu_right, logvar_right):
    gauss_klds = 0.5 * (logvar_right - logvar_left +
                        (tensor.exp(logvar_left) / tensor.exp(logvar_right)) +
                        ((mu_left - mu_right) ** 2.0 /
                         tensor.exp(logvar_right)) - 1.0)
    return gauss_klds


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
def dropout_layer(state_before, use_noise, trng, p=0.2):
    proj = tensor.switch(
        use_noise >= 0.5,
        state_before * trng.binomial(state_before.shape, p=(1. - p), n=1,
                                     dtype=state_before.dtype),
        state_before * (1. - p))
    return proj


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]
    return params


def save_params(path, tparams):
    params = {}
    for kk, vv in tparams.iteritems():
        params[kk] = vv.get_value()
    cPickle.dump(params, open(path, 'wb'))


# layers: 'name': ('parameter initializer', 'feedforward')
layers = {
    'ff': ('param_init_fflayer', 'fflayer'),
    'nf': ('param_init_nflayer', 'nflayer'),
    'gru': ('param_init_gru', 'gru_layer'),
    'lstm': ('param_init_lstm', 'lstm_layer'),
    'hsoftmax': ('param_init_hsoftmax', 'hsoftmax_layer'),
    'latent_lstm': ('param_init_lstm', 'latent_lstm_layer'),
}


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# orthogonal initialization for weights
# see Saxe et al. ICLR'14
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


# weight initializer, normal by default
def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def lrelu(x):
    return tensor.clip(tensor.nnet.relu(x, 1. / 3), -3.0, 3.0)


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
       >>> x, y = theano.tensor.matrices('x', 'y')
       >>> c = concatenate([x, y], axis=1)
    :parameters:
       - tensor_list : list
          list of Theano tensor expressions that should be concatenated.
       - axis : int
          the tensors will be joined along this axis.
    :returns:
       - out : tensor
          the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]
    return out


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff',
                       nin=None, nout=None, ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')
    return params


def fflayer(tparams, state_below, options, prefix='ff', **kwargs):
    W = tparams[_p(prefix, 'W')]
    b = tparams[_p(prefix, 'b')]
    if state_below.dtype == 'int32' or state_below.dtype == 'int64':
        out = W[state_below] + b
    else:
        out = tensor.dot(state_below, W) + b
    return out


def param_init_lstm(options, params, prefix='lstm', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    W = numpy.concatenate([
        norm_weight(nin, dim),
        norm_weight(nin, dim),
        norm_weight(nin, dim),
        norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([
        ortho_weight(dim),
        ortho_weight(dim),
        ortho_weight(dim),
        ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U
    params[_p(prefix, 'b')] = numpy.zeros((4 * dim,)).astype('float32')
    return params


def lstm_layer(tparams, state_below, options,
               prefix='lstm', mask=None, one_step=False,
               init_state=None, init_memory=None, nsteps=None,
               **kwargs):

    if nsteps is None:
        nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # helper for getting params
    def param(name):
        return tparams[_p(prefix, name)]

    dim = param('U').shape[0]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # initial/previous state
    if init_state is None:
        if not options['learn_h0']:
            init_state = tensor.alloc(0., n_samples, dim)
        else:
            init_state0 = theano.shared(numpy.zeros((options['dim'])),
                                        name=_p(prefix, "h0"))
            init_state = tensor.alloc(init_state0, n_samples, dim)
            tparams[_p(prefix, 'h0')] = init_state0

    U = param('U')
    b = param('b')
    W = param('W')
    non_seqs = [U, b, W]

    # initial/previous memory
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]


    def _step(mask, sbelow, h_tm1, c_tm1, *args):
        preact = tensor.dot(h_tm1, param('U'))
        preact += sbelow
        preact += param('b')

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_tm1 + i * c
        c = mask * c + (1. - mask) * c_tm1
        h = o * tensor.tanh(c)
        h = mask * h + (1. - mask) * h_tm1

        return h, c

    lstm_state_below = tensor.dot(state_below, param('W')) + param('b')
    if state_below.ndim == 3:
        lstm_state_below = lstm_state_below.reshape((
            state_below.shape[0], state_below.shape[1], -1))

    if one_step:
        mask = mask.dimshuffle(0, 'x')
        h, c = _step(mask, lstm_state_below, init_state, init_memory)
        rval = [h, c]
    else:
        if mask.ndim == 3 and mask.ndim == state_below.ndim:
            mask = mask.reshape((mask.shape[0], mask.shape[1] * mask.shape[2]))
            mask = mask.dimshuffle(0, 1, 'x')
        elif mask.ndim == 2:
            mask = mask.dimshuffle(0, 1, 'x')

        rval, updates = theano.scan(
            _step, sequences=[mask, lstm_state_below],
            outputs_info=[init_state, init_memory],
            name=_p(prefix, '_layers'),
            non_sequences=non_seqs,
            strict=True, n_steps=nsteps)
    return [rval, updates]


def latent_lstm_layer(
       tparams, state_below,
       options, prefix='lstm', back_states=None,
       gaussian_s=None, mask=None, one_step=False,
       init_state=None, init_memory=None, nsteps=None,
       provide_z=False,
       **kwargs):

    dim = options['dim']
    if nsteps is None:
        nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # helper for getting params
    def param(name):
        return tparams[_p(prefix, name)]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # initial/previous state
    if init_state is None:
        if not options['learn_h0']:
            init_state = tensor.alloc(0., n_samples, dim)
        else:
            init_state0 = theano.shared(
                numpy.zeros((options['dim'])),
                name=_p(prefix, "h0"))
            init_state = tensor.alloc(init_state0, n_samples, dim)
            tparams[_p(prefix, 'h0')] = init_state0

    non_seqs = [
        param('U'),
        tparams[_p('pri_ff_1', 'W')],
        tparams[_p('pri_ff_1', 'b')],
        tparams[_p('pri_ff_2', 'W')],
        tparams[_p('pri_ff_2', 'b')],
        tparams[_p('inf_ff_1', 'W')],
        tparams[_p('inf_ff_1', 'b')],
        tparams[_p('inf_ff_2', 'W')],
        tparams[_p('inf_ff_2', 'b')],
        tparams[_p('aux_ff_1', 'W')],
        tparams[_p('aux_ff_1', 'b')],
        tparams[_p('aux_ff_2', 'W')],
        tparams[_p('aux_ff_2', 'b')],
        tparams[_p('gen_ff_1', 'W')],
        tparams[_p('gen_ff_1', 'b')],
        tparams[_p('gen_ff_2', 'W')]
    ]

    # initial/previous memory
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _apply_nf(z0):
        zi = z0
        log_det_sum = 0.
        for i in range(options['num_nf_layers']):
            zi, log_det = get_layer('nf')[1](
                tparams, zi, options,
                prefix='inf_nf_%d' % i)
            log_det_sum += log_det
        return zi, log_det_sum

    def _step(mask, sbelow, d_, zmuv, h_tm1, c_tm1, U,
              pri_ff_1_W, pri_ff_1_b,
              pri_ff_2_W, pri_ff_2_b,
              inf_ff_1_W, inf_ff_1_b,
              inf_ff_2_W, inf_ff_2_b,
              aux_ff_1_W, aux_ff_1_b,
              aux_ff_2_W, aux_ff_2_b,
              gen_ff_1_W, gen_ff_1_b,
              gen_ff_2_W):
        pri_hid = tensor.dot(h_tm1, pri_ff_1_W) + pri_ff_1_b
        pri_hid = lrelu(pri_hid)
        pri_out = tensor.dot(pri_hid, pri_ff_2_W) + pri_ff_2_b
        z_dim = pri_out.shape[-1] / 2
        pri_mu, pri_sigma = pri_out[:, :z_dim], pri_out[:, z_dim:]

        if d_ is not None:
            inf_inp = concatenate([h_tm1, d_], axis=1)
            inf_hid = tensor.dot(inf_inp, inf_ff_1_W) + inf_ff_1_b
            inf_hid = lrelu(inf_hid)
            inf_out = tensor.dot(inf_hid, inf_ff_2_W) + inf_ff_2_b
            inf_mu, inf_sigma = inf_out[:, :z_dim], inf_out[:, z_dim:]
            # first sample z ~ q(z|x)
            z_smp = inf_mu + zmuv * tensor.exp(0.5 * inf_sigma)
            log_qz = tensor.sum(log_prob_gaussian(z_smp, inf_mu, inf_sigma), axis=-1)

            # pass through normalizing flows
            if options['num_nf_layers'] > 0:
                z_smp, log_det_sum = _apply_nf(z_smp)
                log_qz = log_qz - log_det_sum

            log_pz = tensor.sum(log_prob_gaussian(z_smp, pri_mu, pri_sigma), axis=-1)
            kld_qp = log_qz - log_pz

            aux_hid = tensor.dot(z_smp, aux_ff_1_W) + aux_ff_1_b
            aux_hid = lrelu(aux_hid)

            # concatenate with forward state
            if options['use_h_in_aux']:
                aux_hid = tensor.concatenate([aux_hid, h_tm1], axis=1)

            aux_out = tensor.dot(aux_hid, aux_ff_2_W) + aux_ff_2_b
            aux_out = T.clip(aux_out, -15., 15.)

            aux_mu, aux_sigma = aux_out[:, :d_.shape[1]], aux_out[:, d_.shape[1]:]
            aux_mu = tensor.tanh(aux_mu)
            disc_d_ = theano.gradient.disconnected_grad(d_)
            aux_cost = -log_prob_gaussian(disc_d_, aux_mu, aux_sigma)
            aux_cost = tensor.sum(aux_cost, axis=-1)
        else:
            if provide_z:
                print("Zs were provided!")
                z_smp = zmuv
            else:
                z_smp = z_mu + zmuv * tensor.exp(0.5 * z_sigma)

                if options['use_nf']:
                    z_smp, _ = _apply_nf(z_smp)

            kld_qp = tensor.sum(z_smp, axis=-1) * 0.
            aux_cost = tensor.sum(z_smp, axis=-1) * 0.
            log_pz = kld_qp * 0.
            log_qz = kld_qp * 0.

        # transform z
        gen_hid = tensor.dot(z_smp, gen_ff_1_W) + gen_ff_1_b
        gen_hid = lrelu(gen_hid)
        gen_out = tensor.dot(gen_hid, gen_ff_2_W)
        preact = tensor.dot(h_tm1, U) + sbelow + gen_out

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim']))
        c = tensor.tanh(_slice(preact, 3, options['dim']))

        c = f * c_tm1 + i * c
        c = mask * c + (1. - mask) * c_tm1
        h = o * tensor.tanh(c)
        h = mask * h + (1. - mask) * h_tm1
        return h, c, z_smp, log_pz, log_qz, kld_qp, aux_cost

    lstm_state_below = tensor.dot(state_below, param('W')) + param('b') + 0. * is_train
    if state_below.ndim == 3:
        lstm_state_below = lstm_state_below.reshape((
            state_below.shape[0], state_below.shape[1], -1))

    if one_step:
        mask = mask.dimshuffle(0, 'x')
        _step_inps = [mask, lstm_state_below, None, gaussian_s,
                      init_state, init_memory] + non_seqs
        h, c, z, _, _, _, _ = _step(*_step_inps)
        rval = [h, c, z]
        updates = {}
    else:
        if mask.ndim == 3 and mask.ndim == state_below.ndim:
            mask = mask.reshape((mask.shape[0], mask.shape[1] * mask.shape[2]))
        mask = mask.dimshuffle(0, 1, 'x')

        rval, updates = theano.scan(
            _step, sequences=[mask, lstm_state_below, back_states, gaussian_s],
            outputs_info=[init_state, init_memory, None, None, None, None, None],
            name=_p(prefix, '_layers'), non_sequences=non_seqs)
    return [rval, updates]


# initialize all parameters
def init_params(options):
    params = OrderedDict()
    # forward stochastic LSTM
    params = \
        get_layer('latent_lstm')[0](
            options, params, prefix='encoder',
            nin=options['dim_proj'], dim=options['dim'])
    # input layer
    params = \
        get_layer('ff')[0](
            options, params, prefix='ff_in_lstm',
            nin=options['dim_input'], nout=options['dim_proj'],
            ortho=True)
    # output layer
    params = \
        get_layer('ff')[0](
            options, params, prefix='ff_out_mus',
            nin=options['dim'], nout=options['dim_input'],
            ortho=True)

    # backward deterministic LSTM
    params = \
        get_layer(options['encoder'])[0](
            options, params, prefix='encoder_r',
            nin=options['dim_proj'], dim=options['dim'])
    # output layer
    params = \
        get_layer('ff')[0](
            options, params, prefix='ff_out_mus_r',
            nin=options['dim'], nout=options['dim_input'],
            ortho=True)

    # prior network
    params = \
        get_layer('ff')[0](
            options, params, prefix='pri_ff_1',
            nin=options['dim'], nout=options['dim_mlp'],
            ortho=True)
    params = \
        get_layer('ff')[0](
            options, params, prefix='pri_ff_2',
            nin=options['dim_mlp'], nout=2 * options['dim_z'],
            ortho=True)
    # posterior network
    params = \
        get_layer('ff')[0](
            options, params, prefix='inf_ff_1',
            nin=2 * options['dim'], nout=options['dim_mlp'],
            ortho=True)
    params = \
        get_layer('ff')[0](
            options, params, prefix='inf_ff_2',
            nin=options['dim_mlp'], nout=2 * options['dim_z'],
            ortho=True)
    # n-layer deep nf
    for i in range(options['num_nf_layers']):
        params = get_layer('nf')[0](
                options, params, prefix='inf_nf_%d' % i,
                nz=options['dim_z'])
    # Auxiliary network
    params = \
        get_layer('ff')[0](
            options, params, prefix='aux_ff_1',
            nin=options['dim_z'], nout=options['dim_mlp'],
            ortho=True)
    if options['use_h_in_aux']:
        dim_aux = options['dim_mlp'] + options['dim']
    else:
        dim_aux = options['dim']
    params = \
        get_layer('ff')[0](
            options, params, prefix='aux_ff_2',
            nin=dim_aux,
            nout=2 * options['dim'],
            ortho=True)

    # Decoder/Generative network
    params = \
        get_layer('ff')[0](
            options, params, prefix='gen_ff_1',
            nin=options['dim_z'], nout=options['dim_mlp'],
            ortho=True)
    U = numpy.concatenate([
        norm_weight(options['dim_mlp'], options['dim']),
        norm_weight(options['dim_mlp'], options['dim']),
        norm_weight(options['dim_mlp'], options['dim']),
        norm_weight(options['dim_mlp'], options['dim'])],
        axis=1)
    params[_p('gen_ff_2', 'W')] = U
    return params


def build_rev_model(tparams, options, x, y, x_mask):
    # for the backward rnn, we just need to invert x and x_mask
    # concatenate first x and all targets y
    # x = [x1, x2, x3]
    # y = [x2, x3, x4]
    xc = tensor.concatenate([x[:1, :], y], axis=0)
    # xc = [x1, x2, x3, x4]
    x0 = tensor.alloc(1, 1, x_mask.shape[1])
    xc_mask = tensor.concatenate([x0, x_mask], axis=0)
    # xc_mask = [1, 1, 1, 0]
    # xr = [x4, x3, x2, x1]
    xr = xc[::-1]
    # xr_mask = [0, 1, 1, 1]
    xr_mask = xc_mask[::-1]

    xr_emb = \
        get_layer('ff')[1](
            tparams, xr, options,
            prefix='ff_in_lstm', activ='lambda x: x')
    (states_rev, _), updates_rev = \
        get_layer(options['encoder'])[1](tparams, xr_emb, options,
                                         prefix='encoder_r', mask=xr_mask)
    # shift mus for prediction [o4, o3, o2]
    # targets are [x3, x2, x1]
    out = states_rev[:-1]
    targets = xr[1:]
    targets_mask = xr_mask[1:]
    out_logits = \
        get_layer('ff')[1](
            tparams, out, options,
            prefix='ff_out_mus_r', activ='linear')
    out_probs = masked_softmax(out_logits, axis=-1)
    nll_rev = categorical_crossentropy(targets, out_probs)
    # states_rev = [s4, s3, s2, s1]
    # posterior sees (s2, s3, s4) in order to predict x2, x3, x4
    states_rev = states_rev[:-1][::-1]
    # ...
    assert xr.ndim == 2
    assert xr_mask.ndim == 2
    nll_rev = (nll_rev * targets_mask).sum(0)
    return nll_rev, states_rev, updates_rev


def build_gen_model(tparams, options, x, y, x_mask, zmuv, states_rev):
    # disconnecting reconstruction gradient from going in the backward encoder
    x_emb = get_layer('ff')[1](tparams, x, options, prefix='ff_in_lstm', activ='lambda x: x')
    # small dropout
    trng = RandomStreams(seed)
    x_emb = dropout_layer(x_emb, is_train, trng, p=options['dropout'])
    rvals, updates_gen = get_layer('latent_lstm')[1](
       tparams, state_below=x_emb, options=options,
       prefix='encoder', mask=x_mask, gaussian_s=zmuv,
       back_states=states_rev)

    states_gen, memories_gen, z, log_pz, log_qzIx, kld, rec_cost_rev = (
            rvals[0], rvals[1], rvals[2], rvals[3], rvals[4], rvals[5], rvals[6])
    out_logits = get_layer('ff')[1](tparams, states_gen, options, prefix='ff_out_mus')
    out_probs = masked_softmax(out_logits, axis=-1)

    nll_gen = categorical_crossentropy(y, out_probs)
    nll_gen = (nll_gen * x_mask).sum(0)
    log_pxIz = -nll_gen
    log_pz = (log_pz * x_mask).sum(0)
    log_qzIx = (log_qzIx * x_mask).sum(0)
    kld = (kld * x_mask).sum(0)
    rec_cost_rev = (rec_cost_rev * x_mask).sum(0)
    return nll_gen, states_gen, kld, rec_cost_rev, updates_gen, log_pxIz, log_pz, log_qzIx, z, memories_gen


def ELBOcost(rec_cost, kld, kld_weight=1.):
    assert kld.ndim == 1
    assert rec_cost.ndim == 1
    return rec_cost + kld_weight * kld


# build a sampler
def build_sampler(tparams, options, trng, provide_z=False):
    last_word = T.lvector('last_word')
    init_state = tensor.matrix('init_state', dtype='float32')
    init_memory = tensor.matrix('init_memory', dtype='float32')
    gaussian_sampled = tensor.matrix('gaussian', dtype='float32')
    last_word.tag.test_value = -1 * np.ones((2,), dtype="int64")
    init_state.tag.test_value = np.zeros((2, options['dim']), dtype="float32")
    init_memory.tag.test_value = np.zeros((2, options['dim']), dtype="float32")
    gaussian_sampled.tag.test_value = np.random.randn(2, 100).astype("float32")

    # if it's the first word, emb should be all zero
    x_emb = get_layer('ff')[1](
        tparams, last_word, options,
        prefix='ff_in_lstm', activ='lambda x: x')

    # apply one step of gru layer
    rvals, update_gen = get_layer('latent_lstm')[1](
        tparams, x_emb, options,
        prefix='encoder', mask=None,
        one_step=True, gaussian_s=gaussian_sampled,
        back_states=None, init_state=init_state,
        init_memory=init_memory, provide_z=provide_z)
    next_state, next_memory, z = rvals

    # Compute parameters of the output distribution
    logits = get_layer('ff')[1](
        tparams, next_state, options,
        prefix='ff_out_mus')
    next_probs = masked_softmax(logits, axis=-1)

    # next word probability
    print('Building f_next..')
    inps = [last_word, init_state, init_memory, gaussian_sampled]
    outs = [next_probs, next_state, next_memory]
    f_next = theano.function(inps, outs, name='f_next')
    print('Done')

    return f_next


def beam_sample(tparams, f_next, options, trng=None, maxlen=30,
                zmuv=None, unk_id=None, eos_id=None, bos_id=None,
                init_states=None, init_memories=None, beam_size=10):
    assert bos_id is not None
    samples = []
    samples_scores = 0
    nb_samples = 1

    if zmuv is not None:
        nb_samples = zmuv.shape[1]
        zmuv = np.repeat(zmuv, beam_size, axis=1)

    # initial token is indicated by a -1 and initial state is zero
    next_w = bos_id * numpy.ones((nb_samples,)).astype('int64')
    next_state = numpy.zeros((nb_samples, options['dim'])).astype('float32')
    next_memory = numpy.zeros((nb_samples, options['dim'])).astype('float32')
    next_w = np.repeat(next_w, beam_size, axis=0)
    next_state = np.repeat(next_state, beam_size, axis=0)
    next_memory = np.repeat(next_memory, beam_size, axis=0)

    assert init_states is None
    assert init_memories is None

    samples = [[[] for _ in range(beam_size)] for _ in range(nb_samples)]
    probs = [[0. for _ in range(beam_size)] for _ in range(nb_samples)]
    completed_beams = [[] for _ in range(nb_samples)]
    completed_probs = [[] for _ in range(nb_samples)]

    for ii in range(maxlen):
        if zmuv is None:
            zmuv_t = numpy.random.normal(
                loc=0.0, scale=1.0,
                size=(next_w.shape[0], options['dim_z'])).astype('float32')
        else:
            if ii >= zmuv.shape[0]:
                zmuv_t = numpy.random.normal(
                    loc=0.0, scale=1.0,
                    size=(next_w.shape[0], options['dim_z'])).astype('float32')
            else:
                zmuv_t = zmuv[ii, :, :]

        inps = [next_w, next_state, next_memory, zmuv_t]
        ret = f_next(*inps)
        next_p, next_state, next_memory = ret

        next_p[:, 0] = 0.
        if bos_id is not None:
            next_p[:, bos_id] = 0.
        if unk_id is not None:
            next_p[:, unk_id] = 0.
        if (eos_id is not None) and ii < 5:
            next_p[:, eos_id] = 0.
        next_p = next_p / numpy.sum(next_p, axis=1)[:, None]

        topk_words = numpy.argsort(next_p, axis=-1)[:, ::-1]
        topk_probs = numpy.log(numpy.sort(next_p, axis=-1)[:, ::-1])

        # build the new beams for each input example
        all_sources = []
        all_prev_word = []
        for i in range(nb_samples):
            old_beams = samples[i]
            old_probs = probs[i]
            new_beams = []
            new_probs = []
            new_sources = []
            # for each beam
            for j in range(beam_size):
                # remove duplicate words, these can happen due to the pointer softmax
                # e.g. if the word appears twice in the document.
                added_words = set()
                k = 0
                while len(new_beams) < beam_size:
                    candidate = topk_words[i * beam_size + j, k]
                    assert candidate != unk_id
                    if candidate not in added_words:
                        if candidate == eos_id:
                            completed_beams[i].append(old_beams[j] + [candidate])
                            completed_probs[i].append(old_probs[j] + topk_probs[i * beam_size + j, k])
                            completed_probs[i][-1] /= len(completed_beams[i][-1])
                        else:
                            new_beams.append(old_beams[j] + [candidate])
                            new_probs.append(old_probs[j] + topk_probs[i * beam_size + j, k])
                            new_sources.append(i * beam_size + j)
                        added_words.add(candidate)
                    k += 1

            # compare all beams for this particular example between them
            best_beams = sorted(zip(new_beams, new_probs, new_sources), key=lambda x: x[1], reverse=True)
            bb, bp, bs = zip(*best_beams)
            samples[i] = bb[:beam_size]
            probs[i] = bp[:beam_size]

            # keep track of which line the selected beam originates from
            all_sources.extend(bs[:beam_size])
            all_prev_word.extend([s[-1] for s in samples[i]])

        next_state = next_state[all_sources]
        next_memory = next_memory[all_sources]
        next_w = numpy.array(all_prev_word).astype('int64')

    # at the end of sampling steps
    for i in range(nb_samples):
        for j in range(beam_size):
            completed_beams[i].append(samples[i][j])
            completed_probs[i].append(probs[i][j] / len(samples[i][j]))
    # order generated candidates by their log-probability
    # and return the top-scoring candidate for each input example
    best_samples = []
    best_scores = []
    for i in range(nb_samples):
        bsp = sorted(zip(completed_beams[i], completed_probs[i]), key=lambda x: x[1], reverse=True)
        best_samples.append(bsp[0][0])
        best_scores.append(bsp[0][1])
    return best_samples, best_scores


# generate sample
def gen_sample(tparams, f_next, options, trng=None, maxlen=30,
               argmax=False, kickstart=None, zmuv=None,
               unk_id=None, eos_id=None, bos_id=None,
               init_states=None, init_memories=None):
    assert bos_id is not None
    samples = []
    samples_scores = 0
    nb_samples = 1

    if zmuv is not None:
        nb_samples = zmuv.shape[1]

    # initial token is indicated by a -1 and initial state is zero
    next_w = bos_id * numpy.ones((nb_samples,)).astype('int64')
    next_state = numpy.zeros((nb_samples, options['dim'])).astype('float32')
    next_memory = numpy.zeros((nb_samples, options['dim'])).astype('float32')

    if init_states is not None:
        next_state = init_states
    if init_memories is not None:
        next_memory = init_memories
    if next_state.shape[0] != nb_samples:
        next_state = np.tile(next_state, reps=(nb_samples, 1))
    if next_memory.shape[0] != nb_samples:
        next_memory = np.tile(next_memory, reps=(nb_samples, 1))

    for ii in range(maxlen):
        if zmuv is None:
            zmuv_t = numpy.random.normal(
                loc=0.0, scale=1.0,
                size=(next_w.shape[0], options['dim_z'])).astype('float32')
        else:
            if ii >= zmuv.shape[0]:
                zmuv_t = numpy.random.normal(
                    loc=0.0, scale=1.0,
                    size=(next_w.shape[0], options['dim_z'])).astype('float32')
            else:
                zmuv_t = zmuv[ii, :, :]

        inps = [next_w, next_state, next_memory, zmuv_t]
        ret = f_next(*inps)
        next_p, next_state, next_memory = ret

        next_p[:, 0] = 0.
        if bos_id is not None:
            next_p[:, bos_id] = 0.
        if unk_id is not None:
            next_p[:, unk_id] = 0.
        if (eos_id is not None) and ii < 5:
            next_p[:, eos_id] = 0.
        next_p = next_p / numpy.sum(next_p, axis=1)[:, None]

        if argmax:
            nw = next_p.argmax(axis=1)
            samples_scores += np.log(next_p[np.arange(next_w.shape[0]), nw])
        else:
            nw = []
            next_pi = numpy.argsort(next_p, axis=1)[:, ::-1][:, :10]
            next_pp = numpy.sort(next_p, axis=1)[:, ::-1][:, :10]
            next_pp = next_pp / numpy.sum(next_pp, axis=1)[:, None]
            for i in range(next_p.shape[0]):
                nw_i = numpy.random.choice(next_pi[i], 1, p=next_pp[i, :])
                nw.append(nw_i[0])
            nw = numpy.asarray(nw)

        next_w = nw
        samples.append(nw)
        samples_scores += np.log(next_p[np.arange(next_w.shape[0]), nw])

    samples = np.stack(samples)
    return samples, samples_scores


def pred_probs(f_log_probs, f_iwae_eval, options, data, source='valid'):
    rvals = []
    iwae_rvals = []
    n_done = 0

    def get_data(data, source):
        if source == 'valid':
            return data.get_valid_batch()
        elif source == 'test':
            return data.get_test_batch()
        else:
            train_batches = []
            iterator = data.get_train_batch()
            for i in range(100):
                train_batches.append(next(iterator))
            return train_batches

    data_iterator = get_data(data, source)
    for num, (x, y, x_mask) in enumerate(data_iterator):
        x = x.transpose(1, 0)
        y = y.transpose(1, 0)
        x_mask = x_mask.transpose(1, 0)
        n_done += numpy.sum(x_mask)
        n_steps = x.shape[0]
        n_samps = x.shape[1]
        zmuv = numpy.random.normal(
            loc=0.0, scale=1.0,
            size=(n_steps, n_samps, options['dim_z']))
        zmuv = zmuv.astype('float32')
        elbo = f_log_probs(x, y, x_mask, zmuv)
        for val in elbo:
            rvals.append(val)
        # IWAE numbers
        iwae = iwae_multi_eval(
            x, y, x_mask, num_iwae_iters, f_iwae_eval,
            num_iwae_samps, options['dim_z'])
        iwae = np.ravel(iwae)
        assert len(iwae) == x.shape[1]
        for val in iwae:
            iwae_rvals.append(val)
    return numpy.exp(numpy.array(rvals).sum() / n_done), \
        numpy.exp(numpy.array(iwae_rvals).sum() / n_done)


def log_mean_exp(x, axis):
    m = tensor.max(x, axis=axis, keepdims=True)
    return m + tensor.log(tensor.mean(tensor.exp(x - m), axis=axis, keepdims=True))


# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, gshared, beta1=0.9, beta2=0.99, e=1e-5):
    updates = []
    t_prev = theano.shared(numpy.float32(0.))
    t = t_prev + 1.
    lr_t = lr * tensor.sqrt(1. - beta2**t) / (1. - beta1**t)
    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0., p.name + '_mean')
        v = theano.shared(p.get_value() * 0., p.name + '_variance')
        m_t = beta1 * m + (1. - beta1) * g
        v_t = beta2 * v + (1. - beta2) * g**2
        step = lr_t * m_t / (tensor.sqrt(v_t) + e)
        p_t = p - step
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((t_prev, t))
    f_update = theano.function([lr], [], updates=updates, profile=profile)
    return f_update


def train(dim_input=200,  # input vector dimensionality
          dim=2000,  # the number of GRU units
          dim_proj=600,  # the number of GRU units
          encoder='lstm',
          patience=10,  # early stopping patience
          max_epochs=10,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          lrate=0.0002,
          maxlen=100,  # maximum length of the description
          optimizer='adam',
          batch_size=16,
          valid_batch_size=16,
          data_dir='experiments/data',
          model_dir='experiments/imdb',
          log_dir='experiments/imdb',
          validFreq=1000,
          saveFreq=1000,  # save the parameters after every saveFreq updates
          sampleFreq=100,  # generate some samples after every sampleFreq
          dataset=None,  # Not used
          valid_dataset=None,  # Not used
          dictionary=None,  # Not used
          dropout=0.,
          reload_=False,
          use_iwae=False,
          use_h_in_aux=False,
          num_nf_layers=0,
          kl_start=0.2,
          weight_aux=0.,
          kl_rate=0.0003):

    dim_z = 64
    dim_mlp = dim
    learn_h0 = False

    desc = 'seed{}_aux{}_aux_zh{}_iwae{}_nfl{}_dr{:.1f}'.format(
        seed, weight_aux, str(use_h_in_aux), str(use_iwae),
        str(num_nf_layers), dropout)
    logs = '{}/{}_log.txt'.format(log_dir, desc)
    opts = '{}/{}_opts.pkl'.format(model_dir, desc)
    pars = '{}/{}_pars.npz'.format(model_dir, desc)
    print(desc)

    data = IMDB_JMARS(data_dir, seq_len=16,
                      batch_size=batch_size, topk=16000)
    dim_input = data.voc_size

    # Model options
    model_options = locals().copy()
    pkl.dump(model_options, open(opts, 'wb'))

    print('Options:')
    print(model_options)
    print('Building model')
    params = init_params(model_options)
    # load model
    if os.path.exists(pars):
        print("Reloading model from {}".format(pars))
        params = load_params(pars, params)
    tparams = init_tparams(params)

    x = tensor.lmatrix('x')
    y = tensor.lmatrix('y')
    x_mask = tensor.matrix('x_mask')
    # Debug test_value
    x.tag.test_value = np.random.rand(11, 20).astype("int64")
    y.tag.test_value = np.random.rand(11, 20).astype("int64")
    x_mask.tag.test_value = np.ones((11, 20)).astype("float32")

    zmuv = tensor.tensor3('zmuv')
    weight_f = tensor.scalar('weight_f')
    lr = tensor.scalar('lr')

    # build the symbolic computational graph
    nll_rev, states_rev, updates_rev = \
        build_rev_model(tparams, model_options, x, y, x_mask)
    nll_gen, states_gen, kld, rec_cost_rev, updates_gen, \
        log_pxIz, log_pz, log_qzIx, z, _ = \
        build_gen_model(tparams, model_options, x, y, x_mask, zmuv, states_rev)

    if model_options['use_iwae']:
        log_ws = log_pxIz - log_qzIx + log_pz
        log_ws_matrix = log_ws.reshape((x.shape[1] / num_iwae_samps_train, num_iwae_samps_train))
        log_ws_minus_max = log_ws_matrix - tensor.max(log_ws_matrix, axis=1, keepdims=True)
        ws = tensor.exp(log_ws_minus_max)
        ws_norm = ws / T.sum(ws, axis=1, keepdims=True)
        ws_norm = theano.gradient.disconnected_grad(ws_norm)
        vae_cost = -tensor.sum(log_ws_matrix * ws_norm, axis=1).mean() + 0. * weight_f
        elbo_cost = -log_mean_exp(log_ws_matrix, axis=1).mean()
    else:
        vae_cost = ELBOcost(nll_gen, kld, kld_weight=weight_f).mean()
        elbo_cost = ELBOcost(nll_gen, kld, kld_weight=1.).mean()

    aux_cost = (numpy.float32(weight_aux) * (rec_cost_rev + nll_rev)).mean()
    reg_cost = 1e-6 * tensor.sum([tensor.sum(p ** 2) for p in tparams.values()])
    tot_cost = vae_cost + aux_cost + reg_cost
    nll_gen_cost = nll_gen.mean()
    nll_rev_cost = nll_rev.mean()
    kld_cost = kld.mean()

    nbatch = 0
    for batch in data.get_valid_batch():
        nbatch += 1
    print('Total valid batches: {}'.format(nbatch))

    print('Building f_log_probs...')
    inps = [x, y, x_mask, zmuv, weight_f]
    f_log_probs = theano.function(
       inps[:-1], ELBOcost(nll_gen, kld, kld_weight=1.),
       updates=(updates_gen + updates_rev), profile=profile,
       givens={is_train: numpy.float32(0.)})
    f_iwae_eval = theano.function(
       inps[:-1], [log_pxIz, log_pz, log_qzIx],
       updates=(updates_gen + updates_rev),
       givens={is_train: numpy.float32(0.)})
    print('Done')

    print('Computing gradient...')
    grads = tensor.grad(tot_cost, itemlist(tparams))
    print('Done')

    all_grads, non_finite, clipped = gradient_clipping(grads, tparams, 100.)
    # update function
    all_gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
                   for k, p in tparams.iteritems()]
    all_gsup = [(gs, g) for gs, g in zip(all_gshared, all_grads)]
    # forward pass + gradients
    outputs = [vae_cost, aux_cost, tot_cost, kld_cost,
               elbo_cost, nll_rev_cost, nll_gen_cost, non_finite]
    print('Fprop')
    f_prop = theano.function(
        inps, outputs, updates=all_gsup,
        givens={is_train: numpy.float32(1.)})
    print('Fupdate')
    f_update = eval(optimizer)(lr, tparams, all_gshared)

    print('Optimization')
    # Training loop
    uidx = 0
    estop = False
    bad_counter = 0
    kl_start = model_options['kl_start']
    kl_rate = model_options['kl_rate']
    best_valid_err = 99999

    # append to logs
    if os.path.exists(logs):
        print("Appending to {}".format(logs))
        log_file = open(logs, 'a')
    else:
        log_file = open(logs, 'w')

    # count minibatches in one epoch
    num_train_batches = 0.
    for x, y, x_mask in data.get_train_batch():
        num_train_batches += 1.
    num_total_batches = num_train_batches * max_epochs

    # epochs loop
    for eidx in range(max_epochs):
        print("Epoch: {}".format(eidx))
        n_samples = 0
        tr_costs = [[], [], [], [], [], [], []]

        for x, y, x_mask in data.get_train_batch():
            # Repeat if we're using IWAE
            if model_options['use_iwae']:
                x = numpy.repeat(x, num_iwae_samps_train, axis=0)
                y = numpy.repeat(y, num_iwae_samps_train, axis=0)
                x_mask = numpy.repeat(x_mask, num_iwae_samps_train, axis=0)

            # Transpose data to have the time steps on dimension 0.
            x = x.transpose(1, 0).astype('int32')
            y = y.transpose(1, 0).astype('int32')
            x_mask = x_mask.transpose(1, 0).astype('float32')
            n_steps = x.shape[0]
            n_samps = x.shape[1]

            uidx += 1
            if kl_start < 1.:
                kl_start += kl_rate

            ud_start = time.time()
            # compute cost, grads and copy grads to shared variables
            zmuv = numpy.random.normal(loc=0.0, scale=1.0, size=(
                n_steps, n_samps, model_options['dim_z'])).astype('float32')
            vae_cost_np, aux_cost_np, tot_cost_np, kld_cost_np, \
                elbo_cost_np, nll_rev_cost_np, nll_gen_cost_np, not_finite_np = \
                f_prop(x, y, x_mask, zmuv, np.float32(kl_start))
            if numpy.isnan(tot_cost_np) or numpy.isinf(tot_cost_np) or not_finite_np:
                print('Nan cost... skipping')
                continue
            else:
                f_update(numpy.float32(lrate))

            # update costs
            tr_costs[0].append(vae_cost_np)
            tr_costs[1].append(aux_cost_np)
            tr_costs[2].append(tot_cost_np)
            tr_costs[3].append(kld_cost_np)
            tr_costs[4].append(elbo_cost_np)
            tr_costs[5].append(nll_rev_cost_np)
            tr_costs[6].append(nll_gen_cost_np)
            tr_costs = [trc[-dispFreq:] for trc in tr_costs]
            ud = time.time() - ud_start

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print('PROGRESS: 00.00%')
                str1 = 'Epoch {:d}  Update {:d}  VaeCost {:.2f}  AuxCost {:.2f}  KldCost {:.2f}  TotCost {:.2f}  ElboCost {:.2f}  NllRev {:.2f}  NllGen {:.2f}  KL_start {:.2f}'.format(
                    eidx, uidx, np.mean(tr_costs[0]), np.mean(tr_costs[1]), np.mean(tr_costs[3]),
                    np.mean(tr_costs[2]), np.mean(tr_costs[4]), np.mean(tr_costs[5]), np.mean(tr_costs[6]),
                    kl_start)
                print(str1)
                log_file.write(str1 + '\n')
                log_file.flush()

        print('Starting validation...')
        train_err = pred_probs(f_log_probs, f_iwae_eval, model_options, data, source='train')
        str1 = 'Train ELBO: {:.2f}, IWAE: {:.2f}'.format(train_err[0], train_err[1])
        valid_err = pred_probs(f_log_probs, f_iwae_eval, model_options, data, source='valid')
        str2 = 'Valid ELBO: {:.2f}, IWAE: {:.2f}'.format(valid_err[0], valid_err[1])
        test_err = pred_probs(f_log_probs, f_iwae_eval, model_options, data, source='test')
        str3 = 'Test ELBO:  {:.2f}, IWAE: {:.2f}'.format(test_err[0], test_err[1])
        str4 = '\n'.join([str1, str2, str3])
        print(str4)
        log_file.write(str4 + '\n')

        if (best_valid_err < valid_err[1]):
            if lrate > 0.00001:
                print('Decaying learning rate to {}'.format(lrate))
                lrate = lrate / 2.0
        else:
            # Save best model and best error
            best_valid_err = valid_err[1]
            save_params(pars, tparams)

        # finish after this many updates
        if uidx >= finish_after:
            print('Finishing after %d iterations!' % uidx)
            break

    return best_valid_err


if __name__ == '__main__':
    pass
