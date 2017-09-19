'''
Build a simple neural language model using GRU units
'''

import numpy as np
import os
import theano
import theano.tensor as T
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
from philly_utils import print_philly_hb
import numpy
import copy

import warnings
import time

from collections import OrderedDict
from blizzard import Blizzard_tbptt
from util import Iterator
profile = False


def gradient_clipping(grads, tparams, clip_c=100):
    g2 = 0.
    for g in grads:
        g2 += (g**2).sum()
    g2 = tensor.sqrt(g2)
    not_finite = tensor.or_(tensor.isnan(g2), tensor.isinf(g2))
    new_grads = []
    lr = tensor.scalar(name='lr')
    for p, g in zip(tparams.values(), grads):
        new_grads.append(tensor.switch(
            g2 > clip_c, g * (clip_c / g2), g))
    return new_grads, not_finite, tensor.lt(clip_c, g2)


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
    gauss_klds = 0.5 * (logvar_right - logvar_left + (tensor.exp(logvar_left) / tensor.exp(logvar_right)) + ((mu_left - mu_right)**2.0 / tensor.exp(logvar_right)) - 1.0)
    return gauss_klds


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
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


# layers: 'name': ('parameter initializer', 'feedforward')
layers = {
    'ff': ('param_init_fflayer', 'fflayer'),
    'gru': ('param_init_gru', 'gru_layer'),
    'lstm': ('param_init_lstm', 'lstm_layer'),
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
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


def param_init_lstm(options,
                     params,
                     prefix='lstm',
                     nin=None,
                     dim=None):
     if nin is None:
         nin = options['dim_proj']

     if dim is None:
         dim = options['dim_proj']

     W = numpy.concatenate([norm_weight(nin,dim),
                            norm_weight(nin,dim),
                            norm_weight(nin,dim),
                            norm_weight(nin,dim)],
                            axis=1)

     params[_p(prefix,'W')] = W
     U = numpy.concatenate([ortho_weight(dim),
                            ortho_weight(dim),
                            ortho_weight(dim),
                            ortho_weight(dim)],
                            axis=1)

     params[_p(prefix,'U')] = U
     params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

     return params

def lstm_layer(tparams, state_below,
                options,
                prefix='lstm',
                mask=None, one_step=False,
                init_state=None,
                init_memory=None,
                nsteps=None,
                **kwargs):

     if nsteps is None:
         nsteps = state_below.shape[0]

     if state_below.ndim == 3:
         n_samples = state_below.shape[1]
     else:
         n_samples = 1

     param = lambda name: tparams[_p(prefix, name)]
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
             return _x[:, :, n*dim:(n+1)*dim]
         return _x[:, n*dim:(n+1)*dim]

     def _step(mask, sbelow, sbefore, cell_before, *args):
         preact = tensor.dot(sbefore, param('U'))
         preact += sbelow
         preact += param('b')

         i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
         f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
         o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
         c = tensor.tanh(_slice(preact, 3, dim))

         c = f * cell_before + i * c
         c = mask * c + (1. - mask) * cell_before
         h = o * tensor.tanh(c)
         h = mask * h + (1. - mask) * sbefore

         return h, c

     lstm_state_below = tensor.dot(state_below, param('W')) + param('b')
     if state_below.ndim == 3:
         lstm_state_below = lstm_state_below.reshape((state_below.shape[0],
                                                      state_below.shape[1],
                                                      -1))
     if one_step:
         mask = mask.dimshuffle(0, 'x')
         h, c = _step(mask, lstm_state_below, init_state, init_memory)
         rval = [h, c]
     else:
         if mask.ndim == 3 and mask.ndim == state_below.ndim:
             mask = mask.reshape((mask.shape[0], \
                                  mask.shape[1]*mask.shape[2])).dimshuffle(0, 1, 'x')
         elif mask.ndim == 2:
             mask = mask.dimshuffle(0, 1, 'x')

         rval, updates = theano.scan(_step,
                                     sequences=[mask, lstm_state_below],
                                     outputs_info=[init_state, init_memory],
                                     name=_p(prefix, '_layers'),
                                     non_sequences=non_seqs,
                                     strict=True,
                                     n_steps=nsteps)
     return [rval, updates]


def latent_lstm_layer(
        tparams, state_below,
        options, prefix='lstm', back_states = None,
        gaussian_s=None, mask=None, one_step=False,
        init_state=None, init_memory=None, nsteps=None,
        **kwargs):

    if nsteps is None:
        nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    param = lambda name: tparams[_p(prefix, name)]
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
    non_seqs = [U, b, W, tparams[_p('z_cond', 'W')],
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
                tparams[_p('aux_ff_2', 'b')]]

    # initial/previous memory
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(mask, sbelow, d_, g_s, sbefore, cell_before,
              U, b, W, W_cond,
              pri_ff_1_w, pri_ff_1_b,
              pri_ff_2_w, pri_ff_2_b,
              inf_ff_1_w, inf_ff_1_b,
              inf_ff_2_w, inf_ff_2_b,
              aux_ff_1_w, aux_ff_1_b,
              aux_ff_2_w, aux_ff_2_b):

        p_z = lrelu(tensor.dot(sbefore, pri_ff_1_w) + pri_ff_1_b)
        z_mus = tensor.dot(p_z, pri_ff_2_w) + pri_ff_2_b
        z_dim = z_mus.shape[-1] / 2
        z_mu, z_sigma = z_mus[:, :z_dim], z_mus[:, z_dim:]

        if d_ is not None:
            encoder_hidden = lrelu(tensor.dot(concatenate([sbefore, d_], axis=1), inf_ff_1_w) + inf_ff_1_b)
            encoder_mus = tensor.dot(encoder_hidden, inf_ff_2_w) + inf_ff_2_b
            encoder_mu, encoder_sigma = encoder_mus[:, :z_dim], encoder_mus[:, z_dim:]
            tild_z_t = encoder_mu + g_s * tensor.exp(0.5 * encoder_sigma)
            kld = gaussian_kld(encoder_mu, encoder_sigma, z_mu, z_sigma)
            kld = tensor.sum(kld, axis=-1)

            aux_hid = tensor.dot(tild_z_t, aux_ff_1_w) + aux_ff_1_b
            aux_hid = lrelu(aux_hid)

            # concatenate with forward state
            if options['use_h_in_aux']:
                disc_s_ = theano.gradient.disconnected_grad(sbefore)
                aux_hid = tensor.concatenate([aux_hid, disc_s_], axis=1)

            aux_out = tensor.dot(aux_hid, aux_ff_2_w) + aux_ff_2_b
            aux_out = T.clip(aux_out, -8., 8.)
            aux_mu, aux_sigma = aux_out[:, :d_.shape[1]], aux_out[:, d_.shape[1]:]
            aux_mu = tensor.tanh(aux_mu)
            disc_d_ = theano.gradient.disconnected_grad(d_)
            aux_cost = -log_prob_gaussian(disc_d_, aux_mu, aux_sigma)
            #aux_cost = (disc_d_ - aux_mu) ** 2.0
            aux_cost = tensor.sum(aux_cost, axis=-1)
        else:
            tild_z_t = z_mu + g_s * tensor.exp(0.5 * z_sigma)
            kld = tensor.sum(tild_z_t, axis=-1) * 0.
            aux_cost = tensor.sum(tild_z_t, axis=-1) * 0.

        z = tild_z_t
        preact = tensor.dot(sbefore, param('U')) + tensor.dot(z, W_cond)
        preact += sbelow
        preact += param('b')

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * cell_before + i * c
        c = mask * c + (1. - mask) * cell_before
        h = o * tensor.tanh(c)
        h = mask * h + (1. - mask) * sbefore
        return h, c, z, kld, aux_cost

    lstm_state_below = tensor.dot(state_below, param('W')) + param('b')
    if state_below.ndim == 3:
        lstm_state_below = lstm_state_below.reshape((state_below.shape[0],
                                                     state_below.shape[1],
                                                     -1))
    if one_step:
        mask = mask.dimshuffle(0, 'x')
        h, c = _step(mask, lstm_state_below, init_state, init_memory)
        rval = [h, c]
    else:
        if mask.ndim == 3 and mask.ndim == state_below.ndim:
            mask = mask.reshape((mask.shape[0], mask.shape[1] * mask.shape[2])).dimshuffle(0, 1, 'x')
        elif mask.ndim == 2:
            mask = mask.dimshuffle(0, 1, 'x')

        rval, updates = theano.scan(
            _step, sequences=[mask, lstm_state_below, back_states, gaussian_s],
            outputs_info = [init_state, init_memory, None, None, None],
            name=_p(prefix, '_layers'), non_sequences=non_seqs, strict=True, n_steps=nsteps)
    return [rval, updates]


# initialize all parameters
def init_params(options):
    params = OrderedDict()
    params = get_layer('latent_lstm')[0](options, params,
                                         prefix='encoder',
                                         nin=options['dim_proj'],
                                         dim=options['dim'])
    params = get_layer('ff')[0](options, params, prefix='ff_in_lstm',
                                nin=options['dim_input'], nout=options['dim_proj'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_out_lstm',
                                nin=options['dim'], nout=options['dim'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_out_prev',
                                nin=options['dim_proj'],
                                nout=options['dim'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_out_mus',
                                nin=options['dim'],
                                nout=2 * options['dim_input'],
                                ortho=False)
    U = numpy.concatenate([norm_weight(options['dim_z'], options['dim']),
                           norm_weight(options['dim_z'], options['dim']),
                           norm_weight(options['dim_z'], options['dim']),
                           norm_weight(options['dim_z'], options['dim'])], axis=1)
    params[_p('z_cond', 'W')] = U

    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder_r',
                                              nin=options['dim_proj'],
                                              dim=options['dim'])
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_in_lstm_r',
                                nin=options['dim_input'], nout=options['dim_proj'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_out_lstm_r',
                                nin=options['dim'], nout=options['dim'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_out_prev_r',
                                nin=options['dim_proj'],
                                nout=options['dim'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_out_mus_r',
                                nin=options['dim'],
                                nout=2 * options['dim_input'],
                                ortho=False)

    # Prior Network params
    params = get_layer('ff')[0](options, params, prefix='pri_ff_1', nin=options['dim'], nout=options['dim_proj'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='pri_ff_2', nin=options['dim_proj'], nout=2 * options['dim_z'], ortho=False)
    # Inference network params
    params = get_layer('ff')[0](options, params, prefix='inf_ff_1', nin=2 * options['dim'], nout=options['dim_proj'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='inf_ff_2', nin=options['dim_proj'], nout=2 * options['dim_z'], ortho=False)
    # Auxiliary network params
    params = \
        get_layer('ff')[0](options, params, prefix='aux_ff_1',
                           nin=options['dim_z'], nout=options['dim_proj'],
                           ortho=False)
    if options['use_h_in_aux']:
        dim_aux = options['dim_proj'] + options['dim']
    else:
        dim_aux = options['dim_proj']
    params = \
        get_layer('ff')[0](options, params, prefix='aux_ff_2',
                           nin=dim_aux, nout=2 * options['dim'],
                           ortho=False)

    return params


def build_rev_model(tparams, options, x, y, x_mask):
    # for the backward rnn, we just need to invert x and x_mask
    # concatenate first x and all targets y
    # x = [x1, x2, x3]
    # y = [x2, x3, x4]
    xc = tensor.concatenate([x[:1, :, :], y], axis=0)
    # xc = [x1, x2, x3, x4]
    xc_mask = tensor.concatenate([tensor.alloc(1, 1, x_mask.shape[1]), x_mask], axis=0)
    # xc_mask = [1, 1, 1, 0]
    # xr = [x4, x3, x2, x1]
    xr = xc[::-1]
    # xr_mask = [0, 1, 1, 1]
    xr_mask = xc_mask[::-1]

    xr_emb = get_layer('ff')[1](tparams, xr, options, prefix='ff_in_lstm_r', activ='lrelu')
    (states_rev, _), updates_rev = get_layer(options['encoder'])[1](tparams, xr_emb, options, prefix='encoder_r', mask=xr_mask)
    out_lstm = get_layer('ff')[1](tparams, states_rev, options, prefix='ff_out_lstm_r', activ='linear')
    out_prev = get_layer('ff')[1](tparams, xr_emb, options, prefix='ff_out_prev_r', activ='linear')
    out = lrelu(out_lstm + out_prev)
    out_mus = get_layer('ff')[1](tparams, out, options, prefix='ff_out_mus_r', activ='linear')
    out_mu, out_logvar = out_mus[:, :, :options['dim_input']], out_mus[:, :, options['dim_input']:]

    # shift mus for prediction [o4, o3, o2]
    # targets are [x3, x2, x1]
    out_mu = out_mu[:-1]
    out_logvar = out_logvar[:-1]
    targets = xr[1:]
    targets_mask = xr_mask[1:]
    # states_rev = [s4, s3, s2, s1]
    # cut first state out (info about x4 is in s3)
    # posterior sees (s2, s3, s4) in order to predict x2, x3, x4
    states_rev = states_rev[:-1][::-1]
    # ...
    assert xr_mask.ndim == 2
    assert xr.ndim == 3
    log_p_y = log_prob_gaussian(targets, mean=out_mu, log_var=out_logvar)
    log_p_y = T.sum(log_p_y, axis=-1)     # Sum over output dim.
    nll_rev = -log_p_y                    # NLL
    nll_rev = (nll_rev * targets_mask).sum(0)
    return nll_rev, states_rev, updates_rev


# build a training model
def build_gen_model(tparams, options, x, y, x_mask, zmuv, states_rev):
    # disconnecting reconstruction gradient from going in the backward encoder
    x_emb = get_layer('ff')[1](tparams, x, options, prefix='ff_in_lstm', activ='lrelu')
    rvals, updates_gen = get_layer('latent_lstm')[1](
        tparams, state_below=x_emb, options=options,
        prefix='encoder', mask=x_mask, gaussian_s=zmuv,
        back_states=states_rev)

    states_gen, z, kld, rec_cost_rev = rvals[0], rvals[2], rvals[3], rvals[4]
    # Compute parameters of the output distribution
    out_lstm = get_layer('ff')[1](tparams, states_gen, options, prefix='ff_out_lstm', activ='linear')
    out_prev = get_layer('ff')[1](tparams, x_emb, options, prefix='ff_out_prev', activ='linear')
    out = lrelu(out_lstm + out_prev)
    out_mus = get_layer('ff')[1](tparams, out, options, prefix='ff_out_mus', activ='linear')
    out_mu, out_logvar = out_mus[:, :, :options['dim_input']], out_mus[:, :, options['dim_input']:]

    # Compute gaussian log prob of target
    log_p_y = log_prob_gaussian(y, mean=out_mu, log_var=out_logvar)
    log_p_y = T.sum(log_p_y, axis=-1)  # Sum over output dim.
    nll_gen = -log_p_y  # NLL
    nll_gen = (nll_gen * x_mask).sum(0)
    kld = (kld * x_mask).sum(0)
    rec_cost_rev = (rec_cost_rev * x_mask).sum(0)
    return nll_gen, states_gen, kld, rec_cost_rev, updates_gen


def ELBOcost(rec_cost, kld, kld_weight=1.):
    assert kld.ndim == 1
    assert rec_cost.ndim == 1
    return rec_cost + kld_weight * kld


def pred_probs(f_log_probs, options, data, source='valid'):
    rvals = []
    n_done = 0

    for data_ in data:
        x = data_[0][0]
        y = data_[1][0]
        x_mask = np.ones((x.shape[0], x.shape[1]), dtype='float32')
        n_done += x.shape[1]

        zmuv = numpy.random.normal(loc=0.0, scale=1.0, size=(
            x.shape[0], x.shape[1], options['dim_z'])).astype('float32')
        elbo = f_log_probs(x, y, x_mask, zmuv)
        for val in elbo:
            rvals.append(val)
    return numpy.array(rvals).mean()


# optimizers
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


def train(dim_input=200,          # input vector dimensionality
          dim=2048,               # the number of recurrent units
          dim_proj=1024,          # the number of hidden units
          encoder='lstm',
          patience=10,            # early stopping patience
          max_epochs=100,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,             # L2 weight decay penalty
          lrate=0.001,
          optimizer='adam',
          batch_size=16,
          valid_batch_size=16,
          data_dir='experiments/data',
          model_dir='experiments/blizzard',
          log_dir='experiments/blizzard',
          saveto='model.npz',
          validFreq=1000,
          use_dropout=False,
          reload_=False,
          use_h_in_aux=False,
          weight_aux_gen=0.,
          weight_aux_nll=0.,
          dim_z=256,
          kl_start=0.2,
          kl_rate=0.0003):

    learn_h0 = False
    seed = 0.

    desc = 'seed{}_aux_gen{}_aux_nll{}_aux_zh{}_klrate{}'.format(
        seed, weight_aux_gen, weight_aux_nll, str(use_h_in_aux), kl_rate)
    logs = '{}/{}_log.txt'.format(log_dir, desc)
    opts = '{}/{}_opts.pkl'.format(model_dir, desc)

    print("- log file: {}".format(logs))
    print("- opts file: {}".format(opts))

    # Model options
    model_options = locals().copy()
    pkl.dump(model_options, open(opts, 'wb'))
    log_file = open(logs, 'w')


    x_dim = 200
    data_path = '/scratch/macote/blizzard_unseg/'
    file_name = 'blizzard_unseg_tbptt'

    normal_params = np.load(data_path + file_name + '_normal.npz')
    X_mean = normal_params['X_mean']
    X_std = normal_params['X_std']
    train_data = Blizzard_tbptt(name='train',
                                path=data_path,
                                frame_size=x_dim,
                                file_name=file_name,
                                X_mean=X_mean,
                                X_std=X_std)

    valid_data = Blizzard_tbptt(name='valid',
                                path=data_path,
                                frame_size=x_dim,
                                file_name=file_name,
                                X_mean=X_mean,
                                X_std=X_std)

    test_data = Blizzard_tbptt(name='test',
                               path=data_path,
                               frame_size=x_dim,
                               file_name=file_name,
                               X_mean=X_mean,
                               X_std=X_std)

    # The following numbers are for batch_size of 128.
    assert batch_size == 128
    train_d_ = Iterator(train_data, batch_size, start=0, end=2040064)
    valid_d_ = Iterator(valid_data, batch_size, start=2040064, end=2152704)
    test_d_ = Iterator(test_data, batch_size, start=2152704, end=2267008-128)  # Use complete batch only.

    print('Building model')
    params = init_params(model_options)
    tparams = init_tparams(params)

    x = tensor.tensor3('x')
    y = tensor.tensor3('y')
    x_mask = tensor.matrix('x_mask')
    zmuv = tensor.tensor3('zmuv')
    weight_f = tensor.scalar('weight_f')
    lr = tensor.scalar('lr')

    # build the symbolic computational graph
    nll_rev, states_rev, updates_rev = \
        build_rev_model(tparams, model_options, x, y, x_mask)
    nll_gen, states_gen, kld, rec_cost_rev, updates_gen = \
        build_gen_model(tparams, model_options, x, y, x_mask, zmuv, states_rev)

    vae_cost = ELBOcost(nll_gen, kld, kld_weight=weight_f).mean()
    elbo_cost = ELBOcost(nll_gen, kld, kld_weight=1.).mean()
    aux_cost = (numpy.float32(weight_aux_gen) * rec_cost_rev + weight_aux_nll * nll_rev).mean()
    tot_cost = (vae_cost + aux_cost)
    nll_gen_cost = nll_gen.mean()
    nll_rev_cost = nll_rev.mean()
    kld_cost = kld.mean()

    print('- Building f_log_probs...')
    inps = [x, y, x_mask, zmuv, weight_f]
    f_log_probs = theano.function(
        inps[:-1], ELBOcost(nll_gen, kld, kld_weight=1.),
        updates=(updates_gen + updates_rev), profile=profile)

    print('- Building gradient...')
    grads = tensor.grad(tot_cost, itemlist(tparams))
    all_grads, non_finite, clipped = gradient_clipping(grads, tparams, 100.)
    # update function
    all_gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
                   for k, p in tparams.iteritems()]
    all_gsup = [(gs, g) for gs, g in zip(all_gshared, all_grads)]
    # forward pass + gradients
    outputs = [vae_cost, aux_cost, tot_cost, kld_cost, elbo_cost, nll_rev_cost, nll_gen_cost, non_finite]
    print('- Building f_prop...')
    f_prop = theano.function(inps, outputs, updates=all_gsup)
    print('- Building f_update...')
    f_update = eval(optimizer)(lr, tparams, all_gshared)
    print('DONE.')

    print('- Starting optimization...')
    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = list(numpy.load(saveto)['history_errs'])
    best_p = None
    bad_count = 0

    # Training loop
    uidx = 0
    estop = False
    bad_counter = 0
    kl_start = model_options['kl_start']
    kl_rate = model_options['kl_rate']
    old_valid_err = numpy.inf
    start = time.time()

    for eidx in range(max_epochs):
        print("Epoch: {}".format(eidx))
        n_samples = 0
        tr_costs = [[], [], [], [], [], [], []]

        for data_ in train_d_:
            x = data_[0][0]
            y = data_[1][0]
            x_mask = np.ones((x.shape[0], x.shape[1]), dtype='float32')

            n_samples += x.shape[1]
            uidx += 1
            kl_start = min(1., kl_start + kl_rate)

            # build samples for the reparametrization trick
            zmuv = numpy.random.normal(loc=0.0, scale=1.0, size=(x.shape[0], x.shape[1], model_options['dim_z'])).astype('float32')
            # propagate samples forward into the network
            vae_cost_np, aux_cost_np, tot_cost_np, kld_cost_np, elbo_cost_np, nll_rev_cost_np, nll_gen_cost_np, not_finite = \
                f_prop(x, y, x_mask, zmuv, np.float32(kl_start))

            # skip nan gradients
            if not_finite:
                continue

            # update weights given learning rate
            f_update(numpy.float32(lrate))

            # update costs
            tr_costs[0].append(vae_cost_np)
            tr_costs[1].append(aux_cost_np)
            tr_costs[2].append(tot_cost_np)
            tr_costs[3].append(kld_cost_np)
            tr_costs[4].append(elbo_cost_np)
            tr_costs[5].append(nll_rev_cost_np)
            tr_costs[6].append(nll_gen_cost_np)

            # average of last 10 batches
            for n in range(len(tr_costs)):
                tr_costs[n] = tr_costs[n][-10:]

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print_philly_hb()
                checkpoint = time.time()
                str1 = 'Epoch {:d}  Update {:d}  VaeCost {:.2f}  AuxCost {:.2f}  KldCost {:.2f}  TotCost {:.2f}  ElboCost {:.2f}  NllRev {:.2f}  NllGen {:.2f}  KL_start {:.2f} Speed {:.2f}it/s'.format(
                    eidx, uidx, np.mean(tr_costs[0]), np.mean(tr_costs[1]), np.mean(tr_costs[3]),
                    np.mean(tr_costs[2]), np.mean(tr_costs[4]), np.mean(tr_costs[5]), np.mean(tr_costs[6]),
                    kl_start, float(uidx) / (checkpoint - start))
                print(str1)
                log_file.write(str1 + '\n')
                log_file.flush()

        print('Starting validation...')
        valid_err = pred_probs(f_log_probs, model_options, valid_d_, source='valid')
        test_err = pred_probs(f_log_probs, model_options, test_d_, source='test')
        history_errs.append(valid_err)
        str1 = 'Valid/Test ELBO: {:.2f}, {:.2f}'.format(valid_err, test_err)

        # decay learning rate if validation error increases
        if (old_valid_err < valid_err) and lrate > 0.0001:
            lrate = lrate / 2.0

        old_valid_err = history_errs[-1]
        print(str1)
        log_file.write(str1 + '\n')

        # finish after this many updates
        if uidx >= finish_after:
            print('Finishing after %d iterations!' % uidx)
            break

    valid_err = pred_probs(f_log_probs, model_options, valid_d_, source='valid')
    test_err = pred_probs(f_log_probs, model_options, test_d_, source='test')
    str1 = 'Valid/Test ELBO: {:.2f}, {:.2f}'.format(valid_err, test_err)
    print(str1)
    log_file.write(str1 + '\n')
    log_file.close()
    return valid_err


if __name__ == '__main__':
    pass
