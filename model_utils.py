
import numpy as np
import os
import theano
import theano.tensor as T
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import copy
import warnings
import time
from collections import OrderedDict

profile = False

# some input flags
t_reset_states = T.scalar('reset_states')


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
def parname(pp, name):
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
    outf = open(path, 'wb')
    pkl.dump(params, outf)
    outf.close()


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


def ortho_weight(rng, ndim, scale=1.1):
    W = rng.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32') * scale


# weight initializer, normal by default
def norm_weight(rng, nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * rng.randn(nin, nout)
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
    assert nin is not None
    assert nout is not None
    rng = options['rng']
    params[parname(prefix, 'W')] = norm_weight(rng, nin, nout, scale=0.01, ortho=ortho)
    params[parname(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[parname(prefix, 'W')]) +
        tparams[parname(prefix, 'b')])


def param_init_lstm(options,
                    params,
                    prefix='lstm',
                    nin=None,
                    dim=None):
    rng = options['rng']
    assert nin is not None
    assert dim is not None
    W = numpy.concatenate([norm_weight(rng, nin, dim),
                           norm_weight(rng, nin, dim),
                           norm_weight(rng, nin, dim),
                           norm_weight(rng, nin, dim)],
                           axis=1)

    params[parname(prefix,'W')] = W
    U = numpy.concatenate([ortho_weight(rng, dim),
                           ortho_weight(rng, dim),
                           ortho_weight(rng, dim),
                           ortho_weight(rng, dim)],
                           axis=1)

    params[parname(prefix,'U')] = U
    params[parname(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')
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

    param = lambda name: tparams[parname(prefix, name)]
    dim = param('U').shape[0]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    init_state = tensor.alloc(0., n_samples, dim)

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

    def _step(mask, sbelow, h_tm1, c_tm1, *args):
        preact = tensor.dot(h_tm1, param('U'))
        preact += sbelow

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
                                    name=parname(prefix, '_layers'),
                                    non_sequences=non_seqs,
                                    strict=True,
                                    n_steps=nsteps)
    return [rval, updates]


def latent_lstm_layer(
        tparams, state_below,
        options, prefix='lstm', back_states=None,
        gaussian_s=None, mask=None, one_step=False,
        init_state=None, nsteps=None, **kwargs):

    if nsteps is None:
        nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    param = lambda name: tparams[parname(prefix, name)]
    dim = param('U').shape[0]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # initial/previous state
    if init_state is None:
        if options.get('carry_h0', False):
            print('Carrying states over...')
            init_ary = numpy.zeros((
                options['batch_size'], 2 * options['dim'])).astype('float32')
            init_state = theano.shared(
                    init_ary, name=parname(prefix, 'init_state'))
            tparams[parname(prefix, 'init_state')] = init_state
        else:
            init_state = tensor.alloc(0., n_samples, 2 * dim)

    init_h0 = init_state[:, :dim]
    init_c0 = init_state[:, dim:]

    U = param('U')
    non_seqs = [U, tparams[parname('z_cond', 'W')],
                tparams[parname('pri_ff_1', 'W')],
                tparams[parname('pri_ff_1', 'b')],
                tparams[parname('pri_ff_2', 'W')],
                tparams[parname('pri_ff_2', 'b')],
                tparams[parname('inf_ff_1', 'W')],
                tparams[parname('inf_ff_1', 'b')],
                tparams[parname('inf_ff_2', 'W')],
                tparams[parname('inf_ff_2', 'b')],
                tparams[parname('aux_ff_1', 'W')],
                tparams[parname('aux_ff_1', 'b')],
                tparams[parname('aux_ff_2', 'W')],
                tparams[parname('aux_ff_2', 'b')]]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(mask, xbelow, sbelow, d_, zmuv, h_tm1, c_tm1,
              U, W_cond, pri_ff_1_w, pri_ff_1_b,
              pri_ff_2_w, pri_ff_2_b,
              inf_ff_1_w, inf_ff_1_b,
              inf_ff_2_w, inf_ff_2_b,
              aux_ff_1_w, aux_ff_1_b,
              aux_ff_2_w, aux_ff_2_b):

        # previous state and current input
        pri_inp = concatenate([h_tm1, xbelow], axis=1)
        p_z = lrelu(tensor.dot(pri_inp, pri_ff_1_w) + pri_ff_1_b)
        z_mus = tensor.dot(p_z, pri_ff_2_w) + pri_ff_2_b
        z_dim = z_mus.shape[-1] / 2
        z_mu, z_sigma = z_mus[:, :z_dim], z_mus[:, z_dim:]

        if d_ is not None:
            # previous state, backward state and current input
            inf_inp = concatenate([h_tm1, d_, xbelow], axis=1)
            inf_inp = lrelu(tensor.dot(inf_inp, inf_ff_1_w) + inf_ff_1_b)
            encoder_mus = tensor.dot(inf_inp, inf_ff_2_w) + inf_ff_2_b
            encoder_mus = T.clip(encoder_mus, -10., 10.)
            encoder_mu, encoder_sigma = encoder_mus[:, :z_dim], encoder_mus[:, z_dim:]
            z_smp = encoder_mu + zmuv * tensor.exp(0.5 * encoder_sigma)
            kld = gaussian_kld(encoder_mu, encoder_sigma, z_mu, z_sigma)
            kld = tensor.sum(kld, axis=-1)

            aux_hid = tensor.dot(z_smp, aux_ff_1_w) + aux_ff_1_b
            aux_hid = lrelu(aux_hid)
            # concatenate with forward state
            if options['use_h_in_aux']:
                print("Using h_in_aux...")
                aux_hid = tensor.concatenate([aux_hid, h_tm1], axis=1)

            aux_out = tensor.dot(aux_hid, aux_ff_2_w) + aux_ff_2_b
            aux_out = T.clip(aux_out, -10., 10.)
            aux_mu, aux_sigma = aux_out[:, :d_.shape[1]], aux_out[:, d_.shape[1]:]
            aux_mu = tensor.tanh(aux_mu)
            disc_d_ = theano.gradient.disconnected_grad(d_)
            aux_cost = -log_prob_gaussian(disc_d_, mean=aux_mu, log_var=aux_sigma)
            aux_cost = tensor.sum(aux_cost, axis=-1)
        else:
            z_smp = z_mu + zmuv * tensor.exp(0.5 * z_sigma)
            kld = tensor.sum(z_smp, axis=-1) * 0.
            aux_cost = tensor.sum(z_smp, axis=-1) * 0.

        # transform z
        gen_out = tensor.dot(z_smp, W_cond)
        preact = tensor.dot(h_tm1, U) + sbelow + gen_out

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_tm1 + i * c
        c = mask * c + (1. - mask) * c_tm1
        h = o * tensor.tanh(c)
        h = mask * h + (1. - mask) * h_tm1
        return h, c, z_smp, kld, aux_cost

    lstm_state_below = tensor.dot(state_below, param('W')) + param('b')
    if state_below.ndim == 3:
        lstm_state_below = lstm_state_below.reshape((
            state_below.shape[0], state_below.shape[1], -1))
    if one_step:
        mask = mask.dimshuffle(0, 'x')
        h, c = _step(mask, state_below, lstm_state_below, init_h0, init_c0)
        rval = [h, c]
    else:
        if mask.ndim == 3 and mask.ndim == state_below.ndim:
            mask = mask.reshape((mask.shape[0], mask.shape[1] * mask.shape[2])).dimshuffle(0, 1, 'x')
        elif mask.ndim == 2:
            mask = mask.dimshuffle(0, 1, 'x')

        rval, updates = theano.scan(
                _step, sequences=[mask, state_below, lstm_state_below, back_states, gaussian_s],
            outputs_info=[init_h0, init_c0, None, None, None],
            name=parname(prefix, '_layers'), non_sequences=non_seqs, strict=True, n_steps=nsteps)
        if options.get('carry_h0', False):
            print('- Adding update state...')
            updates[init_state] = tensor.concatenate([
                rval[0][-1], rval[1][-1]], axis=1)
    return [rval, updates]
