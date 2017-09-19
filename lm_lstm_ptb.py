'''
Build a simple neural language model using GRU units
'''

import numpy as np
import os
import theano
import theano.tensor as T
import theano.tensor as tensor
<<<<<<< HEAD

import cPickle as pkl
import numpy
import reader
import warnings
import time

=======
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lm_data import PTB

import cPickle as pkl
import ipdb
import numpy
import copy
from costs import iwae_multi_eval
from tqdm import tqdm
import warnings
import time
import cPickle
>>>>>>> master
from collections import OrderedDict

#from char_data_iterator import TextIterator

profile = False
seed = 1234
<<<<<<< HEAD
numpy.random.seed(seed)
=======
num_iwae_samps = 25
num_iwae_iters = 1
numpy.random.seed(seed)
is_train = tensor.scalar('is_train')


def param_init_hsoftmax(options, params, nin, ncls, nout, prefix='hsoftmax'):
    nout_per_cls = (nout + ncls - 1) / ncls
    W1 = numpy.asarray(numpy.random.normal(
        0, 0.01, size=(nin, ncls)), dtype=theano.config.floatX)
    b1 = numpy.asarray(numpy.zeros((ncls,)), dtype=theano.config.floatX)

    # Second level of h_softmax
    W2 = numpy.asarray(numpy.random.normal(
        0, 0.01, size=(ncls, nin, nout_per_cls)), dtype=theano.config.floatX)
    b2 = numpy.asarray(numpy.zeros((ncls, nout_per_cls)), dtype=theano.config.floatX)

    # store some private vars
    options['hsoftmax_ncls'] = ncls
    options['nvocab'] = nout
    params[_p(prefix, 'W1')] = W1
    params[_p(prefix, 'W2')] = W2
    params[_p(prefix, 'b1')] = b1
    params[_p(prefix, 'b2')] = b2
    return params


def hsoftmax_layer(tparams, state_below, options, y_indexes=None,
                   prefix='hsoftmax', compute_all=False, **kwargs):
    """
    shape of state_below is expected to be: (#tsteps, #batchsize, #dim)
    y_indexes: a theano variable of true targets.
    """
    ncls = options['hsoftmax_ncls']
    nout = options['nvocab']
    nout_per_cls = (nout + ncls - 1) / ncls
    state_shp = state_below.shape

    if state_below.ndim == 3:
        reshaped = 1
        state_reshp = state_below.reshape([state_shp[0] * state_shp[1], state_shp[2]])
        batch_size = state_shp[1] * state_shp[0]
    else:
        reshaped = 0
        state_reshp = state_below
        batch_size = state_shp[0]

    if compute_all:
        # shape: (batch_size, output_size)  (batch size after reshaping)
        output = tensor.nnet.h_softmax(state_reshp, batch_size, nout,
                                       ncls, nout_per_cls,
                                       tparams[_p(prefix, 'W1')],
                                       tparams[_p(prefix, 'b1')],
                                       tparams[_p(prefix, 'W2')],
                                       tparams[_p(prefix, 'b2')])
        if reshaped:
            output = output.reshape([state_shp[0], state_shp[1], -1])
    else:
        if y_indexes != None:
            y_indexes = y_indexes.flatten()
        # shape: (batch_size,)
        output = tensor.nnet.h_softmax(state_reshp, batch_size, nout,
                                       ncls, nout_per_cls,
                                       tparams[_p(prefix, 'W1')],
                                       tparams[_p(prefix, 'b1')],
                                       tparams[_p(prefix, 'W2')],
                                       tparams[_p(prefix, 'b2')], y_indexes)
        if reshaped:
            output = output.reshape([state_shp[0], state_shp[1]])
    return output


def masked_softmax(x, axis=-1, mask=None):
    if mask is not None:
        x = (mask * x) + (1 - mask) * (-10)
        x = tensor.clip(x, -10., 10.)
    e_x = tensor.exp(x - tensor.max(x, axis=axis, keepdims=True))
    if mask is not None:
        e_x = e_x * mask
    softmax = e_x / (tensor.sum(e_x, axis=axis, keepdims=True) + 1e-6)
    return softmax

>>>>>>> master

def gradient_clipping(grads, tparams, clip_c=1.0):
    g2 = 0.
    for g in grads:
        g2 += (g**2).sum()
    g2 = tensor.sqrt(g2)
    not_finite = tensor.or_(tensor.isnan(g2), tensor.isinf(g2))
    new_grads = []
<<<<<<< HEAD
=======
    lr = tensor.scalar(name='lr')
>>>>>>> master
    for p, g in zip(tparams.values(), grads):
        new_grads.append(tensor.switch(
            g2 > clip_c, g * (clip_c / g2), g))
    return new_grads, not_finite, tensor.lt(clip_c, g2)


<<<<<<< HEAD
=======
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


>>>>>>> master
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


<<<<<<< HEAD
=======
def save_params(path, tparams):
    params = {}
    for kk, vv in tparams.iteritems():
        params[kk] = vv.get_value()
    cPickle.dump(params, open(path, 'wb'))


>>>>>>> master
# layers: 'name': ('parameter initializer', 'feedforward')
layers = {
    'ff': ('param_init_fflayer', 'fflayer'),
    'gru': ('param_init_gru', 'gru_layer'),
    'lstm': ('param_init_lstm', 'lstm_layer'),
<<<<<<< HEAD
=======
    'hsoftmax': ('param_init_hsoftmax', 'hsoftmax_layer'),
>>>>>>> master
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


<<<<<<< HEAD
class TimitData():
    def __init__(self, fn, batch_size):
        import numpy as np
        data = np.load(fn)

        ####
        # IMPORTANT: u_train is the input and x_train is the target.
        ##
        u_train, x_train = data['u_train'], data['x_train']
        u_valid, x_valid = data['u_valid'], data['x_valid']
        (u_test, x_test, mask_test) = data['u_test'],  data['x_test'], data['mask_test']

        # assert u_test.shape[0] == 1680
        # assert x_test.shape[0] == 1680
        # assert mask_test.shape[0] == 1680

        self.u_train = u_train
        self.x_train = x_train
        self.u_valid = u_valid
        self.x_valid = x_valid

        # make multiple of batchsize
        n_test_padded = ((u_test.shape[0] // batch_size) + 1)*batch_size
        assert n_test_padded > u_test.shape[0]
        pad = n_test_padded - u_test.shape[0]
        u_test = np.pad(u_test, ((0, pad), (0, 0), (0, 0)), mode='constant')
        x_test = np.pad(x_test, ((0, pad), (0, 0), (0, 0)), mode='constant')
        mask_test = np.pad(mask_test, ((0, pad), (0, 0)), mode='constant')
        self.u_test = u_test
        self.x_test = x_test
        self.mask_test = mask_test

        self.n_train = u_train.shape[0]
        self.n_valid = u_valid.shape[0]
        self.n_test = u_test.shape[0]
        self.batch_size = batch_size

        print("TRAINING SAMPLES LOADED", self.u_train.shape)
        print("TEST SAMPLES LOADED", self.u_test.shape)
        print("VALID SAMPLES LOADED", self.u_valid.shape)
        print("TEST AVG LEN        ", np.mean(self.mask_test.sum(axis=1)) * 200)
        # test that x and u are correctly shifted
        assert np.sum(self.u_train[:, 1:] - self.x_train[:, :-1]) == 0.0
        assert np.sum(self.u_valid[:, 1:] - self.x_valid[:, :-1]) == 0.0
        for row in range(self.u_test.shape[0]):
            l = int(self.mask_test[row].sum())
            if l > 0:  # if l is zero the sequence is fully padded.
                assert np.sum(self.u_test[row, 1:l] -
                              self.x_test[row, :l-1]) == 0.0, row

    def _iter_data(self, u, x, mask=None):
        # IMPORTANT: In SRNN (where the data come from) u refers to the input whereas x, to the target.
        indices = range(len(u))
        for idx in chunk(indices, n=self.batch_size):
            u_batch, x_batch = u[idx], x[idx]
            if mask is None:
                mask_batch = np.ones((x_batch.shape[0], x_batch.shape[1]), dtype='float32')
            else:
                mask_batch = mask[idx]
            yield u_batch, x_batch, mask_batch

    def get_train_batch(self):
        return iter(self._iter_data(self.u_train, self.x_train))

    def get_valid_batch(self):
        return iter(self._iter_data(self.u_valid, self.x_valid))

    def get_test_batch(self):
        return iter(self._iter_data(self.u_test, self.x_test, mask=self.mask_test))


=======
>>>>>>> master
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
<<<<<<< HEAD
=======
    if state_below.dtype == 'int32' or state_below.dtype == 'int64':
        return tparams[_p(prefix, 'W')][state_below] + tparams[_p(prefix, 'b')]
>>>>>>> master
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
<<<<<<< HEAD

=======
>>>>>>> master
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
<<<<<<< HEAD
        options, prefix='lstm', back_states = None,
        gaussian_s=None, mask=None, one_step=False,
        init_state=None, init_memory=None, nsteps=None,
=======
        options, prefix='lstm', back_states=None,
        gaussian_s=None, mask=None, one_step=False,
        init_state=None, init_memory=None, nsteps=None,
        provide_z=False,
>>>>>>> master
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
                tparams[_p('trans_1', 'W')],
                tparams[_p('trans_1', 'b')],
                tparams[_p('z_mus', 'W')],
                tparams[_p('z_mus', 'b')],
                tparams[_p('inf', 'W')],
                tparams[_p('inf', 'b')],
                tparams[_p('inf_mus', 'W')],
                tparams[_p('inf_mus', 'b')],
                tparams[_p('gen_mus', 'W')],
                tparams[_p('gen_mus', 'b')]]

    # initial/previous memory
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(mask, sbelow, d_, g_s, sbefore, cell_before,
              U, b, W, W_cond, trans_1_w, trans_1_b,
              z_mus_w, z_mus_b,
              inf_w, inf_b,
              inf_mus_w, inf_mus_b,
<<<<<<< HEAD
              gen_mus_w, gen_mus_b):

        p_z = tensor.nnet.softplus(tensor.dot(sbefore, trans_1_w) + trans_1_b)
=======
              gen_mus_w, gen_mus_b,
              hdrop=None):

        p_z = lrelu(tensor.dot(sbefore, trans_1_w) + trans_1_b)
>>>>>>> master
        z_mus = tensor.dot(p_z, z_mus_w) + z_mus_b
        z_dim = z_mus.shape[-1] / 2
        z_mu, z_sigma = z_mus[:, :z_dim], z_mus[:, z_dim:]

        if d_ is not None:
<<<<<<< HEAD
            encoder_hidden = tensor.nnet.softplus(tensor.dot(concatenate([sbefore, d_], axis=1), inf_w) + inf_b)
            encoder_mus = tensor.dot(encoder_hidden, inf_mus_w) + inf_mus_b
            encoder_mu, encoder_sigma = encoder_mus[:, :z_dim], encoder_mus[:, z_dim:]
            tild_z_t = encoder_mu + g_s * tensor.exp(0.5 * encoder_sigma)
=======
            encoder_hidden = lrelu(tensor.dot(concatenate([sbefore, d_], axis=1), inf_w) + inf_b)
            encoder_mus = tensor.dot(encoder_hidden, inf_mus_w) + inf_mus_b
            encoder_mu, encoder_sigma = encoder_mus[:, :z_dim], encoder_mus[:, z_dim:]
            tild_z_t = encoder_mu + g_s * tensor.exp(0.5 * encoder_sigma)
            log_pz = tensor.sum(log_prob_gaussian(tild_z_t, z_mu, z_sigma), axis=-1)
            log_qzIx = tensor.sum(log_prob_gaussian(tild_z_t, encoder_mu, encoder_sigma), axis=-1)
>>>>>>> master
            kld = gaussian_kld(encoder_mu, encoder_sigma, z_mu, z_sigma)
            kld = tensor.sum(kld, axis=-1)
            decoder_mus = tensor.dot(tild_z_t, gen_mus_w) + gen_mus_b
            decoder_mu, decoder_sigma = decoder_mus[:, :d_.shape[1]], decoder_mus[:, d_.shape[1]:]
            decoder_mu = tensor.tanh(decoder_mu)
            decoder_mu = T.clip(decoder_mu, -10., 10.)
            decoder_sigma = T.clip(decoder_sigma, -10., 10.)
            disc_d_ = theano.gradient.disconnected_grad(d_)
<<<<<<< HEAD
            recon_cost = (tensor.exp(0.5 * decoder_sigma) + tensor.sqr(disc_d_ - decoder_mu)/(2 * tensor.sqr(tensor.exp(0.5 * decoder_sigma))))
            recon_cost = tensor.sum(recon_cost, axis=-1)
        else:
            tild_z_t = z_mu + g_s * tensor.exp(0.5 * z_sigma)
            kld = tensor.sum(tild_z_t, axis=-1) * 0.
            recon_cost = tensor.sum(tild_z_t, axis=-1) * 0.

=======
            recon_cost = -log_prob_gaussian(disc_d_, decoder_mu, decoder_sigma)
            recon_cost = tensor.sum(recon_cost, axis=-1)
        else:
            if provide_z:
                print("Zs were provided!")
                tild_z_t = g_s
            else:
                tild_z_t = z_mu + g_s * tensor.exp(0.5 * z_sigma)

            kld = tensor.sum(tild_z_t, axis=-1) * 0.
            recon_cost = tensor.sum(tild_z_t, axis=-1) * 0.
            log_pz = kld * 0.
            log_qzIx = kld * 0.

        # recurrent dropout
        if hdrop is not None:
            sbefore = sbefore * hdrop
>>>>>>> master
        z = tild_z_t
        preact = tensor.dot(sbefore, param('U')) +  tensor.dot(z, W_cond)
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
<<<<<<< HEAD
        return h, c, z, kld, recon_cost
=======
        return h, c, z, log_pz, log_qzIx, kld, recon_cost
>>>>>>> master

    lstm_state_below = tensor.dot(state_below, param('W')) + param('b')
    if state_below.ndim == 3:
        lstm_state_below = lstm_state_below.reshape((state_below.shape[0],
                                                     state_below.shape[1],
                                                     -1))
    if one_step:
        mask = mask.dimshuffle(0, 'x')
<<<<<<< HEAD
        h, c = _step(mask, lstm_state_below, init_state, init_memory)
        rval = [h, c]
=======
        _step_inps = [mask, lstm_state_below, None, gaussian_s, init_state, init_memory] + non_seqs
        h, c, z, _, _, _, _ = _step(*_step_inps)
        rval = [h, c, z]
        updates = {}

>>>>>>> master
    else:
        if mask.ndim == 3 and mask.ndim == state_below.ndim:
            mask = mask.reshape((mask.shape[0], \
                                 mask.shape[1]*mask.shape[2])).dimshuffle(0, 1, 'x')
        elif mask.ndim == 2:
            mask = mask.dimshuffle(0, 1, 'x')

<<<<<<< HEAD
        rval, updates = theano.scan(
            _step, sequences=[mask, lstm_state_below, back_states, gaussian_s],
            outputs_info = [init_state, init_memory, None, None, None],
            name=_p(prefix, '_layers'), non_sequences=non_seqs, strict=True, n_steps=nsteps)
=======
        trng = RandomStreams(seed)
        hdrop = trng.binomial(
            (lstm_state_below.shape[1], options['dim']), p=0.9, n=1,
            dtype=theano.config.floatX)
        hdrop = is_train * hdrop + (1 - is_train) * tensor.ones_like(hdrop)
        non_seqs.append(hdrop)

        rval, updates = theano.scan(
            _step, sequences=[mask, lstm_state_below, back_states, gaussian_s],
            outputs_info = [init_state, init_memory, None, None, None, None, None],
            name=_p(prefix, '_layers'), non_sequences=non_seqs, strict=True, n_steps=nsteps)

>>>>>>> master
    return [rval, updates]


# initialize all parameters
def init_params(options):
    params = OrderedDict()
<<<<<<< HEAD
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    params = get_layer('latent_lstm')[0](options, params,
                                         prefix='encoder',
                                         nin=options['dim_word'],
                                         dim=options['dim'])
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'])
=======
    params = get_layer('latent_lstm')[0](options, params,
                                         prefix='encoder',
                                         nin=options['dim_proj'],
                                         dim=options['dim'])
    params = get_layer('ff')[0](options, params, prefix='ff_in_lstm',
                                nin=options['dim_input'], nout=options['dim_proj'],
                                ortho=True)
    params = get_layer('ff')[0](options, params, prefix='ff_out_mus',
                                nin=options['dim'],
                                nout=options['dim_input'],
                                ortho=True)
>>>>>>> master
    U = numpy.concatenate([norm_weight(options['dim_z'], options['dim']),
                           norm_weight(options['dim_z'], options['dim']),
                           norm_weight(options['dim_z'], options['dim']),
                           norm_weight(options['dim_z'], options['dim'])], axis=1)
    params[_p('z_cond', 'W')] = U

    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder_r',
<<<<<<< HEAD
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm_r',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev_r',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_r',
                                nin=options['dim_word'],
                                nout=options['n_words'])


=======
                                              nin=options['dim_proj'],
                                              dim=options['dim'])
    params = get_layer('ff')[0](options, params, prefix='ff_out_mus_r',
                                nin=options['dim'],
                                nout=options['dim_input'],
                                ortho=True)
>>>>>>> master
    #Prior Network params
    params = get_layer('ff')[0](options, params, prefix='trans_1', nin=options['dim'], nout=options['prior_hidden'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='z_mus', nin=options['prior_hidden'], nout=2 * options['dim_z'], ortho=False)
    #Inference network params
    params = get_layer('ff')[0](options, params, prefix='inf', nin = 2 * options['dim'], nout=options['encoder_hidden'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='inf_mus', nin = options['encoder_hidden'], nout=2 * options['dim_z'], ortho=False)
    #Generative Network params
    params = get_layer('ff')[0](options, params, prefix='gen_mus', nin = options['dim_z'], nout=2 * options['dim'], ortho=False)
    return params


def build_rev_model(tparams, options, x, y, x_mask):
<<<<<<< HEAD
    #xc = tensor.concatenate([x[:1, :], y], axis=0)
    #xc_mask = tensor.concatenate([tensor.alloc(1, 1, x_mask.shape[1]), x_mask], axis=0)
    xr = x[::-1]
    xr_mask = x_mask[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])
    embr_shifted = tensor.zeros_like(embr)
    embr_shifted = tensor.set_subtensor(embr_shifted[1:], embr[:-1])
    embr = embr_shifted


    (states_rev, _), updates_rev = get_layer(options['encoder'])[1](tparams, embr, options, prefix='encoder_r', mask=xr_mask)
    out_lstm = get_layer('ff')[1](tparams, states_rev, options, prefix='ff_logit_lstm_r', activ='linear')
    out_prev = get_layer('ff')[1](tparams, embr, options, prefix='ff_logit_prev_r', activ='linear')
    out = lrelu(out_lstm + out_prev)

    logit = get_layer('ff')[1](tparams, out, options, prefix='ff_logit_r',
                               activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(
        logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

    states_rev = states_rev[::-1]

    targets = xr
    targets_mask = xr_mask

    # cost
    x_flat = targets.flatten()
    x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
    cost = -tensor.log(probs.flatten()[x_flat_idx])
    cost = cost.reshape([x.shape[0], x.shape[1]])
    cost = (cost * targets_mask).sum(0)

    return cost, states_rev, updates_rev


    #return nll_rev, states_rev, updates_rev
=======
    # for the backward rnn, we just need to invert x and x_mask
    # concatenate first x and all targets y
    # x = [x1, x2, x3]
    # y = [x2, x3, x4]
    xc = tensor.concatenate([x[:1, :], y], axis=0)
    # xc = [x1, x2, x3, x4]
    xc_mask = tensor.concatenate([tensor.alloc(1, 1, x_mask.shape[1]), x_mask], axis=0)
    # xc_mask = [1, 1, 1, 0]
    # xr = [x4, x3, x2, x1]
    xr = xc[::-1]
    # xr_mask = [0, 1, 1, 1]
    xr_mask = xc_mask[::-1]

    xr_emb = get_layer('ff')[1](tparams, xr, options, prefix='ff_in_lstm', activ='lambda x: x')
    (states_rev, _), updates_rev = get_layer(options['encoder'])[1](tparams, xr_emb, options, prefix='encoder_r', mask=xr_mask)
    # shift mus for prediction [o4, o3, o2]
    # targets are [x3, x2, x1]
    out = states_rev[:-1]
    targets = xr[1:]
    targets_mask = xr_mask[1:]
    out_logits = get_layer('ff')[1](tparams, out, options, prefix='ff_out_mus_r', activ='linear')
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
>>>>>>> master


# build a training model
def build_gen_model(tparams, options, x, y, x_mask, zmuv, states_rev):
<<<<<<< HEAD
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # input
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    # disconnecting reconstruction gradient from going in the backward encoder
    rvals, updates_gen = get_layer('latent_lstm')[1](
        tparams, state_below=emb, options=options,
        prefix='encoder', mask=x_mask, gaussian_s=zmuv,
        back_states=states_rev)

    states_gen, kld, rec_cost_rev = rvals[0], rvals[3], rvals[4]
    # Compute parameters of the output distribution
    out_lstm = get_layer('ff')[1](tparams, states_gen, options, prefix='ff_logit_lstm', activ='linear')
    out_prev = get_layer('ff')[1](tparams, emb, options, prefix='ff_logit_prev', activ='linear')
    out = lrelu(out_lstm + out_prev)
    logit = get_layer('ff')[1](tparams, out, options, prefix='ff_logit',
                               activ='linear')
    # Compute gaussian log prob of target
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(
        logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

    # cost
    x_flat = x.flatten()
    x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
    cost = -tensor.log(probs.flatten()[x_flat_idx])
    cost = cost.reshape([x.shape[0], x.shape[1]])
    cost = (cost * x_mask).sum(0)
    kld = (kld * x_mask).sum(0)
    rec_cost_rev = (rec_cost_rev * x_mask).sum(0)
    return cost, states_gen, kld, rec_cost_rev, updates_gen
=======
    opt_ret = dict()
    # disconnecting reconstruction gradient from going in the backward encoder
    x_emb = get_layer('ff')[1](tparams, x, options, prefix='ff_in_lstm', activ='lambda x: x')
    rvals, updates_gen = get_layer('latent_lstm')[1](
        tparams, state_below=x_emb, options=options,
        prefix='encoder', mask=x_mask, gaussian_s=zmuv,
        back_states=states_rev)

    states_gen, z, log_pz, log_qzIx, kld, rec_cost_rev = rvals[0], rvals[2], rvals[3], rvals[4], rvals[5], rvals[6]
    out_logits = get_layer('ff')[1](tparams, states_gen, options, prefix='ff_out_mus', activ='linear')
    out_probs = masked_softmax(out_logits, axis=-1)

    nll_gen = categorical_crossentropy(y, out_probs)
    nll_gen = (nll_gen * x_mask).sum(0)
    log_pxIz = -nll_gen
    log_pz = (log_pz * x_mask).sum(0)
    log_qzIx = (log_qzIx * x_mask).sum(0)
    kld = (kld * x_mask).sum(0)
    rec_cost_rev = (rec_cost_rev * x_mask).sum(0)
    return nll_gen, states_gen, kld, rec_cost_rev, updates_gen, log_pxIz, log_pz, log_qzIx, z
>>>>>>> master


def ELBOcost(rec_cost, kld, kld_weight=1.):
    assert kld.ndim == 1
    assert rec_cost.ndim == 1
    return rec_cost + kld_weight * kld


<<<<<<< HEAD
def pred_probs(f_log_probs, options, data, source='valid'):
    rvals = []
    n_done = 0

    for x, y in reader.ptb_iterator(data, options['batch_size'], options['maxlen']):
        x = x.T
        y = y.T

        x_mask = np.ones((x.shape[0], x.shape[1]), dtype='float32')
        n_done += x.shape[1]

=======
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
    x_emb = tensor.switch(last_word[:, None] < 0,
                          tensor.alloc(0., last_word.shape[0], options['dim_proj']),
                          get_layer('ff')[1](tparams, last_word, options,
                                             prefix='ff_in_lstm',
                                             activ='lambda x: x'))

    # apply one step of gru layer
    rvals, update_gen = get_layer('latent_lstm')[1](tparams, x_emb, options,
                                                    prefix='encoder',
                                                    mask=None,
                                                    one_step=True,
                                                    gaussian_s=gaussian_sampled,
                                                    back_states=None,
                                                    init_state=init_state,
                                                    init_memory=init_memory,
                                                    provide_z=provide_z)
    next_state, next_memory, z = rvals

    # Compute parameters of the output distribution
    logits = get_layer('ff')[1](tparams, next_state, options, prefix='ff_out_mus', activ='linear')
    next_probs = masked_softmax(logits, axis=-1)

    # next word probability
    print('Building f_next..')
    inps = [last_word, init_state, init_memory, gaussian_sampled]
    outs = [next_probs, next_state, next_memory]
    f_next = theano.function(inps, outs, name='f_next')
    print('Done')

    return f_next


# generate sample
def gen_sample(tparams, f_next, options, trng=None, maxlen=30, argmax=False, kickstart=None, zmuv=None,
               unk_id=None, eos_id=None, bos_id=None):
    assert bos_id is not None
    samples = []
    samples_scores = 0
    nb_samples = 1

    if kickstart is not None and zmuv is not None:
        assert kickstart.shape[1] == zmuv.shape[0]

    if kickstart is not None:
        maxlen = maxlen + len(kickstart)
        nb_samples = kickstart.shape[1]

    if zmuv is not None:
        nb_samples = zmuv.shape[0]

    # initial token is indicated by a -1 and initial state is zero
    next_w = bos_id * numpy.ones((nb_samples,)).astype('int64')
    next_state = numpy.zeros((nb_samples, options['dim'])).astype('float32')
    next_memory = numpy.zeros((nb_samples, options['dim'])).astype('float32')

    for ii in range(maxlen):
        if zmuv is None:
            zmuv_t = numpy.random.normal(
                loc=0.0, scale=1.0,
                size=(next_w.shape[0], options['dim_z'])).astype('float32')
        else:
            zmuv_t = zmuv[:, ii, :]

        inps = [next_w, next_state, next_memory, zmuv_t]
        ret = f_next(*inps)
        next_p, next_state, next_memory = ret

        if unk_id is not None:
            next_p[:, unk_id] = 0.
        if (eos_id is not None) and ii < 5:
            next_p[:, eos_id] = 0.
        next_p = next_p / numpy.sum(next_p, axis=1)[:, None]

        if argmax:
            nw = next_p.argmax(axis=1)
        else:
            nw = []
            for i in range(next_p.shape[0]):
                nw_i = numpy.random.choice(range(next_p.shape[1]), 1, p=next_p[i, :])
                nw.append(nw_i[0])
            nw = numpy.asarray(nw)

        if kickstart is not None and ii < len(kickstart):
            nw = kickstart[ii]

        next_w = nw
        samples.append(nw)
        samples_scores += np.log(next_p[np.arange(next_w.shape[0]), nw])

    samples = np.stack(samples)
    return samples, samples_scores


def pred_probs(f_log_probs, f_iwae_eval, options, data, source='valid'):
    rvals = []
    iwae_rvals = []
    n_done = 0

    next_batch = (lambda: data.get_valid_batch()) \
        if source == 'valid' else (lambda: data.get_test_batch())
    nbatches = 0
    for batch in next_batch():
        nbatches += 1
    iterate = next_batch()
    for idx in tqdm(range(nbatches), ncols=80, ascii=True):
        x, y, x_mask = next(iterate)
        x = x.transpose(1, 0)
        y = y.transpose(1, 0)
        x_mask = x_mask.transpose(1, 0)
        n_done += numpy.sum(x_mask)
>>>>>>> master
        zmuv = numpy.random.normal(loc=0.0, scale=1.0, size=(
            x.shape[0], x.shape[1], options['dim_z'])).astype('float32')
        elbo = f_log_probs(x, y, x_mask, zmuv)
        for val in elbo:
            rvals.append(val)
<<<<<<< HEAD
    return numpy.array(rvals).mean()
=======
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
>>>>>>> master


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


<<<<<<< HEAD
def train(dim_word=200,  # input vector dimensionality
=======
def train(dim_input=200,  # input vector dimensionality
>>>>>>> master
          dim=2000,  # the number of GRU units
          dim_proj=600,  # the number of GRU units
          encoder='lstm',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 weight decay penalty
<<<<<<< HEAD
          lrate=0.001,
          maxlen=100,  # maximum length of the description
          optimizer='adam',
          batch_size=16,
          n_words=30000,
=======
          lrate=0.0002,
          maxlen=100,  # maximum length of the description
          optimizer='adam',
          batch_size=16,
>>>>>>> master
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,  # save the parameters after every saveFreq updates
          sampleFreq=100,  # generate some samples after every sampleFreq
          dataset=None,  # Not used
          valid_dataset=None,  # Not used
          dictionary=None,  # Not used
          use_dropout=False,
          reload_=False,
          kl_start=0.2,
          weight_aux=0.,
          kl_rate=0.0003):
<<<<<<< HEAD
    dim_proj = dim
    data_path = '/data/lisatmp4/anirudhg/ptb/'
    prior_hidden = dim
    dim_z = 256
    encoder_hidden = dim
    learn_h0 = False

    desc = saveto + 'seed_' + str(seed) + '_model_' + str(weight_aux) + '_weight_aux_' +  str(kl_start) + '_kl_Start_' + str(kl_rate) +  '_kl_rate_log.txt'
    opts = saveto + 'seed_' + str(seed) + '_model_' + str(weight_aux) + '_weight_aux_' +  str(kl_start) + '_kl_Start_' + str(kl_rate) +  '_kl_rate_opts.pkl'

    print(desc)

    raw_data = reader.ptb_raw_data(data_path)
    train_data, valid_data, test_data, _ = raw_data

=======

    prior_hidden = dim
    dim_z = 32
    encoder_hidden = dim
    learn_h0 = False

    desc = saveto + 'seed_' + str(seed) + '_model_' + str(weight_aux) + '_weight_aux_' + \
        str(kl_start) + '_kl_Start_' + str(kl_rate) +  '_kl_rate_log.txt'
    opts = saveto + 'seed_' + str(seed) + '_model_' + str(weight_aux) + '_weight_aux_' + \
        str(kl_start) + '_kl_Start_' + str(kl_rate) +  '_kl_rate_opts.pkl'
    pars = saveto + 'seed_' + str(seed) + '_model_' + str(weight_aux) + '_weight_aux_' + \
        str(kl_start) + '_kl_Start_' + str(kl_rate) +  '_kl_rate_pars.npz'

    print(desc)

    data = PTB("./experiments/data", 35, batch_size)
    dim_input = data.voc_size
>>>>>>> master

    # Model options
    model_options = locals().copy()
    pkl.dump(model_options, open(opts, 'wb'))
    log_file = open(desc, 'w')

<<<<<<< HEAD
    #data = TimitData("timit_raw_batchsize64_seqlen40.npz", batch_size=model_options['batch_size'])

=======
>>>>>>> master
    print('Building model')
    params = init_params(model_options)
    tparams = init_tparams(params)

<<<<<<< HEAD
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
=======
    x = tensor.lmatrix('x')
    y = tensor.lmatrix('y')
    x_mask = tensor.matrix('x_mask')
    # Debug test_value
    x.tag.test_value = np.random.rand(11, 20).astype("int64")
    y.tag.test_value = np.random.rand(11, 20).astype("int64")
    x_mask.tag.test_value = np.ones((11, 20)).astype("float32")
>>>>>>> master

    zmuv = tensor.tensor3('zmuv')
    weight_f = tensor.scalar('weight_f')
    lr = tensor.scalar('lr')

    # build the symbolic computational graph
    nll_rev, states_rev, updates_rev = \
        build_rev_model(tparams, model_options, x, y, x_mask)
<<<<<<< HEAD
    nll_gen, states_gen, kld, rec_cost_rev, updates_gen = \
=======
    nll_gen, states_gen, kld, rec_cost_rev, updates_gen, \
        log_pxIz, log_pz, log_qzIx, z = \
>>>>>>> master
        build_gen_model(tparams, model_options, x, y, x_mask, zmuv, states_rev)

    vae_cost = ELBOcost(nll_gen, kld, kld_weight=weight_f).mean()
    elbo_cost = ELBOcost(nll_gen, kld, kld_weight=1.).mean()
    aux_cost = (numpy.float32(weight_aux) * (rec_cost_rev + nll_rev)).mean()
    tot_cost = (vae_cost + aux_cost)
    nll_gen_cost = nll_gen.mean()
    nll_rev_cost = nll_rev.mean()
    kld_cost = kld.mean()

<<<<<<< HEAD
=======
    nbatch = 0
    for batch in data.get_valid_batch():
        nbatch += 1
    print('Total valid batches: {}'.format(nbatch))

>>>>>>> master
    print('Building f_log_probs...')
    inps = [x, y, x_mask, zmuv, weight_f]
    f_log_probs = theano.function(
        inps[:-1], ELBOcost(nll_gen, kld, kld_weight=1.),
<<<<<<< HEAD
        updates=(updates_gen + updates_rev), profile=profile, on_unused_input='ignore')
=======
        updates=(updates_gen + updates_rev), profile=profile,
        givens={is_train: numpy.float32(0.)})
    f_iwae_eval = theano.function(
        inps[:-1], [log_pxIz, log_pz, log_qzIx],
        updates=(updates_gen + updates_rev),
        givens={is_train: numpy.float32(0.)})
>>>>>>> master
    print('Done')

    print('Computing gradient...')
    grads = tensor.grad(tot_cost, itemlist(tparams))
    print('Done')

<<<<<<< HEAD
    all_grads, non_finite, clipped = gradient_clipping(grads, tparams, 5.)
=======
    all_grads, non_finite, clipped = gradient_clipping(grads, tparams, 100.)
>>>>>>> master
    # update function
    all_gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
                   for k, p in tparams.iteritems()]
    all_gsup = [(gs, g) for gs, g in zip(all_gshared, all_grads)]
    # forward pass + gradients
    outputs = [vae_cost, aux_cost, tot_cost, kld_cost, elbo_cost, nll_rev_cost, nll_gen_cost, non_finite]
    print('Fprop')
<<<<<<< HEAD
    f_prop = theano.function(inps, outputs, updates=all_gsup, on_unused_input='ignore')
=======
    f_prop = theano.function(inps, outputs, updates=all_gsup,
                             givens={is_train: numpy.float32(1.)})
>>>>>>> master
    print('Fupdate')
    f_update = eval(optimizer)(lr, tparams, all_gshared)

    print('Optimization')
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
<<<<<<< HEAD

=======
    old_valid_err = 99999

    # epochs loop
>>>>>>> master
    for eidx in range(max_epochs):
        print("Epoch: {}".format(eidx))
        n_samples = 0
        tr_costs = [[], [], [], [], [], [], []]

<<<<<<< HEAD
        #for x, y, x_mask in data.get_train_batch():
        for x, y in reader.ptb_iterator(train_data, batch_size, maxlen):
          # Transpose data to have the time steps on dimension 0.
            x = x.T
            y = y.T

            x_mask = np.ones((x.shape[0], x.shape[1]), dtype='float32')
=======
        for x, y, x_mask in data.get_train_batch():
            # Transpose data to have the time steps on dimension 0.
            x = x.transpose(1, 0).astype('int32')
            y = y.transpose(1, 0).astype('int32')
            x_mask = x_mask.transpose(1, 0).astype('float32')

>>>>>>> master
            n_samples += x.shape[1]
            uidx += 1
            if kl_start < 1.:
                kl_start += kl_rate

            ud_start = time.time()
            # compute cost, grads and copy grads to shared variables
<<<<<<< HEAD
            zmuv = numpy.random.normal(loc=0.0, scale=1.0, size=(x.shape[0], x.shape[1], model_options['dim_z'])).astype('float32')
            vae_cost_np, aux_cost_np, tot_cost_np, kld_cost_np, elbo_cost_np, nll_rev_cost_np, nll_gen_cost_np, not_finite_np = \
=======
            zmuv = numpy.random.normal(loc=0.0, scale=1.0, size=(
                x.shape[0], x.shape[1], model_options['dim_z'])).astype('float32')
            vae_cost_np, aux_cost_np, tot_cost_np, kld_cost_np, \
                elbo_cost_np, nll_rev_cost_np, nll_gen_cost_np, not_finite_np = \
>>>>>>> master
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
            ud = time.time() - ud_start

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                str1 = 'Epoch {:d}  Update {:d}  VaeCost {:.2f}  AuxCost {:.2f}  KldCost {:.2f}  TotCost {:.2f}  ElboCost {:.2f}  NllRev {:.2f}  NllGen {:.2f}  KL_start {:.2f}'.format(
                    eidx, uidx, np.mean(tr_costs[0]), np.mean(tr_costs[1]), np.mean(tr_costs[3]), np.mean(tr_costs[2]), np.mean(tr_costs[4]), \
                    np.mean(tr_costs[5]), np.mean(tr_costs[6]), kl_start)
                print(str1)
                log_file.write(str1 + '\n')
                log_file.flush()

<<<<<<< HEAD
        if eidx in [10, 20]:
            lrate = lrate / 2.0

        print 'Starting validation...'
        valid_err = pred_probs(f_log_probs, model_options, valid_data, source='valid')
        test_err = pred_probs(f_log_probs, model_options, test_data, source='test')
        history_errs.append(valid_err)
        str1 = 'Valid/Test ELBO: {:.2f}, {:.2f}'.format(valid_err, test_err)
        print(str1)
        log_file.write(str1 + '\n')

=======

        print('Starting validation...')
        valid_err = pred_probs(f_log_probs, f_iwae_eval, model_options, data, source='valid')
        test_err = pred_probs(f_log_probs, f_iwae_eval, model_options, data, source='test')
        history_errs.append(valid_err[0])
        str1 = 'Valid/Test ELBO: {:.2f}, {:.2f}'.format(valid_err[0], test_err[0])
        str2 = 'Valid/Test IWAE: {:.2f}, {:.2f}'.format(valid_err[1], test_err[1])
        str1 = str1 + '\n' + str2
        print(str1)
        log_file.write(str1 + '\n')

        if (old_valid_err < history_errs[-1]):
            if lrate > 0.0001:
                lrate = lrate / 2.0
        else:
            # Save better model.
            save_params(pars, tparams)

        old_valid_err = history_errs[-1]

>>>>>>> master
        # finish after this many updates
        if uidx >= finish_after:
            print('Finishing after %d iterations!' % uidx)
            break

<<<<<<< HEAD
    valid_err = pred_probs(f_log_probs, model_options, valid_data, source='valid')
    test_err = pred_probs(f_log_probs, model_options, test_data, source='test')
    str1 = 'Valid/Test ELBO: {:.2f}, {:.2f}'.format(valid_err, test_err)
=======
    valid_err = pred_probs(f_log_probs, f_iwae_eval, model_options, data, source='valid')
    test_err = pred_probs(f_log_probs, f_iwae_eval, model_options, data, source='test')
    str1 = 'Valid/Test ELBO: {:.2f}, {:.2f}'.format(valid_err[0], test_err[0])
    str2 = 'Valid/Test IWAE: {:.2f}, {:.2f}'.format(valid_err[1], test_err[1])
    str1 = str1 + '\n' + str2
>>>>>>> master
    print(str1)
    log_file.write(str1 + '\n')
    log_file.close()
    return valid_err


if __name__ == '__main__':
    pass
