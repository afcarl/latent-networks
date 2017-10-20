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
import numpy
import copy

import warnings
import time
from collections import OrderedDict
from blizzard import Blizzard_tbptt
from model_utils import *
profile = False


class Iterator(object):
    """
    Dataset iterator
    Parameters
    ----------
    .. todo::
    """
    def __init__(self, data, batch_size=None, nbatch=None,
                 start=0, end=None, shuffle=False, infinite_data=0,
                 pseudo_n=1000000):
        if (batch_size or nbatch) is None:
            raise ValueError("Either batch_size or nbatch should be given.")
        if (batch_size and nbatch) is not None:
            raise ValueError("Provide either batch_size or nbatch.")
        self.infinite_data = infinite_data
        if not infinite_data:
            self.start = start
            self.end = data.num_examples() if end is None else end
            if self.start >= self.end or self.start < 0:
                raise ValueError("Got wrong value for start %d." % self.start)
            self.nexp = self.end - self.start
            if nbatch is not None:
                self.batch_size = int(np.float(self.nexp / float(nbatch)))
                self.nbatch = nbatch
            elif batch_size is not None:
                self.batch_size = batch_size
                self.nbatch = int(np.float(self.nexp / float(batch_size)))
            self.shuffle = shuffle
        else:
            self.pseudo_n = pseudo_n
        self.data = data
        self.name = self.data.name

    def __iter__(self):
        if self.infinite_data:
            for i in xrange(self.pseudo_n):
                yield self.data.slices()
        else:
            if self.shuffle:
                self.data.shuffle()
            start = self.start
            end = self.end - self.end % self.batch_size
            for idx in xrange(start, end, self.batch_size):
                yield [self.data.slices(idx, idx + self.batch_size),
                        self.data.slices(idx + 1, idx + self.batch_size + 1)]


def build_rev_model(tparams, options, x, y, x_mask):
    xc = tensor.concatenate([x[:1, :, :], y], axis=0)
    xc_mask = tensor.concatenate([tensor.alloc(1, 1, x_mask.shape[1]), x_mask], axis=0)
    xr = xc[::-1]
    xr_mask = xc_mask[::-1]

    xr_emb = get_layer('ff')[1](tparams, xr, options, prefix='ff_in_lstm_r', activ='lrelu')
    (states_rev, _), updates_rev = get_layer(options['encoder'])[1](tparams, xr_emb, options, prefix='encoder_r', mask=xr_mask)
    out_lstm = get_layer('ff')[1](tparams, states_rev, options, prefix='ff_out_lstm_r', activ='linear')
    out_prev = get_layer('ff')[1](tparams, xr_emb, options, prefix='ff_out_prev_r', activ='linear')
    out = lrelu(out_lstm + out_prev)
    out_mus = get_layer('ff')[1](tparams, out, options, prefix='ff_out_mus_r', activ='linear')
    out_mu, out_logvar = out_mus[:, :, :options['dim_input']], out_mus[:, :, options['dim_input']:]

    out_mu = out_mu[:-1]
    out_logvar = out_logvar[:-1]
    targets = xr[1:]
    targets_mask = xr_mask[1:]
    states_rev = states_rev[:-1][::-1]

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

    states_gen, cells_gen, z, kld, rec_cost_rev = (
            rvals[0], rvals[1], rvals[2], rvals[3], rvals[4])
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


# initialize all parameters
def init_params(options):
    rng = options['rng']
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
    params = get_layer('ff')[0](options, params, prefix='pri_ff_1', nin=options['dim'] + options['dim_proj'], nout=options['dim_proj'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='pri_ff_2', nin=options['dim_proj'], nout=2 * options['dim_z'], ortho=False)
    # Inference network params
    params = get_layer('ff')[0](
            options, params, prefix='inf_ff_1', nin=2 * options['dim'] + options['dim_proj'],
            nout=options['dim_proj'], ortho=False)
    params = get_layer('ff')[0](
            options, params, prefix='inf_ff_2', nin=options['dim_proj'],
            nout=2 * options['dim_z'], ortho=False)
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
    U = numpy.concatenate([
        norm_weight(rng, options['dim_z'], options['dim']),
        norm_weight(rng, options['dim_z'], options['dim']),
        norm_weight(rng, options['dim_z'], options['dim']),
        norm_weight(rng, options['dim_z'], options['dim'])],
        axis=1)
    params[parname('z_cond', 'W')] = U
    return params


def ELBOcost(rec_cost, kld, kld_weight=1.):
    assert kld.ndim == 1
    assert rec_cost.ndim == 1
    return rec_cost + kld_weight * kld


def pred_probs(f_log_probs, options, data, source='valid'):
    rng = options['rng']
    rvals = []
    n_done = 0

    for data_ in data:
        x = data_[0][0]
        y = data_[1][0]
        x_mask = np.ones((x.shape[0], x.shape[1]), dtype='float32')
        n_done += x.shape[1]

        zmuv = rng.normal(loc=0.0, scale=1.0, size=(
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
          seed=1234,
          kl_start=0.2,
          kl_rate=0.0003):

    rng = numpy.random.RandomState(seed)
    carry_h0 = True
    desc = 'seed{:d}_aux-gen{}_aux-nll{}_aux-zh{}_klrate{}'.format(
        seed, weight_aux_gen, weight_aux_nll, str(use_h_in_aux), kl_rate)
    logs = '{}/{}_log.txt'.format(log_dir, desc)
    diag = '{}/{}_diag.pkl'.format(log_dir, desc)
    opts = '{}/{}_opts.pkl'.format(model_dir, desc)
    pars = '{}/{}_pars.pkl'.format(model_dir, desc)

    print("- logs file: {}".format(logs))
    print("- opts file: {}".format(opts))
    print("- pars file: {}".format(pars))
    print("- diag file: {}".format(diag))

    # Model options
    model_options = locals().copy()
    pkl.dump(model_options, open(opts, 'wb'))
    log_file = open(logs, 'w')

    # save diagnostics into a pkl
    diags = {
        'train_costs': [[], [], [], [], [], [], []],
        'valid_elbo': [],
        'test_elbo': [],
    }

    x_dim = 200
    file_name = 'blizzard_unseg_tbptt'
    normal_params = np.load(data_dir + file_name + '_normal.npz')
    X_mean = normal_params['X_mean']
    X_std = normal_params['X_std']
    train_data = Blizzard_tbptt(name='train',
                                path=data_dir,
                                frame_size=x_dim,
                                file_name=file_name,
                                X_mean=X_mean,
                                X_std=X_std)

    valid_data = Blizzard_tbptt(name='valid',
                                path=data_dir,
                                frame_size=x_dim,
                                file_name=file_name,
                                X_mean=X_mean,
                                X_std=X_std)

    test_data = Blizzard_tbptt(name='test',
                               path=data_dir,
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
    reset_state = tensor.scalar('reset_state')

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

    print('- Building update init state...')
    init_state_shared = tparams[parname('encoder', 'init_state')]
    reset_updates = [(init_state_shared, init_state_shared * np.float32(0.))]
    f_reset_states = theano.function([], [], updates=reset_updates)

    print('- Building gradient...')
    grads = tensor.grad(tot_cost, itemlist(tparams))
    all_grads, non_finite, clipped = gradient_clipping(grads, tparams, 100.)
    # update function
    all_gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
                   for k, p in tparams.iteritems()]
    all_gsup = [(gs, g) for gs, g in zip(all_gshared, all_grads)]
    # forward pass + gradients
    outputs = [vae_cost, aux_cost, tot_cost, kld_cost, elbo_cost,
               nll_rev_cost, nll_gen_cost, non_finite]
    print('- Building f_prop...')
    all_updates = all_gsup + updates_gen.items() + updates_rev.items()
    f_prop = theano.function(inps, outputs, updates=all_gsup)
    print('- Building f_update...')
    f_update = eval(optimizer)(lr, tparams, all_gshared)

    history_errs = [c for c in diags['valid_elbo']]
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
        f_reset_states()

        for data_ in train_d_:
            x = data_[0][0]
            y = data_[1][0]
            x_mask = np.ones((x.shape[0], x.shape[1]), dtype='float32')

            n_samples += x.shape[1]
            uidx += 1
            kl_start = min(1., kl_start + kl_rate)

            # build samples for the reparametrization trick
            zmuv = rng.normal(loc=0.0, scale=1.0, size=(x.shape[0], x.shape[1], model_options['dim_z'])).astype('float32')
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
                diags['train_costs'][n].append(np.mean(tr_costs[n]))

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                checkpoint = time.time()
                str1 = 'Epoch {:d}  Update {:d}  VaeCost {:.2f}  AuxCost {:.2f}  KldCost {:.2f} ' \
                        'TotCost {:.2f}  ElboCost {:.2f}  NllRev {:.2f}  NllGen {:.2f} ' \
                        'KL_start {:.2f}  Speed {:.2f}it/s  PROGRESS: {:2.2f}'.format(
                                eidx, uidx, np.mean(tr_costs[0]), np.mean(tr_costs[1]), np.mean(tr_costs[3]),
                                np.mean(tr_costs[2]), np.mean(tr_costs[4]), np.mean(tr_costs[5]), np.mean(tr_costs[6]),
                                kl_start, float(uidx) / (checkpoint - start), (float(eidx + 1) / max_epochs) * 100.)
                print(str1)
                log_file.write(str1 + '\n')
                log_file.flush()

        print('Starting validation...')
        f_reset_states()
        valid_err = pred_probs(f_log_probs, model_options, valid_d_, source='valid')
        f_reset_states()
        test_err = pred_probs(f_log_probs, model_options, test_d_, source='test')
        history_errs.append(valid_err)

        diags['valid_elbo'].append(valid_err)
        diags['test_elbo'].append(test_err)

        # save diags
        diagf = open(diag, "wb")
        pkl.dump(diags, diagf)
        diagf.close()

        # decay learning rate if validation error increases
        if (old_valid_err < valid_err) and lrate > 0.0001:
            lrate = lrate / 2.0

        old_valid_err = history_errs[-1]
        str1 = 'Valid/Test ELBO: {:.2f}, {:.2f}'.format(valid_err, test_err)
        log_file.write(str1 + '\n')
        print(str1)

        # finish after this many updates
        if uidx >= finish_after:
            print('Finishing after %d iterations!' % uidx)
            break

    return valid_err


if __name__ == '__main__':
    pass
