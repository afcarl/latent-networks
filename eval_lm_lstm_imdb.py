'''
Build a simple neural language model using GRU units
'''

import argparse
import numpy as np
import os
import theano
import theano.tensor as T
import theano.tensor as tensor
from lm_data import IMDB_JMARS
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lm_lstm_imdb import (init_params, init_tparams, load_params,
        is_train, build_rev_model, build_gen_model,
        build_sampler, gen_sample, beam_sample, ELBOcost)

import cPickle as pkl
import numpy
import copy
from costs import iwae_multi_eval
from tqdm import tqdm
import warnings
import time
from collections import OrderedDict

profile = False
seed = 1234
num_iwae_samps = 25
num_iwae_iters = 1
num_iwae_samps_train = 5
numpy.random.seed(seed)


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


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_prefix", help="Model path")
    return parser

def eval():
    parser = build_parser()
    args = parser.parse_args()

    model_file = args.model_prefix + "_pars.npz"
    model_opts = args.model_prefix + "_opts.pkl"
    model_options = pkl.load(open(model_opts, 'rb'))

    # Load data
    data = IMDB_JMARS("./experiments/data", seq_len=16,
                      batch_size=50, topk=16000)
    model_options["dim_input"] = data.voc_size

    params = init_params(model_options)
    print('Loading model parameters...')
    params = load_params(model_file, params)
    tparams = init_tparams(params)

    x = T.lmatrix('x')
    y = T.lmatrix('y')
    x_mask = T.matrix('x_mask')
    zmuv = T.tensor3('zmuv')

    # build the symbolic computational graph
    nll_rev, states_rev, updates_rev = \
        build_rev_model(tparams, model_options, x, y, x_mask)
    nll_gen, states_gen, kld, rec_cost_rev, updates_gen, \
        log_pxIz, log_pz, log_qzIx, z, _ = \
        build_gen_model(tparams, model_options, x, y, x_mask, zmuv, states_rev)

    inps = [x, y, x_mask, zmuv]
    f_log_probs = theano.function(
       inps, ELBOcost(nll_gen, kld, kld_weight=1.),
       updates=(updates_gen + updates_rev), profile=profile,
       givens={is_train: numpy.float32(0.)})
    f_iwae_eval = theano.function(
       inps, [log_pxIz, log_pz, log_qzIx],
       updates=(updates_gen + updates_rev),
       givens={is_train: numpy.float32(0.)})
    print('Done')

    print('Starting validation...')
    train_err = pred_probs(f_log_probs, f_iwae_eval, model_options, data, source='train')
    print('Train ELBO: {:.2f}, IWAE: {:.2f}'.format(train_err[0], train_err[1]))
    valid_err = pred_probs(f_log_probs, f_iwae_eval, model_options, data, source='valid')
    print('Valid ELBO: {:.2f}, IWAE: {:.2f}'.format(valid_err[0], valid_err[1]))
    test_err = pred_probs(f_log_probs, f_iwae_eval, model_options, data, source='test')
    print('Test ELBO:  {:.2f}, IWAE: {:.2f}'.format(test_err[0], test_err[1]))


if __name__ == '__main__':
    eval()
