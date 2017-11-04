'''
Build a simple neural language model using GRU units
'''
from __future__ import print_function

import sys
import argparse
import numpy as np
import cPickle as pkl
from ipdb import set_trace as dbg

import theano
import theano.tensor as T
from lm_data import IMDB_JMARS
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lm_lstm_imdb import (init_params, init_tparams, load_params,
        is_train, build_rev_model, build_gen_model,
        build_sampler, gen_sample, beam_sample)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_prefix", help="Model path")
    parser.add_argument("--seqlen", type=int, default=50,
                        help="Sequence length. Default: %(default)s")
    parser.add_argument("--nb-samples", type=int, default=10,
                        help="Number of samples. Default: %(default)s")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Seed for the random generator. Default: always different")
    parser.add_argument("--show-real-data", action="store_true",
                        help="Show real data from validset instead sampling.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    trng = RandomStreams(args.seed)
    rng = np.random.RandomState(args.seed + 1)
    model_file = args.model_prefix + "_pars.npz"
    model_opts = args.model_prefix + "_opts.pkl"
    model_options = pkl.load(open(model_opts, 'rb'))

    # Load data
    data = IMDB_JMARS("./experiments/data", seq_len=16,
                      batch_size=args.nb_samples, topk=16000)
    model_options["dim_input"] = data.voc_size

    for num, (x, y, x_mask) in enumerate(data.get_train_batch()):
        data.print_batch(x)
        if num == 1:
            break

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
    # Build sampler
    f_next = build_sampler(tparams, model_options, trng, provide_z=True)
    # Build inference
    get_latents = theano.function([x, y, x_mask, zmuv], [z, log_pxIz, log_pz, log_qzIx],
            updates=(updates_gen + updates_rev),
            givens={is_train: np.float32(0.)})

    def _get_iwae_latents(batch, nrep=50):
        batch_rep = [np.repeat(x, nrep, 0).T for x in batch]
        zmuv = rng.normal(loc=0.0, scale=1.0, size=(
            batch_rep[0].shape[0], nrep, model_options['dim_z'])).astype('float32')
        z, log_pxIz, log_pz, log_qzIx = \
                get_latents(batch_rep[0], batch_rep[1], batch_rep[2], zmuv)
        log_ws = np.ravel(log_pxIz - log_qzIx + log_pz)
        log_ws_minus_max = log_ws - np.max(log_ws)
        ws = np.exp(log_ws_minus_max)
        ws_norm = ws / np.sum(ws)
        max_z = np.argmax(ws_norm)
        return z[:, [max_z], :]

    s1_id = [data.word2idx.get(word, data.unk_id) for word in 'the film is great'.split()]
    batch = data.prepare_batch([s1_id])
    _get_iwae_latents(batch)

    while True:
        s1 = raw_input("s1:").strip().split()
        s2 = raw_input("s2:").strip().split()

        s1_id = [data.word2idx.get(word, data.unk_id) for word in s1]
        s2_id = [data.word2idx.get(word, data.unk_id) for word in s2]

        batch = data.prepare_batch(data.pad_sent(s1_id))
        z1 = _get_iwae_latents(batch, nrep=200)
        batch = data.prepare_batch(data.pad_sent(s2_id))
        z2 = _get_iwae_latents(batch, nrep=200)

        # Interpolation
        print("Samples")
        for i in np.linspace(0, 1, 11):
            print("{}: ".format(i), end="")
            z = ((1 - i) * z1) + (i * z2)  # Interpolate latent
            z = np.repeat(z, 20, axis=1)
            sample, sample_score = gen_sample(
                    tparams, f_next, model_options,
                    maxlen=20, argmax=False, zmuv=z,
                    unk_id=data.unk_id, eos_id=data.eos_id, bos_id=data.bos_id)
            sample = [sample.T[np.argsort(sample_score)[-1]]]
            data.print_batch(sample, eos_id=data.eos_id, print_number=False)

        print("Argmax")
        for i in np.linspace(0, 1, 11):
            print("{}: ".format(i), end="")
            z = ((1 - i) * z1) + (i * z2)  # Interpolate latent
            sample, sample_score = gen_sample(
                tparams, f_next, model_options,
                maxlen=20, argmax=True, zmuv=z,
                unk_id=data.unk_id, eos_id=data.eos_id, bos_id=data.bos_id)
            data.print_batch(sample.T, eos_id=data.eos_id, print_number=False)
        raw_input("-- Next --")


if __name__ == '__main__':
    main()
