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
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


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
    rng = np.random.RandomState(args.seed+1)
    model_file = args.model_prefix + "_pars.npz"
    model_opts = args.model_prefix + "_opts.pkl"
    model_options = pkl.load(open(model_opts, 'rb'))

    # Load data
    from lm_data import IMDB_JMARS
    data = IMDB_JMARS("./experiments/data", seq_len=16, batch_size=args.nb_samples, topk=16000)
    model_options["dim_input"] = data.voc_size

    if args.show_real_data:
        for x, y, x_mask in data.get_valid_batch():
            data.print_batch(x)
            raw_input("-- More --")
        sys.exit(0)

    from lm_lstm_imdb import init_params, init_tparams, load_params
    from lm_lstm_imdb import is_train, build_rev_model, build_gen_model, build_sampler, gen_sample, beam_sample
    params = init_params(model_options)
    print('Loading model parameters...')
    params = load_params(model_file, params)
    tparams = init_tparams(params)

    x = T.lmatrix('x')
    y = T.lmatrix('y')
    x_mask = T.matrix('x_mask')
    # Debug test_value
    x.tag.test_value = np.random.rand(11, 20).astype("int64")
    y.tag.test_value = np.random.rand(11, 20).astype("int64")
    x_mask.tag.test_value = np.ones((11, 20)).astype("float32")
    is_train.tag.test_value = np.float32(0.)

    zmuv = T.tensor3('zmuv')
    zmuv.tag.test_value = np.ones((11, 20, model_options['dim_z'])).astype("float32")

    # build the symbolic computational graph
    nll_rev, states_rev, updates_rev = \
        build_rev_model(tparams, model_options, x, y, x_mask)
    nll_gen, states_gen, kld, rec_cost_rev, updates_gen, \
        log_pxIz, log_pz, log_qzIx, z, _ = \
        build_gen_model(tparams, model_options, x, y, x_mask, zmuv, states_rev)
    # Build sampler
    f_next = build_sampler(tparams, model_options, trng, provide_z=True)
    # Build inference
    get_latents = theano.function([x, y, x_mask, zmuv], z,
        updates=(updates_gen + updates_rev),
        givens={is_train: np.float32(0.)})

    indices = np.arange(len(data.va_words))
    while True:
        s1 = raw_input("s1:").strip().split()
        s2 = raw_input("s2:").strip().split()

        s1_id = [data.word2idx.get(x, data.unk_id) for x in s1]
        s2_id = [data.word2idx.get(x, data.unk_id) for x in s2]

        batch = data.prepare_batch([s1_id, s2_id])
        data.print_batch(batch[0])

        zmuv = rng.normal(loc=0.0, scale=1.0, size=(
            batch[0].shape[1], 2, model_options['dim_z'])).astype('float32')
        batch_z = get_latents(batch[0].T, batch[1].T, batch[2].T, zmuv)
        z1 = batch_z[:, [0], :]
        z2 = batch_z[:, [1], :]

        print("Beam Search")
        data.print_batch(batch[0][[0]], eos_id=data.eos_id, print_number=False)
        for i in np.linspace(0, 1, 11):
            print("{}: ".format(i), end="")
            z = ((1 - i) * z1) + (i * z2)  # Interpolate latent
            z = np.repeat(z, 10, axis=1)
            sample, sample_score = beam_sample(tparams, f_next, model_options,
                maxlen=20, zmuv=z, unk_id=data.unk_id,
                eos_id=data.eos_id, bos_id=data.bos_id)
            sample = [sample[0]]
            data.print_batch(sample, eos_id=data.eos_id, print_number=False)

        data.print_batch(batch[0][[1]], eos_id=data.eos_id, print_number=False)

        # Interpolation
        print("Samples")
        data.print_batch(batch[0][[0]], eos_id=data.eos_id, print_number=False)
        for i in np.linspace(0, 1, 11):
            print("{}: ".format(i), end="")
            z = ((1 - i) * z1) + (i * z2)  # Interpolate latent
            z = np.repeat(z, 10, axis=1)
            sample, sample_score = gen_sample(tparams, f_next, model_options,
                    maxlen=20, argmax=False, zmuv=z,
                    unk_id=data.unk_id, eos_id=data.eos_id, bos_id=data.bos_id)
            sample = [sample.T[np.argsort(sample_score)[-1]]]
            data.print_batch(sample, eos_id=data.eos_id, print_number=False)

        data.print_batch(batch[0][[1]], eos_id=data.eos_id, print_number=False)

        print("Argmax")
        data.print_batch(batch[0][[0]], eos_id=data.eos_id, print_number=False)
        for i in np.linspace(0, 1, 11):
            print("{}: ".format(i), end="")
            z = ((1 - i) * z1) + (i * z2)  # Interpolate latent
            sample, sample_score = gen_sample(tparams, f_next, model_options,
                maxlen=20, argmax=True, zmuv=z,
                unk_id=data.unk_id, eos_id=data.eos_id, bos_id=data.bos_id)
            data.print_batch(sample.T, eos_id=data.eos_id, print_number=False)
        data.print_batch(batch[0][[1]], eos_id=data.eos_id, print_number=False)
        raw_input("-- Next --")
        sys.exit(0)


if __name__ == '__main__':
    main()

