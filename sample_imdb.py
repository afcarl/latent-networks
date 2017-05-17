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

    parser.add_argument("model", help="Model params (.npz)")
    parser.add_argument("options", help="Model params (.npz)")

    parser.add_argument("--seqlen", type=int, default=50,
                        help="Sequence length. Default: %(default)s")
    parser.add_argument("--nb-samples", type=int, default=10,
                        help="Number of samples. Default: %(default)s")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Seed for the random generator. Default: always different")

    parser.add_argument("--show-real-data", action="store_true",
                        help="Show real data from validset instead sampling.")

    parser.add_argument("--eval", action="store_true", help="Run evaluation.")
    parser.add_argument("--kickstart", action="store_true", help="Kickstart sequences with real data.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode.")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed+1)
    model_file = args.model
    opts = args.options
    model_options = pkl.load(open(opts, 'rb'))

    # Load data
    from lm_data import IMDB_JMARS
    data = IMDB_JMARS("./experiments/data", seq_len=16, batch_size=args.nb_samples, topk=16000)
    model_options["dim_input"] = data.voc_size

    if args.show_real_data:
        for x, y, x_mask in data.get_valid_batch():
            data.print_batch(x)
            raw_input("-- More --")

        sys.exit(0)

    print('Loading model')
    from lm_lstm_imdb import init_params, init_tparams, load_params
    params = init_params(model_options)
    params = load_params(model_file, params)
    tparams = init_tparams(params)

    if args.kickstart:
        trng = RandomStreams(args.seed)
        from lm_lstm_imdb import build_sampler, gen_sample
        f_next = build_sampler(tparams, model_options, trng)

        data.batch_size = 1
        for x, y, x_mask in data.get_valid_batch():
            x = x.transpose(1, 0).astype('int32')
            y = y.transpose(1, 0).astype('int32')
            x_mask = x_mask.transpose(1, 0).astype('float32')
            # x.shape : seq_len, batch_size
            half = len(x) // 2
            print("Ground truth: {}".format(data.batch2text(x.T)[0]))
            print("Half sentence: {}".format(data.batch2text(x[:half].T)[0]))

            zmuv = rng.normal(loc=0.0, scale=1.0,
                              size=(1, model_options['dim_z'])).astype('float32')
            zmuv = np.tile(zmuv, reps=(args.nb_samples, 1))

            print("Samples (fixed latent)")
            kickstart = np.tile(x[:half], reps=(1, args.nb_samples))
            sample, sample_score = gen_sample(tparams, f_next, model_options, maxlen=args.seqlen, argmax=False, kickstart=kickstart, zmuv=zmuv)
            #print("LL: {}".format(sample_score))
            data.print_batch(sample.T, eos_id=data.eos_id)

            print("Argmax (sample latent)")
            zmuv = rng.normal(loc=0.0, scale=1.0,
                              size=(args.nb_samples, model_options['dim_z'])).astype('float32')
            sample, sample_score = gen_sample(tparams, f_next, model_options, maxlen=args.seqlen, argmax=True, kickstart=kickstart, zmuv=zmuv)
            #print("LL: {}".format(sample_score))
            data.print_batch(sample.T, eos_id=data.eos_id)

            raw_input("-- More --")

        sys.exit(0)

    if args.eval:
        from lm_lstm_imdb import is_train, ELBOcost, build_rev_model, build_gen_model, pred_probs

        x = T.lmatrix('x')
        y = T.lmatrix('y')
        x_mask = T.matrix('x_mask')
        # Debug test_value
        x.tag.test_value = np.random.rand(11, 20).astype("int64")
        y.tag.test_value = np.random.rand(11, 20).astype("int64")
        x_mask.tag.test_value = np.ones((11, 20)).astype("float32")
        zmuv = T.tensor3('zmuv')

        # build the symbolic computational graph
        nll_rev, states_rev, updates_rev = \
            build_rev_model(tparams, model_options, x, y, x_mask)
        nll_gen, states_gen, kld, rec_cost_rev, updates_gen = \
            build_gen_model(tparams, model_options, x, y, x_mask, zmuv, states_rev)

        print('Building f_log_probs...')
        inps = [x, y, x_mask, zmuv]
        f_log_probs = theano.function(
            inps,
            ELBOcost(nll_gen, kld, kld_weight=1.),
            updates=(updates_gen + updates_rev),
            givens={is_train: np.float32(0.)}
        )

        print('Done')
        valid_err = pred_probs(f_log_probs, model_options, data, source='valid')
        print("Valid: {}".format(valid_err))
        test_err = pred_probs(f_log_probs, model_options, data, source='test')
        print("Test: {}".format(test_err))

    trng = RandomStreams(args.seed)
    from lm_lstm_imdb import build_sampler, gen_sample
    f_next = build_sampler(tparams, model_options, trng)

    unk_id = data.unk_id
    eos_id = data.eos_id
    while True:
        print("Samples (fixed latent)")
        zmuv = rng.normal(loc=0.0, scale=1.0, size=(1, args.seqlen, model_options['dim_z'])).astype('float32')
        zmuv = np.tile(zmuv, reps=(args.nb_samples, 1, 1))
        sample, sample_score = gen_sample(tparams, f_next, model_options, maxlen=args.seqlen, argmax=False, zmuv=zmuv,
                                          unk_id=unk_id, eos_id=eos_id)
        #print("LL: {}".format(sample_score))
        data.print_batch(sample.T, eos_id=data.eos_id)

        print("Argmax (sample latent)")
        zmuv = rng.normal(loc=0.0, scale=1.0, size=(args.nb_samples, args.seqlen, model_options['dim_z'])).astype('float32')
        sample, sample_score = gen_sample(tparams, f_next, model_options, maxlen=args.seqlen, argmax=True,
                                          zmuv=zmuv, unk_id=unk_id, eos_id=eos_id)
        #print("LL: {}".format(sample_score))
        data.print_batch(sample.T, eos_id=data.eos_id)

        raw_input("-- More --")


if __name__ == '__main__':
    main()
