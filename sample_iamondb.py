'''
Build a simple neural language model using GRU units
'''
from __future__ import print_function

import sys
import argparse
import numpy as np
import cPickle as pkl
from ipdb import set_trace as dbg

import matplotlib.pyplot as plt

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("model", help="Model params (.npz)")
    parser.add_argument("opts", help="Model options (.pkl)")

    parser.add_argument("--seqlen", type=int, default=500,
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
    opts = args.opts
    model_options = pkl.load(open(opts, 'rb'))

    # Load data
    data_path = './datasets/iamondb/'
    from iamondb import IAMOnDB
    iamondb = IAMOnDB(name='train',
                      prep='normalize',
                      cond=False,
                      path=data_path)

    X_mean = iamondb.X_mean
    X_std = iamondb.X_std


    print('Loading model')
    from lm_lstm_iamondb import init_params, init_tparams, load_params
    params = init_params(model_options)
    params = load_params(model_file, params)
    tparams = init_tparams(params)

    if args.show_real_data:
        iamondb_valid = IAMOnDB(name='valid',
                          prep='normalize',
                          cond=False,
                          path=data_path,
                          X_mean=X_mean, X_std=X_std)

        dset_size = len(iamondb_valid.data[0])
        for start in range(0, dset_size, 1):
            end = start + 10
            # x.shape : seq_len, batch_size, input_dim
            x, x_mask = iamondb_valid.slices(start, end)

            from iamondb_utils import plot_lines_iamondb_example
            #print("NLL: {}".format(sample_score))

            plot_lines_iamondb_example(x, offsets_provided=True,
                                       mask=x_mask,
                                       mean=X_mean, std=X_std, colored=True,
                                       show=True)

    if args.kickstart:
        iamondb_valid = IAMOnDB(name='valid',
                          prep='normalize',
                          cond=False,
                          path=data_path,
                          X_mean=X_mean, X_std=X_std)


        trng = RandomStreams(args.seed)
        from lm_lstm_iamondb import build_sampler, gen_sample, build_rev_model, build_gen_model
        from iamondb_utils import plot_lines_iamondb_example
        f_next = build_sampler(tparams, model_options, trng)

        x = T.tensor3('x')
        y = T.tensor3('y')
        x_mask = T.matrix('x_mask')
        # Debug test_value
        x.tag.test_value = np.random.rand(11, 20, 3).astype("int64")
        y.tag.test_value = np.random.rand(11, 20, 3).astype("int64")
        x_mask.tag.test_value = np.ones((11, 20)).astype("float32")
        zmuv = T.tensor3('zmuv')
        zmuv.tag.test_value = np.ones((11, 20, model_options['dim_z'])).astype("float32")

        # build the symbolic computational graph
        nll_rev, states_rev, updates_rev = \
            build_rev_model(tparams, model_options, x, y, x_mask)
        nll_gen, states_gen, kld, rec_cost_rev, updates_gen, memories_gen = \
            build_gen_model(tparams, model_options, x, y, x_mask, zmuv, states_rev)

        # Build inference
        get_states = theano.function([x, y, x_mask, zmuv], [states_gen[-1], memories_gen[-1]],
                                      updates=(updates_gen + updates_rev))

        dset_size = len(iamondb_valid.data[0])
        for start in range(0, dset_size, 1):
            end = start + 1
            # x.shape : seq_len, batch_size, input_dim
            x, x_mask = iamondb_valid.slices(start, end)
            y = x
            x = np.concatenate([np.zeros_like(x[[0]]), x[:-1]], axis=0)

            seqlen = len(x)
            zmuv = rng.normal(loc=0.0, scale=1.0, size=(seqlen, 1, model_options['dim_z'])).astype('float32')
            states, memories = get_states(x, y, x_mask, zmuv)
            sample, sample_score = gen_sample(tparams, f_next, model_options, maxlen=args.seqlen, argmax=False,
                                              init_states=states, init_memories=memories)
            print("NLL: {}".format(sample_score))

            sample = np.concatenate([y[:, 0, :], sample[0]], axis=0)
            plot_lines_iamondb_example(sample, offsets_provided=True,
                                       mean=X_mean, std=X_std, colored=True,
                                       show=True)

    if args.eval:
        from lm_lstm_iamondb import ELBOcost, build_rev_model, build_gen_model, pred_probs

        iamondb_valid = IAMOnDB(name='valid',
                          prep='normalize',
                          cond=False,
                          path=data_path,
                          X_mean=X_mean, X_std=X_std)

        x = T.tensor3('x')
        y = T.tensor3('y')
        x_mask = T.matrix('x_mask')
        zmuv = T.tensor3('zmuv')
        weight_f = T.scalar('weight_f')

        # build the symbolic computational graph
        nll_rev, states_rev, updates_rev = \
            build_rev_model(tparams, model_options, x, y, x_mask)
        nll_gen, states_gen, kld, rec_cost_rev, updates_gen, _ = \
            build_gen_model(tparams, model_options, x, y, x_mask, zmuv, states_rev)

        print('Building f_log_probs...')
        inps = [x, y, x_mask, zmuv, weight_f]
        f_log_probs = theano.function(inps[:-1], ELBOcost(nll_gen, kld, kld_weight=1.),
                                      updates=(updates_gen + updates_rev), profile=False)
        print('Done')

        valid_err = pred_probs(f_log_probs, model_options, iamondb_valid, 20, source='valid')
        print("Valid: {}".format(valid_err))

    trng = RandomStreams(args.seed)
    from lm_lstm_iamondb import build_sampler, gen_sample
    from iamondb_utils import plot_lines_iamondb_example
    f_next = build_sampler(tparams, model_options, trng)
    samples = []
    for i in range(args.nb_samples):
        sample, sample_score = gen_sample(tparams, f_next, model_options, maxlen=args.seqlen, argmax=False)
        samples += [sample]
        print("NLL: {}".format(sample_score))

        plot_lines_iamondb_example(sample[0], offsets_provided=True,
                                   mean=X_mean, std=X_std, colored=True,
                                   show=True)

    # Print all of them
    samples = np.concatenate(samples, axis=0)
    plot_lines_iamondb_example(samples.transpose(1, 0, 2), offsets_provided=True,
                               mean=X_mean, std=X_std, colored=True,
                               show=True)

    dbg()


if __name__ == '__main__':
    main()
