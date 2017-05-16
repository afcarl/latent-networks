'''
Build a simple neural language model using GRU units
'''
from __future__ import print_function

import sys
import argparse
import numpy as np
import cPickle as pkl
#from ipdb import set_trace as dbg
#import matplotlib.pyplot as plt

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("model", help="Model params (.npz)")

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
import ipdb

def get_interpolation(a1, a2, alpha):
    return alpha*a1 + (1-alpha)*a2

def main():
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed+1)
    model_file = args.model
    opts = model_file[:-len("_pars.npz")] + "_opts.pkl"
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

    x = T.lmatrix('x')
    y = T.lmatrix('y')
    x_mask = T.matrix('x_mask')
    zmuv = T.tensor3('zmuv')
    #is_train = T.scalar('is_train')

    if args.kickstart:
        trng = RandomStreams(args.seed)
        from lm_lstm_imdb import is_train, build_sampler, gen_sample,build_rev_model, build_interpolate_model
        nll_rev, states_rev, updates_rev, get_states_rev =\
            build_rev_model(tparams, model_options, x, y, x_mask)

        f_next = build_sampler(tparams, model_options, trng)
        out_probs, states_gen, updates_gen, z = \
            build_interpolate_model(tparams, model_options, x, x_mask, zmuv, states_rev, trng)

        get_latent_states = theano.function([x, x_mask, zmuv, states_rev],[z],
                                             updates=(updates_gen),
                                             givens={is_train: np.float32(0.)})
        get_new_states = theano.function([x, x_mask, zmuv, states_rev, z], [out_probs],
                                             updates=(updates_gen),on_unused_input='ignore')

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
    from lm_lstm_imdb import is_train, build_sampler, gen_sample,build_rev_model, build_interpolate_model
    f_next = build_sampler(tparams, model_options, trng)
    nll_rev, states_rev, updates_rev, get_states_rev =\
        build_rev_model(tparams, model_options, x, y, x_mask)

    f_next = build_sampler(tparams, model_options, trng)
    out_probs, states_gen, updates_gen, z = \
        build_interpolate_model(tparams, model_options, x, x_mask, zmuv, states_rev, trng)

    get_latent_states = theano.function([x, x_mask, zmuv, states_rev],[z],
                                         updates=(updates_gen),
                                         givens={is_train: np.float32(0.)})
    get_new_states = theano.function([x, x_mask, zmuv, states_rev, z], [out_probs],
                                         updates=(updates_gen),on_unused_input='ignore',
                                         givens={is_train: np.float32(0.)})


    for x, y, x_mask in data.get_valid_batch():
        x = x.transpose(1, 0).astype('int32')
        y = y.transpose(1, 0).astype('int32')
        x_mask = x_mask.transpose(1, 0).astype('float32')
        zmuv_2 = np.random.normal(loc=0.0, scale=1.0, size=(x.shape[0], x.shape[1], model_options['dim_z'])).astype('float32')
        states_reversed = get_states_rev(x, y, x_mask)
        z_1 = get_latent_states(x, x_mask, zmuv_2, states_reversed[0])

        z1 = z_1[0][:,0,:]
        z2 = z_1[0][:,1,:]

        int_z = []

        coef = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for i in range(len(coef)):
            int_z.append(get_interpolation(z1, z2, coef[i]))
        int_z = np.asarray(int_z)
        #print args.nb_samples

        print(int_z.shape)
        outprobs = get_new_states(x, x_mask, zmuv_2, states_reversed[0], int_z)
        output_ = np.zeros((outprobs[0].shape[0], outprobs[0].shape[1]))
        for i in range(output_.shape[0]):
            for j in range(output_.shape[1]):
                out_ = np.where(outprobs[0][i][j][:] ==1)
                output_[i][j] = out_[0][0]
        print("Printing something", outprobs[0].shape)
        output_ = output_.astype('int32')
        ipdb.set_trace()
        data.print_batch(output_, eos_id=data.eos_id)
        raw_input("-- More --")
    '''
    while True:
        zmuv = rng.normal(loc=0.0, scale=1.0,
                          size=(1, model_options['dim_z'])).astype('float32')
        zmuv = np.tile(zmuv, reps=(args.nb_samples, 1))



        print("Samples (fixed latent)")
        sample, sample_score = gen_sample(tparams, f_next, model_options, maxlen=args.seqlen, argmax=False, zmuv=zmuv)
        #print("LL: {}".format(sample_score))
        data.print_batch(sample.T, eos_id=data.eos_id)

        print("Argmax (sample latent)")
        zmuv = rng.normal(loc=0.0, scale=1.0,
                          size=(args.nb_samples, model_options['dim_z'])).astype('float32')
        sample, sample_score = gen_sample(tparams, f_next, model_options, maxlen=args.seqlen, argmax=True, zmuv=zmuv)
        #print("LL: {}".format(sample_score))
        data.print_batch(sample.T, eos_id=data.eos_id)

        raw_input("-- More --")
    '''

if __name__ == '__main__':
    main()
