#!/usr/bin/env python

import os
import argparse
import pprint
from lm_lstm_timit import train


def main(job_id, params):
    print("Parameters:")
    pprint.pprint(params)
    validerr = train(
        data_dir=params['data_dir'],
        model_dir=params['model_dir'],
        log_dir=params['log_dir'],
        reload_=params['reload'],
        dim_input=params['dim_input'],
        dim=params['dim'],
        decay_c=params['decay_c'],
        lrate=params['learning_rate'],
        optimizer=params['optimizer'],
        dim_proj=params['dim_proj'],
        batch_size=32,
        valid_batch_size=32,
        dispFreq=10,
        seed=params['seed'],
        weight_aux_gen=params['weight_aux_gen'],
        weight_aux_nll=params['weight_aux_nll'],
        use_dropout=params['use_dropout'],
        use_h_in_aux=params['use_h_in_aux'],
        dim_z=params['dim_z'],
        kl_start=params['kl_start'],
        kl_rate=params['kl_rate'])
    return validerr


if __name__ == '__main__':
    try:
        # Created experiments folder, if needed.
        os.makedirs("./experiments/timit/")
    except:
        pass

    #
    parser = argparse.ArgumentParser("TIMIT experiments for VRNN with auxiliary costs.")
    parser.add_argument('--philly_datadir', type=str, default='./experiments/data',
                        nargs='?', help='path of the input data directory (HDFS)')
    parser.add_argument('--philly_logdir', type=str, default='./experiments/timit',
                        nargs='?', help='path of the log directory (NFS)')
    parser.add_argument('--philly_modeldir', type=str, default='./experiments/timit',
                        help='path of the output directory (HDFS)')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--weight_aux_gen', type=float, default=0.)
    parser.add_argument('--weight_aux_nll', type=float, default=0.)
    parser.add_argument('--use_h_in_aux', action='store_true')
    args = parser.parse_args()

    main(0, {
        'dim_input': 200,
        'dim': 1024,
        'dim_proj': 512,
        'optimizer': 'adam',
        'seed': args.seed,
        'decay_c': 0.,
        'data_dir': args.philly_datadir,
        'log_dir': args.philly_logdir,
        'model_dir': args.philly_modeldir,
        'use_h_in_aux': args.use_h_in_aux,
        'weight_aux_gen': args.weight_aux_gen,
        'weight_aux_nll': args.weight_aux_nll,
        'use_dropout': False,
        'kl_start': 0.2,
        'kl_rate': 0.00005,
        'dim_z': 256,
        'learning_rate': 0.001,
        'reload': False})
