#!/usr/bin/env python

import os
from lm_lstm_imdb import train
import argparse


def main(job_id, params):
    print(params)
    validerr = train(
        model_dir=params['model_dir'],
        data_dir=params['data_dir'],
        log_dir=params['log_dir'],
        reload_=params['reload'],
        dim_input=params['dim_input'],
        dim=params['dim'],
        lrate=params['learning-rate'],
        optimizer=params['optimizer'],
        dim_proj=params['dim_proj'],
        weight_aux=params['weight_aux'],
        use_iwae=params['use_iwae'],
        batch_size=32,
        valid_batch_size=32,
        dispFreq=10,
        saveFreq=1000,
        sampleFreq=1000,
        dataset=None,
        valid_dataset=None,
        dictionary=None,
        dropout=params['dropout'],
        kl_start=params['kl_start'],
        kl_rate=0.0001)
    return validerr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #
    parser.add_argument('--philly_datadir', type=str, default='./experiments/data', nargs='?', help='path of the input data directory (HDFS)')
    parser.add_argument('--philly_logdir', type=str, default='./experiments/imdb', nargs='?', help='path of the log directory (NFS)')
    parser.add_argument('--philly_modeldir', type=str, default='./experiments/imdb', help='path of the output directory (HDFS)')
    #
    parser.add_argument('--weight_aux', type=float, default=0.)
    parser.add_argument('--use_iwae', action='store_true')
    args = parser.parse_args()

    main(0, {
        'model_dir': args.philly_modeldir,
        'log_dir': args.philly_logdir,
        'data_dir': args.philly_datadir,
        'use_iwae': args.use_iwae,
        'weight_aux': args.weight_aux,
        'dim_input': -1,
        'dim': 500,
        'dim_proj': 300,
        'optimizer': 'adam',
        'kl_start': 1.,
        'dropout': 0.,
        'learning-rate': 0.001,
        'reload': False
    })
