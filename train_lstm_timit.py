#!/usr/bin/env python

import os
import argparse
from lm_lstm_timit import train

def main(job_id, params):
    print(params)
    validerr = train(
        data_dir='experiments/data',
        model_dir='experiments/timit',
        log_dir='experiments/timit',
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
        saveFreq=1000,
        sampleFreq=1000,
        dataset=None,
        valid_dataset=None,
        dictionary=None,
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

    parser = argparse.ArgumentParser("TIMIT experiments for VRNN with auxiliary costs.")
    parser.add_argument('--weight_aux_gen', type=float, default=0.)
    parser.add_argument('--weight_aux_nll', type=float, default=0.)
    parser.add_argument('--use_h_in_aux', action='store_true')
    args = parser.parse_args()

    main(0, {
        'dim_input': 200,
        'dim': 1024,
        'dim_proj': 512,
        'optimizer': 'adam',
        'decay_c': 0.,
        'use_h_in_aux': args.use_h_in_aux,
        'weight_aux_gen': args.weight_aux_gen,
        'weight_aux_nll': args.weight_aux_nll,
        'use_dropout': False,
        'kl_start': 0.2,
        'kl_rate': 0.0003,
        'dim_z': 256,
        'learning_rate': 0.001,
        'reload': False})
