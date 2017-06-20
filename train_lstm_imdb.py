#!/usr/bin/env python

import os
from lm_lstm_imdb import train


def main(job_id, params):
    print(params)
    validerr = train(
        saveto=params['model'],
        reload_=params['reload'],
        dim_input=params['dim_input'],
        dim=params['dim'],
        decay_c=params['decay-c'],
        lrate=params['learning-rate'],
        optimizer=params['optimizer'],
        dim_proj=params['dim_proj'],
        weight_aux=params['weight_aux'],
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
    try:
        # Created experiments folder, if needed.
        os.makedirs("./experiments/imdb/")
    except:
        pass

    main(0, {
        'model': './experiments/imdb/',
        'dim_input': -1,  # Determine but the dataset.
        'dim': 500,
        'dim_proj': 300,
        'optimizer': 'adam',
        'decay-c': 0.,
        'kl_start': 1.,
        'weight_aux': 0.0005,
        'dropout': 0.2,
        'learning-rate': 0.001,
        'reload': False})
