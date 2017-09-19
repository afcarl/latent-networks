#!/usr/bin/env python

from lm_lstm_mnist import train


def main(job_id, params):
    print params
    validerr = train(
        saveto=params['model'],
        reload_=params['reload'],
        dim_word=params['dim_word'],
        dim=params['dim'],
        n_words=params['n-words'],
        decay_c=params['decay-c'],
        weight_aux=params['weight_aux'],
        lrate=params['learning-rate'],
        optimizer=params['optimizer'],
        maxlen=784,
        batch_size=32,
        valid_batch_size=32,
        validFreq=5000,
        dispFreq=10,
        saveFreq=1000,
        sampleFreq=1000,
        kl_start=params['kl_start'],
        kl_rate=0.00005,
        use_dropout=params['use-dropout'][0])
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': '/data/lisatmp4/anirudhg/latent_autoregressive/models/model_len30_0.05_kl.npz',
        'dim_word': 512,
        'dim': 1024,
        'n-words': 50,
        'optimizer': 'adam',
        'decay-c': 0.,
        'kl_start': 0.2,
        'weight_aux': 0.,
        'use-dropout': False,
        'learning-rate': 0.0001,
        'reload': False})
