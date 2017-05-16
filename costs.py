import numpy as np


def np_log_mean_exp(x, axis=None):
    assert (axis is not None), "please provide an axis..."
    m = np.max(x, axis=axis, keepdims=True)
    lme = m + np.log(np.mean(np.exp(x - m), axis=axis, keepdims=True))
    return lme


def iwae_multi_eval(x, y, x_mask, iters, cost_func, iwae_num, dim_z):
    # slow multi-pass evaluation of IWAE bound.
    rx = []
    # all the inputs are transpose, axis=1 is batch_size
    for elt in (x, y, x_mask):
        rx.append(np.repeat(elt, iwae_num, axis=1))
    log_p_xIz = []
    log_p_z = []
    log_q_zIx = []
    # produce multiple samples for the iwae bound
    for i in range(iters):
        x_ = rx[0]
        zmuv = np.random.normal(loc=0.0, scale=1.0, size=(
            x_.shape[0], x_.shape[1], dim_z)).astype('float32')
        inps = rx + [zmuv]
        result = cost_func(*inps)
        b_size = int(result[0].shape[0] / iwae_num)
        log_p_xIz.append(result[0].reshape((b_size, iwae_num)))
        log_p_z.append(result[1].reshape((b_size, iwae_num)))
        log_q_zIx.append(result[2].reshape((b_size, iwae_num)))
    # stack up results from multiple passes
    log_p_xIz = np.concatenate(log_p_xIz, axis=1)
    log_p_z = np.concatenate(log_p_z, axis=1)
    log_q_zIx = np.concatenate(log_q_zIx, axis=1)
    # compute the IWAE bound for each example in x
    log_ws_mat = log_p_xIz + log_p_z - log_q_zIx
    iwae_bounds = -1.0 * np_log_mean_exp(log_ws_mat, axis=1)
    return iwae_bounds
