import numpy as np
import numpy.random as npr
import scipy
import scipy.misc

nprs = npr.RandomState(1234)


def shuffle_arr(*arrays):
    # shuffle each array in the list while keeping
    # the relative order of elements across the arrays
    indices = range(arrays[0].shape[0])
    nprs.shuffle(indices)
    return [A[indices] for A in arrays]


def load_binarized_mnist(data_path):
    # binarized_mnist_test.amat  binarized_mnist_train.amat  binarized_mnist_valid.amat
    print('loading binary MNIST, sampled version (de Larochelle)')
    train_x = np.loadtxt(data_path + '/binarized_mnist_train.amat').astype('int32')
    valid_x = np.loadtxt(data_path + '/binarized_mnist_valid.amat').astype('int32')
    test_x = np.loadtxt(data_path + '/binarized_mnist_test.amat').astype('int32')
    # shuffle dataset
    train_x = shuffle_arr(train_x)
    valid_x = shuffle_arr(valid_x)
    test_x = shuffle_arr(test_x)
    print('DONE.')
    return train_x[0], valid_x[0], test_x[0]


def get_mnist_iterator(data, batch_size):
    data = shuffle_arr(data)[0]
    for i in range(0, len(data), batch_size):
        batch = data[i: i + batch_size]
        batch = np.concatenate([np.zeros((batch_size, 1)), batch], axis=1).astype('int64')
        x_mask = np.ones((batch.shape[0], 784)).astype('float32')
        x = batch[:, :-1].T
        y = batch[:, 1:].T
        x_mask = x_mask.T
        yield x, y, x_mask
