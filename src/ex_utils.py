import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from scipy.stats import kstest


def batch_iterator(sample, train, test, batch_size):
    """Create iterator for iterating over train and test batches.

    Parameters
    ----------
    get_data : function
        Function that can be called to get batch_size amount of data with 
        get_data(batch_size=batch_size)
    train : np.ndarray
        Object of size [num train samples, dim of datapoint] to hold train data
    test : np.ndarray
        Object of size [num test samples, dim of datapoint] to hold test data
    batch_size : int

    Returns
    -------
    batch_iterators : list
    """
    num_train_batches = train.shape[0]//batch_size
    for i in range(num_train_batches):
        batch = sample(batch_size=batch_size, train=True)
        if len(batch.shape) == 1:
            batch = np.reshape(batch, (-1, 1))
        train[(i*batch_size):((i+1)*batch_size),:] = batch

    num_test_batches = test.shape[0]//batch_size
    for i in range(num_test_batches):
        batch = sample(batch_size=batch_size, train=False)
        if len(batch.shape) == 1:
            batch = np.reshape(batch, (-1, 1))
        test[(i*batch_size):((i+1)*batch_size),:] = batch

    batch_iterators = [batch_iterator_factory(a, batch_size) for a in [train, test]] 

    return batch_iterators


def batch_iterator_factory(x, batch_size):
    batch_idx = 0
    num_batches = x.shape[0]//batch_size

    def next_batch():
        nonlocal batch_idx, num_batches
        minibatch = x[batch_idx*batch_size:(batch_idx+1)*batch_size,:]
        batch_idx += 1
        if batch_idx == num_batches:
            batch_idx = 0
        return minibatch

    return next_batch


def sample_batch(sampler, batch_size):
    return sampler(batch_size)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def kolmogorov_smirnov(z_infer, z_true):
    ks = kstest(z_infer, z_true)
    return ks