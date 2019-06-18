# -*- coding:utf-8 -*-

import numpy as np


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def mean_absolute_error(y_true, y_pred):
    '''
    mean absolute error

    Parameters
    ----------
    y_true, y_pred: np.ndarray, shape is (batch_size, num_of_features)

    Returns
    ----------
    np.float64

    '''

    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    '''
    mean squared error

    Parameters
    ----------
    y_true, y_pred: np.ndarray, shape is (batch_size, num_of_features)

    Returns
    ----------
    np.float64

    '''
    return np.mean((y_true - y_pred) ** 2)
