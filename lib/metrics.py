# -*- coding:utf-8 -*-

import numpy as np

def masked_mape_np(labels, preds, null_val = np.nan):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100
