# -*- coding:utf-8 -*-

import sys
import pytest
import numpy as np

sys.path.append('.')


def test_normalize():
    from lib import data_preparation
    train = np.random.uniform(size=(32, 12, 307, 3))
    val = np.random.uniform(size=(32, 12, 307, 3))
    test = np.random.uniform(size=(32, 12, 307, 3))
    stats, train_norm, val_norm, test_norm = data_preparation.normalization(
        train, val, test)
    assert train.shape == train_norm.shape
    assert val.shape == val_norm.shape
    assert test.shape == test_norm.shape
