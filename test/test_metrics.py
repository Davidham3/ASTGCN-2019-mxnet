# -*- coding:utf-8 -*-

import sys
import pytest
import numpy as np

sys.path.append('.')


def test_mape():
    from lib.metrics import masked_mape_np
    a = np.random.uniform(size=(2, 2))
    b = np.random.uniform(size=(2, 2))
    mape = masked_mape_np(a, b, 0)
    assert type(mape) == np.float64


def test_mse():
    from lib.metrics import mean_squared_error
    a = np.random.uniform(size=(2, 2))
    b = np.random.uniform(size=(2, 2))
    mse = mean_squared_error(a, b)
    assert type(mse) == np.float64


def test_mae():
    from lib.metrics import mean_absolute_error
    a = np.random.uniform(size=(2, 2))
    b = np.random.uniform(size=(2, 2))
    mae = mean_absolute_error(a, b)
    assert type(mae) == np.float64
