# -*- coding:utf-8 -*-

import sys
import pytest
import numpy as np
from mxnet import nd

sys.path.append('.')


def test_ASTGCN_submodule():
    from model.astgcn import ASTGCN_submodule
    x = nd.random_uniform(shape=(32, 307, 3, 24))
    K = 3
    cheb_polynomials = [nd.random_uniform(shape=(307, 307)) for i in range(K)]
    backbone = [
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": 2,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]
    net = ASTGCN_submodule(12, backbone)
    net.initialize()
    output = net(x)
    assert output.shape == (32, 307, 12)
    assert type(output.mean().asscalar()) == np.float32


def test_predict1():
    from model.astgcn import ASTGCN
    from model.model_config import get_backbones
    import mxnet as mx
    ctx = mx.cpu()
    all_backbones = get_backbones('configurations/PEMS04.conf',
                                  'data/PEMS04/distance.csv', ctx)

    net = ASTGCN(12, all_backbones)
    net.initialize(ctx=ctx)
    test_w = nd.random_uniform(shape=(16, 307, 3, 12), ctx=ctx)
    test_d = nd.random_uniform(shape=(16, 307, 3, 12), ctx=ctx)
    test_r = nd.random_uniform(shape=(16, 307, 3, 36), ctx=ctx)
    output = net([test_w, test_d, test_r])
    assert output.shape == (16, 307, 12)
    assert type(output.mean().asscalar()) == np.float32


def test_predict2():
    from model.astgcn import ASTGCN
    from model.model_config import get_backbones
    import mxnet as mx
    ctx = mx.cpu()
    all_backbones = get_backbones('configurations/PEMS08.conf',
                                  'data/PEMS08/distance.csv', ctx)

    net = ASTGCN(12, all_backbones)
    net.initialize(ctx=ctx)
    test_w = nd.random_uniform(shape=(8, 170, 3, 12), ctx=ctx)
    test_d = nd.random_uniform(shape=(8, 170, 3, 12), ctx=ctx)
    test_r = nd.random_uniform(shape=(8, 170, 3, 36), ctx=ctx)
    output = net([test_w, test_d, test_r])
    assert output.shape == (8, 170, 12)
    assert type(output.mean().asscalar()) == np.float32
