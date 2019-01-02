# -*- coding:utf-8 -*-
import mxnet as mx

from model.model_config import get_backbones
from mxnet import nd
from model.astgcn import ASTGCN
ctx = mx.cpu()
all_backbones = get_backbones('configurations/PEMS04.conf', 'data/PEMS04/distance.csv', ctx)

net = ASTGCN(12, all_backbones)
net.initialize(ctx = mx.cpu())
test_w = nd.random_uniform(shape = (16, 307, 3, 12))
test_d = nd.random_uniform(shape = (16, 307, 3, 12))
test_r = nd.random_uniform(shape = (16, 307, 3, 36))
print(net([test_w, test_d, test_r]).shape)