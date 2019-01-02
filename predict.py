# -*- coding:utf-8 -*-

import mxnet as mx
from mxnet import gluon
from mxnet import nd
from model.model_config import get_backbones
from lib.utils import generate_x_y
from lib.utils import predict

import numpy as np
import os
import configparser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type = str, help = "configuration file path", required = True)
args = parser.parse_args()

# read configuration
config = configparser.ConfigParser()
config.read(args.config)
data_config = config['Data']
training_config = config['Training']
predict_config = config['Predict']

adj_filename                  = data_config['adj_filename']
graph_signal_matrix_filename  = data_config['graph_signal_matrix_filename']
num_of_vertices               = int(data_config['num_of_vertices'])
num_of_features               = int(data_config['num_of_features'])
points_per_hour               = int(data_config['points_per_hour'])
num_for_predict               = int(data_config['num_for_predict'])

model_name                    = training_config['model_name']
ctx                           = training_config['ctx']
batch_size                    = int(training_config['batch_size'])
num_of_weeks                  = int(training_config['num_of_weeks'])
num_of_days                   = int(training_config['num_of_days'])
num_of_hours                  = int(training_config['num_of_hours'])

params_file                   = predict_config['params_file']
data_file                     = predict_config['data_file']

# select devices
if ctx.startswith('cpu'):
    ctx = mx.cpu()
elif ctx.startswith('gpu'):
    ctx = mx.gpu(int(ctx.split('-')[1]))

# import model
print('model is %s'%(model_name))
if model_name == 'MSTGCN':
    from model.mstgcn import MSTGCN as model
elif model_name == 'ASTGCN':
    from model.astgcn import ASTGCN as model
else:
    raise SystemExit('Wrong type of model!')

# get model's structure
all_backbones = get_backbones(args.config, adj_filename, ctx)

# load parameters
print('loading parameters')
net = model(num_for_predict, all_backbones)
net.load_parameters(params_file, ctx = ctx)
print('model initialization finished!')

# load data and normalization statistics
transformer = np.load(os.path.join(os.path.split(params_file)[0], 'transformer_data.npz'))
data = np.load(data_file)['data']

def normalize(data, mean, std):
    norm = (data.reshape(data.shape[0], -1) - mean) / std
    return norm.reshape(*data.shape)

# generate data
test_week, test_day, test_recent, test_target = generate_x_y(data, num_of_weeks, num_of_days, num_of_hours, points_per_hour, num_for_predict)
print(test_week.shape, test_day.shape, test_recent.shape)

# normalization
test_week_norm = normalize(test_week, transformer['week_mean'], transformer['week_std'])
test_day_norm = normalize(test_day, transformer['day_mean'], transformer['day_std'])
test_recent_norm = normalize(test_recent, transformer['recent_mean'], transformer['recent_std'])
print(test_week_norm.shape, test_day_norm.shape, test_recent.shape)

# create data loader
data_loader = gluon.data.DataLoader(
                    gluon.data.ArrayDataset(
                        nd.array(test_week_norm, ctx = ctx),
                        nd.array(test_day_norm, ctx = ctx),
                        nd.array(test_recent_norm, ctx = ctx)
                    ),
                    batch_size = batch_size,
                    shuffle = False
                )

if 'prediction_filename' in predict_config:
    prediction_path = predict_config['prediction_filename']

    # predict
    loader_length = len(data_loader)
    prediction = []
    for index, (w, d, r) in enumerate(data_loader):
        # pylint: disable=no-member
        prediction.append(net([w, d, r]).asnumpy())
        print('predicting batch %s / %s'%(index + 1, loader_length))
    prediction = np.concatenate(prediction, 0)

    # save results
    np.savez_compressed(
        os.path.normpath(prediction_path), 
        prediction = prediction
    )