# -*- coding:utf-8 -*-

import os
import shutil
from time import time
from datetime import datetime
import configparser
import argparse

import numpy as np

import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxboard import SummaryWriter

from lib.utils import compute_val_loss, evaluate, predict
from lib.data_preparation import read_and_generate_dataset
from model.model_config import get_backbones

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str,
                    help="configuration file path", required=True)
parser.add_argument("--force", type=str, default=False,
                    help="remove params dir", required=False)
args = parser.parse_args()

# mxboard log dir
if os.path.exists('logs'):
    shutil.rmtree('logs')
    print('Remove log dir')

# read configuration
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])

model_name = training_config['model_name']
ctx = training_config['ctx']
optimizer = training_config['optimizer']
learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
merge = bool(int(training_config['merge']))

# select devices
if ctx.startswith('cpu'):
    ctx = mx.cpu()
elif ctx.startswith('gpu'):
    ctx = mx.gpu(int(ctx[ctx.index('-') + 1:]))

# import model
print('Model is %s' % (model_name))
if model_name == 'MSTGCN':
    from model.mstgcn import MSTGCN as model
elif model_name == 'ASTGCN':
    from model.astgcn import ASTGCN as model
else:
    raise SystemExit('Wrong type of model!')

# make model params dir
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
if 'params_dir' in training_config and training_config['params_dir'] != "None":
    params_path = os.path.join(training_config['params_dir'], model_name)
else:
    params_path = 'params/%s_%s/' % (model_name, timestamp)

# check parameters file
if os.path.exists(params_path) and not args.force:
    raise SystemExit("Params folder exists! Select a new params path please!")
else:
    if os.path.exists(params_path):
        shutil.rmtree(params_path)
    os.makedirs(params_path)
    print('Create params directory %s' % (params_path))


class MyInit(mx.init.Initializer):
    xavier = mx.init.Xavier()
    uniform = mx.init.Uniform()

    def _init_weight(self, name, data):
        if len(data.shape) < 2:
            self.uniform._init_weight(name, data)
            print('Init', name, data.shape, 'with Uniform')
        else:
            self.xavier._init_weight(name, data)
            print('Init', name, data.shape, 'with Xavier')

if __name__ == "__main__":
    # read all data from graph signal matrix file
    print("Reading data...")
    all_data = read_and_generate_dataset(graph_signal_matrix_filename,
                                         num_of_weeks,
                                         num_of_days,
                                         num_of_hours,
                                         num_for_predict,
                                         points_per_hour,
                                         merge)

    # test set ground truth
    true_value = (all_data['test']['target'].transpose((0, 2, 1))
                  .reshape(all_data['test']['target'].shape[0], -1))

    # training set data loader
    train_loader = gluon.data.DataLoader(
                        gluon.data.ArrayDataset(
                            nd.array(all_data['train']['week'], ctx=ctx),
                            nd.array(all_data['train']['day'], ctx=ctx),
                            nd.array(all_data['train']['recent'], ctx=ctx),
                            nd.array(all_data['train']['target'], ctx=ctx)
                        ),
                        batch_size=batch_size,
                        shuffle=True
    )

    # validation set data loader
    val_loader = gluon.data.DataLoader(
                    gluon.data.ArrayDataset(
                        nd.array(all_data['val']['week'], ctx=ctx),
                        nd.array(all_data['val']['day'], ctx=ctx),
                        nd.array(all_data['val']['recent'], ctx=ctx),
                        nd.array(all_data['val']['target'], ctx=ctx)
                    ),
                    batch_size=batch_size,
                    shuffle=False
    )

    # testing set data loader
    test_loader = gluon.data.DataLoader(
                    gluon.data.ArrayDataset(
                        nd.array(all_data['test']['week'], ctx=ctx),
                        nd.array(all_data['test']['day'], ctx=ctx),
                        nd.array(all_data['test']['recent'], ctx=ctx),
                        nd.array(all_data['test']['target'], ctx=ctx)
                    ),
                    batch_size=batch_size,
                    shuffle=False
    )

    # save Z-score mean and std
    stats_data = {}
    for type_ in ['week', 'day', 'recent']:
        stats = all_data['stats'][type_]
        stats_data[type_ + '_mean'] = stats['mean']
        stats_data[type_ + '_std'] = stats['std']
    np.savez_compressed(
        os.path.join(params_path, 'stats_data'),
        **stats_data
    )

    # loss function MSE
    loss_function = gluon.loss.L2Loss()

    # get model's structure
    all_backbones = get_backbones(args.config, adj_filename, ctx)

    net = model(num_for_predict, all_backbones)
    net.initialize(ctx=ctx)
    for val_w, val_d, val_r, val_t in val_loader:
        net([val_w, val_d, val_r])
        break
    net.initialize(ctx=ctx, init=MyInit(), force_reinit=True)

    # initialize a trainer to train model
    trainer = gluon.Trainer(net.collect_params(), optimizer,
                            {'learning_rate': learning_rate})

    # initialize a SummaryWriter to write information into logs dir
    sw = SummaryWriter(logdir=params_path, flush_secs=5)

    # compute validation loss before training
    compute_val_loss(net, val_loader, loss_function, sw, epoch=0)

    # compute testing set MAE, RMSE, MAPE before training
    evaluate(net, test_loader, true_value, num_of_vertices, sw, epoch=0)

    # train model
    global_step = 1
    for epoch in range(1, epochs + 1):

        for train_w, train_d, train_r, train_t in train_loader:

            start_time = time()

            with autograd.record():
                output = net([train_w, train_d, train_r])
                l = loss_function(output, train_t)
            l.backward()
            trainer.step(train_t.shape[0])
            training_loss = l.mean().asscalar()

            sw.add_scalar(tag='training_loss',
                          value=training_loss,
                          global_step=global_step)

            print('global step: %s, training loss: %.2f, time: %.2fs'
                  % (global_step, training_loss, time() - start_time))
            global_step += 1

        # logging the gradients of parameters for checking convergence
        for name, param in net.collect_params().items():
            try:
                sw.add_histogram(tag=name + "_grad",
                                 values=param.grad(),
                                 global_step=global_step,
                                 bins=1000)
            except:
                print("can't plot histogram of {}_grad".format(name))

        # compute validation loss
        compute_val_loss(net, val_loader, loss_function, sw, epoch)

        # evaluate the model on testing set
        evaluate(net, test_loader, true_value, num_of_vertices, sw, epoch)

        params_filename = os.path.join(params_path,
                                       '%s_epoch_%s.params' % (model_name,
                                                               epoch))
        net.save_parameters(params_filename)
        print('save parameters to file: %s' % (params_filename))

    # close SummaryWriter
    sw.close()

    if 'prediction_filename' in training_config:
        prediction_path = training_config['prediction_filename']

        prediction = predict(net, test_loader)

        np.savez_compressed(
            os.path.normpath(prediction_path),
            prediction=prediction,
            ground_truth=all_data['test']['target']
        )
