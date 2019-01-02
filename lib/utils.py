# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from .metrics import masked_mape_np

def search_day_data(train, num_of_days, label_start_idx, points_per_hour, num_for_predict):
    '''
    find data in previous day given current start index.
    for example, if current start index is 8:00 am on Wed, it will return start and end index of 8:00 am on Tue

    Parameters
    ----------
    train: np.ndarray

    num_of_days: int, how many days will be used

    label_start_idx: current start index

    points_per_hour: number of points per hour

    num_for_predict: number of points will be predict

    Returns
    ----------
    list[(start_index, end_index)]: length is num_of_days, for example, if label_start_idx represents 8:00 am Wed, 
                                    num_of_days is 2, it will return [(8:00 am Mon, 9:00 am Mon), (8:00 am Tue, 9:00 am Tue)]
    the second returned value is (label_start_idx, label_start_idx + num_for_predict), e.g. (8:00 am Wed, 9:00 am Wed)

    '''
    if label_start_idx + num_for_predict > len(train):
        return None
    x_idx = []
    for i in range(1, num_of_days + 1):
        start_idx, end_idx = label_start_idx - 12 * (24 * i), label_start_idx - 12 * (24 * i) + points_per_hour
        if start_idx >= 0 and end_idx >= 0:
            x_idx.append((start_idx, end_idx))
    if len(x_idx) != num_of_days:
        return None
    return list(reversed(x_idx)), (label_start_idx, label_start_idx + num_for_predict)

def search_week_data(train, num_of_weeks, label_start_idx, points_per_hour, num_for_predict):
    '''
    just like search_day_data, this function search previous week data
    '''
    if label_start_idx + num_for_predict > len(train):
        return None
    x_idx = []
    for i in range(1, num_of_weeks + 1):
        start_idx, end_idx = label_start_idx - 12 * (24 * 7 * i), label_start_idx - 12 * (24 * 7 * i) + points_per_hour
        if start_idx >= 0 and end_idx >= 0:
            x_idx.append((start_idx, end_idx))
    if len(x_idx) != num_of_weeks:
        return None
    return list(reversed(x_idx)), (label_start_idx, label_start_idx + num_for_predict)

def search_recent_data(train, num_of_hours, label_start_idx, points_per_hour, num_for_predict):
    '''
    just like search_day_data, this function search previous hour data
    '''
    if label_start_idx + num_for_predict > len(train):
        return None
    x_idx = []
    for i in range(1, num_of_hours + 1):
        start_idx, end_idx = label_start_idx - 12 * i, label_start_idx - 12 * i + points_per_hour
        if start_idx >= 0 and end_idx >= 0:
            x_idx.append((start_idx, end_idx))
    if len(x_idx) != num_of_hours:
        return None
    return list(reversed(x_idx)), (label_start_idx, label_start_idx + num_for_predict)

def generate_x_y(train, num_of_weeks, num_of_days, num_of_hours, points_per_hour, num_for_predict):
    '''
    Parameters
    ----------
    train: np.ndarray, shape is (num_of_samples, num_of_vertices, num_of_features)
    
    num_of_weeks, num_of_days, num_of_hours: int
    
    Returns
    ----------
    week_data: np.ndarray, shape is (num_of_samples, num_of_vertices, num_of_features, points_per_hour * num_of_weeks)
    
    day_data: np.ndarray, shape is (num_of_samples, num_of_vertices, num_of_features, points_per_hour * num_of_days)
    
    recent_data: np.ndarray, shape is (num_of_samples, num_of_vertices, num_of_features, points_per_hour * num_of_hours)
    
    target: np.ndarray, shape is (num_of_samples, num_of_vertices, num_for_predict)
    
    '''
    length = len(train)
    data = []
    for i in range(length):
        week = search_week_data(train, num_of_weeks, i, points_per_hour, num_for_predict)
        day = search_day_data(train, num_of_days, i, points_per_hour, num_for_predict)
        recent = search_recent_data(train, num_of_hours, i, points_per_hour, num_for_predict)
        if week and day and recent:
            assert week[1] == day[1]
            assert day[1] == recent[1]
            week_data = np.concatenate([train[i: j] for i, j in week[0]], axis = 0)
            day_data = np.concatenate([train[i: j] for i, j in day[0]], axis = 0)
            recent_data = np.concatenate([train[i: j] for i, j in recent[0]], axis = 0)
            data.append(((week_data, day_data, recent_data), train[week[1][0]: week[1][1]]))
    
    features, label = zip(*data)
    week_data, day_data, recent_data = (np.concatenate([np.expand_dims(x.transpose((1, 2, 0)), 0) for x in i], 0) for i in zip(*features))
    target = np.concatenate([np.expand_dims(x.transpose((1, 2, 0)), 0) for x in label], axis = 0)[:, :, 0, :]
    return week_data, day_data, recent_data, target

def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix
    
    '''
    distance_df = pd.read_csv(distance_df_filename, dtype={'from': 'int', 'to': 'int'})
    # pylint: disable=no-member
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype = np.float32)

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        i, j = int(row[0]), int(row[1])
        A[i, j] = 1
        A[j, i] = 1
    
    return A

def compute_val_loss(net, val_loader, loss_function, sw, epoch):
    '''
    compute mean loss on validation set

    Parameters
    ----------
    net: model

    val_loader: gluon.data.DataLoader

    loss_function: func

    sw: mxboard.SummaryWriter

    epoch: int, current epoch

    '''
    val_loader_length = len(val_loader)
    tmp = []
    for index, (val_w, val_d, val_r, val_t) in enumerate(val_loader):
        output = net([val_w, val_d, val_r])
        l = loss_function(output, val_t)
        tmp.extend(l.asnumpy().tolist())
        print('validation batch %s / %s, loss: %.2f'%(index + 1, val_loader_length, l.mean().asscalar()))
    validation_loss = sum(tmp) / len(tmp)
    sw.add_scalar(tag = 'validation_loss', value = validation_loss, global_step = epoch)
    print('epoch: %s, validation loss: %.2f'%(epoch, validation_loss))

def predict(net, test_loader):
    '''
    predict

    Parameters
    ----------
    net: model

    test_loader: gluon.data.DataLoader

    Returns
    ----------
    prediction: np.ndarray, shape is (num_of_samples, num_of_vertices, num_for_predict)

    '''

    test_loader_length = len(test_loader)
    prediction = []
    for index, (test_w, test_d, test_r, _) in enumerate(test_loader):
        prediction.append(net([test_w, test_d, test_r]).asnumpy())
        print('predicting testing set batch %s / %s'%(index + 1, test_loader_length))
    prediction = np.concatenate(prediction, 0)
    return prediction

def evaluate(net, test_loader, true_value, num_of_vertices, sw, epoch):
    '''
    compute MAE, RMSE, MAPE scores of the prediction for 3, 6, 12 points on testing set

    Parameters
    ----------
    net: model

    test_loader: gluon.data.DataLoader

    true_value: np.ndarray, all ground truth of testing set

    num_of_vertices: int, number of vertices

    sw: mxboard.SummaryWriter

    epoch: int, current epoch

    '''
    prediction = predict(net, test_loader)
    prediction = prediction.transpose((0, 2, 1)).reshape(prediction.shape[0], -1)
    for i in [3, 6, 12]:
        print('current epoch: %s, predict %s points'%(epoch, i))
        mae = mean_absolute_error(true_value[:, : i * num_of_vertices], prediction[:, : i * num_of_vertices])
        rmse = mean_squared_error(true_value[:, : i * num_of_vertices], prediction[:, : i * num_of_vertices]) ** 0.5
        mape = masked_mape_np(true_value[:, : i * num_of_vertices], prediction[:, : i * num_of_vertices], 0)
        print('MAE: %.2f'%(mae))
        print('RMSE: %.2f'%(rmse))
        print('MAPE: %.2f'%(mape))
        print()
        sw.add_scalar(tag = 'MAE_%s_points'%(i), value = mae, global_step = epoch)
        sw.add_scalar(tag = 'RMSE_%s_points'%(i), value = rmse, global_step = epoch)
        sw.add_scalar(tag = 'MAPE_%s_points'%(i), value = mape, global_step = epoch)