# -*- coding:utf-8 -*-

import numpy as np
import mxnet as mx

from sklearn.preprocessing import StandardScaler

from .utils import generate_x_y

def normalization(train, val, test, num_of_vertices, num_of_features, points_per_hour, length):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray, shape is (num_of_samples, num_of_vertices, num_of_features, points_per_hour * length)

    num_of_vertices: int, number of vertices

    num_of_features: int, number of features

    points_per_hour: int, number of points per hour

    length: number of hours will be used

    Returns
    ----------
    transformer: sklearn.preprocessing.data.StandardScaler

    train_norm, val_norm, test_norm: np.ndarray, shape is (num_of_samples, num_of_vertices, num_of_features, points_per_hour * length)

    '''
    transformer = StandardScaler()

    train_norm = transformer.fit_transform(train.reshape(train.shape[0], -1))\
                .reshape(train.shape[0], num_of_vertices, num_of_features, points_per_hour * length)

    val_norm = transformer.transform(val.reshape(val.shape[0], -1))\
                .reshape(val.shape[0], num_of_vertices, num_of_features, points_per_hour * length)

    test_norm = transformer.transform(test.reshape(test.shape[0], -1))\
                .reshape(test.shape[0], num_of_vertices, num_of_features, points_per_hour * length)

    return transformer, train_norm, val_norm, test_norm

def read_and_generate_dataset(graph_signal_matrix_filename, num_of_vertices, num_of_features, num_of_weeks, num_of_days, num_of_hours, points_per_hour, num_for_predict):
    data = np.load(graph_signal_matrix_filename)
    train, val, test = data['train'], data['val'], data['test']
    print(train.shape, val.shape, test.shape)

    train_week, train_day, train_recent, train_target = generate_x_y(train, num_of_weeks, num_of_days, num_of_hours, points_per_hour, num_for_predict)
    val_week, val_day, val_recent, val_target = generate_x_y(val, num_of_weeks, num_of_days, num_of_hours, points_per_hour, num_for_predict)
    test_week, test_day, test_recent, test_target = generate_x_y(test, num_of_weeks, num_of_days, num_of_hours, points_per_hour, num_for_predict)

    print('training size:', train_week.shape, train_day.shape, train_recent.shape, train_target.shape)
    print('validation size:', val_week.shape, val_day.shape, val_recent.shape, val_target.shape)
    print('testing size:', test_week.shape, test_day.shape, test_recent.shape, test_target.shape)

    week_transformer, train_week_norm, val_week_norm, test_week_norm = normalization(train_week, val_week, test_week, num_of_vertices, num_of_features, points_per_hour, num_of_weeks)
    day_transformer, train_day_norm, val_day_norm, test_day_norm = normalization(train_day, val_day, test_day, num_of_vertices, num_of_features, points_per_hour, num_of_days)
    recent_transformer, train_recent_norm, val_recent_norm, test_recent_norm = normalization(train_recent, val_recent, test_recent, num_of_vertices, num_of_features, points_per_hour, num_of_hours)

    all_data = {
        'train': {
            'week': train_week_norm,
            'day': train_day_norm,
            'recent': train_recent_norm,
            'target': train_target,
        },
        'val': {
            'week': val_week_norm,
            'day': val_day_norm,
            'recent': val_recent_norm,
            'target': val_target
        },
        'test': {
            'week': test_week_norm,
            'day': test_day_norm,
            'recent': test_recent_norm,
            'target': test_target
        },
        'transformer': {
            'week': week_transformer,
            'day': day_transformer,
            'recent': recent_transformer
        }
    }

    return all_data