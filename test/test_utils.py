# -*- coding:utf-8 -*-

import sys
import pytest
import numpy as np

sys.path.append('.')


def test_search_data1():
    from lib.utils import search_data
    week_indices = search_data(2016 + 12, 1, 2016, 12, 7 * 24, 12)
    assert week_indices == [(0, 12)]


def test_search_data2():
    from lib.utils import search_data
    day_indices = search_data(288 * 2 + 12, 2, 288 * 2, 12, 24, 12)
    assert day_indices == [(0, 12), (288, 300)]


def test_search_data3():
    from lib.utils import search_data
    hour_indices = search_data(12 + 12, 1, 12, 12, 1, 12)
    assert hour_indices == [(0, 12)]


def test_search_data4():
    from lib.utils import search_data
    hour_indices = search_data(12 + 12, 2, 12, 12, 1, 12)
    assert hour_indices is None


def test_search_data5():
    from lib.utils import search_data
    hour_indices = search_data(12 * 2 + 12, 2, 12 * 2, 12, 1, 12)
    assert hour_indices == [(0, 12), (12, 24)]


def test_get_sample_indices1():
    from lib.utils import get_sample_indices
    data = np.random.uniform(size=(2016 + 12, 307, 3))
    week, day, hour, target = get_sample_indices(data, 1, 1, 3, 2016, 12, 12)
    assert week.shape == (12, 307, 3)
    assert day.shape == (12, 307, 3)
    assert hour.shape == (12 * 3, 307, 3)
    assert target.shape == (12, 307, 3)


def test_get_sample_indices2():
    from lib.utils import get_sample_indices
    data = np.random.uniform(size=(2016 + 12, 307, 3))
    sample = get_sample_indices(data, 2, 3, 3, 2016, 12, 12)
    assert sample is None


def test_get_sample_indices3():
    from lib.utils import get_sample_indices
    data = np.random.uniform(size=(7 * 24 * 12 * 2 + 12, 307, 3))
    week, day, hour, target = get_sample_indices(data, 2, 4, 3,
                                                 7 * 24 * 12 * 2, 12, 12)
    assert week.shape == (12 * 2, 307, 3)
    assert day.shape == (12 * 4, 307, 3)
    assert hour.shape == (12 * 3, 307, 3)
    assert target.shape == (12, 307, 3)


def test_get_adjacency_matrix1():
    from lib.utils import get_adjacency_matrix
    filename = 'data/PEMS04/distance.csv'
    num_of_vertices = 307
    A = get_adjacency_matrix(filename, num_of_vertices)
    assert A.shape == (num_of_vertices, num_of_vertices)


def test_get_adjacency_matrix2():
    from lib.utils import get_adjacency_matrix
    filename = 'data/PEMS08/distance.csv'
    num_of_vertices = 170
    A = get_adjacency_matrix(filename, num_of_vertices)
    assert A.shape == (num_of_vertices, num_of_vertices)


def test_scaled_Laplacian():
    from lib.utils import get_adjacency_matrix, scaled_Laplacian
    adj = get_adjacency_matrix('data/PEMS04/distance.csv', 307)
    assert scaled_Laplacian(adj).shape == adj.shape


def test_cheb_polynomial1():
    from lib.utils import (get_adjacency_matrix,
                           scaled_Laplacian, cheb_polynomial)
    adj = get_adjacency_matrix('data/PEMS04/distance.csv', 307)
    L = scaled_Laplacian(adj)
    cheb_polys = cheb_polynomial(L, 3)
    assert len(cheb_polys) == 3
    for i in cheb_polys:
        assert i.shape == adj.shape


def test_cheb_polynomial2():
    from lib.utils import (get_adjacency_matrix,
                           scaled_Laplacian, cheb_polynomial)
    adj = get_adjacency_matrix('data/PEMS08/distance.csv', 170)
    L = scaled_Laplacian(adj)
    cheb_polys = cheb_polynomial(L, 4)
    assert len(cheb_polys) == 4
    for i in cheb_polys:
        assert i.shape == adj.shape
