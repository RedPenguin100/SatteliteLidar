import itertools
import time
from math import comb

import pytest

import numpy as np
from numba import njit

from sklearn.preprocessing import PolynomialFeatures

from SatteliteLidar.research.src.comb_utils import n_choose_r, combinations_numba, compositions_numba, \
    my_polynomial_features
from SatteliteLidar.research.src.mls import find_close_fast, quick_linalg_norm_axis_minus_1, moving_least_squares


@pytest.mark.parametrize('n,r', [(2, 1), (3, 1), (5, 2), (10, 3)])
def test_n_choose_r(n, r):
    assert n_choose_r(n, r) == comb(n, r)


def test_combinations():
    arr = np.array([1, 2, 3, 4, 5])

    assert np.all(combinations_numba(arr, 2) == [a for a in itertools.combinations(arr, 2)])


def test_compositions():
    print(compositions_numba(3, 2))


def test_polynomial_features():
    arr = np.array([[1, 1], [2, 2]])
    poly = PolynomialFeatures(degree=2)
    # poly.fit_transform(arr)

    a = np.array([5, 2])
    print(poly.fit_transform(a.reshape(1, a.shape[0])))
    print(my_polynomial_features(a.reshape(1, a.shape[0]), 2))

    # print(np.polynomial.polynomial.polyval(2, a))


def test_sanity_2d():
    n = 50
    base = np.linspace(0, 2 * np.pi, n)
    X, Y = np.meshgrid(base, base)
    Z = np.sin(X * Y / (2 * np.pi))
    z_flat = Z.ravel()

    xy = np.column_stack((X.ravel(), Y.ravel()))
    c = moving_least_squares(xy, z_flat, m=0, delta=1)

    assert c.shape == (n ** 2,)
    # TODO: write better regression testing
    assert pytest.approx(c[146]) == 0.4408272963225725


def test_find_close():
    arr = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    points = find_close_fast(arr, 2, 0.9)
    assert np.all(points == np.array([1]))

    points = find_close_fast(arr, 1, 1.1)
    assert np.all(points == np.array([0, 1]))

    points = find_close_fast(arr, 3, 2.1)
    assert np.all(points == np.array([0, 1, 2, 3, 4]))

    arr2 = np.array([[1, 1], [2, 2], [1, 2], [2, 1], [3, 3]], dtype=np.float64)
    points = find_close_fast(arr2, (2, 2), delta=0.1)
    assert np.all(points == np.array([1]))

    points = find_close_fast(arr2, (2, 2), delta=1.1)
    assert np.all(points == np.array([1, 2, 3]))

    points = find_close_fast(arr2, (2, 2), delta=1.5)
    assert np.all(points == np.array([0, 1, 2, 3, 4]))


def find_close(x_vec, x_val, delta):
    if len(x_vec.shape) == 1:
        x_vec = np.atleast_2d(x_vec)
        diff = np.subtract(x_vec, x_val)
        distances = np.linalg.norm(diff, axis=0)
    else:
        diff = np.subtract(x_vec, x_val)
        distances = np.linalg.norm(diff, axis=1)

    indices = np.where(distances < delta)[0]
    return indices


def test_performance():
    start = time.time()
    for i in range(100000):
        arr2 = np.array([[1, 1], [2, 2], [1, 2], [2, 1], [3, 3]], dtype=np.float64)
        points = find_close(arr2, (2, 2), delta=0.1)
    end = time.time()
    print(f"Took: {end - start} seconds")
    arr2 = np.array([[1, 1], [2, 2], [1, 2], [2, 1], [3, 3]], dtype=np.float64)
    points = find_close_fast(arr2, (2, 2), delta=0.1)
    start = time.time()
    for i in range(100000):
        arr2 = np.array([[1, 1], [2, 2], [1, 2], [2, 1], [3, 3]], dtype=np.float64)
        points = find_close_fast(arr2, (2, 2), delta=0.1)
    end = time.time()
    print(f"Took: {end - start} seconds")


def test_stub():
    a = np.array([[1, 1], [2, 2]], dtype=np.float64)
    b = quick_linalg_norm_axis_minus_1(a)


def test_quick_norm2():
    a = np.array([[1, 1], [2, 2]], dtype=np.float64)

    iterations = 100000

    for i in range(iterations):
        b = quick_linalg_norm_axis_minus_1(a)

    assert np.all(np.linalg.norm(a, axis=-1) == quick_linalg_norm_axis_minus_1(a))


def test_quick_norm3():
    a = np.array([[1, 1], [2, 2]], dtype=np.float64)

    iterations = 100000

    for i in range(iterations):
        b = np.linalg.norm(a, axis=-1)

    assert np.all(np.linalg.norm(a, axis=-1) == quick_linalg_norm_axis_minus_1(a))
