import time

import pytest

import numpy as np
from numba import njit

from util import moving_least_squares, find_close


@njit
def find_close_fast(x_vec, x_val, delta):
    # If x_vec is 1D, treat it as a single row (and x_val is assumed to be scalar)
    if x_vec.ndim == 1:
        n = x_vec.shape[0]
        count = 0
        # First pass: count indices where |x_vec[i] - x_val| < delta.
        for i in range(n):
            diff = x_vec[i] - x_val
            if diff < 0:
                diff = -diff
            if diff < delta:
                count += 1
        result = np.empty(count, np.int64)
        idx = 0
        # Second pass: store qualifying indices.
        for i in range(n):
            diff = x_vec[i] - x_val
            if diff < 0:
                diff = -diff
            if diff < delta:
                result[idx] = i
                idx += 1
        return result
    else:
        delta_sqr = delta ** 2
        # For 2D x_vec, compute the Euclidean norm for each row.
        m = x_vec.shape[0]
        n = x_vec.shape[1]

        l = []
        idx = 0
        # Second pass: store qualifying row indices.
        for i in range(m):
            s = 0.0
            for j in range(n):
                diff = x_vec[i, j] - x_val[j]
                s += diff * diff
            if s < delta_sqr:
                l.append(i)

        return np.array(l)


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
    arr = np.array([1,2,3,4,5], dtype=np.float64)
    points = find_close_fast(arr, 2, 0.9)
    assert np.all(points == np.array([1]))

    points = find_close_fast(arr, 1, 1.1)
    assert np.all(points == np.array([0, 1]))

    points = find_close_fast(arr, 3, 2.1)
    assert np.all(points == np.array([0,1,2,3,4]))

    arr2 = np.array([[1, 1], [2, 2], [1, 2], [2, 1], [3, 3]], dtype=np.float64)
    points = find_close_fast(arr2, (2, 2), delta=0.1)
    assert np.all(points == np.array([1]))

    points = find_close_fast(arr2, (2, 2), delta=1.1)
    assert np.all(points == np.array([1, 2, 3]))

    points = find_close_fast(arr2, (2, 2), delta=1.5)
    assert np.all(points == np.array([0, 1, 2, 3, 4]))



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
