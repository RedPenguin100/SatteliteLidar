import numpy as np

from numba import njit
from SatteliteLidar.research.src.comb_utils import my_polynomial_features


@njit
def quick_linalg_norm_axis_minus_1(a: np.array):
    r = a.shape[0]
    res = np.empty(r)
    for i in range(r):
        res[i] = np.linalg.norm(a[i])
    return res


@njit
def gaussian_weights(x, x_i, h):
    norm = np.linalg.norm(x - x_i)
    pow = -np.power(norm / h, 2)
    return np.exp(pow)


@njit
def gaussian_weights_vec(xs: np.array, x_i: float, h: float):
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs).T
    diff = np.subtract(xs, x_i)
    # norm = quick_linalg_norm_axis_minus_1(diff)
    # norm = np.linalg.norm(diff, axis=-1)
    # norm_sq = (norm ** 2)
    norm_sq = np.sum(diff ** 2, axis=-1)  # much faster than taking norm and then squaring
    norm = np.sqrt(norm_sq)
    norm_sq = np.power(norm, 2)

    return np.exp(-norm_sq / (h ** 2))


@njit
def calc_s(xs: np.array, x: float, y: np.array, h):
    s = 0.

    gaussian_weights = gaussian_weights_vec(xs, x, h)
    gaussian_sum = np.sum(gaussian_weights)

    for j in range(len(xs)):
        w = gaussian_sum

        w = gaussian_weights[j] / w
        s += y[j] * w

    return s


"""
Find all values in x_vec \in (x_val-delta, x_val+delta), quickly
"""


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

        # Second pass: store qualifying row indices.
        for i in range(m):
            s = 0.0
            for j in range(n):
                diff = x_vec[i, j] - x_val[j]
                s += diff * diff
            if s < delta_sqr:
                l.append(i)

        return np.array(l)


@njit
def shepard_kernel(x, y, delta, x_eval=None):
    if x_eval is None:
        x_eval = x

    kernel = np.empty(x_eval.shape[:1])

    h = len(x)

    # TODO: efficiency
    for z in range(len(x_eval)):
        indices = find_close_fast(x, x_eval[z], delta)
        s = calc_s(x[indices], x_eval[z], y[indices], h)
        kernel[z] = s

    return kernel


@njit
def mls_matrices_multiply(ys, weights, P):
    return (ys.T * weights) @ P @ np.linalg.inv((P.T * weights) @ P)


def moving_least_squares_old(all_xs: np.array, all_ys: np.array, m=0, delta: float = 1,
                             x_eval: np.array = None) -> np.array:
    from sklearn.preprocessing import PolynomialFeatures

    if len(all_xs.shape) == 1:
        all_xs = np.atleast_2d(all_xs).T

    if x_eval is None:
        x_eval = all_xs

    if m == 0:
        return shepard_kernel(all_xs, all_ys, delta, x_eval)
    kernel = np.empty(x_eval.shape[:1])

    h = len(all_xs)
    poly = PolynomialFeatures(degree=m)
    poly.fit_transform(x_eval[0].reshape(1, x_eval.shape[1]))

    for z in range(len(x_eval)):
        indices = find_close_fast(all_xs, x_eval[z], delta)

        xs, ys = all_xs[indices], all_ys[indices]
        gaussian_weights = gaussian_weights_vec(xs, x_eval[z], h)
        P = poly.transform(xs)

        # gaussian_weights could be viewed as the diagonal D, it is faster to do cartesian
        b = mls_matrices_multiply(ys, gaussian_weights, P)

        s = poly.transform(x_eval[z].reshape(1, x_eval.shape[1])) * b
        value = np.sum(s)
        kernel[z] = value

    return kernel


@njit
def moving_least_squares(all_xs: np.array, all_ys: np.array, m=0, delta: float = 1,
                         x_eval: np.array = None) -> np.array:
    if len(all_xs.shape) == 1:
        all_xs = np.atleast_2d(all_xs).T

    if x_eval is None:
        x_eval = all_xs

    if m == 0:
        return shepard_kernel(all_xs, all_ys, delta, x_eval)
    kernel = np.empty(x_eval.shape[:1])

    h = len(all_xs)

    for z in range(len(x_eval)):
        indices = find_close_fast(all_xs, x_eval[z], delta)

        xs, ys = all_xs[indices], all_ys[indices]
        gaussian_weights = gaussian_weights_vec(xs, x_eval[z], h)
        P = my_polynomial_features(xs, m)

        # gaussian_weights could be viewed as the diagonal D, it is faster to do cartesian
        b = mls_matrices_multiply(ys, gaussian_weights, P)

        # temp = my_polynomial_features(x_eval[z].reshape(1, x_eval.shape[1]), m)

        temp = my_polynomial_features(x_eval[z][None, :], m)

        s = temp * b
        value = np.sum(s)
        kernel[z] = value

    return kernel
