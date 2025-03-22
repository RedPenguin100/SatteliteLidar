import numpy as np

from numba import njit
from stepwise_mls.comb_utils import my_polynomial_features


@njit(cache=True)
def quick_linalg_norm_axis_minus_1(a: np.array):
    r = a.shape[0]
    res = np.empty(r)
    for i in range(r):
        res[i] = np.linalg.norm(a[i])
    return res


@njit(cache=True)
def gaussian_weights(x, x_i, h):
    norm = np.linalg.norm(x - x_i)
    pow = -np.power(norm / h, 2)
    return np.exp(pow)


@njit(cache=True)
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


@njit(cache=True)
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
Find all values in x_vec in (x_val-delta, x_val+delta), quickly
"""


@njit(cache=True)
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
                if len(l) == 0:
                    next_hint = i
                l.append(i)

        return np.array(l)


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
def mls_novel(XY_I, Z_I, XY_J, Z_J, m, delta):
    I_approx_at_I = moving_least_squares(XY_I, Z_I.ravel(), m=m, delta=delta, x_eval=XY_I)

    I_approx_at_J = moving_least_squares(XY_I, Z_I.ravel(), m=m, delta=delta, x_eval=XY_J)
    e_at_J = I_approx_at_J - Z_J.ravel()
    MLS_approx_at_error = moving_least_squares(XY_J, e_at_J, m=m, delta=delta, x_eval=XY_I)

    return I_approx_at_I - MLS_approx_at_error


@njit(cache=True)
def mls_novel_all(XY_I, Z_I, XY_J, Z_J, m, delta):
    stacked_XY = np.vstack((XY_I, XY_J))

    I_approx_at_IJ = moving_least_squares(XY_I, Z_I.ravel(), m=m, delta=delta, x_eval=stacked_XY)

    e_at_J = I_approx_at_IJ[len(XY_I):] - Z_J.ravel()

    MLS_approx_at_error = moving_least_squares(XY_J, e_at_J, m=m, delta=delta, x_eval=stacked_XY)

    return I_approx_at_IJ - MLS_approx_at_error


@njit(cache=True)
def mls_novel_all_3(XY_I, Z_I, XY_J, Z_J, XY_K, Z_K, m, delta):
    stacked_XY = np.vstack((XY_I, XY_J))
    super_stacked = np.vstack((stacked_XY, XY_K))

    I_approx_at_IJ = moving_least_squares(XY_I, Z_I.ravel(), m=m, delta=delta, x_eval=super_stacked)

    e_at_J = I_approx_at_IJ[len(XY_I):len(XY_I) + len(XY_J)] - Z_J.ravel()

    MLS_approx_at_error = moving_least_squares(XY_J, e_at_J, m=m, delta=delta, x_eval=super_stacked)

    IJ_approx = I_approx_at_IJ - MLS_approx_at_error
    #################

    approx_K = moving_least_squares(stacked_XY, IJ_approx, m=m, delta=delta, x_eval=XY_K)
    e_at_K = approx_K - Z_K.ravel()

    MLS_approx_at_error_K = moving_least_squares(XY_K, e_at_K, m=m, delta=delta, x_eval=super_stacked)

    return IJ_approx - MLS_approx_at_error_K


# @njit(cache=True)
def mls_combined(XY_I, Z_I, XY_J, Z_J, m, delta):
    stacked_XY = np.vstack((XY_I, XY_J))
    stacked_Z = np.vstack((Z_I, Z_J)).ravel()
    IJ_approx_at_I = moving_least_squares(stacked_XY, stacked_Z, m=m, delta=delta, x_eval=XY_I)
    return IJ_approx_at_I


@njit(cache=True)
def mls_combined_all(stacked_XY, stacked_Z, m, delta):
    IJ_approx_at_IJ = moving_least_squares(stacked_XY, stacked_Z, m=m, delta=delta, x_eval=stacked_XY)
    return IJ_approx_at_IJ
