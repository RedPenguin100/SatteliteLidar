import time

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from numba import njit


@njit
def gaussian_weights(x, x_i, h):
    norm = np.linalg.norm(x - x_i)
    pow = -np.power(norm / h, 2)
    return np.exp(pow)


def gaussian_weights_vec(xs, x_i, h):
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs).T
    diff = np.subtract(xs, x_i)
    norm = np.linalg.norm(diff, axis=-1)
    return np.exp(-np.power(norm / h, 2))


def calc_s(xs: np.array, x: float, y: np.array, h):
    s = 0.

    gaussian_weights = gaussian_weights_vec(xs, x, h)
    gaussian_sum = np.sum(gaussian_weights)

    for j in range(len(xs)):
        w = gaussian_sum

        w = gaussian_weights[j] / w
        s += y[j] * w

    return s


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
    poly = PolynomialFeatures(degree=m)
    poly.fit_transform(x_eval[0].reshape(1, x_eval.shape[1]))

    for z in range(len(x_eval)):
        indices = find_close_fast(all_xs, x_eval[z], delta)

        xs, ys = all_xs[indices], all_ys[indices]
        gaussian_weights = gaussian_weights_vec(xs, x_eval[z], h)
        D = np.diag(gaussian_weights)

        P = poly.transform(xs)

        b = ys.T @ D @ P @ np.linalg.inv(P.T @ D @ P)

        s = poly.transform(x_eval[z].reshape(1, x_eval.shape[1])) * b
        value = np.sum(s)
        kernel[z] = value

    return kernel


def get_sine_data_2d(n, line, dev, bias: float = 0.):
    start, end = line
    base = np.linspace(start, end, n)
    X, Y = np.meshgrid(base, base)
    Z = np.sin(X * Y / (2 * np.pi))
    Z = Z + np.random.normal(bias, scale=dev, size=Z.shape)
    return X, Y, Z


def basic_experiment(n, m):
    line = (0, 2 * np.pi)

    delta = 1
    extended_line = (line[0] - 2 * delta, line[1] + 2 * delta)

    small_n = n
    # I_base = np.random.uniform(extended_line[0], extended_line[1], small_n)
    X_base = np.random.uniform(extended_line[0], extended_line[1], (small_n, small_n))
    Y_base = np.random.uniform(extended_line[0], extended_line[1], (small_n, small_n))
    XY_base = np.column_stack((X_base.ravel(), Y_base.ravel()))
    Z_base = np.sin(X_base * Y_base / (2 * np.pi))

    J_base = np.random.uniform(extended_line[0], extended_line[1], small_n)
    X_J = np.random.uniform(extended_line[0], extended_line[1], (small_n, small_n))
    Y_J = np.random.uniform(extended_line[0], extended_line[1], (small_n, small_n))
    XY_J = np.column_stack((X_J.ravel(), Y_J.ravel()))
    Z_J = np.sin(X_J * Y_J / (2 * np.pi))

    I_approx_at_base = moving_least_squares(XY_base, Z_base.ravel(), m=m, delta=delta, x_eval=XY_base)
    I_approx_at_J = moving_least_squares(XY_base, Z_base.ravel(), m=m, delta=delta, x_eval=XY_J)
    e_at_J = I_approx_at_J - Z_J.ravel()

    MLS_approx_at_error = moving_least_squares(XY_J, e_at_J, m=m, delta=delta, x_eval=XY_base)
    # combined approach

    stacked_XY = np.vstack((XY_base, XY_J))
    stacked_Z = np.vstack((Z_base, Z_J)).ravel()
    I_J_approx_at_base = moving_least_squares(stacked_XY, stacked_Z, m=m, delta=delta, x_eval=XY_base)

    # mask = np.ones_like(X_base, dtype=bool)
    mask = (X_base >= line[0]) & (X_base <= line[1]) & (Y_base >= line[0]) & (Y_base <= line[1])

    I_approx_at_base = I_approx_at_base.reshape(small_n, small_n)
    X_base_masked = X_base[mask]
    Y_base_masked = Y_base[mask]
    I_approx_at_base_masked = I_approx_at_base[mask]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.scatter(X_base_masked, Y_base_masked, I_approx_at_base_masked, color='blue', alpha=0.5)
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    mask_at_J = (X_J >= line[0]) & (X_J <= line[1]) & (Y_J >= line[0]) & (Y_J <= line[1])
    I_approx_at_J = I_approx_at_J.reshape(small_n, small_n)
    X_J_masked = X_J[mask_at_J]
    Y_J_masked = Y_J[mask_at_J]
    I_approx_at_J_masked = I_approx_at_J[mask_at_J]

    # fig2 = plt.figure()
    # ax = fig2.add_subplot(111, projection='3d')
    #
    # ax.scatter(X_J_masked, Y_J_masked, I_approx_at_J_masked, color='blue', alpha=0.5)
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    mask_at_J = (X_J >= line[0]) & (X_J <= line[1]) & (Y_J >= line[0]) & (Y_J <= line[1])
    e_at_J = e_at_J.reshape(small_n, small_n)
    X_J_masked = X_J[mask_at_J]
    Y_J_masked = Y_J[mask_at_J]
    e_at_J_masked = e_at_J[mask_at_J]

    # fig3 = plt.figure()
    # ax = fig3.add_subplot(111, projection='3d')
    #
    # ax.scatter(X_J_masked, Y_J_masked, e_at_J_masked, color='blue', alpha=0.5)
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    mask_base = (X_base >= line[0]) & (X_base <= line[1]) & (Y_base >= line[0]) & (Y_base <= line[1])
    MLS_approx_at_error = MLS_approx_at_error.reshape(small_n, small_n)
    X_base_masked = X_base[mask_base]
    Y_base_masked = Y_base[mask_base]
    MLS_approx_at_error_masked = MLS_approx_at_error[mask_base]

    # fig4 = plt.figure()
    # ax = fig4.add_subplot(111, projection='3d')
    # ax.scatter(X_base_masked, Y_base_masked, MLS_approx_at_error_masked, color='blue', alpha=0.5)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    mask_base = (X_base >= line[0]) & (X_base <= line[1]) & (Y_base >= line[0]) & (Y_base <= line[1])
    I_J_approx_at_base = I_J_approx_at_base.reshape(small_n, small_n)
    X_base_masked = X_base[mask_base]
    Y_base_masked = Y_base[mask_base]
    I_J_approx_at_base_masked = I_J_approx_at_base[mask_base]

    # fig5 = plt.figure()
    # ax = fig5.add_subplot(111, projection='3d')
    # ax.scatter(X_base_masked, Y_base_masked, I_J_approx_at_base_masked, color='blue', alpha=0.5)
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    mls_i = np.linalg.norm(I_approx_at_base[mask_base] - Z_base[mask_base])
    mls_i_mls_e = np.linalg.norm(I_approx_at_base[mask_base] - MLS_approx_at_error_masked - Z_base[mask_base])
    mls_i_union_j = np.linalg.norm(I_J_approx_at_base[mask_base] - Z_base[mask_base])

    print("Error of MLS_I(x): ", mls_i)
    print("Error MLS_I(x) + MLS_e(x): ", mls_i_mls_e)
    print("Error MLS_(I union J)(x): ", mls_i_union_j)

    # plt.show(block=True)
    return (mls_i, mls_i_mls_e, mls_i_union_j)


def error_diff_main():
    n = 50

    line = (0, 2 * np.pi)

    X, Y, Z = get_sine_data_2d(n, line, 0.1)
    xy = np.column_stack((X.ravel(), Y.ravel()))
    # X / X2, Y / Y2 may be shared
    X2, Y2, Z2 = get_sine_data_2d(n, line, 0.1, bias=0.)
    xy2 = np.column_stack((X.ravel(), Y.ravel()))

    z_flat = Z.ravel()
    z2_flat = Z2.ravel()
    z_diff_flag = z2_flat - z_flat

    small_n = n // 3
    small_base = np.linspace(line[0], line[1], small_n)
    x_eval, y_eval = np.meshgrid(small_base, small_base)
    xy_eval = np.column_stack((x_eval.ravel(), y_eval.ravel()))

    z_approximation = moving_least_squares(xy, z_flat, m=1, delta=1, x_eval=xy_eval)
    # c = moving_least_squares(xy, z_diff_flag, m=1, delta=1, x_eval=xy_eval)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, cmap='viridis')
    # ax.scatter(x_eval, y_eval, c.reshape(n, n), color='blue', alpha=0.5)
    # ax.scatter(x_eval, y_eval, c2.reshape(n, n), color='orange', alpha=0.5)
    ax.scatter(x_eval, y_eval, z_approximation.reshape(small_n, small_n), color='blue', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    fig2 = plt.figure()
    ax3 = fig2.add_subplot(111, projection='3d')

    ax3.scatter(X, Y, z_flat.reshape(n, n), color='blue', alpha=0.5)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    # fig3 = plt.figure()
    # ax2 = fig3.add_subplot(111, projection='3d')
    #
    # ax2.plot_surface(x_eval, y_eval, z2_flat.reshape(n, n), color='blue')
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Y')
    # ax2.set_zlabel('Z')

    plt.show(block=True)


def run_experiment(n, m, tries, func):
    res = []
    start = time.time()
    for i in range(tries):
        res.append(func(n=n, m=m))
    end = time.time()
    print(f"Experiment took: {end - start}")
    return res


if __name__ == "__main__":

    tries = 4

    res_list = []
    for j in [0, 1, 2]:
        for i in [35, 40, 45]:
            res_list.append(run_experiment(i, j, tries, func=basic_experiment))

    res_size = len(res_list)
    errors = np.empty((tries, res_size), dtype=float)
    mid_mls = np.empty((tries, res_size), dtype=float)
    mid_mls2 = np.empty((tries, res_size), dtype=float)
    mls_i = np.empty((tries, res_size), dtype=float)
    mls_plus = np.empty((tries, res_size), dtype=float)
    mls_union = np.empty((tries, res_size), dtype=float)
    for i in range(tries):
        errors[i, :] = np.array(
            [res[i][0] - res[i][2] for res in res_list])
        mid_mls[i, :] = np.array(
            [res[i][0] - res[i][1] for res in res_list])
        mid_mls2[i, :] = np.array(
            [res[i][2] - res[i][1] for res in res_list])
        mls_i[i, :] = np.array(
            [res[i][0] for res in res_list])
        mls_plus[i, :] = np.array(
            [res[i][1] for res in res_list])
        mls_union[i, :] = np.array(
            [res[i][2] for res in res_list])

    print("Averages: ", np.mean(errors, axis=0))
    print("Averages mid_mls: ", np.mean(mid_mls, axis=0))
    print("Averages mid_mls2: ", np.mean(mid_mls2, axis=0))
    print("Averages mls_i: ", np.mean(mls_i, axis=0))
    print("Averages mls_plus: ", np.mean(mls_plus, axis=0))
    print("Averages mls_union: ", np.mean(mls_union, axis=0))

    # n = 50
    #
    # line = (0, 2 * np.pi)
    #
    # X, Y, Z = get_sine_data_2d(n, line, 0.1)
    # xy = np.column_stack((X.ravel(), Y.ravel()))
    # # X / X2, Y / Y2 may be shared
    # X2, Y2, Z2 = get_sine_data_2d(n, line, 0.1, bias=0.)
    # xy2 = np.column_stack((X.ravel(), Y.ravel()))
    #
    # z_flat = Z.ravel()
    # z2_flat = Z2.ravel()
    #
    # small_base = np.linspace(line[0], line[1], n)
    # x_eval, y_eval = np.meshgrid(small_base, small_base)
    # xy_eval = np.column_stack((x_eval.ravel(), y_eval.ravel()))
    #
    # c = moving_least_squares(xy, z_flat, m=1, delta=1, x_eval=xy_eval)
    # c2 = moving_least_squares(xy, z2_flat, m=1, delta=1, x_eval=xy_eval)
    #
    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.plot_surface(X, Y, Z, cmap='viridis')
    # # ax.scatter(x_eval, y_eval, c.reshape(n, n), color='blue', alpha=0.5)
    # # ax.scatter(x_eval, y_eval, c2.reshape(n, n), color='orange', alpha=0.5)
    # print((c -c2).reshape(n, n))
    # ax.scatter(x_eval, y_eval, (c - c2).reshape(n, n), color='blue', alpha=0.5)
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show(block=True)
