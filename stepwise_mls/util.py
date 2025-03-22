import numpy as np
from matplotlib import pyplot as plt

from stepwise_mls.error_utils import Error
from stepwise_mls.mls import moving_least_squares, mls_combined, mls_novel_all, mls_combined_all
from stepwise_mls.timer import Timer
from stepwise_mls.experiment import MLSResult, Experiment, AggregatedExperiment, ExperimentStore


def get_sine_data_2d(n, line, dev, bias: float = 0.):
    start, end = line
    base = np.linspace(start, end, n)
    X, Y = np.meshgrid(base, base)
    Z = np.sin(X * Y / (2 * np.pi))
    noise = np.random.normal(bias, scale=dev, size=Z.shape)
    Z = Z + noise
    return X, Y, Z


def generate_2d_base(line, n, base_points='uniform'):
    if base_points == 'uniform':
        X_base = np.random.uniform(line[0], line[1], (n, n))
        Y_base = np.random.uniform(line[0], line[1], (n, n))
        return X_base, Y_base
    elif base_points == 'halton':
        from scipy.stats import qmc
        if not hasattr(generate_2d_base, 'sampler'):
            generate_2d_base.sampler = qmc.Halton(d=2, scramble=False)
        base = generate_2d_base.sampler.random(n=n ** 2)
        base = qmc.scale(base, line[0], line[1])
        X_base, Y_base = base[:, 0].reshape(n, n), base[:, 1].reshape(n, n)
        return X_base, Y_base

    ValueError(f'Unknown base_points={base_points} argument')


def basic_experiment(n, m, delta, base_points='halton', base_function='sine', plot=False):
    line = (0, np.pi)
    gap = 1
    extended_line = (line[0] - 2 * gap, line[1] + 2 * gap)

    small_n = n
    # I_base = np.random.uniform(extended_line[0], extended_line[1], small_n)
    X_base, Y_base = generate_2d_base(line, n, base_points)
    X_J, Y_J = generate_2d_base(line, n, base_points)

    if base_function == 'sine':
        Z_base = np.sin(X_base * Y_base * 2)
        Z_J = np.sin(X_J * Y_J * 2)
        # Z_base = np.sin(X_base * Y_base)
        # Z_J = np.sin(X_J * Y_J)
    elif base_function == 'exp':
        Z_base = np.exp(X_base * Y_base / (2 * np.pi))
        Z_J = np.exp(X_J * Y_J / (2 * np.pi))
        # Z_base = np.exp(X_base * Y_base )
        # Z_J = np.exp(X_J * Y_J)

    XY_base = np.column_stack((X_base.ravel(), Y_base.ravel()))
    XY_J = np.column_stack((X_J.ravel(), Y_J.ravel()))

    mls_i = MLSResult()
    mls_plus = MLSResult()
    mls_union = MLSResult()

    with Timer() as t:
        I_approx_at_base = moving_least_squares(XY_base, Z_base.ravel(), m=m, delta=delta, x_eval=XY_base)
    mls_i.time = t.elapsed_time

    with Timer() as t:
        I_approx_at_J = moving_least_squares(XY_base, Z_base.ravel(), m=m, delta=delta, x_eval=XY_J)
        e_at_J = I_approx_at_J - Z_J.ravel()

        MLS_approx_at_error = moving_least_squares(XY_J, e_at_J, m=m, delta=delta, x_eval=XY_base)
    mls_plus.time = t.elapsed_time

    # combined approach

    with Timer() as t:
        I_J_approx_at_base = mls_combined(XY_base, Z_base, XY_J, Z_J, m, delta)
    mls_union.time = t.elapsed_time

    # mask = np.ones_like(X_base, dtype=bool)
    mask = (X_base >= line[0]) & (X_base <= line[1]) & (Y_base >= line[0]) & (Y_base <= line[1])

    I_approx_at_base = I_approx_at_base.reshape(small_n, small_n)
    X_base_masked = X_base[mask]
    Y_base_masked = Y_base[mask]
    I_approx_at_base_masked = I_approx_at_base[mask]
    if plot:
        Z_base_masked = Z_base[mask]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X_base_masked, Y_base_masked, I_approx_at_base_masked, color='blue', alpha=0.5)
        ax.plot_trisurf(X_base_masked, Y_base_masked, Z_base_masked, color='orange', alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('MLS on set I')

    mask_at_J = (X_J >= line[0]) & (X_J <= line[1]) & (Y_J >= line[0]) & (Y_J <= line[1])
    I_approx_at_J = I_approx_at_J.reshape(small_n, small_n)
    X_J_masked = X_J[mask_at_J]
    Y_J_masked = Y_J[mask_at_J]
    I_approx_at_J_masked = I_approx_at_J[mask_at_J]

    if plot:
        Z_J_base_masked = Z_J[mask_at_J]

        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')

        ax.scatter(X_J_masked, Y_J_masked, I_approx_at_J_masked, color='blue', alpha=0.5)
        ax.plot_trisurf(X_J_masked, Y_J_masked, Z_J_base_masked, color='orange', alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Iterative first approx')

    mask_at_J = (X_J >= line[0]) & (X_J <= line[1]) & (Y_J >= line[0]) & (Y_J <= line[1])
    e_at_J = e_at_J.reshape(small_n, small_n)
    X_J_masked = X_J[mask_at_J]
    Y_J_masked = Y_J[mask_at_J]
    e_at_J_masked = e_at_J[mask_at_J]

    mask_base = (X_base >= line[0]) & (X_base <= line[1]) & (Y_base >= line[0]) & (Y_base <= line[1])
    X_base_masked = X_base[mask_base]
    Y_base_masked = Y_base[mask_base]
    MLS_approx_at_error = MLS_approx_at_error.reshape(small_n, small_n)
    MLS_approx_at_error_masked = MLS_approx_at_error[mask_base]

    if plot:
        fig3 = plt.figure()
        ax = fig3.add_subplot(111, projection='3d')

        ax.scatter(X_J_masked, Y_J_masked, e_at_J_masked, label='Error on set I', color='blue', alpha=0.5)
        ax.plot_trisurf(X_base_masked, Y_base_masked, MLS_approx_at_error_masked, label='Approximated error',
                        color='orange', alpha=0.5)

        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Error on I vs approximated error func={base_function}')

    mask_base = (X_base >= line[0]) & (X_base <= line[1]) & (Y_base >= line[0]) & (Y_base <= line[1])
    MLS_approx_at_error = MLS_approx_at_error.reshape(small_n, small_n)
    X_base_masked = X_base[mask_base]
    Y_base_masked = Y_base[mask_base]
    Z_base_masked = Z_base[mask_base]
    MLS_approx_at_error_masked = MLS_approx_at_error[mask_base]

    if plot:
        fig4 = plt.figure()
        ax = fig4.add_subplot(111, projection='3d')
        # ax.scatter(X_base_masked, Y_base_masked, MLS_approx_at_error_masked, color='blue', alpha=0.5)
        ax.scatter(X_base_masked, Y_base_masked, I_approx_at_base[mask_base] - MLS_approx_at_error_masked,
                   label='Improved approximation', color='green', alpha=0.5)
        ax.scatter(X_base_masked, Y_base_masked, I_approx_at_base[mask_base], label='Approximation on I', color='blue',
                   alpha=0.5)

        ax.plot_trisurf(X_base_masked, Y_base_masked, Z_base_masked, color='orange', alpha=0.5)
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Iterative MLS approximation, func={base_function}')

    mask_base = (X_base >= line[0]) & (X_base <= line[1]) & (Y_base >= line[0]) & (Y_base <= line[1])
    I_J_approx_at_base = I_J_approx_at_base.reshape(small_n, small_n)
    X_base_masked = X_base[mask_base]
    Y_base_masked = Y_base[mask_base]
    I_J_approx_at_base_masked = I_J_approx_at_base[mask_base]

    if plot:
        fig5 = plt.figure()
        ax = fig5.add_subplot(111, projection='3d')
        ax.scatter(X_base_masked, Y_base_masked, I_J_approx_at_base_masked, color='blue', alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Union approximation')

    mls_i.error = Error.calculate_error(I_approx_at_base[mask_base], Z_base[mask_base])
    mls_plus.error = Error.calculate_error(I_approx_at_base[mask_base] - MLS_approx_at_error_masked, Z_base[mask_base])
    mls_union.error = Error.calculate_error(I_J_approx_at_base[mask_base], Z_base[mask_base])

    print("Error of MLS_I(x): ", mls_i.error)
    print("Error MLS_I(x) + MLS_e(x): ", mls_plus.error)
    print("Error MLS_(I union J)(x): ", mls_union.error)

    experiment = Experiment()
    experiment.result_i = mls_i
    experiment.result_plus = mls_plus
    experiment.result_union = mls_union
    experiment.base_function = base_function
    experiment.m = m
    experiment.n = n
    experiment.delta = delta

    if plot:
        plt.show(block=True)
    return experiment


def hetero_experiment(n, m, delta, base_points='halton', base_function='sine'):
    np.random.seed(0)
    line = (0, np.pi)
    gap = 1
    extended_line = (line[0] - gap, line[1] + gap)

    mean_I, mean_J = 0, 0
    stddev_I, stddev_J = 0.1, 0.3

    mls_plus = MLSResult()
    mls_union = MLSResult()

    X_I, Y_I = generate_2d_base(extended_line, n, base_points=base_points)
    Z_truth_I = np.sin(X_I * Y_I * 2)
    Z_I = Z_truth_I + np.random.normal(mean_I, stddev_I, X_I.shape)
    XY_I = np.column_stack((X_I.ravel(), Y_I.ravel()))

    X_J, Y_J = generate_2d_base(extended_line, n, base_points=base_points)
    Z_truth_J = np.sin(X_J * Y_J * 2)
    Z_J = Z_truth_J + np.random.normal(mean_J, stddev_J, X_J.shape)
    XY_J = np.column_stack((X_J.ravel(), Y_J.ravel()))

    # Consider if this is right
    with Timer(verbose=False) as timer:
        mls_approx_novel_all = mls_novel_all(XY_I, Z_I, XY_J, Z_J, m, delta)
    mls_plus.time = timer.elapsed_time

    with Timer(verbose=False) as timer:
        mls_approx_combined = mls_combined_all(XY_I, Z_I, XY_J, Z_J, m, delta)
    mls_union.time = timer.elapsed_time

    stacked_XY = np.vstack((XY_I, XY_J))
    stacked_Z = np.vstack((Z_I, Z_J)).ravel()

    # mask_I = (XY_I >= line[0]) & (XY_I <= line[1])
    # mask_I = np.logical_and(mask_I[:, 0], mask_I[:, 1])

    mask_IJ = (stacked_XY >= line[0]) & (stacked_XY <= line[1])
    mask_IJ = np.logical_and(mask_IJ[:, 0], mask_IJ[:, 1])
    #        mask_IJ = (X_I >= line[0]) & (X_I <= line[1]) & (Y_I >= line[0]) & (Y_I <= line[1])

    mls_approx_novel_masked = mls_approx_novel_all[mask_IJ]
    mls_approx_combined_masked = mls_approx_combined[mask_IJ]

    # error_plus = Error.calculate_error(mls_approx_novel_masked, Z_truth_I[mask_I])
    # error_combined = Error.calculate_error(mls_approx_combined_masked, Z_truth_I[mask_I])

    mls_plus.error = Error.calculate_error(mls_approx_novel_masked, stacked_Z[mask_IJ])
    mls_union.error = Error.calculate_error(mls_approx_combined_masked, stacked_Z[mask_IJ])

    mls_i = MLSResult()
    mls_i.time = 0
    mls_i.error = Error(0, 0, 0, 0)

    experiment = Experiment()
    experiment.result_i = mls_i
    experiment.result_plus = mls_plus
    experiment.result_union = mls_union
    experiment.base_function = base_function
    experiment.m = m
    experiment.n = n
    experiment.delta = delta
    return experiment

    # mask_I_linear = (X_I >= line[0]) & (X_I <= line[1]) & (Y_I >= line[0]) & (Y_I <= line[1])
    #
    # mls_approx_novel_all_2 = mls_approx_novel_all[:(setting.n ** 2)].reshape(setting.n, setting.n)
    # mls_approx_novel_masked_2 = mls_approx_novel_all_2[mask_I_linear]
    # mls_approx_combined_2 = mls_approx_combined[:(setting.n ** 2)].reshape(setting.n, setting.n)
    # mls_approx_combined_masked_2 = mls_approx_combined_2[mask_I_linear]
    #
    # error_plus = Error.calculate_error(mls_approx_novel_masked_2, Z_truth_I[mask_I_linear])
    # error_combined = Error.calculate_error(mls_approx_combined_masked_2, Z_truth_I[mask_I_linear])
    # print(setting)
    # print(error_plus)
    # print(error_combined)
    # print()


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


def run_experiment(n, m, delta, tries, base_function, func):
    aggregated_experiment = AggregatedExperiment()
    with Timer() as timer:
        for i in range(tries):
            if func == 'basic_experiment':
                aggregated_experiment.add(basic_experiment(n=n, m=m, delta=delta, base_function=base_function))
            elif func == 'hetero_experiment':
                aggregated_experiment.add(hetero_experiment(n=n, m=m, delta=delta, base_function=base_function))

    print(f"Experiment took: {timer.elapsed_time}")
    mls_i_time, mls_plus_time, mls_union_time = aggregated_experiment.get_times()
    print(f"mls_i / mls_plus / mls_union : {mls_i_time, mls_plus_time, mls_union_time}")

    return aggregated_experiment


def run_all_experiments():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    tries = 4

    tasks = []
    res_list = ExperimentStore()

    experiment = 'hetero_experiment'

    for base_function in ['sine']:
        for m in [0, 1, 2]:
            for n in [45, 50, 55, 60, 65]:
                for delta in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2]:
                    # res_list.add_experiment(run_experiment(n, m, delta, tries, base_function, experiment))
                    tasks.append((n, m, delta, tries, base_function, experiment))

    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(run_experiment, *arg) for arg in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing'):
            res_list.add_experiment(future.result())

    error_averages = res_list.get_error_df(errors=['rmse', 'score'])
    error_averages.to_csv(f'experiment-{experiment}.csv')
    print("mls_i, mls_plus, mls_union: \n", error_averages)


if __name__ == "__main__":
    # print("Program begin")
    # # error_diff_main()
    # # error_diff_main()
    # # exp = basic_experiment(m=1, n=40, delta=0.4, base_function='sine', plot=True)
    # # print(exp.result_i)
    # # print(exp.result_plus)

    with Timer(verbose=False) as timer:
        run_all_experiments()
    # print(f"Time took total: {timer.elapsed_time}s")

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
    pass
