import time
import numpy as np
from error_utils import rmse, Error
from matplotlib import pyplot as plt

from mls import moving_least_squares
from timer import Timer


def get_sine_data_2d(n, line, dev, bias: float = 0.):
    start, end = line
    base = np.linspace(start, end, n)
    X, Y = np.meshgrid(base, base)
    Z = np.sin(X * Y / (2 * np.pi))
    noise = np.random.normal(bias, scale=dev, size=Z.shape)
    Z = Z + noise
    return X, Y, Z


class MLSResult:
    def __init__(self):
        self.time = None
        self.error = None


class Experiment:
    def __init__(self):
        self.result_i = None
        self.result_plus = None
        self.result_union = None

        self.n = None
        self.m = None
        self.delta = None
        self.base_function = None


class AggregatedExperiment:
    def __init__(self):
        self.results_i = []
        self.results_plus = []
        self.results_union = []
        self.base_function = None
        self.m = None
        self.n = None
        self.delta = None

    def add(self, experiment: Experiment):
        assert len(self.results_i) == len(self.results_plus)
        assert len(self.results_i) == len(self.results_union)
        self.results_i.append(experiment.result_i)
        self.results_plus.append(experiment.result_plus)
        self.results_union.append(experiment.result_union)

        if self.base_function is None:
            self.base_function = experiment.base_function
        elif self.base_function != experiment.base_function:
            self.base_function = "INVALID"

        if self.m is None:
            self.m = experiment.m
        elif self.m != experiment.m:
            self.m = "INVALID"

        if self.n is None:
            self.n = experiment.n
        elif self.n != experiment.n:
            self.n = "INVALID"

        if self.delta is None:
            self.delta = experiment.delta
        elif self.delta != experiment.delta:
            self.delta = "INVALID"

    def size(self):
        return len(self.results_i)

    def get_times(self):
        mls_i_time = 0
        mls_plus_time = 0
        mls_union_time = 0
        for i in range(len(self.results_i)):
            mls_i_time += self.results_i[i].time
            mls_plus_time += self.results_plus[i].time
            mls_union_time += self.results_union[i].time

        return (mls_i_time, mls_plus_time, mls_union_time)

    def get_error_average(self, error='rmse'):
        mls_i_error = 0
        mls_plus_error = 0
        mls_union_error = 0

        for i in range(len(self.results_i)):
            if error == 'rmse':
                mls_i_error += self.results_i[i].error.root_err
                mls_plus_error += self.results_plus[i].error.root_err
                mls_union_error += self.results_union[i].error.root_err
            else:
                raise ValueError(f"Unknown error type: {error}")

        return mls_i_error, mls_plus_error, mls_union_error


class ExperimentStore:
    def __init__(self):
        self.experiments = dict()

    def add_experiment(self, experiment: AggregatedExperiment):
        if (experiment.n, experiment.m, experiment.delta, experiment.base_function) in self.experiments:
            self.experiments[(experiment.n, experiment.m, experiment.delta, experiment.base_function)].add(experiment)
        else:
            self.experiments[(experiment.n, experiment.m, experiment.delta, experiment.base_function)] = experiment

    def get_error_df(self, error='rmse'):
        import pandas as pd

        df = pd.DataFrame(columns=['n', 'm', 'delta', 'base_function', 'mls_i', 'mls_plus', 'mls_union'])

        for setting, experiment in self.experiments.items():
            n, m, delta, base_function = setting
            error_averages = experiment.get_error_average(error)
            df.loc[len(df)] = (n, m, delta, base_function, error_averages[0], error_averages[1], error_averages[2])

        return df


def basic_experiment(n, m, delta, base_points='halton', base_function='sine', plot=False):
    line = (0, np.pi)
    gap = 1
    extended_line = (line[0] - 2 * gap, line[1] + 2 * gap)

    small_n = n
    # I_base = np.random.uniform(extended_line[0], extended_line[1], small_n)
    if base_points == 'uniform':
        X_base = np.random.uniform(extended_line[0], extended_line[1], (small_n, small_n))
        Y_base = np.random.uniform(extended_line[0], extended_line[1], (small_n, small_n))
        X_J = np.random.uniform(extended_line[0], extended_line[1], (small_n, small_n))
        Y_J = np.random.uniform(extended_line[0], extended_line[1], (small_n, small_n))
    elif base_points == 'halton':
        from scipy.stats import qmc
        sampler = qmc.Halton(d=2, scramble=False)
        base = sampler.random(n=small_n ** 2)
        base = qmc.scale(base, extended_line[0], extended_line[1])
        base_J = sampler.random(n=small_n ** 2)
        base_J = qmc.scale(base_J, extended_line[0], extended_line[1])
        X_base, Y_base = base[:, 0].reshape(small_n, small_n), base[:, 1].reshape(small_n, small_n)
        X_J, Y_J = base_J[:, 0].reshape(small_n, small_n), base_J[:, 1].reshape(small_n, small_n)
    else:
        ValueError(f'Unknown base_points={base_points} argument')

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
        stacked_XY = np.vstack((XY_base, XY_J))
        stacked_Z = np.vstack((Z_base, Z_J)).ravel()
        I_J_approx_at_base = moving_least_squares(stacked_XY, stacked_Z, m=m, delta=delta, x_eval=XY_base)
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
        ax.plot_trisurf(X_base_masked, Y_base_masked, MLS_approx_at_error_masked, label='Approximated error', color='orange', alpha=0.5)

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
        ax.scatter(X_base_masked, Y_base_masked, I_approx_at_base[mask_base] - MLS_approx_at_error_masked, label='Improved approximation', color='green', alpha=0.5)
        ax.scatter(X_base_masked, Y_base_masked, I_approx_at_base[mask_base], label='Approximation on I', color='blue', alpha=0.5)

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
    for base_function in ['sine', 'exp']:
        for m in [0, 1, 2]:
            for n in [45, 50, 55, 60, 65]:
                for delta in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2]:
                    # res_list.add_experiment(run_experiment(n, m, delta, tries, base_function, 'basic_experiment'))
                    tasks.append((n, m, delta, tries, base_function, 'basic_experiment'))

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_experiment, *arg) for arg in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing'):
            res_list.add_experiment(future.result())

    error_averages = res_list.get_error_df(error='rmse')
    error_averages.to_csv('experiments-halton.csv')
    print("mls_i, mls_plus, mls_union: \n", error_averages)


if __name__ == "__main__":
    print("Program begin")
    # error_diff_main()
    # error_diff_main()
    exp = basic_experiment(m=1, n=40, delta=0.4, base_function='sine', plot=True)
    print(exp.result_i)
    print(exp.result_plus)

    start = time.time()
    # run_all_experiments()
    end = time.time()
    print(f"Time took total: {end - start}s")

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
