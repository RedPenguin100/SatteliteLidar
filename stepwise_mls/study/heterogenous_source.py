from dataclasses import dataclass

import numpy as np

from stepwise_mls.error_utils import Error
from stepwise_mls.mls import mls_novel, mls_combined, mls_novel_all, mls_combined_all, mls_novel_all_3
from stepwise_mls.timer import Timer
from stepwise_mls.util import generate_2d_base


def add_noise(Z, mean, stddev, generator='normal'):
    if generator == 'normal':
        return np.random.normal(mean, stddev, Z.shape)
    else:
        raise ValueError('Unknown generator {}'.format(generator))


@dataclass
class Setting:
    base_points: str
    delta: float
    m: int
    n: int


if __name__ == '__main__':
    np.random.seed(0)
    line = (0, np.pi)
    gap = 1
    extended_line = (line[0] - gap, line[1] + gap)

    settings = []
    for base_points in ['halton']:
        for delta in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for n in [120]:
                for m in [2]:
                    settings.append(Setting(base_points=base_points, delta=delta, m=m, n=n))

    for setting in settings:
        mean_I, mean_J, mean_K = 0, 0, 0
        stddev_I, stddev_J, stddev_K = 0.3, 0.1, 0.3

        X_I, Y_I = generate_2d_base(extended_line, (setting.n * 2) // 3, base_points=base_points)
        Z_truth_I = np.sin(X_I * Y_I * 2)
        Z_I = Z_truth_I + np.random.normal(mean_I, stddev_I, X_I.shape)
        XY_I = np.column_stack((X_I.ravel(), Y_I.ravel()))

        X_J, Y_J = generate_2d_base(extended_line, setting.n // 3, base_points=base_points)
        Z_truth_J = np.sin(X_J * Y_J * 2)
        Z_J = Z_truth_J + np.random.normal(mean_J, stddev_J, X_J.shape)
        XY_J = np.column_stack((X_J.ravel(), Y_J.ravel()))

        X_K, Y_K = generate_2d_base(extended_line, setting.n // 3, base_points=base_points)
        Z_truth_K = np.sin(X_K * Y_K * 2)
        Z_K = Z_truth_K + np.random.normal(mean_K, stddev_K, X_K.shape)
        XY_K = np.column_stack((X_K.ravel(), Y_K.ravel()))

        # Consider if this is right
        with Timer(verbose=True) as timer:
            mls_approx_novel_all = mls_novel_all_3(XY_I, Z_I, XY_J, Z_J, XY_K, Z_K, setting.m, setting.delta)
        with Timer(verbose=True) as timer:
            stacked_XY_super = np.vstack([XY_I, XY_J, XY_K])
            stacked_Z_super = np.concatenate([Z_I.ravel(), Z_J.ravel(), Z_K.ravel()])

            mls_approx_combined = mls_combined_all(stacked_XY_super, stacked_Z_super,
                                                   setting.m, setting.delta)

        stacked_XY = np.vstack((XY_I, XY_J))
        stacked_Z = np.concatenate((Z_I.ravel(), Z_J.ravel()))

        mask_I = (XY_I >= line[0]) & (XY_I <= line[1])
        mask_I = np.logical_and(mask_I[:, 0], mask_I[:, 1])

        mask_IJ = (stacked_XY >= line[0]) & (stacked_XY <= line[1])
        mask_IJ = np.logical_and(mask_IJ[:, 0], mask_IJ[:, 1])

        mask_IJK = (stacked_XY_super >= line[0]) & (stacked_XY_super <= line[1])
        mask_IJK = np.logical_and(mask_IJK[:, 0], mask_IJK[:, 1])

        mls_approx_novel_masked = mls_approx_novel_all[mask_IJK]

        mls_approx_combined_masked = mls_approx_combined[mask_IJK]

        # error_plus = Error.calculate_error(mls_approx_novel_masked, Z_truth_I[mask_I])
        # error_combined = Error.calculate_error(mls_approx_combined_masked, Z_truth_I[mask_I])

        error_plus = Error.calculate_error(mls_approx_novel_masked, stacked_Z_super[mask_IJK])
        error_combined = Error.calculate_error(mls_approx_combined_masked, stacked_Z_super[mask_IJK])

        print(setting)
        print(error_plus)
        print(error_combined)
        print()

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
