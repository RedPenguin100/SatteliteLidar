from dataclasses import dataclass

import numpy as np

import finufft

from matplotlib import pyplot as plt
from numba import njit

from stepwise_mls.error_utils import mse
from stepwise_mls.timer import Timer
from stepwise_mls.util import generate_2d_base


@dataclass
class Setting:
    base_points: str
    dropout: float
    delta: float


@njit
def point_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def low_pass_filter(frequencies, k_cut):
    idx = np.arange(len(frequencies))
    k = np.where(idx <= len(frequencies) // 2, idx, idx - len(frequencies))
    k_cut = 10
    filt = frequencies.copy()
    filt[np.abs(k) > k_cut] = 0
    return filt


def line_between_points(a, b, t):
    return a + (b - a) * t


if __name__ == '__main__':
    np.random.seed(0)
    line = (0, np.pi)
    gap = 1
    extended_line = (line[0] - 1, line[1] + 1)

    settings = []
    for base_points in ['halton']:
        # settings.append(Setting(base_points=base_points, dropout=0.2, delta=0.1))
        # settings.append(Setting(base_points=base_points, dropout=0.3, delta=0.1))
        # settings.append(Setting(base_points=base_points, dropout=0.4, delta=0.1))
        settings.append(Setting(base_points=base_points, dropout=0, delta=0.2))
        # settings.append(Setting(base_points=base_points, dropout=0.2, delta=0.05))
        # settings.append(Setting(base_points=base_points, dropout=0.2, delta=0.001))

    for setting in settings:
        lines = []
        for intercept in [-1.5, -1, -0.5, 0., 0.5, 1, 1.5]:
            slope = 1
            point_a = (0, intercept)
            point_b = (np.pi, np.pi * slope + intercept)
            d = point_distance(point_a, point_b)
            delta = setting.delta
            num_points = int(d / delta)
            print("Num points: ", num_points)

            t = np.linspace(0, 1, num_points, endpoint=False)
            x = line_between_points(point_a[0], point_b[0], t)
            y = line_between_points(point_a[1], point_b[1], t)

            even_points = num_points // 2 - 1
            to_remove = np.random.choice(even_points, size=int(even_points * setting.dropout),
                                         replace=False)

            mask = np.ones(num_points, dtype=bool)
            mask[to_remove * 2] = False

            t_masked = t[mask]
            lines.append((x, y, t, mask))

        with Timer(verbose=True) as _:
            X_J, Y_J = generate_2d_base(extended_line, 100, base_points=base_points)
            # Z_truth_J = np.ones_like(X_J * X_J, dtype=np.float64)
            Z_truth_J = np.sin(X_J * Y_J * 0.5, dtype=np.float64)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(X_J.ravel(), Y_J.ravel(), Z_truth_J.ravel(), color='orange', alpha=0.5)

        for line in lines:
            x, y, t, mask = line
            x_masked = x[mask]
            y_masked = y[mask]
            t_masked = t[mask]

            # z = np.sin(x * y * 0.5)
            z = np.ones(len(t), dtype=np.float64)
            # z = np.exp(x * y * 0.5)
            z_masked = z[mask]

            odds = np.zeros(num_points, dtype=bool)
            odds[1::2] = True
            z_odd = z[odds]
            z_odd_tag = np.fft.fft(z_odd)
            z_odd_approx = np.fft.ifft(z_odd_tag).real
            print(f"Diff odds: ", mse(z_odd, z_odd_approx))

            # ax.scatter(x[odds], y[odds], z_odd_approx, color='red', alpha=0.5)
            # ax.scatter(x[odds], y[odds], z_odd, color='green', alpha=0.5)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('MLS on set I')

        ver_lines = []
        for i in range(len(lines) - 1):
            first_line = lines[i]
            second_line = lines[i + 1]

            x1, y1, t1, mask1 = first_line
            x2, y2, t2, mask2 = second_line
            z1 = np.sin(x1 * y1 * 0.5)
            z2 = np.sin(x2 * y2 * 0.5)
            for j in range(len(x1) - 1):
                # TODO: deal with t1=t2 assumption
                t = np.linspace(0, 1, 5, endpoint=False)

                x_ver = line_between_points(x1[j], x2[j + 1], t)
                y_ver = line_between_points(y1[j], y2[j + 1], t)
                # Linear interpolation
                z_ver = line_between_points(z1[j], z2[j + 1], t)
                ver_lines.append((x_ver, y_ver, z_ver))

                ax.scatter(x_ver, y_ver, z_ver, color='red', alpha=0.5)

        err = 0
        pts = 0
        for ver_line in ver_lines:
            x_ver, y_ver, z_ver = ver_line
            z_true = np.sin(x_ver * y_ver * 0.5)
            err += np.linalg.norm(z_true - z_ver)
            pts += len(z_true)
        print("Error: ", err / pts)

        plt.show(block=True)

    ## backup code for nufft which seems to work horribly
    """
                w = 2 * np.pi * (t - 0.5)
                w_masked = w[mask]
    
                # M = len(t) * 2
                M = len(t)
                c = finufft.nufft1d1(w_masked, z_masked.astype(np.complex128), M, eps=1e-12, isign=1)
                c_filt = low_pass_filter(c, 5)
    
                z_rec = (finufft.nufft1d2(w, c, eps=1e-12, isign=-1) / M).real
    
                print(f"Diff {setting.delta}", mse(z_rec, z))
    
                ax.scatter(x, y, z_rec.real, color='blue', alpha=0.5)
    """
