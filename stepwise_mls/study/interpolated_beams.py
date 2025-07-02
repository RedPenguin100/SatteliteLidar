from dataclasses import dataclass

import numpy as np

import finufft
from matplotlib import pyplot as plt
from numba import njit

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


if __name__ == '__main__':
    np.random.seed(0)
    line = (0, np.pi)
    gap = 1
    extended_line = (line[0] - 1, line[1] + 1)

    settings = []
    for base_points in ['halton']:
        settings.append(Setting(base_points=base_points, dropout=0.1, delta=0.1))
        settings.append(Setting(base_points=base_points, dropout=0.1, delta=0.01))
        settings.append(Setting(base_points=base_points, dropout=0.1, delta=0.001))

    for setting in settings:
        lines = []
        for intercept in [0., -1, 1, -1.5, 1.5]:
            slope = 1
            point_a = (0, intercept)
            point_b = (np.pi, np.pi * slope + intercept)
            d = point_distance(point_a, point_b)
            delta = setting.delta
            num_points = int(d / delta)

            t = np.linspace(0, 1, num_points)
            x = point_a[0] + (point_b[0] - point_a[0]) * t
            y = point_a[1] + (point_b[1] - point_a[1]) * t

            to_remove = np.random.choice(num_points, size=int(num_points * setting.dropout),
                                         replace=False)

            mask = np.ones(num_points, dtype=bool)
            mask[to_remove] = False

            t_remain = t[mask]
            x = x[mask]
            y = y[mask]
            lines.append((x, y, t, mask))

        with Timer(verbose=True) as _:
            X_J, Y_J = generate_2d_base(extended_line, 50, base_points=base_points)
            Z_truth_J = np.sin(X_J * Y_J * 2)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(X_J.ravel(), Y_J.ravel(), Z_truth_J.ravel(), color='orange', alpha=0.5)

        for line in lines:
            x, y, t, mask = line
            t_remain = t[mask]

            z = np.sin(x * y * 2)
            # z_remain = z[mask]

            w = 2 * np.pi * (t_remain - 0.5)
            # c = np.fft.fft(z)
            # z_rec = np.fft.ifft(c)
            sp_size = 32
            upsampfac=3
            spread_kerevalmeth=0

            c = finufft.nufft1d1(w, z.astype(np.complex128), len(t_remain), eps=1e-14, isign=1,
                                 upsampfac=upsampfac,
                                 spread_kerevalmeth=spread_kerevalmeth,
                                 spread_max_sp_size=sp_size
                                 )
            z_rec = finufft.nufft1d2(w, c, eps=1e-14, isign=-1,
                                     upsampfac=upsampfac,
                                     spread_kerevalmeth=spread_kerevalmeth,
                                     spread_max_sp_size=sp_size
                                     ) / len(t_remain)
            print(f"RMSE {setting.delta}: ", np.linalg.norm((z_rec - z)) / np.sqrt(len(z)))
            print(f"Mean error {setting.delta}: ", np.mean(np.abs(z_rec - z)))

            ax.scatter(x, y, z_rec.real, color='blue', alpha=0.5)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('MLS on set I')

        plt.show(block=True)
