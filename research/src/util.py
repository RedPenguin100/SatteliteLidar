import numpy as np
import matplotlib.pyplot as plt


def gaussian_weights(x, x_i, h):
    norm = np.linalg.norm(x - x_i)
    pow = -np.power(norm / h, 2)
    return np.exp(pow)


def calc_s(x: np.array, y: np.array, h):
    s = 0.

    for i in range(len(x)):
        w = 0.
        for j in range(len(x)):
            w += gaussian_weights(x, x[j], h)
        w = gaussian_weights(x, x[i], h) / w
        s += y[i] * w
    return s


def find_close(x, index, delta):
    x = np.atleast_2d(x)
    diff = np.subtract(x, x[:, index])
    distances = np.linalg.norm(diff, axis=0)
    indices = np.where(distances < delta)[0]
    return indices


def shepard_kernel(x, y, delta):
    kernel = np.empty_like(x)

    h = len(x)

    # TODO: efficiency
    for z in range(len(x)):
        indices = find_close(x, z, delta)
        print(indices)
        s = calc_s(x[indices], y[indices], h)
        kernel[z] = s

    return kernel


def moving_least_squares(x: np.array, y: np.array, m=0, delta: float = 1) -> np.array:
    if m == 0:
        return shepard_kernel(x, y, delta)


n = 100
x = np.linspace(0, 2 * np.pi, n)

y = np.sin(x) + 0.1 * np.random.randn(n)

m = 0
c = moving_least_squares(x, y, m=0, delta=1)

x_high_res = np.linspace(0, 2 * np.pi, n * 10)
y_high_res = np.sin(x_high_res)

plt.scatter(x, y, label='Noisy data', s=10)
plt.plot(x_high_res, y_high_res, color="red", label="True Function")
# plt.step(x, c, color="green", label="Moving least squares")
plt.plot(x, c, color="green", label="Moving least squares")

plt.legend()
plt.show()
