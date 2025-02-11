import numpy as np

import matplotlib.pyplot as plt
from fontTools.unicodedata import block

from util import moving_least_squares

n = 100
x = np.linspace(0, 2 * np.pi, n)

y = np.sin(x) + 0.1 * np.random.randn(n)

m = 0
c_shepard = moving_least_squares(x, y, m=0, delta=1)
c_linear = moving_least_squares(x, y, m=1, delta=1)

x_high_res = np.linspace(0, 2 * np.pi, n * 10)
y_high_res = np.sin(x_high_res)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(x, y, label='Noisy data', s=10)
ax.plot(x_high_res, y_high_res, color="red", label="True Function")
# plt.step(x, c, color="green", label="Moving least squares")
ax.plot(x, c_shepard, color="green", label="Moving least squares")

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

ax2.scatter(x, y, label='Noisy data', s=10)
ax2.plot(x_high_res, y_high_res, color="red", label="True Function")
# plt.step(x, c, color="green", label="Moving least squares")
ax2.plot(x, c_linear, color="green", label="Moving least squares")


# plt.legend()
plt.show(block=True)
