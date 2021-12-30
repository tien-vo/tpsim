from deck import e1, e2, e3, theta, Nx, Nz, w_wce
from scipy.special import jv
import matplotlib.pyplot as plt
import tpsim as tp
import numpy as np


def dv(vz0, vp0, n=1):
    g0 = 1 / np.sqrt(1 - vp0 ** 2 - vz0 ** 2)
    a = w_wce * Nx * vp0

    C1 = 2 * Nz / np.sqrt(Nz ** 2 - 1)
    C2 = (vz0 * np.sin(theta) * e1 + g0 * e3) * jv(n, a)
    C3 = (
        (e2 + e1 * np.cos(theta)) * jv(n - 1, a)
        - (e2 - e1 * np.cos(theta)) * jv(n + 1, a)
    )

    return C1 * np.sqrt(-C2 + C3 * vp0 / 2)


lim = 0.15
vp0 = np.linspace(0, lim, 10000)

fig, ax = plt.subplots(1, 1)

resonances = np.arange(-2, 3)
colors = ["r", "k", "b", "g", "y"]
for i, n in enumerate(resonances):
    vz0 = 1 / Nz * (1 - n / w_wce)
    d = dv(vz0, vp0, n=n)
    v2 = vz0 + d
    v1 = vz0 - d
    ax.plot(v2, vp0, c=colors[i], label=f"$n={n}$")
    ax.plot(v1, vp0, c=colors[i])

ax.set_aspect("equal")
ax.set_xlim(-lim, lim)
ax.legend()
plt.show()
