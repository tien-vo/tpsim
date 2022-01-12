from scipy.optimize import curve_fit
from scipy.special import jv
import matplotlib.pyplot as plt
import mpl_extras as me
import tpsim as tp
import numpy as np


def calculate_C(E0, B0, n, w_wce=0.15, theta=65):

    theta = np.radians(theta)
    smax = tp.KE2S(3000) / tp.c
    wce = tp.cyclotron_frequency(B0, particle="e-")
    wpe = tp.plasma_frequency(n, particle="e-")

    E0n = tp.normalize_E(E0, B0)
    Nx, Nz, pex, pey, pez, pbx, pby, pbz = tp.whistler_polarization(
        theta, B0, n, w_wce * wce,
    )
    # Scaling factor
    Exw = E0n / np.sqrt(pex ** 2 + pez ** 2)
    # Normalized wave amplitudes
    N = np.sqrt(Nx ** 2 + Nz ** 2)
    e1 = (np.cos(theta) * pex - np.sin(theta) * pez) * Exw / w_wce
    e2 = Exw * pey / w_wce
    e3 =-(np.sin(theta) * pex + np.cos(theta) * pez) * Exw / N / w_wce

    def v_res(l):
        return 1 / Nz * (1 - l / w_wce)

    def dv(vz0, vp0, l):
        """ Resonance width """
        g0 = 1 / np.sqrt(1 - vp0 ** 2 - vz0 ** 2)
        a = w_wce * Nx * vp0

        C1 = 2 * Nz / np.sqrt(Nz ** 2 - 1)
        C2 = (vz0 * np.sin(theta) * e1 + g0 * e3) * jv(l, a)
        C3 = (
            (e2 + e1 * np.cos(theta)) * jv(l - 1, a)
            - (e2 - e1 * np.cos(theta)) * jv(l + 1, a)
        )
        return C1 * np.sqrt(np.abs(-C2 + C3 * vp0 / 2))

    vp0 = np.linspace(0, smax, 10000)
    delta_p = Nz / (1 - Nz ** 2) / w_wce

    ls = np.arange(-2, 3)
    C = np.zeros(ls.shape)
    for (i, l) in enumerate(ls):
        vz0 = v_res(l)
        d = dv(vz0, vp0, l)
        C[i] = np.max(np.abs(2 * d / delta_p))

    return Exw * pby, np.max(C)


def fit_model(x, a, b):
    return a * x + b


calculate_C_vec = np.vectorize(calculate_C)


E = np.linspace(0.01, 20, 100)
dB_1, C_1 = calculate_C_vec(E, B0=10, n=5, theta=65)
dB_2, C_2 = calculate_C_vec(E, B0=60, n=350, theta=65)
p1, pc1 = curve_fit(fit_model, np.log(C_1), np.log(dB_1))
p2, pc2 = curve_fit(fit_model, np.log(C_2), np.log(dB_2))
perr1 = np.sqrt(np.diag(pc1))
perr2 = np.sqrt(np.diag(pc2))


me.setup_mpl(tex=True)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
fig.suptitle(r"$\theta=65^\circ$; Fit model: $\delta B/B_0=\alpha C^\beta$")
ax.plot(
    C_1, dB_1, "-k", lw=3,
    label=fr"1 AU parameters: $\alpha={np.exp(p1[1]):.1f},\beta={p1[0]}$",
)
ax.plot(
    C_2, dB_2, "--r", lw=3,
    label=fr"0.3 AU parameters: $\alpha={np.exp(p2[1]):.1f},\beta={p2[0]}$",
)

ax.set_xlabel(r"$C=|2\Delta\hat{P}_\parallel/\delta \hat{P}_\parallel|$")
ax.set_ylabel("$\delta B/B_0$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
ax.tick_params(which="both", **me.params)

fig.savefig("C_oblique.png")
