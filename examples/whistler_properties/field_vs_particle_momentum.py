import matplotlib.pyplot as plt
import astropy.constants as c
import astropy.units as u
import mpl_extras as me
import tpsim as tp
import numpy as np


def A(theta, B0=10, n=5, w_wce=0.15):
    wce = tp.cyclotron_frequency(B0, particle="e-")
    E0 = tp.normalize_E(20, B0)
    V_max = tp.KE2S(1000)

    theta = np.radians(theta)
    Nx, Nz, pex, pey, pez, pbx, pby, pbz = tp.whistler_polarization(
        theta, B0, n, w_wce * wce
    )
    Ex = E0 / np.sqrt(pex ** 2 + pez ** 2)
    d1 = (np.cos(theta) * pex - np.sin(theta) * pez) * Ex / w_wce
    d2 = Ex * pey / w_wce
    return d2 * tp.c / V_max


P = np.linspace(-90, 90, 1000)

me.setup_mpl(tex=True)
fig, ax = plt.subplots(1, 1, figsize=(8, 3))
fig.subplots_adjust(bottom=0.2)
ax.plot(P, A(P, B0=10, n=5), "-k", label="1 AU")
ax.plot(P, A(P, B0=60, n=350), "-r", label="0.3 AU")
ax.set_xlim(P.min(), P.max())
ax.set_ylim(0, 0.8)
ax.set_xlabel("$\\theta$ [deg]")
ax.set_ylabel(r"$|qA/mu_{\max}|$")
ax.tick_params(**me.params)
ax.legend()

fig.savefig(f"field_vs_particle.png", dpi=fig.dpi)
