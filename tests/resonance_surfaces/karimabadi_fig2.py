from scipy.special import jv, jvp
import matplotlib.pyplot as plt
import mpl_extras as me
import tpsim as tp
import numpy as np
import warnings


warnings.filterwarnings("ignore")


e = 0.01
alpha = np.radians(60)
#N = 3.27
N = 298.828
Nx = N * np.sin(alpha)
Nz = N * np.cos(alpha)
#w_wce = 1.6
w_wce = 0.15
H0 = 0.01


def R_surface(pz, l):
    bl = -l * Nz / w_wce
    cl = 1 - Nz ** 2

    rho2 = (l / w_wce) ** 2 - 1 + bl ** 2 / cl
    return np.sqrt(rho2 - cl * (pz + bl / cl) ** 2)


def H_surface(pz, pz0):
    aH = 1 / Nz ** 2 - 1
    H0 = np.sqrt(1 + pz0 ** 2)
    bH = H0 / Nz

    R2 = bH ** 2 / aH + 1 - H0 ** 2
    return np.sqrt(aH * (pz + bH / aH) ** 2 - R2)


#pz = np.linspace(-2.4, 2.4, 10000)
pz = np.linspace(-0.09, 0.09, 10000)

me.setup_mpl(tex=True)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
fig.suptitle(f"$w/\Omega_{{ce}}={w_wce:.2f}$, $N_\parallel={Nz:.2f}$")



ax.plot(pz, H_surface(pz, H0), "-k")

c = ["r", "k", "b"]
#Pz_0 = np.array([0.367, 0.979, 1.591])
Pz_0 = np.array([0.0066])
#for (j, l) in enumerate(range(-1, 2)):
for (j, l) in enumerate([0]):
    ax.plot(pz, R_surface(pz, l), f"--{c[j]}", label=f"$l={l}$")

    Pzr = Nz * (l / w_wce - H0) / (1 - Nz ** 2)
    Ppr = np.sqrt(np.abs(
        ((l / w_wce) ** 2 - (Nz * H0) ** 2 - 1 + Nz ** 2) / (1 - Nz ** 2)
    ))
    g0r = (l / w_wce - H0 * Nz ** 2) / (1 - Nz ** 2)
    rho = Nx * w_wce * Ppr

    dP = 2 * np.sqrt(e) * Nz / np.sqrt(Nz ** 2 - 1) * np.sqrt(np.abs(
        (Pzr * np.sin(alpha) + g0r) * jv(l, rho) + 0.5 * Ppr * (
            (1 - np.cos(alpha)) * jv(l - 1, rho) -
            (1 + np.cos(alpha)) * jv(l + 1, rho)
        )
    ))

    pz_w = np.linspace(Pz_0[j] - dP, Pz_0[j] + dP, 1000)
    #ax.plot(pz_w, H_surface(pz_w, H0), "-r", lw=10)
    ax.axvspan(Pz_0[j] - dP, Pz_0[j] + dP, color="r", alpha=0.2)

ax.set_aspect("equal")
ax.set_xlim(-0.09, 0.09)
ax.set_ylim(0, 0.09)
#ax.set_xlim(-0.4, 2.4)
#ax.set_ylim(0, 1.4)
ax.tick_params(**me.params)
ax.set_xlabel("$P_\\parallel$")
ax.set_ylabel("$P_\\bot$")
ax.legend(ncol=3, bbox_to_anchor=(0.98, 0.98), loc=1, borderaxespad=0)

fig.savefig("karimabadi_fig2.png", dpi=fig.dpi)

