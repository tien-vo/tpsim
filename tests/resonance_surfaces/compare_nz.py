import matplotlib.pyplot as plt
import mpl_extras as me
import tpsim as tp
import numpy as np
import warnings


warnings.filterwarnings("ignore")


alpha = np.linspace(0, 2 * np.pi, 1000)

me.setup_mpl(fontsize=22, tex=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.subplots_adjust(wspace=0.3)

W = [0.15, 0.15]
NZ = [20, 200]


for (j, l) in enumerate(range(0, 6)):
    vrp, _ = tp.R_surface(l, Nz=NZ[0], w=W[0])
    vz_H, vp_H = tp.H_surface(alpha, vrp, Nz=NZ[0], w=W[0])
    axes[0].plot(vz_H, vp_H, "-k")
    axes[0].scatter([vrp], [0], c="w", marker="X", edgecolors="k", s=100)

    vrp, _ = tp.R_surface(l, Nz=NZ[1], w=W[1])
    vz_H, vp_H = tp.H_surface(alpha, vrp, Nz=NZ[1], w=W[1])
    axes[1].plot(vz_H, vp_H, "-k")
    axes[1].scatter([vrp], [0], c="w", marker="X", edgecolors="k", s=100)


for (j, ax) in enumerate(axes):
    ax.set_title(
        f"$\omega/\Omega_{{ce}}={W[j]:.2f}$, $N_\parallel={NZ[j]:.2f}$"
    )
    ax.set_aspect("equal")
    if j == 0:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    else:
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
    ax.tick_params(**me.params)
    ax.set_xlabel("$v_\\parallel/c$")
    ax.set_ylabel("$v_\\bot/c$")
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")

fig.savefig(f"compare_nz.png", dpi=fig.dpi)
