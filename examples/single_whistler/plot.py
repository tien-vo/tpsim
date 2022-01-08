from pathos.pools import _ProcessPool as Pool
from contextlib import closing
from scipy.special import jv
import matplotlib.pyplot as plt
import mpl_extras as me
import tables as tb
import tpsim as tp
import numpy as np
import deck as d
import warnings
import os


if not os.path.exists("plot_frames"): os.mkdir("plot_frames")
warnings.filterwarnings("ignore")


def get_state(n):

    with tb.open_file(f"particles/{n}.h5") as f:
        t = f.root.t[0]
        ux = f.root.ux[:]
        uy = f.root.uy[:]
        uz = f.root.uz[:]

    g = np.sqrt(1 + ux ** 2 + uy ** 2 + uz ** 2)
    v_para = uz / g
    v_perp = np.sqrt(ux ** 2 + uy ** 2) / g
    return t, v_para, v_perp


def get_H_surface(vr0):
    alpha = np.linspace(0, 2 * np.pi, 1000)
    return tp.H_surface(alpha, vr0, Nz=d.Nz, w=d.w_wce)


# --------------------------------------------------------------------------- #
#                         Calculate resonance width
# --------------------------------------------------------------------------- #
Nx, Nz = d.Nx, d.Nz
N = np.sqrt(Nx ** 2 + Nz ** 2)
E0n = d.E0n
w_wce = d.w_wce
theta = d.theta
pex, pey, pez, pbx, pby, pbz = d.pex, d.pey, d.pez, d.pbx, d.pby, d.pbz
Exw = E0n / np.sqrt(pex ** 2 + pez ** 2)
e1 = (np.cos(theta) * pex - np.sin(theta) * pez) * Exw / w_wce
e2 = Exw * pey / w_wce
e3 =-(np.sin(theta) * pex + np.cos(theta) * pez) * Exw / N / w_wce


def v_res(n):
    return 1 / Nz * (1 - n / w_wce)


def dv(_vz0, _vp0, n):
    _g0 = 1 / np.sqrt(1 - _vp0 ** 2 - _vz0 ** 2)
    a = w_wce * Nx * _vp0

    C1 = 2 * Nz / np.sqrt(Nz ** 2 - 1)
    C2 = (_vz0 * np.sin(theta) * e1 + _g0 * e3) * jv(n, a)
    C3 = (
        (e2 + e1 * np.cos(theta)) * jv(n - 1, a)
        - (e2 - e1 * np.cos(theta)) * jv(n + 1, a)
    )
    return C1 * np.sqrt(np.abs(-C2 + C3 * _vp0 / 2))


# --------------------------------------------------------------------------- #
#                         Plot method
# --------------------------------------------------------------------------- #
def plot(plot_idx):

    with tb.open_file(f"particles/{plot_idx}.h5") as f:
        t = f.root.t[0]
        ux = f.root.ux[:]
        uy = f.root.uy[:]
        uz = f.root.uz[:]

    g = np.sqrt(1 + ux ** 2 + uy ** 2 + uz ** 2)
    v_para = uz / g
    v_perp = np.sqrt(ux ** 2 + uy ** 2) / g

    me.setup_mpl(tex=True, fontsize=20, cache=True)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 9))
    fig.suptitle(f"$t/T_w$ = {t * w_wce / 2 / np.pi:.1f}")
    sizes = np.ones(d.Np)
    sizes[idx0] = 20

    im = axes[0].scatter(
        v_para, v_perp, c=np.degrees(P0), cmap="jet", s=sizes
    )
    cb0 = fig.colorbar(im, cax=me.add_cbar(axes[0]))
    cb0.set_label("Initial pitch angle [deg]")

    im = axes[1].scatter(
        v_para[idx0], v_perp[idx0], c=S0[idx0], cmap="jet", s=20
    )
    cb1 = fig.colorbar(im, cax=me.add_cbar(axes[1]))
    cb1.set_label("Initial speed [$c$]")

    for j, ax in enumerate(axes):
        # Plot resonance islands
        for (i, n) in enumerate(ns_plot):
            _vz0 = v_res(n)
            dv_res = dv(_vz0, _vp0, n)
            ax.plot(_vz0 + dv_res, _vp0, "-k", alpha=0.2)
            ax.plot(_vz0 - dv_res, _vp0, "-k", alpha=0.2)

        ax.set_xlim(-smax, smax)
        ax.set_ylim(0, smax)
        ax.set_aspect("equal")
        ax.set_facecolor("gray")
        ax.tick_params(**me.params)
        if j == 1: ax.set_xlabel("$v_z/c$")
        ax.set_ylabel("$v_\perp/c$")

    plot_idx //= d.save_interval
    fig.savefig(f"plot_frames/{plot_idx}.png")
    plt.close(fig)
    print(f"Plottting frame #{plot_idx}")


if __name__ == "__main__":

    indices = np.arange(0, d.Nt, d.save_interval)
    P0 = d.P
    S0 = d.S
    ux0, uy0, uz0 = d.uxn, d.uyn, d.uzn
    g0 = np.sqrt(1 + ux0 ** 2 + uy0 ** 2 + uz0 ** 2)
    vz0 = uz0 / g0
    idx0 = np.where((np.degrees(P0) == 0) | (np.degrees(P0) == 180))
    smax = max(d.S_range) * 1.1
    _vp0 = np.linspace(0, smax, 10000)

    ns_plot = np.arange(-3, 4)
    with closing(Pool(8, maxtasksperchild=8)) as p:
        for _ in p.imap_unordered(plot, indices):
            pass

