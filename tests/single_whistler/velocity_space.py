import matplotlib.pyplot as plt
import mpl_extras as me
import deck as d
import tpsim as tp
import tables as tb
import numpy as np
import warnings


warnings.filterwarnings("ignore")


def get_state(n):

    with tb.open_file("particles/0.h5") as f:
        ux0 = f.root.ux[:]
        uy0 = f.root.uy[:]
        uz0 = f.root.uz[:]
        # Convert to KE0
        KE0, _, _ = tp.US2ES(ux0, uy0, uz0)

    with tb.open_file(f"particles/{n}.h5") as f:
        t = f.root.t[0]
        ux = f.root.ux[:]
        uy = f.root.uy[:]
        uz = f.root.uz[:]

    g = np.sqrt(1 + ux ** 2 + uy ** 2 + uz ** 2)
    v_para = uz / g
    v_perp = np.sqrt(ux ** 2 + uy ** 2) / g
    return t, v_para, v_perp, KE0


if __name__ == "__main__":

    indices = np.arange(0, d.Nt, d.save_interval)

    plt.rc("font", **dict(family="serif", size=12))
    fig, ax = plt.subplots(1, 1)

    for n in indices:
        plt.cla()

        t, v_para, v_perp, KE0 = get_state(n)
        ax.scatter(v_para, v_perp, c=KE0, cmap="jet", s=5)

        fig.suptitle(f"$t\\Omega_{{ce}}$ = {t:.2f}")
        ax.set_xlim(-0.08, 0.08)
        ax.set_ylim(0, 0.08)
        ax.grid()
        ax.set_aspect("equal")
        ax.set_facecolor("gray")
        ax.tick_params(**me.params)
        ax.set_xlabel("$v_z/c$")
        ax.set_ylabel("$v_\perp/c$")

        plt.pause(0.01)

    plt.show()
