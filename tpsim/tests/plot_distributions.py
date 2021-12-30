from ..distribution import Distribution
from ..conversions import KE2S
from ..constants import c
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np


def test_1AU_VDF():

    max_speed = KE2S(1700) / c

    v_para = np.linspace(-max_speed, max_speed, 200)
    v_perp = np.linspace(-max_speed, max_speed, 200)

    bins = (v_para, v_perp)
    VXMESH, VYMESH = np.meshgrid(*bins)
    VX = VXMESH.flatten()
    VY = VYMESH.flatten()

    dist = Distribution(v_perp=VY, v_para=VX, r="1 AU")
    log_norm = LogNorm(vmin=1e-17, vmax=1e-9)

    fig, axes = plt.subplots(1, 4, figsize=(17, 5), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.02, left=0.07, right=0.95)
    fig.suptitle("VDF at 1 AU")

    H_core, _, _ = np.histogram2d(VX, VY, bins=bins, weights=dist.core())
    im_core = axes[0].pcolormesh(
        VXMESH, VYMESH, H_core.T, norm=log_norm, cmap="jet"
    )

    H_halo, _, _ = np.histogram2d(VX, VY, bins=bins, weights=dist.halo())
    im_halo = axes[1].pcolormesh(
        VXMESH, VYMESH, H_halo.T, norm=log_norm, cmap="jet"
    )

    H_strahl, _, _ = np.histogram2d(VX, VY, bins=bins, weights=dist.strahl())
    im_strahl = axes[2].pcolormesh(
        VXMESH, VYMESH, H_strahl.T, norm=log_norm, cmap="jet"
    )

    H_total, _, _ = np.histogram2d(VX, VY, bins=bins, weights=dist.total())
    im_total = axes[3].pcolormesh(
        VXMESH, VYMESH, H_total.T, norm=log_norm, cmap="jet"
    )
    cb = fig.colorbar(
        im_total,
        ax=axes,
        pad=0.01,
        fraction=0.02,
        shrink=0.5,
    )
    cb.set_label(r"$f(v_z,v_{\bot})$ [$cm^{-3} km^{-3} s^3$]")

    axes[0].set_title("Core")
    axes[1].set_title("Halo")
    axes[2].set_title("Strahl")
    axes[3].set_title("Total")

    for j, ax in enumerate(axes):
        ax.set_aspect("equal")
        ax.set_xlabel("$v_z/c$")
        if j == 0:
            ax.set_ylabel(r"$v_\bot/c$")
        else:
            ax.set_yticklabels([])
        ax.set_xlim(-max_speed, max_speed)
        ax.set_ylim(-max_speed, max_speed)
        ax.axhline(0, color="k")
        ax.axvline(0, color="k")

    fig.savefig(f"plots/test_1AU_VDF.png", dpi=fig.dpi)
    plt.close(fig)


def test_03AU_VDF():

    max_speed = 0.14

    v_para = np.linspace(-max_speed, max_speed, 200)
    v_perp = np.linspace(-max_speed, max_speed, 200)

    bins = (v_para, v_perp)
    VXMESH, VYMESH = np.meshgrid(*bins)
    VX = VXMESH.flatten()
    VY = VYMESH.flatten()

    dist = Distribution(v_perp=VY, v_para=VX, r="0.3 AU")
    log_norm = LogNorm(vmin=1e-17, vmax=1e-9)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.02, left=0.07, right=0.95)
    fig.suptitle("VDF at 0.3 AU")

    H_core, _, _ = np.histogram2d(VX, VY, bins=bins, weights=dist.core())
    im_core = axes[0].pcolormesh(
        VXMESH, VYMESH, H_core.T, norm=log_norm, cmap="jet"
    )

    H_strahl, _, _ = np.histogram2d(VX, VY, bins=bins, weights=dist.strahl())
    im_strahl = axes[1].pcolormesh(
        VXMESH, VYMESH, H_strahl.T, norm=log_norm, cmap="jet"
    )

    H_total, _, _ = np.histogram2d(VX, VY, bins=bins, weights=dist.total())
    im_total = axes[2].pcolormesh(
        VXMESH, VYMESH, H_total.T, norm=log_norm, cmap="jet"
    )
    cb = fig.colorbar(
        im_total,
        ax=axes,
        pad=0.01,
        fraction=0.02,
        shrink=0.5,
    )
    cb.set_label(r"$f(v_z,v_{\bot})$ [$cm^{-3} km^{-3} s^3$]")

    axes[0].set_title("Core")
    axes[1].set_title("Strahl")
    axes[2].set_title("Total")
    axes[0].set_ylabel(r"$v_\bot/c$")
    for j, ax in enumerate(axes):
        ax.set_aspect("equal")
        ax.set_xlabel("$v_z/c$")
        ax.set_xlim(-max_speed, max_speed)
        ax.set_ylim(-max_speed, max_speed)
        ax.axhline(0, color="k")
        ax.axvline(0, color="k")

    fig.savefig(f"plots/test_03AU_VDF.png", dpi=fig.dpi)
    plt.close(fig)
