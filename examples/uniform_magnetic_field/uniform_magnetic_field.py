from itertools import product
import matplotlib.pyplot as plt
import mpl_extras as me
import tpsim as tp
import numpy as np
import warnings
import os


# --------------------------------------------------------------------------- #
#                           Simulation parameters
# --------------------------------------------------------------------------- #
## ---------- Simulation time
# Start time [1/wce]
t_start = 0
# Stop time [1/wce]
t_stop = 2 * np.pi
# Time step [1/wce]
dt = np.pi * 1e-3
# Number of time steps
Nt = int(t_stop / dt)
# Interval to log
log_interval = Nt // 10

## ---------- Background parameters
# Background magnetic field [nT]
B0 = 10
# Number density [1/cc]
n = 5
# Electron cyclotron frequency
wce = tp.cyclotron_frequency(B0)
# Electron plasma frequency
wpe = tp.plasma_frequency(n)

## ---------- Particle parameters
kinetic_energy = np.array([10, 50])
gyrophase = np.array([0, 0])
pitch_angle = np.array([45, 90])
Np = len(kinetic_energy)
# Normalized position
xn, yn, zn = np.zeros((3, Np))
# Normalized velocity
uxn, uyn, uzn = tp.ES2US(
    kinetic_energy,
    np.radians(gyrophase),
    np.radians(pitch_angle)
) / tp.c

## ---------- Electromagnetic field model
## Define the electromagnetic field here (background + perturbations)
def EM_model(t, x, y, z, ux, uy, uz):
    """Returns `Np`-dimensional arrays `Ex`, `Ey`, `Ez`, `Bx`, `By`, `Bz` in
    normalized units.
    """
    Ex = np.zeros(Np)
    Ey = np.zeros(Np)
    Ez = np.zeros(Np)
    Bx = np.zeros(Np)
    By = np.zeros(Np)
    Bz = np.ones(Np)
    return Ex, Ey, Ez, Bx, By, Bz

# Print some information of the script
print(f"""
------------------------------------------------------------------
    Simulation of particle dynamics in uniform magnetic field.
------------------------------------------------------------------
 * dtwce                                    : {dt:.4f}
 * Number of time steps                     : {Nt}
----------
 * Number of particles                      : {Np}
----------
 * Background magnetic field                : {B0} nT
 * Background electron density              : {n} /cc
 * Electron plasma frequency                : {wpe:.4f} rad/s
 * Electron cyclotron frequency             : {wce:.4f} rad/s
 * wpe / wce                                : {wpe / wce:.4f}
------------------------------------------------------------------
""")


# --------------------------------------------------------------------------- #
#                              Post-processing
# --------------------------------------------------------------------------- #
def check_solution(X, Y, Z, UX, UY, UZ, tol=1e-3):
    r"""
    The analytical solution is given by
        x(t) = x_0 + v_\bot \sin(t + \delta)
        y(t) = y_0 + s v_\bot \cos(t + \delta)
        v_x(t) = v_\bot \cos(t + \delta)
        v_y(t) = -s v_\bot \sin(t + \delta)
    """
    s = -1

    T = np.arange(Nt) * dt
    vperp = np.sqrt(uxn ** 2 + uyn ** 2)
    delta = np.arctan2(-s * uyn, uxn)
    # Solve for IC
    x0 = xn - vperp * np.sin(delta)
    y0 = yn - s * vperp * np.cos(delta)
    # Create solution arrays
    XS, YS, ZS, UXS, UYS, UZS = np.zeros((6, Np, Nt))
    # Loop through particles
    for i in range(Np):
        XS[i, :] = x0[i] + vperp[i] * np.sin(T + delta[i])
        YS[i, :] = y0[i] + s * vperp[i] * np.cos(T + delta[i])
        ZS[i, :] = zn[i] + uzn[i] * T
        UXS[i, :] = vperp[i] * np.cos(T + delta[i])
        UYS[i, :] = -s * vperp[i] * np.sin(T + delta[i])
        UZS[i, :] = uzn[i]

    # Check
    assert np.isclose(X, XS, rtol=tol, atol=tol).all()
    assert np.isclose(Y, YS, rtol=tol, atol=tol).all()
    assert np.isclose(Z, ZS, rtol=tol, atol=tol).all()
    assert np.isclose(UX, UXS, rtol=tol, atol=tol).all()
    assert np.isclose(UY, UYS, rtol=tol, atol=tol).all()
    assert np.isclose(UZ, UZS, rtol=tol, atol=tol).all()

    me.setup_mpl(tex=True)
    # Loop through particles
    for i in range(X.shape[0]):
        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=(12, 6), sharex=True)
        fig.subplots_adjust(wspace=0.3)
        fig.suptitle(
            f"KE0 = {kinetic_energy[i]} eV; P0 = {pitch_angle[i]}$^\circ$"
        )

        # Plot solved solutions
        axes[0, 0].plot(T, X[i, :], "-k")
        axes[1, 0].plot(T, Y[i, :], "-k")
        axes[2, 0].plot(T, Z[i, :], "-k")
        axes[0, 1].plot(T, UX[i, :], "-k")
        axes[1, 1].plot(T, UY[i, :], "-k")
        axes[2, 1].plot(T, UZ[i, :], "-k")
        # Plot analytical solutions
        axes[0, 0].plot(T, XS[i, :], "--r")
        axes[1, 0].plot(T, YS[i, :], "--r")
        axes[2, 0].plot(T, ZS[i, :], "--r")
        axes[0, 1].plot(T, UXS[i, :], "--r")
        axes[1, 1].plot(T, UYS[i, :], "--r")
        axes[2, 1].plot(T, UZS[i, :], "--r")
        # Formats
        axes[0, 0].set_ylabel("$x\\Omega_{ce}/c$")
        axes[1, 0].set_ylabel("$y\\Omega_{ce}/c$")
        axes[2, 0].set_ylabel("$z\\Omega_{ce}/c$")
        axes[0, 1].set_ylabel("$u_x/c$")
        axes[1, 1].set_ylabel("$u_y/c$")
        axes[2, 1].set_ylabel("$u_z/c$")
        for (m, n) in np.ndindex(axes.shape):
            ax = axes[m, n]
            ax.tick_params(**me.params)
            ax.set_xlim(T.min(), T.max())
            if n == 2:
                ax.set_xlabel("$t\\Omega_{ce}$")

        fig.savefig(f"particle_{i}.png")
        plt.close(fig)


# --------------------------------------------------------------------------- #
#                               Run simulation
# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    # Initial conditions
    t, x, y, z, ux, uy, uz = t_start, xn, yn, zn, uxn, uyn, uzn
    # History arrays
    X, Y, Z, UX, UY, UZ = np.zeros((6, Np, Nt))
    X[:, 0] = x
    Y[:, 0] = y
    Z[:, 0] = z
    UX[:, 0] = ux
    UY[:, 0] = uy
    UZ[:, 0] = uz

    # Main loop
    print(f"Starting main loop")
    advance = tp.advance
    for n in range(1, Nt):
        # Advance particles
        t, x, y, z, ux, uy, uz = advance(t, x, y, z, ux, uy, uz, EM_model, dt)

        # Save to history arrays
        X[:, n] = x
        Y[:, n] = y
        Z[:, n] = z
        UX[:, n] = ux
        UY[:, n] = uy
        UZ[:, n] = uz

        # Log
        if n % log_interval == 0: print(f"Pushed {n} steps")

    print(f"Done!")

    # Post-processing
    check_solution(X, Y, Z, UX, UY, UZ)

