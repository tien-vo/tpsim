r"""
    Main script for running simulation. Include processing functions here.
"""


import matplotlib.pyplot as plt
import mpl_extras as me
import deck as d
import tpsim as tp
import numpy as np


#######################################################
# Assignments to reduce namespace in loop
#######################################################
# Pusher
advance = tp.f_advance
# Model
EM_model = d.EM_model
# Time
Nt = d.Nt
dt = d.dt
save_interval = d.save_interval

#######################################################
# Processing functions
#######################################################
def check_solution(X, Y, Z, UX, UY, UZ, tol=1e-3, plot=False):
    r"""
    The analytical solution is given by
        x(t) = x_0 + v_\bot \sin(t+\delta)
        y(t) = y_0 + s v_\bot \cos(t+\delta) - |ExB| t
        v_x(t) = v_\bot\cos(t + \delta)
        v_y(t) = -s v_\bot\sin(t + \delta) - |ExB|  (|B|=1)
    """
    s = -1

    T = np.arange(d.Nt) * dt
    vperp = np.sqrt(d.uxb ** 2 + d.uyb ** 2)
    delta = np.arctan2(-s * (d.uyb + d.eps), d.uxb)
    # Solve for IC
    x0 = d.xb - vperp * np.sin(delta)
    y0 = d.yb - s * vperp * np.cos(delta)
    # Create solution arrays
    XS, YS, ZS, UXS, UYS, UZS = np.zeros((6, d.Np, d.Nt), dtype=np.float64)
    # Loop through particles
    for i in range(X.shape[0]):
        XS[i, :] = x0[i] + vperp[i] * np.sin(T + delta[i])
        YS[i, :] = y0[i] + s * vperp[i] * np.cos(T + delta[i]) - d.eps * T
        ZS[i, :] = d.zb[i] + d.uzb[i] * T
        UXS[i, :] = vperp[i] * np.cos(T + delta[i])
        UYS[i, :] = -s * vperp[i] * np.sin(T + delta[i]) - d.eps
        UZS[i, :] = d.uzb[i]

    # Check
    assert np.isclose(X, XS, rtol=tol, atol=tol).all()
    assert np.isclose(Y, YS, rtol=tol, atol=tol).all()
    assert np.isclose(Z, ZS, rtol=tol, atol=tol).all()
    assert np.isclose(UX, UXS, rtol=tol, atol=tol).all()
    assert np.isclose(UY, UYS, rtol=tol, atol=tol).all()
    assert np.isclose(UZ, UZS, rtol=tol, atol=tol).all()

    if plot:
        me.setup_mpl(tex=True)
        # Loop through particles
        for i in range(X.shape[0]):
            # Create figure
            fig, axes = plt.subplots(3, 2, figsize=(17, 11), sharex=True)

            fig.suptitle(
                f"KE0 = {d.kinetic_energy[i]} \n" f"P0 = {d.pitch_angle[i]}"
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

            fig.savefig(f"particle_{i}.png", dpi=fig.dpi)
            plt.close(fig)


if __name__ == "__main__":

    # Unpack
    t, x, y, z, ux, uy, uz = d.t_start, d.xb, d.yb, d.zb, d.uxb, d.uyb, d.uzb
    # History arrays
    X, Y, Z, UX, UY, UZ = np.zeros((6, d.Np, d.Nt), dtype=np.float64)
    X[:, 0] = x
    Y[:, 0] = y
    Z[:, 0] = z
    UX[:, 0] = ux
    UY[:, 0] = uy
    UZ[:, 0] = uz

    ##################
    # Main loop
    ##################
    print(f"Starting main loop")
    for n in range(1, Nt):

        ## Advance particles
        t, x, y, z, ux, uy, uz = advance(t, x, y, z, ux, uy, uz, EM_model, dt)

        ## Do other stuff here
        # Save to history arrays
        X[:, n] = x
        Y[:, n] = y
        Z[:, n] = z
        UX[:, n] = ux
        UY[:, n] = uy
        UZ[:, n] = uz
        # At save interval
        if n % save_interval == 0:
            print(f"Pushed {n} steps")

    ###################
    # Post-processing
    ###################
    check_solution(X, Y, Z, UX, UY, UZ, plot=True)
