"""
    Solve the IVP in Qin et al. with the Boris pusher and compare results
    with RK4 output.
"""

import matplotlib.pyplot as plt
import mpl_extras as me
import tpsim as tp
import numpy as np


s = -1
advance_boris = tp.advance


class QinProblem:
    def __init__(
        self,
        IC=np.array(
            [
                1,
                0.0,
                0.0,
                0.0,
                0.1,
                0.0,
            ]
        ),
    ):
        self.IC = IC

    def solve(self, which, dt=np.pi / 10, Nt=2000 * 170, plot=True):
        """
        The trajectory isn't a closed loop. So we can only estimate
        the number of steps it takes to close 1 large gyration, which
        is about ~ 2000 steps.
        """

        advance = advance_boris if which == "boris" else self.advance_RK4

        # Unpack IC
        t = 0
        x, y, z, ux, uy, uz = self.IC
        # Set up history arrays
        X, Y = np.zeros((2, Nt))
        X[0] = x
        Y[0] = y

        # Model
        model = self.model
        #############
        # Main loop
        #############
        for n in range(1, Nt):
            # Push particles
            t, x, y, z, ux, uy, uz = advance(t, x, y, z, ux, uy, uz, model, dt)
            # Save to history
            X[n] = x
            Y[n] = y
            # Output progress
            if n % (Nt // 10) == 0:
                print(f"Pushed {n} steps")

        if plot:
            me.setup_mpl(tex=True)
            # Create figure
            fig, axes = plt.subplots(
                1, 2, sharey=True, sharex=True, figsize=(17, 9)
            )
            fig.subplots_adjust(wspace=0.05)
            fig.suptitle(
                f"Initial state: {qin.IC}; "
                f"$\\delta_t$ = {dt:.4f}; "
                f"$N_t$ = {Nt}"
            )
            # Plot first 2000 steps
            axes[0].plot(X[:2000], Y[:2000], "-k")
            # Plot last 2000 steps
            axes[1].plot(X[-2000:], Y[-2000:], "-k")
            for ax in axes:
                ax.grid()
                ax.set_aspect("equal")
                ax.tick_params(**me.params)
                ax.set_xlabel("$x$")

            axes[0].set_ylabel("$y$")
            axes[0].set_title("1st turn")
            axes[1].set_title("170th turn")
            fig.savefig(f"qin_{which}.png", dpi=fig.dpi)

    def solve_RK45(self, dt=np.pi / 10, Nt=2000 * 170, plot=True):
        pass

    @staticmethod
    def model(t, x, y, z, ux, uy, uz):
        eps = 1e-2
        r = np.sqrt(x ** 2 + y ** 2)
        Ex = -eps * x / r ** 3
        Ey = -eps * y / r ** 3
        Ez = np.zeros(x.shape)
        Bx = np.zeros(x.shape)
        By = np.zeros(x.shape)
        Bz = r
        return Ex, Ey, Ez, Bx, By, Bz

    @staticmethod
    def advance_RK4(t, x, y, z, ux, uy, uz, model, dt):
        def _k(t, x, y, z, ux, uy, uz):
            Ex, Ey, Ez, Bx, By, Bz = model(t, x, y, uz, ux, uy, uz)
            g = np.sqrt(1 + ux ** 2 + uy ** 2 + uz ** 2)
            dx = ux / g
            dy = uy / g
            dz = uz / g
            dux = s * (Ex + (uy * Bz - uz * By) / g)
            duy = s * (Ey + (uz * Bx - ux * Bz) / g)
            duz = s * (Ez + (ux * By - uy * Bx) / g)
            return np.array([dx, dy, dz, dux, duy, duz])

        hs = t + dt / 2
        fs = t + dt
        state = np.array([x, y, z, ux, uy, uz])
        k1 = _k(t, *state)
        k2 = _k(hs, *(state + dt * k1 / 2))
        k3 = _k(hs, *(state + dt * k2 / 2))
        k4 = _k(fs, *(state + dt * k3))
        new_state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return (fs, *new_state)


if __name__ == "__main__":

    qin = QinProblem()
    qin.solve("boris")
    qin.solve("rk4")
