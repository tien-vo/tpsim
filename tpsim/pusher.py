"""
    Contains the Boris pusher.
"""


__all__ = [
    "advance",
    "f_advance",
]


from numba import njit
import numpy as np


def advance(t, x, y, z, ux, uy, uz, model, dt):
    r"""Advance the particles in time through a step size dt with the given
    electromagnetic field model using the Boris algorithm. See more details in
    Ripperda et al., 2018, or Section 3.1 in the thesis.

    ----------
    Parameters
    ----------
    t: unitless
        Time in natural units (normalized to 1/wce)
    x: unitless
        x position in natural units (normalized to c/wce)
    y: unitless
        y position in natural units (normalized to c/wce)
    z: unitless
        z position in natural units (normalized to c/wce)
    ux: unitless
        x relativistic velocity in natural units (normalized to c)
    uy: unitless
        y relativistic velocity in natural units (normalized to c)
    uz: unitless
        z relativistic velocity in natural units (normalized to c)
    model: callable
        A function that takes in the state (t, x, y, z, ux, uy, uz) and returns
        (Ex, Ey, Ez, Bx, By, Bz)
    dt: unitless
        Step size in natural units (normalized to 1/wce)

    -------
    Returns
    -------
    (t_new, x_new, y_new, z_new, ux_new, uy_new, uz_new): unitless
        New electron state in natural units (same as input parameters)
    """
    # Charge sign
    s = -1
    # Electromagnetic field
    Ex, Ey, Ez, Bx, By, Bz = model(t, x, y, z, ux, uy, uz)
    # u_minus
    uxm = ux + s * dt * Ex / 2
    uym = uy + s * dt * Ey / 2
    uzm = uz + s * dt * Ez / 2
    # Lorentz factor
    g = np.sqrt(1 + (uxm ** 2 + uym ** 2 + uzm ** 2))
    # Auxiliary vectors
    Tx = s * Bx * dt / 2 / g
    Ty = s * By * dt / 2 / g
    Tz = s * Bz * dt / 2 / g
    T2 = Tx ** 2 + Ty ** 2 + Tz ** 2
    Sx = 2 * Tx / (1 + T2)
    Sy = 2 * Ty / (1 + T2)
    Sz = 2 * Tz / (1 + T2)
    # u_plus with up = um + ( (um + um x T) x S )
    uxp = (
        uxm
        - Sy * Ty * uxm
        - Sz * Tz * uxm
        + Sz * uym
        + Sy * Tx * uym
        - Sy * uzm
        + Sz * Tx * uzm
    )
    uyp = (
        -Sz * uxm
        + Sx * Ty * uxm
        + uym
        - Sx * Tx * uym
        - Sz * Tz * uym
        + Sx * uzm
        + Sz * Ty * uzm
    )
    uzp = (
        Sy * uxm
        + Sx * Tz * uxm
        - Sx * uym
        + Sy * Tz * uym
        + uzm
        - Sx * Tx * uzm
        - Sy * Ty * uzm
    )
    # New relativistic velocity
    ux_new = uxp + s * dt * Ex / 2
    uy_new = uyp + s * dt * Ey / 2
    uz_new = uzp + s * dt * Ez / 2
    g_new = np.sqrt(1 + ux_new ** 2 + uy_new ** 2 + uz_new ** 2)
    # New position
    x_new = x + dt * ux_new / g_new
    y_new = y + dt * uy_new / g_new
    z_new = z + dt * uz_new / g_new
    # New time
    t_new = t + dt
    return t_new, x_new, y_new, z_new, ux_new, uy_new, uz_new


@njit
def f_advance(t, x, y, z, ux, uy, uz, model, dt):
    # Charge sign
    s = -1
    # Electromagnetic field
    Ex, Ey, Ez, Bx, By, Bz = model(t, x, y, z, ux, uy, uz)
    # New time
    t_new = t + dt
    # Preallocate
    N_ = len(x)
    x_new, y_new, z_new, ux_new, uy_new, uz_new = np.zeros((6, N_))
    for i in range(N_):
        # u_minus
        uxm = ux[i] + s * dt * Ex[i] / 2
        uym = uy[i] + s * dt * Ey[i] / 2
        uzm = uz[i] + s * dt * Ez[i] / 2
        # Lorentz factor
        g = np.sqrt(1 + (uxm ** 2 + uym ** 2 + uzm ** 2))
        # Auxiliary vectors
        Tx = s * Bx[i] * dt / 2 / g
        Ty = s * By[i] * dt / 2 / g
        Tz = s * Bz[i] * dt / 2 / g
        T2 = Tx ** 2 + Ty ** 2 + Tz ** 2
        Sx = 2 * Tx / (1 + T2)
        Sy = 2 * Ty / (1 + T2)
        Sz = 2 * Tz / (1 + T2)
        # u_plus with up = um + ( (um + um x T) x S )
        uxp = (
            uxm
            - Sy * Ty * uxm
            - Sz * Tz * uxm
            + Sz * uym
            + Sy * Tx * uym
            - Sy * uzm
            + Sz * Tx * uzm
        )
        uyp = (
            -Sz * uxm
            + Sx * Ty * uxm
            + uym
            - Sx * Tx * uym
            - Sz * Tz * uym
            + Sx * uzm
            + Sz * Ty * uzm
        )
        uzp = (
            Sy * uxm
            + Sx * Tz * uxm
            - Sx * uym
            + Sy * Tz * uym
            + uzm
            - Sx * Tx * uzm
            - Sy * Ty * uzm
        )
        # New relativistic velocity
        ux_new[i] = uxp + s * dt * Ex[i] / 2
        uy_new[i] = uyp + s * dt * Ey[i] / 2
        uz_new[i] = uzp + s * dt * Ez[i] / 2
        g_new = np.sqrt(1 + ux_new[i] ** 2 + uy_new[i] ** 2 + uz_new[i] ** 2)
        # New position
        x_new[i] = x[i] + dt * ux_new[i] / g_new
        y_new[i] = y[i] + dt * uy_new[i] / g_new
        z_new[i] = z[i] + dt * uz_new[i] / g_new

    return t_new, x_new, y_new, z_new, ux_new, uy_new, uz_new

