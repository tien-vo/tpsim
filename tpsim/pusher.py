"""
    Contains the Boris pusher.
"""


__all__ = [
    "advance",
]


import numpy as np


def advance(t, x, y, z, ux, uy, uz, model, dt, s="e-"):
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
    particle: str
        "e-" for electron, "p" for ion

    -------
    Returns
    -------
    (t_new, x_new, y_new, z_new, ux_new, uy_new, uz_new): unitless
        New electron state in natural units (same as input parameters)
    """
    # Charge sign
    qq = -1 if s == "e-" else 1
    # Electromagnetic field
    Ex, Ey, Ez, Bx, By, Bz = model(t, x, y, z, ux, uy, uz)
    # u_minus
    uxm = ux + qq * dt * Ex / 2
    uym = uy + qq * dt * Ey / 2
    uzm = uz + qq * dt * Ez / 2
    # Lorentz factor
    g = np.sqrt(1 + (uxm ** 2 + uym ** 2 + uzm ** 2))
    # Auxiliary vectors
    Tx = qq * Bx * dt / 2 / g
    Ty = qq * By * dt / 2 / g
    Tz = qq * Bz * dt / 2 / g
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
    ux_new = uxp + qq * dt * Ex / 2
    uy_new = uyp + qq * dt * Ey / 2
    uz_new = uzp + qq * dt * Ez / 2
    g_new = np.sqrt(1 + ux_new ** 2 + uy_new ** 2 + uz_new ** 2)
    # New position
    x_new = x + dt * ux_new / g_new
    y_new = y + dt * uy_new / g_new
    z_new = z + dt * uz_new / g_new
    # New time
    t_new = t + dt
    return t_new, x_new, y_new, z_new, ux_new, uy_new, uz_new

