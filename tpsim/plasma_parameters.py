"""
    Plasma physics functions
"""


__all__ = [
    "SDP",
    "ds",
    "wcs",
    "wps",
]


from .constants import q, m, qq, eps0, c
import numpy as np


def wcs(B, s="e-"):
    """Calculate the unsigned cyclotron frequency in SI units.

    ----------
    Parameters
    ----------
    B : nT
        Background magnetic field
    s : str
        "e-" for electron or "i" for ion (default: "e-")

    -------
    Returns
    -------
    wcs : rad/s
        Cyclotron frequency
    """
    return abs(q[s]) * B * 1e-9 / m[s]


def wps(n, s="e-"):
    """"Calculate the plasma frequency in SI units.

    ----------
    Parameters
    ----------
    n           : cm-3
        Number density of particles
    s    : str
        "e" for electron or "i" for ion

    -------
    Returns
    -------
    wps : rad/s
        Plasma frequency
    """
    return np.sqrt((n * 100 ** 3 * q[s] ** 2) / (eps0 * m[s]))


def ds(n, s="e-"):
    """"Calculate the inertial length in SI units.

    ----------
    Parameters
    ----------
    n           : cm-3
        Number density of particles
    s    : str
        "e" for electron or "i" for ion

    -------
    Returns
    -------
    ds : rad/s
        Inertial length
    """
    return c / wps(n, s)


def SDP(B, n, w):
    """Calculate the Stix parameters.

    ----------
    Parameters
    ----------
    B : nT
        Background magnetic field
    n : cm-3
        Number density of particles
    w : rad s-1
        wave frequency

    -------
    Returns
    -------
    S, D, P : unitless
        Stix coefficients
    """
    S, D, P = 1, 0, 1
    for s in ["e-", "i"]:
        wc_ = wcs(B, s=s)
        wp_ = wps(n, s=s)
        S += - (wp_ ** 2) / (w ** 2 - wc_ ** 2)
        D += qq[s] * (wc_ * wp_ ** 2) / (w * (w ** 2 - wc_ ** 2))
        P += - (wp_ ** 2) / (w ** 2)

    return (S, D, P)

