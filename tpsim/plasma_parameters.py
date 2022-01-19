"""
    Plasma physics functions
"""


__all__ = [
    "calculate_SDP",
    "cyclotron_frequency",
    "plasma_frequency",
]


from .constants import q, m, s, eps0, c
import numpy as np


def cyclotron_frequency(B, particle="e-"):
    """Calculate the unsigned cyclotron frequency in SI units.

    ----------
    Parameters
    ----------
    B           : nT
        Background magnetic field
    particle    : str
        "e-" for electron or "i" for ion

    -------
    Returns
    -------
    wc : rad/s
        Cyclotron frequency
    """
    wc = abs(q[particle]) * B * 1e-9 / m[particle]
    return wc


def plasma_frequency(n, particle="e-"):
    """"Calculate the plasma frequency in SI units.

    ----------
    Parameters
    ----------
    n           : cm-3
        Number density of particles
    particle    : str
        "e" for electron or "i" for ion

    -------
    Returns
    -------
    wp : rad/s
        Plasma frequency
    """
    wp = np.sqrt((n * 100 ** 3 * q[particle] ** 2) / (eps0 * m[particle]))
    return wp


def inertial_length(n, particle="e-"):
    """"Calculate the inertial length in SI units.

    ----------
    Parameters
    ----------
    n           : cm-3
        Number density of particles
    particle    : str
        "e" for electron or "i" for ion

    -------
    Returns
    -------
    d : rad/s
        Inertial length
    """
    return c / plasma_frequency(n, particle)


def calculate_SDP(B, n, w):
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
    for ptcl in ["e-", "i"]:
        wc = cyclotron_frequency(B, particle=ptcl)
        wp = plasma_frequency(n, particle=ptcl)
        S += - (wp ** 2) / (w ** 2 - wc ** 2)
        D += s[ptcl] * (wc * wp ** 2) / (w * (w ** 2 - wc ** 2))
        P += - (wp ** 2) / (w ** 2)

    return (S, D, P)

