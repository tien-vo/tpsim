"""
    Plasma physics functions
"""


__all__ = [
    "calculate_SDP",
    "cyclotron_frequency",
    "plasma_frequency",
]


from .constants import e, m_e, m_i, eps0
import numpy as np


def cyclotron_frequency(B, m=m_e):
    """Calculate the unsigned cyclotron frequency in SI units.

    ----------
    Parameters
    ----------
    B : nT
        Background magnetic field
    m : kg
        Particle mass (default: electron mass)

    -------
    Returns
    -------
    wc : rad/s
        Cyclotron frequency
    """
    wc = e * B * 1e-9 / m
    return wc


def plasma_frequency(n, m=m_e):
    """"Calculate the plasma frequency in SI units.

    ----------
    Parameters
    ----------
    n : cm-3
        Number density of particles
    m : kg
        Particle mass (default: electron mass)

    -------
    Returns
    -------
    wp : rad/s
        Plasma frequency
    """
    wp = np.sqrt((n * 100 ** 3 * e ** 2) / (eps0 * m))
    return wp


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
    for i, m in enumerate([m_e, m_i]):
        s = -1 if i == 0 else  1    #sign for electron and ion charge
        wc = cyclotron_frequency(B, m)
        wp = plasma_frequency(n, m)
        S += - (wp ** 2) / (w ** 2 - wc ** 2)
        D += s * (wc * wp**2) / (w * (w ** 2 - wc ** 2))
        P += - (wp ** 2) / (w ** 2)

    return (S, D, P)

