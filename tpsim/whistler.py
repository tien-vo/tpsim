"""
    Contains whistler calculations.
"""


__all__ = [
    "whistler_polarization",
    "R_surface",
    "H_surface",
]


from .plasma_parameters import SDP
import astropy.units as u
import numpy as np


def whistler_polarization(theta, B0, n, w):
    r"""Solves cold plasma dispersion relation for the wave polarizations from
    background and wave parameters. Consult eqns (4-8) in Tao & Bortnik, 2010
    or Section 2.1 of the thesis in `docs/thesis/thesis.pdf` for the details.

    ----------
    Parameters
    ----------
    theta : rad
        Wave propagation angle (obliquity)
    B0 : nT
        Background magnetic field
    n : cm-3
        Background density
    w : rad/s
        Wave frequency

    -------
    Returns
    -------
    Nx : unitless
        x component of the refractive index vector N = (Nx, 0, Nz)
    Nz : unitless
        z component of the refractive index vector N = (Nx, 0, Nz)
    pex : unitless
        |Exw| / |Exw|
    pey : unitless
        |Eyw| / |Exw|
    pez : unitless
        |Ezw| / |Exw|
    pbx : unitless
        c |Bxw| / |Exw|
    pby : unitless
        c |Byw| / |Exw|
    pbz : unitless
        c |Bzw| / |Exw|
    """
    # Calculate Stix coefficients
    S, D, P = calculate_SDP(B=B0, n=n, w=w)
    # Auxilliary constants
    A = S * np.sin(theta) ** 2 + P * np.cos(theta) ** 2
    B = S * P + S * A - D ** 2 * np.sin(theta) ** 2
    C = P * (S ** 2 - D ** 2)
    # Index of refraction
    N = np.sqrt((B - np.sqrt(B ** 2 - 4 * A * C)) / 2 / A)
    Nx = N * np.sin(theta)
    Nz = N * np.cos(theta)
    # Calculate polarizations (relative component is Ex)
    pex = np.float64(1.0)
    pey = D / (N ** 2 - S)
    pez = Nx * Nz / (Nx ** 2 - P)
    pbx = Nz * D / (N ** 2 - S)
    pby = Nz * P / (P - Nx ** 2)
    pbz = Nx * D / (S - N ** 2)
    return (Nx, Nz, pex, pey, pez, pbx, pby, pbz)


def R_surface(n, Nz, w):
    r"""Solves for the intersection of the R surface and the v_perp = 0 axis.
    See details in Section 2.3 of the thesis in `docs/thesis/thesis.pdf`.

    ----------
    Parameters
    ----------
    n : unitless
        The resonance harmonic
    Nz : unitless
        The parallel refractive index in natural units (nz = c kz / w)
    w : unitless
        The wave frequency in natural units (w/wce)

    -------
    Returns
    -------
    (vrp, vrm): unitless
        The positive and negative roots of the resonant velocity in natural
        units
    """
    # Phase velocity
    v_phase = 1 / Nz
    alpha = n / w / Nz
    A = np.sqrt(
        alpha ** 2 / (1 + alpha ** 2) * (1 - v_phase ** 2 / (1 + alpha ** 2))
    )
    vrp = v_phase / (1 + alpha ** 2) + A
    vrm = v_phase / (1 + alpha ** 2) - A
    return vrp, vrm


def H_surface(alpha, vz0, Nz, w):
    r"""Solves for the constant H surface. See details in Section 2.3 of
    the thesis in `docs/thesis/thesis.pdf`.

    ----------
    Parameters
    ----------
    alpha : rad, array_like
        The pitch angle in radians
    vz0 : unitless
        The intersection of H with v_perp = 0
    Nz : unitless
        The parallel refractive index in natural units (nz = c kz / w)
    w : unitless
        The wave frequency in natural units (w/wce)

    -------
    Returns
    -------
    (vz, vp): unitless
        The parallel and perpendicular velocity in natural units
    """
    # Original Lorentz factor
    g0 = 1 / np.sqrt(1 - vz0 ** 2)
    # Phase velocity
    v_phase = 1 / Nz
    # Auxilliary constants
    aH = v_phase ** 2 - 1
    H0 = g0 * (1 - v_phase * vz0)
    bH = v_phase * H0

    R = np.sqrt(H0 ** 2 - 1 + v_phase ** 2 / (H0 ** 2 + v_phase ** 2))
    vz = v_phase / (H0 ** 2 + v_phase ** 2) + R * np.cos(alpha) / np.sqrt(
        H0 ** 2 + v_phase ** 2
    )
    vp = R * np.sin(alpha) / H0
    return vz, vp
