"""
    Coordinate transformations
"""


__all__ = [
    "gamma",
    "KE2S",
    "S2KE",
    "VS2ES",
    "ES2VS",
    "US2ES",
    "ES2US",
    "normalize_E",
]


from .constants import c, mc2
import astropy.units as u
import numpy as np


def unvectorized_gamma(speed):
    r"""Calculate the Lorentz factor. Check that the speed is smaller than
    the speed of light.

    ----------
    Parameters
    ----------
    speed: km/s
        Must be non-relativistic speed.

    -------
    Returns
    -------
    gamma_: unitless
        Lorentz factor
    """
    assert speed <= c
    gamma_ = 1 / np.sqrt(1 - (speed / c) ** 2)
    return gamma_

# Vectorize this function
gamma = np.vectorize(unvectorized_gamma)


def KE2S(kinetic_energy):
    r"""Convert relativistic kinetic energy to speed. It is impossible to
    violate relativity with this calculation.

    ----------
    Parameters
    ----------
    kinetic_energy: eV
        The kinetic energy of the electron defined by W = (gamma - 1) m c^2

    -------
    Returns
    -------
    speed: km s-1
        The speed calculated from inverting for the Lorentz factor
    """
    # Calculate speed
    speed = c * np.sqrt(1 - (1 + kinetic_energy / mc2) ** (-2))
    return speed


def S2KE(speed):
    r"""Convert relativistic kinetic energy to speed. It is impossible to
    violate relativity with this calculation. This is the inversion of `KE2S`.

    ----------
    Parameters
    ----------
    speed: km/s
        The speed of the particle relative to plasma frame.

    -------
    Returns
    -------
    kinetic_energy: eV
        The kinetic energy of the electron defined by W = (gamma - 1) m c^2
    """
    kinetic_energy = (gamma(speed) - 1) * mc2
    return kinetic_energy


def VS2ES(vx, vy, vz):
    r"""Converts 3-D non-relativistic velocity vector v=(vx, vy, vz) to the
    energy space (spherical coordinate with energy as the radius).

    ----------
    Parameters
    ----------
    vx: km/s
        The x component of the velocity vector
    vy: km/s
        The y component of the velocity vector
    vz: km/s
        The z component of the velocity vector

    -------
    Returns
    -------
    kinetic_energy: eV
        The kinetic energy of the electron defined by W = (gamma - 1) m c^2
    gyrophase: rad
        The azimuthal angle
    pitch_angle: rad
        The polar angle
    """
    speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    kinetic_energy = S2KE(speed)
    gyrophase = np.arctan2(vy, vx)
    pitch_angle = np.arccos(vz / speed)
    return (kinetic_energy, gyrophase, pitch_angle)


def ES2VS(kinetic_energy, gyrophase, pitch_angle):
    r"""Converts 3-D vector in energy space
    (kinetic_energy, gyrophase, pitch_angle) to the 3-D non-relativistic
    velocity vector. This is the inversion of `VS2ES`.

    ----------
    Parameters
    ----------
    kinetic_energy: eV
        The kinetic energy of the electron defined by W = (gamma - 1) m c^2
    gyrophase: rad
        The azimuthal angle
    pitch_angle: rad
        The polar angle

    -------
    Returns
    -------
    vx: km/s
        The x component of the velocity vector
    vy: km/s
        The y component of the velocity vector
    vz: km/s
        The z component of the velocity vector
    """
    speed = KE2S(kinetic_energy)
    return speed * np.array(
        [
            np.sin(pitch_angle) * np.cos(gyrophase),
            np.sin(pitch_angle) * np.sin(gyrophase),
            np.cos(pitch_angle),
        ]
    )


def US2ES(ux, uy, uz):
    r"""Converts 3-D relativistic velocity vector u=gamma (vx, vy, vz) to the
    energy space (spherical coordinate with energy as the radius).

    ----------
    Parameters
    ----------
    ux: km/s
        The x component of the velocity vector
    uy: km/s
        The y component of the velocity vector
    uz: km/s
        The z component of the velocity vector

    -------
    Returns
    -------
    kinetic_energy: eV
        The kinetic energy of the electron defined by W = (gamma - 1) m c^2
    gyrophase: rad
        The azimuthal angle
    pitch_angle: rad
        The polar angle
    """
    unorm = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
    g = np.sqrt(1 + (unorm / c) ** 2)
    kinetic_energy = (g - 1) * mc2
    gyrophase = np.arctan2(uy, ux)
    pitch_angle = np.arccos(uz / unorm)
    return (kinetic_energy, gyrophase, pitch_angle)


def ES2US(kinetic_energy, gyrophase, pitch_angle):
    r"""Converts 3-D vector in energy space
    (kinetic_energy, gyrophase, pitch_angle) to the 3-D relativistic
    velocity vector. This is the inversion of `US2ES`.

    ----------
    Parameters
    ----------
    kinetic_energy: eV
        The kinetic energy of the electron defined by W = (gamma - 1) m c^2
    gyrophase: rad
        The azimuthal angle
    pitch_angle: rad
        The polar angle

    -------
    Returns
    -------
    ux: km/s
        The x component of the velocity vector
    uy: km/s
        The y component of the velocity vector
    uz: km/s
        The z component of the velocity vector
    """
    speed = KE2S(kinetic_energy)
    return (
        gamma(speed)
        * speed
        * np.array(
            [
                np.sin(pitch_angle) * np.cos(gyrophase),
                np.sin(pitch_angle) * np.sin(gyrophase),
                np.cos(pitch_angle),
            ]
        )
    )


def normalize_E(E, B0):
    r"""Normalize the electric field to natural units.

    ----------
    Parameters
    ----------
    E: mV/m
        The electric field.
    B0: nT
        The background magnetic field.

    -------
    Returns
    -------
    E_bar: unitless
        Normalized electric field (E / cB0)
    """
    E_bar = 1e3 * E / c / B0
    return E_bar
