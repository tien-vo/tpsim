"""
    HDF5 helpers.
"""


__all__ = [
    "dump_particles",
]


import tables as tb


def dump_particles(n, t, x, y, z, ux, uy, uz):
    r"""Dump particles' state vector into an HDF5 database.

    ----------
    Parameters
    ----------
    n: int
        The nth step in the simulation
    t: unitles
        The time corresponding to n
    x: unitless
        The x position
    y: unitless
        The y position
    z: unitless
        The z position
    ux: unitless
        The x velocity
    uy: unitless
        The y velocity
    uz: unitless
        The z velocity
    """

    f = tb.open_file(f"particles/{n}.h5", "w")
    f.create_array("/", "t", [t], "Normalized time twce")
    f.create_array("/", "x", x, "Normalized position x wce / c")
    f.create_array("/", "y", y, "Normalized position y wce / c")
    f.create_array("/", "z", z, "Normalized position z wce / c")
    f.create_array(
        "/", "ux", ux, "Normalized relativistic velocity ux = g vx / c"
    )
    f.create_array(
        "/", "uy", uy, "Normalized relativistic velocity uy = g vx / c"
    )
    f.create_array(
        "/", "uz", uz, "Normalized relativistic velocity uz = g vx / c"
    )
    f.close()

