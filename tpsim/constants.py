"""
    Constants used in the simulation.
"""


__all__ = [
    "c",
    "e",
    "eps0",
    "m",
    "mc2",
    "m_e",
    "m_i",
    "q",
    "s"
]


import astropy.constants as c
import astropy.units as u


# Charge [C]
e = c.e.to(u.C).value
# Electron mass [kg]
m_e = c.m_e.to(u.kg).value
# Ion mass [kg]
m_i = c.m_p.to(u.kg).value

# Electric permittivity in vacuum [C2 s2 kg-1 m-3]
eps0 = c.eps0.to(u.Unit("C2 s2 kg-1 m-3")).value

# Speed of light [km s-1]
c = c.c.to(u.Unit("km s-1")).value

# Electron rest mass [eV]
mc2 = m_e * (c * 1e3) ** 2 / e

# Create dictionary for mass, charge, and charge sign
m = {"e-": m_e, "p": m_i}
q = {"e-": -e, "p": e}
s = {"e-": -1, "p": 1}
