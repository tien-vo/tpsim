"""
    Input deck for simulations.
"""


from itertools import product
from numba import njit
import tpsim as tp
import numpy as np
import warnings
import os


####################################################
# Simulation parameters
####################################################
## Simulation time
# Start time
t_start = 0
# Stop time
t_stop = 2 * np.pi
# Time step
dt = np.pi * 1e-3
# Number of time steps
Nt = int(t_stop / dt)
# Interval to save data
save_interval = Nt // 10

####################################################
# Background parameters
####################################################
# Background magnetic field [nT]
B0 = 10
# Number density (per cc)
n = 5
# Electron cyclotron frequency
wce = tp.cyclotron_frequency(B0)
# Electron plasma frequency
wpe = tp.plasma_frequency(n)

####################################################
# Electromagnetic field model
# (Define how the electromagnetic field is determined here, together with
# wave parameters & whatnot)
####################################################
@njit(fastmath=True, cache=True)
def EM_model(t, x, y, z, ux, uy, uz):
    """
    Returns `Np`-dimensional arrays `Ex`, `Ey`, `Ez`, `Bx`, `By`, `Bz`
    in normalized units.
    """
    N = x.size
    Ex = np.zeros(N)
    Ey = np.zeros(N)
    Ez = np.zeros(N)
    Bx = np.zeros(N)
    By = np.zeros(N)
    Bz = np.ones(N)
    return Ex, Ey, Ez, Bx, By, Bz


####################################################
# Particle parameters
####################################################
# Energy space
kinetic_energy = np.array([10, 30])
gyrophase = np.array([0, 0])
pitch_angle = np.array([45, 90])
# Number of particles
Np = kinetic_energy.size
# Spatial positions
xb, yb, zb = np.zeros((3, Np))
uxb, uyb, uzb = tp.ES2US(
    kinetic_energy, np.radians(gyrophase), np.radians(pitch_angle)
) / tp.c

print(f"""
------------------------------------------------------------------
    Simulation of particle dynamics in finite electric field.
------------------------------------------------------------------
 * dtwce                                    : {dt:.4f}
 * Number of time steps                     : {Nt}
----------
 * Number of particles                      : {Np}
----------
 * Background magnetic field                : {B0} nT
 * Background electron density              : {n} /cc
 * Electron plasma frequency                : {wpe:.4f} rad/s
 * Electron cyclotron frequency             : {wce:.4f} rad/s
 * wpe / wce                                : {wpe / wce:.4f}
------------------------------------------------------------------
""")

