"""
    Input deck for simulations.
"""


from itertools import product
from numba import njit
import astropy.units as u
import tpsim as tp
import numpy as np
import warnings


####################################################
# Simulation parameters
####################################################
## Simulation time
# Start time
t_start = np.float64(0)
# Stop time
t_stop = np.float64(7 * 2 * np.pi)
# Time step
dt = np.float64(np.pi * 1e-1)
# Number of time steps
Nt = np.int(t_stop / dt)
# Interval to save data
save_interval = 1

####################################################
# Background parameters
####################################################
# Background magnetic field [nT]
B0 = np.float64(10)
# Number density (per cc)
n = np.float64(5)
# Electron cyclotron frequency [rad/s]
wce = tp.cyclotron_frequency(B0)
# Electron plasma frequency [rad/s]
wpe = tp.plasma_frequency(n)

####################################################
# Wave parameters
####################################################
# w/wce
w_wce = np.float64(0.15)
# Obliquity [rad]
theta = np.radians(65)
# Amplitude [mV/m]
E0 = np.float64(20)
# Ampitude in normalized units
E0b = tp.normalize_E(E0, B0)
Nx, Nz, ex, ey, ez, bx, by, bz = tp.single_wave_polarization(
    theta,
    B0,
    n,
    w_wce * wce,
)
## NOTE:
# In this coordinate system, the electric field is determined analytically
# as Ew = E0 ( ex cos(psi), - ey sin(psi), ez cos(psi) ). So the original
# electric field magnitude is Ew0 = E0b * sqrt(ex ** 2 + ez ** 2) if psi0 = 0.
# Since we wish the wave amplitude to be E0b, we have to rescale it such that
# Ew0 = E0b.
E0b /= np.sqrt(ex ** 2 + ez ** 2)
# Recalculate original amplitude in physical units
_Ew0 = E0b * np.sqrt(ex ** 2 + ez ** 2) * tp.c * B0 * 1e-3
_Bw0 = E0b * by * B0

N = np.sqrt(Nx ** 2 + Nz ** 2)
e1 = (np.cos(theta) * ex - np.sin(theta) * ez) * E0b / w_wce
e2 = E0b * ey / w_wce
e3 = (np.sin(theta) * ex + np.cos(theta) * ez) * E0b / w_wce / N

####################################################
# Electromagnetic field model
# (Define how the electromagnetic field is determined here, together with
# wave parameters & whatnot)
####################################################
@njit(cache=True, fastmath=True)
def EM_model(t, x, y, z, ux, uy, uz):
    """
    Returns `Np`-dimensional arrays `Ex`, `Ey`, `Ez`, `Bx`, `By`, `Bz`
    in normalized units.
    """
    # Preallocate
    N_ = len(x)
    Ex, Ey, Ez, Bx, By, Bz = np.zeros((6, N_))
    for i in range(N_):
        # Wave phase
        psi = w_wce * (Nx * x[i] + Nz * z[i] - t)
        # Calculate oscillations
        sp, cp = np.sin(psi), np.cos(psi)
        # Rescale
        Ex[i] = E0b * ex * cp
        Ey[i] = -E0b * ey * sp
        Ez[i] = E0b * ez * cp
        Bx[i] = E0b * bx * sp
        By[i] = E0b * by * cp
        # Has background magnetic field
        Bz[i] = E0b * bz * sp + 1

    return Ex, Ey, Ez, Bx, By, Bz


####################################################
# Particle parameters
####################################################
# Energy space
KE_range = np.arange(0, 1001, 100)
G_range = np.arange(0, 1, 20)
P_range = np.arange(0, 181, 30)
kinetic_energy, gyrophase, pitch_angle = np.array(
    list(product(KE_range, G_range, P_range))
).T
# Number of particles
Np = kinetic_energy.size
# Normalized position
xb, yb, zb = np.zeros((3, Np), dtype=np.float64)
# Normalized velocity
uxb, uyb, uzb = tp.ES2US(
    kinetic_energy, np.radians(gyrophase), np.radians(pitch_angle)
) / tp.c

