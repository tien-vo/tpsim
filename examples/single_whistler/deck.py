from itertools import product
import matplotlib.pyplot as plt
import tpsim as tp
import numpy as np


# --------------------------------------------------------------------------- #
#                           Simulation parameters
# --------------------------------------------------------------------------- #
## ---------- Simulation time
# Start time [1/wce]
t_start = 0
# Stop time [1/wce]
t_stop = 70 * 2 * np.pi
# Time step [1/wce]
dt = np.pi * 1e-3
# Number of time steps
Nt = int(t_stop / dt)
# Interval to save data
save_interval = Nt // 1000

## ---------- Background parameters
# Background magnetic field [nT]
B0 = 60
# Number density [1/cc]
n = 350
# Electron cyclotron frequency
wce = tp.cyclotron_frequency(B0, particle="e-")
# Electron plasma frequency
wpe = tp.plasma_frequency(n, particle="e-")

## ---------- Wave parameters
# w/wce
w_wce = 0.15
# Obliquity [rad]
theta = np.radians(65)
# Amplitude [mV/m]
E0 = 20.0
# Normalized amplitude
E0n = tp.normalize_E(E0, B0)
# Calculate polarizations
Nx, Nz, pex, pey, pez, pbx, pby, pbz = tp.whistler_polarization(
    theta, B0, n, w_wce * wce
)
"""NOTE:
    In this coordinate system, the electric field is determined analytically as
Ew = E0 ( pex cos(psi), -ey sin(psi), ez cos(psi) ). So the original electric
field magnitude is Ew0 = E0n sqrt(pex^2 + pez^2) if psi0 = 0. Since we wish the
wave amplitude to be E0n, we have to rescale it such that Ew0 = E0n.
"""
E0n /= np.sqrt(pex ** 2 + pez ** 2)
# Recalculate original amplitude in physical units (just to make sure)
Ew0_recal = E0n * np.sqrt(pex ** 2 + pez ** 2) * tp.c * B0 * 1e-3
Bw0_recal = E0n * pby * B0

## ---------- Particle parameters
# Speed
S_range = np.linspace(0, tp.KE2S(3000), 30) / tp.c
# Gyrophase
G_range = np.radians(np.arange(0, 1, 20))
# Pitch angle
P_range = np.radians(np.arange(0, 181, 5))
S, G, P = np.array(list(product(S_range, G_range, P_range))).T
# Number of particles
Np = len(S)
# Normalized position
xn, yn, zn = np.zeros((3, Np))
# Normalized velocity
uxn, uyn, uzn = S * np.array([
    np.sin(P) * np.cos(G),
    np.sin(P) * np.sin(G),
    np.cos(P),
])

## ---------- Electromagnetic field model
## Define the electromagnetic field here (background + perturbations)
def EM_model(t, x, y, z, ux, uy, uz):
    """Returns `Np`-dimensional arrays `Ex`, `Ey`, `Ez`, `Bx`, `By`, `Bz` in
    normalized units.
    """
    psi = w_wce * (Nx * x + Nz * z - t)
    Ex = E0n * pex * np.cos(psi)
    Ey =-E0n * pey * np.sin(psi)
    Ez = E0n * pez * np.cos(psi)
    Bx = E0n * pbx * np.sin(psi)
    By = E0n * pby * np.cos(psi)
    Bz = E0n * pbz * np.sin(psi) + 1
    return Ex, Ey, Ez, Bx, By, Bz

# Simulation information
sim_info = f"""
------------------------------------------------------------------
    Simulation of particle dynamics in a single uniform whistler.
------------------------------------------------------------------
 * dtwce                                    : {dt:.4f}
 * Number of time steps                     : {Nt}
----------
 * Number of particles                      : {Np}
----------
 * Background magnetic field                : {B0} nT
 * Background electron density              : {n} cm-3
 * Electron plasma frequency                : {wpe:.4f} rad/s
 * Electron cyclotron frequency             : {wce:.4f} rad/s
 * wpe / wce                                : {wpe / wce:.4f}
----------
 * Original wave electric field amplitude   : {Ew0_recal:.4f} mV/m
 * Original wave magnetic field amplitude   : {Bw0_recal:.4f} nT
 * Wave frequency                           : {w_wce} wce
 * Wave obliquity                           : {np.degrees(theta)} deg
------------------------------------------------------------------
"""

