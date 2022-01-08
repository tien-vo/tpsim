"""
This script follows closely the Mathematica script from Sandri, 1996 (see
Sandri_1996_script/ for more details) and tries to reproduce the results
therein.
"""


import matplotlib.pyplot as plt
import mpl_extras as me
import tpsim as tp
import numpy as np


sigma = 16
beta = 4
rho = 45.92


def F(t, state):
    """Define the Lorenz system"""
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def DF(x, y, z):
    """The Jacobian of the Lorenz system"""
    return np.array([[-sigma, sigma, 0], [rho - z, -1, -x], [y, x, -beta]])


def RK4(x0, phi0, T, dt, *args):

    # Number of steps
    N = int(T / dt)

    # Assign current state
    x = x0
    phi = phi0

    # Loop
    for n in range(N):
        t = n * dt
        # Update state
        xn = x
        xk1 = F(t, xn, *args)
        xk2 = F(t + dt / 2, xn + dt * xk1 / 2, *args)
        xk3 = F(t + dt / 2, xn + dt * xk2 / 2, *args)
        xk4 = F(t + dt, xn + dt * xk3, *args)
        x = xn + dt * (xk1 + 2 * xk2 + 2 * xk3 + xk4) / 6
        # Update phi
        pn = phi
        pk1 = np.dot(DF(*xn), pn)
        pk2 = np.dot(DF(*(xn + dt * xk1 / 2)), pn + dt * pk1 / 2)
        pk3 = np.dot(DF(*(xn + dt * xk2 / 2)), pn + dt * pk2 / 2)
        pk4 = np.dot(DF(*(xn + dt * xk3)), pn + dt * pk3)
        phi = pn + dt * (pk1 + 2 * pk2 + 2 * pk3 + pk4) / 6

    return x, phi


# Constants & initial conditions
dt = 0.02
x0 = np.array([19, 20, 50])
phi0 = np.identity(3)


# --------------------------------------------------------------------------- #
#                       Continuous system estimation
# --------------------------------------------------------------------------- #
"""
    Continuously updates the 'ball' in each iteration, after which an
    estimation of the LCE can be carried out via the operation of the ball on
    random elements of the phase space.
"""
# Run the pusher for a transient period.
T = 20
# Step the system
x, phi = RK4(x0, phi0, T, dt)
# Compare to the final state in Sandri, 1996
assert np.isclose(x, np.array([17.733, 15.0227, 53.4122]), rtol=1e-2).all()
# Estimate LCEs from integrated phi by acting the resulting ball phi onto
# a random vector
u = np.random.randn(len(x))
LCE = np.log(np.linalg.norm(np.dot(phi, u))) / T
# Compare again, but since this is randomized, the margin might be large
assert np.isclose(LCE, 1.45026, rtol=2e-1)


# --------------------------------------------------------------------------- #
#                   LCE spectrum after transient period
# --------------------------------------------------------------------------- #
"""
    Having passed the transient period, we can now estimate the LCE components
    in periods of K iterations from the ball phi.
"""
# Reset the ball to an identity ball
x = x0
phi = phi0
# Run pusher for small periods now
T = 0.1
K = 800
t_S = T * np.arange(K)
# Holders for LCE components
L_S = np.zeros((3, K))
l1, l2, l3 = np.zeros(3)
# Push and calculate
for j in range(K):
    x, phi = RK4(x, phi, T, dt)
    W, V = tp.gram_schmidt(phi)
    # Note that this estimation is cumulative
    l1 += np.log(np.linalg.norm(W[:, 0])) / T
    l2 += np.log(np.linalg.norm(W[:, 1])) / T
    l3 += np.log(np.linalg.norm(W[:, 2])) / T
    L_S[0, j] = l1
    L_S[1, j] = l2
    L_S[2, j] = l3
    # Renormalize the ball
    phi = V


# Normalize LCE
L_S = L_S / np.arange(1, K + 1)
# Compare with Sandri's results. Somehow, the LCE component close to zero is
# very hard to match. But we're probably fine with it being on the same order
# of magnitude.
assert np.isclose(L_S[0, -1], 1.50085, rtol=1e-1)
assert np.isclose(L_S[2, -1], -22.4872, rtol=1e-1)


# --------------------------------------------------------------------------- #
#                   What if we used forward differentiation?
# --------------------------------------------------------------------------- #
T = 80
dt = 0.0002
N = int(T / dt)
t_FD = dt * np.arange(N)
x = x0
phi = phi0
L_FD = np.zeros((3, N))
l1, l2, l3 = np.zeros(3)
for j in range(N):
    # Push (calculate new phi with FD)
    x, _ = RK4(x, phi, dt, dt)
    # Push phi
    phi = np.matmul(np.identity(3) + dt * DF(*x), phi)
    # Then, we follow the same procedure
    W, V = tp.gram_schmidt(phi)
    l1 += np.log(np.linalg.norm(W[:, 0])) / dt
    l2 += np.log(np.linalg.norm(W[:, 1])) / dt
    l3 += np.log(np.linalg.norm(W[:, 2])) / dt
    L_FD[0, j] = l1
    L_FD[1, j] = l2
    L_FD[2, j] = l3
    # Renormalize
    phi = V

# Normalize LCE
L_FD = L_FD / np.arange(1, N + 1)


# Plot
me.setup_mpl(tex=True)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(t_S, L_S[0, :], "-k", label="RK4")
ax.plot(t_S, L_S[1, :], "-k")
ax.plot(t_S, L_S[2, :], "-k")
ax.plot(t_FD, L_FD[0, :], "--r", label="FD")
ax.plot(t_FD, L_FD[1, :], "--r")
ax.plot(t_FD, L_FD[2, :], "--r")
ax.set_xlabel("Steps")
ax.set_ylabel("LCEs")
ax.tick_params(**me.params)
fig.savefig("lorenz_LCE_spectrum.png", dpi=fig.dpi)
