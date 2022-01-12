from ..pusher import advance
import numpy as np
import pytest


def model(t, x, y, z, ux, uy, uz):
    """Some arbitrary model with constant Ex and Bz"""
    N = x.size
    Ex = np.ones(N)
    Ey = np.zeros(N)
    Ez = np.zeros(N)
    Bx = np.zeros(N)
    By = np.zeros(N)
    Bz = np.ones(N)
    return Ex, Ey, Ez, Bx, By, Bz


def test_pusher_calculations():
    """Check a case by hand"""

    state = np.array([0, 0, 0, 0.2, 0, 0.03])
    _, x, y, z, ux, uy, uz = advance(0, *state.T, model, 1e-5, particle="e-")
    assert np.isclose(x, 1.96024e-6)
    assert np.isclose(y, 1.92138e-11)
    assert np.isclose(z, 2.94048e-7)
    assert np.isclose(ux, 0.199991)
    assert np.isclose(uy, 1.96027e-6)
    assert np.isclose(uz, 0.03)
