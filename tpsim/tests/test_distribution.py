from ..distribution import Distribution, maxwellian, kappa
from scipy.integrate import dblquad
import numpy as np


def test_maxwellian():
    n = 5.0

    def density_function(vx, vy):
        vth_x = 1828.0
        vth_y = 1828.0
        return 2 * np.pi * vy * maxwellian(vx, vy, vth_x, vth_y, 0, 0, n)

    ub = 1e5
    assert np.isclose(
        dblquad(density_function, 0, ub, lambda x: -ub, lambda x: ub)[0], n
    )


def test_kappa():
    n = 3.0

    def density_function(vx, vy):
        k = 6.7
        vth_x = 3012.5
        vth_y = 3012.5
        vo_x = 0
        vo_y = 0
        return 2 * np.pi * vy * kappa(vx, vy, vth_x, vth_y, vo_x, vo_y, n, k)

    ub = 1e5
    assert np.isclose(
        dblquad(density_function, 0, ub, lambda x: -ub, lambda x: ub)[0], n
    )
