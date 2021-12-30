"""
    Distribution functions.
"""


__all__ = [
    "maxwellian",
    "kappa",
    "Distribution",
]


from .constants import c
from scipy.special import gamma
import numpy as np


def maxwellian(vx, vy, vth_x, vth_y, vo_x, vo_y, n):
    r"""Bi-Maxwellian velocity distribution function.

    ----------
    Parameters
    ----------
    vx: km/s
        Parallel (non-relativistic) velocity
    vy: km/s
        Perpendicular (non-relativistic) velocity
    vth_x: km/s
        Parallel thermal velocity
    vth_y: km/s
        Perpendicular thermal velocity
    n: cm-3
        Number density

    -------
    Returns
    -------
    f: cm-3 km-3 s3
        The weight
    """
    A = n / (np.pi ** 1.5) / (vth_y ** 2) / vth_x
    return A * np.exp(
        -(((vx - vo_x) / vth_x) ** 2 + ((vy - vo_y) / vth_y) ** 2)
    )


def kappa(vx, vy, vth_x, vth_y, vo_x, vo_y, n, k):
    r"""Bi-Kappa velocity distribution function.

    ----------
    Parameters
    ----------
    vx: km/s
        Parallel (non-relativistic) velocity
    vy: km/s
        Perpendicular (non-relativistic) velocity
    vth_x: km/s
        Parallel thermal velocity
    vth_y: km/s
        Perpendicular thermal velocity
    vo_x: km/s
        Parallel drift velocity
    vo_y: km/s
        Perpendicular drift velocity
    n: cm-3
        Number density
    k: unitless
        The kappa number

    -------
    Returns
    -------
    f: cm-3 km-3 s3
        The weight
    """
    Ak = (
        (np.pi * (k - 1.5)) ** (-1.5)
        * n
        * gamma(k + 1)
        / gamma(k - 0.5)
        / (vth_y ** 2)
        / vth_x
    )
    Bk = ((vx - vo_x) / vth_x) ** 2 + ((vy - vo_y) / vth_y) ** 2
    return Ak * (1 + Bk / (k - 1.5)) ** (-(k + 1))


def general_VDF(*, n, vx, vy, vth_x, vth_y, vo_x, vo_y, k=None):
    r"""A general VDF. It is bi-Maxwellian when no kappa is provided, otherwise
    it is bi-Kappa.
    """
    if k is None:
        return maxwellian(
            n=n,
            vx=vx,
            vy=vy,
            vth_x=vth_x,
            vth_y=vth_y,
            vo_x=vo_x,
            vo_y=vo_y,
        )
    else:
        return kappa(
            n=n,
            k=k,
            vx=vx,
            vy=vy,
            vth_x=vth_x,
            vth_y=vth_y,
            vo_x=vo_x,
            vo_y=vo_y,
        )


class Distribution:
    r"""Presets for 1 AU and 0.3 AU parameters. See Section 4.2 in the thesis
    for more details.

    1. At 0.3 AU: the total distribution has 2 components: a core and a strahl.
    Both are bi-Maxwellian.

    2. At 1 AU: the total distribution has 3 components: a core, halo, and
    strahl. The core is bi-Maxwellian. The halo & strahl are bi-Kappa.
    """

    core_params = {
        "1 AU": dict(n=13.7, vth_x=1828, vth_y=1828, vo_x=0, vo_y=0),
        "0.3 AU": dict(
            n=332.5,
            vth_x=3889.2,
            vth_y=3889.2,
            vo_x=-479.68,
            vo_y=0,
        ),
    }
    halo_params = {
        "1 AU": dict(
            n=0.52, k=4.62, vth_x=3012.5, vth_y=3012.5, vo_x=0, vo_y=0
        ),
        "0.3 AU": None,
    }
    strahl_params = {
        "1 AU": dict(
            n=0.21,
            k=4.57,
            vth_x=3600,
            vth_y=1300,
            vo_x=2000,
            vo_y=0,
        ),
        "0.3 AU": dict(
            n=17.5,
            vth_x=7935.09,
            vth_y=5626.61,
            vo_x=9293.8,
            vo_y=0,
        ),
    }

    def __init__(self, v_para, v_perp, r="1 AU"):
        r"""Calculate the weight from the velocity.

        ----------
        Parameters
        ----------
        v_para: unitless
            Parallel (non-relativistic) velocity, normalized to c.
        v_perp: unitless
            Perpendicular (non-relativistic) velocity, normalized to c.
        r: str
            Either '1 AU' or '0.3 AU'.
        """
        # To physical units
        self.v_para = v_para
        self.v_perp = v_perp
        self.r = r

    def core(self):
        return general_VDF(
            vx=self.v_para, vy=self.v_perp, **self.core_params[self.r]
        )

    def halo(self):
        return general_VDF(
            vx=self.v_para, vy=self.v_perp, **self.halo_params[self.r]
        )

    def strahl(self):
        return general_VDF(
            vx=self.v_para, vy=self.v_perp, **self.strahl_params[self.r]
        )

    def total(self):
        if self.r == "1 AU":
            return self.core() + self.halo() + self.strahl()
        elif self.r == "0.3 AU":
            return self.core() + self.strahl()
