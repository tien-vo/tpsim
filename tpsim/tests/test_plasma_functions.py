from ..constants import m
from ..plasma_functions import (
    cyclotron_frequency,
    plasma_frequency,
    calculate_SDP,
)
from plasmapy.formulary.dielectric import cold_plasma_permittivity_SDP as SDP
from plasmapy.formulary import wc_, wp_
import astropy.units as u
import numpy as np
import pytest


@pytest.mark.parametrize("B", [1, 10, 500, 1000])
@pytest.mark.parametrize("ptcl", ["p", "e-"])
def test_cyclotron_freq(B, ptcl):
    wc = cyclotron_frequency(B, particle=ptcl)
    wc_official = wc_(B * u.nT, particle=ptcl)
    assert np.isclose(wc, wc_official.value)


@pytest.mark.parametrize("n", [5, 50, 500])
@pytest.mark.parametrize("ptcl", ["p", "e-"])
def test_plasma_freq(n, ptcl):
    wp = plasma_frequency(n, particle=ptcl)
    wp_official = wp_(n/u.Unit("cm3"), particle=ptcl)
    assert np.isclose(wp, wp_official.value)


@pytest.mark.parametrize("B", [1, 10, 50])
@pytest.mark.parametrize("n", [5, 300])
@pytest.mark.parametrize("w_wce", [0.05, 0.1, 0.3])
def test_SDP_calculation(B, n, w_wce):

    # Calculate using our functions
    wce = cyclotron_frequency(B, "e-")
    S, D, P = calculate_SDP(B, n, w_wce * wce)
    # Calculate using plasmapy's function
    S_official, D_official, P_official = SDP(
        B * u.nT,
        ["e-", "p"],
        [n / u.Unit("cm3"), n / u.Unit("cm3")],
        (w_wce * wce) * u.rad / u.s
    )
    assert np.isclose(S, S_official.value)
    assert np.isclose(P, P_official.value)
    assert np.isclose(D, D_official.value)
