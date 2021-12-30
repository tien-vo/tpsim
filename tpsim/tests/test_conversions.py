from ..conversions import (
    KE2S, S2KE, ES2VS, VS2ES, ES2US, US2ES, gamma, normalize_E,
)
from ..plasma_functions import cyclotron_frequency
from ..constants import c, m_e
from plasmapy.formulary import Lorentz_factor
from plasmapy.formulary import wc_
import astropy.units as u
import numpy as np
import pytest


@pytest.mark.parametrize("V", np.linspace(0, 10000, 10))
def test_gamma(V):
    assert np.isclose(gamma(V), Lorentz_factor(V * u.km / u.s))


@pytest.mark.parametrize("E", np.linspace(1, 100, 10))
def test_electric_field_conversion(E):
    B0 = 10

    E_check = (E * u.Unit("mV/m") / (c * u.km / u.s) / (B0 * u.nT)).decompose()
    assert np.isclose(normalize_E(E, B0), E_check.value)


def test_energy_conversion_to_speed():
    """
    Test energy conversion to speed. Test value was calculated from
    CASIO fx-115ES PLUS calculator.
    """
    energy = 100  # eV
    calculated_speed = KE2S(energy)
    assert np.isclose(calculated_speed, 5930.099381)
    assert np.isclose(S2KE(calculated_speed), energy)


class TestVelocitySpaceConversion:
    """
    Test conversion from velocity space to 'energy space'
    (polar coordinate with r=KE)
    """

    KE_ref = 843.671725  # eV
    G_ref = np.radians(1.145762838)  # rad
    P_ref = np.radians(35.54309676)  # rad
    # Non-relativistic velocity corresponding to reference KE, G, P
    V = np.array([10000, 200, 14000])

    def test_non_relativistic_velocity_conversion(self):
        KE_test, G_test, P_test = VS2ES(*self.V)
        assert np.isclose(KE_test, self.KE_ref)
        assert np.isclose(G_test, self.G_ref)
        assert np.isclose(P_test, self.P_ref)
        # Invert conversion
        vx, vy, vz = ES2VS(KE_test, G_test, P_test)
        assert np.isclose(vx, self.V[0])
        assert np.isclose(vy, self.V[1])
        assert np.isclose(vz, self.V[2])

    def test_relativistic_velocity_conversion(self):
        # Calculate gamma
        g = gamma(np.linalg.norm(self.V))
        # Calculate relativistic velocity
        U = ES2US(self.KE_ref, self.G_ref, self.P_ref)
        assert np.isclose(U / g, self.V).all()
        # Invert conversion
        KE_test, G_test, P_test = US2ES(*U)
        assert np.isclose(KE_test, self.KE_ref)
        assert np.isclose(G_test, self.G_ref)
        assert np.isclose(P_test, self.P_ref)
