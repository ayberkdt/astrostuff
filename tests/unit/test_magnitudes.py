import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from kozmos.core import constants
from kozmos.radiation import magnitudes


def test_solar_bolometric_magnitude():
    # IAU 2015 B2: the nominal solar luminosity corresponds to M_Bol ≈ 4.74 mag [Mamajek2015].
    assert float(magnitudes.absolute_bolometric_magnitude(constants.SOLAR_LUMINOSITY.value)) == pytest.approx(
        4.74, abs=0.005
    )
    assert float(magnitudes.absolute_bolometric_magnitude(constants.BOLOMETRIC_ZERO_POINT_LUMINOSITY.value)) == 0.0


def test_distance_modulus():
    assert float(magnitudes.distance_modulus_from_pc(10.0)) == 0.0
    assert float(magnitudes.distance_modulus_from_pc(1e6)) == pytest.approx(25.0)


def test_planck_integral_stefan_boltzmann():
    # ∫ B_ν dν = σ T⁴ / π
    t = 5772.0
    integral, _ = quad(lambda nu: float(magnitudes.planck_frequency(nu, t)), 1e11, 3e16, limit=200)
    assert integral == pytest.approx(float(magnitudes.blackbody_flux(t)) / math.pi, rel=1e-8, abs=0)


def test_wien_peak_wavelength():
    t = 5772.0
    peak = minimize_scalar(
        lambda lam: -float(magnitudes.planck_wavelength(lam, t)),
        bounds=(1e-7, 2e-6),
        method="bounded",
        options={"xatol": 1e-14},
    ).x
    assert peak * t == pytest.approx(constants.WIEN_WAVELENGTH.value, rel=1e-6, abs=0)


def test_solar_luminosity_from_nominal_values():
    # L = 4π R² σ T⁴; the nominal R, T_eff and L agree to one part in a thousand [Prsa2016].
    luminosity = (
        4
        * math.pi
        * constants.SOLAR_RADIUS.value**2
        * float(magnitudes.blackbody_flux(constants.SOLAR_EFFECTIVE_TEMPERATURE.value))
    )
    assert luminosity == pytest.approx(constants.SOLAR_LUMINOSITY.value, rel=1e-3, abs=0)


def test_derived_constants_match_nist_digits():
    # Truncated digits in the NIST table: 5.670 374 419... e-8, 1.438 776 877... e-2 and 2.897 771 955... e-3.
    # The table truncates instead of rounding; the comparison therefore uses the leading digits.
    assert f"{constants.STEFAN_BOLTZMANN.value:.14e}".startswith("5.670374419")
    assert f"{constants.SECOND_RADIATION_CONSTANT.value:.14e}".startswith("1.438776877")
    assert np.isclose(constants.PARSEC.value, 3.0856775814913673e16, rtol=1e-15)
    assert f"{constants.WIEN_WAVELENGTH.value:.14e}".startswith("2.897771955")
    # the root of x = 5 (1 - exp(-x)) behind it, against the Lambert-W closed form x = 5 + W(-5 exp(-5))
    from scipy.special import lambertw

    root = 5.0 + float(np.real(lambertw(-5.0 * np.exp(-5.0))))
    expected = constants.PLANCK.value * constants.SPEED_OF_LIGHT.value / (constants.BOLTZMANN.value * root)
    assert constants.WIEN_WAVELENGTH.value == pytest.approx(expected, rel=1e-15, abs=0)
