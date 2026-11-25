import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from kozmos.core import constants
from kozmos.radiation import magnitudes


def test_gunes_bolometrik_kadiri():
    # IAU 2015 B2: nominal Güneş ışınım gücü M_Bol ≈ 4,74 mag'a karşılık gelir [Mamajek2015].
    assert float(magnitudes.absolute_bolometric_magnitude(constants.SOLAR_LUMINOSITY.value)) == pytest.approx(
        4.74, abs=0.005
    )
    assert float(magnitudes.absolute_bolometric_magnitude(constants.BOLOMETRIC_ZERO_POINT_LUMINOSITY.value)) == 0.0


def test_uzaklik_modulu():
    assert float(magnitudes.distance_modulus_from_pc(10.0)) == 0.0
    assert float(magnitudes.distance_modulus_from_pc(1e6)) == pytest.approx(25.0)


def test_planck_integrali_stefan_boltzmann():
    # ∫ B_ν dν = σ T⁴ / π
    t = 5772.0
    integral, _ = quad(lambda nu: float(magnitudes.planck_frequency(nu, t)), 1e11, 3e16, limit=200)
    assert integral == pytest.approx(float(magnitudes.blackbody_flux(t)) / math.pi, rel=1e-8)


def test_wien_tepe_dalgaboyu():
    t = 5772.0
    peak = minimize_scalar(
        lambda lam: -float(magnitudes.planck_wavelength(lam, t)),
        bounds=(1e-7, 2e-6),
        method="bounded",
        options={"xatol": 1e-14},
    ).x
    assert peak * t == pytest.approx(constants.WIEN_WAVELENGTH.value, rel=1e-6)


def test_gunes_isinim_gucu_nominal_degerlerden():
    # L = 4π R² σ T⁴; nominal R, T_eff ve L birbiriyle binde bir içinde tutarlıdır [Prsa2016].
    isinim_gucu = (
        4
        * math.pi
        * constants.SOLAR_RADIUS.value**2
        * float(magnitudes.blackbody_flux(constants.SOLAR_EFFECTIVE_TEMPERATURE.value))
    )
    assert isinim_gucu == pytest.approx(constants.SOLAR_LUMINOSITY.value, rel=1e-3)


def test_turetilmis_sabitler_nist_basamaklariyla():
    # NIST tablosundaki kesik basamaklar: 5.670 374 419... e-8 ve 1.438 776 877... e-2.
    # Tablo yuvarlamaz, keser; bu yüzden karşılaştırma baştaki basamaklarla yapılır.
    assert f"{constants.STEFAN_BOLTZMANN.value:.14e}".startswith("5.670374419")
    assert f"{constants.SECOND_RADIATION_CONSTANT.value:.14e}".startswith("1.438776877")
    assert np.isclose(constants.PARSEC.value, 3.0856775814913673e16, rtol=1e-15)
