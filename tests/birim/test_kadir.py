import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from kozmos.isinim import kadir
from kozmos.temel import sabitler as s


def test_gunes_bolometrik_kadiri():
    # IAU 2015 B2: nominal Güneş ışınım gücü M_Bol ≈ 4,74 mag'a karşılık gelir [Mamajek2015].
    assert float(kadir.mutlak_bolometrik_kadir(s.GUNES_ISINIM_GUCU.deger)) == pytest.approx(4.74, abs=0.005)
    assert float(kadir.mutlak_bolometrik_kadir(s.BOLOMETRIK_SIFIR_ISINIM_GUCU.deger)) == 0.0


def test_uzaklik_modulu():
    assert float(kadir.uzaklik_modulu_pc(10.0)) == 0.0
    assert float(kadir.uzaklik_modulu_pc(1e6)) == pytest.approx(25.0)


def test_planck_integrali_stefan_boltzmann():
    # ∫ B_ν dν = σ T⁴ / π
    t = 5772.0
    integral, _ = quad(lambda nu: float(kadir.planck_frekans(nu, t)), 1e11, 3e16, limit=200)
    assert integral == pytest.approx(float(kadir.kara_cisim_akisi(t)) / math.pi, rel=1e-8)


def test_wien_tepe_dalgaboyu():
    t = 5772.0
    tepe = minimize_scalar(
        lambda lam: -float(kadir.planck_dalgaboyu(lam, t)),
        bounds=(1e-7, 2e-6),
        method="bounded",
        options={"xatol": 1e-14},
    ).x
    assert tepe * t == pytest.approx(s.WIEN_DALGABOYU.deger, rel=1e-6)


def test_gunes_isinim_gucu_nominal_degerlerden():
    # L = 4π R² σ T⁴; nominal R, T_eff ve L birbiriyle binde bir içinde tutarlıdır [Prsa2016].
    isinim_gucu = 4 * math.pi * s.GUNES_YARICAPI.deger**2 * float(kadir.kara_cisim_akisi(s.GUNES_ETKIN_SICAKLIGI.deger))
    assert isinim_gucu == pytest.approx(s.GUNES_ISINIM_GUCU.deger, rel=1e-3)


def test_turetilmis_sabitler_nist_basamaklariyla():
    # NIST tablosundaki kesik basamaklar: 5.670 374 419... e-8 ve 1.438 776 877... e-2.
    # Tablo yuvarlamaz, keser; bu yüzden karşılaştırma baştaki basamaklarla yapılır.
    assert f"{s.STEFAN_BOLTZMANN.deger:.14e}".startswith("5.670374419")
    assert f"{s.IKINCI_ISINIM_SABITI.deger:.14e}".startswith("1.438776877")
    assert np.isclose(s.PARSEK.deger, 3.0856775814913673e16, rtol=1e-15)
