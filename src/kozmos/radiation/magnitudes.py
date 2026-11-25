"""Kadir, uzaklık modülü ve kara cisim ışınımı.

Bolometrik kadir sıfır noktası IAU 2015 B2 kararından [Mamajek2015], uzaklık
modülü tanımı [Hogg1999] Denk. 25'ten okunmuştur. Planck fonksiyonu
astropy'nin ``BlackBody`` modelinin kaynak kodundan okunmuştur [Astropy2022]. Sabitler CODATA 2022 değerleridir [CODATA2022].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.constants import BOLOMETRIC_ZERO_POINT_LUMINOSITY, BOLTZMANN, PLANCK, SPEED_OF_LIGHT, STEFAN_BOLTZMANN

Array = NDArray[np.float64]


@source("Mamajek2015", where="M_Bol tanımı; L_0 = 3,0128e28 W", status="read", note="M_Bol = −2,5 log10(L / L_0)")
def absolute_bolometric_magnitude(luminosity_w: ArrayLike) -> Array:
    """Işınım gücünden mutlak bolometrik kadiri hesaplar.

    M_Bol = −2,5 log10(L / L_0), L_0 = 3,0128 × 10²⁸ W.

    Parameters
    ----------
    isinim_gucu_w : float ya da ndarray
        Işınım gücü [W], pozitif.

    References
    ----------
    [Mamajek2015] L_0 tanımı.

    Verification
    ------------
    tests/unit/test_magnitudes.py: nominal Güneş ışınım gücü için 4,74 mag (kararda bildirilen değer).
    """
    return -2.5 * np.log10(np.asarray(luminosity_w, dtype=float) / BOLOMETRIC_ZERO_POINT_LUMINOSITY.value)


@source("Mamajek2015", where="M_Bol tanımı; L_0 = 3,0128e28 W", status="read")
def luminosity_from_bolometric_magnitude_w(absolute_bolometric_magnitude_mag: ArrayLike) -> Array:
    """Mutlak bolometrik kadirden ışınım gücünü hesaplar: L = L_0 · 10^(−0,4 M_Bol).

    References
    ----------
    [Mamajek2015] L_0 tanımı.

    Verification
    ------------
    tests/property/test_magnitudes_property.py: ``mutlak_bolometrik_kadir`` ile gidiş-dönüş.
    """
    m = np.asarray(absolute_bolometric_magnitude_mag, dtype=float)
    return BOLOMETRIC_ZERO_POINT_LUMINOSITY.value * 10.0 ** (-0.4 * m)


@source("Hogg1999", where="Denk. 25", status="read", note="DM = 5 log10(D_L / 10 pc)")
def distance_modulus_from_pc(distance_pc: ArrayLike) -> Array:
    """Parsek cinsinden uzaklıktan uzaklık modülünü hesaplar.

    References
    ----------
    [Hogg1999] Denk. 25.

    Validity
    --------
    Sönümleme ve K düzeltmesi içermez.

    Verification
    ------------
    tests/unit/test_magnitudes.py: 10 pc için 0 mag.
    """
    return 5.0 * np.log10(np.asarray(distance_pc, dtype=float) / 10.0)


@source("Hogg1999", where="Denk. 25", status="read")
def pc_from_distance_modulus(distance_modulus_mag: ArrayLike) -> Array:
    """Uzaklık modülünden parsek cinsinden uzaklığı hesaplar: d = 10^(μ/5 + 1).

    References
    ----------
    [Hogg1999] Denk. 25.

    Verification
    ------------
    tests/property/test_magnitudes_property.py: ``uzaklik_modulu_pc`` ile gidiş-dönüş.
    """
    return 10.0 ** (np.asarray(distance_modulus_mag, dtype=float) / 5.0 + 1.0)


@source(
    "Astropy2022",
    where="modeling/physical_models.py, BlackBody.evaluate",
    status="read",
    note="bb_nu = 2 h ν³ / (c² expm1(hν/kT)); astropy 8.0.0 kaynak kodu. Ders kitabı künyesi [RybickiLightman1979] okunmadı",
)
def planck_frequency(frequency_hz: ArrayLike, temperature_k: ArrayLike) -> Array:
    """Kara cismin özgül yeğinliğini frekans başına hesaplar.

    B_ν(T) = (2 h ν³ / c²) / (exp(hν / kT) − 1)

    Returns
    -------
    float ya da ndarray
        B_ν [W m⁻² Hz⁻¹ sr⁻¹].

    References
    ----------
    [Astropy2022] BlackBody.evaluate.

    Validity
    --------
    ν > 0, T > 0. Küçük hν/kT için ``expm1`` kullanılır; büyük değerlerde üs
    taşması sıfıra iner.

    Verification
    ------------
    tests/unit/test_magnitudes.py: frekans integrali σT⁴/π; tests/oracle/test_radiation_oracle.py: astropy BlackBody.
    """
    nu = np.asarray(frequency_hz, dtype=float)
    t = np.asarray(temperature_k, dtype=float)
    x = PLANCK.value * nu / (BOLTZMANN.value * t)
    with np.errstate(over="ignore"):
        return 2.0 * PLANCK.value * nu**3 / SPEED_OF_LIGHT.value**2 / np.expm1(x)


@source(
    "Astropy2022",
    where="modeling/physical_models.py, BlackBody.evaluate",
    status="read",
    note="B_λ = B_ν c/λ² dönüşümüyle (türetim bu projede): B_λ = (2hc²/λ⁵) / (exp(hc/λkT) − 1)",
)
def planck_wavelength(wavelength_m: ArrayLike, temperature_k: ArrayLike) -> Array:
    """Kara cismin özgül yeğinliğini dalga boyu başına hesaplar.

    Returns
    -------
    float ya da ndarray
        B_λ [W m⁻³ sr⁻¹].

    References
    ----------
    [Astropy2022] BlackBody.evaluate.

    Verification
    ------------
    tests/unit/test_magnitudes.py: tepe dalga boyu Wien sabitiyle uyuşur [CODATA2022].
    """
    lam = np.asarray(wavelength_m, dtype=float)
    t = np.asarray(temperature_k, dtype=float)
    x = PLANCK.value * SPEED_OF_LIGHT.value / (lam * BOLTZMANN.value * t)
    with np.errstate(over="ignore"):
        return 2.0 * PLANCK.value * SPEED_OF_LIGHT.value**2 / lam**5 / np.expm1(x)


@source(
    "CODATA2022",
    where="Stefan-Boltzmann constant",
    status="read",
    note="F = σ T⁴; yasanın kendisi için ders kitabı künyesi bekliyor",
)
def blackbody_flux(temperature_k: ArrayLike) -> Array:
    """Kara cisim yüzeyinden çıkan toplam akıyı hesaplar: F = σ T⁴ [W m⁻²].

    References
    ----------
    [CODATA2022] Stefan-Boltzmann sabiti.

    Verification
    ------------
    tests/unit/test_magnitudes.py: nominal Güneş yarıçapı ve etkin sıcaklığından nominal ışınım gücü [Prsa2016].
    """
    return STEFAN_BOLTZMANN.value * np.asarray(temperature_k, dtype=float) ** 4
