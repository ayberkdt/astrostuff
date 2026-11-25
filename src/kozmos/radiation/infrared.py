"""Kızılötesi fotometri: 2MASS ve WISE Vega sıfır noktaları, kadir-akı dönüşümleri.

Sıfır kadir akı yoğunlukları kaynak makalelerin arXiv TeX dosyalarından
okunmuştur: 2MASS için [Cohen2003] Tablo 2 (F_ν(iso) ve λ(iso)), WISE için
[Wright2010] Bölüm 4.4 (F_ν° ve F_λ°). Wien kayma sabiti CODATA 2022
değeridir [CODATA2022].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import infrastructure, source
from kozmos.core.constants import WIEN_WAVELENGTH

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Band:
    """Bir fotometrik bandın Vega sistemindeki sıfır noktası."""

    name: str
    wavelength_um: float
    zero_flux_jy: float
    zero_flux_uncertainty_jy: float
    source: str


# [Cohen2003] Tablo 2: λ(iso) [µm], F_ν(iso) [Jy] ve belirsizliği.
# [Wright2010] Bölüm 4.4: F_ν° = 306.681, 170.663, 29.0448, 8.2839 Jy (W1..W4); genel sistematik belirsizlik ±%1,5.
BANDS: dict[str, Band] = {
    "J": Band("2MASS J", 1.235, 1594.0, 27.80, "Cohen2003"),
    "H": Band("2MASS H", 1.662, 1024.0, 19.95, "Cohen2003"),
    "Ks": Band("2MASS Ks", 2.159, 666.7, 12.55, "Cohen2003"),
    "W1": Band("WISE W1", 3.4, 306.681, 0.015 * 306.681, "Wright2010"),
    "W2": Band("WISE W2", 4.6, 170.663, 0.015 * 170.663, "Wright2010"),
    "W3": Band("WISE W3", 12.0, 29.0448, 0.015 * 29.0448, "Wright2010"),
    "W4": Band("WISE W4", 22.0, 8.2839, 0.015 * 8.2839, "Wright2010"),
}


@infrastructure("bant sözlüğünü sorgular; denklem uygulamaz")
def band(name: str) -> Band:
    """Bant adından (``"J"``, ``"H"``, ``"Ks"``, ``"W1"`` ... ``"W4"``) sıfır noktası kaydını döndürür."""
    try:
        return BANDS[name]
    except KeyError:
        raise ValueError(f"bilinmeyen bant {name!r}; bilinenler: {', '.join(BANDS)}") from None


@source("Cohen2003", where="Tablo 2: F_ν(iso)", status="read", note="2MASS J, H, Ks: 1594, 1024, 666.7 Jy")
@source("Wright2010", where="Bölüm 4.4: F_ν°", status="read", note="WISE W1..W4: 306.681, 170.663, 29.0448, 8.2839 Jy")
def flux_jy_from_vega_magnitude(magnitude: ArrayLike, band_name: str) -> Array:
    """Vega sistemindeki kadirden akı yoğunluğunu hesaplar: F_ν = F_ν° · 10^(−0,4 m).

    Parameters
    ----------
    kadir : float ya da ndarray
        Vega kadiri.
    bant_adi : str
        ``"J"``, ``"H"``, ``"Ks"``, ``"W1"``, ``"W2"``, ``"W3"``, ``"W4"``.

    Returns
    -------
    float ya da ndarray
        Akı yoğunluğu [Jy].

    References
    ----------
    [Cohen2003] Tablo 2; [Wright2010] Bölüm 4.4.

    Validity
    --------
    Sıfır noktaları belirli bir tayf biçimi için tanımlıdır; çok kırmızı ya da
    çok mavi kaynaklarda renk düzeltmesi gerekir. [Wright2010] W3 ve W4 için
    kırmızı kaynaklarda %8 ve %4 düzeyinde düzeltmeden söz eder; burada uygulanmaz.

    Verification
    ------------
    tests/unit/test_infrared.py: sıfır kadirde tablo değeri; 2,5 kadirde onda biri.
    """
    return band(band_name).zero_flux_jy * 10.0 ** (-0.4 * np.asarray(magnitude, dtype=float))


@source("Cohen2003", where="Tablo 2: F_ν(iso)", status="read")
@source("Wright2010", where="Bölüm 4.4: F_ν°", status="read")
def vega_magnitude_from_flux(flux_jy: ArrayLike, band_name: str) -> Array:
    """Akı yoğunluğundan Vega kadirini hesaplar: m = −2,5 log10(F_ν / F_ν°).

    References
    ----------
    [Cohen2003] Tablo 2; [Wright2010] Bölüm 4.4.

    Verification
    ------------
    tests/unit/test_infrared.py: ``vega_kadirinden_aki_jy`` ile gidiş-dönüş.
    """
    return -2.5 * np.log10(np.asarray(flux_jy, dtype=float) / band(band_name).zero_flux_jy)


@source(
    "Cohen2003",
    where="Tablo 2: F_ν(iso)",
    status="read",
    note="m_AB = −2,5 log10(F_ν / 3631 Jy) tanımıyla fark: m_AB − m_Vega = 2,5 log10(3631 / F_ν°). "
    "3631 Jy sabitinin künyesi [OkeGunn1983] okunmadı",
)
@source("Wright2010", where="Bölüm 4.4: F_ν°", status="read")
def vega_ab_offset(band_name: str) -> float:
    """Bir bantta AB ve Vega kadirleri arasındaki farkı hesaplar: m_AB − m_Vega.

    References
    ----------
    [Cohen2003] Tablo 2; [Wright2010] Bölüm 4.4.

    Validity
    --------
    AB sıfır noktası 3631 Jy alınır; bu değerin kaynağı bu çalışmada okunmamıştır.

    Verification
    ------------
    tests/unit/test_infrared.py: Ks bandında yaklaşık 1,84 mag (tablo değerinden hesaplanan).
    """
    return float(2.5 * np.log10(3631.0 / band(band_name).zero_flux_jy))


@source("CODATA2022", where="Wien wavelength displacement law constant", status="read", note="λ_tepe · T = b")
def peak_wavelength_um(temperature_k: ArrayLike) -> Array:
    """Kara cisim ışınımının (B_λ) tepe dalga boyunu hesaplar: λ_tepe = b / T.

    Returns
    -------
    float ya da ndarray
        Tepe dalga boyu [µm].

    References
    ----------
    [CODATA2022] Wien kayma sabiti.

    Validity
    --------
    Tepe, B_λ için tanımlıdır; B_ν'nün tepesi başka bir dalga boyundadır.

    Verification
    ------------
    tests/unit/test_infrared.py: 5772 K için 0,502 µm; 30 K toz için yaklaşık 97 µm.
    """
    return WIEN_WAVELENGTH.value / np.asarray(temperature_k, dtype=float) * 1e6


@source("CODATA2022", where="Wien wavelength displacement law constant", status="read", note="T = b / λ_tepe")
def peak_temperature_k(wavelength_um: ArrayLike) -> Array:
    """Tepe dalga boyundan kara cisim sıcaklığını hesaplar: T = b / λ_tepe.

    Bir bandın en duyarlı olduğu ısıl kaynak sıcaklığının mertebesini verir:
    2MASS Ks için yaklaşık 1340 K, WISE W4 için yaklaşık 130 K.

    References
    ----------
    [CODATA2022] Wien kayma sabiti.

    Verification
    ------------
    tests/unit/test_infrared.py: ``tepe_dalgaboyu_um`` ile gidiş-dönüş.
    """
    return WIEN_WAVELENGTH.value / (np.asarray(wavelength_um, dtype=float) * 1e-6)
