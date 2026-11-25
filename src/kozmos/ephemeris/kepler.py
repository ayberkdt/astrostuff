"""İki cisim problemi: Kepler denklemi ve temel bağıntılar.

Denklemler Murray ve Correia'nın derleme bölümünün arXiv TeX kaynağından
(1009.1738, ``Chapter2.tex``) okunmuştur [MurrayCorreia2010]; denklemler
oradaki TeX etiketleriyle anılır.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.angles import TWO_PI, wrap_pm_pi

Array = NDArray[np.float64]


@source(
    "MurrayCorreia2010",
    where="TeX etiketi eq37 (Kepler denklemi), eq38-eq39 (Newton-Raphson)",
    status="read",
    note="kaynak başlangıç için E0 = M önerir; burada E0 = M + e sin M, e > 0,8 için π·işaret(M) (bu projenin seçimi)",
)
def eccentric_anomaly(
    mean_anomaly_rad: ArrayLike, eccentricity: ArrayLike, *, tolerance: float = 1e-14, max_iterations: int = 50
) -> Array:
    """Kepler denklemini eliptik yörünge için çözer: M = E − e sin E.

    Parameters
    ----------
    ortalama_anomali_rad : float ya da ndarray
        Ortalama anomali M [rad].
    dis_merkezlik : float ya da ndarray
        Dış merkezlik, 0 <= e < 1.
    tolerans : float
        Yinelemenin durma ölçütü, |ΔE| [rad].

    Returns
    -------
    float ya da ndarray
        Dış merkezli anomali E [rad]; M ile aynı 2π dalında.

    References
    ----------
    [MurrayCorreia2010] eq37, eq38-eq39.

    Validity
    --------
    0 <= e < 1. e → 1 ve M → 0 yakınında Newton yinelemesi yavaşlar;
    e <= 0,999 için sınanmıştır.

    Verification
    ------------
    tests/property/test_kepler_property.py: |E − e sin E − M| < 1e-12.
    """
    m = np.asarray(mean_anomaly_rad, dtype=float)
    e = np.asarray(eccentricity, dtype=float)
    if np.any((e < 0) | (e >= 1)):
        raise ValueError("dış merkezlik [0, 1) aralığında olmalı")
    m, e = np.broadcast_arrays(m, e)
    m_ind = wrap_pm_pi(m)
    buyuk_e = m_ind + e * np.sin(m_ind)
    # Yüksek dış merkezlikte E0 = π·işaret(M) Newton için güvenli başlangıçtır.
    buyuk_e = np.where(e > 0.8, np.pi * np.sign(m_ind + 1e-300), buyuk_e)
    for _ in range(max_iterations):
        f = buyuk_e - e * np.sin(buyuk_e) - m_ind
        delta = f / (1.0 - e * np.cos(buyuk_e))
        buyuk_e = buyuk_e - delta
        if np.all(np.abs(delta) < tolerance):
            break
    return buyuk_e + (m - m_ind)


@source(
    "MurrayCorreia2010",
    where="TeX etiketi eq17 ve eq33",
    status="read",
    note="r = a(1−e²)/(1+e cos f) = a(1−e cos E); tan(f/2) = sqrt((1+e)/(1−e)) tan(E/2) bu ikisinden türetilir (türetim bu projede)",
)
def true_anomaly(eccentric_anomaly_rad: ArrayLike, eccentricity: ArrayLike) -> Array:
    """Dış merkezli anomaliden gerçek anomaliyi hesaplar.

    References
    ----------
    [MurrayCorreia2010] eq17, eq33.

    Verification
    ------------
    tests/property/test_kepler_property.py: yörünge denklemi r = a(1 − e cos E) = a(1 − e²)/(1 + e cos f).
    """
    buyuk_e = np.asarray(eccentric_anomaly_rad, dtype=float)
    e = np.asarray(eccentricity, dtype=float)
    f = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(buyuk_e / 2.0), np.sqrt(1.0 - e) * np.cos(buyuk_e / 2.0))
    # E ile aynı 2π dalında kal.
    return f + TWO_PI * np.round((buyuk_e - f) / TWO_PI)


@source("MurrayCorreia2010", where="TeX etiketi eq19", status="read", note="T² = 4π² a³ / [G(m1 + m2)]")
def orbital_period_s(semi_major_axis_m: ArrayLike, gm_m3_s2: ArrayLike) -> Array:
    """Yörünge dönemini hesaplar: P = 2π sqrt(a³/μ).

    Parameters
    ----------
    yari_buyuk_eksen_m : float ya da ndarray
        Yarı büyük eksen [m].
    kutle_parametresi_m3_s2 : float ya da ndarray
        μ = G(m1 + m2) [m³ s⁻²].

    References
    ----------
    [MurrayCorreia2010] eq19.

    Verification
    ------------
    tests/unit/test_kepler.py: 1 au ve nominal GM_Güneş için bir Gauss yılı dolayı [Prsa2016].
    """
    a = np.asarray(semi_major_axis_m, dtype=float)
    return TWO_PI * np.sqrt(a**3 / np.asarray(gm_m3_s2, dtype=float))


@source("MurrayCorreia2010", where="TeX etiketi eq30", status="read", note="v² = G(m1 + m2)(2/r − 1/a)")
def orbital_speed_m_s(distance_m: ArrayLike, semi_major_axis_m: ArrayLike, gm_m3_s2: ArrayLike) -> Array:
    """Yörünge hızını vis-viva denkleminden hesaplar.

    References
    ----------
    [MurrayCorreia2010] eq30.

    Verification
    ------------
    tests/unit/test_kepler.py: dairesel yörüngede v = sqrt(μ/r).
    """
    r = np.asarray(distance_m, dtype=float)
    a = np.asarray(semi_major_axis_m, dtype=float)
    return np.sqrt(np.asarray(gm_m3_s2, dtype=float) * (2.0 / r - 1.0 / a))
