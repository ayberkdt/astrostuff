"""Koordinat sistemleri arası dönüşümler.

Galaktik dönüşüm matrisi ve ufuk dönüşümü ERFA kaynak dosyalarından
okunmuştur [ERFA2023]: ``icrs2g.c``, ``hd2ae.c``, ``ae2hd.c``. Galaktik
sistemin tanımı Hipparcos kataloğundaki biçimdir; ERFA matrisi bu tanımdan
tam duyarlıkla türetir ve katalog kaynağı olarak ESA SP-1200'ü anar.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.angles import spherical_to_unit_vector, unit_vector_to_spherical, wrap_0_2pi

Array = NDArray[np.float64]

# ICRS → galaktik dönme matrisi; icrs2g.c içindeki 30 basamaklı değerler.
ICRS_TO_GALACTIC = np.array(
    [
        [-0.054875560416215368492398900454, -0.873437090234885048760383168409, -0.483835015548713226831774175116],
        [+0.494109427875583673525222371358, -0.444829629960011178146614061616, +0.746982244497218890527388004556],
        [-0.867666149019004701181616534570, -0.198076373431201528180486091412, +0.455983776175066922272100478348],
    ]
)


@source(
    "ERFA2023",
    where="icrs2g.c",
    status="read",
    note="kuzey galaktik kutup α = 192.85948°, δ = 27.12825°; çıkış düğümü boylamı 32.93192° (Hipparcos tanımı)",
)
def icrs_to_galactic(ra_rad: ArrayLike, dec_rad: ArrayLike) -> tuple[Array, Array]:
    """ICRS sağ açıklık ve dik açıklığını galaktik boylam ve enleme çevirir.

    Parameters
    ----------
    sag_acikllik_rad, dik_acikllik_rad : float ya da ndarray
        ICRS koordinatları [rad].

    Returns
    -------
    (l, b) : galaktik boylam [0, 2π) ve enlem [−π/2, π/2] [rad].

    References
    ----------
    [ERFA2023] icrs2g.c.

    Validity
    --------
    Dönme matrisi sabittir; dönem ya da öz hareket düzeltmesi yapılmaz.

    Verification
    ------------
    tests/oracle/test_coordinates_oracle.py: erfa.icrs2g ile fark < 1e-14 rad.
    """
    v = spherical_to_unit_vector(ra_rad, dec_rad)
    return unit_vector_to_spherical(v @ ICRS_TO_GALACTIC.T)


@source("ERFA2023", where="icrs2g.c", status="read", note="ters dönüşüm matrisin devriğidir; g2icrs.c ayrıca okunmadı")
def galactic_to_icrs(longitude_rad: ArrayLike, latitude_rad: ArrayLike) -> tuple[Array, Array]:
    """Galaktik boylam ve enlemi ICRS sağ açıklık ve dik açıklığına çevirir.

    References
    ----------
    [ERFA2023] icrs2g.c.

    Verification
    ------------
    tests/oracle/test_coordinates_oracle.py: erfa.g2icrs ile fark < 1e-14 rad; gidiş-dönüş.
    """
    v = spherical_to_unit_vector(longitude_rad, latitude_rad)
    return unit_vector_to_spherical(v @ ICRS_TO_GALACTIC)


@source(
    "ERFA2023",
    where="hd2ae.c",
    status="read",
    note="azimut kuzeyden doğuya; birim vektör bileşenleri hd2ae.c içindeki x, y, z",
)
def hour_angle_to_horizontal(
    hour_angle_rad: ArrayLike, dec_rad: ArrayLike, latitude_rad: ArrayLike
) -> tuple[Array, Array]:
    """Saat açısı ve dik açıklığı azimut ve yüksekliğe çevirir.

    Parameters
    ----------
    saat_acisi_rad : float ya da ndarray
        Yerel saat açısı H = yerel yıldız zamanı − sağ açıklık [rad].
    dik_acikllik_rad : float ya da ndarray
    enlem_rad : float ya da ndarray
        Gözlemcinin enlemi [rad].

    Returns
    -------
    (azimut, yükseklik) : azimut [0, 2π), kuzeyden doğuya; yükseklik [−π/2, π/2] [rad].

    References
    ----------
    [ERFA2023] hd2ae.c.

    Validity
    --------
    Saf geometrik dönmedir: kırılma, aberasyon ve günlük paralaks içermez.
    Zenitte azimut tanımsızdır ve 0 döner.

    Verification
    ------------
    tests/oracle/test_coordinates_oracle.py: erfa.hd2ae ile fark < 1e-13 rad.
    """
    h = np.asarray(hour_angle_rad, dtype=float)
    d = np.asarray(dec_rad, dtype=float)
    p = np.asarray(latitude_rad, dtype=float)
    x = -np.cos(h) * np.cos(d) * np.sin(p) + np.sin(d) * np.cos(p)
    y = -np.sin(h) * np.cos(d)
    z = np.cos(h) * np.cos(d) * np.cos(p) + np.sin(d) * np.sin(p)
    r = np.hypot(x, y)
    azimut = np.where(r != 0.0, np.arctan2(y, x), 0.0)
    return wrap_0_2pi(azimut), np.arctan2(z, r)


@source("ERFA2023", where="ae2hd.c", status="read")
def horizontal_to_hour_angle(
    azimuth_rad: ArrayLike, altitude_rad: ArrayLike, latitude_rad: ArrayLike
) -> tuple[Array, Array]:
    """Azimut ve yüksekliği saat açısı ve dik açıklığa çevirir.

    Returns
    -------
    (saat açısı, dik açıklık) : saat açısı (−π, π] [rad].

    References
    ----------
    [ERFA2023] ae2hd.c.

    Verification
    ------------
    tests/oracle/test_coordinates_oracle.py: erfa.ae2hd ile fark < 1e-13 rad; gidiş-dönüş.
    """
    a = np.asarray(azimuth_rad, dtype=float)
    e = np.asarray(altitude_rad, dtype=float)
    p = np.asarray(latitude_rad, dtype=float)
    x = -np.cos(a) * np.cos(e) * np.sin(p) + np.sin(e) * np.cos(p)
    y = -np.sin(a) * np.cos(e)
    z = np.cos(a) * np.cos(e) * np.cos(p) + np.sin(e) * np.sin(p)
    r = np.hypot(x, y)
    return np.where(r != 0.0, np.arctan2(y, x), 0.0), np.arctan2(z, r)


@source(
    "ERFA2023",
    where="hd2ae.c (z bileşeni)",
    status="read",
    note="sec z yalnızca düzlem-paralel atmosfer yaklaşımıdır; yaklaşımın kaynağı [KastenYoung1989] okunmadı",
)
def airmass_plane_parallel(altitude_rad: ArrayLike) -> Array:
    """Düzlem-paralel atmosfer yaklaşımıyla hava kütlesini hesaplar: X = 1 / sin(h).

    References
    ----------
    [ERFA2023] hd2ae.c (yükseklik tanımı).

    Validity
    --------
    Yükseklik 30° üstünde yaklaşık binde birkaç doğruluk beklenir (sınanmadı);
    ufka yakın bölgede küresel atmosfer formülleri gerekir. Ufkun altında ``nan`` döner.

    Verification
    ------------
    tests/oracle/test_coordinates_oracle.py: zenitte 1, 30° yükseklikte 2.
    """
    h = np.asarray(altitude_rad, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(h > 0.0, 1.0 / np.sin(h), np.nan)
