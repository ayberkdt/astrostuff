"""Açı aritmetiği ve küre üzerinde ayrıklık.

İç gösterim radyandır. Dönüşüm sabitleri ERFA ``erfam.h`` dosyasından
okunmuştur [ERFA2023].
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source

TWO_PI = 6.283185307179586476925287  # erfam.h ERFA_D2PI
DEG_RAD = 1.745329251994329576923691e-2  # erfam.h ERFA_DD2R
ARCSEC_RAD = 4.848136811095359935899141e-6  # erfam.h ERFA_DAS2R
HOUR_RAD = 15.0 * DEG_RAD

Array = NDArray[np.float64]


@source("ERFA2023", where="erfam.h ERFA_DD2R", status="read")
def deg_to_rad(angle_deg: ArrayLike) -> Array:
    """Dereceyi radyana çevirir.

    References
    ----------
    [ERFA2023] erfam.h, ERFA_DD2R.

    Verification
    ------------
    tests/unit/test_angles.py
    """
    return np.asarray(angle_deg, dtype=float) * DEG_RAD


@source("ERFA2023", where="erfam.h ERFA_DD2R", status="read")
def rad_to_deg(angle_rad: ArrayLike) -> Array:
    """Radyanı dereceye çevirir.

    References
    ----------
    [ERFA2023] erfam.h, ERFA_DD2R.

    Verification
    ------------
    tests/unit/test_angles.py
    """
    return np.asarray(angle_rad, dtype=float) / DEG_RAD


@source("ERFA2023", where="erfam.h ERFA_DAS2R", status="read")
def arcsec_to_rad(angle_arcsec: ArrayLike) -> Array:
    """Yay saniyesini radyana çevirir.

    References
    ----------
    [ERFA2023] erfam.h, ERFA_DAS2R.

    Verification
    ------------
    tests/unit/test_angles.py
    """
    return np.asarray(angle_arcsec, dtype=float) * ARCSEC_RAD


@source(
    "ERFA2023",
    where="anp.c",
    status="read",
    note="w = fmod(a, 2π); eksi ise 2π eklenir",
)
def wrap_0_2pi(angle_rad: ArrayLike) -> Array:
    """Açıyı [0, 2π) aralığına indirger.

    References
    ----------
    [ERFA2023] erfam.h, ERFA_D2PI.

    Verification
    ------------
    tests/property/test_angles_property.py: sonuç aralıkta ve girdiyle 2π'nin katı kadar farklı.
    """
    w = np.mod(np.asarray(angle_rad, dtype=float), TWO_PI)
    # Çok küçük eksi girdide mod sonucu yuvarlamayla 2π'ye eşitlenebilir.
    return np.where(w >= TWO_PI, 0.0, w)


@source("ERFA2023", where="anpm.c", status="read", note="ERFA aralığı (−π, π]; burada [−π, π)")
def wrap_pm_pi(angle_rad: ArrayLike) -> Array:
    """Açıyı [−π, π) aralığına indirger.

    References
    ----------
    [ERFA2023] erfam.h, ERFA_D2PI.

    Verification
    ------------
    tests/property/test_angles_property.py
    """
    return wrap_0_2pi(np.asarray(angle_rad, dtype=float) + math.pi) - math.pi


@source("ERFA2023", where="s2c.c", status="read", note="küresel → Kartezyen birim vektör")
def spherical_to_unit_vector(longitude_rad: ArrayLike, latitude_rad: ArrayLike) -> Array:
    """Boylam ve enlemden birim vektör üretir: (cos b cos l, cos b sin l, sin b).

    Returns
    -------
    ndarray, biçim (..., 3)

    References
    ----------
    [ERFA2023] s2c.c.

    Verification
    ------------
    tests/property/test_angles_property.py: norm 1, gidiş-dönüş.
    """
    boylam = np.asarray(longitude_rad, dtype=float)
    enlem = np.asarray(latitude_rad, dtype=float)
    cb = np.cos(enlem)
    return np.stack(np.broadcast_arrays(cb * np.cos(boylam), cb * np.sin(boylam), np.sin(enlem)), axis=-1)


@source("ERFA2023", where="c2s.c", status="read")
def unit_vector_to_spherical(vector: ArrayLike) -> tuple[Array, Array]:
    """Vektörden boylam [0, 2π) ve enlem [−π/2, π/2] üretir.

    References
    ----------
    [ERFA2023] c2s.c.

    Verification
    ------------
    tests/property/test_angles_property.py: gidiş-dönüş.
    """
    v = np.asarray(vector, dtype=float)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return wrap_0_2pi(np.arctan2(y, x)), np.arctan2(z, np.hypot(x, y))


@source(
    "ERFA2023",
    where="sepp.c",
    status="read",
    note="açı = atan2(|a×b|, a·b); kosinüs formülünün küçük açıdaki duyarlılık kaybı yoktur",
)
def angular_separation(
    longitude1_rad: ArrayLike, latitude1_rad: ArrayLike, longitude2_rad: ArrayLike, latitude2_rad: ArrayLike
) -> Array:
    """Küre üzerinde iki yön arasındaki açıyı hesaplar.

    θ = atan2(|a × b|, a · b); a ve b yönlerin birim vektörleridir.

    Parameters
    ----------
    boylam1_rad, enlem1_rad, boylam2_rad, enlem2_rad : float ya da ndarray
        İki yönün boylam ve enlemi [rad].

    Returns
    -------
    float ya da ndarray
        Ayrıklık [rad], [0, π].

    References
    ----------
    [ERFA2023] sepp.c.

    Validity
    --------
    Bütün açılarda kararlıdır; çakışık ve karşıt noktalar dahil.

    Verification
    ------------
    tests/oracle/test_angles_oracle.py: erfa.seps ile fark < 1e-14 rad.
    """
    a = spherical_to_unit_vector(longitude1_rad, latitude1_rad)
    b = spherical_to_unit_vector(longitude2_rad, latitude2_rad)
    capraz = np.linalg.norm(np.cross(a, b), axis=-1)
    nokta = np.sum(a * b, axis=-1)
    return np.arctan2(capraz, nokta)
