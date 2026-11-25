"""Gökyüzünde iki kaynak listesinin konuma göre eşlenmesi.

Açısal ayrıklık birim vektörler arasındaki açıdır (ERFA ``sepp.c``)
[ERFA2023]. En yakın komşu araması birim vektörlerin üç boyutlu uzayında
k-d ağacıyla yapılır; kiriş uzunluğu 2 sin(θ/2) açıyla tekdüze arttığı için
kirişe göre en yakın komşu, açıya göre en yakın komşudur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from kozmos._sources import source
from kozmos.core.angles import spherical_to_unit_vector

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Match:
    """Birinci listedeki her kaynağın ikinci listedeki en yakın komşusu.

    Attributes
    ----------
    indis : ndarray[int]
        İkinci listedeki en yakın kaynağın indisi.
    ayriklik_yaysaniye : ndarray
        O kaynağa açısal uzaklık [yay saniyesi].
    eslesti : ndarray[bool]
        Ayrıklık verilen yarıçapın içindeyse doğru.
    """

    index: NDArray[np.int64]
    separation_arcsec: Array
    matched: NDArray[np.bool_]


@source(
    "ERFA2023",
    where="sepp.c, s2c.c",
    status="read",
    note="kiriş = |a − b| = 2 sin(θ/2) ⇒ θ = 2 asin(kiriş/2) (türetim bu projede); komşu araması scipy cKDTree",
)
def cross_match(
    ra1_deg: ArrayLike,
    dec1_deg: ArrayLike,
    ra2_deg: ArrayLike,
    dec2_deg: ArrayLike,
    radius_arcsec: float,
) -> Match:
    """Birinci listedeki her kaynak için ikinci listedeki en yakın kaynağı bulur.

    Parameters
    ----------
    sag_acikllik1_derece, dik_acikllik1_derece : ndarray (N,)
    sag_acikllik2_derece, dik_acikllik2_derece : ndarray (M,)
    yaricap_yaysaniye : float
        Eşleşme sayılacak en büyük ayrıklık.

    References
    ----------
    [ERFA2023] sepp.c, s2c.c.

    Validity
    --------
    Eşleme tek yönlüdür: ikinci listedeki bir kaynak birden çok kaynağın en
    yakın komşusu olabilir. Öz hareket ve dönem farkı düzeltilmez.

    Verification
    ------------
    tests/oracle/test_wcs_oracle.py: astropy match_to_catalog_sky ile aynı indisler ve ayrıklıklar.
    """
    a = spherical_to_unit_vector(np.radians(ra1_deg), np.radians(dec1_deg))
    b = spherical_to_unit_vector(np.radians(ra2_deg), np.radians(dec2_deg))
    kiris, index = cKDTree(np.atleast_2d(b)).query(np.atleast_2d(a), k=1)
    ayriklik = np.degrees(2.0 * np.arcsin(np.clip(kiris / 2.0, 0.0, 1.0))) * 3600.0
    return Match(index.astype(np.int64), ayriklik, ayriklik <= radius_arcsec)
