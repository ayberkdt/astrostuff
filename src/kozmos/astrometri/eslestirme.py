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

from kozmos._kaynak import kaynak
from kozmos.temel.aci import kuresel_den_birim_vektor

Dizi = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Eslesme:
    """Birinci listedeki her kaynağın ikinci listedeki en yakın komşusu.

    Öznitelikler
    ------------
    indis : ndarray[int]
        İkinci listedeki en yakın kaynağın indisi.
    ayriklik_yaysaniye : ndarray
        O kaynağa açısal uzaklık [yay saniyesi].
    eslesti : ndarray[bool]
        Ayrıklık verilen yarıçapın içindeyse doğru.
    """

    indis: NDArray[np.int64]
    ayriklik_yaysaniye: Dizi
    eslesti: NDArray[np.bool_]


@kaynak(
    "ERFA2023",
    yer="sepp.c, s2c.c",
    teyit="okundu",
    not_="kiriş = |a − b| = 2 sin(θ/2) ⇒ θ = 2 asin(kiriş/2) (türetim bu projede); komşu araması scipy cKDTree",
)
def capraz_esle(
    sag_acikllik1_derece: ArrayLike,
    dik_acikllik1_derece: ArrayLike,
    sag_acikllik2_derece: ArrayLike,
    dik_acikllik2_derece: ArrayLike,
    yaricap_yaysaniye: float,
) -> Eslesme:
    """Birinci listedeki her kaynak için ikinci listedeki en yakın kaynağı bulur.

    Parametreler
    ------------
    sag_acikllik1_derece, dik_acikllik1_derece : ndarray (N,)
    sag_acikllik2_derece, dik_acikllik2_derece : ndarray (M,)
    yaricap_yaysaniye : float
        Eşleşme sayılacak en büyük ayrıklık.

    Kaynaklar
    ---------
    [ERFA2023] sepp.c, s2c.c.

    Geçerlilik
    ----------
    Eşleme tek yönlüdür: ikinci listedeki bir kaynak birden çok kaynağın en
    yakın komşusu olabilir. Öz hareket ve dönem farkı düzeltilmez.

    Doğrulama
    ---------
    tests/kahin/test_wcs_kahin.py: astropy match_to_catalog_sky ile aynı indisler ve ayrıklıklar.
    """
    a = kuresel_den_birim_vektor(np.radians(sag_acikllik1_derece), np.radians(dik_acikllik1_derece))
    b = kuresel_den_birim_vektor(np.radians(sag_acikllik2_derece), np.radians(dik_acikllik2_derece))
    kiris, indis = cKDTree(np.atleast_2d(b)).query(np.atleast_2d(a), k=1)
    ayriklik = np.degrees(2.0 * np.arcsin(np.clip(kiris / 2.0, 0.0, 1.0))) * 3600.0
    return Eslesme(indis.astype(np.int64), ayriklik, ayriklik <= yaricap_yaysaniye)
