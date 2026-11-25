"""Açı aritmetiği ve küre üzerinde ayrıklık.

İç gösterim radyandır. Dönüşüm sabitleri ERFA ``erfam.h`` dosyasından
okunmuştur [ERFA2023].
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._kaynak import kaynak

IKI_PI = 6.283185307179586476925287  # erfam.h ERFA_D2PI
DERECE_RAD = 1.745329251994329576923691e-2  # erfam.h ERFA_DD2R
YAYSANIYE_RAD = 4.848136811095359935899141e-6  # erfam.h ERFA_DAS2R
SAAT_RAD = 15.0 * DERECE_RAD

Dizi = NDArray[np.float64]


@kaynak("ERFA2023", yer="erfam.h ERFA_DD2R", teyit="okundu")
def derece_den_radyan(aci_derece: ArrayLike) -> Dizi:
    """Dereceyi radyana çevirir.

    Kaynaklar
    ---------
    [ERFA2023] erfam.h, ERFA_DD2R.

    Doğrulama
    ---------
    tests/birim/test_aci.py
    """
    return np.asarray(aci_derece, dtype=float) * DERECE_RAD


@kaynak("ERFA2023", yer="erfam.h ERFA_DD2R", teyit="okundu")
def radyan_dan_derece(aci_rad: ArrayLike) -> Dizi:
    """Radyanı dereceye çevirir.

    Kaynaklar
    ---------
    [ERFA2023] erfam.h, ERFA_DD2R.

    Doğrulama
    ---------
    tests/birim/test_aci.py
    """
    return np.asarray(aci_rad, dtype=float) / DERECE_RAD


@kaynak("ERFA2023", yer="erfam.h ERFA_DAS2R", teyit="okundu")
def yaysaniye_den_radyan(aci_yaysaniye: ArrayLike) -> Dizi:
    """Yay saniyesini radyana çevirir.

    Kaynaklar
    ---------
    [ERFA2023] erfam.h, ERFA_DAS2R.

    Doğrulama
    ---------
    tests/birim/test_aci.py
    """
    return np.asarray(aci_yaysaniye, dtype=float) * YAYSANIYE_RAD


@kaynak(
    "ERFA2023",
    yer="anp.c",
    teyit="okundu",
    not_="w = fmod(a, 2π); eksi ise 2π eklenir",
)
def sifir_iki_pi(aci_rad: ArrayLike) -> Dizi:
    """Açıyı [0, 2π) aralığına indirger.

    Kaynaklar
    ---------
    [ERFA2023] erfam.h, ERFA_D2PI.

    Doğrulama
    ---------
    tests/ozellik/test_aci_ozellik.py: sonuç aralıkta ve girdiyle 2π'nin katı kadar farklı.
    """
    w = np.mod(np.asarray(aci_rad, dtype=float), IKI_PI)
    # Çok küçük eksi girdide mod sonucu yuvarlamayla 2π'ye eşitlenebilir.
    return np.where(w >= IKI_PI, 0.0, w)


@kaynak("ERFA2023", yer="anpm.c", teyit="okundu", not_="ERFA aralığı (−π, π]; burada [−π, π)")
def eksi_pi_arti_pi(aci_rad: ArrayLike) -> Dizi:
    """Açıyı [−π, π) aralığına indirger.

    Kaynaklar
    ---------
    [ERFA2023] erfam.h, ERFA_D2PI.

    Doğrulama
    ---------
    tests/ozellik/test_aci_ozellik.py
    """
    return sifir_iki_pi(np.asarray(aci_rad, dtype=float) + math.pi) - math.pi


@kaynak("ERFA2023", yer="s2c.c", teyit="okundu", not_="küresel → Kartezyen birim vektör")
def kuresel_den_birim_vektor(boylam_rad: ArrayLike, enlem_rad: ArrayLike) -> Dizi:
    """Boylam ve enlemden birim vektör üretir: (cos b cos l, cos b sin l, sin b).

    Döndürür
    --------
    ndarray, biçim (..., 3)

    Kaynaklar
    ---------
    [ERFA2023] s2c.c.

    Doğrulama
    ---------
    tests/ozellik/test_aci_ozellik.py: norm 1, gidiş-dönüş.
    """
    boylam = np.asarray(boylam_rad, dtype=float)
    enlem = np.asarray(enlem_rad, dtype=float)
    cb = np.cos(enlem)
    return np.stack(np.broadcast_arrays(cb * np.cos(boylam), cb * np.sin(boylam), np.sin(enlem)), axis=-1)


@kaynak("ERFA2023", yer="c2s.c", teyit="okundu")
def birim_vektor_den_kuresel(vektor: ArrayLike) -> tuple[Dizi, Dizi]:
    """Vektörden boylam [0, 2π) ve enlem [−π/2, π/2] üretir.

    Kaynaklar
    ---------
    [ERFA2023] c2s.c.

    Doğrulama
    ---------
    tests/ozellik/test_aci_ozellik.py: gidiş-dönüş.
    """
    v = np.asarray(vektor, dtype=float)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return sifir_iki_pi(np.arctan2(y, x)), np.arctan2(z, np.hypot(x, y))


@kaynak(
    "ERFA2023",
    yer="sepp.c",
    teyit="okundu",
    not_="açı = atan2(|a×b|, a·b); kosinüs formülünün küçük açıdaki duyarlılık kaybı yoktur",
)
def acisal_ayriklik(
    boylam1_rad: ArrayLike, enlem1_rad: ArrayLike, boylam2_rad: ArrayLike, enlem2_rad: ArrayLike
) -> Dizi:
    """Küre üzerinde iki yön arasındaki açıyı hesaplar.

    θ = atan2(|a × b|, a · b); a ve b yönlerin birim vektörleridir.

    Parametreler
    ------------
    boylam1_rad, enlem1_rad, boylam2_rad, enlem2_rad : float ya da ndarray
        İki yönün boylam ve enlemi [rad].

    Döndürür
    --------
    float ya da ndarray
        Ayrıklık [rad], [0, π].

    Kaynaklar
    ---------
    [ERFA2023] sepp.c.

    Geçerlilik
    ----------
    Bütün açılarda kararlıdır; çakışık ve karşıt noktalar dahil.

    Doğrulama
    ---------
    tests/kahin/test_aci_kahin.py: erfa.seps ile fark < 1e-14 rad.
    """
    a = kuresel_den_birim_vektor(boylam1_rad, enlem1_rad)
    b = kuresel_den_birim_vektor(boylam2_rad, enlem2_rad)
    capraz = np.linalg.norm(np.cross(a, b), axis=-1)
    nokta = np.sum(a * b, axis=-1)
    return np.arctan2(capraz, nokta)
