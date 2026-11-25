"""Koordinat sistemleri arası dönüşümler.

Galaktik dönüşüm matrisi ve ufuk dönüşümü ERFA kaynak dosyalarından
okunmuştur [ERFA2023]: ``icrs2g.c``, ``hd2ae.c``, ``ae2hd.c``. Galaktik
sistemin tanımı Hipparcos kataloğundaki biçimdir; ERFA matrisi bu tanımdan
tam duyarlıkla türetir ve katalog kaynağı olarak ESA SP-1200'ü anar.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._kaynak import kaynak
from kozmos.temel.aci import birim_vektor_den_kuresel, kuresel_den_birim_vektor, sifir_iki_pi

Dizi = NDArray[np.float64]

# ICRS → galaktik dönme matrisi; icrs2g.c içindeki 30 basamaklı değerler.
ICRS_DEN_GALAKTIK = np.array(
    [
        [-0.054875560416215368492398900454, -0.873437090234885048760383168409, -0.483835015548713226831774175116],
        [+0.494109427875583673525222371358, -0.444829629960011178146614061616, +0.746982244497218890527388004556],
        [-0.867666149019004701181616534570, -0.198076373431201528180486091412, +0.455983776175066922272100478348],
    ]
)


@kaynak(
    "ERFA2023",
    yer="icrs2g.c",
    teyit="okundu",
    not_="kuzey galaktik kutup α = 192.85948°, δ = 27.12825°; çıkış düğümü boylamı 32.93192° (Hipparcos tanımı)",
)
def icrs_den_galaktik(sag_acikllik_rad: ArrayLike, dik_acikllik_rad: ArrayLike) -> tuple[Dizi, Dizi]:
    """ICRS sağ açıklık ve dik açıklığını galaktik boylam ve enleme çevirir.

    Parametreler
    ------------
    sag_acikllik_rad, dik_acikllik_rad : float ya da ndarray
        ICRS koordinatları [rad].

    Döndürür
    --------
    (l, b) : galaktik boylam [0, 2π) ve enlem [−π/2, π/2] [rad].

    Kaynaklar
    ---------
    [ERFA2023] icrs2g.c.

    Geçerlilik
    ----------
    Dönme matrisi sabittir; dönem ya da öz hareket düzeltmesi yapılmaz.

    Doğrulama
    ---------
    tests/kahin/test_koordinat_kahin.py: erfa.icrs2g ile fark < 1e-14 rad.
    """
    v = kuresel_den_birim_vektor(sag_acikllik_rad, dik_acikllik_rad)
    return birim_vektor_den_kuresel(v @ ICRS_DEN_GALAKTIK.T)


@kaynak("ERFA2023", yer="icrs2g.c", teyit="okundu", not_="ters dönüşüm matrisin devriğidir; g2icrs.c ayrıca okunmadı")
def galaktik_den_icrs(boylam_rad: ArrayLike, enlem_rad: ArrayLike) -> tuple[Dizi, Dizi]:
    """Galaktik boylam ve enlemi ICRS sağ açıklık ve dik açıklığına çevirir.

    Kaynaklar
    ---------
    [ERFA2023] icrs2g.c.

    Doğrulama
    ---------
    tests/kahin/test_koordinat_kahin.py: erfa.g2icrs ile fark < 1e-14 rad; gidiş-dönüş.
    """
    v = kuresel_den_birim_vektor(boylam_rad, enlem_rad)
    return birim_vektor_den_kuresel(v @ ICRS_DEN_GALAKTIK)


@kaynak(
    "ERFA2023",
    yer="hd2ae.c",
    teyit="okundu",
    not_="azimut kuzeyden doğuya; birim vektör bileşenleri hd2ae.c içindeki x, y, z",
)
def saat_acisi_den_ufuk(
    saat_acisi_rad: ArrayLike, dik_acikllik_rad: ArrayLike, enlem_rad: ArrayLike
) -> tuple[Dizi, Dizi]:
    """Saat açısı ve dik açıklığı azimut ve yüksekliğe çevirir.

    Parametreler
    ------------
    saat_acisi_rad : float ya da ndarray
        Yerel saat açısı H = yerel yıldız zamanı − sağ açıklık [rad].
    dik_acikllik_rad : float ya da ndarray
    enlem_rad : float ya da ndarray
        Gözlemcinin enlemi [rad].

    Döndürür
    --------
    (azimut, yükseklik) : azimut [0, 2π), kuzeyden doğuya; yükseklik [−π/2, π/2] [rad].

    Kaynaklar
    ---------
    [ERFA2023] hd2ae.c.

    Geçerlilik
    ----------
    Saf geometrik dönmedir: kırılma, aberasyon ve günlük paralaks içermez.
    Zenitte azimut tanımsızdır ve 0 döner.

    Doğrulama
    ---------
    tests/kahin/test_koordinat_kahin.py: erfa.hd2ae ile fark < 1e-13 rad.
    """
    h = np.asarray(saat_acisi_rad, dtype=float)
    d = np.asarray(dik_acikllik_rad, dtype=float)
    p = np.asarray(enlem_rad, dtype=float)
    x = -np.cos(h) * np.cos(d) * np.sin(p) + np.sin(d) * np.cos(p)
    y = -np.sin(h) * np.cos(d)
    z = np.cos(h) * np.cos(d) * np.cos(p) + np.sin(d) * np.sin(p)
    r = np.hypot(x, y)
    azimut = np.where(r != 0.0, np.arctan2(y, x), 0.0)
    return sifir_iki_pi(azimut), np.arctan2(z, r)


@kaynak("ERFA2023", yer="ae2hd.c", teyit="okundu")
def ufuk_den_saat_acisi(azimut_rad: ArrayLike, yukseklik_rad: ArrayLike, enlem_rad: ArrayLike) -> tuple[Dizi, Dizi]:
    """Azimut ve yüksekliği saat açısı ve dik açıklığa çevirir.

    Döndürür
    --------
    (saat açısı, dik açıklık) : saat açısı (−π, π] [rad].

    Kaynaklar
    ---------
    [ERFA2023] ae2hd.c.

    Doğrulama
    ---------
    tests/kahin/test_koordinat_kahin.py: erfa.ae2hd ile fark < 1e-13 rad; gidiş-dönüş.
    """
    a = np.asarray(azimut_rad, dtype=float)
    e = np.asarray(yukseklik_rad, dtype=float)
    p = np.asarray(enlem_rad, dtype=float)
    x = -np.cos(a) * np.cos(e) * np.sin(p) + np.sin(e) * np.cos(p)
    y = -np.sin(a) * np.cos(e)
    z = np.cos(a) * np.cos(e) * np.cos(p) + np.sin(e) * np.sin(p)
    r = np.hypot(x, y)
    return np.where(r != 0.0, np.arctan2(y, x), 0.0), np.arctan2(z, r)


@kaynak(
    "ERFA2023",
    yer="hd2ae.c (z bileşeni)",
    teyit="okundu",
    not_="sec z yalnızca düzlem-paralel atmosfer yaklaşımıdır; yaklaşımın kaynağı [KastenYoung1989] okunmadı",
)
def hava_kutlesi_duzlem_paralel(yukseklik_rad: ArrayLike) -> Dizi:
    """Düzlem-paralel atmosfer yaklaşımıyla hava kütlesini hesaplar: X = 1 / sin(h).

    Kaynaklar
    ---------
    [ERFA2023] hd2ae.c (yükseklik tanımı).

    Geçerlilik
    ----------
    Yükseklik 30° üstünde yaklaşık binde birkaç doğruluk beklenir (sınanmadı);
    ufka yakın bölgede küresel atmosfer formülleri gerekir. Ufkun altında ``nan`` döner.

    Doğrulama
    ---------
    tests/kahin/test_koordinat_kahin.py: zenitte 1, 30° yükseklikte 2.
    """
    h = np.asarray(yukseklik_rad, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(h > 0.0, 1.0 / np.sin(h), np.nan)
