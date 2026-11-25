"""Aykırı değerlere dayanıklı konum ve ölçek kestiricileri.

Çift ağırlıklı (biweight) kestiricilerin kaynağı [Beers1990] makalesidir;
makalenin metni bu çalışmada açılmamıştır ve kayıtlar "bekliyor"
durumundadır. Formüller ``astropy.stats`` ile sayısal olarak karşılaştırılır.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._kaynak import kaynak

Dizi = NDArray[np.float64]
Maske = NDArray[np.bool_]

# Gauss dağılımında σ = MAD / Φ⁻¹(3/4); 1 / 0.6744897501960817.
MAD_DEN_SIGMA = 1.482602218505602


@kaynak(
    "Beers1990",
    yer="ortanca mutlak sapma (MAD) ve Gauss ölçeklemesi",
    teyit="bekliyor",
    not_="σ ≈ 1,4826 · ortanca(|x − ortanca(x)|); katsayı 1/Φ⁻¹(3/4)",
)
def mad_olcek(x: ArrayLike, eksen: int | None = None) -> Dizi:
    """Ortanca mutlak sapmadan Gauss eşdeğeri standart sapmayı kestirir.

    Kaynaklar
    ---------
    [Beers1990] MAD tanımı.

    Geçerlilik
    ----------
    Örneklerin yarısına kadar aykırı değere dayanır. Gauss veride verimi
    standart sapmanınkinin yaklaşık %37'sidir (sınanmadı).

    Doğrulama
    ---------
    tests/kahin/test_saglam_kahin.py: astropy.stats.mad_std ile tam eşitlik.
    """
    a = np.asarray(x, dtype=float)
    ortanca = np.median(a, axis=eksen, keepdims=True)
    return MAD_DEN_SIGMA * np.median(np.abs(a - ortanca), axis=eksen)


@kaynak(
    "Beers1990",
    yer="çift ağırlıklı konum kestiricisi",
    teyit="bekliyor",
    not_="ζ = M + Σ(x−M)(1−u²)² / Σ(1−u²)², u = (x−M)/(c·MAD), |u| < 1; c = 6",
)
def ciftagirlik_konum(x: ArrayLike, c: float = 6.0) -> float:
    """Çift ağırlıklı (biweight) konum kestiricisini hesaplar.

    Kaynaklar
    ---------
    [Beers1990] çift ağırlıklı konum.

    Geçerlilik
    ----------
    MAD sıfırsa (örneklerin yarısından fazlası aynı değerde) ortanca döner.

    Doğrulama
    ---------
    tests/kahin/test_saglam_kahin.py: astropy.stats.biweight_location ile bağıl fark < 1e-12.
    """
    a = np.asarray(x, dtype=float).ravel()
    m = float(np.median(a))
    mad = float(np.median(np.abs(a - m)))
    if mad == 0.0:
        return m
    u = (a - m) / (c * mad)
    agirlik = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)
    return m + float(np.sum((a - m) * agirlik) / np.sum(agirlik))


@kaynak(
    "Beers1990",
    yer="çift ağırlıklı ölçek kestiricisi",
    teyit="bekliyor",
    not_="S = √n · sqrt(Σ(x−M)²(1−u²)⁴) / |Σ(1−u²)(1−5u²)|, u = (x−M)/(c·MAD), |u| < 1; c = 9",
)
def ciftagirlik_olcek(x: ArrayLike, c: float = 9.0) -> float:
    """Çift ağırlıklı (biweight) ölçek kestiricisini hesaplar.

    Kaynaklar
    ---------
    [Beers1990] çift ağırlıklı ölçek.

    Geçerlilik
    ----------
    MAD sıfırsa 0 döner.

    Doğrulama
    ---------
    tests/kahin/test_saglam_kahin.py: astropy.stats.biweight_scale ile bağıl fark < 1e-12.
    """
    a = np.asarray(x, dtype=float).ravel()
    m = float(np.median(a))
    mad = float(np.median(np.abs(a - m)))
    if mad == 0.0:
        return 0.0
    u = (a - m) / (c * mad)
    ic = np.abs(u) < 1.0
    d, u2 = (a - m)[ic], u[ic] ** 2
    pay = np.sqrt(np.sum(d**2 * (1.0 - u2) ** 4))
    payda = np.abs(np.sum((1.0 - u2) * (1.0 - 5.0 * u2)))
    return float(np.sqrt(a.size) * pay / payda)


@kaynak(
    "Beers1990",
    yer="yinelemeli kırpma",
    teyit="bekliyor",
    not_="her yinelemede merkez = ortanca, ölçek = standart sapma; |x − merkez| > kσ olanlar atılır. "
    "Bu kural astropy.stats.sigma_clip varsayılanlarıyla aynıdır; yöntemin özgün künyesi belirlenmedi",
)
def sigma_kirp(x: ArrayLike, sigma: float = 3.0, yineleme: int = 5, eksen: int | None = None) -> Maske:
    """Yinelemeli sigma kırpma uygular ve atılan elemanların maskesini döndürür.

    Parametreler
    ------------
    x : ndarray
    sigma : float
        Kırpma eşiği (standart sapma katı).
    yineleme : int
        En çok yineleme sayısı; hiçbir eleman atılmayınca erken durur.
    eksen : int, isteğe bağlı
        Kırpmanın uygulanacağı eksen; verilmezse bütün dizi tek örneklemdir.

    Döndürür
    --------
    ndarray[bool]
        ``True``: eleman atıldı.

    Kaynaklar
    ---------
    [Beers1990] yinelemeli kırpma.

    Geçerlilik
    ----------
    Ölçek kestiricisi standart sapmadır; tek bir çok büyük aykırı değer ilk
    yinelemede ölçeği şişirir ve komşu aykırı değerleri gizleyebilir.

    Doğrulama
    ---------
    tests/kahin/test_saglam_kahin.py: astropy.stats.sigma_clip maskesiyle tam eşitlik.
    """
    a = np.asarray(x, dtype=float)
    maske = ~np.isfinite(a)
    for _ in range(yineleme):
        dolu = np.where(maske, np.nan, a)
        merkez = np.nanmedian(dolu, axis=eksen, keepdims=True)
        olcek = np.nanstd(dolu, axis=eksen, keepdims=True)
        yeni = maske | (np.abs(a - merkez) > sigma * olcek)
        if np.array_equal(yeni, maske):
            break
        maske = yeni
    return maske
