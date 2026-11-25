"""Çerçeve yanlılığı ve presesyon: ICRS koordinatlarını tarihin ortalama ekvatoruna taşıma.

IAU 2006 presesyonunun Fukushima-Williams açıları ve bunlardan kurulan dönme
matrisi ERFA kaynak kodundan okunmuştur [ERFA2023]: ``pfw06.c`` (açıların
polinomları), ``fw2m.c`` (matris), ``pmat06.c`` (ikisinin birleşimi).

Nutasyon (en çok yaklaşık 17 yay saniyesi) ve aberasyon (yaklaşık 20 yay
saniyesi) burada yoktur; sonuç "tarihin ortalama" koordinatlarıdır, "görünür"
koordinatlar değil.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._kaynak import kaynak
from kozmos.temel.aci import IKI_PI
from kozmos.zaman.julyen import ortalama_egiklik

Dizi = NDArray[np.float64]

_JD_J2000 = 2451545.0  # erfam.h ERFA_DJ00
_JULYEN_YUZYILI_GUN = 36525.0  # erfam.h ERFA_DJC
_YAYSANIYE_RAD = np.pi / (180.0 * 3600.0)

# pfw06.c: P03 yanlılık + presesyon açılarının polinom katsayıları [yay saniyesi], t'nin artan kuvvetleri.
_GAMB = (-0.052928, 10.556378, 0.4932044, -0.00031238, -0.000002788, 0.0000000260)
_PHIB = (84381.412819, -46.811016, 0.0511268, 0.00053289, -0.000000440, -0.0000000176)
_PSIB = (-0.041775, 5038.481484, 1.5584175, -0.00018522, -0.000026452, -0.0000000148)


def _polinom(katsayilar: tuple[float, ...], t: Dizi) -> Dizi:
    toplam = np.zeros_like(t)
    for k in reversed(katsayilar):
        toplam = toplam * t + k
    return toplam * _YAYSANIYE_RAD


@kaynak(
    "ERFA2023",
    yer="pfw06.c",
    teyit="okundu",
    not_="t = ((date1 − 2451545,0) + date2) / 36525 (TT, Jülyen yüzyılı); γ̄, φ̄, ψ̄ beşinci dereceden polinomlar; "
    "ε_A = eraObl06",
)
def fukushima_williams_acilari(jd1: ArrayLike, jd2: ArrayLike = 0.0) -> tuple[Dizi, Dizi, Dizi, Dizi]:
    """IAU 2006 yanlılık-presesyon Fukushima-Williams açılarını hesaplar.

    Parametreler
    ------------
    jd1, jd2 : float ya da ndarray
        İki parçalı Jülyen tarihi (TT).

    Döndürür
    --------
    (γ̄, φ̄, ψ̄, ε_A) [rad]

    Kaynaklar
    ---------
    [ERFA2023] pfw06.c.

    Geçerlilik
    ----------
    Polinomlar J2000 çevresinde birkaç yüzyıl için kurulmuştur; uzak tarihlerde doğruluk düşer.

    Doğrulama
    ---------
    tests/kahin/test_presesyon_kahin.py: erfa.pfw06 ile eşitlik.
    """
    t = ((np.asarray(jd1, dtype=float) - _JD_J2000) + np.asarray(jd2, dtype=float)) / _JULYEN_YUZYILI_GUN
    return _polinom(_GAMB, t), _polinom(_PHIB, t), _polinom(_PSIB, t), ortalama_egiklik(jd1, jd2)


def _rz(aci: Dizi) -> Dizi:
    c, s, sifir, bir = np.cos(aci), np.sin(aci), np.zeros_like(aci), np.ones_like(aci)
    return np.stack([np.stack([c, s, sifir], -1), np.stack([-s, c, sifir], -1), np.stack([sifir, sifir, bir], -1)], -2)


def _rx(aci: Dizi) -> Dizi:
    c, s, sifir, bir = np.cos(aci), np.sin(aci), np.zeros_like(aci), np.ones_like(aci)
    return np.stack([np.stack([bir, sifir, sifir], -1), np.stack([sifir, c, s], -1), np.stack([sifir, -s, c], -1)], -2)


@kaynak(
    "ERFA2023",
    yer="pmat06.c, fw2m.c",
    teyit="okundu",
    not_="R = R_x(−ε_A) R_z(−ψ̄) R_x(φ̄) R_z(γ̄) (fw2m.c'deki çağrı sırası: Rz(γ̄), Rx(φ̄), Rz(−ψ̄), Rx(−ε)). "
    "Temel dönme matrislerinin işaret kuralı rx.c ve rz.c'den okunmadı; erfa.pmat06 ile eşitlik sınandı",
)
def yanlilik_presesyon_matrisi(jd1: ArrayLike, jd2: ArrayLike = 0.0) -> Dizi:
    """GCRS (ICRS yönelimli) vektörleri tarihin ortalama ekvator ve ekinoksuna döndüren matrisi kurar.

    v(tarih) = R · v(GCRS).

    Döndürür
    --------
    ndarray (..., 3, 3)

    Kaynaklar
    ---------
    [ERFA2023] pmat06.c, fw2m.c.

    Geçerlilik
    ----------
    Çerçeve yanlılığı ve presesyonu içerir; nutasyonu içermez.

    Doğrulama
    ---------
    tests/kahin/test_presesyon_kahin.py: erfa.pmat06 ile 1e-14 içinde eşitlik; matris dik (ortogonal).
    """
    gamb, phib, psib, epsa = fukushima_williams_acilari(jd1, jd2)
    return _rx(-epsa) @ _rz(-psib) @ _rx(phib) @ _rz(gamb)


@kaynak(
    "ERFA2023",
    yer="pmat06.c (matris), s2c.c ve c2s.c (küresel ↔ birim vektör)",
    teyit="okundu",
    not_="yön birim vektöre çevrilir, yanlılık-presesyon matrisiyle döndürülür ve küresel koordinata geri çevrilir",
)
def icrs_den_tarihin_ortalamasina(
    sag_acikllik_rad: ArrayLike, dik_acikllik_rad: ArrayLike, jd1: ArrayLike, jd2: ArrayLike = 0.0
) -> tuple[Dizi, Dizi]:
    """ICRS koordinatlarını tarihin ortalama ekvator ve ekinoksuna taşır.

    Parametreler
    ------------
    sag_acikllik_rad, dik_acikllik_rad : float ya da ndarray
        ICRS koordinatları [rad].
    jd1, jd2 : float ya da ndarray
        İki parçalı Jülyen tarihi (TT); koordinatlarla yayınlanabilir biçimde.

    Döndürür
    --------
    (sağ açıklık [0, 2π), dik açıklık) [rad]

    Kaynaklar
    ---------
    [ERFA2023] pmat06.c, s2c.c, c2s.c.

    Geçerlilik
    ----------
    Öz hareket, paralaks, nutasyon ve aberasyon uygulanmaz; kalan fark yıldızlar için
    yaklaşık yarım yay dakikasına kadardır.

    Doğrulama
    ---------
    tests/kahin/test_presesyon_kahin.py: erfa.pmat06 ile döndürülmüş vektörle eşitlik; astropy'nin
    tarihin ortalama ekvatoru çerçevesiyle (PrecessedGeocentric) karşılaştırma.
    """
    a, d = np.asarray(sag_acikllik_rad, dtype=float), np.asarray(dik_acikllik_rad, dtype=float)
    v = np.stack(np.broadcast_arrays(np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)), axis=-1)
    w = np.einsum("...ij,...j->...i", yanlilik_presesyon_matrisi(jd1, jd2), v)
    return np.arctan2(w[..., 1], w[..., 0]) % IKI_PI, np.arctan2(w[..., 2], np.hypot(w[..., 0], w[..., 1]))
