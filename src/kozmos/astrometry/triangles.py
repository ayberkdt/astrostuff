"""Üçgen eşleme: iki nokta listesini ölçek, dönme, öteleme ve ayna bilinmeden eşleştirme.

Yöntem Groth'un makalesinin ADS taramasından okunmuştur [Groth1986]:

- Her nokta üçlüsü bir üçgendir. Köşeler, 1-2 kenarı en kısa, 2-3 kenarı orta,
  3-1 kenarı en uzun olacak biçimde sıralanır.
- Üçgenin ölçekten, dönmeden ve ötelemeden bağımsız iki sayısı vardır: en uzun
  kenarın en kısa kenara oranı R (Denk. 1) ve 1. köşedeki açının kosinüsü C
  (Denk. 5). Oranı büyük (tipik sınır 10) üçgenler atılır.
- Köşelerin dolanım yönü (saat yönü ya da tersi) ayna dönüşümünü ayırt eder:
  gerçek eşleşmelerin hepsi aynı yönlü ya da hepsi ters yönlüdür.
- Eşleşen her üçgen, köşe çiftlerine birer oy verir; noktalar oy sırasına göre
  atanır ve oy yarıya düştüğünde ya da bir nokta ikinci kez atanacakken durulur.

Bu uygulamada kaynaktan ayrılan yerler: R ve C toleransları kaynaktaki hata
yayılımıyla (Denk. 4 ve 6) değil sabit sayılarla verilir; yanlış eşleşmeleri
ayıklayan büyütme süzgeci (kaynakta yinelemeli, Denk. 10-12) çevre oranının
ortancası çevresinde tek bir sağlam kırpmaya indirgenmiştir.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from kozmos._sources import source
from kozmos.astrometry.matching import cross_match
from kozmos.astrometry.solve import WcsSolution, _tangent_plane, fit_wcs, refine_wcs
from kozmos.astrometry.wcs import pixel_to_sky

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Triangles:
    """Bir nokta listesinin üçgenleri.

    Attributes
    ----------
    kose : (T, 3) int
        Köşe indisleri, [Groth1986] sırasıyla (1-2 en kısa, 3-1 en uzun kenar).
    oran : (T,) float
        R: en uzun kenarın en kısa kenara oranı.
    kosinus : (T,) float
        C: 1. köşedeki açının kosinüsü.
    log_cevre : (T,) float
        Çevrenin doğal logaritması.
    yon : (T,) int
        Köşelerin dolanım yönü: +1 saat yönünün tersi, −1 saat yönü.
    """

    vertices: NDArray[np.int64]
    ratio: Array
    cosine: Array
    log_perimeter: Array
    orientation: NDArray[np.int64]


@source(
    "Groth1986",
    where="Bölüm II.b: köşe sıralaması; Denk. 1-3 (R) ve Denk. 5 (C); oran sınırı",
    status="read",
    note="R = r3/r2, r3 = |köşe3 − köşe1| (en uzun kenar), r2 = |köşe2 − köşe1| (en kısa kenar); "
    "C = (Δx3 Δx2 + Δy3 Δy2)/(r3 r2). Kaynak oran sınırı için tipik değer 10 verir",
)
def build_triangles(x: ArrayLike, y: ArrayLike, ratio_limit: float = 10.0) -> Triangles:
    """Bir nokta listesinin bütün üçgenlerini ve değişmezlerini kurar.

    Parameters
    ----------
    x, y : ndarray (N,)
    oran_siniri : float
        R bu değeri aşan (çok ince) üçgenler atılır.

    Returns
    -------
    Ucgenler

    References
    ----------
    [Groth1986] Bölüm II.b, Denk. 1-3, 5.

    Validity
    --------
    Üçgen sayısı N³/6 ile büyür; N birkaç onu aşmamalıdır. Çakışık noktalar
    (sıfır uzunluklu kenar) atılır.

    Verification
    ------------
    tests/unit/test_triangles.py: bilinen üçgende R ve C; değişmezlerin ölçek, dönme, öteleme ve ayna altında korunması.
    """
    p = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    uclu = np.array(list(combinations(range(len(p)), 3)), dtype=np.int64).reshape(-1, 3)
    if uclu.size == 0:
        bos = np.empty(0)
        return Triangles(uclu, bos, bos, bos, np.empty(0, dtype=np.int64))
    a, b, c = p[uclu[:, 0]], p[uclu[:, 1]], p[uclu[:, 2]]
    # Kenar k, k. köşenin karşısındadır: kenar0 = b-c, kenar1 = c-a, kenar2 = a-b.
    kenar = np.column_stack([np.hypot(*(b - c).T), np.hypot(*(c - a).T), np.hypot(*(a - b).T)])
    sira = np.argsort(kenar, axis=1)  # en kısa, orta, en uzun kenarın karşı köşeleri
    satir = np.arange(len(uclu))
    # Köşe 1 en kısa ve en uzun kenarın ortak ucudur: orta kenarın karşısındaki köşe. Köşe 2 en uzun kenarın,
    # köşe 3 en kısa kenarın karşısındaki köşedir.
    k1, k2, k3 = uclu[satir, sira[:, 1]], uclu[satir, sira[:, 2]], uclu[satir, sira[:, 0]]
    r2, r3 = kenar[satir, sira[:, 0]], kenar[satir, sira[:, 2]]
    gecerli = r2 > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = r3 / r2
    gecerli &= ratio <= ratio_limit
    k1, k2, k3, r2, r3, ratio = (d[gecerli] for d in (k1, k2, k3, r2, r3, ratio))
    d2, d3 = p[k2] - p[k1], p[k3] - p[k1]
    cosine = np.sum(d2 * d3, axis=1) / (r2 * r3)
    orientation = np.where(d2[:, 0] * d3[:, 1] - d2[:, 1] * d3[:, 0] >= 0, 1, -1).astype(np.int64)
    cevre = kenar[gecerli].sum(axis=1)
    return Triangles(np.column_stack([k1, k2, k3]), ratio, cosine, np.log(cevre), orientation)


@source(
    "Groth1986",
    where="Bölüm II.c (üçgen eşleme, yön), II.d (yanlış eşleşmelerin ayıklanması), II.e (oylama ve atama)",
    status="read",
    note="eşleşen üçgen üç köşe çiftine birer oy verir; çiftler oy sırasıyla atanır; oy yarıya düşünce, bir nokta "
    "ikinci kez atanacakken ya da oy sıfıra inince durulur; hiçbir çift birden çok oy almadıysa eşleşme yoktur. "
    "Çoğunluktaki yön tutulur. Sabit toleranslar ve tek adımlı büyütme kırpması bu projenin sadeleştirmesidir",
)
def match_triangles(
    x1: ArrayLike,
    y1: ArrayLike,
    x2: ArrayLike,
    y2: ArrayLike,
    *,
    ratio_tolerance: float = 0.01,
    cosine_tolerance: float = 0.01,
    ratio_limit: float = 10.0,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], int]:
    """İki nokta listesi arasındaki karşılıkları üçgen oylamasıyla bulur.

    Parameters
    ----------
    x1, y1, x2, y2 : ndarray
        İki listenin koordinatları; birimleri, ölçekleri ve yönelimleri farklı olabilir.
    oran_toleransi : float
        log R farkının üst sınırı.
    kosinus_toleransi : float
        C farkının üst sınırı.

    Returns
    -------
    (indis1, indis2, oy, yön) : eşleşen noktaların iki listedeki indisleri, aldıkları oylar ve
    eşleşmelerin yönü (+1 aynı yön, −1 ayna).

    References
    ----------
    [Groth1986] Bölüm II.c-e.

    Validity
    --------
    İki listenin ortak noktaları ikisinde de çoğunlukta olmalıdır; listelerden biri ötekinin
    küçük bir alt bölgesiyse ortak üçgen azdır ve eşleşme bulunmaz. Optik bozulma toleransı aşarsa
    oylar dağılır. Eşleşme bulunamazsa boş diziler döner.

    Verification
    ------------
    tests/unit/test_triangles.py: döndürülmüş, ölçeklenmiş, aynalanmış ve gürültülü listede karşılıkların bulunması.
    """
    u1 = build_triangles(x1, y1, ratio_limit)
    u2 = build_triangles(x2, y2, ratio_limit)
    bos = np.empty(0, dtype=np.int64)
    if u1.ratio.size == 0 or u2.ratio.size == 0:
        return bos, bos, bos, 0
    olcek = np.array([1.0 / ratio_tolerance, 1.0 / cosine_tolerance])
    uzay1 = np.column_stack([np.log(u1.ratio), u1.cosine]) * olcek
    uzay2 = np.column_stack([np.log(u2.ratio), u2.cosine]) * olcek
    komsu = cKDTree(uzay2).query_ball_point(uzay1, r=1.0)
    i1 = np.repeat(np.arange(len(komsu)), [len(k) for k in komsu])
    i2 = np.fromiter((j for k in komsu for j in k), dtype=np.int64, count=i1.size)
    if i1.size == 0:
        return bos, bos, bos, 0
    # Yön: gerçek eşleşmelerin hepsi aynı yönlü ya da hepsi ters yönlüdür; çoğunluk tutulur.
    ayni = u1.orientation[i1] == u2.orientation[i2]
    orientation = 1 if ayni.sum() >= (~ayni).sum() else -1
    sec = ayni if orientation == 1 else ~ayni
    i1, i2 = i1[sec], i2[sec]
    # Büyütme: gerçek eşleşmelerde çevre oranı aynıdır (iki listenin ölçek oranı).
    buyutme = u1.log_perimeter[i1] - u2.log_perimeter[i2]
    sayim, kenarlar = np.histogram(buyutme, bins=max(10, int(np.ptp(buyutme) / 0.02) + 1))
    peak = 0.5 * (kenarlar[np.argmax(sayim)] + kenarlar[np.argmax(sayim) + 1])
    sec = np.abs(buyutme - peak) < 0.03
    i1, i2 = i1[sec], i2[sec]
    n1, n2 = len(np.asarray(x1)), len(np.asarray(x2))
    oy = np.zeros((n1, n2), dtype=np.int64)
    for k in range(3):
        np.add.at(oy, (u1.vertices[i1, k], u2.vertices[i2, k]), 1)
    sira = np.argsort(-oy, axis=None)
    a1, a2, oylar = [], [], []
    onceki = int(oy.flat[sira[0]])
    if onceki <= 1:
        return bos, bos, bos, orientation
    for s in sira:
        p1, p2 = divmod(int(s), n2)
        v = int(oy[p1, p2])
        if v == 0 or v < 0.5 * onceki or p1 in a1 or p2 in a2:
            break
        a1.append(p1)
        a2.append(p2)
        oylar.append(v)
        onceki = v
    return np.array(a1, dtype=np.int64), np.array(a2, dtype=np.int64), np.array(oylar, dtype=np.int64), orientation


@source(
    "Groth1986",
    where="Bölüm II (algoritma) ve Bölüm III (katalog ile plak listesinin eşlenmesi örneği)",
    status="read",
    note="en parlak noktalarla üçgen eşleme, ardından eşleşen çiftlerden doğrusal çözüm. Eşleşmeden WCS'e geçiş "
    "(wcs_uydur, wcs_iyilestir) bu projenin birleştirmesidir",
)
def blind_solve(
    x: ArrayLike,
    y: ArrayLike,
    brightness: ArrayLike,
    ra_deg: ArrayLike,
    dec_deg: ArrayLike,
    catalog_magnitude: ArrayLike,
    *,
    n_brightest: int = 25,
    final_radius_arcsec: float = 1.5,
    tolerance: float = 0.01,
    min_validation_fraction: float = 0.4,
    min_matches: int = 8,
) -> tuple[WcsSolution, int]:
    """Başlangıç WCS'i olmadan, saptanan kaynakları katalogla eşleyip TAN izdüşümlü WCS çözer.

    Parameters
    ----------
    x, y : ndarray
        Saptanan kaynakların sıfır tabanlı piksel konumları.
    parlaklik : ndarray
        Saptanan kaynakların akısı (büyük = parlak); en parlakları seçmek için.
    sag_acikllik_derece, dik_acikllik_derece, katalog_kadiri : ndarray
        Görüntünün kapladığı bölgenin katalogu ve kadirleri (küçük = parlak).
    en_parlak : int
        Üçgen eşlemeye giren nokta sayısı (her iki listeden).
    tolerans : float
        Üçgen değişmezlerinin (log R ve C) eşleşme toleransı.
    en_az_dogrulama_kesri : float
        Çözüm, üçgen eşlemeye giren parlak kaynakların en az bu kesrini katalogla eşleştirmelidir;
        yoksa çözüm reddedilir.
    en_az_eslesme : int
        Doğrulamada eşleşmesi gereken parlak kaynak sayısının mutlak alt sınırı. Birkaç yıldızlık
        "çözüm" rastlantıdan ayırt edilemez (gerçek, seyrek bir SOFI alanında 5 yıldızla 471 yay
        saniyesi yanlış bir çözüm kesir ölçütünden geçmiştir).

    Returns
    -------
    (WcsCozumu, üçgen eşlemesinde atanan çift sayısı)

    References
    ----------
    [Groth1986] Bölüm II, III.

    Validity
    --------
    Ölçek, yönelim, ayna ve teğet noktası bilinmeden çalışır; ama katalog görüntünün
    kapladığı bölgeyle yaklaşık aynı alanı örtmelidir (yaklaşık bir gök konumu ve alan
    genişliği gerekir). Bütün gökte arama yapan dizinli bir çözücü değildir. Üçten az
    çift atanırsa ya da bulunan çözüm doğrulamadan geçmezse ``ValueError`` verir. Doğrulama
    gereklidir: birkaç yanlış çiftten kurulan çözüm, yoğun bir katalogda rastlantıyla onlarca
    yıldız "eşleştirebilir" (gerçek SOFI yığınında görüldü).

    Verification
    ------------
    tests/unit/test_triangles.py: bilinen WCS'ten üretilen alanda başlangıç bilgisi olmadan çözüm.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ra, dec = np.asarray(ra_deg, dtype=float), np.asarray(dec_deg, dtype=float)
    g = np.argsort(-np.asarray(brightness, dtype=float))[:n_brightest]
    magnitude = np.asarray(catalog_magnitude, dtype=float)
    k = np.argsort(np.where(np.isfinite(magnitude), magnitude, np.inf))[:n_brightest]
    v = np.array(
        [
            np.cos(np.radians(dec)) * np.cos(np.radians(ra)),
            np.cos(np.radians(dec)) * np.sin(np.radians(ra)),
            np.sin(np.radians(dec)),
        ]
    ).mean(axis=1)
    crval = np.array([np.degrees(np.arctan2(v[1], v[0])) % 360.0, np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))])
    xi, eta = _tangent_plane(ra[k], dec[k], crval)
    i1, i2, _, _ = match_triangles(x[g], y[g], xi, eta, ratio_tolerance=tolerance, cosine_tolerance=tolerance)
    if i1.size < 3:
        raise ValueError(f"üçgen eşleme yalnızca {i1.size} çift atadı; listeler aynı alanı örtmüyor olabilir")
    ilk = fit_wcs(x[g][i1], y[g][i1], ra[k][i2], dec[k][i2], crval)
    radius = max(3.0 * ilk.residual_rms_arcsec, 2.0 * final_radius_arcsec)
    try:
        cozum = refine_wcs(ilk.wcs, x, y, ra, dec, radius_arcsec=radius, final_radius_arcsec=final_radius_arcsec)
    except ValueError as hata:
        raise ValueError(f"üçgen eşlemesinden kurulan çözüm iyileştirilemedi: {hata}") from hata
    # Doğrulama: parlak kaynakların yeterli bir kesri katalogda karşılık bulmalıdır.
    ra_p, dec_p = pixel_to_sky(cozum.wcs, x[g], y[g])
    # Ayrı katalog yıldızları sayılır: ölçeği çökmüş yanlış bir çözüm bütün kaynakları tek yıldıza düşürebilir.
    e = cross_match(ra_p, dec_p, ra, dec, final_radius_arcsec)
    eslesen = int(np.unique(e.index[e.matched]).size)
    gereken = max(float(min_matches), min_validation_fraction * min(g.size, k.size))
    if eslesen < gereken:
        raise ValueError(
            f"çözüm doğrulanamadı: {g.size} parlak kaynak yalnızca {eslesen} ayrı katalog yıldızıyla eşleşti "
            f"(en az {gereken:.0f} gerekir)"
        )
    return cozum, int(i1.size)
