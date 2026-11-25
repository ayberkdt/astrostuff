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

from kozmos._kaynak import kaynak
from kozmos.astrometri.cozum import WcsCozumu, _teget_duzlem, wcs_iyilestir, wcs_uydur
from kozmos.astrometri.eslestirme import capraz_esle
from kozmos.astrometri.wcs import piksel_den_gok

Dizi = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Ucgenler:
    """Bir nokta listesinin üçgenleri.

    Öznitelikler
    ------------
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

    kose: NDArray[np.int64]
    oran: Dizi
    kosinus: Dizi
    log_cevre: Dizi
    yon: NDArray[np.int64]


@kaynak(
    "Groth1986",
    yer="Bölüm II.b: köşe sıralaması; Denk. 1-3 (R) ve Denk. 5 (C); oran sınırı",
    teyit="okundu",
    not_="R = r3/r2, r3 = |köşe3 − köşe1| (en uzun kenar), r2 = |köşe2 − köşe1| (en kısa kenar); "
    "C = (Δx3 Δx2 + Δy3 Δy2)/(r3 r2). Kaynak oran sınırı için tipik değer 10 verir",
)
def ucgenleri_kur(x: ArrayLike, y: ArrayLike, oran_siniri: float = 10.0) -> Ucgenler:
    """Bir nokta listesinin bütün üçgenlerini ve değişmezlerini kurar.

    Parametreler
    ------------
    x, y : ndarray (N,)
    oran_siniri : float
        R bu değeri aşan (çok ince) üçgenler atılır.

    Döndürür
    --------
    Ucgenler

    Kaynaklar
    ---------
    [Groth1986] Bölüm II.b, Denk. 1-3, 5.

    Geçerlilik
    ----------
    Üçgen sayısı N³/6 ile büyür; N birkaç onu aşmamalıdır. Çakışık noktalar
    (sıfır uzunluklu kenar) atılır.

    Doğrulama
    ---------
    tests/birim/test_ucgen.py: bilinen üçgende R ve C; değişmezlerin ölçek, dönme, öteleme ve ayna altında korunması.
    """
    p = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    uclu = np.array(list(combinations(range(len(p)), 3)), dtype=np.int64).reshape(-1, 3)
    if uclu.size == 0:
        bos = np.empty(0)
        return Ucgenler(uclu, bos, bos, bos, np.empty(0, dtype=np.int64))
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
        oran = r3 / r2
    gecerli &= oran <= oran_siniri
    k1, k2, k3, r2, r3, oran = (d[gecerli] for d in (k1, k2, k3, r2, r3, oran))
    d2, d3 = p[k2] - p[k1], p[k3] - p[k1]
    kosinus = np.sum(d2 * d3, axis=1) / (r2 * r3)
    yon = np.where(d2[:, 0] * d3[:, 1] - d2[:, 1] * d3[:, 0] >= 0, 1, -1).astype(np.int64)
    cevre = kenar[gecerli].sum(axis=1)
    return Ucgenler(np.column_stack([k1, k2, k3]), oran, kosinus, np.log(cevre), yon)


@kaynak(
    "Groth1986",
    yer="Bölüm II.c (üçgen eşleme, yön), II.d (yanlış eşleşmelerin ayıklanması), II.e (oylama ve atama)",
    teyit="okundu",
    not_="eşleşen üçgen üç köşe çiftine birer oy verir; çiftler oy sırasıyla atanır; oy yarıya düşünce, bir nokta "
    "ikinci kez atanacakken ya da oy sıfıra inince durulur; hiçbir çift birden çok oy almadıysa eşleşme yoktur. "
    "Çoğunluktaki yön tutulur. Sabit toleranslar ve tek adımlı büyütme kırpması bu projenin sadeleştirmesidir",
)
def ucgen_esle(
    x1: ArrayLike,
    y1: ArrayLike,
    x2: ArrayLike,
    y2: ArrayLike,
    *,
    oran_toleransi: float = 0.01,
    kosinus_toleransi: float = 0.01,
    oran_siniri: float = 10.0,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], int]:
    """İki nokta listesi arasındaki karşılıkları üçgen oylamasıyla bulur.

    Parametreler
    ------------
    x1, y1, x2, y2 : ndarray
        İki listenin koordinatları; birimleri, ölçekleri ve yönelimleri farklı olabilir.
    oran_toleransi : float
        log R farkının üst sınırı.
    kosinus_toleransi : float
        C farkının üst sınırı.

    Döndürür
    --------
    (indis1, indis2, oy, yön) : eşleşen noktaların iki listedeki indisleri, aldıkları oylar ve
    eşleşmelerin yönü (+1 aynı yön, −1 ayna).

    Kaynaklar
    ---------
    [Groth1986] Bölüm II.c-e.

    Geçerlilik
    ----------
    İki listenin ortak noktaları ikisinde de çoğunlukta olmalıdır; listelerden biri ötekinin
    küçük bir alt bölgesiyse ortak üçgen azdır ve eşleşme bulunmaz. Optik bozulma toleransı aşarsa
    oylar dağılır. Eşleşme bulunamazsa boş diziler döner.

    Doğrulama
    ---------
    tests/birim/test_ucgen.py: döndürülmüş, ölçeklenmiş, aynalanmış ve gürültülü listede karşılıkların bulunması.
    """
    u1 = ucgenleri_kur(x1, y1, oran_siniri)
    u2 = ucgenleri_kur(x2, y2, oran_siniri)
    bos = np.empty(0, dtype=np.int64)
    if u1.oran.size == 0 or u2.oran.size == 0:
        return bos, bos, bos, 0
    olcek = np.array([1.0 / oran_toleransi, 1.0 / kosinus_toleransi])
    uzay1 = np.column_stack([np.log(u1.oran), u1.kosinus]) * olcek
    uzay2 = np.column_stack([np.log(u2.oran), u2.kosinus]) * olcek
    komsu = cKDTree(uzay2).query_ball_point(uzay1, r=1.0)
    i1 = np.repeat(np.arange(len(komsu)), [len(k) for k in komsu])
    i2 = np.fromiter((j for k in komsu for j in k), dtype=np.int64, count=i1.size)
    if i1.size == 0:
        return bos, bos, bos, 0
    # Yön: gerçek eşleşmelerin hepsi aynı yönlü ya da hepsi ters yönlüdür; çoğunluk tutulur.
    ayni = u1.yon[i1] == u2.yon[i2]
    yon = 1 if ayni.sum() >= (~ayni).sum() else -1
    sec = ayni if yon == 1 else ~ayni
    i1, i2 = i1[sec], i2[sec]
    # Büyütme: gerçek eşleşmelerde çevre oranı aynıdır (iki listenin ölçek oranı).
    buyutme = u1.log_cevre[i1] - u2.log_cevre[i2]
    sayim, kenarlar = np.histogram(buyutme, bins=max(10, int(np.ptp(buyutme) / 0.02) + 1))
    tepe = 0.5 * (kenarlar[np.argmax(sayim)] + kenarlar[np.argmax(sayim) + 1])
    sec = np.abs(buyutme - tepe) < 0.03
    i1, i2 = i1[sec], i2[sec]
    n1, n2 = len(np.asarray(x1)), len(np.asarray(x2))
    oy = np.zeros((n1, n2), dtype=np.int64)
    for k in range(3):
        np.add.at(oy, (u1.kose[i1, k], u2.kose[i2, k]), 1)
    sira = np.argsort(-oy, axis=None)
    a1, a2, oylar = [], [], []
    onceki = int(oy.flat[sira[0]])
    if onceki <= 1:
        return bos, bos, bos, yon
    for s in sira:
        p1, p2 = divmod(int(s), n2)
        v = int(oy[p1, p2])
        if v == 0 or v < 0.5 * onceki or p1 in a1 or p2 in a2:
            break
        a1.append(p1)
        a2.append(p2)
        oylar.append(v)
        onceki = v
    return np.array(a1, dtype=np.int64), np.array(a2, dtype=np.int64), np.array(oylar, dtype=np.int64), yon


@kaynak(
    "Groth1986",
    yer="Bölüm II (algoritma) ve Bölüm III (katalog ile plak listesinin eşlenmesi örneği)",
    teyit="okundu",
    not_="en parlak noktalarla üçgen eşleme, ardından eşleşen çiftlerden doğrusal çözüm. Eşleşmeden WCS'e geçiş "
    "(wcs_uydur, wcs_iyilestir) bu projenin birleştirmesidir",
)
def kor_cozum(
    x: ArrayLike,
    y: ArrayLike,
    parlaklik: ArrayLike,
    sag_acikllik_derece: ArrayLike,
    dik_acikllik_derece: ArrayLike,
    katalog_kadiri: ArrayLike,
    *,
    en_parlak: int = 25,
    son_yaricap_yaysaniye: float = 1.5,
    tolerans: float = 0.01,
    en_az_dogrulama_kesri: float = 0.4,
    en_az_eslesme: int = 8,
) -> tuple[WcsCozumu, int]:
    """Başlangıç WCS'i olmadan, saptanan kaynakları katalogla eşleyip TAN izdüşümlü WCS çözer.

    Parametreler
    ------------
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

    Döndürür
    --------
    (WcsCozumu, üçgen eşlemesinde atanan çift sayısı)

    Kaynaklar
    ---------
    [Groth1986] Bölüm II, III.

    Geçerlilik
    ----------
    Ölçek, yönelim, ayna ve teğet noktası bilinmeden çalışır; ama katalog görüntünün
    kapladığı bölgeyle yaklaşık aynı alanı örtmelidir (yaklaşık bir gök konumu ve alan
    genişliği gerekir). Bütün gökte arama yapan dizinli bir çözücü değildir. Üçten az
    çift atanırsa ya da bulunan çözüm doğrulamadan geçmezse ``ValueError`` verir. Doğrulama
    gereklidir: birkaç yanlış çiftten kurulan çözüm, yoğun bir katalogda rastlantıyla onlarca
    yıldız "eşleştirebilir" (gerçek SOFI yığınında görüldü).

    Doğrulama
    ---------
    tests/birim/test_ucgen.py: bilinen WCS'ten üretilen alanda başlangıç bilgisi olmadan çözüm.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ra, dec = np.asarray(sag_acikllik_derece, dtype=float), np.asarray(dik_acikllik_derece, dtype=float)
    g = np.argsort(-np.asarray(parlaklik, dtype=float))[:en_parlak]
    kadir = np.asarray(katalog_kadiri, dtype=float)
    k = np.argsort(np.where(np.isfinite(kadir), kadir, np.inf))[:en_parlak]
    v = np.array(
        [
            np.cos(np.radians(dec)) * np.cos(np.radians(ra)),
            np.cos(np.radians(dec)) * np.sin(np.radians(ra)),
            np.sin(np.radians(dec)),
        ]
    ).mean(axis=1)
    crval = np.array([np.degrees(np.arctan2(v[1], v[0])) % 360.0, np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))])
    xi, eta = _teget_duzlem(ra[k], dec[k], crval)
    i1, i2, _, _ = ucgen_esle(x[g], y[g], xi, eta, oran_toleransi=tolerans, kosinus_toleransi=tolerans)
    if i1.size < 3:
        raise ValueError(f"üçgen eşleme yalnızca {i1.size} çift atadı; listeler aynı alanı örtmüyor olabilir")
    ilk = wcs_uydur(x[g][i1], y[g][i1], ra[k][i2], dec[k][i2], crval)
    yaricap = max(3.0 * ilk.artik_rms_yaysaniye, 2.0 * son_yaricap_yaysaniye)
    try:
        cozum = wcs_iyilestir(
            ilk.wcs, x, y, ra, dec, yaricap_yaysaniye=yaricap, son_yaricap_yaysaniye=son_yaricap_yaysaniye
        )
    except ValueError as hata:
        raise ValueError(f"üçgen eşlemesinden kurulan çözüm iyileştirilemedi: {hata}") from hata
    # Doğrulama: parlak kaynakların yeterli bir kesri katalogda karşılık bulmalıdır.
    ra_p, dec_p = piksel_den_gok(cozum.wcs, x[g], y[g])
    # Ayrı katalog yıldızları sayılır: ölçeği çökmüş yanlış bir çözüm bütün kaynakları tek yıldıza düşürebilir.
    e = capraz_esle(ra_p, dec_p, ra, dec, son_yaricap_yaysaniye)
    eslesen = int(np.unique(e.indis[e.eslesti]).size)
    gereken = max(float(en_az_eslesme), en_az_dogrulama_kesri * min(g.size, k.size))
    if eslesen < gereken:
        raise ValueError(
            f"çözüm doğrulanamadı: {g.size} parlak kaynak yalnızca {eslesen} ayrı katalog yıldızıyla eşleşti "
            f"(en az {gereken:.0f} gerekir)"
        )
    return cozum, int(i1.size)
