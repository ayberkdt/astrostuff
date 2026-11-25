"""Görüntü indirgeme: kare birleştirme, bias/karanlık/düz alan düzeltmesi, foton aktarımı.

İndirgeme denklemi astropy CCD indirgeme rehberinden [CraigChambers2026], kazanç
ve okuma gürültüsü formülleri IRAF ``findgain`` görevinin belgesinden
[IRAF2026] okunmuştur. Varyans yayılımı bu projenin modelidir ve sentetik
karelerle sınanır.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._kaynak import kaynak
from kozmos.istatistik.saglam import sigma_kirp

Dizi = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class IndirgenmisKare:
    """İndirgenmiş görüntü ve piksel başına varyansı.

    Öznitelikler
    ------------
    veri : ndarray (Y, X)
        Akı [e⁻/s/piksel].
    varyans : ndarray (Y, X)
        ``veri``nin varyansı [(e⁻/s)²].
    """

    veri: Dizi
    varyans: Dizi


@kaynak(
    "CraigChambers2026",
    yer="defter 01-05 'Calibration overview': 'Every image has noise'",
    teyit="okundu",
    not_="kalibrasyon kareleri gürültüyü azaltmak için çok sayıda kareden birleştirilir; sigma kırpmalı "
    "birleştirmenin ayrıntısı (defter 01-06) okunmadı",
)
def birlestir(kareler: ArrayLike, *, yontem: str = "ortalama", sigma: float | None = 3.0) -> tuple[Dizi, Dizi]:
    """Aynı sahnenin karelerini piksel piksel birleştirir.

    Parametreler
    ------------
    kareler : ndarray (K, Y, X)
    yontem : {"ortalama", "ortanca"}
    sigma : float, isteğe bağlı
        Verilirse birleştirmeden önce kare ekseninde sigma kırpma uygulanır.

    Döndürür
    --------
    (birleşik, kullanılan) : birleşik kare ve piksel başına kullanılan kare sayısı.

    Kaynaklar
    ---------
    [CraigChambers2026] defter 01-05.

    Geçerlilik
    ----------
    Kareler aynı düzeyde olmalıdır (aynı poz süresi, aynı aydınlatma); ölçekleme
    yapılmaz. Gauss gürültüsünde ortancanın varyansı ortalamanınkinin yaklaşık
    π/2 katıdır.

    Doğrulama
    ---------
    tests/birim/test_indirgeme.py: kozmik ışın eklenmiş karelerde kırpmalı ortalamanın yansızlığı.
    """
    k = np.asarray(kareler, dtype=float)
    if k.ndim != 3:
        raise ValueError("kareler (K, Y, X) biçiminde olmalı")
    if yontem not in ("ortalama", "ortanca"):
        raise ValueError("yöntem 'ortalama' ya da 'ortanca' olmalı")
    maske = sigma_kirp(k, sigma=sigma, eksen=0) if sigma is not None else np.zeros(k.shape, dtype=bool)
    dolu = np.where(maske, np.nan, k)
    birlesik = np.nanmean(dolu, axis=0) if yontem == "ortalama" else np.nanmedian(dolu, axis=0)
    return birlesik, (~maske).sum(axis=0).astype(float)


@kaynak(
    "IRAF2026",
    yer="noao/obsutil/src/doc/findgain.hlp, ALGORITHM",
    teyit="okundu",
    not_="gain = ((mean(flat1)+mean(flat2)) − (mean(zero1)+mean(zero2))) / (σ²(flatdif) − σ²(zerodif)); "
    "readnoise = gain · σ(zerodif) / √2",
)
def kazanc_ve_okuma_gurultusu(
    duz1: ArrayLike, duz2: ArrayLike, bias1: ArrayLike, bias2: ArrayLike
) -> tuple[float, float]:
    """İki düz alan ve iki bias karesinden kazancı ve okuma gürültüsünü ölçer.

    Fark görüntüleri sabit desenleri (düz alan yapısı, bias yapısı) götürür;
    geriye yalnızca rastgele gürültü kalır.

    Döndürür
    --------
    (kazanç [e⁻/ADU], okuma gürültüsü [e⁻ RMS])

    Kaynaklar
    ---------
    [IRAF2026] findgain.hlp.

    Geçerlilik
    ----------
    İki düz kare aynı aydınlatma düzeyinde ve doğrusal bölgede olmalıdır.
    Karanlık akımın katkısı ihmal edilir (kısa poz).

    Doğrulama
    ---------
    tests/birim/test_indirgeme.py: sentetik karelerde girilen değerlerin geri kazanımı.
    """
    f1, f2 = np.asarray(duz1, dtype=float), np.asarray(duz2, dtype=float)
    b1, b2 = np.asarray(bias1, dtype=float), np.asarray(bias2, dtype=float)
    var_bias = np.var(b1 - b2, ddof=1)
    kazanc = ((f1.mean() + f2.mean()) - (b1.mean() + b2.mean())) / (np.var(f1 - f2, ddof=1) - var_bias)
    return float(kazanc), float(kazanc * np.sqrt(var_bias / 2.0))


@kaynak(
    "CraigChambers2026",
    yer="defter 01-05 'Calibration overview': indirgeme denklemi",
    teyit="okundu",
    not_="yıldızlar + gürültü = (ham − bias − karanlık akım)/düz − gökyüzü; karanlık akım = (karanlık kare − bias)/süre. "
    "Varyans σ² = (sinyal_e + R²)/(düz·t)² bu projenin modelidir (kaynakta okunmadı)",
)
def kare_indirge(
    ham_adu: ArrayLike,
    ana_bias_adu: ArrayLike,
    ana_karanlik_e_s: ArrayLike | float,
    ana_duz: ArrayLike | float,
    *,
    kazanc_e_adu: float,
    okuma_gurultusu_e: float,
    poz_s: float,
) -> IndirgenmisKare:
    """Ham kareyi bias, karanlık ve düz alan düzeltmesiyle akıya çevirir.

    Sıra: bias çıkar, kazançla elektrona çevir, karanlık akımı çıkar, düz
    alana ve poz süresine böl.

    Parametreler
    ------------
    ham_adu : ndarray (Y, X)
    ana_bias_adu : ndarray (Y, X) ya da float
    ana_karanlik_e_s : ndarray (Y, X) ya da float
        Karanlık akım haritası [e⁻/s/piksel].
    ana_duz : ndarray (Y, X) ya da float
        Normalleştirilmiş düz alan (ortalaması 1).
    kazanc_e_adu, okuma_gurultusu_e, poz_s : float

    Döndürür
    --------
    IndirgenmisKare
        Akı [e⁻/s/piksel] ve varyansı.

    Kaynaklar
    ---------
    [CraigChambers2026] defter 01-05.

    Geçerlilik
    ----------
    Varyans, bias çıkarılmış sinyalin Poisson gürültüsü ile okuma gürültüsünün
    toplamıdır; ana kalibrasyon karelerinin kendi gürültüsü eklenmez (çok
    sayıda kareden birleştirildikleri varsayılır). Poisson terimi gözlenen
    sayımdan kestirildiği için düşük sayımda yanlıdır.

    Doğrulama
    ---------
    tests/birim/test_indirgeme.py: indirgenmiş karenin bilinen sahneye göre artığı ve indirgenmiş χ².
    """
    if poz_s <= 0:
        raise ValueError("poz süresi pozitif olmalı")
    sinyal_e = (np.asarray(ham_adu, dtype=float) - ana_bias_adu) * kazanc_e_adu
    duz = np.asarray(ana_duz, dtype=float)
    veri = (sinyal_e - np.asarray(ana_karanlik_e_s, dtype=float) * poz_s) / (duz * poz_s)
    varyans = (np.maximum(sinyal_e, 0.0) + okuma_gurultusu_e**2) / (duz * poz_s) ** 2
    return IndirgenmisKare(veri, varyans)
