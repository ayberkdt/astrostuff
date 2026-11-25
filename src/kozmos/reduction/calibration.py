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

from kozmos._sources import source
from kozmos.statistics.robust import sigma_clip

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReducedFrame:
    """İndirgenmiş görüntü ve piksel başına varyansı.

    Attributes
    ----------
    veri : ndarray (Y, X)
        Akı [e⁻/s/piksel].
    varyans : ndarray (Y, X)
        ``veri``nin varyansı [(e⁻/s)²].
    """

    data: Array
    variance: Array


@source(
    "CraigChambers2026",
    where="defter 01-05 'Calibration overview': 'Every image has noise'",
    status="read",
    note="kalibrasyon kareleri gürültüyü azaltmak için çok sayıda kareden birleştirilir; sigma kırpmalı "
    "birleştirmenin ayrıntısı (defter 01-06) okunmadı",
)
def combine(frames: ArrayLike, *, method: str = "ortalama", sigma: float | None = 3.0) -> tuple[Array, Array]:
    """Aynı sahnenin karelerini piksel piksel birleştirir.

    Parameters
    ----------
    kareler : ndarray (K, Y, X)
    yontem : {"ortalama", "ortanca"}
    sigma : float, isteğe bağlı
        Verilirse birleştirmeden önce kare ekseninde sigma kırpma uygulanır.

    Returns
    -------
    (birleşik, kullanılan) : birleşik kare ve piksel başına kullanılan kare sayısı.

    References
    ----------
    [CraigChambers2026] defter 01-05.

    Validity
    --------
    Kareler aynı düzeyde olmalıdır (aynı poz süresi, aynı aydınlatma); ölçekleme
    yapılmaz. Gauss gürültüsünde ortancanın varyansı ortalamanınkinin yaklaşık
    π/2 katıdır.

    Verification
    ------------
    tests/unit/test_reduction.py: kozmik ışın eklenmiş karelerde kırpmalı ortalamanın yansızlığı.
    """
    k = np.asarray(frames, dtype=float)
    if k.ndim != 3:
        raise ValueError("kareler (K, Y, X) biçiminde olmalı")
    if method not in ("ortalama", "ortanca"):
        raise ValueError("yöntem 'ortalama' ya da 'ortanca' olmalı")
    mask = sigma_clip(k, sigma=sigma, axis=0) if sigma is not None else np.zeros(k.shape, dtype=bool)
    dolu = np.where(mask, np.nan, k)
    birlesik = np.nanmean(dolu, axis=0) if method == "ortalama" else np.nanmedian(dolu, axis=0)
    return birlesik, (~mask).sum(axis=0).astype(float)


@source(
    "IRAF2026",
    where="noao/obsutil/src/doc/findgain.hlp, ALGORITHM",
    status="read",
    note="gain = ((mean(flat1)+mean(flat2)) − (mean(zero1)+mean(zero2))) / (σ²(flatdif) − σ²(zerodif)); "
    "readnoise = gain · σ(zerodif) / √2",
)
def gain_and_read_noise(flat1: ArrayLike, flat2: ArrayLike, bias1: ArrayLike, bias2: ArrayLike) -> tuple[float, float]:
    """İki düz alan ve iki bias karesinden kazancı ve okuma gürültüsünü ölçer.

    Fark görüntüleri sabit desenleri (düz alan yapısı, bias yapısı) götürür;
    geriye yalnızca rastgele gürültü kalır.

    Returns
    -------
    (kazanç [e⁻/ADU], okuma gürültüsü [e⁻ RMS])

    References
    ----------
    [IRAF2026] findgain.hlp.

    Validity
    --------
    İki düz kare aynı aydınlatma düzeyinde ve doğrusal bölgede olmalıdır.
    Karanlık akımın katkısı ihmal edilir (kısa poz).

    Verification
    ------------
    tests/unit/test_reduction.py: sentetik karelerde girilen değerlerin geri kazanımı.
    """
    f1, f2 = np.asarray(flat1, dtype=float), np.asarray(flat2, dtype=float)
    b1, b2 = np.asarray(bias1, dtype=float), np.asarray(bias2, dtype=float)
    var_bias = np.var(b1 - b2, ddof=1)
    kazanc = ((f1.mean() + f2.mean()) - (b1.mean() + b2.mean())) / (np.var(f1 - f2, ddof=1) - var_bias)
    return float(kazanc), float(kazanc * np.sqrt(var_bias / 2.0))


@source(
    "CraigChambers2026",
    where="defter 01-05 'Calibration overview': indirgeme denklemi",
    status="read",
    note="yıldızlar + gürültü = (ham − bias − karanlık akım)/düz − gökyüzü; karanlık akım = (karanlık kare − bias)/süre. "
    "Varyans σ² = (sinyal_e + R²)/(düz·t)² bu projenin modelidir (kaynakta okunmadı)",
)
def reduce_frame(
    raw_adu: ArrayLike,
    master_bias_adu: ArrayLike,
    master_dark_e_s: ArrayLike | float,
    master_flat: ArrayLike | float,
    *,
    gain_e_adu: float,
    read_noise_e: float,
    exposure_s: float,
) -> ReducedFrame:
    """Ham kareyi bias, karanlık ve düz alan düzeltmesiyle akıya çevirir.

    Sıra: bias çıkar, kazançla elektrona çevir, karanlık akımı çıkar, düz
    alana ve poz süresine böl.

    Parameters
    ----------
    ham_adu : ndarray (Y, X)
    ana_bias_adu : ndarray (Y, X) ya da float
    ana_karanlik_e_s : ndarray (Y, X) ya da float
        Karanlık akım haritası [e⁻/s/piksel].
    ana_duz : ndarray (Y, X) ya da float
        Normalleştirilmiş düz alan (ortalaması 1).
    kazanc_e_adu, okuma_gurultusu_e, poz_s : float

    Returns
    -------
    IndirgenmisKare
        Akı [e⁻/s/piksel] ve varyansı.

    References
    ----------
    [CraigChambers2026] defter 01-05.

    Validity
    --------
    Varyans, bias çıkarılmış sinyalin Poisson gürültüsü ile okuma gürültüsünün
    toplamıdır; ana kalibrasyon karelerinin kendi gürültüsü eklenmez (çok
    sayıda kareden birleştirildikleri varsayılır). Poisson terimi gözlenen
    sayımdan kestirildiği için düşük sayımda yanlıdır.

    Verification
    ------------
    tests/unit/test_reduction.py: indirgenmiş karenin bilinen sahneye göre artığı ve indirgenmiş χ².
    """
    if exposure_s <= 0:
        raise ValueError("poz süresi pozitif olmalı")
    sinyal_e = (np.asarray(raw_adu, dtype=float) - master_bias_adu) * gain_e_adu
    flat = np.asarray(master_flat, dtype=float)
    data = (sinyal_e - np.asarray(master_dark_e_s, dtype=float) * exposure_s) / (flat * exposure_s)
    variance = (np.maximum(sinyal_e, 0.0) + read_noise_e**2) / (flat * exposure_s) ** 2
    return ReducedFrame(data, variance)
