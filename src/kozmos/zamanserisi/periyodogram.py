"""Eşit aralıklı olmayan zaman serileri için Lomb-Scargle periyodogramı.

Tanımlar VanderPlas'ın derlemesinin arXiv TeX kaynağından (1703.09824)
okunmuştur [VanderPlas2018]; denklemler oradaki TeX etiketleriyle anılır.
Yöntemin özgün makaleleri [Lomb1976] ve [Scargle1982] bu çalışmada
okunmamıştır.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._kaynak import kaynak

Dizi = NDArray[np.float64]


@kaynak(
    "VanderPlas2018",
    yer="eq:lomb-scargle-periodogram, eq:tau-def",
    teyit="okundu",
    not_="g_n ortalaması çıkarılmış veri; PSD normalleştirmesi, birimi veri biriminin karesi",
)
def lomb_scargle_klasik(zaman: ArrayLike, deger: ArrayLike, frekans: ArrayLike) -> Dizi:
    """Klasik Lomb-Scargle periyodogramını hesaplar.

    P(f) = ½ { [Σ g cos(2πf(t−τ))]² / Σ cos²(2πf(t−τ)) + [Σ g sin(2πf(t−τ))]² / Σ sin²(2πf(t−τ)) },
    tan(4πfτ) = Σ sin(4πft) / Σ cos(4πft).

    Parametreler
    ------------
    zaman : ndarray (N,)
    deger : ndarray (N,)
        Ölçümler; ortalaması içeride çıkarılır.
    frekans : ndarray (M,)
        Frekanslar [1 / zaman birimi], pozitif.

    Döndürür
    --------
    ndarray (M,)

    Kaynaklar
    ---------
    [VanderPlas2018] eq:lomb-scargle-periodogram, eq:tau-def.

    Geçerlilik
    ----------
    Ölçüm belirsizliklerini kullanmaz ve verinin ortalamasının sinyalin
    ortalamasına eşit olduğunu varsayar; faz kapsaması eksikse yanlı olur.
    Hesap yükü O(N·M).

    Doğrulama
    ---------
    tests/kahin/test_periyodogram_kahin.py: astropy LombScargle(fit_mean=False, normalization="psd").
    """
    t = np.asarray(zaman, dtype=float)
    g = np.asarray(deger, dtype=float)
    g = g - g.mean()
    f = np.atleast_1d(np.asarray(frekans, dtype=float))
    omega = 2.0 * np.pi * f[:, None]
    tau = np.arctan2(np.sin(2 * omega * t).sum(axis=1), np.cos(2 * omega * t).sum(axis=1)) / (2 * omega[:, 0])
    faz = omega * (t[None, :] - tau[:, None])
    c, s = np.cos(faz), np.sin(faz)
    return 0.5 * ((c @ g) ** 2 / (c * c).sum(axis=1) + (s @ g) ** 2 / (s * s).sum(axis=1))


@kaynak(
    "VanderPlas2018",
    yer="eq:floating-mean-model, eq:chi2-with-errors, eq:lomb-scargle-normalized",
    teyit="okundu",
    not_="her frekansta y0 + a sin + b cos modelinin ağırlıklı en küçük kareler uyumu; P = 1 − χ̂²(f)/χ̂²₀",
)
def lomb_scargle(zaman: ArrayLike, deger: ArrayLike, frekans: ArrayLike, sigma: ArrayLike | None = None) -> Dizi:
    """Yüzen ortalamalı, ağırlıklı Lomb-Scargle periyodogramını hesaplar.

    Her frekansta y(t) = y₀ + A sin(2πf(t − φ)) modeli ağırlıklı en küçük
    karelerle uydurulur; periyodogram sabit modele göre χ² iyileşmesidir:
    P(f) = 1 − χ̂²(f) / χ̂²₀, 0 <= P <= 1.

    Parametreler
    ------------
    zaman, deger : ndarray (N,)
    frekans : ndarray (M,)
    sigma : ndarray (N,), isteğe bağlı
        Ölçüm standart belirsizlikleri; verilmezse eşit ağırlık.

    Kaynaklar
    ---------
    [VanderPlas2018] eq:floating-mean-model, eq:chi2-with-errors, eq:lomb-scargle-normalized.

    Geçerlilik
    ----------
    Tek sinüs bileşenli model varsayar; sinüs dışı biçimlerde (örten çift,
    geçiş) güç harmoniklere dağılır. Hesap yükü O(N·M).

    Doğrulama
    ---------
    tests/kahin/test_periyodogram_kahin.py: astropy LombScargle(fit_mean=True, normalization="standard").
    """
    t = np.asarray(zaman, dtype=float)
    y = np.asarray(deger, dtype=float)
    w = np.ones_like(y) if sigma is None else 1.0 / np.asarray(sigma, dtype=float) ** 2
    w = w / w.sum()
    f = np.atleast_1d(np.asarray(frekans, dtype=float))
    ortalama = w @ y
    yc = y - ortalama
    ki2_sifir = w @ (yc * yc)
    faz = 2.0 * np.pi * f[:, None] * (t - t.mean())[None, :]
    taban = np.stack([np.ones_like(faz), np.sin(faz), np.cos(faz)], axis=-1)  # (M, N, 3)
    wb = taban * w[None, :, None]
    normal = np.einsum("mni,mnj->mij", wb, taban)
    sag = np.einsum("mni,n->mi", wb, yc)
    cozum = np.linalg.solve(normal, sag[..., None])[..., 0]
    aciklanan = np.einsum("mi,mi->m", sag, cozum)  # χ̂²₀ − χ̂²(f)
    return aciklanan / ki2_sifir


@kaynak(
    "VanderPlas2018",
    yer="eq:LS-prob-z ve eq:FAP-neff çevresi",
    teyit="bekliyor",
    not_="standart normalleştirmede tek frekans dağılımı ve bağımsız frekans sayısı yaklaşımı; "
    "normalleştirmeye göre dağılım biçimi kaynaktan teyit edilmedi",
)
def yanlis_alarm_olasiligi(guc: ArrayLike, veri_sayisi: int, bagimsiz_frekans_sayisi: float) -> Dizi:
    """Standart normalleştirilmiş periyodogram tepesi için yanlış alarm olasılığını kestirir.

    Tek frekansta P(Z < z) = 1 − (1 − z)^{(N−3)/2}; M bağımsız frekansta
    FAP = 1 − [P(Z < z)]^M.

    Kaynaklar
    ---------
    [VanderPlas2018] eq:LS-prob-z, eq:FAP-neff.

    Geçerlilik
    ----------
    Gauss beyaz gürültü varsayar. Bağımsız frekans sayısı iyi tanımlı değildir;
    sonuç mertebe kestirimidir. Güvenilir değer için bootstrap gerekir.

    Doğrulama
    ---------
    tests/kahin/test_periyodogram_kahin.py: astropy false_alarm_probability(method="naive") ile aynı biçim.
    """
    z = np.asarray(guc, dtype=float)
    tek_fap = (1.0 - z) ** (0.5 * (veri_sayisi - 3))
    # 1 − (1 − p)^M; küçük p için iptal hatasını önlemek üzere log1p/expm1 ile.
    with np.errstate(divide="ignore"):
        return -np.expm1(bagimsiz_frekans_sayisi * np.log1p(-tek_fap))


@kaynak(
    "VanderPlas2018",
    yer="Bölüm 7.1 (frekans ızgarası), eq:n-eval",
    teyit="bekliyor",
    not_="ızgara adımı 1/(n_o T); n_o = 5 önerisi kaynaktan teyit edilmedi",
)
def frekans_izgarasi(zaman: ArrayLike, en_yuksek_frekans: float, asiri_ornekleme: float = 5.0) -> Dizi:
    """Periyodogram için düzgün frekans ızgarası üretir: Δf = 1 / (n_o T).

    Parametreler
    ------------
    zaman : ndarray (N,)
    en_yuksek_frekans : float
        Izgaranın üst sınırı. Eşit aralıklı olmayan veride Nyquist frekansı iyi
        tanımlı değildir; üst sınır aranan en kısa döneme göre seçilir.
    asiri_ornekleme : float
        Tepe genişliği (1/T) başına ızgara noktası sayısı.

    Kaynaklar
    ---------
    [VanderPlas2018] Bölüm 7.1.

    Doğrulama
    ---------
    tests/kahin/test_periyodogram_kahin.py
    """
    t = np.asarray(zaman, dtype=float)
    adim = 1.0 / (asiri_ornekleme * (t.max() - t.min()))
    return np.arange(adim, en_yuksek_frekans + adim, adim)
