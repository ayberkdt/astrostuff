"""Eşit aralıklı olmayan zaman serileri için Lomb-Scargle periyodogramı.

Tanımlar VanderPlas'ın derlemesinin arXiv TeX kaynağından (1703.09824)
okunmuştur [VanderPlas2018]; denklemler oradaki TeX etiketleriyle anılır.
Yöntemin özgün makaleleri [Lomb1976] ve [Scargle1982] bu çalışmada
okunmamıştır.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source

Array = NDArray[np.float64]


@source(
    "VanderPlas2018",
    where="eq:lomb-scargle-periodogram, eq:tau-def",
    status="read",
    note="g_n ortalaması çıkarılmış veri; PSD normalleştirmesi, birimi veri biriminin karesi",
)
def lomb_scargle_classical(times: ArrayLike, value: ArrayLike, frequency: ArrayLike) -> Array:
    """Klasik Lomb-Scargle periyodogramını hesaplar.

    P(f) = ½ { [Σ g cos(2πf(t−τ))]² / Σ cos²(2πf(t−τ)) + [Σ g sin(2πf(t−τ))]² / Σ sin²(2πf(t−τ)) },
    tan(4πfτ) = Σ sin(4πft) / Σ cos(4πft).

    Parameters
    ----------
    zaman : ndarray (N,)
    deger : ndarray (N,)
        Ölçümler; ortalaması içeride çıkarılır.
    frekans : ndarray (M,)
        Frekanslar [1 / zaman birimi], pozitif.

    Returns
    -------
    ndarray (M,)

    References
    ----------
    [VanderPlas2018] eq:lomb-scargle-periodogram, eq:tau-def.

    Validity
    --------
    Ölçüm belirsizliklerini kullanmaz ve verinin ortalamasının sinyalin
    ortalamasına eşit olduğunu varsayar; faz kapsaması eksikse yanlı olur.
    Hesap yükü O(N·M).

    Verification
    ------------
    tests/oracle/test_periodogram_oracle.py: astropy LombScargle(fit_mean=False, normalization="psd").
    """
    t = np.asarray(times, dtype=float)
    g = np.asarray(value, dtype=float)
    g = g - g.mean()
    f = np.atleast_1d(np.asarray(frequency, dtype=float))
    omega = 2.0 * np.pi * f[:, None]
    tau = np.arctan2(np.sin(2 * omega * t).sum(axis=1), np.cos(2 * omega * t).sum(axis=1)) / (2 * omega[:, 0])
    faz = omega * (t[None, :] - tau[:, None])
    c, s = np.cos(faz), np.sin(faz)
    return 0.5 * ((c @ g) ** 2 / (c * c).sum(axis=1) + (s @ g) ** 2 / (s * s).sum(axis=1))


@source(
    "VanderPlas2018",
    where="eq:floating-mean-model, eq:chi2-with-errors, eq:lomb-scargle-normalized",
    status="read",
    note="her frekansta y0 + a sin + b cos modelinin ağırlıklı en küçük kareler uyumu; P = 1 − χ̂²(f)/χ̂²₀",
)
def lomb_scargle(times: ArrayLike, value: ArrayLike, frequency: ArrayLike, sigma: ArrayLike | None = None) -> Array:
    """Yüzen ortalamalı, ağırlıklı Lomb-Scargle periyodogramını hesaplar.

    Her frekansta y(t) = y₀ + A sin(2πf(t − φ)) modeli ağırlıklı en küçük
    karelerle uydurulur; periyodogram sabit modele göre χ² iyileşmesidir:
    P(f) = 1 − χ̂²(f) / χ̂²₀, 0 <= P <= 1.

    Parameters
    ----------
    zaman, deger : ndarray (N,)
    frekans : ndarray (M,)
    sigma : ndarray (N,), isteğe bağlı
        Ölçüm standart belirsizlikleri; verilmezse eşit ağırlık.

    References
    ----------
    [VanderPlas2018] eq:floating-mean-model, eq:chi2-with-errors, eq:lomb-scargle-normalized.

    Validity
    --------
    Tek sinüs bileşenli model varsayar; sinüs dışı biçimlerde (örten çift,
    geçiş) güç harmoniklere dağılır. Hesap yükü O(N·M).

    Verification
    ------------
    tests/oracle/test_periodogram_oracle.py: astropy LombScargle(fit_mean=True, normalization="standard").
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(value, dtype=float)
    w = np.ones_like(y) if sigma is None else 1.0 / np.asarray(sigma, dtype=float) ** 2
    w = w / w.sum()
    f = np.atleast_1d(np.asarray(frequency, dtype=float))
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


@source(
    "VanderPlas2018",
    where="TeX etiketi eq:FAP-neff ve izleyen paragraf",
    status="read",
    note="FAP(z) ≈ 1 − [P_tek(z)]^N_eff; basit kestirim N_eff = f_max · T",
)
@source(
    "Astropy2022",
    where="timeseries/periodograms/lombscargle/_statistics.py, fap_single ve fap_naive",
    status="read",
    note="standart normalleştirmede tek frekans yanlış alarm olasılığı (1 − z)^{(N−3)/2}; log1p/expm1 biçimi",
)
def false_alarm_probability(power: ArrayLike, n_data: int, n_independent_frequencies: float) -> Array:
    """Standart normalleştirilmiş periyodogram tepesi için yanlış alarm olasılığını kestirir.

    Tek frekansta P(Z < z) = 1 − (1 − z)^{(N−3)/2}; M bağımsız frekansta
    FAP = 1 − [P(Z < z)]^M.

    References
    ----------
    [VanderPlas2018] eq:FAP-neff; [Astropy2022] fap_single, fap_naive.

    Validity
    --------
    Gauss beyaz gürültü varsayar. Bağımsız frekans sayısı iyi tanımlı değildir;
    sonuç mertebe kestirimidir. Güvenilir değer için bootstrap gerekir.

    Verification
    ------------
    tests/oracle/test_periodogram_oracle.py: astropy false_alarm_probability(method="naive") ile aynı biçim.
    """
    z = np.asarray(power, dtype=float)
    tek_fap = (1.0 - z) ** (0.5 * (n_data - 3))
    # 1 − (1 − p)^M; küçük p için iptal hatasını önlemek üzere log1p/expm1 ile.
    with np.errstate(divide="ignore"):
        return -np.expm1(n_independent_frequencies * np.log1p(-tek_fap))


@source(
    "VanderPlas2018",
    where="Bölüm 7.1, TeX etiketi eq:n-eval",
    status="read",
    note="Δf = 1/(n_o T); kaynak n_o = 5 kullanır ve 5-10 aralığını anar",
)
def frequency_grid(times: ArrayLike, max_frequency: float, oversampling: float = 5.0) -> Array:
    """Periyodogram için düzgün frekans ızgarası üretir: Δf = 1 / (n_o T).

    Parameters
    ----------
    zaman : ndarray (N,)
    en_yuksek_frekans : float
        Izgaranın üst sınırı. Eşit aralıklı olmayan veride Nyquist frekansı iyi
        tanımlı değildir; üst sınır aranan en kısa döneme göre seçilir.
    asiri_ornekleme : float
        Tepe genişliği (1/T) başına ızgara noktası sayısı.

    References
    ----------
    [VanderPlas2018] Bölüm 7.1, eq:n-eval.

    Verification
    ------------
    tests/oracle/test_periodogram_oracle.py
    """
    t = np.asarray(times, dtype=float)
    adim = 1.0 / (oversampling * (t.max() - t.min()))
    return np.arange(adim, max_frequency + adim, adim)
