"""Tek karede kozmik ışın saptama: Laplace kenar saptama (L.A.Cosmic).

Algoritma van Dokkum'un makalesinin arXiv TeX kaynağından (astro-ph/0108003,
``cosmic.tex``) okunmuştur [vanDokkum2001]. Adımlar ve denklemler:

1. Görüntü iki kat alt örneklenir, Laplace çekirdeğiyle (TeX etiketi
   ``kernel.eq``) evriştirilir, eksi değerler sıfırlanır ve özgün çözünürlüğe
   geri ortalanır: L⁺.
2. Gürültü modeli (``noise.eq``): N = g⁻¹ sqrt(g (M₅ ∘ I) + σ_rn²).
3. Anlamlılık (``sigmap.eq``): S = L⁺ / (f_s N); büyük ölçekli yapı çıkarılır:
   S′ = S − (S ∘ M₅).
4. İnce yapı görüntüsü: F = (M₃ ∘ I) − ([M₃ ∘ I] ∘ M₇).
5. Kozmik ışın: S′ > σ_lim ve L⁺ / F > f_lim.
6. İşaretlenen pikseller çevredeki iyi piksellerin ortancasıyla değiştirilir ve
   işlem yinelenir.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from kozmos._sources import source

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]

# [vanDokkum2001] kernel.eq: (1/4) [[0, −1, 0], [−1, 4, −1], [0, −1, 0]]
_LAPLACE = 0.25 * np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
_SUBSAMPLING = 2  # f_s


@source(
    "vanDokkum2001",
    where="TeX etiketleri kernel.eq, noise.eq, sigmap.eq ve izleyen denklemler (S′, F, seçim ölçütü)",
    status="read",
    note="alt örnekleme çarpanı f_s = 2. f_lim: kaynak iyi örneklenmiş veride 2 (varsayılanı), eksik örneklenmiş "
    "veride daha büyük değer önerir (WFPC2 geniş alan çipleri için yaklaşık 5). Kaynak, işaretlenen piksellerin "
    "komşularına daha düşük eşik uygulanabildiğini söyler ama oranı vermez; 0,5 oranı ve komşularda da ince "
    "yapı ölçütünün aranması bu projenin seçimidir (aranmazsa kozmik ışına bitişik yıldız çekirdekleri işaretlenir). σ_lim = 5 varsayılanı da bu projenin seçimidir",
)
def detect_cosmic_rays(
    image_adu: ArrayLike,
    *,
    gain_e_adu: float,
    read_noise_e: float,
    sigma_lim: float = 5.0,
    f_lim: float = 2.0,
    max_iterations: int = 4,
    neighbor_threshold_ratio: float = 0.5,
) -> tuple[Array, Mask]:
    """Tek karede kozmik ışınları saptar ve temizler.

    Parameters
    ----------
    goruntu_adu : ndarray (Y, X)
        Görüntü [ADU]; gökyüzü çıkarılmamış olmalıdır (gürültü modeli toplam sayıma dayanır).
    kazanc_e_adu : float
        Kazanç g [e⁻/ADU].
    okuma_gurultusu_e : float
        Okuma gürültüsü σ_rn [e⁻].
    sigma_lim : float
        Laplace anlamlılık eşiği σ_lim.
    f_lim : float
        Laplace görüntüsünün ince yapı görüntüsüne en küçük oranı f_lim; yıldızları kozmik ışınlardan ayırır.
    en_cok_yineleme : int
        Yeni piksel işaretlenmeyince erken durur.
    komsu_esik_orani : float
        İşaretlenen piksellere bitişik piksellerde eşik ``komsu_esik_orani × sigma_lim`` olur; kozmik ışınların
        sönük kenarlarını yakalar. 1 verilirse komşulara ayrı eşik uygulanmaz.

    Returns
    -------
    (temiz görüntü, maske) : işaretlenen pikseller çevredeki iyi piksellerin
    ortancasıyla değiştirilmiş görüntü ve ``True`` = kozmik ışın maskesi.

    References
    ----------
    [vanDokkum2001] kernel.eq, noise.eq, sigmap.eq ve izleyen denklemler.

    Validity
    --------
    PSF iyi örneklenmişse (FWHM yaklaşık 2,5 piksel ve üstü) yıldızlar güvenle
    ayrılır; eksik örneklenmiş veride ``f_lim`` büyütülmelidir. Kaynak, 5σ
    eşiğinde özgün görüntüdeki 5σ tepelerin yaklaşık yarısının işaretlendiğini
    belirtir. Gerçek HST STIS karesinde iki pozun farkına karşı ölçülen tamlık ve
    saflık ``figures/cosmic_rays_real_data.json`` dosyasındadır.

    Verification
    ------------
    tests/unit/test_cosmic_rays.py: sentetik yıldız alanına eklenen kozmik ışınların tamlığı ve yıldızların korunması.
    """
    g = np.asarray(image_adu, dtype=float)
    temiz = g.copy()
    mask = np.zeros(g.shape, dtype=bool)
    ny, nx = g.shape
    for _ in range(max_iterations):
        low = np.repeat(np.repeat(temiz, _SUBSAMPLING, axis=0), _SUBSAMPLING, axis=1)
        lap = np.clip(ndimage.convolve(low, _LAPLACE, mode="nearest"), 0.0, None)
        l_arti = lap.reshape(ny, _SUBSAMPLING, nx, _SUBSAMPLING).mean(axis=(1, 3))
        m5 = ndimage.median_filter(temiz, size=5, mode="nearest")
        gurultu = np.sqrt(np.clip(gain_e_adu * m5, 0.0, None) + read_noise_e**2) / gain_e_adu
        s = l_arti / (_SUBSAMPLING * gurultu)
        s_ussu = s - ndimage.median_filter(s, size=5, mode="nearest")
        m3 = ndimage.median_filter(temiz, size=3, mode="nearest")
        ince = np.clip(m3 - ndimage.median_filter(m3, size=7, mode="nearest"), 0.01, None)
        yeni = (s_ussu > sigma_lim) & (l_arti / ince > f_lim) & ~mask
        if neighbor_threshold_ratio < 1.0 and (yeni | mask).any():
            bitisik = ndimage.binary_dilation(yeni | mask, structure=np.ones((3, 3)))
            yeni |= bitisik & (s_ussu > neighbor_threshold_ratio * sigma_lim) & (l_arti / ince > f_lim) & ~mask
        if not yeni.any():
            break
        mask |= yeni
        # Yalnızca maskeli piksellerin çevresinde maskeli ortanca hesaplanır (hız için).
        ys, xs = np.nonzero(mask)
        for y0, x0 in zip(ys, xs, strict=True):
            y1, y2, x1, x2 = max(0, y0 - 2), min(ny, y0 + 3), max(0, x0 - 2), min(nx, x0 + 3)
            iyi = ~mask[y1:y2, x1:x2]
            if iyi.any():
                temiz[y0, x0] = np.median(g[y1:y2, x1:x2][iyi])
    return temiz, mask
