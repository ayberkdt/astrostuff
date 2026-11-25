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

from kozmos._kaynak import kaynak

Dizi = NDArray[np.float64]
Maske = NDArray[np.bool_]

# [vanDokkum2001] kernel.eq: (1/4) [[0, −1, 0], [−1, 4, −1], [0, −1, 0]]
_LAPLACE = 0.25 * np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
_ALT_ORNEKLEME = 2  # f_s


@kaynak(
    "vanDokkum2001",
    yer="TeX etiketleri kernel.eq, noise.eq, sigmap.eq ve izleyen denklemler (S′, F, seçim ölçütü)",
    teyit="okundu",
    not_="alt örnekleme çarpanı f_s = 2. σ_lim ve f_lim için makalede önerilen sayısal değerler aranmadı; "
    "varsayılanlar (5 ve 2) bu projenin seçimidir. Komşu piksellere daha düşük eşik uygulama adımı yoktur",
)
def kozmik_isin_bul(
    goruntu_adu: ArrayLike,
    *,
    kazanc_e_adu: float,
    okuma_gurultusu_e: float,
    sigma_lim: float = 5.0,
    f_lim: float = 2.0,
    en_cok_yineleme: int = 4,
) -> tuple[Dizi, Maske]:
    """Tek karede kozmik ışınları saptar ve temizler.

    Parametreler
    ------------
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

    Döndürür
    --------
    (temiz görüntü, maske) : işaretlenen pikseller çevredeki iyi piksellerin
    ortancasıyla değiştirilmiş görüntü ve ``True`` = kozmik ışın maskesi.

    Kaynaklar
    ---------
    [vanDokkum2001] kernel.eq, noise.eq, sigmap.eq ve izleyen denklemler.

    Geçerlilik
    ----------
    PSF iyi örneklenmişse (FWHM yaklaşık 2,5 piksel ve üstü) yıldızlar güvenle
    ayrılır; eksik örneklenmiş veride ``f_lim`` büyütülmelidir. Kaynak, 5σ
    eşiğinde özgün görüntüdeki 5σ tepelerin yaklaşık yarısının işaretlendiğini
    belirtir.

    Doğrulama
    ---------
    tests/birim/test_kozmik_isin.py: sentetik yıldız alanına eklenen kozmik ışınların tamlığı ve yıldızların korunması.
    """
    g = np.asarray(goruntu_adu, dtype=float)
    temiz = g.copy()
    maske = np.zeros(g.shape, dtype=bool)
    ny, nx = g.shape
    for _ in range(en_cok_yineleme):
        alt = np.repeat(np.repeat(temiz, _ALT_ORNEKLEME, axis=0), _ALT_ORNEKLEME, axis=1)
        lap = np.clip(ndimage.convolve(alt, _LAPLACE, mode="nearest"), 0.0, None)
        l_arti = lap.reshape(ny, _ALT_ORNEKLEME, nx, _ALT_ORNEKLEME).mean(axis=(1, 3))
        m5 = ndimage.median_filter(temiz, size=5, mode="nearest")
        gurultu = np.sqrt(np.clip(kazanc_e_adu * m5, 0.0, None) + okuma_gurultusu_e**2) / kazanc_e_adu
        s = l_arti / (_ALT_ORNEKLEME * gurultu)
        s_ussu = s - ndimage.median_filter(s, size=5, mode="nearest")
        m3 = ndimage.median_filter(temiz, size=3, mode="nearest")
        ince = np.clip(m3 - ndimage.median_filter(m3, size=7, mode="nearest"), 0.01, None)
        yeni = (s_ussu > sigma_lim) & (l_arti / ince > f_lim) & ~maske
        if not yeni.any():
            break
        maske |= yeni
        # Yalnızca maskeli piksellerin çevresinde maskeli ortanca hesaplanır (hız için).
        ys, xs = np.nonzero(maske)
        for y0, x0 in zip(ys, xs, strict=True):
            y1, y2, x1, x2 = max(0, y0 - 2), min(ny, y0 + 3), max(0, x0 - 2), min(nx, x0 + 3)
            iyi = ~maske[y1:y2, x1:x2]
            if iyi.any():
                temiz[y0, x0] = np.median(g[y1:y2, x1:x2][iyi])
    return temiz, maske
