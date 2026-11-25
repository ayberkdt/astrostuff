"""Astronomik görüntülerin ekrana germe (stretch) fonksiyonları ve renkli bileşim.

Astronomik görüntünün dinamik aralığı ekranınkinden çok geniştir: doğrusal
ölçekte ya soluk yapı ya parlak çekirdek görünür. asinh germesi soluk uçta
doğrusal, parlak uçta logaritmik davranır. Tanımlar ve renk koruyan bileşim
Lupton ve arkadaşlarının makalesinin arXiv TeX kaynağından
(astro-ph/0312483) okunmuştur [Lupton2004].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source

Array = NDArray[np.float64]


@source(
    "Lupton2004",
    where="Bölüm 2, dipnot: f(x) = asinh(α Q (x − m)) / Q",
    status="read",
    note="Q → 0 limitinde doğrusal germe α(x − m); m en küçük değer (siyah nokta)",
)
def asinh_stretch(image: ArrayLike, *, black: float, alpha: float, q: float = 8.0) -> Array:
    """asinh germesi uygular: f(x) = asinh(α Q (x − m)) / Q.

    Parameters
    ----------
    goruntu : ndarray
    siyah : float
        Siyaha karşılık gelen değer ``m`` (genellikle gökyüzü düzeyi).
    alfa : float
        Doğrusal bölgenin eğimi α; ``1/alfa`` soluk yapının ölçeğidir.
    q : float
        Doğrusaldan logaritmiğe geçişin sertliği; ``q → 0`` doğrusal germedir.

    Returns
    -------
    ndarray
        Gerilmiş değerler; alt sınır 0, üst sınır uygulanmaz.

    References
    ----------
    [Lupton2004] Bölüm 2, dipnot.

    Validity
    --------
    ``siyah`` altındaki değerler 0'a kırpılır.

    Verification
    ------------
    tests/unit/test_visualization.py: küçük sinyalde doğrusal limit, tekdüzelik.
    """
    x = np.asarray(image, dtype=float) - black
    if q <= 0:
        return np.clip(alpha * x, 0.0, None)
    return np.clip(np.arcsinh(alpha * q * x) / q, 0.0, None)


@source(
    "Lupton2004",
    where="Bölüm 2, TeX etiketi EqIntensityStretch",
    status="read",
    note="I = (r+g+b)/3; R = r f(I)/I, G = g f(I)/I, B = b f(I)/I; en büyük bileşen 1'i aşarsa üçü de ona bölünür",
)
def lupton_rgb(
    red: ArrayLike, green: ArrayLike, blue: ArrayLike, *, black: float = 0.0, alpha: float, q: float = 8.0
) -> Array:
    """Üç banttan renk koruyan RGB bileşimi üretir.

    Germe toplam yeğinliğe I = (r + g + b)/3 uygulanır ve üç bant aynı
    çarpanla f(I)/I ölçeklenir; böylece bir cismin rengi parlaklığına bağlı
    olarak değişmez. Bileşenlerden biri 1'i aşarsa üçü birden en büyüğe
    bölünür: yeğinlik kırpılır, renk korunur.

    Parameters
    ----------
    kirmizi, yesil, mavi : ndarray (Y, X)
        Uzun, orta ve kısa dalga boyu görüntüleri; aynı ızgarada, gökyüzü çıkarılmış.
    siyah, alfa, q : float
        ``asinh_ger`` ile aynı anlamda.

    Returns
    -------
    ndarray (Y, X, 3)
        [0, 1] aralığında RGB.

    References
    ----------
    [Lupton2004] Bölüm 2, EqIntensityStretch.

    Validity
    --------
    Bantların göreli ölçeği çağıran tarafından ayarlanır; ölçekleme renk dengesini belirler.

    Verification
    ------------
    tests/unit/test_visualization.py: renk oranlarının korunumu, doymuş pikselde rengin korunması.
    """
    r, g, b = (np.clip(np.asarray(k, dtype=float) - black, 0.0, None) for k in (red, green, blue))
    yeginlik = (r + g + b) / 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        carpan = np.where(yeginlik > 0, asinh_stretch(yeginlik, black=0.0, alpha=alpha, q=q) / yeginlik, 0.0)
    rgb = np.stack([r * carpan, g * carpan, b * carpan], axis=-1)
    en_buyuk = rgb.max(axis=-1, keepdims=True)
    return np.where(en_buyuk > 1.0, rgb / np.maximum(en_buyuk, 1e-300), rgb)


@source(
    "Lupton2004",
    where="Bölüm 2: doğrusal eşleme ve en büyük değer M",
    status="read",
    note="alt ve üst sınırın yüzdeliklerden seçilmesi bu projenin kuralıdır; kaynak m ve M'yi kullanıcıya bırakır",
)
def percentile_scale(image: ArrayLike, low_percent: float = 1.0, high_percent: float = 99.5) -> Array:
    """Görüntüyü iki yüzdelik arasında [0, 1] aralığına doğrusal eşler.

    References
    ----------
    [Lupton2004] Bölüm 2.

    Validity
    --------
    Sonlu olmayan pikseller yüzdelik hesabına girmez ve çıktıda 0 olur.

    Verification
    ------------
    tests/unit/test_visualization.py
    """
    x = np.asarray(image, dtype=float)
    low, high = np.nanpercentile(x, [low_percent, high_percent])
    if high <= low:
        return np.zeros_like(x)
    return np.nan_to_num(np.clip((x - low) / (high - low), 0.0, 1.0), nan=0.0)
