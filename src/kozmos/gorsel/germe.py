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

from kozmos._kaynak import kaynak

Dizi = NDArray[np.float64]


@kaynak(
    "Lupton2004",
    yer="Bölüm 2, dipnot: f(x) = asinh(α Q (x − m)) / Q",
    teyit="okundu",
    not_="Q → 0 limitinde doğrusal germe α(x − m); m en küçük değer (siyah nokta)",
)
def asinh_ger(goruntu: ArrayLike, *, siyah: float, alfa: float, q: float = 8.0) -> Dizi:
    """asinh germesi uygular: f(x) = asinh(α Q (x − m)) / Q.

    Parametreler
    ------------
    goruntu : ndarray
    siyah : float
        Siyaha karşılık gelen değer ``m`` (genellikle gökyüzü düzeyi).
    alfa : float
        Doğrusal bölgenin eğimi α; ``1/alfa`` soluk yapının ölçeğidir.
    q : float
        Doğrusaldan logaritmiğe geçişin sertliği; ``q → 0`` doğrusal germedir.

    Döndürür
    --------
    ndarray
        Gerilmiş değerler; alt sınır 0, üst sınır uygulanmaz.

    Kaynaklar
    ---------
    [Lupton2004] Bölüm 2, dipnot.

    Geçerlilik
    ----------
    ``siyah`` altındaki değerler 0'a kırpılır.

    Doğrulama
    ---------
    tests/birim/test_gorsel.py: küçük sinyalde doğrusal limit, tekdüzelik.
    """
    x = np.asarray(goruntu, dtype=float) - siyah
    if q <= 0:
        return np.clip(alfa * x, 0.0, None)
    return np.clip(np.arcsinh(alfa * q * x) / q, 0.0, None)


@kaynak(
    "Lupton2004",
    yer="Bölüm 2, TeX etiketi EqIntensityStretch",
    teyit="okundu",
    not_="I = (r+g+b)/3; R = r f(I)/I, G = g f(I)/I, B = b f(I)/I; en büyük bileşen 1'i aşarsa üçü de ona bölünür",
)
def lupton_rgb(
    kirmizi: ArrayLike, yesil: ArrayLike, mavi: ArrayLike, *, siyah: float = 0.0, alfa: float, q: float = 8.0
) -> Dizi:
    """Üç banttan renk koruyan RGB bileşimi üretir.

    Germe toplam yeğinliğe I = (r + g + b)/3 uygulanır ve üç bant aynı
    çarpanla f(I)/I ölçeklenir; böylece bir cismin rengi parlaklığına bağlı
    olarak değişmez. Bileşenlerden biri 1'i aşarsa üçü birden en büyüğe
    bölünür: yeğinlik kırpılır, renk korunur.

    Parametreler
    ------------
    kirmizi, yesil, mavi : ndarray (Y, X)
        Uzun, orta ve kısa dalga boyu görüntüleri; aynı ızgarada, gökyüzü çıkarılmış.
    siyah, alfa, q : float
        ``asinh_ger`` ile aynı anlamda.

    Döndürür
    --------
    ndarray (Y, X, 3)
        [0, 1] aralığında RGB.

    Kaynaklar
    ---------
    [Lupton2004] Bölüm 2, EqIntensityStretch.

    Geçerlilik
    ----------
    Bantların göreli ölçeği çağıran tarafından ayarlanır; ölçekleme renk dengesini belirler.

    Doğrulama
    ---------
    tests/birim/test_gorsel.py: renk oranlarının korunumu, doymuş pikselde rengin korunması.
    """
    r, g, b = (np.clip(np.asarray(k, dtype=float) - siyah, 0.0, None) for k in (kirmizi, yesil, mavi))
    yeginlik = (r + g + b) / 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        carpan = np.where(yeginlik > 0, asinh_ger(yeginlik, siyah=0.0, alfa=alfa, q=q) / yeginlik, 0.0)
    rgb = np.stack([r * carpan, g * carpan, b * carpan], axis=-1)
    en_buyuk = rgb.max(axis=-1, keepdims=True)
    return np.where(en_buyuk > 1.0, rgb / np.maximum(en_buyuk, 1e-300), rgb)


@kaynak(
    "Lupton2004",
    yer="Bölüm 2: doğrusal eşleme ve en büyük değer M",
    teyit="okundu",
    not_="alt ve üst sınırın yüzdeliklerden seçilmesi bu projenin kuralıdır; kaynak m ve M'yi kullanıcıya bırakır",
)
def yuzdelik_olcekle(goruntu: ArrayLike, alt_yuzde: float = 1.0, ust_yuzde: float = 99.5) -> Dizi:
    """Görüntüyü iki yüzdelik arasında [0, 1] aralığına doğrusal eşler.

    Kaynaklar
    ---------
    [Lupton2004] Bölüm 2.

    Geçerlilik
    ----------
    Sonlu olmayan pikseller yüzdelik hesabına girmez ve çıktıda 0 olur.

    Doğrulama
    ---------
    tests/birim/test_gorsel.py
    """
    x = np.asarray(goruntu, dtype=float)
    alt, ust = np.nanpercentile(x, [alt_yuzde, ust_yuzde])
    if ust <= alt:
        return np.zeros_like(x)
    return np.nan_to_num(np.clip((x - alt) / (ust - alt), 0.0, 1.0), nan=0.0)
