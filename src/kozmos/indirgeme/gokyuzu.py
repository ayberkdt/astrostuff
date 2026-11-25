"""Kaydırmalı (dither) karelerden gökyüzü modeli kurma ve çıkarma.

Yakın kızılötesinde gökyüzü kaynaklardan çok daha parlaktır ve dakikalar
içinde değişir. Çözüm, teleskobu pozlar arasında kaydırıp aynı gecenin
karelerinden gökyüzünün kendisini modellemektir: yıldızlar her karede başka
piksele düştüğü için piksel piksel ortanca onları eler.

Durağan ve dinamik model ayrımı ve kuralları THELI makalesinin arXiv TeX
kaynağından (1308.4989, ``theli.tex``, "Static and dynamic background models")
okunmuştur [Schirmer2013]: durağan model bir dizi pozun tek ortanca
görüntüsüdür ve genel düzey değişimi için yeniden ölçeklenir; dinamik model
k'ıncı poz için zamanca en yakın m pozdan kurulur; her iki durumda nesneler
önce maskelenir ve yığındaki en yüksek piksel atılır.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._kaynak import kaynak

Dizi = NDArray[np.float64]


def _ortanca_model(yigin: Dizi, en_yuksegi_at: int) -> Dizi:
    if en_yuksegi_at > 0 and yigin.shape[0] > en_yuksegi_at + 1:
        yigin = np.sort(yigin, axis=0)  # nan'lar sona gider
        sayi = np.sum(np.isfinite(yigin), axis=0)
        sira = np.arange(yigin.shape[0])[:, None, None]
        yigin = np.where(sira >= sayi - en_yuksegi_at, np.nan, yigin)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # bütün değerleri maskeli pikseller: aşağıda doldurulur
        return np.nanmedian(yigin, axis=0)


def _model(kareler: Dizi, maskeler: NDArray[np.bool_] | None, en_yuksegi_at: int) -> Dizi:
    """Her karesi kendi ortancasına bölünmüş yığının piksel piksel ortancası (birim düzeyli model).

    Bir piksel bütün karelerde maskeliyse (ya da atma sonrası değeri kalmıyorsa) o pikselde maskesiz
    ortanca kullanılır; model hiçbir pikselde ``nan`` kalmaz.
    """
    duzey = np.nanmedian(np.where(maskeler, np.nan, kareler) if maskeler is not None else kareler, axis=(1, 2))
    yigin = kareler / duzey[:, None, None]
    if maskeler is None:
        return _ortanca_model(yigin, en_yuksegi_at)
    model = _ortanca_model(np.where(maskeler, np.nan, yigin), en_yuksegi_at)
    bos = ~np.isfinite(model)
    if bos.any():
        model[bos] = _ortanca_model(yigin, en_yuksegi_at)[bos]
    return model


@kaynak(
    "Schirmer2013",
    yer="'Static and dynamic background models' bölümü: durağan model",
    teyit="okundu",
    not_="tek ortanca görüntü; uygulanacağı kareye yeniden ölçeklenir; nesneler maskelenir ve yığındaki en yüksek "
    "piksel atılır. Ölçeğin karenin (maskesiz) ortancasından alınması bu projenin seçimidir",
)
def duragan_gokyuzu_cikar(
    kareler: ArrayLike, *, maskeler: ArrayLike | None = None, en_yuksegi_at: int = 1
) -> tuple[Dizi, Dizi]:
    """Bütün karelerden tek bir gökyüzü modeli kurar ve her kareden ölçekleyerek çıkarır.

    Parametreler
    ------------
    kareler : ndarray (K, Y, X)
        Aynı filtrede, pozlar arasında kaydırılmış kareler (düz alan düzeltmesi yapılmış).
    maskeler : ndarray[bool] (K, Y, X), isteğe bağlı
        ``True`` = nesne pikseli; modele girmez.
    en_yuksegi_at : int
        Her pikselde yığının en yüksek kaç değerinin atılacağı.

    Döndürür
    --------
    (gökyüzü çıkarılmış kareler, model) : model birim düzeylidir (ortancası 1 dolayında).

    Kaynaklar
    ---------
    [Schirmer2013] 'Static and dynamic background models'.

    Geçerlilik
    ----------
    Gökyüzünün biçimi dizi boyunca değişmiyorsa (kaynak: 10-30 dakikadan kısa
    yakın kızılötesi dizileri) yeterlidir. Kaydırma adımından büyük nesneler
    modele sızar ve kendi kendini çıkarır. Kalabalık alanda maske şarttır: 12
    yıldızlı sentetik dizide açıklık akısı maskesiz modelle %6,8 eksik, nesne
    maskesiyle %0,6 eksik ölçülmüştür.
    En yüksek değeri atmak yıldız kalıntısını azaltır ama boş gökyüzünde ortancayı
    aşağı çeker: 12 karelik sentetik dizide artık, kare gürültüsünün yaklaşık onda
    biri kadar artıya kayar (tests/birim/test_gokyuzu.py). Nesne maskesi verilmişse
    ``en_yuksegi_at=0`` bu yanlılığı kaldırır. Gerçek SOFI Ks dizisinde (16 kare, 28 dakika)
    durağan model büyük ölçekli artığı 156 ADU'dan 0,8 ADU'ya indirmiştir; gökadanın alanı
    doldurduğu ikinci dizide artık 4,6 ADU'da kalmıştır (``belgeler/kizilotesi_indirgeme.json``).

    Doğrulama
    ---------
    tests/birim/test_gokyuzu.py: değişken düzeyli sentetik dizide artığın gürültü düzeyine inmesi ve yıldız akısının korunması.
    """
    k = np.asarray(kareler, dtype=float)
    if k.ndim != 3 or k.shape[0] < 3:
        raise ValueError("en az üç kare (K, Y, X) gerekir")
    m = None if maskeler is None else np.asarray(maskeler, dtype=bool)
    model = _model(k, m, en_yuksegi_at)
    duzey = np.nanmedian(np.where(m, np.nan, k) if m is not None else k, axis=(1, 2))
    return k - duzey[:, None, None] * model[None, :, :], model


@kaynak(
    "Schirmer2013",
    yer="'Static and dynamic background models' bölümü: dinamik model",
    teyit="okundu",
    not_="k'ıncı pozun modeli zamanca en yakın m pozdan kurulur; pencere k'ıncı pozu simetrik çevrelemek zorunda "
    "değildir. Karenin kendisi modele girmez (bu projenin seçimi; kaynak belirtmez)",
)
def dinamik_gokyuzu_cikar(
    kareler: ArrayLike,
    zamanlar: ArrayLike,
    *,
    komsu_sayisi: int = 6,
    maskeler: ArrayLike | None = None,
    en_yuksegi_at: int = 1,
) -> Dizi:
    """Her kare için zamanca en yakın karelerden gökyüzü modeli kurar ve çıkarır.

    Parametreler
    ------------
    kareler : ndarray (K, Y, X)
    zamanlar : ndarray (K,)
        Poz ortası zamanları (herhangi bir birim).
    komsu_sayisi : int
        Her model için kullanılacak en yakın kare sayısı m (karenin kendisi hariç).
    maskeler, en_yuksegi_at
        ``duragan_gokyuzu_cikar`` ile aynı.

    Döndürür
    --------
    ndarray (K, Y, X)
        Gökyüzü çıkarılmış kareler.

    Kaynaklar
    ---------
    [Schirmer2013] 'Static and dynamic background models'.

    Geçerlilik
    ----------
    Gökyüzünün biçimi dizi içinde değişiyorsa (kaynak: 10-30 dakikadan uzun
    yakın kızılötesi dizileri) gereklidir. Az komşu gürültüyü artırır: m kareden
    kurulan modelin gürültüsü çıkarılan kareye yaklaşık 1/√m oranında eklenir. Gerçek SOFI
    dizisinde (28 dakika) 8 komşulu dinamik model durağan modelden gürültülü çıkmıştır
    (8,2 ve 7,1 ADU); kısa dizide durağan model yeğlenir.

    Doğrulama
    ---------
    tests/birim/test_gokyuzu.py: biçimi zamanla değişen gökyüzünde durağan modelden küçük artık.
    """
    k = np.asarray(kareler, dtype=float)
    t = np.asarray(zamanlar, dtype=float)
    if k.ndim != 3 or t.shape != (k.shape[0],):
        raise ValueError("kareler (K, Y, X), zamanlar (K,) olmalı")
    if komsu_sayisi < 3 or komsu_sayisi >= k.shape[0]:
        raise ValueError("komşu sayısı en az 3 ve kare sayısından küçük olmalı")
    m = None if maskeler is None else np.asarray(maskeler, dtype=bool)
    cikti = np.empty_like(k)
    for i in range(k.shape[0]):
        sira = np.argsort(np.abs(t - t[i]), kind="stable")
        komsu = sira[sira != i][:komsu_sayisi]
        model = _model(k[komsu], None if m is None else m[komsu], en_yuksegi_at)
        duzey = np.nanmedian(np.where(m[i], np.nan, k[i]) if m is not None else k[i])
        cikti[i] = k[i] - duzey * model
    return cikti
