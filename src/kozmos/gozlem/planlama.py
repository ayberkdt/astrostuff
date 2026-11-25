"""Gözlem planlama: bir hedefin gece boyunca yüksekliği, azimutu ve hava kütlesi.

Var olan yapı taşlarını birleştirir: Greenwich ortalama yıldız zamanı
(``zaman.julyen``, ERFA ``gmst06.c``) ve saat açısından ufuk koordinatlarına
dönüşüm (``koordinat.donusum``, ERFA ``hd2ae.c``) [ERFA2023].

Planlama doğruluğundadır: presesyon, nutasyon, aberasyon ve kırılma
uygulanmaz. J2000 koordinatlarıyla 2026'da hata yaklaşık 0,4 dereceye kadar
çıkar; teleskop yönlendirmesi için yeterli değildir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._kaynak import kaynak
from kozmos.koordinat.donusum import hava_kutlesi_duzlem_paralel, saat_acisi_den_ufuk
from kozmos.temel.aci import IKI_PI, eksi_pi_arti_pi
from kozmos.zaman.julyen import greenwich_ortalama_yildiz_zamani

Dizi = NDArray[np.float64]
YILDIZ_GUNU_ORANI = 1.00273781191135448  # era00.c: UT1 günü başına Dünya dönüşü (tur)


@dataclass(frozen=True, slots=True)
class Gorunurluk:
    """Bir hedefin verilen anlardaki ufuk koordinatları."""

    saat_acisi_rad: Dizi
    azimut_rad: Dizi
    yukseklik_rad: Dizi
    hava_kutlesi: Dizi


@kaynak(
    "ERFA2023",
    yer="gmst06.c, hd2ae.c",
    teyit="okundu",
    not_="H = GMST + doğu boylamı − sağ açıklık; ardından saat açısı → azimut, yükseklik. UT1 ≈ UTC ≈ TT alınır",
)
def gorunurluk(
    sag_acikllik_rad: float,
    dik_acikllik_rad: float,
    enlem_rad: float,
    dogu_boylami_rad: float,
    jd1: ArrayLike,
    jd2: ArrayLike = 0.0,
) -> Gorunurluk:
    """Bir hedefin verilen anlardaki saat açısını, azimutunu, yüksekliğini ve hava kütlesini hesaplar.

    Parametreler
    ------------
    sag_acikllik_rad, dik_acikllik_rad : float
        Hedefin koordinatları [rad].
    enlem_rad, dogu_boylami_rad : float
        Gözlemcinin enlemi ve doğuya artı boylamı [rad].
    jd1, jd2 : float ya da ndarray
        İki parçalı Jülyen tarihi (UTC).

    Döndürür
    --------
    Gorunurluk

    Kaynaklar
    ---------
    [ERFA2023] gmst06.c, hd2ae.c.

    Geçerlilik
    ----------
    UT1, UTC ve TT aynı sayılır (yıldız zamanında en çok birkaç saniyelik
    etki). Koordinatlar tarihin ekinoksuna taşınmaz: J2000 koordinatlarıyla
    hata yılda yaklaşık 50 yay saniyesi birikir. Hava kütlesi düzlem-paralel
    yaklaşımdır; ufkun altında ``nan``.

    Doğrulama
    ---------
    tests/kahin/test_planlama_kahin.py: astropy AltAz ile 0,6 derece içinde uyum.
    """
    gmst = greenwich_ortalama_yildiz_zamani(jd1, jd2, jd1, jd2)
    saat_acisi = eksi_pi_arti_pi(gmst + dogu_boylami_rad - sag_acikllik_rad)
    azimut, yukseklik = saat_acisi_den_ufuk(saat_acisi, dik_acikllik_rad, enlem_rad)
    return Gorunurluk(saat_acisi, azimut, yukseklik, hava_kutlesi_duzlem_paralel(yukseklik))


@kaynak(
    "ERFA2023",
    yer="era00.c (dönme hızı), hd2ae.c",
    teyit="okundu",
    not_="meridyen geçişi H = 0; saat açısı UT1 günü başına 1,00273781191135448 tur ilerler",
)
def meridyen_gecisi_jd(sag_acikllik_rad: float, dogu_boylami_rad: float, jd_baslangic: float) -> float:
    """Verilen andan sonraki ilk üst meridyen geçişinin Jülyen tarihini bulur.

    Kaynaklar
    ---------
    [ERFA2023] era00.c, hd2ae.c.

    Geçerlilik
    ----------
    ``gorunurluk`` ile aynı yaklaşımlar. Sonuç birkaç saniye doğruluğundadır.

    Doğrulama
    ---------
    tests/kahin/test_planlama_kahin.py: bulunan anda saat açısı sıfır, yükseklik en büyük.
    """
    gmst = float(greenwich_ortalama_yildiz_zamani(jd_baslangic, 0.0, jd_baslangic, 0.0))
    kalan = (sag_acikllik_rad - gmst - dogu_boylami_rad) % IKI_PI
    return jd_baslangic + kalan / IKI_PI / YILDIZ_GUNU_ORANI


@kaynak(
    "ERFA2023",
    yer="hd2ae.c (yükseklik bileşeni)",
    teyit="okundu",
    not_="sin h = sin φ sin δ + cos φ cos δ cos H denkleminin H için çözümü (türetim bu projede)",
)
def ufuk_ustu_suresi_saat(dik_acikllik_rad: ArrayLike, enlem_rad: float, yukseklik_siniri_rad: float = 0.0) -> Dizi:
    """Bir hedefin günde verilen yükseklik sınırının üstünde kaldığı süreyi hesaplar.

    cos H₀ = (sin h₀ − sin φ sin δ) / (cos φ cos δ); süre = 2 H₀ (yıldız zamanıyla).

    Döndürür
    --------
    float ya da ndarray
        Süre [yıldız saati]; hiç doğmuyorsa 0, hiç batmıyorsa 24.

    Kaynaklar
    ---------
    [ERFA2023] hd2ae.c.

    Geçerlilik
    ----------
    Kırılma ve hedefin açısal boyutu hesaba katılmaz.

    Doğrulama
    ---------
    tests/kahin/test_planlama_kahin.py: ekvatorda her yıldız 12 saat; kutup yıldızları 24 saat.
    """
    d = np.asarray(dik_acikllik_rad, dtype=float)
    kosinus = (np.sin(yukseklik_siniri_rad) - np.sin(enlem_rad) * np.sin(d)) / (np.cos(enlem_rad) * np.cos(d))
    return 2.0 * np.arccos(np.clip(kosinus, -1.0, 1.0)) / IKI_PI * 24.0
