"""Gözlem planlama: bir hedefin gece boyunca yüksekliği, azimutu ve hava kütlesi.

Var olan yapı taşlarını birleştirir: Greenwich ortalama yıldız zamanı
(``zaman.julyen``, ERFA ``gmst06.c``) ve saat açısından ufuk koordinatlarına
dönüşüm (``koordinat.donusum``, ERFA ``hd2ae.c``) [ERFA2023].

Koordinatlar çerçeve yanlılığı ve presesyonla tarihin ortalama ekvatoruna
taşınır (``koordinat.presesyon``, ERFA ``pmat06.c``); istenirse atmosfer
kırılması eklenir (``gozlem.kirilma``, ERFA ``refco.c``). Nutasyon, aberasyon,
kutup hareketi ve UT1 − UTC farkı uygulanmaz; astropy ile ölçülen fark bir yay
dakikasının altındadır (tests/oracle/test_planning_oracle.py). Teleskop
yönlendirmesi için yeterli değildir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.coordinates.precession import icrs_to_mean_of_date
from kozmos.coordinates.transforms import airmass_plane_parallel, hour_angle_to_horizontal
from kozmos.core.angles import TWO_PI, wrap_pm_pi
from kozmos.observing.refraction import observed_altitude
from kozmos.time.julian import greenwich_mean_sidereal_time

Array = NDArray[np.float64]
SIDEREAL_RATE = 1.00273781191135448  # era00.c: UT1 günü başına Dünya dönüşü (tur)


@dataclass(frozen=True, slots=True)
class Visibility:
    """Bir hedefin verilen anlardaki ufuk koordinatları."""

    hour_angle_rad: Array
    azimuth_rad: Array
    altitude_rad: Array
    airmass: Array


@source(
    "ERFA2023",
    where="gmst06.c, hd2ae.c, pmat06.c, refco.c",
    status="read",
    note="koordinatlar yanlılık-presesyon matrisiyle tarihe taşınır; H = GMST + doğu boylamı − sağ açıklık; ardından "
    "saat açısı → azimut, yükseklik; istenirse kırılma. UT1 ≈ UTC ≈ TT alınır",
)
def visibility(
    ra_rad: float,
    dec_rad: float,
    latitude_rad: float,
    east_longitude_rad: float,
    jd1: ArrayLike,
    jd2: ArrayLike = 0.0,
    *,
    to_date: bool = True,
    refraction: tuple[float, float] | None = None,
) -> Visibility:
    """Bir hedefin verilen anlardaki saat açısını, azimutunu, yüksekliğini ve hava kütlesini hesaplar.

    Parameters
    ----------
    sag_acikllik_rad, dik_acikllik_rad : float
        Hedefin ICRS (J2000) koordinatları [rad].
    enlem_rad, dogu_boylami_rad : float
        Gözlemcinin enlemi ve doğuya artı boylamı [rad].
    jd1, jd2 : float ya da ndarray
        İki parçalı Jülyen tarihi (UTC).
    tarihe_tasi : bool
        Doğruysa koordinatlar tarihin ortalama ekvatoruna taşınır; yanlışsa verildiği gibi kullanılır
        (koordinatlar zaten tarihin ekinoksundaysa).
    kirilma : (A, B), isteğe bağlı
        ``gozlem.kirilma.kirilma_sabitleri`` çıktısı; verilirse yükseklik gözlenen (kırılmış) yüksekliktir.

    Returns
    -------
    Gorunurluk

    References
    ----------
    [ERFA2023] gmst06.c, hd2ae.c, pmat06.c, refco.c.

    Validity
    --------
    UT1, UTC ve TT aynı sayılır (yıldız zamanında en çok birkaç saniyelik
    etki). Nutasyon ve aberasyon uygulanmaz. ``tarihe_tasi=False`` ve J2000
    koordinatlarıyla hata yılda yaklaşık 50 yay saniyesi birikir. Kırılma
    modeli yaklaşık 5° yüksekliğin altında ``nan`` verir. Hava kütlesi
    düzlem-paralel yaklaşımdır; ufkun altında ``nan``.

    Verification
    ------------
    tests/oracle/test_planning_oracle.py: astropy AltAz ile bir yay dakikası içinde uyum (kırılmalı ve kırılmasız).
    """
    ra, dec = ra_rad, dec_rad
    if to_date:
        ra, dec = icrs_to_mean_of_date(ra, dec, jd1, jd2)
    gmst = greenwich_mean_sidereal_time(jd1, jd2, jd1, jd2)
    saat_acisi = wrap_pm_pi(gmst + east_longitude_rad - ra)
    azimut, yukseklik = hour_angle_to_horizontal(saat_acisi, dec, latitude_rad)
    if refraction is not None:
        yukseklik = observed_altitude(yukseklik, *refraction)
    return Visibility(saat_acisi, azimut, yukseklik, airmass_plane_parallel(yukseklik))


@source(
    "ERFA2023",
    where="era00.c (dönme hızı), hd2ae.c",
    status="read",
    note="meridyen geçişi H = 0; saat açısı UT1 günü başına 1,00273781191135448 tur ilerler",
)
def transit_jd(
    ra_rad: float,
    east_longitude_rad: float,
    jd_start: float,
    *,
    dec_rad: float | None = None,
) -> float:
    """Verilen andan sonraki ilk üst meridyen geçişinin Jülyen tarihini bulur.

    ``dik_acikllik_rad`` verilirse koordinatlar ICRS sayılır ve sağ açıklık başlangıç
    anının ortalama ekinoksuna taşınır (presesyon dik açıklığa da bağlıdır); verilmezse
    sağ açıklık olduğu gibi kullanılır.

    References
    ----------
    [ERFA2023] era00.c, hd2ae.c.

    Validity
    --------
    ``gorunurluk`` ile aynı yaklaşımlar. Sonuç birkaç saniye doğruluğundadır.

    Verification
    ------------
    tests/oracle/test_planning_oracle.py: bulunan anda saat açısı sıfır, yükseklik en büyük.
    """
    if dec_rad is not None:
        ra_rad = float(icrs_to_mean_of_date(ra_rad, dec_rad, jd_start)[0])
    gmst = float(greenwich_mean_sidereal_time(jd_start, 0.0, jd_start, 0.0))
    kalan = (ra_rad - gmst - east_longitude_rad) % TWO_PI
    return jd_start + kalan / TWO_PI / SIDEREAL_RATE


@source(
    "ERFA2023",
    where="hd2ae.c (yükseklik bileşeni)",
    status="read",
    note="sin h = sin φ sin δ + cos φ cos δ cos H denkleminin H için çözümü (türetim bu projede)",
)
def time_above_horizon_hr(dec_rad: ArrayLike, latitude_rad: float, altitude_limit_rad: float = 0.0) -> Array:
    """Bir hedefin günde verilen yükseklik sınırının üstünde kaldığı süreyi hesaplar.

    cos H₀ = (sin h₀ − sin φ sin δ) / (cos φ cos δ); süre = 2 H₀ (yıldız zamanıyla).

    Returns
    -------
    float ya da ndarray
        Süre [yıldız saati]; hiç doğmuyorsa 0, hiç batmıyorsa 24.

    References
    ----------
    [ERFA2023] hd2ae.c.

    Validity
    --------
    Kırılma ve hedefin açısal boyutu hesaba katılmaz.

    Verification
    ------------
    tests/oracle/test_planning_oracle.py: ekvatorda her yıldız 12 saat; kutup yıldızları 24 saat.
    """
    d = np.asarray(dec_rad, dtype=float)
    cosine = (np.sin(altitude_limit_rad) - np.sin(latitude_rad) * np.sin(d)) / (np.cos(latitude_rad) * np.cos(d))
    return 2.0 * np.arccos(np.clip(cosine, -1.0, 1.0)) / TWO_PI * 24.0
