"""Takvim, Jülyen tarihi, Dünya dönme açısı ve yıldız zamanı.

Algoritmalar ve katsayılar ERFA kaynak dosyalarından okunmuştur [ERFA2023]:
``cal2jd.c``, ``jd2cal.c``, ``era00.c``, ``gmst06.c``, ``obl06.c``.

Jülyen tarihi iki parçalı tutulur (``jd1 + jd2``); toplam tek ``float64`` ile
J2000 yakınında yaklaşık 40 mikrosaniye çözünürlük verir.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.angles import ARCSEC_RAD, TWO_PI, wrap_0_2pi

J2000_JD = 2451545.0  # erfam.h ERFA_DJ00
MJD_ZERO = 2400000.5  # erfam.h ERFA_DJM0
JULIAN_CENTURY = 36525.0  # erfam.h ERFA_DJC
TT_MINUS_TAI_S = 32.184  # erfam.h ERFA_TTMTAI
DAY_S = 86400.0  # erfam.h ERFA_DAYSEC

Array = NDArray[np.float64]
_MONTH_LENGTHS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


@source(
    "ERFA2023",
    where="cal2jd.c",
    status="read",
    note="Gregoryen takvim; C'deki sıfıra doğru tamsayı bölmesi (im-14)/12 açıkça uygulanır",
)
def calendar_to_mjd(year: ArrayLike, month: ArrayLike, day: ArrayLike) -> NDArray[np.int64]:
    """Gregoryen takvim tarihinin 0h anındaki Değiştirilmiş Jülyen Tarihini verir.

    Parameters
    ----------
    yil, ay, gun : int ya da tamsayı dizisi
        Proleptik Gregoryen takvimde tarih; yıl >= −4799.

    Returns
    -------
    int ya da tamsayı dizisi
        MJD (0h). Jülyen tarihi ``2400000.5 + MJD`` olur.

    References
    ----------
    [ERFA2023] cal2jd.c.

    Validity
    --------
    Yıl >= −4799. Jülyen takvimi (1582 öncesi sivil kullanım) desteklenmez;
    bütün tarihler proleptik Gregoryen yorumlanır.

    Verification
    ------------
    tests/oracle/test_time_oracle.py: erfa.cal2jd ile tam eşitlik.
    """
    iy = np.asarray(year, dtype=np.int64)
    im = np.asarray(month, dtype=np.int64)
    id_ = np.asarray(day, dtype=np.int64)
    if np.any(iy < -4799):
        raise ValueError("yıl -4799'dan küçük olamaz")
    if np.any((im < 1) | (im > 12)):
        raise ValueError("ay 1-12 aralığında olmalı")
    residual = (im == 2) & (iy % 4 == 0) & ((iy % 100 != 0) | (iy % 400 == 0))
    if np.any((id_ < 1) | (id_ > _MONTH_LENGTHS[im - 1] + residual)):
        raise ValueError("gün o ay için geçersiz")
    my = np.where(im <= 2, -1, 0)  # C: (im - 14) / 12, sıfıra doğru
    iypmy = iy + my
    return (
        (1461 * (iypmy + 4800)) // 4
        + (367 * (im - 2 - 12 * my)) // 12
        - (3 * ((iypmy + 4900) // 100)) // 4
        + id_
        - 2432076
    )


@source("ERFA2023", where="cal2jd.c; erfam.h ERFA_DJM0", status="read")
def calendar_to_jd(year: ArrayLike, month: ArrayLike, day: ArrayLike, day_fraction: ArrayLike = 0.0) -> Array:
    """Gregoryen takvim tarihinden Jülyen tarihini verir.

    Parameters
    ----------
    yil, ay, gun : int
        Takvim tarihi.
    gun_kesri : float
        Gece yarısından itibaren gün kesri, [0, 1).

    Returns
    -------
    float ya da ndarray
        Jülyen tarihi (tek sayı). Yüksek duyarlık için iki parçayı ayrı tutun:
        ``2400000.5`` ve ``takvim_den_mjd(...) + gun_kesri``.

    References
    ----------
    [ERFA2023] cal2jd.c; erfam.h ERFA_DJM0.

    Verification
    ------------
    tests/oracle/test_time_oracle.py
    """
    return MJD_ZERO + calendar_to_mjd(year, month, day) + np.asarray(day_fraction, dtype=float)


@source(
    "ERFA2023",
    where="jd2cal.c",
    status="read",
    note="takvim çevrimi aynen; gün kesrinin dengeli toplaması sadeleştirilmiştir",
)
def jd_to_calendar(
    jd1: ArrayLike, jd2: ArrayLike = 0.0
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], Array]:
    """İki parçalı Jülyen tarihinden Gregoryen takvim tarihini verir.

    Returns
    -------
    (yıl, ay, gün, gün_kesri)

    References
    ----------
    [ERFA2023] jd2cal.c.

    Validity
    --------
    −68569.5 <= JD <= 1e9. Gün kesri iki parçanın kesirlerinden toplanır;
    ERFA'daki dengeli toplama uygulanmadığından kesir son basamakta farklı olabilir.

    Verification
    ------------
    tests/oracle/test_time_oracle.py: erfa.jd2cal ile takvim alanlarında tam eşitlik.
    """
    a = np.asarray(jd1, dtype=float)
    b = np.asarray(jd2, dtype=float)
    if np.any((a + b < -68569.5) | (a + b > 1e9)):
        raise ValueError("Jülyen tarihi desteklenen aralığın dışında")
    da, db = np.rint(a), np.rint(b)
    f = (a - da) + (b - db) + 0.5
    tasma = np.floor(f)
    jd = (da + db + tasma).astype(np.int64)
    f = f - tasma
    ell = jd + 68569
    n = (4 * ell) // 146097
    ell = ell - (146097 * n + 3) // 4
    i = (4000 * (ell + 1)) // 1461001
    ell = ell - (1461 * i) // 4 + 31
    k = (80 * ell) // 2447
    day = ell - (2447 * k) // 80
    ell = k // 11
    month = k + 2 - 12 * ell
    year = 100 * (n - 49) + i + ell
    return year, month, day, f


@source("ERFA2023", where="erfam.h ERFA_TTMTAI, ERFA_DAYSEC", status="read")
def tai_to_tt(jd1: ArrayLike, jd2: ArrayLike = 0.0) -> tuple[Array, Array]:
    """TAI'yi TT'ye çevirir: TT = TAI + 32,184 s.

    References
    ----------
    [ERFA2023] erfam.h ERFA_TTMTAI.

    Verification
    ------------
    tests/oracle/test_time_oracle.py: erfa.taitt.
    """
    return np.asarray(jd1, dtype=float), np.asarray(jd2, dtype=float) + TT_MINUS_TAI_S / DAY_S


@source("ERFA2023", where="erfam.h ERFA_TTMTAI, ERFA_DAYSEC", status="read")
def tt_to_tai(jd1: ArrayLike, jd2: ArrayLike = 0.0) -> tuple[Array, Array]:
    """TT'yi TAI'ye çevirir: TAI = TT − 32,184 s.

    References
    ----------
    [ERFA2023] erfam.h ERFA_TTMTAI.

    Verification
    ------------
    tests/oracle/test_time_oracle.py: erfa.tttai.
    """
    return np.asarray(jd1, dtype=float), np.asarray(jd2, dtype=float) - TT_MINUS_TAI_S / DAY_S


@source(
    "ERFA2023",
    where="era00.c",
    status="read",
    note="θ = 2π (f + 0.7790572732640 + 0.00273781191135448 t); t = UT1 − J2000 [gün]",
)
def earth_rotation_angle(ut1_jd1: ArrayLike, ut1_jd2: ArrayLike = 0.0) -> Array:
    """Dünya Dönme Açısını (ERA, IAU 2000) hesaplar.

    Parameters
    ----------
    ut1_jd1, ut1_jd2 : float ya da ndarray
        UT1 ölçeğinde iki parçalı Jülyen tarihi.

    Returns
    -------
    float ya da ndarray
        ERA [rad], [0, 2π).

    References
    ----------
    [ERFA2023] era00.c.

    Verification
    ------------
    tests/oracle/test_time_oracle.py: erfa.era00 ile fark < 1e-12 rad.
    """
    a = np.asarray(ut1_jd1, dtype=float)
    b = np.asarray(ut1_jd2, dtype=float)
    d1, d2 = np.minimum(a, b), np.maximum(a, b)
    t = d1 + (d2 - J2000_JD)
    f = np.fmod(d1, 1.0) + np.fmod(d2, 1.0)
    return wrap_0_2pi(TWO_PI * (f + 0.7790572732640 + 0.00273781191135448 * t))


@source("ERFA2023", where="gmst06.c", status="read", note="ERA + TT'de 5. derece polinom; katsayılar yay saniyesi")
def greenwich_mean_sidereal_time(ut1_jd1: ArrayLike, ut1_jd2: ArrayLike, tt_jd1: ArrayLike, tt_jd2: ArrayLike) -> Array:
    """Greenwich ortalama yıldız zamanını (IAU 2006 presesyonuyla tutarlı) hesaplar.

    Returns
    -------
    float ya da ndarray
        GMST [rad], [0, 2π).

    References
    ----------
    [ERFA2023] gmst06.c.

    Validity
    --------
    UT1 ve TT ayrı verilir; ikisi için aynı değer verilirse hata birkaç
    mikroyay saniyesi mertebesindedir (ERFA belge notu; bu çalışmada ölçülmedi).

    Verification
    ------------
    tests/oracle/test_time_oracle.py: erfa.gmst06 ile fark < 1e-12 rad.
    """
    t = ((np.asarray(tt_jd1, dtype=float) - J2000_JD) + np.asarray(tt_jd2, dtype=float)) / JULIAN_CENTURY
    polinom = (
        0.014506 + (4612.156534 + (1.3915817 + (-0.00000044 + (-0.000029956 + (-0.0000000368) * t) * t) * t) * t) * t
    )
    return wrap_0_2pi(earth_rotation_angle(ut1_jd1, ut1_jd2) + polinom * ARCSEC_RAD)


@source("ERFA2023", where="obl06.c", status="read", note="ε0 = 84381.406″ ve 5. derece polinom")
def mean_obliquity(tt_jd1: ArrayLike, tt_jd2: ArrayLike = 0.0) -> Array:
    """Ekliptiğin ortalama eğikliğini (IAU 2006) hesaplar.

    Returns
    -------
    float ya da ndarray
        Ortalama eğiklik [rad].

    References
    ----------
    [ERFA2023] obl06.c.

    Verification
    ------------
    tests/oracle/test_time_oracle.py: erfa.obl06 ile fark < 1e-15 rad.
    """
    t = ((np.asarray(tt_jd1, dtype=float) - J2000_JD) + np.asarray(tt_jd2, dtype=float)) / JULIAN_CENTURY
    eps = 84381.406 + (-46.836769 + (-0.0001831 + (0.00200340 + (-0.000000576 + (-0.0000000434) * t) * t) * t) * t) * t
    return eps * ARCSEC_RAD


@source("ERFA2023", where="gmst06.c", status="read")
@source(
    "Astropy2022",
    where="time/core.py, Time._sid_time_or_earth_rot_ang",
    status="read",
    note="açı = boylam + θ (doğu boylamı artı); kutup hareketi ve TIO konumu eklenmez",
)
def local_mean_sidereal_time(
    ut1_jd1: ArrayLike, ut1_jd2: ArrayLike, tt_jd1: ArrayLike, tt_jd2: ArrayLike, east_longitude_rad: ArrayLike
) -> Array:
    """Yerel ortalama yıldız zamanını hesaplar.

    References
    ----------
    [ERFA2023] gmst06.c; [Astropy2022] time/core.py.

    Validity
    --------
    Ekinoksların denklemi (nutasyon) eklenmez; görünür yıldız zamanından
    farkı yaklaşık 1 saniyeye kadar çıkar.

    Verification
    ------------
    tests/oracle/test_time_oracle.py
    """
    gmst = greenwich_mean_sidereal_time(ut1_jd1, ut1_jd2, tt_jd1, tt_jd2)
    return wrap_0_2pi(gmst + np.asarray(east_longitude_rad, dtype=float))
