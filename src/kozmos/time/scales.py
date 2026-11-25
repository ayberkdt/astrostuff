"""Zaman ölçekleri arasındaki farklar: TAI − UTC ve türevleri.

Artık saniye tablosu ve 1972 öncesi sürüklenme katsayıları ERFA ``dat.c``
dosyasından okunmuştur [ERFA2023]. Tablo yeni bir artık saniye ilan
edildiğinde güncellenmelidir; ``TABLO_SURUM_YILI`` tablonun ait olduğu ERFA
sürüm yılıdır.
"""

from __future__ import annotations

import warnings

from kozmos._sources import source
from kozmos.time.julian import TT_MINUS_TAI_S, calendar_to_mjd

TABLE_RELEASE_YEAR = 2023  # dat.c IYV

# (yıl, ay, TAI − UTC [s]); dat.c içindeki _changes tablosu.
_CHANGES: tuple[tuple[int, int, float], ...] = (
    (1960, 1, 1.4178180),
    (1961, 1, 1.4228180),
    (1961, 8, 1.3728180),
    (1962, 1, 1.8458580),
    (1963, 11, 1.9458580),
    (1964, 1, 3.2401300),
    (1964, 4, 3.3401300),
    (1964, 9, 3.4401300),
    (1965, 1, 3.5401300),
    (1965, 3, 3.6401300),
    (1965, 7, 3.7401300),
    (1965, 9, 3.8401300),
    (1966, 1, 4.3131700),
    (1968, 2, 4.2131700),
    (1972, 1, 10.0),
    (1972, 7, 11.0),
    (1973, 1, 12.0),
    (1974, 1, 13.0),
    (1975, 1, 14.0),
    (1976, 1, 15.0),
    (1977, 1, 16.0),
    (1978, 1, 17.0),
    (1979, 1, 18.0),
    (1980, 1, 19.0),
    (1981, 7, 20.0),
    (1982, 7, 21.0),
    (1983, 7, 22.0),
    (1985, 7, 23.0),
    (1988, 1, 24.0),
    (1990, 1, 25.0),
    (1991, 1, 26.0),
    (1992, 7, 27.0),
    (1993, 7, 28.0),
    (1994, 7, 29.0),
    (1996, 1, 30.0),
    (1997, 7, 31.0),
    (1999, 1, 32.0),
    (2006, 1, 33.0),
    (2009, 1, 34.0),
    (2012, 7, 35.0),
    (2015, 7, 36.0),
    (2017, 1, 37.0),
)

# 1972 öncesi: (referans MJD, sürüklenme [s/gün]); dat.c içindeki drift tablosu.
_DRIFT: tuple[tuple[float, float], ...] = (
    (37300.0, 0.0012960),
    (37300.0, 0.0012960),
    (37300.0, 0.0012960),
    (37665.0, 0.0011232),
    (37665.0, 0.0011232),
    (38761.0, 0.0012960),
    (38761.0, 0.0012960),
    (38761.0, 0.0012960),
    (38761.0, 0.0012960),
    (38761.0, 0.0012960),
    (38761.0, 0.0012960),
    (38761.0, 0.0012960),
    (39126.0, 0.0025920),
    (39126.0, 0.0025920),
)


@source("ERFA2023", where="dat.c", status="read", note="tablo 2017-01-01 girişiyle biter; ERFA sürüm yılı 2023")
def tai_minus_utc_s(year: int, month: int, day: int, day_fraction: float = 0.0) -> float:
    """Verilen UTC tarihinde TAI − UTC farkını (ΔAT) saniye olarak verir.

    Parameters
    ----------
    yil, ay, gun : int
        UTC takvim tarihi (Gregoryen).
    gun_kesri : float
        Günün kesri, [0, 1]; yalnızca 1972 öncesi sürüklenme döneminde sonucu etkiler.

    Returns
    -------
    float
        TAI − UTC [s].

    References
    ----------
    [ERFA2023] dat.c.

    Validity
    --------
    1960 ve sonrası. Tablonun sürüm yılından beş yıldan fazla ileri tarihlerde
    uyarı verilir: o tarihe kadar yeni artık saniye ilan edilmiş olabilir.
    Artık saniyenin eklendiği günün içindeki 86401. saniye bu fonksiyonla
    ayırt edilmez.

    Verification
    ------------
    tests/oracle/test_time_scales_oracle.py: erfa.dat ile tam eşitlik.
    """
    if not 0.0 <= day_fraction <= 1.0:
        raise ValueError("gün kesri [0, 1] aralığında olmalı")
    mjd = int(calendar_to_mjd(year, month, day))
    if year < _CHANGES[0][0]:
        raise ValueError("UTC 1960 öncesinde tanımlı değil")
    if year > TABLE_RELEASE_YEAR + 5:
        warnings.warn(
            f"artık saniye tablosu {TABLE_RELEASE_YEAR} sürümüdür; {year} için güncelliği doğrulanmalı",
            stacklevel=2,
        )
    m = 12 * year + month
    for i in range(len(_CHANGES) - 1, -1, -1):
        if m >= 12 * _CHANGES[i][0] + _CHANGES[i][1]:
            break
    else:
        raise ValueError("tarih tablonun başlangıcından önce")
    fark = _CHANGES[i][2]
    if i < len(_DRIFT):
        fark += (mjd + day_fraction - _DRIFT[i][0]) * _DRIFT[i][1]
    return fark


@source("ERFA2023", where="dat.c; erfam.h ERFA_TTMTAI", status="read", note="TT − UTC = (TAI − UTC) + 32,184 s")
def tt_minus_utc_s(year: int, month: int, day: int, day_fraction: float = 0.0) -> float:
    """Verilen UTC tarihinde TT − UTC farkını saniye olarak verir.

    References
    ----------
    [ERFA2023] dat.c; erfam.h ERFA_TTMTAI.

    Verification
    ------------
    tests/oracle/test_time_scales_oracle.py
    """
    return tai_minus_utc_s(year, month, day, day_fraction) + TT_MINUS_TAI_S
