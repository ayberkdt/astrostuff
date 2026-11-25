"""Zaman ölçekleri arasındaki farklar: TAI − UTC ve türevleri.

Artık saniye tablosu ve 1972 öncesi sürüklenme katsayıları ERFA ``dat.c``
dosyasından okunmuştur [ERFA2024]. Tablo yeni bir artık saniye ilan
edildiğinde güncellenmelidir; ``TABLO_SURUM_YILI`` tablonun ait olduğu ERFA
sürüm yılıdır.
"""

from __future__ import annotations

import warnings

from kozmos._kaynak import kaynak
from kozmos.zaman.julyen import TT_EKSI_TAI_S, takvim_den_mjd

TABLO_SURUM_YILI = 2023  # dat.c IYV

# (yıl, ay, TAI − UTC [s]); dat.c içindeki _changes tablosu.
_DEGISIKLIKLER: tuple[tuple[int, int, float], ...] = (
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
_SURUKLENME: tuple[tuple[float, float], ...] = (
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


@kaynak("ERFA2024", yer="dat.c", teyit="okundu", not_="tablo 2017-01-01 girişiyle biter; ERFA sürüm yılı 2023")
def tai_eksi_utc_s(yil: int, ay: int, gun: int, gun_kesri: float = 0.0) -> float:
    """Verilen UTC tarihinde TAI − UTC farkını (ΔAT) saniye olarak verir.

    Parametreler
    ------------
    yil, ay, gun : int
        UTC takvim tarihi (Gregoryen).
    gun_kesri : float
        Günün kesri, [0, 1]; yalnızca 1972 öncesi sürüklenme döneminde sonucu etkiler.

    Döndürür
    --------
    float
        TAI − UTC [s].

    Kaynaklar
    ---------
    [ERFA2024] dat.c.

    Geçerlilik
    ----------
    1960 ve sonrası. Tablonun sürüm yılından beş yıldan fazla ileri tarihlerde
    uyarı verilir: o tarihe kadar yeni artık saniye ilan edilmiş olabilir.
    Artık saniyenin eklendiği günün içindeki 86401. saniye bu fonksiyonla
    ayırt edilmez.

    Doğrulama
    ---------
    tests/kahin/test_olcekler_kahin.py: erfa.dat ile tam eşitlik.
    """
    if not 0.0 <= gun_kesri <= 1.0:
        raise ValueError("gün kesri [0, 1] aralığında olmalı")
    mjd = int(takvim_den_mjd(yil, ay, gun))
    if yil < _DEGISIKLIKLER[0][0]:
        raise ValueError("UTC 1960 öncesinde tanımlı değil")
    if yil > TABLO_SURUM_YILI + 5:
        warnings.warn(
            f"artık saniye tablosu {TABLO_SURUM_YILI} sürümüdür; {yil} için güncelliği doğrulanmalı",
            stacklevel=2,
        )
    m = 12 * yil + ay
    for i in range(len(_DEGISIKLIKLER) - 1, -1, -1):
        if m >= 12 * _DEGISIKLIKLER[i][0] + _DEGISIKLIKLER[i][1]:
            break
    else:
        raise ValueError("tarih tablonun başlangıcından önce")
    fark = _DEGISIKLIKLER[i][2]
    if i < len(_SURUKLENME):
        fark += (mjd + gun_kesri - _SURUKLENME[i][0]) * _SURUKLENME[i][1]
    return fark


@kaynak("ERFA2024", yer="dat.c; erfam.h ERFA_TTMTAI", teyit="okundu", not_="TT − UTC = (TAI − UTC) + 32,184 s")
def tt_eksi_utc_s(yil: int, ay: int, gun: int, gun_kesri: float = 0.0) -> float:
    """Verilen UTC tarihinde TT − UTC farkını saniye olarak verir.

    Kaynaklar
    ---------
    [ERFA2024] dat.c; erfam.h ERFA_TTMTAI.

    Doğrulama
    ---------
    tests/kahin/test_olcekler_kahin.py
    """
    return tai_eksi_utc_s(yil, ay, gun, gun_kesri) + TT_EKSI_TAI_S
