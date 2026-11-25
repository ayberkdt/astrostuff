"""Takvim, Jülyen tarihi, Dünya dönme açısı ve yıldız zamanı.

Algoritmalar ve katsayılar ERFA kaynak dosyalarından okunmuştur [ERFA2023]:
``cal2jd.c``, ``jd2cal.c``, ``era00.c``, ``gmst06.c``, ``obl06.c``.

Jülyen tarihi iki parçalı tutulur (``jd1 + jd2``); toplam tek ``float64`` ile
J2000 yakınında yaklaşık 40 mikrosaniye çözünürlük verir.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._kaynak import kaynak
from kozmos.temel.aci import IKI_PI, YAYSANIYE_RAD, sifir_iki_pi

J2000_JD = 2451545.0  # erfam.h ERFA_DJ00
MJD_SIFIR = 2400000.5  # erfam.h ERFA_DJM0
JULYEN_YUZYILI = 36525.0  # erfam.h ERFA_DJC
TT_EKSI_TAI_S = 32.184  # erfam.h ERFA_TTMTAI
GUN_S = 86400.0  # erfam.h ERFA_DAYSEC

Dizi = NDArray[np.float64]
_AY_UZUNLUKLARI = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


@kaynak(
    "ERFA2023",
    yer="cal2jd.c",
    teyit="okundu",
    not_="Gregoryen takvim; C'deki sıfıra doğru tamsayı bölmesi (im-14)/12 açıkça uygulanır",
)
def takvim_den_mjd(yil: ArrayLike, ay: ArrayLike, gun: ArrayLike) -> NDArray[np.int64]:
    """Gregoryen takvim tarihinin 0h anındaki Değiştirilmiş Jülyen Tarihini verir.

    Parametreler
    ------------
    yil, ay, gun : int ya da tamsayı dizisi
        Proleptik Gregoryen takvimde tarih; yıl >= −4799.

    Döndürür
    --------
    int ya da tamsayı dizisi
        MJD (0h). Jülyen tarihi ``2400000.5 + MJD`` olur.

    Kaynaklar
    ---------
    [ERFA2023] cal2jd.c.

    Geçerlilik
    ----------
    Yıl >= −4799. Jülyen takvimi (1582 öncesi sivil kullanım) desteklenmez;
    bütün tarihler proleptik Gregoryen yorumlanır.

    Doğrulama
    ---------
    tests/kahin/test_zaman_kahin.py: erfa.cal2jd ile tam eşitlik.
    """
    iy = np.asarray(yil, dtype=np.int64)
    im = np.asarray(ay, dtype=np.int64)
    id_ = np.asarray(gun, dtype=np.int64)
    if np.any(iy < -4799):
        raise ValueError("yıl -4799'dan küçük olamaz")
    if np.any((im < 1) | (im > 12)):
        raise ValueError("ay 1-12 aralığında olmalı")
    artik = (im == 2) & (iy % 4 == 0) & ((iy % 100 != 0) | (iy % 400 == 0))
    if np.any((id_ < 1) | (id_ > _AY_UZUNLUKLARI[im - 1] + artik)):
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


@kaynak("ERFA2023", yer="cal2jd.c; erfam.h ERFA_DJM0", teyit="okundu")
def takvim_den_jd(yil: ArrayLike, ay: ArrayLike, gun: ArrayLike, gun_kesri: ArrayLike = 0.0) -> Dizi:
    """Gregoryen takvim tarihinden Jülyen tarihini verir.

    Parametreler
    ------------
    yil, ay, gun : int
        Takvim tarihi.
    gun_kesri : float
        Gece yarısından itibaren gün kesri, [0, 1).

    Döndürür
    --------
    float ya da ndarray
        Jülyen tarihi (tek sayı). Yüksek duyarlık için iki parçayı ayrı tutun:
        ``2400000.5`` ve ``takvim_den_mjd(...) + gun_kesri``.

    Kaynaklar
    ---------
    [ERFA2023] cal2jd.c; erfam.h ERFA_DJM0.

    Doğrulama
    ---------
    tests/kahin/test_zaman_kahin.py
    """
    return MJD_SIFIR + takvim_den_mjd(yil, ay, gun) + np.asarray(gun_kesri, dtype=float)


@kaynak(
    "ERFA2023",
    yer="jd2cal.c",
    teyit="okundu",
    not_="takvim çevrimi aynen; gün kesrinin dengeli toplaması sadeleştirilmiştir",
)
def jd_den_takvim(
    jd1: ArrayLike, jd2: ArrayLike = 0.0
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], Dizi]:
    """İki parçalı Jülyen tarihinden Gregoryen takvim tarihini verir.

    Döndürür
    --------
    (yıl, ay, gün, gün_kesri)

    Kaynaklar
    ---------
    [ERFA2023] jd2cal.c.

    Geçerlilik
    ----------
    −68569.5 <= JD <= 1e9. Gün kesri iki parçanın kesirlerinden toplanır;
    ERFA'daki dengeli toplama uygulanmadığından kesir son basamakta farklı olabilir.

    Doğrulama
    ---------
    tests/kahin/test_zaman_kahin.py: erfa.jd2cal ile takvim alanlarında tam eşitlik.
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
    gun = ell - (2447 * k) // 80
    ell = k // 11
    ay = k + 2 - 12 * ell
    yil = 100 * (n - 49) + i + ell
    return yil, ay, gun, f


@kaynak("ERFA2023", yer="erfam.h ERFA_TTMTAI, ERFA_DAYSEC", teyit="okundu")
def tai_den_tt(jd1: ArrayLike, jd2: ArrayLike = 0.0) -> tuple[Dizi, Dizi]:
    """TAI'yi TT'ye çevirir: TT = TAI + 32,184 s.

    Kaynaklar
    ---------
    [ERFA2023] erfam.h ERFA_TTMTAI.

    Doğrulama
    ---------
    tests/kahin/test_zaman_kahin.py: erfa.taitt.
    """
    return np.asarray(jd1, dtype=float), np.asarray(jd2, dtype=float) + TT_EKSI_TAI_S / GUN_S


@kaynak("ERFA2023", yer="erfam.h ERFA_TTMTAI, ERFA_DAYSEC", teyit="okundu")
def tt_den_tai(jd1: ArrayLike, jd2: ArrayLike = 0.0) -> tuple[Dizi, Dizi]:
    """TT'yi TAI'ye çevirir: TAI = TT − 32,184 s.

    Kaynaklar
    ---------
    [ERFA2023] erfam.h ERFA_TTMTAI.

    Doğrulama
    ---------
    tests/kahin/test_zaman_kahin.py: erfa.tttai.
    """
    return np.asarray(jd1, dtype=float), np.asarray(jd2, dtype=float) - TT_EKSI_TAI_S / GUN_S


@kaynak(
    "ERFA2023",
    yer="era00.c",
    teyit="okundu",
    not_="θ = 2π (f + 0.7790572732640 + 0.00273781191135448 t); t = UT1 − J2000 [gün]",
)
def dunya_donme_acisi(ut1_jd1: ArrayLike, ut1_jd2: ArrayLike = 0.0) -> Dizi:
    """Dünya Dönme Açısını (ERA, IAU 2000) hesaplar.

    Parametreler
    ------------
    ut1_jd1, ut1_jd2 : float ya da ndarray
        UT1 ölçeğinde iki parçalı Jülyen tarihi.

    Döndürür
    --------
    float ya da ndarray
        ERA [rad], [0, 2π).

    Kaynaklar
    ---------
    [ERFA2023] era00.c.

    Doğrulama
    ---------
    tests/kahin/test_zaman_kahin.py: erfa.era00 ile fark < 1e-12 rad.
    """
    a = np.asarray(ut1_jd1, dtype=float)
    b = np.asarray(ut1_jd2, dtype=float)
    d1, d2 = np.minimum(a, b), np.maximum(a, b)
    t = d1 + (d2 - J2000_JD)
    f = np.fmod(d1, 1.0) + np.fmod(d2, 1.0)
    return sifir_iki_pi(IKI_PI * (f + 0.7790572732640 + 0.00273781191135448 * t))


@kaynak("ERFA2023", yer="gmst06.c", teyit="okundu", not_="ERA + TT'de 5. derece polinom; katsayılar yay saniyesi")
def greenwich_ortalama_yildiz_zamani(
    ut1_jd1: ArrayLike, ut1_jd2: ArrayLike, tt_jd1: ArrayLike, tt_jd2: ArrayLike
) -> Dizi:
    """Greenwich ortalama yıldız zamanını (IAU 2006 presesyonuyla tutarlı) hesaplar.

    Döndürür
    --------
    float ya da ndarray
        GMST [rad], [0, 2π).

    Kaynaklar
    ---------
    [ERFA2023] gmst06.c.

    Geçerlilik
    ----------
    UT1 ve TT ayrı verilir; ikisi için aynı değer verilirse hata birkaç
    mikroyay saniyesi mertebesindedir (ERFA belge notu; bu çalışmada ölçülmedi).

    Doğrulama
    ---------
    tests/kahin/test_zaman_kahin.py: erfa.gmst06 ile fark < 1e-12 rad.
    """
    t = ((np.asarray(tt_jd1, dtype=float) - J2000_JD) + np.asarray(tt_jd2, dtype=float)) / JULYEN_YUZYILI
    polinom = (
        0.014506 + (4612.156534 + (1.3915817 + (-0.00000044 + (-0.000029956 + (-0.0000000368) * t) * t) * t) * t) * t
    )
    return sifir_iki_pi(dunya_donme_acisi(ut1_jd1, ut1_jd2) + polinom * YAYSANIYE_RAD)


@kaynak("ERFA2023", yer="obl06.c", teyit="okundu", not_="ε0 = 84381.406″ ve 5. derece polinom")
def ortalama_egiklik(tt_jd1: ArrayLike, tt_jd2: ArrayLike = 0.0) -> Dizi:
    """Ekliptiğin ortalama eğikliğini (IAU 2006) hesaplar.

    Döndürür
    --------
    float ya da ndarray
        Ortalama eğiklik [rad].

    Kaynaklar
    ---------
    [ERFA2023] obl06.c.

    Doğrulama
    ---------
    tests/kahin/test_zaman_kahin.py: erfa.obl06 ile fark < 1e-15 rad.
    """
    t = ((np.asarray(tt_jd1, dtype=float) - J2000_JD) + np.asarray(tt_jd2, dtype=float)) / JULYEN_YUZYILI
    eps = 84381.406 + (-46.836769 + (-0.0001831 + (0.00200340 + (-0.000000576 + (-0.0000000434) * t) * t) * t) * t) * t
    return eps * YAYSANIYE_RAD


@kaynak(
    "ERFA2023",
    yer="gmst06.c; boylam doğuya artı",
    teyit="bekliyor",
    not_="yerel yıldız zamanı = GMST + doğu boylamı; ortalama (görünür değil)",
)
def yerel_ortalama_yildiz_zamani(
    ut1_jd1: ArrayLike, ut1_jd2: ArrayLike, tt_jd1: ArrayLike, tt_jd2: ArrayLike, dogu_boylami_rad: ArrayLike
) -> Dizi:
    """Yerel ortalama yıldız zamanını hesaplar.

    Kaynaklar
    ---------
    [ERFA2023] gmst06.c.

    Geçerlilik
    ----------
    Ekinoksların denklemi (nutasyon) eklenmez; görünür yıldız zamanından
    farkı yaklaşık 1 saniyeye kadar çıkar.

    Doğrulama
    ---------
    tests/kahin/test_zaman_kahin.py
    """
    gmst = greenwich_ortalama_yildiz_zamani(ut1_jd1, ut1_jd2, tt_jd1, tt_jd2)
    return sifir_iki_pi(gmst + np.asarray(dogu_boylami_rad, dtype=float))
