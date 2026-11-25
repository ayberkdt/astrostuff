"""İki cisim problemi: Kepler denklemi ve temel bağıntılar.

Bu modüldeki denklemler ders kitabı biçimleriyle yazılmıştır; atıf verilen
kaynakların metni bu çalışmada açılmamıştır ve bütün kayıtlar "bekliyor"
durumundadır. Doğruluk, denklemin kendisine dayanan özellik testleriyle
(artık, gidiş-dönüş, korunum) sınanır.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._kaynak import kaynak
from kozmos.temel.aci import IKI_PI, eksi_pi_arti_pi

Dizi = NDArray[np.float64]


@kaynak(
    "MurrayDermott1999",
    yer="Bölüm 2.4, Kepler denklemi",
    teyit="bekliyor",
    not_="M = E − e sin E; Newton-Raphson yinelemesi; başlangıç E0 = M + e sin M",
)
def dis_merkezli_anomali(
    ortalama_anomali_rad: ArrayLike, dis_merkezlik: ArrayLike, *, tolerans: float = 1e-14, en_cok_yineleme: int = 50
) -> Dizi:
    """Kepler denklemini eliptik yörünge için çözer: M = E − e sin E.

    Parametreler
    ------------
    ortalama_anomali_rad : float ya da ndarray
        Ortalama anomali M [rad].
    dis_merkezlik : float ya da ndarray
        Dış merkezlik, 0 <= e < 1.
    tolerans : float
        Yinelemenin durma ölçütü, |ΔE| [rad].

    Döndürür
    --------
    float ya da ndarray
        Dış merkezli anomali E [rad]; M ile aynı 2π dalında.

    Kaynaklar
    ---------
    [MurrayDermott1999] Bölüm 2.4.

    Geçerlilik
    ----------
    0 <= e < 1. e → 1 ve M → 0 yakınında Newton yinelemesi yavaşlar;
    e <= 0,999 için sınanmıştır.

    Doğrulama
    ---------
    tests/ozellik/test_kepler_ozellik.py: |E − e sin E − M| < 1e-12.
    """
    m = np.asarray(ortalama_anomali_rad, dtype=float)
    e = np.asarray(dis_merkezlik, dtype=float)
    if np.any((e < 0) | (e >= 1)):
        raise ValueError("dış merkezlik [0, 1) aralığında olmalı")
    m, e = np.broadcast_arrays(m, e)
    m_ind = eksi_pi_arti_pi(m)
    buyuk_e = m_ind + e * np.sin(m_ind)
    # Yüksek dış merkezlikte E0 = π·işaret(M) Newton için güvenli başlangıçtır.
    buyuk_e = np.where(e > 0.8, np.pi * np.sign(m_ind + 1e-300), buyuk_e)
    for _ in range(en_cok_yineleme):
        f = buyuk_e - e * np.sin(buyuk_e) - m_ind
        delta = f / (1.0 - e * np.cos(buyuk_e))
        buyuk_e = buyuk_e - delta
        if np.all(np.abs(delta) < tolerans):
            break
    return buyuk_e + (m - m_ind)


@kaynak(
    "MurrayDermott1999",
    yer="Bölüm 2.4, gerçek anomali",
    teyit="bekliyor",
    not_="tan(f/2) = sqrt((1+e)/(1−e)) tan(E/2); atan2 biçimi",
)
def gercek_anomali(dis_merkezli_anomali_rad: ArrayLike, dis_merkezlik: ArrayLike) -> Dizi:
    """Dış merkezli anomaliden gerçek anomaliyi hesaplar.

    Kaynaklar
    ---------
    [MurrayDermott1999] Bölüm 2.4.

    Doğrulama
    ---------
    tests/ozellik/test_kepler_ozellik.py: yörünge denklemi r = a(1 − e cos E) = a(1 − e²)/(1 + e cos f).
    """
    buyuk_e = np.asarray(dis_merkezli_anomali_rad, dtype=float)
    e = np.asarray(dis_merkezlik, dtype=float)
    f = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(buyuk_e / 2.0), np.sqrt(1.0 - e) * np.cos(buyuk_e / 2.0))
    # E ile aynı 2π dalında kal.
    return f + IKI_PI * np.round((buyuk_e - f) / IKI_PI)


@kaynak("MurrayDermott1999", yer="Bölüm 2.3, Kepler'in üçüncü yasası", teyit="bekliyor", not_="P = 2π sqrt(a³ / μ)")
def yorunge_donemi_s(yari_buyuk_eksen_m: ArrayLike, kutle_parametresi_m3_s2: ArrayLike) -> Dizi:
    """Yörünge dönemini hesaplar: P = 2π sqrt(a³/μ).

    Parametreler
    ------------
    yari_buyuk_eksen_m : float ya da ndarray
        Yarı büyük eksen [m].
    kutle_parametresi_m3_s2 : float ya da ndarray
        μ = G(m1 + m2) [m³ s⁻²].

    Kaynaklar
    ---------
    [MurrayDermott1999] Bölüm 2.3.

    Doğrulama
    ---------
    tests/birim/test_kepler.py: 1 au ve nominal GM_Güneş için bir Gauss yılı dolayı [Prsa2016].
    """
    a = np.asarray(yari_buyuk_eksen_m, dtype=float)
    return IKI_PI * np.sqrt(a**3 / np.asarray(kutle_parametresi_m3_s2, dtype=float))


@kaynak("MurrayDermott1999", yer="Bölüm 2.4, vis-viva", teyit="bekliyor", not_="v² = μ (2/r − 1/a)")
def yorunge_hizi_m_s(uzaklik_m: ArrayLike, yari_buyuk_eksen_m: ArrayLike, kutle_parametresi_m3_s2: ArrayLike) -> Dizi:
    """Yörünge hızını vis-viva denkleminden hesaplar.

    Kaynaklar
    ---------
    [MurrayDermott1999] Bölüm 2.4.

    Doğrulama
    ---------
    tests/birim/test_kepler.py: dairesel yörüngede v = sqrt(μ/r).
    """
    r = np.asarray(uzaklik_m, dtype=float)
    a = np.asarray(yari_buyuk_eksen_m, dtype=float)
    return np.sqrt(np.asarray(kutle_parametresi_m3_s2, dtype=float) * (2.0 / r - 1.0 / a))
