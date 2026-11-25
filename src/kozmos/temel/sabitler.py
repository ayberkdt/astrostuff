"""Fiziksel ve astronomik sabitler.

Her sabit değeri, birimi, belirsizliği ve kaynağıyla tanımlıdır. Değerler
kaynak metinlerinden 25 Kasım 2026'da okunmuştur:

- CODATA 2022: NIST ``allascii.txt`` tablosu [CODATA2022].
- IAU 2015 B3 nominal Güneş ve gezegen değerleri: [Prsa2016], Tablo 1.
- IAU 2015 B2 bolometrik sıfır noktası: [Mamajek2015].
- Astronomik birim, Jülyen yılı ve yüzyılı, J2000: ERFA ``erfam.h`` [ERFA2024].

Türetilmiş sabitler (``PARSEK``, ``MEGAPARSEK`` ...) tanımlarından hesaplanır
ve ``turetim`` alanında formülü taşır.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from kozmos._kaynak import altyapi


@dataclass(frozen=True, slots=True)
class Sabit:
    """Kaynağıyla birlikte bir sabit."""

    ad: str
    deger: float
    birim: str
    belirsizlik: float  # 0.0: tanım gereği kesin
    kaynak: str
    yer: str
    teyit: str = "okundu"
    turetim: str | None = None

    def __float__(self) -> float:
        return self.deger


# --- CODATA 2022 [CODATA2022]; NIST allascii.txt -----------------------------------
ISIK_HIZI = Sabit("ışık hızı", 299_792_458.0, "m s^-1", 0.0, "CODATA2022", "speed of light in vacuum")
PLANCK = Sabit("Planck sabiti", 6.626_070_15e-34, "J Hz^-1", 0.0, "CODATA2022", "Planck constant")
BOLTZMANN = Sabit("Boltzmann sabiti", 1.380_649e-23, "J K^-1", 0.0, "CODATA2022", "Boltzmann constant")
CEKIM = Sabit(
    "Newton çekim sabiti",
    6.674_30e-11,
    "m^3 kg^-1 s^-2",
    0.000_15e-11,
    "CODATA2022",
    "Newtonian constant of gravitation",
)
TEMEL_YUK = Sabit("temel yük", 1.602_176_634e-19, "C", 0.0, "CODATA2022", "elementary charge")
ELEKTRON_KUTLESI = Sabit(
    "elektron kütlesi", 9.109_383_7139e-31, "kg", 0.000_000_0028e-31, "CODATA2022", "electron mass"
)
PROTON_KUTLESI = Sabit("proton kütlesi", 1.672_621_925_95e-27, "kg", 0.000_000_000_52e-27, "CODATA2022", "proton mass")
ATOMIK_KUTLE_BIRIMI = Sabit(
    "atomik kütle sabiti",
    1.660_539_068_92e-27,
    "kg",
    0.000_000_000_52e-27,
    "CODATA2022",
    "atomic mass constant",
)
AVOGADRO = Sabit("Avogadro sabiti", 6.022_140_76e23, "mol^-1", 0.0, "CODATA2022", "Avogadro constant")
THOMSON_KESITI = Sabit(
    "Thomson tesir kesiti",
    6.652_458_7051e-29,
    "m^2",
    0.000_000_0062e-29,
    "CODATA2022",
    "Thomson cross section",
)
# NIST tablosu bu üç sabiti kesik biçimde ("5.670 374 419...") verir; tam değer
# kesin sabitlerden türetilir ve tablodaki basamaklarla testte karşılaştırılır.
STEFAN_BOLTZMANN = Sabit(
    "Stefan-Boltzmann sabiti",
    2 * math.pi**5 * BOLTZMANN.deger**4 / (15 * PLANCK.deger**3 * ISIK_HIZI.deger**2),
    "W m^-2 K^-4",
    0.0,
    "CODATA2022",
    "Stefan-Boltzmann constant (5.670 374 419... e-8)",
    turetim="2 pi^5 k^4 / (15 h^3 c^2)",
    teyit="okundu",
)
WIEN_DALGABOYU = Sabit(
    "Wien kayma sabiti (dalga boyu)",
    2.897_771_955e-3,
    "m K",
    0.0,
    "CODATA2022",
    "Wien wavelength displacement law constant (2.897 771 955... e-3)",
)
IKINCI_ISINIM_SABITI = Sabit(
    "ikinci ışınım sabiti",
    PLANCK.deger * ISIK_HIZI.deger / BOLTZMANN.deger,
    "m K",
    0.0,
    "CODATA2022",
    "second radiation constant (1.438 776 877... e-2)",
    turetim="h c / k",
)

# --- IAU 2015 B3 nominal değerleri [Prsa2016], Tablo 1 ------------------------------
GUNES_YARICAPI = Sabit("nominal Güneş yarıçapı", 6.957e8, "m", 0.0, "Prsa2016", "Tablo 1")
GUNES_ISINIM_GUCU = Sabit("nominal Güneş ışınım gücü", 3.828e26, "W", 0.0, "Prsa2016", "Tablo 1")
GUNES_ETKIN_SICAKLIGI = Sabit("nominal Güneş etkin sıcaklığı", 5772.0, "K", 0.0, "Prsa2016", "Tablo 1")
GUNES_SABITI = Sabit("nominal toplam Güneş ışınımı", 1361.0, "W m^-2", 0.0, "Prsa2016", "Tablo 1")
GM_GUNES = Sabit("nominal Güneş kütle parametresi", 1.327_124_4e20, "m^3 s^-2", 0.0, "Prsa2016", "Tablo 1")
GM_YER = Sabit("nominal Yer kütle parametresi", 3.986_004e14, "m^3 s^-2", 0.0, "Prsa2016", "Tablo 1")
GM_JUPITER = Sabit("nominal Jüpiter kütle parametresi", 1.266_865_3e17, "m^3 s^-2", 0.0, "Prsa2016", "Tablo 1")
YER_EKVATOR_YARICAPI = Sabit("nominal Yer ekvator yarıçapı", 6.3781e6, "m", 0.0, "Prsa2016", "Tablo 1")
YER_KUTUP_YARICAPI = Sabit("nominal Yer kutup yarıçapı", 6.3568e6, "m", 0.0, "Prsa2016", "Tablo 1")
JUPITER_EKVATOR_YARICAPI = Sabit("nominal Jüpiter ekvator yarıçapı", 7.1492e7, "m", 0.0, "Prsa2016", "Tablo 1")
JUPITER_KUTUP_YARICAPI = Sabit("nominal Jüpiter kutup yarıçapı", 6.6854e7, "m", 0.0, "Prsa2016", "Tablo 1")

# --- IAU 2015 B2 bolometrik sıfır noktası [Mamajek2015] ------------------------------
BOLOMETRIK_SIFIR_ISINIM_GUCU = Sabit("M_bol = 0 ışınım gücü", 3.0128e28, "W", 0.0, "Mamajek2015", "L_0 tanımı")

# --- ERFA erfam.h [ERFA2024] ---------------------------------------------------------
ASTRONOMIK_BIRIM = Sabit("astronomik birim", 149_597_870.7e3, "m", 0.0, "ERFA2024", "erfam.h ERFA_DAU")
GUN_SANIYE = Sabit("gün", 86_400.0, "s", 0.0, "ERFA2024", "erfam.h ERFA_DAYSEC")
JULYEN_YILI_GUN = Sabit("Jülyen yılı", 365.25, "gün", 0.0, "ERFA2024", "erfam.h ERFA_DJY")
JULYEN_YUZYILI_GUN = Sabit("Jülyen yüzyılı", 36_525.0, "gün", 0.0, "ERFA2024", "erfam.h ERFA_DJC")
J2000_JD = Sabit("J2000.0 Jülyen tarihi", 2_451_545.0, "gün", 0.0, "ERFA2024", "erfam.h ERFA_DJ00")
MJD_SIFIR_JD = Sabit("MJD sıfır noktası", 2_400_000.5, "gün", 0.0, "ERFA2024", "erfam.h ERFA_DJM0")
TT_EKSI_TAI = Sabit("TT − TAI", 32.184, "s", 0.0, "ERFA2024", "erfam.h ERFA_TTMTAI")

# --- Türetilmiş sabitler -------------------------------------------------------------
# Parsek: 1 yay saniyelik paralaksa karşılık gelen uzaklık, au · 648000/π.
# Bu tanım IAU 2015 B2 kararının notunda geçer; metinden okunmadığı için teyit bekliyor.
PARSEK = Sabit(
    "parsek",
    ASTRONOMIK_BIRIM.deger * 648_000.0 / math.pi,
    "m",
    0.0,
    "Mamajek2015",
    "parsek tanımı notu",
    teyit="bekliyor",
    turetim="au · 648000 / pi",
)
MEGAPARSEK = Sabit(
    "megaparsek",
    1e6 * PARSEK.deger,
    "m",
    0.0,
    "Mamajek2015",
    "parsek tanımı notu",
    teyit="bekliyor",
    turetim="1e6 · parsek",
)
JULYEN_YILI_SANIYE = Sabit(
    "Jülyen yılı",
    JULYEN_YILI_GUN.deger * GUN_SANIYE.deger,
    "s",
    0.0,
    "ERFA2024",
    "erfam.h ERFA_DJY · ERFA_DAYSEC",
    turetim="365.25 · 86400",
)
ISIK_HIZI_KM_S = ISIK_HIZI.deger / 1e3


@altyapi("modül içeriğini listeler; denklem uygulamaz")
def tum_sabitler() -> dict[str, Sabit]:
    """Modüldeki bütün ``Sabit`` nesnelerini adlarıyla döndürür."""
    return {ad: nesne for ad, nesne in globals().items() if isinstance(nesne, Sabit)}
