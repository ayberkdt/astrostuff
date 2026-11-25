"""Fiziksel ve astronomik sabitler.

Her sabit değeri, birimi, belirsizliği ve kaynağıyla tanımlıdır. Değerler
kaynak metinlerinden 25 Kasım 2026'da okunmuştur:

- CODATA 2022: NIST ``allascii.txt`` tablosu [CODATA2022].
- IAU 2015 B3 nominal Güneş ve gezegen değerleri: [Prsa2016], Tablo 1.
- IAU 2015 B2 bolometrik sıfır noktası: [Mamajek2015].
- Astronomik birim, Jülyen yılı ve yüzyılı, J2000: ERFA ``erfam.h`` [ERFA2023].

Türetilmiş sabitler (``PARSEK``, ``MEGAPARSEK`` ...) tanımlarından hesaplanır
ve ``turetim`` alanında formülü taşır.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from kozmos._sources import infrastructure


@dataclass(frozen=True, slots=True)
class Constant:
    """Kaynağıyla birlikte bir sabit."""

    name: str
    value: float
    unit: str
    uncertainty: float  # 0.0: tanım gereği kesin
    source: str
    where: str
    status: str = "read"
    derivation: str | None = None

    def __float__(self) -> float:
        return self.value


# --- CODATA 2022 [CODATA2022]; NIST allascii.txt -----------------------------------
SPEED_OF_LIGHT = Constant("ışık hızı", 299_792_458.0, "m s^-1", 0.0, "CODATA2022", "speed of light in vacuum")
PLANCK = Constant("Planck sabiti", 6.626_070_15e-34, "J Hz^-1", 0.0, "CODATA2022", "Planck constant")
BOLTZMANN = Constant("Boltzmann sabiti", 1.380_649e-23, "J K^-1", 0.0, "CODATA2022", "Boltzmann constant")
GRAVITATIONAL_CONSTANT = Constant(
    "Newton çekim sabiti",
    6.674_30e-11,
    "m^3 kg^-1 s^-2",
    0.000_15e-11,
    "CODATA2022",
    "Newtonian constant of gravitation",
)
ELEMENTARY_CHARGE = Constant("temel yük", 1.602_176_634e-19, "C", 0.0, "CODATA2022", "elementary charge")
ELECTRON_MASS = Constant(
    "elektron kütlesi", 9.109_383_7139e-31, "kg", 0.000_000_0028e-31, "CODATA2022", "electron mass"
)
PROTON_MASS = Constant("proton kütlesi", 1.672_621_925_95e-27, "kg", 0.000_000_000_52e-27, "CODATA2022", "proton mass")
ATOMIC_MASS_UNIT = Constant(
    "atomik kütle sabiti",
    1.660_539_068_92e-27,
    "kg",
    0.000_000_000_52e-27,
    "CODATA2022",
    "atomic mass constant",
)
AVOGADRO = Constant("Avogadro sabiti", 6.022_140_76e23, "mol^-1", 0.0, "CODATA2022", "Avogadro constant")
THOMSON_CROSS_SECTION = Constant(
    "Thomson tesir kesiti",
    6.652_458_7051e-29,
    "m^2",
    0.000_000_0062e-29,
    "CODATA2022",
    "Thomson cross section",
)
# NIST tablosu bu üç sabiti kesik biçimde ("5.670 374 419...") verir; tam değer
# kesin sabitlerden türetilir ve tablodaki basamaklarla testte karşılaştırılır.
STEFAN_BOLTZMANN = Constant(
    "Stefan-Boltzmann sabiti",
    2 * math.pi**5 * BOLTZMANN.value**4 / (15 * PLANCK.value**3 * SPEED_OF_LIGHT.value**2),
    "W m^-2 K^-4",
    0.0,
    "CODATA2022",
    "Stefan-Boltzmann constant (5.670 374 419... e-8)",
    derivation="2 pi^5 k^4 / (15 h^3 c^2)",
    status="read",
)
WIEN_WAVELENGTH = Constant(
    "Wien kayma sabiti (dalga boyu)",
    2.897_771_955e-3,
    "m K",
    0.0,
    "CODATA2022",
    "Wien wavelength displacement law constant (2.897 771 955... e-3)",
)
SECOND_RADIATION_CONSTANT = Constant(
    "ikinci ışınım sabiti",
    PLANCK.value * SPEED_OF_LIGHT.value / BOLTZMANN.value,
    "m K",
    0.0,
    "CODATA2022",
    "second radiation constant (1.438 776 877... e-2)",
    derivation="h c / k",
)

# --- IAU 2015 B3 nominal değerleri [Prsa2016], Tablo 1 ------------------------------
SOLAR_RADIUS = Constant("nominal Güneş yarıçapı", 6.957e8, "m", 0.0, "Prsa2016", "Tablo 1")
SOLAR_LUMINOSITY = Constant("nominal Güneş ışınım gücü", 3.828e26, "W", 0.0, "Prsa2016", "Tablo 1")
SOLAR_EFFECTIVE_TEMPERATURE = Constant("nominal Güneş etkin sıcaklığı", 5772.0, "K", 0.0, "Prsa2016", "Tablo 1")
SOLAR_CONSTANT = Constant("nominal toplam Güneş ışınımı", 1361.0, "W m^-2", 0.0, "Prsa2016", "Tablo 1")
GM_SUN = Constant("nominal Güneş kütle parametresi", 1.327_124_4e20, "m^3 s^-2", 0.0, "Prsa2016", "Tablo 1")
GM_EARTH = Constant("nominal Yer kütle parametresi", 3.986_004e14, "m^3 s^-2", 0.0, "Prsa2016", "Tablo 1")
GM_JUPITER = Constant("nominal Jüpiter kütle parametresi", 1.266_865_3e17, "m^3 s^-2", 0.0, "Prsa2016", "Tablo 1")
EARTH_EQUATORIAL_RADIUS = Constant("nominal Yer ekvator yarıçapı", 6.3781e6, "m", 0.0, "Prsa2016", "Tablo 1")
EARTH_POLAR_RADIUS = Constant("nominal Yer kutup yarıçapı", 6.3568e6, "m", 0.0, "Prsa2016", "Tablo 1")
JUPITER_EQUATORIAL_RADIUS = Constant("nominal Jüpiter ekvator yarıçapı", 7.1492e7, "m", 0.0, "Prsa2016", "Tablo 1")
JUPITER_POLAR_RADIUS = Constant("nominal Jüpiter kutup yarıçapı", 6.6854e7, "m", 0.0, "Prsa2016", "Tablo 1")

# --- IAU 2015 B2 bolometrik sıfır noktası [Mamajek2015] ------------------------------
BOLOMETRIC_ZERO_POINT_LUMINOSITY = Constant("M_bol = 0 ışınım gücü", 3.0128e28, "W", 0.0, "Mamajek2015", "L_0 tanımı")

# --- ERFA erfam.h [ERFA2023] ---------------------------------------------------------
ASTRONOMICAL_UNIT = Constant("astronomik birim", 149_597_870.7e3, "m", 0.0, "ERFA2023", "erfam.h ERFA_DAU")
DAY_SECONDS = Constant("gün", 86_400.0, "s", 0.0, "ERFA2023", "erfam.h ERFA_DAYSEC")
JULIAN_YEAR_DAYS = Constant("Jülyen yılı", 365.25, "gün", 0.0, "ERFA2023", "erfam.h ERFA_DJY")
JULIAN_CENTURY_DAYS = Constant("Jülyen yüzyılı", 36_525.0, "gün", 0.0, "ERFA2023", "erfam.h ERFA_DJC")
J2000_JD = Constant("J2000.0 Jülyen tarihi", 2_451_545.0, "gün", 0.0, "ERFA2023", "erfam.h ERFA_DJ00")
MJD_ZERO_JD = Constant("MJD sıfır noktası", 2_400_000.5, "gün", 0.0, "ERFA2023", "erfam.h ERFA_DJM0")
TT_MINUS_TAI = Constant("TT − TAI", 32.184, "s", 0.0, "ERFA2023", "erfam.h ERFA_TTMTAI")

# --- Türetilmiş sabitler -------------------------------------------------------------
# Parsek: 1 yay saniyelik paralaksa karşılık gelen uzaklık, au · 648000/π.
# Tanım IAU 2015 B2 kararının arXiv metnindeki son nottan okundu [Mamajek2015].
PARSEC = Constant(
    "parsek",
    ASTRONOMICAL_UNIT.value * 648_000.0 / math.pi,
    "m",
    0.0,
    "Mamajek2015",
    "son not: parsek tam olarak (648000/π) au",
    status="read",
    derivation="au · 648000 / pi",
)
MEGAPARSEC = Constant(
    "megaparsek",
    1e6 * PARSEC.value,
    "m",
    0.0,
    "Mamajek2015",
    "son not: parsek tam olarak (648000/π) au",
    status="read",
    derivation="1e6 · parsek",
)
JULIAN_YEAR_SECONDS = Constant(
    "Jülyen yılı",
    JULIAN_YEAR_DAYS.value * DAY_SECONDS.value,
    "s",
    0.0,
    "ERFA2023",
    "erfam.h ERFA_DJY · ERFA_DAYSEC",
    derivation="365.25 · 86400",
)
SPEED_OF_LIGHT_KM_S = SPEED_OF_LIGHT.value / 1e3


@infrastructure("modül içeriğini listeler; denklem uygulamaz")
def all_constants() -> dict[str, Constant]:
    """Modüldeki bütün ``Sabit`` nesnelerini adlarıyla döndürür."""
    return {name: nesne for name, nesne in globals().items() if isinstance(nesne, Constant)}
