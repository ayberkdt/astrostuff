"""Physical and astronomical constants.

Each constant is defined with its value, unit, uncertainty and source. The values
were read from the source texts on 25 November 2026:

- CODATA 2022: NIST ``allascii.txt`` table [CODATA2022].
- IAU 2015 B3 nominal solar and planetary values: [Prsa2016], Table 1.
- IAU 2015 B2 bolometric zero point: [Mamajek2015].
- Astronomical unit, Julian year and century, J2000: ERFA ``erfam.h`` [ERFA2023].

Derived constants (``PARSEC``, ``MEGAPARSEC`` ...) are computed from their definitions
and carry the formula in the ``derivation`` field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from kozmos._sources import infrastructure


@dataclass(frozen=True, slots=True)
class Constant:
    """A constant together with its source."""

    name: str
    value: float
    unit: str
    uncertainty: float  # 0.0: exact by definition
    source: str
    where: str
    status: str = "read"
    derivation: str | None = None

    def __float__(self) -> float:
        return self.value


# --- CODATA 2022 [CODATA2022]; NIST allascii.txt -----------------------------------
SPEED_OF_LIGHT = Constant("speed of light", 299_792_458.0, "m s^-1", 0.0, "CODATA2022", "speed of light in vacuum")
PLANCK = Constant("Planck constant", 6.626_070_15e-34, "J Hz^-1", 0.0, "CODATA2022", "Planck constant")
BOLTZMANN = Constant("Boltzmann constant", 1.380_649e-23, "J K^-1", 0.0, "CODATA2022", "Boltzmann constant")
GRAVITATIONAL_CONSTANT = Constant(
    "Newtonian constant of gravitation",
    6.674_30e-11,
    "m^3 kg^-1 s^-2",
    0.000_15e-11,
    "CODATA2022",
    "Newtonian constant of gravitation",
)
ELEMENTARY_CHARGE = Constant("elementary charge", 1.602_176_634e-19, "C", 0.0, "CODATA2022", "elementary charge")
ELECTRON_MASS = Constant("electron mass", 9.109_383_7139e-31, "kg", 0.000_000_0028e-31, "CODATA2022", "electron mass")
PROTON_MASS = Constant("proton mass", 1.672_621_925_95e-27, "kg", 0.000_000_000_52e-27, "CODATA2022", "proton mass")
ATOMIC_MASS_UNIT = Constant(
    "atomic mass constant",
    1.660_539_068_92e-27,
    "kg",
    0.000_000_000_52e-27,
    "CODATA2022",
    "atomic mass constant",
)
AVOGADRO = Constant("Avogadro constant", 6.022_140_76e23, "mol^-1", 0.0, "CODATA2022", "Avogadro constant")
THOMSON_CROSS_SECTION = Constant(
    "Thomson cross section",
    6.652_458_7051e-29,
    "m^2",
    0.000_000_0062e-29,
    "CODATA2022",
    "Thomson cross section",
)
# The NIST table gives these three constants in truncated form ("5.670 374 419..."); the full value
# is derived from the exact constants and compared with the tabulated digits in a test.
STEFAN_BOLTZMANN = Constant(
    "Stefan-Boltzmann constant",
    2 * math.pi**5 * BOLTZMANN.value**4 / (15 * PLANCK.value**3 * SPEED_OF_LIGHT.value**2),
    "W m^-2 K^-4",
    0.0,
    "CODATA2022",
    "Stefan-Boltzmann constant (5.670 374 419... e-8)",
    derivation="2 pi^5 k^4 / (15 h^3 c^2)",
    status="read",
)
WIEN_WAVELENGTH = Constant(
    "Wien wavelength displacement law constant",
    2.897_771_955e-3,
    "m K",
    0.0,
    "CODATA2022",
    "Wien wavelength displacement law constant (2.897 771 955... e-3)",
)
SECOND_RADIATION_CONSTANT = Constant(
    "second radiation constant",
    PLANCK.value * SPEED_OF_LIGHT.value / BOLTZMANN.value,
    "m K",
    0.0,
    "CODATA2022",
    "second radiation constant (1.438 776 877... e-2)",
    derivation="h c / k",
)

# --- IAU 2015 B3 nominal values [Prsa2016], Table 1 ------------------------------
SOLAR_RADIUS = Constant("nominal solar radius", 6.957e8, "m", 0.0, "Prsa2016", "Table 1")
SOLAR_LUMINOSITY = Constant("nominal solar luminosity", 3.828e26, "W", 0.0, "Prsa2016", "Table 1")
SOLAR_EFFECTIVE_TEMPERATURE = Constant("nominal solar effective temperature", 5772.0, "K", 0.0, "Prsa2016", "Table 1")
SOLAR_CONSTANT = Constant("nominal total solar irradiance", 1361.0, "W m^-2", 0.0, "Prsa2016", "Table 1")
GM_SUN = Constant("nominal solar mass parameter", 1.327_124_4e20, "m^3 s^-2", 0.0, "Prsa2016", "Table 1")
GM_EARTH = Constant("nominal Earth mass parameter", 3.986_004e14, "m^3 s^-2", 0.0, "Prsa2016", "Table 1")
GM_JUPITER = Constant("nominal Jupiter mass parameter", 1.266_865_3e17, "m^3 s^-2", 0.0, "Prsa2016", "Table 1")
EARTH_EQUATORIAL_RADIUS = Constant("nominal Earth equatorial radius", 6.3781e6, "m", 0.0, "Prsa2016", "Table 1")
EARTH_POLAR_RADIUS = Constant("nominal Earth polar radius", 6.3568e6, "m", 0.0, "Prsa2016", "Table 1")
JUPITER_EQUATORIAL_RADIUS = Constant("nominal Jupiter equatorial radius", 7.1492e7, "m", 0.0, "Prsa2016", "Table 1")
JUPITER_POLAR_RADIUS = Constant("nominal Jupiter polar radius", 6.6854e7, "m", 0.0, "Prsa2016", "Table 1")

# --- IAU 2015 B2 bolometric zero point [Mamajek2015] ------------------------------
BOLOMETRIC_ZERO_POINT_LUMINOSITY = Constant(
    "M_bol = 0 luminosity", 3.0128e28, "W", 0.0, "Mamajek2015", "L_0 definition"
)

# --- ERFA erfam.h [ERFA2023] ---------------------------------------------------------
ASTRONOMICAL_UNIT = Constant("astronomical unit", 149_597_870.7e3, "m", 0.0, "ERFA2023", "erfam.h ERFA_DAU")
DAY_SECONDS = Constant("day", 86_400.0, "s", 0.0, "ERFA2023", "erfam.h ERFA_DAYSEC")
JULIAN_YEAR_DAYS = Constant("Julian year", 365.25, "day", 0.0, "ERFA2023", "erfam.h ERFA_DJY")
JULIAN_CENTURY_DAYS = Constant("Julian century", 36_525.0, "day", 0.0, "ERFA2023", "erfam.h ERFA_DJC")
J2000_JD = Constant("J2000.0 Julian date", 2_451_545.0, "day", 0.0, "ERFA2023", "erfam.h ERFA_DJ00")
MJD_ZERO_JD = Constant("MJD zero point", 2_400_000.5, "day", 0.0, "ERFA2023", "erfam.h ERFA_DJM0")
TT_MINUS_TAI = Constant("TT − TAI", 32.184, "s", 0.0, "ERFA2023", "erfam.h ERFA_TTMTAI")

# --- Derived constants -------------------------------------------------------------
# Parsec: the distance corresponding to a parallax of 1 arcsecond, au · 648000/π.
# The definition was read from the final note in the arXiv text of IAU 2015 Resolution B2 [Mamajek2015].
PARSEC = Constant(
    "parsec",
    ASTRONOMICAL_UNIT.value * 648_000.0 / math.pi,
    "m",
    0.0,
    "Mamajek2015",
    "final note: the parsec is exactly (648000/π) au",
    status="read",
    derivation="au · 648000 / pi",
)
MEGAPARSEC = Constant(
    "megaparsec",
    1e6 * PARSEC.value,
    "m",
    0.0,
    "Mamajek2015",
    "final note: the parsec is exactly (648000/π) au",
    status="read",
    derivation="1e6 · parsec",
)
JULIAN_YEAR_SECONDS = Constant(
    "Julian year",
    JULIAN_YEAR_DAYS.value * DAY_SECONDS.value,
    "s",
    0.0,
    "ERFA2023",
    "erfam.h ERFA_DJY · ERFA_DAYSEC",
    derivation="365.25 · 86400",
)
SPEED_OF_LIGHT_KM_S = SPEED_OF_LIGHT.value / 1e3


@infrastructure("lists the module contents; implements no equation")
def all_constants() -> dict[str, Constant]:
    """Return all ``Constant`` objects in the module, keyed by name."""
    return {name: obj for name, obj in globals().items() if isinstance(obj, Constant)}
