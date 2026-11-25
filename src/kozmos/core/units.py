"""Physical-unit checking: a small dimension and unit algebra, a suffix parser and a boundary checker.

The library encodes the unit of a quantity in the identifier (``mass_kg``, ``radius_m``,
``surface_density_msun_pc2``); nothing used to check that a caller passes the right unit or that the
suffixes are used consistently. This module adds, with numpy only:

``Dimension``
    Exponents (exact rationals) of the seven SI base dimensions plus four counting dimensions.
``Unit``
    A dimension, a scale factor to the coherent SI unit and a display name. Offset-free only.
``Quantity``
    A float array together with a ``Unit``: multiplication, division, rational powers, addition and
    subtraction with dimension checking and conversion, and comparison.
``unit_from_suffix(name)``
    Maps an identifier such as ``sfr_surface_density_msun_yr_kpc2`` to a ``Unit``.
``with_units(function)``
    Wraps a plain function so that it accepts ``Quantity`` arguments, converts each to the unit named by
    the parameter suffix, raises ``UnitError`` on a dimension mismatch and returns a ``Quantity`` in the
    unit named by the function's own suffix.

Angles (this project's choice)
------------------------------
In the SI, plane and solid angle are quantities with the unit one ([SIBrochure9] Section 2.3.3, which refers to 2.3.4); Table 4
footnote (b) adds that ``rad = m/m`` "is not intrinsic and may be misleading since angle is not the same
kind of quantity as other length ratios". This module therefore gives angle its own dimension ``ANGLE``
(radian, steradian = ``ANGLE^2``). Consequences: degrees, arcminutes, arcseconds, milliarcseconds and
microarcseconds convert to one another and to radians; ``rad s^-1`` and ``Hz`` are different dimensions
(the 2 pi is not lost silently, and Table 4 footnote (d) restricts the hertz to periodic phenomena);
a length times an angle has the unit ``m rad``; ``Quantity.drop_angle()`` gives the SI view in which the
radian is the unit one. A dimensionless parameter without a suffix does not accept an angle.

Counting dimensions (this project's choice)
-------------------------------------------
The suffixes ``_pix``, ``_adu`` and ``_e`` (detector pixel, analogue-to-digital unit, electron) are
dimension one in the SI. They get their own dimensions ``PIXEL``, ``ADU`` and ``ELECTRON`` so that a gain
``gain_e_adu`` cannot be mistaken for a rate ``flux_e_s``. Reference for the symbols: [Rots2021] (``pix``,
``adu``).

Suffix grammar (derived from the identifiers that exist in the package)
-----------------------------------------------------------------------
A name is split on ``_``. The *suffix* is the maximal run of trailing tokens that are unit tokens, with at
least one stem token kept in front. A unit token is a lower-case symbol from ``SUFFIX_UNITS`` optionally
followed by an integer power written without a caret (``m2``, ``kpc2``, ``s2``). Within one suffix:

* the first unit token is a numerator and keeps its power: ``mass_kg`` -> kg, ``gm_m3_s2`` -> m^3 s^-2;
* every following token is a denominator and its power is negated: ``velocity_km_s`` -> km s^-1,
  ``surface_density_msun_pc2`` -> Msun pc^-2, ``sfr_surface_density_msun_yr_kpc2`` -> Msun yr^-1 kpc^-2,
  ``opacity_m2_kg`` -> m^2 kg^-1, ``omega_km_s_kpc`` -> km s^-1 kpc^-1, ``gain_e_adu`` -> e adu^-1;
* exception, this project's choice: when the suffix is a single token and the stem ends in
  ``number_density`` or ``column_density`` the power is negated (``number_density_m3`` -> m^-3,
  ``column_density_m2`` -> m^-2); without the exception the grammar would read them as m^3 and m^2, the
  reading of ``comoving_volume_mpc3``;
* ``X_from_Y``: the unit of the result is read from ``X`` when ``X`` carries one (``flux_jy_from_...``,
  ``pc_from_distance_modulus``); a suffix that is only a unit after ``from`` (``distance_modulus_from_pc``)
  names the source quantity, not the result; otherwise the trailing suffix of the whole name is the result
  (``luminosity_from_bolometric_magnitude_w``).

Some suffixes are recognised but cannot be represented, and raise ``UnsupportedSuffixError``: ``c``
(degree Celsius, an offset unit), ``jd`` and ``mjd`` (Julian dates are instants, also offset), ``mag`` and
``cgs`` (logarithmic quantities: magnitudes, ``log10 g`` in cgs), ``rg`` (the gravitational radius,
a unit that depends on the mass) and ``si`` (a coefficient whose SI dimension depends on a polytropic
index). ``with_units`` leaves such parameters and results unchecked and refuses a ``Quantity`` for them.

Parameters and functions without a suffix are treated as dimensionless / unchecked: a plain number passes
through unchanged, a ``Quantity`` must be dimensionless (this project's choice; a mass passed for
``efficiency`` is an error, not silently accepted). ``with_units(function, units={...}, returns=...)``
declares units for names that carry no suffix.

Scale factors
-------------
Every numerical scale factor is taken from ``kozmos.core.constants`` (CODATA 2022, IAU 2015, ERFA) or
``kozmos.core.angles`` where it exists there; the rest are definitions read in [SIBrochure9] (prefixes
Table 7, non-SI units Table 8, derived units Table 4), [IAU2012B2] (astronomical unit) and [Rots2021]
(jansky, erg, Julian year). ``UNIT_SOURCES`` records the origin of each symbol. Not modelled: temperature
offsets, logarithmic units (magnitude, dex), uncertainty propagation; see ``Validity`` of
``with_units``.
"""

from __future__ import annotations

import functools
import inspect
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.angles import ARCSEC_RAD, DEG_RAD
from kozmos.core.constants import (
    ASTRONOMICAL_UNIT,
    ATOMIC_MASS_UNIT,
    DAY_SECONDS,
    EARTH_EQUATORIAL_RADIUS,
    ELEMENTARY_CHARGE,
    GM_EARTH,
    GM_JUPITER,
    GM_SUN,
    GRAVITATIONAL_CONSTANT,
    JULIAN_YEAR_SECONDS,
    JUPITER_EQUATORIAL_RADIUS,
    PARSEC,
    SOLAR_LUMINOSITY,
    SOLAR_RADIUS,
)

Array = NDArray[np.float64]

# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class UnitError(ValueError):
    """A dimension mismatch, an unknown unit or an invalid operation on units."""


class UnitSuffixError(UnitError):
    """An identifier that cannot be read as a name with a unit suffix."""


class UnsupportedSuffixError(UnitSuffixError):
    """A recognised suffix whose unit this system cannot represent.

    ``kind`` is one of ``"offset"``, ``"epoch"``, ``"logarithmic"``, ``"context"``.
    """

    def __init__(self, message: str, token: str, kind: str) -> None:
        super().__init__(message)
        self.token = token
        self.kind = kind


# --------------------------------------------------------------------------------------
# Dimension
# --------------------------------------------------------------------------------------

#: SI base dimensions in the symbols of [SIBrochure9] Table 3, then the project's four counting dimensions.
BASE_DIMENSIONS = ("T", "L", "M", "I", "Theta", "N", "J", "ANGLE", "PIXEL", "ELECTRON", "ADU")


def _fraction(p: Any) -> Fraction:
    """Convert an exponent to an exact rational; refuse floats that are not (close to) a small rational."""
    if isinstance(p, bool):
        raise UnitError("an exponent must be an integer or a rational number")
    if isinstance(p, Fraction):
        return p
    if isinstance(p, int | np.integer):
        return Fraction(int(p))
    if isinstance(p, float | np.floating):
        f = Fraction(float(p)).limit_denominator(64)
        if abs(float(f) - float(p)) > 1e-12 * max(1.0, abs(float(p))):
            raise UnitError(f"exponent {p!r} is not a rational number with denominator <= 64")
        return f
    raise UnitError(f"an exponent must be an integer or a rational number, got {type(p).__name__}")


@dataclass(frozen=True, slots=True)
class Dimension:
    """Exponents (exact rationals) of the base dimensions in ``BASE_DIMENSIONS``."""

    exponents: tuple[Fraction, ...] = (Fraction(0),) * len(BASE_DIMENSIONS)

    def __post_init__(self) -> None:
        if len(self.exponents) != len(BASE_DIMENSIONS):
            raise UnitError(f"a dimension needs {len(BASE_DIMENSIONS)} exponents")
        object.__setattr__(self, "exponents", tuple(_fraction(e) for e in self.exponents))

    @classmethod
    def of(cls, **exponents: Any) -> Dimension:
        """Build a dimension from named exponents, e.g. ``Dimension.of(L=1, T=-1)``."""
        unknown = set(exponents) - set(BASE_DIMENSIONS)
        if unknown:
            raise UnitError(f"unknown base dimension(s): {sorted(unknown)}")
        return cls(tuple(_fraction(exponents.get(name, 0)) for name in BASE_DIMENSIONS))

    def is_dimensionless(self) -> bool:
        """True when every exponent is zero (the angle exponent included)."""
        return not any(self.exponents)

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension(tuple(a + b for a, b in zip(self.exponents, other.exponents, strict=True)))

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension(tuple(a - b for a, b in zip(self.exponents, other.exponents, strict=True)))

    def __pow__(self, p: Any) -> Dimension:
        q = _fraction(p)
        return Dimension(tuple(a * q for a in self.exponents))

    def __str__(self) -> str:
        parts = []
        for name, e in zip(BASE_DIMENSIONS, self.exponents, strict=True):
            if e == 1:
                parts.append(name)
            elif e:
                parts.append(f"{name}^{e}")
        return " ".join(parts) if parts else "1"


_DIMENSIONLESS = Dimension()
_SI_SYMBOL = {
    "T": "s",
    "L": "m",
    "M": "kg",
    "I": "A",
    "Theta": "K",
    "N": "mol",
    "J": "cd",
    "ANGLE": "rad",
    "PIXEL": "pix",
    "ELECTRON": "e",
    "ADU": "adu",
}


# --------------------------------------------------------------------------------------
# Unit
# --------------------------------------------------------------------------------------

_Parts = tuple[tuple[str, Fraction], ...]


def _merge_parts(a: _Parts, b: _Parts, sign: int) -> _Parts:
    merged: dict[str, Fraction] = dict(a)
    for symbol, e in b:
        merged[symbol] = merged.get(symbol, Fraction(0)) + sign * e
    return tuple((s, e) for s, e in merged.items() if e != 0)


def _is_number(x: Any) -> bool:
    return isinstance(x, int | float | np.number | np.ndarray) and not isinstance(x, bool)


@dataclass(frozen=True, slots=True, eq=False)
class Unit:
    """A dimension and the scale factor to the coherent SI unit (``value_in_SI = value * scale``).

    ``parts`` is the display name as (symbol, exponent) pairs; it does not take part in comparison.
    Offset-free only: there is no unit with a zero-point shift (no degree Celsius).
    """

    dimension: Dimension
    scale: float
    parts: _Parts = ()

    def __post_init__(self) -> None:
        if not (math.isfinite(self.scale) and self.scale > 0.0):
            raise UnitError("the scale factor of a unit must be finite and positive")
        object.__setattr__(self, "scale", float(self.scale))

    # -- comparison -------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            return NotImplemented
        return self.dimension == other.dimension and math.isclose(self.scale, other.scale, rel_tol=1e-14, abs_tol=0.0)

    def __hash__(self) -> int:
        return hash(self.dimension)

    # -- algebra ----------------------------------------------------------------------
    def __mul__(self, other: Any) -> Any:
        if isinstance(other, Unit):
            return Unit(
                self.dimension * other.dimension, self.scale * other.scale, _merge_parts(self.parts, other.parts, 1)
            )
        if _is_number(other):
            return Quantity(other, self)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Any:
        if isinstance(other, Unit):
            return Unit(
                self.dimension / other.dimension, self.scale / other.scale, _merge_parts(self.parts, other.parts, -1)
            )
        return NotImplemented

    def __pow__(self, p: Any) -> Unit:
        q = _fraction(p)
        return Unit(self.dimension**q, self.scale ** float(q), tuple((s, e * q) for s, e in self.parts if e * q != 0))

    # -- conversion -------------------------------------------------------------------
    def is_convertible_to(self, other: Unit) -> bool:
        """True when both units have the same dimension."""
        return self.dimension == other.dimension

    def conversion_numerator_denominator(self, other: Unit) -> tuple[float, float]:
        """Return ``(self.scale, other.scale)`` after checking that the dimensions agree."""
        if self.dimension != other.dimension:
            raise UnitError(
                f"cannot convert '{self}' ({self.dimension}) to '{other}' ({other.dimension}): dimensions differ"
            )
        return self.scale, other.scale

    @classmethod
    def parse(cls, expression: str) -> Unit:
        """Parse a unit expression such as ``"kg m^-3"``, ``"km/s"``, ``"Msun pc^-2"`` or ``"W m-2 Hz-1"``.

        Symbols are case sensitive and follow [SIBrochure9]: an optional prefix (Table 7) and a unit symbol.
        Products are written with a space, ``*`` or a centred dot; at most one solidus is allowed and
        everything after it is a denominator ([SIBrochure9] Section 5.2). Powers are written ``^-3``,
        ``^{-4/3}``, as trailing digits (``m3 s^-2``) or as Unicode superscripts (``m³ s⁻²``).
        ``u`` and the Greek mu are accepted for the micro prefix. A degree Celsius is refused (offset unit).
        """
        return _parse_expression(expression)

    def __str__(self) -> str:
        if not self.parts:
            return "1"
        return " ".join(s if e == 1 else f"{s}^{e}" for s, e in self.parts)

    def __repr__(self) -> str:
        return f"Unit('{self}', dimension='{self.dimension}', scale={self.scale!r})"


ONE = Unit(_DIMENSIONLESS, 1.0)


def _named(dimension: Dimension, scale: float, symbol: str) -> Unit:
    return Unit(dimension, scale, ((symbol, Fraction(1)),))


# -- registry ---------------------------------------------------------------------------

#: SI prefixes, [SIBrochure9] Table 7 (``u`` and the Greek mu are accepted spellings of micro).
SI_PREFIXES: dict[str, int] = {
    "Q": 30,
    "R": 27,
    "Y": 24,
    "Z": 21,
    "E": 18,
    "P": 15,
    "T": 12,
    "G": 9,
    "M": 6,
    "k": 3,
    "h": 2,
    "da": 1,
    "d": -1,
    "c": -2,
    "m": -3,
    "µ": -6,
    "μ": -6,
    "u": -6,
    "n": -9,
    "p": -12,
    "f": -15,
    "a": -18,
    "z": -21,
    "y": -24,
    "r": -27,
    "q": -30,
}

_T, _L, _M, _I, _TH, _N, _J, _ANG, _PIX, _E, _ADU = (Dimension.of(**{n: 1}) for n in BASE_DIMENSIONS)

_REGISTRY: dict[str, tuple[Unit, bool]] = {}  # symbol -> (unit, accepts an SI prefix)
UNIT_SOURCES: dict[str, str] = {}  # symbol -> where the definition / scale factor comes from


def _register(symbols: tuple[str, ...], unit: Unit, prefixable: bool, origin: str) -> None:
    named = _named(unit.dimension, unit.scale, symbols[0])
    for s in symbols:
        _REGISTRY[s] = (named, prefixable)
        UNIT_SOURCES[s] = origin


def _build_registry() -> None:
    si2 = "SIBrochure9 Table 2"
    si4 = "SIBrochure9 Table 4"
    si8 = "SIBrochure9 Table 8"
    _register(("m",), Unit(_L, 1.0), True, si2)
    _register(("g",), Unit(_M, 1e-3), True, si2 + " (gram; the kilogram is the base unit, Section 3)")
    _register(("kg",), Unit(_M, 1.0), False, si2)
    _register(("s",), Unit(_T, 1.0), True, si2)
    _register(("A",), Unit(_I, 1.0), True, si2)
    _register(("K",), Unit(_TH, 1.0), True, si2)
    _register(("mol",), Unit(_N, 1.0), True, si2)
    _register(("cd",), Unit(_J, 1.0), True, si2)

    rad = Unit(_ANG, 1.0)
    sr = Unit(_ANG**2, 1.0)
    m, kg, s, amp = Unit(_L, 1.0), Unit(_M, 1.0), Unit(_T, 1.0), Unit(_I, 1.0)
    cd = Unit(_J, 1.0)
    _register(("rad",), rad, True, si4 + " (this project: own dimension ANGLE)")
    _register(("sr",), sr, True, si4 + " (this project: ANGLE^2)")
    hz = ONE / s
    newton = kg * m / s**2
    pascal = newton / m**2
    joule = newton * m
    watt = joule / s
    coulomb = amp * s
    volt = watt / amp
    _register(("Hz",), hz, True, si4)
    _register(("N",), newton, True, si4)
    _register(("Pa",), pascal, True, si4)
    _register(("J",), joule, True, si4)
    _register(("W",), watt, True, si4)
    _register(("C",), coulomb, True, si4)
    _register(("V",), volt, True, si4)
    _register(("F",), coulomb / volt, True, si4)
    _register(("ohm", "Ω"), volt / amp, True, si4)
    _register(("S",), amp / volt, True, si4)
    _register(("Wb",), volt * s, True, si4)
    _register(("T",), volt * s / m**2, True, si4)
    _register(("H",), volt * s / amp, True, si4)
    _register(("lm",), cd * sr, True, si4)
    _register(("lx",), cd * sr / m**2, True, si4)

    # Table 8 (non-SI units accepted for use with the SI)
    _register(("min",), Unit(_T, 60.0), False, si8)
    _register(("h",), Unit(_T, 3600.0), False, si8)
    _register(("d",), Unit(_T, DAY_SECONDS.value), False, "constants.DAY_SECONDS; " + si8 + " (86 400 s)")
    _register(("deg", "°"), Unit(_ANG, DEG_RAD), False, "angles.DEG_RAD (ERFA); " + si8 + " (pi/180 rad)")
    _register(("arcmin", "′"), Unit(_ANG, DEG_RAD / 60.0), False, "angles.DEG_RAD / 60; " + si8 + " (pi/10 800 rad)")
    _register(
        ("arcsec", "as", "″"), Unit(_ANG, ARCSEC_RAD), True, "angles.ARCSEC_RAD (ERFA); " + si8 + " (pi/648 000 rad)"
    )
    _register(("angstrom", "Å"), Unit(_L, 1e-10), False, si8 + " (0.1 nm)")
    _register(("bar",), Unit(_M / _L / _T**2, 1e5), False, si8 + " (0.1 MPa)")
    _register(("Da",), Unit(_M, ATOMIC_MASS_UNIT.value), True, "constants.ATOMIC_MASS_UNIT; " + si8 + " footnote (i)")
    _register(("eV",), Unit(_M * _L**2 / _T**2, ELEMENTARY_CHARGE.value), True, "constants.ELEMENTARY_CHARGE; " + si8)
    _register(("erg",), Unit(_M * _L**2 / _T**2, 1e-7), True, "Rots2021 (erg = 1e-7 J)")
    _register(("au",), Unit(_L, ASTRONOMICAL_UNIT.value), False, "constants.ASTRONOMICAL_UNIT; IAU2012B2; " + si8)

    # astronomical units
    _register(
        ("pc",), Unit(_L, PARSEC.value), True, "constants.PARSEC = au 648000/pi (IAU 2015 B2 final note, Mamajek2015)"
    )
    _register(("yr",), Unit(_T, JULIAN_YEAR_SECONDS.value), True, "constants.JULIAN_YEAR_SECONDS (365.25 d); Rots2021")
    solar_mass = GM_SUN.value / GRAVITATIONAL_CONSTANT.value
    _register(
        ("Msun", "M☉", "M_⊙"),
        Unit(_M, solar_mass),
        False,
        "constants.GM_SUN / constants.GRAVITATIONAL_CONSTANT (as accretion.py and laws.py do)",
    )
    _register(("Rsun", "R☉", "R_⊙"), Unit(_L, SOLAR_RADIUS.value), False, "constants.SOLAR_RADIUS (IAU 2015 B3)")
    _register(
        ("Lsun", "L☉", "L_⊙"),
        Unit(_M * _L**2 / _T**3, SOLAR_LUMINOSITY.value),
        False,
        "constants.SOLAR_LUMINOSITY (IAU 2015 B3)",
    )
    _register(("Mearth",), Unit(_M, GM_EARTH.value / GRAVITATIONAL_CONSTANT.value), False, "constants.GM_EARTH / G")
    _register(("Mjup",), Unit(_M, GM_JUPITER.value / GRAVITATIONAL_CONSTANT.value), False, "constants.GM_JUPITER / G")
    _register(("Rearth",), Unit(_L, EARTH_EQUATORIAL_RADIUS.value), False, "constants.EARTH_EQUATORIAL_RADIUS")
    _register(("Rjup",), Unit(_L, JUPITER_EQUATORIAL_RADIUS.value), False, "constants.JUPITER_EQUATORIAL_RADIUS")
    jansky = Unit(_M / _T**2, 1e-26)  # W m^-2 Hz^-1 = kg s^-2
    _register(("Jy",), jansky, True, "Rots2021 Section 3.3 and Table 5 (1e-26 W m^-2 Hz^-1)")

    # counting units (this project's dimensions) and the plain percent
    _register(("pix", "pixel"), Unit(_PIX, 1.0), False, "Rots2021 (pixel); this project: own dimension PIXEL")
    _register(("adu", "ADU"), Unit(_ADU, 1.0), False, "Rots2021 (adu); this project: own dimension ADU")
    _register(("e",), Unit(_E, 1.0), False, "this project: electron count, own dimension ELECTRON")
    _register(("%", "percent"), Unit(_DIMENSIONLESS, 0.01), False, "Rots2021 (percent)")


_build_registry()


def _lookup_symbol(symbol: str) -> Unit | None:
    """Resolve a case-sensitive symbol: exact registry entry first, then prefix + prefixable symbol."""
    if symbol in _REGISTRY:
        return _REGISTRY[symbol][0]
    for prefix in sorted(SI_PREFIXES, key=len, reverse=True):
        if symbol.startswith(prefix) and len(symbol) > len(prefix):
            rest = symbol[len(prefix) :]
            if rest in _REGISTRY and _REGISTRY[rest][1]:
                base = _REGISTRY[rest][0]
                factor = float(f"1e{SI_PREFIXES[prefix]}")
                return Unit(base.dimension, factor * base.scale, ((symbol, Fraction(1)),))
    return None


_SUPERSCRIPT = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")
_SUPERSCRIPT_RUN = re.compile("[⁻⁰¹²³⁴-⁹]+")
_BRACED = re.compile(r"\{[^}]*\}")
_OFFSET_SPELLINGS = ("°C", "degC", "deg_C", "Celsius")


def _parse_power(text: str, token: str) -> Fraction:
    body = text[1:-1] if text.startswith("{") and text.endswith("}") else text
    try:
        if "/" in body:
            num, den = body.split("/")
            return Fraction(int(num), int(den))
        return Fraction(int(body))
    except (ValueError, ZeroDivisionError):
        raise UnitError(f"cannot read the power '{text}' in '{token}'") from None


def _parse_token(token: str) -> Unit:
    if token == "1":
        return ONE
    symbol, power = token, Fraction(1)
    if "^" in token:
        symbol, _, ptext = token.partition("^")
        power = _parse_power(ptext, token)
    unit = _lookup_symbol(symbol)
    if unit is None and "^" not in token:
        match = re.fullmatch(r"(.*?[^\d\s])(\d+)", token)
        if match:
            unit = _lookup_symbol(match.group(1))
            power = Fraction(int(match.group(2)))
    if unit is None:
        raise UnitError(f"unknown unit symbol in '{token}'")
    return unit**power


@functools.lru_cache(maxsize=2048)
def _parse_expression(expression: str) -> Unit:
    if not isinstance(expression, str) or not expression.strip():
        raise UnitError("a unit expression must be a non-empty string")
    if any(s in expression for s in _OFFSET_SPELLINGS):
        raise UnitError(
            "the degree Celsius is an offset unit; this system has no offset units (use K for a temperature "
            "interval, convert Celsius temperatures yourself)"
        )
    text = _SUPERSCRIPT_RUN.sub(lambda m: "^" + m.group(0).translate(_SUPERSCRIPT), expression)
    text = re.sub(r"e⁻|e\^-(?![\d{])", "e", text).replace("⋅", " ").replace("·", " ").replace("*", " ")
    # protect slashes inside braced powers
    protected: list[str] = []

    def keep(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\u0000{len(protected) - 1}\u0000"

    text = _BRACED.sub(keep, text)
    halves = text.split("/")
    if len(halves) > 2:
        raise UnitError(f"'{expression}': more than one solidus; use negative exponents or one solidus")

    def restore(t: str) -> str:
        return re.sub("\u0000(\\d+)\u0000", lambda m: protected[int(m.group(1))], t)

    result = ONE
    for i, half in enumerate(halves):
        tokens = restore(half).split()
        if not tokens:
            raise UnitError(f"'{expression}': empty numerator or denominator")
        for token in tokens:
            unit = _parse_token(token)
            result = result * unit if i == 0 else result / unit
    return result


# --------------------------------------------------------------------------------------
# Quantity
# --------------------------------------------------------------------------------------


def _as_unit(unit: Unit | str) -> Unit:
    if isinstance(unit, Unit):
        return unit
    if isinstance(unit, str):
        return Unit.parse(unit)
    raise UnitError(f"expected a Unit or a unit expression, got {type(unit).__name__}")


class Quantity:
    """A float array together with a ``Unit``.

    Addition and subtraction need equal dimensions and convert the right operand to the unit of the left
    one. Multiplication and division combine the units; a dimensionless result is returned as a plain
    unit-one quantity. ``**`` accepts integers and rationals (``0.5``, ``Fraction(1, 3)``). ``==`` is
    exact after conversion (no tolerance; use ``isclose``); ordering comparisons need equal dimensions.
    NumPy functions are not supported (``__array_ufunc__ = None``): take ``to_value`` first.
    """

    __slots__ = ("_unit", "_value")
    __array_priority__ = 1000
    __array_ufunc__ = None
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, value: ArrayLike | Quantity, unit: Unit | str = ONE) -> None:
        target = _as_unit(unit)
        if isinstance(value, Quantity):
            self._value = value.to_value(target)
        else:
            self._value = np.asarray(value, dtype=float)
        self._unit = target

    # -- accessors --------------------------------------------------------------------
    @property
    def value(self) -> Any:
        """The numerical value in ``self.unit`` (a numpy scalar for a 0-d quantity)."""
        return self._value[()] if self._value.ndim == 0 else self._value

    @property
    def unit(self) -> Unit:
        return self._unit

    @property
    def dimension(self) -> Dimension:
        return self._unit.dimension

    @property
    def shape(self) -> tuple[int, ...]:
        return self._value.shape

    def __len__(self) -> int:
        return len(self._value)

    def __getitem__(self, index: Any) -> Quantity:
        return Quantity(self._value[index], self._unit)

    def __repr__(self) -> str:
        return f"Quantity({self._value.tolist()!r}, '{self._unit}')"

    # -- conversion -------------------------------------------------------------------
    def to_value(self, unit: Unit | str) -> Any:
        """Return the value in ``unit``; raises ``UnitError`` when the dimensions differ."""
        target = _as_unit(unit)
        s_from, s_to = self._unit.conversion_numerator_denominator(target)
        out = self._value if s_from == s_to else self._value * s_from / s_to
        return out[()] if out.ndim == 0 else out

    def to(self, unit: Unit | str) -> Quantity:
        """Return the quantity converted to ``unit``."""
        target = _as_unit(unit)
        return Quantity(self.to_value(target), target)

    def drop_angle(self) -> Quantity:
        """Return the SI view in which the radian is the unit one (the angle exponent becomes zero).

        The value is converted to the coherent SI unit first, so ``Quantity(1, "deg").drop_angle()`` is the
        dimensionless 0.017453... and ``Quantity(2, "rad s^-1").drop_angle()`` is ``2 s^-1``.
        """
        index = BASE_DIMENSIONS.index("ANGLE")
        if self._unit.dimension.exponents[index] == 0:
            return self
        exps = list(self._unit.dimension.exponents)
        exps[index] = Fraction(0)
        dimension = Dimension(tuple(exps))
        parts = tuple((_SI_SYMBOL[n], e) for n, e in zip(BASE_DIMENSIONS, exps, strict=True) if e != 0)
        return Quantity(self._value * self._unit.scale, Unit(dimension, 1.0, parts))

    def isclose(self, other: Quantity, rtol: float = 1e-9, atol: float = 0.0) -> Any:
        """Element-wise ``np.isclose`` after converting ``other`` to this unit (tolerance in this unit)."""
        return np.isclose(self._value, other.to_value(self._unit), rtol=rtol, atol=atol)

    # -- arithmetic -------------------------------------------------------------------
    def _plain_operand(self, other: Any, op: str) -> Array:
        """A plain number is acceptable only next to a dimensionless quantity."""
        if not self._unit.dimension.is_dimensionless():
            raise UnitError(f"cannot {op} a plain number and a quantity with unit '{self._unit}' ({self.dimension})")
        return np.asarray(other, dtype=float) / self._unit.scale

    def _add(self, other: Any, sign: float, op: str) -> Quantity:
        if isinstance(other, Quantity):
            return Quantity(self._value + sign * other.to_value(self._unit), self._unit)
        if _is_number(other):
            return Quantity(self._value + sign * self._plain_operand(other, op), self._unit)
        return NotImplemented  # type: ignore[return-value]

    def __add__(self, other: Any) -> Quantity:
        return self._add(other, 1.0, "add")

    __radd__ = __add__

    def __sub__(self, other: Any) -> Quantity:
        return self._add(other, -1.0, "subtract")

    def __rsub__(self, other: Any) -> Quantity:
        return (-self)._add(other, 1.0, "subtract")

    def __neg__(self) -> Quantity:
        return Quantity(-self._value, self._unit)

    def __abs__(self) -> Quantity:
        return Quantity(np.abs(self._value), self._unit)

    @staticmethod
    def _finish(value: Array, unit: Unit) -> Quantity:
        if unit.dimension.is_dimensionless() and unit.scale != 1.0:
            return Quantity(value * unit.scale, ONE)
        if unit.dimension.is_dimensionless():
            return Quantity(value, ONE)
        return Quantity(value, unit)

    def __mul__(self, other: Any) -> Quantity:
        if isinstance(other, Quantity):
            return self._finish(self._value * other._value, self._unit * other._unit)
        if isinstance(other, Unit):
            return self._finish(self._value, self._unit * other)
        if _is_number(other):
            return Quantity(self._value * np.asarray(other, dtype=float), self._unit)
        return NotImplemented  # type: ignore[return-value]

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Quantity:
        if isinstance(other, Quantity):
            return self._finish(self._value / other._value, self._unit / other._unit)
        if isinstance(other, Unit):
            return self._finish(self._value, self._unit / other)
        if _is_number(other):
            return Quantity(self._value / np.asarray(other, dtype=float), self._unit)
        return NotImplemented  # type: ignore[return-value]

    def __rtruediv__(self, other: Any) -> Quantity:
        if _is_number(other):
            return self._finish(np.asarray(other, dtype=float) / self._value, ONE / self._unit)
        return NotImplemented  # type: ignore[return-value]

    def __pow__(self, p: Any) -> Quantity:
        q = _fraction(p)
        return self._finish(self._value ** float(q), self._unit**q)

    def sqrt(self) -> Quantity:
        """Square root (exponent 1/2 of the dimension)."""
        return self ** Fraction(1, 2)

    # -- comparison -------------------------------------------------------------------
    def _comparable(self, other: Any) -> Any:
        if isinstance(other, Quantity):
            return other.to_value(self._unit)
        if _is_number(other):
            return self._plain_operand(other, "compare")
        return None

    def __eq__(self, other: object) -> Any:
        if isinstance(other, Quantity) and other.dimension != self.dimension:
            return False
        if not isinstance(other, Quantity) and not _is_number(other):
            return NotImplemented
        if not isinstance(other, Quantity) and not self._unit.dimension.is_dimensionless():
            return False
        return self._value == self._comparable(other)

    def __ne__(self, other: object) -> Any:
        result = self.__eq__(other)
        return result if result is NotImplemented else np.logical_not(result)

    def _order(self, other: Any, op: Callable[[Any, Any], Any]) -> Any:
        rhs = self._comparable(other)
        if rhs is None:
            return NotImplemented
        return op(self._value, rhs)

    def __lt__(self, other: Any) -> Any:
        return self._order(other, np.less)

    def __le__(self, other: Any) -> Any:
        return self._order(other, np.less_equal)

    def __gt__(self, other: Any) -> Any:
        return self._order(other, np.greater)

    def __ge__(self, other: Any) -> Any:
        return self._order(other, np.greater_equal)


# --------------------------------------------------------------------------------------
# Suffix grammar
# --------------------------------------------------------------------------------------

#: Lower-case suffix tokens -> unit expression. Derived from the identifiers that exist in the package;
#: ``mpc`` is the megaparsec, ``mas`` the milliarcsecond, ``myr`` the megayear, ``ms`` the millisecond.
SUFFIX_UNITS: dict[str, str] = {
    "kg": "kg",
    "g": "g",
    "m": "m",
    "km": "km",
    "cm": "cm",
    "mm": "mm",
    "um": "um",
    "nm": "nm",
    "angstrom": "angstrom",
    "au": "au",
    "pc": "pc",
    "kpc": "kpc",
    "mpc": "Mpc",
    "gpc": "Gpc",
    "rsun": "Rsun",
    "lsun": "Lsun",
    "msun": "Msun",
    "mearth": "Mearth",
    "mjup": "Mjup",
    "rearth": "Rearth",
    "rjup": "Rjup",
    "s": "s",
    "ms": "ms",
    "hr": "h",
    "day": "d",
    "yr": "yr",
    "kyr": "kyr",
    "myr": "Myr",
    "gyr": "Gyr",
    "hz": "Hz",
    "k": "K",
    "jy": "Jy",
    "w": "W",
    "j": "J",
    "erg": "erg",
    "ev": "eV",
    "pa": "Pa",
    "hpa": "hPa",
    "t": "T",
    "wb": "Wb",
    "rad": "rad",
    "sr": "sr",
    "deg": "deg",
    "arcmin": "arcmin",
    "arcsec": "arcsec",
    "mas": "mas",
    "uas": "uas",
    "pix": "pix",
    "adu": "adu",
    "e": "e",
    "percent": "%",
}

#: Recognised suffix tokens that this system cannot represent: token -> (kind, reason).
UNSUPPORTED_SUFFIXES: dict[str, tuple[str, str]] = {
    "c": ("offset", "degree Celsius is an offset unit (zero point shifted); no offset units in this system"),
    "jd": ("epoch", "a Julian date is an instant (zero point shifted); no offset units in this system"),
    "mjd": ("epoch", "a modified Julian date is an instant (zero point shifted); no offset units in this system"),
    "mag": ("logarithmic", "magnitudes are logarithmic quantities; no logarithmic units in this system"),
    "cgs": ("logarithmic", "log10 of a quantity in cgs units (dex); no logarithmic units in this system"),
    "rg": ("context", "the gravitational radius G M / c^2 depends on the mass; it is not a fixed unit"),
    "si": ("context", "a coefficient in SI units whose dimension depends on another argument"),
}

#: Stems after which a single positive power in a suffix means a reciprocal (this project's choice).
IMPLICIT_INVERSE_STEMS = ("number_density", "column_density")

_POWER_TOKEN = re.compile(r"([a-z]+)([0-9]+)?")
_SUFFIX_CACHE: dict[str, Unit] = {}


def _token_kind(token: str) -> str | None:
    if token in UNSUPPORTED_SUFFIXES:
        return "unsupported"
    match = _POWER_TOKEN.fullmatch(token)
    if match and match.group(1) in SUFFIX_UNITS:
        return "unit"
    return None


def _trailing_run(tokens: list[str]) -> int:
    """Index where the maximal trailing run of unit tokens starts (at least one stem token is kept)."""
    i = len(tokens)
    while i > 1 and _token_kind(tokens[i - 1]):
        i -= 1
    return i


def _suffix_tokens(name: str) -> tuple[list[str], list[str]]:
    """Split ``name`` into (stem tokens, suffix tokens); the suffix is empty when there is none."""
    if not isinstance(name, str) or not name or name != name.strip() or " " in name:
        raise UnitSuffixError(f"not a valid identifier: {name!r}")
    tokens = name.split("_")
    if "from" in tokens[1:] or tokens[0] == "from":
        k = tokens.index("from")
        left, right = tokens[:k], tokens[k + 1 :]
        right_is_unit_only = bool(right) and all(_token_kind(t) for t in right)
        if right_is_unit_only:  # the unit after "from" names the source quantity, not the result
            tokens = left
        else:
            i = _trailing_run(tokens)
            if i < len(tokens):
                return tokens[:i], tokens[i:]
            if len(left) == 1 and _token_kind(left[0]):  # pc_from_distance_modulus
                return [], left
            tokens = left
    i = _trailing_run(tokens)
    return tokens[:i], tokens[i:]


@source(
    "SIBrochure9",
    where="Table 7 (SI prefixes, p. 138), Table 8 (non-SI units, pp. 140-141), Section 5.2 (unit symbols, p. 142)",
    status="read",
    note="Unit symbols and prefixes follow the Brochure. The suffix grammar (first token numerator, following tokens "
    "denominators, the number_density/column_density exception, the 'from' rule, the table of lower-case tokens) is "
    "this project's choice, derived from the identifiers that exist in the package, not from the Brochure.",
)
def unit_from_suffix(name: str) -> Unit | None:
    """Return the unit named by the suffix of an identifier, or ``None`` when it has no unit suffix.

    ``mass_kg`` -> kg; ``velocity_km_s`` -> km s^-1; ``gm_m3_s2`` -> m^3 s^-2;
    ``surface_density_msun_pc2`` -> Msun pc^-2; ``number_density_m3`` -> m^-3.

    Grammar: the suffix is the maximal trailing run of unit tokens (``SUFFIX_UNITS``, optional integer power
    glued to the symbol) behind at least one stem token; the first token is a numerator, the following ones
    are denominators; see the module docstring for the ``number_density`` exception and the ``from`` rule.

    Parameters
    ----------
    name : str
        A parameter or function name.

    Returns
    -------
    Unit or None
        The unit, or ``None`` when the name carries no unit suffix (dimensionless / unchecked).

    Raises
    ------
    UnsupportedSuffixError
        The suffix is recognised but cannot be represented (``c``, ``jd``, ``mjd``, ``mag``, ``cgs``,
        ``rg``, ``si``).
    UnitSuffixError
        ``name`` is not a valid identifier.

    References
    ----------
    [SIBrochure9] Table 7, Table 8, Section 5.2 (symbols); the grammar is this project's choice.

    Validity
    --------
    Lexical only: the grammar cannot know that a token is a unit in this position and not a word. It was
    checked against every parameter and function name of the package (tests/unit/test_unit_conventions.py);
    names it cannot explain are listed there. ``yr`` is the Julian year of 365.25 d (not a calendar year)
    and is also used for epochs (``epoch_yr``), which are instants.

    Verification
    ------------
    tests/unit/test_units.py: the documented examples; tests/unit/test_unit_conventions.py: every name of
    the package.
    """
    cached = _SUFFIX_CACHE.get(name)
    if cached is not None:
        return cached
    stem, tokens = _suffix_tokens(name)
    if not tokens:
        return None
    for token in tokens:
        if token in UNSUPPORTED_SUFFIXES:
            kind, reason = UNSUPPORTED_SUFFIXES[token]
            raise UnsupportedSuffixError(f"suffix '_{token}' of '{name}' is not supported: {reason}", token, kind)
    inverse = len(tokens) == 1 and "_".join(stem).endswith(IMPLICIT_INVERSE_STEMS)
    unit = ONE
    for position, token in enumerate(tokens):
        match = _POWER_TOKEN.fullmatch(token)
        assert match is not None
        factor = Unit.parse(SUFFIX_UNITS[match.group(1)]) ** (int(match.group(2)) if match.group(2) else 1)
        unit = unit * (ONE / factor if position > 0 or inverse else factor)
    _SUFFIX_CACHE[name] = unit
    return unit


# --------------------------------------------------------------------------------------
# Boundary checker
# --------------------------------------------------------------------------------------


def _unit_or_flag(name: str) -> tuple[Unit | None, UnsupportedSuffixError | None]:
    try:
        return unit_from_suffix(name), None
    except UnsupportedSuffixError as error:
        return None, error


@source(
    "SIBrochure9",
    where="Section 2.3.3 and Table 3 (dimensions of quantities, p. 132), Section 5.2 (p. 142)",
    status="read",
    note="Dimension checking follows the dimension algebra of Section 2.3.3; the wrapper, the rule that an unsuffixed "
    "parameter accepts only a dimensionless Quantity and the 'units' / 'returns' overrides are this project's choice.",
)
def with_units[F: Callable[..., Any]](
    function: F, *, units: Mapping[str, Unit | str] | None = None, returns: Unit | str | None = None
) -> F:
    """Wrap a plain function so that it accepts and returns ``Quantity`` objects.

    Each ``Quantity`` argument is converted to the unit named by the parameter suffix before the plain
    function is called; a dimension mismatch raises ``UnitError`` naming the function, the parameter and
    both dimensions. Plain numbers and arrays pass through unchanged. The result is returned as a
    ``Quantity`` in the unit named by the function's own suffix (each element of a tuple result gets it);
    a function without a unit suffix returns its plain result unchanged.

    Parameters
    ----------
    function : callable
        A function whose parameter names carry unit suffixes.
    units : mapping of str to Unit or str, optional
        Units for parameters that carry no (or an unsupported) suffix, e.g. ``{"m_low": "Msun"}``.
    returns : Unit or str, optional
        Unit of the result when the function name carries no (or an unsupported) suffix.

    Returns
    -------
    callable
        The wrapper, with the attributes ``parameter_units`` (name -> Unit or None) and ``return_unit``.

    References
    ----------
    [SIBrochure9] Section 2.3.3 (dimensions); the wrapper is this project's design.

    Validity
    --------
    A parameter or function without a unit suffix is dimensionless / unchecked: a plain number passes, a
    ``Quantity`` must be dimensionless (an angle is not) or the call raises. Parameters with an unsupported
    suffix (``_c``, ``_jd``, ``_mag``, ``_cgs``, ``_rg``, ``_si``) refuse a ``Quantity``. The suffix says
    what the function expects, not what it computes: a name that contradicts its docstring is wrapped
    wrongly (see tests/unit/test_unit_conventions.py: ``time_above_horizon_hr`` is in sidereal hours,
    ``comoving_volume_element_mpc3`` is per steradian, ``sfr_calibration_msun_yr`` takes W/Hz for one band).
    Not modelled: temperature offsets (no degree Celsius), logarithmic units (magnitude, dex), uncertainty
    propagation, ``Quantity`` inside containers, defaults (they stay plain numbers), NumPy ufuncs on a
    ``Quantity``. Cost, measured on Python 3.12.1 / numpy 2.2.6 (Windows 11, median of 7 repeats of 20 000
    calls): ``schwarzschild_radius_m`` takes 9.2 us plain, 18.5 us wrapped with a plain argument (+9 us, of
    which ``inspect.Signature.bind`` alone is 3.2 us; the rest was not itemised) and 29.8 us wrapped with one
    ``Quantity`` argument (+21 us, including the result ``Quantity``); two ``Quantity`` arguments cost +27 us
    for ``luminosity_w``. For an array of 200 000 elements the plain call takes 2.3 ms and the wrapped call
    with a conversion 3.9 ms; the conversion multiplies and then divides, 1.4 ms for the two passes. Use the
    wrapper at the boundary, not inside a loop.

    Verification
    ------------
    tests/unit/test_units.py: wrapper behaviour and argument errors; the same file runs the checker on 53
    functions of black_holes, stars, star_formation, cosmology, radiation and ephemeris with inputs in
    other units and with swapped arguments.
    """
    signature = inspect.signature(function)
    overrides = {k: _as_unit(v) for k, v in (units or {}).items()}
    unknown = set(overrides) - set(signature.parameters)
    if unknown:
        raise UnitError(f"'units' names parameters that {function.__name__} does not have: {sorted(unknown)}")

    plans: dict[str, tuple[Unit | None, UnsupportedSuffixError | None]] = {}
    for name, parameter in signature.parameters.items():
        if name in overrides:
            plans[name] = (overrides[name], None)
        elif parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            plans[name] = (None, None)
        else:
            plans[name] = _unit_or_flag(name)

    if returns is not None:
        return_unit: Unit | None = _as_unit(returns)
    else:
        return_unit = _unit_or_flag(function.__name__)[0]
    fname = function.__name__

    def convert(name: str, q: Quantity) -> Any:
        unit, flag = plans.get(name, (None, None))
        if flag is not None:
            raise UnitError(
                f"{fname}(): parameter '{name}' has an unsupported unit suffix ({flag}); pass a plain number"
            )
        if unit is None:
            if q.dimension.is_dimensionless():
                return q.to_value(ONE)
            raise UnitError(
                f"{fname}(): parameter '{name}' carries no unit suffix, so it is dimensionless; got a quantity in "
                f"'{q.unit}' ({q.dimension}). Pass a plain number or declare the unit with units={{'{name}': ...}}"
            )
        try:
            return q.to_value(unit)
        except UnitError:
            raise UnitError(
                f"{fname}(): parameter '{name}' expects '{unit}' ({unit.dimension}), got '{q.unit}' ({q.dimension})"
            ) from None

    def wrap_result(result: Any) -> Any:
        if return_unit is None:
            return result
        if isinstance(result, tuple):
            return tuple(Quantity(r, return_unit) for r in result)
        return Quantity(result, return_unit)

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        for name, value in bound.arguments.items():
            if isinstance(value, Quantity):
                bound.arguments[name] = convert(name, value)
        return wrap_result(function(*bound.args, **bound.kwargs))

    wrapper.parameter_units = {k: v[0] for k, v in plans.items()}  # type: ignore[attr-defined]
    wrapper.return_unit = return_unit  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]
