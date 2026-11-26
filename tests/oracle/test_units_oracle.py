"""Oracle for kozmos.core.units: conversion factors and dimensions against astropy.units.

astropy is a test oracle only. The scale factors of kozmos.core.units come from kozmos.core.constants
(CODATA 2022, IAU 2015, ERFA); astropy 8 adopts IAU 2015 B3 nominal values and CODATA 2022
(``astropy.constants.M_sun.reference``: "IAU 2015 Resolution B 3 + CODATA 2022"), so no difference from
different constants is expected for the installed version. The measured differences are at the rounding level
(1.1e-16 for the angstrom and the erg, which astropy builds from a prefix on another unit). With an older
astropy whose constants are CODATA 2018 the dalton would differ by 1.4e-9 (CODATA 2018 1.66053906660e-27 kg
against CODATA 2022 1.66053906892e-27 kg); the dalton row is the one to look at then.
"""

from __future__ import annotations

import numpy as np
import pytest

astropy_units = pytest.importorskip("astropy.units")
u = astropy_units

from kozmos.black_holes import basics  # noqa: E402
from kozmos.core.units import BASE_DIMENSIONS, Quantity, Unit, with_units  # noqa: E402
from kozmos.stars import scaling  # noqa: E402

pytestmark = pytest.mark.oracle

ASTROPY_BASE = {
    "m": "L",
    "kg": "M",
    "s": "T",
    "A": "I",
    "K": "Theta",
    "mol": "N",
    "cd": "J",
    "rad": "ANGLE",
    "pix": "PIXEL",
    "electron": "ELECTRON",
    "adu": "ADU",
}

# (kozmos expression, astropy expression)
ROWS = [
    ("m", "m"), ("kg", "kg"), ("g", "g"), ("s", "s"), ("A", "A"), ("K", "K"), ("mol", "mol"), ("cd", "cd"),
    ("rad", "rad"), ("sr", "sr"), ("Hz", "Hz"), ("N", "N"), ("Pa", "Pa"), ("J", "J"), ("W", "W"), ("C", "C"),
    ("V", "V"), ("F", "F"), ("ohm", "Ohm"), ("S", "S"), ("Wb", "Wb"), ("T", "T"), ("H", "H"), ("lm", "lm"),
    ("lx", "lx"), ("min", "min"), ("h", "h"), ("d", "d"), ("deg", "deg"), ("arcmin", "arcmin"),
    ("arcsec", "arcsec"), ("as", "arcsec"), ("mas", "mas"), ("uas", "uas"), ("angstrom", "angstrom"),
    ("bar", "bar"), ("Da", "Da"), ("eV", "eV"), ("erg", "erg"), ("au", "au"), ("pc", "pc"), ("kpc", "kpc"),
    ("Mpc", "Mpc"), ("Gpc", "Gpc"), ("yr", "yr"), ("Gyr", "Gyr"), ("Myr", "Myr"), ("Msun", "Msun"),
    ("Rsun", "Rsun"), ("Lsun", "Lsun"), ("Mearth", "Mearth"), ("Mjup", "Mjup"), ("Rearth", "Rearth"),
    ("Rjup", "Rjup"), ("Jy", "Jy"), ("mJy", "mJy"), ("pix", "pix"), ("adu", "adu"), ("e", "electron"),
    ("%", "percent"), ("hPa", "hPa"),
    ("km s^-1", "km / s"), ("Msun pc^-2", "Msun / pc2"), ("Msun yr^-1 kpc^-2", "Msun / (yr kpc2)"),
    ("m^3 s^-2", "m3 / s2"), ("kg m^-3", "kg / m3"), ("m^2 kg^-1", "m2 / kg"),
    ("W m^-2 Hz^-1", "W / (m2 Hz)"), ("km s^-1 kpc^-1", "km / (s kpc)"), ("e adu^-1", "electron / adu"),
    ("g cm^-3", "g / cm3"), ("kg m^-2", "kg / m2"), ("rad s^-1", "rad / s"), ("Jy", "1e-26 W / (m2 Hz)"),
]  # fmt: skip

# Rows whose scale differs from astropy by more than 4 rounding errors, with the measured difference and why.
# Empty for astropy 8.0.0 (see the module docstring); fill in when a different astropy is installed.
KNOWN_DIFFERENCES: dict[str, tuple[float, str]] = {}


def _astropy_scale_and_dimension(expression):
    decomposed = u.Unit(expression).decompose()
    dimension = {ASTROPY_BASE[str(b)]: float(p) for b, p in zip(decomposed.bases, decomposed.powers, strict=True)}
    return float(decomposed.scale), dimension


@pytest.mark.parametrize(("ours", "theirs"), ROWS, ids=[r[0] for r in ROWS])
def test_scale_factor_and_dimension_against_astropy(ours, theirs):
    unit = Unit.parse(ours)
    scale, dimension = _astropy_scale_and_dimension(theirs)
    exponents = dict(zip(BASE_DIMENSIONS, (float(e) for e in unit.dimension.exponents), strict=True))
    assert {k: v for k, v in exponents.items() if v} == {k: v for k, v in dimension.items() if v}
    tolerance, _ = KNOWN_DIFFERENCES.get(ours, (4.5e-16, "rounding: 4 double-precision roundings"))
    assert unit.scale == pytest.approx(scale, rel=tolerance, abs=0), (ours, unit.scale / scale - 1.0)


def test_rows_cover_every_registered_symbol_with_an_astropy_equivalent():
    from kozmos.core.units import UNIT_SOURCES

    covered = {r[0] for r in ROWS}
    # symbols that are alternative spellings of a covered one, or that astropy has no unit for
    alias_of = {
        "Ω": "ohm", "°": "deg", "′": "arcmin", "″": "arcsec", "Å": "angstrom",
        "M☉": "Msun", "M_⊙": "Msun", "R☉": "Rsun", "R_⊙": "Rsun", "L☉": "Lsun",
        "L_⊙": "Lsun", "pixel": "pix", "ADU": "adu", "percent": "%",
    }  # fmt: skip
    missing = {s for s in UNIT_SOURCES if s not in covered and s not in alias_of}
    assert not missing, missing


def test_numerical_conversions_against_astropy_on_random_values():
    rng = np.random.default_rng(20261125)
    values = rng.uniform(1e-3, 1e3, 200)
    pairs = [
        ("Msun pc^-2", "kg m^-2", "Msun / pc2", "kg / m2"),
        ("km s^-1 kpc^-1", "Hz", "km / (s kpc)", "Hz"),
        ("Jy", "W m^-2 Hz^-1", "Jy", "W / (m2 Hz)"),
        ("Gyr", "s", "Gyr", "s"),
        ("uas", "arcsec", "uas", "arcsec"),
        ("deg", "mas", "deg", "mas"),
        ("Lsun", "erg s^-1", "Lsun", "erg / s"),
        ("Rsun", "au", "Rsun", "au"),
        ("pc", "Mpc", "pc", "Mpc"),
        ("eV", "J", "eV", "J"),
        ("hPa", "bar", "hPa", "bar"),
    ]
    for ours_from, ours_to, theirs_from, theirs_to in pairs:
        ours = Quantity(values, ours_from).to_value(ours_to)
        theirs = (values * u.Unit(theirs_from)).to_value(u.Unit(theirs_to))
        np.testing.assert_allclose(ours, theirs, rtol=1e-14, atol=0.0, err_msg=f"{ours_from} -> {ours_to}")


def test_with_units_matches_plain_calls_whose_inputs_astropy_converted():
    """Independent conversion of the inputs by astropy, plain call, against the wrapper with a Quantity."""
    mass = 3.0e6
    expected = basics.schwarzschild_radius_m((mass * u.Msun).to_value(u.kg))
    got = with_units(basics.schwarzschild_radius_m)(Quantity(mass, "Msun"))
    assert got.value == pytest.approx(expected, rel=1e-14, abs=0)

    radius_rsun, teff_mk = 2.5, 6.2e6
    expected = scaling.luminosity_w((radius_rsun * u.Rsun).to_value(u.m), (teff_mk * u.mK).to_value(u.K))
    got = with_units(scaling.luminosity_w)(Quantity(radius_rsun, "Rsun"), Quantity(teff_mk, "mK"))
    assert got.value == pytest.approx(expected, rel=1e-14, abs=0)
    assert got.to_value("Lsun") == pytest.approx((expected * u.W).to_value(u.Lsun), rel=1e-14, abs=0)
