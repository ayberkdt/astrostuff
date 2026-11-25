"""Tests of the unit algebra, the suffix grammar and the boundary checker (kozmos.core.units).

Expected values come from the sources read for the module: the SI Brochure 9th edition (Tables 4, 7, 8,
Section 5.2), IAU 2012 Resolution B2 (astronomical unit), Rots, Gray and Derriere 2021 (jansky, Julian year,
milliarcsecond) and the constants module (CODATA 2022, IAU 2015, ERFA).
"""

from __future__ import annotations

import math
import statistics
import timeit
from fractions import Fraction

import numpy as np
import pytest

from kozmos.black_holes import accretion, basics
from kozmos.core import units as U
from kozmos.core.angles import ARCSEC_RAD, DEG_RAD
from kozmos.core.constants import (
    ASTRONOMICAL_UNIT,
    ATOMIC_MASS_UNIT,
    ELEMENTARY_CHARGE,
    GM_SUN,
    GRAVITATIONAL_CONSTANT,
    JULIAN_YEAR_SECONDS,
    MEGAPARSEC,
    PARSEC,
    SOLAR_LUMINOSITY,
    SOLAR_RADIUS,
)
from kozmos.core.units import (
    BASE_DIMENSIONS,
    Dimension,
    Quantity,
    Unit,
    UnitError,
    UnitSuffixError,
    UnsupportedSuffixError,
    unit_from_suffix,
    with_units,
)
from kozmos.cosmology import distances
from kozmos.ephemeris import kepler
from kozmos.radiation import infrared, magnitudes
from kozmos.star_formation import clouds, laws
from kozmos.stars import scaling, structure

MSUN = GM_SUN.value / GRAVITATIONAL_CONSTANT.value
PC = PARSEC.value
AU = ASTRONOMICAL_UNIT.value
DAY = 86400.0
YEAR = JULIAN_YEAR_SECONDS.value


def q(value, unit):
    return Quantity(value, unit)


# --------------------------------------------------------------------------------------
# Closed-form conversions with the values printed in the sources
# --------------------------------------------------------------------------------------


def test_parsec_and_megaparsec_come_from_the_constants_module_exactly():
    assert q(1, "pc").to_value("m") == PARSEC.value
    assert q(1, "Mpc").to_value("m") == MEGAPARSEC.value
    assert q(1, "kpc").to_value("pc") == pytest.approx(1000.0, rel=1e-15)


def test_parsec_is_648000_over_pi_au():
    """IAU 2015 B2 final note, as implemented in constants.PARSEC (same expression, so exact)."""
    assert Unit.parse("pc").scale == AU * 648_000.0 / math.pi


def test_au_is_149597870700_m_exactly():
    """IAU 2012 Resolution B2, recommendation 1 and SI Brochure Table 8: 1 au = 149 597 870 700 m."""
    assert Unit.parse("au").scale == 149_597_870_700.0


def test_jansky_is_1e_minus_26_si():
    """Rots et al. 2021, Section 3.3 and Table 5: 1 Jy = 1e-26 W m^-2 Hz^-1."""
    assert q(1, "Jy").to_value("W m^-2 Hz^-1") == 1e-26
    assert Unit.parse("Jy").dimension == Unit.parse("W m^-2 Hz^-1").dimension
    assert q(3631, "Jy").to_value("W m^-2 Hz^-1") == pytest.approx(3.631e-23, rel=1e-15)


def test_julian_year_is_31557600_s():
    """Rots et al. 2021, Table 5 and the unit list: Julian year = 365.25 d = 31 557 600 s."""
    assert q(1, "yr").to_value("s") == 31_557_600.0
    assert q(1, "yr").to_value("d") == 365.25
    assert q(1, "Gyr").to_value("yr") == pytest.approx(1e9, rel=1e-15)


@pytest.mark.parametrize(
    ("expression", "in_si", "rel"),
    [
        ("min", 60.0, 0.0),
        ("h", 3600.0, 0.0),
        ("d", 86_400.0, 0.0),
        ("angstrom", 1e-10, 1e-15),
        ("bar", 1e5, 0.0),
        ("eV", 1.602_176_634e-19, 0.0),
        ("Da", 1.660_539_068_92e-27, 0.0),
        ("g", 1e-3, 0.0),
        ("kg", 1.0, 0.0),
        ("erg", 1e-7, 1e-15),
    ],
)
def test_non_si_units_of_table_8(expression, in_si, rel):
    """SI Brochure 9th ed., Table 8 (pp. 140-141) and the Rots et al. 2021 erg."""
    assert Unit.parse(expression).scale == pytest.approx(in_si, rel=rel)


def test_angle_units_of_table_8():
    """Table 8: 1 deg = pi/180 rad, 1' = pi/10 800 rad, 1'' = pi/648 000 rad (ERFA digits: 1e-15)."""
    assert q(1, "deg").to_value("rad") == pytest.approx(math.pi / 180.0, rel=1e-15)
    assert q(1, "arcmin").to_value("rad") == pytest.approx(math.pi / 10_800.0, rel=1e-15)
    assert q(1, "arcsec").to_value("rad") == pytest.approx(math.pi / 648_000.0, rel=1e-15)
    assert q(1, "deg").to_value("rad") == DEG_RAD
    assert q(1, "arcsec").to_value("rad") == ARCSEC_RAD
    assert q(1, "deg").to_value("arcmin") == pytest.approx(60.0, rel=1e-15)
    assert q(1, "deg").to_value("arcsec") == pytest.approx(3600.0, rel=1e-15)


def test_milliarcsecond_is_one_3600000th_of_a_degree():
    """Rots et al. 2021 unit list: mas = 1/3 600 000 deg; text: 1 nrad = 0.2 mas (approximately)."""
    assert q(1, "mas").to_value("deg") == pytest.approx(1.0 / 3_600_000.0, rel=1e-15)
    assert q(1, "uas").to_value("mas") == pytest.approx(1e-3, rel=1e-15)
    assert 0.20 < q(1, "nrad").to_value("mas") < 0.21
    assert q(1, "as").to_value("arcsec") == 1.0  # SI Brochure Table 8 footnote (a)


def test_solar_units_use_the_constants_module():
    assert q(1, "Rsun").to_value("m") == SOLAR_RADIUS.value
    assert q(1, "Lsun").to_value("W") == SOLAR_LUMINOSITY.value
    assert q(1, "Msun").to_value("kg") == MSUN
    assert q(1, "Msun").to_value("kg") == GM_SUN.value / GRAVITATIONAL_CONSTANT.value
    assert q(1, "M☉").to_value("kg") == MSUN


def test_atomic_mass_and_electron_volt_share_the_codata_values():
    assert Unit.parse("Da").scale == ATOMIC_MASS_UNIT.value
    assert Unit.parse("eV").scale == ELEMENTARY_CHARGE.value


@pytest.mark.parametrize(
    ("prefix", "exponent"),
    [
        ("Q", 30), ("R", 27), ("Y", 24), ("Z", 21), ("E", 18), ("P", 15), ("T", 12), ("G", 9), ("M", 6),
        ("k", 3), ("h", 2), ("da", 1), ("d", -1), ("c", -2), ("m", -3), ("µ", -6), ("n", -9), ("p", -12),
        ("f", -15), ("a", -18), ("z", -21), ("y", -24), ("r", -27), ("q", -30),
    ],
)  # fmt: skip
def test_si_prefixes_of_table_7(prefix, exponent):
    """SI Brochure 9th ed., Table 7 (all 24 prefixes)."""
    assert q(1, prefix + "m").to_value("m") == pytest.approx(10.0**exponent, rel=1e-15)


def test_prefix_on_gram_and_on_non_prefixable_units():
    assert q(1, "mg").to_value("kg") == pytest.approx(1e-6, rel=1e-15)  # Brochure Section 3: 1e-6 kg is a mg
    assert q(1, "Mg").to_value("kg") == pytest.approx(1e3, rel=1e-15)  # Table 8: 1 t = 1 Mg = 1e3 kg
    with pytest.raises(UnitError):
        Unit.parse("kMsun")
    with pytest.raises(UnitError):
        Unit.parse("mh")


@pytest.mark.parametrize(
    ("unit", "base_units"),
    [
        ("Hz", "s^-1"),
        ("N", "kg m s^-2"),
        ("Pa", "kg m^-1 s^-2"),
        ("J", "kg m^2 s^-2"),
        ("W", "kg m^2 s^-3"),
        ("C", "A s"),
        ("V", "kg m^2 s^-3 A^-1"),
        ("F", "kg^-1 m^-2 s^4 A^2"),
        ("ohm", "kg m^2 s^-3 A^-2"),
        ("S", "kg^-1 m^-2 s^3 A^2"),
        ("Wb", "kg m^2 s^-2 A^-1"),
        ("T", "kg s^-2 A^-1"),
        ("H", "kg m^2 s^-2 A^-2"),
        ("lm", "cd sr"),
        ("lx", "cd sr m^-2"),
        ("sr", "rad^2"),
    ],
)
def test_derived_units_of_table_4_in_terms_of_base_units(unit, base_units):
    """SI Brochure 9th ed., Table 4 (the 22 units with special names; the base-unit column)."""
    assert Unit.parse(unit) == Unit.parse(base_units)


def test_table_4_relations_between_derived_units():
    assert Unit.parse("J") == Unit.parse("N m")
    assert Unit.parse("W") == Unit.parse("J/s")
    assert Unit.parse("V") == Unit.parse("W/A")
    assert Unit.parse("T") == Unit.parse("Wb/m^2")
    assert Unit.parse("Pa") == Unit.parse("N/m^2")


def test_dimension_string_uses_the_symbols_of_table_3():
    assert str(Unit.parse("kg m^-3").dimension) == "L^-3 M"
    assert str(Unit.parse("Pa").dimension) == "T^-2 L^-1 M"
    assert str(Unit.parse("1").dimension) == "1"


# --------------------------------------------------------------------------------------
# Powers and roots with rational exponents
# --------------------------------------------------------------------------------------


def test_rational_powers_and_roots_of_units():
    m3 = Unit.parse("m^3")
    assert (m3 ** Fraction(1, 3)) == Unit.parse("m")
    assert Unit.parse("m^2") ** 0.5 == Unit.parse("m")
    assert Unit.parse("kg") ** Fraction(-4, 3) == Unit.parse("kg^{-4/3}")
    assert (Unit.parse("pc") ** 3) ** Fraction(1, 3) == Unit.parse("pc")
    assert Unit.parse("s") ** 0 == Unit.parse("1")


def test_white_dwarf_coefficient_unit_has_the_dimension_in_its_docstring():
    """kozmos.stars.scaling.white_dwarf_k_coefficient_si is documented as Pa m^4 kg^{-4/3}."""
    unit = Unit.parse("Pa m^4 kg^{-4/3}")
    assert unit.dimension == Dimension.of(M=Fraction(-1, 3), L=3, T=-2)


def test_quantity_powers_and_roots():
    volume = q(8.0, "m^3")
    assert (volume ** Fraction(1, 3)).to_value("m") == pytest.approx(2.0, rel=1e-15)
    assert q(4.0, "m^2").sqrt().unit == Unit.parse("m")
    assert (q(2.0, "km") ** 2).to_value("m^2") == pytest.approx(4e6, rel=1e-15)
    assert (q(3.0, "s") ** -1).to_value("Hz") == pytest.approx(1.0 / 3.0, rel=1e-15)
    cube_root = q(27.0, "Msun^3") ** (1 / 3)  # a float that is 1/3 within 1e-12
    assert cube_root.unit == Unit.parse("Msun")
    assert cube_root.value == pytest.approx(3.0, rel=1e-14)


def test_irrational_exponent_is_refused():
    with pytest.raises(UnitError, match="rational"):
        q(1.0, "m") ** math.pi
    with pytest.raises(UnitError):
        Dimension.of(L=1) ** 0.123456789


def test_dimension_exponents_are_exact_rationals():
    d = Dimension.of(L=1, M=1, T=-2) ** Fraction(1, 3)
    assert d.exponents[BASE_DIMENSIONS.index("L")] == Fraction(1, 3)
    assert (d**3) == Dimension.of(L=1, M=1, T=-2)
    assert str(d) == "T^-2/3 L^1/3 M^1/3"


# --------------------------------------------------------------------------------------
# Dimension errors, arithmetic, comparison
# --------------------------------------------------------------------------------------


def test_addition_converts_to_the_left_unit():
    total = q(3.0, "km") + q(500.0, "m")
    assert total.unit == Unit.parse("km")
    assert total.value == pytest.approx(3.5, rel=1e-15)
    diff = q(1.0, "pc") - q(1.0, "au")
    assert diff.to_value("m") == pytest.approx(PC - AU, rel=1e-15)


def test_addition_of_different_dimensions_raises_and_names_both():
    with pytest.raises(UnitError, match="dimensions differ"):
        q(1.0, "m") + q(1.0, "kg")
    with pytest.raises(UnitError):
        q(1.0, "m") - q(1.0, "s")
    with pytest.raises(UnitError):
        q(1.0, "m") + 1.0
    with pytest.raises(UnitError):
        _ = q(1.0, "m") < q(1.0, "kg")


def test_plain_numbers_combine_only_with_dimensionless_quantities():
    assert (q(0.5, "1") + 1.0).value == 1.5
    assert (q(50.0, "%") + 1.0).to_value("%") == pytest.approx(150.0, rel=1e-15)
    with pytest.raises(UnitError):
        q(1.0, "rad") + 1.0  # an angle is not dimensionless in this system


def test_conversion_between_dimensions_raises_with_both_dimensions_in_the_message():
    with pytest.raises(UnitError, match=r"kg.*M.*m.*L"):
        q(1.0, "kg").to("m")
    with pytest.raises(UnitError):
        q(1.0, "m").to_value("s")


def test_multiplication_and_division_combine_units():
    speed = q(10.0, "km") / q(2.0, "s")
    assert speed.to_value("m s^-1") == pytest.approx(5000.0, rel=1e-15)
    force = q(2.0, "kg") * q(3.0, "m s^-2")
    assert force.to_value("N") == pytest.approx(6.0, rel=1e-15)
    assert (2.0 * q(1.0, "m")).value == 2.0
    assert (q(6.0, "m") / 2.0).value == 3.0
    assert (1.0 / q(4.0, "s")).to_value("Hz") == 0.25


def test_dimensionless_ratio_is_returned_as_a_plain_number():
    ratio = q(3.0, "km") / q(1.0, "m")
    assert ratio.unit == Unit.parse("1")
    assert ratio.value == pytest.approx(3000.0, rel=1e-15)


def test_comparison_and_equality_after_conversion():
    assert q(1.0, "km") == q(1000.0, "m")
    assert q(1.0, "km") != q(1.0, "m")
    assert q(1.0, "m") != q(1.0, "kg")  # equality across dimensions is False, not an error
    assert q(1.0, "km") > q(999.0, "m")
    assert q(1.0, "km") >= q(1000.0, "m")
    assert q(1.0, "m") < q(1.0, "km")
    assert np.all(q([1.0, 2.0], "km") == q([1000.0, 2000.0], "m"))
    assert q(1.0, "km").isclose(q(1000.0 * (1 + 1e-12), "m"))
    assert Quantity.__hash__ is None


def test_array_quantities_keep_shape_and_index():
    x = q(np.arange(6.0).reshape(2, 3), "pc")
    assert x.shape == (2, 3)
    assert x[1].unit == Unit.parse("pc")
    assert x.to("kpc").value[1, 2] == pytest.approx(5e-3, rel=1e-15)
    assert len(x) == 2


# --------------------------------------------------------------------------------------
# Round trips
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", sorted(U.UNIT_SOURCES))
def test_round_trip_through_the_coherent_si_unit(symbol):
    unit = Unit.parse(symbol)
    si = Unit(unit.dimension, 1.0)
    values = np.array([1e-9, 0.37, 1.0, 12.5, 4.2e11])
    back = q(values, symbol).to(si).to(symbol)
    np.testing.assert_allclose(back.value, values, rtol=4e-16 * 3)


def test_round_trip_through_a_chain_of_astronomical_units():
    v = q(0.7, "Mpc")
    chain = v.to("kpc").to("pc").to("au").to("m").to("km").to("Mpc")
    assert chain.value == pytest.approx(0.7, rel=1e-14)


def test_unit_algebra_is_consistent_with_quantity_algebra():
    a, b = q(3.0, "Msun"), q(2.0, "pc^-2")
    assert (a * b).to_value("kg m^-2") == pytest.approx(3.0 * MSUN * 2.0 / PC**2, rel=1e-14)


# --------------------------------------------------------------------------------------
# Angle and counting-dimension policy
# --------------------------------------------------------------------------------------


def test_radian_per_second_is_not_a_hertz():
    """Own ANGLE dimension (SI Brochure 2.3.3, Table 4 footnotes (b) and (d)): the 2 pi is not lost silently."""
    with pytest.raises(UnitError):
        q(1.0, "rad s^-1").to("Hz")
    assert q(1.0, "rad s^-1").drop_angle().to_value("Hz") == 1.0


def test_angle_is_not_dimensionless_but_drop_angle_gives_the_si_view():
    assert not q(1.0, "deg").dimension.is_dimensionless()
    assert q(180.0, "deg").drop_angle().to_value("1") == pytest.approx(math.pi, rel=1e-15)
    assert (q(2.0, "m") * q(0.5, "rad")).unit == Unit.parse("m rad")
    assert (q(2.0, "m") * q(0.5, "rad")).drop_angle().to_value("m") == 1.0
    assert Unit.parse("sr") == Unit.parse("rad^2")


def test_counting_dimensions_are_separate():
    assert Unit.parse("e adu^-1").dimension != Unit.parse("e s^-1").dimension
    with pytest.raises(UnitError):
        q(1.0, "adu").to("e")
    with pytest.raises(UnitError):
        q(1.0, "pix").to("1")


# --------------------------------------------------------------------------------------
# Unit expressions: syntax of SI Brochure Section 5.2 and the spellings used in docstrings
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("m/s", "m s^-1"),
        ("kg/m^3", "kg m^-3"),
        ("kg m^-3", "kg⋅m^-3"),
        ("m³ s⁻²", "m^3 s^-2"),
        ("M☉ pc⁻²", "Msun pc^-2"),
        ("M_⊙", "Msun"),
        ("W/m^2 Hz", "W m^-2 Hz^-1"),
        ("e⁻/s", "e s^-1"),
        ("µm", "um"),
        ("μm", "um"),
        ("Å", "angstrom"),
        ("km s⁻¹ kpc⁻¹", "km/s kpc"),
        ("m3 s^-2", "m^3 s^-2"),
        ("1/s", "Hz"),
    ],
)
def test_unit_expression_spellings(a, b):
    assert Unit.parse(a) == Unit.parse(b)


@pytest.mark.parametrize("bad", ["", "  ", "foo", "m//s", "m/s/kg", "m^x", "m^{1/0}", "kg^{3(1+1/n)}", "°C", "degC"])
def test_bad_unit_expressions_raise(bad):
    with pytest.raises(UnitError):
        Unit.parse(bad)


def test_celsius_is_refused_with_an_explanation():
    with pytest.raises(UnitError, match="offset"):
        Unit.parse("°C")


def test_unit_equality_and_display():
    assert Unit.parse("km") == Unit(Dimension.of(L=1), 1000.0)
    assert Unit.parse("km") != Unit.parse("m")
    assert str(Unit.parse("kg m^-3")) == "kg m^-3"
    assert str(Unit.parse("1")) == "1"
    assert 3 * Unit.parse("m") == q(3.0, "m")


# --------------------------------------------------------------------------------------
# Suffix grammar
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expression"),
    [
        ("mass_kg", "kg"),
        ("radius_m", "m"),
        ("ra_deg", "deg"),
        ("separation_arcsec", "arcsec"),
        ("luminosity_w", "W"),
        ("distance_mpc", "Mpc"),
        ("time_gyr", "Gyr"),
        ("frequency_hz", "Hz"),
        ("flux_jy", "Jy"),
        ("temperature_k", "K"),
        ("velocity_km_s", "km s^-1"),
        ("density_kg_m3", "kg m^-3"),
        ("surface_density_msun_pc2", "Msun pc^-2"),
        ("rate_kg_s", "kg s^-1"),
        ("sfr_surface_density_msun_yr_kpc2", "Msun yr^-1 kpc^-2"),
        ("opacity_m2_kg", "m^2 kg^-1"),
        ("gm_m3_s2", "m^3 s^-2"),
        ("omega_km_s_kpc", "km s^-1 kpc^-1"),
        ("surface_gravity_m_s2", "m s^-2"),
        ("gw_frequency_derivative_hz_s", "Hz s^-1"),
        ("horizon_angular_velocity_rad_s", "rad s^-1"),
        ("gain_e_adu", "e adu^-1"),
        ("master_dark_e_s", "e s^-1"),
        ("sky_e_pix", "e pix^-1"),
        ("comoving_volume_mpc3", "Mpc^3"),
        ("shadow_angular_diameter_uas", "uas"),
        ("pressure_hpa", "hPa"),
        ("magnetic_field_t", "T"),
        ("magnetic_flux_wb", "Wb"),
        ("polytrope_gravitational_energy_j", "J"),
        ("wavelength_um", "um"),
        ("time_above_horizon_hr", "h"),
    ],
)
def test_suffix_grammar_examples(name, expression):
    assert unit_from_suffix(name) == Unit.parse(expression)


def test_first_token_is_numerator_following_tokens_are_denominators():
    assert unit_from_suffix("a_m3_s2").dimension == Dimension.of(L=3, T=-2)
    assert unit_from_suffix("a_s2_m3").dimension == Dimension.of(L=-3, T=2)
    assert unit_from_suffix("a_msun_yr_kpc2").dimension == Dimension.of(M=1, T=-1, L=-2)


def test_number_and_column_density_exception():
    """this project's choice: a single positive power after these stems is a reciprocal."""
    assert unit_from_suffix("number_density_m3") == Unit.parse("m^-3")
    assert unit_from_suffix("column_density_m2") == Unit.parse("m^-2")
    assert unit_from_suffix("mass_density_kg_m3") == Unit.parse("kg m^-3")  # two tokens: ordinary rule
    assert unit_from_suffix("comoving_volume_mpc3") == Unit.parse("Mpc^3")


@pytest.mark.parametrize(
    ("name", "expression"),
    [
        ("flux_jy_from_vega_magnitude", "Jy"),
        ("pc_from_distance_modulus", "pc"),
        ("luminosity_from_bolometric_magnitude_w", "W"),
        ("gw_frequency_from_time_to_merger_hz", "Hz"),
    ],
)
def test_from_rule_result_unit(name, expression):
    assert unit_from_suffix(name) == Unit.parse(expression)


def test_unit_after_from_names_the_source_quantity_not_the_result():
    assert unit_from_suffix("distance_modulus_from_pc") is None


@pytest.mark.parametrize(
    "name",
    ["spin", "efficiency", "z", "x", "m", "s", "day", "day_fraction", "jd1", "ut1_jd1", "m_low", "mean_molecular_weight",
     "min_pixels", "n_points", "lomb_scargle", "e_function", "c_phi", "sigma_y", "spin_one"],
)  # fmt: skip
def test_names_without_a_unit_suffix(name):
    """A lone token is a stem, never a suffix; words that merely look like symbols are not units."""
    assert unit_from_suffix(name) is None


@pytest.mark.parametrize(
    ("name", "token", "kind"),
    [
        ("temperature_c", "c", "offset"),
        ("transit_jd", "jd", "epoch"),
        ("calendar_to_mjd", "mjd", "epoch"),
        ("distance_modulus_mag", "mag", "logarithmic"),
        ("log_surface_gravity_cgs", "cgs", "logarithmic"),
        ("kerr_isco_radius_rg", "rg", "context"),
        ("k_si", "si", "context"),
    ],
)
def test_unsupported_suffixes_raise_with_kind(name, token, kind):
    with pytest.raises(UnsupportedSuffixError) as info:
        unit_from_suffix(name)
    assert info.value.token == token
    assert info.value.kind == kind
    assert isinstance(info.value, UnitError)


@pytest.mark.parametrize("bad", ["", " mass_kg", "mass kg", None, 3])
def test_invalid_identifiers_raise(bad):
    with pytest.raises(UnitSuffixError):
        unit_from_suffix(bad)


def test_every_suffix_token_resolves_to_a_unit_with_a_source():
    for token, expression in U.SUFFIX_UNITS.items():
        unit = Unit.parse(expression)
        assert unit_from_suffix(f"x_{token}") == unit, token
    for symbol, origin in U.UNIT_SOURCES.items():
        assert origin.strip(), symbol


# --------------------------------------------------------------------------------------
# Boundary checker on toy functions
# --------------------------------------------------------------------------------------


def _kinetic_energy_j(mass_kg, speed_m_s, scale=1.0):
    return 0.5 * mass_kg * speed_m_s**2 * scale


def test_wrapper_converts_arguments_and_wraps_the_result():
    f = with_units(_kinetic_energy_j)
    plain = _kinetic_energy_j(2.0, 3.0)
    out = f(q(2000.0, "g"), q(0.003, "km s^-1"))
    assert isinstance(out, Quantity)
    assert out.unit == Unit.parse("J")
    assert out.value == pytest.approx(plain, rel=1e-15)
    assert out.to_value("erg") == pytest.approx(plain * 1e7, rel=1e-15)


def test_plain_numbers_and_arrays_pass_through_unchanged():
    f = with_units(_kinetic_energy_j)
    out = f(2.0, speed_m_s=np.array([3.0, 4.0]))
    np.testing.assert_array_equal(out.value, _kinetic_energy_j(2.0, np.array([3.0, 4.0])))
    mixed = f(q(2.0, "kg"), 3.0)  # a Quantity and a plain number in one call
    assert mixed.value == _kinetic_energy_j(2.0, 3.0)


def test_wrapper_error_names_function_parameter_and_both_dimensions():
    f = with_units(_kinetic_energy_j)
    with pytest.raises(UnitError) as info:
        f(q(2.0, "m"), 3.0)  # a length where a mass is expected
    text = str(info.value)
    assert "_kinetic_energy_j" in text
    assert "'mass_kg'" in text
    assert "'kg' (M)" in text
    assert "'m' (L)" in text


def test_wrapper_keeps_metadata_and_exposes_the_plan():
    f = with_units(_kinetic_energy_j)
    assert f.__name__ == "_kinetic_energy_j"
    assert f.__wrapped__ is _kinetic_energy_j
    assert f.parameter_units["mass_kg"] == Unit.parse("kg")
    assert f.parameter_units["speed_m_s"] == Unit.parse("m/s")
    assert f.parameter_units["scale"] is None
    assert f.return_unit == Unit.parse("J")


def test_unsuffixed_parameter_accepts_only_dimensionless_quantities():
    f = with_units(_kinetic_energy_j)
    assert f(2.0, 3.0, scale=q(50.0, "%")).value == _kinetic_energy_j(2.0, 3.0, 0.5)
    with pytest.raises(UnitError, match="'scale'.*no unit suffix"):
        f(2.0, 3.0, scale=q(1.0, "kg"))
    with pytest.raises(UnitError):
        f(2.0, 3.0, scale=q(1.0, "rad"))  # an angle is not dimensionless in this system


def test_function_without_a_suffix_returns_the_plain_result():
    def ratio(mass_kg, other_mass_kg):
        return mass_kg / other_mass_kg

    out = with_units(ratio)(q(1.0, "Msun"), q(1.0, "kg"))
    assert not isinstance(out, Quantity)
    assert out == pytest.approx(MSUN, rel=1e-15)


def test_units_and_returns_overrides():
    def mean_mass(m_low, m_high):
        return 0.5 * (m_low + m_high)

    f = with_units(mean_mass, units={"m_low": "Msun", "m_high": "Msun"}, returns="Msun")
    out = f(q(2.0 * MSUN, "kg"), q(4.0, "Msun"))
    assert out.unit == Unit.parse("Msun")
    assert out.value == pytest.approx(3.0, rel=1e-15)
    with pytest.raises(UnitError):
        f(q(1.0, "m"), 1.0)
    with pytest.raises(UnitError, match="does not have"):
        with_units(mean_mass, units={"nope": "m"})


def test_tuple_results_are_wrapped_element_wise():
    def split_m(length_m):
        return length_m / 2, length_m / 4

    a, b = with_units(split_m)(q(4.0, "km"))
    assert (a.to_value("m"), b.to_value("m")) == (2000.0, 1000.0)


def test_unsupported_suffix_parameters_refuse_a_quantity_but_take_plain_numbers():
    def boil(temperature_c, pressure_hpa):
        return temperature_c + pressure_hpa

    f = with_units(boil)
    assert f(20.0, q(1.0, "bar")) == 1020.0  # 1 bar = 1000 hPa; the Celsius number stays a plain number
    with pytest.raises(UnitError, match="unsupported unit suffix"):
        f(q(20.0, "K"), 1.0)
    assert f.parameter_units["temperature_c"] is None


def test_exceptions_of_the_wrapped_function_pass_through():
    with pytest.raises(ValueError, match="positive"):
        with_units(basics.schwarzschild_radius_m)(q(-1.0, "kg"))


def test_wrapper_rejects_wrong_call_signature_like_the_function_does():
    f = with_units(_kinetic_energy_j)
    with pytest.raises(TypeError):
        f(1.0)


# --------------------------------------------------------------------------------------
# The checker on real functions of the package
# --------------------------------------------------------------------------------------

COSMO = distances.Cosmology.flat(h0=70.0, omega_m=0.3)
SPEC = {}  # id -> (function, native kwargs, alternative inputs {param: (value, unit)}, overrides)


def case(function, native, other_units, **overrides):
    SPEC[f"{function.__module__.split('.', 1)[1]}.{function.__name__}"] = (function, native, other_units, overrides)


# black holes ----------------------------------------------------------------------------
case(basics.schwarzschild_radius_m, {"mass_kg": 2.0e30}, {"mass_kg": (2.0e30 / MSUN, "Msun")})
case(basics.gravitational_radius_m, {"mass_kg": 1.5e31}, {"mass_kg": (1.5e34, "g")})
case(basics.photon_sphere_radius_m, {"mass_kg": 8.0e36}, {"mass_kg": (8.0e36 / MSUN, "Msun")})
case(basics.schwarzschild_isco_radius_m, {"mass_kg": 8.0e36}, {"mass_kg": (8.0e33, "Mg")})
case(basics.isco_orbital_frequency_hz, {"mass_kg": 2.0e31}, {"mass_kg": (2.0e31 / MSUN, "Msun")})
case(
    basics.shadow_angular_diameter_uas,
    {"mass_kg": 6.5e9 * MSUN, "distance_m": 16.8 * MEGAPARSEC.value},
    {"mass_kg": (6.5e9, "Msun"), "distance_m": (16.8e3, "kpc")},
)
case(basics.hawking_temperature_k, {"mass_kg": 1.0e12}, {"mass_kg": (1.0e15, "g")})
case(
    basics.horizon_angular_velocity_rad_s,
    {"mass_kg": 3.0e31, "spin": 0.7},
    {"mass_kg": (3.0e31 / MSUN, "Msun"), "spin": (0.7, "1")},
)
case(
    basics.chirp_mass_kg,
    {"mass1_kg": 36.0 * MSUN, "mass2_kg": 29.0 * MSUN},
    {"mass1_kg": (36.0, "Msun"), "mass2_kg": (29.0, "Msun")},
)
case(
    basics.time_to_merger_s,
    {"chirp_mass_kg": 30.0 * MSUN, "frequency_hz": 35.0},
    {"chirp_mass_kg": (30.0, "Msun"), "frequency_hz": (0.035, "kHz")},
)
case(accretion.eddington_luminosity_w, {"mass_kg": 4.0e6 * MSUN}, {"mass_kg": (4.0e6, "Msun")})
case(
    accretion.bondi_radius_m,
    {"mass_kg": 4.0e6 * MSUN, "sound_speed_m_s": 5.0e5},
    {"mass_kg": (4.0e6, "Msun"), "sound_speed_m_s": (500.0, "km s^-1")},
)
case(
    accretion.tidal_disruption_radius_m,
    {"black_hole_mass_kg": 1.0e6 * MSUN, "stellar_mass_kg": MSUN, "stellar_radius_m": SOLAR_RADIUS.value},
    {"black_hole_mass_kg": (1.0e6, "Msun"), "stellar_mass_kg": (1.0, "Msun"), "stellar_radius_m": (1.0, "Rsun")},
)
# stars ----------------------------------------------------------------------------------
case(
    scaling.luminosity_w,
    {"radius_m": SOLAR_RADIUS.value, "effective_temperature_k": 5772.0},
    {"radius_m": (1.0, "Rsun"), "effective_temperature_k": (5.772e6, "mK")},
)
case(
    scaling.effective_temperature_k,
    {"luminosity_w": SOLAR_LUMINOSITY.value, "radius_m": SOLAR_RADIUS.value},
    {"luminosity_w": (1.0, "Lsun"), "radius_m": (1.0, "Rsun")},
)
case(
    scaling.surface_gravity_m_s2,
    {"mass_kg": MSUN, "radius_m": SOLAR_RADIUS.value},
    {"mass_kg": (1.0, "Msun"), "radius_m": (1.0, "Rsun")},
)
case(
    scaling.mean_density_kg_m3,
    {"mass_kg": MSUN, "radius_m": SOLAR_RADIUS.value},
    {"mass_kg": (1.0, "Msun"), "radius_m": (SOLAR_RADIUS.value / PC, "pc")},
)
case(scaling.free_fall_time_s, {"density_kg_m3": 1410.0}, {"density_kg_m3": (1.41, "g cm^-3")})
case(
    scaling.kelvin_helmholtz_timescale_s,
    {"mass_kg": MSUN, "radius_m": SOLAR_RADIUS.value, "luminosity_w": SOLAR_LUMINOSITY.value},
    {"mass_kg": (1.0, "Msun"), "radius_m": (1.0, "Rsun"), "luminosity_w": (1.0, "Lsun")},
)
case(
    scaling.nuclear_timescale_yr,
    {"mass_kg": MSUN, "luminosity_w": SOLAR_LUMINOSITY.value, "yield_fraction": 0.007, "fuel_fraction": 0.1},
    {"mass_kg": (1.0, "Msun"), "luminosity_w": (1.0, "Lsun")},
)
case(
    scaling.eddington_luminosity_w,
    {"mass_kg": 10.0 * MSUN, "opacity_m2_kg": 0.034},
    {"mass_kg": (10.0, "Msun"), "opacity_m2_kg": (0.34, "cm^2 g^-1")},
)
case(
    structure.polytrope_central_density_kg_m3,
    {"mass_kg": MSUN, "radius_m": SOLAR_RADIUS.value, "n": 3.0},
    {"mass_kg": (1.0, "Msun"), "radius_m": (1.0, "Rsun")},
)
case(
    structure.polytrope_central_pressure_pa,
    {"mass_kg": MSUN, "radius_m": SOLAR_RADIUS.value, "n": 3.0},
    {"mass_kg": (1.0, "Msun"), "radius_m": (1.0, "Rsun")},
)
case(
    structure.virial_mean_temperature_k,
    {"mass_kg": MSUN, "radius_m": SOLAR_RADIUS.value, "mean_molecular_weight": 0.6},
    {"mass_kg": (1.0, "Msun"), "radius_m": (1.0, "Rsun")},
)
# star formation -------------------------------------------------------------------------
case(
    clouds.sound_speed_m_s,
    {"temperature_k": 10.0, "mean_molecular_weight": 2.33},
    {"temperature_k": (1.0e4, "mK")},
)
case(
    clouds.jeans_length_m,
    {"sound_speed_m_s": 200.0, "density_kg_m3": 1e-18},
    {"sound_speed_m_s": (0.2, "km s^-1"), "density_kg_m3": (1e-21, "g cm^-3")},
)
case(
    clouds.jeans_mass_kg,
    {"sound_speed_m_s": 200.0, "density_kg_m3": 1e-18},
    {"sound_speed_m_s": (0.2, "km s^-1"), "density_kg_m3": (1e-21, "g cm^-3")},
)
case(
    clouds.crossing_time_s,
    {"radius_m": 3.0 * PC, "velocity_dispersion_m_s": 1.0e3},
    {"radius_m": (3.0, "pc"), "velocity_dispersion_m_s": (1.0, "km s^-1")},
)
case(
    clouds.virial_parameter,
    {"velocity_dispersion_m_s": 1.0e3, "radius_m": 3.0 * PC, "mass_kg": 1.0e3 * MSUN},
    {"velocity_dispersion_m_s": (1.0, "km s^-1"), "radius_m": (3.0, "pc"), "mass_kg": (1.0e3, "Msun")},
)
case(
    clouds.mass_density_kg_m3,
    {"number_density_m3": 1.0e10, "mean_molecular_weight": 2.33},
    {"number_density_m3": (1.0e4, "cm^-3")},
)
case(clouds.column_density_m2, {"surface_density_kg_m2": 10.0}, {"surface_density_kg_m2": (1.0, "g cm^-2")})
case(
    clouds.linewidth_size_km_s,
    {"size_pc": 10.0, "coefficient_km_s": 0.72, "exponent": 0.5},
    {"size_pc": (0.01, "kpc"), "coefficient_km_s": (720.0, "m s^-1")},
)
case(
    laws.depletion_time_yr,
    {"gas_surface_density_msun_pc2": 10.0, "sfr_surface_density_msun_yr_kpc2": 1.0e-3},
    {
        "gas_surface_density_msun_pc2": (10.0 * MSUN / PC**2, "kg m^-2"),
        "sfr_surface_density_msun_yr_kpc2": (1.0, "Msun Gyr^-1 pc^-2"),
    },
)
case(
    laws.toomre_q_gas,
    {"kappa_km_s_kpc": 40.0, "sound_speed_km_s": 8.0, "surface_density_msun_pc2": 10.0},
    {
        "kappa_km_s_kpc": (40.0e3, "m s^-1 kpc^-1"),
        "sound_speed_km_s": (8.0e3, "m s^-1"),
        "surface_density_msun_pc2": (10.0e6, "Msun kpc^-2"),
    },
)
case(
    laws.two_dimensional_jeans_mass_msun,
    {"sound_speed_km_s": 8.0, "surface_density_msun_pc2": 10.0},
    {"sound_speed_km_s": (8.0e3, "m s^-1"), "surface_density_msun_pc2": (10.0e6, "Msun kpc^-2")},
)
# cosmology ------------------------------------------------------------------------------
case(distances.doppler_redshift, {"velocity_km_s": 3000.0}, {"velocity_km_s": (3.0e6, "m s^-1")})
case(distances.doppler_velocity_km_s, {"z": 0.01}, {"z": (1.0, "%")})
case(distances.hubble_distance_mpc, {"cosmology": COSMO}, {})
case(
    distances.luminosity_distance_mpc,
    {"z": np.array([0.1, 0.5]), "cosmology": COSMO},
    {"z": (np.array([10.0, 50.0]), "%")},
)
case(distances.lookback_time_gyr, {"z": 0.5, "cosmology": COSMO}, {})
case(distances.peculiar_velocity_km_s, {"z_observed": 0.011, "z_cosmological": 0.01}, {"z_observed": (1.1, "%")})
# radiation ------------------------------------------------------------------------------
case(infrared.peak_wavelength_um, {"temperature_k": 300.0}, {"temperature_k": (3.0e5, "mK")})
case(infrared.peak_temperature_k, {"wavelength_um": 10.0}, {"wavelength_um": (1.0e5, "angstrom")})
case(
    magnitudes.absolute_bolometric_magnitude, {"luminosity_w": SOLAR_LUMINOSITY.value}, {"luminosity_w": (1.0, "Lsun")}
)
case(
    magnitudes.planck_frequency,
    {"frequency_hz": 2.0e11, "temperature_k": 2.725},
    {"frequency_hz": (200.0, "GHz"), "temperature_k": (2725.0, "mK")},
    returns="W m^-2 Hz^-1 sr^-1",
)
case(magnitudes.blackbody_flux, {"temperature_k": 5772.0}, {"temperature_k": (5.772, "kK")}, returns="W m^-2")
case(magnitudes.distance_modulus_from_pc, {"distance_pc": 10.0}, {"distance_pc": (0.01, "kpc")})
case(infrared.flux_jy_from_vega_magnitude, {"magnitude": 10.0, "band_name": "J"}, {})
case(infrared.vega_magnitude_from_flux, {"flux_jy": 0.5, "band_name": "J"}, {"flux_jy": (0.5e-26, "W m^-2 Hz^-1")})
# ephemeris ------------------------------------------------------------------------------
case(
    kepler.orbital_period_s,
    {"semi_major_axis_m": AU, "gm_m3_s2": GM_SUN.value},
    {"semi_major_axis_m": (1.0, "au"), "gm_m3_s2": (GM_SUN.value * 1e-9, "km^3 s^-2")},
)
case(
    kepler.orbital_speed_m_s,
    {"distance_m": AU, "semi_major_axis_m": 1.5 * AU, "gm_m3_s2": GM_SUN.value},
    {"distance_m": (1.0, "au"), "semi_major_axis_m": (1.5, "au"), "gm_m3_s2": (GM_SUN.value * 1e-9, "km^3 s^-2")},
)
case(
    kepler.eccentric_anomaly,
    {"mean_anomaly_rad": 1.0, "eccentricity": 0.3},
    {"mean_anomaly_rad": (1.0 / DEG_RAD, "deg")},
    returns="rad",
)
case(
    kepler.true_anomaly,
    {"eccentric_anomaly_rad": 1.0, "eccentricity": 0.3},
    {"eccentric_anomaly_rad": (1.0 / ARCSEC_RAD, "arcsec")},
    returns="rad",
)

IDS = sorted(SPEC)


def test_at_least_25_functions_in_the_six_packages_are_checked():
    packages = {i.split(".")[0] for i in IDS}
    assert packages == {"black_holes", "stars", "star_formation", "cosmology", "radiation", "ephemeris"}
    assert len(IDS) >= 25


def _wrapped(case_id):
    function, native, other_units, overrides = SPEC[case_id]
    return with_units(function, **overrides), function, native, other_units


def _as_numbers(result):
    return result.value if isinstance(result, Quantity) else result


@pytest.mark.parametrize("case_id", IDS)
def test_inputs_in_other_units_give_the_plain_result(case_id):
    wrapped, function, native, other_units = _wrapped(case_id)
    expected = function(**native)
    kwargs = dict(native)
    for name, (value, unit) in other_units.items():
        kwargs[name] = q(value, unit)
    got = wrapped(**kwargs)
    # 1e-12: every conversion is one multiplication and one division (relative error 2.2e-16 each); the
    # largest exponent of an argument in these functions is 3 and results involve at most 4 arguments.
    np.testing.assert_allclose(_as_numbers(got), expected, rtol=1e-12, atol=0.0)
    if wrapped.return_unit is not None:
        assert isinstance(got, Quantity)
        assert got.unit == wrapped.return_unit
    else:
        assert not isinstance(got, Quantity)


@pytest.mark.parametrize("case_id", IDS)
def test_inputs_in_the_suffix_unit_are_bitwise_identical_to_the_plain_call(case_id):
    wrapped, function, native, _ = _wrapped(case_id)
    expected = function(**native)
    kwargs = dict(native)
    for name, unit in wrapped.parameter_units.items():
        if name in kwargs and unit is not None and not isinstance(kwargs[name], distances.Cosmology | str):
            kwargs[name] = q(kwargs[name], unit)
    np.testing.assert_array_equal(_as_numbers(wrapped(**kwargs)), expected)


@pytest.mark.parametrize("case_id", IDS)
def test_a_quantity_of_the_wrong_kind_is_refused_for_every_unit_parameter(case_id):
    wrapped, _, native, _ = _wrapped(case_id)
    checked = [name for name, unit in wrapped.parameter_units.items() if unit is not None and name in native]
    for name in checked:
        kwargs = dict(native)
        kwargs[name] = q(1.0, "A")  # an electric current is a length, mass, time ... of nothing in this package
        with pytest.raises(UnitError, match=f"'{name}'"):
            wrapped(**kwargs)


@pytest.mark.parametrize("case_id", IDS)
def test_swapping_two_arguments_of_different_dimension_raises(case_id):
    wrapped, _, native, _ = _wrapped(case_id)
    units = {
        name: unit
        for name, unit in wrapped.parameter_units.items()
        if unit is not None and name in native and not isinstance(native[name], distances.Cosmology | str)
    }
    names = sorted(units)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :] if units[a].dimension != units[b].dimension]
    for a, b in pairs:
        kwargs = dict(native)
        for name in names:
            kwargs[name] = q(native[name], units[name])
        kwargs[a], kwargs[b] = kwargs[b], kwargs[a]
        with pytest.raises(UnitError, match=f"'{a}'|'{b}'"):
            wrapped(**kwargs)


def test_the_swap_test_is_not_vacuous():
    """Most functions of the table have a pair of unit parameters of different dimension to swap."""
    with_pairs = 0
    for case_id in IDS:
        wrapped, _, native, _ = _wrapped(case_id)
        dims = {u.dimension for n, u in wrapped.parameter_units.items() if u is not None and n in native}
        with_pairs += len(dims) >= 2
    assert with_pairs >= 20


def test_a_mass_where_a_length_is_expected_is_the_documented_example():
    f = with_units(basics.schwarzschild_radius_m)
    with pytest.raises(UnitError, match=r"'mass_kg' expects 'kg' \(M\), got 'm' \(L\)"):
        f(q(1.0, "m"))
    g = with_units(clouds.crossing_time_s)
    with pytest.raises(UnitError, match="'radius_m'"):
        g(q(1.0, "km s^-1"), q(3.0, "pc"))  # radius and velocity dispersion swapped


def test_results_are_quantities_in_the_unit_of_the_function_suffix():
    r = with_units(basics.schwarzschild_radius_m)(q(1.0, "Msun"))
    assert r.unit == Unit.parse("m")
    assert r.to_value("km") == pytest.approx(2.953, rel=1e-3)  # 2 G Msun / c^2 = 2.953 km
    shadow = with_units(basics.shadow_angular_diameter_uas)(q(6.5e9, "Msun"), q(16.8, "Mpc"))
    assert shadow.unit == Unit.parse("uas")
    assert shadow.to_value("arcsec") == pytest.approx(shadow.value * 1e-6, rel=1e-14)
    assert 30.0 < shadow.value < 60.0  # M87*: a few tens of microarcseconds


def test_astronomical_unit_period_matches_a_gaussian_year():
    """kozmos.ephemeris.kepler test idea: P(1 au, GM_sun) is about one year (within 1e-4)."""
    period = with_units(kepler.orbital_period_s)(q(1.0, "au"), q(GM_SUN.value, "m^3 s^-2"))
    assert period.to_value("yr") == pytest.approx(1.0, rel=1e-4)


def test_distance_modulus_from_pc_is_not_wrapped_as_a_length():
    wrapped = with_units(magnitudes.distance_modulus_from_pc)
    assert wrapped.return_unit is None
    assert wrapped(q(10.0, "pc")) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------------------
# Overhead (measured; the numbers are quoted in the Validity section of with_units)
# --------------------------------------------------------------------------------------


def _microseconds(call, number=20_000, repeat=5):
    return statistics.median(timeit.repeat(call, number=number, repeat=repeat)) / number * 1e6


def test_overhead_per_call_is_tens_of_microseconds():
    """Loose upper bounds (10x the measured values, to survive a slow CI machine); the numbers are printed."""
    plain = basics.schwarzschild_radius_m
    wrapped = with_units(plain)
    quantity = q(1.0, "Msun")
    t_plain = _microseconds(lambda: plain(2.0e30))
    t_plain_args = _microseconds(lambda: wrapped(2.0e30))
    t_quantity = _microseconds(lambda: wrapped(quantity))
    print(
        f"\nplain call {t_plain:.1f} us; wrapped, plain argument {t_plain_args:.1f} us; "
        f"wrapped, Quantity argument {t_quantity:.1f} us (overhead {t_quantity - t_plain:.1f} us)"
    )
    assert t_plain_args - t_plain < 200.0
    assert t_quantity - t_plain < 400.0


def test_overhead_is_independent_of_array_size():
    x = np.linspace(1e30, 1e31, 200_000)
    plain = basics.schwarzschild_radius_m
    wrapped = with_units(plain)
    qx = q(x / MSUN, "Msun")
    t_plain = _microseconds(lambda: plain(x), number=50, repeat=5)
    t_wrapped = _microseconds(lambda: wrapped(qx), number=50, repeat=5)
    print(f"\n200 000 elements: plain {t_plain / 1e3:.2f} ms, wrapped with conversion {t_wrapped / 1e3:.2f} ms")
    assert t_wrapped < 5.0 * t_plain + 2000.0  # one extra pass over the array for the conversion
