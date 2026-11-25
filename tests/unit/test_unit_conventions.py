"""Convention test: every unit suffix in the package parses with the grammar of kozmos.core.units.

The test walks every public function of every public module under ``kozmos`` (the same walk as
``tests/test_source_rule.py``) and checks, for every parameter name and every function name:

1. a trailing unit suffix parses (``unit_from_suffix``); the suffixes that are recognised but cannot be
   represented (offset, epoch, logarithmic, context-dependent units) are listed in ``UNSUPPORTED`` with the
   reason, and the list must match exactly (a new one fails, a stale one fails);
2. a name that looks like a physical quantity but carries no unit suffix is listed in ``NO_UNIT_SUFFIX`` with
   the reason (this is the allowlist asked for in the task); the list must match exactly;
3. the unit in square brackets in the docstring of a parameter, and in the Returns section or summary line of a
   function, agrees in dimension and scale with the suffix; the disagreements found are listed in
   ``DOCSTRING_DISAGREEMENTS``;
4. the same unit is not spelled two ways, and a suffix agrees in dimension with the quantity word before it;
5. inconsistencies that need a human reading (not computable from names) are pinned in ``KNOWN_FINDINGS`` with
   a phrase of evidence that must still be in the source; when the code is fixed the pin fails and is removed.

Nothing is renamed by this test; it only records.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from collections import defaultdict

import pytest

import kozmos
from kozmos.core.units import (
    IMPLICIT_INVERSE_STEMS,
    SUFFIX_UNITS,
    UNSUPPORTED_SUFFIXES,
    Dimension,
    Unit,
    UnitError,
    UnitSuffixError,
    UnsupportedSuffixError,
    _suffix_tokens,
    unit_from_suffix,
)


def _public_functions():
    for info in pkgutil.walk_packages(kozmos.__path__, "kozmos."):
        if info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        module = importlib.import_module(info.name)
        for name, obj in vars(module).items():
            if name.startswith("_") or not inspect.isfunction(obj) or obj.__module__ != module.__name__:
                continue
            yield module.__name__.removeprefix("kozmos."), obj


FUNCTIONS = list(_public_functions())


def _names() -> dict[str, list[str]]:
    """Identifier -> where it occurs (``module.function`` for a function name, ``module.function(param)``)."""
    found: dict[str, list[str]] = defaultdict(list)
    for module, function in FUNCTIONS:
        found[function.__name__].append(f"{module}.{function.__name__}()")
        for parameter in inspect.signature(function).parameters:
            found[parameter].append(f"{module}.{function.__name__}({parameter})")
    return found


NAMES = _names()


def _group(reason: str, *names: str) -> dict[str, str]:
    return dict.fromkeys(names, reason)


# --------------------------------------------------------------------------------------
# 1. suffixes that are recognised but cannot be represented
# --------------------------------------------------------------------------------------

UNSUPPORTED: dict[str, str] = {
    **_group(
        "offset unit: degree Celsius has a shifted zero point (temperature_c of the refraction model)", "temperature_c"
    ),
    **_group(
        "epoch: a Julian date is an instant with a shifted zero point (the function returns a date, not an interval)",
        "calendar_to_jd",
        "calendar_to_mjd",
        "transit_jd",
    ),
    **_group(
        "logarithmic quantity: magnitude (mag) or log10 of a cgs quantity (dex)",
        "absolute_bolometric_magnitude_mag",
        "distance_modulus_mag",
        "log_surface_gravity_cgs",
    ),
    **_group(
        "context-dependent unit: radii in units of the gravitational radius G M / c^2 (depends on the mass)",
        "kerr_ergosphere_radius_rg",
        "kerr_horizon_radius_rg",
        "kerr_isco_radius_rg",
        "kerr_photon_orbit_radius_rg",
    ),
    **_group(
        "context-dependent unit: coefficient in SI units whose dimension depends on the polytropic index n",
        "k_si",
        "polytrope_k_coefficient_si",
    ),
    **_group(
        "context-dependent unit: coefficient in SI units (white-dwarf K; Pa m^4 kg^-4/3 for mu_e fixed)",
        "white_dwarf_k_coefficient_si",
    ),
}


def test_every_unit_suffix_in_the_package_parses():
    failures = []
    flagged: set[str] = set()
    for name in sorted(NAMES):
        try:
            unit_from_suffix(name)
        except UnsupportedSuffixError:
            flagged.add(name)
        except UnitSuffixError as error:
            failures.append(f"{name}: {error}")
    assert not failures, "\n".join(failures)
    unexpected = flagged - set(UNSUPPORTED)
    stale = set(UNSUPPORTED) - flagged
    assert not unexpected, f"unsupported suffix not in the allowlist (add with a reason): {sorted(unexpected)}"
    assert not stale, f"allowlisted unsupported names that no longer exist: {sorted(stale)}"


def test_unsupported_tokens_have_the_kinds_the_allowlist_assumes():
    kinds = {token: kind for token, (kind, _) in UNSUPPORTED_SUFFIXES.items()}
    assert kinds == {
        "c": "offset",
        "jd": "epoch",
        "mjd": "epoch",
        "mag": "logarithmic",
        "cgs": "logarithmic",
        "rg": "context",
        "si": "context",
    }


def test_a_substantial_part_of_the_package_carries_suffixes():
    """Guard against a grammar that silently parses nothing."""
    with_unit = [n for n in NAMES if _has_unit(n)]
    assert len(with_unit) > 150
    assert len(FUNCTIONS) > 100


def _has_unit(name: str) -> bool:
    try:
        return unit_from_suffix(name) is not None
    except UnsupportedSuffixError:
        return True


# --------------------------------------------------------------------------------------
# 2. names that look like physical quantities but carry no unit suffix
# --------------------------------------------------------------------------------------

QUANTITY_WORDS = {
    "mass", "radius", "distance", "length", "wavelength", "luminosity", "temperature", "velocity", "speed",
    "density", "frequency", "pressure", "energy", "time", "timescale", "opacity", "flux", "width", "size",
    "period", "rate", "angle", "anomaly", "obliquity", "separation", "longitude", "latitude", "altitude",
    "azimuth", "magnitude", "gravity", "humidity", "noise", "amplitude", "modulus", "dispersion", "jd",
    "times", "wavenumber", "fwhm",
}  # fmt: skip

NO_UNIT_SUFFIX: dict[str, str] = {
    **_group(
        "dimensionless by definition: a ratio, fraction, redshift or coefficient (the quantity word names the "
        "numerator, not the unit)",
        "chabrier_mass_fraction_above",
        "kroupa_mass_fraction_above",
        "salpeter_mass_fraction_above",
        "lane_emden_central_to_mean_density",
        "lane_emden_mass_coefficient",
        "kerr_isco_specific_energy",
        "normalized_mass_to_flux_ratio",
        "virial_parameter_from_surface_density",
        "relative_humidity",
        "signal_to_noise",
        "equivalent_width_sigma_vollmann_eversberg",
        "resolving_power_from_velocity",
        "half_amplitude_wavenumber",
    ),
    **_group(
        "solar masses, documented [M_sun] in the docstring (mass_msun is the only suffixed spelling)",
        "chabrier_mean_mass",
        "kroupa_mean_mass",
        "salpeter_mean_mass",
    ),
    **_group(
        "Msun yr^-1 kpc^-2, documented in the docstring; the function name states no unit",
        "amplitude",
        "free_fall_sfr_surface_density",
        "kennicutt_orbital_sfr_surface_density",
        "kennicutt_schmidt_sfr_surface_density",
        "life_cycle_sfr_surface_density",
    ),
    **_group(
        "angle in radians, documented [rad] in the docstring; the function name states no unit",
        "angular_separation",
        "earth_rotation_angle",
        "eccentric_anomaly",
        "true_anomaly",
        "horizontal_to_hour_angle",
        "hour_angle_to_horizontal",
        "mean_obliquity",
        "observed_altitude",
    ),
    **_group(
        "time scale returned as a sidereal angle in radians / hours, documented in the docstring",
        "greenwich_mean_sidereal_time",
        "local_mean_sidereal_time",
    ),
    **_group(
        "magnitude (logarithmic quantity) with no suffix; the system has no logarithmic units",
        "absolute_bolometric_magnitude",
        "absolute_magnitude",
        "apparent_magnitude",
        "best_absolute_magnitude",
        "catalog_magnitude",
        "distance_modulus",
        "distance_modulus_from_pc",
        "magnitude",
        "vega_magnitude_from_flux",
    ),
    **_group(
        "Julian date: an instant with a shifted zero point (offset-like); the two-part jd1/jd2 convention of ERFA",
        "jd_start",
        "jd_to_calendar",
    ),
    **_group(
        "spectral radiance / flux with a unit fixed by the docstring (W m^-2 Hz^-1 sr^-1, W m^-3 sr^-1, W m^-2)",
        "blackbody_flux",
        "planck_frequency",
        "planck_wavelength",
    ),
    **_group(
        "unit chosen by the caller: any consistent unit of the sampled quantity (frequency 1 / time unit of times)",
        "frequency",
        "frequency_grid",
        "max_frequency",
        "times",
        "autocorrelation_time",
    ),
    **_group(
        "image / detector quantity in counts or pixels, not a physical unit of the SI",
        "flux",
        "flux_sigma",
        "noise",
        "gain_and_read_noise",
        "radius",
        "filter_size",
        "max_group_size",
        "size",
        "fit_psf_width",
        "smoothing_fwhm",
        "mean_continuum_flux",
        "mean_line_flux",
        "template_flux",
    ),
    **_group(
        "spectroscopy (another module): wavelength / velocity with the unit chosen by the caller or fixed in the docstring",
        "delta_wavelength",
        "doppler_shifted_wavelength",
        "equivalent_width",
        "rebin_log_wavelength",
        "radial_velocity_cross_correlation",
        "radial_velocity_from_wavelengths",
        "wavelength",
        "wavelength_observed",
        "wavelength_rest",
        "highest_wavenumber",
        "velocity_resolution_km_s_value",
    ),
}


def _looks_physical(name: str) -> bool:
    return bool(QUANTITY_WORDS & set(name.split("_")))


def test_names_that_look_physical_but_carry_no_unit_are_all_explained():
    unsuffixed = {n for n in NAMES if _looks_physical(n) and not _has_unit(n)}
    unexplained = unsuffixed - set(NO_UNIT_SUFFIX)
    stale = set(NO_UNIT_SUFFIX) - unsuffixed
    assert not unexplained, (
        "names that look like physical quantities and have no unit suffix and no entry in NO_UNIT_SUFFIX:\n"
        + "\n".join(f"  {n}: {NAMES[n][:2]}" for n in sorted(unexplained))
    )
    assert not stale, f"NO_UNIT_SUFFIX entries that are gone or now carry a suffix: {sorted(stale)}"
    assert all(reason.strip() for reason in NO_UNIT_SUFFIX.values())


# --------------------------------------------------------------------------------------
# 3. suffix against the unit written in the docstring
# --------------------------------------------------------------------------------------


def _brackets(text: str) -> list[tuple[str, Unit]]:
    parsed = []
    for content in re.findall(r"\[([^\]\[]+)\]", text):
        try:
            parsed.append((content.strip(), Unit.parse(content.strip())))
        except UnitError:
            pass
    return parsed


def _parameter_docs(doc: str) -> dict[str, str]:
    if "Parameters\n" not in doc:
        return {}
    section = doc.split("Parameters\n", 1)[1]
    for stop in ("\nReturns\n", "\nReferences\n", "\nRaises\n", "\nValidity\n"):
        section = section.split(stop, 1)[0]
    docs: dict[str, str] = {}
    current = None
    for line in section.splitlines():
        match = re.match(r"^(\w+) : ", line)
        if match:
            current = match.group(1)
            docs[current] = line
        elif current and line.startswith("    "):
            docs[current] += "\n" + line
    return docs


def _return_doc(doc: str) -> str:
    summary = doc.split("\n\n")[0]
    returns = doc.split("\nReturns\n", 1)[1].split("\nReferences\n", 1)[0] if "\nReturns\n" in doc else ""
    return summary + "\n" + returns


DOCSTRING_DISAGREEMENTS: dict[tuple[str, str], str] = {}


def _docstring_scan():
    checked = 0
    disagreements = {}
    for module, function in FUNCTIONS:
        doc = inspect.getdoc(function) or ""
        qualified = f"{module}.{function.__name__}"
        parameter_docs = _parameter_docs(doc)
        for parameter in inspect.signature(function).parameters:
            try:
                unit = unit_from_suffix(parameter)
            except UnsupportedSuffixError:
                continue
            brackets = _brackets(parameter_docs.get(parameter, ""))
            if unit is None or not brackets:
                continue
            checked += 1
            if not any(b == unit for _, b in brackets):
                disagreements[(qualified, parameter)] = f"suffix {unit}, docstring {[t for t, _ in brackets]}"
        try:
            unit = unit_from_suffix(function.__name__)
        except UnsupportedSuffixError:
            continue
        brackets = _brackets(_return_doc(doc))
        if unit is not None and brackets:
            checked += 1
            if not any(b == unit for _, b in brackets):
                disagreements[(qualified, "return")] = f"suffix {unit}, docstring {[t for t, _ in brackets]}"
    return checked, disagreements


def test_docstring_units_agree_with_the_suffixes():
    checked, disagreements = _docstring_scan()
    assert checked > 150, "the docstring scan covers too few names; the bracket syntax may have changed"
    assert set(disagreements) == set(DOCSTRING_DISAGREEMENTS), disagreements
    assert all(reason.strip() for reason in DOCSTRING_DISAGREEMENTS.values())


# --------------------------------------------------------------------------------------
# 4. one spelling per unit; the suffix matches the quantity word
# --------------------------------------------------------------------------------------

SPELLED_OUT = {
    "hour", "hours", "secs", "second", "seconds", "meter", "metre", "meters", "degree", "degrees", "kilogram",
    "parsec", "solar", "kelvin", "watt", "year", "years", "micron", "jansky", "radian", "radians", "arcseconds",
    "minutes", "msol", "msolar", "solmass", "ha", "yrs", "hrs", "deg2",
}  # fmt: skip


def _used_tokens() -> set[str]:
    tokens: set[str] = set()
    for name in NAMES:
        _, suffix = _suffix_tokens(name)
        tokens |= {re.sub(r"\d+$", "", t) for t in suffix}
    return tokens


def test_no_unit_is_spelled_two_ways():
    used = sorted(t for t in _used_tokens() if t in SUFFIX_UNITS)
    by_unit: dict[str, list[str]] = defaultdict(list)
    for token in used:
        unit = Unit.parse(SUFFIX_UNITS[token])
        key = f"{unit.dimension}|{unit.scale:.12g}"
        by_unit[key].append(token)
    duplicates = {k: v for k, v in by_unit.items() if len(v) > 1}
    assert not duplicates, f"the same unit under two suffix spellings: {duplicates}"


def test_no_spelled_out_unit_names_end_an_identifier():
    offenders = [n for n in NAMES if "_" in n and n.split("_")[-1] in SPELLED_OUT]
    assert not offenders, offenders


L, M, T = Dimension.of(L=1), Dimension.of(M=1), Dimension.of(T=1)
SKY = Dimension.of(ANGLE=1)
PIX = Dimension.of(PIXEL=1)
# head noun (the last word of the stem) -> dimensions that a suffix after it may have
HEAD_NOUN_DIMENSIONS = {
    "mass": {M},
    "radius": {L, SKY, PIX},  # a search or fit radius on the sky is an angle, in an image it is in pixels
    "distance": {L},
    "length": {L},
    "wavelength": {L},
    "luminosity": {Dimension.of(M=1, L=2, T=-3)},
    "temperature": {Dimension.of(Theta=1)},
    "velocity": {Dimension.of(L=1, T=-1), Dimension.of(ANGLE=1, T=-1)},  # angular velocity
    "speed": {Dimension.of(L=1, T=-1)},
    "pressure": {Dimension.of(M=1, L=-1, T=-2)},
    "opacity": {Dimension.of(L=2, M=-1)},
    "timescale": {T},
    "time": {T},
    "energy": {Dimension.of(M=1, L=2, T=-2)},
    "frequency": {Dimension.of(T=-1)},
}


def test_suffix_agrees_in_dimension_with_the_head_noun_in_front_of_it():
    """The last word of the stem (``mass``, ``radius``, ``distance`` ...) fixes the dimension of the suffix."""
    problems = []
    for name in sorted(NAMES):
        try:
            unit = unit_from_suffix(name)
        except UnsupportedSuffixError:
            continue
        stem, _ = _suffix_tokens(name)
        if unit is None or not stem:
            continue
        allowed = HEAD_NOUN_DIMENSIONS.get(stem[-1])
        if allowed and unit.dimension not in allowed:
            problems.append(f"{name}: '{stem[-1]}' with suffix of dimension {unit.dimension} ({NAMES[name][0]})")
    assert not problems, "\n".join(problems)


STRONG_TOKENS = {
    "km", "kpc", "mpc", "msun", "hz", "deg", "arcsec", "kg", "pc", "yr", "gyr", "jy", "um", "angstrom", "mas",
    "uas", "rad", "mm", "cm", "nm", "lsun", "rsun", "au",
}  # fmt: skip
EMBEDDED_UNIT: dict[str, str] = {
    "velocity_resolution_km_s_value": (
        "spectroscopy: the unit km s^-1 sits in the middle of the name and the name ends in 'value'; the grammar "
        "reads no suffix, so with_units treats the parameter as dimensionless (pass units={...})"
    ),
}


def test_no_unit_hides_in_the_middle_of_a_name_the_grammar_cannot_explain():
    embedded = {
        n for n in NAMES if not _has_unit(n) and "from" not in n.split("_") and STRONG_TOKENS & set(n.split("_")[1:])
    }
    assert embedded == set(EMBEDDED_UNIT), sorted(embedded ^ set(EMBEDDED_UNIT))


def test_the_number_and_column_density_exception_is_used_only_where_listed():
    hits = sorted(
        n for n in NAMES if "_".join(_suffix_tokens(n)[0]).endswith(IMPLICIT_INVERSE_STEMS) and _suffix_tokens(n)[1]
    )
    assert hits == ["column_density_m2", "number_density_m3"], hits


# --------------------------------------------------------------------------------------
# 5. findings that need a human reading
# --------------------------------------------------------------------------------------

KNOWN_FINDINGS = [
    (
        "radiation.magnitudes.distance_modulus_from_pc",
        "def distance_modulus_from_pc",
        "the suffix _pc names the unit of the argument while every other 'X_from_Y' function names the unit of "
        "the result (pc_from_distance_modulus, flux_jy_from_vega_magnitude); the grammar needs the 'from' rule",
    ),
    (
        "star_formation.clouds.number_density_m3",
        "m⁻³",
        "_m3 means m^-3 here and in column_density_m2 (m^-2) but m^3 in comoving_volume_mpc3: the grammar needs "
        "the number_density / column_density exception",
    ),
]


def _source_text(qualified: str) -> str:
    module_name, function_name = qualified.rsplit(".", 1)
    module = importlib.import_module(f"kozmos.{module_name}")
    return inspect.getsource(module.__dict__[function_name]) if function_name in module.__dict__ else ""


@pytest.mark.parametrize(("qualified", "evidence", "finding"), KNOWN_FINDINGS, ids=[k[0] for k in KNOWN_FINDINGS])
def test_known_findings_are_still_present(qualified, evidence, finding):
    """When the code is corrected the evidence disappears and this pin must be removed with the finding."""
    assert evidence in _source_text(qualified), f"no longer in {qualified}: {evidence!r} ({finding})"


# --------------------------------------------------------------------------------------
# vocabulary bookkeeping
# --------------------------------------------------------------------------------------

KNOWN_MIXED_UNITS = {
    "angle": "rad, deg and arcsec spellings of the same quantity word",
    "dec": "dec_deg (catalogue functions) against dec_rad (coordinate transforms)",
    "depletion_time": "s (clouds) against yr (laws)",
    "distance": "m (black holes, ephemeris) against pc (radiation)",
    "free_fall_time": "s (clouds, stars) against yr (laws)",
    "mass": "kg everywhere against msun in stars.imf and star_formation.laws",
    "ra": "ra_deg against ra_rad",
    "sound_speed": "m_s (clouds, black holes) against km_s (laws)",
    "surface_density": "kg_m2 (clouds) against msun_pc2 (laws)",
    "velocity": "m_s against km_s",
    "wavelength": "m, um and angstrom",
}


def test_mixed_unit_systems_for_the_same_quantity_word_are_recorded():
    """SI in one module, astronomical units in another: legitimate, but a unit-aware caller must convert."""
    by_stem: dict[str, dict[str, Unit]] = defaultdict(dict)
    for name in NAMES:
        try:
            unit = unit_from_suffix(name)
        except UnsupportedSuffixError:
            continue
        if unit is not None:
            by_stem["_".join(_suffix_tokens(name)[0])][name] = unit
    mixed = {
        stem
        for stem, variants in by_stem.items()
        if len({u.dimension for u in variants.values()}) == 1
        and len({round(u.scale, 12) for u in variants.values()}) > 1
    }
    assert set(KNOWN_MIXED_UNITS) <= mixed, sorted(set(KNOWN_MIXED_UNITS) - mixed)
