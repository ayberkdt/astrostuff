"""Tests of kozmos.relativity.weak_field.

Published numbers come from Will's review [Will2014] (TeX source read in this project): the solar-limb deflection
1.7505 arcsec (Eq. E29), the Mercury perihelion shift 42.98 arcsec per century (Eq. E34), the Shapiro delay
(240 - 20 ln(d^2/r)) microseconds (Eq. E32), the Gravity Probe B geodetic precession 6.63 arcsec per year and the
Earth-Moon value "about 2 arcseconds per century" (section 'Geodetic precession').

Inputs that Will does not print are taken from the sources named at each use:
 * Mercury orbital elements: semi-major axis 0.38709927 au, eccentricity 0.20563593 and mean-longitude rate
   149472.67411175 degrees per century, Table 1 (1800 AD - 2050 AD) of the JPL page 'Approximate Positions of the
   Planets' (https://ssd.jpl.nasa.gov/planets/approx_pos.html, read 2026-11-26);
 * G M_sun, solar radius, G M_earth, Earth equatorial radius, astronomical unit: kozmos.core.constants (nominal
   values of Prsa et al. 2016 and the ERFA definitions).
The values of Will are rounded (three or four digits), which sets the lower limit of every comparison.
"""

from __future__ import annotations

import warnings
from decimal import Decimal, getcontext

import numpy as np
import pytest

from kozmos.core.constants import (
    ASTRONOMICAL_UNIT,
    EARTH_EQUATORIAL_RADIUS,
    GM_EARTH,
    GM_SUN,
    GRAVITATIONAL_CONSTANT,
    JULIAN_YEAR_SECONDS,
    SOLAR_RADIUS,
    SPEED_OF_LIGHT,
)
from kozmos.relativity.weak_field import (
    geodetic_precession_circular_rad_s,
    geodetic_precession_vector_rad_s,
    light_deflection_angle_rad,
    perihelion_advance_per_orbit_rad,
    perihelion_advance_rate_rad_s,
    schwarzschild_clock_ratio,
    shapiro_round_trip_delay_s,
    static_clock_ratio,
)

G = GRAVITATIONAL_CONSTANT.value
C = SPEED_OF_LIGHT.value
M_SUN = GM_SUN.value / G
M_EARTH = GM_EARTH.value / G
AU = ASTRONOMICAL_UNIT.value
R_SUN = SOLAR_RADIUS.value
ARCSEC = 180.0 / np.pi * 3600.0
CENTURY = 36525.0 * 86400.0

# Mercury: JPL approximate-positions table 1, 1800 AD - 2050 AD
MERCURY_A = 0.38709927 * AU
MERCURY_E = 0.20563593
MERCURY_LONGITUDE_RATE_DEG_PER_CENTURY = 149472.67411175


# --------------------------------------------------------------------------------------
# light deflection
# --------------------------------------------------------------------------------------


def test_solar_limb_deflection_is_1_7505_arcsec():
    # Eq. E29 prints 1.''7505 for (1 + gamma)/2 = 1. Computed 1.75119 arcsec (nominal solar radius 6.957e8 m): difference
    # +3.9e-4 relative, from the rounding of the printed value and the unknown radius used there; tolerance 1e-3.
    assert light_deflection_angle_rad(M_SUN, R_SUN) * ARCSEC == pytest.approx(1.7505, rel=1e-3, abs=0)


def test_deflection_closed_form_scaling_with_gamma_angle_and_distance():
    # geometric mass 1 m: d = 1000 m gives 4 / 1000 for gamma = 1, Phi = 0 (Eq. E28 in G = c = 1)
    mass = C**2 / G
    assert light_deflection_angle_rad(mass, 1000.0) == pytest.approx(4.0e-3, rel=1e-14, abs=0)
    assert light_deflection_angle_rad(mass, 2000.0) == pytest.approx(2.0e-3, rel=1e-14, abs=0)
    assert light_deflection_angle_rad(mass, 1000.0, gamma_ppn=0.0) == pytest.approx(
        2.0e-3, rel=1e-14, abs=0
    )  # Newtonian half
    assert light_deflection_angle_rad(mass, 1000.0, incoming_angle_rad=np.pi / 2) == pytest.approx(
        2.0e-3, rel=1e-14, abs=0
    )
    assert light_deflection_angle_rad(mass, 1000.0, incoming_angle_rad=np.pi) == pytest.approx(0.0, abs=1e-18)
    assert light_deflection_angle_rad(mass, 1000.0, gamma_ppn=1.5, incoming_angle_rad=0.0) == pytest.approx(
        5.0e-3, rel=1e-14, abs=0
    )
    # array input
    d = np.array([1e3, 2e3, 4e3])
    assert light_deflection_angle_rad(mass, d) == pytest.approx(4.0 / d, rel=1e-14, abs=0)


def test_deflection_rejects_nonpositive_inputs():
    with pytest.raises(ValueError):
        light_deflection_angle_rad(0.0, 1.0)
    with pytest.raises(ValueError):
        light_deflection_angle_rad(1.0, -1.0)


# --------------------------------------------------------------------------------------
# perihelion shift
# --------------------------------------------------------------------------------------


def test_mercury_perihelion_advance_is_42_98_arcsec_per_century():
    # Eq. E34: 42.''98 for gamma = beta = 1 (J_2 term dropped). Computed 42.9805 (difference 5e-4 arcsec, 1.1e-5
    # relative; the printed value has four digits, so +-0.005); tolerance 0.005 arcsec per century.
    rate = perihelion_advance_rate_rad_s(M_SUN, MERCURY_A, MERCURY_E) * CENTURY * ARCSEC
    assert rate == pytest.approx(42.98, abs=0.005)


def test_mercury_advance_per_orbit_times_orbits_per_century_with_the_period_from_the_jpl_longitude_rate():
    # The period from the mean-longitude rate of the JPL table is 87.96926 d, Kepler's third law with the nominal G M_sun
    # gives 87.96947 d (relative difference 2.4e-6); the per-orbit advance times the orbits per century agrees with the
    # rate function to 2.4e-6.
    period_jpl = 360.0 / MERCURY_LONGITUDE_RATE_DEG_PER_CENTURY * CENTURY
    per_orbit = perihelion_advance_per_orbit_rad(M_SUN, MERCURY_A, MERCURY_E)
    assert per_orbit == pytest.approx(
        5.0187e-7, rel=1e-4, abs=0
    )  # 6 pi G M / (c^2 a (1 - e^2)), see the closed form below
    assert per_orbit * CENTURY / period_jpl * ARCSEC == pytest.approx(42.98, abs=0.005)
    kepler_period = 2.0 * np.pi * np.sqrt(MERCURY_A**3 / GM_SUN.value)
    assert kepler_period == pytest.approx(period_jpl, rel=1e-5, abs=0)  # measured 2.4e-6


def test_advance_per_orbit_closed_form_and_the_gr_value_of_the_ppn_factor():
    a, e = 3.0e10, 0.3
    expected = 6.0 * np.pi * G * M_SUN / (C**2 * a * (1.0 - e**2))
    assert perihelion_advance_per_orbit_rad(M_SUN, a, e) == pytest.approx(expected, rel=1e-14, abs=0)
    # PPN factor (2 + 2 gamma - beta) / 3: 1 in GR, 2/3 for gamma = beta = 0, 1/3 + ... for (gamma, beta) = (0.5, 1.5)
    assert perihelion_advance_per_orbit_rad(M_SUN, a, e, gamma_ppn=0.0, beta_ppn=0.0) == pytest.approx(
        expected * 2.0 / 3.0, rel=1e-14, abs=0
    )
    assert perihelion_advance_per_orbit_rad(M_SUN, a, e, gamma_ppn=0.5, beta_ppn=1.5) == pytest.approx(
        expected * (2.0 + 1.0 - 1.5) / 3.0, rel=1e-14, abs=0
    )


def test_rate_equals_carrolls_apsidal_frequency_formula():
    # Carroll Eq. 7.56: omega_a = 3 (G M)^(3/2) / (c^2 (1 - e^2) a^(5/2)); the rate function is the same expression
    for a, e in [(MERCURY_A, MERCURY_E), (1.0e11, 0.0), (2.0e9, 0.7)]:
        expected = 3.0 * GM_SUN.value**1.5 / (C**2 * (1.0 - e**2) * a**2.5)
        got = perihelion_advance_rate_rad_s(M_SUN, a, e)
        assert got == pytest.approx(
            expected, rel=1e-12, abs=0
        )  # G M_sun = G * (GM_sun / G) to rounding, measured 2e-16


def test_quadrupole_term_matches_the_coefficient_printed_in_eq_e34():
    # Eq. E34: the J_2 term is 42.98 * 3e-4 * (J_2 / 1e-7) arcsec per century. The coefficient is printed with one digit;
    # the computed value is 2.955e-4, which rounds to 3e-4. Tolerance: the range that rounds to 3e-4.
    base = perihelion_advance_rate_rad_s(M_SUN, MERCURY_A, MERCURY_E)
    with_j2 = perihelion_advance_rate_rad_s(M_SUN, MERCURY_A, MERCURY_E, quadrupole_j2=1e-7, body_radius_m=R_SUN)
    coefficient = (with_j2 - base) / base
    assert 2.5e-4 < coefficient < 3.5e-4
    assert coefficient == pytest.approx(2.955e-4, rel=2e-3, abs=0)
    doubled = perihelion_advance_rate_rad_s(M_SUN, MERCURY_A, MERCURY_E, quadrupole_j2=2e-7, body_radius_m=R_SUN)
    assert (doubled - base) / (with_j2 - base) == pytest.approx(2.0, rel=1e-12, abs=0)  # linear in J_2


def test_perihelion_input_checks():
    with pytest.raises(ValueError):
        perihelion_advance_per_orbit_rad(M_SUN, MERCURY_A, 1.0)
    with pytest.raises(ValueError):
        perihelion_advance_per_orbit_rad(M_SUN, MERCURY_A, -0.1)
    with pytest.raises(ValueError):
        perihelion_advance_per_orbit_rad(-1.0, MERCURY_A, 0.2)


# --------------------------------------------------------------------------------------
# Shapiro delay
# --------------------------------------------------------------------------------------


def _conjunction_geometry(d_solar_radii: float, target_au: float):
    """Earth at 1 au and a target at target_au on opposite sides, the straight line missing the Sun by d."""
    d = d_solar_radii * R_SUN
    r_earth, r_target = AU, target_au * AU
    theta_e = np.arcsin(d / r_earth)
    theta_t = np.arcsin(d / r_target)
    earth = np.array([r_earth * np.cos(theta_e), r_earth * np.sin(theta_e), 0.0])
    target = np.array([-r_target * np.cos(theta_t), r_target * np.sin(theta_t), 0.0])
    return earth, target


def test_shapiro_delay_closed_form_at_superior_conjunction():
    # (r_E + x_E.n)(r_e - x_e.n) / d^2 -> 4 r_E r_e / d^2 for d << r: the delay is 4 G M / c^3 ln(4 r_E r_e / d^2)
    # (2 (1 + gamma) = 4 for gamma = 1). Measured relative difference -2.1e-5 for d = 5 R_sun, tolerance 2e-4.
    earth, target = _conjunction_geometry(5.0, 1.52)
    delay = shapiro_round_trip_delay_s(M_SUN, earth, target)
    d = 5.0 * R_SUN
    approx_delay = 4.0 * G * M_SUN / C**3 * np.log(4.0 * AU * 1.52 * AU / d**2)
    assert delay == pytest.approx(approx_delay, rel=2e-4, abs=0)


@pytest.mark.parametrize("d_solar_radii", [1.0, 2.0, 5.0, 10.0, 20.0])
@pytest.mark.parametrize("target_au", [0.72, 1.52])
def test_shapiro_delay_matches_eq_e32_to_its_rounding(d_solar_radii, target_au):
    # Eq. E32 for gamma = 1: 240 - 20 ln(d^2 / r) microseconds with d in solar radii, r in au. The printed numbers have
    # two digits; the computed delay differs by -0.48 % ... +0.62 % over d = 1..20 R_sun, r = 0.72 and 1.52 au (the 20 is
    # 19.70 = 4 G M_sun / c^3 in microseconds); tolerance 1 %.
    earth, target = _conjunction_geometry(d_solar_radii, target_au)
    delay_us = shapiro_round_trip_delay_s(M_SUN, earth, target) * 1e6
    printed = 240.0 - 20.0 * np.log(d_solar_radii**2 / target_au)
    assert delay_us == pytest.approx(printed, rel=1e-2, abs=0)


def test_shapiro_delay_scales_with_gamma_and_is_zero_outside_the_logarithm_ratio():
    earth, target = _conjunction_geometry(3.0, 1.0)
    full = shapiro_round_trip_delay_s(M_SUN, earth, target)
    assert shapiro_round_trip_delay_s(M_SUN, earth, target, gamma_ppn=0.0) == pytest.approx(
        0.5 * full, rel=1e-14, abs=0
    )
    assert shapiro_round_trip_delay_s(2.0 * M_SUN, earth, target) == pytest.approx(2.0 * full, rel=1e-14, abs=0)
    # the ray passes the Sun on the same side of a target in front of the Sun: the delay is much smaller
    earth_side = np.array([AU, 0.0, 0.0])
    near = np.array([0.5 * AU, 0.3 * AU, 0.0])
    assert shapiro_round_trip_delay_s(M_SUN, earth_side, near) < 0.5 * full


def test_shapiro_delay_identity_for_the_logarithm_argument():
    # d^2 = r_E^2 - (x_E.n)^2 gives (r_E + x_E.n)(r_e - x_e.n)/d^2 = (r_e - x_e.n)/(r_E - x_E.n); the function must equal
    # the delay computed with the second form (an algebraic identity, so the agreement is rounding)
    earth, target = _conjunction_geometry(4.0, 0.72)
    n = (earth - target) / np.linalg.norm(earth - target)
    argument = (np.linalg.norm(target) - target @ n) / (np.linalg.norm(earth) - earth @ n)
    assert shapiro_round_trip_delay_s(M_SUN, earth, target) == pytest.approx(
        4.0 * G * M_SUN / C**3 * np.log(argument),
        rel=1e-11,
        abs=0,  # measured 6e-15; 1.2e-13 with the norms moved by one unit in the last place (r - x.n cancels)
    )


def test_shapiro_delay_rejects_a_line_through_the_mass_and_wrong_shapes():
    with pytest.raises(ValueError):
        shapiro_round_trip_delay_s(M_SUN, np.array([AU, 0.0, 0.0]), np.array([-AU, 0.0, 0.0]))
    with pytest.raises(ValueError):
        shapiro_round_trip_delay_s(M_SUN, np.array([AU, 0.0]), np.array([-AU, 1.0, 0.0]))


def _printed_form(earth, source):
    """Eq. E31 as printed (n from the source to the Earth), in double precision."""
    n = (earth - source) / np.linalg.norm(earth - source)
    r_e, r_s = np.linalg.norm(earth), np.linalg.norm(source)
    return np.log((r_e + earth @ n) * (r_s - source @ n) / (r_e**2 - (earth @ n) ** 2))


def _printed_form_exact(earth, source):
    """Eq. E31 as printed, evaluated with 60 significant digits (decimal module) from the double inputs."""
    getcontext().prec = 60
    xe = [Decimal(float(v)) for v in earth]
    xs = [Decimal(float(v)) for v in source]
    chord = sum((a - b) ** 2 for a, b in zip(xe, xs, strict=True)).sqrt()
    n = [(a - b) / chord for a, b in zip(xe, xs, strict=True)]
    r_e, r_s = sum(a * a for a in xe).sqrt(), sum(a * a for a in xs).sqrt()
    e_n, s_n = sum(a * b for a, b in zip(xe, n, strict=True)), sum(a * b for a, b in zip(xs, n, strict=True))
    return ((r_e + e_n) * (r_s - s_n) / (r_e * r_e - e_n * e_n)).ln()


_SHAPIRO_PREFACTOR = 4.0 * G * M_SUN / C**3  # 2 (1 + gamma) G M / c^3 for gamma = 1


def test_shapiro_delay_collinear_same_side_is_finite_and_equals_ln_2():
    # Earth at 1 au, reflector at 0.5 au on the same side of the Sun, collinear with it: the segment does not touch
    # the Sun, d^2 of the printed form is 0 (0/0) but (r_E + r_e + R)/(r_E + r_e - R) = 2 / 1. Before the fix this raised
    # ValueError. Measured: 1.3656361e-5 s, relative difference to 4 G M / c^3 ln 2 is 0, tolerance 1e-14.
    earth = np.array([AU, 0.0, 0.0])
    source = np.array([0.5 * AU, 0.0, 0.0])
    delay = shapiro_round_trip_delay_s(M_SUN, earth, source)
    assert delay == pytest.approx(_SHAPIRO_PREFACTOR * np.log(2.0), rel=1e-14, abs=0)
    assert delay == pytest.approx(1.3656e-5, rel=1e-4, abs=0)  # value quoted by the audit; measured 1.36564e-5 s


def test_shapiro_delay_has_no_cancellation_near_the_collinear_geometry():
    # A transverse offset of 1e-6 au: the printed form in double precision cancels (measured error against the 60-digit
    # value -2.19e-5); the function agrees with the 60-digit value to rounding (measured 2.2e-16, tolerance 1e-14).
    earth = np.array([AU, 0.0, 0.0])
    source = np.array([0.5 * AU, 1e-6 * AU, 0.0])
    reference = _SHAPIRO_PREFACTOR * float(_printed_form_exact(earth, source))
    assert shapiro_round_trip_delay_s(M_SUN, earth, source) == pytest.approx(reference, rel=1e-14, abs=0)
    assert abs(_printed_form(earth, source) * _SHAPIRO_PREFACTOR / reference - 1.0) > 1e-6  # the cancellation is real


def test_shapiro_delay_equals_the_printed_form_for_generic_geometries():
    # The printed Eq. E31 form is the reference for generic geometry; the two agree to rounding. 300 random pairs with
    # both distances in [0.3 au, ...): worst relative difference measured 4.0e-14, tolerance 4e-13.
    rng = np.random.default_rng(20260926)
    worst = 0.0
    count = 0
    while count < 300:
        earth, source = rng.normal(size=3) * AU, rng.normal(size=3) * AU
        if min(np.linalg.norm(earth), np.linalg.norm(source)) < 0.3 * AU:
            continue
        count += 1
        printed = _SHAPIRO_PREFACTOR * _printed_form(earth, source)
        worst = max(worst, abs(shapiro_round_trip_delay_s(M_SUN, earth, source) / printed - 1.0))
    assert worst < 4e-13


def test_shapiro_delay_rejects_coincident_positions_without_a_numpy_warning():
    # Before the fix: a divide warning (NaN) followed by the misleading "passes through the mass" message.
    point = np.array([AU, 0.2 * AU, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="coincide"):
            shapiro_round_trip_delay_s(M_SUN, point, point)


def test_shapiro_delay_rejects_only_a_segment_that_contains_the_mass():
    # opposite sides: r_E + r_e = R exactly (R = 3 au + 1 au) -> rejected with the "through the mass" message
    with pytest.raises(ValueError, match="through the mass"):
        shapiro_round_trip_delay_s(M_SUN, np.array([3.0 * AU, 0.0, 0.0]), np.array([-AU, 0.0, 0.0]))


# --------------------------------------------------------------------------------------
# gravitational time dilation
# --------------------------------------------------------------------------------------


def test_schwarzschild_clock_ratio_at_the_solar_surface():
    # sqrt(1 - 2 G M / (R c^2)) = 1 - x/2 - x^2/8 with x = 2 G M / (R c^2) = 4.2453e-6: closed form series
    x = 2.0 * GM_SUN.value / (R_SUN * C**2)
    assert schwarzschild_clock_ratio(R_SUN, M_SUN) == pytest.approx(1.0 - x / 2.0 - x**2 / 8.0, rel=1e-15, abs=0)
    assert 1.0 - schwarzschild_clock_ratio(R_SUN, M_SUN) == pytest.approx(
        2.1225e-6, rel=1e-4, abs=0
    )  # measured 2.12250e-6
    assert schwarzschild_clock_ratio(1e30, M_SUN) == pytest.approx(1.0, abs=1e-15)


def test_schwarzschild_clock_ratio_horizon_and_frequency_shift():
    rs = 2.0 * G * M_SUN / C**2
    with pytest.raises(ValueError):
        schwarzschild_clock_ratio(rs, M_SUN)
    # Eq. 7.61: omega_2 / omega_1 = ratio of the clock rates of the emitter and the receiver
    r1, r2 = 3.0 * rs, 40.0 * rs
    shift = np.sqrt((1.0 - rs / r1) / (1.0 - rs / r2))
    assert schwarzschild_clock_ratio(r1, M_SUN) / schwarzschild_clock_ratio(r2, M_SUN) == pytest.approx(
        shift, rel=1e-14, abs=0
    )


def test_static_clock_ratio_weak_field_agrees_with_the_exact_ratio_and_with_eq_7_62():
    phi = -GM_SUN.value / R_SUN
    # Hand computation from literal constants (not from the code under test): G M_sun = 1.3271244e20 m^3 s^-2
    # (Prsa et al. 2016 nominal value, as in kozmos.core.constants), R_sun = 6.957e8 m (same source), c = 299792458 m/s
    # (SI definition). x = 2 G M / (R c^2) = 4.245005140290713e-6 and sqrt(1 - x) = 1 - x/2 - x^2/8 - x^3/16 - 5 x^4/128
    # (binomial series; the next term is 7e-27) = 0.9999978774951773. Measured relative difference 0, tolerance 1e-15.
    x_hand = 2.0 * 1.3271244e20 / (6.957e8 * 299792458.0**2)
    hand = 1.0 - x_hand / 2.0 - x_hand**2 / 8.0 - x_hand**3 / 16.0 - 5.0 * x_hand**4 / 128.0
    assert static_clock_ratio(-1.3271244e20 / 6.957e8) == pytest.approx(hand, rel=1e-15, abs=0)
    # and the exact Schwarzschild value for Phi = -G M / r is the same function of the potential (a consistency check,
    # not a source of the reference)
    assert static_clock_ratio(phi) == pytest.approx(schwarzschild_clock_ratio(R_SUN, M_SUN), rel=1e-15, abs=0)
    # two clocks: omega_2/omega_1 = 1 + Phi_1 - Phi_2 (c = 1) to second order in Phi / c^2; the neglected term is of order
    # (Phi_1/c^2)^2 / 2 = 2.3e-12 for the solar surface (measured difference -2.3e-12), tolerance 3e-11
    phi_1, phi_2 = -GM_SUN.value / R_SUN, -GM_SUN.value / AU
    ratio = static_clock_ratio(phi_1) / static_clock_ratio(phi_2)
    assert ratio == pytest.approx(1.0 + (phi_1 - phi_2) / C**2, abs=3e-11)
    assert static_clock_ratio(0.0) == 1.0
    with pytest.raises(ValueError):
        static_clock_ratio(-0.6 * C**2)


# --------------------------------------------------------------------------------------
# geodetic precession
# --------------------------------------------------------------------------------------


def test_gravity_probe_b_geodetic_precession_is_6_63_arcsec_per_year():
    # Will: 6.63 arcsec per year (6630 mas/yr) for an orbit at 642 km altitude. Computed with the nominal Earth equatorial
    # radius 6378.1 km and G M_earth: 6.6211 arcsec per year (difference -1.3e-3 relative: the printed value has three
    # digits and was computed for the real orbit); tolerance 3e-3.
    r = EARTH_EQUATORIAL_RADIUS.value + 642e3
    rate = geodetic_precession_circular_rad_s(M_EARTH, r) * JULIAN_YEAR_SECONDS.value * ARCSEC
    assert rate == pytest.approx(6.63, rel=3e-3, abs=0)


def test_earth_moon_geodetic_precession_is_about_two_arcsec_per_century():
    # Will: "about 2 arcseconds per century" (de Sitter precession of the Earth-Moon system in the field of the Sun)
    rate = geodetic_precession_circular_rad_s(M_SUN, AU) * JULIAN_YEAR_SECONDS.value * 100.0 * ARCSEC
    assert rate == pytest.approx(2.0, abs=0.1)  # computed 1.9188; the printed "about 2" is rounded to one digit


def test_earth_moon_geodetic_precession_tight_hand_computation_from_constants():
    # Circular orbit, gamma = 1: Omega = (3/2) (G M)^(3/2) / (c^2 r^(5/2)) (de Sitter; (gamma + 1/2) v x grad U / c^2
    # with v = sqrt(G M / r)). Constants typed here: G M_sun = 1.3271244e20 m^3 s^-2 (Prsa et al. 2016 nominal value),
    # r = 1 au = 149597870700 m (IAU 2012 exact definition), c = 299792458 m/s (SI definition), one Julian year =
    # 365.25 * 86400 s (IAU), arcsec = 180 * 3600 / pi. Hand result 1.918814 arcsec per century (the source's "about 2"
    # is 4.2 % larger, one rounded digit); measured relative difference to the code 0, tolerance 1e-12.
    gm, r, c = 1.3271244e20, 149597870700.0, 299792458.0
    omega = 1.5 * gm**1.5 / (c**2 * r**2.5)
    hand = omega * 365.25 * 86400.0 * 100.0 * 180.0 / np.pi * 3600.0
    assert hand == pytest.approx(1.918814, abs=1e-6)
    rate = geodetic_precession_circular_rad_s(M_SUN, AU) * JULIAN_YEAR_SECONDS.value * 100.0 * ARCSEC
    assert rate == pytest.approx(hand, rel=1e-12, abs=0)


def test_geodetic_vector_form_matches_the_circular_orbit_formula_and_points_along_the_orbit_normal():
    r = 7.0e6
    speed = np.sqrt(GM_EARTH.value / r)
    position = np.array([r, 0.0, 0.0])
    velocity = np.array([0.0, speed, 0.0])
    grad_u = -GM_EARTH.value / r**2 * position / r  # U = G M / r increases towards the mass
    omega = geodetic_precession_vector_rad_s(velocity, grad_u)
    assert omega[0] == 0.0 and omega[1] == 0.0
    assert omega[2] > 0.0  # prograde: along r x v
    assert omega[2] == pytest.approx(geodetic_precession_circular_rad_s(M_EARTH, r), rel=1e-13, abs=0)  # measured 0
    # gamma dependence: (gamma + 1/2)
    assert geodetic_precession_vector_rad_s(velocity, grad_u, gamma_ppn=0.0)[2] == pytest.approx(
        omega[2] / 3.0, rel=1e-14, abs=0
    )
    # a retrograde orbit reverses the vector
    assert geodetic_precession_vector_rad_s(-velocity, grad_u)[2] == pytest.approx(-omega[2], rel=1e-14, abs=0)
    with pytest.raises(ValueError):
        geodetic_precession_vector_rad_s(velocity[:2], grad_u)


# --------------------------------------------------------------------------------------
# mutation-testing survivors: argument order of the PPN parameters, the factor gamma + 1/2, the clock boundary
# --------------------------------------------------------------------------------------


def test_perihelion_advance_rate_has_the_ppn_parameters_in_the_right_places():
    """Hand value of the rate: advance / period = (2 + 2 gamma - beta) (G M)^(3/2) / (c^2 a^(5/2) (1 - e^2)) with
    gamma = 2, beta = 1/2 (factor 5.5; swapped, gamma = 1/2 and beta = 2 would give 1.0), plus the J_2 term
    3 pi J_2 R^2 / p^2 divided by the period (J_2 = 1e-3, R = 7e8 m, not exchangeable with each other)."""
    gm, a, e, gamma, beta, j2, radius = 1.32712440018e20, 5.79e10, 0.2056, 2.0, 0.5, 1e-3, 7.0e8
    mass = gm / G
    p = a * (1.0 - e**2)
    period = 2.0 * np.pi * np.sqrt(a**3 / gm)
    post_newtonian = (2.0 + 2.0 * gamma - beta) * gm**1.5 / (C**2 * a**2.5 * (1.0 - e**2))
    quadrupole = 3.0 * np.pi * j2 * radius**2 / p**2 / period
    got = perihelion_advance_rate_rad_s(mass, a, e, gamma, beta, j2, radius)
    assert got == pytest.approx(post_newtonian + quadrupole, rel=1e-12, abs=0)
    swapped = perihelion_advance_rate_rad_s(mass, a, e, beta, gamma, j2, radius)
    assert abs(got / swapped - 1.0) > 0.4  # measured 0.451: the two orders are far apart
    # the rate is the per-orbit advance over the period for the same arguments
    per_orbit = perihelion_advance_per_orbit_rad(mass, a, e, gamma, beta, j2, radius)
    assert got == pytest.approx(per_orbit / period, rel=1e-14, abs=0)


def test_geodetic_precession_vector_uses_gamma_plus_one_half():
    """v = (0, 7000, 0) m/s, grad U = (-4, 0, 0) m/s^2: v x grad U = (0, 0, 28000), so for gamma = 2 the vector is
    (0, 0, 2.5 * 28000 / c^2) = 7.79e-13 rad/s; gamma + 1 would give 3 * 28000 / c^2."""
    v, grad = [0.0, 7000.0, 0.0], [-4.0, 0.0, 0.0]
    expected = 2.5 * 28000.0 / C**2
    got = geodetic_precession_vector_rad_s(v, grad, gamma_ppn=2.0)
    assert got[2] == pytest.approx(expected, rel=1e-15, abs=0)
    assert got[0] == 0.0 and got[1] == 0.0
    default = geodetic_precession_vector_rad_s(v, grad)  # gamma = 1: factor 1.5
    assert default[2] == pytest.approx(1.5 * 28000.0 / C**2, rel=1e-15, abs=0)
    batch = geodetic_precession_vector_rad_s([v, v], [grad, grad], gamma_ppn=0.0)  # gamma = 0: factor 1/2
    assert batch[1, 2] == pytest.approx(0.5 * 28000.0 / C**2, rel=1e-15, abs=0)
    circular = geodetic_precession_circular_rad_s(M_EARTH, 7.0e6, gamma_ppn=2.0)
    assert circular == pytest.approx(2.5 * (G * M_EARTH) ** 1.5 / (C**2 * (7.0e6) ** 2.5), rel=1e-14, abs=0)


def test_static_clock_ratio_boundary_phi_equal_minus_half_c_squared_raises():
    """The argument 1 + 2 Phi / c^2 must be strictly positive; at Phi = -c^2 / 2 it is exactly 0 (the arithmetic is
    exact there), at Phi = -0.6 c^2 it is negative."""
    for phi in (-0.5 * C**2, -0.6 * C**2, [-1.0, -0.5 * C**2]):
        with pytest.raises(ValueError, match="Phi > -c"):
            static_clock_ratio(phi)
    just_above = static_clock_ratio(-0.5 * C**2 * (1.0 - 1e-6))
    assert just_above == pytest.approx(1e-3, rel=1e-6, abs=0)  # sqrt(1e-6)


def test_shapiro_delay_rejects_a_line_through_the_mass_whatever_the_last_bits_of_the_norms():
    """Opposite sides of the mass with the end points moved by a few units in the last place.

    The exact test r_E + r_e - R > 0 saw the line through the mass only when the three norms rounded alike, which
    they do on one platform and not on another; then a delay of a path through the centre was returned.
    """
    earth, source = np.array([3.0 * AU, 0.0, 0.0]), np.array([-AU, 0.0, 0.0])
    for k_earth in range(-3, 4):
        for k_source in range(-3, 4):
            e, s = earth.copy(), source.copy()
            for _ in range(abs(k_earth)):
                e[0] = np.nextafter(e[0], np.inf if k_earth > 0 else -np.inf)
            for _ in range(abs(k_source)):
                s[0] = np.nextafter(s[0], np.inf if k_source > 0 else -np.inf)
            with pytest.raises(ValueError, match="through the mass"):
                shapiro_round_trip_delay_s(M_SUN, e, s)
    # an impact parameter of one solar radius is far from the limit
    grazing = shapiro_round_trip_delay_s(M_SUN, np.array([AU, 6.957e8, 0.0]), np.array([-AU, 6.957e8, 0.0]))
    assert 2.0e-4 < grazing < 3.0e-4  # about 240 microseconds for a round trip at superior conjunction


def test_shapiro_delay_behind_the_mass_against_50_digit_arithmetic():
    """Earth at 1.496e11 m, reflector at 2.28e11 m on the far side, impact parameters from the solar radius down.

    With r_E + r_e - R formed by subtraction the delay was 1.4e-4 off for a ray 3e4 m from the centre.
    """
    from decimal import Decimal, getcontext

    getcontext().prec = 50
    r_earth, r_source = 1.496e11, 2.28e11
    for impact in (7e8, 1e7, 1e6, 3e5, 3e4):
        earth = np.array([-np.sqrt(r_earth**2 - impact**2), impact, 0.0])
        source = np.array([np.sqrt(r_source**2 - impact**2), impact, 0.0])
        e = [Decimal(float(c)) for c in earth]
        s = [Decimal(float(c)) for c in source]
        norm_e = sum(c * c for c in e).sqrt()
        norm_s = sum(c * c for c in s).sqrt()
        chord = sum((a - b) ** 2 for a, b in zip(e, s, strict=True)).sqrt()
        exact = (
            4
            * Decimal(G)
            * Decimal(M_SUN)
            / Decimal(C) ** 3
            * ((norm_e + norm_s + chord) / (norm_e + norm_s - chord)).ln()
        )
        value = shapiro_round_trip_delay_s(M_SUN, earth, source)
        assert abs(Decimal(float(value)) / exact - 1) < Decimal("1e-13")  # measured 2e-16
