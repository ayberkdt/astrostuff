"""Tests of kozmos.black_holes.kerr.

Numbers come from closed forms, limits, the formulas that were read in the sources (re-typed here independently of
the module) and from the independently sourced functions of kozmos.black_holes.basics. Tolerances were measured:
the comment above each one gives the largest deviation found.
"""

from __future__ import annotations

import numpy as np
import pytest

from kozmos.black_holes import basics, kerr
from kozmos.core.constants import BOLTZMANN, GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT

G = GRAVITATIONAL_CONSTANT.value
C = SPEED_OF_LIGHT.value
KB = BOLTZMANN.value
MSUN = 2.0e30  # kg; a round number, only a mass scale (every test is independent of its exact value)
M10 = 10.0 * MSUN
SPINS = [-0.9, -0.5, 0.0, 0.3, 0.7, 0.95, 0.998]


def _derivative(f, x, h=1e-5):
    """Central derivative of a scalar function."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


# --------------------------------------------------------------------------------------
# circular geodesics
# --------------------------------------------------------------------------------------


def test_schwarzschild_limits_of_the_orbit():
    x = np.array([4.0, 6.0, 10.0, 50.0])
    # measured: energy exact, angular momentum and Omega 2.4e-16 and 1.7e-16 relative
    energy = kerr.kerr_circular_orbit_specific_energy(x, 0.0)
    assert energy == pytest.approx((1 - 2 / x) / np.sqrt(1 - 3 / x), rel=1e-14, abs=0)
    ang_mom = kerr.kerr_circular_orbit_specific_angular_momentum_rg(x, 0.0)
    assert ang_mom == pytest.approx(np.sqrt(x) / np.sqrt(1 - 3 / x), rel=1e-14, abs=0)
    # Kepler's law, Omega^2 = G M / r^3 (Abramowicz2013, Schwarzschild summary), with r = x G M / c^2
    omega = kerr.kerr_circular_orbit_angular_velocity_rad_s(M10, x, 0.0)
    r = x * G * M10 / C**2
    assert omega == pytest.approx(np.sqrt(G * M10 / r**3), rel=1e-14, abs=0)
    # the vertical epicyclic frequency equals the orbital one, Omega^2 = (omega_theta*)^2
    assert kerr.kerr_vertical_epicyclic_frequency_hz(M10, x, 0.0) == pytest.approx(
        kerr.kerr_circular_orbit_frequency_hz(M10, x, 0.0), rel=1e-14, abs=0
    )
    # the radial one is Omega sqrt(1 - 6/x)
    assert kerr.kerr_radial_epicyclic_frequency_hz(M10, 10.0, 0.0) == pytest.approx(
        kerr.kerr_circular_orbit_frequency_hz(M10, 10.0, 0.0) * np.sqrt(1 - 6 / 10.0), rel=1e-14, abs=0
    )


def test_frequency_equals_the_isco_frequency_of_basics():
    # basics.isco_orbital_frequency_hz uses Omega^2 = G M / r^3 at r = 6 G M / c^2; measured relative difference 2.6e-16
    assert kerr.kerr_circular_orbit_frequency_hz(M10, 6.0, 0.0) == pytest.approx(
        float(basics.isco_orbital_frequency_hz(M10)), rel=1e-12, abs=0
    )
    # the frequency scales as 1/M
    assert kerr.kerr_circular_orbit_frequency_hz(2 * M10, 8.0, 0.5) == pytest.approx(
        0.5 * kerr.kerr_circular_orbit_frequency_hz(M10, 8.0, 0.5), rel=1e-14, abs=0
    )


@pytest.mark.parametrize("spin", SPINS)
def test_energy_at_the_isco_equals_basics(spin):
    r_isco = float(basics.kerr_isco_radius_rg(spin))
    # measured: the largest relative difference over SPINS is 9.3e-15; the bound leaves a factor 1e3
    assert kerr.kerr_circular_orbit_specific_energy(r_isco, spin) == pytest.approx(
        float(basics.kerr_isco_specific_energy(spin)), rel=1e-11, abs=0
    )


@pytest.mark.parametrize("spin", SPINS)
def test_energy_and_angular_momentum_are_extremal_at_the_isco(spin):
    r_isco = float(basics.kerr_isco_radius_rg(spin))
    # central derivative with h = 1e-5 (measured |dE/dr| <= 9.3e-10 and |dL/dr| <= 1.8e-9 over SPINS and -1, the
    # worst at spin 0.998; the truncation error is O(h^2) times the third derivative; the slope away from the ISCO,
    # at r_isco + 1, is above 1.1e-3)
    h = 1e-5
    de = _derivative(lambda x: float(kerr.kerr_circular_orbit_specific_energy(x, spin)), r_isco, h)
    dl = _derivative(lambda x: float(kerr.kerr_circular_orbit_specific_angular_momentum_rg(x, spin)), r_isco, h)
    assert abs(de) < 1e-8
    assert abs(dl) < 1e-8
    far = r_isco + 1.0
    assert abs(_derivative(lambda x: float(kerr.kerr_circular_orbit_specific_energy(x, spin)), far, h)) > 1e-3


@pytest.mark.parametrize("spin", SPINS)
def test_angular_momentum_over_energy_is_the_keplerian_l_of_the_source(spin):
    # l_K = M^{1/2} (r^2 - 2 a M^{1/2} r^{1/2} + a^2) / (r^{3/2} - 2 M r^{1/2} + a M^{1/2}), M = 1 (Abramowicz2013,
    # TeX label keplerian-momentum-kerr; l = L / E, TeX label constant-motion-specific). Retyped here.
    x = np.array([8.0, 12.0, 30.0, 100.0])
    x = x[x > float(basics.kerr_isco_radius_rg(spin))]
    l_k = (x**2 - 2 * spin * np.sqrt(x) + spin**2) / (x**1.5 - 2 * np.sqrt(x) + spin)
    ratio = kerr.kerr_circular_orbit_specific_angular_momentum_rg(x, spin) / kerr.kerr_circular_orbit_specific_energy(
        x, spin
    )
    # measured largest relative difference: 3.0e-16
    assert ratio == pytest.approx(l_k, rel=1e-12, abs=0)


@pytest.mark.parametrize("spin", SPINS)
def test_dE_equals_omega_dL_along_circular_orbits(spin):
    # for circular geodesics dE = Omega dL (a property of the effective potential extremum, see the effective-potential
    # discussion of Abramowicz2013). Finite differences with h = 1e-5 at r = r_isco + 3; measured relative 2.5e-9 (worst over SPINS).
    x0 = float(basics.kerr_isco_radius_rg(spin)) + 3.0
    h = 1e-5
    de = _derivative(lambda x: float(kerr.kerr_circular_orbit_specific_energy(x, spin)), x0, h)
    dl = _derivative(lambda x: float(kerr.kerr_circular_orbit_specific_angular_momentum_rg(x, spin)), x0, h)
    omega = float(kerr.kerr_circular_orbit_angular_velocity_rad_s(1.0 / (G / C**3), x0, spin))  # c^3/(G M) = 1
    assert de == pytest.approx(omega * dl, rel=1e-6, abs=0)


@pytest.mark.parametrize("spin", SPINS)
def test_energy_diverges_at_the_photon_orbit(spin):
    r_ph = float(basics.kerr_photon_orbit_radius_rg(spin))
    values = [float(kerr.kerr_circular_orbit_specific_energy(r_ph * (1 + eps), spin)) for eps in (1e-3, 1e-6, 1e-9)]
    # E grows as the square root of the distance to the photon orbit: each step of 1e-3 in the distance raises E by
    # about sqrt(1e3) = 31.6 (measured factors 31.2 to 31.6 for all spins in SPINS)
    assert values[1] / values[0] == pytest.approx(31.6, rel=0.1, abs=0)
    assert values[2] / values[1] == pytest.approx(31.6, rel=0.1, abs=0)
    assert values[2] > 1e3


def test_no_circular_orbit_inside_the_photon_orbit():
    with pytest.raises(ValueError):
        kerr.kerr_circular_orbit_specific_energy(3.0, 0.0)
    with pytest.raises(ValueError):
        kerr.kerr_circular_orbit_angular_velocity_rad_s(M10, 2.9, 0.0)
    with pytest.raises(ValueError):
        kerr.kerr_circular_orbit_specific_energy(3.9, -1.0)  # retrograde photon orbit of the extreme hole is at 4
    with pytest.raises(ValueError):
        kerr.kerr_circular_orbit_specific_energy(-1.0, 0.0)


def test_orbital_frequency_of_prograde_and_retrograde_orbits():
    # at the same Boyer-Lindquist radius the prograde orbit is SLOWER: Omega = 1 / (x^{3/2} + a_*); the prograde orbit
    # is stable down to a smaller radius, where it is faster
    x = 10.0
    assert kerr.kerr_circular_orbit_frequency_hz(M10, x, 0.9) < kerr.kerr_circular_orbit_frequency_hz(M10, x, -0.9)
    assert kerr.kerr_circular_orbit_frequency_hz(M10, float(basics.kerr_isco_radius_rg(0.9)), 0.9) > (
        kerr.kerr_circular_orbit_frequency_hz(M10, float(basics.kerr_isco_radius_rg(-0.9)), -0.9)
    )
    # Omega = c^3 / (G M) / (x^{3/2} + a_*): closed form for a_* = 1, x = 4 is 1/9 in units of c^3/(G M)
    omega = kerr.kerr_circular_orbit_angular_velocity_rad_s(M10, 4.0, 1.0)
    assert omega == pytest.approx(C**3 / (G * M10) / 9.0, rel=1e-14, abs=0)


# --------------------------------------------------------------------------------------
# marginally bound orbit
# --------------------------------------------------------------------------------------


def test_marginally_bound_radius_limits():
    assert float(kerr.kerr_marginally_bound_radius_rg(0.0)) == pytest.approx(4.0, rel=1e-14, abs=0)
    assert float(kerr.kerr_marginally_bound_radius_rg(1.0)) == pytest.approx(1.0, rel=1e-14, abs=0)
    assert float(kerr.kerr_marginally_bound_radius_rg(-1.0)) == pytest.approx(
        3.0 + 2.0 * np.sqrt(2.0), rel=1e-14, abs=0
    )


@pytest.mark.parametrize("spin", [-1.0, *SPINS, 0.9999])
def test_energy_is_one_at_the_marginally_bound_orbit(spin):
    r_mb = float(kerr.kerr_marginally_bound_radius_rg(spin))
    # measured: |E - 1| below 4.2e-13 for all spins (the largest at spin 0.9999, where r_mb is close to the photon
    # orbit and the metric terms cancel); the bound 1e-9 leaves a factor of 2e3
    assert float(kerr.kerr_circular_orbit_specific_energy(r_mb, spin)) == pytest.approx(1.0, abs=1e-9)
    # orbits between r_mb and the ISCO are bound, orbits between the photon orbit and r_mb are unbound
    r_ph = float(basics.kerr_photon_orbit_radius_rg(spin))
    r_isco = float(basics.kerr_isco_radius_rg(spin))
    assert float(kerr.kerr_circular_orbit_specific_energy(0.5 * (r_mb + r_isco), spin)) < 1.0
    assert float(kerr.kerr_circular_orbit_specific_energy(0.5 * (r_mb + r_ph), spin)) > 1.0


@pytest.mark.parametrize("spin", [-0.7, 0.0, 0.5, 0.9])
def test_radii_are_ordered(spin):
    assert (
        float(basics.kerr_photon_orbit_radius_rg(spin))
        < float(kerr.kerr_marginally_bound_radius_rg(spin))
        < float(basics.kerr_isco_radius_rg(spin))
    )


# --------------------------------------------------------------------------------------
# epicyclic and precession frequencies
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("spin", [-1.0, *SPINS, 0.9999])
def test_radial_epicyclic_frequency_vanishes_at_the_isco(spin):
    r_isco = float(basics.kerr_isco_radius_rg(spin))
    f_orbit = float(kerr.kerr_circular_orbit_frequency_hz(M10, r_isco, spin))
    f_radial = float(kerr.kerr_radial_epicyclic_frequency_hz(M10, r_isco, spin))
    # measured: f_r / f_orbit below 2.1e-8 at the ISCO radius of basics (square root of the roundoff ~1e-16 in the
    # bracket, which sits well inside the 1e-12 clip of the module)
    assert f_radial / f_orbit < 1e-6
    outside = float(kerr.kerr_radial_epicyclic_frequency_hz(M10, r_isco * 1.5, spin))
    assert outside > 0.0


@pytest.mark.parametrize("spin", [-0.9, 0.0, 0.5, 0.9])
def test_radial_epicyclic_frequency_is_nan_inside_the_isco(spin):
    r_isco = float(basics.kerr_isco_radius_rg(spin))
    r_inside = 0.5 * (r_isco + float(basics.kerr_photon_orbit_radius_rg(spin)))
    assert np.isnan(kerr.kerr_radial_epicyclic_frequency_hz(M10, r_inside, spin))
    assert np.isnan(kerr.kerr_periastron_precession_frequency_hz(M10, r_inside, spin))
    # the vertical frequency stays real
    assert np.isfinite(kerr.kerr_vertical_epicyclic_frequency_hz(M10, r_inside, spin))


def test_radial_epicyclic_frequency_has_a_maximum_outside_the_isco():
    # the radial epicyclic frequency of the Schwarzschild geometry, f_r^2 = f^2 (1 - 6/x), has its maximum at x = 8
    x = np.linspace(6.5, 20.0, 2701)
    f = kerr.kerr_radial_epicyclic_frequency_hz(M10, x, 0.0)
    assert x[np.argmax(f)] == pytest.approx(8.0, abs=0.01)


def test_vertical_epicyclic_frequency_expansion_and_positivity():
    # (omega_theta*)^2 = Omega^2 (1 - 4 a x^{-3/2} + 3 a^2 x^{-2}); square root to second order in x^{-1/2}:
    # 1 - 2 a x^{-3/2} + (3/2) a^2 x^{-2} (the next term is -2 a^2 x^{-3}); measured residual 1e-12 at x = 1e4
    x = 1.0e4
    for spin in (-0.9, 0.9):
        f_orbit = float(kerr.kerr_circular_orbit_frequency_hz(M10, x, spin))
        f_theta = float(kerr.kerr_vertical_epicyclic_frequency_hz(M10, x, spin))
        expansion = 1.0 - 2.0 * spin * x**-1.5 + 1.5 * spin**2 * x**-2
        assert f_theta / f_orbit == pytest.approx(expansion, abs=1e-10)
    # the bracket 1 - 4 a x^-3/2 + 3 a^2 x^-2 is positive on a grid down to the photon orbit
    for spin in SPINS + [1.0, -1.0]:
        r_ph = float(basics.kerr_photon_orbit_radius_rg(spin))
        x = r_ph + np.logspace(-6, 2, 200)
        assert np.all(np.isfinite(kerr.kerr_vertical_epicyclic_frequency_hz(M10, x, spin)))


def test_periastron_precession_weak_field_and_consistency():
    # Schwarzschild: Omega - Omega sqrt(1 - 6/x) -> 3 Omega / x = 3 x^{-5/2} (units c^3/(G M)) for large x
    x = 1.0e4
    unit = C**3 / (G * M10) / (2.0 * np.pi)
    value = float(kerr.kerr_periastron_precession_frequency_hz(M10, x, 0.0)) / unit
    # measured relative deviation from 3 x^{-5/2}: 1.5e-4 = 1.5 / x (the next term of the expansion)
    assert value == pytest.approx(3.0 * x**-2.5, rel=5e-4, abs=0)
    # difference of the two source frequencies
    x2, a = 12.0, 0.6
    assert float(kerr.kerr_periastron_precession_frequency_hz(M10, x2, a)) == pytest.approx(
        float(kerr.kerr_circular_orbit_frequency_hz(M10, x2, a) - kerr.kerr_radial_epicyclic_frequency_hz(M10, x2, a)),
        rel=1e-13,
        abs=0,
    )


def test_nodal_precession_is_the_lense_thirring_frequency():
    unit = C**3 / (G * M10) / (2.0 * np.pi)
    assert float(kerr.kerr_nodal_precession_frequency_hz(M10, 10.0, 0.0)) == pytest.approx(0.0, abs=1e-14)
    x = 1.0e4
    for spin in (0.5, 0.9):
        value = float(kerr.kerr_nodal_precession_frequency_hz(M10, x, spin)) / unit
        # Omega - omega_theta -> 2 a x^{-3/2} Omega = 2 a x^{-3} (1 - (3/4) a x^{-1/2}) for large x, where the second
        # term is the expansion of Omega = 1/(x^{3/2} + a); measured relative deviation of the leading term alone:
        # 3.8e-3 (a = 0.5) and 6.8e-3 (a = 0.9); with the correction included: 2e-6 and 6e-6
        assert value == pytest.approx(2.0 * spin * x**-3 * (1.0 - 0.75 * spin * x**-0.5), rel=1e-4, abs=0)
    # sign reverses with the sign of the spin (counted along the direction of the orbit)
    assert float(kerr.kerr_nodal_precession_frequency_hz(M10, 8.0, 0.7)) > 0.0
    assert float(kerr.kerr_nodal_precession_frequency_hz(M10, 8.0, -0.7)) < 0.0


# --------------------------------------------------------------------------------------
# frame dragging
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("spin", [-0.9, 0.0, 0.3, 0.7, 0.99, 1.0])
def test_frame_dragging_at_the_horizon_is_the_horizon_angular_velocity(spin):
    r_plus = float(basics.kerr_horizon_radius_rg(spin))
    omega = float(kerr.frame_dragging_angular_velocity_rad_s(M10, r_plus, spin))
    # measured relative difference: 2.2e-16; the horizon angular velocity is the same at any angle (rigid rotation)
    assert omega == pytest.approx(float(basics.horizon_angular_velocity_rad_s(M10, spin)), rel=1e-12, abs=0.0)
    omega_pole = float(kerr.frame_dragging_angular_velocity_rad_s(M10, r_plus, spin, polar_angle_rad=0.3))
    # measured relative difference 0 (identical at the horizon, where the angle drops out)
    assert omega_pole == pytest.approx(omega, rel=1e-12, abs=0.0)


def test_frame_dragging_equals_minus_g_tphi_over_g_phiphi():
    # Kerr metric of Abramowicz2013 (TeX label kerr-metric), M = 1, retyped here
    a, x, theta = 0.8, 3.5, 1.1
    rho2 = x**2 + a**2 * np.cos(theta) ** 2
    g_tp = -2.0 * a * x * np.sin(theta) ** 2 / rho2
    g_pp = (x**2 + a**2 + 2.0 * a**2 * x * np.sin(theta) ** 2 / rho2) * np.sin(theta) ** 2
    omega = float(kerr.frame_dragging_angular_velocity_rad_s(M10, x, a, polar_angle_rad=theta))
    # measured relative difference: 2e-16
    assert omega == pytest.approx(C**3 / (G * M10) * (-g_tp / g_pp), rel=1e-12, abs=0)


def test_frame_dragging_far_away_and_sign():
    x = 1.0e3
    unit = C**3 / (G * M10)
    # measured relative deviation from 2 a / x^3: 3.6e-7 (next term is of order 1/x)
    assert float(kerr.frame_dragging_angular_velocity_rad_s(M10, x, 0.6)) / unit == pytest.approx(
        1.2 / x**3, rel=1e-5, abs=0
    )
    assert float(kerr.frame_dragging_angular_velocity_rad_s(M10, 5.0, -0.6)) < 0.0
    assert float(kerr.frame_dragging_angular_velocity_rad_s(M10, 5.0, 0.0)) == 0.0


# --------------------------------------------------------------------------------------
# horizon thermodynamics
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("spin", [-1.0, -0.5, 0.0, 0.4, 0.9, 1.0])
def test_horizon_area_matches_the_horizon_radius_of_basics(spin):
    r_g = G * M10 / C**2
    r_plus = float(basics.kerr_horizon_radius_rg(spin))
    expected = 4.0 * np.pi * (r_plus**2 + spin**2) * r_g**2
    # measured relative difference: 0 (exact to rounding); tolerance 1e-12
    assert float(kerr.kerr_horizon_area_m2(M10, spin)) == pytest.approx(expected, rel=1e-12, abs=0)


def test_horizon_area_limits():
    r_g = G * M10 / C**2
    assert float(kerr.kerr_horizon_area_m2(M10, 0.0)) == pytest.approx(16.0 * np.pi * r_g**2, rel=1e-14, abs=0)
    assert float(kerr.kerr_horizon_area_m2(M10, 1.0)) == pytest.approx(8.0 * np.pi * r_g**2, rel=1e-14, abs=0)
    # a_* = 0 agrees with the Schwarzschild radius of basics: 4 pi R_S^2
    assert float(kerr.kerr_horizon_area_m2(M10, 0.0)) == pytest.approx(
        4.0 * np.pi * float(basics.schwarzschild_radius_m(M10)) ** 2, rel=1e-14, abs=0
    )


def test_irreducible_mass_limits_and_the_carroll_form():
    assert float(kerr.kerr_irreducible_mass_kg(M10, 0.0)) == pytest.approx(M10, rel=1e-14, abs=0)
    assert float(kerr.kerr_irreducible_mass_kg(M10, 1.0)) == pytest.approx(M10 / np.sqrt(2.0), rel=1e-14, abs=0)
    assert float(kerr.kerr_irreducible_mass_kg(M10, -1.0)) == pytest.approx(M10 / np.sqrt(2.0), rel=1e-14, abs=0)
    # M_irr^2 = A c^4 / (16 pi G^2)
    for spin in (0.3, 0.8):
        area = float(kerr.kerr_horizon_area_m2(M10, spin))
        # measured relative difference: 0 (exact to rounding); tolerance 1e-12
        assert float(kerr.kerr_irreducible_mass_kg(M10, spin)) == pytest.approx(
            np.sqrt(area * C**4 / (16.0 * np.pi * G**2)), rel=1e-12, abs=0
        )
    # Carroll Eq. 7.149 last line, in G = c = 1 with J = M^2 a_* (retyped): M_irr = sqrt{[M^2 + sqrt(M^4 - J^2)] / 2}
    spin = 0.6
    assert float(kerr.kerr_irreducible_mass_kg(1.0, spin)) == pytest.approx(
        np.sqrt((1.0 + np.sqrt(1.0 - spin**2)) / 2.0), rel=1e-14, abs=0
    )


def test_extractable_fraction():
    assert float(kerr.kerr_penrose_extractable_fraction(0.0)) == 0.0
    # Carroll: "approximately 29%" for the extreme Kerr black hole; exact value 1 - 1/sqrt(2) = 0.29289
    assert float(kerr.kerr_penrose_extractable_fraction(1.0)) == pytest.approx(
        1.0 - 1.0 / np.sqrt(2.0), rel=1e-14, abs=0
    )
    assert round(float(kerr.kerr_penrose_extractable_fraction(1.0)), 2) == 0.29
    a = np.linspace(0.0, 1.0, 21)
    assert np.all(np.diff(kerr.kerr_penrose_extractable_fraction(a)) > 0.0)
    assert float(kerr.kerr_penrose_extractable_fraction(0.5)) == pytest.approx(
        1.0 - float(kerr.kerr_irreducible_mass_kg(1.0, 0.5)), rel=1e-14, abs=0
    )


@pytest.mark.parametrize("spin", [0.0, 0.5, 0.9, 0.99])
def test_surface_gravity_forms_agree(spin):
    r_g = G * M10 / C**2
    r_plus = float(basics.kerr_horizon_radius_rg(spin))
    area = float(kerr.kerr_horizon_area_m2(M10, spin))
    # Page2005 eq:3: kappa = 4 pi (r_+ - M) / A in geometrized units, converted with c^2 / r_G ... (retyped);
    # measured relative difference: 3.3e-16 at most
    kappa_page = 4.0 * np.pi * (r_plus - 1.0) * r_g / area * C**2
    assert float(kerr.kerr_surface_gravity_m_s2(M10, spin)) == pytest.approx(kappa_page, rel=1e-11, abs=0)
    # Carroll Eq. 7.155 with G = c = 1, M = 1, a = a_*: sqrt(1 - a^2) / [2 (1 + sqrt(1 - a^2))]
    kappa_carroll = np.sqrt(1.0 - spin**2) / (2.0 * (1.0 + np.sqrt(1.0 - spin**2)))
    assert float(kerr.kerr_surface_gravity_m_s2(M10, spin)) == pytest.approx(
        kappa_carroll * C**2 / r_g, rel=1e-12, abs=0
    )


def test_temperature_limits_and_schwarzschild_value():
    # measured: relative difference to basics at a_* = 0 is 0
    assert float(kerr.kerr_hawking_temperature_k(M10, 0.0)) == pytest.approx(
        float(basics.hawking_temperature_k(M10)), rel=1e-12, abs=0
    )
    assert float(kerr.kerr_hawking_temperature_k(M10, 1.0)) == 0.0
    a = np.array([0.0, 0.5, 0.9, 0.99, 0.9999])
    t = kerr.kerr_hawking_temperature_k(M10, a)
    assert np.all(np.diff(t) < 0.0)
    # kappa ratio = 2 s / (1 + s), s = sqrt(1 - a^2); measured agreement 1e-15
    s_root = np.sqrt(1.0 - a**2)
    assert t / t[0] == pytest.approx(2.0 * s_root / (1.0 + s_root), rel=1e-12, abs=0)
    assert float(kerr.kerr_hawking_temperature_k(M10, -0.5)) == pytest.approx(float(t[1]), rel=1e-14, abs=0)


def test_entropy_limits():
    assert float(kerr.kerr_bekenstein_hawking_entropy(M10, 0.0)) == pytest.approx(
        float(basics.bekenstein_hawking_entropy(M10)), rel=1e-12, abs=0
    )
    assert float(kerr.kerr_bekenstein_hawking_entropy(M10, 1.0)) == pytest.approx(
        0.5 * float(basics.bekenstein_hawking_entropy(M10)), rel=1e-12, abs=0
    )


@pytest.mark.parametrize("spin", [0.3, 0.7, 0.9])
def test_first_law_by_finite_differences(spin):
    """dM c^2 = T dS + Omega_H dJ with J = a_* G M^2 / c, for independent small changes of M and J."""
    m0 = M10
    j0 = spin * G * m0**2 / C

    def state(m, j):
        s = j * C / (G * m**2)
        return (
            float(kerr.kerr_hawking_temperature_k(m, s)),
            float(kerr.kerr_bekenstein_hawking_entropy(m, s)) * KB,
            float(basics.horizon_angular_velocity_rad_s(m, s)),
        )

    for dm_rel, dj_rel in ((1e-4, 0.0), (0.0, 1e-4)):
        m_up, m_dn = m0 * (1 + dm_rel), m0 * (1 - dm_rel)
        j_up, j_dn = j0 * (1 + dj_rel), j0 * (1 - dj_rel)
        t_mid, _, om_mid = state(m0, j0)
        ds = state(m_up, j_up)[1] - state(m_dn, j_dn)[1]
        dj = j_up - j_dn
        dmc2 = (m_up - m_dn) * C**2
        # measured |dM c^2 - T dS - Omega dJ| / max(|dM c^2|, |Omega dJ|) for central differences of relative step 1e-4:
        # 6e-10 (a_* = 0.3), 1.6e-8 (0.7), 2.8e-7 (0.9); the error is second order in the step and grows with spin
        # because the third derivatives of S(M, J) grow toward extremality; the tolerance 1e-6 leaves a factor 3.5
        scale = max(abs(dmc2), abs(om_mid * dj))
        assert abs(dmc2 - t_mid * ds - om_mid * dj) < 1e-6 * scale


# --------------------------------------------------------------------------------------
# area theorem
# --------------------------------------------------------------------------------------


def test_area_margin_vanishes_at_the_limit_for_equal_non_spinning_holes():
    # two Schwarzschild holes of mass m merge into a Schwarzschild hole of mass sqrt(2) m: the areas just balance
    m = 20.0 * MSUN
    # measured: the margin relative to one initial area is 8e-16
    margin = float(kerr.kerr_area_theorem_margin_m2(m, 0.0, m, 0.0, np.sqrt(2.0) * m, 0.0))
    assert abs(margin) / float(kerr.kerr_horizon_area_m2(m, 0.0)) < 1e-12
    assert float(kerr.kerr_area_theorem_margin_m2(m, 0.0, m, 0.0, 1.5 * m, 0.0)) > 0.0
    assert float(kerr.kerr_area_theorem_margin_m2(m, 0.0, m, 0.0, 1.3 * m, 0.0)) < 0.0


def test_area_margin_for_generic_merger_numbers():
    # generic numbers (not an observed event): 30 + 20 solar masses, spins 0.1 and 0.3, final 48 solar masses, spin 0.7
    margin = float(kerr.kerr_area_theorem_margin_m2(30 * MSUN, 0.1, 20 * MSUN, 0.3, 48 * MSUN, 0.7))
    a_initial = float(kerr.kerr_horizon_area_m2(30 * MSUN, 0.1) + kerr.kerr_horizon_area_m2(20 * MSUN, 0.3))
    assert margin > 0.0
    # the relative increase is a few tens of per cent for these numbers (measured 0.53)
    assert 0.2 < margin / a_initial < 0.8
    # a final mass that is too small violates the theorem
    assert float(kerr.kerr_area_theorem_margin_m2(30 * MSUN, 0.1, 20 * MSUN, 0.3, 34 * MSUN, 0.7)) < 0.0


def test_radiated_fraction_bound():
    m = 15.0 * MSUN
    # 1 - 1/sqrt(2) for two equal non-spinning holes
    assert float(kerr.merger_radiated_fraction_bound(m, 0.0, m, 0.0)) == pytest.approx(
        1.0 - 1.0 / np.sqrt(2.0), rel=1e-13, abs=0
    )
    assert float(kerr.merger_radiated_fraction_bound(m, 0.0, 3 * m, 0.0)) == pytest.approx(
        1.0 - np.sqrt(10.0) / 4.0, rel=1e-13, abs=0
    )
    # spin raises the bound: the irreducible masses are smaller
    assert float(kerr.merger_radiated_fraction_bound(m, 0.9, m, 0.9)) > float(
        kerr.merger_radiated_fraction_bound(m, 0.0, m, 0.0)
    )
    # at the bound the area margin is zero for a non-spinning final hole
    m1, m2 = 30.0 * MSUN, 20.0 * MSUN
    bound = float(kerr.merger_radiated_fraction_bound(m1, 0.5, m2, 0.2))
    final = (1.0 - bound) * (m1 + m2)
    # measured: relative to the initial area, 1.7e-16
    a_initial = float(kerr.kerr_horizon_area_m2(m1, 0.5) + kerr.kerr_horizon_area_m2(m2, 0.2))
    assert float(kerr.kerr_area_theorem_margin_m2(m1, 0.5, m2, 0.2, final, 0.0)) / a_initial == pytest.approx(
        0.0, abs=1e-10
    )


# --------------------------------------------------------------------------------------
# Kerr-Newman
# --------------------------------------------------------------------------------------


def test_kerr_newman_horizon_radius():
    for spin in (-0.8, 0.0, 0.5, 1.0):
        assert float(kerr.kerr_newman_horizon_radius_rg(spin, 0.0)) == pytest.approx(
            float(basics.kerr_horizon_radius_rg(spin)), rel=1e-14, abs=0
        )
    # Reissner-Nordstrom: r_+ = 1 + sqrt(1 - Q_*^2); extremal at Q_* = 1
    assert float(kerr.kerr_newman_horizon_radius_rg(0.0, 0.6)) == pytest.approx(1.8, rel=1e-14, abs=0)
    assert float(kerr.kerr_newman_horizon_radius_rg(0.0, 1.0)) == 1.0
    # the root of Delta = r^2 - 2 r + a^2 + Q^2
    a, q = 0.5, 0.6
    r = float(kerr.kerr_newman_horizon_radius_rg(a, q))
    assert r**2 - 2.0 * r + a**2 + q**2 == pytest.approx(0.0, abs=1e-14)
    with pytest.raises(ValueError):
        kerr.kerr_newman_horizon_radius_rg(0.8, 0.8)


# --------------------------------------------------------------------------------------
# input validation and shapes
# --------------------------------------------------------------------------------------


def test_input_validation_and_broadcasting():
    with pytest.raises(ValueError):
        kerr.kerr_circular_orbit_frequency_hz(-1.0, 10.0, 0.0)
    with pytest.raises(ValueError):
        kerr.kerr_circular_orbit_frequency_hz(M10, 10.0, 1.1)
    with pytest.raises(ValueError):
        kerr.kerr_hawking_temperature_k(M10, -1.5)
    with pytest.raises(ValueError):
        kerr.kerr_marginally_bound_radius_rg(2.0)
    x = np.array([6.0, 8.0, 10.0])
    assert kerr.kerr_circular_orbit_frequency_hz(M10, x, 0.5).shape == (3,)
    assert kerr.kerr_circular_orbit_specific_energy(x, np.array([0.1, 0.2, 0.3])).shape == (3,)
    assert kerr.kerr_horizon_area_m2(np.array([M10, 2 * M10]), 0.5).shape == (2,)


def _pugliese_energy_and_momentum(r, a, prograde):
    """E and L as printed in Pugliese, Quevedo and Ruffini (arXiv:1105.2959, Kerr_A.tex lines 284-288 and 299-316), M = 1, a >= 0.

    Typed from the TeX, independently of the module: prograde is the lower sign (L_-, E^(+)_-), retrograde the
    upper sign with the negative angular momentum (L_+, E^(-)_+).
    """
    s = -1.0 if prograde else 1.0  # the upper sign of the source is +1
    momentum = abs(a**2 + s * 2.0 * a * np.sqrt(r) + r**2) / np.sqrt(r**2 * (r - 3.0) - s * 2.0 * a * np.sqrt(r**3))
    first = (
        (r**5) ** 0.25
        * abs((a**2 + (r - 2.0) * r) * (a - s * np.sqrt(r**3)))
        / np.sqrt((r - 3.0) * np.sqrt(r) - s * 2.0 * a)
    )
    sign = 1.0 if prograde else -1.0  # E^(+) for L = +L_-, E^(-) for L = -L_+
    energy = (first + sign * 2.0 * a * r * momentum) / (r * (r**3 + a**2 * (r + 2.0)))
    return energy, momentum


@pytest.mark.parametrize("spin", [0.0, 0.1, 0.5, 0.9, 0.998, 1.0, -0.1, -0.5, -0.9, -0.998, -1.0])
def test_energy_and_angular_momentum_equal_the_printed_closed_forms(spin):
    photon = float(basics.kerr_photon_orbit_radius_rg(spin))
    radii = photon + np.concatenate([[0.05, 0.2, 0.5], np.geomspace(1.0, 1000.0, 14)])
    for radius in radii:
        energy, momentum = _pugliese_energy_and_momentum(radius, abs(spin), spin >= 0)
        # measured: at most 6e-13 (spin 1 at 0.05 above the photon orbit), below 2e-14 elsewhere
        assert float(kerr.kerr_circular_orbit_specific_energy(radius, spin)) == pytest.approx(energy, rel=1e-11, abs=0)
        assert float(kerr.kerr_circular_orbit_specific_angular_momentum_rg(radius, spin)) == pytest.approx(
            momentum, rel=1e-11, abs=0
        )


@pytest.mark.parametrize("spin", [0.0, 0.6, 1.0, -0.6, -1.0])
def test_angular_momentum_equals_the_covariant_velocity_component(spin):
    # the derivation that the module used before: L = u_phi = A (g_tphi + Omega g_phiphi), -A^-2 = g_tt + 2 Omega g_tphi + Omega^2 g_phiphi
    for radius in float(basics.kerr_photon_orbit_radius_rg(spin)) + np.array([0.3, 1.0, 5.0, 40.0]):
        omega = 1.0 / (radius**1.5 + spin)
        g_tt, g_tp = -(1.0 - 2.0 / radius), -2.0 * spin / radius
        g_pp = radius**2 + spin**2 + 2.0 * spin**2 / radius
        redshift = (-(g_tt + 2.0 * omega * g_tp + omega**2 * g_pp)) ** -0.5
        assert float(kerr.kerr_circular_orbit_specific_angular_momentum_rg(radius, spin)) == pytest.approx(
            redshift * (g_tp + omega * g_pp), rel=1e-11, abs=0
        )
        assert float(kerr.kerr_circular_orbit_specific_energy(radius, spin)) == pytest.approx(
            -redshift * (g_tt + omega * g_tp), rel=1e-11, abs=0
        )


def test_energy_stays_finite_and_accurate_just_above_the_photon_orbit():
    # close to the photon orbit E ~ (photon-orbit coefficient) / sqrt(eps): the product E sqrt(eps) tends to a constant;
    # the earlier normalisation lost 3e-3 at spin 1, eps 1e-7 and returned inf at 1e-9
    for spin in (0.5, 0.998, 1.0):
        photon = float(basics.kerr_photon_orbit_radius_rg(spin))
        products = []
        for eps in (1e-5, 1e-7):
            radius = photon * (1.0 + eps)
            energy = float(kerr.kerr_circular_orbit_specific_energy(radius, spin))
            assert np.isfinite(energy)
            reference, _ = _pugliese_energy_and_momentum(radius, spin, True)
            assert energy == pytest.approx(reference, rel=1e-6, abs=0)
            products.append(energy * np.sqrt(eps))
        if spin < 1.0:  # at spin 1 the photon orbit coincides with the horizon radius and E stays finite (0.577)
            assert products[1] == pytest.approx(products[0], rel=1e-3, abs=0)


# --------------------------------------------------------------------------------------
# mutation-testing survivors: width of the rounding band of the radial epicyclic frequency
# --------------------------------------------------------------------------------------


def _radial_bracket(r, a):
    """Independent restatement of the bracket (omega_r / omega_phi)^2 = 1 - 6/r + 8 a r^(-3/2) - 3 a^2 / r^2."""
    return 1.0 - 6.0 / r + 8.0 * a * r**-1.5 - 3.0 * a**2 / r**2


def _radius_with_bracket(target, a):
    """Radius near the ISCO at which the bracket equals `target` (brentq on the restated bracket)."""
    from scipy.optimize import brentq

    r_isco = float(basics.kerr_isco_radius_rg(a))
    r = brentq(lambda x: _radial_bracket(x, a) - target, 0.97 * r_isco, 1.03 * r_isco, xtol=1e-15, rtol=1e-15)
    assert _radial_bracket(r, a) == pytest.approx(
        target, rel=1e-3, abs=0
    )  # the bracket rounds at 1e-16, so a 1e-3 match is loose
    return r


@pytest.mark.parametrize("spin", [-0.5, 0.0, 0.9])
@pytest.mark.parametrize("target", [-3e-10, -1e-11, -3e-12])
def test_radial_epicyclic_frequency_is_nan_just_inside_the_isco_beyond_the_rounding_band(spin, target):
    """The rounding band is 1e-12 of the orbital frequency squared. A radius with the bracket at -3e-10, -1e-11 or
    -3e-12 lies outside it (a band widened to 1e-10 would swallow the last two, one widened to 1e-6 all three), is
    radially unstable and must give NaN, not 0; so must the precession frequency that is built from it."""
    r = _radius_with_bracket(target, spin)
    assert np.isnan(kerr.kerr_radial_epicyclic_frequency_hz(M10, r, spin))
    assert np.isnan(kerr.kerr_periastron_precession_frequency_hz(M10, r, spin))


@pytest.mark.parametrize("spin", [-0.5, 0.0, 0.9])
def test_radial_epicyclic_frequency_is_zero_inside_the_rounding_band(spin):
    """A bracket of -5e-13 (inside the documented 1e-12 band) is rounding noise: the frequency is exactly 0."""
    r = _radius_with_bracket(-5e-13, spin)
    assert kerr.kerr_radial_epicyclic_frequency_hz(M10, r, spin) == 0.0
    f_orbit = kerr.kerr_circular_orbit_frequency_hz(M10, r, spin)
    assert kerr.kerr_periastron_precession_frequency_hz(M10, r, spin) == pytest.approx(f_orbit, rel=1e-15, abs=0)


def test_nodal_precession_and_penrose_fraction_for_small_spins():
    """1 - sqrt(q) with q = 1 - 4 a r^-3/2 + 3 a^2 r^-2 and 1 - M_irr / M, formed without their cancellations.

    Formed directly, the nodal precession was 76 % off at a = 1e-12, r = 1000, and the Penrose fraction was noise
    below a = 1e-6 (true a^2 / 8). Reference: 50-digit arithmetic.
    """
    from decimal import Decimal, getcontext

    getcontext().prec = 50
    mass = 10.0 * 1.989e30
    for a, r in ((1e-12, 1000.0), (1e-9, 10.0), (1e-6, 1e6), (1e-3, 50.0), (0.3, 8.0), (-1e-9, 10.0)):
        da, dr = Decimal(a), Decimal(r)
        q = 1 - 4 * da / (dr * dr.sqrt()) + 3 * da * da / (dr * dr)
        exact = 1 - q.sqrt()
        ratio = kerr.kerr_nodal_precession_frequency_hz(mass, r, a) / kerr.kerr_circular_orbit_frequency_hz(mass, r, a)
        assert abs(Decimal(float(ratio)) / exact - 1) < Decimal("1e-13")  # measured 1e-16
    for a in (1e-12, 1e-8, 1e-6, 1e-4, 1e-2, 0.5, 1.0):
        da = Decimal(a)
        exact = 1 - ((1 + (1 - da * da).sqrt()) / 2).sqrt()
        assert abs(Decimal(float(kerr.kerr_penrose_extractable_fraction(a))) / exact - 1) < Decimal("1e-14")
