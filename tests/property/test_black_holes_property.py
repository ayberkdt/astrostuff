"""Property tests of the black hole module: monotonicity, scaling, ordering and round trips."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kozmos.black_holes import accretion, basics

masses = st.floats(1e-3, 1e10).map(lambda x: x * 1.988e30)  # 1e-3 to 1e10 M☉
spins = st.floats(-1.0, 1.0)
factors = st.floats(0.1, 100.0)
SETTINGS = settings(max_examples=60, deadline=None)


@given(st.floats(0.0, 1.0), st.floats(0.0, 1.0))
@SETTINGS
def test_isco_monotonic_in_spin(a1, a2):
    # For prograde orbits the ISCO shrinks with spin, for retrograde orbits it grows: r(−a) increasing, r(+a) decreasing.
    lo, hi = sorted((a1, a2))
    assert basics.kerr_isco_radius_rg(hi) <= basics.kerr_isco_radius_rg(lo) + 1e-12
    assert basics.kerr_isco_radius_rg(-hi) >= basics.kerr_isco_radius_rg(-lo) - 1e-12


@given(spins)
@SETTINGS
def test_isco_bounds_and_ordering(a):
    isco_radius = basics.kerr_isco_radius_rg(a)
    photon = basics.kerr_photon_orbit_radius_rg(a)
    horizon = basics.kerr_horizon_radius_rg(a)
    assert 1.0 - 1e-12 <= isco_radius <= 9.0 + 1e-12
    # Horizon ≤ photon orbit ≤ ISCO; the ergosphere lies outside the horizon.
    assert horizon <= photon + 1e-9
    assert photon <= isco_radius + 1e-9
    assert basics.kerr_ergosphere_radius_rg(a) >= horizon - 1e-12
    # The prograde orbit lies inside the retrograde orbit.
    assert basics.kerr_isco_radius_rg(abs(a)) <= basics.kerr_isco_radius_rg(-abs(a)) + 1e-12


@given(st.floats(1e-6, 1e-3))
@SETTINGS
def test_kerr_isco_continuous_at_low_spin(eps):
    # a* → 0: the ISCO tends continuously to 6 r_G from both signs (slope ≈ 3.27 < 3.3).
    assert abs(basics.kerr_isco_radius_rg(eps) - 6.0) <= 3.3 * eps + 1e-9
    assert abs(basics.kerr_isco_radius_rg(-eps) - 6.0) <= 3.3 * eps + 1e-9


@given(st.floats(-1.0, 1.0), st.floats(-1.0, 1.0))
@SETTINGS
def test_efficiency_increases_with_spin(a1, a2):
    lo, hi = sorted((a1, a2))
    assert basics.isco_radiative_efficiency(hi) >= basics.isco_radiative_efficiency(lo) - 1e-12
    assert 0.0 < basics.isco_radiative_efficiency(a1) < 0.4227


@given(spins, st.floats(0.0, np.pi))
@SETTINGS
def test_ergosphere_between_horizon_and_equator(a, theta):
    r = basics.kerr_ergosphere_radius_rg(a, theta)
    assert basics.kerr_horizon_radius_rg(a) - 1e-12 <= r <= 2.0 + 1e-12


@given(masses, factors)
@SETTINGS
def test_geometry_linear(m, k):
    for f in (
        basics.gravitational_radius_m,
        basics.schwarzschild_radius_m,
        basics.photon_sphere_radius_m,
        basics.critical_impact_parameter_m,
        basics.schwarzschild_isco_radius_m,
    ):
        assert f(k * m) == pytest.approx(k * f(m), rel=1e-12, abs=0)
    for f in (basics.isco_orbital_frequency_hz, basics.gw_frequency_at_isco_hz, basics.hawking_temperature_k):
        assert f(k * m) == pytest.approx(f(m) / k, rel=1e-12, abs=0)


@given(masses, st.floats(1e20, 1e26), factors)
@SETTINGS
def test_shadow_diameter_depends_on_mass_distance_ratio(m, d, k):
    assert basics.shadow_angular_diameter_uas(k * m, k * d) == pytest.approx(
        basics.shadow_angular_diameter_uas(m, d), rel=1e-12, abs=0
    )


@given(masses, st.floats(1.0001, 1e6), st.floats(1.0001, 1e6))
@SETTINGS
def test_redshift_decreases_with_radius(m, x1, x2):
    rs = basics.schwarzschild_radius_m(m)
    lo, hi = sorted((x1, x2))
    z_lo = basics.gravitational_redshift(lo * rs, m)
    z_hi = basics.gravitational_redshift(hi * rs, m)
    assert z_lo >= z_hi > 0.0


@given(masses, factors)
@SETTINGS
def test_thermodynamic_scaling(m, k):
    assert basics.bekenstein_hawking_entropy(k * m) == pytest.approx(
        k**2 * basics.bekenstein_hawking_entropy(m), rel=1e-12, abs=0
    )
    assert basics.hawking_evaporation_time_s(k * m) == pytest.approx(
        k**3 * basics.hawking_evaporation_time_s(m), rel=1e-12, abs=0
    )
    # T·M is constant: the temperature falls as the mass grows.
    assert basics.hawking_temperature_k(k * m) * k * m == pytest.approx(
        basics.hawking_temperature_k(m) * m, rel=1e-12, abs=0
    )


@given(masses, masses)
@SETTINGS
def test_chirp_mass_symmetry_and_bound(m1, m2):
    mc = basics.chirp_mass_kg(m1, m2)
    assert mc == pytest.approx(basics.chirp_mass_kg(m2, m1), rel=1e-13, abs=0)
    # ν ≤ 1/4 ⇒ 𝓜 = ν^{3/5} M ≤ 2^{-6/5} M; equality for equal masses.
    assert mc <= 2.0**-1.2 * (m1 + m2) * (1.0 + 1e-12)
    assert basics.chirp_mass_kg(2.0 * m1, 2.0 * m2) == pytest.approx(2.0 * mc, rel=1e-13, abs=0)


@given(st.floats(1.0, 100.0), st.floats(1.0, 400.0), st.floats(1.0, 400.0))
@SETTINGS
def test_merger_time_decreases_with_frequency_and_round_trip(mc_sun, f1, f2):
    mc = mc_sun * 1.988e30
    lo, hi = sorted((f1, f2))
    assert basics.time_to_merger_s(mc, hi) <= basics.time_to_merger_s(mc, lo)
    assert basics.gw_frequency_derivative_hz_s(mc, hi) >= basics.gw_frequency_derivative_hz_s(mc, lo)
    tau = basics.time_to_merger_s(mc, lo)
    assert basics.gw_frequency_from_time_to_merger_hz(mc, tau) == pytest.approx(lo, rel=1e-12, abs=0)
    # τ ∝ f^{-8/3}.
    assert basics.time_to_merger_s(mc, 2.0 * lo) == pytest.approx(
        basics.time_to_merger_s(mc, lo) * 2.0 ** (-8.0 / 3.0), rel=1e-12, abs=0
    )


@given(masses, factors, st.floats(0.01, 0.4))
@SETTINGS
def test_eddington_linear_and_efficiency_scaling(m, k, eta):
    assert accretion.eddington_luminosity_w(k * m) == pytest.approx(
        k * accretion.eddington_luminosity_w(m), rel=1e-12, abs=0
    )
    # The luminosity is independent of the efficiency: L = η Ṁ_Edd(η) c² = L_Edd.
    mdot = accretion.eddington_accretion_rate_kg_s(m, eta)
    assert accretion.accretion_luminosity_w(mdot, eta) == pytest.approx(
        accretion.eddington_luminosity_w(m), rel=1e-12, abs=0
    )


@given(st.floats(0.01, 0.4), st.floats(0.01, 0.4))
@SETTINGS
def test_salpeter_time_increases_with_efficiency(e1, e2):
    lo, hi = sorted((e1, e2))
    assert accretion.salpeter_time_s(lo) <= accretion.salpeter_time_s(hi)


@given(masses, st.floats(1e3, 1e6), st.floats(1e-25, 1e-15), factors)
@SETTINGS
def test_bondi_scaling(m, cs, rho, k):
    assert accretion.bondi_radius_m(k * m, cs) == pytest.approx(k * accretion.bondi_radius_m(m, cs), rel=1e-12, abs=0)
    assert accretion.bondi_radius_m(m, k * cs) == pytest.approx(
        accretion.bondi_radius_m(m, cs) / k**2, rel=1e-12, abs=0
    )
    assert accretion.bondi_accretion_rate_kg_s(k * m, rho, cs) == pytest.approx(
        k**2 * accretion.bondi_accretion_rate_kg_s(m, rho, cs), rel=1e-12, abs=0
    )
    assert accretion.bondi_accretion_rate_kg_s(m, k * rho, cs) == pytest.approx(
        k * accretion.bondi_accretion_rate_kg_s(m, rho, cs), rel=1e-12, abs=0
    )
    assert accretion.bondi_accretion_rate_kg_s(m, rho, k * cs) == pytest.approx(
        accretion.bondi_accretion_rate_kg_s(m, rho, cs) / k**3, rel=1e-12, abs=0
    )


@given(masses, st.floats(1e12, 1e18), st.floats(1e4, 1e8), factors)
@SETTINGS
def test_thin_disk_scaling_and_envelope(m, mdot, r_in, k):
    # T_max ∝ (M Ṁ)^{1/4} r_in^{-3/4}.
    t0 = accretion.thin_disk_peak_temperature_k(m, mdot, r_in)
    assert accretion.thin_disk_peak_temperature_k(k * m, mdot, r_in) == pytest.approx(k**0.25 * t0, rel=1e-12, abs=0)
    assert accretion.thin_disk_peak_temperature_k(m, k * mdot, r_in) == pytest.approx(k**0.25 * t0, rel=1e-12, abs=0)
    assert accretion.thin_disk_peak_temperature_k(m, mdot, k * r_in) == pytest.approx(k**-0.75 * t0, rel=1e-12, abs=0)
    # The profile does not exceed T_max at any radius.
    r = r_in * np.linspace(1.0, 50.0, 400)
    assert np.all(accretion.thin_disk_effective_temperature_k(r, m, mdot, r_in) <= t0 * (1.0 + 1e-12))


@given(masses, masses, st.floats(1e6, 1e10), factors)
@SETTINGS
def test_tidal_radius_and_hills_scaling(m_bh, m_star, r_star, k):
    rt = accretion.tidal_disruption_radius_m(m_bh, m_star, r_star)
    assert accretion.tidal_disruption_radius_m(k * m_bh, m_star, r_star) == pytest.approx(
        k ** (1.0 / 3.0) * rt, rel=1e-12, abs=0
    )
    assert accretion.tidal_disruption_radius_m(m_bh, k * m_star, r_star) == pytest.approx(
        k ** (-1.0 / 3.0) * rt, rel=1e-12, abs=0
    )
    # At M_H, r_t = R_S; for a larger black hole r_t < R_S (the star is swallowed without being disrupted).
    mh = accretion.hills_mass_kg(m_star, r_star)
    assert accretion.tidal_disruption_radius_m(mh, m_star, r_star) == pytest.approx(
        basics.schwarzschild_radius_m(mh), rel=1e-12, abs=0
    )
    assert accretion.tidal_disruption_radius_m(2.0 * mh, m_star, r_star) < basics.schwarzschild_radius_m(2.0 * mh)
