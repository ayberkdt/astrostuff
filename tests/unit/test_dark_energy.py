"""Closed forms, published numbers and internal identities for kozmos.cosmology.dark_energy.

No number below comes from the functions under test: closed forms are derived in the comments, the one
published number is the critical density printed in [Frieman2008] (sec2.tex line 94), and the numerical
references are scipy quadratures written in the test.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from kozmos.core.constants import GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT
from kozmos.cosmology import dark_energy as de
from kozmos.cosmology import distances

Z = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0])

LCDM = distances.Cosmology.flat(70.0, 0.3)
EDS = distances.Cosmology.flat(100.0, 1.0)
CPL = distances.Cosmology.flat(70.0, 0.3, w=-0.8, wa=0.5)
# curved, w0-wa, with radiation: every term of E(z) is active
FULL = distances.Cosmology(68.0, 0.28, 0.69, w=-0.9, wa=-0.3, omega_r=9e-5)
MODELS = [LCDM, EDS, CPL, FULL, distances.Cosmology.flat(70.0, 0.3, w=-1.2)]


def test_eos_limits():
    assert de.de_eos(0.0, CPL) == pytest.approx(-0.8, abs=1e-15)
    assert de.de_eos(1.0, CPL) == pytest.approx(-0.8 + 0.25, abs=1e-15)  # w0 + wa z/(1+z), z = 1
    assert de.de_eos(1e9, CPL) == pytest.approx(-0.3, abs=1e-8)  # w0 + wa as z -> infinity


def test_de_rho_ratio_equals_the_integral_of_frieman_eq_rho_scaling():
    # [Frieman2008] Eq. rho_scaling: rho_DE(z)/rho_DE(0) = exp[3 int_0^z (1+w(z')) dln(1+z')]
    for k in (CPL, FULL):
        for z in (0.1, 1.0, 5.0, 50.0):
            integral = quad(
                lambda lnp, k=k: 1.0 + de.de_eos(np.expm1(lnp), k), 0.0, np.log1p(z), epsabs=0, epsrel=1e-13
            )[0]
            assert de.de_rho_ratio(z, k) == pytest.approx(np.exp(3.0 * integral), rel=1e-12)


def test_constant_w_density_ratio():
    assert de.de_rho_ratio(3.0, distances.Cosmology.flat(70, 0.3, w=-0.9)) == pytest.approx(4.0 ** (3 * 0.1), rel=1e-14)
    assert np.all(de.de_rho_ratio(Z, LCDM) == 1.0)


@pytest.mark.parametrize("k", MODELS)
def test_density_parameters_add_to_one(k):
    total = de.omega_matter(Z, k) + de.omega_radiation(Z, k) + de.omega_de(Z, k) + de.omega_curvature(Z, k)
    assert np.max(np.abs(total - 1.0)) < 1e-14


def test_lcdm_closed_forms():
    # flat LCDM without radiation: Omega_m(z) = Om (1+z)^3 / (Om (1+z)^3 + OL); q = Omega_m(z)/2 - Omega_L(z)
    # (Eq. q of [Frieman2008] with w = -1, derived here)
    zp = 1.0 + Z
    om = 0.3 * zp**3 / (0.3 * zp**3 + 0.7)
    assert np.allclose(de.omega_matter(Z, LCDM), om, rtol=1e-14)
    assert np.allclose(de.omega_de(Z, LCDM), 1.0 - om, rtol=1e-13)
    assert np.allclose(de.deceleration_parameter(Z, LCDM), 0.5 * om - (1.0 - om), rtol=1e-12, atol=1e-15)
    assert np.all(de.omega_radiation(Z, LCDM) == 0.0) and np.all(np.abs(de.omega_curvature(Z, LCDM)) < 1e-15)


def test_einstein_de_sitter():
    assert np.allclose(de.omega_matter(Z, EDS), 1.0, rtol=1e-15)
    assert np.allclose(de.deceleration_parameter(Z, EDS), 0.5, rtol=1e-15)
    assert np.allclose(de.total_eos(Z, EDS), 0.0, atol=1e-15)


def test_pure_radiation_universe():
    k = distances.Cosmology(70.0, 0.0, 0.0, omega_r=1.0)
    assert np.allclose(de.deceleration_parameter(Z, k), 1.0, rtol=1e-15)  # Eq. q: (1/2)(1 + 3/3)
    assert np.allclose(de.total_eos(Z, k), 1.0 / 3.0, rtol=1e-15)


@pytest.mark.parametrize("k", [CPL, FULL, LCDM])
def test_q_equals_minus_one_minus_half_dlnh2_dlna(k):
    # independent definition: q = -a a_ddot / a_dot^2 = -1 - dlnH/dlna; central difference in ln a, step h
    h = 1e-4
    lna = -np.log1p(Z[:-1])
    e2 = lambda n: distances.e_function(np.expm1(-n), k) ** 2  # noqa: E731
    dln = (np.log(e2(lna + h)) - np.log(e2(lna - h))) / (2 * h)
    assert np.allclose(de.deceleration_parameter(Z[:-1], k), -1.0 - 0.5 * dln, atol=2e-8)


@pytest.mark.parametrize("k", [LCDM, CPL, distances.Cosmology.flat(70.0, 0.3, w=-1.2, omega_r=1e-4)])
def test_flat_q_from_total_eos(k):
    assert np.allclose(de.deceleration_parameter(Z, k), 0.5 * (1.0 + 3.0 * de.total_eos(Z, k)), atol=1e-14)


def test_acceleration_onset_lcdm_closed_form():
    # q = 0 for flat LCDM: Om (1+z)^3 / 2 = OL  ->  (1+z)^3 = 2 OL / Om
    for om in (0.25, 0.3, 0.4):
        k = distances.Cosmology.flat(70.0, om)
        assert de.acceleration_onset_redshift(k) == pytest.approx(
            (2.0 * (1.0 - om) / om) ** (1.0 / 3.0) - 1.0, rel=1e-12
        )


def test_acceleration_onset_constant_w_closed_form():
    # flat wCDM, q = 0: Om (1+z)^3 + (1+3w) OL (1+z)^{3(1+w)} = 0  ->  (1+z)^{-3w} = -(1+3w) OL / Om
    for w in (-0.8, -1.2):
        k = distances.Cosmology.flat(70.0, 0.3, w=w)
        expected = (-(1.0 + 3.0 * w) * 0.7 / 0.3) ** (-1.0 / (3.0 * w)) - 1.0
        assert de.acceleration_onset_redshift(k) == pytest.approx(expected, rel=1e-12)


def test_acceleration_onset_raises_without_sign_change():
    with pytest.raises(ValueError):
        de.acceleration_onset_redshift(EDS)  # q = 1/2 everywhere
    with pytest.raises(ValueError):
        de.acceleration_onset_redshift(LCDM, z_max=0.5)  # onset at z = 0.67


def test_matter_de_equality_constant_w_closed_form():
    # [Frieman2008] sec2.tex line 278: 1 + z = (Omega_M/Omega_DE)^{1/(3w)}; checked for both signs of 1 + w
    for w in (-1.0, -0.8, -1.2):
        k = distances.Cosmology.flat(70.0, 0.3, w=w)
        assert de.matter_de_equality_redshift(k) == pytest.approx((0.3 / 0.7) ** (1.0 / (3.0 * w)) - 1.0, rel=1e-12)


def test_matter_de_equality_cpl_equal_densities():
    z_eq = de.matter_de_equality_redshift(CPL)
    assert 0.0 < z_eq < 2.0
    assert de.omega_matter(z_eq, CPL) == pytest.approx(de.omega_de(z_eq, CPL), rel=1e-12)


def test_matter_de_equality_raises():
    with pytest.raises(ValueError):
        de.matter_de_equality_redshift(distances.Cosmology.flat(70.0, 0.8))  # matter dominates at z = 0 and beyond
    with pytest.raises(ValueError):
        de.matter_de_equality_redshift(EDS)


def test_matter_radiation_equality():
    k = distances.Cosmology.flat(70.0, 0.3, omega_r=8.5e-5)
    z_eq = de.matter_radiation_equality_redshift(k)
    assert z_eq == pytest.approx(0.3 / 8.5e-5 - 1.0, rel=1e-14)
    assert de.omega_matter(z_eq, k) == pytest.approx(de.omega_radiation(z_eq, k), rel=1e-13)
    with pytest.raises(ValueError):
        de.matter_radiation_equality_redshift(LCDM)


def test_critical_density_matches_the_published_number():
    # [Frieman2008] sec2.tex line 94: rho_crit = 1.88e-29 h^2 g cm^-3 = 1.88e-26 h^2 kg m^-3 (three digits)
    for h in (0.5, 0.7, 1.0):
        k = distances.Cosmology.flat(100.0 * h, 0.3)
        assert de.critical_density_kg_m3(0.0, k) == pytest.approx(1.88e-26 * h**2, rel=3e-3)


def test_critical_density_scales_as_e_squared():
    assert np.allclose(
        de.critical_density_kg_m3(Z, FULL) / de.critical_density_kg_m3(0.0, FULL),
        distances.e_function(Z, FULL) ** 2,
        rtol=1e-14,
    )


def test_hubble_parameter_is_h0_times_e():
    assert np.allclose(de.hubble_parameter_km_s_mpc(Z, CPL), 70.0 * distances.e_function(Z, CPL), rtol=1e-15)
    assert de.hubble_parameter_km_s_mpc(0.0, FULL) == pytest.approx(68.0, rel=1e-15)


def test_de_mass_density_and_lambda():
    k = distances.Cosmology.flat(70.0, 0.3)
    rho = de.critical_density_kg_m3(0.0, k) * 0.7
    assert np.allclose(de.de_mass_density_kg_m3(Z, k), rho, rtol=1e-14)  # constant for w = -1
    # [Frieman2008] sec2.tex line 62: Lambda = 8 pi G rho_VAC (c = 1); in SI divide by c^2
    lam = 8.0 * np.pi * GRAVITATIONAL_CONSTANT.value * rho / SPEED_OF_LIGHT.value**2
    assert de.cosmological_constant_lambda(k) == pytest.approx(lam, rel=1e-14)
    assert 1.0e-52 < de.cosmological_constant_lambda(k) < 1.4e-52  # the usual order of magnitude, 1.1e-52 m^-2
    with pytest.raises(ValueError):
        de.cosmological_constant_lambda(distances.Cosmology.flat(70.0, 0.3, w=-0.9))
    with pytest.raises(ValueError):
        de.cosmological_constant_lambda(CPL)
    # phantom energy grows with time, so it was smaller in the past: rho_DE(z=1) < rho_DE(0), ratio 2^{-0.6}
    phantom = distances.Cosmology.flat(70.0, 0.3, w=-1.2)
    assert de.de_mass_density_kg_m3(1.0, phantom) < de.de_mass_density_kg_m3(0.0, phantom)  # (1+z)^{-0.6}


def test_age_einstein_de_sitter_closed_form():
    # E = (1+z)^{3/2}: t = (2/3) t_H (1+z)^{-3/2}
    t_h = distances.hubble_time_gyr(EDS)
    assert np.allclose(de.age_gyr(Z, EDS), (2.0 / 3.0) * t_h * (1.0 + Z) ** -1.5, rtol=1e-13)


def test_age_flat_lcdm_closed_form():
    # [Frieman2008] sec2.tex line 131: a(t) = (Om/OL)^{1/3} sinh^{2/3}(3 sqrt(OL) H0 t / 2); invert for t
    a = 1.0 / (1.0 + Z)
    t = 2.0 / (3.0 * np.sqrt(0.7)) * distances.hubble_time_gyr(LCDM) * np.arcsinh(np.sqrt(0.7 / 0.3) * a**1.5)
    assert np.allclose(de.age_gyr(Z, LCDM), t, rtol=1e-12)


def test_age_with_radiation_against_quad():
    # t(z) = t_H int_z^inf dz'/((1+z') E(z')), done in u = ln(1+z') to infinity with scipy
    k = distances.Cosmology.flat(70.0, 0.3, w=-0.9, wa=0.2, omega_r=8.5e-5)
    for z in (0.0, 1.0, 10.0, 1100.0):
        ref = quad(
            lambda u: 1.0 / distances.e_function(np.expm1(u), k), np.log1p(z), 80.0, epsabs=0, epsrel=1e-13, limit=500
        )[0]
        assert de.age_gyr(z, k) == pytest.approx(distances.hubble_time_gyr(k) * ref, rel=1e-12)


def test_age_requires_matter_or_radiation():
    with pytest.raises(ValueError):
        de.age_gyr(0.0, distances.Cosmology(70.0, 0.0, 1.0))


def test_big_rip_time_exact_for_pure_phantom_and_estimate_with_matter():
    # Omega_DE = 1: E = a^{-3(1+w)/2} = a^{beta/2}, t_rip - t0 = int_1^inf da/(a H0 E) = 2/(beta H0), exact
    beta = -3.0 * (1.0 - 1.5)
    pure = distances.Cosmology(70.0, 0.0, 1.0, w=-1.5)
    exact = (
        distances.hubble_time_gyr(pure)
        * quad(lambda a: 1.0 / (a * distances.e_function(1.0 / a - 1.0, pure)), 1.0, np.inf)[0]
    )
    assert de.big_rip_time_gyr(pure) == pytest.approx(exact, rel=1e-12)
    assert exact == pytest.approx(2.0 * distances.hubble_time_gyr(pure) / beta, rel=1e-12)
    # with matter the source's formula is an estimate: measured difference from the integral
    k = distances.Cosmology.flat(70.0, 0.3, w=-1.5)
    numerical = (
        distances.hubble_time_gyr(k)
        * quad(lambda a: 1.0 / (a * distances.e_function(1.0 / a - 1.0, k)), 1.0, np.inf)[0]
    )
    ratio = de.big_rip_time_gyr(k) / numerical
    # measured 0.859: the estimate is 14 percent short because Omega_DE = 0.7 < 1 makes E smaller than a^{beta/2}
    assert 0.85 < ratio < 0.87, ratio
    for bad in (LCDM, CPL, distances.Cosmology.flat(70.0, 0.3, w=-0.9)):
        with pytest.raises(ValueError):
            de.big_rip_time_gyr(bad)


def test_cosmology_validation():
    with pytest.raises(ValueError):
        distances.Cosmology(70.0, 0.3, 0.7, omega_r=-1e-5)
    k = distances.Cosmology.flat(70.0, 0.3, w=-0.9, wa=0.1, omega_r=1e-4)
    assert k.omega_k == pytest.approx(0.0, abs=1e-15)
    assert k.omega_lambda == pytest.approx(1.0 - 0.3 - 1e-4)


# --- bouncing models: E^2 <= 0 somewhere ------------------------------------------------------------------------
# Cosmology(70, 0.1, 1.5): Omega_k = -0.6, E^2 = 0.1 x^3 - 0.6 x^2 + 1.5 with x = 1 + z; E^2 = 0 at z = 0.9166 and
# negative for 0.9166 < z < 4.4 (written here, not taken from the code under test).
BOUNCE = distances.Cosmology(70.0, 0.1, 1.5)
# E^2 = 0 exactly at z = 1: 0.5 x^3 - 1.5 x^2 + 2 = 4 - 6 + 2 = 0 for x = 2
ZERO_E2 = distances.Cosmology(70.0, 0.5, 2.0)


def _bounce_e2(z):
    x = 1.0 + np.asarray(z, dtype=float)
    return 0.1 * x**3 - 0.6 * x**2 + 1.5


def test_bouncing_model_gives_nan_where_e_squared_is_not_positive():
    # before the fix: q(z = 1) = 11.0 and Omega_m(z = 1) = -8.0 (finite values from a negative denominator)
    z = np.array([0.0, 0.5, 0.9, 1.0, 2.0, 4.0, 5.0, 20.0])
    x = 1.0 + z
    e2 = _bounce_e2(z)
    bad = e2 <= 0.0
    assert bad.sum() == 3 and (~bad).sum() == 5  # z = 1, 2, 4 are inside the negative window
    # closed forms for w = -1, no radiation: Omega_i(z) = (term of E^2) / E^2
    closed = {
        de.omega_matter: 0.1 * x**3 / e2,
        de.omega_de: 1.5 / e2,
        de.omega_curvature: -0.6 * x**2 / e2,
        de.omega_radiation: 0.0 * x,
        de.deceleration_parameter: (0.05 * x**3 - 1.5) / e2,  # q = Omega_m/2 - Omega_DE
        de.total_eos: -1.5 / (0.1 * x**3 + 1.5),  # (p_DE)/(rho_m + rho_DE), the E^2 cancels
    }
    for function, expected in closed.items():
        value = function(z, BOUNCE)
        assert np.all(np.isnan(value[bad])), function.__name__
        assert np.allclose(value[~bad], expected[~bad], rtol=1e-13, atol=0.0), function.__name__
    # scalars, and E^2 exactly zero
    assert np.isnan(de.deceleration_parameter(1.0, BOUNCE)) and np.isnan(de.omega_matter(1.0, BOUNCE))
    assert de.deceleration_parameter(0.0, BOUNCE) == pytest.approx(-1.45, abs=1e-15)  # 0.05 - 1.5
    for function in (de.omega_matter, de.omega_de, de.omega_radiation, de.omega_curvature, de.deceleration_parameter):
        assert np.isnan(function(1.0, ZERO_E2)), function.__name__


def test_acceleration_onset_rejects_a_pole_of_q():
    # before the fix: returned z = 0.9166, where E^2 crosses zero and q = 1.7e14
    with pytest.raises(ValueError, match=r"E\^2"):
        de.acceleration_onset_redshift(BOUNCE)


def test_sign_change_across_a_pole_is_not_a_root():
    # 1/(z - 0.5) changes sign at z = 0.5 with |f| -> infinity; before the fix this returned 0.5
    with pytest.raises(ValueError, match="pole"):
        de._first_sign_change(lambda z: 1.0 / (np.asarray(z) - 0.5), 2.0)
    assert de._first_sign_change(lambda z: np.asarray(z) - 0.5, 2.0) == pytest.approx(0.5, abs=1e-14)


def test_acceleration_onset_requires_acceleration_today():
    # w0 = -0.3, wa = -0.9, flat, Omega_m = 0.3: q(0) = 0.3/2 + (0.7/2)(1 + 3 (-0.3)) = 0.15 + 0.035 = 0.185 > 0, but
    # q < 0 for 0.390 < z < 0.860 (measured; the audit text said 2): the first zero outwards (0.390 before the fix)
    # is where acceleration ENDS going forward
    k = distances.Cosmology.flat(70.0, 0.3, w=-0.3, wa=-0.9)
    assert de.deceleration_parameter(0.0, k) == pytest.approx(0.185, abs=1e-12)  # measured 0.18500000000000003
    assert de.deceleration_parameter(0.6, k) < 0.0  # measured -0.018
    with pytest.raises(ValueError, match="today"):
        de.acceleration_onset_redshift(k)


def test_acceleration_onset_is_where_the_present_accelerating_phase_began():
    # q(0) < 0 and q > 0 beyond the onset: q changes sign from negative to positive there, and q < 0 before it
    for k in (distances.Cosmology.flat(70.0, 0.3), CPL, distances.Cosmology.flat(70.0, 0.3, w=-1.2)):
        z_acc = de.acceleration_onset_redshift(k)
        z = np.linspace(0.0, z_acc, 200, endpoint=False)
        assert np.all(de.deceleration_parameter(z, k) < 0.0)
        assert de.deceleration_parameter(z_acc * 1.001 + 1e-3, k) > 0.0
        assert abs(de.deceleration_parameter(z_acc, k)) < 1e-12  # measured at most 1.1e-16


def test_first_sign_change_root_exactly_on_a_grid_node():
    # before the fix a root on a node (zero times anything is not < 0) was skipped: the result was None
    grid = np.expm1(np.linspace(0.0, np.log1p(3.0), de._SCAN_POINTS))
    node = grid[len(grid) // 2]
    assert de._first_sign_change(lambda z: np.asarray(z) - node, 3.0) == node
    # a zero at the first node is a root too, and a later sign change is not preferred over an earlier node root
    assert de._first_sign_change(lambda z: np.asarray(z) * (np.asarray(z) - 1.0), 3.0) == 0.0


def _two_roots_close_together(z):
    # f < 0 except in a window of half-width 1.25e-3 in ln(1+z) around ln 2 (roots at ln(1+z) = ln 2 +- 1.25e-3)
    u = np.log1p(np.asarray(z))
    return 1.25e-3**2 - (u - np.log(2.0)) ** 2


def test_scan_resolves_two_roots_2p5e_3_apart_in_ln_1pz(monkeypatch):
    # the grid step is 9.8e-4 for z_max = 50, so a window of 2.5e-3 holds at least two nodes and is found: the first
    # root is the lower edge of the window
    root = de._first_sign_change(_two_roots_close_together, 50.0)
    assert root == pytest.approx(np.expm1(np.log(2.0) - 1.25e-3), rel=1e-11)  # measured difference 3.3e-16
    # a 50-point grid (step 0.08) steps over the window: the pair of roots is missed
    monkeypatch.setattr(de, "_SCAN_POINTS", 50)
    assert de._first_sign_change(_two_roots_close_together, 50.0) is None


def test_matter_de_equality_requires_dark_energy_dominance_today():
    # flat, w = +0.5: rho_DE ∝ (1+z)^4.5 grows faster than matter into the past, Omega_m = 0.6 > Omega_DE today:
    # ln(rho_m/rho_DE) = ln 1.5 - 1.5 ln(1+z) vanishes at z = 1.5^(1/1.5) - 1 = 0.310, where dark energy STARTS to
    # dominate going back (the end of an earlier era); this was returned before the fix
    k = distances.Cosmology.flat(70.0, 0.6, w=0.5)
    assert 1.5 ** (1.0 / 1.5) - 1.0 == pytest.approx(0.3104, abs=1e-4)
    with pytest.raises(ValueError, match="today"):
        de.matter_de_equality_redshift(k)
    # equal today: the equality is at z = 0 (Omega_m = Omega_DE = 0.5 exactly)
    assert de.matter_de_equality_redshift(distances.Cosmology.flat(70.0, 0.5)) == 0.0
