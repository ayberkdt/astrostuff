"""Tests of kozmos.relativity.special.

Every reference value comes from a closed form, from an invariance property (Lorentz condition, invariant
interval, invariant four-velocity product) or from an independent computation with the boost matrix. Tolerances
are about ten times the maximum measured over the random samples used here (seed fixed); the measured values are
in the comments.
"""

from __future__ import annotations

import numpy as np
import pytest

from kozmos.core.constants import SPEED_OF_LIGHT
from kozmos.relativity.special import (
    aberrated_angle_rad,
    add_velocities_collinear_m_s,
    add_velocities_m_s,
    boost_four_vector,
    boost_matrix,
    contracted_length_m,
    dilated_time_s,
    doppler_factor,
    doppler_factor_source_frame,
    energy_from_momentum_j,
    four_momentum,
    invariant_mass_kg,
    kinetic_energy_j,
    lorentz_factor,
    minkowski_inner_product,
    proper_time_s,
    rapidity,
    relativistic_momentum,
    spacetime_interval_squared_m2,
    total_energy_j,
    velocity_from_rapidity_m_s,
)

C = SPEED_OF_LIGHT.value
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
RNG = np.random.default_rng(20261126)


def _random_direction(rng: np.random.Generator) -> np.ndarray:
    d = rng.normal(size=3)
    return d / np.linalg.norm(d)


def _random_velocity(rng: np.random.Generator, beta_max: float) -> np.ndarray:
    return rng.uniform(0.0, beta_max) * C * _random_direction(rng)


# --------------------------------------------------------------------------------------
# Lorentz factor and rapidity
# --------------------------------------------------------------------------------------


def test_lorentz_factor_closed_forms():
    assert lorentz_factor(0.0) == 1.0
    assert lorentz_factor(0.6 * C) == pytest.approx(1.25, rel=1e-15, abs=0)
    assert lorentz_factor(0.8 * C) == pytest.approx(5.0 / 3.0, rel=1e-15, abs=0)
    assert lorentz_factor(-0.6 * C) == lorentz_factor(0.6 * C)


def test_lorentz_factor_low_velocity_limit():
    beta = np.array([1e-4, 1e-3, 1e-2])
    # gamma = 1 + beta^2/2 + 3 beta^4/8 + 5 beta^6/16 + 35 beta^8/128 + ... The three-term sum differs from gamma by
    # 3.1e-13 at beta = 1e-2 (the 5 beta^6/16 term; the rel=1e-15 only ever passed thanks to the hidden absolute
    # tolerance 1e-12 of pytest.approx). With four terms the omitted term is 2.7e-17 and the measured relative
    # difference is 2.2e-16 (tolerance 1e-15).
    series = 1.0 + beta**2 / 2.0 + 3.0 * beta**4 / 8.0 + 5.0 * beta**6 / 16.0
    assert lorentz_factor(beta * C) == pytest.approx(series, rel=1e-15, abs=0)
    assert np.allclose((lorentz_factor(beta * C) - 1.0) / (beta**2 / 2.0), 1.0, atol=1e-3)


def test_lorentz_factor_equals_cosh_of_rapidity():
    beta = np.linspace(-0.99999, 0.99999, 41)
    assert lorentz_factor(beta * C) == pytest.approx(np.cosh(rapidity(beta * C)), rel=1e-12, abs=0)


@pytest.mark.parametrize("bad", [C, -C, 1.1 * C])
def test_speed_at_or_above_c_is_rejected(bad):
    with pytest.raises(ValueError):
        lorentz_factor(bad)
    with pytest.raises(ValueError):
        rapidity(bad)


def test_rapidity_round_trip_and_additivity():
    # The round trip loses digits as 1 - tanh(phi) shrinks: relative error 1.8e-11 at phi = 8 (1 - beta = 2e-7), so the
    # check is limited to |phi| <= 5, where the maximum relative error measured is 9.7e-14.
    phi = np.linspace(-5.0, 5.0, 33)
    assert rapidity(velocity_from_rapidity_m_s(phi)) == pytest.approx(phi, rel=1e-12, abs=1e-15)
    a, b = 0.3, 1.7
    composed = add_velocities_collinear_m_s(velocity_from_rapidity_m_s(a), velocity_from_rapidity_m_s(b))
    assert rapidity(composed) == pytest.approx(a + b, rel=1e-13, abs=0)
    assert np.all(np.abs(velocity_from_rapidity_m_s(np.linspace(-30, 30, 61))) <= C)


# --------------------------------------------------------------------------------------
# Boost matrix, inner product, interval
# --------------------------------------------------------------------------------------


def test_boost_matrix_along_x_is_eq_1_18_of_carroll():
    phi = 0.7
    v = np.array([C * np.tanh(phi), 0.0, 0.0])
    expected = np.eye(4)
    expected[:2, :2] = [[np.cosh(phi), -np.sinh(phi)], [-np.sinh(phi), np.cosh(phi)]]  # Eq. 1.18
    assert boost_matrix(v) == pytest.approx(expected, abs=1e-14)


def test_boost_matrix_zero_velocity_is_identity():
    assert np.array_equal(boost_matrix(np.zeros(3)), np.eye(4))


def test_boost_matrix_satisfies_lorentz_condition():
    # Eq. 1.14: Lambda^T eta Lambda = eta. Measured maximum defect 7.7e-14 for these 500 random boosts with beta <
    # 0.9999 (gamma up to 70); tolerance 1e-12.
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(500):
        lam = boost_matrix(_random_velocity(rng, 0.9999))
        worst = max(worst, np.abs(lam.T @ ETA @ lam - ETA).max())
    assert worst < 1e-12


def test_boost_inverse_is_boost_of_minus_velocity():
    rng = np.random.default_rng(2)
    for _ in range(100):
        v = _random_velocity(rng, 0.999)
        assert boost_matrix(-v) @ boost_matrix(v) == pytest.approx(np.eye(4), abs=1e-13)  # measured 1.3e-14


def test_collinear_boosts_compose_by_adding_rapidity():
    d = _random_direction(np.random.default_rng(3))
    v1, v2 = 0.4 * C * d, 0.75 * C * d
    composed = boost_matrix(v1) @ boost_matrix(v2)
    w = velocity_from_rapidity_m_s(rapidity(0.4 * C) + rapidity(0.75 * C)) * d
    assert composed == pytest.approx(boost_matrix(w), abs=1e-13)


def test_boosts_in_different_directions_do_not_commute():
    a = boost_matrix(np.array([0.6 * C, 0.0, 0.0]))
    b = boost_matrix(np.array([0.0, 0.6 * C, 0.0]))
    assert np.abs(a @ b - b @ a).max() > 0.1  # the composition contains a Wigner rotation


def test_minkowski_inner_product_signs_and_symmetry():
    timelike = np.array([2.0, 1.0, 0.0, 0.0])
    spacelike = np.array([1.0, 2.0, 0.0, 0.0])
    null = np.array([1.0, 0.0, 0.0, 1.0])
    assert minkowski_inner_product(timelike, timelike) == -3.0
    assert minkowski_inner_product(spacelike, spacelike) == 3.0
    assert minkowski_inner_product(null, null) == 0.0
    a, b = RNG.normal(size=4), RNG.normal(size=4)
    assert minkowski_inner_product(a, b) == minkowski_inner_product(b, a)
    assert minkowski_inner_product(a, b) == pytest.approx(a @ ETA @ b, abs=1e-15)


def test_interval_and_inner_product_are_invariant_under_random_boosts():
    # Measured maximum relative change of the interval and of the inner product 3.9e-14 (500 boosts, beta < 0.9999);
    # tolerance 5e-13.
    rng = np.random.default_rng(4)
    worst = 0.0
    for _ in range(500):
        event = rng.normal(size=4) * 1e9
        other = rng.normal(size=4) * 1e9
        v = _random_velocity(rng, 0.9999)
        scale = np.sum(event**2)
        before = spacetime_interval_squared_m2(event)
        after = spacetime_interval_squared_m2(boost_four_vector(event, v))
        worst = max(worst, abs(after - before) / scale)
        dot_after = minkowski_inner_product(boost_four_vector(event, v), boost_four_vector(other, v))
        worst = max(worst, abs(dot_after - minkowski_inner_product(event, other)) / np.sqrt(scale * np.sum(other**2)))
    assert worst < 5e-13


def test_light_like_separation_has_zero_interval_in_every_frame():
    n = _random_direction(np.random.default_rng(5))
    event = 1e9 * np.concatenate([[1.0], n])
    v = np.array([0.3 * C, -0.5 * C, 0.2 * C])
    scale = 1e18  # squares of the components; the rounding of a sum of such squares is a few times 1e-16 of it
    assert abs(spacetime_interval_squared_m2(event)) < 1e-14 * scale
    assert abs(spacetime_interval_squared_m2(boost_four_vector(event, v))) < 1e-14 * scale


def test_event_at_rest_in_new_frame_moves_with_the_frame_velocity():
    # events with x = v t (the origin of S') map to x' = 0 (Eq. 1.20: v = x / t)
    v = np.array([0.3 * C, 0.2 * C, -0.4 * C])
    t = 3.7e-3
    event = np.concatenate([[C * t], v * t])
    moved = boost_four_vector(event, v)
    assert moved[1:] == pytest.approx(np.zeros(3), abs=1e-3)  # lengths of order 1e6 m, rounding 1e-10
    gamma = lorentz_factor(np.linalg.norm(v))
    assert moved[0] == pytest.approx(C * t / gamma, rel=1e-13, abs=0)  # proper time of the moving origin


# --------------------------------------------------------------------------------------
# Velocity addition
# --------------------------------------------------------------------------------------


def test_collinear_addition_matches_rapidity_addition():
    # Measured maximum difference 1.8e-15 c over these 500 random pairs; tolerance 2e-14 c.
    rng = np.random.default_rng(6)
    a = rng.uniform(-0.9999, 0.9999, 500)
    b = rng.uniform(-0.9999, 0.9999, 500)
    ref = C * np.tanh(np.arctanh(a) + np.arctanh(b))
    assert np.abs(add_velocities_collinear_m_s(a * C, b * C) - ref).max() < 2e-14 * C


def test_collinear_addition_never_exceeds_c():
    rng = np.random.default_rng(7)
    a = rng.uniform(-1.0, 1.0, 2000) * (1.0 - 1e-12)
    b = rng.uniform(-1.0, 1.0, 2000) * (1.0 - 1e-12)
    w = add_velocities_collinear_m_s(a * C, b * C)
    assert np.all(np.abs(w) <= C)
    # adding a velocity to a speed close to c stays below c
    assert add_velocities_collinear_m_s(C * (1 - 1e-12), 0.9 * C) < C


def test_collinear_addition_low_velocity_and_light_invariance():
    assert add_velocities_collinear_m_s(10.0, 20.0) == pytest.approx(30.0, rel=1e-14, abs=0)
    # photon-like limit: any u added to v -> c gives c
    assert add_velocities_collinear_m_s(C * (1 - 1e-13), 0.7 * C) == pytest.approx(C, rel=1e-12, abs=0)
    # 0.5c + 0.5c = 0.8c
    assert add_velocities_collinear_m_s(0.5 * C, 0.5 * C) == pytest.approx(0.8 * C, rel=1e-15, abs=0)


def test_vector_addition_parallel_equals_collinear_formula():
    d = _random_direction(np.random.default_rng(8))
    w = add_velocities_m_s(0.6 * C * d, 0.5 * C * d)
    assert w == pytest.approx(add_velocities_collinear_m_s(0.6 * C, 0.5 * C) * d, rel=1e-13, abs=0)
    antiparallel = add_velocities_m_s(0.6 * C * d, -0.6 * C * d)
    assert np.linalg.norm(antiparallel) < 1e-6  # rounding of order c * 1e-16 / ... in m/s


def test_vector_addition_satisfies_the_invariant_four_velocity_identity_and_stays_below_c():
    # Invariance of -U_frame . U_object gives gamma_v = gamma_u gamma_w (1 - u.w / c^2) for w the velocity in S, v the
    # velocity in S' and u the velocity of S' in S. Measured maximum relative defect 1.5e-14 and round-trip defect
    # 6.7e-15 c for beta < 0.99 (the identity loses digits when beta approaches 1: 1.5e-11 was measured for
    # beta < 0.9999 in a separate run); tolerance 2e-13.
    rng = np.random.default_rng(9)
    worst = 0.0
    for _ in range(500):
        v = _random_velocity(rng, 0.99)
        u = _random_velocity(rng, 0.99)
        w = add_velocities_m_s(v, u)
        assert np.linalg.norm(w) < C
        gv = lorentz_factor(np.linalg.norm(v))
        gu = lorentz_factor(np.linalg.norm(u))
        gw = lorentz_factor(np.linalg.norm(w))
        worst = max(worst, abs(gv - gu * gw * (1.0 - u @ w / C**2)) / gv)
        back = add_velocities_m_s(w, -u)
        worst = max(worst, np.linalg.norm(back - v) / C)
    assert worst < 2e-13


def test_vector_addition_never_exceeds_c_near_the_limit():
    rng = np.random.default_rng(10)
    for _ in range(200):
        v = _random_velocity(rng, 1.0 - 1e-9)
        u = _random_velocity(rng, 1.0 - 1e-9)
        assert np.linalg.norm(add_velocities_m_s(v, u)) <= C


# --------------------------------------------------------------------------------------
# Time dilation and length contraction
# --------------------------------------------------------------------------------------


def test_time_dilation_matches_the_boosted_event_pair():
    tau = 3.0e-6
    v = np.array([0.3 * C, -0.5 * C, 0.2 * C])
    # two ticks of a clock at rest in S': separation (c tau, 0, 0, 0). S coordinates are obtained with boost(-v).
    in_s = boost_four_vector(np.array([C * tau, 0.0, 0.0, 0.0]), -v)
    speed = np.linalg.norm(v)
    assert in_s[0] / C == pytest.approx(dilated_time_s(tau, speed), rel=1e-13, abs=0)
    assert proper_time_s(dilated_time_s(tau, speed), speed) == pytest.approx(tau, rel=1e-14, abs=0)
    assert np.sqrt(-spacetime_interval_squared_m2(in_s)) / C == pytest.approx(tau, rel=1e-12, abs=0)


def test_time_dilation_low_velocity_limit():
    beta = 1e-3
    assert dilated_time_s(1.0, beta * C) == pytest.approx(1.0 + beta**2 / 2.0, rel=1e-11, abs=0)
    assert dilated_time_s(2.0, 0.6 * C) == pytest.approx(2.5, rel=1e-15, abs=0)


def test_length_contraction_matches_the_boosted_rod():
    # rod of proper length L0 along x' in S' (moving with v along x relative to S). Its end points are read at equal
    # time t in S: the second end point event is chosen with t = 0, found from the boost matrix itself.
    length0 = 12.5
    speed = 0.8 * C
    lam_inverse = boost_matrix(np.array([-speed, 0.0, 0.0]))  # S' -> S
    # S time of the event (c t', L0) is the first row; solve for t' such that it is zero
    t_prime = -lam_inverse[0, 1] * length0 / lam_inverse[0, 0] / C
    end_in_s = lam_inverse @ np.array([C * t_prime, length0, 0.0, 0.0])
    assert end_in_s[0] == pytest.approx(0.0, abs=1e-12)
    assert end_in_s[1] == pytest.approx(contracted_length_m(length0, speed), rel=1e-14, abs=0)
    assert contracted_length_m(length0, speed) == pytest.approx(0.6 * length0, rel=1e-15, abs=0)
    assert contracted_length_m(length0, 1e-3 * C) == pytest.approx(length0 * (1.0 - 0.5e-6), rel=1e-9, abs=0)


# --------------------------------------------------------------------------------------
# Doppler effect and aberration
# --------------------------------------------------------------------------------------


def test_doppler_and_aberration_match_the_boosted_null_vector():
    # Photon four-vector k = (1, n) in the observer frame S; the source frame S' moves with V relative to S, so
    # k' = Lambda(V) k. Then nu_obs / nu_src = k^0 / k'^0, the angle theta is the one between V and n, theta' the one
    # between V and the direction of k'. Measured maximum relative defect of the Doppler factor 6.7e-15 (source-frame
    # form 1.4e-14), of the angle 2.1e-14 rad (500 samples, beta < 0.9999); tolerances 1e-13, 2e-13, 3e-13.
    rng = np.random.default_rng(11)
    worst_d = worst_a = worst_d2 = 0.0
    for _ in range(500):
        beta = rng.uniform(0.0, 0.9999)
        d = _random_direction(rng)
        n = _random_direction(rng)
        k_source = boost_matrix(beta * C * d) @ np.concatenate([[1.0], n])
        factor = 1.0 / k_source[0]
        theta = np.arccos(np.clip(d @ n, -1.0, 1.0))
        n_source = k_source[1:] / k_source[0]
        theta_source = np.arccos(np.clip(d @ n_source, -1.0, 1.0))
        worst_d = max(worst_d, abs(doppler_factor(beta * C, theta) / factor - 1.0))
        worst_d2 = max(worst_d2, abs(doppler_factor_source_frame(beta * C, theta_source) / factor - 1.0))
        worst_a = max(worst_a, abs(aberrated_angle_rad(beta * C, theta_source) - theta))
    assert worst_d < 1e-13
    assert worst_d2 < 2e-13
    assert worst_a < 3e-13


def test_doppler_longitudinal_and_transverse_limits():
    beta = 0.6
    assert doppler_factor(beta * C, 0.0) == pytest.approx(np.sqrt((1 + beta) / (1 - beta)), rel=1e-15, abs=0)  # = 2
    assert doppler_factor(beta * C, np.pi) == pytest.approx(np.sqrt((1 - beta) / (1 + beta)), rel=1e-15, abs=0)  # = 1/2
    gamma = 1.25
    assert doppler_factor(beta * C, np.pi / 2) == pytest.approx(1.0 / gamma, rel=1e-15, abs=0)  # transverse redshift
    assert doppler_factor_source_frame(beta * C, np.pi / 2) == pytest.approx(
        gamma, rel=1e-15, abs=0
    )  # transverse blueshift
    assert doppler_factor(0.0, 1.234) == 1.0
    small = 1e-6
    assert doppler_factor(small * C, 0.9) == pytest.approx(1.0 + small * np.cos(0.9), abs=2e-12)


def test_aberration_limits():
    beta = 0.6
    assert aberrated_angle_rad(beta * C, 0.7) == pytest.approx(
        np.arctan2(np.sin(0.7), 1.25 * (np.cos(0.7) + beta)), rel=1e-15, abs=0
    )
    assert aberrated_angle_rad(0.0, 1.1) == pytest.approx(1.1, rel=1e-15, abs=0)
    # emission at right angles in the source frame: cos(theta) = beta
    assert np.cos(aberrated_angle_rad(beta * C, np.pi / 2)) == pytest.approx(beta, rel=1e-15, abs=0)
    # forward beaming: theta < theta' for 0 < theta' < pi
    angles = np.linspace(0.1, 3.0, 20)
    assert np.all(aberrated_angle_rad(beta * C, angles) < angles)
    assert aberrated_angle_rad(beta * C, 0.0) == 0.0
    assert aberrated_angle_rad(beta * C, np.pi) == pytest.approx(np.pi, abs=1e-15)


# --------------------------------------------------------------------------------------
# Four-momentum, energy
# --------------------------------------------------------------------------------------


def test_four_momentum_norm_is_minus_mass_squared_times_c_squared():
    # measured maximum relative defect 1.2e-14 (200 samples, beta < 0.99); tolerance 2e-13
    rng = np.random.default_rng(12)
    m = 9.109e-31
    for _ in range(200):
        v = _random_velocity(rng, 0.99)
        p = four_momentum(m, v)
        assert minkowski_inner_product(p, p) == pytest.approx(-((m * C) ** 2), rel=2e-13, abs=0)


def test_four_momentum_of_a_particle_at_rest_boosts_to_four_momentum_of_the_boost_velocity():
    m = 1.67e-27
    v = np.array([0.3 * C, 0.1 * C, -0.2 * C])
    rest = four_momentum(m, np.zeros(3))
    assert rest == pytest.approx([m * C, 0.0, 0.0, 0.0], abs=0.0)
    # the particle moves with v in S, i.e. S' = rest frame moves with v: S coordinates from boost(-v)
    assert boost_four_vector(rest, -v) == pytest.approx(four_momentum(m, v), rel=1e-13, abs=0)


def test_energy_momentum_relation_and_limits():
    m = 1.67e-27
    rng = np.random.default_rng(13)
    for _ in range(100):
        v = _random_velocity(rng, 0.999)
        speed = np.linalg.norm(v)
        p = relativistic_momentum(m, v)
        assert np.linalg.norm(p) == pytest.approx(lorentz_factor(speed) * m * speed, rel=1e-13, abs=0)
        assert energy_from_momentum_j(m, np.linalg.norm(p)) == pytest.approx(total_energy_j(m, speed), rel=1e-13, abs=0)
        # the two routines form |v| separately, and one unit in the last place of |v|^2 is amplified by gamma^2
        # (up to 500 here): 1.2e-14 on the CI machines, where the limit of 1e-14 failed on every platform but one
        assert total_energy_j(m, speed) == pytest.approx(C * four_momentum(m, v)[0], rel=1e-12, abs=0)
    assert total_energy_j(m, 0.0) == m * C**2
    assert energy_from_momentum_j(0.0, 3.0e-20) == pytest.approx(3.0e-20 * C, rel=1e-15, abs=0)  # photon
    assert energy_from_momentum_j(m, 0.0) == m * C**2
    assert relativistic_momentum(m, np.array([10.0, 0.0, 0.0])) == pytest.approx([m * 10.0, 0.0, 0.0], rel=1e-12, abs=0)


def test_kinetic_energy_limits():
    m = 2.0
    beta = np.array([1e-5, 1e-3, 1e-2])
    newtonian = 0.5 * m * (beta * C) ** 2
    # relative difference from the Newtonian value is 3 beta^2 / 4 to leading order
    assert kinetic_energy_j(m, beta * C) / newtonian - 1.0 == pytest.approx(0.75 * beta**2, rel=1e-3, abs=0)
    fast = 0.99
    assert kinetic_energy_j(m, fast * C) == pytest.approx(total_energy_j(m, fast * C) - m * C**2, rel=1e-12, abs=0)
    assert kinetic_energy_j(m, 0.6 * C) == pytest.approx(0.25 * m * C**2, rel=1e-15, abs=0)  # gamma = 1.25


# --------------------------------------------------------------------------------------
# Invariant mass
# --------------------------------------------------------------------------------------


def test_invariant_mass_of_two_photons():
    # M c^2 = sqrt(2 E1 E2 (1 - cos alpha)) for photons of energies E1, E2 with opening angle alpha
    e1, e2, alpha = 3.2e-11, 1.1e-11, 0.83
    n1 = np.array([1.0, 0.0, 0.0])
    n2 = np.array([np.cos(alpha), np.sin(alpha), 0.0])
    photons = np.array([e1 / C * np.concatenate([[1.0], n1]), e2 / C * np.concatenate([[1.0], n2])])
    expected = np.sqrt(2.0 * e1 * e2 * (1.0 - np.cos(alpha))) / C**2
    assert invariant_mass_kg(photons) == pytest.approx(expected, rel=1e-13, abs=0)


def test_invariant_mass_symmetric_pion_decay_closed_form():
    # two photons of equal energy E and opening angle alpha: M c^2 = 2 E sin(alpha / 2)
    energy, alpha = 67.5e6 * 1.602176634e-19, 1.9
    half = alpha / 2.0
    p1 = energy / C * np.array([1.0, np.sin(half), np.cos(half), 0.0])
    p2 = energy / C * np.array([1.0, -np.sin(half), np.cos(half), 0.0])
    assert invariant_mass_kg(np.array([p1, p2])) == pytest.approx(2.0 * energy * np.sin(half) / C**2, rel=1e-13, abs=0)


def test_invariant_mass_is_invariant_under_boosts_and_reduces_to_the_rest_mass():
    rng = np.random.default_rng(14)
    m = 1.0e-27
    single = four_momentum(m, np.array([0.3 * C, 0.1 * C, 0.0]))
    assert invariant_mass_kg(single[None, :]) == pytest.approx(m, rel=1e-14, abs=0)
    particles = np.array([four_momentum(m, _random_velocity(rng, 0.9)) for _ in range(5)])
    reference = invariant_mass_kg(particles)
    for _ in range(20):
        boost = _random_velocity(rng, 0.99)
        boosted = np.array([boost_four_vector(p, boost) for p in particles])
        assert invariant_mass_kg(boosted) == pytest.approx(reference, rel=1e-10, abs=0)
    # total energy in the centre-of-momentum frame is at least the sum of the rest masses
    assert reference >= 5.0 * m


def test_invariant_mass_collinear_photons_is_zero_and_spacelike_is_rejected():
    e = 1.0e-12
    collinear = np.array([e / C * np.array([1.0, 1.0, 0.0, 0.0])] * 2)
    assert invariant_mass_kg(collinear) == pytest.approx(0.0, abs=1e-30)
    with pytest.raises(ValueError):
        invariant_mass_kg(np.array([[0.0, 1.0, 0.0, 0.0]]))


def test_shape_errors():
    with pytest.raises(ValueError):
        boost_matrix(np.zeros(4))
    with pytest.raises(ValueError):
        boost_four_vector(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError):
        add_velocities_m_s(np.zeros((2, 3)), np.zeros(3))


@pytest.mark.parametrize("function", [doppler_factor, doppler_factor_source_frame, aberrated_angle_rad])
def test_angle_based_functions_reject_a_negative_speed(function):
    # Decision: the speed argument is a speed (0 <= v < c, as the docstrings state); the direction of motion is carried
    # by the angle. Before the fix a negative speed was silently treated as a signed velocity.
    with pytest.raises(ValueError, match="non-negative"):
        function(-0.5 * C, 0.7)
    with pytest.raises(ValueError, match="non-negative"):
        function(np.array([0.1 * C, -0.1 * C]), 0.7)
    with pytest.raises(ValueError):
        function(C, 0.7)  # v = c remains rejected
    assert np.isfinite(function(0.0, 0.7))  # zero stays valid


def test_signed_speed_equals_a_rotated_angle_for_the_doppler_factor():
    # The sign convention removed above is recovered with the angle: a source moving at v along -x is the source moving
    # along +x seen at the angle pi - theta. Exact identity of the closed form (measured difference 0, tolerance 1e-15).
    beta, theta = 0.6, 0.4
    forward = doppler_factor(beta * C, np.pi - theta)
    assert forward == pytest.approx(1.0 / (1.0 / np.sqrt(1 - beta**2) * (1.0 + beta * np.cos(theta))), rel=1e-15, abs=0)


# --------------------------------------------------------------------------------------
# mutation-testing survivors: units of the invariant mass, the rounding band, batches, v = c and NaN
# --------------------------------------------------------------------------------------

M_ELECTRON = 9.1093837015e-31  # kg, only a convenient SI scale of 1e-30


def test_invariant_mass_of_particles_at_rest_is_in_kilograms():
    """Two electrons at rest: P = (2 m c, 0, 0, 0), M = 2 m = 1.8e-30 kg. A missing division by c would return
    5.5e-22; a pytest.approx with rel only would not notice it (default absolute tolerance 1e-12), so abs=0."""
    p = four_momentum(M_ELECTRON, np.zeros(3))
    assert invariant_mass_kg(np.stack([p, p])) == pytest.approx(2.0 * M_ELECTRON, rel=1e-14, abs=0)
    assert invariant_mass_kg(np.stack([p, p])) < 1e-29
    # moving back to back with equal speed: M = 2 gamma m
    v = 0.6 * C
    pa, pb = four_momentum(M_ELECTRON, [v, 0, 0]), four_momentum(M_ELECTRON, [-v, 0, 0])
    assert invariant_mass_kg(np.stack([pa, pb])) == pytest.approx(2.0 * 1.25 * M_ELECTRON, rel=1e-14, abs=0)


def _almost_null_pair(excess: float) -> np.ndarray:
    """A single four-momentum (1, 1 + excess, 0, 0): -P.P = 1 - (1 + excess)^2 = -(2 excess + excess^2) < 0."""
    return np.array([[[1.0, 1.0 + excess, 0.0, 0.0]]])


def test_invariant_mass_clips_a_slightly_negative_square_to_zero():
    """Documented: a -P.P negative by less than 1e-12 (P^0)^2 is set to zero. Here -P.P = -4e-13 (measured below):
    the result is exactly 0, not NaN (no clipping) and not an error."""
    p = _almost_null_pair(2e-13)
    s = -minkowski_inner_product(p[0, 0], p[0, 0])
    assert -1e-12 < s < -3e-13  # measured -4.0e-13
    result = invariant_mass_kg(p)
    assert result.shape == (1,)
    assert result[0] == 0.0


def test_invariant_mass_raises_for_a_spacelike_sum_beyond_the_1e_minus_12_band():
    """-P.P = -4e-11 is 40 times the documented band 1e-12 (P^0)^2 and 2.5 times smaller than 1e-10, so a band
    widened to 1e-10 would accept it; it must raise."""
    p = _almost_null_pair(2e-11)
    assert -minkowski_inner_product(p[0, 0], p[0, 0]) == pytest.approx(-4e-11, rel=1e-3, abs=0)
    with pytest.raises(ValueError, match="spacelike"):
        invariant_mass_kg(p)
    with pytest.raises(ValueError, match="spacelike"):
        invariant_mass_kg(_almost_null_pair(1e-11))  # -2e-11, still beyond the band
    # and the band scales with (P^0)^2: the same relative excess at P^0 = 1e-20 still raises
    with pytest.raises(ValueError, match="spacelike"):
        invariant_mass_kg(1e-20 * _almost_null_pair(2e-11))


def test_four_momentum_of_a_batch_of_velocities_equals_the_single_results():
    """A (2, 3) batch with different speeds (0.6 c and 0.8 c) against the two single calls; gamma depends on the
    LAST axis only (a sum over the first axis would mix the two particles)."""
    v = np.array([[0.6 * C, 0.0, 0.0], [0.0, 0.0, 0.8 * C]])
    batch = four_momentum(M_ELECTRON, v)
    assert batch.shape == (2, 4)
    for i in range(2):
        assert np.array_equal(batch[i], four_momentum(M_ELECTRON, v[i]))
    assert batch[0, 0] == pytest.approx(1.25 * M_ELECTRON * C, rel=1e-15, abs=0)
    assert batch[1, 0] == pytest.approx(M_ELECTRON * C / 0.6, rel=1e-15, abs=0)
    masses = np.array([M_ELECTRON, 3.0 * M_ELECTRON])
    mixed = four_momentum(masses, v)
    assert np.array_equal(mixed[1], four_momentum(3.0 * M_ELECTRON, v[1]))
    deep = four_momentum(M_ELECTRON, np.stack([v, v[::-1]]))  # shape (2, 2, 3)
    assert deep.shape == (2, 2, 4) and np.array_equal(deep[1, 0], batch[1])


@pytest.mark.parametrize("bad", [[C, 0.0, 0.0], [0.0, -C, 0.0], [C / np.sqrt(3.0)] * 3, [np.nan, 0.0, 0.0]])
def test_four_momentum_and_boost_matrix_refuse_the_speed_of_light_and_nan(bad):
    """beta^2 = 1 exactly (or NaN) is outside the open interval beta^2 < 1: both functions must raise."""
    with pytest.raises(ValueError, match="smaller than the speed of light"):
        four_momentum(M_ELECTRON, bad)
    with pytest.raises(ValueError, match="smaller than the speed of light"):
        boost_matrix(bad)


def test_four_momentum_and_boost_matrix_accept_a_speed_just_below_c():
    v = [(1.0 - 1e-9) * C, 0.0, 0.0]
    assert np.isfinite(four_momentum(M_ELECTRON, v)).all()
    assert np.isfinite(boost_matrix(v)).all()
