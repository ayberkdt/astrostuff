"""Tests of kozmos.mechanics.integrators.

Every reference number comes from a closed form (harmonic oscillator, Kepler energy), a conservation law
(angular momentum of a central force), or a property of the map itself (determinant of its Jacobian, reversibility);
none comes from the functions under test. The tolerances were measured on 26 November 2026 (Windows, numpy
float64) and are stated next to each test with the measured value.
"""

from __future__ import annotations

import numpy as np
import pytest

from kozmos.mechanics import integrators as ig
from kozmos.mechanics.integrators import (
    forest_ruth_step,
    integrate,
    leapfrog_dkd_step,
    leapfrog_kdk_step,
    relative_drift,
    symplectic_step_separable,
)

SCHEMES = ("kdk", "dkd", "forest_ruth")
STEPS = {"kdk": leapfrog_kdk_step, "dkd": leapfrog_dkd_step, "forest_ruth": forest_ruth_step}
TWO_PI = 2.0 * np.pi


def oscillator(q, t):
    return -q


def pendulum(q, t):
    return -np.sin(q)


def kepler(q, t):
    """Kepler problem with G M = 1 in the plane."""
    return -q / np.linalg.norm(q) ** 3


def kepler_energy(q, v):
    return 0.5 * np.sum(v**2, axis=-1) - 1.0 / np.linalg.norm(q, axis=-1)


def kepler_start(e):
    """Periapsis state of an orbit with a = 1, G M = 1 (period 2 pi)."""
    return np.array([1.0 - e, 0.0]), np.array([0.0, np.sqrt((1.0 + e) / (1.0 - e))])


# --------------------------------------------------------------------------------------
# coefficients and interface
# --------------------------------------------------------------------------------------


def test_forest_ruth_weights_satisfy_the_order_conditions():
    """Sum of weights is 1 (consistency) and the sum of cubes is 0 (cancels the third-order error term of T2).

    Both are algebraic identities of w1 = 1/(2 - 2^(1/3)), w0 = -2^(1/3) w1; measured residuals 0 and 4e-16.
    """
    w1, w0 = ig._W_OUTER, ig._W_MIDDLE
    assert 2.0 * w1 + w0 == pytest.approx(1.0, abs=1e-15)
    assert 2.0 * w1**3 + w0**3 == pytest.approx(0.0, abs=1e-14)
    assert w1 == pytest.approx(1.3512071919596578, abs=1e-15)  # 1/(2 - 2^(1/3)) evaluated in float64 in this session
    assert w0 < 0.0


def test_integrate_returns_synchronised_trajectory_with_expected_shapes():
    t, q, v = integrate(oscillator, [1.0, 0.0], [0.0, 1.0], 0.1, 50, t0=2.0, record_every=5)
    assert t.shape == (11,) and q.shape == (11, 2) and v.shape == (11, 2)
    assert t[0] == 2.0 and t[-1] == pytest.approx(7.0, abs=1e-12)
    assert np.allclose(q[0], [1.0, 0.0]) and np.allclose(v[0], [0.0, 1.0])


def test_zero_steps_returns_the_initial_state():
    t, q, v = integrate(oscillator, [1.0], [0.5], 0.1, 0)
    assert t.shape == (1,) and q[0, 0] == 1.0 and v[0, 0] == 0.5


def test_input_validation():
    with pytest.raises(ValueError):
        integrate(oscillator, [1.0], [0.0], 0.1, 3, scheme="rk4")
    with pytest.raises(ValueError):
        integrate(oscillator, [1.0], [0.0], 0.0, 3)
    with pytest.raises(ValueError):
        integrate(oscillator, [1.0], [0.0, 1.0], 0.1, 3)
    with pytest.raises(ValueError):
        integrate(oscillator, [1.0], [0.0], 0.1, -1)
    with pytest.raises(ValueError):
        integrate(oscillator, [1.0], [0.0], 0.1, 3, record_every=0)
    with pytest.raises(ValueError):
        integrate(lambda q, t: np.zeros(3), [1.0], [0.0], 0.1, 3)
    with pytest.raises(ValueError):
        symplectic_step_separable([0.0], [1.0], 0.1, lambda p: p, lambda q: q, order=3)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_driver_equals_repeated_single_steps(scheme):
    """The driver (which reuses the end acceleration in the KDK form) is the same map as the one-step function."""
    q0, v0 = kepler_start(0.5)
    _, q, v = integrate(kepler, q0, v0, 0.05, 40, scheme=scheme)
    qq, vv = q0.copy(), v0.copy()
    for k in range(40):
        qq, vv = STEPS[scheme](qq, vv, 0.05 * k, 0.05, kepler)
    assert np.max(np.abs(q[-1] - qq)) < 1e-14
    assert np.max(np.abs(v[-1] - vv)) < 1e-14


def test_time_dependent_force_receives_the_time():
    """x'' = cos(w t) has the exact solution x = (1 - cos(w t)) / w^2 for x(0) = x'(0) = 0 ... with a(t) = cos(w t)."""
    w = 2.0
    t, q, v = integrate(lambda x, time: np.cos(w * time) * np.ones_like(x), [0.0], [0.0], 0.001, 2000)
    # x'' = cos(w t): x' = sin(w t)/w, x = (1 - cos(w t))/w^2
    assert q[-1, 0] == pytest.approx((1.0 - np.cos(w * t[-1])) / w**2, abs=1e-6)
    assert v[-1, 0] == pytest.approx(np.sin(w * t[-1]) / w, abs=1e-6)


W_DRIVE = 1.7
F_DRIVE = 1.0 / (1.0 - W_DRIVE**2)


def driven(q, t):
    return -q + np.cos(W_DRIVE * t)


def driven_exact(t, x0=1.0, v0=0.5):
    """x'' = -x + cos(w t): x = (x0 - F) cos t + v0 sin t + F cos(w t), F = 1 / (1 - w^2); returns (x, v)."""
    x = (x0 - F_DRIVE) * np.cos(t) + v0 * np.sin(t) + F_DRIVE * np.cos(W_DRIVE * t)
    v = -(x0 - F_DRIVE) * np.sin(t) + v0 * np.cos(t) - F_DRIVE * W_DRIVE * np.sin(W_DRIVE * t)
    return x, v


@pytest.mark.parametrize(
    ("scheme", "ratio", "tolerance", "final_error_bound"),
    [("kdk", 4.0, 0.05, 1e-4), ("dkd", 4.0, 0.05, 1e-4), ("forest_ruth", 16.0, 0.12, 3e-8)],
)
def test_time_dependent_force_converges_with_the_nominal_order_for_every_scheme(
    scheme, ratio, tolerance, final_error_bound
):
    """x'' = -x + cos(1.7 t), x(0) = 1, v(0) = 0.5, against the closed form at t = 8 (a force that depends on q
    and on t, so a wrong time argument in any sub-step changes the order).

    Measured errors for dt = 0.2, 0.1, 0.05, 0.025, 0.0125 (max of |dx|, |dv|):
    kdk 2.4e-2, 6.0e-3, 1.5e-3, 3.7e-4, 9.3e-5; dkd 2.4e-2, 5.9e-3, 1.5e-3, 3.7e-4, 9.3e-5;
    forest_ruth 1.6e-3, 9.7e-5, 6.0e-6, 3.8e-7, 2.4e-8. Halving ratios: kdk 4.034, 4.008, 4.002, 4.0005; dkd 4.015,
    4.004, 4.001, 4.0002; forest_ruth 16.073, 16.019, 16.005, 16.001 (largest deviation 0.034 and 0.073). The
    tolerances are 1.5 times those deviations; the bounds on the finest-step error are 1.1, 1.1 and 1.3 times the
    measured values.
    """
    errors = []
    for dt in (0.2, 0.1, 0.05, 0.025, 0.0125):
        n = int(round(8.0 / dt))
        t, q, v = integrate(driven, [1.0], [0.5], dt, n, scheme=scheme)
        x_exact, v_exact = driven_exact(t[-1])
        errors.append(max(abs(q[-1, 0] - x_exact), abs(v[-1, 0] - v_exact)))
    for a, b in zip(errors[:-1], errors[1:], strict=True):
        assert a / b == pytest.approx(ratio, abs=tolerance)
    assert errors[-1] < final_error_bound


# --------------------------------------------------------------------------------------
# convergence order (harmonic oscillator, exact solution cos t, -sin t at t = 10)
# --------------------------------------------------------------------------------------


def _final_error(scheme, dt):
    n = int(round(10.0 / dt))
    _, q, v = integrate(oscillator, [1.0], [0.0], dt, n, scheme=scheme)
    return max(abs(q[-1, 0] - np.cos(10.0)), abs(v[-1, 0] + np.sin(10.0)))


@pytest.mark.parametrize(
    ("scheme", "ratio", "tolerance"),
    [("kdk", 4.0, 0.02), ("dkd", 4.0, 0.02), ("forest_ruth", 16.0, 0.05)],
)
def test_error_ratio_when_the_step_is_halved(scheme, ratio, tolerance):
    """Measured ratios for dt = 0.1/0.05, 0.05/0.025, 0.025/0.0125.

    kdk 3.9940, 3.9985, 3.9996; dkd 4.0035, 4.0009, 4.0002; forest_ruth 16.0228, 16.0056, 16.0014.
    The largest deviation from the nominal ratio is 0.006 (leapfrog) and 0.023 (fourth order).
    """
    errors = [_final_error(scheme, dt) for dt in (0.1, 0.05, 0.025, 0.0125)]
    for a, b in zip(errors[:-1], errors[1:], strict=True):
        assert a / b == pytest.approx(ratio, abs=tolerance)


def test_fourth_order_is_far_more_accurate_than_leapfrog_at_the_same_step():
    assert _final_error("forest_ruth", 0.05) < 1e-2 * _final_error("kdk", 0.05)  # measured 3.6e-6 versus 7.0e-4


# --------------------------------------------------------------------------------------
# time reversibility
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", SCHEMES)
def test_pendulum_forward_then_backward_returns_to_the_start(scheme):
    """500 steps of 0.1 forward and 500 backward. Measured residual: kdk 2e-14, dkd 3e-14, forest_ruth 3e-14."""
    t, q, v = integrate(pendulum, [2.0], [0.3], 0.1, 500, scheme=scheme)
    _, q2, v2 = integrate(pendulum, q[-1], v[-1], -0.1, 500, t0=t[-1], scheme=scheme)
    assert abs(q2[-1, 0] - 2.0) < 1e-12
    assert abs(v2[-1, 0] - 0.3) < 1e-12


@pytest.mark.parametrize("scheme", SCHEMES)
def test_kepler_forward_then_backward_returns_to_the_start(scheme):
    """2000 steps of P/100 (20 periods, e = 0.5) forward and backward.

    Measured residual: kdk 6e-13, dkd 5e-12, forest_ruth 7e-12 (rounding amplified along the orbit).
    """
    q0, v0 = kepler_start(0.5)
    dt = TWO_PI / 100.0
    t, q, v = integrate(kepler, q0, v0, dt, 2000, scheme=scheme)
    _, q2, v2 = integrate(kepler, q[-1], v[-1], -dt, 2000, t0=t[-1], scheme=scheme)
    assert np.max(np.abs(q2[-1] - q0)) < 1e-10
    assert np.max(np.abs(v2[-1] - v0)) < 1e-10


# --------------------------------------------------------------------------------------
# energy: bounded, no secular drift, dt^2 scaling
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("scheme", "steps_per_period"), [("kdk", 100), ("dkd", 100), ("forest_ruth", 50)])
def test_kepler_energy_error_is_bounded_over_1000_periods(scheme, steps_per_period):
    """e = 0.5, 1000 periods, 100 (forest_ruth: 50) steps per period, state recorded 10 times per period.

    Measured: max |dE/E| in the first tenth of the run, and the relative difference of the last-tenth maximum from it:
    kdk 1.0845e-2 and -5.6e-7; dkd 2.504e-3 and -9.8e-8; forest_ruth (dt = P/50) 2.002e-3 and -1.7e-9.
    The means of the signed error in the first and the last tenth differ by 2.5e-4 (kdk), 5.7e-4 (dkd), 1.8e-3
    (forest_ruth) relative. Tolerances: 1e-4 on the maxima ratio (180 times the worst measured), 5% on the means.
    A secular drift would raise the last-tenth maximum above the first: linear growth of 1e-3 of the error
    amplitude per 100 periods, for instance, would give a ratio of 1.01 and fail.
    """
    q0, v0 = kepler_start(0.5)
    dt = TWO_PI / steps_per_period
    record_every = steps_per_period // 10
    _, q, v = integrate(kepler, q0, v0, dt, steps_per_period * 1000, scheme=scheme, record_every=record_every)
    drift = relative_drift(kepler_energy(q, v))
    tenth = len(drift) // 10
    first, last = np.abs(drift[:tenth]).max(), np.abs(drift[-tenth:]).max()
    assert last / first == pytest.approx(1.0, abs=1e-4)
    assert drift[:tenth].mean() == pytest.approx(drift[-tenth:].mean(), rel=0.05)
    # the error itself is small for the step used and of the expected size
    assert last < 5e-2


@pytest.mark.parametrize(
    ("scheme", "ratio", "tolerance"), [("kdk", 4.0, 0.1), ("dkd", 4.0, 0.1), ("forest_ruth", 16.0, 1.0)]
)
def test_energy_error_scales_as_dt_to_the_order(scheme, ratio, tolerance):
    """Modified-energy behaviour: max |dE/E| over 5 periods for dt = P/100, P/200, P/400, e = 0.5.

    Measured maxima: kdk 1.0845e-2, 2.689e-3, 6.710e-4 (ratios 4.032, 4.008); dkd 2.504e-3, 6.317e-4,
    1.583e-4 (ratios 3.964, 3.991); forest_ruth 1.428e-4, 9.237e-6, 5.824e-7 (ratios 15.45, 15.86). The tolerance is
    the largest measured deviation rounded up.
    """
    q0, v0 = kepler_start(0.5)
    maxima = []
    for steps in (100, 200, 400):
        _, q, v = integrate(kepler, q0, v0, TWO_PI / steps, 5 * steps, scheme=scheme)
        maxima.append(np.abs(relative_drift(kepler_energy(q, v))).max())
    for a, b in zip(maxima[:-1], maxima[1:], strict=True):
        assert a / b == pytest.approx(ratio, abs=tolerance)


# --------------------------------------------------------------------------------------
# angular momentum of a central force
# --------------------------------------------------------------------------------------


def _central(q, t):
    """Central, non-Keplerian force: -r^(-2) rhat - 0.3 r (a softened inner repulsion plus a harmonic well)."""
    return -q / np.linalg.norm(q) ** 3 - 0.3 * q


@pytest.mark.parametrize("scheme", SCHEMES)
def test_angular_momentum_is_conserved_to_rounding_for_a_central_force(scheme):
    """20000 steps of 0.01 (r between 0.47 and 1.07). Measured max relative change of L_z: kdk 2.6e-14, dkd 1.6e-14,
    forest_ruth 2.5e-14. Tolerance 1e-12. (The energy of this scheme is not conserved, only bounded.)"""
    q0, v0 = np.array([1.0, 0.2]), np.array([0.1, 0.8])
    _, q, v = integrate(_central, q0, v0, 0.01, 20000, scheme=scheme)
    lz = q[:, 0] * v[:, 1] - q[:, 1] * v[:, 0]
    assert np.max(np.abs(lz / lz[0] - 1.0)) < 1e-12


def test_angular_momentum_vector_in_three_dimensions():
    """The full vector r x v: measured max absolute change 9.3e-15 over 20000 steps (|L| = 0.68); tolerance 1e-12."""

    def force(q, t):
        return -q / np.linalg.norm(q) ** 3 - 0.2 * q / np.linalg.norm(q)

    _, q, v = integrate(force, [1.0, 0.2, 0.3], [0.1, 0.7, 0.4], 0.01, 20000)
    assert np.max(np.abs(np.cross(q, v) - np.cross(q[0], v[0]))) < 1e-12


# --------------------------------------------------------------------------------------
# symplecticity: Jacobian determinant of the one-step map of the pendulum
# --------------------------------------------------------------------------------------


def _jacobian_determinant(step, z, h=1e-4):
    """Determinant of the Jacobian of z -> step(z) by central finite differences with increment h."""
    jac = np.empty((2, 2))
    for j in range(2):
        dz = np.zeros(2)
        dz[j] = h
        jac[:, j] = (step(z + dz) - step(z - dz)) / (2.0 * h)
    return np.linalg.det(jac)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_one_step_map_of_the_pendulum_has_unit_jacobian_determinant(scheme):
    """dt = 0.1, state (q, v) = (2.0, 0.3). Measured |det - 1| with h = 1e-4: kdk 4.6e-12, dkd 5.8e-12,
    forest_ruth 6.7e-12 (finite-difference noise; h = 1e-6 gives 2e-10). Tolerance 1e-9."""

    def step(z):
        q1, v1 = STEPS[scheme](z[:1], z[1:], 0.0, 0.1, pendulum)
        return np.array([q1[0], v1[0]])

    assert abs(_jacobian_determinant(step, np.array([2.0, 0.3])) - 1.0) < 1e-9


def test_the_determinant_test_can_fail_explicit_euler_is_not_symplectic():
    """Negative control: explicit Euler has det = 1 + dt^2 cos(q) exactly; measured det - 1 = -4.16e-3 at q = 2."""

    def euler(z):
        return np.array([z[0] + 0.1 * z[1], z[1] - 0.1 * np.sin(z[0])])

    deviation = _jacobian_determinant(euler, np.array([2.0, 0.3])) - 1.0
    assert deviation == pytest.approx(0.01 * np.cos(2.0), abs=1e-8)
    assert abs(deviation) > 1e-3


@pytest.mark.parametrize("order", [2, 4])
def test_general_step_with_relativistic_kinetic_energy_has_unit_determinant(order):
    """H = sqrt(1 + p^2) - cos q. The kinetic term is not quadratic; each sub-map is an exact flow, so det = 1.

    Measured |det - 1| (h = 1e-4): about 1e-11 for both orders; tolerance 1e-9.
    """

    def step(z):
        q1, p1 = symplectic_step_separable(
            z[:1], z[1:], 0.1, lambda p: p / np.sqrt(1.0 + p**2), lambda q: np.sin(q), order=order
        )
        return np.array([q1[0], p1[0]])

    assert abs(_jacobian_determinant(step, np.array([2.0, 0.3])) - 1.0) < 1e-9


@pytest.mark.parametrize(("order", "scheme"), [(2, "dkd"), (4, "forest_ruth")])
def test_general_step_equals_the_specialised_steps_for_quadratic_kinetic_energy(order, scheme):
    q0, v0 = kepler_start(0.5)
    q1, p1 = symplectic_step_separable(q0, v0, 0.05, lambda p: p, lambda q: -kepler(q, 0.0), order=order)
    q2, v2 = STEPS[scheme](q0, v0, 0.0, 0.05, kepler)
    assert np.max(np.abs(q1 - q2)) < 1e-15
    assert np.max(np.abs(p1 - v2)) < 1e-15


# --------------------------------------------------------------------------------------
# relative_drift
# --------------------------------------------------------------------------------------


def test_relative_drift_values_and_sign():
    d = relative_drift([2.0, 2.2, 1.8, 2.0])
    assert np.allclose(d, [0.0, 0.1, -0.1, 0.0], atol=1e-15)
    assert np.allclose(relative_drift([-2.0, -2.2]), [0.0, -0.1], atol=1e-15)  # reference sign does not flip the sign
    with pytest.raises(ValueError):
        relative_drift([0.0, 1.0])


def test_harmonic_oscillator_leapfrog_conserves_the_modified_energy_exactly():
    """The one-step matrix of the oscillator, M = [[1 - dt^2/2, dt (1 - dt^2/4)], [-dt, 1 - dt^2/2]] for kick-drift-
    kick (derived by hand from the three sub-steps), has det 1 and preserves Q = v^2 + (1 - dt^2/4) q^2; the
    drift-kick-drift form preserves (1 - dt^2/4) v^2 + q^2. The true energy then oscillates with relative amplitude
    dt^2/4 (q = 1, v = 0 start). Measured for dt = 0.1, 10000 steps: Q drift 1.0e-14 (kdk), 1.4e-14 (dkd);
    max |dE/E| = 2.49999993e-3 against dt^2/4 = 2.5e-3 (relative difference 3e-8). Tolerances 1e-12 and 1e-6.
    """
    dt = 0.1
    _, q, v = integrate(oscillator, [1.0], [0.0], dt, 10000, scheme="kdk")
    invariant = v[:, 0] ** 2 + (1.0 - dt**2 / 4.0) * q[:, 0] ** 2
    assert np.max(np.abs(invariant / invariant[0] - 1.0)) < 1e-12
    drift = relative_drift(0.5 * (q[:, 0] ** 2 + v[:, 0] ** 2))
    assert np.abs(drift).max() == pytest.approx(dt**2 / 4.0, rel=1e-6)
    _, q, v = integrate(oscillator, [1.0], [0.0], dt, 10000, scheme="dkd")
    invariant = (1.0 - dt**2 / 4.0) * v[:, 0] ** 2 + q[:, 0] ** 2
    assert np.max(np.abs(invariant / invariant[0] - 1.0)) < 1e-12
