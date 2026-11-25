"""Hamiltonian mechanics against closed forms, Lie-algebra identities and the Lagrangian integrator.

Oracles (none calls the code under test): H = p^2/2m + k q^2/2; H = sqrt(p^2 + m^2) for the free relativistic
particle; hyperbolic motion of a relativistic particle in a constant force (p = p0 - F t, q = q0 - (E - E0)/F);
H = (p - e A)^2 / 2m for a charged particle; the textbook angular-momentum algebra {Lx, Ly} = Lz; the sl(2, R)
algebra and the relation 2KH + 2Q^2 + m alpha = 0 printed in the source; the explicit-time derivative dQ/dt = 0.

Every tolerance is a round number (1, 2 or 5 times a power of ten) at least ten times the error measured on the
reference platform (Python 3.12, numpy 2.2, scipy 1.14); where the measured error is exactly zero the tolerance is a
floor (1e-14, or 1e-13 for symmetry defects that are built from finite-difference derivatives). ``pytest -s`` prints
the measured value next to the tolerance (lines starting with MEASURED).

Every test that draws random numbers owns its generator (``np.random.default_rng(<own seed>)``), so the result does
not depend on the order of the tests. The tolerance of such a test is a round number at least 3 times the maximum of
the error over seeds 0..299 (comment next to each test; measured with the script that replaces ``_check`` and calls
the ``_case`` function with ``default_rng(seed)``).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import j0, j1, y0, y1

from kozmos.mechanics import hamiltonian as hm
from kozmos.mechanics import lagrangian as lg
from kozmos.mechanics.hamiltonian import LegendreTransformError


def _check(name: str, error: float, tolerance: float) -> None:
    print(f"MEASURED {name}: {error:.3e} (tolerance {tolerance:.1e})")
    assert error <= tolerance, f"{name}: {error:.3e} > {tolerance:.1e}"


M, K = 1.7, 2.3


def oscillator(q, v, t):
    return 0.5 * M * np.sum(v * v) - 0.5 * K * np.sum(q * q)


def oscillator_h(q, p, t):
    return 0.5 * np.sum(p * p) / M + 0.5 * K * np.sum(q * q)


# --------------------------------------------------------------------------------------
# Legendre transform
# --------------------------------------------------------------------------------------


def test_quadratic_lagrangian_is_inverted_in_one_newton_step():
    h = hm.hamiltonian_from_lagrangian(oscillator)
    calls = []
    original = h._newton_matrix

    def counting(q, v, t):
        calls.append(1)
        return original(q, v, t)

    h._newton_matrix = counting
    q, p = np.array([0.4, -1.1]), np.array([0.9, 0.35])
    _check("oscillator H(q, p)", abs(h(q, p) - oscillator_h(q, p, 0.0)), 1e-15)
    assert len(calls) == 1, "a Lagrangian quadratic in v must need exactly one Newton step"
    _check("v(p) = p / m", float(np.max(np.abs(h.velocity(q, p) - p / M))), 1e-14)


@pytest.mark.parametrize("momentum", [0.0, 0.1, 1.0, 10.0, 50.0, 60.0, 100.0, 1000.0, 6000.0])
def test_free_relativistic_particle(momentum):
    mass = 1.4

    def relativistic(q, v, t):
        return -mass * np.sqrt(1.0 - np.sum(v * v))

    h = hm.hamiltonian_from_lagrangian(relativistic)
    p = momentum * np.array([0.6, -0.8, 0.0])
    exact = np.sqrt(momentum**2 + mass**2)
    # measured over |p| = 0, 0.1, 1, 10, 50, 60, 100, 1000, 6000 (mass 1.4, so up to 4286 m): H error <= 1.8e-16,
    # dq/dt error <= 1.3e-15 (at |p| = 6000)
    _check(f"H = sqrt(p^2 + m^2), |p| = {momentum}", abs(h(np.zeros(3), p) - exact) / exact, 1e-14)
    dq, dp = hm.hamilton_equations(h, np.zeros(3), p)
    _check(f"dq/dt = p / H, |p| = {momentum}", float(np.max(np.abs(dq - p / exact))), 2e-14)
    assert np.all(dp == 0.0)


def test_inversion_boundary_region_of_the_relativistic_particle():
    """L = -m sqrt(1 - v^2): the inversion works up to p = 2.125e5 m (bisection, see the docstring of
    ``hamiltonian_from_lagrangian``; it was 50.3 m and then 6683 m in earlier versions) and raises
    ``LegendreTransformError`` above, where 1 - v is below 1.1e-11. If the differentiation helpers change, re-measure the
    limit and update the docstring and this test together."""
    mass = 1.4

    def relativistic(q, v, t):
        return -mass * np.sqrt(1.0 - np.sum(v * v))

    h = hm.hamiltonian_from_lagrangian(relativistic)
    # measured for p / m = 40 ... 2e5: relative error of H <= 2.2e-16, error of v <= 1.9e-15 (at 3000 m)
    for ratio in (40.0, 50.0, 55.0, 60.0, 100.0, 1000.0, 3000.0, 6000.0, 6600.0, 6800.0, 1e4, 1e5, 2e5):
        p = np.array([ratio * mass])
        exact = np.hypot(p[0], mass)
        _check(f"H(p = {ratio} m)", abs(h(np.zeros(1), p) - exact) / exact, 1e-14)
        _check(f"v(p = {ratio} m)", abs(h.velocity(np.zeros(1), p)[0] - p[0] / exact), 2e-14)
    for ratio in (2.3e5, 1e6, 1e8):
        with pytest.raises(LegendreTransformError):
            h(np.zeros(1), np.array([ratio * mass]))


def test_inversion_failures_are_reported_not_hidden():
    def relativistic(q, v, t):
        return -np.sqrt(1.0 - np.sum(v * v))

    h = hm.hamiltonian_from_lagrangian(relativistic)
    with pytest.raises(LegendreTransformError):
        h(np.zeros(1), np.array([1e6]))  # beyond 2.125e5 m (see the boundary test)

    def no_solution(q, v, t):
        return -np.cos(v[0])  # p = sin(v) never reaches 2

    with pytest.raises(LegendreTransformError):
        hm.hamiltonian_from_lagrangian(no_solution)(np.zeros(1), np.array([2.0]))


def test_charged_particle_hamiltonian_and_equations():
    charge, field, mass = 0.8, 1.9, 1.3

    def magnetic(q, v, t):
        return 0.5 * mass * np.sum(v * v) + charge * 0.5 * field * (-q[1] * v[0] + q[0] * v[1])

    h = hm.hamiltonian_from_lagrangian(magnetic)
    q, p = np.array([0.7, -0.4, 0.1]), np.array([0.5, 0.9, -0.3])
    a_vec = 0.5 * field * np.array([-q[1], q[0], 0.0])
    kinetic = p - charge * a_vec
    _check("H = (p - e A)^2 / 2m", abs(h(q, p) - kinetic @ kinetic / (2 * mass)), 1e-15)
    dq, dp = hm.hamilton_equations(h, q, p)
    _check("dq/dt = (p - e A) / m", float(np.max(np.abs(dq - kinetic / mass))), 1e-14)
    v = kinetic / mass
    dp_exact = np.array([0.5 * charge * field * v[1], -0.5 * charge * field * v[0], 0.0])
    _check("dp/dt = e grad(A.v)", float(np.max(np.abs(dp - dp_exact))), 1e-14)


# L = (1 + c t) v^2 / 2 - q^2 / 2  =>  p = (1 + c t) v,  H = p^2 / (2 (1 + c t)) + q^2 / 2.
TD_C = 0.3


def time_dependent_mass(q, v, t):
    return 0.5 * (1.0 + TD_C * t) * np.sum(v * v) - 0.5 * np.sum(q * q)


def test_legendre_derivatives_include_the_explicit_time_component():
    h = hm.hamiltonian_from_lagrangian(time_dependent_mass)
    for q0, p0, t in ((0.7, 1.3, 0.9), (-0.4, -0.8, 0.0), (1.1, 0.2, 2.5)):
        q, p = np.array([q0]), np.array([p0])
        m_t = 1.0 + TD_C * t
        exact_h = p0**2 / (2.0 * m_t) + 0.5 * q0**2
        exact = (q0, p0 / m_t, -0.5 * TD_C * p0**2 / m_t**2)  # dH/dq, dH/dp, dH/dt
        dh_dq, dh_dp, dh_dt = h.derivatives(q, p, t)
        # measured over the three points: H 1.1e-16, dH/dq 0, dH/dp 2.2e-16, dH/dt 8.3e-17 (absolute)
        _check(f"H(q, p, t = {t})", abs(h(q, p, t) - exact_h), 2e-15)
        _check(f"dH/dq (t = {t})", abs(dh_dq[0] - exact[0]), 1e-15)
        _check(f"dH/dp (t = {t})", abs(dh_dp[0] - exact[1]), 2e-15)
        _check(f"dH/dt (t = {t})", abs(dh_dt - exact[2]), 1e-15)
    # the sign of dH/dt matters: it is -dL/dt at fixed v, which is -c v^2 / 2 = -c p^2 / (2 m^2)
    assert h.derivatives(np.array([0.7]), np.array([1.3]), 0.9)[2] < 0.0


def _bessel_solution(times, q0, p0):
    """Closed form of m(t) q'' + c q' + q = 0, m = 1 + c t: q = A J0(x) + B Y0(x), x = 2 sqrt(m) / c, p = m q'."""
    x0 = 2.0 / TD_C
    # q'(0) = p0 (m(0) = 1) and dq/dt = -(A J1 + B Y1) / sqrt(m)
    matrix = np.array([[j0(x0), y0(x0)], [-j1(x0), -y1(x0)]])
    a, b = np.linalg.solve(matrix, [q0, p0])
    m_t = 1.0 + TD_C * times
    x = 2.0 * np.sqrt(m_t) / TD_C
    q = a * j0(x) + b * y0(x)
    p = -np.sqrt(m_t) * (a * j1(x) + b * y1(x))
    return q, p


def test_integrated_trajectory_with_explicit_time_dependence_against_bessel_closed_form():
    times = np.linspace(0.0, 8.0, 17)
    q_exact, p_exact = _bessel_solution(times, 1.0, 0.5)
    h = hm.hamiltonian_from_lagrangian(time_dependent_mass)
    q_h, p_h = hm.integrate_hamiltonian(h, [1.0], [0.5], times)
    # measured: q 2.5e-11, p 6.2e-11 (DOP853, rtol 1e-10, atol 1e-12)
    _check("time-dependent mass q(t) (Legendre Hamiltonian)", float(np.max(np.abs(q_h[:, 0] - q_exact))), 1e-9)
    _check("time-dependent mass p(t) (Legendre Hamiltonian)", float(np.max(np.abs(p_h[:, 0] - p_exact))), 1e-9)

    def hamiltonian(q, p, t):  # the closed-form Hamiltonian, complex-step differentiable
        return p[0] ** 2 / (2.0 * (1.0 + TD_C * t)) + 0.5 * q[0] ** 2

    q_c, p_c = hm.integrate_hamiltonian(hamiltonian, [1.0], [0.5], times)
    _check("time-dependent mass q(t) (closed-form H)", float(np.max(np.abs(q_c[:, 0] - q_exact))), 1e-9)
    assert abs(q_exact[-1] - 1.0) > 0.1  # the trajectory is not trivial


def test_legendre_hamiltonian_rejects_complex_step_and_works_with_finite():
    h = hm.hamiltonian_from_lagrangian(oscillator)
    q, p = np.array([0.4]), np.array([0.9])
    with pytest.raises(ValueError, match="not complex-analytic"):
        h(q.astype(complex), p)
    value = hm.poisson_bracket(h, oscillator_h, q, p, method="finite")  # [H, H'] with the same H: zero
    _check("[H_legendre, H] by finite differences", abs(value), 1e-14)


# --------------------------------------------------------------------------------------
# Hamilton's equations and integration
# --------------------------------------------------------------------------------------


def test_hamilton_equations_oscillator_both_methods():
    q, p = np.array([0.4, -1.1]), np.array([0.9, 0.35])
    for method, tolerance in (("complex", 1e-15), ("finite", 1e-11)):
        dq, dp = hm.hamilton_equations(oscillator_h, q, p, method=method)
        _check(f"dq/dt = p/m ({method})", float(np.max(np.abs(dq - p / M))), tolerance)
        _check(f"dp/dt = -k q ({method})", float(np.max(np.abs(dp + K * q))), tolerance)


def test_integrate_hamiltonian_oscillator_and_energy():
    omega = np.sqrt(K / M)
    times = np.linspace(0.0, 10.0, 21)
    q, p = hm.integrate_hamiltonian(oscillator_h, [1.0], [0.5], times)
    q_exact = np.cos(omega * times) + 0.5 / (M * omega) * np.sin(omega * times)
    p_exact = M * (-omega * np.sin(omega * times) + 0.5 / M * np.cos(omega * times))
    _check("Hamilton oscillator q(t)", float(np.max(np.abs(q[:, 0] - q_exact))), 1e-9)
    _check("Hamilton oscillator p(t)", float(np.max(np.abs(p[:, 0] - p_exact))), 1e-9)
    energy = np.array([oscillator_h(q[i], p[i], 0.0) for i in range(len(times))])
    _check("Hamilton oscillator energy drift", float(np.ptp(energy)), 1e-8)


def test_lagrangian_and_hamiltonian_integrations_agree_on_the_pendulum():
    g, length, bob = 9.81, 1.3, 0.8

    def pendulum(q, v, t):
        return 0.5 * bob * length**2 * v[0] ** 2 + bob * g * length * np.cos(q[0])

    times = np.linspace(0.0, 4.0, 17)
    q_l, v_l = lg.integrate_lagrangian(pendulum, [2.0], [0.0], times)
    p_l = bob * length**2 * v_l  # p = dL/dv written out
    h = hm.hamiltonian_from_lagrangian(pendulum)
    q_h, p_h = hm.integrate_hamiltonian(h, [2.0], [0.0], times)
    _check("pendulum q: Lagrange vs Hamilton", float(np.max(np.abs(q_l - q_h))), 1e-10)
    _check("pendulum p: Lagrange vs Hamilton", float(np.max(np.abs(p_l - p_h))), 5e-10)
    # closed form of the Hamiltonian from the Legendre transform
    point_q, point_p = np.array([0.7]), np.array([1.3])
    exact = point_p[0] ** 2 / (2 * bob * length**2) - bob * g * length * np.cos(0.7)
    _check("pendulum H(q, p) closed form", abs(h(point_q, point_p) - exact), 1e-14)
    dq, dp = hm.hamilton_equations(h, point_q, point_p)
    _check("pendulum dq/dt", abs(dq[0] - point_p[0] / (bob * length**2)), 1e-14)
    _check("pendulum dp/dt = -m g l sin q", abs(dp[0] + bob * g * length * np.sin(0.7)), 1e-14)


def test_relativistic_particle_in_a_constant_force_hyperbolic_motion():
    mass, force, q0 = 1.1, 0.9, 0.3

    def lagrangian(q, v, t):
        return -mass * np.sqrt(1.0 - v[0] ** 2) - force * q[0]

    times = np.linspace(0.0, 2.0, 9)
    p_exact = -force * times
    energy = np.sqrt(p_exact**2 + mass**2)
    q_exact = q0 + (mass - energy) / force  # E + F q is conserved
    h = hm.hamiltonian_from_lagrangian(lagrangian)
    q_h, p_h = hm.integrate_hamiltonian(h, [q0], [0.0], times)
    _check("hyperbolic motion q (Hamilton)", float(np.max(np.abs(q_h[:, 0] - q_exact))), 5e-10)
    _check("hyperbolic motion p (Hamilton)", float(np.max(np.abs(p_h[:, 0] - p_exact))), 2e-14)
    q_l, v_l = lg.integrate_lagrangian(lagrangian, [q0], [0.0], times)
    _check("hyperbolic motion q (Lagrange)", float(np.max(np.abs(q_l[:, 0] - q_exact))), 5e-10)
    _check("hyperbolic motion v (Lagrange)", float(np.max(np.abs(v_l[:, 0] - p_exact / energy))), 1e-9)
    # total energy H = sqrt(p^2 + m^2) + F q is conserved
    total = np.array([h(q_h[i], p_h[i]) for i in range(len(times))])
    _check("relativistic H drift", float(np.ptp(total)), 5e-10)


# --------------------------------------------------------------------------------------
# Poisson brackets
# --------------------------------------------------------------------------------------


def _lx(q, p, t):
    return q[1] * p[2] - q[2] * p[1]


def _ly(q, p, t):
    return q[2] * p[0] - q[0] * p[2]


def _lz(q, p, t):
    return q[0] * p[1] - q[1] * p[0]


def _kepler_h(q, p, t):
    return 0.5 * np.sum(p * p) - 1.0 / np.sqrt(np.sum(q * q))


def _fundamental_brackets_case(rng):
    q, p = rng.normal(size=3), rng.normal(size=3)
    worst = 0.0
    for i in range(3):
        for j in range(3):
            worst = max(
                worst,
                abs(
                    hm.poisson_bracket(lambda a, b, t, i=i: a[i], lambda a, b, t, j=j: b[j], q, p)
                    - (1.0 if i == j else 0.0)
                ),
                abs(hm.poisson_bracket(lambda a, b, t, i=i: a[i], lambda a, b, t, j=j: a[j], q, p)),
                abs(hm.poisson_bracket(lambda a, b, t, i=i: b[i], lambda a, b, t, j=j: b[j], q, p)),
            )
    _check("[q_i, p_j] = delta_ij, [q, q] = [p, p] = 0", worst, 1e-15)  # max over 300 seeds: 0 (floor 1e-15)


def test_fundamental_brackets():
    _fundamental_brackets_case(np.random.default_rng(101))


def _angular_momentum_case(rng):
    q, p = rng.normal(size=3), rng.normal(size=3)
    lx, ly, lz = _lx(q, p, 0), _ly(q, p, 0), _lz(q, p, 0)
    errors = [
        abs(hm.poisson_bracket(_lx, _ly, q, p) - lz),
        abs(hm.poisson_bracket(_ly, _lz, q, p) - lx),
        abs(hm.poisson_bracket(_lz, _lx, q, p) - ly),
        abs(hm.poisson_bracket(_ly, _lx, q, p) + lz),  # antisymmetry
    ]
    _check("{Lx, Ly} = Lz and cyclic", max(errors), 5e-15)  # max over 300 seeds: 8.9e-16

    def l_squared(a, b, t):
        return _lx(a, b, t) ** 2 + _ly(a, b, t) ** 2 + _lz(a, b, t) ** 2

    _check("{L^2, Lz} = 0", abs(hm.poisson_bracket(l_squared, _lz, q, p)), 5e-14)  # max over 300 seeds: 1.4e-14


def test_angular_momentum_algebra_and_casimir():
    _angular_momentum_case(np.random.default_rng(102))


def _kepler_case(rng):
    q, p = rng.normal(size=3), rng.normal(size=3)

    def runge_lenz_x(a, b, t):
        # A = p x L - mu r_hat (unit mass, mu = 1), x component
        return b[1] * _lz(a, b, t) - b[2] * _ly(a, b, t) - a[0] / np.sqrt(np.sum(a * a))

    worst = max(abs(hm.poisson_bracket(f, _kepler_h, q, p)) for f in (_lx, _ly, _lz, runge_lenz_x))
    _check("{L, H} = 0 and {A_x, H} = 0 for the Kepler Hamiltonian", worst, 2e-14)  # max over 300 seeds: 3.1e-15


def test_kepler_hamiltonian_commutes_with_angular_momentum_and_runge_lenz():
    _kepler_case(np.random.default_rng(103))


def _torque_case(rng):
    def linear_potential_h(q, p, t):
        return 0.5 * np.sum(p * p) + q[0]  # force (-1, 0, 0)

    q, p = rng.normal(size=3), rng.normal(size=3)
    # torque z = x F_y - y F_x = y (closed form); {Lz, H} = y
    _check(
        "{Lz, H} = y", abs(hm.poisson_bracket(_lz, linear_potential_h, q, p) - q[1]), 1e-15
    )  # max over 300 seeds: 2.2e-16 (floor 1e-15)
    _check("dLz/dt = {Lz, H}", abs(hm.observable_derivative(_lz, linear_potential_h, q, p) - q[1]), 1e-15)


def test_non_central_force_torque_through_observable_derivative():
    _torque_case(np.random.default_rng(104))


def _jacobi_leibniz_case(rng):
    q, p = rng.normal(size=3), rng.normal(size=3)

    def f(a, b, t):
        return a[0] * b[1] + np.sin(a[2]) * b[0]

    def g(a, b, t):
        return a[1] ** 2 * b[2] + np.cos(b[0]) * a[0]

    def r(a, b, t):
        return b[0] * b[1] * a[2] + np.exp(a[1])

    def nested(x, y, z):
        # [x, [y, z]] with the inner bracket by complex step and the outer one by finite differences
        return hm.poisson_bracket(x, lambda a, b, t: hm.poisson_bracket(y, z, a, b, t), q, p, method="finite")

    total = nested(f, g, r) + nested(g, r, f) + nested(r, f, g)
    scale = max(abs(nested(f, g, r)), abs(nested(g, r, f)), abs(nested(r, f, g)), 1.0)
    _check("Jacobi identity (relative to the largest term)", abs(total) / scale, 2e-11)  # max over 300 seeds: 3.3e-12

    def product(a, b, t):
        return f(a, b, t) * g(a, b, t)

    left = hm.poisson_bracket(product, r, q, p)
    right = f(q, p, 0) * hm.poisson_bracket(g, r, q, p) + g(q, p, 0) * hm.poisson_bracket(f, r, q, p)
    _check("Leibniz rule", abs(left - right) / max(1.0, abs(left)), 5e-15)  # max over 300 seeds: 8.9e-16


def test_jacobi_identity_and_leibniz_rule():
    _jacobi_leibniz_case(np.random.default_rng(105))


# --------------------------------------------------------------------------------------
# conformal particle: sl(2, R) algebra and time-dependent charges
# --------------------------------------------------------------------------------------

MC, ALPHA = 1.3, 0.7


def _charge(index):
    def charge(q, p, t):
        return hm.conformal_particle_charges(q[0], p[0], t, MC, ALPHA)[index]

    return charge


def _conformal_case(rng):
    for _ in range(5):
        q, p, t = np.array([rng.uniform(0.5, 2.0)]), rng.normal(size=1), rng.uniform(-2.0, 2.0)
        h, charge, k = (c(q, p, t) for c in (_charge(0), _charge(1), _charge(2)))
        worst = max(
            abs(hm.poisson_bracket(_charge(1), _charge(0), q, p, t) - h),  # [Q, H] = H
            abs(hm.poisson_bracket(_charge(1), _charge(2), q, p, t) + k),  # [Q, K] = -K
            abs(hm.poisson_bracket(_charge(0), _charge(2), q, p, t) - 2.0 * charge),  # [H, K] = 2 Q
            abs(2.0 * k * h + 2.0 * charge**2 + MC * ALPHA),  # 2KH + 2Q^2 + m alpha = 0
        )
        _check("sl(2,R) algebra and 2KH + 2Q^2 + m alpha = 0", worst, 2e-13)  # max over 300 seeds: 3.2e-14


def test_conformal_charges_algebra_and_relation():
    _conformal_case(np.random.default_rng(106))


def test_conformal_charges_are_conserved_by_the_flow_including_explicit_time():
    h_fun = _charge(0)
    q, p, t = np.array([1.1]), np.array([0.4]), 0.9
    worst = max(abs(hm.observable_derivative(_charge(i), h_fun, q, p, t)) for i in range(3))
    _check("dQ/dt = {Q, H} + dQ/dt (explicit) = 0 for H, Q, K", worst, 1e-14)
    # Q and K depend explicitly on t, so the bracket alone is not zero for Q
    assert abs(hm.poisson_bracket(_charge(1), h_fun, q, p, t)) > 0.1

    times = np.linspace(0.0, 3.0, 13)
    q_t, p_t = hm.integrate_hamiltonian(h_fun, [1.1], [0.4], times)
    values = np.array([[_charge(i)(q_t[j], p_t[j], times[j]) for i in range(3)] for j in range(len(times))])
    _check("H, Q, K drift along the orbit", float(np.max(np.ptp(values, axis=0))), 2e-9)


# --------------------------------------------------------------------------------------
# Noether's inverse theorem
# --------------------------------------------------------------------------------------


def test_canonical_symmetry_variation():
    q, p = np.array([0.7, -0.4, 0.1]), np.array([0.5, 0.9, -0.3])
    dq, dp = hm.canonical_symmetry_variation(_lz, q, p)
    _check("Lz generates (-y, x, 0)", float(np.max(np.abs(dq - np.array([-q[1], q[0], 0.0])))), 1e-15)
    _check("Lz generates (-py, px, 0) on the momenta", float(np.max(np.abs(dp - np.array([-p[1], p[0], 0.0])))), 1e-15)
    dq_h, dp_h = hm.canonical_symmetry_variation(oscillator_h, q[:2], p[:2])
    flow_q, flow_p = hm.hamilton_equations(oscillator_h, q[:2], p[:2])
    assert np.allclose(dq_h, flow_q, atol=1e-15, rtol=0) and np.allclose(dp_h, flow_p, atol=1e-15, rtol=0)


def test_input_validation():
    with pytest.raises(ValueError, match="same length"):
        hm.hamilton_equations(oscillator_h, [0.0, 1.0], [0.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        hm.integrate_hamiltonian(oscillator_h, [0.0], [1.0], [1.0])
