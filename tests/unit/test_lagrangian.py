"""Lagrangian mechanics against closed forms, conservation laws and independent oracles.

Oracles used (none of them calls the code under test):
* harmonic oscillator and driven oscillator: analytic solutions, scipy.integrate.quad;
* plane pendulum: the Jacobi elliptic solution theta(t) = 2 arcsin(k sn(K + w t | k^2)) (scipy.special.ellipj);
* double pendulum: the textbook 2x2 system for the accelerations, written out here;
* Kepler: the solution of Kepler's equation by brentq, the closed-form energy, angular momentum and the
  Laplace-Runge-Lenz vector;
* conical pendulum: constant polar angle with phi_dot^2 = g / (l cos theta);
* charged particle in a uniform magnetic field: the circular motion with the cyclotron frequency e B / m;
* free relativistic particle: p = m v / sqrt(1 - v^2), W = m (1 - v^2)^(-3/2), h = m / sqrt(1 - v^2).

Every tolerance is a round number (1, 2 or 5 times a power of ten) at least ten times the error measured on the
reference platform (Python 3.12, numpy 2.2, scipy 1.14); where the measured error is exactly zero the tolerance is a
floor (1e-14, or 1e-13 for symmetry defects that are built from finite-difference derivatives). ``pytest -s`` prints
the measured value next to the tolerance (lines starting with MEASURED).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ellipj, ellipk

from kozmos.mechanics import lagrangian as lg
from kozmos.mechanics.lagrangian import SingularInertiaError


def _check(name: str, error: float, tolerance: float) -> None:
    print(f"MEASURED {name}: {error:.3e} (tolerance {tolerance:.1e})")
    assert error <= tolerance, f"{name}: {error:.3e} > {tolerance:.1e}"


# --------------------------------------------------------------------------------------
# harmonic oscillator
# --------------------------------------------------------------------------------------

M, K = 1.7, 2.3
OMEGA = np.sqrt(K / M)


def oscillator(q, v, t):
    return 0.5 * M * np.sum(v * v) - 0.5 * K * np.sum(q * q)


def test_oscillator_momentum_force_inertia_energy():
    q, v = np.array([0.4, -1.1]), np.array([0.7, 0.2])
    _check("p = m v", np.max(np.abs(lg.generalized_momenta(oscillator, q, v) - M * v)), 1e-15)
    _check("F = -k q", np.max(np.abs(lg.generalized_forces(oscillator, q, v) + K * q)), 1e-15)
    _check("W = m 1", np.max(np.abs(lg.inertia_matrix(oscillator, q, v) - M * np.eye(2))), 1e-12)
    energy = 0.5 * M * v @ v + 0.5 * K * q @ q
    _check("h = T + V", abs(lg.jacobi_function(oscillator, q, v) - energy), 1e-15)


def test_oscillator_accelerations_and_residual():
    q, v = np.array([0.4, -1.1]), np.array([0.7, 0.2])
    a = lg.euler_lagrange_accelerations(oscillator, q, v)
    _check("a = -k q / m", np.max(np.abs(a + K * q / M)), 5e-13)
    assert np.max(np.abs(lg.euler_lagrange_residual(oscillator, q, v, a))) < 1e-10
    wrong = a + np.array([0.3, -0.2])
    residual = lg.euler_lagrange_residual(oscillator, q, v, wrong)
    _check("residual = m (a - a_true)", np.max(np.abs(residual - M * np.array([0.3, -0.2]))), 2e-13)


def test_oscillator_integration_against_closed_form():
    times = np.linspace(0.0, 10.0, 21)
    q, v = lg.integrate_lagrangian(oscillator, [1.0, -0.5], [0.5, 0.3], times)
    q_exact = np.stack(
        [
            np.cos(OMEGA * times) + 0.5 / OMEGA * np.sin(OMEGA * times),
            -0.5 * np.cos(OMEGA * times) + 0.3 / OMEGA * np.sin(OMEGA * times),
        ],
        axis=1,
    )
    _check("oscillator q(t)", np.max(np.abs(q - q_exact)), 1e-9)
    h = np.array([lg.jacobi_function(oscillator, q[i], v[i]) for i in range(len(times))])
    _check("oscillator h drift", float(np.ptp(h)), 5e-9)


def test_finite_method_agrees_with_complex_step():
    q, v = np.array([0.4, -1.1]), np.array([0.7, 0.2])
    a_complex = lg.euler_lagrange_accelerations(oscillator, q, v, method="complex")
    a_finite = lg.euler_lagrange_accelerations(oscillator, q, v, method="finite")
    _check("finite vs complex accelerations", np.max(np.abs(a_complex - a_finite)), 1e-9)


def test_non_analytic_lagrangian_needs_the_finite_method():
    def with_abs(q, v, t):
        return 0.5 * abs(v[0]) ** 2 * (1.0 + 0.1 * abs(q[0])) - 0.5 * q[0] ** 2

    q, v = np.array([0.4]), np.array([0.7])
    with pytest.raises(ValueError, match="not analytic"):
        lg.generalized_momenta(with_abs, q, v)
    exact = 0.7 * (1.0 + 0.04)
    _check("finite method with abs()", abs(lg.generalized_momenta(with_abs, q, v, method="finite")[0] - exact), 1e-13)


def test_singular_inertia_matrix_raises_a_clear_error():
    def linear_in_velocity(q, v, t):
        return q[0] * v[0] - 0.5 * q[0] ** 2

    def degenerate(q, v, t):
        return 0.5 * (v[0] + v[1]) ** 2 - 0.5 * (q[0] ** 2 + q[1] ** 2)

    with pytest.raises(SingularInertiaError, match="zero on its diagonal"):
        lg.euler_lagrange_accelerations(linear_in_velocity, [0.3], [0.1])
    with pytest.raises(SingularInertiaError, match="condition number"):
        lg.euler_lagrange_accelerations(degenerate, [0.3, 0.2], [0.1, 0.0])


def test_max_condition_rejects_at_a_tight_limit_and_accepts_at_a_loose_one():
    """The condition number is taken after scaling W to unit diagonal, so that it measures degeneracy and not units.

    W = [[1, 0.98], [0.98, 1]] has condition number 1.98 / 0.02 = 99: rejected at max_condition = 10, accepted at
    1e3 and 1e12. W = diag(1, 50) has condition number 1 after scaling and is accepted at any limit above 1.
    """

    def coupled(q, v, t):
        return 0.5 * (v[0] ** 2 + v[1] ** 2) + 0.98 * v[0] * v[1] - 0.5 * (q[0] ** 2 + q[1] ** 2)

    def unequal_masses(q, v, t):
        return 0.5 * v[0] ** 2 + 0.5 * 50.0 * v[1] ** 2 - 0.5 * (q[0] ** 2 + q[1] ** 2)

    q, v = np.array([0.3, 0.2]), np.array([0.1, -0.4])
    with pytest.raises(SingularInertiaError, match="condition number 99"):
        lg.euler_lagrange_accelerations(coupled, q, v, max_condition=10.0)
    with pytest.raises(SingularInertiaError, match="condition number"):
        lg.integrate_lagrangian(coupled, q, v, [0.0, 1.0], max_condition=10.0)
    exact = np.linalg.solve(np.array([[1.0, 0.98], [0.98, 1.0]]), -q)
    for limit in (1e3, 1e12):
        a = lg.euler_lagrange_accelerations(coupled, q, v, max_condition=limit)
        _check(f"coupled: W a = -q (max_condition = {limit:g})", float(np.max(np.abs(a - exact))), 5e-11)
    a = lg.euler_lagrange_accelerations(unequal_masses, q, v, max_condition=2.0)
    _check("a = -q / m with masses 1 and 50", float(np.max(np.abs(a + q / np.array([1.0, 50.0])))), 5e-14)


def test_coordinates_of_different_units_do_not_look_singular():
    """Polar Kepler problem in SI units: W = diag(m, m r^2) with r = 1.5e11 m has a plain condition number of 2e22."""
    mass, gm = 6e24, 6.674e-11 * 1.989e30

    def kepler(q, v, t):
        return 0.5 * mass * (v[0] ** 2 + q[0] ** 2 * v[1] ** 2) + gm * mass / q[0]

    r, omega, v_r = 1.5e11, 2e-7, 1e3
    a = lg.euler_lagrange_accelerations(kepler, [r, 1.3], [v_r, omega])
    exact = np.array([r * omega**2 - gm / r**2, -2.0 * v_r * omega / r])
    _check("polar Kepler in SI units", float(np.max(np.abs(a / exact - 1.0))), 5e-12)  # measured 4e-14


def test_planet_in_si_units_with_a_coordinate_at_zero():
    """Cartesian Kepler problem at (7.8e11 m, 0): the first versions returned a_y = 3.4e-5 m/s^2 instead of 0."""
    mass, gm = 1e27, 6.674e-11 * 1.989e30

    def planet(q, v, t):
        return 0.5 * mass * (v[0] ** 2 + v[1] ** 2) + gm * mass / np.sqrt(q[0] ** 2 + q[1] ** 2)

    for method, tolerance in (("complex", 1e-12), ("finite", 1e-6)):  # measured 1e-16 and 8e-9
        a = lg.euler_lagrange_accelerations(planet, [7.8e11, 0.0], [0.0, 3e4], 0.0, method)
        expected = -gm / 7.8e11**2
        _check(f"a_x of the planet ({method})", abs(a[0] / expected - 1.0), tolerance)
        _check(f"a_y of the planet ({method})", abs(a[1] / expected), tolerance)


def test_accelerations_that_cannot_be_determined_are_refused():
    """v = 5e5 with unit mass and a unit spring, finite method: the force is 1e-12 of the Lagrangian per unit of q.

    The first versions returned -0.413 for the true -0.3.
    """
    with pytest.raises(lg.DerivativePrecisionError, match="Rescale"):
        lg.euler_lagrange_accelerations(lambda q, v, t: 0.5 * v[0] ** 2 - 0.5 * q[0] ** 2, [0.3], [5e5], 0.0, "finite")
    a = lg.euler_lagrange_accelerations(lambda q, v, t: 0.5 * v[0] ** 2 - 0.5 * q[0] ** 2, [0.3], [5e5])
    assert a[0] == pytest.approx(
        -0.3, rel=1e-9, abs=0
    )  # the complex step is exact whatever the size of the kinetic term


def test_input_validation():
    with pytest.raises(ValueError, match="same length"):
        lg.generalized_momenta(oscillator, [0.0, 1.0], [0.0])
    with pytest.raises(ValueError, match="1-D"):
        lg.generalized_forces(oscillator, 1.0, 1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        lg.integrate_lagrangian(oscillator, [0.0], [1.0], [0.0, 0.0, 1.0])


# --------------------------------------------------------------------------------------
# plane pendulum (non-quadratic in q)
# --------------------------------------------------------------------------------------

G_ACC, LEN, BOB = 9.81, 1.3, 0.8


def pendulum(q, v, t):
    return 0.5 * BOB * LEN**2 * v[0] ** 2 + BOB * G_ACC * LEN * np.cos(q[0])


def test_pendulum_accelerations_closed_form():
    for theta in (0.2, 1.0, 2.5):
        a = lg.euler_lagrange_accelerations(pendulum, [theta], [0.4])
        _check("pendulum a", abs(a[0] + G_ACC / LEN * np.sin(theta)), 2e-12)
        assert lg.generalized_forces(pendulum, [theta], [0.4])[0] == pytest.approx(
            -BOB * G_ACC * LEN * np.sin(theta), rel=1e-14, abs=0
        )


def test_pendulum_against_jacobi_elliptic_solution():
    theta0 = 2.0
    k = np.sin(theta0 / 2.0)
    omega = np.sqrt(G_ACC / LEN)
    times = np.linspace(0.0, 6.0, 25)
    q, v = lg.integrate_lagrangian(pendulum, [theta0], [0.0], times)
    sn, _, _, _ = ellipj(ellipk(k**2) + omega * times, k**2)
    exact = 2.0 * np.arcsin(k * sn)
    _check("pendulum theta(t) vs elliptic", np.max(np.abs(q[:, 0] - exact)), 1e-9)
    h = np.array([lg.jacobi_function(pendulum, q[i], v[i]) for i in range(len(times))])
    _check("pendulum h drift", float(np.ptp(h)), 5e-9)
    # closed-form energy: h = -m g l cos(theta0) on this orbit
    _check("pendulum h value", abs(h[0] + BOB * G_ACC * LEN * np.cos(theta0)), 1e-14)


# --------------------------------------------------------------------------------------
# double pendulum (non-trivial, configuration-dependent mass matrix)
# --------------------------------------------------------------------------------------

M1, M2, L1, L2 = 1.1, 0.7, 1.0, 0.6


def double_pendulum(q, v, t):
    t1, t2 = q
    w1, w2 = v
    kinetic = 0.5 * (M1 + M2) * L1**2 * w1**2 + 0.5 * M2 * L2**2 * w2**2 + M2 * L1 * L2 * w1 * w2 * np.cos(t1 - t2)
    potential = -(M1 + M2) * G_ACC * L1 * np.cos(t1) - M2 * G_ACC * L2 * np.cos(t2)
    return kinetic - potential


def test_double_pendulum_mass_matrix_and_accelerations():
    q, v = np.array([0.9, -0.4]), np.array([0.3, 1.2])
    c = np.cos(q[0] - q[1])
    s = np.sin(q[0] - q[1])
    mass = np.array([[(M1 + M2) * L1**2, M2 * L1 * L2 * c], [M2 * L1 * L2 * c, M2 * L2**2]])
    _check("double pendulum W", np.max(np.abs(lg.inertia_matrix(double_pendulum, q, v) - mass)), 5e-13)
    rhs = np.array(
        [
            -M2 * L1 * L2 * v[1] ** 2 * s - (M1 + M2) * G_ACC * L1 * np.sin(q[0]),
            M2 * L1 * L2 * v[0] ** 2 * s - M2 * G_ACC * L2 * np.sin(q[1]),
        ]
    )
    a_exact = np.linalg.solve(mass, rhs)
    a = lg.euler_lagrange_accelerations(double_pendulum, q, v)
    _check("double pendulum a", np.max(np.abs(a - a_exact)), 2e-11)
    assert np.max(np.abs(lg.euler_lagrange_residual(double_pendulum, q, v, a_exact))) < 1e-8


def test_double_pendulum_energy_conservation():
    times = np.linspace(0.0, 4.0, 17)
    q, v = lg.integrate_lagrangian(double_pendulum, [1.2, 0.5], [0.0, 0.0], times)
    h = np.array([lg.jacobi_function(double_pendulum, q[i], v[i]) for i in range(len(times))])
    # closed-form energy at the start (at rest): V = -(m1+m2) g l1 cos t1 - m2 g l2 cos t2
    expected = -(M1 + M2) * G_ACC * L1 * np.cos(1.2) - M2 * G_ACC * L2 * np.cos(0.5)
    _check("double pendulum h(0)", abs(h[0] - expected), 1e-14)
    _check("double pendulum h drift", float(np.max(np.abs(h - expected))), 5e-9)


# --------------------------------------------------------------------------------------
# spherical pendulum
# --------------------------------------------------------------------------------------


def spherical_pendulum(q, v, t):
    theta = q[0]
    return 0.5 * BOB * LEN**2 * (v[0] ** 2 + np.sin(theta) ** 2 * v[1] ** 2) + BOB * G_ACC * LEN * np.cos(theta)


def test_conical_pendulum_is_a_steady_solution():
    theta0 = 0.7
    phi_dot = np.sqrt(G_ACC / (LEN * np.cos(theta0)))
    a = lg.euler_lagrange_accelerations(spherical_pendulum, [theta0, 0.0], [0.0, phi_dot])
    _check("conical pendulum accelerations", float(np.max(np.abs(a))), 2e-14)
    times = np.linspace(0.0, 4.0, 9)
    q, v = lg.integrate_lagrangian(spherical_pendulum, [theta0, 0.0], [0.0, phi_dot], times)
    # measured 3.8e-12; 5.7e-11 with sin and cos moved by one unit in the last place (the steady solution is only
    # neutrally stable, so a rounding error of the start grows along the orbit)
    _check("conical pendulum theta(t)", float(np.max(np.abs(q[:, 0] - theta0))), 1e-9)
    _check("conical pendulum phi(t)", float(np.max(np.abs(q[:, 1] - phi_dot * times))), 1e-9)  # 6.9e-11 likewise


def test_spherical_pendulum_general_orbit_conserves_h_and_pphi():
    times = np.linspace(0.0, 3.0, 13)
    q, v = lg.integrate_lagrangian(spherical_pendulum, [0.8, 0.0], [0.3, 1.1], times)
    h = np.array([lg.jacobi_function(spherical_pendulum, q[i], v[i]) for i in range(len(times))])
    p_phi = np.array([lg.generalized_momenta(spherical_pendulum, q[i], v[i])[1] for i in range(len(times))])
    _check(
        "spherical pendulum p_phi vs closed form",
        float(np.max(np.abs(p_phi - BOB * LEN**2 * np.sin(q[:, 0]) ** 2 * v[:, 1]))),
        1e-14,
    )
    _check("spherical pendulum p_phi drift", float(np.ptp(p_phi)), 2e-9)
    _check("spherical pendulum h drift", float(np.ptp(h)), 1e-8)
    assert np.ptp(q[:, 0]) > 0.05  # the polar angle really moves


# --------------------------------------------------------------------------------------
# Kepler problem, polar and Cartesian
# --------------------------------------------------------------------------------------

MU, MP, ECC, SMA = 1.0, 1.3, 0.5, 1.0  # G M, particle mass, eccentricity, semi-major axis


def kepler_polar(q, v, t):
    return 0.5 * MP * (v[0] ** 2 + q[0] ** 2 * v[1] ** 2) + MP * MU / q[0]


def kepler_cartesian(q, v, t):
    return 0.5 * MP * np.sum(v * v) + MP * MU / np.sqrt(np.sum(q * q))


def _kepler_exact(times):
    mean_motion = np.sqrt(MU / SMA**3)
    anomaly = np.array(
        [brentq(lambda e, m=mean_motion * t: e - ECC * np.sin(e) - m, -1.0, 4 * np.pi + 1.0, xtol=1e-15) for t in times]
    )
    x = SMA * (np.cos(anomaly) - ECC)
    y = SMA * np.sqrt(1.0 - ECC**2) * np.sin(anomaly)
    return x, y


E_KEPLER = -MP * MU / (2.0 * SMA)
LZ_KEPLER = MP * np.sqrt(MU * SMA * (1.0 - ECC**2))
R_PERI = SMA * (1.0 - ECC)
V_PERI = np.sqrt(MU * (1.0 + ECC) / (SMA * (1.0 - ECC)))
TIMES_KEPLER = np.linspace(0.0, 4.0 * np.pi, 33)  # two periods


def test_kepler_polar():
    q, v = lg.integrate_lagrangian(
        kepler_polar, [R_PERI, 0.0], [0.0, V_PERI / R_PERI], TIMES_KEPLER, rtol=1e-12, atol=1e-13
    )
    h = np.array([lg.jacobi_function(kepler_polar, q[i], v[i]) for i in range(len(q))])
    p_phi = np.array([lg.generalized_momenta(kepler_polar, q[i], v[i])[1] for i in range(len(q))])
    _check("Kepler polar energy vs closed form", float(np.max(np.abs(h - E_KEPLER))), 5e-11)
    _check("Kepler polar angular momentum vs closed form", float(np.max(np.abs(p_phi - LZ_KEPLER))), 5e-11)
    x_exact, y_exact = _kepler_exact(TIMES_KEPLER)
    x, y = q[:, 0] * np.cos(q[:, 1]), q[:, 0] * np.sin(q[:, 1])
    _check("Kepler polar orbit vs Kepler's equation", float(np.max(np.hypot(x - x_exact, y - y_exact))), 5e-10)


def test_kepler_cartesian_with_runge_lenz_vector():
    q, v = lg.integrate_lagrangian(kepler_cartesian, [R_PERI, 0.0], [0.0, V_PERI], TIMES_KEPLER, rtol=1e-12, atol=1e-13)
    h = np.array([lg.jacobi_function(kepler_cartesian, q[i], v[i]) for i in range(len(q))])
    p = np.array([lg.generalized_momenta(kepler_cartesian, q[i], v[i]) for i in range(len(q))])
    l_z = q[:, 0] * p[:, 1] - q[:, 1] * p[:, 0]
    r = np.hypot(q[:, 0], q[:, 1])
    # Laplace-Runge-Lenz vector A = p x L - m^2 mu r_hat (planar: L = l_z e_z); |A| = m^2 mu e, pointing along +x
    a_x = p[:, 1] * l_z - MP**2 * MU * q[:, 0] / r
    a_y = -p[:, 0] * l_z - MP**2 * MU * q[:, 1] / r
    _check("Kepler Cartesian energy", float(np.max(np.abs(h - E_KEPLER))), 1e-10)
    _check("Kepler Cartesian angular momentum", float(np.max(np.abs(l_z - LZ_KEPLER))), 2e-10)
    _check("Kepler Cartesian LRL x", float(np.max(np.abs(a_x - MP**2 * MU * ECC))), 2e-10)
    _check("Kepler Cartesian LRL y", float(np.max(np.abs(a_y))), 2e-10)
    x_exact, y_exact = _kepler_exact(TIMES_KEPLER)
    _check(
        "Kepler Cartesian orbit vs Kepler's equation",
        float(np.max(np.hypot(q[:, 0] - x_exact, q[:, 1] - y_exact))),
        5e-10,
    )


def test_kepler_energy_function_is_the_same_number_in_both_coordinate_systems():
    r, phi, r_dot, phi_dot = 0.8, 0.3, 0.2, 1.1
    polar = lg.jacobi_function(kepler_polar, [r, phi], [r_dot, phi_dot])
    q = np.array([r * np.cos(phi), r * np.sin(phi)])
    v = np.array([r_dot * np.cos(phi) - r * phi_dot * np.sin(phi), r_dot * np.sin(phi) + r * phi_dot * np.cos(phi)])
    cartesian = lg.jacobi_function(kepler_cartesian, q, v)
    _check("h invariant under the change of coordinates", abs(polar - cartesian), 1e-14)


# --------------------------------------------------------------------------------------
# time-dependent Lagrangian: energy not conserved, dh/dt = -dL/dt
# --------------------------------------------------------------------------------------

FORCE_AMPLITUDE, DRIVE = 0.9, 1.7  # F and Omega of the driving; unit mass, w^2 = 3


def driven_oscillator(q, v, t):
    return 0.5 * v[0] ** 2 - 0.5 * 3.0 * q[0] ** 2 - FORCE_AMPLITUDE * np.cos(DRIVE * t) * q[0]


def _driven_exact(times, x0, v0):
    w = np.sqrt(3.0)
    c = -FORCE_AMPLITUDE / (w**2 - DRIVE**2)
    x = (x0 - c) * np.cos(w * times) + (v0 / w) * np.sin(w * times) + c * np.cos(DRIVE * times)
    v = -(x0 - c) * w * np.sin(w * times) + v0 * np.cos(w * times) - c * DRIVE * np.sin(DRIVE * times)
    return x, v


def test_time_dependent_lagrangian_energy_balance():
    x0, v0 = 0.3, -0.2
    times = np.linspace(0.0, 8.0, 17)
    q, v = lg.integrate_lagrangian(driven_oscillator, [x0], [v0], times, rtol=1e-12, atol=1e-13)
    x_exact, v_exact = _driven_exact(times, x0, v0)
    _check("driven oscillator x(t)", float(np.max(np.abs(q[:, 0] - x_exact))), 2e-11)
    h = np.array([lg.jacobi_function(driven_oscillator, q[i], v[i], t) for i, t in enumerate(times)])
    h_closed = 0.5 * v_exact**2 + 1.5 * x_exact**2 + FORCE_AMPLITUDE * np.cos(DRIVE * times) * x_exact
    _check("h against closed form", float(np.max(np.abs(h - h_closed))), 1e-10)
    assert np.ptp(h) > 0.5, "the energy function must change when L depends on t"

    # dh/dt = -dL/dt = F Omega sin(Omega t) x: integrate the right-hand side by quad over the analytic motion
    def power(t):
        x_t, _ = _driven_exact(np.array([t]), x0, v0)
        return -FORCE_AMPLITUDE * DRIVE * np.sin(DRIVE * t) * x_t[0]

    for i in (4, 8, 16):
        change, _ = quad(power, 0.0, times[i], epsabs=1e-13, epsrel=1e-13, limit=200)
        _check(f"h(t_{i}) - h(0) = -integral dL/dt", abs((h[i] - h[0]) - change), 1e-10)


# --------------------------------------------------------------------------------------
# charged particle in a uniform magnetic field (velocity-dependent potential)
# --------------------------------------------------------------------------------------

CHARGE, BFIELD, MQ = 0.8, 1.9, 1.3


def magnetic(q, v, t):
    return 0.5 * MQ * np.sum(v * v) + CHARGE * 0.5 * BFIELD * (-q[1] * v[0] + q[0] * v[1])


def test_canonical_versus_kinetic_momentum():
    q, v = np.array([0.7, -0.4, 0.1]), np.array([0.5, 0.9, -0.3])
    p = lg.generalized_momenta(magnetic, q, v)
    a_vec = 0.5 * BFIELD * np.array([-q[1], q[0], 0.0])  # A = (B/2) (-y, x, 0)
    _check("p = m v + e A", np.max(np.abs(p - (MQ * v + CHARGE * a_vec))), 1e-14)
    assert abs(p[0] - MQ * v[0]) > 0.1  # canonical and kinetic momentum really differ
    _check(
        "h = m v^2 / 2 (magnetic field does no work)", abs(lg.jacobi_function(magnetic, q, v) - 0.5 * MQ * v @ v), 1e-14
    )


def test_charged_particle_circular_motion():
    omega = CHARGE * BFIELD / MQ
    speed = 0.9
    times = np.linspace(0.0, 8.0, 17)
    q, v = lg.integrate_lagrangian(magnetic, [0.2, -0.1, 0.0], [speed, 0.0, 0.4], times, rtol=1e-12, atol=1e-13)
    x_exact = 0.2 + speed / omega * np.sin(omega * times)
    y_exact = -0.1 + speed / omega * (np.cos(omega * times) - 1.0)
    _check("cyclotron orbit", float(np.max(np.hypot(q[:, 0] - x_exact, q[:, 1] - y_exact))), 2e-11)
    _check("vz constant", float(np.max(np.abs(v[:, 2] - 0.4))), 1e-12)
    a = lg.euler_lagrange_accelerations(magnetic, q[5], v[5])
    _check("a = (e B / m) (vy, -vx, 0)", float(np.max(np.abs(a - omega * np.array([v[5, 1], -v[5, 0], 0.0])))), 5e-13)
    h = np.array([lg.jacobi_function(magnetic, q[i], v[i]) for i in range(len(times))])
    _check("kinetic energy constant", float(np.ptp(h)), 5e-11)
    p_x = np.array([lg.generalized_momenta(magnetic, q[i], v[i])[0] for i in range(len(times))])
    assert np.ptp(p_x) > 0.1  # canonical momentum is NOT constant, the kinetic momentum magnitude is


# --------------------------------------------------------------------------------------
# explicit time dependence through the velocity: the cross term D = d2L/dv dt
# --------------------------------------------------------------------------------------


def linear_in_t_and_v(q, v, t):
    return 0.5 * v[0] ** 2 + t * v[0]  # p = v + t, dp/dt = a + 1 = 0  =>  a = -1 (D = 1, a sign error gives +1)


M0, MC, FORCE = 1.2, 0.5, 0.9


def mass_ramp(q, v, t):
    return 0.5 * M0 * (1.0 + MC * t) * v[0] ** 2 - FORCE * q[0]  # p = m(t) v, dp/dt = -F (constant force)


def breathing_oscillator(q, v, t):
    return 0.5 * M0 * (1.0 + MC * t) * v[0] ** 2 - 0.5 * K * q[0] ** 2  # m a + mdot v = -k q


def test_cross_term_d2l_dvdt_linear_lagrangian_gives_a_minus_one():
    for q0, v0, t in ((0.3, 0.7, 0.0), (-1.2, 0.0, 1.5), (0.8, -2.0, -0.6)):
        q, v = np.array([q0]), np.array([v0])
        a = lg.euler_lagrange_accelerations(linear_in_t_and_v, q, v, t)
        # measured over the three points (central difference for the mixed derivative): |a + 1| <= 3.0e-13,
        # residuals <= 4.1e-13
        _check(f"a = -1 (t = {t})", abs(a[0] + 1.0), 5e-12)
        # residual = W a + C v + D - dL/dq = a + 1: zero at a = -1, equal to a + 1 elsewhere (a sign error in D
        # would give a - 1 and fail both)
        _check(
            f"residual at a = -1 (t = {t})",
            abs(lg.euler_lagrange_residual(linear_in_t_and_v, q, v, [-1.0], t)[0]),
            5e-12,
        )
        _check(
            "residual at a = 0 equals 1",
            abs(lg.euler_lagrange_residual(linear_in_t_and_v, q, v, [0.0], t)[0] - 1.0),
            5e-12,
        )
        _check(
            "residual at a = 2 equals 3",
            abs(lg.euler_lagrange_residual(linear_in_t_and_v, q, v, [2.0], t)[0] - 3.0),
            5e-12,
        )
    # finite-difference differentiation gives the same acceleration
    a_finite = lg.euler_lagrange_accelerations(linear_in_t_and_v, [0.3], [0.7], 0.4, method="finite")
    _check("a = -1 (finite method)", abs(a_finite[0] + 1.0), 5e-10)  # measured 2.3e-11


def test_cross_term_d2l_dvdt_linear_lagrangian_trajectory():
    times = np.linspace(0.0, 4.0, 9)
    q, v = lg.integrate_lagrangian(linear_in_t_and_v, [0.5], [0.8], times)
    _check(
        "q(t) = q0 + v0 t - t^2/2", float(np.max(np.abs(q[:, 0] - (0.5 + 0.8 * times - 0.5 * times**2)))), 1e-10
    )  # measured 5.2e-12
    _check("v(t) = v0 - t", float(np.max(np.abs(v[:, 0] - (0.8 - times)))), 1e-10)  # measured 4.2e-12


def test_time_dependent_mass_accelerations_and_residual():
    # m(t) a + mdot v = -k q for L = m(t) v^2 / 2 - k q^2 / 2
    for q0, v0, t in ((0.4, 0.7, 0.0), (-1.1, 0.3, 1.3), (0.9, -1.6, 2.4)):
        m_t, mdot = M0 * (1.0 + MC * t), M0 * MC
        exact = -(K * q0 + mdot * v0) / m_t
        q, v = np.array([q0]), np.array([v0])
        a = lg.euler_lagrange_accelerations(breathing_oscillator, q, v, t)
        # measured over the three points: a <= 3.6e-14, residual <= 4.3e-14 (central difference for the mixed
        # derivative)
        _check(f"time-dependent mass a (t = {t})", abs(a[0] - exact), 5e-13)
        residual = lg.euler_lagrange_residual(breathing_oscillator, q, v, [exact + 0.25], t)
        _check(f"residual = m (a - a_true) (t = {t})", abs(residual[0] - 0.25 * m_t), 5e-13)
        # a sign error in D would shift the acceleration by 2 mdot v / m, at least 0.1 at each of these points
        assert abs(2.0 * mdot * v0 / m_t) > 0.1


def test_time_dependent_mass_trajectory_against_closed_form():
    # constant force F: m(t) v = m0 v0 - F t with m = m0 (1 + c t)  =>
    # v = (v0 + F / (m0 c)) / (1 + c t) - F / (m0 c),  q = q0 + (v0 + F / (m0 c)) ln(1 + c t) / c - F t / (m0 c)
    q0, v0 = 0.4, 0.9
    times = np.linspace(0.0, 6.0, 13)
    b = FORCE / (M0 * MC)
    q_exact = q0 + (v0 + b) * np.log1p(MC * times) / MC - b * times
    v_exact = (v0 + b) / (1.0 + MC * times) - b
    q, v = lg.integrate_lagrangian(mass_ramp, [q0], [v0], times)
    # measured: q 3.1e-11, v 7.8e-12 (DOP853, rtol 1e-10, atol 1e-12)
    _check("time-dependent mass q(t)", float(np.max(np.abs(q[:, 0] - q_exact))), 5e-10)
    _check("time-dependent mass v(t)", float(np.max(np.abs(v[:, 0] - v_exact))), 5e-10)
    assert abs(q_exact[-1] - (q0 + v0 * times[-1])) > 1.0  # the mass ramp and the force matter


# --------------------------------------------------------------------------------------
# free relativistic particle (non-quadratic kinetic term)
# --------------------------------------------------------------------------------------

MR = 1.4


def relativistic(q, v, t):
    return -MR * np.sqrt(1.0 - np.sum(v * v))


def test_relativistic_particle_momentum_inertia_energy():
    v = np.array([0.3, -0.5, 0.2])
    gamma = 1.0 / np.sqrt(1.0 - v @ v)
    q = np.zeros(3)
    _check("p = m gamma v", np.max(np.abs(lg.generalized_momenta(relativistic, q, v) - MR * gamma * v)), 1e-14)
    w_exact = MR * gamma * (np.eye(3) + gamma**2 * np.outer(v, v))
    _check("W = m gamma (1 + gamma^2 v v^T)", np.max(np.abs(lg.inertia_matrix(relativistic, q, v) - w_exact)), 1e-9)
    _check("h = m gamma", abs(lg.jacobi_function(relativistic, q, v) - MR * gamma), 1e-14)
    a = lg.euler_lagrange_accelerations(relativistic, q, v)
    _check("free particle a = 0", float(np.max(np.abs(a))), 2e-13)


# --------------------------------------------------------------------------------------
# action
# --------------------------------------------------------------------------------------


def test_action_free_particle_and_oscillator_closed_forms():
    def free(q, v, t):
        return 0.5 * 2.0 * np.sum(v * v)

    times = np.linspace(0.0, 3.0, 13)
    path = 0.7 * times[:, None] + 0.1
    _check("free particle action", abs(lg.action(free, times, path) - 0.5 * 2.0 * 0.7**2 * 3.0), 1e-14)

    def exact_l(t):
        x = np.cos(OMEGA * t) + 0.5 / OMEGA * np.sin(OMEGA * t)
        v = -OMEGA * np.sin(OMEGA * t) + 0.5 * np.cos(OMEGA * t)
        return 0.5 * M * v**2 - 0.5 * K * x**2

    exact, _ = quad(exact_l, 0.0, 2.0, epsabs=1e-13, epsrel=1e-13)
    errors = []
    for n in (11, 21, 41, 81):
        t = np.linspace(0.0, 2.0, n)
        x = np.cos(OMEGA * t) + 0.5 / OMEGA * np.sin(OMEGA * t)
        errors.append(abs(lg.action(oscillator, t, x[:, None]) - exact))
    print("MEASURED action errors for n = 11, 21, 41, 81:", errors)
    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(3)]
    print("MEASURED convergence rates (log2 of the error ratio per halving of h):", rates)
    assert errors[-1] < 1e-8
    assert min(rates) > 3.0  # at least the O(h^3) of the docstring (measured about 7 for this analytic path)


def test_action_is_stationary_on_the_true_path_and_not_on_a_false_one():
    t_end = 2.0
    times = np.linspace(0.0, t_end, 81)
    x_true = np.cos(OMEGA * times) + 0.5 / OMEGA * np.sin(OMEGA * times)
    bump = np.sin(np.pi * times / t_end)  # vanishes at both ends
    x_false = x_true + 0.4 * bump  # same end points, not a solution

    def change(path, eps):
        return lg.action(oscillator, times, (path + eps * bump)[:, None]) - lg.action(oscillator, times, path[:, None])

    eps_big, eps_small = 1e-2, 5e-3
    true_big, true_small = change(x_true, eps_big), change(x_true, eps_small)
    false_big, false_small = change(x_false, eps_big), change(x_false, eps_small)
    ratio_true = true_big / true_small
    ratio_false = false_big / false_small
    print(f"MEASURED dS(true): {true_big:.3e}, {true_small:.3e}, ratio {ratio_true:.4f}")
    print(f"MEASURED dS(false): {false_big:.3e}, {false_small:.3e}, ratio {ratio_false:.4f}")
    # second variation of S for eta = bump: (1/2) integral (m eta'^2 - k eta^2) dt per eps^2
    second, _ = quad(
        lambda s: 0.5 * (M * (np.pi / t_end * np.cos(np.pi * s / t_end)) ** 2 - K * np.sin(np.pi * s / t_end) ** 2),
        0.0,
        t_end,
    )
    _check("true path: dS = eps^2 S2", abs(true_small - second * eps_small**2) / abs(second * eps_small**2), 5e-6)
    assert ratio_true == pytest.approx(4.0, rel=1e-4, abs=0)  # O(eps^2): halving eps divides dS by 4
    assert ratio_false == pytest.approx(2.0, rel=2e-2, abs=0)  # O(eps): halving eps divides dS by 2

    # first variation on the false path from Eq. (dosI2)-type integration by parts: dS/deps = integral (-m xdd - k x) bump
    def first(t):
        x = np.cos(OMEGA * t) + 0.5 / OMEGA * np.sin(OMEGA * t)
        xdd = -(OMEGA**2) * x
        # x_false'' = xdd + 0.4 * (-(pi/T)^2 sin)
        xf = x + 0.4 * np.sin(np.pi * t / t_end)
        xfdd = xdd - 0.4 * (np.pi / t_end) ** 2 * np.sin(np.pi * t / t_end)
        return (-M * xfdd - K * xf) * np.sin(np.pi * t / t_end)

    slope, _ = quad(first, 0.0, t_end, epsabs=1e-14, epsrel=1e-14)
    # Richardson: dS(eps) = a eps + b eps^2  =>  a = (4 dS(eps/2) - dS(eps)) / eps
    first_order = (4.0 * false_small - false_big) / eps_big
    _check("false path: dS/deps from the Euler-Lagrange residual", abs(first_order - slope) / abs(slope), 1e-7)


def test_explicit_time_term_with_two_coordinates():
    """L = v.v/2 - q.q/2 + t (v0 + 2 v1): d2L/dv_i dt = (1, 2), so a = (-q0 - 1, -q1 - 2).

    With one coordinate only, a transposed index in that term cannot be seen.
    """

    def lagrangian(q, v, t):
        return 0.5 * (v[0] ** 2 + v[1] ** 2) - 0.5 * (q[0] ** 2 + q[1] ** 2) + t * (v[0] + 2.0 * v[1])

    q, v = np.array([0.4, 0.2]), np.array([0.1, -0.3])
    for method in ("complex", "finite"):
        a = lg.euler_lagrange_accelerations(lagrangian, q, v, 0.7, method)
        _check(f"a = -q - (1, 2) ({method})", float(np.max(np.abs(a - (-q - np.array([1.0, 2.0]))))), 5e-9)
    residual = lg.euler_lagrange_residual(lagrangian, q, v, -q - np.array([1.0, 2.0]), 0.7)
    _check("residual on the solution", float(np.max(np.abs(residual))), 5e-9)


def test_action_needs_four_samples():
    with pytest.raises(ValueError):
        lg.action(oscillator, [0.0, 0.5, 1.0], np.zeros((3, 1)))


def _coupled_pendula(q, v, t):
    return 0.5 * (v[0] ** 2 + v[1] ** 2) + 0.3 * v[0] * v[1] * np.cos(q[0] - q[1]) + np.cos(q[0]) + np.cos(q[1])


def test_finite_method_at_unit_scale_agrees_with_the_complex_step_at_many_states():
    """Second derivatives by nested finite differences, at 40 random states of two coupled pendula.

    The function that the outer difference sees is itself a numerical derivative, with noise far above rounding.
    Before the noise floor of the step search, 13 of 120 elements of d2L/dv dv and d2L/dv dq were wrong at such
    states (the worst by 2.8, with an error estimate that made the acceleration be refused, or the plan of an
    integration useless). Measured now on 300 states: worst error 6.5e-10.
    """
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(40):
        q, v = rng.uniform(-1.0, 1.0, 2), rng.uniform(-1.0, 1.0, 2)
        exact = lg._terms(_coupled_pendula, q, v, 0.0, "complex", False)
        finite = lg._terms(_coupled_pendula, q, v, 0.0, "finite", False)
        for index in (1, 2):  # d2L/dv dv and d2L/dv dq
            worst = max(worst, float(np.max(np.abs(finite[index][0] - exact[index][0]))))
        a_exact = lg.euler_lagrange_accelerations(_coupled_pendula, q, v)
        a_finite = lg.euler_lagrange_accelerations(_coupled_pendula, q, v, 0.0, "finite")  # must not be refused
        worst = max(worst, float(np.max(np.abs(a_finite - a_exact))))
    _check("finite against complex, 40 states", worst, 2e-8)


def test_finite_method_integrates_at_the_default_tolerances():
    """The same system over one unit of time: the finite method follows the complex step and stays affordable.

    Measured: largest difference of the coordinates 1.7e-10 with 106 right-hand sides (the complex step needs
    94). With the steps that the search chose before the noise floor, 299 right-hand sides did not cover 0.02.
    """
    count = [0]

    def counted(q, v, t):
        count[0] += 1
        return _coupled_pendula(q, v, t)

    times = np.linspace(0.0, 1.0, 11)
    q_exact, v_exact = lg.integrate_lagrangian(_coupled_pendula, [0.3, 0.2], [0.1, -0.4], times)
    q_finite, v_finite = lg.integrate_lagrangian(counted, [0.3, 0.2], [0.1, -0.4], times, method="finite")
    _check("coordinates, finite against complex", float(np.max(np.abs(q_finite - q_exact))), 5e-9)
    _check("velocities, finite against complex", float(np.max(np.abs(v_finite - v_exact))), 5e-9)
    assert count[0] < 300_000  # measured 91546 evaluations of the Lagrangian


def test_action_uses_exactly_the_requested_number_of_gauss_legendre_nodes():
    """L = v^4 on a cubic spline is a polynomial of degree 8 on every interval.

    n nodes are exact up to degree 2 n - 1, so five nodes integrate it exactly and four do not (measured
    difference 1.4e-10 relative). A routine that used one node fewer than asked would fail the first assertion.
    """
    times = np.linspace(0.0, 2.0, 9)
    path = np.sin(1.3 * times)[:, None]

    def quartic(q, v, t):
        return v[0] ** 4

    exact = lg.action(quartic, times, path, nodes=8)
    assert lg.action(quartic, times, path, nodes=5) == pytest.approx(exact, rel=1e-14, abs=0)
    assert lg.action(quartic, times, path) == pytest.approx(exact, rel=1e-14, abs=0)  # the default is five
    assert abs(lg.action(quartic, times, path, nodes=4) / exact - 1.0) > 2e-11
    assert abs(lg.action(quartic, times, path, nodes=6) / exact - 1.0) < 1e-14
