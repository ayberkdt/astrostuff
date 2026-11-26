"""Guards of the mechanics and relativity foundations that a mutation run found untested.

372 hand-written mutants of core/_differentiation.py, mechanics/{lagrangian, hamiltonian, noether, energy,
integrators}.py and relativity/curvature.py were run against the tests of those modules; 112 survived. Most were
constants of heuristics or equivalent rewritings. The ones below changed what the library returns or refuses, and
no test saw it. Each test names the branch it pins (plan/AUDIT_2026-11-26.md, ninth round).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from kozmos.core import _differentiation as d
from kozmos.core.constants import GRAVITATIONAL_CONSTANT
from kozmos.mechanics import energy as en
from kozmos.mechanics import hamiltonian as hm
from kozmos.mechanics import integrators as it
from kozmos.mechanics import lagrangian as lg
from kozmos.mechanics import noether as nt
from kozmos.relativity import curvature
from kozmos.relativity.metrics import minkowski_metric

G = GRAVITATIONAL_CONSTANT.value


# --------------------------------------------------------------------------------------
# core/_differentiation.py
# --------------------------------------------------------------------------------------


def test_one_undetermined_element_of_a_vector_function_is_enough_to_refuse():
    """[x^2, |x|] at 0: the first element is fine, the second has a kink at the point.

    A mutant that refused only when every element was undetermined returned [[0], [0]].
    """
    with pytest.raises(ValueError, match="could not be determined"):
        d.jacobian(lambda x: np.array([x[0] ** 2, np.abs(x[0])]), [0.0], "finite")
    result = d.jacobian(lambda x: np.array([x[0] ** 2, x[0] ** 3 + 2.0 * x[0]]), [0.0], "finite")
    assert np.allclose(result[:, 0], [0.0, 2.0], rtol=1e-10, atol=1e-12)


def test_second_derivatives_need_a_scalar_function():
    with pytest.raises(ValueError, match="scalar function"):
        d.hessian_with_error(lambda x: x**2, [1.0, 2.0])
    with pytest.raises(ValueError, match="scalar function"):
        d.hessian(lambda x: x**2, [1.0, 2.0])
    with pytest.raises(ValueError, match="scalar function"):
        d.mixed_second(lambda x, y: x * y[0], [1.0, 2.0], [0.5])
    with pytest.raises(ValueError, match="scalar function"):
        d.gradient(lambda x: x**2, [1.0, 2.0])


def test_error_estimates_are_symmetric_for_a_hessian_and_at_rounding_level_for_the_complex_step():
    def f(x):
        return np.sin(x[0]) * np.exp(x[1]) + x[0] * x[1] ** 3

    for method in ("complex", "finite"):
        result, error = d.hessian_with_error(f, [0.7, 1.3], method)
        assert np.array_equal(error, error.T) and np.array_equal(result, result.T)
        assert np.all(error > 0.0)
    result, error = d.jacobian_with_error(lambda x: np.sin(x[0]) * x[1], [0.7, 1.3])
    # 4 eps |result|: measured 8.8e-16 and 5.7e-16
    assert np.allclose(error, 4.0 * np.finfo(float).eps * np.abs(result), rtol=1e-12, atol=0)


def test_a_reused_plan_carries_the_safety_factor_of_nested_differences():
    """The estimate of the plan-reuse path against that of a fresh search, for a function that differentiates.

    Both must be above the true error. A mutant that dropped the factor of nested differences on the reuse path
    left its estimate below the true error.
    """

    def inner(y):
        return d.jacobian_with_error(lambda x: np.sin(x[0]) * np.exp(0.3 * y[0]) + x[0] ** 3 * y[1], [0.4], "finite")[0]

    def exact(y):  # d/dy of [cos(0.4) exp(0.3 y0) + 3 * 0.16 * y1]
        return np.array([[0.3 * np.cos(0.4) * np.exp(0.3 * y[0]), 3.0 * 0.4**2]])

    plan: dict = {}
    first = np.array([0.7, 1.1])
    fresh, fresh_error = d._finite_jacobian(inner, first, plan)
    second = np.array([0.71, 1.12])
    reused, reused_error = d._finite_jacobian(inner, second, plan)
    again, again_error = d._finite_jacobian(inner, second)
    assert np.allclose(reused, exact(second), rtol=1e-7, atol=0)
    assert np.all(np.abs(reused - exact(second)) <= reused_error)
    assert np.all(np.abs(again - exact(second)) <= again_error)


# --------------------------------------------------------------------------------------
# mechanics/lagrangian.py
# --------------------------------------------------------------------------------------


def test_public_lagrangian_functions_refuse_a_non_analytic_lagrangian_with_the_complex_step():
    """L = v^2/2 - |q - 0.3|: the force is -1 at q = 0.7 and the complex step would give 0."""

    def kinked(q, v, t):
        return 0.5 * v[0] ** 2 - np.abs(q[0] - 0.3)

    for function in (lg.euler_lagrange_accelerations, lg.generalized_forces):
        with pytest.raises(ValueError, match="not analytic"):
            function(kinked, [0.7], [0.1])
    with pytest.raises(ValueError, match="not analytic"):
        lg.integrate_lagrangian(kinked, [0.7], [0.1], [0.0, 0.1])
    assert lg.euler_lagrange_accelerations(kinked, [0.7], [0.1], 0.0, "finite")[0] == pytest.approx(
        -1.0, rel=1e-8, abs=0
    )


def test_state_and_time_arguments_are_validated():
    def free(q, v, t):
        return 0.5 * np.sum(v * v)

    for q, v in (([[1.0, 2.0]], [[0.1, 0.2]]), ([], []), ([1.0, 2.0], [0.1])):
        with pytest.raises(ValueError):
            lg.generalized_momenta(free, q, v)
    with pytest.raises(ValueError, match="at least two values"):
        lg.integrate_lagrangian(free, [0.0], [1.0], [0.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        lg.integrate_lagrangian(free, [0.0], [1.0], [0.0, 1.0, 1.0])


def test_an_integration_that_fails_is_reported():
    """q'' = q^2 from q = v = 1 reaches infinity before t = 5: the integrator gives up, and that is an error."""
    with pytest.raises(RuntimeError, match="integration failed"):
        lg.integrate_lagrangian(lambda q, v, t: 0.5 * v[0] ** 2 + q[0] ** 3 / 3.0, [1.0], [1.0], [0.0, 5.0])


# --------------------------------------------------------------------------------------
# mechanics/hamiltonian.py, noether.py, energy.py, integrators.py
# --------------------------------------------------------------------------------------


def test_legendre_velocity_needs_coordinates_and_momenta_of_equal_length():
    h = hm.hamiltonian_from_lagrangian(lambda q, v, t: 0.5 * np.sum(v * v))
    with pytest.raises(ValueError, match="same length"):
        h.velocity([1.0, 2.0], [0.5])
    assert np.allclose(h.velocity([1.0, 2.0], [0.5, -0.25]), [0.5, -0.25], rtol=1e-14, atol=0)


def test_generator_nonlinear_in_the_velocity_is_differentiated_at_the_velocity():
    """L = v^2/2, delta q = v^2, a = 0.5 at v = 2: delta L = (dL/dv) d(delta q)/dt = v * 2 v a = 4.

    A mutant that took d(delta q)/dv at q instead of v returned 2.
    """
    value = nt.symmetry_variation(
        lambda q, v, t: 0.5 * v[0] ** 2, [0.3], [2.0], 0.0, lambda q, v, t: np.array([v[0] ** 2]), a=[0.5]
    )
    assert value == pytest.approx(4.0, rel=1e-12, abs=0)


def test_boost_and_translation_generators_need_whole_particles():
    for factory in (nt.boost_generator, nt.translation_generator):
        generator = factory([1.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            generator(np.zeros(4), np.zeros(4), 0.0)
        assert np.shape(generator(np.zeros(6), np.zeros(6), 0.0)) == (6,)


def test_potential_energy_of_laboratory_masses_and_shape_mismatches():
    """Energies of 1e-10 J: a spurious self term of -G m_i m_i / 2 per particle would show (it is invisible beside
    astronomical energies, which is all the other tests use)."""
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    exact = -G * (1.0 * 2.0 / 1.0 + 1.0 * 3.0 / 2.0 + 2.0 * 3.0 / np.sqrt(5.0))
    assert en.potential_energy_j([1.0, 2.0, 3.0], x) == pytest.approx(exact, rel=1e-14, abs=0)
    with pytest.raises(ValueError, match="shape"):
        en.potential_energy_j([1.0], x[:2])
    with pytest.raises(ValueError, match="same shape"):
        en.total_angular_momentum([1.0, 2.0], x[:2], [[0.0, 1.0, 0.0]])


def test_relative_drift_needs_at_least_one_value():
    for values in (np.array([]), np.float64(1.0)):
        with pytest.raises(ValueError, match="at least one entry"):
            it.relative_drift(values)


# --------------------------------------------------------------------------------------
# relativity/curvature.py
# --------------------------------------------------------------------------------------


def test_riemann_tensor_validates_the_point_and_refuses_a_non_analytic_metric():
    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        curvature.riemann_tensor(minkowski_metric, [0.0, 1.0, 2.0])

    def with_abs(x):
        return np.diag([-1.0, 1.0 + np.abs(x[1]), 1.0, 1.0])

    with pytest.raises(ValueError, match="not analytic"):
        curvature.riemann_tensor(with_abs, [0.0, 0.5, 0.3, 0.2])


def test_kretschmann_scalar_does_not_depend_on_the_unit_of_a_coordinate():
    """Schwarzschild written with a rescaled azimuth, phi = 1e9 phi': the same geometry, with g_phi'phi' 1e18 times
    larger and curvature components in the coordinate basis that differ by as much. The scalar must be the same,
    and neither call may warn or refuse.

    This does not pin the orthonormal-frame weighting of the precision decision: a mutant that weights with |g|
    instead of sqrt|g| passes this test too (the decision is the same here). No case was found where it differs.
    """
    mass, stretch = 1.0, 1e9

    def schwarzschild(x, scale):
        r, theta = x[1], x[2]
        f = 1.0 - 2.0 * mass / r
        return np.diag([-f, 1.0 / f, r**2, (scale * r * np.sin(theta)) ** 2])

    point = [0.0, 6.0, 1.1, 0.3]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plain = curvature.kretschmann_scalar(lambda x: schwarzschild(x, 1.0), point)
        stretched = curvature.kretschmann_scalar(lambda x: schwarzschild(x, stretch), point)
    exact = 48.0 * mass**2 / 6.0**6
    assert plain == pytest.approx(exact, rel=1e-9, abs=0)
    assert stretched == pytest.approx(exact, rel=1e-9, abs=0)


# --------------------------------------------------------------------------------------
# found by the second adversarial probe (plan/AUDIT_2026-11-26.md, tenth round)
# --------------------------------------------------------------------------------------


def test_a_dependence_on_a_galactic_scale_is_not_taken_for_independence():
    """G(t) = 40 + 0.3 t / T with T = 7e15 s: dG/dt = 4.3e-17, and G does not change within a rounding error over
    a thousand seconds.

    The finite method returned exactly 0 with a zero estimate (a function "independent of the coordinate"). The
    independence is now looked for 3e6 max(1, |x|) away as well, and where it is found the estimate is one
    rounding error of f over that distance, not zero.
    """
    period = 7e15

    def slow(t):
        return np.array([40.0 + 0.3 * t[0] / period])

    exact = 0.3 / period
    assert d.jacobian(slow, [0.0])[0, 0] == pytest.approx(exact, rel=1e-12, abs=0)
    value, error = d.jacobian_with_error(slow, [0.0], "finite")
    assert abs(value[0, 0] - exact) <= error[0, 0] and error[0, 0] < 1e-2 * exact  # measured 4.3e-20
    with pytest.raises(ValueError, match="could not be determined"):
        d.jacobian(slow, [0.0], "finite")
    # a function that really does not depend on the coordinate: zero, with an estimate at the rounding level
    value, error = d.jacobian_with_error(lambda x: np.array([40.0 + x[1]]), [0.0, 0.3], "finite")
    assert value[0, 0] == 0.0 and 0.0 < error[0, 0] < 1e-20
    assert d.jacobian(lambda x: np.array([40.0 + x[1]]), [0.0, 0.3], "finite")[0, 0] == 0.0


def test_forces_at_a_coordinate_that_is_zero_in_a_problem_of_galactic_size():
    """Kepler potential centred 3e20 m away, particle at x = 0: dL/dx = 1.27e-10. The finite method returned 0."""
    gm, centre = 6.6743e-11 * 2e41, 3e20

    def lagrangian(q, v, t):
        return 0.5 * (v[0] ** 2 + v[1] ** 2) + gm / np.sqrt((q[0] - centre) ** 2 + q[1] ** 2)

    for x in (0.0, 0.5, 1e9):
        q = [x, 1e20]
        radius = np.hypot(centre - x, 1e20)
        exact = gm * (centre - x) / radius**3
        assert lg.generalized_forces(lagrangian, q, [0.0, 0.0])[0] == pytest.approx(exact, rel=1e-12, abs=0)
        try:
            force = lg.generalized_forces(lagrangian, q, [0.0, 0.0], 0.0, "finite")[0]
        except ValueError:
            continue  # refused: the scale of the problem cannot be seen from x = 0
        assert force == pytest.approx(exact, rel=1e-6, abs=0)


def test_coriolis_term_at_the_origin_of_a_slowly_rotating_frame():
    """m = 1e30 kg, Omega = 1e-15 1/s, v = (1e5, 0) m/s at the origin: a = (0, -2 Omega v_x) = (0, -2e-10).

    The finite method returned (0, 0): the only acceleration there is was lost without an error. It is refused
    now (the term changes L by less than its rounding over any distance the search reaches).
    """
    mass, omega = 1e30, 1e-15

    def rotating(q, v, t):
        kinetic = 0.5 * mass * (v[0] ** 2 + v[1] ** 2)
        return kinetic + mass * omega * (q[0] * v[1] - q[1] * v[0]) + 0.5 * mass * omega**2 * (q[0] ** 2 + q[1] ** 2)

    for q in ([0.0, 0.0], [0.0, 0.5], [3000.0, 0.0]):
        a = lg.euler_lagrange_accelerations(rotating, q, [1e5, 0.0])
        exact = np.array([omega**2 * q[0], omega**2 * q[1] - 2.0 * omega * 1e5])
        assert a[1] == pytest.approx(exact[1], rel=1e-10, abs=0)
        assert a[0] == pytest.approx(exact[0], rel=1e-6, abs=1e-25)
        with pytest.raises((lg.DerivativePrecisionError, ValueError)):
            lg.euler_lagrange_accelerations(rotating, q, [1e5, 0.0], 0.0, "finite")
        with pytest.raises(ValueError):
            lg.generalized_forces(rotating, q, [1e5, 0.0], 0.0, "finite")


def test_residual_is_refused_where_the_accelerations_are():
    """euler_lagrange_residual combined the derivatives without looking at their estimates and returned a residual
    of 19 % of the force at a state (a double pendulum in galactic units) where the accelerations were refused."""
    mass, omega = 1e30, 1e-15

    def rotating(q, v, t):
        kinetic = 0.5 * mass * (v[0] ** 2 + v[1] ** 2)
        return kinetic + mass * omega * (q[0] * v[1] - q[1] * v[0]) + 0.5 * mass * omega**2 * (q[0] ** 2 + q[1] ** 2)

    q, v = [0.0, 0.0], [1e5, 0.0]
    a = lg.euler_lagrange_accelerations(rotating, q, v)
    residual = lg.euler_lagrange_residual(rotating, q, v, a)
    assert np.max(np.abs(residual)) <= 1e-10 * mass * 2.0 * omega * 1e5
    with pytest.raises((lg.DerivativePrecisionError, ValueError)):
        lg.euler_lagrange_residual(rotating, q, v, a, 0.0, "finite")

    def oscillator(q, v, t):
        return 0.5 * v[0] ** 2 - 0.5 * q[0] ** 2

    for method in ("complex", "finite"):
        assert lg.euler_lagrange_residual(oscillator, [0.3], [0.1], [-0.3], 0.0, method)[0] == pytest.approx(
            0.0, abs=1e-8
        )
        assert lg.euler_lagrange_residual(oscillator, [0.3], [0.1], [0.2], 0.0, method)[0] == pytest.approx(
            0.5, rel=1e-8, abs=0
        )


def test_a_kink_inside_or_at_the_stencil_is_found_by_the_test_of_analyticity():
    """L = v^2/2 - eps |q - q0|: the complex step of abs gives no force, the true acceleration is -eps sign(q - q0).

    With the kink within the first step of the reference (1e-3) the test passed for eps up to 1e-6 (and up to 1e-2
    beside a kinetic energy of 5e5): the roughness that the kink puts into the estimate of the reference was part
    of the tolerance. The comparison is made within the rounding of the reference now, and a second-difference
    probe finds a kink that no step separates from the point.
    """
    q0 = 0.4

    def kinked(eps):
        return lambda q, v, t: 0.5 * v[0] ** 2 - eps * np.abs(q[0] - q0)

    for eps in (1e-6, 1e-8, 1e-10):
        for distance in (1e-3, 1e-6, 1e-10, 0.0):
            with pytest.raises(ValueError, match="not analytic|could not be checked"):
                lg.euler_lagrange_accelerations(kinked(eps), [q0 + distance], [1.0])
    # away from the kink the finite method gives the force of the side the state is on
    a = lg.euler_lagrange_accelerations(kinked(1e-6), [q0 + 0.3], [1.0], 0.0, "finite")
    assert a[0] == pytest.approx(-1e-6, rel=1e-6, abs=0)
    # an analytic Lagrangian with the same numbers is accepted
    smooth = lg.euler_lagrange_accelerations(
        lambda q, v, t: 0.5 * v[0] ** 2 - 1e-8 * (q[0] - q0) ** 2, [q0 + 1e-3], [1.0]
    )
    assert smooth[0] == pytest.approx(-2e-11, rel=1e-9, abs=0)


def test_inertia_matrix_with_a_zero_on_its_diagonal_is_not_singular():
    """L = v0 v1 - V(q): W = [[0, 1], [1, 0]], determinant -1. The complex step refused it as singular."""

    def lagrangian(q, v, t):
        return v[0] * v[1] - 0.3 * q[0] ** 2 - 0.4 * q[1] ** 2

    for method in ("complex", "finite"):
        a = lg.euler_lagrange_accelerations(lagrangian, [0.5, 1.0], [0.2, -0.1], 0.0, method)
        assert np.allclose(a, [-0.8, -0.3], rtol=1e-8, atol=0)  # a0 = dL/dq1, a1 = dL/dq0
    with pytest.raises(lg.SingularInertiaError, match="row of the matrix"):
        lg.euler_lagrange_accelerations(lambda q, v, t: 0.5 * v[0] ** 2 + q[1] * v[1], [0.5, 1.0], [0.2, -0.1])


def test_legendre_inversion_with_the_finite_method_converges_at_the_noise_of_the_differences():
    """With the default tolerance of 1e-13 the finite method refused 37 of 60 states of a 3-D oscillator ("no
    descent step"): dL/dv by finite differences is known to about 1e-10. Its tolerance is not taken below 1e-8."""

    def coupled(q, v, t):
        return 0.5 * (2.0 + np.cos(q[0])) * v[0] ** 2 + 0.3 * v[0] * v[1] + 0.5 * v[1] ** 2 + np.cos(q[1])

    finite, exact = hm.hamiltonian_from_lagrangian(coupled, method="finite"), hm.hamiltonian_from_lagrangian(coupled)
    rng = np.random.default_rng(0)
    for _ in range(12):
        q, p = rng.normal(size=2), rng.normal(size=2)
        assert np.allclose(finite.velocity(q, p), exact.velocity(q, p), rtol=0, atol=1e-6)  # measured 1e-8
        assert finite(q, p) == pytest.approx(exact(q, p), rel=1e-10, abs=1e-12)


def test_two_bodies_in_si_units_are_integrated():
    """Two bodies of 2e30 kg, 1.5e11 m apart on a circular orbit: integrate_lagrangian refused them at once, while
    the accelerations at the first state were accepted. On the path where the steps of an earlier state are
    reused, an element that does not depend on a coordinate (dp/dt) got the rounding estimate of a difference at
    a step of 1e-3 s instead of the estimate of its independence."""
    mass, separation = 2e30, 1.5e11
    speed = np.sqrt(G * mass / (2.0 * separation))

    def binary(q, v, t):
        distance = np.sqrt((q[0] - q[2]) ** 2 + (q[1] - q[3]) ** 2)
        return 0.5 * mass * np.sum(v * v) + G * mass * mass / distance

    q0 = [-0.5 * separation, 0.0, 0.5 * separation, 0.0]
    v0 = [0.0, -speed, 0.0, speed]
    period = 2.0 * np.pi * 0.5 * separation / speed
    q, v = lg.integrate_lagrangian(binary, q0, v0, [0.0, 0.25 * period])
    expected = [0.0, -0.5 * separation, 0.0, 0.5 * separation]  # a quarter of a turn
    assert np.allclose(q[-1], expected, rtol=0, atol=1e-7 * separation)


# --------------------------------------------------------------------------------------
# untested in the second mutation run (plan/AUDIT_2026-11-26.md, eleventh round)
# --------------------------------------------------------------------------------------


def test_complex_step_takes_its_scale_from_the_coordinates():
    """The step is 1e-30 of the coordinate; a zero coordinate takes the smallest non-zero one below 1, or 1; the
    step is not below 1e-280. Each line fails for one mutant of those rules (a zero where 1e-100 is right, sinh 1
    where 1 is right, an overflow, a false refusal)."""
    assert np.array_equal(d.jacobian(lambda x: 1e-100 * x[0] + x[1], [0.0, 1.0]), [1e-100, 1.0])
    assert np.array_equal(d.jacobian(lambda x: 1e-100 * (x[0] + x[1]), [0.0, 0.0]), [1e-100, 1e-100])
    fine = d.jacobian(lambda x: 1e-30 * np.sin(x[0] / 1e-30) + x[1], [0.0, 1e-30], check=False)
    assert np.allclose(fine, [1.0, 1.0], rtol=1e-12, atol=0)
    finer = d.jacobian(lambda x: 1e-45 * np.sin(x[0] / 1e-45) + x[1] + x[2], [0.0, 1e-30, 1e-10], check=False)
    assert np.allclose(finer, [1.0, 1.0, 1.0], rtol=1e-12, atol=0)
    assert d.jacobian(lambda x: 1e-30 * x[0], [1e-270])[0] == pytest.approx(1e-30, rel=1e-12, abs=0)


def test_finite_jacobian_of_an_offset_function_at_large_arguments():
    """f = C + sin(x0) cos(x1) at (1e6, 2e6), where the first step (1e3) spans 160 periods.

    With C = 1e10 the rounding of f allows 1e-4 at best: the estimate says so and the call that returns no estimate
    refuses. Mutants of the noise floor and of the fill of undetermined elements returned -0.0038 and 1.5e-13 for
    the true 0.707 and -0.229.
    """
    x = np.array([1e6, 2e6])
    exact = np.array([np.cos(1e6) * np.cos(2e6), -np.sin(1e6) * np.sin(2e6)])
    value, error = d.jacobian_with_error(lambda y: np.array([1e10 + np.sin(y[0]) * np.cos(y[1])]), x, "finite")
    assert np.all(np.abs(value[0] - exact) <= error[0]) and np.all(error[0] < 1e-3)  # measured 6e-6 and 1e-4
    with pytest.raises(ValueError, match="could not be determined"):
        d.jacobian(lambda y: np.array([1e10 + np.sin(y[0]) * np.cos(y[1])]), x, "finite")
    for offset, tolerance in ((0.0, 1e-8), (1e6, 1e-7)):  # measured 9e-10 and 9e-9
        value = d.jacobian(lambda y, c=offset: np.array([c + np.sin(y[0]) * np.cos(y[1])]), x, "finite")[0]
        assert np.allclose(value, exact, rtol=tolerance, atol=0)


def test_strict_rule_of_the_functions_that_return_no_estimate():
    """Undetermined elements are accepted only below 1e-7 of the largest determined one; determined ones always."""

    def offset(constant):
        return lambda y: np.array([constant + np.sin(y[0]) * np.cos(y[1])])

    with pytest.raises(ValueError, match="could not be determined"):
        d.jacobian(offset(1e14), [0.3, 1.1], "finite")  # true derivatives 0.43 and -0.26; a mutant returned -1.6e-7, 0
    point = np.array([1e-3, 5.0])
    exact = np.array([np.cos(1e-3) * np.cos(5.0), -np.sin(1e-3) * np.sin(5.0)])
    assert np.allclose(d.jacobian(offset(1e6), point, "finite")[0], exact, rtol=1e-6, atol=1e-9)
    # a vector function with one element whose dependence is below the spacing of its values (1e16 + x) and one that
    # varies: the far confirmation must be made for the first although the second does not look constant
    value, error = d.jacobian_with_error(lambda y: np.array([1e16 + y[0], y[0]]), [0.3], "finite")
    assert abs(value[0, 0] - 1.0) <= error[0, 0] and value[1, 0] == pytest.approx(1.0, rel=1e-12, abs=0)
    # an element that is identically zero only on a part of the steps is not anchored at zero
    value, _ = d.hessian_with_error(lambda y: np.sin(y[0]) * y[1] ** 2 + 1e6, [0.0, 2.0], "finite")
    assert value[0, 1] == pytest.approx(4.0, rel=1e-6, abs=0)  # a mutant returned 2.0


def test_second_derivatives_without_an_estimate_are_strict():
    def kinked(y):
        return y[0] * np.abs(y[0]) + y[1]

    def mixed(a, b):
        return a[0] * np.abs(b[0]) + 1e12

    with pytest.raises(ValueError):
        d.hessian(kinked, [0.0, 1.0], "finite")
    with pytest.raises(ValueError):
        d.hessian_with_error(kinked, [0.0, 1.0], "finite", strict=True)
    with pytest.raises(ValueError):
        d.mixed_second(mixed, [1.0], [0.0], "finite")
    with pytest.raises(ValueError):
        d.mixed_second_with_error(mixed, [1.0], [0.0], "finite", strict=True)
    value, error = d.hessian_with_error(kinked, [0.0, 1.0], "finite")  # not strict: the estimate tells
    assert error[0, 0] > 0.1


def test_a_test_that_cannot_be_made_says_so():
    with pytest.raises(ValueError, match="could not be checked against finite differences"):
        d.gradient(lambda x: np.abs(x[0]), [0.0])


@pytest.mark.parametrize("where", ["first state", "middle time", "last time", "velocity window", "time window"])
def test_analyticity_is_tested_at_the_first_state_and_at_every_output_time(where):
    """A free particle from q = 0 with v = 1 and a term 0.3 |q - c| that is switched on in a window which the path
    crosses at one output time only (or in a window of the velocity, or of the time). Each case fails for a mutant
    that skips, or reads wrongly, one of the tests."""
    times = np.array([0.0, 1.0, 2.0, 3.0])
    start_q, start_v = 0.0, 1.0

    def kink(q):
        return 0.3 * np.abs(q[0] - 10.0)

    def lagrangian(q, v, t):
        position, velocity = np.real(q[0]), np.real(v[0])
        if where == "first state":
            active = np.abs(position) < 0.3
        elif where == "middle time":
            active = np.abs(position - 2.0) < 0.3
        elif where == "last time":
            active = np.abs(position - 3.0) < 0.3
        elif where == "velocity window":
            active = (np.abs(velocity - 1.0) < 0.3) & (np.abs(position - 1.0) < 0.3)
        else:
            active = np.abs(np.real(t) - 2.0) < 0.3
        return 0.5 * v[0] ** 2 + np.where(active, kink(q), 0.0)

    with pytest.raises(ValueError, match="not analytic|could not be checked"):
        lg.integrate_lagrangian(lagrangian, [start_q], [start_v], times, rtol=1e-6, atol=1e-8)


def test_generator_with_a_negative_velocity_coefficient_needs_the_acceleration():
    def free(q, v, t):
        return 0.5 * v[0] ** 2

    for sign in (1.0, -1.0):
        with pytest.raises(ValueError, match="depends on the velocity"):
            nt.symmetry_variation(free, [0.3], [2.0], 0.0, lambda q, v, t, s=sign: np.array([s * 0.5 * v[0]]))


def test_virial_ratio_is_formed_about_the_barycentre():
    """Circular binary displaced by 1e22 m: 6.0e-6 from 1 with positions relative to the barycentre, 1.04e-5 with the
    plain positions, 3.6e-5 with a wrong sign. Also: an infinite G M has an infinite escape speed, not an error."""
    mass, separation, offset = 2e30, 1.5e11, 1e22
    speed = np.sqrt(G * mass / (2.0 * separation))
    x = np.array([[offset - 0.5 * separation, 0.0, 0.0], [offset + 0.5 * separation, 0.0, 0.0]])
    v = np.array([[0.0, -speed, 0.0], [0.0, speed, 0.0]])
    assert abs(en.virial_ratio([mass, mass], x, v) - 1.0) < 8e-6
    assert en.escape_speed_m_s(np.inf, 1.0) == np.inf


def test_precision_guard_of_the_accelerations_at_its_thresholds(monkeypatch):
    """The guard, with derivative terms and estimates given by hand: a = W^-1 (F - C v - D), one coordinate with
    W = 1 and F = 1 (so |a| + size = 2 and the allowed error is 1e-6 * (2 + 1e-3 * 2) = 2.002e-6), then two
    coordinates of which one vanishes (allowed there: 1e-3 of that, 2e-9)."""

    def terms(force, force_error, w, w_error, n):
        zero = np.zeros((n, n))
        return lambda *args, **kwargs: (
            (np.array(force), np.array(force_error)),
            (np.array(w), np.array(w_error)),
            (zero, zero),
            (np.zeros(n), np.zeros(n)),
        )

    def accelerations(n=1):
        return lg._accelerations(lambda q, v, t: 0.0, np.zeros(n), np.zeros(n), 0.0, "finite", 1e12, False)

    monkeypatch.setattr(lg, "_terms", terms([1.0], [1.9e-6], [[1.0]], [[0.0]], 1))
    assert accelerations()[0] == 1.0
    monkeypatch.setattr(lg, "_terms", terms([1.0], [2.1e-6], [[1.0]], [[0.0]], 1))
    with pytest.raises(lg.DerivativePrecisionError):
        accelerations()
    # the estimate of W counts, multiplied by |a|
    monkeypatch.setattr(lg, "_terms", terms([1.0], [0.0], [[1.0]], [[2.1e-6]], 1))
    with pytest.raises(lg.DerivativePrecisionError):
        accelerations()
    monkeypatch.setattr(lg, "_terms", terms([1.0], [0.0], [[1.0]], [[1.9e-6]], 1))
    assert accelerations()[0] == 1.0
    # a coordinate without any term is allowed a thousandth of the tolerance on the common scale
    identity = [[1.0, 0.0], [0.0, 1.0]]
    nothing = [[0.0, 0.0], [0.0, 0.0]]
    monkeypatch.setattr(lg, "_terms", terms([1.0, 0.0], [0.0, 1.9e-9], identity, nothing, 2))
    assert np.array_equal(accelerations(2), [1.0, 0.0])
    monkeypatch.setattr(lg, "_terms", terms([1.0, 0.0], [0.0, 2.1e-9], identity, nothing, 2))
    with pytest.raises(lg.DerivativePrecisionError):
        accelerations(2)


def test_argument_checks_name_what_is_wrong():
    def free(q, v, t):
        return 0.5 * np.sum(v * v)

    with pytest.raises(ValueError, match="non-empty 1-D"):
        lg.generalized_momenta(free, [[1.0, 2.0]], [[0.1, 0.2]])
    with pytest.raises(ValueError, match="non-empty 1-D"):
        lg.generalized_momenta(free, [], [])
    with pytest.raises(ValueError, match="same length as q"):
        lg.euler_lagrange_residual(free, [0.1, 0.2], [0.3, 0.4], [0.0])
    with pytest.raises(ValueError, match=r"shape \(len\(times\), n\)"):
        lg.action(free, [0.0, 1.0, 2.0, 3.0], np.zeros((3, 1)))
    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        curvature.riemann_tensor(lambda x: np.diag([-1.0, 1.0, 1.0, 1.0]), [0.0, 1.0, 2.0])


def test_an_undetermined_element_needs_1e_minus_7_of_the_largest_determined_one():
    """[x, C + sin x] at 0.3: the first element is determined (derivative 1); the second is known to 5.6e-6 for
    C = 1e8 and to 1.4e-5 for C = 1e9, which is not 1e-7 of 1, and the call without an estimate refuses. (For
    C = 1e7 it is known to 1e-6 of its own value and is returned.)"""
    for constant in (1e8, 1e9):
        with pytest.raises(ValueError, match="could not be determined"):
            d.jacobian(lambda x, c=constant: np.array([x[0], c + np.sin(x[0])]), [0.3], "finite")
        value, error = d.jacobian_with_error(lambda x, c=constant: np.array([x[0], c + np.sin(x[0])]), [0.3], "finite")
        assert abs(value[1, 0] - np.cos(0.3)) <= error[1, 0] < 1e-4
    value = d.jacobian(lambda x: np.array([x[0], 1e7 + np.sin(x[0])]), [0.3], "finite")
    assert value[1, 0] == pytest.approx(np.cos(0.3), rel=1e-6, abs=0)


def test_precision_guard_of_the_residual_at_its_threshold(monkeypatch):
    """W = 1, F = 1, a = 1: the terms have size 2 and the residual may be uncertain by 1e-6 * (2 + 1e-3 * 2)."""

    def terms(force_error):
        zero = np.zeros((1, 1))
        return lambda *args, **kwargs: (
            (np.array([1.0]), np.array([force_error])),
            (np.array([[1.0]]), zero),
            (zero, zero),
            (np.zeros(1), np.zeros(1)),
        )

    def residual():
        return lg.euler_lagrange_residual(lambda q, v, t: 0.0, [0.0], [0.0], [1.0], 0.0, "finite")

    monkeypatch.setattr(lg, "_terms", terms(1.9e-6))
    assert residual()[0] == 0.0
    monkeypatch.setattr(lg, "_terms", terms(2.1e-6))
    with pytest.raises(lg.DerivativePrecisionError, match="residual"):
        residual()
