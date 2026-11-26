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
