"""Numerical derivatives against closed forms, over wide ranges of size and scale.

The helper was rewritten twice after independent audits: the first version used fixed steps and an
absolute tolerance, the second adapted the step but still accepted wrong numbers in several regimes.
Every test of the section "found by the audits" reproduces one of the reported failures; the measured
errors quoted in the comments are those of the present version.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from kozmos.core import _differentiation as d


def _f(x):
    return np.sin(x[0]) * np.exp(x[1]) + x[0] ** 3 / x[1]


def _gradient(x):
    return np.array(
        [np.cos(x[0]) * np.exp(x[1]) + 3 * x[0] ** 2 / x[1], np.sin(x[0]) * np.exp(x[1]) - x[0] ** 3 / x[1] ** 2]
    )


def _hessian(x):
    mixed = np.cos(x[0]) * np.exp(x[1]) - 3 * x[0] ** 2 / x[1] ** 2
    return np.array(
        [
            [-np.sin(x[0]) * np.exp(x[1]) + 6 * x[0] / x[1], mixed],
            [mixed, np.sin(x[0]) * np.exp(x[1]) + 2 * x[0] ** 3 / x[1] ** 3],
        ]
    )


X = np.array([0.7, 1.3])


# --------------------------------------------------------------------------------------
# Unit-scale functions
# --------------------------------------------------------------------------------------


def test_complex_step_gradient_is_exact_to_rounding():
    assert np.allclose(d.gradient(_f, X), _gradient(X), rtol=1e-15, atol=0)


def test_finite_gradient_and_its_error_estimate():
    result, error = d.jacobian_with_error(_f, X, "finite")
    assert np.allclose(result, _gradient(X), rtol=1e-12, atol=0)
    assert np.all(np.abs(result - _gradient(X)) <= 5.0 * error)  # an estimate, not a bound
    assert np.all(error <= 1e-9 * np.abs(result))


@pytest.mark.parametrize(("method", "rtol"), [("complex", 1e-11), ("finite", 1e-9)])
def test_hessian(method, rtol):
    result, error = d.hessian_with_error(_f, X, method)
    assert np.allclose(result, _hessian(X), rtol=rtol, atol=0)
    assert np.array_equal(result, result.T)
    # The estimate of a nested finite difference carries a safety factor of 30, because the true error was measured
    # at up to 13.4 raw estimates (2400 elements); with it the true error stayed below 0.33 estimates (960 elements).
    assert np.all(np.abs(result - _hessian(X)) <= error)


def test_jacobian_of_a_tensor_valued_function():
    def metric(x):
        return np.array([[x[0] ** 2, x[0] * x[1]], [x[0] * x[1], np.cos(x[1])]])

    result = d.jacobian(metric, X)
    assert result.shape == (2, 2, 2)
    assert np.allclose(result[..., 0], [[2 * X[0], X[1]], [X[1], 0.0]], rtol=1e-15)
    assert np.allclose(result[..., 1], [[0.0, X[0]], [X[0], -np.sin(X[1])]], rtol=1e-15)


def test_mixed_second():
    result = d.mixed_second(lambda x, y: x[0] ** 2 * np.sin(y[0]) + x[1] * y[1] ** 2, [0.5, 2.0], [0.3, 1.5])
    assert np.allclose(result, [[2 * 0.5 * np.cos(0.3), 0.0], [0.0, 2 * 1.5]], rtol=1e-11, atol=1e-11)


def test_argument_checks():
    with pytest.raises(ValueError):
        d.jacobian(_f, [[1.0, 2.0]])
    with pytest.raises(ValueError):
        d.jacobian(_f, X, "other")
    with pytest.raises(ValueError):
        d.gradient(lambda x: x, X)


def test_error_orders_of_the_formulas():
    # the six-point formula is exact for polynomials up to degree six; on x^7 at 0 its error is 36 h^6
    # (h^6 f^(7) / 140 with f^(7) = 5040)
    h = 1e-2
    points = [np.asarray((k * h) ** 7) for k in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)]
    assert float(d._six(points, h)) == pytest.approx(36.0 * h**6, rel=1e-10, abs=0)
    points = [np.asarray((k * h) ** 6 + (k * h) ** 3) for k in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)]
    assert float(d._six(points, h)) == pytest.approx(0.0, abs=1e-18)
    # complex step: no truncation error at any size of the argument
    for x in (1e-8, 1.0, 1e8):
        assert d.gradient(lambda y: y[0] ** 3, [x])[0] == pytest.approx(3.0 * x**2, rel=1e-15, abs=0)


def test_function_independent_of_a_coordinate_gives_exactly_zero():
    result, error = d.jacobian_with_error(lambda x: np.sin(x[0]) + 7.0, [0.3, 5.0], "finite")
    # the derivative is zero; the estimate is what the confirmation leaves open: one rounding error of f over
    # the distance at which the independence was confirmed (3e6 max(1, |x_i|)); measured 1.1e-22
    assert result[1] == 0.0 and 0.0 < error[1] < 1e-20
    assert result[0] == pytest.approx(np.cos(0.3), rel=1e-12, abs=0)


def test_unchecked_complex_step_costs_one_evaluation_per_coordinate():
    calls = []

    def counted(x):
        calls.append(1)
        return np.sin(x[0]) * np.exp(x[1])

    d.jacobian(counted, X, check=False)
    assert len(calls) == 2
    calls.clear()
    d.jacobian(counted, X)
    # the check: seven real values per coordinate and six more for the confirming step, f(x), and the probe for a
    # kink (two values per decade of the step until two decades stand above the rounding); measured 74 in all
    assert 2 + 2 * (7 + 6) + 1 < len(calls) <= 150


# --------------------------------------------------------------------------------------
# Non-analytic functions
# --------------------------------------------------------------------------------------


def test_non_analytic_function_is_detected():
    # abs() of a complex number is its modulus: the imaginary part is lost and the derivative would be 0
    with pytest.raises(ValueError, match="not analytic"):
        d.gradient(lambda x: np.abs(x[0]) * x[1], X)
    assert np.allclose(d.gradient(lambda x: np.abs(x[0]) * x[1], X, "finite"), [X[1], X[0]], rtol=1e-11)


@pytest.mark.parametrize("factor", [1.0, 1e-3, 1e-5, 1e-8, 1e-20, 1e12])
def test_non_analytic_function_is_detected_whatever_its_size(factor):
    with pytest.raises(ValueError, match="not analytic"):
        d.gradient(lambda x: factor * np.abs(x[0]), [0.7])


def test_non_analytic_lagrangian_in_si_units_is_detected():
    # kinetic energy of an electron written with norm(): the complex-step momentum would be 0 instead of 9.1e-31 v
    def kinetic(v):
        return 0.5 * 9.1e-31 * np.linalg.norm(v) ** 2

    with pytest.raises(ValueError, match="not analytic"):
        d.gradient(kinetic, [1.0, 2.0])
    assert np.allclose(d.gradient(kinetic, [1.0, 2.0], "finite"), [9.1e-31, 1.82e-30], rtol=1e-10, atol=0)
    exact = d.gradient(lambda v: 0.5 * 9.1e-31 * (v[0] ** 2 + v[1] ** 2), [1.0, 2.0])
    assert np.allclose(exact, [9.1e-31, 1.82e-30], rtol=1e-15, atol=0)


def test_non_analytic_part_on_top_of_a_large_constant_is_detected():
    with pytest.raises(ValueError, match="not analytic"):
        d.gradient(lambda x: 1e6 + np.abs(x[0]), [0.7])


def test_small_non_analytic_part_next_to_a_smooth_slope_is_detected():
    # audit: with a tolerance of 1e-3 between the two derivatives this returned 0.001 instead of 0.0010004
    with pytest.raises(ValueError, match="not analytic"):
        d.gradient(lambda x: 1.0 + 1e-3 * x[0] + 4e-7 * np.abs(x[0]), [0.37])


_NON_ANALYTIC = {
    "abs": (np.abs, np.sign),
    "conjugate": (lambda z: np.sqrt(z * np.conj(z)), np.sign),
    "real part": (lambda z: np.real(z) ** 2, lambda x: 2.0 * x),
    "cast to float": (lambda z: np.sin(np.real(z)), np.cos),
}


@pytest.mark.parametrize("name", sorted(_NON_ANALYTIC))
def test_no_wrong_derivative_is_accepted_over_a_grid_of_sizes_and_offsets(name):
    """A non-analytic term of any size on top of an offset of any size is refused, or its derivative is right.

    The one exception is a derivative that real arithmetic cannot see: smaller than 1e-10 of the function value
    per unit of the argument.
    """
    g, slope = _NON_ANALYTIC[name]
    visible_cases = 0
    for offset in (0.0, 1.0, 1e3, 1e6, 1e9):
        for factor in (1e-12, 1e-9, 1e-6, 1e-3, 1.0, 1e3):
            for x in (0.7, 3e4):
                true = factor * slope(x)
                value = abs(offset) + factor * abs(float(np.real(g(complex(x)))))
                visible = abs(true) * max(1.0, abs(x)) > 1e-10 * value
                visible_cases += visible
                try:
                    result = d.gradient(lambda y, o=offset, s=factor: o + s * g(y[0]) + 0.0 * y[0], [x])[0]
                except ValueError:
                    continue
                if visible:
                    assert result == pytest.approx(true, rel=1e-3, abs=0), (name, offset, factor, x)
    assert visible_cases >= 30  # the grid does test something


def test_kink_at_the_point_is_refused_by_the_finite_method():
    with pytest.raises(ValueError, match="could not be determined"):
        d.gradient(lambda x: np.abs(x[0]), [0.0], "finite")


# --------------------------------------------------------------------------------------
# Scales far from one
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("length", [1.0, 1e-2, 1e-5, 1e-7])
def test_analytic_function_with_a_small_length_scale_is_accepted(length):
    x = 0.3
    result = d.gradient(lambda y: np.sin(y[0] / length), [x])[0]
    assert result * length == pytest.approx(np.cos(x / length), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize(("length", "tolerance"), [(1.0, 1e-12), (1e-2, 1e-10), (1e-5, 1e-9), (1e-7, 1e-6)])
def test_finite_gradient_resolves_small_length_scales(length, tolerance):
    # x + h is rounded to 2e-16 |x|, which a step of 1e-9 feels: the error grows as the scale shrinks
    x = 0.3
    result, error = d.jacobian_with_error(lambda y: np.sin(y[0] / length), [x], "finite")
    assert result[0] * length == pytest.approx(np.cos(x / length), rel=tolerance, abs=0)
    assert abs(result[0] - np.cos(x / length) / length) <= 3.0 * error[0]


@pytest.mark.parametrize("method", ["complex", "finite"])
def test_hessian_resolves_small_length_scales(method):
    # the first version returned a relative error of -1.09 for sin(1e4 x)
    # measured: 4e-11 (complex-step gradient), 2e-8 (finite gradient, with an estimate of 1e-8)
    x, length = 0.3, 1e-4
    result, error = d.hessian_with_error(lambda y: np.sin(y[0] / length), [x], method)
    tolerance = 1e-9 if method == "complex" else 1e-6
    assert result[0, 0] * length**2 == pytest.approx(-np.sin(x / length), rel=tolerance, abs=0)
    assert abs(result[0, 0] + np.sin(x / length) / length**2) <= 5.0 * error[0, 0]


def test_function_that_cannot_be_resolved_is_refused():
    with pytest.raises(ValueError, match="could not be determined"):
        d.gradient(lambda y: np.sin(y[0] / 1e-14), [0.3], "finite")


@pytest.mark.parametrize("point", [0.0, 1e-8, 1e-3])
def test_analytic_function_near_a_stationary_point_is_accepted(point):
    # derivative and error estimate shrink together near x^n at 0; the size of f in the wider neighbourhood decides
    assert d.gradient(lambda y: y[0] ** 3 + 2.0, [point])[0] == pytest.approx(3.0 * point**2, rel=1e-15, abs=1e-50)
    assert d.gradient(lambda y: -0.25 * y[0] ** 4, [point])[0] == pytest.approx(-(point**3), rel=1e-15, abs=1e-50)
    expected = -np.sin(point / 1e-3) / 1e-3
    assert d.gradient(lambda y: np.cos(y[0] / 1e-3), [point])[0] == pytest.approx(expected, rel=1e-12, abs=1e-50)
    assert abs(d.gradient(lambda y: y[0] ** 5, [0.0], "finite")[0]) < 1e-30


def test_analytic_functions_are_accepted_over_a_grid_of_sizes_offsets_and_length_scales():
    accepted = total = 0
    for offset in (0.0, 1.0, 1e3, 1e6, 1e9):
        for factor in (1e-12, 1e-9, 1e-6, 1e-3, 1.0, 1e3):
            for x in (0.0, 1e-6, 0.7, 3e4):
                for length in (1e-6, 1e-3, 1.0, 1e3):
                    total += 1
                    try:
                        result = d.gradient(lambda y, o=offset, s=factor, w=length: o + s * np.sin(y[0] / w), [x])[0]
                    except ValueError:
                        continue
                    accepted += 1
                    assert result == pytest.approx(factor * np.cos(x / length) / length, rel=1e-12, abs=0)
    # a variation that is tiny against a large offset cannot be checked by real differences and is refused
    assert accepted >= _ACCEPTED_OF_480


_ACCEPTED_OF_480 = 420  # measured: 430


# --------------------------------------------------------------------------------------
# Found by the audits
# --------------------------------------------------------------------------------------


def test_dependence_below_the_spacing_of_the_values_is_not_taken_for_independence():
    # 1e16 + q: f(q + h) == f(q) for the first step, yet the derivative is 1
    result, error = d.jacobian_with_error(lambda x: 1e16 + x[0], [0.0], "finite")
    assert result[0] == pytest.approx(1.0, abs=5.0 * error[0])
    assert error[0] < 1e-3
    assert d.gradient(lambda x: 1e16 + x[0], [0.0])[0] == 1.0  # and the complex step is not refused for it


def test_narrow_feature_between_the_nodes_is_resolved():
    # a Gaussian of width 1e-6 seen from a stencil of width 1e-3: the first version returned 0 for both
    def bump(x):
        return np.exp(-((x[0] / 1e-6) ** 2))

    x = 3e-7
    first = -2.0 * x / 1e-12 * np.exp(-0.09)
    second = (-2.0 / 1e-12 + 4.0 * x**2 / 1e-24) * np.exp(-0.09)
    assert d.gradient(bump, [x], "finite")[0] == pytest.approx(first, rel=1e-11, abs=0)
    assert d.hessian(bump, [x])[0, 0] == pytest.approx(second, rel=1e-11, abs=0)


def test_pole_between_the_nodes_is_resolved():
    # Coulomb potential at 5e-11 m: the first stencil (1e-3) straddles the origin
    def coulomb(q):
        return 2.3e-28 / np.sqrt(np.sum(q * q))

    result = d.gradient(coulomb, np.array([5e-11, 0.0, 0.0]), "finite")
    assert result[0] == pytest.approx(-2.3e-28 / 25e-22, rel=1e-10, abs=0)
    assert np.all(np.abs(result[1:]) <= 1e-12 * abs(result[0]))


def test_edge_of_the_domain_on_a_stencil_node():
    # log(x - 0.369) at 0.37: the node x - h of the first step is the edge itself (the complex-step gradient
    # is finite garbage beyond it); it returned -7.85e32
    assert d.hessian(lambda x: np.log(x[0] - 0.369), [0.37])[0, 0] == pytest.approx(-1e6, rel=1e-8, abs=0)


def test_coordinate_at_zero_in_a_problem_of_large_scale():
    # 1/r potential at (7.8e11, 0): along y the natural scale is 7.8e11, the first step 1e-3
    result = d.hessian(lambda q: 1e37 / np.sqrt(q[0] ** 2 + q[1] ** 2), [7.8e11, 0.0])
    expected = 1e37 * np.array([[2.0 / 7.8e11**3, 0.0], [0.0, -1.0 / 7.8e11**3]])
    assert np.allclose(result, expected, rtol=1e-10, atol=1e-10 * abs(expected[0, 0]))


def test_large_step_that_only_averages_over_structure_is_not_trusted():
    # sin(x / 1e-7): at a step of 1e9 the difference quotient of a bounded function is tiny and looked like a
    # vanishing derivative; larger steps must agree with the result of the smaller ones
    x = 0.3
    result = d.gradient(lambda y: np.sin(y[0] / 1e-7), [x], "finite")[0]
    assert result * 1e-7 == pytest.approx(np.cos(x / 1e-7), rel=1e-6, abs=0)

    # the same with a small amplitude on an offset, where the smallest steps are pure rounding noise
    # (the derivative 6e-10 of a function of size 1 is known to 5e-5 only: its estimate says so, and the call
    # that returns no estimate refuses)
    def small(y):
        return 1.0 + 1e-9 * np.sin(y[0])

    result, error = d.jacobian_with_error(small, [3e4], "finite")
    assert result[0] == pytest.approx(1e-9 * np.cos(3e4), rel=1e-3, abs=0)
    assert abs(result[0] - 1e-9 * np.cos(3e4)) <= error[0] < 1e-3 * abs(result[0])
    with pytest.raises(ValueError, match="could not be determined"):
        d.gradient(small, [3e4], "finite")


def test_steps_of_inner_differences_do_not_collide():
    # two inner differences with the same code and different closures, scales 1e3 and 1e-6: the first design
    # keyed the frozen steps by code object and gave -280 for the second derivative of the second function
    def make(amplitude, scale):
        return lambda x: amplitude * np.sin(x[0] / scale)

    def inner(y):
        return np.array(
            [d.gradient(make(1e3, 1e3), [y[0]], "finite")[0], d.gradient(make(1e-6, 1e-6), [y[0]], "finite")[0]]
        )

    result = d.jacobian(inner, [0.3e-6], "finite")[1, 0]
    assert result == pytest.approx(-1e-6 * np.sin(0.3) / 1e-12, rel=1e-8, abs=0)


def test_third_derivative_through_nested_differences():
    result = d.jacobian(lambda y: d.hessian(lambda z: np.sin(z[0]) * np.exp(z[1]), y), X, "finite")
    assert result[0, 0, 0] == pytest.approx(-np.cos(X[0]) * np.exp(X[1]), rel=1e-8, abs=0)
    assert d._stack() == []  # the bookkeeping is released


def test_bookkeeping_is_released_after_an_exception_inside_a_nested_difference():
    def broken(y):
        if y[0] > 0.7:
            raise RuntimeError("boom")
        return d.gradient(lambda z: np.sin(z[0]), y, "finite")

    with pytest.raises(RuntimeError):
        d._finite_column(broken, np.array([0.7]), 0)
    assert d._stack() == []
    assert d.gradient(lambda z: np.sin(z[0]), [0.3], "finite")[0] == pytest.approx(np.cos(0.3), rel=1e-12, abs=0)


def test_bookkeeping_is_separate_for_every_thread():
    # six threads differentiate functions of scales 1e-4, 1 and 1e5 at once; with one shared stack up to a third
    # of the results were wrong
    failures = []

    def work(scale):
        for _ in range(40):
            inner = lambda y, s=scale: d.gradient(lambda x: np.sin(x[0] / s), y, "finite")  # noqa: E731
            result = d.jacobian(inner, [0.3 * scale], "finite")[0, 0]
            if abs(result * scale**2 + np.sin(0.3)) > 1e-6:
                failures.append((scale, result))

    threads = [threading.Thread(target=work, args=(s,)) for s in (1e-4, 1e5, 1.0, 1e-4, 1e5, 1.0)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []


def test_vanishing_derivative_of_a_noisy_function_is_returned_with_an_honest_error():
    rng = np.random.default_rng(5)

    def noisy_constant(y):
        return 0.333 + 1e-13 * rng.standard_normal()

    result, error = d.jacobian_with_error(noisy_constant, [2.0], "finite")
    assert abs(result[0]) <= 5.0 * error[0]
    assert error[0] < 1e-9


def test_identically_zero_component_next_to_the_edge_of_the_domain():
    # a tensor component that vanishes, in a function that is undefined for x < 0 (so that the far probes fail)
    def f(x):
        return np.array([0.0 * x[0], x[0] ** (2.0 / 3.0)])

    result, error = d.jacobian_with_error(f, [1.2e-3], "finite")
    assert result[0, 0] == 0.0 and error[0, 0] == 0.0
    assert result[1, 0] == pytest.approx(2.0 / 3.0 * 1.2e-3 ** (-1.0 / 3.0), rel=1e-9, abs=0)


def test_second_derivatives_refuse_a_non_analytic_function():
    # the inner gradients are taken without the check, so the check at the point itself must not be dropped
    with pytest.raises(ValueError, match="not analytic"):
        d.hessian(lambda x: np.abs(x[0]) ** 3 + x[1] ** 2, [0.7, 0.2])
    with pytest.raises(ValueError, match="not analytic"):
        d.mixed_second(lambda x, y: np.abs(x[0]) * y[0] ** 2, [0.7], [0.3])


def test_steps_of_one_point_are_reused_at_the_next_and_searched_again_when_they_fail():
    calls = []

    def f(x):
        calls.append(1)
        return np.array([np.sin(x[0] / 1e-4), x[1] ** 2, 3.0])

    plan: dict = {}
    first, _ = d._finite_jacobian(f, np.array([0.3e-4, 2.0]), plan)
    searched = len(calls)
    calls.clear()
    second, error = d._finite_jacobian(f, np.array([0.31e-4, 2.001]), plan)
    assert len(calls) < searched / 2  # six values per coordinate and step
    expected = np.array([[np.cos(0.31) / 1e-4, 0.0], [0.0, 2.0 * 2.001], [0.0, 0.0]])
    assert np.allclose(second, expected, rtol=1e-8, atol=0)
    assert np.all(np.abs(second - expected) <= 5.0 * error + 1e-300)
    # far away the stored step no longer resolves the first element: the plan is refused and rebuilt
    ratios, independent, reach = plan["steps"][0]
    plan["steps"][0] = (ratios * 1e6, independent, reach)
    calls.clear()
    third, _ = d._finite_jacobian(f, np.array([0.3e-4, 2.0]), plan)
    assert np.allclose(third, first, rtol=1e-9, atol=0)
    assert len(calls) >= searched


def test_reused_steps_see_a_dependence_that_was_absent_at_the_first_point():
    # p = m(q) v vanishes for every q while v = 0, so its q-derivative is exactly zero there; later it is not
    def momentum(z):
        return np.array([(2.0 + np.cos(z[0])) * z[1]])

    plan: dict = {}
    first, _ = d._finite_jacobian(momentum, np.array([0.4, 0.0]), plan)
    assert first[0, 0] == 0.0
    later, _ = d._finite_jacobian(momentum, np.array([0.5, 1.5]), plan)
    assert later[0, 0] == pytest.approx(-np.sin(0.5) * 1.5, rel=1e-9, abs=0)
    assert later[0, 1] == pytest.approx(2.0 + np.cos(0.5), rel=1e-9, abs=0)
