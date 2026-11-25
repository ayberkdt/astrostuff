"""Numerical derivatives against closed forms."""

from __future__ import annotations

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


def test_complex_step_gradient_is_exact_to_rounding():
    assert np.allclose(d.gradient(_f, X), _gradient(X), rtol=1e-15, atol=0)


def test_finite_gradient():
    assert np.allclose(d.gradient(_f, X, "finite"), _gradient(X), rtol=1e-11, atol=0)


@pytest.mark.parametrize(("method", "rtol"), [("complex", 1e-11), ("finite", 1e-7)])
def test_hessian(method, rtol):
    result = d.hessian(_f, X, method)
    assert np.allclose(result, _hessian(X), rtol=rtol, atol=0)
    assert np.array_equal(result, result.T)


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


def test_non_analytic_function_is_detected():
    # abs() of a complex number is its modulus: the imaginary part is lost and the derivative would be 0.
    with pytest.raises(ValueError, match="not analytic"):
        d.gradient(lambda x: np.abs(x[0]) * x[1], X)
    assert np.allclose(d.gradient(lambda x: np.abs(x[0]) * x[1], X, "finite"), [X[1], X[0]], rtol=1e-11)


def test_argument_checks():
    with pytest.raises(ValueError):
        d.jacobian(_f, [[1.0, 2.0]])
    with pytest.raises(ValueError):
        d.jacobian(_f, X, "other")
    with pytest.raises(ValueError):
        d.gradient(lambda x: x, X)


# --------------------------------------------------------------------------------------
# Scale independence. An audit of the first version of this helper showed that its fixed steps (1e-6 for the
# check, 1e-3 for second derivatives) and the absolute tolerance of the check gave wrong results without an error
# for functions whose values or length scales are far from one. Each test below failed with that version.
# --------------------------------------------------------------------------------------


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
    # the same energy written analytically passes and is exact
    assert np.allclose(
        d.gradient(lambda v: 0.5 * 9.1e-31 * (v[0] ** 2 + v[1] ** 2), [1.0, 2.0]), [9.1e-31, 1.82e-30], rtol=1e-15
    )


def test_non_analytic_part_on_top_of_a_large_constant_is_detected():
    with pytest.raises(ValueError, match="not analytic"):
        d.gradient(lambda x: 1e6 + np.abs(x[0]), [0.7])


@pytest.mark.parametrize("length", [1.0, 1e-2, 1e-5, 1e-7])
def test_analytic_function_with_a_small_length_scale_is_accepted(length):
    # measured relative error of the complex-step gradient: at most 3e-12 (the argument x / length is rounded)
    x = 0.3
    result = d.gradient(lambda y: np.sin(y[0] / length), [x])[0]
    assert result * length == pytest.approx(np.cos(x / length), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize(("length", "tolerance"), [(1.0, 1e-11), (1e-2, 1e-9), (1e-4, 1e-8), (1e-6, 1e-6)])
def test_finite_gradient_resolves_small_length_scales(length, tolerance):
    # measured: 2e-13, 3e-11, 5e-10, 3e-8; the rounding error grows as 2e-16 / (h / length) with the smaller step
    x = 0.3
    result = d.gradient(lambda y: np.sin(y[0] / length), [x], "finite")[0]
    assert result * length == pytest.approx(np.cos(x / length), rel=tolerance)


@pytest.mark.parametrize(
    ("method", "length", "tolerance"), [("complex", 1e-4, 1e-9), ("complex", 2e-3, 1e-10), ("finite", 2e-3, 1e-8)]
)
def test_hessian_resolves_small_length_scales(method, length, tolerance):
    # the first version returned a relative error of -1.09 for sin(1e4 x); measured now 2e-11, 4e-12, 5e-10
    x = 0.3
    result = d.hessian(lambda y: np.sin(y[0] / length), [x], method)[0, 0]
    assert result * length**2 == pytest.approx(-np.sin(x / length), rel=tolerance)


def test_function_that_cannot_be_resolved_is_refused():
    with pytest.raises(ValueError, match="did not settle"):
        d.gradient(lambda y: np.sin(y[0] / 1e-12), [0.3], "finite")


def test_error_orders_of_the_two_formulas():
    # four-point formula on x^5 at 0: the error is exactly -4 h^4 = -h^4 f^(5) / 30 with f^(5) = 120
    h = 1e-3
    assert d._stencil(lambda y: y[0] ** 5, np.array([0.0]), 0, h)[0] == pytest.approx(-4.0 * h**4, rel=1e-12)
    # complex step: no truncation error at any size of the argument
    for x in (1e-8, 1.0, 1e8):
        assert d.gradient(lambda y: y[0] ** 3, [x])[0] == pytest.approx(3.0 * x**2, rel=1e-15)


def test_step_of_an_inner_difference_is_kept_inside_an_outer_one():
    # third derivative as a finite difference of a Hessian element: measured relative error 2e-9
    result = d.jacobian(lambda y: d.hessian(lambda z: np.sin(z[0]) * np.exp(z[1]), y), X, "finite")
    expected = -np.cos(X[0]) * np.exp(X[1])  # d^3/dx^3
    assert result[0, 0, 0] == pytest.approx(expected, rel=1e-7)
    assert not d._FROZEN  # the bookkeeping is released


def test_vanishing_derivative_of_a_noisy_function_is_returned_not_refused():
    rng = np.random.default_rng(5)

    def noisy_constant(y):
        return 0.333 + 1e-13 * rng.standard_normal()

    assert abs(d.gradient(noisy_constant, [2.0], "finite")[0]) < 1e-9


@pytest.mark.parametrize("point", [0.0, 1e-8, 1e-3])
def test_analytic_function_near_a_stationary_point_is_accepted(point):
    # central differences of x^3 and x^4 at 0 are pure truncation error (h^2 and 0); the first rule set refused x^3
    assert d.gradient(lambda y: y[0] ** 3 + 2.0, [point])[0] == pytest.approx(
        3.0 * point**2, rel=1e-15, abs=1e-50
    )  # complex step: -h^2 = -1e-60 at 0
    assert d.gradient(lambda y: -0.25 * y[0] ** 4, [point])[0] == pytest.approx(
        -(point**3), rel=1e-15, abs=1e-50
    )  # complex step: -h^2 = -1e-60 at 0
    assert d.gradient(lambda y: np.cos(y[0] / 1e-3), [point])[0] == pytest.approx(
        -np.sin(point / 1e-3) / 1e-3, rel=1e-12, abs=1e-300
    )


_NON_ANALYTIC = {
    "abs": (np.abs, np.sign),
    "conjugate": (lambda z: np.sqrt(z * np.conj(z)), np.sign),
    "real part": (lambda z: np.real(z) ** 2, lambda x: 2.0 * x),
    "cast to float": (lambda z: np.sin(np.real(z)), np.cos),
}


@pytest.mark.parametrize("name", sorted(_NON_ANALYTIC))
def test_no_wrong_derivative_is_accepted_over_a_grid_of_sizes_and_offsets(name):
    """A non-analytic term of any size on top of an offset of any size is refused, or its derivative is right.

    The one exception is a derivative that real arithmetic cannot see at all: smaller than 1e-10 of the function
    value per unit of the argument. Measured over the 90 cases of each kind: every visible case is refused.
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
                    assert result == pytest.approx(true, rel=1e-3), (name, offset, factor, x)
    assert visible_cases >= 30  # the grid does test something


def test_analytic_functions_are_accepted_over_a_grid_of_sizes_offsets_and_length_scales():
    # measured: 418 of 480 accepted (the refused ones vary by less than rounding against their offset), worst
    # relative error of an accepted derivative 7.4e-14
    accepted = 0
    for offset in (0.0, 1.0, 1e3, 1e6, 1e9):
        for factor in (1e-12, 1e-9, 1e-6, 1e-3, 1.0, 1e3):
            for x in (0.0, 1e-6, 0.7, 3e4):
                for length in (1e-6, 1e-3, 1.0, 1e3):
                    try:
                        result = d.gradient(lambda y, o=offset, s=factor, w=length: o + s * np.sin(y[0] / w), [x])[0]
                    except ValueError:
                        assert factor <= 1e-6 * max(
                            offset, 1e-300
                        )  # only a variation tiny against the offset is refused
                        continue
                    accepted += 1
                    assert result == pytest.approx(factor * np.cos(x / length) / length, rel=1e-12)
    assert accepted >= 400
