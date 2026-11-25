"""Numerical derivatives of user-supplied functions (private helper, no physics).

Two methods, both derived here from the Taylor series of ``f`` and tied to no publication:

``"complex"``
    f(x + i h) = f(x) + i h f'(x) − h² f''(x)/2 − ..., so Im f(x + i h) / h = f'(x) + O(h²).
    There is no subtraction, hence no cancellation, and h can be 1e-30: the result is exact to
    rounding whatever the scale of the function. It needs a function that is analytic and accepts
    complex arguments. ``abs``, ``hypot``, ``norm``, ``arctan2``, comparisons and ``.real`` break
    this silently (the imaginary part is lost and the derivative comes out as 0), so every
    complex-step derivative is compared with real central differences and refused when they
    contradict it (see ``_check_analytic`` for what the comparison can and cannot detect).
``"finite"``
    Four-point central difference, [f(x−2h) − 8 f(x−h) + 8 f(x+h) − f(x+2h)] / (12 h), error
    h⁴ f⁽⁵⁾/30. The step starts at 1e-3 · max(1, |x_i|) and is divided by ten until the two-point
    difference (f(x+h) − f(x−h)) / (2h), whose error is h² f‴/6, agrees with the four-point value
    to 3e-6: the step then resolves the scale on which the function varies, whatever that scale
    is, and the four-point error is below about 1e-11 relative. If no step meets that rule (the
    function is itself a noisy numerical derivative), the step where the two formulas agree best
    is used provided they agree to 1e-3; the accuracy is then that agreement squared at best. A
    derivative below 1e-6 |f| / max(1, |x_i|) at the first step is returned as it is, because such
    a value cannot be told from the noise of a function that is itself computed numerically. A
    function that varies on a scale below 1e-9 of the coordinate is refused. At a kink the result
    is the symmetric difference quotient (0 for ``abs`` at 0), not an error.

Second derivatives are four-point central differences (same step rule) of the first derivatives:
about 1e-12 relative with the complex-step first derivative for a function whose scale of
variation is near the coordinate scale, and 2e-16 · |f'| / (h |f''|) in general, because the
step has to shrink with the scale of the function.

Verification: tests/unit/test_differentiation.py.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]
Method = Literal["complex", "finite"]

_EPS = float(np.finfo(float).eps)
_COMPLEX_STEP = 1e-30
_FIRST_STEP = 1e-3  # relative to max(1, |x_i|)
_STEP_DECADES = 7  # steps 1e-3 ... 1e-9
_RESOLVED = 3e-6  # agreement of the two-point and four-point differences that accepts a step
_NEGLIGIBLE = 1e-6  # a derivative this small relative to |f| / max(1, |x_i|) is returned from the first step
_LOOSE = 1e-3  # fallback for noisy functions: best agreement of the two formulas over all steps
_CHECK_AGREEMENT = 1e-3  # agreement of a central difference with the complex-step derivative
_CHECK_DECADES = (
    10  # the check goes on to 1e-12: close to the edge of a domain only a very small real step stays inside
)


def _scale(x: Array) -> Array:
    return np.maximum(1.0, np.abs(x))


def _real(value: ArrayLike) -> Array:
    return np.asarray(value, dtype=float)


_FROZEN: list[dict] = []  # one dictionary per finite difference that is being evaluated, innermost last


def _stencil(f: Callable[[Array], ArrayLike], x: Array, i: int, h: float) -> tuple[Array, Array, float]:
    e = np.zeros_like(x)
    e[i] = h
    fm2, fm1, fp1, fp2 = _real(f(x - 2 * e)), _real(f(x - e)), _real(f(x + e)), _real(f(x + 2 * e))
    four = (fm2 - 8.0 * fm1 + 8.0 * fp1 - fp2) / (12.0 * h)
    two = (fp1 - fm1) / (2.0 * h)
    values = max(float(np.max(np.abs(v))) if v.size else 0.0 for v in (fm2, fm1, fp1, fp2))
    return four, two, values


def _finite_column(f: Callable[[Array], ArrayLike], x: Array, i: int) -> Array:
    """Four-point derivative along coordinate ``i`` with a step that resolves the function.

    When this derivative is itself evaluated inside another finite difference (a second derivative, or the
    derivative of a curvature tensor), the step chosen at the first stencil point is kept for the other points:
    a step that changed from point to point would make the error of the inner derivative jump, and the outer
    difference would amplify the jump.
    """
    base = _FIRST_STEP * _scale(x)[i]
    key = (getattr(f, "__code__", None), i, x.size)
    if _FROZEN and key in _FROZEN[-1]:
        return _stencil(f, x, i, _FROZEN[-1][key])[0]
    best, best_mismatch, best_step = None, np.inf, base
    _FROZEN.append({})
    try:
        for decade in range(_STEP_DECADES):
            h = base * 10.0**-decade
            _FROZEN[-1].clear()
            four, two, values = _stencil(f, x, i, h)
            size = float(np.max(np.abs(four))) if four.size else 0.0
            if decade == 0:
                first, first_small = four, size <= _NEGLIGIBLE * values / _scale(x)[i]
            noise = 50.0 * _EPS * values / h
            if np.all(np.abs(four - two) <= _RESOLVED * size + noise):
                best, best_mismatch, best_step = four, 0.0, h
                break
            mismatch = float(np.max(np.abs(four - two))) / size if size > 0.0 else np.inf
            if mismatch < best_mismatch:
                best, best_mismatch, best_step = four, mismatch, h
    finally:
        _FROZEN.pop()
    # A function that is itself a numerical derivative carries more noise than the estimate above knows of, and
    # no step may meet the strict rule; the step where the two formulas agree best is then used if they agree to 1e-3.
    if (best is None or best_mismatch > _LOOSE) and first_small:
        # a derivative below 1e-6 of |f| / max(1, |x_i|): the two formulas differ only by the noise of the function
        best, best_mismatch, best_step = first, 0.0, base
    if best is None or best_mismatch > _LOOSE:
        raise ValueError(
            f"finite-difference derivative along coordinate {i} did not settle for steps down to "
            f"{base * 10.0 ** -(_STEP_DECADES - 1):.1e}: the function is not smooth there or varies on a much "
            "smaller scale than its argument; rescale the coordinate"
        )
    if _FROZEN:
        _FROZEN[-1][key] = best_step
    return best


def _finite_jacobian(f: Callable[[Array], ArrayLike], x: Array) -> Array:
    return np.stack([_finite_column(f, x, i) for i in range(x.size)], axis=-1)


def _complex_jacobian(f: Callable[[Array], ArrayLike], x: Array) -> Array:
    columns = []
    for i in range(x.size):
        z = x.astype(complex)
        z[i] += 1j * _COMPLEX_STEP
        columns.append(np.imag(np.asarray(f(z), dtype=complex)) / _COMPLEX_STEP)
    return np.stack(columns, axis=-1)


def _check_analytic(f: Callable[[Array], ArrayLike], x: Array, result: Array) -> None:
    """Raise when real central differences contradict the complex-step derivative ``result``.

    Central differences are taken with steps h = 1e-3, 1e-4, ... 1e-12 times max(1, |x_i|) until
    every element of ``result`` is vouched for. With the rounding noise of a difference,
    floor = 20 eps |f| / h for each element of f, an element is vouched for

    - when a difference agrees with it to 1e-3 at a step whose floor is below 1 % of the element;
    - when element and difference agree within ten times the floor at the first step (a derivative
      that vanishes or is too small for real arithmetic to see more of it);
    - when the gap between element and difference shrinks like the h² truncation error of the
      central difference, as it does where the derivative is small against the higher derivatives:
      by a factor of fifty or more over each of two successive decades, or into the floor after
      shrinking in every decade before. The constant gap left by a non-analytic function does not
      shrink.

    What this cannot detect: a non-analytic part whose derivative is below
    4e-11 |f| / max(1, |x_i|), or a kink closer to x than the step that would be needed to see it. Such a derivative is below the resolution of double precision
    for real differences too, so ``method="finite"`` could not compute it either. In the other
    direction, an analytic function whose variation is that small against a large offset, or whose
    length scale is so short that no step resolves it above the floor, is refused.
    """
    pending = np.ones(result.shape, dtype=bool)
    scale = _scale(x)
    previous = np.full(result.shape, np.inf)
    streak = np.zeros(result.shape, dtype=int)
    clean = np.ones(result.shape, dtype=bool)  # the gap has shrunk in every decade so far
    seen = np.zeros(result.shape, dtype=bool)  # a step inside the domain of f has been taken
    for decade in range(_CHECK_DECADES):
        reference = np.empty_like(result)
        floor = np.empty_like(result)  # rounding noise of each difference, from the size of that element of f
        for i in range(x.size):
            if not np.any(pending[..., i]):
                reference[..., i] = result[..., i]
                floor[..., i] = 0.0
                continue
            h = _FIRST_STEP * scale[i] * 10.0**-decade
            e = np.zeros_like(x)
            e[i] = h
            with np.errstate(all="ignore"):  # a step that leaves the domain gives NaN and vouches for nothing
                plus, minus = _real(f(x + e)), _real(f(x - e))
            reference[..., i] = (plus - minus) / (2.0 * h)
            floor[..., i] = 20.0 * _EPS * np.maximum(np.abs(plus), np.abs(minus)) / h
        finite = np.isfinite(reference) & np.isfinite(floor)
        size = np.maximum(np.abs(result), np.abs(reference))
        gap = np.abs(result - reference)
        with np.errstate(invalid="ignore"):
            vouched = finite & (gap <= _CHECK_AGREEMENT * size) & (floor <= 1e-2 * size)
            # the first step that stays inside the domain: agreement within the rounding noise of the difference
            vouched |= finite & ~seen & (gap <= 10.0 * floor)
            # later steps: the gap must have stood clear of the noise of this step before it shrank; a gap that is
            # exactly zero because the step is below the spacing of the function values proves nothing
            shrunk = finite & seen & (gap <= 0.02 * previous) & (floor <= previous)
        streak = np.where(shrunk, streak + 1, np.where(finite & seen, 0, streak))
        vouched |= streak >= 2
        vouched |= clean & shrunk & (gap <= floor)
        clean &= shrunk | ~(finite & seen)
        previous = np.where(finite, gap, previous)
        seen |= finite
        pending &= ~vouched
        if not pending.any():
            return
    raise ValueError(
        "complex-step derivative disagrees with central differences: the function is not analytic "
        "in its argument (abs, hypot, norm, arctan2, comparisons or a cast to float), or its variation is "
        "too small against its value to be checked; use method='finite'"
    )


def _first(f: Callable[[Array], ArrayLike], x: Array, method: Method, check: bool) -> Array:
    if method == "finite":
        return _finite_jacobian(f, x)
    if method != "complex":
        raise ValueError("method must be 'complex' or 'finite'")
    result = _complex_jacobian(f, x)
    if check:
        _check_analytic(f, x, result)
    return result


def jacobian(f: Callable[[Array], ArrayLike], x: ArrayLike, method: Method = "complex") -> Array:
    """Derivative of ``f`` with respect to the 1-D vector ``x``.

    Returns an array of shape ``f(x).shape + (x.size,)``: the last axis runs over the
    components of ``x``. A scalar ``f`` therefore gives the gradient.

    Raises
    ------
    ValueError
        With ``method="complex"`` when the complex-step result is contradicted by real central
        differences, which means that ``f`` is not analytic in its argument (or ignores the
        imaginary part); use ``method="finite"`` for such a function. With ``method="finite"``
        when no step between 1e-3 and 1e-9 of the coordinate scale resolves the function.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D vector")
    return _first(f, x, method, check=True)


def gradient(f: Callable[[Array], ArrayLike], x: ArrayLike, method: Method = "complex") -> Array:
    """Gradient of a scalar function; shape ``(x.size,)``."""
    result = jacobian(f, x, method)
    if result.ndim != 1:
        raise ValueError("gradient needs a scalar function")
    return result


def hessian(f: Callable[[Array], ArrayLike], x: ArrayLike, method: Method = "complex") -> Array:
    """Matrix of second derivatives of a scalar function, symmetrized; shape ``(x.size, x.size)``.

    Four-point central difference of the gradient with the step rule of the module docstring. The
    analyticity check of the complex-step gradient is made once, at ``x``.
    """
    x = np.asarray(x, dtype=float)
    gradient(f, x, method)  # shape and analyticity check at the point itself
    result = _finite_jacobian(lambda y: _first(f, y, method, check=False), x)
    return 0.5 * (result + result.T)


def mixed_second(
    f: Callable[[Array, Array], ArrayLike], x: ArrayLike, y: ArrayLike, method: Method = "complex"
) -> Array:
    """Mixed derivatives ∂²f/∂x_i∂y_j of a scalar ``f(x, y)``; shape ``(x.size, y.size)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    gradient(lambda u: f(u, y), x, method)  # analyticity check at the point itself
    return _finite_jacobian(lambda v: _first(lambda u: f(u, v), x, method, check=False), y)
