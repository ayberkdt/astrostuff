"""Numerical derivatives of user-supplied functions (private helper, no physics).

Two methods, both derived here from the Taylor series of ``f`` and tied to no publication.

``"complex"``
    f(x + i h) = f(x) + i h f'(x) − h² f''(x)/2 − ..., so Im f(x + i h) / h = f'(x) + O(h²). There
    is no subtraction, hence no cancellation, and h is 1e-30 of the coordinate (of the smallest
    non-zero coordinate below 1, or 1, for a coordinate that is zero): the result is exact to rounding
    whatever the scale of the function. It needs a function that is analytic and accepts complex
    arguments. ``abs``, ``norm``, comparisons and ``.real`` break this silently (the imaginary part
    is lost), so every complex-step derivative is compared with the finite-difference derivative
    and refused when the two differ by more than twenty times the rounding of the latter (plus its
    truncation, measured against the next smaller step, where the step came out of a search), when a
    second-difference probe finds a kink at the point (see ``_kink_at``), or when no
    finite-difference step determines the derivative at all (the test cannot be made). For this
    test the finite difference is judged against the size of f (see below), because a derivative
    that is small against |f| / h cannot contradict the complex step, which does not suffer from
    the size of f.
``"finite"``
    Seven function values on a symmetric stencil, f(x) and f(x ± h), f(x ± 2h), f(x ± 3h), give

    - the six-point derivative D6 = [(f₊₃ − f₋₃) − 9 (f₊₂ − f₋₂) + 45 (f₊₁ − f₋₁)] / (60 h), error
      h⁶ f⁽⁷⁾/140 (written with differences of opposite points, so that a function that does not
      depend on the coordinate gives exactly zero), which is the value returned;
    - the four-point derivative D4 = [(f₋₂ − f₊₂) + 8 (f₊₁ − f₋₁)] / (12 h), error h⁴ f⁽⁵⁾/30;
    - the differences Δ² = f₋₁ − 2 f₀ + f₊₁ and Δ⁴ = f₋₂ − 4 f₋₁ + 6 f₀ − 4 f₊₁ + f₊₂.

    The error of D6 is estimated, for every element of f separately, as

        err = 1.1 σ / h  +  4 eps |x_i| |D6| / h  +  |D6 − D4|  +  |Δ⁴| / (2h) · min(1, |Δ⁴| / |Δ²|)

    with σ = max(4 eps max|f|, |Δ⁶| / 30), Δ⁶ being the sixth difference of the seven values,
    that is: the noise of the function values (rounding, or what Δ⁶ shows of it); rounding of x + h; the truncation error of D4, which
    exceeds that of D6 and also picks up the noise of a function that is itself computed
    numerically; and roughness, which counts in full when the stencil does not resolve the function
    (a pole or a narrow feature between the nodes: Δ⁴ is then as large as Δ²) and for little when
    it does (Δ⁴ ≈ (h/L)² Δ²). A step is acceptable for an element if

        err ≤ 1e-6 |D6| + 1e-7 S / h,

    where S is the variation of f over the stencil (largest minus smallest of the seven values,
    less eight times σ, and not below zero): S / h is the size of a difference quotient at that
    step (below the first step: the largest such size between that step and the first one; above
    it: not more than at the first step). The size of f itself is no scale for its derivatives
    (L + 1e12 has those of L), and with it in the place of S a derivative of 2 was returned as 2.02.
    The rule does not depend on the units of f or of x. The first step is 1e-3 · max(1, |x_i|). If the estimate there
    is not already a hundred times below the limit, or if a step a hundred times smaller does not
    give the same derivative within the estimates, the steps from 1e-12 to 1e9 times the first one
    are all tried. For each element the smallest acceptable step is taken first (truncation is least
    there), and a larger step replaces the current choice only if it gives the same derivative within
    the two estimates and has the smaller estimate; this is how rounding noise is reduced for a function
    whose scale of variation is much larger than max(1, |x_i|), without trusting a large step that
    merely averages over structure. An element for which no step is acceptable gets the step with the
    smallest estimate up to the first one, improved by larger steps under the same rule, and keeps
    that estimate. The functions that return estimates (``*_with_error``) return it, and their
    callers decide (the accelerations and the Riemann tensor refuse what their propagated estimate
    does not support). The functions that return the derivative alone accept such an element only
    if its estimate is below 1e-7 of the largest element that is determined (a vanishing element of
    a Jacobian whose other elements are known) and raise ValueError otherwise.
    A kink at x itself is refused (the estimate never falls below the limit there). A function that
    does not depend on the coordinate (equal values on the first stencil, and 3e9 first steps away
    on both sides within eight rounding errors) gives exactly zero; its estimate is what that
    leaves open, one rounding error of f (or the difference seen) over that distance. A term that
    changes f by less than that over 3e6 max(1, |x_i|) is not seen: at a coordinate that is zero
    the scale of the problem is unknown to the helper.

    What is not seen: structure finer than a hundredth of the first step whose amplitude is below
    about 1e-10 of the function value there (the first step and its confirmation then both look
    perfect), a feature in an element that is itself below 2e-13 of the largest element of f, and,
    as with every sampling method, features that fall between the nodes of all the steps tried. The
    functions that return the error estimate (``*_with_error``) let a caller test what it needs.

Second derivatives are finite differences, by the same rule, of the first derivatives. Inside an
outer difference the steps of inner differences are chosen once, at the central point, and reused
at the other stencil points: a step that changed from point to point would make the error of the
inner derivative jump, and the outer difference would amplify the jump. The bookkeeping is kept
per thread.

Verification: tests/unit/test_differentiation.py.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]
Method = Literal["complex", "finite"]

_EPS = float(np.finfo(float).eps)
_COMPLEX_STEP = 1e-30
_FIRST_STEP = 1e-3  # relative to max(1, |x_i|)
_DOWN, _UP = 12, 9  # decades searched below and above the first step
_ACCEPT_REL, _ACCEPT_ABS = 1e-6, 1e-7  # a result is returned only below this error estimate
_GOOD_REL, _GOOD_ABS = 1e-8, 1e-10  # below this estimate at the first step no other step is tried
_CONFIRM = -2  # decade of the step that must confirm a first step that looks perfect
# In first steps: where a function must still be unchanged to count as independent of a coordinate. The largest
# distance is three times the largest step that is searched: nothing nearer could have shown a dependence either.
_FAR = (3e9, 1e6, 1e3, 1e1)
_NOISE_SLACK = 100.0  # how many times the noise part of an estimate counts when two steps are compared
# The estimate of a difference of a function that is itself a finite difference was measured against exact
# derivatives (2400 elements, two functions, x of order 1 and of order 1e6): the true error reached 13.4 estimates.
# Differences of exact functions and of complex-step derivatives stayed below 2.5 estimates. The estimate of a
# nested difference is therefore multiplied by this factor before it is returned.
_NESTED_SAFETY = 30.0
_CHECK_FACTOR = 20.0  # complex-step and finite results may differ by this many error estimates

_LOCAL = threading.local()


class _Trace:
    """Steps chosen by the finite differences made during one evaluation of a function, in call order."""

    __slots__ = ("center_error", "index", "inner_error", "recording", "steps")

    def __init__(self) -> None:
        self.steps: list[tuple[tuple[int, int], Array, _Trace]] = []
        self.index = 0
        self.recording = True
        self.inner_error = 0.0  # largest relative error estimate of those differences, at the points used so far
        self.center_error = 0.0  # the same at the central point alone


def _stack() -> list[_Trace]:
    stack = getattr(_LOCAL, "stack", None)
    if stack is None:
        stack = _LOCAL.stack = []
    return stack


def _scale(x: Array) -> Array:
    return np.maximum(1.0, np.abs(x))


def _call(f: Callable[[Array], ArrayLike], x: Array, trace: _Trace, shape: tuple[int, ...] | None = None) -> Array:
    """Evaluate ``f`` with ``trace`` governing the finite differences made inside it.

    With ``shape`` (a stencil point, not the central one) a point at which ``f`` cannot be evaluated gives NaN:
    outside the domain of ``f`` a point carries no information, and the step that used it is discarded.
    """
    stack = _stack()
    depth = len(stack)
    trace.index = 0
    stack.append(trace)
    try:
        with np.errstate(all="ignore"):
            return np.asarray(f(x), dtype=float)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError):
        if shape is None:
            raise
        return np.full(shape, np.nan)
    finally:
        del stack[depth:]


def _differences(f: Callable[[Array], ArrayLike], x: Array, i: int, h: float, trace: _Trace, shape: tuple[int, ...]):
    e = np.zeros_like(x)
    e[i] = h
    return [_call(f, x + k * e, trace, shape) for k in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)]


def _six(values: list[Array], h: float) -> Array:
    fm3, fm2, fm1, fp1, fp2, fp3 = values
    return ((fp3 - fm3) - 9.0 * (fp2 - fm2) + 45.0 * (fp1 - fm1)) / (60.0 * h)


def _stencil(
    f: Callable[[Array], ArrayLike], x: Array, i: int, h: float, center: Array, trace: _Trace, loose: bool = False
):
    """Six-point derivative, its error estimate, the difference-quotient scale, the constancy flag and the data
    that classify an element as a rounding residue, for one step."""
    trace.inner_error = trace.center_error
    points = _differences(f, x, i, h, trace, np.shape(center))
    fm3, fm2, fm1, fp1, fp2, fp3 = points
    with np.errstate(all="ignore"):
        six = _six(points, h)
        four = ((fm2 - fp2) + 8.0 * (fp1 - fm1)) / (12.0 * h)
        second = fm1 - 2.0 * center + fp1
        fourth = fm2 - 4.0 * fm1 + 6.0 * center - 4.0 * fp1 + fp2
        values = np.maximum.reduce([np.abs(v) for v in (*points, center)])
        # how much the function changes over the stencil: the scale of its difference quotients. The size of
        # the function itself is no such scale (L + 1e12 has the derivatives of L).
        spread = np.maximum.reduce([*points, center]) - np.minimum.reduce([*points, center])
        # roughness: for a resolved function the fourth difference is (h / L)^2 times the second one and counts
        # for little; for a feature between the nodes, or for noise, the two are comparable and it counts in full
        rough = 0.5 * np.abs(fourth) / h * np.minimum(1.0, np.abs(fourth) / (np.abs(second) + 1e-300))
        sixth = (fm3 + fp3) - 6.0 * (fm2 + fp2) + 15.0 * (fm1 + fp1) - 20.0 * center
        # noise of one function value: at least a few rounding errors, and what the sixth difference shows (its
        # coefficients have a root sum of squares of 30.4); D6 amplifies it by 1.1 / h
        noise = np.maximum(4.0 * _EPS * values, np.abs(sixth) / 30.0)
        # seven noisy samples of a constant differ by a few times the noise: that is no variation
        spread = np.maximum(spread - 8.0 * noise, 0.0)
        error = (
            1.1 * noise / h  # rounding and noise of the function values
            + 4.0 * _EPS * abs(float(x[i])) / h * np.abs(six)  # rounding of x + h
            + np.abs(six - four)  # truncation of the four-point formula, and noise
            + rough
            + trace.inner_error * np.abs(six)  # a smooth error of inner differences passes through unnoticed
        )
        constant = np.all([v == center for v in points], axis=0)
        largest = float(np.max(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else 0.0
        # is this element, at this step, noise at the rounding level of the largest element?
        # (or at the level of the errors of the inner differences, if the function is itself a numerical derivative)
        tiny = values <= max(1e3 * _EPS, 10.0 * trace.inner_error) * largest
        noisy = np.abs(sixth) / 30.0 >= 0.05 * values
        # for such an element |D6| <= (2 + 18 + 90) / 60 max|f| / h whatever the six values are
        bound = 110.0 / 60.0 * values / h
        inner = trace.inner_error * np.abs(six)
        # the part of the estimate that is noise of the function values and of the step
        noise_part = 1.1 * noise / h + 4.0 * _EPS * abs(float(x[i])) / h * np.abs(six)
    return (
        six,
        error,
        (values if loose else spread) / h,
        constant,
        (tiny, noisy, bound, largest / h, inner, np.abs(sixth) / 30.0, noise_part, values, constant),
    )


def _ratio(six: Array, error: Array, quotient: Array, constant: Array, decade: int) -> Array:
    """Error estimate in units of the allowed error; inf where the step says nothing."""
    with np.errstate(all="ignore"):
        ratio = error / (_ACCEPT_REL * np.abs(six) + _ACCEPT_ABS * quotient)
        ratio = np.where(np.isfinite(six) & np.isfinite(ratio), ratio, np.inf)
    # ``constant`` is the confirmed flag of the first step (see _finite_column): the derivative is exactly zero
    return np.where(constant, 0.0, ratio)


def _residues(results: dict, extras: dict, constant: Array) -> dict:
    """Give the elements that are rounding residues their proper error estimate and size.

    An element of a tensor that vanishes is often computed as a rounding residue of terms that cancel. It is
    recognised by its values: at the rounding level of the largest element at every step, and noise (a sixth
    difference as large as the values) at half of the steps or more. A small but smooth element is not one. The
    derivative of a residue is zero within (2 + 18 + 90) / 60 max|f| / h, which bounds the six-point formula whatever
    the values are, and that estimate is judged against the size of the tensor, not of the residue.
    """
    decades = sorted(results)
    inside = np.stack([np.isfinite(results[d][0]) for d in decades])
    tiny = np.stack([extras[d][0] for d in decades])
    noisy = np.stack([extras[d][1] for d in decades])
    count = np.sum(inside, axis=0)
    residue = np.all(tiny | ~inside, axis=0) & (np.sum(noisy & inside, axis=0) * 2 >= count) & (count > 0) & ~constant
    out = {}
    for d in decades:
        six, error, quotient, flag = results[d]
        bound, largest = extras[d][2], extras[d][3]
        out[d] = (six, np.where(residue, bound, error), np.where(residue, largest, quotient), flag)
    return out


def _finite_column(
    f: Callable[[Array], ArrayLike],
    x: Array,
    i: int,
    record: list | None = None,
    loose: bool = False,
    noise_out: list | None = None,
) -> tuple[Array, Array, Array]:
    """Derivative along coordinate ``i`` and its error estimate (see the module docstring)."""
    stack = _stack()
    parent = stack[-1] if stack else None
    key = (i, x.size)
    if parent is not None and not parent.recording:
        # the steps were chosen when the enclosing function was evaluated at its central point
        if parent.index < len(parent.steps) and parent.steps[parent.index][0] == key:
            _, steps, child = parent.steps[parent.index]
            parent.index += 1
            result = None
            for h in np.unique(steps):
                points = _differences(f, x, i, float(h), child, np.shape(steps))
                six = _six(points, float(h))
                result = six if result is None else np.where(steps == h, six, result)
                # the error of this difference with the frozen step, for the enclosing difference to count
                fm3, fm2, fm1, fp1, fp2, fp3 = points
                with np.errstate(all="ignore"):
                    four = ((fm2 - fp2) + 8.0 * (fp1 - fm1)) / (12.0 * float(h))
                    values = np.maximum.reduce([np.abs(v) for v in points])
                    error = 1.1 * 4.0 * _EPS * values / float(h) + np.abs(six - four)
                    relative = np.where((steps == h) & (np.abs(six) > 0.0), error / np.abs(six), 0.0)
                relative = relative[np.isfinite(relative)]
                if relative.size:
                    parent.inner_error = max(parent.inner_error, min(1.0, float(np.max(relative))))
            if noise_out is not None:
                noise_out.append((np.zeros_like(result), np.zeros_like(result)))
            return result, np.zeros_like(result), np.ones(np.shape(result), dtype=bool)
        parent = None  # the call sequence differs from the recorded one: choose the steps afresh

    child = _Trace()
    center = _call(f, x, child)
    child.recording = False
    child.center_error = child.inner_error
    base = _FIRST_STEP * max(1.0, abs(float(x[i])))
    good = _GOOD_REL / _ACCEPT_REL  # in units of the allowed error

    first = _stencil(f, x, i, base, center, child, loose)
    six, error, quotient, constant, extra0 = first
    extras = {0: extra0}
    raw = constant.copy()
    # for an element found independent of the coordinate: how far that was confirmed, and how much the values differed
    reach = np.full(np.shape(center), np.inf)
    gap = np.zeros(np.shape(center))
    if np.any(constant):
        # Equal values on the first stencil may hide a dependence that is below the spacing of the values (the
        # q in 1e16 + q, or a term that varies on a galactic scale at a coordinate that is zero). The function counts
        # as independent of the coordinate only if it is also unchanged 3e9 first steps away, on both sides. If the
        # function cannot be evaluated that far away (the edge of its domain, an overflow), the largest of 1e6, 1e3
        # and 10 first steps at which it can be is used instead.
        shape = np.shape(center)
        confirmed = np.zeros(shape, dtype=bool)
        decided = np.zeros(shape, dtype=bool)
        for distance in _FAR:
            far = np.zeros_like(x)
            far[i] = distance * base
            plus, minus = _call(f, x + far, _Trace(), shape), _call(f, x - far, _Trace(), shape)
            usable = np.isfinite(plus) & np.isfinite(minus) & ~decided
            # equal to within a few rounding errors: far from x the same value is reached through other operands
            tolerance = 8.0 * _EPS * np.abs(center)
            with np.errstate(invalid="ignore"):
                differ = np.maximum(np.abs(plus - center), np.abs(minus - center))
                agree = usable & (differ <= tolerance)
            confirmed |= agree
            reach = np.where(agree, distance * base, reach)
            gap = np.where(agree, differ, gap)
            decided |= usable
            if np.all(decided | ~constant):
                break
        constant &= confirmed
        first = (six, error, quotient, constant)
    error = np.where(constant, 0.0, np.where(raw & ~constant, np.inf, error))
    first = (six, error, quotient, constant)
    results = {0: first}
    fine = _stencil(f, x, i, base * 10.0**_CONFIRM, center, child, loose)
    extras[_CONFIRM] = fine[4]
    results[_CONFIRM] = (fine[0], np.where(fine[3] & ~constant, np.inf, fine[1]), fine[2], constant)
    six, error, quotient = _residues(results, extras, constant)[0][:3]
    # The error that inner differences pass on is the same at every step of this one: searching other steps cannot
    # reduce it, so it does not count in the decision to search (it stays in the estimate that is returned).
    with np.errstate(invalid="ignore"):
        own = np.where(np.isfinite(error), np.maximum(error - extras[0][4], 0.0), error)
    settled = not np.any(_ratio(six, own, quotient, constant, 0) > good)
    first_noise = np.where(np.isfinite(extras[0][6]), extras[0][6], 0.0)
    if settled and loose:
        # For the test of a complex-step derivative the first step is taken only if its estimate is noise, not
        # truncation or roughness: the comparison is made within the noise (see _check_analytic).
        with np.errstate(invalid="ignore"):
            settled = bool(np.all((own <= 10.0 * first_noise) | constant | ~np.isfinite(own)))
    if settled:
        # A first step that looks perfect may still sit on structure finer than itself (a ripple of small amplitude
        # on a large offset gives small differences of every order). One step a hundred times smaller must give the
        # same derivative; if it does not, the whole range is searched.
        fine = _residues(results, extras, constant)[_CONFIRM]
        with np.errstate(invalid="ignore"):
            same = np.abs(six - fine[0]) <= 2.0 * (error + fine[1]) + 1e-300
        settled = bool(np.all(same | constant))
    if settled:
        best_six, best_error, best_step = six, error, np.full(np.shape(six), base)
        best_noise = first_noise
        determined = np.ones(np.shape(six), dtype=bool)
    else:
        for decade in [*range(-_DOWN, 0), *range(1, _UP + 1)]:
            if decade in results:
                continue
            other = _stencil(f, x, i, base * 10.0**decade, center, child, loose)
            extras[decade] = other[4]
            # Equal values at this step say nothing unless the function is confirmed independent of the
            # coordinate: the step may simply be below the spacing of the function values. Such a step is dead.
            dead = other[3] & ~constant
            results[decade] = (other[0], np.where(dead, np.inf, other[1]), other[2], constant)
        # The noise of the function values does not depend on the step, and one sixth difference may show it far
        # too small by chance (seven samples, and values that repeat at the smallest steps). The largest sixth
        # difference met at this step or at a smaller one is therefore a floor for the noise at this step: for a
        # function that the step resolves, the smaller steps show noise alone. Without the floor a function that
        # is itself a numerical derivative can anchor the choice on a step that is all noise.
        floor = np.zeros(np.shape(six))
        noises = {}
        for decade in sorted(results):
            sample = extras[decade][5]
            floor = np.maximum(floor, np.where(np.isfinite(sample), sample, 0.0))
            r_six, r_error, r_quotient, r_flag = results[decade]
            with np.errstate(invalid="ignore"):
                r_error = np.maximum(r_error, 1.1 * floor / (base * 10.0**decade))
                noises[decade] = np.maximum(extras[decade][6], 1.1 * floor / (base * 10.0**decade))
            results[decade] = (r_six, r_error, r_quotient, r_flag)
        results = _residues(results, extras, constant)
        shape = np.shape(six)
        # Below the first step a result is also acceptable against the largest difference quotient met between
        # that step and the first one: near a stationary point of high order (x^4 at 0) the derivative and its
        # error estimate shrink together, and only the size of f in the wider neighbourhood says that both are small.
        widest = np.zeros(shape)
        ratios = {}
        for decade in range(0, -_DOWN - 1, -1):
            six, error, quotient, constant = results[decade]
            widest = np.maximum(widest, np.where(np.isfinite(quotient), quotient, 0.0))
            ratios[decade] = _ratio(six, error, widest, constant, decade)
        # Above the first step the difference quotient of a curved function grows with the step and says nothing
        # about the derivative at the point: the allowance stays that of the first step.
        first_quotient = results[0][2]
        for decade in range(1, _UP + 1):
            six, error, quotient, constant = results[decade]
            with np.errstate(invalid="ignore"):
                capped = np.where(np.isfinite(first_quotient), np.minimum(quotient, first_quotient), quotient)
            if loose:
                capped = quotient
            ratios[decade] = _ratio(six, error, capped, constant, decade)
        # An element that is exactly zero at every point of every step inside the domain is identically zero in
        # the neighbourhood (a vanishing component of a tensor): its derivative is zero. It is anchored at once.
        quotients = np.stack([extras[d][7] for d in results])  # the largest |f| on each stencil
        inside = np.isfinite(quotients)
        anchored = np.all((quotients == 0.0) | ~inside, axis=0) & np.any(inside, axis=0)
        best_six = np.zeros(shape)
        best_error = np.full(shape, np.inf)
        best_step = np.full(shape, base)
        best_noise = np.zeros(shape)
        for decade in range(-_DOWN, _UP + 1):  # from the smallest step to the largest
            six, error, quotient, constant = results[decade]
            error = np.where(constant, 0.0, error)
            acceptable = ratios[decade] <= 1.0
            # the smallest acceptable step is the anchor: truncation is least there
            new = acceptable & ~anchored
            anchored |= new
            # A larger step replaces the current choice only where it gives the same derivative within the two
            # estimates and has the smaller estimate. The comparison is with the current choice, not with the
            # anchor: an anchor with a large (honest) error would agree with anything, including a large step
            # that merely averages over the structure of the function. The noise part of an estimate comes from
            # a handful of samples and was measured to fall short of the true noise by a factor of up to 45 for a
            # function that is itself a numerical derivative: it counts _NOISE_SLACK times in the comparison.
            noise = np.where(np.isfinite(noises[decade]), noises[decade], 0.0)
            with np.errstate(invalid="ignore"):
                agrees = (
                    np.abs(six - best_six) <= 3.0 * (error + best_error) + _NOISE_SLACK * (noise + best_noise) + 1e-300
                )
                take = new | (acceptable & anchored & agrees & (error < best_error))
            best_six = np.where(take, six, best_six)
            best_error = np.where(take, error, best_error)
            best_step = np.where(take, base * 10.0**decade, best_step)
            best_noise = np.where(take, noise, best_noise)
        best_error = np.where(anchored & ~np.isfinite(best_error), 0.0, best_error)  # the identically zero elements
        determined = anchored.copy()
        if not np.all(anchored):
            # No step determines these elements by itself. They get the step with the smallest estimate among those
            # up to the first one, and then a larger step wherever it gives the same value within the estimates and
            # a smaller estimate (the rule of the chain above): a larger step that averages over the structure of
            # the function disagrees and is not taken. Whether the estimate that remains is good enough is decided
            # for the Jacobian as a whole (an element that vanishes is known well enough if its error is small
            # against the other elements). An element without any finite estimate keeps an infinite one.
            for decade in range(-_DOWN, _UP + 1):
                six, error, quotient, constant = results[decade]
                noise = np.where(np.isfinite(noises[decade]), noises[decade], 0.0)
                with np.errstate(invalid="ignore"):
                    usable = ~anchored & np.isfinite(six) & np.isfinite(error) & (error < best_error)
                    if decade > 0:
                        usable &= np.abs(six - best_six) <= 3.0 * (error + best_error) + _NOISE_SLACK * (
                            noise + best_noise
                        )
                best_six = np.where(usable, six, best_six)
                best_error = np.where(usable, error, best_error)
                best_step = np.where(usable, base * 10.0**decade, best_step)
                best_noise = np.where(usable, noise, best_noise)
    # An element confirmed as independent of the coordinate (equal, within a few rounding errors, at the stencil and
    # far away) has the derivative zero. The quotient of its rounding residues by the step is not returned: at the
    # smallest steps it is large.
    # Its estimate is what the confirmation leaves open: a change of f by the difference seen, or by one rounding
    # error, over the distance at which it was confirmed. (It was zero, and a term of a Lagrangian that varies on
    # a scale of 1e20 at a coordinate that is zero was returned as absent, without any error.)
    independent = results[0][3]
    with np.errstate(all="ignore"):
        open_error = np.maximum(gap, _EPS * np.abs(center)) / reach
    best_six = np.where(independent, 0.0, best_six)
    best_error = np.where(independent, open_error, best_error)
    best_step = np.where(independent, base, best_step)
    if noise_out is not None:
        # What the test of a complex-step derivative may attribute to the reference, in two parts. The rounding of
        # the function values and of x + h at the step that was chosen. And, where the step came out of a search,
        # the truncation of the six-point formula there, taken as its difference from the next smaller step that
        # was tried (a first step that was accepted at once has none worth counting: it was accepted because its
        # estimate was within ten times its rounding). Not the noise inferred from the sixth difference, nor the
        # roughness term: across a kink both are as large as the kink, and the test would pass whatever the complex
        # step gives.
        decades = sorted(extras)
        with np.errstate(all="ignore"):
            rounding = np.zeros(np.shape(best_six))
            truncation = np.zeros(np.shape(best_six))
            for position, decade in enumerate(decades):
                chosen = best_step == base * 10.0**decade
                h = base * 10.0**decade
                own = (1.1 * 4.0 * _EPS * extras[decade][7] + 4.0 * _EPS * abs(float(x[i])) * np.abs(best_six)) / h
                rounding = np.where(chosen, own, rounding)
                if position > 0 and not settled:
                    six_below = results[decades[position - 1]][0]
                    usable = np.isfinite(six_below) & ~extras[decades[position - 1]][8]
                    truncation = np.where(chosen & usable, np.abs(best_six - six_below), truncation)
        noise_out.append((np.where(independent, open_error, rounding), np.where(independent, 0.0, truncation)))
    if child.steps:  # finite differences were taken inside f
        best_error = best_error * _NESTED_SAFETY
    if parent is not None:
        parent.steps.append((key, best_step, child))
        with np.errstate(all="ignore"):
            relative = np.where(np.abs(best_six) > 0.0, best_error / np.abs(best_six), 0.0)
        relative = relative[np.isfinite(relative)]
        if relative.size:
            parent.inner_error = max(parent.inner_error, min(1.0, float(np.max(relative))))
    if record is not None:
        unit = max(1.0, abs(float(x[i])))  # steps and distances are stored in units of max(1, |x_i|)
        record.append((best_step / unit, np.broadcast_to(independent, np.shape(best_six)).copy(), reach / unit))
    return best_six, best_error, determined


def _planned(f: Callable[[Array], ArrayLike], x: Array, plan: dict) -> tuple[Array, Array] | None:
    """Derivative with the steps of an earlier call at a nearby point; None if they no longer serve.

    Each step is verified where it is used: the six-point and four-point formulas must agree within the limit of the
    module docstring, or, for an element that vanishes, within that limit of the largest element of the Jacobian.
    Every element is evaluated again: one that did not depend on a coordinate at the earlier point (a momentum that
    vanished with the velocity) may depend on it now.
    """
    columns, errors, slack, factors = [], [], [], []
    for i, (ratios, independent, reach) in enumerate(plan["steps"]):
        scale = max(1.0, abs(float(x[i])))
        six_all = np.zeros(ratios.shape)
        error_all = np.zeros(ratios.shape)
        slack_all = np.zeros(ratios.shape)
        nested = 1.0
        for ratio in np.unique(ratios):
            h = float(ratio) * scale
            trace = _Trace()
            points = _differences(f, x, i, h, trace, ratios.shape)
            fm3, fm2, fm1, fp1, fp2, fp3 = points
            if trace.steps:
                nested = _NESTED_SAFETY
            with np.errstate(all="ignore"):
                six = _six(points, h)
                four = ((fm2 - fp2) + 8.0 * (fp1 - fm1)) / (12.0 * h)
                values = np.maximum.reduce([np.abs(v) for v in points])
                error = (
                    1.1 * 4.0 * _EPS * values / h + 4.0 * _EPS * abs(float(x[i])) / h * np.abs(six) + np.abs(six - four)
                )
                spread = np.maximum.reduce(points) - np.minimum.reduce(points)
                own = _ACCEPT_REL * np.abs(six) + _ACCEPT_ABS * spread / h
                # an element that was independent of the coordinate at the earlier point and whose six values
                # are equal here is independent still, with the estimate of that confirmation
                still = independent & (spread == 0.0)
                error = np.where(still, _EPS * values / (reach * scale), error)
            use = ratios == ratio
            six_all = np.where(use, six, six_all)
            error_all = np.where(use, error, error_all)
            slack_all = np.where(use, own, slack_all)
        columns.append(six_all)
        errors.append(error_all)
        slack.append(slack_all)
        factors.append(nested)
    result = np.stack(columns, axis=-1)
    error = np.stack(errors, axis=-1)
    allowed = np.stack(slack, axis=-1)
    finite = np.abs(result[np.isfinite(result)])
    largest = float(np.max(finite)) if finite.size else 0.0
    with np.errstate(invalid="ignore"):
        good = error <= np.maximum(allowed, _ACCEPT_ABS * largest)
    if not np.all(good):
        return None
    # the steps are judged as they were when they were chosen; the estimate of a nested difference is not
    return result, error * np.array(factors)


def _finite_jacobian(
    f: Callable[[Array], ArrayLike],
    x: Array,
    plan: dict | None = None,
    strict: bool = False,
    loose: bool = False,
    noise_out: list | None = None,
) -> tuple[Array, Array]:
    """Finite-difference Jacobian with error estimates.

    ``plan`` is a dictionary that the caller keeps between calls at successive nearby points (the right-hand sides of
    an integration): the first call stores the steps it chose, later calls reuse them as long as each one still
    passes its check where it is used, and search afresh otherwise.
    """
    if plan is not None and "steps" in plan:
        planned = _planned(f, x, plan)
        if planned is not None:
            return planned
    record: list | None = [] if plan is not None else None
    columns = [_finite_column(f, x, i, record, loose, noise_out) for i in range(x.size)]
    result = np.stack([c[0] for c in columns], axis=-1)
    error = np.stack([c[1] for c in columns], axis=-1)
    determined = np.stack([c[2] for c in columns], axis=-1)
    if loose and not np.all(determined):
        index = np.argwhere(~determined)[0]
        raise ValueError(
            f"finite-difference derivative along coordinate {int(index[-1])} could not be determined: no step "
            "between 1e-15 and 1e6 times max(1, |x_i|) gives an error estimate below the limit"
        )
    if strict and not np.all(determined):
        # ``strict`` is for the callers that return the derivative without its estimate. An element that no step
        # determines by itself (its error is not small against its own value, nor against the variation of the
        # function over the stencil) is accepted if the error is small against the largest determined element: the
        # vanishing element of a Jacobian whose other elements are known. Where nothing is determined there is no
        # scale to accept it against (the size of f is none: L + 1e20 has the derivatives of L), and it is refused.
        size = float(np.max(np.abs(result[determined]))) if np.any(determined) else 0.0
        with np.errstate(all="ignore"):
            bad = ~determined & ~(error <= _ACCEPT_ABS * size)
        if np.any(bad):
            index = np.argwhere(bad)[0]
            raise ValueError(
                f"finite-difference derivative along coordinate {int(index[-1])} could not be determined: the best "
                f"step leaves an error estimate of {float(error[tuple(index)]):.1e} for a value of "
                f"{float(result[tuple(index)]):.1e}. The function is not smooth there, is too noisy (a large constant "
                "part), or varies on a scale that no step between 1e-15 and 1e6 times max(1, |x_i|) resolves"
            )
    if plan is not None:
        plan["steps"] = record
    return result, error


def _complex_jacobian(f: Callable[[Array], ArrayLike], x: Array) -> Array:
    # The step is 1e-30 of the coordinate: the error of the complex step is of relative order (h / L)^2 for a
    # function that varies on the scale L, and L can be as small as the coordinate itself (q^4 at q = 1e-30, where
    # an absolute step of 1e-30 returned zero). A coordinate that is exactly zero takes the scale of the smallest
    # non-zero one below 1, or 1. Below 1e-280 the imaginary part would underflow.
    size = np.abs(x)
    nonzero = size[size > 0.0]
    fallback = min(1.0, float(np.min(nonzero))) if nonzero.size else 1.0
    columns = []
    for i in range(x.size):
        h = max(_COMPLEX_STEP * (float(size[i]) if size[i] > 0.0 else fallback), 1e-280)
        z = x.astype(complex)
        z[i] += 1j * h
        columns.append(np.imag(np.asarray(f(z), dtype=complex)) / h)
    return np.stack(columns, axis=-1)


def _kink_at(f: Callable[[Array], ArrayLike], x: Array, result: Array) -> bool:
    """Whether f has a kink at x or so near that no step separates the two.

    The central difference of a function with a kink at the point gives the mean of the two slopes, and so does the
    complex step of abs: the comparison of the two sees nothing. The second difference does. For a smooth function
    Q(h) = [f(x + h) - 2 f(x) + f(x - h)] / h is h f'' + O(h^3) and falls by ten (or more) from one decade of h
    to the next; across a kink it is the jump of the slope and stays. Q is taken on the decades from the first
    step down while it stands above twenty rounding errors, and the two smallest such steps decide: a kink is
    reported if Q has not fallen below half between them. A jump of the slope below about 1e-6 sqrt(|f| |f''|) is
    hidden by the curvature at every step and is not found.
    """
    center = _call(f, x, _Trace())
    shape = np.shape(center)
    for i in range(x.size):
        base = _FIRST_STEP * max(1.0, abs(float(x[i])))
        slope = np.abs(result[..., i])
        previous = None  # (decade, |Q|, above the noise)
        kink = np.zeros(shape, dtype=bool)
        decided = np.zeros(shape, dtype=bool)
        for decade in range(-_DOWN, 1):  # from the smallest step up: the first pair above the noise decides
            h = base * 10.0**decade
            e = np.zeros_like(x)
            e[i] = h
            plus, minus = _call(f, x + e, _Trace(), shape), _call(f, x - e, _Trace(), shape)
            with np.errstate(all="ignore"):
                quotient = np.abs(plus - 2.0 * center + minus) / h
                values = np.maximum.reduce([np.abs(plus), np.abs(minus), np.abs(center)])
                noise = 20.0 * (4.0 * 1.1 * _EPS * values + 4.0 * _EPS * abs(float(x[i])) * slope) / h
                above = np.isfinite(quotient) & (quotient > noise)
            if previous is not None:
                both = above & previous[1] & ~decided
                with np.errstate(invalid="ignore"):
                    kink |= both & (previous[0] >= 0.5 * quotient)
                decided |= both
            previous = (quotient, above)
            if np.all(decided):
                break
        if np.any(kink):
            return True
    return False


def _check_analytic(f: Callable[[Array], ArrayLike], x: Array, result: Array) -> None:
    """Raise when the finite-difference derivative contradicts the complex-step derivative ``result``.

    The two may differ by twenty error estimates of the finite difference (plus 1e-9 of the value). What
    this cannot detect is therefore a non-analytic part whose derivative is below that estimate: about
    2e-11 |f| / max(1, |x_i|) for a function evaluated to rounding, more for a noisy one or close to
    the edge of its domain, where only a very small step stays inside.
    """
    try:
        # ``loose``: the reference is judged against the size of f, as a test of the complex step needs (a
        # derivative that is small against |f| / h cannot be contradicted, and need not be: the complex step does
        # not suffer from the size of f). A reference that no step determines even so means that the test cannot
        # be made, and that is an error.
        noise_parts: list = []
        reference, error = _finite_jacobian(f, x, loose=True, noise_out=noise_parts)
    except ValueError as exc:
        raise ValueError(
            "complex-step derivative could not be checked against finite differences "
            f"({exc}); if the function is analytic, rescale its argument, otherwise it has no derivative there"
        ) from exc
    # The estimate is zero where the function is, for real arithmetic, independent of a coordinate; a dependence
    # below the spacing of the function values is then invisible to finite differences and proves nothing
    # against the complex-step result. The same noise floor as for every other element is used there.
    with np.errstate(all="ignore"):
        value = np.abs(np.asarray(f(x), dtype=float))
    floor = 1.1 * 4.0 * _EPS * value[..., None] / (_FIRST_STEP * _scale(x))
    # The two may differ by the noise of the reference, not by its whole estimate: a kink inside the stencil
    # leaves the value of the reference near the mean slope and puts its own size into the roughness and
    # truncation terms of the estimate, and with those in the tolerance the test passed whatever the complex
    # step gave (L = v^2/2 - 1e-8 |q - q0| at q - q0 = 1e-3: acceleration 0, no error). For an analytic function
    # those terms are far below the noise at the step that the search settles on.
    noise = np.stack([part[0] for part in noise_parts], axis=-1)
    truncation = np.stack([part[1] for part in noise_parts], axis=-1)
    allowed = _CHECK_FACTOR * np.maximum(noise, floor) + truncation + 1e-9 * np.abs(result) + 1e-300
    disagree = ~(np.abs(result - reference) <= allowed)
    if not np.any(disagree) and _kink_at(f, x, result):
        raise ValueError(
            "the function has a kink at the point or very near it: it is not analytic in its argument (abs, norm, "
            "comparisons), and its derivative depends on the side; use method='finite' away from the kink"
        )
    if np.any(disagree):
        raise ValueError(
            "complex-step derivative disagrees with the finite-difference derivative: the function is not analytic "
            "in its argument (abs, norm, np.real, comparisons or a cast to float); use method='finite'"
        )


def _first(
    f: Callable[[Array], ArrayLike], x: Array, method: Method, check: bool, strict: bool = False
) -> tuple[Array, Array]:
    if method == "finite":
        return _finite_jacobian(f, x, strict=strict)
    if method != "complex":
        raise ValueError("method must be 'complex' or 'finite'")
    result = _complex_jacobian(f, x)
    if check:
        _check_analytic(f, x, result)
    return result, 4.0 * _EPS * np.abs(result)


def jacobian_with_error(
    f: Callable[[Array], ArrayLike], x: ArrayLike, method: Method = "complex", check: bool = True, strict: bool = False
) -> tuple[Array, Array]:
    """Derivative of ``f`` with respect to the 1-D vector ``x`` and an estimate of its absolute error.

    ``check=False`` skips the comparison of the complex-step result with finite differences. It is meant for
    the inside of an integration whose first and last points are checked: the comparison costs seven real
    evaluations per coordinate, the complex step one.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D vector")
    return _first(f, x, method, check=check, strict=strict)


def jacobian(f: Callable[[Array], ArrayLike], x: ArrayLike, method: Method = "complex", check: bool = True) -> Array:
    """Derivative of ``f`` with respect to the 1-D vector ``x``.

    Returns an array of shape ``f(x).shape + (x.size,)``: the last axis runs over the
    components of ``x``. A scalar ``f`` therefore gives the gradient.

    Raises
    ------
    ValueError
        With ``method="complex"`` when the complex-step result is contradicted by finite
        differences, which means that ``f`` is not analytic in its argument (or ignores the
        imaginary part); use ``method="finite"`` for such a function. With ``method="finite"``
        when no step gives an error estimate below the limit of the module docstring.
    """
    return jacobian_with_error(f, x, method, check, strict=True)[0]


def gradient(f: Callable[[Array], ArrayLike], x: ArrayLike, method: Method = "complex") -> Array:
    """Gradient of a scalar function; shape ``(x.size,)``."""
    result = jacobian(f, x, method)
    if result.ndim != 1:
        raise ValueError("gradient needs a scalar function")
    return result


def _checked_at(f: Callable[[Array], ArrayLike], x: Array, method: Method) -> None:
    if _first(f, x, method, check=True)[0].ndim != 1:
        raise ValueError("a scalar function is needed")


def hessian_with_error(
    f: Callable[[Array], ArrayLike], x: ArrayLike, method: Method = "complex", strict: bool = False
) -> tuple[Array, Array]:
    """Symmetrized matrix of second derivatives of a scalar function and an estimate of its absolute error.

    Finite difference of the gradient. The analyticity check of the complex-step gradient is made once, at ``x``.
    """
    x = np.asarray(x, dtype=float)
    _checked_at(f, x, method)
    result, error = _finite_jacobian(lambda y: _first(f, y, method, check=False)[0], x, strict=strict)
    return 0.5 * (result + result.T), 0.5 * (error + error.T)


def hessian(f: Callable[[Array], ArrayLike], x: ArrayLike, method: Method = "complex") -> Array:
    """Matrix of second derivatives of a scalar function, symmetrized; shape ``(x.size, x.size)``."""
    return hessian_with_error(f, x, method, strict=True)[0]


def mixed_second_with_error(
    f: Callable[[Array, Array], ArrayLike],
    x: ArrayLike,
    y: ArrayLike,
    method: Method = "complex",
    strict: bool = False,
) -> tuple[Array, Array]:
    """Mixed derivatives ∂²f/∂x_i∂y_j of a scalar ``f(x, y)`` and an estimate of their absolute error."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    _checked_at(lambda u: f(u, y), x, method)
    return _finite_jacobian(lambda v: _first(lambda u: f(u, v), x, method, check=False)[0], y, strict=strict)


def mixed_second(
    f: Callable[[Array, Array], ArrayLike], x: ArrayLike, y: ArrayLike, method: Method = "complex"
) -> Array:
    """Mixed derivatives ∂²f/∂x_i∂y_j of a scalar ``f(x, y)``; shape ``(x.size, y.size)``."""
    return mixed_second_with_error(f, x, y, method, strict=True)[0]
