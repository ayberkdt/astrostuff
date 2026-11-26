"""Numerical derivatives of user-supplied functions (private helper, no physics).

Two methods, both derived here from the Taylor series of ``f`` and tied to no publication.

``"complex"``
    f(x + i h) = f(x) + i h f'(x) − h² f''(x)/2 − ..., so Im f(x + i h) / h = f'(x) + O(h²). There
    is no subtraction, hence no cancellation, and h can be 1e-30: the result is exact to rounding
    whatever the scale of the function. It needs a function that is analytic and accepts complex
    arguments. ``abs``, ``norm``, comparisons and ``.real`` break this silently (the imaginary part
    is lost), so every complex-step derivative is compared with the finite-difference derivative
    and refused when the two differ by more than the error estimate of the latter.
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

        err ≤ 1e-6 |D6| + 1e-7 max|f| / h,

    the second term being the size of a difference quotient at that step (below the first step: the
    largest such size between that step and the first one), so that the rule does not depend on the
    units of f or of x. The first step is 1e-3 · max(1, |x_i|). If the estimate there
    is not already a hundred times below the limit, or if a step a hundred times smaller does not
    give the same derivative within the estimates, the steps from 1e-12 to 1e9 times the first one
    are all tried. For each element the smallest acceptable step is taken first (truncation is least
    there), and a larger step replaces the current choice only if it gives the same derivative within
    the two estimates and has the smaller estimate; this is how rounding noise is reduced for a function
    whose scale of variation is much larger than max(1, |x_i|), without trusting a large step that
    merely averages over structure. If no step is acceptable for some element, ValueError is raised.
    A kink at x itself is refused (the estimate never falls below the limit there). A function that
    does not depend on the coordinate (equal values on the first stencil and a million first steps
    away on both sides) gives exactly zero with a zero estimate.

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
_FAR = (1e6, 1e3, 1e1)  # in first steps: where a function must still be unchanged to count as independent
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


def _stencil(f: Callable[[Array], ArrayLike], x: Array, i: int, h: float, center: Array, trace: _Trace):
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
        # roughness: for a resolved function the fourth difference is (h / L)^2 times the second one and counts
        # for little; for a feature between the nodes, or for noise, the two are comparable and it counts in full
        rough = 0.5 * np.abs(fourth) / h * np.minimum(1.0, np.abs(fourth) / (np.abs(second) + 1e-300))
        sixth = (fm3 + fp3) - 6.0 * (fm2 + fp2) + 15.0 * (fm1 + fp1) - 20.0 * center
        # noise of one function value: at least a few rounding errors, and what the sixth difference shows (its
        # coefficients have a root sum of squares of 30.4); D6 amplifies it by 1.1 / h
        noise = np.maximum(4.0 * _EPS * values, np.abs(sixth) / 30.0)
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
    return six, error, values / h, constant, (tiny, noisy, bound, largest / h, inner, np.abs(sixth) / 30.0, noise_part)


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
    f: Callable[[Array], ArrayLike], x: Array, i: int, record: list | None = None
) -> tuple[Array, Array]:
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
            return result, np.zeros_like(result)
        parent = None  # the call sequence differs from the recorded one: choose the steps afresh

    child = _Trace()
    center = _call(f, x, child)
    child.recording = False
    child.center_error = child.inner_error
    base = _FIRST_STEP * max(1.0, abs(float(x[i])))
    good = _GOOD_REL / _ACCEPT_REL  # in units of the allowed error

    first = _stencil(f, x, i, base, center, child)
    six, error, quotient, constant, extra0 = first
    extras = {0: extra0}
    raw = constant.copy()
    if np.any(constant):
        # Equal values on the first stencil may hide a dependence that is below the spacing of the values (the
        # q in 1e16 + q). The function counts as independent of the coordinate only if it is also unchanged a
        # million first steps away, on both sides. If the function cannot be evaluated that far away (the edge of
        # its domain, an overflow), the largest of 1e3 and 10 first steps at which it can be is used instead.
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
            confirmed |= usable & (np.abs(plus - center) <= tolerance) & (np.abs(minus - center) <= tolerance)
            decided |= usable
            if np.all(decided | ~constant):
                break
        constant &= confirmed
        first = (six, error, quotient, constant)
    error = np.where(constant, 0.0, np.where(raw & ~constant, np.inf, error))
    first = (six, error, quotient, constant)
    results = {0: first}
    fine = _stencil(f, x, i, base * 10.0**_CONFIRM, center, child)
    extras[_CONFIRM] = fine[4]
    results[_CONFIRM] = (fine[0], np.where(fine[3] & ~constant, np.inf, fine[1]), fine[2], constant)
    six, error, quotient = _residues(results, extras, constant)[0][:3]
    # The error that inner differences pass on is the same at every step of this one: searching other steps cannot
    # reduce it, so it does not count in the decision to search (it stays in the estimate that is returned).
    with np.errstate(invalid="ignore"):
        own = np.where(np.isfinite(error), np.maximum(error - extras[0][4], 0.0), error)
    settled = not np.any(_ratio(six, own, quotient, constant, 0) > good)
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
    else:
        for decade in [*range(-_DOWN, 0), *range(1, _UP + 1)]:
            if decade in results:
                continue
            other = _stencil(f, x, i, base * 10.0**decade, center, child)
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
        for decade in range(1, _UP + 1):
            six, error, quotient, constant = results[decade]
            ratios[decade] = _ratio(six, error, quotient, constant, decade)
        # An element that is exactly zero at every point of every step inside the domain is identically zero in
        # the neighbourhood (a vanishing component of a tensor): its derivative is zero. It is anchored at once.
        quotients = np.stack([results[d][2] for d in results])
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
        best_error = np.where(np.isfinite(best_error), best_error, 0.0)  # the identically zero elements
        if not np.all(anchored):
            worst = min(float(np.max(r)) for r in ratios.values())
            raise ValueError(
                f"finite-difference derivative along coordinate {i} could not be determined: at the best step the "
                f"error estimate is {worst:.1e} times the allowed one. The function is not smooth there, is too "
                "noisy, or varies on a scale that no step between 1e-15 and 1e6 times max(1, |x_i|) resolves"
            )
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
        record.append(best_step / max(1.0, abs(float(x[i]))))  # the steps in units of max(1, |x_i|)
    return best_six, best_error


def _planned(f: Callable[[Array], ArrayLike], x: Array, plan: dict) -> tuple[Array, Array] | None:
    """Derivative with the steps of an earlier call at a nearby point; None if they no longer serve.

    Each step is verified where it is used: the six-point and four-point formulas must agree within the limit of the
    module docstring. Every element is evaluated again: one that did not depend on a coordinate at the earlier
    point (a momentum that vanished with the velocity) may depend on it now.
    """
    columns, errors = [], []
    for i, ratios in enumerate(plan["steps"]):
        scale = max(1.0, abs(float(x[i])))
        six_all = np.zeros(ratios.shape)
        error_all = np.zeros(ratios.shape)
        for ratio in np.unique(ratios):
            h = float(ratio) * scale
            trace = _Trace()
            points = _differences(f, x, i, h, trace, ratios.shape)
            fm3, fm2, fm1, fp1, fp2, fp3 = points
            nested = _NESTED_SAFETY if trace.steps else 1.0
            with np.errstate(all="ignore"):
                six = _six(points, h)
                four = ((fm2 - fp2) + 8.0 * (fp1 - fm1)) / (12.0 * h)
                values = np.maximum.reduce([np.abs(v) for v in points])
                error = (
                    1.1 * 4.0 * _EPS * values / h + 4.0 * _EPS * abs(float(x[i])) / h * np.abs(six) + np.abs(six - four)
                )
                good = error <= _ACCEPT_REL * np.abs(six) + _ACCEPT_ABS * values / h
                error = nested * error  # the step is judged as it was when it was chosen; the estimate is not
            use = ratios == ratio
            if not np.all(good[use]):
                return None
            six_all = np.where(use, six, six_all)
            error_all = np.where(use, error, error_all)
        columns.append(six_all)
        errors.append(error_all)
    return np.stack(columns, axis=-1), np.stack(errors, axis=-1)


def _finite_jacobian(f: Callable[[Array], ArrayLike], x: Array, plan: dict | None = None) -> tuple[Array, Array]:
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
    columns = [_finite_column(f, x, i, record) for i in range(x.size)]
    if plan is not None:
        plan["steps"] = record
    return np.stack([c[0] for c in columns], axis=-1), np.stack([c[1] for c in columns], axis=-1)


def _complex_jacobian(f: Callable[[Array], ArrayLike], x: Array) -> Array:
    columns = []
    for i in range(x.size):
        z = x.astype(complex)
        z[i] += 1j * _COMPLEX_STEP
        columns.append(np.imag(np.asarray(f(z), dtype=complex)) / _COMPLEX_STEP)
    return np.stack(columns, axis=-1)


def _check_analytic(f: Callable[[Array], ArrayLike], x: Array, result: Array) -> None:
    """Raise when the finite-difference derivative contradicts the complex-step derivative ``result``.

    The two may differ by twenty error estimates of the finite difference (plus 1e-9 of the value). What
    this cannot detect is therefore a non-analytic part whose derivative is below that estimate: about
    2e-11 |f| / max(1, |x_i|) for a function evaluated to rounding, more for a noisy one or close to
    the edge of its domain, where only a very small step stays inside.
    """
    try:
        reference, error = _finite_jacobian(f, x)
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
    allowed = _CHECK_FACTOR * np.maximum(error, floor) + 1e-9 * np.abs(result) + 1e-300
    if not np.all(np.abs(result - reference) <= allowed):
        raise ValueError(
            "complex-step derivative disagrees with the finite-difference derivative: the function is not analytic "
            "in its argument (abs, norm, np.real, comparisons or a cast to float); use method='finite'"
        )


def _first(f: Callable[[Array], ArrayLike], x: Array, method: Method, check: bool) -> tuple[Array, Array]:
    if method == "finite":
        return _finite_jacobian(f, x)
    if method != "complex":
        raise ValueError("method must be 'complex' or 'finite'")
    result = _complex_jacobian(f, x)
    if check:
        _check_analytic(f, x, result)
    return result, 4.0 * _EPS * np.abs(result)


def jacobian_with_error(
    f: Callable[[Array], ArrayLike], x: ArrayLike, method: Method = "complex", check: bool = True
) -> tuple[Array, Array]:
    """Derivative of ``f`` with respect to the 1-D vector ``x`` and an estimate of its absolute error.

    ``check=False`` skips the comparison of the complex-step result with finite differences. It is meant for
    the inside of an integration whose first and last points are checked: the comparison costs seven real
    evaluations per coordinate, the complex step one.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D vector")
    return _first(f, x, method, check=check)


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
    return jacobian_with_error(f, x, method, check)[0]


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
    f: Callable[[Array], ArrayLike], x: ArrayLike, method: Method = "complex"
) -> tuple[Array, Array]:
    """Symmetrized matrix of second derivatives of a scalar function and an estimate of its absolute error.

    Finite difference of the gradient. The analyticity check of the complex-step gradient is made once, at ``x``.
    """
    x = np.asarray(x, dtype=float)
    _checked_at(f, x, method)
    result, error = _finite_jacobian(lambda y: _first(f, y, method, check=False)[0], x)
    return 0.5 * (result + result.T), 0.5 * (error + error.T)


def hessian(f: Callable[[Array], ArrayLike], x: ArrayLike, method: Method = "complex") -> Array:
    """Matrix of second derivatives of a scalar function, symmetrized; shape ``(x.size, x.size)``."""
    return hessian_with_error(f, x, method)[0]


def mixed_second_with_error(
    f: Callable[[Array, Array], ArrayLike], x: ArrayLike, y: ArrayLike, method: Method = "complex"
) -> tuple[Array, Array]:
    """Mixed derivatives ∂²f/∂x_i∂y_j of a scalar ``f(x, y)`` and an estimate of their absolute error."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    _checked_at(lambda u: f(u, y), x, method)
    return _finite_jacobian(lambda v: _first(lambda u: f(u, v), x, method, check=False)[0], y)


def mixed_second(
    f: Callable[[Array, Array], ArrayLike], x: ArrayLike, y: ArrayLike, method: Method = "complex"
) -> Array:
    """Mixed derivatives ∂²f/∂x_i∂y_j of a scalar ``f(x, y)``; shape ``(x.size, y.size)``."""
    return mixed_second_with_error(f, x, y, method)[0]
