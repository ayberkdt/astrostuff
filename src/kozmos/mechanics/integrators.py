"""Symplectic time integration of separable Hamiltonian systems (leapfrog and a fourth-order composition).

The equations were read from the TeX sources of Dehnen and Read [DehnenRead2011] (arXiv:1105.1082,
``nbody.tex``, section "The Leapfrog integrator", TeX lines 358-480) and of Chin and Chen [Chin2003]
(arXiv:astro-ph/0304223, ``chin.tex``, TeX lines 105-172). Equations are cited by the TeX labels used there,
with the TeX line number of the label.

Conventions
-----------
The system is separable, H = T(p) + V(q); the accelerations depend on the positions (and possibly on the time
explicitly) but not on the velocities: ``acceleration(q, t) -> array`` with the same shape as ``q``. For the
second-order and fourth-order schemes here the "momentum" is the velocity v (unit mass, or per unit mass); a
system with masses is handled by passing the accelerations, which already contain the masses.
The integrators use a fixed step ``dt`` (a negative ``dt`` integrates backwards). The drivers return the times
``t0 + k dt`` and the positions and velocities at those times: positions and velocities are synchronised at every
recorded time (the intermediate half-step positions of the drift-kick-drift form and the intermediate
velocities of the kick-drift-kick form are auxiliary and are not recorded).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source

Array = NDArray[np.float64]
Acceleration = Callable[[Array, float], Array]

_S = 2.0 ** (1.0 / 3.0)
# weights of the three-fold composition of the second-order step [Chin2003, TeX label fr]:
# T4(eps) = T2(eps/(2-s)) T2(-s eps/(2-s)) T2(eps/(2-s)), s = 2^(1/3)
_W_OUTER = 1.0 / (2.0 - _S)
_W_MIDDLE = -_S / (2.0 - _S)

SCHEMES = ("kdk", "dkd", "forest_ruth")


def _as_state(q: ArrayLike, v: ArrayLike) -> tuple[Array, Array]:
    qa = np.array(q, dtype=float)
    va = np.array(v, dtype=float)
    if qa.shape != va.shape:
        raise ValueError(f"positions and velocities must have the same shape, got {qa.shape} and {va.shape}")
    return qa, va


def _acc(acceleration: Acceleration, q: Array, t: float) -> Array:
    a = np.asarray(acceleration(q, t), dtype=float)
    if a.shape != q.shape:
        raise ValueError(f"acceleration returned shape {a.shape}, expected {q.shape}")
    return a


def _kdk(q: Array, v: Array, t: float, dt: float, acceleration: Acceleration, a0: Array | None):
    """One kick-drift-kick step; returns (q1, v1, a1) with a1 the acceleration at the new position."""
    if a0 is None:
        a0 = _acc(acceleration, q, t)
    v_half = v + 0.5 * dt * a0
    q1 = q + dt * v_half
    a1 = _acc(acceleration, q1, t + dt)
    v1 = v_half + 0.5 * dt * a1
    return q1, v1, a1


def _dkd(q: Array, v: Array, t: float, dt: float, acceleration: Acceleration):
    """One drift-kick-drift step (the force is evaluated at the half-step position and time)."""
    q_half = q + 0.5 * dt * v
    a = _acc(acceleration, q_half, t + 0.5 * dt)
    v1 = v + dt * a
    q1 = q_half + 0.5 * dt * v1
    return q1, v1


def _forest_ruth(q: Array, v: Array, t: float, dt: float, acceleration: Acceleration):
    """Three drift-kick-drift substeps with the weights of the fourth-order composition."""
    for weight in (_W_OUTER, _W_MIDDLE, _W_OUTER):
        h = weight * dt
        q, v = _dkd(q, v, t, h, acceleration)
        t = t + h
    return q, v


@source(
    "DehnenRead2011",
    where="TeX label eq:leapfrog (line 437) and eqn:kdk:a (line 445): kick, drift, kick; eqn:kdk:b (line 471)",
    status="read",
    note="the source prints the kick-drift-kick sequence for a time-independent acceleration; the evaluation of "
    "a at t + dt for the second kick (explicit time dependence) is this project's extension",
)
def leapfrog_kdk_step(
    q: ArrayLike, v: ArrayLike, t: float, dt: float, acceleration: Acceleration
) -> tuple[Array, Array]:
    """Advance one kick-drift-kick leapfrog step: v' = v + a(q) dt/2, q1 = q + v' dt, v1 = v' + a(q1) dt/2.

    Parameters
    ----------
    q : array_like
        Positions at time t (any shape).
    v : array_like
        Velocities at time t, same shape as ``q``.
    t : float
        Time at the start of the step (passed to ``acceleration``).
    dt : float
        Step; may be negative (backward integration).
    acceleration : callable
        ``acceleration(q, t)`` returning an array of the shape of ``q``; independent of the velocity.

    Returns
    -------
    tuple of ndarray
        (q1, v1) at time t + dt, synchronised.

    References
    ----------
    [DehnenRead2011] eq:leapfrog, eqn:kdk:a, eqn:kdk:b.

    Validity
    --------
    Separable systems whose force does not depend on the velocity. Second-order accurate, symplectic and time
    symmetric for fixed dt; with a time-dependent force it is no longer a symplectic map of (q, v) alone.
    One force evaluation at the end of the step plus one at the start (two per call; the driver ``integrate``
    reuses the end value of the previous step).

    Verification
    ------------
    tests/unit/test_integrators.py: error ratio 4 when dt is halved (harmonic oscillator against the exact
    solution), reversibility, Jacobian determinant of the one-step map equal to 1 (pendulum), exact angular
    momentum for a central force.
    """
    qa, va = _as_state(q, v)
    q1, v1, _ = _kdk(qa, va, float(t), float(dt), acceleration, None)
    return q1, v1


@source(
    "DehnenRead2011",
    where="TeX label eqn:opsplit (line 393), eqn:kick:drift (line 406); footnote after eqn:kdk:a (line 462) names "
    "the drift-kick-drift version",
    status="read",
    note="the source names the DKD form but does not print its formulas; the drift e^{dt T/2} e^{dt V} e^{dt T/2} "
    "is the same composition printed in Chin2003 TeX label second (line 160)",
)
def leapfrog_dkd_step(
    q: ArrayLike, v: ArrayLike, t: float, dt: float, acceleration: Acceleration
) -> tuple[Array, Array]:
    """Advance one drift-kick-drift leapfrog step: q' = q + v dt/2, v1 = v + a(q') dt, q1 = q' + v1 dt/2.

    Parameters
    ----------
    q : array_like
        Positions at time t (any shape).
    v : array_like
        Velocities at time t, same shape as ``q``.
    t : float
        Time at the start of the step.
    dt : float
        Step; may be negative.
    acceleration : callable
        ``acceleration(q, t)`` returning an array of the shape of ``q``; it is called once, at the half-step
        position and the half-step time t + dt/2.

    Returns
    -------
    tuple of ndarray
        (q1, v1) at time t + dt, synchronised.

    References
    ----------
    [DehnenRead2011] eqn:opsplit, eqn:kick:drift; [Chin2003] label second.

    Validity
    --------
    As ``leapfrog_kdk_step``. The force is evaluated at an auxiliary half-step position, which is why the
    source prefers the kick-drift-kick form for step-size control. Both forms are second order; this is measured
    in the tests, not taken from the source.

    Verification
    ------------
    tests/unit/test_integrators.py: error ratio 4 when dt is halved, reversibility, Jacobian determinant 1.
    """
    qa, va = _as_state(q, v)
    q1, v1 = _dkd(qa, va, float(t), float(dt), acceleration)
    return q1, v1


@source(
    "Chin2003",
    where="TeX label fr (line 167): T4(eps) = T2(w1 eps) T2(w0 eps) T2(w1 eps), s = 2^(1/3), w1 = 1/(2 - s), "
    "w0 = -s/(2 - s); T2 of label second (line 160) is drift-kick-drift",
    status="read",
    note="the source states the composition and the weights in words and symbols (eps-tilde = eps/(2 - s), middle "
    "step -s eps-tilde); w0 = -s w1 is read from 'T2(-s eps-tilde)'; Forest-Ruth / Yoshida triple jump",
)
def forest_ruth_step(
    q: ArrayLike, v: ArrayLike, t: float, dt: float, acceleration: Acceleration
) -> tuple[Array, Array]:
    """Advance one fourth-order Forest-Ruth step: three drift-kick-drift steps of sizes w1 dt, w0 dt, w1 dt.

    w1 = 1 / (2 - 2^(1/3)) = 1.3512..., w0 = -2^(1/3) / (2 - 2^(1/3)) = -1.7024...

    Parameters
    ----------
    q : array_like
        Positions at time t (any shape).
    v : array_like
        Velocities at time t, same shape as ``q``.
    t : float
        Time at the start of the step.
    dt : float
        Step; may be negative.
    acceleration : callable
        ``acceleration(q, t)``; called three times per step.

    Returns
    -------
    tuple of ndarray
        (q1, v1) at time t + dt.

    References
    ----------
    [Chin2003] label fr, label second.

    Validity
    --------
    Fourth-order accurate, symplectic and time symmetric for a time-independent force. The middle sub-step is
    negative (backward in time), which the source states is unavoidable for a product of this kind beyond second
    order; this makes it unsuitable for variable-step schemes and for irreversible (diffusive) operators.

    Verification
    ------------
    tests/unit/test_integrators.py: error ratio 16 when dt is halved (harmonic oscillator), reversibility,
    energy error of a Kepler orbit, Jacobian determinant 1.
    """
    qa, va = _as_state(q, v)
    q1, v1 = _forest_ruth(qa, va, float(t), float(dt), acceleration)
    return q1, v1


@source(
    "DehnenRead2011",
    where="TeX label eqn:opsplit (line 393) and eqn:kick:drift (line 406): H = T(p) + V(x) and the two exact maps "
    "(drift, kick)",
    status="read",
    note="derivation in this project from eqn:kick:drift: for a general T(p) the exact flow of T is "
    "q -> q + dt dT/dp with p unchanged, and the exact flow of V is p -> p - dt dV/dq with q unchanged; the "
    "source prints these two maps for T = p^2/2 only (unit mass). The composition implemented here is "
    "drift-kick-drift, not the kick-drift-kick of eq:leapfrog (line 437); it is cited from Chin2003",
)
@source(
    "Chin2003",
    where="TeX label second (line 160): e^(eps T/2) e^(eps V) e^(eps T/2) with T = p d/dq the drift and "
    "V = F d/dp the kick (lines 124-125), i.e. drift-kick-drift; order 4: label fr (line 167)",
    status="read",
    note="order 4 uses the three-fold composition of label fr with the weights eps/(2 - s), -s eps/(2 - s), "
    "eps/(2 - s), s = 2^(1/3)",
)
def symplectic_step_separable(
    q: ArrayLike,
    p: ArrayLike,
    dt: float,
    dkinetic_dp: Callable[[Array], Array],
    dpotential_dq: Callable[[Array], Array],
    *,
    order: int = 2,
) -> tuple[Array, Array]:
    """Advance one symplectic step of H = T(p) + V(q) with arbitrary T and V (autonomous).

    Second order: q' = q + (dt/2) T'(p); p1 = p - dt V'(q'); q1 = q' + (dt/2) T'(p1).
    Fourth order: three such steps with the Forest-Ruth weights.

    Parameters
    ----------
    q : array_like
        Generalised coordinates.
    p : array_like
        Canonical momenta, same shape as ``q``.
    dt : float
        Step; may be negative.
    dkinetic_dp : callable
        p -> dT/dp (the velocity as a function of the momentum).
    dpotential_dq : callable
        q -> dV/dq.
    order : int
        2 or 4.

    Returns
    -------
    tuple of ndarray
        (q1, p1).

    References
    ----------
    [DehnenRead2011] eqn:opsplit, eqn:kick:drift (the two maps); [Chin2003] label second (the drift-kick-drift
    composition, line 160) and label fr (order 4). The kick-drift-kick form of [DehnenRead2011] eq:leapfrog is
    ``leapfrog_kdk_step``, not this function.

    Validity
    --------
    H must be separable and not depend explicitly on time. Each sub-map is the exact flow of T or of V, so the
    composition is symplectic for any T and V (relativistic kinetic energy included). For T = p^2/2 it equals
    ``leapfrog_dkd_step`` (order 2) and ``forest_ruth_step`` (order 4).

    Verification
    ------------
    tests/unit/test_integrators.py: agreement with leapfrog_dkd_step and forest_ruth_step; Jacobian determinant 1
    for the pendulum.
    """
    if order not in (2, 4):
        raise ValueError("order must be 2 or 4")
    qa, pa = _as_state(q, p)
    weights = (1.0,) if order == 2 else (_W_OUTER, _W_MIDDLE, _W_OUTER)
    for weight in weights:
        h = weight * float(dt)
        qa = qa + 0.5 * h * np.asarray(dkinetic_dp(pa), dtype=float)
        pa = pa - h * np.asarray(dpotential_dq(qa), dtype=float)
        qa = qa + 0.5 * h * np.asarray(dkinetic_dp(pa), dtype=float)
    return qa, pa


@source(
    "Chin2003",
    where="TeX label second (line 160) for scheme 'dkd' and label fr (line 167) for scheme 'forest_ruth'",
    status="read",
    note="driver loop of this project around the steps of the cited equations",
)
@source(
    "DehnenRead2011",
    where="TeX label eq:leapfrog (line 437), eqn:kdk:a (line 445) for scheme 'kdk'",
    status="read",
    note="driver loop of this project around the steps of the cited equations; for the kick-drift-kick form the "
    "acceleration at the end of one step is reused as the start acceleration of the next step",
)
def integrate(
    acceleration: Acceleration,
    q0: ArrayLike,
    v0: ArrayLike,
    dt: float,
    n_steps: int,
    *,
    t0: float = 0.0,
    scheme: str = "kdk",
    record_every: int = 1,
) -> tuple[Array, Array, Array]:
    """Integrate with a fixed step and return the synchronised trajectory.

    Parameters
    ----------
    acceleration : callable
        ``acceleration(q, t)`` returning an array of the shape of ``q``.
    q0 : array_like
        Initial positions.
    v0 : array_like
        Initial velocities, same shape as ``q0``.
    dt : float
        Step (nonzero; negative integrates backwards).
    n_steps : int
        Number of steps, >= 0.
    t0 : float
        Initial time.
    scheme : {"kdk", "dkd", "forest_ruth"}
        Second-order kick-drift-kick, second-order drift-kick-drift or the fourth-order composition.
    record_every : int
        Store the state every this many steps (the initial state is always stored; the last state is stored
        if ``n_steps`` is a multiple of ``record_every``).

    Returns
    -------
    tuple of ndarray
        (t, q, v): times t0 + k dt, positions and velocities at those times (first axis is time). Positions and
        velocities are both at the full-step times.

    References
    ----------
    [DehnenRead2011] eq:leapfrog, eqn:kdk:a; [Chin2003] label fr.

    Validity
    --------
    Memory is (n_steps / record_every + 1) times the size of the state. The step is fixed: the symplectic
    properties are lost if dt is adapted from the state. Times are computed as t0 + k dt, not by accumulation.

    Verification
    ------------
    tests/unit/test_integrators.py: convergence order, reversibility, Kepler orbit energy bound, angular momentum.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    if n_steps < 0 or int(n_steps) != n_steps:
        raise ValueError("n_steps must be a non-negative integer")
    if record_every < 1 or int(record_every) != record_every:
        raise ValueError("record_every must be a positive integer")
    dt = float(dt)
    if dt == 0.0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and nonzero")
    q, v = _as_state(q0, v0)
    t0 = float(t0)
    n_out = n_steps // record_every + 1
    times = np.empty(n_out)
    qs = np.empty((n_out, *q.shape))
    vs = np.empty((n_out, *q.shape))
    times[0], qs[0], vs[0] = t0, q, v
    a = _acc(acceleration, q, t0) if scheme == "kdk" else None
    k_out = 0
    for k in range(n_steps):
        t = t0 + k * dt
        if scheme == "kdk":
            q, v, a = _kdk(q, v, t, dt, acceleration, a)
        elif scheme == "dkd":
            q, v = _dkd(q, v, t, dt, acceleration)
        else:
            q, v = _forest_ruth(q, v, t, dt, acceleration)
        if (k + 1) % record_every == 0:
            k_out += 1
            times[k_out], qs[k_out], vs[k_out] = t0 + (k + 1) * dt, q, v
    return times, qs, vs


@source(
    "DehnenRead2011",
    where="caption of figure fig:force_schematic (TeX lines 544-547): 'Fractional change in energy for the Kepler "
    "problem'",
    status="read",
    note="the caption uses the words 'fractional change in energy' without a formula; (X_k - X_0)/X_0 is the plain "
    "definition of a fractional change and is this project's reading",
)
def relative_drift(values: ArrayLike) -> Array:
    """Fractional change of a conserved quantity along a trajectory: (X_k - X_0) / |X_0|.

    Parameters
    ----------
    values : array_like
        The conserved quantity (for example the total energy) at the recorded times, first axis is time; the
        first entry is the reference. Must be nonzero at k = 0.

    Returns
    -------
    ndarray
        (X_k - X_0) / |X_0| for every k; the first entry is 0. The sign is kept so that a secular drift can be
        told from a bounded oscillation.

    References
    ----------
    [DehnenRead2011] caption of fig:force_schematic.

    Validity
    --------
    Needs X_0 != 0. For a quantity that is zero in the initial state (for example a component of the total
    momentum) divide by a natural scale yourself.

    Verification
    ------------
    tests/unit/test_integrators.py: values of the harmonic oscillator energy; known result on a constant series.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim < 1 or x.shape[0] < 1:
        raise ValueError("values must have at least one entry")
    x0 = x[0]
    if np.any(x0 == 0.0):
        raise ValueError("the reference value (first entry) must be nonzero")
    return (x - x0) / np.abs(x0)
