"""Noether's first theorem for point mechanics: symmetry variations, boundary terms, charges and defects.

Conventions are those of Banados and Reyes [BanadosReyes2016] (arXiv:1601.03616, TeX file AGR-9.tex, section
"Noether's theorem in particle mechanics"; line numbers refer to that file):

* A symmetry is a function delta_s q(t) = q'(t) - q(t), "the difference of the two functions evaluated at the same
  argument" (line 190), with delta_s (dq/dt) = d(delta_s q)/dt (line 193), such that for ANY q(t)
  delta I = integral dK/dt (Eq. (sym), line 139, and Eq. (sym3), line 226), i.e. delta_s L = dK/dt.
* The Noether charge is Q = K - dL/d(qdot^i) delta_s q^i (Eq. (qn), line 236), conserved on solutions of the
  Euler-Lagrange equations.

A ``generator(q, v, t) -> delta_s q`` returns the variation per unit of the infinitesimal parameter (epsilon = 1).
The sign convention matters: the source's time translation is delta_s q = -epsilon qdot (Eq. (dr), line 166),
its rotation is delta_s r = alpha x r (Eq. (rot), line 159), and with K = -epsilon L the charge is +epsilon E.
Note that the sign of the charge is Q = K - p.delta_s q, the opposite of p.delta_s q - K.

delta_s acts only on the dynamical variables q; any explicit dependence of L on t is a background and is not
varied (the source's discussion of background fields, lines 319-365). A generator that depends on the velocity
needs the acceleration to compute delta_s v = d(delta_s q)/dt; pass it as ``a``. The symmetry condition must hold
for arbitrary q(t), hence for arbitrary a: the defect is evaluated at one point (q, v, a, t) and is a number,
not a verdict.

Generalized coordinates for N particles in Cartesian components are flattened particle by particle:
q = (x_1, y_1, z_1, x_2, y_2, z_2, ...).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core._differentiation import Method, gradient, jacobian
from kozmos.mechanics.lagrangian import Lagrangian, _dl_dq, _dl_dv, _state, _vector

Array = NDArray[np.float64]
Generator = Callable[[Array, Array, float], ArrayLike]
BoundaryTerm = Callable[[Array, Array, float], ArrayLike]

# A generator or boundary term that does not depend on the velocity has a derivative with respect to it that is
# exactly zero with either method (the complex step leaves the imaginary part untouched; finite differences of
# equal values vanish). Any non-zero derivative therefore needs the acceleration. An absolute tolerance (it was
# 1e-12) drops the term of a generator such as -1e-13 s * v, whose coefficient is small only in its units.


def _variation_rate(generator: Generator, q: Array, v: Array, a: Array | None, t: float, method: Method) -> Array:
    """d(delta_s q)/dt = (d g/dq) v + (d g/dv) a + d g/dt  (chain rule; the source gives delta_s qdot = d(delta_s q)/dt)."""
    dg_dq = jacobian(lambda x: generator(x, v, t), q, method)
    dg_dv = jacobian(lambda x: generator(q, x, t), v, method)
    dg_dt = jacobian(lambda y: generator(q, v, y[0]), np.array([float(t)]), method)[:, 0]
    if a is None:
        if np.any(dg_dv != 0.0):
            raise ValueError(
                "the generator depends on the velocity: delta_s qdot needs the acceleration, pass it as a="
            )
        return dg_dq @ v + dg_dt
    return dg_dq @ v + dg_dv @ a + dg_dt


def _boundary_rate(
    boundary_term: BoundaryTerm | None, q: Array, v: Array, a: Array | None, t: float, method: Method
) -> float:
    """Total derivative dK/dt = (dK/dq).v + (dK/dv).a + dK/dt (explicit); 0 for no boundary term."""
    if boundary_term is None:
        return 0.0
    dk_dq = gradient(lambda x: boundary_term(x, v, t), q, method)
    dk_dv = gradient(lambda x: boundary_term(q, x, t), v, method)
    dk_dt = float(jacobian(lambda y: boundary_term(q, v, y[0]), np.array([float(t)]), method)[0])
    if a is None:
        if np.any(dk_dv != 0.0):
            raise ValueError("the boundary term depends on the velocity: dK/dt needs the acceleration, pass it as a=")
        return float(dk_dq @ v + dk_dt)
    return float(dk_dq @ v + dk_dv @ a + dk_dt)


def _acceleration(a: ArrayLike | None, n: int) -> Array | None:
    if a is None:
        return None
    a_arr = _vector(a, "a")
    if a_arr.size != n:
        raise ValueError("a must have the same length as q")
    return a_arr


@source(
    "BanadosReyes2016",
    where="Eq. (sym) (line 139) and Eq. (sym3) (line 226): delta I = integral dK/dt; delta_s L = dL/dq.delta_s q + "
    "dL/dv.d(delta_s q)/dt from the variation of Eq. (action0) and line 193",
    status="read",
    note="the explicit expression of delta_s L through the chain rule is derivation in this project",
)
def symmetry_variation(
    lagrangian: Lagrangian,
    q: ArrayLike,
    v: ArrayLike,
    t: float,
    generator: Generator,
    a: ArrayLike | None = None,
    method: Method = "complex",
) -> float:
    """Evaluate the variation of the Lagrangian under delta_s q = generator(q, v, t).

    delta_s L = dL/dq . delta_s q + dL/dv . d(delta_s q)/dt, with d(delta_s q)/dt = (dg/dq) v + (dg/dv) a + dg/dt.

    Parameters
    ----------
    lagrangian : callable
        ``lagrangian(q, v, t) -> float``.
    q, v : array_like
        Generalized coordinates and velocities, 1-D arrays of equal length.
    t : float
        Value of the independent variable.
    generator : callable
        ``generator(q, v, t) -> delta_s q`` (array of length n), the variation per unit epsilon.
    a : array_like, optional
        Accelerations. Required when the generator depends on the velocity (ValueError otherwise).
    method : {"complex", "finite"}, optional
        Differentiation method.

    Returns
    -------
    float
        delta_s L.

    References
    ----------
    [BanadosReyes2016] Eq. (sym), Eq. (sym3), line 193.

    Validity
    --------
    Only the dynamical variables are varied; explicit t dependence of L is a background and is not varied.
    A generator that depends on higher derivatives (qddot) is not supported. The generator must be analytic
    for ``method="complex"`` (avoid np.cross on non-complex-safe code; use explicit component arithmetic).

    Verification
    ------------
    tests/unit/test_noether.py: delta_s L = n.P = dK/dt for the Galilean boost; through symmetry_defect, the
    variation of the time translation, rotation, translation and Weyl generators at random (q, v, a, t).
    """
    q_arr, v_arr = _state(q, v)
    a_arr = _acceleration(a, q_arr.size)
    delta_q = np.asarray(generator(q_arr, v_arr, t), dtype=float)
    rate = _variation_rate(generator, q_arr, v_arr, a_arr, t, method)
    return float(
        _dl_dq(lagrangian, q_arr, v_arr, t, method) @ delta_q + _dl_dv(lagrangian, q_arr, v_arr, t, method) @ rate
    )


@source(
    "BanadosReyes2016",
    where="Eq. (sym) (line 139) and Eq. (sym3) (line 226): a symmetry has delta_s L = dK/dt for any q(t)",
    status="read",
    note="the defect delta_s L - dK/dt is this project's name for the difference of the two sides of Eq. (sym3), "
    "evaluated at one point; the source states only the condition that it vanishes",
)
def symmetry_defect(
    lagrangian: Lagrangian,
    q: ArrayLike,
    v: ArrayLike,
    t: float,
    generator: Generator,
    boundary_term: BoundaryTerm | None = None,
    a: ArrayLike | None = None,
    method: Method = "complex",
) -> float:
    """Evaluate the defect delta_s L - dK/dt of a trial symmetry at one point (q, v, a, t).

    It is zero at every (q, v, a, t) if and only if (generator, boundary_term) is a symmetry in the sense of the
    source. On a solution of the equations of motion, d/dt of the charge Q = K - p.delta_s q equals MINUS the defect
    (derivation in this project: dQ/dt = dK/dt - d/dt(p) . delta_s q - p . delta_s qdot, and with dp/dt = dL/dq
    this is dK/dt - delta_s L).

    Parameters
    ----------
    lagrangian : callable
        ``lagrangian(q, v, t) -> float``.
    q, v : array_like
        Generalized coordinates and velocities, 1-D arrays of equal length.
    t : float
        Value of the independent variable.
    generator : callable
        ``generator(q, v, t) -> delta_s q``.
    boundary_term : callable, optional
        ``K(q, v, t) -> float``; None means K = 0.
    a : array_like, optional
        Accelerations; required when the generator or K depends on the velocity.
    method : {"complex", "finite"}, optional
        Differentiation method.

    Returns
    -------
    float
        delta_s L - dK/dt.

    References
    ----------
    [BanadosReyes2016] Eq. (sym), Eq. (sym3).

    Validity
    --------
    A single point cannot prove a symmetry; the tests sample random (q, v, a, t). The value has the units of L per
    unit time. The size of rounding noise is about 1e-12 times the largest term at unit scale (complex step
    with a finite-difference outer derivative).

    Verification
    ------------
    tests/unit/test_noether.py: zero for rotation, translation, boost, time translation (equal to dL/dt for a
    driven oscillator) and Weyl scaling; non-zero and equal to B y for a rotation of a particle in a uniform force.
    """
    q_arr, v_arr = _state(q, v)
    a_arr = _acceleration(a, q_arr.size)
    variation = symmetry_variation(lagrangian, q_arr, v_arr, t, generator, a_arr, method)
    return variation - _boundary_rate(boundary_term, q_arr, v_arr, a_arr, t, method)


@source(
    "BanadosReyes2016",
    where="Eq. (qn) (line 236): Q = K - dL/d(qdot^i) delta_s q^i",
    status="read",
)
def noether_charge(
    lagrangian: Lagrangian,
    q: ArrayLike,
    v: ArrayLike,
    t: float,
    generator: Generator,
    boundary_term: BoundaryTerm | None = None,
    method: Method = "complex",
) -> float:
    """Evaluate the Noether charge Q = K - p . delta_s q at (q, v, t).

    Parameters
    ----------
    lagrangian : callable
        ``lagrangian(q, v, t) -> float``.
    q, v : array_like
        Generalized coordinates and velocities, 1-D arrays of equal length.
    t : float
        Value of the independent variable.
    generator : callable
        ``generator(q, v, t) -> delta_s q``, per unit epsilon.
    boundary_term : callable, optional
        ``K(q, v, t) -> float`` with delta_s L = dK/dt; None means K = 0.
    method : {"complex", "finite"}, optional
        Differentiation method (only p = dL/dv is differentiated).

    Returns
    -------
    float
        Q. Conserved on solutions when (generator, boundary_term) is a symmetry; otherwise
        dQ/dt = -(delta_s L - dK/dt), see ``symmetry_defect``.

    References
    ----------
    [BanadosReyes2016] Eq. (qn) (boxed); examples Eq. (angular) (line 247) and the energy at line 251.

    Validity
    --------
    The sign is the source's: for rotations Q = -alpha . (r x p), for the source's time translation Q = +E.
    The function does not verify the symmetry.

    Verification
    ------------
    tests/unit/test_noether.py: Q = -alpha . L for rotations (source Eq. (angular)), Q = E for the time translation,
    Q = -(Weyl charge of the source) for the conformal particle, conservation along integrated orbits.
    """
    q_arr, v_arr = _state(q, v)
    delta_q = np.asarray(generator(q_arr, v_arr, t), dtype=float)
    p = _dl_dv(lagrangian, q_arr, v_arr, t, method)
    k_value = 0.0 if boundary_term is None else float(np.real(boundary_term(q_arr, v_arr, t)))
    return k_value - float(p @ delta_q)


# --------------------------------------------------------------------------------------
# ready-made generators
# --------------------------------------------------------------------------------------


@source(
    "BanadosReyes2016",
    where="Eq. (dr) (line 166), Eq. (trans) (line 187), K = -eps L from Eq. (ttt) (line 173-175)",
    status="read",
)
def t_translation_generator(q: ArrayLike, v: ArrayLike, t: float) -> Array:
    """Generator of the translation in the independent variable: delta_s q = -qdot (per unit epsilon).

    Use it with ``t_translation_boundary_term`` (K = -L); then the Noether charge is the energy function h.

    Parameters
    ----------
    q, v : array_like
        Coordinates and velocities, 1-D arrays of equal length.
    t : float
        Value of the independent variable (unused; kept for the generator signature).

    Returns
    -------
    ndarray
        -v.

    References
    ----------
    [BanadosReyes2016] Eq. (dr), Eq. (trans).

    Validity
    --------
    Velocity dependent: ``symmetry_defect`` needs the acceleration. delta_s q = -epsilon qdot corresponds to
    q'(t + epsilon) = q(t) (line 184); the opposite sign would change the sign of the charge.

    Verification
    ------------
    tests/unit/test_noether.py: Q equals the energy function; defect equals dL/dt (zero for t-independent L).
    """
    return -np.asarray(v)


@source(
    "BanadosReyes2016",
    where="K = -eps L of Eq. (ttt) (line 173-175) and line 249 (K = -eps L)",
    status="read",
)
def t_translation_boundary_term(lagrangian: Lagrangian) -> BoundaryTerm:
    """Build the boundary term K = -L of the translation in the independent variable.

    Parameters
    ----------
    lagrangian : callable
        ``lagrangian(q, v, t) -> float``.

    Returns
    -------
    callable
        ``K(q, v, t) = -lagrangian(q, v, t)``.

    References
    ----------
    [BanadosReyes2016] Eq. (ttt): K = -eps (m v^2 / 2 - V) = -eps L; line 249: "K = -epsilon L".

    Validity
    --------
    With delta_s q = -qdot one finds delta_s L - dK/dt = dL/dt (partial), which vanishes only for a Lagrangian that
    has no explicit dependence on t.

    Verification
    ------------
    tests/unit/test_noether.py: defect equals the partial derivative of L with respect to t.
    """

    def boundary_term(q: Array, v: Array, t: float) -> ArrayLike:
        return -lagrangian(q, v, t)

    return boundary_term


@source(
    "BanadosReyes2016",
    where="Eq. (sss) (line 127): translations of x and y are a symmetry; Eq. (Ip2) (line 374): joint translation "
    "of two particles",
    status="read",
    note="the generator delta_s r_i = a (the same constant vector for every particle) is the infinitesimal form of "
    "r -> r + a; K = 0 because L depends on r only through differences (derivation in this project)",
)
def translation_generator(direction: ArrayLike) -> Generator:
    """Build the generator of a rigid translation of N particles along ``direction``: delta_s r_i = direction.

    Parameters
    ----------
    direction : array_like
        Displacement vector of length d (2 or 3 for Cartesian components); q must have length N d.

    Returns
    -------
    callable
        ``generator(q, v, t)`` returning the vector direction repeated for every particle. K = 0.

    References
    ----------
    [BanadosReyes2016] Eq. (sss) (line 127), Eq. (qn); conserved momentum of the pair, lines 374-381.

    Validity
    --------
    A symmetry only if the potential depends on the particle positions through differences (a fixed external centre
    breaks it, see the source's Eq. (Ip), lines 367-370). The Noether charge is -direction . P.

    Verification
    ------------
    tests/unit/test_noether.py: charge -n . P for two bodies; defect zero for the two-body Kepler potential and
    non-zero for a fixed centre.
    """
    d_vec = _vector(direction, "direction")

    def generator(q: Array, v: Array, t: float) -> Array:
        n_particles, remainder = divmod(np.asarray(q).size, d_vec.size)
        if remainder != 0:
            raise ValueError("the length of q must be a multiple of the length of direction")
        return np.tile(d_vec, n_particles)

    return generator


@source(
    "BanadosReyes2016",
    where="Eq. (rot) (line 159): delta_s r = alpha x r; Q = -alpha . (m r x rdot) Eq. (angular) (line 247)",
    status="read",
)
def rotation_generator(axis: ArrayLike) -> Generator:
    """Build the generator of an infinitesimal rotation of N particles in 3-D: delta_s r_i = axis x r_i.

    Parameters
    ----------
    axis : array_like
        Rotation vector alpha (direction = axis, magnitude = angle per unit epsilon), length 3. q has length 3 N.

    Returns
    -------
    callable
        ``generator(q, v, t)``. K = 0.

    References
    ----------
    [BanadosReyes2016] Eq. (rot), Eq. (angular).

    Validity
    --------
    A symmetry if the potential is invariant under a simultaneous rotation of all particles. The Noether charge is
    -alpha . L with L = sum_i r_i x p_i (the source's sign, line 245-247).

    Verification
    ------------
    tests/unit/test_noether.py: Q = -alpha . L against the explicit cross product; defect zero for a central
    force and for the pair potential; broken by a uniform force with defect -alpha . (r x B).
    """
    alpha = _vector(axis, "axis")
    if alpha.size != 3:
        raise ValueError("axis must have three components")

    def generator(q: Array, v: Array, t: float) -> Array:
        r = np.asarray(q).reshape(-1, 3)
        x, y, z = r[:, 0], r[:, 1], r[:, 2]
        return np.stack(
            [alpha[1] * z - alpha[2] * y, alpha[2] * x - alpha[0] * z, alpha[0] * y - alpha[1] * x], axis=1
        ).reshape(-1)

    return generator


@source(
    "BanadosReyes2016",
    where="Eq. (rot) (line 159), specialised to a rotation about the third axis in the plane",
    status="read",
    note="the planar case delta_s (x, y) = (-y, x) is alpha x r of Eq. (rot) with alpha = (0, 0, 1) (derivation in "
    "this project)",
)
def planar_rotation_generator() -> Generator:
    """Build the generator of a unit rotation of N particles in the plane: delta_s (x_i, y_i) = (-y_i, x_i).

    Returns
    -------
    callable
        ``generator(q, v, t)`` for q = (x_1, y_1, x_2, y_2, ...). K = 0.

    References
    ----------
    [BanadosReyes2016] Eq. (rot).

    Validity
    --------
    The Noether charge is -L_z = -sum (x p_y - y p_x). For generalized coordinates such as polar angles, write the
    generator directly (an angle shift delta_s phi = 1).

    Verification
    ------------
    tests/unit/test_noether.py: Q = -L_z for a planar two-body problem, conserved along an integrated orbit.
    """

    def generator(q: Array, v: Array, t: float) -> Array:
        r = np.asarray(q).reshape(-1, 2)
        return np.stack([-r[:, 1], r[:, 0]], axis=1).reshape(-1)

    return generator


@source(
    "BanadosReyes2016",
    where="Eq. (sym), Eq. (sym3) (lines 139, 226) and Eq. (qn) (line 236)",
    status="read",
    note="the Galilean boost is NOT worked out in the source; derivation in this project from Eq. (sym3): "
    "delta_s r_i = t n gives delta_s L = sum m_i n.v_i = dK/dt with K = sum m_i n.r_i, so Eq. (qn) gives "
    "Q = n . (sum m_i r_i - t sum m_i v_i)",
)
def boost_generator(direction: ArrayLike) -> Generator:
    """Build the generator of a Galilean boost of N particles: delta_s r_i = t n (n = direction).

    Use it with ``boost_boundary_term(masses, direction)`` (K != 0). The Noether charge is
    Q = n . (M R_cm - t P): its conservation is the uniform motion of the centre of mass.

    Parameters
    ----------
    direction : array_like
        Boost direction n (length d); q must have length N d.

    Returns
    -------
    callable
        ``generator(q, v, t) = t * (n, n, ..., n)``.

    References
    ----------
    [BanadosReyes2016] Eq. (sym3), Eq. (qn); the example is not in the source (see the note).

    Validity
    --------
    A symmetry for L = sum m_i v_i^2 / 2 - V(differences of positions). It acts on the velocities as
    delta_s v_i = n, so a Lagrangian with an interaction that depends on velocities needs a check with
    ``symmetry_defect``.

    Verification
    ------------
    tests/unit/test_noether.py: defect zero at random (q, v, a, t); charge n . (M R_cm - t P) against the explicit
    sum; conserved along an integrated two-body orbit.
    """
    d_vec = _vector(direction, "direction")

    def generator(q: Array, v: Array, t: float) -> Array:
        n_particles, remainder = divmod(np.asarray(q).size, d_vec.size)
        if remainder != 0:
            raise ValueError("the length of q must be a multiple of the length of direction")
        return t * np.tile(d_vec, n_particles)

    return generator


@source(
    "BanadosReyes2016",
    where="Eq. (sym3) (line 226); derivation in this project (see boost_generator)",
    status="read",
    note="derivation in this project from Eq. (sym3): K = sum_i m_i n . r_i",
)
def boost_boundary_term(masses: ArrayLike, direction: ArrayLike) -> BoundaryTerm:
    """Build the boundary term K = sum_i m_i n . r_i of the Galilean boost.

    Parameters
    ----------
    masses : array_like
        Particle masses, length N.
    direction : array_like
        Boost direction n (length d); q must have length N d.

    Returns
    -------
    callable
        ``K(q, v, t)``.

    References
    ----------
    [BanadosReyes2016] Eq. (sym3); see ``boost_generator`` for the derivation.

    Validity
    --------
    Same as ``boost_generator``.

    Verification
    ------------
    tests/unit/test_noether.py: defect of the boost is zero with this K and non-zero without it.
    """
    m = _vector(masses, "masses")
    d_vec = _vector(direction, "direction")

    def boundary_term(q: Array, v: Array, t: float) -> ArrayLike:
        r = np.asarray(q).reshape(m.size, d_vec.size)
        return np.sum(m[:, None] * r * d_vec[None, :])

    return boundary_term


@source(
    "BanadosReyes2016",
    where="line 292: delta_s x = -eps t xdot + (eps/2) x, for the action of Eq. (Iconformal) (line 264)",
    status="read",
)
def weyl_generator(q: ArrayLike, v: ArrayLike, t: float) -> Array:
    """Generator of the Weyl (scale) symmetry t -> lambda t, x -> sqrt(lambda) x: delta_s x = -t xdot + x / 2.

    Use it with ``weyl_boundary_term`` (K = -t L).

    Parameters
    ----------
    q, v : array_like
        Coordinates and velocities, 1-D arrays of equal length.
    t : float
        Value of the independent variable.

    Returns
    -------
    ndarray
        -t v + q / 2 (per unit epsilon).

    References
    ----------
    [BanadosReyes2016] line 292 (delta_s x), Eq. (dilationsym) (line 278).

    Validity
    --------
    A symmetry for L = m v^2 / 2 - alpha / q^2 (one coordinate; also for several coordinates if the potential is
    homogeneous of degree -2 in q). Velocity dependent: ``symmetry_defect`` needs the acceleration.

    Verification
    ------------
    tests/unit/test_noether.py: defect zero for the 1/x^2 potential at random (x, v, a, t) and non-zero for 1/x^3;
    the charge is minus the source's Eq. (QWeyl) and is conserved along an integrated orbit.
    """
    return -t * np.asarray(v) + 0.5 * np.asarray(q)


@source(
    "BanadosReyes2016",
    where="line 301: 'the boundary term is K = -tL' for the Weyl symmetry",
    status="read",
)
def weyl_boundary_term(lagrangian: Lagrangian) -> BoundaryTerm:
    """Build the boundary term K = -t L of the Weyl symmetry.

    Parameters
    ----------
    lagrangian : callable
        ``lagrangian(q, v, t) -> float``.

    Returns
    -------
    callable
        ``K(q, v, t) = -t * lagrangian(q, v, t)``.

    References
    ----------
    [BanadosReyes2016] line 301 (K = -tL), Eq. (QWeyl) (line 302).

    Validity
    --------
    With delta_s x = -t xdot + x / 2 this K gives Noether charge Q = K - p.delta_s x = -(1/2 m x xdot -
    (1/2 m t xdot^2 + alpha t / x^2)), i.e. minus the expression printed as Eq. (QWeyl). The source says
    "up to a sign and an epsilon factor"; with its own Eq. (qn) and its own K the sign comes out opposite to the
    printed Eq. (QWeyl).

    Verification
    ------------
    tests/unit/test_noether.py: charge against -Eq. (QWeyl); x^2(t) = x0^2 + 4 Q t / m + 2 E t^2 / m built from
    the two charges reproduces the integrated orbit.
    """

    def boundary_term(q: Array, v: Array, t: float) -> ArrayLike:
        return -t * lagrangian(q, v, t)

    return boundary_term
