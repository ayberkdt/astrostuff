"""Lagrangian mechanics for a user-supplied Lagrangian ``lagrangian(q, v, t) -> scalar``.

``q`` and ``v`` are 1-D arrays of generalized coordinates and generalized velocities in whatever
units the user chose (they are user-defined, so the parameter names carry no unit suffix), ``t`` is
a float. Every derivative of the Lagrangian is taken numerically with
``kozmos.core._differentiation``: ``method="complex"`` (complex step, exact to rounding, needs a
function that is analytic and accepts complex arrays: avoid ``abs``, ``np.linalg.norm``,
``np.hypot``, comparisons and casts to float inside the Lagrangian; use ``np.sqrt(np.sum(q * q))``)
or ``method="finite"`` (four-point central difference, about 1e-9 relative).

The equations were read from Banados and Reyes [BanadosReyes2016] (arXiv:1601.03616, TeX file
AGR-9.tex; line numbers below refer to that file). That source states the Euler-Lagrange equations
(label ``eom``, line 101) and the action (label ``action0``, line 97); it does not write out the
expansion into accelerations, the matrix of second derivatives, or the numerical methods. Those are
derivations or numerics of this project and the ``note`` of each record says so.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from kozmos._sources import source
from kozmos.core._differentiation import Method, gradient, hessian, jacobian, mixed_second

Array = NDArray[np.float64]
Lagrangian = Callable[[Array, Array, float], ArrayLike]

_MAX_CONDITION = 1e12  # default limit on cond(W) in euler_lagrange_accelerations
# Smallest singular value of W below which W is treated as zero, in units of max |p|. W is a difference of
# momenta over a step of 1e-3 (when the Lagrangian varies on the scale of its arguments), so its rounding error
# is about 2.2e-16 / 1e-3 * max |p| = 2e-13 max |p| with the complex-step momenta and about 1e-9 max |p| with
# the finite ones; the thresholds sit two to three orders above that. They must not sit higher: a momentum with
# a large velocity-independent part (a charge in a strong vector potential, L = v^2/2 + 1e10 v) has a W that is
# small compared with max |p| and still well determined. A 1 x 1 matrix always has condition number 1, so the
# condition number alone cannot detect a Lagrangian that is linear in the single velocity.
_ZERO_FLOOR = {"complex": 2e-11, "finite": 1e-6}


class SingularInertiaError(ValueError):
    """The matrix of second velocity derivatives of the Lagrangian is (numerically) singular."""


def _vector(x: ArrayLike, name: str) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    return arr


def _state(q: ArrayLike, v: ArrayLike) -> tuple[Array, Array]:
    q_arr = _vector(q, "q")
    v_arr = _vector(v, "v")
    if q_arr.size != v_arr.size:
        raise ValueError("q and v must have the same length")
    return q_arr, v_arr


def _dl_dv(lagrangian: Lagrangian, q: Array, v: Array, t: float, method: Method) -> Array:
    return gradient(lambda x: lagrangian(q, x, t), v, method)


def _dl_dq(lagrangian: Lagrangian, q: Array, v: Array, t: float, method: Method) -> Array:
    return gradient(lambda x: lagrangian(x, v, t), q, method)


def _dl_dt(lagrangian: Lagrangian, q: Array, v: Array, t: float, method: Method) -> float:
    return float(jacobian(lambda y: lagrangian(q, v, y[0]), np.array([float(t)]), method)[0])


def _d2l_dvdv(lagrangian: Lagrangian, q: Array, v: Array, t: float, method: Method) -> Array:
    return hessian(lambda x: lagrangian(q, x, t), v, method)


def _d2l_dvdq(lagrangian: Lagrangian, q: Array, v: Array, t: float, method: Method) -> Array:
    """Matrix C[i, j] = d2 L / dv_i dq_j."""
    return mixed_second(lambda x, y: lagrangian(y, x, t), v, q, method)


def _d2l_dvdt(lagrangian: Lagrangian, q: Array, v: Array, t: float, method: Method) -> Array:
    return mixed_second(lambda x, y: lagrangian(q, x, y[0]), v, np.array([float(t)]), method)[:, 0]


@source(
    "BanadosReyes2016",
    where="Eq. (eom) (line 101) and Eq. (qn) (line 236): the combination dL/d(qdot^i)",
    status="read",
    note="the source writes dL/d(qdot^i) in Eq. (eom) and in the Noether charge Eq. (qn) but never states "
    "'p_i = dL/d(qdot^i)' as a definition; identifying it with the p_i of its Hamiltonian action Eq. (ha) "
    "(line 393) is the standard Legendre relation implied by comparing Eq. (ha) with Eq. (action0)",
)
def generalized_momenta(
    lagrangian: Lagrangian, q: ArrayLike, v: ArrayLike, t: float = 0.0, method: Method = "complex"
) -> Array:
    """Compute the generalized (canonical) momenta: p_i = dL/dv_i.

    Parameters
    ----------
    lagrangian : callable
        ``lagrangian(q, v, t) -> float``.
    q, v : array_like
        Generalized coordinates and velocities, 1-D arrays of equal length.
    t : float, optional
        Value of the independent variable.
    method : {"complex", "finite"}, optional
        Differentiation method; see the module docstring.

    Returns
    -------
    ndarray
        p, shape (n,).

    References
    ----------
    [BanadosReyes2016] Eq. (eom), Eq. (qn): the quantity dL/d(qdot^i) of the source.

    Validity
    --------
    Needs a Lagrangian that is differentiable in v. With ``method="complex"`` it must also be analytic
    and accept complex arrays; otherwise ValueError is raised.

    Verification
    ------------
    tests/unit/test_lagrangian.py: p = m v for the harmonic oscillator, p = m v + e A for the charged particle,
    p = m v / sqrt(1 - v^2) for the relativistic particle.
    """
    q_arr, v_arr = _state(q, v)
    return _dl_dv(lagrangian, q_arr, v_arr, t, method)


@source(
    "BanadosReyes2016",
    where="Eq. (eom) (line 101): the term dL/dq^i",
    status="read",
)
def generalized_forces(
    lagrangian: Lagrangian, q: ArrayLike, v: ArrayLike, t: float = 0.0, method: Method = "complex"
) -> Array:
    """Compute the generalized forces: F_i = dL/dq_i at fixed v and t.

    Parameters
    ----------
    lagrangian : callable
        ``lagrangian(q, v, t) -> float``.
    q, v : array_like
        Generalized coordinates and velocities, 1-D arrays of equal length.
    t : float, optional
        Value of the independent variable.
    method : {"complex", "finite"}, optional
        Differentiation method; see the module docstring.

    Returns
    -------
    ndarray
        dL/dq, shape (n,).

    References
    ----------
    [BanadosReyes2016] Eq. (eom).

    Validity
    --------
    Same as ``generalized_momenta``.

    Verification
    ------------
    tests/unit/test_lagrangian.py: F = -k q for the harmonic oscillator, F = -m g l sin(theta) for the pendulum.
    """
    q_arr, v_arr = _state(q, v)
    return _dl_dq(lagrangian, q_arr, v_arr, t, method)


@source(
    "BanadosReyes2016",
    where="Eq. (eom) (line 101)",
    status="read",
    note="derivation in this project from Eq. (eom): the chain rule gives d/dt (dL/dv_i) = sum_j W_ij a_j + "
    "sum_j C_ij v_j + D_i with W = d2L/dv dv; the source does not write this matrix",
)
def inertia_matrix(
    lagrangian: Lagrangian, q: ArrayLike, v: ArrayLike, t: float = 0.0, method: Method = "complex"
) -> Array:
    """Compute the matrix of second velocity derivatives (generalized mass matrix): W_ij = d2L / dv_i dv_j.

    Parameters
    ----------
    lagrangian : callable
        ``lagrangian(q, v, t) -> float``.
    q, v : array_like
        Generalized coordinates and velocities, 1-D arrays of equal length.
    t : float, optional
        Value of the independent variable.
    method : {"complex", "finite"}, optional
        Differentiation method; see the module docstring.

    Returns
    -------
    ndarray
        W, shape (n, n), symmetric.

    References
    ----------
    [BanadosReyes2016] Eq. (eom) (the matrix multiplies the accelerations once Eq. (eom) is expanded).

    Validity
    --------
    Four-point central difference of the complex-step gradient: about 1e-12 relative for a smooth
    Lagrangian (about 1e-9 with ``method="finite"``). The Euler-Lagrange equations can be solved for the
    accelerations only where W is non-singular.

    Verification
    ------------
    tests/unit/test_lagrangian.py: W = m for the oscillator, the 2x2 matrix of the double pendulum, the
    relativistic Hessian m gamma (1 + gamma^2 v v^T).
    """
    q_arr, v_arr = _state(q, v)
    return _d2l_dvdv(lagrangian, q_arr, v_arr, t, method)


@source(
    "BanadosReyes2016",
    where="Eq. (qn) (line 236) with Eq. (dr) (line 166) and the boundary term K = -eps L of Eq. (ttt) (line 173-175)",
    status="read",
    note="derivation in this project from Eq. (qn): with delta_s q = -eps qdot and K = -eps L the charge "
    "Q = K - p.delta_s q equals eps (p.qdot - L); the source works this out only for L = T - V "
    "(Q_eps = eps E, line 251)",
)
def jacobi_function(
    lagrangian: Lagrangian, q: ArrayLike, v: ArrayLike, t: float = 0.0, method: Method = "complex"
) -> float:
    """Compute the energy function of the Lagrangian: h = p.v - L.

    It is the Noether charge of the translation in the independent variable (per unit epsilon), and
    it equals the total energy T + V for L = T - V with T quadratic in v. When L depends explicitly on t,
    h is not conserved: dh/dt = -dL/dt (partial derivative) on a solution.

    Parameters
    ----------
    lagrangian : callable
        ``lagrangian(q, v, t) -> float``.
    q, v : array_like
        Generalized coordinates and velocities, 1-D arrays of equal length.
    t : float, optional
        Value of the independent variable.
    method : {"complex", "finite"}, optional
        Differentiation method; see the module docstring.

    Returns
    -------
    float
        h.

    References
    ----------
    [BanadosReyes2016] Eq. (qn), Eq. (dr), Eq. (ttt): Q_eps = eps E with E = (m/2) v^2 + V.

    Validity
    --------
    The sign is that of the source: the charge of delta_s q = -eps qdot is +eps h. Conserved only if L has
    no explicit dependence on t.

    Verification
    ------------
    tests/unit/test_lagrangian.py: h = T + V for oscillator, pendulum and Kepler; dh/dt = -dL/dt for a driven
    oscillator; h of the relativistic particle is m / sqrt(1 - v^2).
    """
    q_arr, v_arr = _state(q, v)
    p = _dl_dv(lagrangian, q_arr, v_arr, t, method)
    return float(p @ v_arr - float(np.real(lagrangian(q_arr, v_arr, t))))


@source(
    "BanadosReyes2016",
    where="Eq. (eom) (line 101)",
    status="read",
    note="derivation in this project from Eq. (eom): M a = dL/dq - (d2L/dv dq) v - d2L/dv dt, M = d2L/dv dv",
)
def euler_lagrange_accelerations(
    lagrangian: Lagrangian,
    q: ArrayLike,
    v: ArrayLike,
    t: float = 0.0,
    method: Method = "complex",
    max_condition: float = _MAX_CONDITION,
) -> Array:
    """Solve the Euler-Lagrange equations for the accelerations.

    d/dt (dL/dv_i) = W_ij a_j + C_ij v_j + D_i = dL/dq_i, with W = d2L/dv dv, C_ij = d2L/dv_i dq_j and
    D_i = d2L/dv_i dt, so that W a = dL/dq - C v - D.

    Parameters
    ----------
    lagrangian : callable
        ``lagrangian(q, v, t) -> float``.
    q, v : array_like
        Generalized coordinates and velocities, 1-D arrays of equal length.
    t : float, optional
        Value of the independent variable.
    method : {"complex", "finite"}, optional
        Differentiation method; see the module docstring.
    max_condition : float, optional
        Largest accepted condition number of W (default 1e12).

    Returns
    -------
    ndarray
        Accelerations a = d2q/dt2, shape (n,).

    Raises
    ------
    SingularInertiaError
        If cond(W) > ``max_condition`` (default 1e12), or if the smallest singular value of W is not above the
        rounding floor 2e-11 max|p| (1e-6 max|p| for ``method="finite"``; this catches a Lagrangian linear in
        the velocity, for which W is zero and a 1 x 1 matrix has condition number 1). Typical causes:
        constraints, gauge freedom, a first-order Lagrangian.

    References
    ----------
    [BanadosReyes2016] Eq. (eom).

    Validity
    --------
    Mixed derivatives are central differences of complex-step gradients: relative error about 1e-12.
    The singularity test is a numerical threshold: a matrix W with a genuine singular value below the floor
    (for example a tiny mass 1e-10 together with momenta of order one) is reported as singular.
    A Lagrangian with a singular W needs the Hamiltonian constraint analysis, which is not implemented.

    Verification
    ------------
    tests/unit/test_lagrangian.py: closed-form accelerations of the oscillator, the plane pendulum, the double
    pendulum and the charged particle; singular case raises.
    """
    q_arr, v_arr = _state(q, v)
    w = _d2l_dvdv(lagrangian, q_arr, v_arr, t, method)
    singular_values = np.linalg.svd(w, compute_uv=False)
    p_scale = float(np.max(np.abs(_dl_dv(lagrangian, q_arr, v_arr, t, method))))
    floor = _ZERO_FLOOR[method] * p_scale
    cond = np.inf if singular_values[-1] <= floor else float(singular_values[0] / singular_values[-1])
    if not np.isfinite(cond) or cond > max_condition:
        raise SingularInertiaError(
            f"the matrix d2L/dv dv is singular (smallest singular value {singular_values[-1]:.3g}, condition "
            f"number {cond:.3g} > {max_condition:.3g}, zero floor {floor:.3g}): the Euler-Lagrange equations "
            "cannot be solved for the accelerations"
        )
    force = _dl_dq(lagrangian, q_arr, v_arr, t, method)
    cross = _d2l_dvdq(lagrangian, q_arr, v_arr, t, method)
    explicit = _d2l_dvdt(lagrangian, q_arr, v_arr, t, method)
    return np.linalg.solve(w, force - cross @ v_arr - explicit)


@source(
    "BanadosReyes2016",
    where="Eq. (eom) (line 101)",
    status="read",
    note="the left-hand side of Eq. (eom) is expanded with the chain rule (derivation in this project)",
)
def euler_lagrange_residual(
    lagrangian: Lagrangian,
    q: ArrayLike,
    v: ArrayLike,
    a: ArrayLike,
    t: float = 0.0,
    method: Method = "complex",
) -> Array:
    """Evaluate the left-hand side of the Euler-Lagrange equations: d/dt (dL/dv) - dL/dq.

    It is zero when (q, v, a) belong to a solution; for arbitrary a it is the "defect" of the trial motion.

    Parameters
    ----------
    lagrangian : callable
        ``lagrangian(q, v, t) -> float``.
    q, v, a : array_like
        Coordinates, velocities and accelerations, 1-D arrays of equal length.
    t : float, optional
        Value of the independent variable.
    method : {"complex", "finite"}, optional
        Differentiation method; see the module docstring.

    Returns
    -------
    ndarray
        W a + C v + D - dL/dq (see ``euler_lagrange_accelerations``), shape (n,).

    References
    ----------
    [BanadosReyes2016] Eq. (eom).

    Validity
    --------
    Same as ``euler_lagrange_accelerations``; no inversion, so a singular W is allowed.

    Verification
    ------------
    tests/unit/test_lagrangian.py: zero for the closed-form accelerations of the oscillator and the double pendulum,
    equal to m (a - a_true) for a perturbed oscillator acceleration.
    """
    q_arr, v_arr = _state(q, v)
    a_arr = _vector(a, "a")
    if a_arr.size != q_arr.size:
        raise ValueError("a must have the same length as q")
    w = _d2l_dvdv(lagrangian, q_arr, v_arr, t, method)
    force = _dl_dq(lagrangian, q_arr, v_arr, t, method)
    cross = _d2l_dvdq(lagrangian, q_arr, v_arr, t, method)
    explicit = _d2l_dvdt(lagrangian, q_arr, v_arr, t, method)
    return w @ a_arr + cross @ v_arr + explicit - force


@source(
    "BanadosReyes2016",
    where="Eq. (eom) (line 101)",
    status="read",
    note="numerical integration of the accelerations of Eq. (eom) with scipy solve_ivp DOP853 (explicit "
    "Runge-Kutta of order 8(5,3)); the integrator itself was not read from a source in this project",
)
def integrate_lagrangian(
    lagrangian: Lagrangian,
    q0: ArrayLike,
    v0: ArrayLike,
    times: ArrayLike,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    method: Method = "complex",
    max_condition: float = _MAX_CONDITION,
) -> tuple[Array, Array]:
    """Integrate the Euler-Lagrange equations from (q0, v0) at ``times[0]``.

    Parameters
    ----------
    lagrangian : callable
        ``lagrangian(q, v, t) -> float``.
    q0, v0 : array_like
        Initial coordinates and velocities, 1-D arrays of equal length.
    times : array_like
        Strictly increasing output values of t (at least two); ``times[0]`` is the initial value.
    rtol, atol : float, optional
        Tolerances passed to ``scipy.integrate.solve_ivp`` (DOP853).
    method : {"complex", "finite"}, optional
        Differentiation method; see the module docstring.
    max_condition : float, optional
        Largest accepted condition number of the matrix d2L/dv dv.

    Returns
    -------
    q, v : ndarray
        Arrays of shape (len(times), n).

    Raises
    ------
    RuntimeError
        If the integrator reports failure.
    SingularInertiaError
        If the matrix d2L/dv dv becomes singular along the way.

    References
    ----------
    [BanadosReyes2016] Eq. (eom).

    Validity
    --------
    The local error is controlled by rtol and atol only; conserved quantities drift at the level the tests
    measure (see tests/unit/test_lagrangian.py), not exactly. Each right-hand side evaluation costs a few
    dozen evaluations of the Lagrangian, so the routine is meant for small n.

    Verification
    ------------
    tests/unit/test_lagrangian.py: harmonic oscillator against the closed form, plane pendulum against the
    Jacobi elliptic solution, Kepler orbit (polar and Cartesian) against the solution of Kepler's equation,
    energy and angular momentum conservation.
    """
    q0_arr, v0_arr = _state(q0, v0)
    n = q0_arr.size
    t_arr = np.asarray(times, dtype=float)
    if t_arr.ndim != 1 or t_arr.size < 2 or np.any(np.diff(t_arr) <= 0.0):
        raise ValueError("times must be a strictly increasing 1-D array with at least two values")

    def rhs(t: float, y: Array) -> Array:
        a = euler_lagrange_accelerations(lagrangian, y[:n], y[n:], t, method, max_condition)
        return np.concatenate([y[n:], a])

    solution = solve_ivp(
        rhs,
        (t_arr[0], t_arr[-1]),
        np.concatenate([q0_arr, v0_arr]),
        method="DOP853",
        t_eval=t_arr,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"integration failed: {solution.message}")
    return solution.y[:n].T, solution.y[n:].T


@source(
    "BanadosReyes2016",
    where="Eq. (action0) (line 97): I[q] = integral dt L(q, qdot, t)",
    status="read",
    note="the quadrature is this project's own: cubic spline with not-a-knot ends through the samples (error O(h^4) in q, "
    "O(h^3) in qdot) and Gauss-Legendre nodes on every interval",
)
def action(lagrangian: Lagrangian, times: ArrayLike, q_path: ArrayLike, nodes: int = 5) -> float:
    """Compute the action S = integral L(q, qdot, t) dt of a sampled path.

    The samples are interpolated by a cubic spline (scipy ``CubicSpline``, not-a-knot ends), which gives q(t)
    and qdot(t); L is integrated on every interval with ``nodes``-point Gauss-Legendre quadrature (exact for
    polynomial integrands of degree 2 * nodes - 1, so the quadrature error is negligible against the spline).
    The spline error is O(h^4) in q and O(h^3) in qdot, so the guaranteed rate of the action of a smooth path is
    O(h^3) in the spacing h; the rate measured for an analytic path (oscillator, 11 to 81 samples) is about 7 because
    the leading spline errors cancel in the integral (see the tests); do not rely on more than O(h^3) in general.

    Parameters
    ----------
    lagrangian : callable
        ``lagrangian(q, v, t) -> float``.
    times : array_like
        Strictly increasing sample values of t, at least four.
    q_path : array_like
        Sampled coordinates, shape (len(times), n).
    nodes : int, optional
        Gauss-Legendre points per interval (default 5).

    Returns
    -------
    float
        S.

    References
    ----------
    [BanadosReyes2016] Eq. (action0), and the variation Eq. (dosI2) (line 220) that makes S stationary on a solution.

    Validity
    --------
    Smooth paths only; the end points are those of the first and last sample. Stationarity holds for variations
    that vanish at both ends (fixed end points).

    Verification
    ------------
    tests/unit/test_lagrangian.py: S of the free particle and the oscillator against the closed form; the change
    of S under an end-point-preserving perturbation is O(eps^2) on the true path and O(eps) on a false one.
    """
    t_arr = np.asarray(times, dtype=float)
    q_arr = np.asarray(q_path, dtype=float)
    if t_arr.ndim != 1 or t_arr.size < 4 or np.any(np.diff(t_arr) <= 0.0):
        raise ValueError("times must be strictly increasing with at least four values")
    if q_arr.ndim == 1:
        q_arr = q_arr[:, None]
    if q_arr.ndim != 2 or q_arr.shape[0] != t_arr.size:
        raise ValueError("q_path must have shape (len(times), n)")
    spline = CubicSpline(t_arr, q_arr, axis=0)
    x, w = np.polynomial.legendre.leggauss(nodes)
    total = 0.0
    for t_left, t_right in zip(t_arr[:-1], t_arr[1:], strict=True):
        half = 0.5 * (t_right - t_left)
        tt = 0.5 * (t_left + t_right) + half * x
        qq = spline(tt)
        vv = spline(tt, 1)
        total += half * sum(wk * float(np.real(lagrangian(qq[k], vv[k], tt[k]))) for k, wk in enumerate(w))
    return float(total)
