"""Hamiltonian mechanics: Legendre transform, Hamilton's equations, Poisson brackets, canonical symmetries.

A Hamiltonian is a callable ``hamiltonian(q, p, t) -> scalar`` with ``q`` and ``p`` 1-D arrays of equal length.
The conventions are those of Banados and Reyes [BanadosReyes2016] (arXiv:1601.03616, TeX file AGR-9.tex; line
numbers refer to that file): Hamiltonian action Eq. (ha) (line 393), Poisson bracket Eq. (Poissonpoint)
(line 397) with [F, G] = dF/dq . dG/dp - dF/dp . dG/dq, equations of motion qdot = dH/dp = [q, H] and
pdot = -dH/dq = [p, H] (lines 402-403), and dG/dt = [G, H] + dG/dt(explicit) Eq. (GH) (line 409).

Derivatives are taken numerically with ``kozmos.core._differentiation`` (``method="complex"`` or ``"finite"``,
see ``kozmos.mechanics.lagrangian``). A Hamiltonian obtained from a Lagrangian by ``hamiltonian_from_lagrangian``
contains a Newton iteration, which is not complex-analytic; that object therefore also provides exact
derivatives (``derivatives`` method, from the envelope relations stated in its docstring) and the functions of
this module use them automatically.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from kozmos._sources import source
from kozmos.core._differentiation import Method, gradient, jacobian
from kozmos.mechanics.lagrangian import Lagrangian, _dl_dq, _dl_dt, _dl_dv, _vector

Array = NDArray[np.float64]
_EPS = float(np.finfo(float).eps)
# The residual dL/dv - p cannot be made smaller than the rounding of v (half an ulp) times |W|, plus the rounding of
# the evaluation itself. For L = -m sqrt(1 - v^2) the residual at the last iterate, in units of
# eps (||W||_inf max|v| + max(1, max|p|)), was at most 8.3 over p / m = 40, 50, 60, 100, 300, 1000, 3000, 5000, 6000,
# 6600, 6683; _RESOLUTION_FACTOR = 16 is about twice that (600 log-uniform p / m in [1, 6600] all converged).
_RESOLUTION_FACTOR = 16.0
Hamiltonian = Callable[[Array, Array, float], ArrayLike]
Observable = Callable[[Array, Array, float], ArrayLike]


class LegendreTransformError(RuntimeError):
    """The inversion p = dL/dv did not converge (no solution, or a singular Hessian)."""


class LegendreHamiltonian:
    """Hamiltonian H(q, p, t) = p.v - L(q, v, t) with v defined by p = dL/dv(q, v, t).

    The inversion is a damped Newton iteration v <- v - W^{-1} (dL/dv - p), W = d2L/dv dv. For a Lagrangian
    quadratic in v the map p(v) is affine and one step is exact. The step is halved (at most 30 times) while
    the trial point is outside the domain of L (non-finite value) or does not reduce the largest residual
    component. Convergence is declared when max |dL/dv - p| <= ``tolerance`` * max(max |p|, max |dL/dv(v_start)|) or when the
    residual has reached the resolution of the map, 16 eps (||W||_inf max|v| + max(1, max|p|)) (half an ulp of v
    times |W|, plus the rounding of the evaluation: no iterate can do better, and H is stationary in v, so the
    resulting error of H is of second order); otherwise ``LegendreTransformError`` is raised.
    Envelope relations (derivation in this project): with p = dL/dv the dependence of H on v drops out of its
    first derivatives, so dH/dp = v, dH/dq = -dL/dq (at fixed v) and dH/dt = -dL/dt (at fixed v).
    """

    def __init__(
        self,
        lagrangian: Lagrangian,
        method: Method = "complex",
        tolerance: float = 1e-13,
        max_iterations: int = 100,
    ) -> None:
        self.lagrangian = lagrangian
        self.method: Method = method
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def _finite_lagrangian(self, q: Array, v: Array, t: float) -> bool:
        with np.errstate(all="ignore"):
            return bool(np.isfinite(np.real(self.lagrangian(q, v, t))))

    def _newton_matrix(self, q: Array, v: Array, t: float) -> Array:
        """Approximate W = d2L/dv dv by differences of dL/dv with a step that stays inside the domain of L.

        For each column the step starts at 1e-7 max(1, |v_i|) and is divided by 10 (at most 7 times) until both
        v + h e_i and v - h e_i are finite points of dL/dv (central difference); if only one side ever is, the
        one-sided difference at the smallest step is used. A fixed step would leave the domain of a Lagrangian such
        as -m sqrt(1 - v^2) once 1 - |v| is below the step (p of the order of 1e3 m) and would then give a slope that
        is wrong by orders of magnitude. The inexact matrix only slows the Newton iteration (the residual, not the
        matrix, decides convergence).
        """
        base = _dl_dv(self.lagrangian, q, v, t, self.method)
        columns = []
        for i in range(v.size):
            scale = max(1.0, abs(float(v[i])))
            column = None
            fallback = None
            for k in range(8):
                step = 1e-7 * scale * 0.1**k
                values = {}
                for sign in (1.0, -1.0):
                    shifted = v.copy()
                    shifted[i] += sign * step
                    try:
                        value = _dl_dv(self.lagrangian, q, shifted, t, self.method)
                    except ValueError:
                        continue
                    if np.all(np.isfinite(value)):
                        values[sign] = value
                if len(values) == 2:
                    column = (values[1.0] - values[-1.0]) / (2.0 * step)
                    break
                for sign, value in values.items():
                    fallback = sign * (value - base) / step
            if column is None:
                column = fallback
            if column is None:
                raise LegendreTransformError("the Newton matrix d2L/dv dv could not be evaluated near the current v")
            columns.append(column)
        return np.stack(columns, axis=1)

    def velocity(self, q: ArrayLike, p: ArrayLike, t: float = 0.0, v_guess: ArrayLike | None = None) -> Array:
        """Solve p = dL/dv for v by damped Newton iteration (see the class docstring)."""
        q_arr = _vector(q, "q")
        p_arr = _vector(p, "p")
        if q_arr.size != p_arr.size:
            raise ValueError("q and p must have the same length")
        v = np.zeros_like(q_arr) if v_guess is None else _vector(v_guess, "v_guess").copy()
        start = _dl_dv(self.lagrangian, q_arr, v, t, self.method)
        # the size of the momenta in the problem, not 1: an electron has p of order 1e-24 in SI units, and a
        # tolerance against max(1, |p|) would accept the starting velocity as the solution
        scale = max(float(np.max(np.abs(p_arr))), float(np.max(np.abs(start))))
        residual = start - p_arr
        for _ in range(self.max_iterations):
            norm = float(np.max(np.abs(residual)))
            if norm <= self.tolerance * scale:
                return v
            w = self._newton_matrix(q_arr, v, t)
            if norm <= _RESOLUTION_FACTOR * _EPS * (
                float(np.max(np.sum(np.abs(w), axis=1))) * float(np.max(np.abs(v))) + scale
            ):
                return v  # the residual is at the resolution of the map v -> p (see the class docstring)
            try:
                step = np.linalg.solve(w, -residual)
            except np.linalg.LinAlgError as error:
                raise LegendreTransformError("the matrix d2L/dv dv is singular during the Newton iteration") from error
            damping = 1.0
            for _halving in range(31):
                trial = v + damping * step
                if self._finite_lagrangian(q_arr, trial, t):
                    try:
                        trial_residual = _dl_dv(self.lagrangian, q_arr, trial, t, self.method) - p_arr
                    except ValueError:
                        trial_residual = np.full_like(p_arr, np.nan)
                    if np.all(np.isfinite(trial_residual)) and float(np.max(np.abs(trial_residual))) < norm:
                        break
                damping *= 0.5
            else:
                raise LegendreTransformError("Newton iteration for p = dL/dv found no descent step")
            v, residual = trial, trial_residual
        raise LegendreTransformError(f"Newton iteration did not converge in {self.max_iterations} steps")

    def __call__(self, q: ArrayLike, p: ArrayLike, t: float = 0.0) -> float:
        if np.iscomplexobj(q) or np.iscomplexobj(p):
            raise ValueError("a Legendre-transformed Hamiltonian is not complex-analytic; use method='finite'")
        q_arr = _vector(q, "q")
        p_arr = _vector(p, "p")
        v = self.velocity(q_arr, p_arr, t)
        return float(p_arr @ v - float(np.real(self.lagrangian(q_arr, v, t))))

    def derivatives(self, q: ArrayLike, p: ArrayLike, t: float = 0.0) -> tuple[Array, Array, float]:
        """Return (dH/dq, dH/dp, dH/dt) from the envelope relations."""
        q_arr = _vector(q, "q")
        p_arr = _vector(p, "p")
        v = self.velocity(q_arr, p_arr, t)
        return (
            -_dl_dq(self.lagrangian, q_arr, v, t, self.method),
            v,
            -_dl_dt(self.lagrangian, q_arr, v, t, self.method),
        )


def _phase_gradients(hamiltonian: Hamiltonian, q: Array, p: Array, t: float, method: Method) -> tuple[Array, Array]:
    """Return (dH/dq, dH/dp), using exact envelope derivatives when the object provides them."""
    derivatives = getattr(hamiltonian, "derivatives", None)
    if derivatives is not None:
        dh_dq, dh_dp, _ = derivatives(q, p, t)
        return dh_dq, dh_dp
    n = q.size
    g = gradient(lambda z: hamiltonian(z[:n], z[n:], t), np.concatenate([q, p]), method)
    return g[:n], g[n:]


def _phase(q: ArrayLike, p: ArrayLike) -> tuple[Array, Array]:
    q_arr = _vector(q, "q")
    p_arr = _vector(p, "p")
    if q_arr.size != p_arr.size:
        raise ValueError("q and p must have the same length")
    return q_arr, p_arr


@source(
    "BanadosReyes2016",
    where="Eq. (ha) (line 393) together with Eq. (action0) (line 97)",
    status="read",
    note="Eq. (ha) is I = integral (p qdot - H) and Eq. (action0) is I = integral L; equating the integrands gives "
    "H = p.v - L with p = dL/dv (the standard Legendre transform, written here as a consequence of the two "
    "equations, not stated by the source); the damped Newton inversion is numerics of this project",
)
def hamiltonian_from_lagrangian(
    lagrangian: Lagrangian,
    method: Method = "complex",
    tolerance: float = 1e-13,
    max_iterations: int = 100,
) -> LegendreHamiltonian:
    """Build the Hamiltonian H(q, p, t) = p.v - L by inverting p = dL/dv numerically.

    Parameters
    ----------
    lagrangian : callable
        ``lagrangian(q, v, t) -> float``.
    method : {"complex", "finite"}, optional
        Differentiation method used in the Newton iteration and in ``derivatives``.
    tolerance : float, optional
        Convergence: max |dL/dv - p| <= tolerance * max(max |p|, max |dL/dv| at the starting velocity).
        Default 1e-13.
    max_iterations : int, optional
        Newton iterations before ``LegendreTransformError`` (default 100).

    Returns
    -------
    LegendreHamiltonian
        A callable ``H(q, p, t=0.0) -> float`` with extra methods ``velocity(q, p, t)`` and
        ``derivatives(q, p, t) -> (dH/dq, dH/dp, dH/dt)``.

    Raises
    ------
    LegendreTransformError
        (when called) if the Newton iteration does not converge.

    References
    ----------
    [BanadosReyes2016] Eq. (ha), Eq. (action0).

    Validity
    --------
    Needs W = d2L/dv dv non-singular on the path of the iteration, and the map v -> p to be invertible near the
    solution (a convex Lagrangian in v is the safe case). The iteration starts at v = 0, so a Lagrangian that is
    undefined there needs ``LegendreHamiltonian.velocity`` with a ``v_guess``. If the map has several branches,
    the branch reached from v = 0 is returned. The Newton matrix uses a step that shrinks until both sides lie in
    the domain of L, and convergence is judged at the resolution of the map (see ``LegendreHamiltonian``), so the
    free relativistic particle L = -m sqrt(1 - v^2) is inverted up to p = 1.14e4 m (measured by bisection in one
    dimension, m = 1). At the limit 1 - v is 4e-9: the check of ``kozmos.core._differentiation`` that the
    Lagrangian is analytic needs real steps that stay inside |v| < 1 on both sides of the trial velocity, and
    for a step that small the rounding of v + h (2e-16 against h) exceeds the accuracy it requires; beyond the
    limit ``LegendreTransformError`` is raised. The object does not accept complex arguments (differentiate it with
    ``method="finite"`` or use its exact ``derivatives``).

    Verification
    ------------
    tests/unit/test_hamiltonian.py: H = sqrt(p^2 + m^2) for the free relativistic particle from |p| = 0 to 1e4 m
    (boundary region included, and the failure above 1.3e4 m), an electron in SI units, H = p^2/2m + k q^2/2 for the oscillator in one
    Newton step, the charged particle H = (p - e A)^2/2m, and, for L = (1 + 0.3 t) v^2/2 - q^2/2, H and all three components of
    ``derivatives`` (including dH/dt) against the closed form.
    """
    return LegendreHamiltonian(lagrangian, method, tolerance, max_iterations)


@source(
    "BanadosReyes2016",
    where="Hamilton's equations after Eq. (Poissonpoint), lines 402-403: qdot = dH/dp, pdot = -dH/dq",
    status="read",
)
def hamilton_equations(
    hamiltonian: Hamiltonian, q: ArrayLike, p: ArrayLike, t: float = 0.0, method: Method = "complex"
) -> tuple[Array, Array]:
    """Evaluate Hamilton's equations: dq/dt = dH/dp, dp/dt = -dH/dq.

    Parameters
    ----------
    hamiltonian : callable
        ``hamiltonian(q, p, t) -> float``, or the object returned by ``hamiltonian_from_lagrangian``.
    q, p : array_like
        Coordinates and momenta, 1-D arrays of equal length.
    t : float, optional
        Value of the independent variable.
    method : {"complex", "finite"}, optional
        Differentiation method (ignored for the exact derivatives of a Legendre Hamiltonian).

    Returns
    -------
    dq_dt, dp_dt : ndarray
        Each of shape (n,).

    References
    ----------
    [BanadosReyes2016] lines 402-403 (equations of motion after Eq. (Poissonpoint)).

    Validity
    --------
    Complex step needs an analytic Hamiltonian that accepts complex arrays.

    Verification
    ------------
    tests/unit/test_hamiltonian.py: oscillator, pendulum and charged-particle closed forms; dq/dt = p / H for the
    relativistic particle (finite and complex-step methods agree to the measured 5e-13).
    """
    q_arr, p_arr = _phase(q, p)
    dh_dq, dh_dp = _phase_gradients(hamiltonian, q_arr, p_arr, t, method)
    return dh_dp, -dh_dq


@source(
    "BanadosReyes2016",
    where="Eq. (Poissonpoint) (line 397): [F, G] = dF/dq dG/dp - dF/dp dG/dq",
    status="read",
)
def poisson_bracket(
    f: Observable, g: Observable, q: ArrayLike, p: ArrayLike, t: float = 0.0, method: Method = "complex"
) -> float:
    """Evaluate the Poisson bracket [F, G] = sum_i (dF/dq_i dG/dp_i - dF/dp_i dG/dq_i) at one phase-space point.

    Parameters
    ----------
    f, g : callable
        Scalar functions ``f(q, p, t)`` and ``g(q, p, t)``.
    q, p : array_like
        Coordinates and momenta, 1-D arrays of equal length.
    t : float, optional
        Value of the independent variable (held fixed).
    method : {"complex", "finite"}, optional
        Differentiation method. A bracket of brackets (Jacobi identity) needs the outer one to be ``"finite"``,
        because the complex step cannot be nested.

    Returns
    -------
    float
        [F, G].

    References
    ----------
    [BanadosReyes2016] Eq. (Poissonpoint); sign convention as in the source ([q, p] = +1).

    Validity
    --------
    The precision is that of the gradients: 1e-15 relative for the complex step with analytic functions, about
    1e-9 for the finite method.

    Verification
    ------------
    tests/unit/test_hamiltonian.py: [q_i, p_j] = delta_ij, [Lx, Ly] = Lz and cyclic, the Jacobi identity, [L, H] = 0
    for the Kepler Hamiltonian, the sl(2, R) algebra of the conformal particle.
    """
    q_arr, p_arr = _phase(q, p)
    fq, fp = _phase_gradients(f, q_arr, p_arr, t, method)
    gq, gp = _phase_gradients(g, q_arr, p_arr, t, method)
    return float(fq @ gp - fp @ gq)


@source(
    "BanadosReyes2016",
    where="Eq. (GH) (line 409): dG/dt = [G, H] + dG/dt (explicit)",
    status="read",
)
def observable_derivative(
    observable: Observable,
    hamiltonian: Hamiltonian,
    q: ArrayLike,
    p: ArrayLike,
    t: float = 0.0,
    method: Method = "complex",
) -> float:
    """Evaluate the total derivative of an observable along the flow: dG/dt = [G, H] + dG/dt (explicit).

    Parameters
    ----------
    observable : callable
        Scalar function ``G(q, p, t)``, possibly explicitly dependent on t.
    hamiltonian : callable
        ``H(q, p, t)``.
    q, p : array_like
        Coordinates and momenta, 1-D arrays of equal length.
    t : float, optional
        Value of the independent variable.
    method : {"complex", "finite"}, optional
        Differentiation method.

    Returns
    -------
    float
        dG/dt at the point (q, p, t) for the motion generated by H.

    References
    ----------
    [BanadosReyes2016] Eq. (GH); conservation condition Eq. (qh1) (line 413) and Eq. (tQ) (line 454).

    Validity
    --------
    Explicit t derivative by the same method as the gradients (complex step needs analytic dependence on t).

    Verification
    ------------
    tests/unit/test_hamiltonian.py: zero for the charges H, Q, K of the conformal particle (Q and K depend
    explicitly on t), and d/dt of the angular momentum under a non-central force equals the torque.
    """
    q_arr, p_arr = _phase(q, p)
    bracket = poisson_bracket(observable, hamiltonian, q_arr, p_arr, t, method)
    explicit = float(jacobian(lambda y: observable(q_arr, p_arr, y[0]), np.array([float(t)]), method)[0])
    return bracket + explicit


@source(
    "BanadosReyes2016",
    where="Hamilton's equations, lines 402-403 (after Eq. (Poissonpoint))",
    status="read",
    note="numerical integration with scipy solve_ivp DOP853",
)
def integrate_hamiltonian(
    hamiltonian: Hamiltonian,
    q0: ArrayLike,
    p0: ArrayLike,
    times: ArrayLike,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    method: Method = "complex",
) -> tuple[Array, Array]:
    """Integrate Hamilton's equations from (q0, p0) at ``times[0]``.

    Parameters
    ----------
    hamiltonian : callable
        ``H(q, p, t)``.
    q0, p0 : array_like
        Initial coordinates and momenta, 1-D arrays of equal length.
    times : array_like
        Strictly increasing output values of t (at least two).
    rtol, atol : float, optional
        Tolerances passed to ``scipy.integrate.solve_ivp`` (DOP853).
    method : {"complex", "finite"}, optional
        Differentiation method.

    Returns
    -------
    q, p : ndarray
        Arrays of shape (len(times), n).

    Raises
    ------
    RuntimeError
        If the integrator reports failure.

    References
    ----------
    [BanadosReyes2016] lines 402-403.

    Validity
    --------
    Not symplectic: the energy drifts at the level of the local error (measured in the tests).

    Verification
    ------------
    tests/unit/test_hamiltonian.py: oscillator against the closed form; the Lagrangian and Hamiltonian
    integrations of the pendulum agree through p = dL/dv; relativistic particle in a constant force against the
    hyperbolic-motion closed form.
    """
    q0_arr, p0_arr = _phase(q0, p0)
    n = q0_arr.size
    t_arr = np.asarray(times, dtype=float)
    if t_arr.ndim != 1 or t_arr.size < 2 or np.any(np.diff(t_arr) <= 0.0):
        raise ValueError("times must be a strictly increasing 1-D array with at least two values")

    def rhs(t: float, y: Array) -> Array:
        dq, dp = hamilton_equations(hamiltonian, y[:n], y[n:], t, method)
        return np.concatenate([dq, dp])

    solution = solve_ivp(
        rhs,
        (t_arr[0], t_arr[-1]),
        np.concatenate([q0_arr, p0_arr]),
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
    where="Eq. (invNoe) (line 421): delta_s q = eps dQ/dp, delta_s p = -eps dQ/dq",
    status="read",
)
def canonical_symmetry_variation(
    charge: Observable, q: ArrayLike, p: ArrayLike, t: float = 0.0, method: Method = "complex"
) -> tuple[Array, Array]:
    """Evaluate the symmetry generated by a charge (Noether's inverse theorem), per unit epsilon.

    Parameters
    ----------
    charge : callable
        Scalar function ``Q(q, p, t)`` satisfying dQ/dt = 0 (Eq. (qh1) of the source).
    q, p : array_like
        Coordinates and momenta, 1-D arrays of equal length.
    t : float, optional
        Value of the independent variable.
    method : {"complex", "finite"}, optional
        Differentiation method.

    Returns
    -------
    delta_q, delta_p : ndarray
        [q, Q] = dQ/dp and [p, Q] = -dQ/dq, each of shape (n,).

    References
    ----------
    [BanadosReyes2016] Eq. (invNoe).

    Validity
    --------
    The function does not check that Q is conserved. Sign remark from this project: in the source the boundary
    term of this symmetry is K = -eps Q + p.delta_s q (last line of the variation below Eq. (invNoe)), so the
    Noether charge of Eq. (qn), K - p.delta_s q, is -eps Q, not +eps Q as the prose of the source says.

    Verification
    ------------
    tests/unit/test_hamiltonian.py: Q = Lz generates (delta x, delta y) = (-y, x); Q = H generates (qdot, pdot).
    tests/unit/test_noether.py: the Noether charge of the generated symmetry is -Q.
    """
    q_arr, p_arr = _phase(q, p)
    dq_charge, dp_charge = _phase_gradients(charge, q_arr, p_arr, t, method)
    return dp_charge, -dq_charge


@source(
    "BanadosReyes2016",
    where="conformal particle in Hamiltonian form, lines 473-475: H = p^2/2m + alpha/q^2, Q = -t H + p q / 2, "
    "K = t^2 H + 2 t Q - m q^2 / 2",
    status="read",
)
def conformal_particle_charges(
    q: ArrayLike, p: ArrayLike, t: float, m: float, alpha: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate the three conserved charges (H, Q, K) of the particle in the potential alpha / q^2.

    H = p^2 / (2 m) + alpha / q^2, Q = -t H + p q / 2, K = t^2 H + 2 t Q - m q^2 / 2. Q and K depend explicitly on t.
    They satisfy [Q, H] = H, [Q, K] = -K, [H, K] = 2 Q and 2 K H + 2 Q^2 + m alpha = 0.

    Parameters
    ----------
    q, p : array_like
        Position and momentum (scalars or arrays; the formulas act elementwise).
    t : float
        Value of the independent variable.
    m : float
        Mass parameter of the particle (the coefficient of the kinetic term).
    alpha : float
        Coefficient of the potential alpha / q^2.

    Returns
    -------
    H, Q, K : ndarray
        The three charges.

    References
    ----------
    [BanadosReyes2016] lines 473-475 (charges), 479 (relation), 483-485 (algebra).

    Validity
    --------
    One degree of freedom; q != 0.

    Verification
    ------------
    tests/unit/test_hamiltonian.py: conserved along the integrated orbit; Poisson-bracket algebra and the
    relation 2KH + 2Q^2 + m alpha = 0 at random phase-space points.
    """
    q_arr = np.asarray(q)  # no cast to float: the function must stay complex-step differentiable
    p_arr = np.asarray(p)
    h = p_arr**2 / (2.0 * m) + alpha / q_arr**2
    charge = -t * h + 0.5 * p_arr * q_arr
    k = t**2 * h + 2.0 * t * charge - 0.5 * m * q_arr**2
    return h, charge, k
