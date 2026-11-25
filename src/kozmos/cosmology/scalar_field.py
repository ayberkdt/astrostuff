"""Quintessence: a minimally coupled scalar field as dark energy.

Everything here follows [Copeland2006] (arXiv hep-th/0603057, TeX ``hepthv3.tex``), the subsection
"Quintessence" (lines 2565-2693) and the autonomous-system analysis (lines 3655-4059).

Units and conventions (the source's, with kappa^2 = 8 pi G, [Copeland2006] line 470):

* The field enters as the dimensionless ``phi = kappa phi_physical`` (reduced Planck units, kappa = 1) in the
  generic integrator, and the potential is given as ``v(phi) = kappa^2 V / (3 H0^2)``, i.e. V in units of the
  critical energy density today. Time is N = ln a (the number of e-folds), so ``dphi_dn = kappa phi_dot / H``.
* ``lam`` is the source's lambda = -V_phi / (kappa V): for V = V0 exp(-kappa lambda phi) the potential falls
  towards positive phi when lambda > 0. A negative lambda is the mirror image phi -> -phi.
* ``x = kappa phi_dot / (sqrt(6) H)`` and ``y = kappa sqrt(V) / (sqrt(3) H)`` ([Copeland2006] Eq. lamGam), with
  y >= 0 and the field kinetic sign epsilon = +1 (quintessence).
* ``w_background`` is w_m of the barotropic background fluid; the source's gamma is 1 + w_m.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from kozmos._sources import source
from kozmos.core._differentiation import gradient

Array = NDArray[np.float64]


@source("Copeland2006", where="text after Eq. (T_mu_nu): rho = phi_dot^2/2 + V (hepthv3.tex line 2614)", status="read")
def scalar_rho(phi_dot: ArrayLike, potential: ArrayLike) -> Array:
    """Compute the energy density of a homogeneous scalar field: rho = phi_dot^2 / 2 + V(phi).

    Parameters
    ----------
    phi_dot : float or ndarray
        Time derivative of the field, in units consistent with ``potential``.
    potential : float or ndarray
        V(phi) in the units of rho.

    Returns
    -------
    float or ndarray
        rho in the units of ``potential`` (no unit is imposed).

    References
    ----------
    [Copeland2006] hepthv3.tex line 2614.

    Validity
    --------
    Flat Friedmann background, minimally coupled field with canonical kinetic term (epsilon = +1).

    Verification
    ------------
    tests/unit/test_scalar_field.py: slow-roll and kinetic limits.
    """
    return 0.5 * np.asarray(phi_dot, dtype=float) ** 2 + np.asarray(potential, dtype=float)


@source("Copeland2006", where="text after Eq. (T_mu_nu): p = phi_dot^2/2 - V (hepthv3.tex line 2615)", status="read")
def scalar_p(phi_dot: ArrayLike, potential: ArrayLike) -> Array:
    """Compute the pressure of a homogeneous scalar field: p = phi_dot^2 / 2 - V(phi).

    Parameters
    ----------
    phi_dot : float or ndarray
        Time derivative of the field, in units consistent with ``potential``.
    potential : float or ndarray
        V(phi) in the units of p.

    Returns
    -------
    float or ndarray
        p in the units of ``potential``.

    References
    ----------
    [Copeland2006] hepthv3.tex line 2615.

    Validity
    --------
    As ``scalar_rho``.

    Verification
    ------------
    tests/unit/test_scalar_field.py: slow-roll and kinetic limits.
    """
    return 0.5 * np.asarray(phi_dot, dtype=float) ** 2 - np.asarray(potential, dtype=float)


@source(
    "Copeland2006",
    where="equation of state w_phi = (phi_dot^2 - 2V)/(phi_dot^2 + 2V) (hepthv3.tex lines 2662-2664)",
    status="read",
)
def scalar_eos(phi_dot: ArrayLike, potential: ArrayLike) -> Array:
    """Compute the equation of state of a scalar field: w_phi = (phi_dot^2 - 2 V) / (phi_dot^2 + 2 V).

    Parameters
    ----------
    phi_dot : float or ndarray
        Time derivative of the field, in units consistent with ``potential``.
    potential : float or ndarray
        V(phi) >= 0 in the units of phi_dot^2 / 2.

    Returns
    -------
    float or ndarray
        w_phi in [-1, 1], dimensionless.

    References
    ----------
    [Copeland2006] hepthv3.tex lines 2659-2665 (the source states -1 <= w_phi <= 1 and the limits
    w = -1 for slow roll, w = 1 for stiff matter).

    Validity
    --------
    V >= 0 is needed for the bounds; a negative V is not rejected.

    Verification
    ------------
    tests/unit/test_scalar_field.py: w = -1 for phi_dot = 0, w = 1 for V = 0.
    """
    kinetic = np.asarray(phi_dot, dtype=float) ** 2
    two_v = 2.0 * np.asarray(potential, dtype=float)
    return (kinetic - two_v) / (kinetic + two_v)


def _check_background(w_background: float) -> float:
    gamma = 1.0 + w_background
    if not 0.0 < gamma < 2.0:
        raise ValueError("w_background must satisfy -1 < w_background < 1 (0 < gamma < 2 in the source)")
    return gamma


@source(
    "Copeland2006",
    where="Eq. autoquin1, autoquin2 (hepthv3.tex lines 3831-3843) with epsilon = +1 and Gamma = 1 (line 3902)",
    status="read",
)
def exponential_quintessence_rhs(n: float, state: ArrayLike, lam: float, w_background: float) -> Array:
    """Evaluate the right-hand side of the autonomous system for V ∝ exp(-lambda phi).

    dx/dN = -3x + (sqrt(6)/2) lambda y^2 + (3/2) x [(1 - w_m) x^2 + (1 + w_m)(1 - y^2)]
    dy/dN = -(sqrt(6)/2) lambda x y     + (3/2) y [(1 - w_m) x^2 + (1 + w_m)(1 - y^2)]

    Parameters
    ----------
    n : float
        N = ln a; unused (the system is autonomous), kept for ``solve_ivp``.
    state : array_like
        (x, y).
    lam : float
        lambda = -V_phi/(kappa V), constant for the exponential potential.
    w_background : float
        Equation of state w_m of the background fluid.

    Returns
    -------
    ndarray
        (dx/dN, dy/dN).

    References
    ----------
    [Copeland2006] Eq. autoquin1 and autoquin2; constraint Eq. confine.

    Validity
    --------
    Constant lambda (exponential potential), epsilon = +1, flat universe, one barotropic fluid with constant w_m.
    The physical region is x^2 + y^2 <= 1, y >= 0.

    Verification
    ------------
    tests/unit/test_scalar_field.py: every fixed point is a zero; finite-difference Jacobian eigenvalues equal the
    source's list.
    """
    x, y = np.asarray(state, dtype=float)
    common = 1.5 * ((1.0 - w_background) * x**2 + (1.0 + w_background) * (1.0 - y**2))
    dx = -3.0 * x + 0.5 * np.sqrt(6.0) * lam * y**2 + x * common
    dy = -0.5 * np.sqrt(6.0) * lam * x * y + y * common
    return np.array([dx, dy])


@source(
    "Copeland2006", where="Eq. Omephiquin (hepthv3.tex line 3871): Omega_phi = x^2 + y^2 (epsilon = +1)", status="read"
)
def scalar_omega_phi(x: ArrayLike, y: ArrayLike) -> Array:
    """Compute the scalar field density parameter: Omega_phi = x^2 + y^2.

    Parameters
    ----------
    x, y : float or ndarray
        The variables x and y of the autonomous system.

    Returns
    -------
    float or ndarray
        Omega_phi = kappa^2 rho_phi / (3 H^2), dimensionless.

    References
    ----------
    [Copeland2006] Eq. Omephiquin.

    Validity
    --------
    epsilon = +1.

    Verification
    ------------
    tests/unit/test_scalar_field.py: fixed-point table values.
    """
    return np.asarray(x, dtype=float) ** 2 + np.asarray(y, dtype=float) ** 2


@source("Copeland2006", where="Eq. wphiquin (hepthv3.tex line 3865): w_phi = (x^2 - y^2)/(x^2 + y^2)", status="read")
def scalar_w_phi(x: ArrayLike, y: ArrayLike) -> Array:
    """Compute the scalar field equation of state from (x, y): w_phi = (x^2 - y^2) / (x^2 + y^2).

    Parameters
    ----------
    x, y : float or ndarray
        The variables x and y of the autonomous system.

    Returns
    -------
    float or ndarray
        w_phi, dimensionless; NaN at x = y = 0.

    References
    ----------
    [Copeland2006] Eq. wphiquin.

    Validity
    --------
    epsilon = +1.

    Verification
    ------------
    tests/unit/test_scalar_field.py: w_phi = -1 + lambda^2/3 at the scalar-dominated point.
    """
    xx = np.asarray(x, dtype=float) ** 2
    yy = np.asarray(y, dtype=float) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return (xx - yy) / (xx + yy)


@source(
    "Copeland2006", where="Eq. weffquin (hepthv3.tex line 3881): w_eff = w_m + (1-w_m) x^2 - (1+w_m) y^2", status="read"
)
def scalar_w_eff(x: ArrayLike, y: ArrayLike, w_background: float) -> Array:
    """Compute the total effective equation of state: w_eff = w_m + (1 - w_m) x^2 - (1 + w_m) y^2.

    Parameters
    ----------
    x, y : float or ndarray
        The variables x and y of the autonomous system.
    w_background : float
        Equation of state w_m of the background fluid.

    Returns
    -------
    float or ndarray
        (p_phi + p_m)/(rho_phi + rho_m), dimensionless; the universe accelerates for w_eff < -1/3.

    References
    ----------
    [Copeland2006] Eq. weffquin.

    Validity
    --------
    epsilon = +1, flat universe, one barotropic fluid.

    Verification
    ------------
    tests/unit/test_scalar_field.py: w_eff = w_phi at the scalar-dominated point and w_eff = w_m at the scaling
    point.
    """
    xx = np.asarray(x, dtype=float) ** 2
    yy = np.asarray(y, dtype=float) ** 2
    return w_background + (1.0 - w_background) * xx - (1.0 + w_background) * yy


@dataclass(frozen=True, slots=True)
class FixedPoint:
    """A fixed point of the exponential-potential system with the entries of the source's Table I.

    Attributes
    ----------
    name : str
        "a", "b1", "b2", "c" or "d" (the source's labels).
    x, y : float
        Coordinates.
    omega_phi : float
        Omega_phi = x^2 + y^2.
    w_phi : float
        Equation of state of the field; NaN for point (a), where Omega_phi = 0.
    w_eff : float
        Total effective equation of state.
    eigenvalues : tuple of complex
        The two eigenvalues mu_1, mu_2 of the source's list.
    stability : str
        "stable node", "unstable node", "saddle point", "stable spiral", "unstable spiral" or "non-hyperbolic".
    """

    name: str
    x: float
    y: float
    omega_phi: float
    w_phi: float
    w_eff: float
    eigenvalues: tuple[complex, complex]
    stability: str


_BOUNDARY_RTOL = 1e-12  # relative tolerance of the existence and stability boundaries (rounding is ~1e-16)


def _sign(value: float, scale: float) -> int:
    """Sign of ``value`` with zero for |value| <= _BOUNDARY_RTOL * scale (``scale``: size of the terms that built it)."""
    if abs(value) <= _BOUNDARY_RTOL * scale:
        return 0
    return 1 if value > 0.0 else -1


def _classify(eigenvalues: tuple[complex, complex], scales: tuple[float, float]) -> str:
    """Stability class by the classification (i)-(iv) of [Copeland2006] lines 3765-3777 (by the eigenvalues).

    ``scales`` are the magnitudes of the terms that built each eigenvalue; a real part that is zero within
    ``_BOUNDARY_RTOL`` times its scale makes the point "non-hyperbolic". The eigenvalues are complex only when the
    caller found a negative discriminant outside the tolerance (an exactly zero imaginary part means real).
    """
    (mu1, mu2), (s1, s2) = eigenvalues, scales
    signs = (_sign(mu1.real, s1), _sign(mu2.real, s2))
    if 0 in signs:
        return "non-hyperbolic"
    if mu1.imag != 0.0 or mu2.imag != 0.0:
        if signs == (-1, -1):
            return "stable spiral"
        if signs == (1, 1):
            return "unstable spiral"
        return "non-hyperbolic"
    if signs == (-1, -1):
        return "stable node"
    if signs == (1, 1):
        return "unstable node"
    return "saddle point"


@source(
    "Copeland2006",
    where="TABLE I (TeX label crit, hepthv3.tex lines 3909-3946) and the eigenvalue list (lines 3949-3987, "
    "Eq. musca for point d)",
    status="read",
    note="existence conditions: (a), (b1), (b2) for all lambda; (c) lambda^2 < 6; (d) lambda^2 > 3 gamma; the "
    "stability is derived from the eigenvalues with the classification of lines 3765-3777",
)
def exponential_fixed_points(lam: float, w_background: float) -> list[FixedPoint]:
    """List the fixed points of the exponential-potential system that exist for ``lam`` and ``w_background``.

    Points (the source's labels): (a) x = y = 0; (b1) x = 1, y = 0; (b2) x = -1, y = 0;
    (c) x = lambda/sqrt(6), y = sqrt(1 - lambda^2/6) for lambda^2 < 6; (d) x = sqrt(3/2) gamma/lambda,
    y = sqrt(3 (2 - gamma) gamma / (2 lambda^2)) for lambda^2 > 3 gamma, with gamma = 1 + w_m.

    Parameters
    ----------
    lam : float
        lambda = -V_phi/(kappa V).
    w_background : float
        Equation of state w_m of the background fluid, -1 < w_m < 1.

    Returns
    -------
    list of FixedPoint
        The points that exist, in the order a, b1, b2, c, d, each with Omega_phi, w_phi, w_eff, the two
        eigenvalues of the source and the stability class.

    Raises
    ------
    ValueError
        If w_background is outside (-1, 1).

    References
    ----------
    [Copeland2006] TABLE I and the eigenvalue list.

    Validity
    --------
    Boundaries are decided with a tolerance of 1e-12 relative to the size of the terms of each quantity (rounding
    is ~1e-16; before the fix an eigenvalue of 4e-16 gave "unstable node" or "saddle point" depending on the
    rounding of lambda). When lambda^2 is within that tolerance of 6 or of 3 gamma, where two points merge, (c)
    (for 6) and (d) (for 3 gamma) are treated as not existing and the surviving point has an eigenvalue that is
    zero within tolerance, hence "non-hyperbolic" ((b1) or (b2) at lambda = +-sqrt(6), (c) at lambda^2 = 3 gamma).
    At the spiral boundary lambda^2 = 24 gamma^2/(9 gamma - 2) the discriminant of (d) is zero within the
    tolerance, the eigenvalues are equal and real, and the class is "stable node" (a degenerate node); the
    spiral is decided from the sign of the discriminant itself, not from the imaginary part of a square root
    (which is ~1e-8 for a rounding-sized negative discriminant). The source's description of the stable spiral ("determinant of M negative")
    is inconsistent with its own eigenvalue formula (complex eigenvalues with negative real part have a
    positive determinant); the classification here uses the eigenvalues. The "unstable spiral" class is this
    project's name for complex eigenvalues with positive real part; the source lists only the stable spiral.

    Verification
    ------------
    tests/unit/test_scalar_field.py: zeros of the right-hand side, finite-difference Jacobian eigenvalues and the
    stability column of the table for several lambda; trajectories ending on the table's attractors.
    """
    gamma = _check_background(w_background)
    s6 = np.sqrt(6.0)
    # (name, x, y, eigenvalues, scales): a scale is the size of the terms that built the eigenvalue
    entries: list[tuple[str, float, float, tuple[complex, complex], tuple[float, float]]] = [
        ("a", 0.0, 0.0, (complex(-1.5 * (2.0 - gamma)), complex(1.5 * gamma)), (1.5 * (2.0 - gamma), 1.5 * gamma)),
        (
            "b1",
            1.0,
            0.0,
            (complex(3.0 - 0.5 * s6 * lam), complex(3.0 * (2.0 - gamma))),
            (3.0 + 0.5 * s6 * abs(lam), 3.0 * (2.0 - gamma)),
        ),
        (
            "b2",
            -1.0,
            0.0,
            (complex(3.0 + 0.5 * s6 * lam), complex(3.0 * (2.0 - gamma))),
            (3.0 + 0.5 * s6 * abs(lam), 3.0 * (2.0 - gamma)),
        ),
    ]
    if lam**2 < 6.0 * (1.0 - _BOUNDARY_RTOL):
        entries.append(
            (
                "c",
                lam / s6,
                float(np.sqrt(1.0 - lam**2 / 6.0)),
                (complex(0.5 * (lam**2 - 6.0)), complex(lam**2 - 3.0 * gamma)),
                (0.5 * lam**2 + 3.0, lam**2 + 3.0 * gamma),
            )
        )
    if lam**2 > 3.0 * gamma * (1.0 + _BOUNDARY_RTOL):
        term = 8.0 * gamma * (lam**2 - 3.0 * gamma) / (lam**2 * (2.0 - gamma))
        discriminant = 1.0 - term  # negative: complex eigenvalues (spiral); decided here, not from a square root
        if abs(discriminant) <= _BOUNDARY_RTOL * (1.0 + term):
            root = 0.0 + 0.0j  # the spiral boundary: equal real eigenvalues (a degenerate node)
        elif discriminant < 0.0:
            root = 1j * np.sqrt(-discriminant)
        else:
            root = complex(np.sqrt(discriminant))
        prefactor = -0.75 * (2.0 - gamma)
        entries.append(
            (
                "d",
                float(np.sqrt(1.5) * gamma / lam),
                float(np.sqrt(3.0 * (2.0 - gamma) * gamma / (2.0 * lam**2))),
                (complex(prefactor * (1.0 + root)), complex(prefactor * (1.0 - root))),
                (abs(prefactor) * (1.0 + abs(root)),) * 2,
            )
        )
    points = []
    for name, x, y, eigenvalues, scales in entries:
        points.append(
            FixedPoint(
                name=name,
                x=x,
                y=y,
                omega_phi=float(scalar_omega_phi(x, y)),
                w_phi=float(scalar_w_phi(x, y)),
                w_eff=float(scalar_w_eff(x, y, w_background)),
                eigenvalues=eigenvalues,
                stability=_classify(eigenvalues, scales),
            )
        )
    return points


@dataclass(frozen=True, slots=True)
class AutonomousSolution:
    """Trajectory of the autonomous system: arrays over N = ln a."""

    n: Array
    x: Array
    y: Array
    omega_phi: Array
    w_phi: Array
    w_eff: Array


@source(
    "Copeland2006",
    where="Eq. autoquin1, autoquin2 (hepthv3.tex lines 3831-3843); Eq. wphiquin, Omephiquin, weffquin",
    status="read",
)
def integrate_exponential_quintessence(
    x0: float,
    y0: float,
    lam: float,
    w_background: float,
    n_span: tuple[float, float],
    n_eval: ArrayLike | None = None,
    rtol: float = 1e-10,
) -> AutonomousSolution:
    """Integrate the autonomous system over N = ln a with ``scipy.integrate.solve_ivp`` (DOP853).

    Parameters
    ----------
    x0, y0 : float
        Initial values of x and y at N = n_span[0]; y0 >= 0 and x0^2 + y0^2 <= 1.
    lam : float
        lambda = -V_phi/(kappa V).
    w_background : float
        Equation of state w_m of the background fluid, -1 < w_m < 1.
    n_span : tuple of float
        (N_start, N_end); N_end < N_start integrates backwards in time.
    n_eval : array_like, optional
        N values at which to report; default a uniform grid of 400 points.
    rtol : float
        Relative tolerance, default 1e-10 (absolute tolerance 1e-13).

    Returns
    -------
    AutonomousSolution
        Arrays n, x, y, omega_phi, w_phi, w_eff.

    Raises
    ------
    ValueError
        If the initial point is outside the physical region or w_background is outside (-1, 1).
    RuntimeError
        If the integrator fails.

    References
    ----------
    [Copeland2006] Eq. autoquin1, autoquin2, wphiquin, Omephiquin, weffquin.

    Validity
    --------
    Constant lambda. The attractors are the fixed points of the table; the approach to a saddle is exponential
    in N with the eigenvalue of the source, so a trajectory started at a distance eps leaves after about
    ln(1/eps)/mu e-folds.

    Verification
    ------------
    tests/unit/test_scalar_field.py: trajectories end on the stable point with the Omega_phi and w_phi of the
    table; agreement with the generic integrator.
    """
    _check_background(w_background)
    if y0 < 0.0 or x0**2 + y0**2 > 1.0 + 1e-12:
        raise ValueError("the initial point must satisfy y >= 0 and x^2 + y^2 <= 1")
    n_eval_array = np.linspace(n_span[0], n_span[1], 400) if n_eval is None else np.asarray(n_eval, dtype=float)
    result = solve_ivp(
        lambda n, s: exponential_quintessence_rhs(n, s, lam, w_background),
        n_span,
        [x0, y0],
        method="DOP853",
        rtol=rtol,
        atol=1e-13,
        t_eval=n_eval_array,
    )
    if not result.success:
        raise RuntimeError(result.message)
    x, y = result.y
    return AutonomousSolution(
        n=result.t,
        x=x,
        y=y,
        omega_phi=scalar_omega_phi(x, y),
        w_phi=scalar_w_phi(x, y),
        w_eff=scalar_w_eff(x, y, w_background),
    )


@dataclass(frozen=True, slots=True)
class QuintessenceSolution:
    """Trajectory of the generic integrator: arrays over N = ln a.

    ``phi`` is kappa times the field, ``dphi_dn`` its derivative with respect to N, ``e`` is H/H0 with H0 the
    normalization of the potential, ``omega_phi`` is kappa^2 rho_phi/(3 H^2) and ``w_phi`` the field equation
    of state.
    """

    n: Array
    phi: Array
    dphi_dn: Array
    e: Array
    omega_phi: Array
    w_phi: Array


@source(
    "Copeland2006",
    where="Eq. phieqF (hepthv3.tex line 2586), Eq. H2 (line 2622), rho and p (lines 2614-2615), Eq. ddota "
    "(line 2625); Hdot = -(kappa^2/2)[phi_dot^2 + (1+w_m) rho_m] (line 3806)",
    status="read",
    note="derivation in this project: the equations in N = ln a with phi_tilde = kappa phi, v = kappa^2 V/(3 H0^2), "
    "E = H/H0: E^2 = (v + Om a^-3 + Or a^-4)/(1 - phi'^2/6); phi'' = -(3 + H'/H) phi' - 3 v_phi/E^2; "
    "H'/H = -phi'^2/2 - (3/2) Omega_m - 2 Omega_r",
)
def integrate_quintessence(
    potential: Callable[[ArrayLike], ArrayLike],
    phi_start: float,
    dphi_dn_start: float,
    n_span: tuple[float, float],
    omega_matter_today: float = 0.0,
    omega_radiation_today: float = 0.0,
    potential_derivative: Callable[[float], float] | None = None,
    n_eval: ArrayLike | None = None,
    rtol: float = 1e-10,
) -> QuintessenceSolution:
    """Integrate the Klein-Gordon and Friedmann equations of a quintessence field over N = ln a.

    State (phi, dphi/dN) with phi = kappa phi_physical. With v(phi) = kappa^2 V/(3 H0^2):

    E^2 = [v(phi) + Om exp(-3N) + Or exp(-4N)] / (1 - phi'^2/6)
    phi'' = -(3 + H'/H) phi' - 3 v'(phi) / E^2, H'/H = -phi'^2/2 - (3/2) Omega_m(N) - 2 Omega_r(N)

    Parameters
    ----------
    potential : callable
        v(phi) = kappa^2 V(phi) / (3 H0^2): the potential in units of the critical energy density 3 H0^2/kappa^2
        (a constant v is a cosmological constant with Omega_Lambda = v). Must accept a float.
    phi_start : float
        phi = kappa phi_physical at N = n_span[0].
    dphi_dn_start : float
        d(phi)/dN = kappa phi_dot / H at N = n_span[0]; |dphi_dn_start| < sqrt(6).
    n_span : tuple of float
        (N_start, N_end); N = 0 is a = 1 (the normalization of the matter and radiation terms).
    omega_matter_today, omega_radiation_today : float
        Matter and radiation density in units of 3 H0^2/kappa^2 at a = 1.
    potential_derivative : callable, optional
        dv/dphi; default is the complex-step derivative of ``potential`` from ``kozmos.core._differentiation``,
        which needs an analytic function (use numpy functions, no ``abs``).
    n_eval : array_like, optional
        N values at which to report; default a uniform grid of 400 points.
    rtol : float
        Relative tolerance, default 1e-10 (absolute tolerance 1e-13).

    Returns
    -------
    QuintessenceSolution
        Arrays n, phi, dphi_dn, e, omega_phi, w_phi. ``e`` equals 1 at a = 1 only if the initial data close the
        Friedmann constraint there; the function does not tune them.

    Raises
    ------
    ValueError
        If |dphi_dn_start| >= sqrt(6), the matter or radiation density is negative, or the total
        v(phi) + rho_m + rho_r is not positive at some N of the integration (then E^2 < 0 has no solution; a
        negative potential is allowed as long as the total stays positive).
    RuntimeError
        If the integrator fails or phi'^2 reaches 6 (E^2 would diverge).

    References
    ----------
    [Copeland2006] Eq. phieqF, Eq. H2, the expressions of rho and p, and the Hdot equation for a field plus
    a fluid.

    Validity
    --------
    Flat universe, canonical field, matter plus radiation as the only other components, no coupling.
    H0 is only the unit of the potential; E(a = 1) = 1 requires tuning the initial data. Energies are
    dimensionless, so no physical units are involved.

    Verification
    ------------
    tests/unit/test_scalar_field.py: a constant potential reproduces lambda-CDM E(z) of
    kozmos.cosmology.distances.e_function; the exponential potential agrees with the autonomous system; an
    independent integration in cosmic time with matter, radiation and a non-constant potential (the radiation
    term of H'/H); ValueError for v = -0.5 with Omega_m = 0.3 (formerly NaN with a numpy warning).
    """
    if abs(dphi_dn_start) >= np.sqrt(6.0):
        raise ValueError("|dphi_dn_start| must be smaller than sqrt(6)")
    if omega_matter_today < 0.0 or omega_radiation_today < 0.0:
        raise ValueError("the matter and radiation densities must not be negative")

    def derivative(phi: float) -> float:
        if potential_derivative is not None:
            return float(potential_derivative(phi))
        return float(gradient(lambda p: potential(p[0]), [phi])[0])

    def background(n: float, phi: float, dphi: float) -> tuple[float, float, float, float]:
        rho_m = omega_matter_today * np.exp(-3.0 * n)
        rho_r = omega_radiation_today * np.exp(-4.0 * n)
        total = float(potential(phi)) + rho_m + rho_r
        if not total > 0.0:  # also catches NaN
            raise ValueError(
                f"the total energy density v(phi) + rho_m + rho_r = {total:.6g} is not positive at N = {n:.6g}"
            )
        if not dphi**2 < 6.0:
            raise RuntimeError(f"phi'^2 = {dphi**2:.6g} reached 6 at N = {n:.6g}: the kinetic energy exceeds E^2")
        e2 = total / (1.0 - dphi**2 / 6.0)
        return e2, rho_m / e2, rho_r / e2, -0.5 * dphi**2 - 1.5 * rho_m / e2 - 2.0 * rho_r / e2

    def rhs(n: float, y: Array) -> list[float]:
        e2, _, _, hprime = background(n, y[0], y[1])
        return [y[1], -(3.0 + hprime) * y[1] - 3.0 * derivative(y[0]) / e2]

    n_eval_array = np.linspace(n_span[0], n_span[1], 400) if n_eval is None else np.asarray(n_eval, dtype=float)
    result = solve_ivp(
        rhs, n_span, [phi_start, dphi_dn_start], method="DOP853", rtol=rtol, atol=1e-13, t_eval=n_eval_array
    )
    if not result.success:
        raise RuntimeError(result.message)
    phi, dphi = result.y
    e2 = np.empty_like(phi)
    for i in range(phi.size):
        e2[i] = background(result.t[i], phi[i], dphi[i])[0]
    v = np.array([float(potential(p)) for p in phi])
    kinetic = e2 * dphi**2
    return QuintessenceSolution(
        n=result.t,
        phi=phi,
        dphi_dn=dphi,
        e=np.sqrt(e2),
        omega_phi=(kinetic / 6.0 + v) / e2,
        w_phi=(kinetic - 6.0 * v) / (kinetic + 6.0 * v),
    )
