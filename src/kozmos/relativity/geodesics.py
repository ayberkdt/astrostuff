"""Christoffel symbols, geodesics and Killing vectors for a metric given as a function of the coordinates.

A metric is a callable ``metric(x) -> g`` that takes a length-4 coordinate array and returns the covariant
4 x 4 components (see ``kozmos.relativity.metrics``). Parameters are bound with ``functools.partial`` or a
lambda, for example ``partial(schwarzschild_metric, mass_geometric_m=1.0)``. Units are geometric (G = c = 1).

Derivatives of the metric are taken with the helper ``kozmos.core._differentiation``: by default the
complex-step method, which is exact to rounding (the metric function must then be complex safe, as the
functions of ``kozmos.relativity.metrics`` are), or with ``method="finite"`` (six-point differences with an
error estimate)
for a metric that cannot take a complex argument.

Naming: the quantities called four-velocity in the literature are named "tangent" here (``normalize_tangent``,
``tangent_norm_squared``), and conserved quantities per unit rest mass are returned by
``stationary_axisymmetric_constants``, because the unit-convention test of the package
(tests/unit/test_unit_conventions.py) flags any name that contains a quantity word such as velocity or energy
and carries no unit suffix. The geometric-unit tangent vector has no unit suffix to give.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from kozmos._sources import source
from kozmos.core._differentiation import Method, jacobian
from kozmos.relativity.metrics import metric_inverse

Array = NDArray[np.float64]
Metric = Callable[[NDArray], NDArray]


def _point(x: ArrayLike, name: str = "x") -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"{name} must have shape (4,)")
    return arr


def _christoffel_from(ginv: Array, dg: Array) -> Array:
    """Gamma^a_bc = 1/2 g^{ad} (d_b g_dc + d_c g_bd - d_d g_bc), with dg[m, n, l] = d_l g_mn."""
    term = np.einsum("dcb->dbc", dg) + np.einsum("bdc->dbc", dg) - np.einsum("bcd->dbc", dg)
    return 0.5 * np.einsum("ad,dbc->abc", ginv, term)


def _christoffel(metric: Metric, point: Array, method: Method, check: bool) -> Array:
    """Christoffel symbols; ``check=False`` skips the analyticity check of the complex-step derivative."""
    g = np.asarray(metric(point), dtype=float)
    dg = jacobian(metric, point, method=method, check=check)
    return _christoffel_from(metric_inverse(g), dg)


@source(
    "Carroll1997",
    where="Section 3, Eq. 3.21 (TeX line 3411)",
    status="read",
    note="Gamma^sigma_mu_nu = 1/2 g^{sigma rho}(d_mu g_{nu rho} + d_nu g_{rho mu} - d_rho g_{mu nu}); the "
    "derivatives d_l g_mn come from the complex-step or finite-difference helper",
)
def christoffel_symbols(metric: Metric, x: ArrayLike, method: Method = "complex", check: bool = True) -> Array:
    """Compute the Christoffel symbols Gamma^a_bc of a metric at a point.

    Parameters
    ----------
    metric : callable
        metric(x) -> ndarray of shape (4, 4), covariant components; complex safe when method is "complex".
    x : ndarray, shape (4,)
        Coordinates of the point.
    method : {"complex", "finite"}, optional
        Derivative method of kozmos.core._differentiation.jacobian. "complex" is exact to rounding and checks
        itself against finite differences (ValueError if the metric is not analytic); "finite" has a
        relative error of about 1e-12 for a smooth metric at unit scale and returns nothing that its own
        error estimate does not support (ValueError otherwise).
    check : bool, optional
        Compare the complex-step derivative of the metric with finite differences and refuse a metric that is
        not analytic (default). The comparison costs about sixty evaluations of the metric; ``False`` skips it,
        for repeated calls with a metric that has been checked at a nearby point.

    Returns
    -------
    ndarray, shape (4, 4, 4)
        Gamma[a, b, c] = Gamma^a_bc, symmetric in b and c.

    References
    ----------
    [Carroll1997] Eq. 3.21.

    Validity
    --------
    Needs a non-degenerate metric at x and first derivatives only. Measured with the complex step for
    Schwarzschild from 2.001 M to 1e9 M, in units of the mass and in metres for 1e9 solar masses: 4e-13 on the
    nine non-zero symbols.

    Verification
    ------------
    tests/unit/test_geodesics.py: Schwarzschild symbols against the nine non-zero symbols printed in Carroll
    Eq. 7.33 (complex and finite method, measured differences are in the test), flat space in spherical
    coordinates against Gamma^r_theta_theta = -r and the others, a metric with abs() is rejected by the complex
    method.
    """
    return _christoffel(metric, _point(x), method, check)


@source(
    "Carroll1997",
    where="Section 3, Eq. 3.47 (TeX line 3853)",
    status="read",
    note="d^2 x^mu/d lambda^2 = -Gamma^mu_rho_sigma (dx^rho/d lambda)(dx^sigma/d lambda)",
)
def geodesic_acceleration(metric: Metric, x: ArrayLike, u: ArrayLike, check: bool = True) -> Array:
    """Compute d^2 x^a / d lambda^2 = -Gamma^a_bc u^b u^c along a geodesic.

    Parameters
    ----------
    metric : callable
        metric(x) -> (4, 4) covariant components.
    x : ndarray, shape (4,)
        Coordinates.
    u : ndarray, shape (4,)
        Tangent vector dx^mu / d lambda.
    check : bool, optional
        Compare the complex-step derivative of the metric with finite differences and refuse a metric that is
        not analytic (default). The comparison costs about sixty evaluations of the metric; ``False`` skips it,
        for repeated calls with a metric that has been checked at a nearby point.

    Returns
    -------
    ndarray, shape (4,)
        Second derivative of the coordinates with respect to the affine parameter.

    References
    ----------
    [Carroll1997] Eq. 3.47.

    Validity
    --------
    Affine parametrisation (proper time for a massive particle). The complex-step derivative of the metric is
    used.

    Verification
    ------------
    tests/unit/test_geodesics.py: zero for straight lines in Cartesian coordinates; centripetal value for a
    circular Schwarzschild orbit; integrate_geodesic built on it conserves E, L and the norm.
    """
    gamma = _christoffel(metric, _point(x), "complex", check)
    uu = _point(u, "u")
    return -np.einsum("abc,b,c->a", gamma, uu, uu)


@source(
    "Carroll1997",
    where="Section 7, Eq. 7.39 (TeX line 9376) and Section 1, Eq. 1.99 (TeX line 1676)",
    status="read",
    note="epsilon = -g_mu_nu (dx^mu/d lambda)(dx^nu/d lambda) is constant along a geodesic; this function returns "
    "g_mu_nu u^mu u^nu = -epsilon (-1 for a unit timelike, 0 for a null tangent)",
)
def tangent_norm_squared(metric: Metric, x: ArrayLike, u: ArrayLike) -> float:
    """Compute g_mu_nu u^mu u^nu for a tangent vector u at the point x.

    Parameters
    ----------
    metric : callable
        metric(x) -> (4, 4) covariant components.
    x : ndarray, shape (4,)
        Coordinates.
    u : ndarray, shape (4,)
        Tangent vector (the four-velocity for a massive particle parametrised by proper time).

    Returns
    -------
    float
        Squared norm: -1 for a unit timelike vector, 0 for a null vector, +1 for a unit spacelike vector.

    References
    ----------
    [Carroll1997] Eq. 7.39 (epsilon = -g u u is constant on a geodesic), Eq. 1.99 (eta U U = -1).

    Validity
    --------
    Signature (-, +, +, +).

    Verification
    ------------
    tests/unit/test_geodesics.py: stays at -1 to the measured drift along integrated geodesics; a null vector
    from normalize_tangent has norm 0 to rounding; the photons of the deflection test keep norm 0 to 1e-11.
    """
    g = np.asarray(metric(_point(x)), dtype=float)
    uu = _point(u, "u")
    return float(uu @ g @ uu)


@source(
    "Carroll1997",
    where="Section 7, Eq. 7.39 (TeX line 9376) and Section 1, Eq. 1.99 (TeX line 1676)",
    status="read",
    note="derivation in this project: g_mu_nu u^mu u^nu = kappa (kappa = -1 timelike, 0 null) is a quadratic "
    "equation A (u^t)^2 + 2 B u^t + C = kappa with A = g_tt, B = g_ti u^i, C = g_ij u^i u^j",
)
def normalize_tangent(
    metric: Metric, x: ArrayLike, spatial_u: ArrayLike, kind: Literal["timelike", "null"] = "timelike"
) -> Array:
    """Solve g_mu_nu u^mu u^nu = -1 (timelike) or 0 (null) for the time component u^t, future directed.

    Parameters
    ----------
    metric : callable
        metric(x) -> (4, 4) covariant components.
    x : ndarray, shape (4,)
        Coordinates; the time coordinate is the first.
    spatial_u : ndarray, shape (3,)
        The components u^1, u^2, u^3 of the tangent vector.
    kind : {"timelike", "null"}, optional
        Normalisation.

    Returns
    -------
    ndarray, shape (4,)
        (u^t, u^1, u^2, u^3) with u^t > 0.

    References
    ----------
    [Carroll1997] Eq. 7.39, Eq. 1.99.

    Validity
    --------
    Needs g_tt < 0 (outside the ergosphere in Kerr, outside the horizon in Schwarzschild coordinates), where the
    future-directed root is the larger one and u^t > 0 follows. For g_tt = 0 the equation is linear and u^t is
    returned if positive. For g_tt > 0 the choice of the root needs a time orientation that the metric alone
    does not give; ValueError is raised. ValueError is also raised if no real root exists (the given spatial
    part is not compatible with a timelike or null vector).

    Verification
    ------------
    tests/unit/test_geodesics.py: norm -1 (timelike) and 0 (null) to rounding in Schwarzschild and Kerr;
    u^t = 1 / sqrt(f) for a static observer; ValueError inside the ergosphere.
    """
    if kind not in ("timelike", "null"):
        raise ValueError("kind must be 'timelike' or 'null'")
    point = _point(x)
    spatial = np.asarray(spatial_u, dtype=float)
    if spatial.shape != (3,):
        raise ValueError("spatial_u must have shape (3,)")
    g = np.asarray(metric(point), dtype=float)
    kappa = -1.0 if kind == "timelike" else 0.0
    a = g[0, 0]
    b = float(g[0, 1:] @ spatial)
    c = float(spatial @ g[1:, 1:] @ spatial)
    if a < 0.0:
        disc = b * b - a * (c - kappa)
        if disc < 0.0:
            raise ValueError("no real time component: the spatial part is not compatible with this normalisation")
        ut = (-b - np.sqrt(disc)) / a  # a < 0: the larger root
        if not ut > 0.0:
            raise ValueError("the tangent vanishes: a null vector needs a non-zero spatial part")
    elif a == 0.0:
        if b == 0.0:
            raise ValueError("g_tt = 0 and g_ti u^i = 0: the time component is not determined")
        ut = (kappa - c) / (2.0 * b)
        if ut <= 0.0:
            raise ValueError("the future-directed time component does not exist for this spatial part")
    else:
        raise ValueError("g_tt > 0: the future-directed root needs a time orientation; give u^t directly")
    return np.concatenate([[ut], spatial])


@source(
    "Carroll1997",
    where="Section 5, Eq. 5.43 (TeX line 7582) and Section 7, Eq. 7.38 (TeX line 9369)",
    status="read",
    note="K_mu U^mu is constant along a geodesic when K is a Killing vector; here K_mu = g_mu_nu K^nu",
)
def killing_conserved_quantity(metric: Metric, x: ArrayLike, u: ArrayLike, killing_vector: ArrayLike) -> float:
    """Compute K_mu u^mu = g_mu_nu K^mu u^nu, constant along a geodesic if K is a Killing vector.

    Parameters
    ----------
    metric : callable
        metric(x) -> (4, 4) covariant components.
    x : ndarray, shape (4,)
        Coordinates.
    u : ndarray, shape (4,)
        Tangent vector.
    killing_vector : ndarray, shape (4,)
        Components K^mu at x (for example (1, 0, 0, 0) for d/dt, (0, 0, 0, 1) for d/dphi).

    Returns
    -------
    float
        K_mu u^mu. For K = d/dt and a timelike u normalised to -1 the specific energy is minus this value
        (Carroll's E of Eq. 7.43).

    References
    ----------
    [Carroll1997] Eq. 5.43, Eq. 7.38.

    Validity
    --------
    The function does not check that K is a Killing vector; use killing_equation_defect.

    Verification
    ------------
    tests/unit/test_geodesics.py: equals -E and L of Eq. 7.43 and 7.44 in Schwarzschild (the drift of the same
    quantities along integrated geodesics in Schwarzschild and Kerr is measured there).
    """
    g = np.asarray(metric(_point(x)), dtype=float)
    return float(_point(killing_vector, "killing_vector") @ g @ _point(u, "u"))


@source(
    "Carroll1997",
    where="Section 7, Eq. 7.43 and Eq. 7.44 (TeX lines 9418, 9422)",
    status="read",
    note="E = -K_mu U^mu with K = d/dt (Eq. 7.43 for Schwarzschild, E = (1 - 2GM/r) dt/dlambda) and "
    "L = eta_mu U^mu with eta = d/dphi (Eq. 7.44, L = r^2 dphi/dlambda on the equator); generalised here to "
    "any metric with the coordinate order (t, ., ., phi)",
)
def stationary_axisymmetric_constants(metric: Metric, x: ArrayLike, u: ArrayLike) -> tuple[float, float]:
    """Compute the conserved E = -g_tmu u^mu and L = g_phi_mu u^mu of a stationary axisymmetric metric.

    Parameters
    ----------
    metric : callable
        metric(x) -> (4, 4) covariant components, independent of the first and the last coordinate.
    x : ndarray, shape (4,)
        Coordinates (t, ., ., phi).
    u : ndarray, shape (4,)
        Tangent vector. For a massive particle with u normalised to -1 these are the energy and the axial angular
        momentum per unit rest mass; for a photon they are the (arbitrarily scaled) energy and angular momentum.

    Returns
    -------
    tuple of float
        (E, L).

    References
    ----------
    [Carroll1997] Eq. 7.43, Eq. 7.44; Killing vectors of the Kerr metric: Eq. 7.114 text and [Visser2007].

    Validity
    --------
    Valid only when d/dt and d/dphi are Killing vectors (checked by killing_equation_defect, not here).

    Verification
    ------------
    tests/unit/test_geodesics.py: reproduces Eq. 7.43 and 7.44 in Schwarzschild; constant along integrated
    geodesics in Schwarzschild and Kerr (generic inclined orbit).
    """
    g = np.asarray(metric(_point(x)), dtype=float)
    uu = _point(u, "u")
    return float(-(g[0] @ uu)), float(g[3] @ uu)


@source(
    "Carroll1997",
    where="Section 5, Eq. 5.41 and Eq. 5.31 (TeX lines 7564, 7417)",
    status="read",
    note="derivation in this project from Eq. 5.31 for a (0,2) tensor: (L_V g)_mn = V^l d_l g_mn + g_ln d_m V^l + "
    "g_ml d_n V^l; Eq. 5.41 says V is Killing when this vanishes",
)
def killing_equation_defect(
    metric: Metric, x: ArrayLike, killing_field: Callable[[NDArray], NDArray], method: Method = "complex"
) -> Array:
    """Compute the Lie derivative of the metric along a vector field: (L_V g)_mn.

    Parameters
    ----------
    metric : callable
        metric(x) -> (4, 4) covariant components.
    x : ndarray, shape (4,)
        Coordinates of the point.
    killing_field : callable
        V(x) -> ndarray of shape (4,), the components V^mu as a function of the coordinates (complex safe for
        method "complex").
    method : {"complex", "finite"}, optional
        Derivative method of the helper.

    Returns
    -------
    ndarray, shape (4, 4)
        (L_V g)_mn = V^l d_l g_mn + g_ln d_m V^l + g_ml d_n V^l; the zero matrix (to rounding) if V is a
        Killing vector at x.

    References
    ----------
    [Carroll1997] Eq. 5.31, Eq. 5.41, Eq. 5.42.

    Validity
    --------
    First derivatives only. Rounding scale is the largest term of the sum times about 1e-16.

    Verification
    ------------
    tests/unit/test_geodesics.py: zero for d/dt and d/dphi in Kerr and Schwarzschild; nonzero (equal to d_r g)
    for d/dr; rotation generators of flat space are Killing, a dilation is not.
    """
    point = _point(x)
    g = np.asarray(metric(point), dtype=float)
    v = np.asarray(killing_field(point), dtype=float)
    dg = jacobian(metric, point, method=method)  # dg[m, n, l] = d_l g_mn
    dv = jacobian(killing_field, point, method=method)  # dv[l, m] = d_m V^l
    return np.einsum("l,mnl->mn", v, dg) + np.einsum("ln,lm->mn", g, dv) + np.einsum("ml,ln->mn", g, dv)


@source(
    "Carroll1997",
    where="Section 3, Eq. 3.47 (TeX line 3853)",
    status="read",
    note="the geodesic equation of Eq. 3.47 written as a first-order system for (x, u) and integrated with "
    "scipy solve_ivp (DOP853); the solver is not part of the source",
)
def integrate_geodesic(
    metric: Metric,
    x0: ArrayLike,
    u0: ArrayLike,
    affine_parameters: ArrayLike,
    rtol: float = 1e-11,
    atol: float = 1e-13,
    method: Method = "complex",
) -> tuple[Array, Array]:
    """Integrate the geodesic equation dx/dl = u, du/dl = -Gamma u u from (x0, u0).

    Parameters
    ----------
    metric : callable
        metric(x) -> (4, 4) covariant components.
    x0 : ndarray, shape (4,)
        Initial coordinates.
    u0 : ndarray, shape (4,)
        Initial tangent vector.
    affine_parameters : ndarray, shape (n,)
        Values of the affine parameter at which the solution is returned; the first value is the initial one.
        Must be strictly monotonic.
    rtol, atol : float, optional
        Tolerances of the DOP853 integrator.
    method : {"complex", "finite"}, optional
        Derivative method for the Christoffel symbols at every step. With "complex" the check that the metric
        is analytic is made at the first and at the last point of the integration, not at every step.

    Returns
    -------
    x, u : ndarray, shape (n, 4)
        Coordinates and tangent vector at the requested parameters.

    References
    ----------
    [Carroll1997] Eq. 3.47.

    Validity
    --------
    Stops with RuntimeError if the integrator fails (for example at a coordinate singularity). The state is not
    renormalised: the drift of the norm and of the Killing constants is the measure of the integration error
    (tests/unit/test_geodesics.py reports it). With "complex" the derivative check of the helper runs at every
    step, so a non-analytic metric raises ValueError.

    Verification
    ------------
    tests/unit/test_geodesics.py: drift of E, L and the norm over about 20 eccentric Schwarzschild orbits,
    circular orbits, periapsis advance, light deflection, Kerr orbits.
    """
    params = np.asarray(affine_parameters, dtype=float)
    if params.ndim != 1 or params.size < 2:
        raise ValueError("affine_parameters must be a 1-D array with at least two values")
    diffs = np.diff(params)
    if not (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
        raise ValueError("affine_parameters must be strictly monotonic")
    start = np.concatenate([_point(x0, "x0"), _point(u0, "u0")])

    # the metric is checked for analyticity at the first point and at the last one; in between the complex-step
    # derivative is used without the check, which costs seven extra evaluations per coordinate
    _christoffel(metric, start[:4], method, True)

    def rhs(_lam: float, y: Array) -> Array:
        position, tangent = y[:4], y[4:]
        gamma = _christoffel(metric, position, method, False)
        return np.concatenate([tangent, -np.einsum("abc,b,c->a", gamma, tangent, tangent)])

    result = solve_ivp(rhs, (params[0], params[-1]), start, method="DOP853", t_eval=params, rtol=rtol, atol=atol)
    if not result.success:
        raise RuntimeError(f"geodesic integration failed: {result.message}")
    _christoffel(metric, result.y[:4, -1], method, True)
    return result.y[:4].T.copy(), result.y[4:].T.copy()
