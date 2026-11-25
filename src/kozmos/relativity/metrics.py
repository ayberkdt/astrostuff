"""Spacetime metrics in geometric units (G = c = 1), signature (-, +, +, +).

Every function takes the coordinates as a length-4 array ``x`` and returns the covariant components
g_{mu nu}(x) as a 4 x 4 array. The mass of a black hole enters as the geometric mass G M / c^2, which is a
length: it is passed in metres with the parameter name ending in ``_m`` (``mass_geometric_m``), and so are the
Kerr parameter a = J / (M c) (``spin_a_m``) and the charge parameter of Reissner-Nordstrom (``charge_m``).
Coordinates that are lengths are in the same unit, so the functions are scale free: the unit only has to be the
same everywhere.

Complex safety: the functions use only analytic numpy operations and build their result with
``dtype = result_type(x, float)``, so that they can be called with a complex argument. This is what the
complex-step derivative of ``kozmos.core._differentiation`` needs (see ``kozmos.relativity.geodesics``).

Sources: Carroll [Carroll1997] for Minkowski, Schwarzschild, Eddington-Finkelstein, Reissner-Nordstrom, Kerr and
Robertson-Walker; Visser [Visser2007] for Kerr in Boyer-Lindquist and in Kerr's original (advanced
Eddington-Finkelstein) coordinates; Abramowicz and Fragile [Abramowicz2013] for the Kerr components.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source

Array = NDArray[np.float64]


def _coordinates(x: ArrayLike) -> NDArray:
    """Return the coordinates as an array and check that there are four of them (complex input is kept)."""
    arr = np.asarray(x)
    if arr.shape != (4,):
        raise ValueError("the coordinates must be an array of shape (4,)")
    return arr


def _zeros(x: NDArray) -> NDArray:
    return np.zeros((4, 4), dtype=np.result_type(x, float))


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.8 (TeX line 447)",
    status="read",
    note="eta_mu_nu = diag(-1, 1, 1, 1); the coordinates are (t, x, y, z)",
)
def minkowski_metric(x: ArrayLike) -> NDArray:
    """Return the Minkowski metric in Cartesian coordinates (t, x, y, z): diag(-1, 1, 1, 1).

    Parameters
    ----------
    x : ndarray, shape (4,)
        Coordinates (t, x, y, z); only the dtype is used (the metric does not depend on position).

    Returns
    -------
    ndarray, shape (4, 4)
        g_{mu nu}.

    References
    ----------
    [Carroll1997] Eq. 1.8.

    Validity
    --------
    Inertial Cartesian coordinates, geometric units. Complex input gives a complex result.

    Verification
    ------------
    tests/unit/test_curvature.py: the Riemann tensor of this metric vanishes; tests/unit/test_geodesics.py:
    geodesics are straight lines.
    """
    arr = _coordinates(x)
    g = _zeros(arr)
    g[0, 0] = -1.0
    g[1, 1] = g[2, 2] = g[3, 3] = 1.0
    return g


@source(
    "Carroll1997",
    where="Section 7, Eq. 7.29 (TeX line 9217)",
    status="read",
    note="G = c = 1; mass_geometric_m is G M / c^2; ds^2 = -(1 - 2M/r) dt^2 + dr^2/(1 - 2M/r) + r^2 dOmega^2",
)
def schwarzschild_metric(x: ArrayLike, mass_geometric_m: float) -> NDArray:
    """Return the Schwarzschild metric in Schwarzschild coordinates (t, r, theta, phi).

    Parameters
    ----------
    x : ndarray, shape (4,)
        Coordinates (t, r, theta, phi); t and r in the unit of the geometric mass, angles in radians.
    mass_geometric_m : float
        Geometric mass M = G M_kg / c^2 [m]. Not checked for sign; r must differ from 2 M.

    Returns
    -------
    ndarray, shape (4, 4)
        g_{mu nu}: diag(-(1 - 2M/r), 1/(1 - 2M/r), r^2, r^2 sin^2 theta).

    References
    ----------
    [Carroll1997] Eq. 7.29.

    Validity
    --------
    Outside the horizon (r > 2 M) the coordinates are regular; the coordinate singularity at r = 2 M and the
    curvature singularity at r = 0 are not guarded. For horizon crossing use
    schwarzschild_eddington_finkelstein_metric.

    Verification
    ------------
    tests/unit/test_curvature.py: Ricci tensor vanishes and Kretschmann scalar equals 48 M^2 / r^6;
    tests/unit/test_geodesics.py: circular orbits, conservation laws, periapsis advance, light deflection.
    """
    arr = _coordinates(x)
    r, theta = arr[1], arr[2]
    f = 1.0 - 2.0 * mass_geometric_m / r
    g = _zeros(arr)
    g[0, 0] = -f
    g[1, 1] = 1.0 / f
    g[2, 2] = r**2
    g[3, 3] = r**2 * np.sin(theta) ** 2
    return g


@source(
    "Carroll1997",
    where="Section 7, Eq. 7.69 (TeX line 9861)",
    status="read",
    note="Eq. 7.69 with u~ = t + r*: ds^2 = -(1 - 2M/r) du~^2 + (du~ dr + dr du~) + r^2 dOmega^2",
)
def schwarzschild_eddington_finkelstein_metric(x: ArrayLike, mass_geometric_m: float) -> NDArray:
    """Return the Schwarzschild metric in Eddington-Finkelstein coordinates (u, r, theta, phi).

    Parameters
    ----------
    x : ndarray, shape (4,)
        Coordinates (u, r, theta, phi) with u = t + r* of Eq. 7.68; r > 0.
    mass_geometric_m : float
        Geometric mass G M / c^2 [m].

    Returns
    -------
    ndarray, shape (4, 4)
        g_{mu nu} with g_uu = -(1 - 2M/r), g_ur = g_ru = 1, g_theta_theta = r^2, g_phi_phi = r^2 sin^2 theta.

    References
    ----------
    [Carroll1997] Eq. 7.68, Eq. 7.69.

    Validity
    --------
    Regular at r = 2 M (the determinant is -r^4 sin^2 theta, Eq. 7.70); singular only at r = 0 and on the
    axis. The coordinates cover the region where future-directed ingoing paths cross the horizon.

    Verification
    ------------
    tests/unit/test_curvature.py: Kretschmann scalar at the horizon is the finite 48 M^2 / (2 M)^6;
    tests/unit/test_geodesics.py: an infalling geodesic crosses r = 2 M with finite acceleration.
    """
    arr = _coordinates(x)
    r, theta = arr[1], arr[2]
    g = _zeros(arr)
    g[0, 0] = -(1.0 - 2.0 * mass_geometric_m / r)
    g[0, 1] = g[1, 0] = 1.0
    g[2, 2] = r**2
    g[3, 3] = r**2 * np.sin(theta) ** 2
    return g


@source(
    "Carroll1997",
    where="Section 7, Eq. 7.110 and Eq. 7.111 (TeX lines 10602, 10607)",
    status="read",
    note="Delta = 1 - 2M/r + (p^2 + q^2)/r^2 with G = 1; charge_m^2 stands for p^2 + q^2 (electric and "
    "magnetic charge enter the metric in the same way, as the text says)",
)
def reissner_nordstrom_metric(x: ArrayLike, mass_geometric_m: float, charge_m: float) -> NDArray:
    """Return the Reissner-Nordstrom metric in coordinates (t, r, theta, phi).

    Parameters
    ----------
    x : ndarray, shape (4,)
        Coordinates (t, r, theta, phi).
    mass_geometric_m : float
        Geometric mass G M / c^2 [m].
    charge_m : float
        Charge parameter Q with Q^2 = G (p^2 + q^2) / c^4 (geometric units), where q is the electric and p the
        magnetic charge [m].

    Returns
    -------
    ndarray, shape (4, 4)
        g_{mu nu}: diag(-D, 1/D, r^2, r^2 sin^2 theta) with D = 1 - 2M/r + Q^2/r^2.

    References
    ----------
    [Carroll1997] Eq. 7.110, Eq. 7.111 (the field F_tr = -q/r^2 is Eq. 7.112).

    Validity
    --------
    Coordinates are regular only for D != 0, that is outside the outer horizon r_+ = M + sqrt(M^2 - Q^2)
    (Eq. 7.113) when |Q| < M. The conversion of Gaussian charge to the geometric length Q is the user's
    responsibility.

    Verification
    ------------
    tests/unit/test_curvature.py: Einstein tensor equals 8 pi T_mu_nu of the Coulomb field of Eq. 7.112
    with the stress tensor of Eq. 7.107; Ricci scalar vanishes; Q = 0 gives Schwarzschild.
    """
    arr = _coordinates(x)
    r, theta = arr[1], arr[2]
    delta = 1.0 - 2.0 * mass_geometric_m / r + charge_m**2 / r**2
    g = _zeros(arr)
    g[0, 0] = -delta
    g[1, 1] = 1.0 / delta
    g[2, 2] = r**2
    g[3, 3] = r**2 * np.sin(theta) ** 2
    return g


@source(
    "Abramowicz2013",
    where="Kerr metric in Boyer-Lindquist coordinates, Eq. kerr-metric (TeX lines 543-580)",
    status="read",
    note="components g_tt, g_t_phi, g_phi_phi, g_rr, g_theta_theta as printed, with M = G M_kg / c^2 and "
    "a = J / (M_kg c) in length units; the same components follow from the line element of Carroll Eq. 7.114 "
    "and of Visser (Boyer-Lindquist section), which were also read and are compared in the tests",
)
def kerr_metric(x: ArrayLike, mass_geometric_m: float, spin_a_m: float) -> NDArray:
    """Return the Kerr metric in Boyer-Lindquist coordinates (t, r, theta, phi).

    With rho^2 = r^2 + a^2 cos^2 theta and Delta = r^2 - 2 M r + a^2:
    g_tt = -(1 - 2 M r / rho^2), g_t_phi = -2 M a r sin^2 theta / rho^2, g_rr = rho^2 / Delta,
    g_theta_theta = rho^2, g_phi_phi = (r^2 + a^2 + 2 M a^2 r sin^2 theta / rho^2) sin^2 theta.

    Parameters
    ----------
    x : ndarray, shape (4,)
        Coordinates (t, r, theta, phi).
    mass_geometric_m : float
        Geometric mass G M / c^2 [m].
    spin_a_m : float
        Kerr parameter a = J / (M c) [m]; positive for rotation in the +phi direction, |a| <= M for a black hole.

    Returns
    -------
    ndarray, shape (4, 4)
        g_{mu nu}.

    References
    ----------
    [Abramowicz2013] Eq. kerr-metric. Same components in [Carroll1997] Eq. 7.114-7.116 and [Visser2007]
    (Boyer-Lindquist section).

    Validity
    --------
    Outside the outer horizon r_+ = M + sqrt(M^2 - a^2) (Delta > 0). Delta = 0 and rho = 0 are not guarded.
    For a = 0 it reduces to schwarzschild_metric.

    Verification
    ------------
    tests/unit/test_geodesics.py: equatorial circular orbits with the Keplerian angular velocity, conserved E
    and L_z, frame dragging, Killing check; tests/unit/test_curvature.py: Ricci tensor vanishes and the
    Kretschmann scalar equals Visser's closed form.
    """
    arr = _coordinates(x)
    r, theta = arr[1], arr[2]
    m, a = mass_geometric_m, spin_a_m
    sin2 = np.sin(theta) ** 2
    rho2 = r**2 + a**2 * np.cos(theta) ** 2
    delta = r**2 - 2.0 * m * r + a**2
    g = _zeros(arr)
    g[0, 0] = -(1.0 - 2.0 * m * r / rho2)
    g[0, 3] = g[3, 0] = -2.0 * m * a * r * sin2 / rho2
    g[1, 1] = rho2 / delta
    g[2, 2] = rho2
    g[3, 3] = (r**2 + a**2 + 2.0 * m * a**2 * r * sin2 / rho2) * sin2
    return g


@source(
    "Visser2007",
    where="section 'Kerr's original coordinates', Eq. E:K1 (TeX lines 203-215)",
    status="read",
    note="ds^2 = -[1 - 2mr/rho^2](du + a sin^2 theta dphi)^2 + 2(du + a sin^2 theta dphi)(dr + a sin^2 theta "
    "dphi) + rho^2 (dtheta^2 + sin^2 theta dphi^2); built here as -F A A + (A B + B A) + rho^2 diag(0,0,1,sin^2) "
    "with the covectors A = (1,0,0,s), B = (0,1,0,s), s = a sin^2 theta",
)
def kerr_eddington_finkelstein_metric(x: ArrayLike, mass_geometric_m: float, spin_a_m: float) -> NDArray:
    """Return the Kerr metric in Kerr's original (advanced Eddington-Finkelstein) coordinates (u, r, theta, phi).

    The azimuth of this form runs opposite to the Boyer-Lindquist azimuth of ``kerr_metric``: the source's
    transformation between the two is phi = -phi_BL - a * integral dr / Delta (its Eq. BL-change-2). For the
    same ``spin_a_m`` the hole therefore rotates towards decreasing phi here, the off-diagonal component has
    the opposite sign (g_u_phi > 0 on the equator for a > 0) and a prograde orbit has negative p_phi.

    Parameters
    ----------
    x : ndarray, shape (4,)
        Coordinates (u, r, theta, phi).
    mass_geometric_m : float
        Geometric mass G M / c^2 [m].
    spin_a_m : float
        Kerr parameter a = J / (M c) [m].

    Returns
    -------
    ndarray, shape (4, 4)
        g_{mu nu}.

    References
    ----------
    [Visser2007] Eq. E:K1 (the Schwarzschild limit a -> 0 is Eq. E:Sch-EF, equal to Eq. 7.69 of
    [Carroll1997]).

    Validity
    --------
    Regular across both horizons; singular only at rho = 0 (r = 0, theta = pi/2) and on the axis. The
    transformation to the Boyer-Lindquist coordinates is Eq. E:BL-change-1 to E:BL-change-3 of the source; it
    is not implemented here.

    Verification
    ------------
    tests/unit/test_curvature.py: Ricci tensor vanishes; the Kretschmann scalar at (r, theta) equals the
    Boyer-Lindquist value (it is a scalar); tests/unit/test_geodesics.py: d/du and d/dphi are Killing vectors.
    """
    arr = _coordinates(x)
    r, theta = arr[1], arr[2]
    m, a = mass_geometric_m, spin_a_m
    sin2 = np.sin(theta) ** 2
    rho2 = r**2 + a**2 * np.cos(theta) ** 2
    s = a * sin2
    zero = np.zeros_like(s)
    one = np.ones_like(s)
    cov_a = np.stack([one, zero, zero, s])
    cov_b = np.stack([zero, one, zero, s])
    f = 1.0 - 2.0 * m * r / rho2
    g = _zeros(arr)
    g += -f * np.outer(cov_a, cov_a) + np.outer(cov_a, cov_b) + np.outer(cov_b, cov_a)
    g[2, 2] += rho2
    g[3, 3] += rho2 * sin2
    return g


@source(
    "Carroll1997",
    where="Section 8, Eq. 8.7 (TeX line 11409)",
    status="read",
    note="Robertson-Walker metric ds^2 = -dt^2 + a^2(t)[dr^2/(1 - k r^2) + r^2 dOmega^2]; the callable "
    "scale_factor gives a(t); coordinates (t, r, theta, phi)",
)
def flrw_metric(x: ArrayLike, scale_factor: Callable[[ArrayLike], ArrayLike], spatial_curvature: float) -> NDArray:
    """Return the Robertson-Walker (FLRW) metric in coordinates (t, r, theta, phi).

    Parameters
    ----------
    x : ndarray, shape (4,)
        Coordinates (t, r, theta, phi); t in the unit in which c = 1, r dimensionless (comoving).
    scale_factor : callable
        a(t), called with the first coordinate. For complex-step derivatives it must accept a complex argument
        and be analytic (use numpy functions, not abs or comparisons).
    spatial_curvature : float
        k of Eq. 8.7 (the sign of the spatial curvature when r is dimensionless: +1, 0 or -1).

    Returns
    -------
    ndarray, shape (4, 4)
        g_{mu nu}: diag(-1, a^2/(1 - k r^2), a^2 r^2, a^2 r^2 sin^2 theta).

    References
    ----------
    [Carroll1997] Eq. 8.7.

    Validity
    --------
    k r^2 < 1 (closed universe has a coordinate singularity at r = 1/sqrt(k)). The function does not check that
    a(t) > 0.

    Verification
    ------------
    tests/unit/test_curvature.py: Einstein tensor for a = t^(2/3) and a = exp(H t) equals the expressions
    obtained from the Ricci tensor and scalar printed in Carroll Eq. 8.13, 8.14 and the Friedmann equations
    8.35, 8.36.
    """
    arr = _coordinates(x)
    t, r, theta = arr[0], arr[1], arr[2]
    a = scale_factor(t)
    k = spatial_curvature
    g = _zeros(arr)
    g[0, 0] = -1.0
    g[1, 1] = a**2 / (1.0 - k * r**2)
    g[2, 2] = a**2 * r**2
    g[3, 3] = a**2 * r**2 * np.sin(theta) ** 2
    return g


@source(
    "Carroll1997",
    where="Section 2, Eq. 2.28 (TeX line 2640)",
    status="read",
    note="g^{mu nu} g_{nu sigma} = delta^mu_sigma",
)
def metric_inverse(metric_matrix: ArrayLike) -> NDArray:
    """Return the inverse metric g^{mu nu} of a covariant metric matrix g_{mu nu}.

    Parameters
    ----------
    metric_matrix : ndarray, shape (4, 4)
        Covariant components g_{mu nu}, real or complex.

    Returns
    -------
    ndarray, shape (4, 4)
        g^{mu nu} with g^{mu nu} g_{nu sigma} = delta^mu_sigma.

    References
    ----------
    [Carroll1997] Eq. 2.28.

    Validity
    --------
    The metric must be non-degenerate (numpy raises LinAlgError otherwise). Complex input is allowed and the
    result is analytic in the entries, so complex-step derivatives pass through.

    Verification
    ------------
    tests/unit/test_geodesics.py: g^{mu nu} g_{nu sigma} = delta for Schwarzschild and Kerr to rounding.
    """
    g = np.asarray(metric_matrix)
    if g.shape != (4, 4):
        raise ValueError("the metric must have shape (4, 4)")
    return np.linalg.inv(g)
