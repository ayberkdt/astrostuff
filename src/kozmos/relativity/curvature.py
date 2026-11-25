"""Curvature tensors of a metric given as a function of the coordinates, in geometric units (G = c = 1).

Conventions follow Carroll [Carroll1997]: signature (-, +, +, +), Riemann tensor
R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma} + Gamma^rho_{mu lambda}
Gamma^lambda_{nu sigma} - Gamma^rho_{nu lambda} Gamma^lambda_{mu sigma} (Eq. 3.67), Ricci tensor
R_{mu nu} = R^lambda_{mu lambda nu} (Eq. 3.90). The sign of the Ricci tensor and of the Ricci scalar is the
one in which a sphere has positive curvature; Visser [Visser2007] warns that some symbolic packages use the
opposite sign for the Ricci tensor and scalar.

Numerical method: the Christoffel symbols are computed with the complex-step derivative of the metric (exact to
rounding), and their derivatives with the four-point central difference of kozmos.core._differentiation
(``method="finite"``). Hence the metric function must be complex safe, the second derivatives of the metric
carry the error of the four-point formula, about h^4 / 30 times a fifth derivative (the step starts at
h = 1e-3 max(1, |x_i|) and is reduced until it resolves the scale on which the Christoffel symbols vary),
and the accuracy is limited by the cancellation of terms that are large in the coordinates (of order 1/r or r)
to leave a curvature of order M/r^3. Measured (tests/unit/test_curvature.py): the Ricci tensor of Schwarzschild
and Kerr vanishes to an absolute 4e-12 ... 1e-9, which is 8e-11 ... 4e-9 of sqrt(K) for 3 M <= r <= 10 M and
1.2e-7 of sqrt(K) at r = 30 M; the Kretschmann scalar has a relative error of 1e-11 ... 7e-10. Coordinates
smaller than 1 in magnitude with 1/x structure (a comoving r = 0.2) need a smaller step, which the helper chooses,
and the rounding error grows in proportion.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core._differentiation import jacobian
from kozmos.relativity.geodesics import Metric, christoffel_symbols
from kozmos.relativity.metrics import metric_inverse

Array = NDArray[np.float64]


def _point(x: ArrayLike) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.shape != (4,):
        raise ValueError("x must have shape (4,)")
    return arr


@source(
    "Carroll1997",
    where="Section 3, Eq. 3.67 (TeX line 4242)",
    status="read",
    note="R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma} + Gamma^rho_{mu lambda} "
    "Gamma^lambda_{nu sigma} - Gamma^rho_{nu lambda} Gamma^lambda_{mu sigma}; the derivative of the "
    "Christoffel symbols is a four-point central difference of complex-step Christoffel symbols",
)
def riemann_tensor(metric: Metric, x: ArrayLike) -> Array:
    """Compute the Riemann tensor R^rho_{sigma mu nu} at a point.

    Parameters
    ----------
    metric : callable
        metric(x) -> ndarray of shape (4, 4), covariant components; must be complex safe.
    x : ndarray, shape (4,)
        Coordinates of the point.

    Returns
    -------
    ndarray, shape (4, 4, 4, 4)
        R[rho, sigma, mu, nu] = R^rho_{sigma mu nu}, antisymmetric in the last two indices.

    References
    ----------
    [Carroll1997] Eq. 3.67; symmetries Eq. 3.78-3.81 and the Bianchi identity Eq. 3.88 are used in the tests.

    Validity
    --------
    Accuracy: second derivatives of the metric by a four-point difference of exact first derivatives, with step
    h = 1e-3 max(1, |x_i|) or smaller (chosen by the helper); the truncation error is about (h / L)^4 / 30 times the fifth derivative scale for a
    variation length L, and rounding errors of the large Christoffel terms add. Measured: Ricci tensor of
    Schwarzschild at r = 10 M has an absolute residual of 4.0e-12 (5.8e-10 of sqrt(K) = 6.9e-3), of Kerr (a = 0.9 M)
    at (r, theta) = (6 M, 1) 4.4e-12 (1.5e-10 of sqrt(K)); see tests/unit/test_curvature.py for more numbers.
    Needs a metric that is smooth in a neighbourhood of x of size 2 h.

    Verification
    ------------
    tests/unit/test_curvature.py: Minkowski gives zero; Schwarzschild and Kerr are Ricci flat; Riemann symmetries
    of Eq. 3.78-3.81; Kretschmann scalar against closed forms; FLRW Ricci tensor against Eq. 8.13.
    """
    point = _point(x)
    gamma = christoffel_symbols(metric, point)

    def christoffel_at(y: Array) -> Array:
        return christoffel_symbols(metric, y)

    dgamma = jacobian(christoffel_at, point, method="finite")  # dgamma[a, b, c, l] = d_l Gamma^a_bc
    return (
        np.einsum("rnsm->rsmn", dgamma)
        - np.einsum("rmsn->rsmn", dgamma)
        + np.einsum("rml,lns->rsmn", gamma, gamma)
        - np.einsum("rnl,lms->rsmn", gamma, gamma)
    )


@source(
    "Carroll1997",
    where="Section 3, Eq. 3.90 (TeX line 4556)",
    status="read",
    note="R_mu_nu = R^lambda_{mu lambda nu}",
)
def ricci_tensor(metric: Metric, x: ArrayLike) -> Array:
    """Compute the Ricci tensor R_mu_nu = R^lambda_{mu lambda nu} at a point.

    Parameters
    ----------
    metric : callable
        metric(x) -> (4, 4) covariant components; must be complex safe.
    x : ndarray, shape (4,)
        Coordinates of the point.

    Returns
    -------
    ndarray, shape (4, 4)
        R_{mu nu}, symmetric.

    References
    ----------
    [Carroll1997] Eq. 3.90, Eq. 3.91.

    Validity
    --------
    Same numerical accuracy as riemann_tensor.

    Verification
    ------------
    tests/unit/test_curvature.py: vanishes for Schwarzschild, Kerr and Kerr in Eddington-Finkelstein
    coordinates to the measured residual; FLRW components against Eq. 8.13.
    """
    return np.einsum("lmln->mn", riemann_tensor(metric, x))


@source(
    "Carroll1997",
    where="Section 3, Eq. 3.92 (TeX line 4572)",
    status="read",
    note="R = g^{mu nu} R_{mu nu}",
)
def ricci_scalar(metric: Metric, x: ArrayLike) -> float:
    """Compute the Ricci scalar R = g^{mu nu} R_{mu nu} at a point.

    Parameters
    ----------
    metric : callable
        metric(x) -> (4, 4) covariant components; must be complex safe.
    x : ndarray, shape (4,)
        Coordinates of the point.

    Returns
    -------
    float
        R.

    References
    ----------
    [Carroll1997] Eq. 3.92.

    Validity
    --------
    Same numerical accuracy as riemann_tensor.

    Verification
    ------------
    tests/unit/test_curvature.py: zero for vacuum and Reissner-Nordstrom metrics; FLRW value 6 (a addot +
    adot^2 + k) / a^2 of Eq. 8.14.
    """
    point = _point(x)
    ginv = metric_inverse(np.asarray(metric(point), dtype=float))
    return float(np.einsum("mn,mn->", ginv, ricci_tensor(metric, point)))


@source(
    "Carroll1997",
    where="Section 3, Eq. 3.95 (TeX line 4591)",
    status="read",
    note="G_mu_nu = R_mu_nu - (1/2) R g_mu_nu",
)
def einstein_tensor(metric: Metric, x: ArrayLike) -> Array:
    """Compute the Einstein tensor G_mu_nu = R_mu_nu - R g_mu_nu / 2 at a point.

    Parameters
    ----------
    metric : callable
        metric(x) -> (4, 4) covariant components; must be complex safe.
    x : ndarray, shape (4,)
        Coordinates of the point.

    Returns
    -------
    ndarray, shape (4, 4)
        G_{mu nu}, symmetric.

    References
    ----------
    [Carroll1997] Eq. 3.95.

    Validity
    --------
    Same numerical accuracy as riemann_tensor. Without a cosmological constant the field equation is
    G_mu_nu = 8 pi G T_mu_nu (Eq. 4.43 with kappa = 8 pi G); the factor is not applied here.

    Verification
    ------------
    tests/unit/test_curvature.py: FLRW G_tt = 3 (adot^2 + k) / a^2 for a = t^(2/3) and a = exp(H t);
    Reissner-Nordstrom G_mu_nu = 8 pi T_mu_nu with the Coulomb field of Eq. 7.112.
    """
    point = _point(x)
    g = np.asarray(metric(point), dtype=float)
    ricci = ricci_tensor(metric, point)
    scalar = float(np.einsum("mn,mn->", metric_inverse(g), ricci))
    return ricci - 0.5 * scalar * g


@source(
    "Carroll1997",
    where="Section 7, Eq. 7.30 (TeX line 9287, left-hand side only) and Eq. 3.76 (TeX line 4406)",
    status="read",
    note="the invariant R^{mu nu rho sigma} R_{mu nu rho sigma} with R_{rho sigma mu nu} = g_{rho lambda} R^lambda_{sigma "
    "mu nu}; the right-hand side printed in Eq. 7.30 (12 G^2 M^2 / r^6) is NOT used: it disagrees with the "
    "value computed here for the same metric and with Visser's printed 48 m^2 / r^6 (a -> 0 of his Kerr "
    "invariant)",
)
def kretschmann_scalar(metric: Metric, x: ArrayLike) -> float:
    """Compute the Kretschmann scalar K = R_{mu nu rho sigma} R^{mu nu rho sigma} at a point.

    Parameters
    ----------
    metric : callable
        metric(x) -> (4, 4) covariant components; must be complex safe.
    x : ndarray, shape (4,)
        Coordinates of the point.

    Returns
    -------
    float
        K, non-negative for a Riemannian metric, indefinite in general in Lorentzian signature.

    References
    ----------
    [Carroll1997] Eq. 7.30 (definition of the invariant), Eq. 3.76 (lowering). The closed form for Kerr is in
    [Visser2007] (section 'Kerr's original coordinates', TeX lines 276-281, and section 'Boyer-Lindquist
    coordinates', TeX line 762).

    Validity
    --------
    Same numerical accuracy as riemann_tensor. For Schwarzschild K = 48 M^2 / r^6 (Visser's formula at a = 0;
    the right-hand side printed in Eq. 7.30 of the TeX source, 12 M^2 / r^6, is a misprint by a factor of 4).

    Verification
    ------------
    tests/unit/test_curvature.py: Schwarzschild 48 M^2 / r^6 (also in Eddington-Finkelstein coordinates at the
    horizon), Kerr against Visser's closed form in Boyer-Lindquist and in Kerr's original coordinates; flat
    space gives zero.
    """
    point = _point(x)
    g = np.asarray(metric(point), dtype=float)
    ginv = metric_inverse(g)
    riemann = riemann_tensor(metric, point)
    lowered = np.einsum("rl,lsmn->rsmn", g, riemann)
    raised = np.einsum("sb,mc,nd,rbcd->rsmn", ginv, ginv, ginv, riemann)
    return float(np.einsum("rsmn,rsmn->", lowered, raised))
