"""Curvature tensors of a metric given as a function of the coordinates, in geometric units (G = c = 1).

Conventions follow Carroll [Carroll1997]: signature (-, +, +, +), Riemann tensor
R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma} + Gamma^rho_{mu lambda}
Gamma^lambda_{nu sigma} - Gamma^rho_{nu lambda} Gamma^lambda_{mu sigma} (Eq. 3.67), Ricci tensor
R_{mu nu} = R^lambda_{mu lambda nu} (Eq. 3.90). The sign of the Ricci tensor and of the Ricci scalar is the
one in which a sphere has positive curvature; Visser [Visser2007] warns that some symbolic packages use the
opposite sign for the Ricci tensor and scalar.

Numerical method: the Christoffel symbols are computed with the complex-step derivative of the metric (exact to
rounding), and their derivatives with the finite differences of kozmos.core._differentiation
(``method="finite"``: six points, an error estimate for every element, a step searched over many decades).
Hence the metric function must be complex safe, and the accuracy is limited by the cancellation of terms that
are large in the coordinates (of order 1/r or r) to leave a curvature of order M/r^3. The error estimates are
carried to the Riemann tensor and compared with it in an orthonormal frame: a tensor that is not known to
``rtol`` (1e-6 by default) raises ``CurvaturePrecisionError``, and one that vanishes within its uncertainty
is returned as exactly zero with a ``CurvaturePrecisionWarning`` that states the bound (flat space in
curvilinear coordinates; also a curvature that is more than about twelve orders of magnitude below the
connection terms, as for a comoving radius whose physical size is tiny against the curvature radius: there the
zero means "not resolved", and every scalar formed from the tensor is zero as well). Measured for Schwarzschild in units of the mass: relative
error of the Kretschmann scalar 7e-13 at 4 M, 4e-12 at 100 M, 6e-10 at 1e4 M, 1e-9 at 1e5 M; refused at 1e6 M
(estimate 7e-6). In metres the refusal comes earlier, because the angular coordinates are of order one beside
radii of 1e12 and more: for 1e9 solar masses 4e-13 at 4 M and 7e-12 at 100 M, refused from 1e3 M; for 1e10
solar masses refused from 100 M. Work in units of the mass far from the hole. Measured (tests/unit/test_curvature.py): the Ricci tensor of Schwarzschild
and Kerr vanishes to an absolute 4e-12 ... 1e-9, which is 8e-11 ... 4e-9 of sqrt(K) for 3 M <= r <= 10 M and
1.2e-7 of sqrt(K) at r = 30 M; the Kretschmann scalar has a relative error of 1e-11 ... 7e-10. Coordinates
smaller than 1 in magnitude with 1/x structure (a comoving r = 0.2) need a smaller step, which the helper
chooses, and the rounding error grows in proportion.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core._differentiation import jacobian_with_error
from kozmos.relativity.geodesics import Metric, _christoffel
from kozmos.relativity.metrics import metric_inverse

Array = NDArray[np.float64]

_ZERO_WITHIN = 10.0  # a Riemann tensor below this many uncertainties counts as zero


class CurvaturePrecisionError(ValueError):
    """The curvature at the point is not determined to the requested accuracy."""


class CurvaturePrecisionWarning(RuntimeWarning):
    """The curvature vanishes within its numerical uncertainty; the returned values are noise around zero."""


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
    "Christoffel symbols is a finite difference (six points, with an error estimate) of complex-step "
    "Christoffel symbols",
)
def riemann_tensor(metric: Metric, x: ArrayLike, rtol: float = 1e-6) -> Array:
    """Compute the Riemann tensor R^rho_{sigma mu nu} at a point.

    Parameters
    ----------
    metric : callable
        metric(x) -> ndarray of shape (4, 4), covariant components; must be complex safe.
    x : ndarray, shape (4,)
        Coordinates of the point.
    rtol : float, optional
        Relative accuracy required of the Riemann tensor (estimated from the finite differences of the
        Christoffel symbols); ``CurvaturePrecisionError`` is raised if it is not reached, unless the tensor
        vanishes within its uncertainty.

    Returns
    -------
    ndarray, shape (4, 4, 4, 4)
        R[rho, sigma, mu, nu] = R^rho_{sigma mu nu}, antisymmetric in the last two indices.

    References
    ----------
    [Carroll1997] Eq. 3.67; symmetries Eq. 3.78-3.81 and the Bianchi identity Eq. 3.88 are used in the tests.

    Validity
    --------
    Accuracy: second derivatives of the metric by a six-point difference of exact first derivatives, with a
    step chosen element by element from its error estimate; rounding errors of the large Christoffel terms add.
    ``CurvaturePrecisionError`` if the tensor is not known to ``rtol`` (see the module docstring). Measured: Ricci tensor of
    Schwarzschild at r = 10 M has an absolute residual of 4.0e-12 (5.8e-10 of sqrt(K) = 6.9e-3), of Kerr (a = 0.9 M)
    at (r, theta) = (6 M, 1) 4.4e-12 (1.5e-10 of sqrt(K)); see tests/unit/test_curvature.py for more numbers.
    Needs a metric that is smooth in a neighbourhood of x.

    Verification
    ------------
    tests/unit/test_curvature.py: Minkowski gives zero; Schwarzschild and Kerr are Ricci flat; Riemann symmetries
    of Eq. 3.78-3.81; Kretschmann scalar against closed forms; FLRW Ricci tensor against Eq. 8.13.
    """
    point = _point(x)
    gamma = _christoffel(metric, point, "complex", True, neighbours=True)

    def christoffel_at(y: Array) -> Array:
        return _christoffel(metric, y, "complex", False)

    # dgamma[a, b, c, l] = d_l Gamma^a_bc, with an estimate of its absolute error
    dgamma, dgamma_error = jacobian_with_error(christoffel_at, point, method="finite")
    riemann = (
        np.einsum("rnsm->rsmn", dgamma)
        - np.einsum("rmsn->rsmn", dgamma)
        + np.einsum("rml,lns->rsmn", gamma, gamma)
        - np.einsum("rnl,lms->rsmn", gamma, gamma)
    )
    error = np.einsum("rnsm->rsmn", dgamma_error) + np.einsum("rmsn->rsmn", dgamma_error)
    # Components are compared in an orthonormal frame, where they all have the unit of curvature: with
    # e_mu = sqrt|g_mu_mu|, R^rho_sigma_mu_nu is multiplied by e_rho / (e_sigma e_mu e_nu). In coordinates whose
    # metric components differ by many orders of magnitude a component that is tiny in the coordinate basis can
    # be the whole curvature, and an absolute comparison would not see its error.
    frame = np.sqrt(np.abs(np.diagonal(np.asarray(metric(point), dtype=float))))
    frame = np.where(frame > 0.0, frame, 1.0)
    weight = np.einsum("r,s,m,n->rsmn", frame, 1.0 / frame, 1.0 / frame, 1.0 / frame)
    size, uncertainty = float(np.max(np.abs(riemann) * weight)), float(np.max(error * weight))
    terms = float(
        max(
            np.max(np.abs(np.einsum("rnsm->rsmn", dgamma)) * weight),
            np.max(np.abs(np.einsum("rml,lns->rsmn", gamma, gamma)) * weight),
        )
    )
    # The tensor must be known to rtol. The one exception is a tensor that vanishes within its uncertainty while
    # that uncertainty is itself below rtol of the connection terms that cancel: flat space in curvilinear
    # coordinates, which is returned as exactly zero, with a warning.
    known = uncertainty <= rtol * size
    vanishing = size <= _ZERO_WITHIN * uncertainty and uncertainty <= rtol * terms
    if not (known or vanishing):
        raise CurvaturePrecisionError(
            f"the Riemann tensor is known to a relative accuracy of {uncertainty / max(size, 1e-300):.1e} only (limit {rtol:.1e}): "
            "the curvature is a small difference of large connection terms at this point in these coordinates "
            "(far from the source, or with coordinates of very different sizes)"
        )
    if not known and size > 0.0:
        bound = max(size, uncertainty)
        warnings.warn(
            f"the Riemann tensor vanishes within its numerical uncertainty ({uncertainty:.1e} in orthonormal "
            f"components, {uncertainty / terms:.1e} of the connection terms that cancel) and is returned as zero. "
            f"If the spacetime is not flat there, its orthonormal components are below about {bound:.1e} and too "
            "small against the connection terms to be seen in these coordinates",
            CurvaturePrecisionWarning,
            stacklevel=2,
        )
        # rounding noise around zero is not returned: scalars formed from it (the Kretschmann scalar is its
        # square) would look like results
        riemann = np.zeros_like(riemann)
    return riemann


@source(
    "Carroll1997",
    where="Section 3, Eq. 3.90 (TeX line 4556)",
    status="read",
    note="R_mu_nu = R^lambda_{mu lambda nu}",
)
def ricci_tensor(metric: Metric, x: ArrayLike, rtol: float = 1e-6) -> Array:
    """Compute the Ricci tensor R_mu_nu = R^lambda_{mu lambda nu} at a point.

    Parameters
    ----------
    metric : callable
        metric(x) -> (4, 4) covariant components; must be complex safe.
    x : ndarray, shape (4,)
        Coordinates of the point.
    rtol : float, optional
        Relative accuracy required of the Riemann tensor (estimated from the finite differences of the
        Christoffel symbols); ``CurvaturePrecisionError`` is raised if it is not reached, unless the tensor
        vanishes within its uncertainty.

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
    return np.einsum("lmln->mn", riemann_tensor(metric, x, rtol))


@source(
    "Carroll1997",
    where="Section 3, Eq. 3.92 (TeX line 4572)",
    status="read",
    note="R = g^{mu nu} R_{mu nu}",
)
def ricci_scalar(metric: Metric, x: ArrayLike, rtol: float = 1e-6) -> float:
    """Compute the Ricci scalar R = g^{mu nu} R_{mu nu} at a point.

    Parameters
    ----------
    metric : callable
        metric(x) -> (4, 4) covariant components; must be complex safe.
    x : ndarray, shape (4,)
        Coordinates of the point.
    rtol : float, optional
        Relative accuracy required of the Riemann tensor (estimated from the finite differences of the
        Christoffel symbols); ``CurvaturePrecisionError`` is raised if it is not reached, unless the tensor
        vanishes within its uncertainty.

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
    return float(np.einsum("mn,mn->", ginv, ricci_tensor(metric, point, rtol)))


@source(
    "Carroll1997",
    where="Section 3, Eq. 3.95 (TeX line 4591)",
    status="read",
    note="G_mu_nu = R_mu_nu - (1/2) R g_mu_nu",
)
def einstein_tensor(metric: Metric, x: ArrayLike, rtol: float = 1e-6) -> Array:
    """Compute the Einstein tensor G_mu_nu = R_mu_nu - R g_mu_nu / 2 at a point.

    Parameters
    ----------
    metric : callable
        metric(x) -> (4, 4) covariant components; must be complex safe.
    x : ndarray, shape (4,)
        Coordinates of the point.
    rtol : float, optional
        Relative accuracy required of the Riemann tensor (estimated from the finite differences of the
        Christoffel symbols); ``CurvaturePrecisionError`` is raised if it is not reached, unless the tensor
        vanishes within its uncertainty.

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
    ricci = ricci_tensor(metric, point, rtol)
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
def kretschmann_scalar(metric: Metric, x: ArrayLike, rtol: float = 1e-6) -> float:
    """Compute the Kretschmann scalar K = R_{mu nu rho sigma} R^{mu nu rho sigma} at a point.

    Parameters
    ----------
    metric : callable
        metric(x) -> (4, 4) covariant components; must be complex safe.
    x : ndarray, shape (4,)
        Coordinates of the point.
    rtol : float, optional
        Relative accuracy required of the Riemann tensor (estimated from the finite differences of the
        Christoffel symbols); ``CurvaturePrecisionError`` is raised if it is not reached, unless the tensor
        vanishes within its uncertainty.

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
    riemann = riemann_tensor(metric, point, rtol)
    lowered = np.einsum("rl,lsmn->rsmn", g, riemann)
    raised = np.einsum("sb,mc,nd,rbcd->rsmn", ginv, ginv, ginv, riemann)
    return float(np.einsum("rsmn,rsmn->", lowered, raised))
