"""PSF-fitting photometry: pixel-integrated Gaussian and Moffat profiles and grouped least-squares fits.

The fit follows the profile-fitting scheme of DAOPHOT [Stetson1987]: a model profile is shifted and scaled to each star,
stars whose profiles overlap are fitted together as a group, and corrections to flux and position come from an
iterated linearized least-squares solution. The model is the pixel response of the star (the profile integrated over
each pixel), as defined in the photutils documentation [PhotutilsPSF2026]; the Moffat form is that of [Moffat1969].
Tested with synthetic images whose truth is known.

Conventions: a pixel center sits at an integer coordinate (``image[y, x]``, the same as
``kozmos.observing.detector.star_field``). The FWHM of a model is that of the profile before it is integrated over the
pixels; the image of a star is slightly wider (see ``fit_psf_width``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, special
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from kozmos._sources import source
from kozmos.statistics.robust import mad_scale

Array = NDArray[np.float64]

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
DEFAULT_MOFFAT_NODES = 5


@dataclass(frozen=True, slots=True)
class PSFPhotometryResult:
    """Result of ``psf_photometry``; every array has one entry per input star, in input order.

    Attributes
    ----------
    flux : ndarray
        Fitted total flux of the star, in the units of the image summed over pixels; may be negative at low S/N.
    flux_err : ndarray
        1-sigma flux uncertainty from the covariance of the group fit (formal if a noise map was given; scaled
        by the residual scatter otherwise, see ``psf_photometry``).
    x, y : ndarray
        Fitted star centers [pixel].
    x_err, y_err : ndarray
        1-sigma center uncertainties [pixel]; NaN if the positions were not fitted.
    chi2_reduced : ndarray
        Weighted residual sum of squares of the pixels within the fit radius of the star divided by their number minus
        the number of parameters per star; NaN if no noise map was given.
    rms_residual : ndarray
        Root-mean-square residual (data minus model) over the same pixels, in image units.
    n_pixels : ndarray
        Number of pixels within the fit radius of the star that entered the fit.
    group_id : ndarray
        Index of the group the star was fitted in.
    group_size : ndarray
        Number of stars in that group.
    converged : ndarray
        False if the iteration did not reach the position tolerance or the fit had too few pixels.
    """

    flux: Array
    flux_err: Array
    x: Array
    y: Array
    x_err: Array
    y_err: Array
    chi2_reduced: Array
    rms_residual: Array
    n_pixels: NDArray[np.int64]
    group_id: NDArray[np.int64]
    group_size: NDArray[np.int64]
    converged: NDArray[np.bool_]


# --------------------------------------------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------------------------------------------


def _gauss_axis(coord: Array, center: float, sigma: float) -> tuple[Array, Array]:
    """Return the integral of a unit 1-D Gaussian over each unit pixel and its derivative with respect to the center."""
    upper = (coord + 0.5 - center) / (np.sqrt(2.0) * sigma)
    lower = (coord - 0.5 - center) / (np.sqrt(2.0) * sigma)
    value = 0.5 * (special.erf(upper) - special.erf(lower))
    derivative = -(np.exp(-(upper**2)) - np.exp(-(lower**2))) / (sigma * np.sqrt(2.0 * np.pi))
    return value, derivative


def _moffat_alpha(fwhm_pix: float, beta: float) -> float:
    """Return the Moffat width parameter R of the profile with the given FWHM: R = FWHM / (2 sqrt(2^(1/beta) - 1))."""
    return fwhm_pix / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))


def _moffat_value_grad(
    xx: Array, yy: Array, x0: float, y0: float, fwhm_pix: float, beta: float, n_nodes: int
) -> tuple[Array, Array, Array]:
    """Return the pixel-integrated unit-flux Moffat profile and its derivatives with respect to x0 and y0.

    The integral over each pixel is a tensor-product Gauss-Legendre rule with n_nodes points per axis.
    """
    radius = _moffat_alpha(fwhm_pix, beta)
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    offsets, w1 = 0.5 * nodes, 0.5 * weights
    xs = xx[..., None, None] + offsets[None, :]  # (..., 1, n) broadcast below
    ys = yy[..., None, None] + offsets[:, None]  # (..., n, 1)
    dx = xs - x0
    dy = ys - y0
    rho = (dx * dx + dy * dy) / (radius * radius)
    norm = (beta - 1.0) / (np.pi * radius * radius)
    base = norm * (1.0 + rho) ** (-beta)
    slope = norm * 2.0 * beta / (radius * radius) * (1.0 + rho) ** (-beta - 1.0)
    w2 = w1[:, None] * w1[None, :]
    value = np.sum(base * w2, axis=(-2, -1))
    d_x0 = np.sum(slope * dx * w2, axis=(-2, -1))
    d_y0 = np.sum(slope * dy * w2, axis=(-2, -1))
    return value, d_x0, d_y0


def _profile(
    model: str, xx: Array, yy: Array, x0: float, y0: float, fwhm_pix: float, beta: float, n_nodes: int
) -> tuple[Array, Array, Array]:
    """Return the unit-flux pixel response and its derivatives with respect to x0 and y0 on pixels (xx, yy)."""
    if model == "gaussian":
        sigma = fwhm_pix * FWHM_TO_SIGMA
        px, dpx = _gauss_axis(xx, x0, sigma)
        py, dpy = _gauss_axis(yy, y0, sigma)
        return px * py, dpx * py, px * dpy
    if model == "moffat":
        return _moffat_value_grad(xx, yy, x0, y0, fwhm_pix, beta, n_nodes)
    raise ValueError(f"unknown model {model!r}; use 'gaussian' or 'moffat'")


@source(
    "PhotutilsPSF2026",
    where="User guide: PSF Photometry, 'Terminology' and 'Analytic PSF Models' (PRF models integrated over the pixels)",
    status="read",
    note="a PRF value is the flux a pixel records, i.e. the integral of the profile over the pixel area for a uniform "
    "pixel response; the Gaussian PRF is integrated analytically. The expression with error functions, "
    "0.5 [erf((x+0.5-x0)/(sqrt(2) sigma)) - erf((x-0.5-x0)/(sqrt(2) sigma))] per axis, is elementary calculus "
    "(this project's derivation, checked in the tests against numerical quadrature); sigma = FWHM / (2 sqrt(2 ln 2))",
)
def gaussian_prf(
    x_pix: ArrayLike, y_pix: ArrayLike, x0_pix: float, y0_pix: float, fwhm_pix: float, flux: float = 1.0
) -> Array:
    """Evaluate the pixel-integrated circular Gaussian profile of a star.

    value(x, y) = flux * Px * Py with
    Px = 0.5 [erf((x + 0.5 - x0) / (sqrt(2) sigma)) - erf((x - 0.5 - x0) / (sqrt(2) sigma))], Py likewise,
    sigma = FWHM / (2 sqrt(2 ln 2)).

    Parameters
    ----------
    x_pix, y_pix : float or ndarray
        Pixel-center coordinates [pixel]; broadcast against each other.
    x0_pix, y0_pix : float
        Star center [pixel].
    fwhm_pix : float
        FWHM of the Gaussian before integration over the pixels [pixel].
    flux : float
        Total flux of the star.

    Returns
    -------
    ndarray
        Flux recorded in each pixel (flux times the integral of the unit profile over the pixel).

    References
    ----------
    [PhotutilsPSF2026] PSF Photometry: Terminology, Analytic PSF Models.

    Validity
    --------
    Exact up to the accuracy of the error function (about 1e-16 relative to the total flux); the sum over all pixels
    equals the flux for a star far from the image edge. A real PSF has broader wings than a Gaussian.

    Verification
    ------------
    tests/unit/test_psf.py: total flux, comparison with numerical quadrature over the pixel, equality with
    observing.detector.star_field, analytic derivative against finite differences.
    """
    xx, yy = np.broadcast_arrays(np.asarray(x_pix, dtype=float), np.asarray(y_pix, dtype=float))
    value, _, _ = _profile("gaussian", xx, yy, float(x0_pix), float(y0_pix), float(fwhm_pix), 0.0, 0)
    return flux * value


@source(
    "Moffat1969",
    where="p. 458, Eq. 7: I*(r) = I0 / {1 + (r/R)^2}^beta (scan of the paper, read)",
    status="read",
    note="the profile shape and the parameters R and beta are from the paper. The normalization to unit total flux, "
    "(beta - 1) / (pi R^2), the relation FWHM = 2 R sqrt(2^(1/beta) - 1), and the integration over a pixel with an "
    "n_nodes x n_nodes Gauss-Legendre rule (default 5 x 5) are this project's derivation or choice, not in the paper; "
    "the symbols beta and R are the paper's",
)
def moffat_prf(
    x_pix: ArrayLike,
    y_pix: ArrayLike,
    x0_pix: float,
    y0_pix: float,
    fwhm_pix: float,
    beta: float = 2.5,
    flux: float = 1.0,
    *,
    n_nodes: int = DEFAULT_MOFFAT_NODES,
) -> Array:
    """Evaluate the pixel-integrated Moffat profile of a star.

    value(x, y) = flux * integral over the pixel of (beta - 1) / (pi R^2) * (1 + r^2 / R^2)^(-beta),
    r^2 = (x - x0)^2 + (y - y0)^2, R = FWHM / (2 sqrt(2^(1/beta) - 1)); the integral over the pixel is taken with a
    tensor-product Gauss-Legendre rule of n_nodes points per axis.

    Parameters
    ----------
    x_pix, y_pix : float or ndarray
        Pixel-center coordinates [pixel]; broadcast against each other.
    x0_pix, y0_pix : float
        Star center [pixel].
    fwhm_pix : float
        FWHM of the Moffat profile before integration over the pixels [pixel].
    beta : float
        Moffat exponent, greater than 1 [dimensionless]; the paper reports values of 3 to 5 for Hamburg Schmidt
        plates and 2.72 for one Lick reflector plate.
    flux : float
        Total flux of the star over the infinite plane; a finite image holds less because of the extended wings.
    n_nodes : int
        Gauss-Legendre nodes per axis for the pixel integral.

    Returns
    -------
    ndarray
        Flux recorded in each pixel.

    References
    ----------
    [Moffat1969] Eq. 7.

    Validity
    --------
    Pixel integral: tensor-product Gauss-Legendre rule, n_nodes x n_nodes per pixel. Measured error against the same
    rule with 64 x 64 nodes (which agrees with 96 x 96 to 1e-12), as the largest error on a 21 x 21 pixel grid for a
    center offset of (0.3, 0.2) pixel, relative to the central pixel value, beta = 1.5, 2.5, 5: with the default 5 x 5
    nodes 2.1e-6, 4.5e-7, 5e-8 at FWHM 1.5 pixel; 1.4e-9 or smaller at FWHM 3 pixel; 5.9e-5, 2.5e-5, 8.8e-6 at FWHM 1
    pixel (the test asserts 3e-6, 1e-8 and 1e-4 respectively). Against scipy dblquad on single pixels the 5 x 5 rule
    agrees to better than 1e-6 relative. Below FWHM of about 1 pixel raise n_nodes (7 x 7 gives 2e-7 at FWHM 1).
    The wings follow a power law: the fraction of the flux outside a radius r is (1 + r^2/R^2)^(1 - beta), so a finite
    image holds less than the total flux (0.9999 of it in a 61 x 61 image for FWHM 3, beta 3; 0.9995 within
    +-30 pixel for beta 2.5).

    Verification
    ------------
    tests/unit/test_psf.py: normalization, FWHM, comparison with adaptive quadrature, convergence of the Gauss-Legendre
    rule with n_nodes, derivative against finite differences, limit beta -> large equals the Gaussian.
    """
    if beta <= 1.0:
        raise ValueError("beta must be greater than 1 for a finite total flux")
    xx, yy = np.broadcast_arrays(np.asarray(x_pix, dtype=float), np.asarray(y_pix, dtype=float))
    value, _, _ = _moffat_value_grad(xx, yy, float(x0_pix), float(y0_pix), float(fwhm_pix), float(beta), n_nodes)
    return flux * value


# --------------------------------------------------------------------------------------------------------------
# Common helpers for the fits
# --------------------------------------------------------------------------------------------------------------


def _as_map(value: ArrayLike | None, shape: tuple[int, int], name: str) -> Array | None:
    """Broadcast a scalar or map to the image shape."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, float(arr))
    if arr.shape != shape:
        raise ValueError(f"{name} must be a scalar or have the shape of the image {shape}, got {arr.shape}")
    return arr


def _pixels_near(
    valid: NDArray[np.bool_], positions: Array, radius: float
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return (y, x) of the valid pixels within radius of any of the positions."""
    ny, nx = valid.shape
    x_lo = max(0, int(np.floor(positions[:, 0].min() - radius)))
    x_hi = min(nx, int(np.ceil(positions[:, 0].max() + radius)) + 1)
    y_lo = max(0, int(np.floor(positions[:, 1].min() - radius)))
    y_hi = min(ny, int(np.ceil(positions[:, 1].max() + radius)) + 1)
    if x_lo >= x_hi or y_lo >= y_hi:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    near = np.zeros(yy.shape, dtype=bool)
    for x0, y0 in positions:
        near |= (xx - x0) ** 2 + (yy - y0) ** 2 <= radius**2
    keep = near & valid[y_lo:y_hi, x_lo:x_hi]
    return yy[keep].astype(np.int64), xx[keep].astype(np.int64)


def _group_labels(x: Array, y: Array, radius: float, max_size: int, min_radius: float) -> NDArray[np.int64]:
    """Group stars by friends-of-friends linking at separations below radius; split groups larger than max_size."""
    n = x.size
    labels = np.zeros(n, dtype=np.int64)
    if n == 0:
        return labels
    pending = [(np.arange(n), radius)]
    next_label = 0
    while pending:
        members, r = pending.pop()
        if members.size == 1:
            labels[members] = next_label
            next_label += 1
            continue
        tree = cKDTree(np.column_stack([x[members], y[members]]))
        pairs = tree.query_pairs(r, output_type="ndarray")
        graph = coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])) if len(pairs) else ([], ([], [])),
            shape=(members.size, members.size),
        )
        count, comp = connected_components(graph, directed=False)
        for c in range(count):
            sub = members[comp == c]
            if sub.size > max_size and r / 2.0 >= min_radius:
                # Stetson: re-run the grouping of an overly large group with a smaller critical separation.
                pending.append((sub, r / 2.0))
            else:
                labels[sub] = next_label
                next_label += 1
    return labels


def _fit_group(
    model: str,
    beta: float,
    n_nodes: int,
    fwhm_pix: float,
    xs: Array,
    ys: Array,
    data: Array,
    weight: Array,
    positions: Array,
    fit_positions: bool,
    max_iterations: int,
    tolerance_pix: float,
    max_shift_pix: float,
    noise_given: bool,
) -> dict:
    """Fit fluxes (and positions) of one group on the given pixels; weight is 1/sigma per pixel."""
    p = positions.shape[0]
    n_par = 3 * p if fit_positions else p
    n_pix = xs.size
    out = {"converged": False}
    if n_pix <= n_par:
        return out | {"too_few": True}
    b = data * weight

    def design(pos: Array):
        cols, dxs, dys = [], [], []
        for x0, y0 in pos:
            v, vx, vy = _profile(model, xs, ys, x0, y0, fwhm_pix, beta, n_nodes)
            cols.append(v)
            dxs.append(vx)
            dys.append(vy)
        return np.array(cols).T, np.array(dxs).T, np.array(dys).T

    pos = positions.copy()
    profile, dprofile_x, dprofile_y = design(pos)
    flux, *_ = np.linalg.lstsq(profile * weight[:, None], b, rcond=None)
    n_iter = 0
    if fit_positions:

        def evaluate(flux_k: Array, pos_k: Array, prof=None):
            prof = design(pos_k) if prof is None else prof
            resid = b - weight * (prof[0] @ flux_k)
            return resid, float(resid @ resid), prof

        resid, chi2, current = evaluate(flux, pos, (profile, dprofile_x, dprofile_y))
        damping = 1e-3
        for iteration in range(1, max_iterations + 1):
            prof, dpx, dpy = current
            jac = np.empty((n_pix, n_par))
            jac[:, 0::3] = prof * weight[:, None]
            jac[:, 1::3] = dpx * flux[None, :] * weight[:, None]
            jac[:, 2::3] = dpy * flux[None, :] * weight[:, None]
            hess = jac.T @ jac
            grad = jac.T @ resid
            accepted = False
            for _ in range(12):
                lhs = hess + damping * np.diag(np.maximum(np.diag(hess), 1e-30))
                step = np.linalg.lstsq(lhs, grad, rcond=None)[0]
                shifts = np.column_stack([step[1::3], step[2::3]])
                big = np.abs(shifts).max()
                if big > max_shift_pix:
                    step = step * (max_shift_pix / big)
                    shifts = shifts * (max_shift_pix / big)
                new_flux = flux + step[0::3]
                new_pos = pos + shifts
                new_resid, new_chi2, new_current = evaluate(new_flux, new_pos)
                if new_chi2 <= chi2:
                    accepted = True
                    break
                damping *= 5.0
            if not accepted:
                break
            flux, pos, resid, chi2, current = new_flux, new_pos, new_resid, new_chi2, new_current
            damping = max(damping / 5.0, 1e-9)
            n_iter = iteration
            if np.abs(shifts).max() < tolerance_pix:
                out["converged"] = True
                break
        profile, dprofile_x, dprofile_y = current
        jac = np.empty((n_pix, n_par))
        jac[:, 0::3] = profile * weight[:, None]
        jac[:, 1::3] = dprofile_x * flux[None, :] * weight[:, None]
        jac[:, 2::3] = dprofile_y * flux[None, :] * weight[:, None]
    else:
        jac = profile * weight[:, None]
        out["converged"] = True
        resid = b - weight * (profile @ flux)
        chi2 = float(resid @ resid)
    hess = jac.T @ jac
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hess)
    if not noise_given:
        cov = cov * (chi2 / (n_pix - n_par))
    var = np.clip(np.diag(cov), 0.0, None)
    if fit_positions:
        flux_err, x_err, y_err = np.sqrt(var[0::3]), np.sqrt(var[1::3]), np.sqrt(var[2::3])
    else:
        flux_err, x_err, y_err = np.sqrt(var), np.full(p, np.nan), np.full(p, np.nan)
    return out | {
        "too_few": False,
        "flux": flux,
        "pos": pos,
        "flux_err": flux_err,
        "x_err": x_err,
        "y_err": y_err,
        "n_iter": n_iter,
    }


# --------------------------------------------------------------------------------------------------------------
# Public fits
# --------------------------------------------------------------------------------------------------------------


@source(
    "Stetson1987",
    where="Section III.D.1 'The Basics' (grouping of stars, iterated linearized least squares, Eq. 8) and Section III.D.2 "
    "subsections a and b (sky held fixed; pixel weights, Eq. 10)",
    status="read",
    note="read in the ADS scan, prose only: the displayed Eqs. 8 and 10 were garbled in the text extraction and were "
    "not seen typeset, so the equation numbers rest on the text's own references to them. A model profile is shifted and scaled to every star; stars closer than a critical "
    "separation (image radius plus fitting radius) are fitted together; the corrections to flux and position follow "
    "from a least-squares solution of Eq. 8 using the derivatives of the profile; only pixels within a fitting radius "
    "of a star enter. Stetson's weights (Eq. 10, with flat-field and interpolation-error terms), the radial taper of "
    "the weights, the reweighting against bad pixels and the empirical PSF are NOT implemented. This project's "
    "choices: an analytic pixel-integrated Gaussian or Moffat profile instead of an empirical PSF; pixel weights "
    "1/sigma^2 from the supplied noise map; Levenberg-Marquardt damping with a step limit instead of undamped "
    "steps; a fixed pixel set per group (from the starting positions); linear least squares for the fluxes at the "
    "starting positions as the first step; defaults fit radius = 1.5 FWHM and critical separation = fit radius + "
    "2 FWHM; groups above max_group_size are regrouped with half the separation, as Stetson suggests; the "
    "uncertainties are (J^T W J)^-1 for the group, scaled by the reduced residual variance when no noise map is "
    "given (not from the paper)",
)
def psf_photometry(
    image: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    fwhm_pix: float,
    *,
    model: str = "gaussian",
    beta: float = 2.5,
    background: ArrayLike | float = 0.0,
    noise: ArrayLike | float | None = None,
    fit_positions: bool = True,
    group_radius_pix: float | None = None,
    fit_radius_pix: float | None = None,
    max_group_size: int = 30,
    max_iterations: int = 40,
    tolerance_pix: float = 1e-3,
    n_nodes: int = DEFAULT_MOFFAT_NODES,
) -> PSFPhotometryResult:
    """Measure star fluxes (and positions) by least-squares fitting of a pixel-integrated PSF model.

    data = background + sum_k flux_k * PRF_k(x - x_k, y - y_k) + noise on the pixels within fit_radius of any star of
    the group; minimize sum ((data - model) / sigma)^2 over the fluxes (linear) and, if requested, the positions
    (iterated, Levenberg-Marquardt). Stars closer than group_radius are fitted together.

    Parameters
    ----------
    image : ndarray (Y, X)
    x, y : ndarray (N,)
        Starting centers [pixel] (for instance from ``detect_sources``); fixed centers if fit_positions is False.
    fwhm_pix : float
        FWHM of the model profile before pixel integration [pixel]; see ``fit_psf_width``.
    model : {"gaussian", "moffat"}
    beta : float
        Moffat exponent (used only for model="moffat").
    background : float or ndarray (Y, X)
        Background level subtracted before the fit, held fixed (as DAOPHOT's NSTAR does), in image units per pixel.
    noise : float or ndarray (Y, X), optional
        1-sigma noise of each pixel in image units, used as weights 1/sigma^2. Without it all pixels have equal
        weight and the uncertainties are scaled by the residual scatter. Pixels with non-finite image value or
        non-positive noise are ignored.
    fit_positions : bool
        Fit the centers as well as the fluxes. If False the problem is linear and solved exactly in one step.
    group_radius_pix : float, optional
        Critical separation for grouping [pixel]; default fit radius + 2 FWHM.
    fit_radius_pix : float, optional
        Only pixels within this distance of a star center enter its fit [pixel]; default 1.5 FWHM.
    max_group_size : int
        Groups with more stars are regrouped with half the separation.
    max_iterations : int
    tolerance_pix : float
        Convergence: the largest center step of an iteration below this [pixel].
    n_nodes : int
        Gauss-Legendre nodes per axis for the Moffat pixel integral.

    Returns
    -------
    PSFPhotometryResult
        Fluxes, fitted centers, uncertainties, goodness of fit per star, group membership.

    References
    ----------
    [Stetson1987] Section III.D, Eq. 8 and Eq. 10 (weights not used).

    Validity
    --------
    Measured with synthetic Gaussian stars of FWHM 3 pixels (the same model as the fit), flat background and Gaussian
    white noise of 5 per pixel, starting positions within 0.2 pixel of the truth; numbers in tests/unit/test_psf.py
    and scratch runs of 300 to 400 trials. S/N below is the nominal flux / (noise sqrt(4 pi sigma^2)); the exact
    S/N is 2.5% lower because the pixel integration widens the profile.

    - Isolated stars: the flux is unbiased within 1.5 standard errors of the mean from S/N 10 to 200, and the scatter
      equals the formal uncertainty to 4% (ratio 0.96 to 1.00) when a noise map is given. Without a noise map the
      uncertainty is scaled by the residual scatter and agrees to 10% (ratio 1.09 in one run of 300 trials).
    - Low S/N: with fitted positions the flux is biased high, +14% at S/N 3 and +5.6% at S/N 5 (the position follows
      noise peaks), +1.6 +- 1.2 flux units (0.7%) at S/N 10; with fit_positions=False it is unbiased at S/N 3 and 5.
    - Against aperture photometry (radius 1.2 FWHM divided by the enclosed fraction, sky from an annulus of 2.5 to
      4 FWHM): the aperture flux scatter is 1.46 to 1.60 times the PSF-fit scatter at every S/N from 3 to 200 (1.56 at
      S/N 5), and 1.26 to 1.45 times with the sky known exactly. These are sky-noise-limited stars; the
      source-noise-limited regime (Poisson noise of bright stars) was not measured.
    - Position: scatter equals the formal x_err to 5% for S/N 10 to 200 (0.19, 0.093, 0.0095 pixel at S/N 10, 20,
      200); at S/N 3 the scatter is 0.84 pixel against a formal 0.61 pixel (non-linear).
    - Blended pairs of equal FWHM 3 pixel stars, bright star S/N 50, 150 trials, random orientation, fitted together:
      for separations of 1, 1.5, 2 FWHM and flux ratios 1 and 4 the flux bias is within 1.6% of the flux of each star
      (within 2 standard errors) and the scatter agrees with the formal uncertainty. Scatter as a percentage of the
      star's flux with known positions / fitted positions: ratio 1: 2.1 / 2.6 (1 FWHM), 2.0 / 2.1 (1.5), 2.0 / 2.0 (2);
      ratio 4 faint star: 8.4 / 10.1 (1), 8.0 / 8.2 (1.5), 7.8 / 8.1 (2); the bright star of ratio 4 keeps 2.0 to 2.6%.
      Fitting the same stars without grouping (noiseless) gives biases of +26.7% (1 FWHM, ratio 1), +5.1% (1.5, ratio
      1), +0.5% (2, ratio 1) and, for the faint star of ratio 4, +107%, +20.4%, +1.9%.
    - Wrong width: the assumed FWHM scales the flux almost in proportion: -9.9% for a width 10% too small, +9.1% for 10%
      too large (noiseless; with noise at S/N 50 -9.8% and +9.4%). Use fit_psf_width, or fit the width on bright stars.
    - Wrong profile: a Moffat star (beta 3, FWHM 3) fitted with a Gaussian of the same FWHM gives 0.77 of the true flux
      for fit radii of 1.5 and 3 FWHM; the reduced chi-square is 2.65 at S/N 100 against 0.98 to 1.02 for the right model.

    The background is held fixed: a background error db shifts a star's flux by db * sum(P) / sum(P^2) over its fit
    pixels (checked exactly in the tests). Starting positions: an isolated star converged to the truth in 100 of 100
    trials for starting offsets of 0.5 to 3 pixels (1 FWHM) at S/N 50 and 99 of 100 at S/N 10 (offsets larger than that
    and offsets in blends were not measured); the step is limited to 0.5 FWHM per iteration. The cost is cubic in the group
    size (30 random stars in a 100 x 100 image take 0.03 s). Stars whose fit region lies outside the image return NaN.

    Verification
    ------------
    tests/unit/test_psf.py: unbiased flux and agreement of the scatter with the formal error; comparison with aperture
    photometry; blends at 1, 1.5 and 2 FWHM; position accuracy; wrong FWHM; grouping; edge handling.
    """
    data_image = np.asarray(image, dtype=float)
    if data_image.ndim != 2:
        raise ValueError("image must be two-dimensional")
    if fwhm_pix <= 0:
        raise ValueError("fwhm_pix must be positive")
    shape = data_image.shape
    x_in = np.atleast_1d(np.asarray(x, dtype=float))
    y_in = np.atleast_1d(np.asarray(y, dtype=float))
    if x_in.shape != y_in.shape or x_in.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    n = x_in.size
    bkg = _as_map(background, shape, "background")
    sigma_map = _as_map(noise, shape, "noise")
    fit_radius = 1.5 * fwhm_pix if fit_radius_pix is None else float(fit_radius_pix)
    group_radius = fit_radius + 2.0 * fwhm_pix if group_radius_pix is None else float(group_radius_pix)

    valid = np.isfinite(data_image)
    if bkg is not None:
        valid &= np.isfinite(bkg)
    if sigma_map is not None:
        valid &= np.isfinite(sigma_map) & (sigma_map > 0)
    residual_image = np.where(valid, data_image - (0.0 if bkg is None else bkg), 0.0)
    weight_image = np.where(valid, 1.0 if sigma_map is None else 1.0 / np.where(valid, sigma_map, 1.0), 0.0)

    flux = np.full(n, np.nan)
    flux_err = np.full(n, np.nan)
    x_fit = x_in.copy()
    y_fit = y_in.copy()
    x_err = np.full(n, np.nan)
    y_err = np.full(n, np.nan)
    chi2_red = np.full(n, np.nan)
    rms_res = np.full(n, np.nan)
    n_pix_star = np.zeros(n, dtype=np.int64)
    converged = np.zeros(n, dtype=bool)
    group_size = np.ones(n, dtype=np.int64)

    labels = _group_labels(x_in, y_in, group_radius, max_group_size, fit_radius)
    for g in range(labels.max() + 1 if n else 0):
        members = np.flatnonzero(labels == g)
        group_size[members] = members.size
        start = np.column_stack([x_in[members], y_in[members]])
        iy, ix = _pixels_near(valid, start, fit_radius)
        res = _fit_group(
            model,
            beta,
            n_nodes,
            float(fwhm_pix),
            ix.astype(float),
            iy.astype(float),
            residual_image[iy, ix],
            weight_image[iy, ix],
            start,
            fit_positions,
            max_iterations,
            tolerance_pix,
            max_shift_pix=0.5 * fwhm_pix,
            noise_given=sigma_map is not None,
        )
        if res.get("too_few"):
            continue
        flux[members] = res["flux"]
        flux_err[members] = res["flux_err"]
        x_fit[members] = res["pos"][:, 0]
        y_fit[members] = res["pos"][:, 1]
        x_err[members] = res["x_err"]
        y_err[members] = res["y_err"]
        converged[members] = res["converged"]
        # Goodness of fit per star: residuals of the group model on the pixels within the fit radius of that star.
        profile = np.zeros(ix.size)
        for k, (xk, yk) in enumerate(res["pos"]):
            profile += (
                res["flux"][k] * _profile(model, ix.astype(float), iy.astype(float), xk, yk, fwhm_pix, beta, n_nodes)[0]
            )
        resid = residual_image[iy, ix] - profile
        w = weight_image[iy, ix]
        n_par = 3 if fit_positions else 1
        for k, m in enumerate(members):
            near = (ix - res["pos"][k, 0]) ** 2 + (iy - res["pos"][k, 1]) ** 2 <= fit_radius**2
            count = int(near.sum())
            n_pix_star[m] = count
            if count > 0:
                rms_res[m] = float(np.sqrt(np.mean(resid[near] ** 2)))
            if sigma_map is not None and count > n_par:
                chi2_red[m] = float(np.sum((resid[near] * w[near]) ** 2) / (count - n_par))
    return PSFPhotometryResult(
        flux=flux,
        flux_err=flux_err,
        x=x_fit,
        y=y_fit,
        x_err=x_err,
        y_err=y_err,
        chi2_reduced=chi2_red,
        rms_residual=rms_res,
        n_pixels=n_pix_star,
        group_id=labels,
        group_size=group_size,
        converged=converged,
    )


@source(
    "PhotutilsPSF2026",
    where="User guide: PSF Photometry, 'Estimating the FWHM of Sources' (fit_fwhm) and 'Terminology'",
    status="read",
    note="read: the FWHM is estimated by fitting each source with a circular Gaussian PRF integrated over the pixels, "
    "so the result is the FWHM before pixel integration; the pixelated image has a variance larger by 1/12 pixel^2. "
    "This project's choices: selection of bright isolated stars by the initial flux and a distance criterion, a free "
    "constant background per star when none is given, the median over stars as the estimate and "
    "1.2533 * MAD-sigma / sqrt(n) as its uncertainty (the asymptotic standard error of a median of Gaussian "
    "data, a textbook result not taken from the cited source)",
)
def fit_psf_width(
    image: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    *,
    background: ArrayLike | float | None = None,
    noise: ArrayLike | float | None = None,
    initial_fwhm_pix: float = 3.0,
    fit_radius_pix: float | None = None,
    isolation_pix: float | None = None,
    bright_fraction: float = 0.5,
    min_stars: int = 3,
) -> tuple[float, float, int]:
    """Estimate the FWHM of the PSF from bright isolated stars.

    Each selected star is fitted with a circular Gaussian pixel response with free flux, center, FWHM (and a free
    constant background when background is None) on a circular cutout; the estimate is the median of the fitted FWHMs.

    Parameters
    ----------
    image : ndarray (Y, X)
    x, y : ndarray (N,)
        Star centers [pixel], for instance from ``detect_sources``.
    background : float or ndarray (Y, X), optional
        Background to subtract; if None a constant is fitted for each star.
    noise : float or ndarray (Y, X), optional
        1-sigma noise per pixel used as weights; equal weights if None.
    initial_fwhm_pix : float
        Starting value of the FWHM [pixel].
    fit_radius_pix : float, optional
        Cutout radius [pixel]; default max(4, 2.5 * initial_fwhm_pix).
    isolation_pix : float, optional
        A star is isolated if no other listed star lies within this distance [pixel]; default 2 * fit radius.
    bright_fraction : float
        Fraction of the isolated stars, taken in order of decreasing starting flux, to fit.
    min_stars : int
        Minimum number of stars to fit; the brightest ones are added if the fraction gives fewer.

    Returns
    -------
    (fwhm_pix, fwhm_err_pix, n_used) : the FWHM of the Gaussian before integration over the pixels [pixel], its
    standard error [pixel] and the number of stars used. To compare with the width of a pixelated image add
    1/12 pixel^2 to the variance: sqrt(fwhm^2 + 8 ln2 / 12).

    References
    ----------
    [PhotutilsPSF2026] PSF Photometry: Estimating the FWHM of Sources.

    Validity
    --------
    Measured with synthetic Gaussian stars on 120 x 120 pixel fields of 12 stars (10 fields per case): the mean
    estimate is 1.994 and 2.002 (true 2.0), 2.980 and 2.998 (3.0), 4.494 and 4.501 (4.5) at S/N 20 and 100, with a
    field-to-field scatter of 0.03 and 0.005 (FWHM 2), 0.03 and 0.007 (3), 0.05 and 0.01 (4.5), close to the stated error;
    typically 3 to 4 stars were used per field. Stars with a non-Gaussian PSF give the width of the best-fitting
    Gaussian, which depends on the fit radius. Unlisted neighbours inside the cutout bias the width high;
    crowded fields need a larger isolation distance. Saturated or cosmic-ray-affected stars must be removed beforehand.

    Verification
    ------------
    tests/unit/test_psf.py: recovery of the input FWHM over a range of widths and noise levels; rejection of blended stars.
    """
    data_image = np.asarray(image, dtype=float)
    shape = data_image.shape
    x_in = np.atleast_1d(np.asarray(x, dtype=float))
    y_in = np.atleast_1d(np.asarray(y, dtype=float))
    radius = max(4.0, 2.5 * initial_fwhm_pix) if fit_radius_pix is None else float(fit_radius_pix)
    isolation = 2.0 * radius if isolation_pix is None else float(isolation_pix)
    bkg = _as_map(background, shape, "background")
    sigma_map = _as_map(noise, shape, "noise")
    ny, nx = shape

    # Isolated stars whose whole cutout lies inside the image.
    inside = (x_in - radius >= 0) & (x_in + radius <= nx - 1) & (y_in - radius >= 0) & (y_in + radius <= ny - 1)
    tree = cKDTree(np.column_stack([x_in, y_in]))
    n_neighbours = np.array(
        [len(tree.query_ball_point([a, b], isolation)) - 1 for a, b in zip(x_in, y_in, strict=True)]
    )
    candidates = np.flatnonzero(inside & (n_neighbours == 0))
    if candidates.size == 0:
        raise ValueError("no isolated star with a complete cutout; widen the list or reduce isolation_pix")

    def cutout(i: int):
        positions = np.array([[x_in[i], y_in[i]]])
        iy, ix = _pixels_near(np.isfinite(data_image), positions, radius)
        d = data_image[iy, ix] - (0.0 if bkg is None else bkg[iy, ix])
        w = np.ones(ix.size) if sigma_map is None else 1.0 / sigma_map[iy, ix]
        return ix.astype(float), iy.astype(float), d, w

    starting_flux = np.array([np.sum(cutout(i)[2]) for i in candidates])
    order = candidates[np.argsort(-starting_flux)]
    n_fit = min(order.size, max(min_stars, int(np.ceil(bright_fraction * order.size))))
    widths = []
    for i in order[:n_fit]:
        xs, ys, d, w = cutout(i)
        if d.size < 8:
            continue
        peak = np.max(d)
        flux0 = max(np.sum(d), 1e-3 * abs(peak) + 1e-12)
        free_bkg = bkg is None

        def residuals(theta, xs=xs, ys=ys, d=d, w=w, free_bkg=free_bkg):
            flux_t, x0, y0, fwhm_t = theta[:4]
            level = theta[4] if free_bkg else 0.0
            sigma = fwhm_t * FWHM_TO_SIGMA
            px, _ = _gauss_axis(xs, x0, sigma)
            py, _ = _gauss_axis(ys, y0, sigma)
            return (d - level - flux_t * px * py) * w

        start = [flux0, x_in[i], y_in[i], initial_fwhm_pix] + ([float(np.median(d))] if free_bkg else [])
        lower = [-np.inf, x_in[i] - 2.0, y_in[i] - 2.0, 0.5] + ([-np.inf] if free_bkg else [])
        upper = [np.inf, x_in[i] + 2.0, y_in[i] + 2.0, radius * 2.0] + ([np.inf] if free_bkg else [])
        fit = optimize.least_squares(residuals, start, bounds=(lower, upper), method="trf", x_scale="jac")
        if fit.success and lower[3] * 1.001 < fit.x[3] < upper[3] * 0.999:
            widths.append(fit.x[3])
    if not widths:
        raise ValueError("no star fit converged to a valid FWHM")
    w_arr = np.asarray(widths)
    fwhm = float(np.median(w_arr))
    err = float(1.2533 * mad_scale(w_arr) / np.sqrt(w_arr.size)) if w_arr.size > 1 else float("nan")
    return fwhm, err, int(w_arr.size)
