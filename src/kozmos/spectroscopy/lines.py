"""Spectral-line measurement: equivalent width, Gaussian line fits, the Voigt profile.

The equivalent-width definition and its error propagation follow Vollmann & Eversberg
[VollmannEversberg2006] (read from the arXiv TeX source astro-ph/0606341 on 25 November 2026).
The Voigt profile is written through the Faddeeva function as in the SciPy documentation
[SciPyDocs2026] and [Zaghloul2011] Eq. (1). Cayrel (1988), the other classical source for the
equivalent-width uncertainty, could not be obtained; its formula is not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit
from scipy.special import wofz

from kozmos._sources import source

Array = NDArray[np.float64]

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))
_FWHM_PER_SIGMA = float(2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass(frozen=True, slots=True)
class EquivalentWidth:
    """Equivalent width with its uncertainty.

    Attributes
    ----------
    value_angstrom : float
        Equivalent width, positive for absorption [wavelength unit of the input, angstrom by convention].
    sigma_angstrom : float
        One-sigma uncertainty [same unit].
    window_width_angstrom : float
        Width of the integration window (lambda_2 - lambda_1) [same unit].
    """

    value_angstrom: float
    sigma_angstrom: float
    window_width_angstrom: float


@dataclass(frozen=True, slots=True)
class GaussianLineFit:
    """Result of a Gaussian line fit on a polynomial continuum; ``*_sigma`` are one-sigma uncertainties.

    The model is flux = continuum(lambda) + amplitude * exp(-(lambda - centre)^2 / (2 sigma^2)); the amplitude is
    negative for an absorption line. ``depth`` is the line depth relative to the continuum at the centre
    (-amplitude / continuum), positive for absorption.
    """

    centre: float
    centre_sigma: float
    sigma: float
    sigma_sigma: float
    fwhm: float
    fwhm_sigma: float
    amplitude: float
    amplitude_sigma: float
    depth: float
    depth_sigma: float
    equivalent_width: float
    equivalent_width_sigma: float
    continuum_at_centre: float
    continuum_coefficients: tuple[float, ...]
    reduced_chi2: float
    covariance: Array


def _pixel_weights(wavelength: Array, lambda_1: float, lambda_2: float) -> Array:
    """Overlap of each pixel (edges at mid-points between centres) with the window [lambda_1, lambda_2]."""
    mid = 0.5 * (wavelength[1:] + wavelength[:-1])
    edges = np.concatenate(
        [[wavelength[0] - (mid[0] - wavelength[0])], mid, [wavelength[-1] + (wavelength[-1] - mid[-1])]]
    )
    lo = np.clip(edges[:-1], lambda_1, lambda_2)
    hi = np.clip(edges[1:], lambda_1, lambda_2)
    return np.maximum(hi - lo, 0.0)


@source(
    "VollmannEversberg2006",
    where="Eq. 1 (definition; TeX label eq1) and the derivatives following Eq. 3, Eq. for sigma^2(W) (TeX label eq10)",
    status="read",
    note=(
        "W = sum_i w_i (1 - F_i/Fc_i), the pixel-wise discretisation of the integral in Eq. 1 with w_i the overlap of "
        "pixel i with [lambda_1, lambda_2] (pixel edges at mid-points; this project's choice). Variance: "
        "sum_i (w_i sigma_i / Fc_i)^2 + ((window - W) sigma_c / Fc)^2, the per-pixel generalisation of the paper's "
        "dW/dF = -window/Fc and dW/dFc = (window - W)/Fc propagated as in TeX label eq10; the per-pixel form and the "
        "use of the weighted mean continuum in the second term are this project's choices."
    ),
)
def equivalent_width(
    wavelength_angstrom: ArrayLike,
    flux: ArrayLike,
    window_angstrom: tuple[float, float],
    *,
    flux_sigma: ArrayLike,
    continuum: ArrayLike = 1.0,
    continuum_sigma: float = 0.0,
) -> EquivalentWidth:
    """Measure the equivalent width of a line by direct integration, with an uncertainty.

    W = integral over [lambda_1, lambda_2] of (1 - F / Fc) d lambda
    var(W) = sum_i (w_i sigma_i / Fc_i)^2 + ((lambda_2 - lambda_1 - W) sigma_c / Fc)^2

    Parameters
    ----------
    wavelength_angstrom : ndarray
        Pixel-centre wavelengths, strictly increasing [angstrom].
    flux : ndarray
        Flux in each pixel (any unit).
    window_angstrom : tuple of float
        Integration limits (lambda_1, lambda_2) [angstrom]; partial pixels are weighted by their overlap.
    flux_sigma : float or ndarray
        One-sigma noise of each pixel flux, same unit as ``flux``.
    continuum : float or ndarray
        Continuum flux (scalar or per pixel), same unit as ``flux``.
    continuum_sigma : float
        One-sigma uncertainty of the continuum level over the line, same unit as ``flux`` (default 0: continuum
        known exactly). Pixel noise is uncorrelated, and so is the continuum error with it.

    Returns
    -------
    EquivalentWidth
        Value, one-sigma uncertainty and window width [angstrom].

    References
    ----------
    [VollmannEversberg2006] Eq. 1 and the error expansion.

    Validity
    --------
    Assumes uncorrelated pixel noise and a continuum known over the window. Pixels are treated as flat bins;
    the integration error is of order (pixel width)^2 times the line curvature, so the profile should span
    several pixels. Measured on Gaussian lines (depth 0.3, sigma 1.5 angstrom, pixel 0.3 angstrom, window
    +-5 sigma, 1500 trials per S/N): mean error within 0.07 formal standard errors from S/N 10 to 300 per
    pixel, and scatter / formal uncertainty between 0.98 and 1.03 (known continuum) and between 0.98 and 1.00
    (continuum from 7 side pixels, its error passed as ``continuum_sigma``); see tests. A continuum estimated
    from the data adds an error that the caller must pass as ``continuum_sigma``.

    Verification
    ------------
    tests/unit/test_spectroscopy.py: exact Gaussian area, box line, partial-pixel weights, Monte Carlo scatter
    versus the formal uncertainty at four S/N.
    """
    lam = np.asarray(wavelength_angstrom, dtype=float)
    f = np.asarray(flux, dtype=float)
    if lam.ndim != 1 or f.shape != lam.shape or lam.size < 3:
        raise ValueError("wavelength_angstrom and flux must be 1-D arrays of the same length (at least 3)")
    if np.any(np.diff(lam) <= 0):
        raise ValueError("wavelength_angstrom must be strictly increasing")
    lambda_1, lambda_2 = float(window_angstrom[0]), float(window_angstrom[1])
    if not lambda_2 > lambda_1:
        raise ValueError("the window must satisfy lambda_2 > lambda_1")
    if lambda_1 < lam[0] or lambda_2 > lam[-1]:
        raise ValueError("the window must lie inside the wavelength array")
    fc = np.broadcast_to(np.asarray(continuum, dtype=float), lam.shape)
    sigma = np.broadcast_to(np.asarray(flux_sigma, dtype=float), lam.shape)
    if np.any(fc <= 0):
        raise ValueError("the continuum must be positive")
    if np.any(sigma < 0) or continuum_sigma < 0:
        raise ValueError("uncertainties must not be negative")
    w = _pixel_weights(lam, lambda_1, lambda_2)
    width = float(np.sum(w))
    value = float(np.sum(w * (1.0 - f / fc)))
    mean_continuum = float(np.sum(w * fc) / width)
    variance = float(np.sum((w * sigma / fc) ** 2) + ((width - value) * continuum_sigma / mean_continuum) ** 2)
    return EquivalentWidth(value, float(np.sqrt(variance)), width)


@source(
    "VollmannEversberg2006",
    where="TeX label eq4 (Sect. 'The error of the equivalent width', low- and high-flux lines)",
    status="read",
    note=(
        "sigma(W) = sqrt(1 + Fc/F) (window - W) / (S/N). In the paper S/N is that of the continuum mean over the "
        "line window; the paper's own correction note says it equals sqrt(M) times the per-pixel S/N for a window "
        "of M pixels. The printed equation numbers follow the order of appearance in the paper; the TeX label is "
        "given instead of a number because the number was not checked against the printed article."
    ),
)
def equivalent_width_sigma_vollmann_eversberg(
    window_width_angstrom: ArrayLike,
    equivalent_width_angstrom: ArrayLike,
    snr_over_window: ArrayLike,
    mean_line_flux: ArrayLike = 1.0,
    mean_continuum_flux: ArrayLike = 1.0,
) -> Array:
    """Compute the equivalent-width uncertainty of Vollmann & Eversberg for photon noise.

    sigma(W) = sqrt(1 + Fc_mean / F_mean) (window - W) / (S/N)

    Parameters
    ----------
    window_width_angstrom : float or ndarray
        Window width lambda_2 - lambda_1 [angstrom].
    equivalent_width_angstrom : float or ndarray
        Equivalent width W [angstrom].
    snr_over_window : float or ndarray
        Signal-to-noise ratio of the mean continuum flux over the window (sqrt(M) times the per-pixel S/N for M
        pixels), dimensionless.
    mean_line_flux : float or ndarray
        Mean flux in the line window, same unit as the continuum flux (only the ratio enters).
    mean_continuum_flux : float or ndarray
        Mean continuum flux over the window.

    Returns
    -------
    ndarray
        One-sigma uncertainty of W [angstrom].

    References
    ----------
    [VollmannEversberg2006] TeX label eq4.

    Validity
    --------
    The paper's model: both the line mean and the continuum level are determined from M pixels each, with Poisson
    noise in the line. Measured by Monte Carlo (40000 trials, see tests) in exactly that model, the scatter of W
    divided by this formula is 1.000 to 1.008 for depth 0.3 and 0.6 at per-pixel S/N 30 and 100 (M = 40
    pixels). If the continuum is instead estimated from 20 times more pixels than the line window, the scatter
    is 0.78 (depth 0.3) and 0.86 (depth 0.6) times the formula, which then overestimates it by 1.28 and 1.17;
    the pixel-wise ``equivalent_width`` is the better choice there.

    Verification
    ------------
    tests/unit/test_spectroscopy.py: weak-line limit sqrt(2) (window - W) / SNR, Monte Carlo in the paper's model.
    """
    window = np.asarray(window_width_angstrom, dtype=float)
    w = np.asarray(equivalent_width_angstrom, dtype=float)
    ratio = np.asarray(mean_continuum_flux, dtype=float) / np.asarray(mean_line_flux, dtype=float)
    return np.sqrt(1.0 + ratio) * (window - w) / np.asarray(snr_over_window, dtype=float)


def _gaussian(x: Array, amplitude: float, centre: float, sigma: float) -> Array:
    return amplitude * np.exp(-0.5 * ((x - centre) / sigma) ** 2)


def _model(x: Array, params: Array, order: int, x0: float, scale: float) -> Array:
    amplitude, centre, sigma = params[:3]
    t = (x - x0) / scale
    continuum = np.polynomial.polynomial.polyval(t, params[3 : 4 + order])
    return continuum + _gaussian(x, amplitude, centre, sigma)


@source(
    "SciPyDocs2026",
    where="scipy.optimize.curve_fit docstring: parameters sigma and absolute_sigma, covariance pcov",
    status="read",
    note=(
        "Bounded least squares through curve_fit (trust-region reflective; centre inside the data, sigma between 0.05 pixel and the data span). With flux_sigma given, absolute_sigma=True (pcov in "
        "absolute units); without it the covariance is scaled by the residual variance (docstring: absolute_sigma "
        "False). The model (Gaussian on a polynomial continuum), the starting values and the propagation of "
        "FWHM, depth and equivalent width from the covariance (first-order, finite-difference Jacobian) are this "
        "project's choices."
    ),
)
def fit_gaussian_line(
    wavelength: ArrayLike,
    flux: ArrayLike,
    flux_sigma: ArrayLike | None = None,
    *,
    continuum_order: int = 1,
) -> GaussianLineFit:
    """Fit a Gaussian line on a polynomial continuum and return parameters with uncertainties.

    flux(lambda) = sum_j c_j t^j + A exp(-(lambda - mu)^2 / (2 s^2)), t = (lambda - lambda_mid) / half_range
    FWHM = 2 sqrt(2 ln 2) s, depth = -A / continuum(mu), EW = -A s sqrt(2 pi) / continuum(mu)

    Parameters
    ----------
    wavelength : ndarray
        Wavelengths of the pixels (any unit; outputs carry it, e.g. angstrom), strictly increasing.
    flux : ndarray
        Flux in each pixel (any unit).
    flux_sigma : float or ndarray, optional
        One-sigma noise per pixel, same unit as ``flux``. If omitted, uncertainties are scaled to the scatter of
        the residuals (reduced chi-square 1 by construction).
    continuum_order : int
        Degree of the polynomial continuum (0 constant, 1 linear, default 1).

    Returns
    -------
    GaussianLineFit
        Centre, sigma, FWHM, amplitude, depth, equivalent width (positive for absorption), continuum at the
        centre, all with one-sigma uncertainties, plus the reduced chi-square and the covariance matrix of
        (amplitude, centre, sigma, continuum coefficients).

    References
    ----------
    [SciPyDocs2026] scipy.optimize.curve_fit (sigma, absolute_sigma, pcov).

    Validity
    --------
    The fit window should contain the line and enough continuum on both sides (at least 3 sigma each side); the
    line is assumed Gaussian and unblended. Raises RuntimeError if the optimiser does not converge. Measured on
    synthetic absorption lines (depth 0.3, sigma 2 pixels, 24 pixels, linear continuum free, 600 trials per
    S/N): for S/N per pixel 20 to 300 (equivalent-width significance 7 to 109) the scatter of centre, sigma,
    amplitude, equivalent width, depth and FWHM divided by the formal uncertainty is between 0.94 and 1.00, and
    the mean error is within 0.06 formal standard errors for centre and sigma and within 0.22 for amplitude
    and depth (S/N 20; 0.07 at S/N 300). At S/N 10 (significance 3.8) the fit has heavy tails (scatter of the
    centre 2.6 pixels against a median formal error of 0.46 pixel) and the formal errors do not apply.

    Verification
    ------------
    tests/unit/test_spectroscopy.py: noiseless recovery, Monte Carlo scatter versus formal errors at three S/N.
    tests/oracle/test_spectroscopy_oracle.py: astropy.modeling LevMarLSQFitter on Gaussian1D.
    """
    x = np.asarray(wavelength, dtype=float)
    y = np.asarray(flux, dtype=float)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("wavelength and flux must be 1-D arrays of the same length")
    if continuum_order < 0:
        raise ValueError("continuum_order must be >= 0")
    n_params = 4 + continuum_order
    if x.size <= n_params:
        raise ValueError("not enough points for the number of parameters")
    sigma_arr = None if flux_sigma is None else np.broadcast_to(np.asarray(flux_sigma, dtype=float), x.shape)
    x0 = 0.5 * (x[0] + x[-1])
    scale = 0.5 * (x[-1] - x[0])

    # Starting values: a grid search over (centre, sigma); for each pair the amplitude and the continuum
    # coefficients follow from a linear weighted least-squares solution. This project's choice.
    w_fit = np.ones(x.size) if sigma_arr is None else 1.0 / sigma_arr
    t = (x - x0) / scale
    cont_basis = np.vander(t, continuum_order + 1, increasing=True)
    dx = float(np.median(np.diff(x)))
    best = (np.inf, None)
    centres = x[:: max(1, x.size // 80)]
    for s_try in np.geomspace(0.6 * dx, 0.3 * (x[-1] - x[0]), 9):
        for mu_try in centres:
            design = np.column_stack([_gaussian(x, 1.0, mu_try, s_try), cont_basis]) * w_fit[:, None]
            sol, res, rank, _ = np.linalg.lstsq(design, y * w_fit, rcond=None)
            if rank < design.shape[1]:
                continue
            chi = float(np.sum((design @ sol - y * w_fit) ** 2))
            if chi < best[0]:
                best = (chi, (sol, mu_try, s_try))
    if best[1] is None:
        raise RuntimeError("no starting values found for the Gaussian fit")
    sol, mu0, s0 = best[1]
    p0 = np.zeros(n_params)
    p0[:3] = [sol[0], mu0, s0]
    p0[3:] = sol[1:]

    def func(xx: Array, *p: float) -> Array:
        return _model(xx, np.asarray(p), continuum_order, x0, scale)

    lower = np.full(n_params, -np.inf)
    upper = np.full(n_params, np.inf)
    lower[1:3] = [x[0], 0.05 * dx]
    upper[1:3] = [x[-1], x[-1] - x[0]]
    popt, pcov = curve_fit(
        func,
        x,
        y,
        p0=p0,
        sigma=sigma_arr,
        absolute_sigma=sigma_arr is not None,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        max_nfev=20000,
    )
    if not np.all(np.isfinite(pcov)):
        raise RuntimeError("the covariance of the Gaussian fit is not finite (degenerate fit)")
    popt = np.asarray(popt)
    amplitude, centre, sig = popt[:3]
    sig = abs(sig)
    popt[2] = sig
    resid = (y - func(x, *popt)) / (1.0 if sigma_arr is None else sigma_arr)
    dof = x.size - n_params
    chi2 = float(np.sum(resid**2))
    reduced = chi2 / dof if sigma_arr is not None else 1.0

    def derived(p: Array) -> Array:
        a, mu, s = p[0], p[1], abs(p[2])
        cont = np.polynomial.polynomial.polyval((mu - x0) / scale, p[3:])
        return np.array([s * _FWHM_PER_SIGMA, -a / cont, -a * s * _SQRT_2PI / cont, cont])

    base = derived(popt)
    jac = np.empty((4, n_params))
    for j in range(n_params):
        step = 1e-6 * max(abs(popt[j]), 1e-3)
        shifted = popt.copy()
        shifted[j] += step
        jac[:, j] = (derived(shifted) - base) / step
    cov_derived = jac @ pcov @ jac.T
    err = np.sqrt(np.diag(pcov))
    err_derived = np.sqrt(np.diag(cov_derived))
    return GaussianLineFit(
        centre=float(centre),
        centre_sigma=float(err[1]),
        sigma=float(sig),
        sigma_sigma=float(err[2]),
        fwhm=float(base[0]),
        fwhm_sigma=float(err_derived[0]),
        amplitude=float(amplitude),
        amplitude_sigma=float(err[0]),
        depth=float(base[1]),
        depth_sigma=float(err_derived[1]),
        equivalent_width=float(base[2]),
        equivalent_width_sigma=float(err_derived[2]),
        continuum_at_centre=float(base[3]),
        continuum_coefficients=tuple(float(c) for c in popt[3:]),
        reduced_chi2=float(reduced),
        covariance=pcov,
    )


@source(
    "SciPyDocs2026",
    where="scipy.special.voigt_profile Notes: V(x; sigma, gamma) = Re[w(z)] / (sigma sqrt(2 pi)), z = (x + i gamma) / (sqrt(2) sigma)",
    status="read",
    note=(
        "Evaluated with scipy.special.wofz (the Faddeeva function). The limits sigma = 0 (Lorentzian) and gamma = 0 "
        "(Gaussian) are handled explicitly, as the docstring states them; the explicit branches are this project's "
        "implementation."
    ),
)
@source(
    "Zaghloul2011",
    where="Eq. (1): w(z) = exp(-z^2) erfc(-iz) and Re w(x + iy) = V(x, y), the real Voigt function (y > 0)",
    status="read",
    note="only the definition of w and its real part is taken from the paper; the sigma/gamma scaling is from SciPyDocs2026.",
)
def voigt_profile(x: ArrayLike, sigma: ArrayLike, gamma: ArrayLike) -> Array:
    """Compute the area-normalised Voigt profile (Gaussian convolved with Lorentzian).

    V(x; sigma, gamma) = Re[w(z)] / (sigma sqrt(2 pi)),  z = (x + i gamma) / (sqrt(2) sigma)
    w(z) = exp(-z^2) erfc(-i z)  (Faddeeva function)

    Parameters
    ----------
    x : float or ndarray
        Offset from the line centre (any unit).
    sigma : float or ndarray
        Standard deviation of the Gaussian part, same unit as ``x``; must be >= 0.
    gamma : float or ndarray
        Half-width at half-maximum of the Lorentzian part, same unit as ``x``; must be >= 0.

    Returns
    -------
    ndarray
        Profile value, integrating to 1 over x, in 1 / unit of ``x``.

    References
    ----------
    [SciPyDocs2026] voigt_profile Notes.
    [Zaghloul2011] Eq. (1).

    Validity
    --------
    sigma = gamma = 0 returns inf at x = 0 and 0 elsewhere. Unit area: the numerical integral over the real line
    equals 1 to better than 1e-9 for gamma/sigma from 1e-3 to 1e3 (measured, see tests). Agreement with
    scipy.special.voigt_profile to a relative 2.4e-14 and with the shape of astropy Voigt1D (method wofz) to a
    relative 7e-15 (see tests/oracle). The accuracy of the
    underlying wofz is that of SciPy's Faddeeva implementation (the SciPy docstring cites Steven G. Johnson's
    package; no accuracy figure was read).

    Verification
    ------------
    tests/unit/test_spectroscopy.py: Gaussian and Lorentzian limits, unit area, peak value, symmetry.
    tests/oracle/test_spectroscopy_oracle.py: scipy.special.voigt_profile and astropy Voigt1D.
    """
    xa, sa, ga = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (x, sigma, gamma)))
    if np.any(sa < 0) or np.any(ga < 0):
        raise ValueError("sigma and gamma must be >= 0")
    out = np.zeros(xa.shape)
    gauss_only = (sa > 0) & (ga == 0)
    lorentz_only = (sa == 0) & (ga > 0)
    both = (sa > 0) & (ga > 0)
    out = np.where(
        gauss_only, np.exp(-0.5 * (xa / np.where(sa > 0, sa, 1.0)) ** 2) / (np.where(sa > 0, sa, 1.0) * _SQRT_2PI), out
    )
    out = np.where(lorentz_only, np.where(ga > 0, ga, 1.0) / (np.pi * (xa**2 + np.where(ga > 0, ga, 1.0) ** 2)), out)
    s_safe = np.where(both, sa, 1.0)
    z = (xa + 1j * np.where(both, ga, 1.0)) / (np.sqrt(2.0) * s_safe)
    out = np.where(both, wofz(z).real / (s_safe * _SQRT_2PI), out)
    out = np.where((sa == 0) & (ga == 0), np.where(xa == 0, np.inf, 0.0), out)
    return out
