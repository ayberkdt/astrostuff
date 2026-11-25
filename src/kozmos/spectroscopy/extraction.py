"""Extraction of a 1-D spectrum from a 2-D spectral image: box (sum) and optimal extraction.

The algorithm is Horne's [Horne1986]. The equations and the steps of its Table I, the profile construction of
Sect. II.D, the cosmic-ray rejection of Sect. II.E and the iteration of Sect. II.F were read on 25 November 2026
from the ADS scan of the paper (page images of PASP 98, 609; the scan has no text layer).

Image layout. The spectrum runs along the detector rows or columns. All functions take ``spatial_axis``: the axis
of the 2-D array that is the spatial (cross-dispersion) direction x; the other axis is the wavelength direction.
Pixel values are in analog-to-digital units (ADU), already bias-subtracted and flat-fielded (Table I step 1), so
that ``D`` of the paper is the input array. ``gain_e_per_adu`` is the paper's Q (effective number of photons per
data number) and ``read_noise_e / gain_e_per_adu`` is sqrt(V_0), the read noise in data numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BoxExtraction:
    """Result of a box (sum) extraction.

    Attributes
    ----------
    flux : ndarray
        Extracted flux per wavelength pixel [ADU].
    variance : ndarray
        Variance of the flux [ADU^2].
    """

    flux: Array
    variance: Array


@dataclass(frozen=True, slots=True)
class OptimalExtraction:
    """Result of an optimal extraction.

    Attributes
    ----------
    flux : ndarray
        Optimal spectrum per wavelength pixel [ADU], shape (n_wavelength,).
    variance : ndarray
        Variance of the optimal spectrum [ADU^2].
    standard_flux : ndarray
        Box (standard) spectrum over the same aperture [ADU] (Table I step 4).
    profile : ndarray
        Normalised spatial profile P, shape (n_spatial, n_wavelength); zero outside the aperture.
    mask : ndarray
        Boolean image, True where the pixel was used (inside the aperture and not rejected as a cosmic ray).
    rejected : ndarray
        Boolean image, True for pixels rejected by the cosmic-ray step.
    sky : ndarray
        Sky image that was subtracted [ADU], shape (n_spatial, n_wavelength).
    iterations : int
        Number of passes of the cosmic-ray loop (Table I steps 6-8).
    """

    flux: Array
    variance: Array
    standard_flux: Array
    profile: Array
    mask: NDArray[np.bool_]
    rejected: NDArray[np.bool_]
    sky: Array
    iterations: int


def _to_spatial_first(data: ArrayLike, spatial_axis: int) -> Array:
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError("the 2-D spectral image must be a 2-D array")
    if spatial_axis not in (0, 1):
        raise ValueError("spatial_axis must be 0 or 1")
    return arr if spatial_axis == 0 else arr.T


def _check_noise(gain_e_per_adu: float, read_noise_e: float) -> tuple[float, float]:
    if not gain_e_per_adu > 0 or read_noise_e < 0:
        raise ValueError("gain_e_per_adu must be positive and read_noise_e must not be negative")
    return float((read_noise_e / gain_e_per_adu) ** 2), float(gain_e_per_adu)


def _aperture_mask(n_spatial: int, aperture: tuple[int, int] | None) -> NDArray[np.bool_]:
    x1, x2 = (0, n_spatial) if aperture is None else (int(aperture[0]), int(aperture[1]))
    if not 0 <= x1 < x2 <= n_spatial:
        raise ValueError("aperture must satisfy 0 <= x1 < x2 <= number of spatial pixels")
    mask = np.zeros(n_spatial, dtype=bool)
    mask[x1:x2] = True
    return mask


def _sky_image(sky: ArrayLike | None, shape: tuple[int, int], spatial_axis: int) -> Array:
    if sky is None:
        return np.zeros(shape)
    s = np.asarray(sky, dtype=float)
    if s.ndim == 2:
        s = s if spatial_axis == 0 else s.T
    return np.broadcast_to(s, shape).astype(float)


def _clipped_fit(t: Array, y: Array, weight: Array, order: int, clip2: float, max_iterations: int = 10) -> Array:
    """Weighted Chebyshev fit with iterative rejection of pixels whose weighted squared residual exceeds clip2."""
    w = np.where(np.isfinite(y) & np.isfinite(weight), weight, 0.0)
    y = np.where(w > 0, y, 0.0)
    keep = w > 0
    if np.count_nonzero(keep) <= order + 1:
        raise ValueError("not enough usable points for the polynomial fit")
    fit = np.zeros_like(y)
    for _ in range(max_iterations):
        coeffs = np.polynomial.chebyshev.chebfit(t[keep], y[keep], order, w=np.sqrt(w[keep]))
        fit = np.polynomial.chebyshev.chebval(t, coeffs)
        new_keep = (w > 0) & (w * (y - fit) ** 2 <= clip2)
        if np.count_nonzero(new_keep) <= order + 1 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return fit


@source(
    "Horne1986",
    where="Eq. 2 and Eq. 3 (standard spectrum and its variance); Eq. 12 (pixel variance); Table I steps 2 and 4",
    status="read",
    note=(
        "f = sum_x (D - S), var f = sum_x V with V = V_0 + |D| / Q (Eq. 12). V_0 = (read_noise_e / gain)^2 and "
        "Q = gain_e_per_adu is this project's mapping of the paper's V_0 and Q onto the detector parameters. The "
        "aperture is the half-open index range x1 <= x < x2; whole pixels only (this project's choice)."
    ),
)
def box_extract(
    data: ArrayLike,
    *,
    gain_e_per_adu: float,
    read_noise_e: float,
    sky: ArrayLike | None = None,
    aperture: tuple[int, int] | None = None,
    spatial_axis: int = 0,
) -> BoxExtraction:
    """Extract a spectrum by summing the sky-subtracted pixels over an aperture (standard extraction).

    f = sum_{x1 <= x < x2} (D - S),  var f = sum V,  V = V_0 + |D| / Q

    Parameters
    ----------
    data : ndarray
        2-D spectral image [ADU], bias-subtracted and flat-fielded.
    gain_e_per_adu : float
        Gain Q [electrons / ADU], positive.
    read_noise_e : float
        Read noise [electrons rms], not negative.
    sky : float or ndarray, optional
        Sky image or value to subtract [ADU]; broadcast against ``data``. Default 0.
    aperture : tuple of int, optional
        Spatial pixel range (x1, x2), x1 <= x < x2. Default: the whole spatial axis.
    spatial_axis : int
        Axis of ``data`` along which the spatial profile varies (0 or 1).

    Returns
    -------
    BoxExtraction
        Flux [ADU] and variance [ADU^2] per wavelength pixel.

    References
    ----------
    [Horne1986] Eqs. 2, 3, 12.

    Validity
    --------
    Unbiased for any profile that lies inside the aperture; the variance counts every aperture pixel with equal
    weight and is therefore larger than the optimal one. Cosmic rays and bad pixels are not treated. Measured
    on synthetic images (Poisson plus read noise, 25 images of 300 columns, 13-pixel aperture): the mean of
    (flux - truth) / sqrt(variance) is -0.01 and its standard deviation 1.010 at 300 e per column and 1.015
    at 30 e per column. Eq. 3 neglects the noise of the sky subtraction, as in the paper.

    Verification
    ------------
    tests/unit/test_spectroscopy.py: known flux, variance versus scatter, comparison with optimal extraction.
    """
    d_img = _to_spatial_first(data, spatial_axis)
    v0, q = _check_noise(gain_e_per_adu, read_noise_e)
    ap = _aperture_mask(d_img.shape[0], aperture)
    s_img = _sky_image(sky, d_img.shape, spatial_axis)
    variance_img = v0 + np.abs(d_img) / q
    flux = np.sum((d_img - s_img)[ap], axis=0)
    variance = np.sum(variance_img[ap], axis=0)
    return BoxExtraction(flux=flux, variance=variance)


@source(
    "Horne1986",
    where="Table I step 3 (fit sky background, FIT_x[D; V]); Sect. II.A (weighted polynomial per wavelength, "
    "weights from Step 2 variances); Sect. II.D (iterative sigma-clipping, threshold sigma_clip^2 about 16)",
    status="read",
    note=(
        "weighted Chebyshev polynomial of degree `order` in x, fitted per wavelength column to the pixels flagged "
        "in `sky_pixels`, then evaluated at every x. Weights are 1/V with V = V_0 + |D|/Q (Eq. 12) in the first of three "
        "passes; in the other two V = V_0 + max(fit, 0)/Q from the fitted sky (this project's choice: weights taken "
        "from the noisy pixel values bias the fit low by about 1/Q ADU, measured). Clipping: pixels with "
        "(D - fit)^2 / V > clip_sigma2 are dropped and the fit repeated until the set is unchanged (at most 10 "
        "passes). The Chebyshev basis, the three passes and the 10-pass cap are this project's choices; the paper says only 'quadratic "
        "or cubic polynomials in x' and iterative sigma-clipping."
    ),
)
def fit_sky_background(
    data: ArrayLike,
    sky_pixels: ArrayLike,
    *,
    gain_e_per_adu: float,
    read_noise_e: float,
    order: int = 1,
    clip_sigma2: float = 16.0,
    spatial_axis: int = 0,
) -> Array:
    """Fit a smooth sky background in the spatial direction at every wavelength.

    S(x, lambda) = weighted polynomial fit over the sky pixels, V = V_0 + |D| / Q, clipped at (D - S)^2 / V > 16

    Parameters
    ----------
    data : ndarray
        2-D spectral image [ADU].
    sky_pixels : ndarray of bool
        Boolean vector over the spatial axis, True for the pixels that contain only sky.
    gain_e_per_adu : float
        Gain Q [electrons / ADU], positive.
    read_noise_e : float
        Read noise [electrons rms], not negative.
    order : int
        Degree of the polynomial in x (default 1).
    clip_sigma2 : float
        Squared normalised residual above which a sky pixel is rejected (default 16, the paper's value).
    spatial_axis : int
        Axis of ``data`` along which the spatial profile varies (0 or 1).

    Returns
    -------
    ndarray
        Sky image [ADU], same shape as ``data``.

    References
    ----------
    [Horne1986] Table I step 3, Sect. II.A, II.D.

    Validity
    --------
    The sky must be well described by a low-order polynomial across the slit; the object must not overlap the sky
    pixels. The fit noise is not propagated into the extraction variance (as in the paper): with a sky fitted
    from 10 pixels and a 13-pixel aperture the extraction scatter is 1.24 to 1.27 times the estimated standard
    error (measured, see tests). Measured for a flat sky of 100 e per pixel on 10 sky pixels (order 0, 20 images
    of 300 columns): the bias is -0.02 of the expected standard error sqrt(V / n) and the scatter is 0.994
    times it. With variances taken from the noisy pixel values instead (the paper's Step 2 weights) the bias
    was -0.28 standard errors (about -1/Q ADU), which is why the later passes use model variances.

    Verification
    ------------
    tests/unit/test_spectroscopy.py: flat and tilted sky recovered, outlier rejection.
    """
    d_img = _to_spatial_first(data, spatial_axis)
    v0, q = _check_noise(gain_e_per_adu, read_noise_e)
    sel = np.asarray(sky_pixels, dtype=bool)
    if sel.shape != (d_img.shape[0],):
        raise ValueError("sky_pixels must be a boolean vector over the spatial axis")
    if order < 0 or np.count_nonzero(sel) <= order + 1:
        raise ValueError("not enough sky pixels for the polynomial order")
    n_x, n_l = d_img.shape
    t_all = np.linspace(-1.0, 1.0, n_x) if n_x > 1 else np.zeros(1)
    variance = v0 + np.abs(d_img) / q
    sky_img = np.empty_like(d_img)
    t_sky = t_all[sel]
    for j in range(n_l):
        y = d_img[sel, j]
        pixel_variance = variance[sel, j]
        coeffs = np.zeros(order + 1)
        for _ in range(3):
            w = 1.0 / pixel_variance
            keep = np.ones(y.size, dtype=bool)
            for _ in range(10):
                coeffs = np.polynomial.chebyshev.chebfit(t_sky[keep], y[keep], order, w=np.sqrt(w[keep]))
                resid2 = w * (y - np.polynomial.chebyshev.chebval(t_sky, coeffs)) ** 2
                new_keep = resid2 <= clip_sigma2
                if np.count_nonzero(new_keep) <= order + 1 or np.array_equal(new_keep, keep):
                    break
                keep = new_keep
            # Second and third pass: variances from the fitted sky, not from the noisy pixel values.
            pixel_variance = v0 + np.maximum(np.polynomial.chebyshev.chebval(t_sky, coeffs), 0.0) / q + 1e-12
        sky_img[:, j] = np.polynomial.chebyshev.chebval(t_all, coeffs)
    return sky_img if spatial_axis == 0 else sky_img.T


@source(
    "Horne1986",
    where="Table I steps 4-9; Eq. 8 and Eq. 9 (optimal spectrum and variance); Eq. 13 (revised variance); "
    "Eq. 14 (initial profile); Sect. II.D (profile fit); Sect. II.E (cosmic-ray rejection, threshold 25); "
    "Sect. II.F (iteration)",
    status="read",
    note=(
        "Profile (steps 5-6): P_raw = (D - S)/f_std (Eq. 14) is fitted per spatial row with a weighted Chebyshev "
        "polynomial of degree `profile_order` in wavelength, weights f_std^2/V (Table I: variance V/f^2), clipped "
        "at squared residual 16; negatives set to 0 and each column renormalised to unit sum over the aperture. "
        "Cosmic rays (steps 7-8): the pixel with the largest (D - S - fP)^2/V in a column is rejected if it exceeds "
        "`cosmic_ray_sigma2` (25), one pixel per column per pass, then f and V (Eq. 13) are recomputed, until no "
        "pixel is rejected; as in the paper's two-stage scheme the profile is fixed during that loop. Following the "
        "paper's remark, negative outliers are rejected too unless `reject_negative` is False. The Chebyshev basis, "
        "the number of profile passes, and the 50-pass cap are this project's choices. The variance uses Eq. 9 with "
        "the mask, var f = 1 / sum_x M P^2/V (default); the Table I form sum_x M P / sum_x M P^2/V is available "
        "with variance_convention='table1'. They are equal when no pixel is rejected."
    ),
)
def optimal_extract(
    data: ArrayLike,
    *,
    gain_e_per_adu: float,
    read_noise_e: float,
    sky: ArrayLike | None = None,
    aperture: tuple[int, int] | None = None,
    profile: ArrayLike | None = None,
    profile_order: int = 3,
    profile_passes: int = 3,
    profile_clip_sigma2: float = 16.0,
    cosmic_ray_sigma2: float = 25.0,
    reject_negative: bool = True,
    variance_convention: str = "eq9",
    spatial_axis: int = 0,
) -> OptimalExtraction:
    """Extract a spectrum with Horne's optimal (profile-weighted) algorithm, rejecting cosmic rays.

    f = sum_x M P (D - S) / V / sum_x M P^2 / V
    var f = 1 / sum_x M P^2 / V     (Eq. 9; Table I writes sum_x M P / sum_x M P^2/V)
    V = V_0 + |f P + S| / Q,   M = 0 where (D - S - f P)^2 > sigma_clip^2 V

    Parameters
    ----------
    data : ndarray
        2-D spectral image [ADU], bias-subtracted and flat-fielded; includes the sky.
    gain_e_per_adu : float
        Gain Q [electrons / ADU], positive.
    read_noise_e : float
        Read noise [electrons rms], not negative.
    sky : float or ndarray, optional
        Sky image or value to subtract [ADU] (for example from ``fit_sky_background``). Default 0.
    aperture : tuple of int, optional
        Spatial pixel range (x1, x2), x1 <= x < x2, over which the profile is defined. Default: whole axis. Use a
        range about +-3 profile sigma wide; the paper says wide limits are safe for optimal extraction.
    profile : ndarray, optional
        Known normalised profile, shape of ``data``. If omitted it is estimated from the data (Sect. II.D).
    profile_order : int
        Degree of the wavelength polynomial for each profile row (default 3).
    profile_passes : int
        Number of passes of profile fit and variance update (default 3).
    profile_clip_sigma2 : float
        Squared-residual clip threshold of the profile fit (default 16).
    cosmic_ray_sigma2 : float
        Squared-residual threshold for rejecting a pixel (default 25).
    reject_negative : bool
        Also reject pixels far below the model (default True).
    variance_convention : {"eq9", "table1"}
        Variance formula, see the note above.
    spatial_axis : int
        Axis of ``data`` along which the spatial profile varies (0 or 1).

    Returns
    -------
    OptimalExtraction
        Optimal flux and variance, the standard flux, profile, masks and sky.

    References
    ----------
    [Horne1986] Table I, Eqs. 8, 9, 13, 14.

    Validity
    --------
    Intended for a point source whose spatial profile varies slowly with wavelength (the paper: unresolved point
    sources only). The profile fit needs a wavelength range much longer than the polynomial degree. Measured
    on synthetic images (31 spatial pixels, Gaussian profile of sigma 2 pixels, 300 columns, 25 images per case,
    gain 2 e/ADU, read noise 5 e, sky 100 e per pixel, known sky; tests/unit/test_spectroscopy.py):
    - aperture of 13 pixels (+-3.25 sigma), 300 e per column: the mean of (flux - truth) / sqrt(variance) is
      -0.02 with the true profile and +0.002 with a cubic profile estimated from the data; its standard
      deviation is 1.009 and 1.012, so the Eq. 9 variance is consistent with the scatter. Tilted trace
      (+-3 pixels over the spectrum, 19-pixel aperture): bias +0.06, standard deviation 1.00.
    - aperture of 25 pixels (+-6 sigma): the positivity step (Table I, P = MAX[P, 0]) biases the cubic-profile
      flux by +0.18 standard errors (+0.08 for a constant profile, -0.01 with the true profile). Use apertures
      of about +-3 sigma, or pass the profile.
    - signal-to-noise gain over box extraction (ratio of rms errors, same 13-pixel aperture): known profile 1.37
      at 30 e per column, 1.30 at 100 e, 1.13 at 1000 e (variance ratio 1.70 at 100 e; Eq. 10 gives 1.69 for
      +-3 sigma limits). With the profile estimated from the same data at 30 e per column: 1.20 for a constant
      profile (order 0) and 1.15 for a cubic one, whose noise also makes the Eq. 9 variance underestimate the
      scatter (standard deviation of the normalised error 1.11).
    - cosmic rays (+1500 ADU in one pixel of 5 percent of the columns): 341 of 341 hits rejected, no other pixel
      rejected, standard deviation of the normalised error 1.015; the box sum has 17.6 times the rms error.
    - variance_convention 'table1' is smaller than 'eq9' by sum(M P): 0.8 in a column with a rejected peak pixel.

    Verification
    ------------
    tests/unit/test_spectroscopy.py: unbiased flux, variance versus scatter, S/N gain over box extraction,
    cosmic-ray rejection, tilted trace, input checks.
    """
    if variance_convention not in ("eq9", "table1"):
        raise ValueError("variance_convention must be 'eq9' or 'table1'")
    d_img = _to_spatial_first(data, spatial_axis)
    v0, q = _check_noise(gain_e_per_adu, read_noise_e)
    n_x, n_l = d_img.shape
    ap = _aperture_mask(n_x, aperture)
    s_img = _sky_image(sky, d_img.shape, spatial_axis)
    d = d_img - s_img
    ap_img = np.broadcast_to(ap[:, None], d.shape)

    # Step 2 and Step 4: initial variance and the standard spectrum.
    variance = v0 + np.abs(d_img) / q
    f_std = np.sum(np.where(ap_img, d, 0.0), axis=0)
    t_lambda = np.linspace(-1.0, 1.0, n_l) if n_l > 1 else np.zeros(1)
    cols = np.arange(n_l)

    p_fixed: Array | None = None
    if profile is not None:
        p_in = _to_spatial_first(profile, spatial_axis)
        if p_in.shape != d.shape:
            raise ValueError("profile must have the shape of data")
        if np.any(p_in < 0):
            raise ValueError("the profile must not be negative")
        p_fixed = np.where(ap_img, p_in, 0.0)
        norm = np.sum(p_fixed, axis=0)
        if np.any(norm <= 0):
            raise ValueError("the profile has no weight inside the aperture in some column")
        p_fixed = p_fixed / norm

    # Steps 5-9: profile (when not given), then the cosmic-ray loop of steps 6-8, iterated (step 9). The first
    # profile uses the standard spectrum; later passes use the cleaned optimal spectrum and leave the pixels
    # rejected in the previous pass out of the profile fit.
    mask = ap_img.copy()
    f = f_std.copy()
    p = np.zeros_like(d) if p_fixed is None else p_fixed
    iterations = 0
    for _ in range(1 if p_fixed is not None else max(1, profile_passes)):
        if p_fixed is None:
            safe_f = np.where(f == 0.0, np.inf, f)
            p_raw = d / safe_f
            weight_all = np.where(mask, (safe_f**2) / variance, 0.0)
            p = np.zeros_like(d)
            for ix in np.flatnonzero(ap):
                p[ix] = _clipped_fit(t_lambda, p_raw[ix], weight_all[ix], profile_order, profile_clip_sigma2)
            p = np.where(ap_img, np.maximum(p, 0.0), 0.0)
            norm = np.sum(p, axis=0)
            p = np.where(norm > 0, p / np.where(norm > 0, norm, 1.0), 0.0)
            variance = v0 + np.abs(f * p + s_img) / q
        mask = ap_img.copy()
        f = f_std.copy()
        for _ in range(50):
            iterations += 1
            denom = np.sum(np.where(mask, p**2 / variance, 0.0), axis=0)
            numer = np.sum(np.where(mask, p * d / variance, 0.0), axis=0)
            f_previous = f
            f = np.where(denom > 0, numer / np.where(denom > 0, denom, 1.0), 0.0)
            variance = v0 + np.abs(f * p + s_img) / q
            resid = d - f * p
            r2 = np.where(mask, resid**2 / variance, 0.0)
            if not reject_negative:
                r2 = np.where(resid > 0, r2, 0.0)
            worst = np.argmax(r2, axis=0)
            reject = r2[worst, cols] > cosmic_ray_sigma2
            if np.any(reject):
                mask[worst[reject], cols[reject]] = False
            elif np.max(np.abs(f - f_previous)) <= 1e-9 * (np.max(np.abs(f)) + 1e-300):
                break
    denom = np.sum(np.where(mask, p**2 / variance, 0.0), axis=0)
    numer = np.sum(np.where(mask, p * d / variance, 0.0), axis=0)
    good = denom > 0
    flux = np.where(good, numer / np.where(good, denom, 1.0), np.nan)
    if variance_convention == "eq9":
        var_flux = np.where(good, 1.0 / np.where(good, denom, 1.0), np.inf)
    else:
        var_flux = np.where(good, np.sum(np.where(mask, p, 0.0), axis=0) / np.where(good, denom, 1.0), np.inf)
    rejected = ap_img & ~mask

    def out(a: Array) -> Array:
        return a if spatial_axis == 0 else a.T

    return OptimalExtraction(
        flux=flux,
        variance=var_flux,
        standard_flux=f_std,
        profile=out(p),
        mask=out(mask),
        rejected=out(rejected),
        sky=out(s_img),
        iterations=iterations,
    )
