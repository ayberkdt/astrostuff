"""Radial velocity by Fourier cross-correlation on a logarithmic wavelength grid.

The method is that of Tonry & Davis [TonryDavis1979]; equations and the data-reduction recipe (Sect. IV) were read
on 25 November 2026 from the ADS scan of the paper (page images of AJ 84, 1511; the scan has no text layer).

Implemented: log-lambda rebinning (Eq. 1), continuum subtraction with a polynomial, cosine end masking and a
Fourier bandpass filter (Sect. IV), the normalised cross-correlation function (Eqs. 3-5), the parabolic fit to
the correlation peak (Sect. III.c), the antisymmetric-noise estimate and the r statistic (Eqs. 20 and 23), the
velocity from the peak position (Eq. 34) and the shift error (Eq. 24, in the form of Eq. 35).

Not implemented: the velocity-dispersion analysis (Eqs. 25-28, 37-38), the confidence level of a peak choice
(Eqs. 29-33), and the removal of sky lines and spikes by interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal.windows import tukey

from kozmos._sources import source
from kozmos.core.constants import SPEED_OF_LIGHT_KM_S
from kozmos.cosmology.distances import doppler_redshift, doppler_velocity_km_s

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LogRebinnedSpectrum:
    """A spectrum sampled linearly in ln(lambda).

    Attributes
    ----------
    wavelength_angstrom : ndarray
        Wavelength of each bin [angstrom], ln(lambda_n) = ln(lambda_0) + n / A.
    flux : ndarray
        Flux in each bin (input flux unit).
    bins_per_ln_lambda : float
        The constant A of Eq. 1, n = A ln(lambda) + B [bins per unit ln(lambda)].
    """

    wavelength_angstrom: Array
    flux: Array
    bins_per_ln_lambda: float


@dataclass(frozen=True, slots=True)
class CrossCorrelation:
    """Normalised cross-correlation function c(n) on the periodic grid.

    Attributes
    ----------
    lag_bins : ndarray
        Lags n [bins], from -N/2 to N/2 - 1.
    correlation : ndarray
        c(n) of Eq. 4 (1 at n = 0 for identical spectra).
    """

    lag_bins: Array
    correlation: Array


@dataclass(frozen=True, slots=True)
class CorrelationPeak:
    """Parabola fitted to the correlation peak.

    Attributes
    ----------
    centre_bins : float
        Position delta of the peak [bins].
    height : float
        Height h of the fitted parabola at its vertex (dimensionless).
    fwhm_bins : float
        Full width at half maximum w of the parabola [bins].
    """

    centre_bins: float
    height: float
    fwhm_bins: float


@dataclass(frozen=True, slots=True)
class RadialVelocityResult:
    """Cross-correlation radial velocity.

    Attributes
    ----------
    velocity_km_s : float
        Radial velocity of the spectrum relative to the template, positive for recession [km/s].
    error_km_s : float
        Estimated one-sigma error [km/s] (first-order conversion of the Eq. 24 shift error).
    shift_bins : float
        Peak position delta [bins].
    shift_error_bins : float
        Estimated error of delta [bins].
    r : float
        The r statistic h / (sqrt(2) sigma_a), dimensionless.
    height : float
        Peak height h.
    fwhm_bins : float
        Peak full width at half maximum [bins].
    highest_wavenumber : float
        The B used in the error estimate [cycles per spectrum].
    bins_per_ln_lambda : float
        The constant A of Eq. 1.
    """

    velocity_km_s: float
    error_km_s: float
    shift_bins: float
    shift_error_bins: float
    r: float
    height: float
    fwhm_bins: float
    highest_wavenumber: float
    bins_per_ln_lambda: float


@source(
    "TonryDavis1979",
    where="Eq. 1 (n = A ln(lambda) + B) and Sect. IV (a section mapped to 2048 bins linear in ln lambda)",
    status="read",
    note=(
        "The grid has N bins with the end bins at the limits: ln(lambda_n) = ln(lambda_min) + n / A, "
        "A = (N - 1) / ln(lambda_max / lambda_min). Linear interpolation of the flux (the paper does not state "
        "the interpolation method) and the end-point convention are this project's choices. The paper's example, "
        "4550-6200 angstrom in 2048 bins, gives about 45 km/s per bin, as printed."
    ),
)
def rebin_log_wavelength(
    wavelength_angstrom: ArrayLike,
    flux: ArrayLike,
    *,
    n_bins: int = 2048,
    wavelength_range_angstrom: tuple[float, float] | None = None,
) -> LogRebinnedSpectrum:
    """Resample a spectrum onto a grid uniform in ln(lambda) by linear interpolation.

    ln(lambda_n) = ln(lambda_min) + n / A,  A = (N - 1) / ln(lambda_max / lambda_min),  n = 0 ... N - 1

    Parameters
    ----------
    wavelength_angstrom : ndarray
        Wavelengths, strictly increasing [angstrom].
    flux : ndarray
        Flux at those wavelengths (any unit).
    n_bins : int
        Number of log-lambda bins N (default 2048, the paper's value).
    wavelength_range_angstrom : tuple of float, optional
        (lambda_min, lambda_max) [angstrom]; default: the full wavelength range of the input.

    Returns
    -------
    LogRebinnedSpectrum
        Grid, interpolated flux and the constant A.

    References
    ----------
    [TonryDavis1979] Eq. 1, Sect. IV.

    Validity
    --------
    Linear interpolation smooths structure narrower than about two input pixels and does not conserve flux; the
    input should be sampled at least as finely as the log grid (the output pixel in velocity is c / A). The
    range must lie inside the input wavelengths (ValueError otherwise).

    Verification
    ------------
    tests/unit/test_spectroscopy.py: uniform step in ln(lambda), the 45 km/s per bin example, linear function
    reproduced exactly.
    """
    lam = np.asarray(wavelength_angstrom, dtype=float)
    f = np.asarray(flux, dtype=float)
    if lam.ndim != 1 or f.shape != lam.shape or lam.size < 4:
        raise ValueError("wavelength_angstrom and flux must be 1-D arrays of the same length (at least 4)")
    if np.any(np.diff(lam) <= 0) or lam[0] <= 0:
        raise ValueError("wavelength_angstrom must be positive and strictly increasing")
    if n_bins < 8:
        raise ValueError("n_bins must be at least 8")
    lo, hi = (lam[0], lam[-1]) if wavelength_range_angstrom is None else wavelength_range_angstrom
    if not (lam[0] <= lo < hi <= lam[-1]):
        raise ValueError("wavelength_range_angstrom must lie inside the input wavelengths and have lo < hi")
    a = (n_bins - 1) / np.log(hi / lo)
    grid = lo * np.exp(np.arange(n_bins) / a)
    grid[-1] = min(grid[-1], hi)
    return LogRebinnedSpectrum(wavelength_angstrom=grid, flux=np.interp(grid, lam, f), bins_per_ln_lambda=float(a))


def _bandpass(n: int, k1: float, k2: float, k3: float, k4: float) -> Array:
    """Real filter on rfft wavenumbers: 0 below k1, cosine rise to 1 at k2, 1 to k3, cosine fall to 0 at k4."""
    k = np.arange(n // 2 + 1, dtype=float)
    filt = np.zeros_like(k)
    rise = (k >= k1) & (k < k2)
    filt[rise] = 0.5 * (1.0 - np.cos(np.pi * (k[rise] - k1) / (k2 - k1)))
    filt[(k >= k2) & (k <= k3)] = 1.0
    fall = (k > k3) & (k < k4)
    filt[fall] = 0.5 * (1.0 + np.cos(np.pi * (k[fall] - k3) / (k4 - k3)))
    return filt


@source(
    "TonryDavis1979",
    where="Sect. IV (spectrum preparation: fourth-order least-squares polynomial subtracted, 5 percent cosine-bell "
    "end masking, bandpass filter rising from wavenumber 10, 1 at 20, falling from 100, 0 at 200 for N = 2048); "
    "Sect. III.a (spectra continuum subtracted, endmasked and filtered)",
    status="read",
    note=(
        "The polynomial is fitted in a Legendre basis. The cosine bell is a Tukey window with 5 percent tapers on "
        "each end. The paper gives only the four corner wavenumbers of the filter; the raised-cosine ramps between "
        "them are this project's choice. The corner wavenumbers are in cycles per spectrum for N = 2048 and are "
        "not rescaled to other N."
    ),
)
def prepare_for_correlation(
    flux: ArrayLike,
    *,
    continuum_order: int = 4,
    end_fraction: float = 0.05,
    bandpass_wavenumbers: tuple[float, float, float, float] = (10.0, 20.0, 100.0, 200.0),
) -> Array:
    """Subtract the continuum, mask the ends and bandpass-filter a log-lambda spectrum.

    g = IFFT[ B(k) FFT[ w(n) (F(n) - poly_4(n)) ] ],  w = cosine bell over 5 percent at each end

    Parameters
    ----------
    flux : ndarray
        Flux on a grid uniform in ln(lambda) (any unit).
    continuum_order : int
        Degree of the least-squares polynomial subtracted (default 4).
    end_fraction : float
        Fraction of the spectrum tapered at each end by the cosine bell (default 0.05).
    bandpass_wavenumbers : tuple of float
        Corners (k1, k2, k3, k4) of the filter [cycles per spectrum]: 0 below k1, rising to 1 at k2, falling
        from k3 to 0 at k4. Default (10, 20, 100, 200).

    Returns
    -------
    ndarray
        Prepared spectrum, same length, same flux unit, approximately zero mean.

    References
    ----------
    [TonryDavis1979] Sect. IV.

    Validity
    --------
    Designed for N = 2048 bins of about 45 km/s; for other N or other line widths the corner wavenumbers should
    be chosen so that the pass band holds the Fourier power of the lines. Not tested against other corner
    wavenumbers; the pass-band gain for a sine of unit amplitude at k = 3, 60 and 400 (N = 2048, measured in
    the tests) is 2e-4, 1.000 and 6e-7.

    Verification
    ------------
    tests/unit/test_spectroscopy.py: polynomial removed exactly, zero mean, ends tapered, low and high
    wavenumbers suppressed, pass band unity.
    """
    f = np.asarray(flux, dtype=float)
    n = f.size
    if f.ndim != 1 or n < 16:
        raise ValueError("flux must be a 1-D array of at least 16 samples")
    if not 0.0 <= end_fraction < 0.5:
        raise ValueError("end_fraction must be in [0, 0.5)")
    k1, k2, k3, k4 = bandpass_wavenumbers
    if not 0.0 <= k1 < k2 <= k3 < k4:
        raise ValueError("bandpass_wavenumbers must satisfy 0 <= k1 < k2 <= k3 < k4")
    t = np.linspace(-1.0, 1.0, n)
    coeffs = np.polynomial.legendre.legfit(t, f, continuum_order)
    resid = f - np.polynomial.legendre.legval(t, coeffs)
    window = tukey(n, alpha=2.0 * end_fraction) if end_fraction > 0 else np.ones(n)
    spectrum = np.fft.rfft(resid * window)
    return np.fft.irfft(spectrum * _bandpass(n, k1, k2, k3, k4), n=n)


@source(
    "TonryDavis1979",
    where="Eqs. 3, 4 and 5 (rms sigma_g, sigma_t; normalised correlation c(n); Fourier form C(k))",
    status="read",
    note=(
        "c(n) = (1/(N sigma_g sigma_t)) sum_m g(m) t(m - n) with periodic indices, evaluated through the FFT as "
        "C(k) = G(k) T*(k) / (N sigma_g sigma_t) (Eq. 5). The spectra are used as given (preparation is a separate "
        "step). Lags are returned fftshifted, from -N/2 to N/2 - 1 (this project's layout)."
    ),
)
def cross_correlate(spectrum: ArrayLike, template: ArrayLike) -> CrossCorrelation:
    """Compute the normalised cross-correlation function of a spectrum with a template.

    c(n) = (1 / (N sigma_g sigma_t)) sum_m g(m) t(m - n),  sigma^2 = (1/N) sum g^2

    A spectrum equal to the template shifted by d bins, g(n) = t(n - d), has its peak at n = d.

    Parameters
    ----------
    spectrum : ndarray
        Prepared spectrum g on the log-lambda grid.
    template : ndarray
        Prepared template t on the same grid, same length.

    Returns
    -------
    CrossCorrelation
        Lags [bins] and c(n); c(0) = 1 for identical inputs.

    References
    ----------
    [TonryDavis1979] Eqs. 3, 4, 5.

    Validity
    --------
    The spectra are treated as periodic with period N, so they must be end-masked (``prepare_for_correlation``)
    to avoid wrap-around steps. A spectrum with zero rms raises ValueError.

    Verification
    ------------
    tests/unit/test_spectroscopy.py: c(0) = 1 for identical inputs, direct-sum check, peak at the injected shift,
    sign convention.
    tests/oracle/test_spectroscopy_oracle.py: scipy.signal.correlate.
    """
    g = np.asarray(spectrum, dtype=float)
    t = np.asarray(template, dtype=float)
    if g.ndim != 1 or g.shape != t.shape:
        raise ValueError("spectrum and template must be 1-D arrays of the same length")
    n = g.size
    sg = float(np.sqrt(np.mean(g**2)))
    st = float(np.sqrt(np.mean(t**2)))
    if sg == 0.0 or st == 0.0:
        raise ValueError("spectrum and template must have non-zero rms")
    spec = np.fft.rfft(g) * np.conj(np.fft.rfft(t)) / (n * sg * st)
    c = np.fft.irfft(spec, n=n)
    lags = np.arange(n) - (n // 2)
    return CrossCorrelation(lag_bins=lags.astype(float), correlation=np.fft.fftshift(c))


@source(
    "TonryDavis1979",
    where="Sect. III.b (quadratic least-squares fit to the correlation peak); Sect. III.c.ii (polynomial fit to 11 "
    "points around the peak); Sect. IV (parabola fitted to the largest peak in -1000 < cZ < 20000 km/s); the "
    "parabola h0 (1 - 2 (n - n0)^2 / w0^2) in the text after Eq. 20 gives the FWHM w",
    status="read",
    note=(
        "Quadratic least-squares fit to n_points samples centred on the largest value inside the search window; "
        "centre = vertex, h = value at the vertex, w = sqrt(2 h / a) for c = h - a (n - delta)^2 (half maximum at "
        "|n - delta| = w / 2). Restricting the maximum to a lag window is the paper's allowed redshift range "
        "(-1000 < cz < 20000 km/s there); the 11-point default is the paper's, the use of unweighted least squares "
        "is this project's choice."
    ),
)
def fit_correlation_peak(
    correlation: CrossCorrelation,
    *,
    lag_window_bins: tuple[float, float],
    n_points: int = 11,
) -> CorrelationPeak:
    """Fit a parabola to the largest peak of the correlation function inside a lag window.

    c(n) = h - a (n - delta)^2,  FWHM w = sqrt(2 h / a)

    Parameters
    ----------
    correlation : CrossCorrelation
        Output of ``cross_correlate``.
    lag_window_bins : tuple of float
        (lowest, highest) lag [bins] in which the peak is searched.
    n_points : int
        Odd number of points fitted around the maximum (default 11).

    Returns
    -------
    CorrelationPeak
        Position delta [bins], height h and FWHM w [bins].

    References
    ----------
    [TonryDavis1979] Sect. III.b, III.c.ii, IV.

    Validity
    --------
    The window must not be so wide that a noise peak exceeds the true peak (the paper's Eqs. 29-33 give the
    probability; not implemented). ValueError if the fitted parabola is not concave or the peak is too close to
    the window edge for n_points samples. For a Gaussian-shaped peak at fractional positions the largest vertex
    error (measured, see tests) is 0.074 bin for FWHM 9 bins and 11 points (the paper's choice), 0.012 bin with 5
    points, 0.029 bin for FWHM 15 bins and 11 points, 0.14 bin for FWHM 6 bins and 11 points: the 11-point
    parabola is a poor model of a narrow peak, and a smaller ``n_points`` trades that bias for noise.

    Verification
    ------------
    tests/unit/test_spectroscopy.py: exact parabola, Gaussian peak at fractional positions, window handling.
    """
    if n_points < 5 or n_points % 2 == 0:
        raise ValueError("n_points must be an odd number of at least 5")
    lags = correlation.lag_bins
    c = correlation.correlation
    inside = np.flatnonzero((lags >= lag_window_bins[0]) & (lags <= lag_window_bins[1]))
    if inside.size == 0:
        raise ValueError("the lag window contains no samples")
    i0 = int(inside[np.argmax(c[inside])])
    half = n_points // 2
    if i0 - half < 0 or i0 + half >= c.size:
        raise ValueError("the peak is too close to the end of the correlation function")
    x = lags[i0 - half : i0 + half + 1] - lags[i0]
    y = c[i0 - half : i0 + half + 1]
    q2, q1, q0 = np.polyfit(x, y, 2)
    if not q2 < 0:
        raise ValueError("the fitted parabola is not concave: no usable peak")
    vertex = -q1 / (2.0 * q2)
    height = q0 - q1**2 / (4.0 * q2)
    fwhm = float(np.sqrt(2.0 * height / -q2)) if height > 0 else float("nan")
    return CorrelationPeak(centre_bins=float(lags[i0] + vertex), height=float(height), fwhm_bins=fwhm)


@source(
    "TonryDavis1979",
    where="Eq. 20 (antisymmetric rms sigma_a) and Eq. 23 (r = h / (sqrt(2) sigma_a))",
    status="read",
    note=(
        "sigma_a^2 = (1/(2N)) sum_n [c(delta + n) - c(delta - n)]^2 over all n of the periodic grid. c at the "
        "non-integer points delta + n is evaluated exactly from the Fourier series of the periodic c (a Fourier "
        "shift); the paper does not say how c is interpolated, so this is this project's choice."
    ),
)
def correlation_r_statistic(correlation: CrossCorrelation, centre_bins: float, height: float) -> float:
    """Compute the r statistic: peak height over the rms of the antisymmetric part of the correlation.

    sigma_a^2 = (1 / (2N)) sum_n [c(delta + n) - c(delta - n)]^2,   r = h / (sqrt(2) sigma_a)

    Parameters
    ----------
    correlation : CrossCorrelation
        Output of ``cross_correlate``.
    centre_bins : float
        Peak position delta [bins].
    height : float
        Peak height h.

    Returns
    -------
    float
        r (dimensionless); large for a high-quality correlation.

    References
    ----------
    [TonryDavis1979] Eqs. 20, 23.

    Validity
    --------
    Eq. 20 assumes the symmetric and antisymmetric parts of the noise have equal rms. For identical noise-free
    spectra sigma_a is zero and r is infinite (returned as inf). Measured with injected shifts on synthetic
    spectra (see tests): the median r is 413 at S/N 200 per pixel, 49 at S/N 20 and 25 at S/N 10, and the
    shift error estimate built on it follows the scatter as quoted for ``shift_error_bins``.

    Verification
    ------------
    tests/unit/test_spectroscopy.py: symmetric function gives infinite r, closed-form sigma_a for an odd sine
    at an integer centre, symmetric peak at a fractional centre.
    """
    c = correlation.correlation
    n = c.size
    c_unshifted = np.fft.ifftshift(c)
    spec = np.fft.rfft(c_unshifted)
    # c(delta + n): Fourier shift by delta (the grid index of lag 0 is 0 after ifftshift).
    k = np.arange(spec.size)
    shifted = np.fft.irfft(spec * np.exp(2j * np.pi * k * centre_bins / n), n=n)
    mirror = shifted[(-np.arange(n)) % n]
    sigma_a = float(np.sqrt(np.sum((shifted - mirror) ** 2) / (2.0 * n)))
    if sigma_a <= 1e-12 * float(np.sqrt(np.mean(c**2))):
        return float("inf")
    return float(height / (np.sqrt(2.0) * sigma_a))


@source(
    "TonryDavis1979",
    where="p. 1519, text before Eq. 34 (correlation functions have a half-amplitude point in Fourier space of about "
    "wavenumber 50); definition of the highest frequency B in Sect. III.c.i, text after Eq. 21",
    status="read",
    note=(
        "The paper states the number (about 50 for its data) but not an algorithm. This project's estimator: "
        "amplitude |C(k)| of the correlation function, smoothed with a 9-sample boxcar; the wavenumber, above the "
        "maximum of the smoothed amplitude, where it first falls below half that maximum. Falls back to N/2 if it "
        "never does."
    ),
)
def half_amplitude_wavenumber(correlation: CrossCorrelation) -> float:
    """Estimate the highest wavenumber B of the correlation function from its Fourier amplitude.

    B = first k above the amplitude maximum where the smoothed |C(k)| < max / 2

    Parameters
    ----------
    correlation : CrossCorrelation
        Output of ``cross_correlate``.

    Returns
    -------
    float
        B [cycles per spectrum].

    References
    ----------
    [TonryDavis1979] p. 1519 (text before Eq. 34), Sect. III.c.i.

    Validity
    --------
    A data-driven estimate that fluctuates with the noise; the paper's own value for its data is about 50. For
    the synthetic spectra of the tests (N = 2048, 60 lines of FWHM 100 km/s, filter 10-20-100-200, S/N 10 to
    200 per pixel) the median was 87 and the range 27 to 92 (measured over 750 trials).

    Verification
    ------------
    tests/unit/test_spectroscopy.py: Gaussian correlation of known width gives the analytic half-amplitude point.
    """
    c = np.fft.ifftshift(correlation.correlation)
    amp = np.abs(np.fft.rfft(c))
    width = 9
    smooth = np.convolve(amp, np.ones(width) / width, mode="same")
    kmax = int(np.argmax(smooth[1:])) + 1
    below = np.flatnonzero(smooth[kmax:] < 0.5 * smooth[kmax])
    return float(kmax + below[0]) if below.size else float(smooth.size - 1)


@source(
    "TonryDavis1979",
    where="Eq. 24 (mean shift error eps = (1/4) (N / 2B) (1 / (1 + r))) and Eq. 35 (the same with numbers)",
    status="read",
    note=(
        "The paper says the coefficient multiplying 1/(1+r) is adjusted to fit external errors and that the "
        "result agrees with an effective N/8B, which is the form implemented. The `scale` factor (default 1) is "
        "this project's hook for such an adjustment."
    ),
)
def shift_error_bins(n_bins: int, r: ArrayLike, highest_wavenumber: float, scale: float = 1.0) -> Array:
    """Compute the expected error of the correlation-peak position.

    epsilon = scale * (1/4) (N / (2 B)) / (1 + r)

    Parameters
    ----------
    n_bins : int
        Number of bins N of the correlation function.
    r : float or ndarray
        The r statistic, non-negative.
    highest_wavenumber : float
        B [cycles per spectrum], positive.
    scale : float
        Multiplicative adjustment of the coefficient (default 1).

    Returns
    -------
    ndarray
        Expected error [bins].

    References
    ----------
    [TonryDavis1979] Eq. 24, Eq. 35.

    Validity
    --------
    Eq. 24 models the error as a noise peak of the antisymmetric part pulling the peak off; the paper itself
    adjusts the coefficient to external velocities. Measured on synthetic absorption-line spectra against
    injected velocities, with r and B (``half_amplitude_wavenumber``) taken from the data: the ratio of the
    measured velocity scatter to this estimate is 0.79 at S/N 10, 0.84 at 20, 1.2 at 50, 2.0 at 100 and 3.3 at
    200 per pixel (table in ``radial_velocity_cross_correlation``).

    Verification
    ------------
    tests/unit/test_spectroscopy.py: the paper's numbers (N = 2048, B = 50 give 5.12 / (1 + r) bins).
    """
    if n_bins <= 0 or not highest_wavenumber > 0:
        raise ValueError("n_bins and highest_wavenumber must be positive")
    r_arr = np.asarray(r, dtype=float)
    if np.any(r_arr < 0):
        raise ValueError("r must not be negative")
    return scale * 0.25 * (n_bins / (2.0 * highest_wavenumber)) / (1.0 + r_arr)


@source(
    "TonryDavis1979",
    where="Eq. 34 (ln(1 + Z) = delta / A) and Eq. 35 (error of ln(1 + Z) = epsilon / A), p. 1519",
    status="read",
    note=(
        "Z = exp(delta/A) - 1, then v from the special-relativistic Doppler relation through "
        "kozmos.cosmology.distances.doppler_velocity_km_s (this project's conversion; the paper quotes cz). The "
        "velocity error is c epsilon / A, first order in epsilon / A. The pipeline order (rebin, prepare, "
        "correlate, peak fit, r, error) follows Sect. IV; the default velocity window of +-1000 km/s is this "
        "project's choice."
    ),
)
def radial_velocity_cross_correlation(
    wavelength_angstrom: ArrayLike,
    flux: ArrayLike,
    template_wavelength_angstrom: ArrayLike,
    template_flux: ArrayLike,
    *,
    n_bins: int = 2048,
    wavelength_range_angstrom: tuple[float, float] | None = None,
    velocity_window_km_s: tuple[float, float] = (-1000.0, 1000.0),
    peak_points: int = 11,
    error_scale: float = 1.0,
) -> RadialVelocityResult:
    """Measure the radial velocity of a spectrum relative to a rest-frame template by cross-correlation.

    rebin both onto ln(lambda), subtract continuum, mask ends, filter, c = g x t, peak delta, v = c tanh(delta / A)

    Parameters
    ----------
    wavelength_angstrom : ndarray
        Wavelengths of the observed spectrum, increasing [angstrom].
    flux : ndarray
        Observed flux (any unit; continuum is subtracted).
    template_wavelength_angstrom : ndarray
        Wavelengths of the template at zero velocity [angstrom].
    template_flux : ndarray
        Template flux.
    n_bins : int
        Number of log-lambda bins (default 2048).
    wavelength_range_angstrom : tuple of float, optional
        Range used for both spectra [angstrom]; default: the overlap of the two wavelength ranges.
    velocity_window_km_s : tuple of float
        Velocities [km/s] between which the correlation peak is searched.
    peak_points : int
        Odd number of points of the parabolic peak fit (default 11, the paper's value; fewer points reduce the
        systematic vertex error of a narrow peak at the cost of more noise).
    error_scale : float
        Adjustment factor passed to ``shift_error_bins`` (default 1).

    Returns
    -------
    RadialVelocityResult
        Velocity, estimated error, peak position, r, height, width.

    References
    ----------
    [TonryDavis1979] Eqs. 1, 4, 20, 23, 24, 34, 35; Sect. III.c, IV.

    Validity
    --------
    Absorption-line spectra of a single object type against a template of the same instrument resolution; the
    wavelength range should be wider than the velocity window requires. The peak is chosen as the largest in the
    window with no confidence level (Eqs. 29-33 not implemented). Measured on synthetic spectra (2048 bins of
    36.5 km/s, 60 absorption lines of FWHM 100 km/s, 0.25 angstrom pixels, white noise, noise-free template,
    shifts uniform in +-150 km/s, 150 trials per S/N; tests/unit/test_spectroscopy.py):
    S/N per pixel   200    100    50     20     10
    scatter [km/s]  0.85   0.92   1.14   2.08   4.03
    bias [km/s]    -0.16  -0.17  -0.20  -0.39  -0.63
    scatter / mean Eq. 24 error   3.3    2.0    1.2    0.84   0.79
    median r        413    233    122    49     25
    The noise-free error is up to 1.63 km/s (rms 0.95) with the default 11-point peak fit, which sets the
    floor at high S/N; with ``peak_points=5`` it is 0.27 km/s (rms 0.17), the scatter is 0.24 km/s at S/N 200
    and 3.80 km/s at S/N 10. The Eq. 24 estimate follows the scatter within 25 percent only for S/N of about
    20 to 50 and overestimates it by 20 percent at S/N 10 (the paper adjusts its coefficient to external
    data; ``error_scale`` is the hook); it does not include the systematic floor, so it underestimates by
    a factor 2 to 3 at S/N of 100 or more.

    Verification
    ------------
    tests/unit/test_spectroscopy.py: injected velocities recovered at five S/N, sign convention, scatter versus
    estimated error, noise-free exactness.
    """
    lam_o = np.asarray(wavelength_angstrom, dtype=float)
    lam_t = np.asarray(template_wavelength_angstrom, dtype=float)
    if wavelength_range_angstrom is None:
        wavelength_range_angstrom = (max(lam_o[0], lam_t[0]), min(lam_o[-1], lam_t[-1]))
    obs = rebin_log_wavelength(lam_o, flux, n_bins=n_bins, wavelength_range_angstrom=wavelength_range_angstrom)
    tmp = rebin_log_wavelength(lam_t, template_flux, n_bins=n_bins, wavelength_range_angstrom=wavelength_range_angstrom)
    corr = cross_correlate(prepare_for_correlation(obs.flux), prepare_for_correlation(tmp.flux))
    a = obs.bins_per_ln_lambda
    window = tuple(a * np.log(1.0 + doppler_redshift(np.asarray(velocity_window_km_s))))
    peak = fit_correlation_peak(corr, lag_window_bins=(window[0], window[1]), n_points=peak_points)
    r = correlation_r_statistic(corr, peak.centre_bins, peak.height)
    b = half_amplitude_wavenumber(corr)
    eps = float(shift_error_bins(n_bins, r if np.isfinite(r) else 1e12, b, scale=error_scale))
    z = np.exp(peak.centre_bins / a) - 1.0
    velocity = float(doppler_velocity_km_s(z))
    return RadialVelocityResult(
        velocity_km_s=velocity,
        error_km_s=float(SPEED_OF_LIGHT_KM_S * eps / a),
        shift_bins=peak.centre_bins,
        shift_error_bins=eps,
        r=r,
        height=peak.height,
        fwhm_bins=peak.fwhm_bins,
        highest_wavenumber=b,
        bins_per_ln_lambda=a,
    )
