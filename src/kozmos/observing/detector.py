"""CCD/CMOS detector signal model, synthetic frame generator and signal-to-noise ratio.

The CCD equation was read from the ADS scan of Howell's paper [Howell1989, Eq. 1].
The generator exists to test reduction algorithms on data where the truth is
known: the true scene underlying the generated frame is returned as well.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Detector:
    """Fixed properties of a detector.

    Attributes
    ----------
    gain_e_adu : float
        Gain [e⁻/ADU].
    read_noise_e : float
        Read noise [e⁻, RMS].
    dark_current_e_s : float
        Dark current [e⁻/s/pixel].
    bias_adu : float
        Bias level [ADU].
    """

    gain_e_adu: float = 1.5
    read_noise_e: float = 5.0
    dark_current_e_s: float = 0.05
    bias_adu: float = 1000.0


@source(
    "CraigChambers2026",
    where="notebook 01-05 'Calibration overview': raw image equation",
    status="read",
    note="raw = bias + noise + dark current + flat × (sky + stars). Modeling the noise as Poisson plus Gaussian "
    "read noise and dividing by the gain is this project's model; this detail was not read in the source",
)
def simulate_raw_frame(
    scene_e_s: ArrayLike,
    exposure_s: float,
    detector: Detector,
    rng: np.random.Generator,
    *,
    flat_field: ArrayLike | float = 1.0,
) -> Array:
    """Generate a noisy raw detector frame from a given scene.

    raw = Poisson[(scene · flat_field + dark_current) · t] / gain + bias + read noise

    Parameters
    ----------
    scene_e_s : ndarray (Y, X)
        Flux reaching the detector [e⁻/s/pixel]: sum of sky and sources. Zero is
        given for bias and dark frames.
    exposure_s : float
        Exposure time [s]; 0 for a bias frame.
    detector : Detector
    rng : numpy.random.Generator
    flat_field : ndarray (Y, X) or float
        Pixel-to-pixel relative sensitivity (mean around 1).

    Returns
    -------
    ndarray (Y, X)
        Raw frame [ADU]. No digitization (rounding to integers) is applied.

    References
    ----------
    [CraigChambers2026] notebook 01-05.

    Validity
    --------
    Linear detector: no saturation, deviation from linearity, charge transfer
    inefficiency, hot pixels or cosmic rays. Dark current is not affected by the flat field.

    Verification
    ------------
    tests/unit/test_reduction.py: recovery of gain and read noise from generated frames.
    """
    scene = np.asarray(scene_e_s, dtype=float)
    expected_e = (scene * flat_field + detector.dark_current_e_s) * exposure_s
    electrons = rng.poisson(np.maximum(expected_e, 0.0)).astype(float)
    read_noise = rng.normal(0.0, detector.read_noise_e, scene.shape)
    return (electrons + read_noise) / detector.gain_e_adu + detector.bias_adu


@source(
    "Photutils2026",
    where="API: photutils.psf.GaussianPRF, equation for the integral over a pixel",
    status="read",
    note="f(x, y) = F/4 [erf((x−x0+0.5)/(√2 σx)) − erf((x−x0−0.5)/(√2 σx))] × (same for y); σx = σy in the "
    "circular case",
)
def star_field(shape: tuple[int, int], x: ArrayLike, y: ArrayLike, flux_e_s: ArrayLike, fwhm_pix: float) -> Array:
    """Generate a scene of point sources with a circular Gaussian PSF.

    Each pixel holds the integral of the profile over the pixel area, not the
    value at the pixel center; the total flux is therefore conserved even for a
    narrow PSF.

    Parameters
    ----------
    shape : (Y, X)
    x, y : ndarray (N,)
        Source centers [pixel]; a pixel center sits at an integer coordinate.
    flux_e_s : ndarray (N,)
        Total flux of the sources [e⁻/s].
    fwhm_pix : float

    Returns
    -------
    ndarray (Y, X) [e⁻/s/pixel]

    References
    ----------
    [Photutils2026] GaussianPRF.

    Validity
    --------
    The wings of a real PSF are broader than a Gaussian (Moffat); this model is
    meant only for testing algorithms.

    Verification
    ------------
    tests/unit/test_reduction.py: conservation of total flux for a source far from the edge.
    """
    from scipy.special import erf

    ny, nx = shape
    sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    edges_x = np.arange(nx + 1) - 0.5
    edges_y = np.arange(ny + 1) - 0.5
    scene = np.zeros(shape)
    for x0, y0, f in zip(np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(flux_e_s), strict=True):
        px = 0.5 * np.diff(erf((edges_x - x0) / (np.sqrt(2.0) * sigma)))
        py = 0.5 * np.diff(erf((edges_y - y0) / (np.sqrt(2.0) * sigma)))
        scene += f * np.outer(py, px)
    return scene


@source(
    "Howell1989",
    where="Eq. 1 ('CCD equation')",
    status="read",
    note="S/N = N* / sqrt(N* + n_pix (N_s + N_d + N_r²)). The (1 + n_pix/n_B) factor is not in the source and was "
    "derived in this project: the variance of a background estimated as the mean of n_B pixels is σ²/n_B, and "
    "subtracting it from n_pix pixels adds n_pix² σ²/n_B. The same term is attributed to Merline and Howell "
    "(1995); that paper was not read",
)
def signal_to_noise(
    source_e: ArrayLike,
    sky_e_pix: ArrayLike,
    dark_e_pix: ArrayLike,
    read_noise_e: float,
    n_pixels: float,
    n_background_pixels: float = np.inf,
) -> Array:
    """Compute the signal-to-noise ratio for aperture photometry with the CCD equation.

    S/N = N* / sqrt( N* + n_pix (1 + n_pix/n_B) (N_S + N_D + N_R²) )

    Parameters
    ----------
    source_e : float or ndarray
        Source electrons N* in the aperture.
    sky_e_pix, dark_e_pix : float or ndarray
        Sky and dark-current electrons per pixel.
    read_noise_e : float
        Read noise [e⁻ RMS].
    n_pixels : float
        Number of pixels n_pix in the aperture.
    n_background_pixels : float
        Number of pixels n_B used in the background estimate; the term becomes 1 if infinite.

    References
    ----------
    [Howell1989] Eq. 1.

    Validity
    --------
    Independent sum of Poisson and Gaussian noise; correlated noise
    (resampling, flat-field error) is not included.

    Verification
    ------------
    tests/unit/test_reduction.py: sqrt(N*) in the source-dominated limit; scatter measured on synthetic frames;
    Monte Carlo test of the background-estimate term.
    """
    n = np.asarray(source_e, dtype=float)
    background_factor = n_pixels * (1.0 + n_pixels / n_background_pixels)
    noise_var = n + background_factor * (
        np.asarray(sky_e_pix, dtype=float) + np.asarray(dark_e_pix, dtype=float) + read_noise_e**2
    )
    return n / np.sqrt(noise_var)
