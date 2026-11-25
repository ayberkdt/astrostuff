"""Image reduction: frame combination, bias/dark/flat-field correction, photon transfer.

The reduction equation comes from the astropy CCD reduction guide [CraigChambers2026]; the gain
and read-noise formulas were read from the documentation of the IRAF ``findgain`` task
[IRAF2026]. The variance propagation is this project's model and is tested with synthetic
frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.statistics.robust import sigma_clip

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReducedFrame:
    """Reduced image and its per-pixel variance.

    Attributes
    ----------
    data : ndarray (Y, X)
        Flux [e⁻/s/pixel].
    variance : ndarray (Y, X)
        Variance of ``data`` [(e⁻/s)²].
    """

    data: Array
    variance: Array


@source(
    "CraigChambers2026",
    where="notebook 01-05 'Calibration overview': 'Every image has noise'",
    status="read",
    note="calibration frames are combined from many frames to reduce the noise; the details of combination "
    "with sigma clipping (notebook 01-06) were not read",
)
def combine(frames: ArrayLike, *, method: str = "mean", sigma: float | None = 3.0) -> tuple[Array, Array]:
    """Combine frames of the same scene pixel by pixel.

    Parameters
    ----------
    frames : ndarray (K, Y, X)
    method : {"mean", "median"}
    sigma : float, optional
        If given, sigma clipping is applied along the frame axis before combining.

    Returns
    -------
    (combined, used) : the combined frame and the number of frames used per pixel.

    References
    ----------
    [CraigChambers2026] notebook 01-05.

    Validity
    --------
    The frames must be at the same level (same exposure time, same illumination); no scaling
    is applied. For Gaussian noise the variance of the median is about
    π/2 times that of the mean.

    Verification
    ------------
    tests/unit/test_reduction.py: unbiasedness of the clipped mean in frames with added cosmic rays.
    """
    k = np.asarray(frames, dtype=float)
    if k.ndim != 3:
        raise ValueError("frames must have shape (K, Y, X)")
    if method not in ("mean", "median"):
        raise ValueError("method must be 'mean' or 'median'")
    mask = sigma_clip(k, sigma=sigma, axis=0) if sigma is not None else np.zeros(k.shape, dtype=bool)
    unmasked = np.where(mask, np.nan, k)
    combined = np.nanmean(unmasked, axis=0) if method == "mean" else np.nanmedian(unmasked, axis=0)
    return combined, (~mask).sum(axis=0).astype(float)


@source(
    "IRAF2026",
    where="noao/obsutil/src/doc/findgain.hlp, ALGORITHM",
    status="read",
    note="gain = ((mean(flat1)+mean(flat2)) − (mean(zero1)+mean(zero2))) / (σ²(flatdif) − σ²(zerodif)); "
    "readnoise = gain · σ(zerodif) / √2",
)
def gain_and_read_noise(flat1: ArrayLike, flat2: ArrayLike, bias1: ArrayLike, bias2: ArrayLike) -> tuple[float, float]:
    """Measure the gain and read noise from two flat-field frames and two bias frames.

    The difference images remove fixed patterns (flat-field structure, bias structure);
    only random noise remains.

    Returns
    -------
    (gain [e⁻/ADU], read noise [e⁻ RMS])

    References
    ----------
    [IRAF2026] findgain.hlp.

    Validity
    --------
    The two flat frames must be at the same illumination level and in the linear regime.
    The dark-current contribution is neglected (short exposure).

    Verification
    ------------
    tests/unit/test_reduction.py: recovery of the input values on synthetic frames.
    """
    f1, f2 = np.asarray(flat1, dtype=float), np.asarray(flat2, dtype=float)
    b1, b2 = np.asarray(bias1, dtype=float), np.asarray(bias2, dtype=float)
    var_bias = np.var(b1 - b2, ddof=1)
    gain = ((f1.mean() + f2.mean()) - (b1.mean() + b2.mean())) / (np.var(f1 - f2, ddof=1) - var_bias)
    return float(gain), float(gain * np.sqrt(var_bias / 2.0))


@source(
    "CraigChambers2026",
    where="notebook 01-05 'Calibration overview': reduction equation",
    status="read",
    note="stars + noise = (raw − bias − dark current)/flat − sky; dark current = (dark frame − bias)/time. "
    "The variance σ² = (signal_e + R²)/(flat·t)² is this project's model (not read in the source)",
)
def reduce_frame(
    raw_adu: ArrayLike,
    master_bias_adu: ArrayLike,
    master_dark_e_s: ArrayLike | float,
    master_flat: ArrayLike | float,
    *,
    gain_e_adu: float,
    read_noise_e: float,
    exposure_s: float,
) -> ReducedFrame:
    """Convert a raw frame to flux with bias, dark and flat-field correction.

    Order: subtract bias, convert to electrons with the gain, subtract dark current, divide
    by the flat field and the exposure time.

    Parameters
    ----------
    raw_adu : ndarray (Y, X)
    master_bias_adu : ndarray (Y, X) or float
    master_dark_e_s : ndarray (Y, X) or float
        Dark-current map [e⁻/s/pixel].
    master_flat : ndarray (Y, X) or float
        Normalized flat field (mean 1).
    gain_e_adu, read_noise_e, exposure_s : float

    Returns
    -------
    ReducedFrame
        Flux [e⁻/s/pixel] and its variance.

    References
    ----------
    [CraigChambers2026] notebook 01-05.

    Validity
    --------
    The variance is the sum of the Poisson noise of the bias-subtracted signal and the read
    noise; the noise of the master calibration frames themselves is not added (they are assumed
    to be combined from many frames). Because the Poisson term is estimated from the observed
    count, it is biased at low counts.

    Verification
    ------------
    tests/unit/test_reduction.py: residual of the reduced frame against the known scene and reduced χ².
    """
    if exposure_s <= 0:
        raise ValueError("exposure time must be positive")
    signal_e = (np.asarray(raw_adu, dtype=float) - master_bias_adu) * gain_e_adu
    flat = np.asarray(master_flat, dtype=float)
    data = (signal_e - np.asarray(master_dark_e_s, dtype=float) * exposure_s) / (flat * exposure_s)
    variance = (np.maximum(signal_e, 0.0) + read_noise_e**2) / (flat * exposure_s) ** 2
    return ReducedFrame(data, variance)
