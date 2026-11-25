"""Robust location and scale estimators that resist outliers.

The formulas were read from the docstrings in the ``astropy.stats`` source
code [Astropy2022]. The original source of the biweight estimators is the
paper [Beers1990]; the text of that paper has not been opened.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]

# For a Gaussian distribution σ = MAD / Φ⁻¹(3/4); 1 / 0.6744897501960817.
MAD_TO_SIGMA = 1.482602218505602


@source(
    "Astropy2022",
    where="stats/funcs.py, median_absolute_deviation and mad_std",
    status="read",
    note="MAD = median(|a − median(a)|); σ = MAD · 1.482602218505602 (= 1/Φ⁻¹(0.75))",
)
def mad_scale(x: ArrayLike, axis: int | None = None) -> Array:
    """Estimate the Gaussian-equivalent standard deviation from the median absolute deviation.

    References
    ----------
    [Astropy2022] stats/funcs.py.

    Validity
    --------
    Tolerates outliers in up to half of the samples. For Gaussian data its
    efficiency is about 37% of that of the standard deviation (not tested).

    Verification
    ------------
    tests/oracle/test_robust_oracle.py: exact equality with astropy.stats.mad_std.
    """
    a = np.asarray(x, dtype=float)
    median = np.median(a, axis=axis, keepdims=True)
    return MAD_TO_SIGMA * np.median(np.abs(a - median), axis=axis)


@source(
    "Astropy2022",
    where="stats/biweight.py, biweight_location docstring",
    status="read",
    note="ζ = M + Σ(x−M)(1−u²)² / Σ(1−u²)², u = (x−M)/(c·MAD), |u| < 1; c = 6. The original source [Beers1990] was not read",
)
def biweight_location(x: ArrayLike, c: float = 6.0) -> float:
    """Compute the biweight location estimator.

    References
    ----------
    [Astropy2022] stats/biweight.py.

    Validity
    --------
    If the MAD is zero (more than half of the samples have the same value), the median is returned.

    Verification
    ------------
    tests/oracle/test_robust_oracle.py: relative difference from astropy.stats.biweight_location < 1e-12.
    """
    a = np.asarray(x, dtype=float).ravel()
    m = float(np.median(a))
    mad = float(np.median(np.abs(a - m)))
    if mad == 0.0:
        return m
    u = (a - m) / (c * mad)
    weight = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)
    return m + float(np.sum((a - m) * weight) / np.sum(weight))


@source(
    "Astropy2022",
    where="stats/biweight.py, biweight_scale docstring",
    status="read",
    note="S = √n · sqrt(Σ(x−M)²(1−u²)⁴) / |Σ(1−u²)(1−5u²)|, |u| < 1; c = 9; n is the total number of elements",
)
def biweight_scale(x: ArrayLike, c: float = 9.0) -> float:
    """Compute the biweight scale estimator.

    References
    ----------
    [Astropy2022] stats/biweight.py.

    Validity
    --------
    Returns 0 if the MAD is zero.

    Verification
    ------------
    tests/oracle/test_robust_oracle.py: relative difference from astropy.stats.biweight_scale < 1e-12.
    """
    a = np.asarray(x, dtype=float).ravel()
    m = float(np.median(a))
    mad = float(np.median(np.abs(a - m)))
    if mad == 0.0:
        return 0.0
    u = (a - m) / (c * mad)
    inside = np.abs(u) < 1.0
    d, u2 = (a - m)[inside], u[inside] ** 2
    numerator = np.sqrt(np.sum(d**2 * (1.0 - u2) ** 4))
    denominator = np.abs(np.sum((1.0 - u2) * (1.0 - 5.0 * u2)))
    return float(np.sqrt(a.size) * numerator / denominator)


@source(
    "Astropy2022",
    where="stats/sigma_clipping.py, SigmaClip docstring",
    status="read",
    note="in each iteration center = median, scale = standard deviation; elements with |x − center| > kσ and non-finite elements are rejected",
)
def sigma_clip(x: ArrayLike, sigma: float = 3.0, iterations: int = 5, axis: int | None = None) -> Mask:
    """Apply iterative sigma clipping and return the mask of rejected elements.

    Parameters
    ----------
    x : ndarray
    sigma : float
        Clipping threshold (multiple of the standard deviation).
    iterations : int
        Maximum number of iterations; stops early when no element is rejected.
    axis : int, optional
        Axis along which to clip; if not given, the whole array is a single sample.

    Returns
    -------
    ndarray[bool]
        ``True``: the element was rejected.

    References
    ----------
    [Astropy2022] stats/sigma_clipping.py.

    Validity
    --------
    The scale estimator is the standard deviation; a single very large outlier
    inflates the scale in the first iteration and can hide neighboring outliers.

    Verification
    ------------
    tests/oracle/test_robust_oracle.py: exact equality with the astropy.stats.sigma_clip mask.
    """
    a = np.asarray(x, dtype=float)
    mask = ~np.isfinite(a)
    for _ in range(iterations):
        kept = np.where(mask, np.nan, a)
        center = np.nanmedian(kept, axis=axis, keepdims=True)
        scale = np.nanstd(kept, axis=axis, keepdims=True)
        new_mask = mask | (np.abs(a - center) > sigma * scale)
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    return mask
