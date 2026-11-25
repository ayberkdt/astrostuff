"""Display stretch functions for astronomical images and color composites.

The dynamic range of an astronomical image is much wider than that of a
screen: on a linear scale either the faint structure or the bright core is
visible. The asinh stretch is linear at the faint end and logarithmic at the
bright end. The definitions and the color-preserving composite were read from
the arXiv TeX source (astro-ph/0312483) of the paper by Lupton and
collaborators [Lupton2004].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source

Array = NDArray[np.float64]


@source(
    "Lupton2004",
    where="Section 2, footnote: f(x) = asinh(α Q (x − m)) / Q",
    status="read",
    note="in the limit Q → 0 the stretch is linear, α(x − m); m is the smallest value (black point)",
)
def asinh_stretch(image: ArrayLike, *, black: float, alpha: float, q: float = 8.0) -> Array:
    """Apply the asinh stretch: f(x) = asinh(α Q (x − m)) / Q.

    Parameters
    ----------
    image : ndarray
    black : float
        Value ``m`` that corresponds to black (usually the sky level).
    alpha : float
        Slope α of the linear region; ``1/alpha`` is the scale of the faint structure.
    q : float
        Sharpness of the transition from linear to logarithmic; ``q → 0`` is a linear stretch.

    Returns
    -------
    ndarray
        Stretched values; the lower limit is 0, no upper limit is applied.

    References
    ----------
    [Lupton2004] Section 2, footnote.

    Validity
    --------
    Values below ``black`` are clipped to 0.

    Verification
    ------------
    tests/unit/test_visualization.py: linear limit for a small signal, monotonicity.
    """
    x = np.asarray(image, dtype=float) - black
    if q <= 0:
        return np.clip(alpha * x, 0.0, None)
    return np.clip(np.arcsinh(alpha * q * x) / q, 0.0, None)


@source(
    "Lupton2004",
    where="Section 2, TeX label EqIntensityStretch",
    status="read",
    note="I = (r+g+b)/3; R = r f(I)/I, G = g f(I)/I, B = b f(I)/I; if the largest component exceeds 1, all three are divided by it",
)
def lupton_rgb(
    red: ArrayLike, green: ArrayLike, blue: ArrayLike, *, black: float = 0.0, alpha: float, q: float = 8.0
) -> Array:
    """Build a color-preserving RGB composite from three bands.

    The stretch is applied to the total intensity I = (r + g + b)/3 and the
    three bands are scaled by the same factor f(I)/I; so the color of an
    object does not change with its brightness. If one of the components
    exceeds 1, all three are divided by the largest: the intensity is
    clipped, the color is preserved.

    Parameters
    ----------
    red, green, blue : ndarray (Y, X)
        Long, middle and short wavelength images; on the same grid, sky subtracted.
    black, alpha, q : float
        Same meaning as in ``asinh_stretch``.

    Returns
    -------
    ndarray (Y, X, 3)
        RGB in the range [0, 1].

    References
    ----------
    [Lupton2004] Section 2, EqIntensityStretch.

    Validity
    --------
    The relative scale of the bands is set by the caller; the scaling determines the color balance.

    Verification
    ------------
    tests/unit/test_visualization.py: preservation of color ratios, preservation of color in a saturated pixel.
    """
    r, g, b = (np.clip(np.asarray(k, dtype=float) - black, 0.0, None) for k in (red, green, blue))
    intensity = (r + g + b) / 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(intensity > 0, asinh_stretch(intensity, black=0.0, alpha=alpha, q=q) / intensity, 0.0)
    rgb = np.stack([r * factor, g * factor, b * factor], axis=-1)
    largest = rgb.max(axis=-1, keepdims=True)
    return np.where(largest > 1.0, rgb / np.maximum(largest, 1e-300), rgb)


@source(
    "Lupton2004",
    where="Section 2: linear mapping and the maximum value M",
    status="read",
    note="choosing the lower and upper limits from percentiles is this project's rule; the source leaves m and M to the user",
)
def percentile_scale(image: ArrayLike, low_percent: float = 1.0, high_percent: float = 99.5) -> Array:
    """Map the image linearly to the range [0, 1] between two percentiles.

    References
    ----------
    [Lupton2004] Section 2.

    Validity
    --------
    Non-finite pixels do not enter the percentile computation and are 0 in the output.

    Verification
    ------------
    tests/unit/test_visualization.py
    """
    x = np.asarray(image, dtype=float)
    low, high = np.nanpercentile(x, [low_percent, high_percent])
    if high <= low:
        return np.zeros_like(x)
    return np.nan_to_num(np.clip((x - low) / (high - low), 0.0, 1.0), nan=0.0)
