"""Cosmic-ray detection in a single frame: Laplacian edge detection (L.A.Cosmic).

The algorithm was read from the arXiv TeX source of van Dokkum's paper (astro-ph/0108003,
``cosmic.tex``) [vanDokkum2001]. Steps and equations:

1. The image is subsampled by a factor of two, convolved with the Laplacian kernel (TeX label
   ``kernel.eq``), negative values are set to zero, and the result is block-averaged
   back to the original resolution: L⁺.
2. Noise model (``noise.eq``): N = g⁻¹ sqrt(g (M₅ ∘ I) + σ_rn²).
3. Significance (``sigmap.eq``): S = L⁺ / (f_s N); large-scale structure is removed:
   S′ = S − (S ∘ M₅).
4. Fine-structure image: F = (M₃ ∘ I) − ([M₃ ∘ I] ∘ M₇).
5. Cosmic ray: S′ > σ_lim and L⁺ / F > f_lim.
6. Flagged pixels are replaced by the median of the surrounding good pixels and the
   procedure is iterated.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from kozmos._sources import source

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]

# [vanDokkum2001] kernel.eq: (1/4) [[0, −1, 0], [−1, 4, −1], [0, −1, 0]]
_LAPLACE = 0.25 * np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
_SUBSAMPLING = 2  # f_s


@source(
    "vanDokkum2001",
    where="TeX labels kernel.eq, noise.eq, sigmap.eq and the following equations (S′, F, selection criterion)",
    status="read",
    note="subsampling factor f_s = 2. f_lim: the source suggests 2 (the default) for well-sampled data and a "
    "larger value for undersampled data (about 5 for WFPC2 wide-field chips). The source says a lower threshold "
    "can be applied to the neighbors of flagged pixels but does not give the ratio; the ratio of 0.5 and the "
    "requirement that neighbors also meet the fine-structure criterion are this project's choice (without it, "
    "stellar cores adjacent to a cosmic ray get flagged). The default σ_lim = 5 is also this project's choice",
)
def detect_cosmic_rays(
    image_adu: ArrayLike,
    *,
    gain_e_adu: float,
    read_noise_e: float,
    sigma_lim: float = 5.0,
    f_lim: float = 2.0,
    max_iterations: int = 4,
    neighbor_threshold_ratio: float = 0.5,
) -> tuple[Array, Mask]:
    """Detect and clean cosmic rays in a single frame.

    Parameters
    ----------
    image_adu : ndarray (Y, X)
        Image [ADU]; the sky must not be subtracted (the noise model relies on the total count).
    gain_e_adu : float
        Gain g [e⁻/ADU].
    read_noise_e : float
        Read noise σ_rn [e⁻].
    sigma_lim : float
        Laplacian significance threshold σ_lim.
    f_lim : float
        Smallest ratio of the Laplacian image to the fine-structure image, f_lim; separates stars from cosmic rays.
    max_iterations : int
        Stops early when no new pixel is flagged.
    neighbor_threshold_ratio : float
        For pixels adjacent to flagged pixels the threshold becomes ``neighbor_threshold_ratio × sigma_lim``;
        this catches the faint edges of cosmic rays. If 1 is given, no separate threshold is applied to neighbors.

    Returns
    -------
    (clean image, mask) : the image with flagged pixels replaced by the median of the
    surrounding good pixels, and ``True`` = cosmic-ray mask.

    References
    ----------
    [vanDokkum2001] kernel.eq, noise.eq, sigmap.eq and the following equations.

    Validity
    --------
    If the PSF is well sampled (FWHM about 2.5 pixels or more), stars are separated
    reliably; for undersampled data ``f_lim`` should be increased. The source states that at
    the 5σ threshold about half of the 5σ peaks in the original image are flagged. The
    completeness and purity measured on a real HST STIS frame against the difference of two
    exposures are in ``figures/cosmic_rays_real_data.json``.

    Verification
    ------------
    tests/unit/test_cosmic_rays.py: completeness of cosmic rays added to a synthetic star field and preservation of the stars.
    """
    image = np.asarray(image_adu, dtype=float)
    cleaned = image.copy()
    mask = np.zeros(image.shape, dtype=bool)
    ny, nx = image.shape
    for _ in range(max_iterations):
        subsampled = np.repeat(np.repeat(cleaned, _SUBSAMPLING, axis=0), _SUBSAMPLING, axis=1)
        laplacian = np.clip(ndimage.convolve(subsampled, _LAPLACE, mode="nearest"), 0.0, None)
        laplacian_pos = laplacian.reshape(ny, _SUBSAMPLING, nx, _SUBSAMPLING).mean(axis=(1, 3))
        m5 = ndimage.median_filter(cleaned, size=5, mode="nearest")
        noise = np.sqrt(np.clip(gain_e_adu * m5, 0.0, None) + read_noise_e**2) / gain_e_adu
        s = laplacian_pos / (_SUBSAMPLING * noise)
        s_prime = s - ndimage.median_filter(s, size=5, mode="nearest")
        m3 = ndimage.median_filter(cleaned, size=3, mode="nearest")
        fine = np.clip(m3 - ndimage.median_filter(m3, size=7, mode="nearest"), 0.01, None)
        new = (s_prime > sigma_lim) & (laplacian_pos / fine > f_lim) & ~mask
        if neighbor_threshold_ratio < 1.0 and (new | mask).any():
            adjacent = ndimage.binary_dilation(new | mask, structure=np.ones((3, 3)))
            new |= adjacent & (s_prime > neighbor_threshold_ratio * sigma_lim) & (laplacian_pos / fine > f_lim) & ~mask
        if not new.any():
            break
        mask |= new
        # The masked median is computed only around masked pixels (for speed).
        ys, xs = np.nonzero(mask)
        for y0, x0 in zip(ys, xs, strict=True):
            y1, y2, x1, x2 = max(0, y0 - 2), min(ny, y0 + 3), max(0, x0 - 2), min(nx, x0 + 3)
            good = ~mask[y1:y2, x1:x2]
            if good.any():
                cleaned[y0, x0] = np.median(image[y1:y2, x1:x2][good])
    return cleaned, mask
