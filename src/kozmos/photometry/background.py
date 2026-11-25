"""Sky background estimation: a global level and a smooth background map.

The estimator was read from the "Modeling the background" section of the
SExtractor documentation [SExtractorDocs2026]: iterative ±3σ clipping around the median,
mode = 2.5 × median − 1.5 × mean, and the median if the mode and the median differ by more
than 30%; for the map, a cell grid, a median filter and a bicubic spline. The paper describing
the method [BertinArnouts1996] was not read in this work.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from kozmos._sources import source

Array = NDArray[np.float64]


@source(
    "SExtractorDocs2026",
    where="Modeling the background: Background estimation, Eq. 4",
    status="read",
    note="±3σ clipping around the median until convergence; Mode = 2.5 × Median − 1.5 × Mean; "
    "the median if the mode and the median differ by more than 30%",
)
def estimate_background(image: ArrayLike, *, sigma: float = 3.0, max_iterations: int = 50) -> tuple[float, float]:
    """Estimate the background level and noise of an image patch.

    Returns
    -------
    (level, rms) : background level and standard deviation of the clipped distribution.

    References
    ----------
    [SExtractorDocs2026] Background estimation.

    Validity
    --------
    Assumes that sources cover a small part of the patch. The "30% difference"
    criterion is applied here as |mode − median| > 0.3 σ; it is not clear in the
    documentation what the 30% is relative to.

    Verification
    ------------
    tests/unit/test_photometry.py: unbiased recovery of the level in a synthetic field with added stars.
    """
    x = np.asarray(image, dtype=float).ravel()
    x = x[np.isfinite(x)]
    for _ in range(max_iterations):
        median, scale = np.median(x), x.std()
        kept = x[np.abs(x - median) <= sigma * scale]
        if kept.size == x.size:
            break
        x = kept
    median, mean, scale = float(np.median(x)), float(x.mean()), float(x.std())
    mode = 2.5 * median - 1.5 * mean
    return (median if abs(mode - median) > 0.3 * scale else mode), scale


@source(
    "SExtractorDocs2026",
    where="Modeling the background: Background map",
    status="read",
    note="local estimate on a cell grid, median filter, bicubic spline between cells",
)
def background_map(image: ArrayLike, cell: int = 64, filter_size: int = 3) -> tuple[Array, Array]:
    """Build a map of the slowly varying background and its noise.

    Parameters
    ----------
    image : ndarray (Y, X)
    cell : int
        Cell side [pixel]; chosen larger than the sources and smaller than the scale of background variation.
    filter_size : int
        Width of the median filter applied to the cell map (in cells); 1 means no filter.

    Returns
    -------
    (background, rms) : two maps with the same shape as the image.

    References
    ----------
    [SExtractorDocs2026] Background map.

    Validity
    --------
    Extended sources larger than a cell (nebulae, galaxies) are counted as background and
    subtracted. The interpolation is a cubic spline with ``scipy.ndimage.zoom``; the edge
    behavior may differ from the "natural bicubic spline" in the documentation.

    Verification
    ------------
    tests/unit/test_photometry.py: recovery of a sloped background within 1%.
    """
    g = np.asarray(image, dtype=float)
    ny, nx = g.shape
    hy, hx = max(1, ny // cell), max(1, nx // cell)
    level, rms = np.empty((hy, hx)), np.empty((hy, hx))
    for i in range(hy):
        for j in range(hx):
            cell_image = g[i * ny // hy : (i + 1) * ny // hy, j * nx // hx : (j + 1) * nx // hx]
            level[i, j], rms[i, j] = estimate_background(cell_image)
    if filter_size > 1:
        level = ndimage.median_filter(level, size=filter_size, mode="nearest")
        rms = ndimage.median_filter(rms, size=filter_size, mode="nearest")
    order = 3 if min(hy, hx) >= 4 else 1
    enlarge = lambda h: ndimage.zoom(h, (ny / hy, nx / hx), order=order, mode="nearest", grid_mode=True)  # noqa: E731
    return enlarge(level)[:ny, :nx], np.clip(enlarge(rms)[:ny, :nx], 1e-300, None)
