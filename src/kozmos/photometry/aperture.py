"""Aperture photometry: a circular aperture with subpixel sampling and an optional sky annulus.

Follows the definitions in the photutils documentation [Photutils2026]; tested with synthetic
images whose truth is known.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source

Array = NDArray[np.float64]


@source(
    "Photutils2026",
    where="User guide: Aperture Photometry, 'subpixel' method",
    status="read",
    note="the pixel is divided into N×N subpixels; the weight is the fraction of subpixels that fall inside the "
    "circular aperture. Subtracting the local sky as the median of an annulus is this project's choice",
)
def aperture_photometry(
    image: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    radius: float,
    *,
    annulus: tuple[float, float] | None = None,
    subpixels: int = 5,
) -> tuple[Array, Array]:
    """Measure flux in a circular aperture; optionally subtract the local sky from an annulus.

    Parameters
    ----------
    image : ndarray (Y, X)
    x, y : ndarray (N,)
        Source centers [pixel].
    radius : float
        Aperture radius [pixel].
    annulus : (inner, outer), optional
        Radii of the sky annulus; if given, the median in the annulus is taken as the sky per pixel.
    subpixels : int
        Number of samples per pixel (along each axis) for the covered fraction of edge pixels.

    Returns
    -------
    (flux, area) : sky-subtracted flux and effective area of the aperture [pixel²].

    References
    ----------
    [Photutils2026] Aperture Photometry.

    Validity
    --------
    Subpixel sampling gives the area of the circle with a relative error of about 1/subpixels².
    The covered fraction of an edge pixel is measured by area and the flux within a pixel is taken
    as uniform; if the aperture cuts the steep part of the PSF (radius smaller than the FWHM) the flux is
    measured a few percent low. For apertures extending beyond the image edge only the pixels inside the
    image are summed.

    Verification
    ------------
    tests/unit/test_photometry.py: area πr² on a flat image; enclosed flux fraction 1 − exp(−r²/2σ²) for a Gaussian star.
    """
    g = np.asarray(image, dtype=float)
    ny, nx = g.shape
    subpixel_offsets = (np.arange(subpixels) + 0.5) / subpixels - 0.5
    fluxes, areas = [], []
    for x0, y0 in zip(
        np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float)), strict=True
    ):
        outer = annulus[1] if annulus else radius
        x1, x2 = max(0, int(np.floor(x0 - outer - 1))), min(nx, int(np.ceil(x0 + outer + 2)))
        y1, y2 = max(0, int(np.floor(y0 - outer - 1))), min(ny, int(np.ceil(y0 + outer + 2)))
        cutout = g[y1:y2, x1:x2]
        yy, xx = np.mgrid[y1:y2, x1:x2]
        dx_sub = xx[..., None, None] + subpixel_offsets[None, None, None, :] - x0
        dy_sub = yy[..., None, None] + subpixel_offsets[None, None, :, None] - y0
        fraction = (dx_sub**2 + dy_sub**2 <= radius**2).mean(axis=(-1, -2))
        sky = 0.0
        if annulus:
            r2 = (xx - x0) ** 2 + (yy - y0) ** 2
            in_annulus = (r2 >= annulus[0] ** 2) & (r2 <= annulus[1] ** 2)
            sky = float(np.median(cutout[in_annulus])) if in_annulus.any() else 0.0
        area = float(fraction.sum())
        fluxes.append(float((cutout * fraction).sum()) - sky * area)
        areas.append(area)
    return np.array(fluxes), np.array(areas)
