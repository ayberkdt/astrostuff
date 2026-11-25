"""Astrometric solution from an image: match detected sources to a catalog and fit a WCS.

The linear part of the FITS WCS connects pixel coordinates to tangent-plane coordinates:
(ξ, η) = CD · (p − CRPIX) [CalabrettaGreisen2002, TeX label eq:px].
The functions here use this relation in the reverse direction: CD and CRPIX are found by
least squares from the pixel and sky positions of matched stars.

The method itself (finding the shift by voting, the match-and-fit iteration) is this
project's choice; it is not taken from a paper. The scope is limited: an approximate WCS
(scale and orientation within a few percent, shift of tens of pixels) is required. There is no
pattern matching ("blind" solution) without any prior information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.astrometry.matching import cross_match
from kozmos.astrometry.wcs import Wcs, pixel_to_sky, sky_to_pixel

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class WcsSolution:
    """Fitted WCS and the quality of the fit.

    Attributes
    ----------
    wcs : Wcs
    n_stars : int
        Number of matched stars that entered the fit.
    residual_rms_arcsec : float
        Root mean square of the position residuals on the sky for the matched stars.
    """

    wcs: Wcs
    n_stars: int
    residual_rms_arcsec: float


def _tangent_plane(ra_deg: Array, dec_deg: Array, crval: Array) -> tuple[Array, Array]:
    """Tangent-plane coordinates (ξ, η) [deg] of sky positions around crval."""
    unit = Wcs(crpix=np.array([1.0, 1.0]), crval=np.asarray(crval, dtype=float), cd=np.eye(2))
    return sky_to_pixel(unit, ra_deg, dec_deg)


@source(
    "CalabrettaGreisen2002",
    where="TeX label eq:px (linear pixel → intermediate coordinate relation)",
    status="read",
    note="the relation (ξ, η) = CD (p − CRPIX) is linear in the form ξ = a0 + a1 x + a2 y, η = b0 + b1 x + b2 y; "
    "the six coefficients are found by least squares, CRPIX = −CD⁻¹ (a0, b0). The fitting method is this "
    "project's choice",
)
def fit_wcs(
    x: ArrayLike,
    y: ArrayLike,
    ra_deg: ArrayLike,
    dec_deg: ArrayLike,
    crval: ArrayLike | None = None,
) -> WcsSolution:
    """Fit a WCS with TAN projection from the pixel and sky positions of matched stars.

    Parameters
    ----------
    x, y : ndarray
        Zero-based pixel coordinates.
    ra_deg, dec_deg : ndarray
        Catalog positions of the same stars [deg].
    crval : (2,), optional
        Tangent point [deg]; if not given, the mean direction of the stars is used.

    Returns
    -------
    WcsSolution

    References
    ----------
    [CalabrettaGreisen2002] eq:px.

    Validity
    --------
    At least three stars are required; stars on the same line do not determine the solution.
    Only linear terms are fitted (scale, rotation, skew, shift); optical
    distortion is not modeled. Outlier matches are not removed.

    Verification
    ------------
    tests/unit/test_astrometry_solve.py: recovery of CD and CRPIX for stars generated from a known WCS.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ra, dec = np.asarray(ra_deg, dtype=float), np.asarray(dec_deg, dtype=float)
    if x.size < 3:
        raise ValueError("at least three matched stars are required to fit a WCS")
    if crval is None:
        v = np.array(
            [
                np.mean(np.cos(np.radians(dec)) * np.cos(np.radians(ra))),
                np.mean(np.cos(np.radians(dec)) * np.sin(np.radians(ra))),
                np.mean(np.sin(np.radians(dec))),
            ]
        )
        crval = np.array(
            [np.degrees(np.arctan2(v[1], v[0])) % 360.0, np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))]
        )
    crval = np.asarray(crval, dtype=float)
    xi, eta = _tangent_plane(ra, dec, crval)
    design = np.column_stack([np.ones(x.size), x + 1.0, y + 1.0])  # FITS pixel = zero-based + 1
    a, *_ = np.linalg.lstsq(design, xi, rcond=None)
    b, *_ = np.linalg.lstsq(design, eta, rcond=None)
    cd = np.array([[a[1], a[2]], [b[1], b[2]]])
    crpix = -np.linalg.solve(cd, np.array([a[0], b[0]]))
    wcs = Wcs(crpix=crpix, crval=crval, cd=cd)
    ra_m, dec_m = pixel_to_sky(wcs, x, y)
    d_ra = (ra_m - ra + 180.0) % 360.0 - 180.0
    residual = np.hypot(d_ra * np.cos(np.radians(dec)), dec_m - dec) * 3600.0
    return WcsSolution(wcs, int(x.size), float(np.sqrt(np.mean(residual**2))))


@source(
    "CalabrettaGreisen2002",
    where="TeX label eq:px: CRPIX is the pixel position of the reference point",
    status="read",
    note="the shift is found from the peak of the two-dimensional histogram of all pairwise differences between "
    "the pixel positions of the catalog computed with the approximate WCS and the detected positions, and is "
    "added to CRPIX. The voting method is this project's choice",
)
def find_shift(
    wcs: Wcs,
    x: ArrayLike,
    y: ArrayLike,
    ra_deg: ArrayLike,
    dec_deg: ArrayLike,
    *,
    max_shift_pix: float = 60.0,
    bin_pix: float = 2.0,
) -> tuple[Wcs, int]:
    """Find the pixel shift of an approximate WCS by voting and return the corrected WCS.

    Parameters
    ----------
    wcs : Wcs
        WCS with approximately correct scale and orientation and a shifted zero point.
    x, y : ndarray
        Zero-based pixel positions of the detected sources.
    ra_deg, dec_deg : ndarray
        Catalog positions [deg].
    max_shift_pix : float
        Largest shift searched (along each axis).
    bin_pix : float
        Bin width of the voting histogram.

    Returns
    -------
    (corrected WCS, number of votes in the peak bin)

    References
    ----------
    [CalabrettaGreisen2002] eq:px.

    Validity
    --------
    If a rotation or scale error causes a shift larger than the bin width at the edge of the
    field, the votes are spread out. If the peak vote is not clearly above the chance level
    (number of sources × number of catalog stars × bin area / search area), the solution cannot be trusted.

    Verification
    ------------
    tests/unit/test_astrometry_solve.py: recovery of a known shift in the presence of false detections and missing catalog stars.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    kx, ky = sky_to_pixel(wcs, ra_deg, dec_deg)
    finite = np.isfinite(kx) & np.isfinite(ky)
    dx = (x[:, None] - kx[finite][None, :]).ravel()
    dy = (y[:, None] - ky[finite][None, :]).ravel()
    nearby = (np.abs(dx) <= max_shift_pix) & (np.abs(dy) <= max_shift_pix)
    edges = np.arange(-max_shift_pix, max_shift_pix + bin_pix, bin_pix)
    votes, _, _ = np.histogram2d(dx[nearby], dy[nearby], bins=(edges, edges))
    # The peak may fall on the boundary of two bins: search in the map summed with neighboring bins.
    pooled = sum(np.roll(np.roll(votes, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1))
    i, j = np.unravel_index(np.argmax(pooled), pooled.shape)
    selected = nearby & (np.abs(dx - (edges[i] + bin_pix / 2)) <= 1.5 * bin_pix)
    selected &= np.abs(dy - (edges[j] + bin_pix / 2)) <= 1.5 * bin_pix
    shift = np.array([np.median(dx[selected]), np.median(dy[selected])])
    return Wcs(crpix=wcs.crpix + shift, crval=wcs.crval, cd=wcs.cd, lonpole=wcs.lonpole), int(pooled[i, j])


@source(
    "CalabrettaGreisen2002",
    where="TeX label eq:px",
    status="read",
    note="the match (cross_match) and fit (fit_wcs) steps are iterated while the matching radius is narrowed. "
    "The iteration scheme is this project's choice",
)
def refine_wcs(
    wcs: Wcs,
    x: ArrayLike,
    y: ArrayLike,
    ra_deg: ArrayLike,
    dec_deg: ArrayLike,
    *,
    radius_arcsec: float = 5.0,
    final_radius_arcsec: float = 1.5,
    iterations: int = 6,
) -> WcsSolution:
    """Improve an approximate WCS against a catalog by match-and-fit iteration.

    Parameters
    ----------
    wcs : Wcs
        Initial WCS; it must place most stars on the correct star within ``radius_arcsec``
        (first ``find_shift`` if necessary).
    radius_arcsec, final_radius_arcsec : float
        First and last matching radius; narrowed geometrically in between.

    Returns
    -------
    WcsSolution
        Solution of the last iteration.

    References
    ----------
    [CalabrettaGreisen2002] eq:px.

    Validity
    --------
    Stars whose initial error exceeds the matching radius are not matched in the first iteration; if
    the error exceeds the radius over a large part of the field, the iteration can converge to a wrong
    solution. Proper motion is not applied.

    Verification
    ------------
    tests/unit/test_astrometry_solve.py: convergence to the true WCS from a rotated, scale-distorted and shifted start.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ra_k, dec_k = np.asarray(ra_deg, dtype=float), np.asarray(dec_deg, dtype=float)
    radii = np.geomspace(radius_arcsec, final_radius_arcsec, iterations)
    solution = None
    for radius in radii:
        ra, dec = pixel_to_sky(wcs, x, y)
        match = cross_match(ra, dec, ra_k, dec_k, float(radius))
        if match.matched.sum() < 3:
            raise ValueError(f"fewer than three matches within {radius:.1f} arcsec; the initial WCS is insufficient")
        solution = fit_wcs(
            x[match.matched],
            y[match.matched],
            ra_k[match.index[match.matched]],
            dec_k[match.index[match.matched]],
            wcs.crval,
        )
        wcs = solution.wcs
    return solution
