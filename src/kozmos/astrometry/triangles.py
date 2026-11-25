"""Triangle matching: match two point lists without knowing scale, rotation, translation or mirroring.

The method was read from the ADS scan of Groth's paper [Groth1986]:

- Every triple of points is a triangle. The vertices are ordered so that side 1-2 is the
  shortest, side 2-3 the middle one and side 3-1 the longest.
- A triangle has two numbers independent of scale, rotation and translation: the ratio R of
  the longest side to the shortest side (Eq. 1) and the cosine C of the angle at vertex 1
  (Eq. 5). Triangles with a large ratio (typical limit 10) are discarded.
- The orientation of the vertices (clockwise or counterclockwise) distinguishes a mirror
  transformation: true matches are either all of the same orientation or all of opposite orientation.
- Each matching triangle casts one vote for each vertex pair; points are assigned in order of
  votes, and assignment stops when the votes fall by half or when a point would be assigned a
  second time.

Where this implementation departs from the source: the R and C tolerances are given as fixed
numbers rather than through the error propagation in the source (Eq. 4 and 6); the magnification
filter that removes false matches (iterative in the source, Eq. 10-12) is reduced to a single
robust clip around the median of the perimeter ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from kozmos._sources import source
from kozmos.astrometry.matching import cross_match
from kozmos.astrometry.solve import WcsSolution, _tangent_plane, fit_wcs, refine_wcs
from kozmos.astrometry.wcs import pixel_to_sky

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Triangles:
    """Triangles of a point list.

    Attributes
    ----------
    vertices : (T, 3) int
        Vertex indices in the order of [Groth1986] (1-2 shortest, 3-1 longest side).
    ratio : (T,) float
        R: ratio of the longest side to the shortest side.
    cosine : (T,) float
        C: cosine of the angle at vertex 1.
    log_perimeter : (T,) float
        Natural logarithm of the perimeter.
    orientation : (T,) int
        Orientation of the vertices: +1 counterclockwise, −1 clockwise.
    """

    vertices: NDArray[np.int64]
    ratio: Array
    cosine: Array
    log_perimeter: Array
    orientation: NDArray[np.int64]


@source(
    "Groth1986",
    where="Section II.b: vertex ordering; Eq. 1-3 (R) and Eq. 5 (C); ratio limit",
    status="read",
    note="R = r3/r2, r3 = |vertex3 − vertex1| (longest side), r2 = |vertex2 − vertex1| (shortest side); "
    "C = (Δx3 Δx2 + Δy3 Δy2)/(r3 r2). The source gives a typical value of 10 for the ratio limit",
)
def build_triangles(x: ArrayLike, y: ArrayLike, ratio_limit: float = 10.0) -> Triangles:
    """Build all triangles of a point list and their invariants.

    Parameters
    ----------
    x, y : ndarray (N,)
    ratio_limit : float
        Triangles with R above this value (very thin) are discarded.

    Returns
    -------
    Triangles

    References
    ----------
    [Groth1986] Section II.b, Eq. 1-3, 5.

    Validity
    --------
    The number of triangles grows as N³/6; N should not exceed a few tens. Coincident points
    (zero-length sides) are discarded.

    Verification
    ------------
    tests/unit/test_triangles.py: R and C for a known triangle; invariance of the invariants under scale, rotation, translation and mirroring.
    """
    p = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    triples = np.array(list(combinations(range(len(p)), 3)), dtype=np.int64).reshape(-1, 3)
    if triples.size == 0:
        empty = np.empty(0)
        return Triangles(triples, empty, empty, empty, np.empty(0, dtype=np.int64))
    a, b, c = p[triples[:, 0]], p[triples[:, 1]], p[triples[:, 2]]
    # Side k is opposite vertex k: side0 = b-c, side1 = c-a, side2 = a-b.
    sides = np.column_stack([np.hypot(*(b - c).T), np.hypot(*(c - a).T), np.hypot(*(a - b).T)])
    order = np.argsort(sides, axis=1)  # vertices opposite the shortest, middle and longest side
    rows = np.arange(len(triples))
    # Vertex 1 is the common end of the shortest and longest sides: the vertex opposite the middle side. Vertex 2 is
    # the vertex opposite the longest side, vertex 3 the vertex opposite the shortest side.
    k1, k2, k3 = triples[rows, order[:, 1]], triples[rows, order[:, 2]], triples[rows, order[:, 0]]
    r2, r3 = sides[rows, order[:, 0]], sides[rows, order[:, 2]]
    valid = r2 > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = r3 / r2
    valid &= ratio <= ratio_limit
    k1, k2, k3, r2, r3, ratio = (d[valid] for d in (k1, k2, k3, r2, r3, ratio))
    d2, d3 = p[k2] - p[k1], p[k3] - p[k1]
    cosine = np.sum(d2 * d3, axis=1) / (r2 * r3)
    orientation = np.where(d2[:, 0] * d3[:, 1] - d2[:, 1] * d3[:, 0] >= 0, 1, -1).astype(np.int64)
    perimeter = sides[valid].sum(axis=1)
    return Triangles(np.column_stack([k1, k2, k3]), ratio, cosine, np.log(perimeter), orientation)


@source(
    "Groth1986",
    where="Section II.c (triangle matching, orientation), II.d (removing false matches), II.e (voting and assignment)",
    status="read",
    note="a matching triangle casts one vote for each of three vertex pairs; pairs are assigned in order of votes; "
    "assignment stops when the votes fall by half, when a point would be assigned a second time, or when the "
    "votes drop to zero; if no pair received more than one vote there is no match. The majority orientation is "
    "kept. The fixed tolerances and the single-step magnification clip are this project's simplification",
)
def match_triangles(
    x1: ArrayLike,
    y1: ArrayLike,
    x2: ArrayLike,
    y2: ArrayLike,
    *,
    ratio_tolerance: float = 0.01,
    cosine_tolerance: float = 0.01,
    ratio_limit: float = 10.0,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], int]:
    """Find the correspondences between two point lists by triangle voting.

    Parameters
    ----------
    x1, y1, x2, y2 : ndarray
        Coordinates of the two lists; units, scales and orientations may differ.
    ratio_tolerance : float
        Upper limit of the log R difference.
    cosine_tolerance : float
        Upper limit of the C difference.
    ratio_limit : float
        Triangles with R above this value are discarded.

    Returns
    -------
    (index1, index2, votes, orientation) : indices of the matched points in the two lists, the votes they
    received, and the orientation of the matches (+1 same orientation, −1 mirror).

    References
    ----------
    [Groth1986] Section II.c-e.

    Validity
    --------
    The common points of the two lists must be the majority in both; if one list is a
    small subregion of the other, there are few common triangles and no match is found. If optical
    distortion exceeds the tolerance the votes are spread out. If no match is found, empty arrays are returned.

    Verification
    ------------
    tests/unit/test_triangles.py: finding the correspondences in a rotated, scaled, mirrored and noisy list.
    """
    u1 = build_triangles(x1, y1, ratio_limit)
    u2 = build_triangles(x2, y2, ratio_limit)
    empty = np.empty(0, dtype=np.int64)
    if u1.ratio.size == 0 or u2.ratio.size == 0:
        return empty, empty, empty, 0
    scale = np.array([1.0 / ratio_tolerance, 1.0 / cosine_tolerance])
    space1 = np.column_stack([np.log(u1.ratio), u1.cosine]) * scale
    space2 = np.column_stack([np.log(u2.ratio), u2.cosine]) * scale
    neighbors = cKDTree(space2).query_ball_point(space1, r=1.0)
    i1 = np.repeat(np.arange(len(neighbors)), [len(k) for k in neighbors])
    i2 = np.fromiter((j for k in neighbors for j in k), dtype=np.int64, count=i1.size)
    if i1.size == 0:
        return empty, empty, empty, 0
    # Orientation: true matches are either all of the same orientation or all of opposite orientation; the majority is kept.
    same = u1.orientation[i1] == u2.orientation[i2]
    orientation = 1 if same.sum() >= (~same).sum() else -1
    keep = same if orientation == 1 else ~same
    i1, i2 = i1[keep], i2[keep]
    # Magnification: for true matches the perimeter ratio is the same (the scale ratio of the two lists).
    magnification = u1.log_perimeter[i1] - u2.log_perimeter[i2]
    counts, bin_edges = np.histogram(magnification, bins=max(10, int(np.ptp(magnification) / 0.02) + 1))
    peak = 0.5 * (bin_edges[np.argmax(counts)] + bin_edges[np.argmax(counts) + 1])
    keep = np.abs(magnification - peak) < 0.03
    i1, i2 = i1[keep], i2[keep]
    n1, n2 = len(np.asarray(x1)), len(np.asarray(x2))
    votes = np.zeros((n1, n2), dtype=np.int64)
    for k in range(3):
        np.add.at(votes, (u1.vertices[i1, k], u2.vertices[i2, k]), 1)
    order = np.argsort(-votes, axis=None)
    a1, a2, vote_counts = [], [], []
    previous = int(votes.flat[order[0]])
    if previous <= 1:
        return empty, empty, empty, orientation
    for s in order:
        p1, p2 = divmod(int(s), n2)
        v = int(votes[p1, p2])
        if v == 0 or v < 0.5 * previous or p1 in a1 or p2 in a2:
            break
        a1.append(p1)
        a2.append(p2)
        vote_counts.append(v)
        previous = v
    return (
        np.array(a1, dtype=np.int64),
        np.array(a2, dtype=np.int64),
        np.array(vote_counts, dtype=np.int64),
        orientation,
    )


@source(
    "Groth1986",
    where="Section II (algorithm) and Section III (example of matching a catalog and a plate list)",
    status="read",
    note="triangle matching with the brightest points, then a linear solution from the matched pairs. The step from "
    "the matching to a WCS (fit_wcs, refine_wcs) is this project's combination",
)
def blind_solve(
    x: ArrayLike,
    y: ArrayLike,
    brightness: ArrayLike,
    ra_deg: ArrayLike,
    dec_deg: ArrayLike,
    catalog_magnitude: ArrayLike,
    *,
    n_brightest: int = 25,
    final_radius_arcsec: float = 1.5,
    tolerance: float = 0.01,
    min_validation_fraction: float = 0.4,
    min_matches: int = 8,
) -> tuple[WcsSolution, int]:
    """Solve a WCS with TAN projection by matching detected sources to a catalog, without an initial WCS.

    Parameters
    ----------
    x, y : ndarray
        Zero-based pixel positions of the detected sources.
    brightness : ndarray
        Flux of the detected sources (large = bright); used to select the brightest.
    ra_deg, dec_deg, catalog_magnitude : ndarray
        Catalog of the region covered by the image, and its magnitudes (small = bright).
    n_brightest : int
        Number of points (from each list) entering the triangle matching.
    final_radius_arcsec : float
        Final matching radius of the refinement and the validation.
    tolerance : float
        Matching tolerance of the triangle invariants (log R and C).
    min_validation_fraction : float
        The solution must match at least this fraction of the bright sources entering the triangle matching
        to the catalog; otherwise the solution is rejected.
    min_matches : int
        Absolute lower limit of the number of bright sources that must match in the validation. A "solution"
        of a few stars cannot be distinguished from chance (in a real, sparse SOFI field a wrong solution
        with an error of 471 arcseconds passed the fraction criterion with 5 stars).

    Returns
    -------
    (WcsSolution, number of pairs assigned in the triangle matching)

    References
    ----------
    [Groth1986] Section II, III.

    Validity
    --------
    Works without knowing scale, orientation, mirroring or tangent point; but the catalog must
    cover about the same area as the region covered by the image (an approximate sky position and
    field width are needed). It is not an indexed solver that searches the whole sky. Raises
    ``ValueError`` if fewer than three pairs are assigned or if the solution found does not pass the
    validation. The validation is necessary: a solution built from a few wrong pairs can "match" tens of
    stars by chance in a dense catalog (seen in a real SOFI stack).

    Verification
    ------------
    tests/unit/test_triangles.py: solution without prior information in a field generated from a known WCS.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ra, dec = np.asarray(ra_deg, dtype=float), np.asarray(dec_deg, dtype=float)
    g = np.argsort(-np.asarray(brightness, dtype=float))[:n_brightest]
    magnitude = np.asarray(catalog_magnitude, dtype=float)
    k = np.argsort(np.where(np.isfinite(magnitude), magnitude, np.inf))[:n_brightest]
    v = np.array(
        [
            np.cos(np.radians(dec)) * np.cos(np.radians(ra)),
            np.cos(np.radians(dec)) * np.sin(np.radians(ra)),
            np.sin(np.radians(dec)),
        ]
    ).mean(axis=1)
    crval = np.array([np.degrees(np.arctan2(v[1], v[0])) % 360.0, np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))])
    xi, eta = _tangent_plane(ra[k], dec[k], crval)
    i1, i2, _, _ = match_triangles(x[g], y[g], xi, eta, ratio_tolerance=tolerance, cosine_tolerance=tolerance)
    if i1.size < 3:
        raise ValueError(f"triangle matching assigned only {i1.size} pairs; the lists may not cover the same field")
    initial = fit_wcs(x[g][i1], y[g][i1], ra[k][i2], dec[k][i2], crval)
    radius = max(3.0 * initial.residual_rms_arcsec, 2.0 * final_radius_arcsec)
    try:
        solution = refine_wcs(initial.wcs, x, y, ra, dec, radius_arcsec=radius, final_radius_arcsec=final_radius_arcsec)
    except ValueError as error:
        raise ValueError(f"the solution built from the triangle matching could not be refined: {error}") from error
    # Validation: a sufficient fraction of the bright sources must find a counterpart in the catalog.
    ra_p, dec_p = pixel_to_sky(solution.wcs, x[g], y[g])
    # Distinct catalog stars are counted: a wrong solution with a collapsed scale can drop all sources onto a single star.
    match = cross_match(ra_p, dec_p, ra, dec, final_radius_arcsec)
    matched_count = int(np.unique(match.index[match.matched]).size)
    required = max(float(min_matches), min_validation_fraction * min(g.size, k.size))
    if matched_count < required:
        raise ValueError(
            f"solution could not be validated: {g.size} bright sources matched only {matched_count} distinct catalog "
            f"stars (at least {required:.0f} required)"
        )
    return solution, int(i1.size)
