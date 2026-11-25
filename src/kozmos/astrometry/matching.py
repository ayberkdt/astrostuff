"""Positional matching of two source lists on the sky.

The angular separation is the angle between unit vectors (ERFA ``sepp.c``)
[ERFA2023]. The nearest-neighbor search is done with a k-d tree in the
three-dimensional space of unit vectors; because the chord length 2 sin(θ/2) increases
monotonically with the angle, the nearest neighbor by chord is the nearest neighbor by angle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from kozmos._sources import source
from kozmos.core.angles import spherical_to_unit_vector

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Match:
    """Nearest neighbor in the second list of each source in the first list.

    Attributes
    ----------
    index : ndarray[int]
        Index of the nearest source in the second list.
    separation_arcsec : ndarray
        Angular distance to that source [arcsec].
    matched : ndarray[bool]
        True if the separation is within the given radius.
    """

    index: NDArray[np.int64]
    separation_arcsec: Array
    matched: NDArray[np.bool_]


@source(
    "ERFA2023",
    where="sepp.c, s2c.c",
    status="read",
    note="chord = |a − b| = 2 sin(θ/2) ⇒ θ = 2 asin(chord/2) (derivation in this project); neighbor search with scipy cKDTree",
)
def cross_match(
    ra1_deg: ArrayLike,
    dec1_deg: ArrayLike,
    ra2_deg: ArrayLike,
    dec2_deg: ArrayLike,
    radius_arcsec: float,
) -> Match:
    """Find the nearest source in the second list for each source in the first list.

    Parameters
    ----------
    ra1_deg, dec1_deg : ndarray (N,)
    ra2_deg, dec2_deg : ndarray (M,)
    radius_arcsec : float
        Largest separation counted as a match.

    Returns
    -------
    Match

    References
    ----------
    [ERFA2023] sepp.c, s2c.c.

    Validity
    --------
    The matching is one-directional: a source in the second list can be the nearest
    neighbor of more than one source. Proper motion and epoch difference are not corrected.

    Verification
    ------------
    tests/oracle/test_wcs_oracle.py: same indices and separations as astropy match_to_catalog_sky.
    """
    a = spherical_to_unit_vector(np.radians(ra1_deg), np.radians(dec1_deg))
    b = spherical_to_unit_vector(np.radians(ra2_deg), np.radians(dec2_deg))
    chord, index = cKDTree(np.atleast_2d(b)).query(np.atleast_2d(a), k=1)
    separation = np.degrees(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))) * 3600.0
    return Match(index.astype(np.int64), separation, separation <= radius_arcsec)
