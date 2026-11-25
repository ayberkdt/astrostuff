"""Frame bias and precession: moving ICRS coordinates to the mean equator of date.

The Fukushima-Williams angles of the IAU 2006 precession and the rotation
matrix built from them were read from the ERFA source code [ERFA2023]:
``pfw06.c`` (polynomials of the angles), ``fw2m.c`` (matrix), ``pmat06.c``
(the combination of the two).

Nutation (up to about 17 arcseconds) and aberration (about 20 arcseconds) are
not included here; the result is in "mean of date" coordinates, not "apparent"
coordinates.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.angles import TWO_PI
from kozmos.time.julian import mean_obliquity

Array = NDArray[np.float64]

_JD_J2000 = 2451545.0  # erfam.h ERFA_DJ00
_JULIAN_CENTURY_DAYS = 36525.0  # erfam.h ERFA_DJC
_ARCSEC_RAD = np.pi / (180.0 * 3600.0)

# pfw06.c: polynomial coefficients of the P03 bias + precession angles [arcsec], in increasing powers of t.
_GAMB = (-0.052928, 10.556378, 0.4932044, -0.00031238, -0.000002788, 0.0000000260)
_PHIB = (84381.412819, -46.811016, 0.0511268, 0.00053289, -0.000000440, -0.0000000176)
_PSIB = (-0.041775, 5038.481484, 1.5584175, -0.00018522, -0.000026452, -0.0000000148)


def _polynomial(coefficients: tuple[float, ...], t: Array) -> Array:
    total = np.zeros_like(t)
    for k in reversed(coefficients):
        total = total * t + k
    return total * _ARCSEC_RAD


@source(
    "ERFA2023",
    where="pfw06.c",
    status="read",
    note="t = ((date1 − 2451545.0) + date2) / 36525 (TT, Julian centuries); γ̄, φ̄, ψ̄ are fifth-degree polynomials; "
    "ε_A = eraObl06",
)
def fukushima_williams_angles(jd1: ArrayLike, jd2: ArrayLike = 0.0) -> tuple[Array, Array, Array, Array]:
    """Compute the IAU 2006 bias-precession Fukushima-Williams angles.

    Parameters
    ----------
    jd1, jd2 : float or ndarray
        Two-part Julian date (TT).

    Returns
    -------
    (γ̄, φ̄, ψ̄, ε_A) [rad]

    References
    ----------
    [ERFA2023] pfw06.c.

    Validity
    --------
    The polynomials are built for a few centuries around J2000; accuracy drops at distant dates.

    Verification
    ------------
    tests/oracle/test_precession_oracle.py: equality with erfa.pfw06.
    """
    t = ((np.asarray(jd1, dtype=float) - _JD_J2000) + np.asarray(jd2, dtype=float)) / _JULIAN_CENTURY_DAYS
    return _polynomial(_GAMB, t), _polynomial(_PHIB, t), _polynomial(_PSIB, t), mean_obliquity(jd1, jd2)


def _rz(angle: Array) -> Array:
    c, s, zero, one = np.cos(angle), np.sin(angle), np.zeros_like(angle), np.ones_like(angle)
    return np.stack([np.stack([c, s, zero], -1), np.stack([-s, c, zero], -1), np.stack([zero, zero, one], -1)], -2)


def _rx(angle: Array) -> Array:
    c, s, zero, one = np.cos(angle), np.sin(angle), np.zeros_like(angle), np.ones_like(angle)
    return np.stack([np.stack([one, zero, zero], -1), np.stack([zero, c, s], -1), np.stack([zero, -s, c], -1)], -2)


@source(
    "ERFA2023",
    where="pmat06.c, fw2m.c",
    status="read",
    note="R = R_x(−ε_A) R_z(−ψ̄) R_x(φ̄) R_z(γ̄) (call order in fw2m.c: Rz(γ̄), Rx(φ̄), Rz(−ψ̄), Rx(−ε)). "
    "The sign convention of the elementary rotation matrices was not read from rx.c and rz.c; equality with erfa.pmat06 was tested",
)
def bias_precession_matrix(jd1: ArrayLike, jd2: ArrayLike = 0.0) -> Array:
    """Build the matrix that rotates GCRS (ICRS-oriented) vectors to the mean equator and equinox of date.

    v(date) = R · v(GCRS).

    Returns
    -------
    ndarray (..., 3, 3)

    References
    ----------
    [ERFA2023] pmat06.c, fw2m.c.

    Validity
    --------
    Includes frame bias and precession; does not include nutation.

    Verification
    ------------
    tests/oracle/test_precession_oracle.py: equality with erfa.pmat06 within 1e-14; the matrix is orthogonal.
    """
    gamb, phib, psib, epsa = fukushima_williams_angles(jd1, jd2)
    return _rx(-epsa) @ _rz(-psib) @ _rx(phib) @ _rz(gamb)


@source(
    "ERFA2023",
    where="pmat06.c (matrix), s2c.c and c2s.c (spherical ↔ unit vector)",
    status="read",
    note="the direction is converted to a unit vector, rotated with the bias-precession matrix and converted back to spherical coordinates",
)
def icrs_to_mean_of_date(
    ra_rad: ArrayLike, dec_rad: ArrayLike, jd1: ArrayLike, jd2: ArrayLike = 0.0
) -> tuple[Array, Array]:
    """Move ICRS coordinates to the mean equator and equinox of date.

    Parameters
    ----------
    ra_rad, dec_rad : float or ndarray
        ICRS coordinates [rad].
    jd1, jd2 : float or ndarray
        Two-part Julian date (TT); broadcastable with the coordinates.

    Returns
    -------
    (right ascension [0, 2π), declination) [rad]

    References
    ----------
    [ERFA2023] pmat06.c, s2c.c, c2s.c.

    Validity
    --------
    Proper motion, parallax, nutation and aberration are not applied; the remaining difference
    for stars is up to about half an arcminute.

    Verification
    ------------
    tests/oracle/test_precession_oracle.py: equality with the vector rotated by erfa.pmat06; comparison with
    astropy's mean-equator-of-date frame (PrecessedGeocentric).
    """
    a, d = np.asarray(ra_rad, dtype=float), np.asarray(dec_rad, dtype=float)
    v = np.stack(np.broadcast_arrays(np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)), axis=-1)
    w = np.einsum("...ij,...j->...i", bias_precession_matrix(jd1, jd2), v)
    return np.arctan2(w[..., 1], w[..., 0]) % TWO_PI, np.arctan2(w[..., 2], np.hypot(w[..., 0], w[..., 1]))
