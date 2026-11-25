"""Observation planning: altitude, azimuth and airmass of a target through the night.

Combines existing building blocks: Greenwich mean sidereal time
(``time.julian``, ERFA ``gmst06.c``) and the conversion from hour angle to
horizontal coordinates (``coordinates.transforms``, ERFA ``hd2ae.c``) [ERFA2023].

Coordinates are moved to the mean equator of date by frame bias and precession
(``coordinates.precession``, ERFA ``pmat06.c``); atmospheric refraction is added
on request (``observing.refraction``, ERFA ``refco.c``). Nutation, aberration,
polar motion and the UT1 − UTC difference are not applied; the difference
measured against astropy is below one arcminute (tests/oracle/test_planning_oracle.py).
It is not sufficient for telescope pointing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.coordinates.precession import icrs_to_mean_of_date
from kozmos.coordinates.transforms import airmass_plane_parallel, hour_angle_to_horizontal
from kozmos.core.angles import TWO_PI, wrap_pm_pi
from kozmos.observing.refraction import observed_altitude
from kozmos.time.julian import greenwich_mean_sidereal_time

Array = NDArray[np.float64]
SIDEREAL_RATE = 1.00273781191135448  # era00.c: Earth rotation per UT1 day (turns)


@dataclass(frozen=True, slots=True)
class Visibility:
    """Horizon coordinates of a target at the given times."""

    hour_angle_rad: Array
    azimuth_rad: Array
    altitude_rad: Array
    airmass: Array


@source(
    "ERFA2023",
    where="gmst06.c, hd2ae.c, pmat06.c, refco.c",
    status="read",
    note="coordinates are moved to the date with the bias-precession matrix; H = GMST + east longitude − right "
    "ascension; then hour angle → azimuth, altitude; refraction on request. UT1 ≈ UTC ≈ TT is assumed",
)
def visibility(
    ra_rad: float,
    dec_rad: float,
    latitude_rad: float,
    east_longitude_rad: float,
    jd1: ArrayLike,
    jd2: ArrayLike = 0.0,
    *,
    to_date: bool = True,
    refraction: tuple[float, float] | None = None,
) -> Visibility:
    """Compute the hour angle, azimuth, altitude and airmass of a target at the given times.

    Parameters
    ----------
    ra_rad, dec_rad : float
        ICRS (J2000) coordinates of the target [rad].
    latitude_rad, east_longitude_rad : float
        Latitude of the observer and longitude, positive east [rad].
    jd1, jd2 : float or ndarray
        Two-part Julian date (UTC).
    to_date : bool
        If true, the coordinates are moved to the mean equator of date; if false, they are used as
        given (when the coordinates are already in the equinox of date).
    refraction : (A, B), optional
        Output of ``observing.refraction.refraction_constants``; if given, the altitude is the
        observed (refracted) altitude.

    Returns
    -------
    Visibility

    References
    ----------
    [ERFA2023] gmst06.c, hd2ae.c, pmat06.c, refco.c.

    Validity
    --------
    UT1, UTC and TT are treated as equal (an effect of at most a few seconds
    in sidereal time). Nutation and aberration are not applied. With ``to_date=False`` and J2000
    coordinates the error grows by about 50 arcseconds per year. The refraction
    model gives ``nan`` below an altitude of about 5°. The airmass is the
    plane-parallel approximation; ``nan`` below the horizon.

    Verification
    ------------
    tests/oracle/test_planning_oracle.py: agreement with astropy AltAz within one arcminute (with and without refraction).
    """
    ra, dec = ra_rad, dec_rad
    if to_date:
        ra, dec = icrs_to_mean_of_date(ra, dec, jd1, jd2)
    gmst = greenwich_mean_sidereal_time(jd1, jd2, jd1, jd2)
    hour_angle = wrap_pm_pi(gmst + east_longitude_rad - ra)
    azimuth, altitude = hour_angle_to_horizontal(hour_angle, dec, latitude_rad)
    if refraction is not None:
        altitude = observed_altitude(altitude, *refraction)
    return Visibility(hour_angle, azimuth, altitude, airmass_plane_parallel(altitude))


@source(
    "ERFA2023",
    where="era00.c (rotation rate), hd2ae.c",
    status="read",
    note="meridian transit at H = 0; the hour angle advances 1.00273781191135448 turns per UT1 day",
)
def transit_jd(
    ra_rad: float,
    east_longitude_rad: float,
    jd_start: float,
    *,
    dec_rad: float | None = None,
) -> float:
    """Find the Julian date of the first upper meridian transit after a given time.

    If ``dec_rad`` is given, the coordinates are taken as ICRS and the right ascension is moved to
    the mean equinox of the start time (precession also depends on declination); if not given,
    the right ascension is used as is.

    References
    ----------
    [ERFA2023] era00.c, hd2ae.c.

    Validity
    --------
    Same approximations as ``visibility``. The result is accurate to a few seconds.

    Verification
    ------------
    tests/oracle/test_planning_oracle.py: hour angle is zero and altitude is maximal at the time found.
    """
    if dec_rad is not None:
        ra_rad = float(icrs_to_mean_of_date(ra_rad, dec_rad, jd_start)[0])
    gmst = float(greenwich_mean_sidereal_time(jd_start, 0.0, jd_start, 0.0))
    remaining = (ra_rad - gmst - east_longitude_rad) % TWO_PI
    return jd_start + remaining / TWO_PI / SIDEREAL_RATE


@source(
    "ERFA2023",
    where="hd2ae.c (altitude component)",
    status="read",
    note="solution for H of the equation sin h = sin φ sin δ + cos φ cos δ cos H (derivation in this project)",
)
def time_above_horizon_hr(dec_rad: ArrayLike, latitude_rad: float, altitude_limit_rad: float = 0.0) -> Array:
    """Compute the time per day a target stays above a given altitude limit.

    cos H₀ = (sin h₀ − sin φ sin δ) / (cos φ cos δ); duration = 2 H₀ (in sidereal time).

    Returns
    -------
    float or ndarray
        Duration [sidereal hours]; 0 if the target never rises, 24 if it never sets.

    References
    ----------
    [ERFA2023] hd2ae.c.

    Validity
    --------
    Refraction and the angular size of the target are not taken into account.

    Verification
    ------------
    tests/oracle/test_planning_oracle.py: every star is up for 12 hours at the equator; circumpolar stars for 24 hours.
    """
    d = np.asarray(dec_rad, dtype=float)
    cosine = (np.sin(altitude_limit_rad) - np.sin(latitude_rad) * np.sin(d)) / (np.cos(latitude_rad) * np.cos(d))
    return 2.0 * np.arccos(np.clip(cosine, -1.0, 1.0)) / TWO_PI * 24.0
