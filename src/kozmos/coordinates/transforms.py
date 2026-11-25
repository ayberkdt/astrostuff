"""Transformations between coordinate systems.

The Galactic transformation matrix and the horizon transformation were read
from the ERFA source files [ERFA2023]: ``icrs2g.c``, ``hd2ae.c``, ``ae2hd.c``.
The definition of the Galactic system is the one in the Hipparcos catalogue;
ERFA derives the matrix from this definition at full precision and cites
ESA SP-1200 as the catalogue source.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.angles import spherical_to_unit_vector, unit_vector_to_spherical, wrap_0_2pi

Array = NDArray[np.float64]

# ICRS to Galactic rotation matrix; the 30-digit values in icrs2g.c.
ICRS_TO_GALACTIC = np.array(
    [
        [-0.054875560416215368492398900454, -0.873437090234885048760383168409, -0.483835015548713226831774175116],
        [+0.494109427875583673525222371358, -0.444829629960011178146614061616, +0.746982244497218890527388004556],
        [-0.867666149019004701181616534570, -0.198076373431201528180486091412, +0.455983776175066922272100478348],
    ]
)


@source(
    "ERFA2023",
    where="icrs2g.c",
    status="read",
    note="north Galactic pole α = 192.85948°, δ = 27.12825°; longitude of the ascending node 32.93192° (Hipparcos definition)",
)
def icrs_to_galactic(ra_rad: ArrayLike, dec_rad: ArrayLike) -> tuple[Array, Array]:
    """Convert ICRS right ascension and declination to Galactic longitude and latitude.

    Parameters
    ----------
    ra_rad, dec_rad : float or ndarray
        ICRS coordinates [rad].

    Returns
    -------
    (l, b) : Galactic longitude [0, 2π) and latitude [−π/2, π/2] [rad].

    References
    ----------
    [ERFA2023] icrs2g.c.

    Validity
    --------
    The rotation matrix is constant; no epoch or proper-motion correction is applied.

    Verification
    ------------
    tests/oracle/test_coordinates_oracle.py: difference from erfa.icrs2g < 1e-14 rad.
    """
    v = spherical_to_unit_vector(ra_rad, dec_rad)
    return unit_vector_to_spherical(v @ ICRS_TO_GALACTIC.T)


@source(
    "ERFA2023",
    where="icrs2g.c",
    status="read",
    note="the inverse transformation is the transpose of the matrix; g2icrs.c was not read separately",
)
def galactic_to_icrs(longitude_rad: ArrayLike, latitude_rad: ArrayLike) -> tuple[Array, Array]:
    """Convert Galactic longitude and latitude to ICRS right ascension and declination.

    References
    ----------
    [ERFA2023] icrs2g.c.

    Verification
    ------------
    tests/oracle/test_coordinates_oracle.py: difference from erfa.g2icrs < 1e-14 rad; round trip.
    """
    v = spherical_to_unit_vector(longitude_rad, latitude_rad)
    return unit_vector_to_spherical(v @ ICRS_TO_GALACTIC)


@source(
    "ERFA2023",
    where="hd2ae.c",
    status="read",
    note="azimuth from north through east; unit-vector components are x, y, z in hd2ae.c",
)
def hour_angle_to_horizontal(
    hour_angle_rad: ArrayLike, dec_rad: ArrayLike, latitude_rad: ArrayLike
) -> tuple[Array, Array]:
    """Convert hour angle and declination to azimuth and altitude.

    Parameters
    ----------
    hour_angle_rad : float or ndarray
        Local hour angle H = local sidereal time − right ascension [rad].
    dec_rad : float or ndarray
        Declination [rad].
    latitude_rad : float or ndarray
        Observer latitude [rad].

    Returns
    -------
    (azimuth, altitude) : azimuth [0, 2π), from north through east; altitude [−π/2, π/2] [rad].

    References
    ----------
    [ERFA2023] hd2ae.c.

    Validity
    --------
    A pure geometric rotation: no refraction, aberration or diurnal parallax.
    At the zenith the azimuth is undefined and 0 is returned.

    Verification
    ------------
    tests/oracle/test_coordinates_oracle.py: difference from erfa.hd2ae < 1e-13 rad.
    """
    h = np.asarray(hour_angle_rad, dtype=float)
    d = np.asarray(dec_rad, dtype=float)
    p = np.asarray(latitude_rad, dtype=float)
    x = -np.cos(h) * np.cos(d) * np.sin(p) + np.sin(d) * np.cos(p)
    y = -np.sin(h) * np.cos(d)
    z = np.cos(h) * np.cos(d) * np.cos(p) + np.sin(d) * np.sin(p)
    r = np.hypot(x, y)
    azimuth = np.where(r != 0.0, np.arctan2(y, x), 0.0)
    return wrap_0_2pi(azimuth), np.arctan2(z, r)


@source("ERFA2023", where="ae2hd.c", status="read")
def horizontal_to_hour_angle(
    azimuth_rad: ArrayLike, altitude_rad: ArrayLike, latitude_rad: ArrayLike
) -> tuple[Array, Array]:
    """Convert azimuth and altitude to hour angle and declination.

    Returns
    -------
    (hour angle, declination) : hour angle (−π, π] [rad].

    References
    ----------
    [ERFA2023] ae2hd.c.

    Verification
    ------------
    tests/oracle/test_coordinates_oracle.py: difference from erfa.ae2hd < 1e-13 rad; round trip.
    """
    a = np.asarray(azimuth_rad, dtype=float)
    e = np.asarray(altitude_rad, dtype=float)
    p = np.asarray(latitude_rad, dtype=float)
    x = -np.cos(a) * np.cos(e) * np.sin(p) + np.sin(e) * np.cos(p)
    y = -np.sin(a) * np.cos(e)
    z = np.cos(a) * np.cos(e) * np.cos(p) + np.sin(e) * np.sin(p)
    r = np.hypot(x, y)
    return np.where(r != 0.0, np.arctan2(y, x), 0.0), np.arctan2(z, r)


@source(
    "ERFA2023",
    where="hd2ae.c (z component)",
    status="read",
    note="the secant of the zenith distance is only the plane-parallel atmosphere approximation; the source of the approximation, [KastenYoung1989], was not read",
)
def airmass_plane_parallel(altitude_rad: ArrayLike) -> Array:
    """Compute the airmass in the plane-parallel atmosphere approximation: X = 1 / sin(h).

    References
    ----------
    [ERFA2023] hd2ae.c (definition of altitude).

    Validity
    --------
    Above 30° altitude an accuracy of a few parts per thousand is expected (not tested);
    near the horizon, spherical-atmosphere formulas are needed. Below the horizon ``nan`` is returned.

    Verification
    ------------
    tests/oracle/test_coordinates_oracle.py: 1 at the zenith, 2 at 30° altitude.
    """
    h = np.asarray(altitude_rad, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(h > 0.0, 1.0 / np.sin(h), np.nan)
