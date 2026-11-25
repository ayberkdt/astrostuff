"""Angle arithmetic and angular separation on the sphere.

The internal representation is radians. The conversion constants are read
from the ERFA file ``erfam.h`` [ERFA2023].
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source

TWO_PI = 6.283185307179586476925287  # erfam.h ERFA_D2PI
DEG_RAD = 1.745329251994329576923691e-2  # erfam.h ERFA_DD2R
ARCSEC_RAD = 4.848136811095359935899141e-6  # erfam.h ERFA_DAS2R
HOUR_RAD = 15.0 * DEG_RAD

Array = NDArray[np.float64]


@source("ERFA2023", where="erfam.h ERFA_DD2R", status="read")
def deg_to_rad(angle_deg: ArrayLike) -> Array:
    """Convert degrees to radians.

    References
    ----------
    [ERFA2023] erfam.h, ERFA_DD2R.

    Verification
    ------------
    tests/unit/test_angles.py
    """
    return np.asarray(angle_deg, dtype=float) * DEG_RAD


@source("ERFA2023", where="erfam.h ERFA_DD2R", status="read")
def rad_to_deg(angle_rad: ArrayLike) -> Array:
    """Convert radians to degrees.

    References
    ----------
    [ERFA2023] erfam.h, ERFA_DD2R.

    Verification
    ------------
    tests/unit/test_angles.py
    """
    return np.asarray(angle_rad, dtype=float) / DEG_RAD


@source("ERFA2023", where="erfam.h ERFA_DAS2R", status="read")
def arcsec_to_rad(angle_arcsec: ArrayLike) -> Array:
    """Convert arcseconds to radians.

    References
    ----------
    [ERFA2023] erfam.h, ERFA_DAS2R.

    Verification
    ------------
    tests/unit/test_angles.py
    """
    return np.asarray(angle_arcsec, dtype=float) * ARCSEC_RAD


@source(
    "ERFA2023",
    where="anp.c",
    status="read",
    note="w = fmod(a, 2π); 2π is added if negative",
)
def wrap_0_2pi(angle_rad: ArrayLike) -> Array:
    """Reduce an angle to the range [0, 2π).

    References
    ----------
    [ERFA2023] erfam.h, ERFA_D2PI.

    Verification
    ------------
    tests/property/test_angles_property.py: the result lies in the range and differs from the input by a multiple of 2π.
    """
    w = np.mod(np.asarray(angle_rad, dtype=float), TWO_PI)
    # For a very small negative input the mod result can round to exactly 2π.
    return np.where(w >= TWO_PI, 0.0, w)


@source("ERFA2023", where="anpm.c", status="read", note="ERFA range is (−π, π]; here [−π, π)")
def wrap_pm_pi(angle_rad: ArrayLike) -> Array:
    """Reduce an angle to the range [−π, π).

    References
    ----------
    [ERFA2023] erfam.h, ERFA_D2PI.

    Verification
    ------------
    tests/property/test_angles_property.py
    """
    return wrap_0_2pi(np.asarray(angle_rad, dtype=float) + math.pi) - math.pi


@source("ERFA2023", where="s2c.c", status="read", note="spherical to Cartesian unit vector")
def spherical_to_unit_vector(longitude_rad: ArrayLike, latitude_rad: ArrayLike) -> Array:
    """Build a unit vector from longitude and latitude: (cos b cos l, cos b sin l, sin b).

    Returns
    -------
    ndarray, shape (..., 3)

    References
    ----------
    [ERFA2023] s2c.c.

    Verification
    ------------
    tests/property/test_angles_property.py: norm 1, round trip.
    """
    longitude = np.asarray(longitude_rad, dtype=float)
    latitude = np.asarray(latitude_rad, dtype=float)
    cb = np.cos(latitude)
    return np.stack(np.broadcast_arrays(cb * np.cos(longitude), cb * np.sin(longitude), np.sin(latitude)), axis=-1)


@source("ERFA2023", where="c2s.c", status="read")
def unit_vector_to_spherical(vector: ArrayLike) -> tuple[Array, Array]:
    """Return longitude [0, 2π) and latitude [−π/2, π/2] from a vector.

    References
    ----------
    [ERFA2023] c2s.c.

    Verification
    ------------
    tests/property/test_angles_property.py: round trip.
    """
    v = np.asarray(vector, dtype=float)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return wrap_0_2pi(np.arctan2(y, x)), np.arctan2(z, np.hypot(x, y))


@source(
    "ERFA2023",
    where="sepp.c",
    status="read",
    note="angle = atan2(|a×b|, a·b); no loss of precision at small angles, unlike the cosine formula",
)
def angular_separation(
    longitude1_rad: ArrayLike, latitude1_rad: ArrayLike, longitude2_rad: ArrayLike, latitude2_rad: ArrayLike
) -> Array:
    """Compute the angle between two directions on the sphere.

    θ = atan2(|a × b|, a · b); a and b are the unit vectors of the directions.

    Parameters
    ----------
    longitude1_rad, latitude1_rad, longitude2_rad, latitude2_rad : float or ndarray
        Longitude and latitude of the two directions [rad].

    Returns
    -------
    float or ndarray
        Separation [rad], [0, π].

    References
    ----------
    [ERFA2023] sepp.c.

    Validity
    --------
    Stable at all angles, including coincident and antipodal points.

    Verification
    ------------
    tests/oracle/test_angles_oracle.py: difference from erfa.seps < 1e-14 rad.
    """
    a = spherical_to_unit_vector(longitude1_rad, latitude1_rad)
    b = spherical_to_unit_vector(longitude2_rad, latitude2_rad)
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    dot = np.sum(a * b, axis=-1)
    return np.arctan2(cross, dot)
