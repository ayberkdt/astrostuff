"""Calendar, Julian date, Earth rotation angle and sidereal time.

The algorithms and coefficients were read from the ERFA source files [ERFA2023]:
``cal2jd.c``, ``jd2cal.c``, ``era00.c``, ``gmst06.c``, ``obl06.c``.

The Julian date is kept in two parts (``jd1 + jd2``); their sum in a single
``float64`` gives a resolution of about 40 microseconds near J2000.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.angles import ARCSEC_RAD, TWO_PI, wrap_0_2pi

J2000_JD = 2451545.0  # erfam.h ERFA_DJ00
MJD_ZERO = 2400000.5  # erfam.h ERFA_DJM0
JULIAN_CENTURY = 36525.0  # erfam.h ERFA_DJC
TT_MINUS_TAI_S = 32.184  # erfam.h ERFA_TTMTAI
DAY_S = 86400.0  # erfam.h ERFA_DAYSEC

Array = NDArray[np.float64]
_MONTH_LENGTHS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


@source(
    "ERFA2023",
    where="cal2jd.c",
    status="read",
    note="Gregorian calendar; the integer division (im-14)/12 of C, which truncates toward zero, is applied explicitly",
)
def calendar_to_mjd(year: ArrayLike, month: ArrayLike, day: ArrayLike) -> NDArray[np.int64]:
    """Return the Modified Julian Date at 0h of a Gregorian calendar date.

    Parameters
    ----------
    year, month, day : int or integer array
        Date in the proleptic Gregorian calendar; year >= −4799.

    Returns
    -------
    int or integer array
        MJD (0h). The Julian date is ``2400000.5 + MJD``.

    References
    ----------
    [ERFA2023] cal2jd.c.

    Validity
    --------
    Year >= −4799. The Julian calendar (civil use before 1582) is not supported;
    all dates are interpreted as proleptic Gregorian.

    Verification
    ------------
    tests/oracle/test_time_oracle.py: exact equality with erfa.cal2jd.
    """
    iy = np.asarray(year, dtype=np.int64)
    im = np.asarray(month, dtype=np.int64)
    id_ = np.asarray(day, dtype=np.int64)
    if np.any(iy < -4799):
        raise ValueError("year cannot be less than -4799")
    if np.any((im < 1) | (im > 12)):
        raise ValueError("month must be in the range 1-12")
    residual = (im == 2) & (iy % 4 == 0) & ((iy % 100 != 0) | (iy % 400 == 0))
    if np.any((id_ < 1) | (id_ > _MONTH_LENGTHS[im - 1] + residual)):
        raise ValueError("day is invalid for that month")
    my = np.where(im <= 2, -1, 0)  # C: (im - 14) / 12, truncated toward zero
    iypmy = iy + my
    return (
        (1461 * (iypmy + 4800)) // 4
        + (367 * (im - 2 - 12 * my)) // 12
        - (3 * ((iypmy + 4900) // 100)) // 4
        + id_
        - 2432076
    )


@source("ERFA2023", where="cal2jd.c; erfam.h ERFA_DJM0", status="read")
def calendar_to_jd(year: ArrayLike, month: ArrayLike, day: ArrayLike, day_fraction: ArrayLike = 0.0) -> Array:
    """Return the Julian date of a Gregorian calendar date.

    Parameters
    ----------
    year, month, day : int
        Calendar date.
    day_fraction : float
        Fraction of the day since midnight, [0, 1).

    Returns
    -------
    float or ndarray
        Julian date (a single number). For high precision keep the two parts
        separate: ``2400000.5`` and ``calendar_to_mjd(...) + day_fraction``.

    References
    ----------
    [ERFA2023] cal2jd.c; erfam.h ERFA_DJM0.

    Verification
    ------------
    tests/oracle/test_time_oracle.py
    """
    return MJD_ZERO + calendar_to_mjd(year, month, day) + np.asarray(day_fraction, dtype=float)


@source(
    "ERFA2023",
    where="jd2cal.c",
    status="read",
    note="calendar conversion as in the source; the balanced summation of the day fraction is simplified",
)
def jd_to_calendar(
    jd1: ArrayLike, jd2: ArrayLike = 0.0
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], Array]:
    """Return the Gregorian calendar date from a two-part Julian date.

    Returns
    -------
    (year, month, day, day_fraction)

    References
    ----------
    [ERFA2023] jd2cal.c.

    Validity
    --------
    −68569.5 <= JD <= 1e9. The day fraction is summed from the fractions of the two parts;
    because the balanced summation of ERFA is not applied, the fraction can differ in the last digit.

    Verification
    ------------
    tests/oracle/test_time_oracle.py: exact equality of the calendar fields with erfa.jd2cal.
    """
    a = np.asarray(jd1, dtype=float)
    b = np.asarray(jd2, dtype=float)
    if np.any((a + b < -68569.5) | (a + b > 1e9)):
        raise ValueError("Julian date is outside the supported range")
    da, db = np.rint(a), np.rint(b)
    f = (a - da) + (b - db) + 0.5
    carry = np.floor(f)
    jd = (da + db + carry).astype(np.int64)
    f = f - carry
    ell = jd + 68569
    n = (4 * ell) // 146097
    ell = ell - (146097 * n + 3) // 4
    i = (4000 * (ell + 1)) // 1461001
    ell = ell - (1461 * i) // 4 + 31
    k = (80 * ell) // 2447
    day = ell - (2447 * k) // 80
    ell = k // 11
    month = k + 2 - 12 * ell
    year = 100 * (n - 49) + i + ell
    return year, month, day, f


@source("ERFA2023", where="erfam.h ERFA_TTMTAI, ERFA_DAYSEC", status="read")
def tai_to_tt(jd1: ArrayLike, jd2: ArrayLike = 0.0) -> tuple[Array, Array]:
    """Convert TAI to TT: TT = TAI + 32.184 s.

    References
    ----------
    [ERFA2023] erfam.h ERFA_TTMTAI.

    Verification
    ------------
    tests/oracle/test_time_oracle.py: erfa.taitt.
    """
    return np.asarray(jd1, dtype=float), np.asarray(jd2, dtype=float) + TT_MINUS_TAI_S / DAY_S


@source("ERFA2023", where="erfam.h ERFA_TTMTAI, ERFA_DAYSEC", status="read")
def tt_to_tai(jd1: ArrayLike, jd2: ArrayLike = 0.0) -> tuple[Array, Array]:
    """Convert TT to TAI: TAI = TT − 32.184 s.

    References
    ----------
    [ERFA2023] erfam.h ERFA_TTMTAI.

    Verification
    ------------
    tests/oracle/test_time_oracle.py: erfa.tttai.
    """
    return np.asarray(jd1, dtype=float), np.asarray(jd2, dtype=float) - TT_MINUS_TAI_S / DAY_S


@source(
    "ERFA2023",
    where="era00.c",
    status="read",
    note="θ = 2π (f + 0.7790572732640 + 0.00273781191135448 t); t = UT1 − J2000 [day]",
)
def earth_rotation_angle(ut1_jd1: ArrayLike, ut1_jd2: ArrayLike = 0.0) -> Array:
    """Compute the Earth Rotation Angle (ERA, IAU 2000).

    Parameters
    ----------
    ut1_jd1, ut1_jd2 : float or ndarray
        Two-part Julian date on the UT1 scale.

    Returns
    -------
    float or ndarray
        ERA [rad], [0, 2π).

    References
    ----------
    [ERFA2023] era00.c.

    Verification
    ------------
    tests/oracle/test_time_oracle.py: difference from erfa.era00 < 1e-12 rad.
    """
    a = np.asarray(ut1_jd1, dtype=float)
    b = np.asarray(ut1_jd2, dtype=float)
    d1, d2 = np.minimum(a, b), np.maximum(a, b)
    t = d1 + (d2 - J2000_JD)
    f = np.fmod(d1, 1.0) + np.fmod(d2, 1.0)
    return wrap_0_2pi(TWO_PI * (f + 0.7790572732640 + 0.00273781191135448 * t))


@source(
    "ERFA2023", where="gmst06.c", status="read", note="ERA + a degree-5 polynomial in TT; coefficients in arcseconds"
)
def greenwich_mean_sidereal_time(ut1_jd1: ArrayLike, ut1_jd2: ArrayLike, tt_jd1: ArrayLike, tt_jd2: ArrayLike) -> Array:
    """Compute Greenwich mean sidereal time (consistent with IAU 2006 precession).

    Returns
    -------
    float or ndarray
        GMST [rad], [0, 2π).

    References
    ----------
    [ERFA2023] gmst06.c.

    Validity
    --------
    UT1 and TT are given separately; if the same value is given for both, the error is
    of the order of a few microarcseconds (ERFA documentation note; not measured in this work).

    Verification
    ------------
    tests/oracle/test_time_oracle.py: difference from erfa.gmst06 < 1e-12 rad.
    """
    t = ((np.asarray(tt_jd1, dtype=float) - J2000_JD) + np.asarray(tt_jd2, dtype=float)) / JULIAN_CENTURY
    polynomial = (
        0.014506 + (4612.156534 + (1.3915817 + (-0.00000044 + (-0.000029956 + (-0.0000000368) * t) * t) * t) * t) * t
    )
    return wrap_0_2pi(earth_rotation_angle(ut1_jd1, ut1_jd2) + polynomial * ARCSEC_RAD)


@source("ERFA2023", where="obl06.c", status="read", note="ε0 = 84381.406″ and a degree-5 polynomial")
def mean_obliquity(tt_jd1: ArrayLike, tt_jd2: ArrayLike = 0.0) -> Array:
    """Compute the mean obliquity of the ecliptic (IAU 2006).

    Returns
    -------
    float or ndarray
        Mean obliquity [rad].

    References
    ----------
    [ERFA2023] obl06.c.

    Verification
    ------------
    tests/oracle/test_time_oracle.py: difference from erfa.obl06 < 1e-15 rad.
    """
    t = ((np.asarray(tt_jd1, dtype=float) - J2000_JD) + np.asarray(tt_jd2, dtype=float)) / JULIAN_CENTURY
    eps = 84381.406 + (-46.836769 + (-0.0001831 + (0.00200340 + (-0.000000576 + (-0.0000000434) * t) * t) * t) * t) * t
    return eps * ARCSEC_RAD


@source("ERFA2023", where="gmst06.c", status="read")
@source(
    "Astropy2022",
    where="time/core.py, Time._sid_time_or_earth_rot_ang",
    status="read",
    note="angle = longitude + θ (east longitude positive); polar motion and the TIO position are not included",
)
def local_mean_sidereal_time(
    ut1_jd1: ArrayLike, ut1_jd2: ArrayLike, tt_jd1: ArrayLike, tt_jd2: ArrayLike, east_longitude_rad: ArrayLike
) -> Array:
    """Compute local mean sidereal time.

    References
    ----------
    [ERFA2023] gmst06.c; [Astropy2022] time/core.py.

    Validity
    --------
    The equation of the equinoxes (nutation) is not added; the difference from apparent
    sidereal time reaches about 1 second.

    Verification
    ------------
    tests/oracle/test_time_oracle.py
    """
    gmst = greenwich_mean_sidereal_time(ut1_jd1, ut1_jd2, tt_jd1, tt_jd2)
    return wrap_0_2pi(gmst + np.asarray(east_longitude_rad, dtype=float))
