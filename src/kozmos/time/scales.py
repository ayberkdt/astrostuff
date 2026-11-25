"""Differences between time scales: TAI − UTC and derived quantities.

The leap-second table and the pre-1972 drift coefficients were read from the
ERFA file ``dat.c`` [ERFA2023]. The table must be updated when a new leap
second is announced; ``TABLE_RELEASE_YEAR`` is the ERFA release year the table
belongs to.
"""

from __future__ import annotations

import warnings

from kozmos._sources import source
from kozmos.time.julian import TT_MINUS_TAI_S, calendar_to_mjd

TABLE_RELEASE_YEAR = 2023  # dat.c IYV

# (year, month, TAI − UTC [s]); the _changes table in dat.c.
_CHANGES: tuple[tuple[int, int, float], ...] = (
    (1960, 1, 1.4178180),
    (1961, 1, 1.4228180),
    (1961, 8, 1.3728180),
    (1962, 1, 1.8458580),
    (1963, 11, 1.9458580),
    (1964, 1, 3.2401300),
    (1964, 4, 3.3401300),
    (1964, 9, 3.4401300),
    (1965, 1, 3.5401300),
    (1965, 3, 3.6401300),
    (1965, 7, 3.7401300),
    (1965, 9, 3.8401300),
    (1966, 1, 4.3131700),
    (1968, 2, 4.2131700),
    (1972, 1, 10.0),
    (1972, 7, 11.0),
    (1973, 1, 12.0),
    (1974, 1, 13.0),
    (1975, 1, 14.0),
    (1976, 1, 15.0),
    (1977, 1, 16.0),
    (1978, 1, 17.0),
    (1979, 1, 18.0),
    (1980, 1, 19.0),
    (1981, 7, 20.0),
    (1982, 7, 21.0),
    (1983, 7, 22.0),
    (1985, 7, 23.0),
    (1988, 1, 24.0),
    (1990, 1, 25.0),
    (1991, 1, 26.0),
    (1992, 7, 27.0),
    (1993, 7, 28.0),
    (1994, 7, 29.0),
    (1996, 1, 30.0),
    (1997, 7, 31.0),
    (1999, 1, 32.0),
    (2006, 1, 33.0),
    (2009, 1, 34.0),
    (2012, 7, 35.0),
    (2015, 7, 36.0),
    (2017, 1, 37.0),
)

# Before 1972: (reference MJD, drift [s/day]); the drift table in dat.c.
_DRIFT: tuple[tuple[float, float], ...] = (
    (37300.0, 0.0012960),
    (37300.0, 0.0012960),
    (37300.0, 0.0012960),
    (37665.0, 0.0011232),
    (37665.0, 0.0011232),
    (38761.0, 0.0012960),
    (38761.0, 0.0012960),
    (38761.0, 0.0012960),
    (38761.0, 0.0012960),
    (38761.0, 0.0012960),
    (38761.0, 0.0012960),
    (38761.0, 0.0012960),
    (39126.0, 0.0025920),
    (39126.0, 0.0025920),
)


@source(
    "ERFA2023", where="dat.c", status="read", note="the table ends with the 2017-01-01 entry; ERFA release year 2023"
)
def tai_minus_utc_s(year: int, month: int, day: int, day_fraction: float = 0.0) -> float:
    """Return the difference TAI − UTC (ΔAT) in seconds at the given UTC date.

    Parameters
    ----------
    year, month, day : int
        UTC calendar date (Gregorian).
    day_fraction : float
        Fraction of the day, [0, 1]; affects the result only in the pre-1972 drift era.

    Returns
    -------
    float
        TAI − UTC [s].

    References
    ----------
    [ERFA2023] dat.c.

    Validity
    --------
    1960 and later. A warning is issued for dates more than five years beyond the
    table release year: a new leap second may have been announced by then.
    The 86401st second within a day on which a leap second is inserted is not
    distinguished by this function.

    Verification
    ------------
    tests/oracle/test_time_scales_oracle.py: exact equality with erfa.dat.
    """
    if not 0.0 <= day_fraction <= 1.0:
        raise ValueError("day fraction must be in the range [0, 1]")
    mjd = int(calendar_to_mjd(year, month, day))
    if year < _CHANGES[0][0]:
        raise ValueError("UTC is not defined before 1960")
    if year > TABLE_RELEASE_YEAR + 5:
        warnings.warn(
            f"the leap-second table is the {TABLE_RELEASE_YEAR} release; verify it is current for {year}",
            stacklevel=2,
        )
    m = 12 * year + month
    for i in range(len(_CHANGES) - 1, -1, -1):
        if m >= 12 * _CHANGES[i][0] + _CHANGES[i][1]:
            break
    else:
        raise ValueError("date is before the start of the table")
    diff = _CHANGES[i][2]
    if i < len(_DRIFT):
        diff += (mjd + day_fraction - _DRIFT[i][0]) * _DRIFT[i][1]
    return diff


@source("ERFA2023", where="dat.c; erfam.h ERFA_TTMTAI", status="read", note="TT − UTC = (TAI − UTC) + 32.184 s")
def tt_minus_utc_s(year: int, month: int, day: int, day_fraction: float = 0.0) -> float:
    """Return the difference TT − UTC in seconds at the given UTC date.

    References
    ----------
    [ERFA2023] dat.c; erfam.h ERFA_TTMTAI.

    Verification
    ------------
    tests/oracle/test_time_scales_oracle.py
    """
    return tai_minus_utc_s(year, month, day, day_fraction) + TT_MINUS_TAI_S
