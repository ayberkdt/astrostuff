import erfa
import numpy as np
import pytest

from kozmos.time import julian

pytestmark = pytest.mark.oracle


def _dates(n=3000, seed=4):
    u = np.random.default_rng(seed)
    return u.integers(-4000, 4000, n), u.integers(1, 13, n), u.integers(1, 29, n)


def _angle_difference(x, y):
    return np.max(np.abs(np.angle(np.exp(1j * (np.asarray(x) - np.asarray(y))))))


def test_calendar_to_mjd_vs_erfa():
    y, a, g = _dates()
    _, mjd = erfa.cal2jd(y, a, g)
    assert np.array_equal(julian.calendar_to_mjd(y, a, g), mjd.astype(np.int64))
    # Leap-year and month-end boundaries.
    for date in [(2000, 2, 29), (1900, 3, 1), (2024, 12, 31), (-4799, 1, 1), (1582, 10, 15)]:
        assert int(julian.calendar_to_mjd(*date)) == int(erfa.cal2jd(*date)[1])


def test_j2000():
    assert float(julian.calendar_to_jd(2000, 1, 1, 0.5)) == 2451545.0


def test_invalid_date():
    for date in [(2023, 2, 29), (2023, 13, 1), (2023, 4, 31), (-5000, 1, 1)]:
        with pytest.raises(ValueError):
            julian.calendar_to_mjd(*date)


def test_jd_to_calendar_vs_erfa():
    u = np.random.default_rng(5)
    jd1 = np.full(3000, 2400000.5)
    jd2 = u.uniform(-600000, 400000, 3000)
    y, a, g, f = julian.jd_to_calendar(jd1, jd2)
    ye, ae, ge, fe = erfa.jd2cal(jd1, jd2)
    assert np.array_equal(y, ye) and np.array_equal(a, ae) and np.array_equal(g, ge)
    assert np.max(np.abs(f - fe)) < 1e-9


def test_calendar_round_trip():
    y, a, g = _dates(seed=6)
    y2, a2, g2, f = julian.jd_to_calendar(julian.MJD_ZERO, julian.calendar_to_mjd(y, a, g).astype(float))
    assert np.array_equal(y, y2) and np.array_equal(a, a2) and np.array_equal(g, g2)
    assert np.all(f == 0.0)


def test_tai_tt_vs_erfa():
    a, b = julian.tai_to_tt(2451545.0, 0.25)
    ea, eb = erfa.taitt(2451545.0, 0.25)
    assert a == ea and abs(b - eb) < 1e-15
    _, b = julian.tt_to_tai(*julian.tai_to_tt(2451545.0, 0.25))
    assert abs(b - 0.25) < 1e-15


def test_rotation_angle_sidereal_time_obliquity_vs_erfa():
    u = np.random.default_rng(7)
    jd1 = np.full(2000, 2400000.5)
    jd2 = u.uniform(30000, 80000, 2000)
    tt2 = jd2 + 69.0 / 86400.0
    assert _angle_difference(julian.earth_rotation_angle(jd1, jd2), erfa.era00(jd1, jd2)) < 1e-12
    gmst = julian.greenwich_mean_sidereal_time(jd1, jd2, jd1, tt2)
    assert _angle_difference(gmst, erfa.gmst06(jd1, jd2, jd1, tt2)) < 1e-12
    assert np.max(np.abs(julian.mean_obliquity(jd1, tt2) - erfa.obl06(jd1, tt2))) < 1e-15
    local = julian.local_mean_sidereal_time(jd1, jd2, jd1, tt2, 0.5)
    assert _angle_difference(local, erfa.gmst06(jd1, jd2, jd1, tt2) + 0.5) < 1e-12
