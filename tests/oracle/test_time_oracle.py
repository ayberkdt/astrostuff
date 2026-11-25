import erfa
import numpy as np
import pytest

from kozmos.time import julian

pytestmark = pytest.mark.oracle


def _tarihler(n=3000, tohum=4):
    u = np.random.default_rng(tohum)
    return u.integers(-4000, 4000, n), u.integers(1, 13, n), u.integers(1, 29, n)


def _aci_farki(x, y):
    return np.max(np.abs(np.angle(np.exp(1j * (np.asarray(x) - np.asarray(y))))))


def test_takvim_den_mjd_erfa_ile():
    y, a, g = _tarihler()
    _, mjd = erfa.cal2jd(y, a, g)
    assert np.array_equal(julian.calendar_to_mjd(y, a, g), mjd.astype(np.int64))
    # Artık yıl ve ay sonu sınırları.
    for tarih in [(2000, 2, 29), (1900, 3, 1), (2024, 12, 31), (-4799, 1, 1), (1582, 10, 15)]:
        assert int(julian.calendar_to_mjd(*tarih)) == int(erfa.cal2jd(*tarih)[1])


def test_j2000():
    assert float(julian.calendar_to_jd(2000, 1, 1, 0.5)) == 2451545.0


def test_gecersiz_tarih():
    for tarih in [(2023, 2, 29), (2023, 13, 1), (2023, 4, 31), (-5000, 1, 1)]:
        with pytest.raises(ValueError):
            julian.calendar_to_mjd(*tarih)


def test_jd_den_takvim_erfa_ile():
    u = np.random.default_rng(5)
    jd1 = np.full(3000, 2400000.5)
    jd2 = u.uniform(-600000, 400000, 3000)
    y, a, g, f = julian.jd_to_calendar(jd1, jd2)
    ye, ae, ge, fe = erfa.jd2cal(jd1, jd2)
    assert np.array_equal(y, ye) and np.array_equal(a, ae) and np.array_equal(g, ge)
    assert np.max(np.abs(f - fe)) < 1e-9


def test_takvim_gidis_donus():
    y, a, g = _tarihler(tohum=6)
    y2, a2, g2, f = julian.jd_to_calendar(julian.MJD_ZERO, julian.calendar_to_mjd(y, a, g).astype(float))
    assert np.array_equal(y, y2) and np.array_equal(a, a2) and np.array_equal(g, g2)
    assert np.all(f == 0.0)


def test_tai_tt_erfa_ile():
    a, b = julian.tai_to_tt(2451545.0, 0.25)
    ea, eb = erfa.taitt(2451545.0, 0.25)
    assert a == ea and abs(b - eb) < 1e-15
    _, b = julian.tt_to_tai(*julian.tai_to_tt(2451545.0, 0.25))
    assert abs(b - 0.25) < 1e-15


def test_donme_acisi_yildiz_zamani_egiklik_erfa_ile():
    u = np.random.default_rng(7)
    jd1 = np.full(2000, 2400000.5)
    jd2 = u.uniform(30000, 80000, 2000)
    tt2 = jd2 + 69.0 / 86400.0
    assert _aci_farki(julian.earth_rotation_angle(jd1, jd2), erfa.era00(jd1, jd2)) < 1e-12
    gmst = julian.greenwich_mean_sidereal_time(jd1, jd2, jd1, tt2)
    assert _aci_farki(gmst, erfa.gmst06(jd1, jd2, jd1, tt2)) < 1e-12
    assert np.max(np.abs(julian.mean_obliquity(jd1, tt2) - erfa.obl06(jd1, tt2))) < 1e-15
    yerel = julian.local_mean_sidereal_time(jd1, jd2, jd1, tt2, 0.5)
    assert _aci_farki(yerel, erfa.gmst06(jd1, jd2, jd1, tt2) + 0.5) < 1e-12
