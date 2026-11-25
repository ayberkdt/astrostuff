"""Comparison of precession and refraction with ERFA and astropy."""

import astropy.units as u
import erfa
import numpy as np
import pytest
from astropy.coordinates import PrecessedGeocentric, SkyCoord
from astropy.time import Time

from kozmos.coordinates import precession
from kozmos.observing import refraction

pytestmark = pytest.mark.oracle

DATES_JD = 2451545.0 + np.array([-36525.0, -3652.5, 0.0, 9826.5, 18262.5, 36525.0])  # 1900-2100


def test_fukushima_williams_and_matrix_vs_erfa():
    for jd in DATES_JD:
        ours = precession.fukushima_williams_angles(jd, 0.25)
        assert np.allclose(ours, erfa.pfw06(jd, 0.25), rtol=0, atol=1e-15)
        r = precession.bias_precession_matrix(jd, 0.25)
        assert np.max(np.abs(r - erfa.pmat06(jd, 0.25))) < 1e-14
        assert np.max(np.abs(r @ r.T - np.eye(3))) < 1e-14
    # Array input: the matrices are stacked.
    assert precession.bias_precession_matrix(DATES_JD).shape == (DATES_JD.size, 3, 3)
    # At J2000 the matrix is only the frame bias: its difference from the identity is at the 1e-7 level (about 23 milliarcseconds).
    j2000 = precession.bias_precession_matrix(2451545.0)
    assert 1e-8 < np.max(np.abs(j2000 - np.eye(3))) < 2e-7


def test_mean_of_date_vs_erfa_and_astropy():
    rs = np.random.default_rng(181)
    ra, dec = rs.uniform(0, 2 * np.pi, 200), np.arcsin(rs.uniform(-1, 1, 200))
    jd = 2461370.5
    a, d = precession.icrs_to_mean_of_date(ra, dec, jd)
    a_k, d_k = erfa.c2s(erfa.rxp(erfa.pmat06(jd, 0.0), erfa.s2c(ra, dec)))
    assert np.max(np.abs(np.angle(np.exp(1j * (a - a_k))))) < 1e-13 and np.max(np.abs(d - d_k)) < 1e-13
    # Over 26 years precession is about 0.36 degrees: the moved direction departs from the original by this much.
    separation = np.degrees(erfa.seps(ra, dec, a, d))
    assert 0.3 < np.median(separation) < 0.4
    # Astropy's mean-equator-of-date frame: the difference is limited by aberration (20.5 arcseconds).
    oracle = SkyCoord(ra * u.rad, dec * u.rad, frame="icrs").transform_to(
        PrecessedGeocentric(equinox=Time(jd, format="jd", scale="tt"), obstime=Time(jd, format="jd", scale="tt"))
    )
    diff = np.degrees(erfa.seps(a, d, oracle.ra.rad, oracle.dec.rad)) * 3600.0
    assert diff.max() < 25.0


def test_refraction_constants_vs_erfa():
    for conditions in [
        (1013.25, 10.0, 0.5, 0.55),
        (780.0, -5.0, 0.2, 2.2),
        (1005.0, 7.0, 0.8, 0.574),
        (600.0, 0.0, 0.0, 10.0),
    ]:
        assert refraction.refraction_constants(*conditions) == pytest.approx(erfa.refco(*conditions), rel=1e-13)
    assert refraction.refraction_constants(0.0, 10.0, 0.5) == (0.0, 0.0)
    with pytest.raises(ValueError):
        refraction.refraction_constants(1013.25, 10.0, 0.5, 1e4)


def test_refraction_source_table():
    # Table in the refco.c comments: 1005 mb, 280.15 K, 80% humidity, 5740 Å; the "eraRefco" column [arcsec].
    # The computation equals erfa.refco exactly but departs from the table by at most 0.015 arcsec (at 76° and 80°);
    # the source does not explain the difference.
    a, b = refraction.refraction_constants(1005.0, 7.0, 0.8, 0.574)
    table = {
        10: 10.27,
        20: 21.20,
        30: 33.61,
        40: 48.83,
        45: 58.18,
        50: 69.30,
        60: 100.54,
        70: 158.68,
        76: 229.43,
        80: 318.55,
    }
    for zd, expected in table.items():
        t = np.tan(np.radians(zd))
        assert np.degrees(a * t + b * t**3) * 3600.0 == pytest.approx(expected, abs=0.02)


def test_observed_altitude_round_trip():
    a, b = refraction.refraction_constants(1013.25, 10.0, 0.5)
    observed = np.radians(np.linspace(10.0, 89.0, 80))
    t = np.tan(np.pi / 2 - observed)
    topocentric = observed - (a * t + b * t**3)  # forward model: Z_topo = Z_obs + dZ
    assert np.max(np.abs(refraction.observed_altitude(topocentric, a, b) - observed)) < 1e-11
    assert np.isnan(refraction.observed_altitude(np.radians(2.0), a, b))
    assert float(refraction.observed_altitude(np.pi / 2, a, b)) == pytest.approx(np.pi / 2)
