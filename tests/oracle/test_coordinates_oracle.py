import math

import erfa
import numpy as np
import pytest

from kozmos.coordinates import transforms
from kozmos.core.angles import angular_separation

pytestmark = pytest.mark.oracle


def _sky(n, seed):
    u = np.random.default_rng(seed)
    return u.uniform(0, 2 * np.pi, n), np.arcsin(u.uniform(-1, 1, n))


def test_galactic_vs_erfa():
    ra, dec = _sky(3000, 41)
    longitude, latitude = transforms.icrs_to_galactic(ra, dec)
    e_lon, e_lat = erfa.icrs2g(ra, dec)
    assert np.max(angular_separation(longitude, latitude, e_lon, e_lat)) < 1e-14
    ra2, dec2 = transforms.galactic_to_icrs(longitude, latitude)
    assert np.max(angular_separation(ra, dec, ra2, dec2)) < 1e-14
    e_ra, e_dec = erfa.g2icrs(longitude, latitude)
    assert np.max(angular_separation(ra2, dec2, e_ra, e_dec)) < 1e-14


def test_galactic_pole_and_matrix():
    # icrs2g.c: north Galactic pole α = 192.85948°, δ = 27.12825°.
    _, latitude = transforms.icrs_to_galactic(math.radians(192.85948), math.radians(27.12825))
    assert float(latitude) == pytest.approx(math.pi / 2, abs=1e-7)
    m = transforms.ICRS_TO_GALACTIC
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-15) and np.linalg.det(m) == pytest.approx(1.0, abs=1e-15)


def test_horizon_vs_erfa():
    u = np.random.default_rng(42)
    h, dec = _sky(3000, 43)
    h = h - np.pi
    latitude = np.arcsin(u.uniform(-1, 1, 3000))
    az, altitude = transforms.hour_angle_to_horizontal(h, dec, latitude)
    eaz, ealt = erfa.hd2ae(h, dec, latitude)
    assert np.max(angular_separation(az, altitude, eaz, ealt)) < 1e-13
    h2, dec2 = transforms.horizontal_to_hour_angle(az, altitude, latitude)
    e_h, e_dec = erfa.ae2hd(az, altitude, latitude)
    assert np.max(angular_separation(h2, dec2, e_h, e_dec)) < 1e-13
    assert np.max(angular_separation(h, dec, h2, dec2)) < 1e-13


def test_horizon_known_cases():
    latitude = math.radians(40.0)
    # On the meridian (H = 0) the altitude is 90° − |φ − δ|; the azimuth of a star south of the zenith is 180°.
    az, altitude = transforms.hour_angle_to_horizontal(0.0, math.radians(10.0), latitude)
    assert float(altitude) == pytest.approx(math.radians(60.0), abs=1e-14)
    assert float(az) == pytest.approx(math.pi, abs=1e-12)
    # The altitude of the celestial pole equals the latitude, and its azimuth is north.
    az, altitude = transforms.hour_angle_to_horizontal(1.0, math.pi / 2, latitude)
    assert float(altitude) == pytest.approx(latitude, abs=1e-14)


def test_airmass():
    assert float(transforms.airmass_plane_parallel(math.pi / 2)) == 1.0
    assert float(transforms.airmass_plane_parallel(math.radians(30.0))) == pytest.approx(2.0, rel=1e-6, abs=0)
    assert np.isnan(transforms.airmass_plane_parallel(-0.1))
