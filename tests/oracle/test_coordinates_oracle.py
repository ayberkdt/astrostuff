import math

import erfa
import numpy as np
import pytest

from kozmos.coordinates import transforms
from kozmos.core.angles import angular_separation

pytestmark = pytest.mark.oracle


def _gokyuzu(n, tohum):
    u = np.random.default_rng(tohum)
    return u.uniform(0, 2 * np.pi, n), np.arcsin(u.uniform(-1, 1, n))


def test_galaktik_erfa_ile():
    ra, dec = _gokyuzu(3000, 41)
    boylam, enlem = transforms.icrs_to_galactic(ra, dec)
    eb, ee = erfa.icrs2g(ra, dec)
    assert np.max(angular_separation(boylam, enlem, eb, ee)) < 1e-14
    ra2, dec2 = transforms.galactic_to_icrs(boylam, enlem)
    assert np.max(angular_separation(ra, dec, ra2, dec2)) < 1e-14
    er, ed = erfa.g2icrs(boylam, enlem)
    assert np.max(angular_separation(ra2, dec2, er, ed)) < 1e-14


def test_galaktik_kutup_ve_matris():
    # icrs2g.c: kuzey galaktik kutup α = 192.85948°, δ = 27.12825°.
    _, enlem = transforms.icrs_to_galactic(math.radians(192.85948), math.radians(27.12825))
    assert float(enlem) == pytest.approx(math.pi / 2, abs=1e-7)
    m = transforms.ICRS_TO_GALACTIC
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-15) and np.linalg.det(m) == pytest.approx(1.0, abs=1e-15)


def test_ufuk_erfa_ile():
    u = np.random.default_rng(42)
    h, dec = _gokyuzu(3000, 43)
    h = h - np.pi
    enlem = np.arcsin(u.uniform(-1, 1, 3000))
    az, yuk = transforms.hour_angle_to_horizontal(h, dec, enlem)
    eaz, eyuk = erfa.hd2ae(h, dec, enlem)
    assert np.max(angular_separation(az, yuk, eaz, eyuk)) < 1e-13
    h2, dec2 = transforms.horizontal_to_hour_angle(az, yuk, enlem)
    eh, ed = erfa.ae2hd(az, yuk, enlem)
    assert np.max(angular_separation(h2, dec2, eh, ed)) < 1e-13
    assert np.max(angular_separation(h, dec, h2, dec2)) < 1e-13


def test_ufuk_bilinen_durumlar():
    enlem = math.radians(40.0)
    # Meridyende (H = 0) yükseklik 90° − |φ − δ|; güneyde kalan yıldızın azimutu 180°.
    az, yuk = transforms.hour_angle_to_horizontal(0.0, math.radians(10.0), enlem)
    assert float(yuk) == pytest.approx(math.radians(60.0), abs=1e-14)
    assert float(az) == pytest.approx(math.pi, abs=1e-12)
    # Gök kutbunun yüksekliği enleme eşittir, azimutu kuzeydir.
    az, yuk = transforms.hour_angle_to_horizontal(1.0, math.pi / 2, enlem)
    assert float(yuk) == pytest.approx(enlem, abs=1e-14)


def test_hava_kutlesi():
    assert float(transforms.airmass_plane_parallel(math.pi / 2)) == 1.0
    assert float(transforms.airmass_plane_parallel(math.radians(30.0))) == pytest.approx(2.0)
    assert np.isnan(transforms.airmass_plane_parallel(-0.1))
