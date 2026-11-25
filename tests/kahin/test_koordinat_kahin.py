import math

import erfa
import numpy as np
import pytest

from kozmos.koordinat import donusum
from kozmos.temel.aci import acisal_ayriklik

pytestmark = pytest.mark.kahin


def _gokyuzu(n, tohum):
    u = np.random.default_rng(tohum)
    return u.uniform(0, 2 * np.pi, n), np.arcsin(u.uniform(-1, 1, n))


def test_galaktik_erfa_ile():
    ra, dec = _gokyuzu(3000, 41)
    boylam, enlem = donusum.icrs_den_galaktik(ra, dec)
    eb, ee = erfa.icrs2g(ra, dec)
    assert np.max(acisal_ayriklik(boylam, enlem, eb, ee)) < 1e-14
    ra2, dec2 = donusum.galaktik_den_icrs(boylam, enlem)
    assert np.max(acisal_ayriklik(ra, dec, ra2, dec2)) < 1e-14
    er, ed = erfa.g2icrs(boylam, enlem)
    assert np.max(acisal_ayriklik(ra2, dec2, er, ed)) < 1e-14


def test_galaktik_kutup_ve_matris():
    # icrs2g.c: kuzey galaktik kutup α = 192.85948°, δ = 27.12825°.
    _, enlem = donusum.icrs_den_galaktik(math.radians(192.85948), math.radians(27.12825))
    assert float(enlem) == pytest.approx(math.pi / 2, abs=1e-7)
    m = donusum.ICRS_DEN_GALAKTIK
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-15) and np.linalg.det(m) == pytest.approx(1.0, abs=1e-15)


def test_ufuk_erfa_ile():
    u = np.random.default_rng(42)
    h, dec = _gokyuzu(3000, 43)
    h = h - np.pi
    enlem = np.arcsin(u.uniform(-1, 1, 3000))
    az, yuk = donusum.saat_acisi_den_ufuk(h, dec, enlem)
    eaz, eyuk = erfa.hd2ae(h, dec, enlem)
    assert np.max(acisal_ayriklik(az, yuk, eaz, eyuk)) < 1e-13
    h2, dec2 = donusum.ufuk_den_saat_acisi(az, yuk, enlem)
    eh, ed = erfa.ae2hd(az, yuk, enlem)
    assert np.max(acisal_ayriklik(h2, dec2, eh, ed)) < 1e-13
    assert np.max(acisal_ayriklik(h, dec, h2, dec2)) < 1e-13


def test_ufuk_bilinen_durumlar():
    enlem = math.radians(40.0)
    # Meridyende (H = 0) yükseklik 90° − |φ − δ|; güneyde kalan yıldızın azimutu 180°.
    az, yuk = donusum.saat_acisi_den_ufuk(0.0, math.radians(10.0), enlem)
    assert float(yuk) == pytest.approx(math.radians(60.0), abs=1e-14)
    assert float(az) == pytest.approx(math.pi, abs=1e-12)
    # Gök kutbunun yüksekliği enleme eşittir, azimutu kuzeydir.
    az, yuk = donusum.saat_acisi_den_ufuk(1.0, math.pi / 2, enlem)
    assert float(yuk) == pytest.approx(enlem, abs=1e-14)


def test_hava_kutlesi():
    assert float(donusum.hava_kutlesi_duzlem_paralel(math.pi / 2)) == 1.0
    assert float(donusum.hava_kutlesi_duzlem_paralel(math.radians(30.0))) == pytest.approx(2.0)
    assert np.isnan(donusum.hava_kutlesi_duzlem_paralel(-0.1))
