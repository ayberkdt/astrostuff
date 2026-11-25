"""Comparison of the WCS transformation and catalog matching with astropy."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from kozmos.astrometry import matching, wcs

pytestmark = pytest.mark.oracle

HEADERS = [
    # SkyView style: CDELT, no rotation.
    {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 225.5,
        "CRPIX2": 225.5,
        "CRVAL1": 85.42,
        "CRVAL2": -1.9,
        "CDELT1": -0.0006666666666666668,
        "CDELT2": 0.0006666666666666668,
    },
    # CD matrix, rotated by 30°, high declination.
    {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 512.0,
        "CRPIX2": 400.5,
        "CRVAL1": 201.3,
        "CRVAL2": 71.2,
        "CD1_1": -2.4e-4,
        "CD1_2": 1.4e-4,
        "CD2_1": 1.4e-4,
        "CD2_2": 2.4e-4,
    },
    # PC + CDELT, right ascension at the 0/360 boundary, southern hemisphere.
    {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 100.0,
        "CRPIX2": 120.0,
        "CRVAL1": 359.98,
        "CRVAL2": -45.0,
        "CDELT1": -5e-4,
        "CDELT2": 5e-4,
        "PC1_1": 0.8,
        "PC1_2": -0.6,
        "PC2_1": 0.6,
        "PC2_2": 0.8,
    },
    # Old convention: CROTA2.
    {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 50.0,
        "CRPIX2": 60.0,
        "CRVAL1": 10.0,
        "CRVAL2": 20.0,
        "CDELT1": -1e-3,
        "CDELT2": 1e-3,
        "CROTA2": 25.0,
    },
]


@pytest.mark.parametrize("header", HEADERS)
def test_pixel_to_sky_against_astropy(header):
    ours = wcs.wcs_from_header(header)
    astropy_wcs = WCS(header)
    rs = np.random.default_rng(121)
    x, y = rs.uniform(-50, 900, 400), rs.uniform(-50, 900, 400)
    ra, dec = wcs.pixel_to_sky(ours, x, y)
    ra_k, dec_k = astropy_wcs.all_pix2world(x, y, 0)
    separation = SkyCoord(ra * u.deg, dec * u.deg).separation(SkyCoord(ra_k * u.deg, dec_k * u.deg)).deg
    assert separation.max() < 1e-9
    x2, y2 = wcs.sky_to_pixel(ours, ra_k, dec_k)
    xk, yk = astropy_wcs.all_world2pix(ra_k, dec_k, 0)
    assert np.max(np.abs(x2 - xk)) < 1e-7 and np.max(np.abs(y2 - yk)) < 1e-7
    assert np.max(np.abs(x2 - x)) < 1e-7 and np.max(np.abs(y2 - y)) < 1e-7  # round trip


def test_reference_pixel_and_errors():
    w = wcs.wcs_from_header(HEADERS[0])
    ra, dec = wcs.pixel_to_sky(w, 224.5, 224.5)  # CRPIX = 225.5 (FITS) ↔ zero-based 224.5
    assert float(ra) == pytest.approx(85.42, abs=1e-12) and float(dec) == pytest.approx(-1.9, abs=1e-12)
    # The opposite hemisphere is undefined in TAN.
    assert np.isnan(wcs.sky_to_pixel(w, 85.42 + 180.0, 1.9)[0])
    with pytest.raises(ValueError):
        wcs.wcs_from_header({**HEADERS[0], "CTYPE1": "RA---SIN"})


def test_cross_match_against_astropy():
    rs = np.random.default_rng(122)
    ra2, dec2 = rs.uniform(84.9, 85.9, 800), rs.uniform(-2.4, -1.4, 800)
    chosen = rs.choice(800, 300, replace=False)
    ra1 = ra2[chosen] + rs.normal(0, 0.3 / 3600, 300) / np.cos(np.radians(dec2[chosen]))
    dec1 = dec2[chosen] + rs.normal(0, 0.3 / 3600, 300)
    result = matching.cross_match(ra1, dec1, ra2, dec2, 2.0)
    index, separation, _ = SkyCoord(ra1 * u.deg, dec1 * u.deg).match_to_catalog_sky(SkyCoord(ra2 * u.deg, dec2 * u.deg))
    assert np.array_equal(result.index, index)
    assert np.max(np.abs(result.separation_arcsec - separation.arcsec)) < 1e-7
    assert np.mean(result.index == chosen) > 0.99 and result.matched.mean() > 0.99
    # Outside the radius: a list shifted by 1 degree does not match.
    assert not matching.cross_match(ra1, dec1 + 1.0, ra2, dec2, 2.0).matched.any()
