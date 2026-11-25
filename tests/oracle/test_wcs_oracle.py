"""WCS dönüşümü ve katalog eşlemesinin astropy ile karşılaştırılması."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from kozmos.astrometry import matching, wcs

pytestmark = pytest.mark.oracle

BASLIKLAR = [
    # SkyView tarzı: CDELT, dönme yok.
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
    # CD matrisi, 30° dönük, yüksek dik açıklık.
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
    # PC + CDELT, sağ açıklık 0/360 sınırında, güney yarıküre.
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
    # Eski kural: CROTA2.
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


@pytest.mark.parametrize("header", BASLIKLAR)
def test_piksel_gok_astropy_ile(header):
    bizim = wcs.wcs_from_header(header)
    kahin = WCS(header)
    rs = np.random.default_rng(121)
    x, y = rs.uniform(-50, 900, 400), rs.uniform(-50, 900, 400)
    ra, dec = wcs.pixel_to_sky(bizim, x, y)
    ra_k, dec_k = kahin.all_pix2world(x, y, 0)
    ayriklik = SkyCoord(ra * u.deg, dec * u.deg).separation(SkyCoord(ra_k * u.deg, dec_k * u.deg)).deg
    assert ayriklik.max() < 1e-9
    x2, y2 = wcs.sky_to_pixel(bizim, ra_k, dec_k)
    xk, yk = kahin.all_world2pix(ra_k, dec_k, 0)
    assert np.max(np.abs(x2 - xk)) < 1e-7 and np.max(np.abs(y2 - yk)) < 1e-7
    assert np.max(np.abs(x2 - x)) < 1e-7 and np.max(np.abs(y2 - y)) < 1e-7  # gidiş-dönüş


def test_referans_piksel_ve_hatalar():
    w = wcs.wcs_from_header(BASLIKLAR[0])
    ra, dec = wcs.pixel_to_sky(w, 224.5, 224.5)  # CRPIX = 225,5 (FITS) ↔ sıfır tabanlı 224,5
    assert float(ra) == pytest.approx(85.42, abs=1e-12) and float(dec) == pytest.approx(-1.9, abs=1e-12)
    # Karşı yarıküre TAN'da tanımsızdır.
    assert np.isnan(wcs.sky_to_pixel(w, 85.42 + 180.0, 1.9)[0])
    with pytest.raises(ValueError):
        wcs.wcs_from_header({**BASLIKLAR[0], "CTYPE1": "RA---SIN"})


def test_capraz_esleme_astropy_ile():
    rs = np.random.default_rng(122)
    ra2, dec2 = rs.uniform(84.9, 85.9, 800), rs.uniform(-2.4, -1.4, 800)
    secim = rs.choice(800, 300, replace=False)
    ra1 = ra2[secim] + rs.normal(0, 0.3 / 3600, 300) / np.cos(np.radians(dec2[secim]))
    dec1 = dec2[secim] + rs.normal(0, 0.3 / 3600, 300)
    sonuc = matching.cross_match(ra1, dec1, ra2, dec2, 2.0)
    index, ayriklik, _ = SkyCoord(ra1 * u.deg, dec1 * u.deg).match_to_catalog_sky(SkyCoord(ra2 * u.deg, dec2 * u.deg))
    assert np.array_equal(sonuc.index, index)
    assert np.max(np.abs(sonuc.separation_arcsec - ayriklik.arcsec)) < 1e-7
    assert np.mean(sonuc.index == secim) > 0.99 and sonuc.matched.mean() > 0.99
    # Yarıçap dışı: 1 derece kaydırılmış liste eşleşmez.
    assert not matching.cross_match(ra1, dec1 + 1.0, ra2, dec2, 2.0).matched.any()
