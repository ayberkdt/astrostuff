"""Test of the astrometric solution from an image against a known WCS."""

import numpy as np
import pytest

from kozmos.astrometry import solve
from kozmos.astrometry.wcs import Wcs, pixel_to_sky

PIXEL_SCALE_DEG = 1.0 / 3600.0  # degree / pixel


def _true_wcs(angle_deg=17.0):
    a = np.radians(angle_deg)
    cd = PIXEL_SCALE_DEG * np.array([[-np.cos(a), np.sin(a)], [np.sin(a), np.cos(a)]])
    return Wcs(crpix=np.array([512.3, 498.7]), crval=np.array([201.3, 41.2]), cd=cd)


def _field(seed, n=250, noise=0.08):
    rng = np.random.default_rng(seed)
    truth = _true_wcs()
    x, y = rng.uniform(0, 1024, n), rng.uniform(0, 1024, n)
    ra, dec = pixel_to_sky(truth, x, y)
    return truth, x + rng.normal(0, noise, n), y + rng.normal(0, noise, n), ra, dec, rng


def _max_difference_arcsec(w1, w2):
    kx, ky = np.meshgrid([0.0, 512.0, 1023.0], [0.0, 512.0, 1023.0])
    r1, d1 = pixel_to_sky(w1, kx, ky)
    r2, d2 = pixel_to_sky(w2, kx, ky)
    return float(np.max(np.hypot((r1 - r2) * np.cos(np.radians(d1)), d1 - d2)) * 3600.0)


def test_fit_wcs_finds_known_solution():
    truth, x, y, ra, dec, _ = _field(201)
    c = solve.fit_wcs(x, y, ra, dec, truth.crval)
    assert np.max(np.abs(c.wcs.cd / truth.cd - 1)) < 2e-4
    assert np.max(np.abs(c.wcs.crpix - truth.crpix)) < 0.03
    # 0.08 pixel noise on two axes: position residual rms ≈ 0.08 · √2 pixel = 0.11 arcseconds.
    assert c.residual_rms_arcsec == pytest.approx(0.08 * np.sqrt(2.0), rel=0.15)
    assert _max_difference_arcsec(c.wcs, truth) < 0.05
    # If the tangent point is not given, the mean direction of the stars is used; the transformation on the sky is unchanged.
    free = solve.fit_wcs(x, y, ra, dec)
    assert _max_difference_arcsec(free.wcs, truth) < 0.05
    with pytest.raises(ValueError):
        solve.fit_wcs(x[:2], y[:2], ra[:2], dec[:2])


def test_exact_solution_on_noiseless_data():
    truth, _, _, _, _, rng = _field(202)
    x, y = rng.uniform(0, 1024, 50), rng.uniform(0, 1024, 50)
    ra, dec = pixel_to_sky(truth, x, y)
    c = solve.fit_wcs(x, y, ra, dec, truth.crval)
    assert np.max(np.abs(c.wcs.cd - truth.cd)) < 1e-13 and np.max(np.abs(c.wcs.crpix - truth.crpix)) < 1e-7
    assert c.residual_rms_arcsec < 1e-6


def test_shift_found_by_voting():
    truth, x, y, ra, dec, rng = _field(203)
    # Detection list: 70% of the stars + 80 false sources; catalog: 80% of the stars.
    detected = rng.random(x.size) < 0.7
    xs = np.concatenate([x[detected], rng.uniform(0, 1024, 80)])
    ys = np.concatenate([y[detected], rng.uniform(0, 1024, 80)])
    in_catalog = rng.random(x.size) < 0.8
    shifted = Wcs(crpix=truth.crpix + np.array([-37.4, 21.8]), crval=truth.crval, cd=truth.cd)
    corrected, votes = solve.find_shift(shifted, xs, ys, ra[in_catalog], dec[in_catalog])
    assert np.max(np.abs(corrected.crpix - truth.crpix)) < 0.3
    assert votes > 0.4 * (detected & in_catalog).sum()  # most of the common stars are in the peak bin


def test_refinement_converges_from_distorted_start():
    truth, x, y, ra, dec, rng = _field(204)
    xs = np.concatenate([x[:200], rng.uniform(0, 1024, 60)])
    ys = np.concatenate([y[:200], rng.uniform(0, 1024, 60)])
    # Start: rotated by 0.15°, scale 0.3% too large, shifted by 12 pixels (an error of a few arcseconds at the edges).
    a = np.radians(0.15)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    distorted = Wcs(crpix=truth.crpix + np.array([12.0, -9.0]), crval=truth.crval, cd=1.003 * rotation @ truth.cd)
    assert _max_difference_arcsec(distorted, truth) > 10.0
    unshifted, _ = solve.find_shift(distorted, xs, ys, ra, dec, max_shift_pix=40.0)
    c = solve.refine_wcs(unshifted, xs, ys, ra, dec, radius_arcsec=6.0)
    assert _max_difference_arcsec(c.wcs, truth) < 0.1
    assert c.n_stars >= 190 and c.residual_rms_arcsec < 0.2
    with pytest.raises(ValueError):
        solve.refine_wcs(truth, xs[:2], ys[:2], ra + 5.0, dec, radius_arcsec=1.0)
