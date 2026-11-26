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
    assert c.residual_rms_arcsec == pytest.approx(0.08 * np.sqrt(2.0), rel=0.15, abs=0)
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


def test_find_shift_scales_to_large_lists():
    # 20,000 detections against 20,000 catalog stars: the all-pairs array would hold 4e8 differences (6.4 GB as
    # two float64 arrays). Only pairs inside the search window are formed, so this runs in a normal process.
    rng = np.random.default_rng(221)
    n = 20000
    scale_deg = 0.5 / 3600.0
    truth = Wcs(
        crpix=np.array([4096.0, 4096.0]),
        crval=np.array([150.0, 20.0]),
        cd=scale_deg * np.array([[-1.0, 0.0], [0.0, 1.0]]),
    )
    x, y = rng.uniform(0, 8192, n), rng.uniform(0, 8192, n)
    ra, dec = pixel_to_sky(truth, x, y)
    shifted = Wcs(crpix=truth.crpix + np.array([11.3, -7.6]), crval=truth.crval, cd=truth.cd)
    corrected, votes = solve.find_shift(
        shifted, x + rng.normal(0, 0.1, n), y + rng.normal(0, 0.1, n), ra, dec, max_shift_pix=30.0
    )
    assert np.max(np.abs(corrected.crpix - truth.crpix)) < 0.05
    assert votes > 0.9 * n


def test_find_shift_rejects_disjoint_lists():
    truth = Wcs(crpix=np.array([50.0, 50.0]), crval=np.array([10.0, 10.0]), cd=np.eye(2) / 3600.0)
    with pytest.raises(ValueError, match="no detected source"):
        solve.find_shift(truth, [10.0, 20.0], [10.0, 20.0], [40.0, 41.0], [-30.0, -31.0])


def test_outlier_rejection_removes_wrong_matches():
    # 200 correct pairs with 0.05-pixel noise plus 30 wrong pairs displaced by 3-10 pixels (mismatches to a
    # neighboring star). Plain least squares is pulled by them; the clipped fit is not.
    rng = np.random.default_rng(222)
    scale_deg = 1.0 / 3600.0
    truth = Wcs(
        crpix=np.array([512.0, 512.0]),
        crval=np.array([80.0, -30.0]),
        cd=scale_deg * np.array([[-1.0, 0.0], [0.0, 1.0]]),
    )
    x, y = rng.uniform(0, 1024, 230), rng.uniform(0, 1024, 230)
    ra, dec = pixel_to_sky(truth, x, y)
    xn, yn = x + rng.normal(0, 0.05, 230), y + rng.normal(0, 0.05, 230)
    angle = rng.uniform(0, 2 * np.pi, 30)
    radius = rng.uniform(3.0, 10.0, 30)
    xn[:30] += radius * np.cos(angle)
    yn[:30] += radius * np.sin(angle)

    def corner_error_arcsec(w):
        cx, cy = np.meshgrid([0.0, 1023.0], [0.0, 1023.0])
        r1, d1 = pixel_to_sky(w, cx, cy)
        r2, d2 = pixel_to_sky(truth, cx, cy)
        return float(np.max(np.hypot((r1 - r2) * np.cos(np.radians(d2)), d1 - d2)) * 3600.0)

    plain = solve.fit_wcs(xn, yn, ra, dec, truth.crval)
    clipped = solve.fit_wcs(xn, yn, ra, dec, truth.crval, clip_sigma=4.0)
    assert clipped.n_stars <= 205 and clipped.n_stars >= 190  # the 30 wrong pairs are gone, few good ones lost
    assert clipped.residual_rms_arcsec < 0.1 < plain.residual_rms_arcsec
    assert corner_error_arcsec(clipped.wcs) < 0.03
    assert corner_error_arcsec(plain.wcs) > 3.0 * corner_error_arcsec(clipped.wcs)
