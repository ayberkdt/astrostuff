"""Test of background, source detection and aperture photometry with synthetic data."""

import numpy as np
import pytest

from kozmos import photometry
from kozmos.observing import detector

SHAPE = (256, 256)
FWHM = 3.0


def _field(seed, n_stars=40, sky=200.0, slope=0.0, noise=5.0):
    rng = np.random.default_rng(seed)
    # Sources are at least 12 pixels apart: overlap is not the subject of this test.
    positions = []
    while len(positions) < n_stars:
        candidate = rng.uniform(15, SHAPE[0] - 15, 2)
        if all(np.hypot(*(candidate - p)) > 12 for p in positions):
            positions.append(candidate)
    x, y = np.array(positions).T
    flux = 10 ** rng.uniform(3.0, 4.5, n_stars)
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]]
    ground = sky + slope * (xx + yy)
    image = ground + detector.star_field(SHAPE, x, y, flux, FWHM) + rng.normal(0, noise, SHAPE)
    return image, ground, x, y, flux


def test_background_level_unbiased():
    image, _, *_ = _field(111)
    level, rms = photometry.estimate_background(image)
    assert level == pytest.approx(200.0, abs=0.3)  # despite the stars; a plain mean would be biased
    assert rms == pytest.approx(5.0, rel=0.1, abs=0)
    assert image.mean() - 200.0 > 1.0


def test_background_map_follows_slope():
    image, ground, *_ = _field(112, slope=0.05)
    bkg_map, rms = photometry.background_map(image, cell=32)
    interior = (slice(32, -32), slice(32, -32))
    assert np.abs(bkg_map[interior] / ground[interior] - 1).max() < 0.01
    assert np.median(rms) == pytest.approx(5.0, rel=0.15, abs=0)
    # On a steep slope the variation within a cell adds to the noise: the rms map comes out larger than the true noise.
    _, rms_steep = photometry.background_map(_field(112, slope=0.3)[0], cell=32)
    assert np.median(rms_steep) > 5.5


def test_source_detection_completeness_and_position():
    image, _, x, y, flux = _field(113)
    bkg_map, rms = photometry.background_map(image, cell=64)
    found = photometry.detect_sources(image, bkg_map, rms, threshold_sigma=4.0, min_pixels=5)
    # Every true star has a detection within 1 pixel; at most 1 extra detection.
    distance = np.hypot(found.x[None, :] - x[:, None], found.y[None, :] - y[:, None])
    assert np.all(distance.min(axis=1) < 1.0)
    assert len(found) - len(x) <= 1
    assert np.all(np.diff(found.flux) <= 0)  # decreasing order of flux
    # For bright stars the center error is below 0.1 pixel.
    bright = flux > 1e4
    assert np.all(distance.min(axis=1)[bright] < 0.1)


def test_few_false_detections_in_pure_noise():
    rng = np.random.default_rng(114)
    noise = rng.normal(0, 1, (512, 512))
    assert len(photometry.detect_sources(noise, 0.0, 1.0, threshold_sigma=3.0, min_pixels=5)) <= 2
    # Without the minimum-pixel condition hundreds of single pixels above 3σ would appear.
    assert len(photometry.detect_sources(noise, 0.0, 1.0, threshold_sigma=3.0, min_pixels=1)) > 200
    assert len(photometry.detect_sources(np.zeros((8, 8)), 0.0, 1.0)) == 0


def test_smoothing_recovers_faint_source():
    rng = np.random.default_rng(115)
    scene = detector.star_field((64, 64), [32.0], [32.0], [60.0], 4.0)  # peak ≈ 3.3; noise 1
    image = scene + rng.normal(0, 1, (64, 64))
    smoothed = photometry.detect_sources(image, 0.0, 1.0, threshold_sigma=5.0, min_pixels=3, smoothing_fwhm=4.0)
    assert len(photometry.detect_sources(image, 0.0, 1.0, threshold_sigma=5.0, min_pixels=3)) == 0
    assert len(smoothed) == 1 and abs(smoothed.x[0] - 32) < 1.5 and abs(smoothed.y[0] - 32) < 1.5


def test_aperture_area_and_enclosed_flux():
    flux, area = photometry.aperture_photometry(np.ones((60, 60)), [30.2], [29.7], 7.5, subpixels=9)
    assert area[0] == pytest.approx(np.pi * 7.5**2, rel=2e-3, abs=0) and flux[0] == pytest.approx(area[0])
    # For a Gaussian PSF the flux fraction within radius r is 1 − exp(−r²/2σ²).
    sigma = FWHM / 2.3548200450309493
    scene = 50.0 + detector.star_field((81, 81), [40.3], [39.8], [1e4], FWHM)

    def fraction(r):
        return photometry.aperture_photometry(scene, [40.3], [39.8], r, annulus=(15.0, 20.0), subpixels=9)[0][0] / 1e4

    for r in (5.0, 7.0, 9.0):
        assert fraction(r) == pytest.approx(1 - np.exp(-(r**2) / (2 * sigma**2)), abs=5e-3)
    # Limit of the method at a narrow aperture: since the flux is taken as uniform within a pixel, at radii
    # where the PSF is steep the fraction comes out low (about 4% for r = 2, FWHM = 3).
    assert 0.02 < (1 - np.exp(-(2.0**2) / (2 * sigma**2))) - fraction(2.0) < 0.06


def test_aperture_photometry_unbiased_in_noisy_field():
    image, _, x, y, flux = _field(116, n_stars=30)
    measured, _ = photometry.aperture_photometry(image, x, y, 6.0, annulus=(9.0, 12.0))
    sigma = FWHM / 2.3548200450309493
    expected = flux * (1 - np.exp(-(6.0**2) / (2 * sigma**2)))
    # Noise in the aperture: sqrt(π r²) · 5 ≈ 53; the mean relative deviation of 30 stars is below one percent.
    assert np.mean(measured / expected - 1) == pytest.approx(0.0, abs=0.01)
    assert np.std((measured - expected) / 60.0) < 2.0


def test_close_pairs_are_deblended():
    rng = np.random.default_rng(171)
    shape = (200, 200)
    # 16 pairs: separation 2 FWHM (with a saddle between them), flux ratio between 1 and 2; pairs far from each other.
    cx, cy = (a.ravel() for a in np.meshgrid(np.arange(25, 200, 50.0), np.arange(25, 200, 50.0)))
    angle = rng.uniform(0, 2 * np.pi, 16)
    fwhm, separation = 3.0, 6.0
    x = np.concatenate([cx - separation / 2 * np.cos(angle), cx + separation / 2 * np.cos(angle)])
    y = np.concatenate([cy - separation / 2 * np.sin(angle), cy + separation / 2 * np.sin(angle)])
    flux = np.concatenate([np.full(16, 4000.0), 4000.0 / rng.uniform(1, 2, 16)])
    image = 100.0 + detector.star_field(shape, x, y, flux, fwhm) + rng.normal(0, 2.0, shape)
    single = photometry.detect_sources(image, 100.0, 2.0, threshold_sigma=4.0, smoothing_fwhm=fwhm)
    deblended = photometry.detect_sources(
        image, 100.0, 2.0, threshold_sigma=4.0, smoothing_fwhm=fwhm, deblend_levels=32
    )
    assert len(single) == 16  # without deblending each pair is a single source
    # In a measurement with 200 realizations (random orientation, subpixel position) this setting separates 92-98% of the ratio 1-2 pairs.
    assert 29 <= len(deblended) <= 32
    distance = np.hypot(deblended.x[None, :] - x[:, None], deblended.y[None, :] - y[:, None]).min(axis=1)
    assert np.sum(distance < 1.0) >= 29  # every separated component is found within one pixel
    # Total flux is conserved: deblending shares pixels, it does not add or remove them.
    assert deblended.flux.sum() == pytest.approx(single.flux.sum(), rel=1e-12, abs=0)
    assert deblended.n_pixels.sum() == single.n_pixels.sum()


def test_deblending_limit_shallow_saddle():
    # At the same separation the saddle gets shallower as the flux ratio grows. Measurement with 200 realizations
    # (2 FWHM separation, random orientation): 32 levels and a contrast of 0.005 separate 14% of the pairs at
    # ratio 3 and 1% at ratio 4; 128 levels and a contrast of 0.001 separate 99% at ratio 3 and 59% at ratio 4.
    rng = np.random.default_rng(173)
    common = {"threshold_sigma": 4.0, "smoothing_fwhm": 3.0}
    coarse = fine = 0
    for _ in range(20):
        image = 100.0 + detector.star_field((60, 60), [27.0, 33.0], [30.0, 30.0], [4000.0, 4000.0 / 3.0], 3.0)
        image = image + rng.normal(0, 2.0, (60, 60))
        coarse += len(photometry.detect_sources(image, 100.0, 2.0, deblend_levels=32, **common)) == 2
        fine += len(photometry.detect_sources(image, 100.0, 2.0, deblend_levels=128, contrast=0.001, **common)) == 2
    assert coarse <= 5 and fine >= 17


def test_deblending_does_not_split_single_stars():
    rng = np.random.default_rng(172)
    shape = (200, 200)
    x, y = (a.ravel().astype(float) for a in np.meshgrid(np.arange(20, 200, 40), np.arange(20, 200, 40)))
    image = (
        100.0 + detector.star_field(shape, x, y, 10 ** rng.uniform(2.6, 4.5, x.size), 3.0) + rng.normal(0, 2.0, shape)
    )
    single = photometry.detect_sources(image, 100.0, 2.0, threshold_sigma=4.0, smoothing_fwhm=3.0)
    deblended = photometry.detect_sources(image, 100.0, 2.0, threshold_sigma=4.0, smoothing_fwhm=3.0, deblend_levels=32)
    assert len(single) == x.size and len(deblended) == x.size


def test_center_of_segment_without_positive_pixels_is_defined():
    # When the detection image and the measurement image diverge (smoothing) the positive weight of a segment can be zero.
    image = np.zeros((40, 40))
    image[18:23, 18:23] = 5.0
    result = photometry.detect_sources(
        image, 6.0, 1.0, threshold_sigma=-2.0, min_pixels=5
    )  # background 6: all pixels negative
    assert len(result) >= 1 and np.all(np.isfinite(result.x)) and np.all(np.isfinite(result.y))
