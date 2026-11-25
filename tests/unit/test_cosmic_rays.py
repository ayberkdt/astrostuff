"""Test of cosmic-ray detection with synthetic data."""

import numpy as np

from kozmos.observing import detector
from kozmos.reduction import cosmic_rays

SHAPE = (160, 160)
DETECTOR = detector.Detector(gain_e_adu=2.0, read_noise_e=5.0, dark_current_e_s=0.0, bias_adu=0.0)


def _frame(seed, fwhm=3.0, n_stars=30, n_rays=60):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(8, SHAPE[1] - 8, n_stars), rng.uniform(8, SHAPE[0] - 8, n_stars)
    scene = 50.0 + detector.star_field(SHAPE, x, y, 10 ** rng.uniform(2.5, 4.3, n_stars), fwhm)  # e/s
    clean = detector.simulate_raw_frame(scene, 20.0, DETECTOR, rng)  # ADU
    dirty = clean.copy()
    truth = np.zeros(SHAPE, dtype=bool)
    # Single-pixel hits and short tracks of 2-4 pixels; every pixel at least ~10 times the noise.
    for _ in range(n_rays):
        y0, x0 = rng.integers(3, SHAPE[0] - 3), rng.integers(3, SHAPE[1] - 3)
        length = rng.integers(1, 5)
        dy, dx = (0, 1) if rng.random() < 0.5 else (1, 0)
        for k in range(length):
            yy, xx = y0 + k * dy, x0 + k * dx
            if 0 <= yy < SHAPE[0] and 0 <= xx < SHAPE[1]:
                dirty[yy, xx] += rng.uniform(400.0, 4000.0)
                truth[yy, xx] = True
    return clean, dirty, truth, (x, y)


def test_cosmic_rays_found_and_stars_preserved():
    clean, dirty, truth, (x, y) = _frame(131)
    result, mask = cosmic_rays.detect_cosmic_rays(dirty, gain_e_adu=2.0, read_noise_e=5.0)
    completeness = (mask & truth).sum() / truth.sum()
    false_flags = (mask & ~truth).sum()
    assert completeness > 0.95
    assert false_flags <= 0.002 * mask.size  # falsely flagged pixels below two per thousand
    # Star centers are not flagged (well-sampled PSF, FWHM = 3 pixels).
    centers = np.zeros(SHAPE, dtype=bool)
    centers[np.round(y).astype(int), np.round(x).astype(int)] = True
    assert not (mask & centers & ~truth).any()
    # In the hit pixels the cleaned image is much closer to the clean frame than the dirty frame is.
    assert np.abs(result - clean)[truth].mean() < 0.05 * np.abs(dirty - clean)[truth].mean()


def test_few_false_alarms_in_clean_frame():
    clean, _, _, _ = _frame(132, n_rays=0)
    _, mask = cosmic_rays.detect_cosmic_rays(clean, gain_e_adu=2.0, read_noise_e=5.0)
    assert mask.sum() <= 0.001 * mask.size


def test_fewer_pixels_as_threshold_grows():
    _, dirty, _, _ = _frame(133)
    _, loose = cosmic_rays.detect_cosmic_rays(dirty, gain_e_adu=2.0, read_noise_e=5.0, sigma_lim=4.0)
    _, strict = cosmic_rays.detect_cosmic_rays(dirty, gain_e_adu=2.0, read_noise_e=5.0, sigma_lim=12.0)
    assert strict.sum() <= loose.sum()
