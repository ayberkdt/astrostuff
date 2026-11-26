"""Test of the detector model and the reduction pipeline with synthetic data.

A scene with known truth is generated, converted to raw frames, reduced, and the
result is compared with the input scene.
"""

import numpy as np
import pytest

from kozmos.observing import detector
from kozmos.reduction import calibration

SHAPE = (120, 160)
DETECTOR = detector.Detector(gain_e_adu=1.7, read_noise_e=6.0, dark_current_e_s=0.2, bias_adu=1000.0)


def _flat_field(rng):
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]]
    vignetting = 1.0 - 0.25 * (((xx - 80) / 110) ** 2 + ((yy - 60) / 110) ** 2)
    flat = vignetting * rng.normal(1.0, 0.02, SHAPE)
    return flat / flat.mean()


def _scene(rng, n_stars=25):
    x, y = rng.uniform(10, SHAPE[1] - 10, n_stars), rng.uniform(10, SHAPE[0] - 10, n_stars)
    flux = 10 ** rng.uniform(2.0, 3.5, n_stars)
    return 20.0 + detector.star_field(SHAPE, x, y, flux, fwhm_pix=3.2)  # sky 20 e/s/pixel


def test_star_field_conserves_flux():
    scene = detector.star_field((61, 61), [30.3], [29.6], [1000.0], fwhm_pix=2.5)
    assert scene.sum() == pytest.approx(1000.0, rel=1e-9, abs=0)
    # Also for a narrow PSF: sampling at the pixel center would not conserve flux, the pixel integral does.
    assert detector.star_field((41, 41), [20.4], [20.1], [500.0], fwhm_pix=0.8).sum() == pytest.approx(
        500.0, rel=1e-9, abs=0
    )


def test_gain_and_read_noise_recovery():
    rng = np.random.default_rng(81)
    flat = _flat_field(rng)
    zeros = np.zeros(SHAPE)
    illumination = np.full(SHAPE, 8000.0)  # e/s; 1 s exposure
    f1, f2 = (detector.simulate_raw_frame(illumination, 1.0, DETECTOR, rng, flat_field=flat) for _ in range(2))
    b1, b2 = (detector.simulate_raw_frame(zeros, 0.0, DETECTOR, rng) for _ in range(2))
    gain, read_noise = calibration.gain_and_read_noise(f1, f2, b1, b2)
    assert gain == pytest.approx(DETECTOR.gain_e_adu, rel=0.03, abs=0)
    assert read_noise == pytest.approx(DETECTOR.read_noise_e, rel=0.03, abs=0)


def test_reduction_returns_the_scene():
    rng = np.random.default_rng(82)
    flat = _flat_field(rng)
    scene = _scene(rng)
    zeros = np.zeros(SHAPE)
    exposure = 60.0

    bias, _ = calibration.combine([detector.simulate_raw_frame(zeros, 0.0, DETECTOR, rng) for _ in range(25)])
    dark_adu, _ = calibration.combine([detector.simulate_raw_frame(zeros, 300.0, DETECTOR, rng) for _ in range(15)])
    dark_e_s = (dark_adu - bias) * DETECTOR.gain_e_adu / 300.0
    flats, _ = calibration.combine(
        [detector.simulate_raw_frame(np.full(SHAPE, 3000.0), 5.0, DETECTOR, rng, flat_field=flat) for _ in range(15)]
    )
    master_flat = (flats - bias) * DETECTOR.gain_e_adu - dark_e_s * 5.0
    master_flat /= master_flat.mean()
    assert (
        np.std(master_flat / flat) < 0.004
    )  # the flat-field pattern is recovered at the level of a few parts per thousand

    raw = detector.simulate_raw_frame(scene, exposure, DETECTOR, rng, flat_field=flat)
    result = calibration.reduce_frame(
        raw,
        bias,
        dark_e_s,
        master_flat,
        gain_e_adu=DETECTOR.gain_e_adu,
        read_noise_e=DETECTOR.read_noise_e,
        exposure_s=exposure,
    )
    residual = (result.data - scene) / np.sqrt(result.variance)
    # Unbiased (mean residual around zero) and the variance plane predicts the true scatter (χ²/N ≈ 1).
    assert abs(residual.mean()) < 0.05
    assert np.mean(residual**2) == pytest.approx(1.0, abs=0.08)
    # Without the flat-field correction a vignetting residual would remain.
    uncorrected = calibration.reduce_frame(
        raw, bias, dark_e_s, 1.0, gain_e_adu=1.7, read_noise_e=6.0, exposure_s=exposure
    )
    assert np.mean(((uncorrected.data - scene) / np.sqrt(uncorrected.variance)) ** 2) > 3.0


def test_combine_rejects_cosmic_rays():
    rng = np.random.default_rng(83)
    frames = rng.normal(100.0, 5.0, (9, 40, 50))
    clean_mean = frames.mean(axis=0)
    hit = rng.integers(0, 9, 30), rng.integers(0, 40, 30), rng.integers(0, 50, 30)
    frames[hit] += 5000.0
    unclipped, _ = calibration.combine(frames, sigma=None)
    clipped, used = calibration.combine(frames, sigma=2.5)
    assert np.abs(unclipped - clean_mean).max() > 400
    assert np.abs(clipped - clean_mean).max() < 6.0
    assert used.min() >= 6 and used[hit[1], hit[2]].max() <= 8
    median, _ = calibration.combine(frames, method="median", sigma=None)
    assert np.abs(median - 100.0).max() < 12
    with pytest.raises(ValueError):
        calibration.combine(frames[0])
    with pytest.raises(ValueError):
        calibration.combine(frames, method="mode")


def test_ccd_equation_limits():
    # Source dominated: S/N → sqrt(N*).
    assert float(detector.signal_to_noise(1e6, 0.0, 0.0, 0.0, 10)) == pytest.approx(1e3, rel=1e-6, abs=0)
    # Sky dominated: S/N → N* / sqrt(n_pix · N_S).
    assert float(detector.signal_to_noise(100.0, 1e5, 0.0, 5.0, 50)) == pytest.approx(
        100 / np.sqrt(50 * 1e5), rel=1e-3, abs=0
    )
    # If the background is estimated from few pixels, the noise increases.
    assert detector.signal_to_noise(1e3, 200.0, 1.0, 5.0, 50, 100) < detector.signal_to_noise(1e3, 200.0, 1.0, 5.0, 50)


def test_ccd_equation_with_synthetic_photometry():
    # The frame-to-frame scatter of the flux of a single star in a fixed aperture is as predicted by the CCD equation.
    rng = np.random.default_rng(84)
    shape, exposure, sky, flux = (41, 41), 30.0, 15.0, 400.0
    scene = sky + detector.star_field(shape, [20.0], [20.0], [flux], fwhm_pix=3.0)
    yy, xx = np.mgrid[:41, :41]
    aperture = (xx - 20) ** 2 + (yy - 20) ** 2 <= 6.0**2
    d = detector.Detector(gain_e_adu=1.0, read_noise_e=8.0, dark_current_e_s=0.0, bias_adu=0.0)
    measured = []
    for _ in range(600):
        frame = detector.simulate_raw_frame(scene, exposure, d, rng)
        measured.append(frame[aperture].sum() - aperture.sum() * sky * exposure)
    measured = np.array(measured)
    source_e = (scene - sky)[aperture].sum() * exposure
    expected = float(detector.signal_to_noise(source_e, sky * exposure, 0.0, 8.0, aperture.sum()))
    assert measured.mean() == pytest.approx(source_e, rel=0.01, abs=0)
    assert measured.mean() / measured.std(ddof=1) == pytest.approx(expected, rel=0.08, abs=0)


def test_background_estimate_term_monte_carlo():
    # The (1 + n_pix/n_B) factor was derived in this project; here it is tested directly by simulation.
    # No source (N* -> 0 limit): the background estimated as the mean of n_B pixels is subtracted from the aperture sum.
    rng = np.random.default_rng(191)
    n_pix, n_b, sky, read_noise = 50, 100, 200.0, 5.0
    trials = 40000
    aperture = rng.poisson(sky, (trials, n_pix)) + rng.normal(0, read_noise, (trials, n_pix))
    annulus = rng.poisson(sky, (trials, n_b)) + rng.normal(0, read_noise, (trials, n_b))
    measured = (aperture.sum(axis=1) - n_pix * annulus.mean(axis=1)).std()
    # Since S/N = N*/noise, with N* = 1e-9 the value 1/(S/N) is the noise itself (source Poisson term neglected).
    expected = 1e-9 / float(detector.signal_to_noise(1e-9, sky, 0.0, read_noise, n_pix, n_b))
    without_term = 1e-9 / float(detector.signal_to_noise(1e-9, sky, 0.0, read_noise, n_pix))
    assert measured == pytest.approx(expected, rel=0.015, abs=0)
    assert (
        expected / without_term == pytest.approx(np.sqrt(1.5), rel=1e-9, abs=0)
        and abs(measured / without_term - 1) > 0.15
    )
