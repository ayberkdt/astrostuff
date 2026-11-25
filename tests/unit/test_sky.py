"""Test of sky subtraction from dithered frames with a synthetic near-infrared sequence."""

import numpy as np
import pytest

from kozmos import photometry
from kozmos.observing import detector
from kozmos.reduction import sky

SHAPE = (96, 96)
NOISE = 3.0


def _sequence(seed, n_frames=12, shape_changes=False):
    """Bright and structured sky + the same star field dithered in each frame."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]]
    pattern = 1.0 + 0.08 * np.sin(xx / 15.0) + 0.05 * (yy / SHAPE[0])  # fixed structure from detector/optics
    slope = (xx - SHAPE[1] / 2) / SHAPE[1]
    x0, y0 = rng.uniform(20, 76, 12), rng.uniform(20, 76, 12)
    flux = 10 ** rng.uniform(2.8, 3.6, 12)
    times = np.arange(n_frames, dtype=float)
    offsets = rng.uniform(-12, 12, (n_frames, 2))
    frames, stars = [], []
    for k in range(n_frames):
        level = 2000.0 * (1.0 + 0.15 * np.sin(times[k] / 2.5))  # the sky varies by 15% within minutes
        sky_shape = pattern * (1.0 + (0.06 * (times[k] - n_frames / 2) / n_frames * slope if shape_changes else 0.0))
        star = detector.star_field(SHAPE, x0 + offsets[k, 0], y0 + offsets[k, 1], flux, 3.0)
        frames.append(level * sky_shape + star + rng.normal(0, NOISE, SHAPE))
        stars.append(star)
    return np.array(frames), np.array(stars), times, (x0, y0, flux, offsets)


def test_static_model_subtracts_sky_and_preserves_stars():
    frames, stars, _, (x0, y0, flux, offsets) = _sequence(141)
    output, model = sky.subtract_static_sky(frames)
    residual = output - stars
    # The sky drops from a level of 2000 to the noise level. The remaining level is not zero: rejecting the highest
    # value pulls the median down and shifts the residual to positive by about a tenth of the noise.
    assert np.abs(np.median(residual, axis=(1, 2))).max() < 0.3 * NOISE
    assert residual.std() < 1.25 * NOISE
    assert np.median(model) == pytest.approx(1.0, abs=0.01)
    # Star flux: compared with the flux measured with the same apertures in the noiseless, skyless star frame.
    # The field is crowded (12 stars, ±12 pixel dither): the unmasked model contains starlight and reduces the flux.
    # The model built with the object mask preserves the flux within 3%.
    k = 5
    xk, yk = x0 + offsets[k, 0], y0 + offsets[k, 1]

    def ratio(image):
        measured, _ = photometry.aperture_photometry(image, xk, yk, 7.0, annulus=(10.0, 13.0))
        truth, _ = photometry.aperture_photometry(stars[k], xk, yk, 7.0, annulus=(10.0, 13.0))
        return float(np.median(measured / truth))

    masked, _ = sky.subtract_static_sky(frames, masks=stars > NOISE)
    print(f"flux ratio: unmasked {ratio(output[k]):.3f}, masked {ratio(masked[k]):.3f}")
    assert ratio(masked[k]) == pytest.approx(1.0, abs=0.03)
    assert ratio(output[k]) < ratio(masked[k])
    assert np.all(flux > 0)


def test_rejecting_highest_pixel_reduces_star_residue():
    frames, stars, _, _ = _sequence(142)
    without_rejection, _ = sky.subtract_static_sky(frames, reject_highest=0)
    with_rejection, _ = sky.subtract_static_sky(frames, reject_highest=1)
    # Pixels where a star falls in other frames but which are empty in this frame: the unmasked median is pulled up
    # here by starlight and leaves a negative dip after subtraction. Rejecting the highest value makes the dip shallower.
    for k in range(frames.shape[0]):
        shadow = (np.delete(stars, k, axis=0).max(axis=0) > 200.0) & (stars[k] < 1.0)
        if shadow.sum() > 30:
            break
    dip_without = np.mean((without_rejection - stars)[k][shadow])
    dip_with = np.mean((with_rejection - stars)[k][shadow])
    assert dip_without < 0.0 and dip_with > dip_without
    # The price: in pixels that see no star the model shifts down and the residual turns positive (about a tenth of the noise).
    empty = stars.max(axis=0) < 1.0
    shift = np.mean((with_rejection - stars)[:, empty]) - np.mean((without_rejection - stars)[:, empty])
    assert 0.0 < shift < 0.25 * NOISE


def test_dynamic_model_better_when_shape_changes():
    frames, stars, times, _ = _sequence(143, n_frames=24, shape_changes=True)
    static, _ = sky.subtract_static_sky(frames)
    dynamic = sky.subtract_dynamic_sky(frames, times, n_neighbors=6)

    # Large-scale residual: averaging in 8×8 boxes suppresses the noise; the remaining sky structure is measured.
    def structure(residual):
        boxes = residual[:, : 96 // 8 * 8, : 96 // 8 * 8].reshape(residual.shape[0], 12, 8, 12, 8).mean(axis=(2, 4))
        return boxes.std()

    assert structure(dynamic - stars) < 0.8 * structure(static - stars)


def test_input_validation():
    frames = np.ones((5, 8, 8))
    with pytest.raises(ValueError):
        sky.subtract_static_sky(frames[:2])
    with pytest.raises(ValueError):
        sky.subtract_dynamic_sky(frames, np.arange(4.0))
    with pytest.raises(ValueError):
        sky.subtract_dynamic_sky(frames, np.arange(5.0), n_neighbors=5)


def test_pixel_masked_in_every_frame_leaves_no_hole_in_model():
    # Case seen in the real SOFI sequence: the mask of a bright object covers a pixel in all frames.
    frames, stars, times, _ = _sequence(144)
    masks = stars > NOISE
    masks[:, 40:44, 40:44] = True
    output, model = sky.subtract_static_sky(frames, masks=masks)
    assert np.all(np.isfinite(model)) and np.all(np.isfinite(output))
    assert np.all(np.isfinite(sky.subtract_dynamic_sky(frames, times, n_neighbors=4, masks=masks)))
