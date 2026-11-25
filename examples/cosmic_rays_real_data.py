"""Test of cosmic-ray cleaning on real HST frames.

Data: Hubble STIS CCD images (MAST). Each file holds two exposures taken one
after the other with the same pointing. Cosmic rays fall on different pixels
in the two exposures, so the difference of the two exposures gives an
independent "truth" for an algorithm that looks at a single frame. Two kinds
of field are tested:

- o5d903060: globular cluster M5 (NGC 5904), a crowded stellar field.
- o4gb010h0: galaxy cluster Abell 1795, a sparse field with diffuse sources.

Measured quantities: the fraction of true cosmic-ray pixels found from the
single frame (completeness), the fraction of flagged pixels that are adjacent
to a true cosmic ray (purity), and how close the cleaned frame is to the
second exposure. The STIS CCD PSF is undersampled; in that case the source
paper says to raise f_lim [vanDokkum2001], so two values are compared.

Usage:
    python examples/cosmic_rays_real_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import _plotting
import numpy as np
from scipy import ndimage

from kozmos.archives import cache
from kozmos.io import fits
from kozmos.reduction import cosmic_rays
from kozmos.visualization import stretch

ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = ("o5d903060", "o4gb010h0")
URL = "https://mast.stsci.edu/api/v0.1/Download/file"
TRUTH_THRESHOLD = (
    6.0  # a pixel counts as cosmic ray if the exposure difference exceeds this multiple of the difference noise
)
CUTOUT = (slice(430, 590), slice(430, 590))
SETTINGS = {
    "f_lim=2, no neighbor threshold": {"f_lim": 2.0, "neighbor_threshold_ratio": 1.0},
    "f_lim=2": {"f_lim": 2.0},
    "f_lim=5": {"f_lim": 5.0},
}


def measure(observation: str) -> tuple[dict, dict]:
    target = ROOT / "data" / "raw" / "hst" / f"{observation}_flt.fits"
    cache.download(
        URL,
        target,
        parameters={"uri": f"mast:HST/product/{observation}_flt.fits"},
        attribution="HST STIS, via MAST",
    )
    units = fits.list_hdus(target)
    primary = units[0][1]
    science = [i for i, (_, b, _) in enumerate(units) if b.get("EXTNAME") == "SCI"]
    if len(science) < 2:
        raise SystemExit(f"{observation}: two exposures expected, found {len(science)}")
    gain, read_noise = float(primary["ATODGAIN"]), float(primary["READNSE"])
    exposure1, header1 = fits.read_fits(target, hdu=science[0])
    exposure2, header2 = fits.read_fits(target, hdu=science[1])
    ratio = float(header1["EXPTIME"]) / float(header2["EXPTIME"])
    exposure2 = (
        exposure2 * ratio
    )  # if the exposure times differ, the second exposure is brought to the scale of the first

    # Truth: the difference of the two exposures. The difference noise is built from the smaller of the two
    # exposures (the value without cosmic rays).
    floor = np.minimum(exposure1, exposure2)
    sigma_difference = np.sqrt((1.0 + ratio) * (np.clip(gain * floor, 0.0, None) + read_noise**2)) / gain
    truth1 = (exposure1 - exposure2) > TRUTH_THRESHOLD * sigma_difference
    truth2 = (exposure2 - exposure1) > TRUTH_THRESHOLD * sigma_difference
    surround = ndimage.binary_dilation(truth1, structure=np.ones((3, 3)))  # neighbors adjacent to true pixels
    bright = truth1 & ((exposure1 - exposure2) > 20.0 * sigma_difference)

    result = {
        "data": {
            "target": str(primary["TARGNAME"]).strip(),
            "program": int(primary["PROPOSID"]),
            "exposure_time_s": [float(header1["EXPTIME"]), float(header2["EXPTIME"])],
            "gain_e_adu": gain,
            "read_noise_e": read_noise,
            "sky_level_adu": float(np.median(exposure1)),
        },
        "truth": {
            "threshold_sigma": TRUTH_THRESHOLD,
            "exposure1_cosmic_ray_pixels": int(truth1.sum()),
            "exposure2_cosmic_ray_pixels": int(truth2.sum()),
            "exposure1_fraction": float(truth1.mean()),
        },
        "settings": {},
    }
    cleaned_images = {}
    for name, option in SETTINGS.items():
        cleaned, mask = cosmic_rays.detect_cosmic_rays(exposure1, gain_e_adu=gain, read_noise_e=read_noise, **option)
        cleaned_images[name] = cleaned
        result["settings"][name] = {
            "flagged_pixels": int(mask.sum()),
            "completeness": float((mask & truth1).sum() / truth1.sum()),
            "completeness_above_20_sigma": float((mask & bright).sum() / bright.sum()),
            "purity_including_adjacent": float((mask & surround).sum() / mask.sum()),
            "non_truth_flagged_fraction": float((mask & ~surround).sum() / mask.size),
            # Cleaning quality: median absolute deviation from the second exposure at true cosmic-ray pixels [difference noise units].
            "deviation_before_cleaning_sigma": float(
                np.median(np.abs(exposure1 - exposure2)[truth1] / sigma_difference[truth1])
            ),
            "deviation_after_cleaning_sigma": float(
                np.median(np.abs(cleaned - exposure2)[truth1] / sigma_difference[truth1])
            ),
        }
    level = float(np.median(exposure1))
    return result, {
        "exposure1": exposure1,
        "cleaned": cleaned_images["f_lim=2"],
        "exposure2": exposure2,
        "level": level,
        "scatter": float(np.sqrt(gain * max(level, 0.0) + read_noise**2) / gain),
    }


def main() -> None:
    results, images = {}, {}
    for observation in OBSERVATIONS:
        results[observation], images[observation] = measure(observation)
    out_dir = ROOT / "figures"
    (out_dir / "cosmic_rays_real_data.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=1))

    # Figure (M5 field): raw exposure, cleaned exposure (f_lim = 2) and second exposure; common stretch.
    g = images[OBSERVATIONS[0]]
    duration = results[OBSERVATIONS[0]]["data"]["exposure_time_s"][0]

    def stretched(image: np.ndarray) -> np.ndarray:
        return stretch.asinh_stretch(
            image[CUTOUT], black=g["level"] - g["scatter"], alpha=1.0 / (10.0 * g["scatter"]), q=6.0
        )

    high = float(np.percentile(stretched(g["exposure2"]), 99.8))
    panels = [
        (f"First exposure ({duration:.0f} s)", g["exposure1"]),
        ("Cleaned first exposure", g["cleaned"]),
        (f"Second exposure ({duration:.0f} s)", g["exposure2"]),
    ]
    fig, axes = _plotting.figure(
        "web", target="web-640", ratio=0.40, mosaic=[["a", "b", "c"]], gridspec_kw={"wspace": 0.04}
    )
    for ax, (title, image) in zip(axes.values(), panels, strict=True):
        # datawiz: allowed as a relative brightness image; a color bar of asinh-stretched values gives the reader no quantity
        ax.imshow(
            np.clip(stretched(image) / high, 0, 1), origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for edge in ax.spines.values():
            edge.set_visible(False)
        ax.set_title(title, loc="left")
    _plotting.save(fig, str(out_dir / "cosmic_rays_real_data"), target="web-640", formats=("png",))


if __name__ == "__main__":
    main()
