"""Optical and infrared images of the same field: the Flame Nebula (NGC 2024).

In star-forming regions dust absorbs visible light; in the near infrared the
extinction is much smaller and young stars inside the dust become visible; in
the mid infrared the emission of the heated dust itself dominates.

The images are downloaded from NASA SkyView (DSS2 Red, 2MASS J/H/K, WISE 12 µm),
background and source detection are done with ``kozmos.photometry``, and the
stretch [Lupton2004] is asinh.

Usage:
    python examples/optical_infrared.py
"""

from __future__ import annotations

from pathlib import Path

import _plotting
import matplotlib.pyplot as plt
import numpy as np

from kozmos import photometry
from kozmos.archives import skyview
from kozmos.visualization import stretch

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "skyview"
RA, DEC, FIELD_DEG, PIXELS = 85.42, -1.90, 0.3, 450
PALETTE = plt.get_cmap("gray").with_extremes(bad="#7c2d12")  # NaN pixels in brick red
PANELS = [
    ("DSS2 Red", "Optical, 0.64 µm"),
    ("2MASS-K", "Near infrared, 2.16 µm"),
    ("WISE 12", "Mid infrared, 12 µm"),
]


def prepare(survey: str) -> tuple[np.ndarray, int]:
    """Download the cutout, subtract the background, count sources and stretch to the range [0, 1]."""
    image, _ = skyview.download_cutout(RA, DEC, survey, CACHE, size_deg=FIELD_DEG, pixels=PIXELS)
    background, rms = photometry.background_map(image, cell=75)
    detected = photometry.detect_sources(image, background, rms, threshold_sigma=4.0, min_pixels=6)
    level, scatter = photometry.estimate_background(image)
    stretched = stretch.asinh_stretch(image, black=level - scatter, alpha=1.0 / (8.0 * scatter), q=6.0)
    scaled = stretch.percentile_scale(stretched, 0.5, 99.9)
    scaled[~np.isfinite(image)] = np.nan  # pixels without data are not painted black, they are shown separately
    return scaled, len(detected)


def main() -> None:
    out_dir = ROOT / "figures"
    out_dir.mkdir(exist_ok=True)

    fig, axes = _plotting.figure(
        "web", target="web-640", ratio=0.40, mosaic=[["a", "b", "c"]], gridspec_kw={"wspace": 0.04}
    )
    counts = {}
    for ax, (survey, title) in zip(axes.values(), PANELS, strict=True):
        image, n = prepare(survey)
        counts[survey] = n
        # datawiz: allowed as a relative brightness image; a color bar of asinh-stretched values gives the reader no quantity
        ax.imshow(image, origin="lower", cmap=PALETTE, vmin=0, vmax=1, interpolation="nearest")
        empty = ~np.isfinite(image)
        if empty.any():
            yy, xx = np.nonzero(empty)
            ax.text(xx.mean(), yy.mean(), "no data", color="white", ha="center", va="center")
            print(f"{survey}: {int(empty.sum())} pixels carry no data (NaN)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for edge in ax.spines.values():
            edge.set_visible(False)
        ax.set_title(f"{title}\n{n} sources detected", loc="left")
    # Scale bar: 5 arcminutes.
    ax = axes["a"]
    length = 5.0 / 60.0 / FIELD_DEG * PIXELS
    ax.plot([22, 22 + length], [24, 24], color="white", lw=2, solid_capstyle="butt")
    ax.text(22, 36, "5′", color="white", va="bottom")
    _plotting.save(fig, str(out_dir / "optical_infrared"), target="web-640", formats=("png",))
    print("source counts:", counts)

    # Near-infrared color composite: K → red, H → green, J → blue.
    bands = []
    for survey in ("2MASS-K", "2MASS-H", "2MASS-J"):
        image, _ = skyview.download_cutout(RA, DEC, survey, CACHE, size_deg=FIELD_DEG, pixels=PIXELS)
        background, rms = photometry.background_map(image, cell=75)
        bands.append((image - background) / np.median(rms))  # scale to noise units: the bands become comparable
    rgb = stretch.lupton_rgb(*bands, black=-1.0, alpha=0.06, q=8.0)
    plt.imsave(out_dir / "flame_nebula_jhk.png", np.clip(rgb, 0, 1), origin="lower")
    print("color composite:", out_dir / "flame_nebula_jhk.png", rgb.shape)


if __name__ == "__main__":
    main()
