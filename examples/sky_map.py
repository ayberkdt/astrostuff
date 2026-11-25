"""All-sky map of naked-eye stars in galactic coordinates.

Data: Bright Star Catalogue (``kozmos.archives.heasarc.bright_stars``).
The coordinate transformation is done with ``kozmos.coordinates.transforms.icrs_to_galactic``;
the projection is matplotlib's Hammer projection.

Usage:
    python examples/sky_map.py
"""

from __future__ import annotations

from pathlib import Path

import _plotting
import numpy as np

from kozmos.archives import heasarc
from kozmos.coordinates.transforms import icrs_to_galactic
from kozmos.core.angles import wrap_pm_pi

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"HR 2491": "Sirius", "HR 2326": "Canopus", "HR 5340": "Arcturus", "HR 7001": "Vega", "HR 1708": "Capella"}


def main() -> None:
    s = heasarc.bright_stars(ROOT / "data" / "raw" / "heasarc")
    valid = np.isfinite(s["vmag"]) & np.isfinite(s["ra"]) & np.isfinite(s["dec"])
    longitude, latitude = icrs_to_galactic(np.radians(s["ra"][valid]), np.radians(s["dec"][valid]))
    magnitude = s["vmag"][valid]
    x = -wrap_pm_pi(longitude)  # sky map convention: longitude increases to the left

    fig, ax = _plotting.figure("web", target="web-640", ratio=0.56, subplot_kw={"projection": "hammer"})
    size = (
        0.25 + 14.0 * (10 ** (-0.4 * (magnitude - magnitude.min()))) ** 0.55
    )  # flux to the power 0.55: so that faint stars stay visible
    ax.scatter(x, latitude, s=size, color="#374151", linewidths=0, alpha=0.75, rasterized=True)
    ax.set_xticklabels([])
    ax.set_yticks(np.radians([-60, -30, 0, 30, 60]))
    ax.set_yticklabels(["−60°", "−30°", "0°", "30°", "60°"])
    ax.grid(True, color="#d1d5db", lw=0.5)
    for hr, name in NAMES.items():
        i = int(np.flatnonzero(s["name"][valid] == hr)[0])
        ax.annotate(
            name, (x[i], latitude[i]), xytext=(6, 6), textcoords="offset points", color="#c2410c", fontweight="bold"
        )
    ax.set_xlabel("Galactic longitude (center in the middle, increasing to the left)")
    ax.set_ylabel("Galactic latitude")

    out_dir = ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    _plotting.save(fig, str(out_dir / "sky_map"), target="web-640", formats=("png",))

    # The |b| < 20° band covers sin(20°) = 34.2% of the sky; a uniform distribution would put that fraction of stars there.
    band = np.abs(latitude) < np.radians(20.0)
    print(
        f"number of stars {magnitude.size}; {100 * band.mean():.1f}% within |b| < 20° (uniform expectation {100 * np.sin(np.radians(20)):.1f}%)"
    )
    bright = magnitude < 3.0
    print(f"{100 * band[bright].mean():.1f}% of the {bright.sum()} stars with V < 3 are in the same band")


if __name__ == "__main__":
    main()
