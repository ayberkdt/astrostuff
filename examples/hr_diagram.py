"""Hertzsprung-Russell diagram of the Sun's neighborhood from Gaia DR3.

Data: stars within 50 pc with parallax S/N > 20 and RUWE < 1.4
(``kozmos.archives.gaia.nearby_stars``). Absolute magnitude
M_G = G + 5 + 5 log10(ϖ/1000) [Babusiaux2018]; extinction is not corrected.

Usage:
    python examples/hr_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import _plotting
import numpy as np

from kozmos.archives import gaia

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    s = gaia.nearby_stars(ROOT / "data" / "raw" / "gaia")
    color = s["bp_rp"]
    magnitude = gaia.absolute_magnitude(s["phot_g_mean_mag"], s["parallax"])

    fig, ax = _plotting.figure("web", target="web-640", ratio=1.05)
    ax.scatter(color, magnitude, s=1.6, alpha=0.35, linewidths=0, color="#6b7280", rasterized=True)
    ax.set_xlim(-0.7, 5.2)
    ax.set_ylim(17.5, -0.5)  # bright at the top
    ax.set_xlabel("Color, BP − RP (mag)")
    ax.set_ylabel("Absolute magnitude, G band (mag)")
    labels = dict(color="#c2410c", fontweight="bold")
    ax.text(2.55, 6.3, "Main sequence", **labels)
    ax.text(0.95, 13.6, "White dwarfs", **labels)
    ax.text(1.42, 2.1, "Giants", **labels)
    count = f"{color.size:,}"
    ax.text(3.05, 1.2, count + " stars\nwithin 50 pc of the Sun", color="#4b5563", va="top")

    out_dir = ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    _plotting.save(fig, str(out_dir / "hr_diagram"), target="web-640", formats=("png",))

    # Region counts: rough boxes, for the caption only.
    white_dwarf = (magnitude > 10.0 + 2.6 * color) & (color < 1.9)
    giant = (magnitude < 3.0) & (color > 0.85)
    print(f"total {color.size}, white dwarf region {white_dwarf.sum()}, giant region {giant.sum()}")
    print(
        f"color range {np.nanmin(color):.2f}..{np.nanmax(color):.2f}, magnitude range {magnitude.min():.1f}..{magnitude.max():.1f}"
    )


if __name__ == "__main__":
    main()
