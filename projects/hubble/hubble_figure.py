"""Hubble diagram for the README: Pantheon+ distance moduli and the best-fit curve.

Input: ``result.json`` (output of hubble_analysis.py) and the data in the local cache.
Output: ``figures/hubble_diagram.png`` (+ PDF and audit report).

The error bars in the figure come from the covariance diagonal and are for
visual purposes only; the fit was done with the full covariance (README
warning of the data repository).

Usage:
    python projects/hubble/hubble_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from kozmos.archives import pantheon
from kozmos.cosmology.distances import Cosmology, distance_modulus

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
import _plotting  # noqa: E402

RESULT = json.loads((Path(__file__).resolve().parent / "result.json").read_text(encoding="utf-8"))


def main() -> None:
    raw = pantheon.load(ROOT / "data" / "raw" / "pantheon")
    s = raw.columns
    flow = (s["zHD"] > 0.01) & (s["IS_CALIBRATOR"] == 0)
    z = s["zHD"][flow]
    a2 = RESULT["analysis_2_sn_and_cepheids"]
    h0, om, m = a2["h0"]["mean"], a2["omega_m"]["mean"], a2["absolute_magnitude"]["mean"]
    cosmology = Cosmology.flat(h0, om)
    mu = s["m_b_corr"][flow] - m  # distance modulus with the absolute magnitude found by this analysis
    residual = mu - distance_modulus(z, cosmology)

    fig, axes = _plotting.figure(
        "web", target="web-640", ratio=0.82, mosaic=[["upper"], ["upper"], ["lower"]], sharex=True
    )
    high, low = axes["upper"], axes["lower"]
    points = dict(s=5, alpha=0.35, linewidths=0, color="#6b7280", rasterized=True)

    high.scatter(z, mu, **points)
    zz = np.logspace(-2, np.log10(2.4), 300)
    (curve,) = high.plot(zz, distance_modulus(zz, cosmology), color="#c2410c", lw=1.8)
    high.set_xscale(
        "log"
    )  # datawiz: allowed as the usual scale of a Hubble diagram; ticks are written as plain numbers
    high.set_ylabel("Distance modulus (mag)")
    high.annotate(
        f"Flat ΛCDM fit\n$H_0$ = {h0:.1f}, $Ω_m$ = {om:.2f}",
        xy=(0.3, float(distance_modulus(0.3, cosmology))),
        xytext=(0.035, 43.3),
        color="#c2410c",
        arrowprops=dict(arrowstyle="-", color="#c2410c", lw=0.8, shrinkB=3),
    )
    high.text(0.25, 35.4, f"{flow.sum()} Type Ia supernovae\n(Pantheon+, z > 0.01)", color="#4b5563", va="top")

    low.scatter(z, residual, **points)
    low.axhline(0.0, color="#c2410c", lw=1.4)
    # Binned median residual: is there a trend?
    edges = np.logspace(-2, np.log10(2.3), 13)
    centers = np.sqrt(edges[:-1] * edges[1:])
    bin_index = np.digitize(z, edges) - 1
    medians = np.array(
        [np.median(residual[bin_index == i]) if np.sum(bin_index == i) >= 5 else np.nan for i in range(centers.size)]
    )
    low.plot(centers, medians, "o", ms=4.5, color="#111827", zorder=5)
    low.set_ylim(-0.75, 0.75)
    low.set_ylabel("Residual (mag)")
    low.set_xlabel("Redshift z")
    low.text(0.012, 0.66, "black: bin median", color="#111827", va="top")
    low.set_xlim(0.0095, 2.6)
    ticks = [0.01, 0.03, 0.1, 0.3, 1.0, 2.0]
    low.set_xticks(ticks)
    low.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "1", "2"])
    low.minorticks_off()

    out_dir = ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    report = _plotting.save(fig, str(out_dir / "hubble_diagram"), target="web-640", cvd=True)
    if report is not None:
        # The datawiz report is keyed in Turkish; show only its counts and file list.
        print({k: v for k, v in report.items() if isinstance(v, int | list)})
    print(
        f"residual: median {np.median(residual):+.4f} mag, robust scatter {1.4826 * np.median(np.abs(residual - np.median(residual))):.3f} mag"
    )


if __name__ == "__main__":
    main()
