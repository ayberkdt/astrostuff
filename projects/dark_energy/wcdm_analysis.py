"""Constrain the dark energy equation of state from the Pantheon+ supernovae (flat wCDM).

Supernovae only, no Cepheid calibrators: the absolute magnitude M (degenerate with H0) is
marginalized analytically with a flat prior, as in analysis 1 of ``projects/hubble``. The posterior
of (Ωm, w) is computed on a grid with flat priors over the grid ranges and compared with the
published values [Brout2022].

The posterior is curved and extends to very negative w at small Ωm, so the marginal mean and
standard deviation depend on the prior range; the range is part of the record.

Usage:
    python projects/dark_energy/wcdm_analysis.py

Output: ``projects/dark_energy/wcdm_result.json`` and ``projects/dark_energy/wcdm_posterior.npz``.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy

import kozmos
from kozmos.archives import pantheon
from kozmos.cosmology.dark_energy import acceleration_onset_redshift, deceleration_parameter
from kozmos.cosmology.distances import Cosmology
from kozmos.cosmology.hubble_diagram import best_absolute_magnitude, prepare_data

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "pantheon"
OUT_DIR = Path(__file__).resolve().parent

# Brout et al. 2022, arXiv:2202.04077 TeX source, main.tex lines 238-239 (macros \wPanSH and \OMPanSH),
# used in the Flat wCDM row of the cosmology table (line 1128) and in the text at line 1101: supernovae
# alone, without the SH0ES calibration. Read on 26 November 2026.
PUBLISHED = {
    "w": {"value": -0.90, "plus": 0.14, "minus": 0.14},
    "omega_m": {"value": 0.309, "plus": 0.063, "minus": 0.069},
    "source": "Brout2022, Flat wCDM, SN only",
}
H0_NOMINAL = 70.0  # degenerate with M; does not affect the result
OMEGA_M_GRID = np.linspace(0.02, 0.60, 59)
W_GRID = np.linspace(-2.5, -0.3, 111)


def _summary(grid: np.ndarray, density: np.ndarray) -> dict[str, float]:
    density = density / np.trapezoid(density, grid)
    mean = float(np.trapezoid(grid * density, grid))
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    low, median, high = np.interp([0.158655, 0.5, 0.841345], cumulative / cumulative[-1], grid)
    return {
        "mean": mean,
        "std": float(np.sqrt(np.trapezoid((grid - mean) ** 2 * density, grid))),
        "median": float(median),
        "plus": float(high - median),
        "minus": float(median - low),
        "density_at_lower_edge_over_peak": float(density[0] / density.max()),
        "density_at_upper_edge_over_peak": float(density[-1] / density.max()),
    }


def main() -> None:
    start = time.time()
    raw = pantheon.load(CACHE)
    data = prepare_data(raw.columns, 0.5 * (raw.covariance + raw.covariance.T), calibrators=False)

    chi2 = np.array(
        [[best_absolute_magnitude(data, Cosmology.flat(H0_NOMINAL, om, w=w))[2] for w in W_GRID] for om in OMEGA_M_GRID]
    )
    posterior = np.exp(-0.5 * (chi2 - chi2.min()))
    i, j = np.unravel_index(np.argmin(chi2), chi2.shape)
    best = Cosmology.flat(H0_NOMINAL, float(OMEGA_M_GRID[i]), w=float(W_GRID[j]))
    lcdm_chi2 = np.array([best_absolute_magnitude(data, Cosmology.flat(H0_NOMINAL, om))[2] for om in OMEGA_M_GRID])

    w = _summary(W_GRID, np.trapezoid(posterior, OMEGA_M_GRID, axis=0))
    omega_m = _summary(OMEGA_M_GRID, np.trapezoid(posterior, W_GRID, axis=1))
    record = {
        "time": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "duration_s": round(time.time() - start, 1),
        "code": {"kozmos": kozmos.__version__},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "data": {"supernovae": len(data), "z_min": 0.01, "calibrators": False},
        "priors": {
            "omega_m": [float(OMEGA_M_GRID[0]), float(OMEGA_M_GRID[-1])],
            "w": [float(W_GRID[0]), float(W_GRID[-1])],
            "absolute_magnitude": "flat, marginalized analytically",
        },
        "grid": {"omega_m_points": len(OMEGA_M_GRID), "w_points": len(W_GRID)},
        "w": w,
        "omega_m": omega_m,
        "best_fit": {
            "omega_m": best.omega_m,
            "w": best.w,
            "chi2": float(chi2.min()),
            "degrees_of_freedom": len(data) - 3,
            "deceleration_parameter_today": float(deceleration_parameter(0.0, best)),
            "acceleration_onset_redshift": float(acceleration_onset_redshift(best)),
        },
        "flat_lcdm": {
            "chi2": float(lcdm_chi2.min()),
            "omega_m": float(OMEGA_M_GRID[np.argmin(lcdm_chi2)]),
            "delta_chi2_for_one_extra_parameter": float(lcdm_chi2.min() - chi2.min()),
        },
        "published": PUBLISHED,
        "comparison": {
            "w_difference_in_published_sigma": (w["median"] - PUBLISHED["w"]["value"]) / PUBLISHED["w"]["plus"],
            "w_lambda_distance_in_our_sigma": (-1.0 - w["median"]) / w["minus"],
        },
    }
    (OUT_DIR / "wcdm_result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.savez_compressed(OUT_DIR / "wcdm_posterior.npz", omega_m=OMEGA_M_GRID, w=W_GRID, chi2=chi2)
    print(f"wrote {OUT_DIR / 'wcdm_result.json'}")


if __name__ == "__main__":
    main()
