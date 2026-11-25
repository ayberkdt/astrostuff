"""Estimate Ωm and H0 from the Pantheon+SH0ES data.

Two analyses are done and the result is compared with the published values [Brout2022]:

1. Supernovae only, flat ΛCDM: Ωm. The absolute magnitude M (degenerate with
   H0) is marginalized analytically with a flat prior; the Ωm posterior is
   computed on a grid.
2. Cepheid calibrators added: MCMC for (H0, Ωm, M) with our own ensemble
   sampler.

Usage:
    python projects/hubble/hubble_analysis.py [--steps 20000] [--seed 20261125]

Output: ``projects/hubble/result.json`` (run record and results) and
``projects/hubble/chain.npy``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy

import kozmos
from kozmos.archives import pantheon
from kozmos.cosmology.distances import Cosmology
from kozmos.cosmology.hubble_diagram import HubbleData, best_absolute_magnitude, log_likelihood, prepare_data
from kozmos.statistics.mcmc import autocorrelation_time, rank_normalized_rhat, sample_until_converged, split_rhat

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "pantheon"
OUT_DIR = Path(__file__).resolve().parent

# Published values: Brout et al. 2022, arXiv:2202.04077 TeX source (main.tex,
# macros \fLCDMOMPan and \fLCDMHPanSH), read on 25 November 2026.
PUBLISHED = {
    "omega_m_flat_lcdm": {"value": 0.334, "uncertainty": 0.018, "source": "Brout2022, flat ΛCDM, SN only"},
    "h0_flat_lcdm": {"value": 73.6, "uncertainty": 1.1, "source": "Brout2022, flat ΛCDM, Pantheon+ and SH0ES"},
}
H0_NOMINAL = 70.0  # Degenerate with M in analysis 1; does not affect the result.


def omega_m_grid(data: HubbleData, grid: np.ndarray) -> dict[str, float | list[float]]:
    """Analysis 1: Ωm posterior with M marginalized (flat prior, 0 < Ωm < 1)."""
    chi2 = np.array([best_absolute_magnitude(data, Cosmology.flat(H0_NOMINAL, om))[2] for om in grid])
    density = np.exp(-0.5 * (chi2 - chi2.min()))
    density /= np.trapezoid(density, grid)
    mean = float(np.trapezoid(grid * density, grid))
    deviation = float(np.sqrt(np.trapezoid((grid - mean) ** 2 * density, grid)))
    return {
        "mean": mean,
        "std": deviation,
        "most_probable": float(grid[np.argmin(chi2)]),
        "chi2_min": float(chi2.min()),
        "degrees_of_freedom": len(data) - 2,
        "grid": grid.tolist(),
        "density": density.tolist(),
    }


def h0_mcmc(data: HubbleData, max_steps: int, rng: np.random.Generator) -> tuple[dict[str, object], np.ndarray]:
    """Analysis 2: ensemble sampler for (H0, Ωm, M), extended until R-hat < 1.01 and length > 50 τ."""

    def log_posterior(p: np.ndarray) -> float:
        h0, om, m = p
        if not (50.0 < h0 < 100.0 and 0.05 < om < 0.95 and -21.0 < m < -18.0):
            return -np.inf
        return log_likelihood(data, Cosmology.flat(h0, om), m)

    walkers = 32
    initial = np.array([73.0, 0.33, -19.25]) + rng.normal(0, [0.5, 0.01, 0.01], (walkers, 3))
    chain, discard, converged = sample_until_converged(
        log_posterior, initial, rng, block_steps=1000, max_steps=max_steps
    )
    steps = chain.samples.shape[0]
    tau = autocorrelation_time(chain.samples)
    kept = chain.samples[discard:]
    flat = kept.reshape(-1, 3)
    names = ["h0", "omega_m", "absolute_magnitude"]
    result: dict[str, object] = {
        name: {
            "mean": float(flat[:, i].mean()),
            "std": float(flat[:, i].std(ddof=1)),
            "percentiles_16_50_84": [float(q) for q in np.percentile(flat[:, i], [16, 50, 84])],
        }
        for i, name in enumerate(names)
    }
    result["diagnostics"] = {
        "steps": steps,
        "max_steps": max_steps,
        "converged": converged,
        "criteria": "rank-normalized folded split R-hat < 1.01 for every parameter and steps > 50 tau",
        "walkers": walkers,
        "burn_in_discarded": discard,
        "autocorrelation_time": tau.tolist(),
        "effective_sample_size": (kept.shape[0] * walkers / tau).tolist(),
        "split_rhat": split_rhat(kept).tolist(),
        "rank_normalized_rhat": rank_normalized_rhat(kept).tolist(),
        "mean_acceptance_fraction": float(chain.acceptance_fraction.mean()),
        "chain_long_enough": bool(steps > 50 * tau.max()),
    }
    best = flat[np.argmax(chain.log_probability[discard:].reshape(-1))]
    result["chi2_min"] = float(-2 * chain.log_probability[discard:].max())
    result["degrees_of_freedom"] = len(data) - 3
    result["most_probable"] = dict(zip(names, map(float, best), strict=True))
    return result, chain.samples


def git_summary() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        return commit + (" (working tree dirty)" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no git repository"


def sigma_difference(value: float, deviation: float, published: dict[str, object]) -> float:
    return (value - float(published["value"])) / float(np.hypot(deviation, float(published["uncertainty"])))  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=20000, help="upper limit; the run stops when converged")
    parser.add_argument("--seed", type=int, default=20261125)
    options = parser.parse_args()
    start = time.time()
    rng = np.random.default_rng(options.seed)

    raw = pantheon.load(CACHE)
    asymmetry = float(np.abs(raw.covariance - raw.covariance.T).max())
    covariance = 0.5 * (raw.covariance + raw.covariance.T)

    sn_only = prepare_data(raw.columns, covariance, calibrators=False)
    a1 = omega_m_grid(sn_only, np.linspace(0.15, 0.60, 181))
    print(
        f"Analysis 1 (SN only, N={len(sn_only)}): Ωm = {a1['mean']:.3f} ± {a1['std']:.3f}"
        f"   χ²/dof = {a1['chi2_min']:.1f}/{a1['degrees_of_freedom']}"
    )

    with_cepheids = prepare_data(raw.columns, covariance, calibrators=True)
    a2, samples = h0_mcmc(with_cepheids, options.steps, rng)
    h0 = a2["h0"]
    print(
        f"Analysis 2 (SN + Cepheids, N={len(with_cepheids)}, calibrators={int(with_cepheids.calibrator.sum())}): "
        f"H0 = {h0['mean']:.2f} ± {h0['std']:.2f}   "  # type: ignore[index]
        f"Ωm = {a2['omega_m']['mean']:.3f} ± {a2['omega_m']['std']:.3f}"
    )  # type: ignore[index]
    print(f"  diagnostics: {a2['diagnostics']}")

    source_record = json.loads((CACHE / "SOURCE_RECORD.json").read_text(encoding="utf-8"))
    record = {
        "time": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "duration_s": round(time.time() - start, 1),
        "code": {"kozmos": kozmos.__version__, "git": git_summary()},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "parameters": {"steps": options.steps, "seed": options.seed, "z_min": 0.01, "h0_nominal": H0_NOMINAL},
        "data": {file: {"sha256": k["sha256"], "bytes": k["bytes"]} for file, k in source_record.items()},
        "covariance_max_asymmetry": asymmetry,
        "analysis_1_sn_only": {k: v for k, v in a1.items() if k not in ("grid", "density")},
        "analysis_2_sn_and_cepheids": a2,
        "published": PUBLISHED,
        "comparison_sigma": {
            "omega_m": sigma_difference(a1["mean"], a1["std"], PUBLISHED["omega_m_flat_lcdm"]),  # type: ignore[arg-type]
            "h0": sigma_difference(h0["mean"], h0["std"], PUBLISHED["h0_flat_lcdm"]),  # type: ignore[index]
        },
    }
    (OUT_DIR / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(OUT_DIR / "chain.npy", samples.astype(np.float32))
    np.savez(OUT_DIR / "omega_m_posterior.npz", grid=np.array(a1["grid"]), density=np.array(a1["density"]))
    print(f"record: {OUT_DIR / 'result.json'}  ({record['duration_s']} s)")
    print(
        f"difference from published: Ωm {record['comparison_sigma']['omega_m']:+.2f}σ, H0 {record['comparison_sigma']['h0']:+.2f}σ"
    )


if __name__ == "__main__":
    main()
