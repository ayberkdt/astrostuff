"""Closed-form check of the H0 uncertainty, independent of the chain.

Question: why does the chain in this repository give H0 = 73.53 ± 1.00 while
the published value is 73.6 ± 1.1 [Brout2022]?

Method: with Ωm fixed, the model is linear in the parameters M and
x = 5 log10(H0 / 70). For supernovae in the Hubble flow the residual is
m − M − μ(z; 70, Ωm) + x, for calibrators m − M − μ_Cepheid. In a linear
Gaussian model the parameter covariance is (Aᵀ C⁻¹ A)⁻¹; it does not depend
on the sampler, the chain length or convergence. The contribution of Ωm is
added through the law of total variance from the scatter of the best x on the
Ωm grid.

The second check is the published chain itself: the four
``Pantheon+SH0ES_*.txt.gz`` chains (emcee) of the Pantheon+ data release are
downloaded and the H0 scatter is measured with and without discarding the
beginning of the chain.

Usage:
    python projects/hubble/h0_uncertainty_check.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from scipy import linalg

from kozmos.archives import cache, pantheon
from kozmos.cosmology.distances import Cosmology, distance_modulus, transverse_comoving_distance_mpc

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
H0_NOMINAL = 70.0


def solve_closed_form(
    s: dict, covariance: np.ndarray, z_min: float, z_max: float = np.inf, use_zhel: bool = False
) -> dict[str, float]:
    """Closed-form solution. ``use_zhel``: the distance definition of the CosmoSIS likelihood in the data release,
    d_L = (1 + z_HEL) D_M(z_HD); if false, d_L = (1 + z_HD) D_M(z_HD)."""
    z = s["zHD"]
    calibrator = s["IS_CALIBRATOR"] == 1
    selection = calibrator | ((z > z_min) & (z < z_max))
    z, calibrator = z[selection], calibrator[selection]
    m = s["m_b_corr"][selection]
    factor = linalg.cho_factor(covariance[np.ix_(selection, selection)], lower=True)
    design = np.column_stack([np.ones(z.size), np.where(calibrator, 0.0, -1.0)])  # columns: M, x
    parameter_covariance = np.linalg.inv(design.T @ linalg.cho_solve(factor, design))

    def fit(omega_m: float) -> tuple[float, float]:
        cosmology = Cosmology.flat(H0_NOMINAL, omega_m)
        if use_zhel:
            model = 5.0 * np.log10((1.0 + s["zHEL"][selection]) * transverse_comoving_distance_mpc(z, cosmology)) + 25.0
        else:
            model = distance_modulus(z, cosmology)
        mu = np.where(calibrator, s["CEPH_DIST"][selection], model)
        p = parameter_covariance @ (design.T @ linalg.cho_solve(factor, m - mu))
        residual = m - mu - design @ p
        return float(p[1]), float(residual @ linalg.cho_solve(factor, residual))

    grid = np.linspace(0.20, 0.50, 121)
    x, chi2 = np.array([fit(o) for o in grid]).T
    weight = np.exp(-0.5 * (chi2 - chi2.min()))
    weight /= weight.sum()
    x_mean = float(weight @ x)
    # Total variance: variance at fixed Ωm + variance of the best x across the Ωm posterior.
    var_x = float(parameter_covariance[1, 1] + weight @ (x - x_mean) ** 2)
    h0 = H0_NOMINAL * 10 ** (x_mean / 5.0)
    conversion = h0 * np.log(10.0) / 5.0
    return {
        "n_sn": int(selection.sum()),
        "calibrators": int(calibrator.sum()),
        "h0": h0,
        "sigma_h0_omega_m_fixed": float(conversion * np.sqrt(parameter_covariance[1, 1])),
        "sigma_h0": float(conversion * np.sqrt(var_x)),
        "omega_m": float(weight @ grid),
    }


CHAIN_URL_ROOT = (
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/5_COSMOLOGY/chains/"
    "Pantheon%2BSH0ES/Pantheon%2BSH0ES_"
)
# Values in the paper: Brout et al. (2022) TeX macros \fLCDMHPanSH, \oLCDMHPanSH, \HPanSH, \wwaHPanSH.
PUBLISHED = {
    "FlatLambdaCDM": (73.6, 1.1),
    "LambdaCDM": (73.4, 1.1),
    "FlatwCDM": (73.5, 1.1),
    "Flatw0waCDM": (73.3, 1.1),
}


def published_chains() -> dict[str, object]:
    """H0 mean and scatter of the four chains in the data release, with and without discarding the beginning."""
    result = {}
    for model, (h0_paper, sigma_paper) in PUBLISHED.items():
        path = cache.download(
            f"{CHAIN_URL_ROOT}{model}.txt.gz",
            ROOT / "data" / "raw" / "pantheon" / f"Pantheon+SH0ES_{model}.txt.gz",
            attribution="Pantheon+SH0ES data release, 5_COSMOLOGY/chains",
        )
        with gzip.open(path, "rt") as file:
            header = file.readline().lstrip("#").split()
            sampler = file.readline().strip()
        chain = np.loadtxt(path, comments="#")
        h0 = 100.0 * chain[:, header.index("cosmological_parameters--h0")]
        n = h0.size
        slices = {}
        for name, start in (
            ("all", 0.0),
            ("first_10_percent_discarded", 0.10),
            ("first_25_percent_discarded", 0.25),
            ("second_half", 0.5),
        ):
            s = h0[int(start * n) :]
            slices[name] = {"samples": int(s.size), "h0": float(s.mean()), "sigma_h0": float(s.std())}
        result[model] = {
            "paper": {"h0": h0_paper, "sigma_h0": sigma_paper},
            "sampler_line": sampler,
            "n_samples": int(n),
            "slices": slices,
        }
    return result


def main() -> None:
    raw = pantheon.load(ROOT / "data" / "raw" / "pantheon")
    s = raw.columns
    stat_only = pantheon.load(ROOT / "data" / "raw" / "pantheon", stat_only=True).covariance
    result = {
        "description": "closed-form linear Gaussian solution; no sampler",
        "published": {"h0": 73.6, "sigma": 1.1, "source": "Brout2022 TeX macro \\fLCDMHPanSH"},
        "chain": json.loads((HERE / "result.json").read_text(encoding="utf-8"))["analysis_2_sn_and_cepheids"]["h0"],
        "options": {
            "stat+sys, z > 0.01 (same selection as the chain)": solve_closed_form(s, raw.covariance, 0.01),
            "stat+sys, z > 0.01, distance definition of the published likelihood (zHEL)": solve_closed_form(
                s, raw.covariance, 0.01, use_zhel=True
            ),
            "stat+sys, no cut (all supernovae)": solve_closed_form(s, raw.covariance, -1.0),
            "stat+sys, z > 0.0233": solve_closed_form(s, raw.covariance, 0.0233),
            "stat+sys, 0.0233 < z < 0.15": solve_closed_form(s, raw.covariance, 0.0233, 0.15),
            "stat only, z > 0.01": solve_closed_form(s, stat_only, 0.01),
        },
    }
    result["published_chains"] = published_chains()
    (HERE / "h0_uncertainty_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, c in result["options"].items():
        print(
            f"{name:68s} N={c['n_sn']:5d} H0={c['h0']:.2f} ± {c['sigma_h0']:.3f} "
            f"(Ωm fixed: {c['sigma_h0_omega_m_fixed']:.3f})  Ωm={c['omega_m']:.3f}"
        )
    print("chain:", result["chain"]["mean"], result["chain"]["std"])
    for model, z in result["published_chains"].items():
        whole, trimmed = z["slices"]["all"], z["slices"]["first_10_percent_discarded"]
        print(
            f"{model:14s} {z['sampler_line']} paper {z['paper']['h0']} ± {z['paper']['sigma_h0']} | whole chain "
            f"{whole['h0']:.2f} ± {whole['sigma_h0']:.3f} | first 10% discarded {trimmed['h0']:.2f} ± {trimmed['sigma_h0']:.3f}"
        )


if __name__ == "__main__":
    main()
