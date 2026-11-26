"""Test of the Hubble diagram likelihood with synthetic data.

In a fake supernova catalog generated with known (H0, Ωm, M), the likelihood
must return the input parameters.
"""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from kozmos.cosmology import hubble_diagram
from kozmos.cosmology.distances import Cosmology, distance_modulus

TRUE = {"h0": 72.0, "omega_m": 0.31, "m": -19.3}


def _synthetic(seed=31, n_flow=400, n_cal=30):
    u = np.random.default_rng(seed)
    cosmology = Cosmology.flat(TRUE["h0"], TRUE["omega_m"])
    z_flow = np.sort(u.uniform(0.012, 1.5, n_flow))
    z_cal = u.uniform(0.001, 0.009, n_cal)
    z_nearby = u.uniform(0.002, 0.009, 20)  # neither calibrator nor Hubble flow: must be dropped
    z = np.concatenate([z_cal, z_nearby, z_flow])
    n = z.size
    is_cal = np.zeros(n, bool)
    is_cal[:n_cal] = True
    cepheid = np.where(is_cal, u.uniform(29, 33, n), -9.0)
    mu = np.where(is_cal, cepheid, distance_modulus(z, cosmology))
    a = u.normal(0, 0.02, (n, 5))
    cov = np.diag(u.uniform(0.1, 0.2, n) ** 2) + a @ a.T  # diagonal + low-rank correlated term
    magnitude = mu + TRUE["m"] + u.multivariate_normal(np.zeros(n), cov)
    columns = {"zHD": z, "IS_CALIBRATOR": is_cal.astype(float), "m_b_corr": magnitude, "CEPH_DIST": cepheid}
    return columns, cov


def test_selection():
    columns, cov = _synthetic()
    sn_only = hubble_diagram.prepare_data(columns, cov, calibrators=False)
    with_cepheids = hubble_diagram.prepare_data(columns, cov, calibrators=True)
    assert len(sn_only) == 400 and not sn_only.calibrator.any()
    assert len(with_cepheids) == 430 and with_cepheids.calibrator.sum() == 30
    assert np.all(sn_only.z > 0.01)


def test_closed_form_absolute_magnitude_matches_numerical_minimization():
    columns, cov = _synthetic()
    data = hubble_diagram.prepare_data(columns, cov, calibrators=True)
    cosmology = Cosmology.flat(TRUE["h0"], TRUE["omega_m"])
    m, sigma, chi2 = hubble_diagram.best_absolute_magnitude(data, cosmology)
    numeric = minimize_scalar(
        lambda x: -hubble_diagram.log_likelihood(data, cosmology, x),
        bounds=(-20, -18.5),
        method="bounded",
        options={"xatol": 1e-10},
    )
    assert m == pytest.approx(numeric.x, abs=1e-7)
    assert chi2 == pytest.approx(2 * numeric.fun, rel=1e-9, abs=0)
    assert abs(m - TRUE["m"]) < 4 * sigma
    # Definition of Δχ² = 1: χ² increases by one at M̂ ± σ_M.
    assert -2 * hubble_diagram.log_likelihood(data, cosmology, m + sigma) - chi2 == pytest.approx(1.0, rel=1e-6, abs=0)


def test_parameter_recovery():
    columns, cov = _synthetic()
    data = hubble_diagram.prepare_data(columns, cov, calibrators=True)

    def profile(om):
        # Minimum χ² over H0 and M for each Ωm: nested one-dimensional minimization.
        inner = minimize_scalar(
            lambda h0: hubble_diagram.best_absolute_magnitude(data, Cosmology.flat(h0, om))[2],
            bounds=(60, 85),
            method="bounded",
        )
        return inner.fun, inner.x

    grid = np.linspace(0.2, 0.45, 26)
    results = [profile(om) for om in grid]
    chi2 = np.array([s[0] for s in results])
    best = int(np.argmin(chi2))
    inside = grid[chi2 - chi2.min() < 4.0]  # roughly the 2σ range
    assert inside.min() <= TRUE["omega_m"] <= inside.max()
    assert abs(results[best][1] - TRUE["h0"]) < 3.0
    assert chi2.min() / (len(data) - 3) == pytest.approx(1.0, abs=0.2)


def test_h0_and_m_degeneracy_without_cepheids():
    # Without calibrators a change in H0 is fully compensated by a shift in M: χ²_min does not depend on H0.
    columns, cov = _synthetic()
    data = hubble_diagram.prepare_data(columns, cov, calibrators=False)
    m1, _, chi2_1 = hubble_diagram.best_absolute_magnitude(data, Cosmology.flat(60.0, 0.31))
    m2, _, chi2_2 = hubble_diagram.best_absolute_magnitude(data, Cosmology.flat(80.0, 0.31))
    assert chi2_1 == pytest.approx(chi2_2, rel=1e-9, abs=0)
    # As H0 grows the model distance shrinks and μ decreases; M = m − μ rises by the same amount.
    assert m2 - m1 == pytest.approx(5 * np.log10(80.0 / 60.0), abs=1e-8)
