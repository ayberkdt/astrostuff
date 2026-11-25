import numpy as np
import pytest
from scipy import stats

from kozmos.statistics import mcmc


def test_stretch_factor_distribution():
    # Cumulative distribution for g(z) ∝ 1/√z, z ∈ [1/a, a]: F(z) = (√z − 1/√a) / (√a − 1/√a).
    a = 2.0
    z = mcmc.stretch_factor(np.random.default_rng(21), 20000, a)
    assert z.min() >= 1 / a and z.max() <= a
    cumulative = lambda x: (np.sqrt(x) - a**-0.5) / (a**0.5 - a**-0.5)  # noqa: E731
    assert stats.kstest(z, cumulative).pvalue > 1e-3
    with pytest.raises(ValueError):
        mcmc.stretch_factor(np.random.default_rng(0), 1, 1.0)


def test_correlated_gaussian_recovery():
    mean = np.array([1.0, -2.0, 0.5])
    cov = np.array([[1.0, 0.8, 0.0], [0.8, 2.0, -0.3], [0.0, -0.3, 0.5]])
    inverse = np.linalg.inv(cov)
    log_p = lambda x: -0.5 * (x - mean) @ inverse @ (x - mean)  # noqa: E731
    u = np.random.default_rng(22)
    chain = mcmc.ensemble_sample(log_p, mean + u.normal(0, 0.1, (24, 3)), 3000, u)
    tau = mcmc.autocorrelation_time(chain.samples)
    assert 3000 > 50 * tau.max()
    sample = chain.flatten(discard=500)
    effective = sample.shape[0] / tau.max()
    assert np.all(np.abs(sample.mean(axis=0) - mean) < 5 * np.sqrt(np.diag(cov) / effective))
    assert np.cov(sample.T) == pytest.approx(cov, abs=0.15)
    assert np.all(mcmc.split_rhat(chain.samples[500:]) < 1.05)
    assert 0.2 < chain.acceptance_fraction.mean() < 0.9


def test_initial_check():
    u = np.random.default_rng(23)
    log_p = lambda x: -0.5 * float(x @ x)  # noqa: E731
    with pytest.raises(ValueError):
        mcmc.ensemble_sample(log_p, u.normal(size=(3, 2)), 10, u)  # odd number of walkers
    with pytest.raises(ValueError):
        mcmc.ensemble_sample(lambda x: -np.inf, u.normal(size=(4, 2)), 10, u)


def test_autocorrelation_time_ar1():
    # Integrated autocorrelation time of an AR(1) process: (1+ρ)/(1−ρ).
    rho, n_steps, k = 0.9, 40000, 8
    u = np.random.default_rng(24)
    x = np.zeros((n_steps, k, 1))
    noise = u.normal(size=(n_steps, k))
    for t in range(1, n_steps):
        x[t, :, 0] = rho * x[t - 1, :, 0] + noise[t]
    assert mcmc.autocorrelation_time(x)[0] == pytest.approx((1 + rho) / (1 - rho), rel=0.15)


def test_rhat_detects_separated_chains():
    u = np.random.default_rng(25)
    good = u.normal(size=(1000, 8, 2))
    assert np.all(np.abs(mcmc.split_rhat(good) - 1) < 0.02)
    bad = good.copy()
    bad[:, :4, 0] += 3.0  # half of the walkers sit on a different peak
    assert mcmc.split_rhat(bad)[0] > 1.1


def test_rank_normalized_rhat():
    u = np.random.default_rng(26)
    good = u.normal(size=(1000, 8, 2))
    r = mcmc.rank_normalized_rhat(good)
    assert np.all(np.abs(r - 1) < 0.02)
    # Half of the walkers at a different location: the rank-normalized component catches it.
    shifted = good.copy()
    shifted[:, :4, 0] += 3.0
    assert mcmc.rank_normalized_rhat(shifted)[0] > 1.1
    # Chains with the same mean but different scales: classical split R̂ misses them, folding catches them.
    scaled = good.copy()
    scaled[:, :4, 1] *= 4.0
    assert mcmc.split_rhat(scaled)[1] < 1.02
    assert mcmc.rank_normalized_rhat(scaled)[1] > 1.1
    # Also defined for infinite-variance (Cauchy) samples, and close to 1.
    cauchy = u.standard_cauchy(size=(2000, 8, 1))
    assert abs(mcmc.rank_normalized_rhat(cauchy)[0] - 1) < 0.02
