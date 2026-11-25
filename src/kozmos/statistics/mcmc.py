"""Affine-invariant ensemble sampler and chain diagnostics.

The stretch move and the parallel update scheme were read from the arXiv TeX
source (1202.3665) of the emcee paper [ForemanMackey2013]; the original source
of the method is the paper [GoodmanWeare2010], which was not read in this work.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtri
from scipy.stats import rankdata

from kozmos._sources import source

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Chain:
    """Output of the ensemble sampler.

    Attributes
    ----------
    samples : ndarray, shape (step, walker, dimension)
    log_probability : ndarray, shape (step, walker)
    acceptance_fraction : ndarray, shape (walker,)
    """

    samples: Array
    log_probability: Array
    acceptance_fraction: Array

    def flatten(self, discard: int = 0, thin: int = 1) -> Array:
        """Discard the burn-in steps and merge the walkers into a single sample array."""
        return self.samples[discard::thin].reshape(-1, self.samples.shape[-1])


@source(
    "ForemanMackey2013",
    where="Eq. goodmanprop (g(z) ∝ 1/√z, z ∈ [1/a, a])",
    status="read",
    note="inverse transform sampling: z = ((a−1)u + 1)² / a, u ~ U(0,1); derivation in this project",
)
def stretch_factor(rng: np.random.Generator, size: int | tuple[int, ...], a: float = 2.0) -> Array:
    """Sample the scale factor of the stretch move from the distribution g(z) ∝ 1/√z.

    The cumulative distribution F(z) = (√z − 1/√a) / (√a − 1/√a) is inverted:
    z = ((a − 1) u + 1)² / a.

    References
    ----------
    [ForemanMackey2013] Eq. goodmanprop.

    Validity
    --------
    a > 1.

    Verification
    ------------
    tests/unit/test_mcmc.py: range of the samples and their cumulative distribution (KS test).
    """
    if a <= 1.0:
        raise ValueError("a must be greater than 1")
    u = rng.random(size)
    return ((a - 1.0) * u + 1.0) ** 2 / a


@source(
    "ForemanMackey2013",
    where="Eq. proposal, acceptance; Algorithm 3 (parallel stretch move)",
    status="read",
    note="Y = X_j + Z (X_k − X_j); acceptance q = min(1, Z^{N−1} p(Y)/p(X_k)); the ensemble is split into two halves",
)
def ensemble_sample(
    log_probability: Callable[[Array], float],
    initial: ArrayLike,
    n_steps: int,
    rng: np.random.Generator,
    *,
    a: float = 2.0,
) -> Chain:
    """Run the affine-invariant ensemble sampler (stretch move).

    At each step the ensemble is split into two halves; every walker in one
    half is updated relative to a walker chosen at random from the other half.

    Parameters
    ----------
    log_probability : callable
        Takes a parameter vector and returns the log posterior density;
        ``-inf`` outside the domain.
    initial : ndarray, shape (walker, dimension)
        Initial positions. The number of walkers must be even and at least twice the dimension.
    n_steps : int
    rng : numpy.random.Generator
    a : float
        Stretch scale; the value recommended in the source is 2.

    References
    ----------
    [ForemanMackey2013] Eq. proposal, acceptance, Algorithm 3.

    Validity
    --------
    Continuous, unimodal or mildly multimodal distributions. Efficiency drops
    in high dimension (above a few tens of parameters).

    Verification
    ------------
    tests/unit/test_mcmc.py: recovery of the mean and covariance for a correlated Gaussian target.
    """
    x = np.array(initial, dtype=float)
    if x.ndim != 2:
        raise ValueError("initial must have the shape (walker, dimension)")
    k, n = x.shape
    if k % 2 or k < 2 * n:
        raise ValueError("the number of walkers must be even and at least twice the dimension")
    lp = np.array([log_probability(v) for v in x])
    if not np.all(np.isfinite(lp)):
        raise ValueError("log probability is not finite at some of the initial positions")
    samples = np.empty((n_steps, k, n))
    lp_record = np.empty((n_steps, k))
    accepted = np.zeros(k)
    half = k // 2
    halves = (np.arange(half), np.arange(half, k))
    for t in range(n_steps):
        for target, complement in (halves, halves[::-1]):
            z = stretch_factor(rng, half, a)
            partner = complement[rng.integers(0, half, half)]
            proposal = x[partner] + z[:, None] * (x[target] - x[partner])
            lp_proposal = np.array([log_probability(v) for v in proposal])
            log_q = (n - 1) * np.log(z) + lp_proposal - lp[target]
            is_accepted = np.log(rng.random(half)) < log_q
            chosen = target[is_accepted]
            x[chosen] = proposal[is_accepted]
            lp[chosen] = lp_proposal[is_accepted]
            accepted[chosen] += 1
        samples[t] = x
        lp_record[t] = lp
    return Chain(samples, lp_record, accepted / n_steps)


@source(
    "ForemanMackey2013",
    where="Section 3, 'Autocorrelation time': τ_f and C_f(T) equations",
    status="read",
    note="τ = 1 + 2 Σ C(T)/C(0). The window rule that truncates the sum (smallest M with M >= c·τ(M), c = 5) is not in the paper; "
    "it is recalled from the implementation in the emcee documentation and has not been verified",
)
def autocorrelation_time(chain: ArrayLike, c: float = 5.0) -> Array:
    """Estimate the integrated autocorrelation time per parameter.

    Parameters
    ----------
    chain : ndarray, shape (step, walker, dimension)
    c : float
        Window coefficient.

    Returns
    -------
    ndarray, shape (dimension,)
        τ in steps. The estimate is unreliable if the chain is shorter than 50 τ.

    References
    ----------
    [ForemanMackey2013] Section 3.

    Verification
    ------------
    tests/unit/test_mcmc.py: closed-form solution (1+ρ)/(1−ρ) for an AR(1) process.
    """
    x = np.asarray(chain, dtype=float)
    n_steps, k, n = x.shape
    tau = np.empty(n)
    fft_length = 1 << int(np.ceil(np.log2(2 * n_steps)))
    for d in range(n):
        acf = np.zeros(n_steps)
        for w in range(k):
            y = x[:, w, d] - x[:, w, d].mean()
            f = np.fft.rfft(y, fft_length)
            g = np.fft.irfft(f * np.conjugate(f))[:n_steps]
            acf += g / g[0] if g[0] > 0 else 0.0
        acf /= k
        taus = 2.0 * np.cumsum(acf) - 1.0
        window = np.arange(n_steps) >= c * taus
        m = int(np.argmax(window)) if np.any(window) else n_steps - 1
        tau[d] = taus[m]
    return tau


@source(
    "Vehtari2021",
    where="Section 3.1: B, W, var⁺ and R̂ equations",
    status="read",
    note="B = N/(M−1) Σ(θ̄_m − θ̄)², W = mean s_m², var⁺ = (N−1)/N·W + B/N, R̂ = sqrt(var⁺/W); split into two halves",
)
def split_rhat(chain: ArrayLike) -> Array:
    """Compute the split R̂ convergence diagnostic per parameter.

    Parameters
    ----------
    chain : ndarray, shape (step, walker, dimension)
        Samples after the burn-in has been discarded.

    Returns
    -------
    ndarray, shape (dimension,)
        R̂; a value close to 1 is consistent with the chains sampling the same distribution.

    References
    ----------
    [Vehtari2021] Section 3.1.

    Validity
    --------
    The walkers of an ensemble sampler are not independent chains; R̂ can
    therefore be optimistic here and should be read together with the
    autocorrelation time.

    Verification
    ------------
    tests/unit/test_mcmc.py: R̂ ≈ 1 for independent Gaussian samples, R̂ > 1.1 for shifted chains.
    """
    x = np.asarray(chain, dtype=float)
    half = x.shape[0] // 2
    parts = np.concatenate([x[:half], x[half : 2 * half]], axis=1)  # (half, 2k, n)
    mean = parts.mean(axis=0)
    w = parts.var(axis=0, ddof=1).mean(axis=0)
    b = half * mean.var(axis=0, ddof=1)
    var_plus = (half - 1) / half * w + b / half
    return np.sqrt(var_plus / w)


@source(
    "Vehtari2021",
    where="Section 4.1-4.2: rank normalization and folding equations",
    status="read",
    note="z = Φ⁻¹((r − 3/8)/(S + 1/4)); folding ζ = |θ − median(θ)|; the larger of the two is reported",
)
def rank_normalized_rhat(chain: ArrayLike) -> Array:
    """Compute the larger of the rank-normalized split R̂ and the folded split R̂.

    For each parameter the samples are replaced by their ranks in the pooled
    sample (average rank for ties) and converted to normal scores with
    z = Φ⁻¹((r − 3/8)/(S + 1/4)); the split R̂ is computed from these scores.
    The same procedure is applied to the absolute deviations from the median
    (folded samples); this measures convergence in the tails. The result is the
    larger of the two.

    Parameters
    ----------
    chain : ndarray, shape (step, walker, dimension)

    Returns
    -------
    ndarray, shape (dimension,)

    References
    ----------
    [Vehtari2021] Section 4.1-4.2.

    Validity
    --------
    Invariant under monotone transformations and also defined for
    distributions with infinite variance. In an ensemble sampler the walkers
    are not independent chains; the value can be optimistic.

    Verification
    ------------
    tests/unit/test_mcmc.py: detection of chains with shifted location and different scale; well-defined for Cauchy samples.
    """
    x = np.asarray(chain, dtype=float)
    n_steps, k, n = x.shape

    def score(d: Array) -> Array:
        rank = rankdata(d.reshape(-1, n), axis=0)  # average rank for ties
        return ndtri((rank - 0.375) / (n_steps * k + 0.25)).reshape(n_steps, k, n)

    folded = np.abs(x - np.median(x.reshape(-1, n), axis=0))
    return np.maximum(split_rhat(score(x)), split_rhat(score(folded)))


@source(
    "Vehtari2021",
    where="Section 'Recommendations for practice': use the sample only if the rank-normalized, folded R-hat < 1.01",
    status="read",
    note="the stopping rule built on that threshold is this project's choice: blocks of block_steps are appended until "
    "R-hat of the kept part is below rhat_limit and the chain is longer than min_tau_multiple autocorrelation "
    "times; the burn-in rule (5 tau, at least 100 steps, at most half the chain) and the factor 50 are this "
    "project's choices and come from no source read here",
)
def sample_until_converged(
    log_probability: Callable[[Array], float],
    initial: ArrayLike,
    rng: np.random.Generator,
    *,
    block_steps: int = 500,
    max_steps: int = 20000,
    rhat_limit: float = 1.01,
    min_tau_multiple: float = 50.0,
    a: float = 2.0,
) -> tuple[Chain, int, bool]:
    """Extend an ensemble chain block by block until the convergence criteria hold.

    After every block: tau = autocorrelation_time(chain), burn-in = min(steps/2, max(100, 5 tau_max));
    the run stops when rank_normalized_rhat(chain[burn-in:]) < rhat_limit for every parameter
    and steps > min_tau_multiple * tau_max, or when max_steps is reached.

    Parameters
    ----------
    log_probability : callable
        Log of the target density (up to a constant) at one position.
    initial : ndarray, shape (walker, dimension)
        Starting positions of the walkers.
    rng : numpy.random.Generator
    block_steps : int
        Steps added between two checks.
    max_steps : int
        Upper limit on the chain length; the last block is shortened to respect it.
    rhat_limit : float
        Threshold on the rank-normalized, folded split R-hat.
    min_tau_multiple : float
        Required chain length in units of the longest integrated autocorrelation time.
    a : float
        Stretch-move scale passed to ``ensemble_sample``.

    Returns
    -------
    (Chain, int, bool)
        The whole chain (burn-in included), the number of burn-in steps to discard, and whether the
        criteria were met. ``False`` means max_steps was reached first; the chain is still returned.

    References
    ----------
    [Vehtari2021] Section "Recommendations for practice" (threshold 1.01).

    Validity
    --------
    The walkers of an ensemble sampler are not independent chains, so R-hat computed over walkers
    can be optimistic; the length criterion in autocorrelation times is the independent second
    check. A chain continued from its last state is the same Markov chain, so appending blocks
    does not bias the sample; checking after every block is a sequential stopping rule and can
    stop on a favorable fluctuation, which the length criterion limits.

    Verification
    ------------
    tests/unit/test_mcmc.py: a correlated Gaussian target converges with both criteria met and the right mean
    and covariance; a too-small max_steps returns the chain flagged as not converged.
    """
    if block_steps < 1 or max_steps < block_steps:
        raise ValueError("block_steps must be positive and not larger than max_steps")
    position = np.array(initial, dtype=float)
    samples: list[Array] = []
    log_probabilities: list[Array] = []
    accepted = np.zeros(position.shape[0])
    steps = 0
    while True:
        n_block = min(block_steps, max_steps - steps)
        block = ensemble_sample(log_probability, position, n_block, rng, a=a)
        samples.append(block.samples)
        log_probabilities.append(block.log_probability)
        accepted += block.acceptance_fraction * n_block
        steps += n_block
        position = block.samples[-1]
        chain = np.concatenate(samples)
        tau_max = float(np.max(autocorrelation_time(chain)))
        discard = int(min(steps // 2, max(100, 5 * tau_max)))
        converged = bool(
            np.isfinite(tau_max)
            and steps > min_tau_multiple * tau_max
            and np.all(rank_normalized_rhat(chain[discard:]) < rhat_limit)
        )
        if converged or steps >= max_steps:
            return Chain(chain, np.concatenate(log_probabilities), accepted / steps), discard, converged
