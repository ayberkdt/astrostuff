"""Lomb-Scargle periodogram for unevenly sampled time series.

The definitions were read from the arXiv TeX source (1703.09824) of
VanderPlas's review [VanderPlas2018]; equations are cited by their TeX labels
there. The original papers of the method, [Lomb1976] and [Scargle1982], were
not read in this work.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source

Array = NDArray[np.float64]


@source(
    "VanderPlas2018",
    where="eq:lomb-scargle-periodogram, eq:tau-def",
    status="read",
    note="g_n is the mean-subtracted data; PSD normalization, unit is the square of the data unit",
)
def lomb_scargle_classical(times: ArrayLike, value: ArrayLike, frequency: ArrayLike) -> Array:
    """Compute the classical Lomb-Scargle periodogram.

    P(f) = ½ { [Σ g cos(2πf(t−τ))]² / Σ cos²(2πf(t−τ)) + [Σ g sin(2πf(t−τ))]² / Σ sin²(2πf(t−τ)) },
    tan(4πfτ) = Σ sin(4πft) / Σ cos(4πft).

    Parameters
    ----------
    times : ndarray (N,)
    value : ndarray (N,)
        Measurements; the mean is subtracted internally.
    frequency : ndarray (M,)
        Frequencies [1 / time unit], positive.

    Returns
    -------
    ndarray (M,)

    References
    ----------
    [VanderPlas2018] eq:lomb-scargle-periodogram, eq:tau-def.

    Validity
    --------
    Does not use measurement uncertainties and assumes that the mean of the
    data equals the mean of the signal; it is biased if the phase coverage is
    incomplete. Computational cost is O(N·M).

    Verification
    ------------
    tests/oracle/test_periodogram_oracle.py: astropy LombScargle(fit_mean=False, normalization="psd").
    """
    t = np.asarray(times, dtype=float)
    g = np.asarray(value, dtype=float)
    g = g - g.mean()
    f = np.atleast_1d(np.asarray(frequency, dtype=float))
    omega = 2.0 * np.pi * f[:, None]
    tau = np.arctan2(np.sin(2 * omega * t).sum(axis=1), np.cos(2 * omega * t).sum(axis=1)) / (2 * omega[:, 0])
    phase = omega * (t[None, :] - tau[:, None])
    c, s = np.cos(phase), np.sin(phase)
    return 0.5 * ((c @ g) ** 2 / (c * c).sum(axis=1) + (s @ g) ** 2 / (s * s).sum(axis=1))


@source(
    "VanderPlas2018",
    where="eq:floating-mean-model, eq:chi2-with-errors, eq:lomb-scargle-normalized",
    status="read",
    note="weighted least squares fit of the model y0 + a sin + b cos at each frequency; P = 1 − χ̂²(f)/χ̂²₀",
)
def lomb_scargle(times: ArrayLike, value: ArrayLike, frequency: ArrayLike, sigma: ArrayLike | None = None) -> Array:
    """Compute the weighted Lomb-Scargle periodogram with a floating mean.

    At each frequency the model y(t) = y₀ + A sin(2πf(t − φ)) is fitted by
    weighted least squares; the periodogram is the improvement in χ² relative
    to the constant model:
    P(f) = 1 − χ̂²(f) / χ̂²₀, 0 <= P <= 1.

    Parameters
    ----------
    times, value : ndarray (N,)
    frequency : ndarray (M,)
    sigma : ndarray (N,), optional
        Measurement standard uncertainties; equal weights if not given.

    References
    ----------
    [VanderPlas2018] eq:floating-mean-model, eq:chi2-with-errors, eq:lomb-scargle-normalized.

    Validity
    --------
    Assumes a model with a single sinusoidal component; for non-sinusoidal
    shapes (eclipsing binaries, transits) the power is spread over harmonics.
    Computational cost is O(N·M).

    Verification
    ------------
    tests/oracle/test_periodogram_oracle.py: astropy LombScargle(fit_mean=True, normalization="standard").
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(value, dtype=float)
    w = np.ones_like(y) if sigma is None else 1.0 / np.asarray(sigma, dtype=float) ** 2
    w = w / w.sum()
    f = np.atleast_1d(np.asarray(frequency, dtype=float))
    mean = w @ y
    yc = y - mean
    chi2_zero = w @ (yc * yc)
    phase = 2.0 * np.pi * f[:, None] * (t - t.mean())[None, :]
    basis = np.stack([np.ones_like(phase), np.sin(phase), np.cos(phase)], axis=-1)  # (M, N, 3)
    wb = basis * w[None, :, None]
    normal = np.einsum("mni,mnj->mij", wb, basis)
    rhs = np.einsum("mni,n->mi", wb, yc)
    solution = np.linalg.solve(normal, rhs[..., None])[..., 0]
    explained = np.einsum("mi,mi->m", rhs, solution)  # χ̂²₀ − χ̂²(f)
    return explained / chi2_zero


@source(
    "VanderPlas2018",
    where="TeX label eq:FAP-neff and the following paragraph",
    status="read",
    note="FAP(z) ≈ 1 − [P_single(z)]^N_eff; simple estimate N_eff = f_max · T",
)
@source(
    "Astropy2022",
    where="timeseries/periodograms/lombscargle/_statistics.py, fap_single and fap_naive",
    status="read",
    note="false alarm probability of a single frequency in the standard normalization (1 − z)^{(N−3)/2}; log1p/expm1 form",
)
def false_alarm_probability(power: ArrayLike, n_data: int, n_independent_frequencies: float) -> Array:
    """Estimate the false alarm probability for a peak of a standard-normalized periodogram.

    For a single frequency P(Z < z) = 1 − (1 − z)^{(N−3)/2}; for M independent
    frequencies FAP = 1 − [P(Z < z)]^M.

    References
    ----------
    [VanderPlas2018] eq:FAP-neff; [Astropy2022] fap_single, fap_naive.

    Validity
    --------
    Assumes Gaussian white noise. The number of independent frequencies is not
    well defined; the result is an order-of-magnitude estimate. A bootstrap is
    needed for a reliable value.

    Verification
    ------------
    tests/oracle/test_periodogram_oracle.py: same form as astropy false_alarm_probability(method="naive").
    """
    z = np.asarray(power, dtype=float)
    single_fap = (1.0 - z) ** (0.5 * (n_data - 3))
    # 1 − (1 − p)^M; use log1p/expm1 to avoid cancellation error for small p.
    with np.errstate(divide="ignore"):
        return -np.expm1(n_independent_frequencies * np.log1p(-single_fap))


@source(
    "VanderPlas2018",
    where="Section 7.1, TeX label eq:n-eval",
    status="read",
    note="Δf = 1/(n_o T); the source uses n_o = 5 and mentions the range 5-10",
)
def frequency_grid(times: ArrayLike, max_frequency: float, oversampling: float = 5.0) -> Array:
    """Build a uniform frequency grid for the periodogram: Δf = 1 / (n_o T).

    Parameters
    ----------
    times : ndarray (N,)
    max_frequency : float
        Upper limit of the grid. The Nyquist frequency is not well defined for
        unevenly sampled data; the upper limit is chosen from the shortest period sought.
    oversampling : float
        Number of grid points per peak width (1/T).

    References
    ----------
    [VanderPlas2018] Section 7.1, eq:n-eval.

    Verification
    ------------
    tests/oracle/test_periodogram_oracle.py
    """
    t = np.asarray(times, dtype=float)
    step = 1.0 / (oversampling * (t.max() - t.min()))
    return np.arange(step, max_frequency + step, step)
