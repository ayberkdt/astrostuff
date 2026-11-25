"""Initial mass functions (IMF): Salpeter, Kroupa and Chabrier.

The forms and parameters were read from the arXiv TeX source of the respective papers on
25 November 2026:

- Kroupa: multi-part power law, ξ(m) ∝ m^{-α_i} [Kroupa2001] (astro-ph/0009005).
- Chabrier: log-normal for m ≤ 1 M_⊙, power law for m > 1 M_⊙, the "IMF" row of Table 1
  [Chabrier2003] (astro-ph/0304382).
- Salpeter: single power law, α = 2.35. The original paper was read from the ADS scan
  [Salpeter1955, Eq. 5]: ξ ≈ 0.03 (M/M_⊙)^−1.35 per logarithmic mass interval, given for
  log(M/M_⊙) between −0.4 and +1.0; outside this range it is extrapolation.

Each IMF has four functions: the normalized number density dN/dm, the mean mass, the mass
fraction above a threshold, and random sampling by inverse transform. Masses are in solar
masses. The density is normalized so that ∫ dN/dm dm = 1 over [m_low, m_high] and is zero
outside this range. All integrals are in closed form (power-law part: simple power integral;
log-normal part: normal distribution function); numerical integration is done only in tests.

The Kroupa segments are joined so that they are continuous at the break masses. The paper
states only proportionality; the continuity assumption is checked in a test by reproducing the
aggregate numbers in the paper (mean mass, BC and M dwarf fractions).
The Chabrier log-normal and power-law coefficients are given as in the table and carry a
discontinuity of about 0.5 percent at m = 1 (measured in a test).

Two boundary notes. In the Chabrier function the exact point m = 1 falls on the power-law branch
([Chabrier2003] Table 1 assigns m <= 1 to the log-normal branch); the density differs by 0.47% at
that single point, and the integrals are not affected.
[Kroupa2001] also gives a form corrected for unresolved binaries (eq:imfr: α1 = 1.8, α2 = 2.7);
this module implements the form the paper gives for single stars, not corrected for binary bias
(eq:imf).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri

from kozmos._sources import source

Array = NDArray[np.float64]

_LN10 = math.log(10.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

# [Kroupa2001] eq:imf: break masses and slopes α_i (m^{-α_i}).
_KROUPA_BREAKS = (0.01, 0.08, 0.5, 1.0, math.inf)
_KROUPA_ALPHA = (0.3, 1.3, 2.3, 2.3)

# [Chabrier2003] Table 1 (TeX label table.diskIMF), "IMF" row: ξ(log m) = A exp{...} (m <= 1)
# and ξ(log m) = A m^{-x} (m > 1).
_CHABRIER_A_LOGNORMAL = 0.158
_CHABRIER_MC = 0.079
_CHABRIER_SIGMA = 0.69
_CHABRIER_A_POWER_LAW = 4.43e-2
_CHABRIER_X = 1.3


def _power_law_integral(coefficient: float, exponent: float, low: ArrayLike, high: ArrayLike) -> Array:
    """Integrate coefficient · m^exponent dm from low to high; logarithm for exponent = -1."""
    lo = np.asarray(low, dtype=float)
    hi = np.asarray(high, dtype=float)
    if abs(exponent + 1.0) < 1e-12:
        return coefficient * np.log(hi / lo)
    return coefficient * (hi ** (exponent + 1.0) - lo ** (exponent + 1.0)) / (exponent + 1.0)


@dataclass(frozen=True, slots=True)
class _PowerLawSegment:
    """dN/dm = coefficient · m^{-alpha} on [low, high)."""

    low: float
    high: float
    coefficient: float
    alpha: float

    def density(self, m: Array) -> Array:
        return self.coefficient * m**-self.alpha

    def number(self, lo: ArrayLike, hi: ArrayLike) -> Array:
        return _power_law_integral(self.coefficient, -self.alpha, lo, hi)

    def mass(self, lo: ArrayLike, hi: ArrayLike) -> Array:
        return _power_law_integral(self.coefficient, 1.0 - self.alpha, lo, hi)

    def inverse(self, u: Array, lo: float, hi: float) -> Array:
        q = 1.0 - self.alpha
        if abs(q) < 1e-12:
            return lo * (hi / lo) ** u
        return (lo**q + u * (hi**q - lo**q)) ** (1.0 / q)


@dataclass(frozen=True, slots=True)
class _LogNormalSegment:
    """dN/dlog₁₀m = amplitude · exp{-(log m - log_mc)² / 2σ²} on [low, high)."""

    low: float
    high: float
    amplitude: float
    log_mc: float
    sigma: float

    def _z(self, m: ArrayLike) -> Array:
        return (np.log10(np.asarray(m, dtype=float)) - self.log_mc) / self.sigma

    def density(self, m: Array) -> Array:
        return self.amplitude * np.exp(-0.5 * self._z(m) ** 2) / (m * _LN10)

    def number(self, lo: ArrayLike, hi: ArrayLike) -> Array:
        return self.amplitude * self.sigma * _SQRT_2PI * (ndtr(self._z(hi)) - ndtr(self._z(lo)))

    def mass(self, lo: ArrayLike, hi: ArrayLike) -> Array:
        # ∫ 10^x exp{-(x-xc)²/2σ²} dx: completing the square gives a normal with mean xc + σ² ln10.
        shift = self.sigma * _LN10
        factor = math.exp(_LN10 * self.log_mc + 0.5 * shift**2)
        return (
            self.amplitude * self.sigma * _SQRT_2PI * factor * (ndtr(self._z(hi) - shift) - ndtr(self._z(lo) - shift))
        )

    def inverse(self, u: Array, lo: float, hi: float) -> Array:
        p_lo, p_hi = float(ndtr(self._z(lo))), float(ndtr(self._z(hi)))
        z = ndtri(p_lo + u * (p_hi - p_lo))
        return 10.0 ** (self.log_mc + self.sigma * z)


class _Imf:
    """Normalized operations of a piecewise IMF on the range [m_low, m_high]."""

    def __init__(self, parts: tuple[_PowerLawSegment | _LogNormalSegment, ...], m_low: float, m_high: float):
        if not 0.0 < m_low < m_high:
            raise ValueError("0 < m_low < m_high is required")
        self.m_low, self.m_high = float(m_low), float(m_high)
        self.segments = [
            (p, max(p.low, m_low), min(p.high, m_high)) for p in parts if min(p.high, m_high) > max(p.low, m_low)
        ]
        self.counts = np.array([float(p.number(lo, hi)) for p, lo, hi in self.segments])
        self.total_count = float(self.counts.sum())

    def density(self, m: ArrayLike) -> Array:
        x = np.asarray(m, dtype=float)
        out = np.zeros_like(x)
        for i, (p, lo, hi) in enumerate(self.segments):
            is_last = i == len(self.segments) - 1
            mask = (x >= lo) & ((x <= hi) if is_last else (x < hi))
            if np.any(mask):
                out[mask] = p.density(x[mask])
        return np.asarray(out / self.total_count)[()]

    def _total_mass(self, low: ArrayLike) -> Array:
        a = np.clip(np.asarray(low, dtype=float), self.m_low, self.m_high)
        return sum(p.mass(np.clip(a, lo, hi), hi) for p, lo, hi in self.segments)

    def mean_mass(self) -> float:
        return float(self._total_mass(self.m_low)) / self.total_count

    def mass_fraction_above(self, threshold: ArrayLike) -> Array:
        return np.asarray(self._total_mass(threshold) / self._total_mass(self.m_low))[()]

    def sample(self, rng: np.random.Generator, n: int) -> Array:
        if n < 0:
            raise ValueError("the sample count cannot be negative")
        u = rng.random(int(n))
        cumulative = np.cumsum(self.counts) / self.total_count
        lower_edge = np.concatenate(([0.0], cumulative[:-1]))
        idx = np.minimum(np.searchsorted(cumulative, u, side="right"), len(self.segments) - 1)
        local_u = (u - lower_edge[idx]) / (cumulative[idx] - lower_edge[idx])
        m = np.empty_like(u)
        for i, (p, lo, hi) in enumerate(self.segments):
            mask = idx == i
            if np.any(mask):
                m[mask] = p.inverse(local_u[mask], lo, hi)
        return m


def _salpeter(alpha: float, m_low: float, m_high: float) -> _Imf:
    if alpha <= 0:
        raise ValueError("α must be positive")
    return _Imf((_PowerLawSegment(0.0, math.inf, 1.0, alpha),), m_low, m_high)


def _kroupa(m_low: float, m_high: float) -> _Imf:
    if m_low < _KROUPA_BREAKS[0]:
        raise ValueError("the Kroupa IMF is not defined below 0.01 M_⊙")
    parts = []
    coefficient = 1.0
    for i, alpha in enumerate(_KROUPA_ALPHA):
        if i > 0:
            coefficient *= _KROUPA_BREAKS[i] ** (alpha - _KROUPA_ALPHA[i - 1])  # continuity at the break
        parts.append(_PowerLawSegment(_KROUPA_BREAKS[i], _KROUPA_BREAKS[i + 1], coefficient, alpha))
    return _Imf(tuple(parts), m_low, m_high)


def _chabrier(m_low: float, m_high: float) -> _Imf:
    parts = (
        _LogNormalSegment(0.0, 1.0, _CHABRIER_A_LOGNORMAL, math.log10(_CHABRIER_MC), _CHABRIER_SIGMA),
        # ξ(log m) = A m^{-x}  ⇒  dN/dm = A/ln10 · m^{-(x+1)}
        _PowerLawSegment(1.0, math.inf, _CHABRIER_A_POWER_LAW / _LN10, _CHABRIER_X + 1.0),
    )
    return _Imf(parts, m_low, m_high)


@source(
    "Chabrier2003",
    where="Section 'Mass function' and 'Functional forms' (TeX label IMF1): α = 2.35, x = 1.35",
    status="read",
)
@source("Kroupa2001", where="Section '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2.35", status="read")
@source(
    "Salpeter1955",
    where="Eq. 2 (definition) and Eq. 5",
    status="read",
    note="read from the ADS scan: dN = ξ(M) d(log10 M) dt/T0 and ξ(M) ≈ 0.03 (M/M_⊙)^−1.35, for log(M/M_⊙) between "
    "−0.4 and +1.0. ξ is per logarithmic mass interval; dN/dM ∝ M^−2.35",
)
def salpeter_dn_dm(mass_msun: ArrayLike, m_low: float = 0.1, m_high: float = 100.0, alpha: float = 2.35) -> Array:
    """Compute the normalized number density of the Salpeter IMF.

    dN/dm = C m^{-α},   ∫_{m_low}^{m_high} dN/dm dm = 1

    Parameters
    ----------
    mass_msun : float or ndarray
        Mass [M_⊙].
    m_low, m_high : float
        Normalization range [M_⊙]; the defaults are a choice specific to this function.
    alpha : float
        Slope; the Salpeter value is 2.35 (x = α - 1 = 1.35).

    Returns
    -------
    float or ndarray
        dN/dm [M_⊙^-1]; 0 outside the range.

    References
    ----------
    [Chabrier2003] IMF1; [Kroupa2001] Salpeter slope; [Salpeter1955] Denk. 5.

    Validity
    --------
    [Kroupa2001]: Salpeter found α = 2.35 for 0.4 < m < 10 M_⊙; going beyond this range is extrapolation.

    Verification
    ------------
    tests/unit/test_stars.py: normalization integral and log-log slope; tests/property/test_stars_property.py.
    """
    return _salpeter(alpha, m_low, m_high).density(mass_msun)


@source("Chabrier2003", where="Section 'Mass function' and 'Functional forms' (TeX label IMF1)", status="read")
@source("Kroupa2001", where="Section '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2.35", status="read")
@source(
    "Salpeter1955",
    where="Eq. 2 (definition) and Eq. 5",
    status="read",
    note="read from the ADS scan: dN = ξ(M) d(log10 M) dt/T0 and ξ(M) ≈ 0.03 (M/M_⊙)^−1.35, for log(M/M_⊙) between "
    "−0.4 and +1.0. ξ is per logarithmic mass interval; dN/dM ∝ M^−2.35",
)
def salpeter_mean_mass(m_low: float = 0.1, m_high: float = 100.0, alpha: float = 2.35) -> float:
    """Compute the number-weighted mean mass of the Salpeter IMF.

    ⟨m⟩ = ∫ m dN/dm dm / ∫ dN/dm dm

    Parameters
    ----------
    m_low, m_high : float
        Range [M_⊙].
    alpha : float
        Slope.

    Returns
    -------
    float
        Mean mass [M_⊙].

    References
    ----------
    [Chabrier2003] IMF1; [Kroupa2001]; [Salpeter1955] Denk. 5.

    Verification
    ------------
    tests/unit/test_stars.py: ratio of closed-form power integrals; numerical integral.
    """
    return _salpeter(alpha, m_low, m_high).mean_mass()


@source("Chabrier2003", where="Section 'Mass function' and 'Functional forms' (TeX label IMF1)", status="read")
@source("Kroupa2001", where="Section '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2.35", status="read")
@source(
    "Salpeter1955",
    where="Eq. 2 (definition) and Eq. 5",
    status="read",
    note="read from the ADS scan: dN = ξ(M) d(log10 M) dt/T0 and ξ(M) ≈ 0.03 (M/M_⊙)^−1.35, for log(M/M_⊙) between "
    "−0.4 and +1.0. ξ is per logarithmic mass interval; dN/dM ∝ M^−2.35",
)
def salpeter_mass_fraction_above(
    m_threshold: ArrayLike, m_low: float = 0.1, m_high: float = 100.0, alpha: float = 2.35
) -> Array:
    """Compute the fraction of the total mass in stars above a threshold (Salpeter).

    f(>m_threshold) = ∫_{m_threshold}^{m_high} m dN/dm dm / ∫_{m_low}^{m_high} m dN/dm dm

    Parameters
    ----------
    m_threshold : float or ndarray
        Threshold mass [M_⊙]; clipped to the range.
    m_low, m_high : float
        Range [M_⊙].
    alpha : float
        Slope.

    Returns
    -------
    float or ndarray
        Mass fraction ∈ [0, 1]; decreases as the threshold increases.

    References
    ----------
    [Chabrier2003] IMF1; [Kroupa2001]; [Salpeter1955] Denk. 5.

    Verification
    ------------
    tests/unit/test_stars.py: closed form; tests/property/test_stars_property.py: monotonicity.
    """
    return _salpeter(alpha, m_low, m_high).mass_fraction_above(m_threshold)


@source("Chabrier2003", where="Section 'Mass function' and 'Functional forms' (TeX label IMF1)", status="read")
@source("Kroupa2001", where="Section '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2.35", status="read")
@source(
    "Salpeter1955",
    where="Eq. 2 (definition) and Eq. 5",
    status="read",
    note="read from the ADS scan: dN = ξ(M) d(log10 M) dt/T0 and ξ(M) ≈ 0.03 (M/M_⊙)^−1.35, for log(M/M_⊙) between "
    "−0.4 and +1.0. ξ is per logarithmic mass interval; dN/dM ∝ M^−2.35",
)
def salpeter_sample(
    rng: np.random.Generator, n: int, m_low: float = 0.1, m_high: float = 100.0, alpha: float = 2.35
) -> Array:
    """Draw random masses from the Salpeter IMF by inverse transform.

    m = [m_low^{1-α} + u (m_high^{1-α} - m_low^{1-α})]^{1/(1-α)},   u ~ U[0, 1)

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    n : int
        Number of samples.
    m_low, m_high : float
        Range [M_⊙].
    alpha : float
        Slope.

    Returns
    -------
    ndarray, shape (n,)
        Masses [M_⊙], within [m_low, m_high].

    References
    ----------
    [Chabrier2003] IMF1; [Kroupa2001]; [Salpeter1955] Denk. 5.

    Verification
    ------------
    tests/unit/test_stars.py: Kolmogorov-Smirnov test (fixed seed); agreement of the mean mass.
    """
    return _salpeter(alpha, m_low, m_high).sample(rng, n)


@source(
    "Kroupa2001",
    where="Section 'The universal IMF', TeX labels eq:xi and eq:imf",
    status="read",
    note="segments are joined continuously at the breaks (the paper states only proportionality); checked against the aggregate numbers in the paper",
)
def kroupa_dn_dm(mass_msun: ArrayLike, m_low: float = 0.01, m_high: float = 50.0) -> Array:
    """Compute the normalized number density of the Kroupa (2001) IMF.

    dN/dm ∝ m^{-α_i};  α = 0.3 (0.01-0.08), 1.3 (0.08-0.5), 2.3 (0.5-1), 2.3 (≥ 1 M_⊙)

    Parameters
    ----------
    mass_msun : float or ndarray
        Mass [M_⊙].
    m_low, m_high : float
        Normalization range [M_⊙]; m_low >= 0.01. The default upper limit is the 50 M_⊙
        the paper uses for the mean mass.

    Returns
    -------
    float or ndarray
        dN/dm [M_⊙^-1], ∫ = 1; 0 outside the range.

    References
    ----------
    [Kroupa2001] eq:xi, eq:imf.

    Validity
    --------
    The paper says the confidence interval is not well determined below 0.08 M_⊙ (α_0 = 0.3 ± 0.7).
    It is for single stars; it contains no correction for binary systems.

    Verification
    ------------
    tests/unit/test_stars.py: mean mass 0.36 M_⊙ and the number/mass fractions in the paper.
    """
    return _kroupa(m_low, m_high).density(mass_msun)


@source("Kroupa2001", where="Section 'The universal IMF', TeX label eq:imf and the mean-mass sentence", status="read")
def kroupa_mean_mass(m_low: float = 0.01, m_high: float = 50.0) -> float:
    """Compute the number-weighted mean mass of the Kroupa IMF.

    Parameters
    ----------
    m_low, m_high : float
        Range [M_⊙]; m_low >= 0.01.

    Returns
    -------
    float
        Mean mass [M_⊙]; 0.36 for 0.01-50 M_⊙.

    References
    ----------
    [Kroupa2001] eq:imf and the sentence "⟨m⟩ = 0.36 M_⊙".

    Verification
    ------------
    tests/unit/test_stars.py: the 0.36 M_⊙ in the paper (two significant digits).
    """
    return _kroupa(m_low, m_high).mean_mass()


@source(
    "Kroupa2001", where="Section 'The universal IMF', TeX label eq:imf and the mass-fraction sentences", status="read"
)
def kroupa_mass_fraction_above(m_threshold: ArrayLike, m_low: float = 0.01, m_high: float = 50.0) -> Array:
    """Compute the fraction of the total mass in stars above a threshold (Kroupa).

    Parameters
    ----------
    m_threshold : float or ndarray
        Threshold mass [M_⊙]; clipped to the range.
    m_low, m_high : float
        Range [M_⊙]; m_low >= 0.01.

    Returns
    -------
    float or ndarray
        Mass fraction ∈ [0, 1]; about 0.5 for a threshold of 1 M_⊙ (paper: half of the mass is below 1 M_⊙).

    References
    ----------
    [Kroupa2001] eq:imf and the mass-fraction sentences.

    Verification
    ------------
    tests/unit/test_stars.py: 50 percent, and 17 percent above 8 M_⊙ ([Kroupa2001]).
    """
    return _kroupa(m_low, m_high).mass_fraction_above(m_threshold)


@source("Kroupa2001", where="Section 'The universal IMF', TeX label eq:imf", status="read")
def kroupa_sample(rng: np.random.Generator, n: int, m_low: float = 0.01, m_high: float = 50.0) -> Array:
    """Draw random masses from the Kroupa IMF by piecewise inverse transform.

    A single uniform number u selects a segment, by cumulative number fraction, and the position within the segment.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    n : int
        Number of samples.
    m_low, m_high : float
        Range [M_⊙]; m_low >= 0.01.

    Returns
    -------
    ndarray, shape (n,)
        Masses [M_⊙].

    References
    ----------
    [Kroupa2001] eq:imf.

    Verification
    ------------
    tests/unit/test_stars.py: Kolmogorov-Smirnov test and sample mean.
    """
    return _kroupa(m_low, m_high).sample(rng, n)


@source(
    "Chabrier2003",
    where="Section 'The disk stellar mass function', TeX label IMFdisk and Table 1 (table.diskIMF), 'IMF' row",
    status="read",
    note="m <= 1: A = 0.158, m_c = 0.079, σ = 0.69; m > 1: A = 4.43e-2, x = 1.3; ξ(log m) is the definition, dN/dm = ξ/(m ln10)",
)
def chabrier_dn_dm(mass_msun: ArrayLike, m_low: float = 0.01, m_high: float = 50.0) -> Array:
    """Compute the normalized number density of the Chabrier (2003) disk IMF.

    ξ(log m) = 0.158 exp{-(log m - log 0.079)² / (2 · 0.69²)}   (m ≤ 1 M_⊙)
    ξ(log m) = 4.43e-2 m^{-1.3}                                 (m > 1 M_⊙)
    dN/dm = ξ(log m) / (m ln 10)

    Parameters
    ----------
    mass_msun : float or ndarray
        Mass [M_⊙].
    m_low, m_high : float
        Normalization range [M_⊙]; the defaults are chosen for comparison with Kroupa.

    Returns
    -------
    float or ndarray
        dN/dm [M_⊙^-1], ∫ = 1; 0 outside the range.

    References
    ----------
    [Chabrier2003] IMFdisk and Table 1.

    Validity
    --------
    This is the disk IMF for single objects; the coefficients for binary systems differ (the IMFsys equation
    of the paper, not implemented here). Parameter uncertainties are given in the paper (e.g. A = 0.158 +0.051/-0.046).

    Verification
    ------------
    tests/unit/test_stars.py: normalization, continuity at m = 1 (1 percent) and the log-normal peak.
    """
    return _chabrier(m_low, m_high).density(mass_msun)


@source("Chabrier2003", where="Section 'The disk stellar mass function', TeX label IMFdisk and Table 1", status="read")
def chabrier_mean_mass(m_low: float = 0.01, m_high: float = 50.0) -> float:
    """Compute the number-weighted mean mass of the Chabrier disk IMF.

    Parameters
    ----------
    m_low, m_high : float
        Range [M_⊙].

    Returns
    -------
    float
        Mean mass [M_⊙].

    References
    ----------
    [Chabrier2003] IMFdisk and Table 1.

    Verification
    ------------
    tests/unit/test_stars.py: relative agreement of 1e-8 with the numerical integral (scipy.integrate.quad).
    """
    return _chabrier(m_low, m_high).mean_mass()


@source("Chabrier2003", where="Section 'The disk stellar mass function', TeX label IMFdisk and Table 1", status="read")
def chabrier_mass_fraction_above(m_threshold: ArrayLike, m_low: float = 0.01, m_high: float = 50.0) -> Array:
    """Compute the fraction of the total mass in stars above a threshold (Chabrier).

    Parameters
    ----------
    m_threshold : float or ndarray
        Threshold mass [M_⊙]; clipped to the range.
    m_low, m_high : float
        Range [M_⊙].

    Returns
    -------
    float or ndarray
        Mass fraction ∈ [0, 1].

    References
    ----------
    [Chabrier2003] IMFdisk and Table 1.

    Verification
    ------------
    tests/unit/test_stars.py: agreement with the numerical integral; tests/property/test_stars_property.py: monotonicity.
    """
    return _chabrier(m_low, m_high).mass_fraction_above(m_threshold)


@source("Chabrier2003", where="Section 'The disk stellar mass function', TeX label IMFdisk and Table 1", status="read")
def chabrier_sample(rng: np.random.Generator, n: int, m_low: float = 0.01, m_high: float = 50.0) -> Array:
    """Draw random masses from the Chabrier disk IMF by inverse transform.

    The log-normal segment uses the inverse of the normal distribution (``ndtri``); the power-law segment uses the closed-form inverse.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    n : int
        Number of samples.
    m_low, m_high : float
        Range [M_⊙].

    Returns
    -------
    ndarray, shape (n,)
        Masses [M_⊙].

    References
    ----------
    [Chabrier2003] IMFdisk and Table 1.

    Verification
    ------------
    tests/unit/test_stars.py: Kolmogorov-Smirnov test (fixed seed).
    """
    return _chabrier(m_low, m_high).sample(rng, n)
