"""Two-body problem: Kepler's equation and basic relations.

The equations were read from the arXiv TeX source (1009.1738, ``Chapter2.tex``)
of the review chapter by Murray and Correia [MurrayCorreia2010]; equations are
cited by the TeX labels used there.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.angles import TWO_PI, wrap_pm_pi

Array = NDArray[np.float64]


@source(
    "MurrayCorreia2010",
    where="TeX label eq37 (Kepler's equation), eq38-eq39 (Newton-Raphson)",
    status="read",
    note="the source suggests E0 = M as the starting value; here E0 = M + e sin M, and π·sign(M) for e > 0.8 (this project's choice)",
)
def eccentric_anomaly(
    mean_anomaly_rad: ArrayLike, eccentricity: ArrayLike, *, tolerance: float = 1e-14, max_iterations: int = 50
) -> Array:
    """Solve Kepler's equation for an elliptic orbit: M = E − e sin E.

    Parameters
    ----------
    mean_anomaly_rad : float or ndarray
        Mean anomaly M [rad].
    eccentricity : float or ndarray
        Eccentricity, 0 <= e < 1.
    tolerance : float
        Stopping criterion of the iteration, |ΔE| [rad].

    Returns
    -------
    float or ndarray
        Eccentric anomaly E [rad], on the same 2π branch as M.

    References
    ----------
    [MurrayCorreia2010] eq37, eq38-eq39.

    Validity
    --------
    0 <= e < 1. The Newton iteration slows down near e → 1 and M → 0;
    tested for e <= 0.999.

    Verification
    ------------
    tests/property/test_kepler_property.py: |E − e sin E − M| < 1e-12.
    """
    m = np.asarray(mean_anomaly_rad, dtype=float)
    e = np.asarray(eccentricity, dtype=float)
    if np.any((e < 0) | (e >= 1)):
        raise ValueError("eccentricity must be in the range [0, 1)")
    m, e = np.broadcast_arrays(m, e)
    m_wrapped = wrap_pm_pi(m)
    ecc_anomaly = m_wrapped + e * np.sin(m_wrapped)
    # At high eccentricity, E0 = π·sign(M) is a safe starting value for Newton.
    ecc_anomaly = np.where(e > 0.8, np.pi * np.sign(m_wrapped + 1e-300), ecc_anomaly)
    for _ in range(max_iterations):
        f = ecc_anomaly - e * np.sin(ecc_anomaly) - m_wrapped
        delta = f / (1.0 - e * np.cos(ecc_anomaly))
        ecc_anomaly = ecc_anomaly - delta
        if np.all(np.abs(delta) < tolerance):
            break
    return ecc_anomaly + (m - m_wrapped)


@source(
    "MurrayCorreia2010",
    where="TeX label eq17 and eq33",
    status="read",
    note="r = a(1−e²)/(1+e cos f) = a(1−e cos E); tan(f/2) = sqrt((1+e)/(1−e)) tan(E/2) is derived from these two (the derivation is in this project)",
)
def true_anomaly(eccentric_anomaly_rad: ArrayLike, eccentricity: ArrayLike) -> Array:
    """Compute the true anomaly from the eccentric anomaly.

    References
    ----------
    [MurrayCorreia2010] eq17, eq33.

    Verification
    ------------
    tests/property/test_kepler_property.py: orbit equation r = a(1 − e cos E) = a(1 − e²)/(1 + e cos f).
    """
    ecc_anomaly = np.asarray(eccentric_anomaly_rad, dtype=float)
    e = np.asarray(eccentricity, dtype=float)
    f = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(ecc_anomaly / 2.0), np.sqrt(1.0 - e) * np.cos(ecc_anomaly / 2.0))
    # Stay on the same 2π branch as E.
    return f + TWO_PI * np.round((ecc_anomaly - f) / TWO_PI)


@source("MurrayCorreia2010", where="TeX label eq19", status="read", note="T² = 4π² a³ / [G(m1 + m2)]")
def orbital_period_s(semi_major_axis_m: ArrayLike, gm_m3_s2: ArrayLike) -> Array:
    """Compute the orbital period: P = 2π sqrt(a³/μ).

    Parameters
    ----------
    yari_ecc_anomalyksen_m : float ya da ndarray
        Semi-major axis [m].
    gm_m3_s2 : float or ndarray
        μ = G(m1 + m2) [m³ s⁻²].

    References
    ----------
    [MurrayCorreia2010] eq19.

    Verification
    ------------
    tests/unit/test_kepler.py: about one Gaussian year for 1 au and the nominal GM_Sun [Prsa2016].
    """
    a = np.asarray(semi_major_axis_m, dtype=float)
    return TWO_PI * np.sqrt(a**3 / np.asarray(gm_m3_s2, dtype=float))


@source("MurrayCorreia2010", where="TeX label eq30", status="read", note="v² = G(m1 + m2)(2/r − 1/a)")
def orbital_speed_m_s(distance_m: ArrayLike, semi_major_axis_m: ArrayLike, gm_m3_s2: ArrayLike) -> Array:
    """Compute the orbital speed from the vis-viva equation.

    References
    ----------
    [MurrayCorreia2010] eq30.

    Verification
    ------------
    tests/unit/test_kepler.py: v = sqrt(μ/r) on a circular orbit.
    """
    r = np.asarray(distance_m, dtype=float)
    a = np.asarray(semi_major_axis_m, dtype=float)
    return np.sqrt(np.asarray(gm_m3_s2, dtype=float) * (2.0 / r - 1.0 / a))
