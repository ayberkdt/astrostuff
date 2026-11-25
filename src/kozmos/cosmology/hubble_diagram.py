"""Supernova Hubble diagram likelihood.

The likelihood definitions were read from the arXiv TeX source of the Pantheon+
cosmology paper (2202.04077) [Brout2022]; equations are cited by their TeX
labels there. The meaning of the data columns comes from the ``README`` file of
the data repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cholesky, solve_triangular

from kozmos._sources import source
from kozmos.cosmology.distances import Cosmology, distance_modulus

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class HubbleData:
    """Supernova data prepared for the likelihood.

    Attributes
    ----------
    z : ndarray
        Hubble diagram redshift (``zHD``).
    magnitude : ndarray
        Standardized magnitude (``m_b_corr``).
    calibrator : ndarray[bool]
        Supernovae in hosts with a Cepheid distance (``IS_CALIBRATOR``).
    cepheid_modulus : ndarray
        Cepheid host distance modulus (``CEPH_DIST``); not used for non-calibrators.
    cholesky_lower : ndarray
        Lower-triangular Cholesky factor of the covariance of the selected rows.
    """

    z: Array
    magnitude: Array
    calibrator: Mask
    cepheid_modulus: Array
    cholesky_lower: Array

    def __len__(self) -> int:
        return int(self.z.shape[0])


@source(
    "Brout2022",
    where="Section 4: z < ~0.01 is not used in the Hubble diagram; eq:dmuprime",
    status="read",
    note="the threshold 0.01 appears in the paper as '~0.01'; the exact value is this project's choice",
)
def prepare_data(
    columns: dict[str, np.ndarray],
    covariance: ArrayLike,
    *,
    z_min: float = 0.01,
    calibrators: bool,
) -> HubbleData:
    """Select rows and compute the Cholesky factor of the covariance submatrix.

    Parameters
    ----------
    columns : dict
        Output of ``kozmos.archives.pantheon.read_table``.
    covariance : ndarray (N, N)
    z_min : float
        Lower redshift limit for non-calibrator supernovae.
    calibrators : bool
        If true, supernovae in Cepheid hosts are included (regardless of their
        redshift); if false, they are excluded entirely.

    References
    ----------
    [Brout2022] Section 4, eq:dmuprime.

    Validity
    --------
    Nearby supernovae, where peculiar velocities dominate, are used only as
    Cepheid anchors; they do not enter the Hubble flow.

    Verification
    ------------
    tests/unit/test_hubble_diagram.py (synthetic data).
    """
    z = np.asarray(columns["zHD"], dtype=float)
    is_calibrator = np.asarray(columns["IS_CALIBRATOR"], dtype=float) == 1.0
    selected = (z > z_min) & ~is_calibrator
    if calibrators:
        selected = selected | is_calibrator
    c = np.asarray(covariance, dtype=float)[np.ix_(selected, selected)]
    return HubbleData(
        z=z[selected],
        magnitude=np.asarray(columns["m_b_corr"], dtype=float)[selected],
        calibrator=is_calibrator[selected],
        cepheid_modulus=np.asarray(columns["CEPH_DIST"], dtype=float)[selected],
        cholesky_lower=cholesky(c, lower=True),
    )


@source(
    "Brout2022",
    where="eq:dmu, eq:dmuprime, eq:dl",
    status="read",
    note="μ_i = m_b_corr,i − M; for calibrators the Cepheid distance modulus replaces the model",
)
def residual_vector(data: HubbleData, cosmology: Cosmology, absolute_magnitude: float) -> Array:
    """Compute the data minus model residual.

    Calibrators: ΔD_i = (m_i − M) − μ_Cepheid,i. Others: ΔD_i = (m_i − M) − μ_model(z_i).

    References
    ----------
    [Brout2022] eq:dmu, eq:dmuprime.

    Validity
    --------
    As in [Brout2022] eq:dl, the model distance is computed with a single
    redshift (``zHD``); the correction factor involving the heliocentric
    redshift (1 + z_hel) is not applied.

    Verification
    ------------
    tests/unit/test_hubble_diagram.py
    """
    mu = data.magnitude - absolute_magnitude
    model = np.where(data.calibrator, data.cepheid_modulus, 0.0)
    hubble_flow = ~data.calibrator
    model[hubble_flow] = distance_modulus(data.z[hubble_flow], cosmology)
    return mu - model


@source(
    "Brout2022",
    where="eq:likelihood, eq:sh0eslikelihood",
    status="read",
    note="−2 ln L = ΔDᵀ C⁻¹ ΔD; C is the STAT+SYS matrix of the data release",
)
def log_likelihood(data: HubbleData, cosmology: Cosmology, absolute_magnitude: float) -> float:
    """Compute the Gaussian log likelihood (without the constant term): ln L = −χ²/2.

    References
    ----------
    [Brout2022] eq:likelihood, eq:sh0eslikelihood.

    Verification
    ------------
    tests/unit/test_hubble_diagram.py: recovery of the injected parameters from synthetic data.
    """
    y = solve_triangular(data.cholesky_lower, residual_vector(data, cosmology, absolute_magnitude), lower=True)
    return -0.5 * float(y @ y)


@source(
    "Brout2022",
    where="eq:likelihood",
    status="read",
    note="M is a linear parameter; for a fixed cosmology the best M is a weighted mean (derivation in this project)",
)
def best_absolute_magnitude(data: HubbleData, cosmology: Cosmology) -> tuple[float, float, float]:
    """Find the M that minimizes χ² for a fixed cosmology, in closed form.

    For ΔD = r − M·1: M̂ = (1ᵀC⁻¹r)/(1ᵀC⁻¹1), σ_M² = 1/(1ᵀC⁻¹1).

    Returns
    -------
    (M̂, σ_M, χ²_min)

    References
    ----------
    [Brout2022] eq:likelihood.

    Verification
    ------------
    tests/unit/test_hubble_diagram.py: comparison with numerical minimization.
    """
    r = solve_triangular(data.cholesky_lower, residual_vector(data, cosmology, 0.0), lower=True)
    ones = solve_triangular(data.cholesky_lower, np.ones(len(data)), lower=True)
    denominator = float(ones @ ones)
    m = float(ones @ r) / denominator
    remainder = r - m * ones
    return m, denominator**-0.5, float(remainder @ remainder)
