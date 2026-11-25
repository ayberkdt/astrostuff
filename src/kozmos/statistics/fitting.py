"""Linear model fitting and χ² with covariance.

The equations were read from the arXiv TeX source (1008.4686,
``straightline.tex``) of the review by Hogg, Bovy and Lang [HoggBovyLang2010].
The source gives TeX labels instead of equation numbers; the numbers were not
verified against the compiled PDF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from kozmos._sources import source

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LinearFit:
    """Result of a linear fit."""

    parameters: Array
    covariance: Array
    chi_square: float
    degrees_of_freedom: int

    @property
    def uncertainties(self) -> Array:
        """Standard uncertainties of the parameters: square root of the covariance diagonal."""
        return np.sqrt(np.diag(self.covariance))


def _covariance_matrix(sigma_or_covariance: ArrayLike, n: int) -> Array:
    c = np.asarray(sigma_or_covariance, dtype=float)
    if c.ndim == 0:
        return np.eye(n) * float(c) ** 2
    if c.ndim == 1:
        if c.shape != (n,):
            raise ValueError("length of the uncertainty array does not match the data")
        return np.diag(c**2)
    if c.shape != (n, n):
        raise ValueError("covariance matrix must be N×N")
    return c


@source(
    "HoggBovyLang2010",
    where="TeX label eq:chisquared",
    status="read",
    note="χ² = [Y − AX]ᵀ C⁻¹ [Y − AX]; the inverse is not formed, the system is solved with the Cholesky factor",
)
def chi_square(residual: ArrayLike, cholesky_lower: ArrayLike) -> float:
    """Compute χ² with covariance: χ² = rᵀ C⁻¹ r.

    Parameters
    ----------
    residual : ndarray, shape (N,)
        Data minus model.
    cholesky_lower : ndarray, shape (N, N)
        Lower-triangular Cholesky factor L of the covariance (C = L Lᵀ).

    Returns
    -------
    float

    References
    ----------
    [HoggBovyLang2010] eq:chisquared.

    Validity
    --------
    C must be symmetric and positive definite.

    Verification
    ------------
    tests/unit/test_fitting.py: comparison with direct matrix inversion.
    """
    y = solve_triangular(np.asarray(cholesky_lower, dtype=float), np.asarray(residual, dtype=float), lower=True)
    return float(y @ y)


@source(
    "HoggBovyLang2010",
    where="TeX label eq:lsf and the parameter covariance equation",
    status="read",
    note="X = [AᵀC⁻¹A]⁻¹[AᵀC⁻¹Y]; parameter covariance [AᵀC⁻¹A]⁻¹",
)
def linear_least_squares(design: ArrayLike, y: ArrayLike, sigma_or_covariance: ArrayLike) -> LinearFit:
    """Fit a linear model by weighted least squares.

    Model Y = A X; solution X = [Aᵀ C⁻¹ A]⁻¹ [Aᵀ C⁻¹ Y].

    Parameters
    ----------
    design : ndarray, shape (N, P)
        Design matrix A.
    y : ndarray, shape (N,)
        Measurements.
    sigma_or_covariance : float, ndarray (N,) or ndarray (N, N)
        Measurement standard uncertainties or the full covariance matrix.

    References
    ----------
    [HoggBovyLang2010] eq:lsf.

    Validity
    --------
    Uncertainties must be in the y axis only, Gaussian and correctly known;
    the source stresses that most of these assumptions do not hold for real data.

    Verification
    ------------
    tests/unit/test_fitting.py: comparison with numpy.polyfit (weighted).
    """
    a = np.asarray(design, dtype=float)
    yy = np.asarray(y, dtype=float)
    if a.ndim != 2 or a.shape[0] != yy.shape[0]:
        raise ValueError("design matrix must be (N, P) and y must be (N,)")
    c = _covariance_matrix(sigma_or_covariance, yy.shape[0])
    factor = cho_factor(c, lower=True)
    cinv_a = cho_solve(factor, a)
    cinv_y = cho_solve(factor, yy)
    normal = a.T @ cinv_a
    cov = np.linalg.inv(normal)
    x = cov @ (a.T @ cinv_y)
    residual = yy - a @ x
    return LinearFit(x, cov, float(residual @ cho_solve(factor, residual)), yy.shape[0] - a.shape[1])


@source(
    "HoggBovyLang2010",
    where="TeX label eq:fofx, eq:lsf",
    status="read",
    note="parameter order as in the source: [b, m]",
)
def fit_line(x: ArrayLike, y: ArrayLike, sigma_y: ArrayLike, *, through_origin: bool = False) -> LinearFit:
    """Fit the line y = m x + b.

    Parameters
    ----------
    x, y : ndarray, shape (N,)
    sigma_y : float or ndarray (N,)
        Uncertainties of y.
    through_origin : bool
        If true, the intercept is fixed to zero and only the slope is fitted.

    Returns
    -------
    LinearFit
        ``parameters`` is ordered [b, m]; with ``through_origin`` it is [m] only.

    References
    ----------
    [HoggBovyLang2010] eq:fofx, eq:lsf.

    Verification
    ------------
    tests/unit/test_fitting.py
    """
    xx = np.asarray(x, dtype=float)
    a = xx[:, None] if through_origin else np.column_stack([np.ones_like(xx), xx])
    return linear_least_squares(a, y, sigma_y)
