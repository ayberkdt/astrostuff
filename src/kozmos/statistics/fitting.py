"""Doğrusal model uydurma ve kovaryanslı χ².

Denklemler Hogg, Bovy ve Lang'ın derlemesinin arXiv TeX kaynağından
(1008.4686, ``straightline.tex``) okunmuştur [HoggBovyLang2010]. Kaynakta
denklem numarası yerine TeX etiketi verilir; numara derlenmiş PDF'ten teyit
edilmemiştir.
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
    """Doğrusal uyumun sonucu."""

    parameters: Array
    covariance: Array
    chi_square: float
    degrees_of_freedom: int

    @property
    def uncertainties(self) -> Array:
        """Parametre standart belirsizlikleri: kovaryans köşegeninin karekökü."""
        return np.sqrt(np.diag(self.covariance))


def _covariance_matrix(sigma_or_covariance: ArrayLike, n: int) -> Array:
    c = np.asarray(sigma_or_covariance, dtype=float)
    if c.ndim == 0:
        return np.eye(n) * float(c) ** 2
    if c.ndim == 1:
        if c.shape != (n,):
            raise ValueError("belirsizlik dizisinin boyu veriyle uyuşmuyor")
        return np.diag(c**2)
    if c.shape != (n, n):
        raise ValueError("kovaryans matrisi N×N olmalı")
    return c


@source(
    "HoggBovyLang2010",
    where="TeX etiketi eq:chisquared",
    status="read",
    note="χ² = [Y − AX]ᵀ C⁻¹ [Y − AX]; ters alınmaz, Cholesky çarpanıyla çözülür",
)
def chi_square(residual: ArrayLike, cholesky_lower: ArrayLike) -> float:
    """Kovaryanslı χ² değerini hesaplar: χ² = rᵀ C⁻¹ r.

    Parameters
    ----------
    artik : ndarray, biçim (N,)
        Veri eksi model.
    cholesky_alt : ndarray, biçim (N, N)
        Kovaryansın alt üçgen Cholesky çarpanı L (C = L Lᵀ).

    Returns
    -------
    float

    References
    ----------
    [HoggBovyLang2010] eq:chisquared.

    Validity
    --------
    C simetrik ve pozitif tanımlı olmalıdır.

    Verification
    ------------
    tests/unit/test_fitting.py: doğrudan ters alma ile karşılaştırma.
    """
    y = solve_triangular(np.asarray(cholesky_lower, dtype=float), np.asarray(residual, dtype=float), lower=True)
    return float(y @ y)


@source(
    "HoggBovyLang2010",
    where="TeX etiketi eq:lsf ve parametre kovaryansı denklemi",
    status="read",
    note="X = [AᵀC⁻¹A]⁻¹[AᵀC⁻¹Y]; parametre kovaryansı [AᵀC⁻¹A]⁻¹",
)
def linear_least_squares(design: ArrayLike, y: ArrayLike, sigma_or_covariance: ArrayLike) -> LinearFit:
    """Doğrusal modeli ağırlıklı en küçük karelerle uydurur.

    Model Y = A X; çözüm X = [Aᵀ C⁻¹ A]⁻¹ [Aᵀ C⁻¹ Y].

    Parameters
    ----------
    tasarim : ndarray, biçim (N, P)
        Tasarım matrisi A.
    y : ndarray, biçim (N,)
        Ölçümler.
    sigma_ya_da_kovaryans : float, ndarray (N,) ya da ndarray (N, N)
        Ölçüm standart belirsizlikleri ya da tam kovaryans matrisi.

    References
    ----------
    [HoggBovyLang2010] eq:lsf.

    Validity
    --------
    Belirsizlikler yalnızca y ekseninde, Gauss ve doğru bilinen olmalıdır;
    kaynak bu varsayımların çoğu gerçek veride sağlanmadığını vurgular.

    Verification
    ------------
    tests/unit/test_fitting.py: numpy.polyfit (ağırlıklı) ile karşılaştırma.
    """
    a = np.asarray(design, dtype=float)
    yy = np.asarray(y, dtype=float)
    if a.ndim != 2 or a.shape[0] != yy.shape[0]:
        raise ValueError("tasarım matrisi (N, P), y (N,) olmalı")
    c = _covariance_matrix(sigma_or_covariance, yy.shape[0])
    carpan = cho_factor(c, lower=True)
    cinv_a = cho_solve(carpan, a)
    cinv_y = cho_solve(carpan, yy)
    normal = a.T @ cinv_a
    kov = np.linalg.inv(normal)
    x = kov @ (a.T @ cinv_y)
    residual = yy - a @ x
    return LinearFit(x, kov, float(residual @ cho_solve(carpan, residual)), yy.shape[0] - a.shape[1])


@source(
    "HoggBovyLang2010",
    where="TeX etiketi eq:fofx, eq:lsf",
    status="read",
    note="parametre sırası kaynaktaki gibi [b, m]",
)
def fit_line(x: ArrayLike, y: ArrayLike, sigma_y: ArrayLike, *, through_origin: bool = False) -> LinearFit:
    """y = m x + b doğrusunu uydurur.

    Parameters
    ----------
    x, y : ndarray, biçim (N,)
    sigma_y : float ya da ndarray (N,)
        y belirsizlikleri.
    orijinden : bool
        Doğruysa kesişim sıfıra sabitlenir ve yalnızca eğim uydurulur.

    Returns
    -------
    DogrusalUyum
        ``parametreler`` sırası [b, m]; ``orijinden`` ise yalnızca [m].

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
