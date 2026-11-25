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

from kozmos._kaynak import kaynak

Dizi = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DogrusalUyum:
    """Doğrusal uyumun sonucu."""

    parametreler: Dizi
    kovaryans: Dizi
    ki_kare: float
    serbestlik_derecesi: int

    @property
    def belirsizlikler(self) -> Dizi:
        """Parametre standart belirsizlikleri: kovaryans köşegeninin karekökü."""
        return np.sqrt(np.diag(self.kovaryans))


def _kovaryans_matrisi(sigma_ya_da_kovaryans: ArrayLike, n: int) -> Dizi:
    c = np.asarray(sigma_ya_da_kovaryans, dtype=float)
    if c.ndim == 0:
        return np.eye(n) * float(c) ** 2
    if c.ndim == 1:
        if c.shape != (n,):
            raise ValueError("belirsizlik dizisinin boyu veriyle uyuşmuyor")
        return np.diag(c**2)
    if c.shape != (n, n):
        raise ValueError("kovaryans matrisi N×N olmalı")
    return c


@kaynak(
    "HoggBovyLang2010",
    yer="TeX etiketi eq:chisquared",
    teyit="okundu",
    not_="χ² = [Y − AX]ᵀ C⁻¹ [Y − AX]; ters alınmaz, Cholesky çarpanıyla çözülür",
)
def ki_kare(artik: ArrayLike, cholesky_alt: ArrayLike) -> float:
    """Kovaryanslı χ² değerini hesaplar: χ² = rᵀ C⁻¹ r.

    Parametreler
    ------------
    artik : ndarray, biçim (N,)
        Veri eksi model.
    cholesky_alt : ndarray, biçim (N, N)
        Kovaryansın alt üçgen Cholesky çarpanı L (C = L Lᵀ).

    Döndürür
    --------
    float

    Kaynaklar
    ---------
    [HoggBovyLang2010] eq:chisquared.

    Geçerlilik
    ----------
    C simetrik ve pozitif tanımlı olmalıdır.

    Doğrulama
    ---------
    tests/birim/test_uyum.py: doğrudan ters alma ile karşılaştırma.
    """
    y = solve_triangular(np.asarray(cholesky_alt, dtype=float), np.asarray(artik, dtype=float), lower=True)
    return float(y @ y)


@kaynak(
    "HoggBovyLang2010",
    yer="TeX etiketi eq:lsf ve parametre kovaryansı denklemi",
    teyit="okundu",
    not_="X = [AᵀC⁻¹A]⁻¹[AᵀC⁻¹Y]; parametre kovaryansı [AᵀC⁻¹A]⁻¹",
)
def dogrusal_en_kucuk_kareler(tasarim: ArrayLike, y: ArrayLike, sigma_ya_da_kovaryans: ArrayLike) -> DogrusalUyum:
    """Doğrusal modeli ağırlıklı en küçük karelerle uydurur.

    Model Y = A X; çözüm X = [Aᵀ C⁻¹ A]⁻¹ [Aᵀ C⁻¹ Y].

    Parametreler
    ------------
    tasarim : ndarray, biçim (N, P)
        Tasarım matrisi A.
    y : ndarray, biçim (N,)
        Ölçümler.
    sigma_ya_da_kovaryans : float, ndarray (N,) ya da ndarray (N, N)
        Ölçüm standart belirsizlikleri ya da tam kovaryans matrisi.

    Kaynaklar
    ---------
    [HoggBovyLang2010] eq:lsf.

    Geçerlilik
    ----------
    Belirsizlikler yalnızca y ekseninde, Gauss ve doğru bilinen olmalıdır;
    kaynak bu varsayımların çoğu gerçek veride sağlanmadığını vurgular.

    Doğrulama
    ---------
    tests/birim/test_uyum.py: numpy.polyfit (ağırlıklı) ile karşılaştırma.
    """
    a = np.asarray(tasarim, dtype=float)
    yy = np.asarray(y, dtype=float)
    if a.ndim != 2 or a.shape[0] != yy.shape[0]:
        raise ValueError("tasarım matrisi (N, P), y (N,) olmalı")
    c = _kovaryans_matrisi(sigma_ya_da_kovaryans, yy.shape[0])
    carpan = cho_factor(c, lower=True)
    cinv_a = cho_solve(carpan, a)
    cinv_y = cho_solve(carpan, yy)
    normal = a.T @ cinv_a
    kov = np.linalg.inv(normal)
    x = kov @ (a.T @ cinv_y)
    artik = yy - a @ x
    return DogrusalUyum(x, kov, float(artik @ cho_solve(carpan, artik)), yy.shape[0] - a.shape[1])


@kaynak(
    "HoggBovyLang2010",
    yer="TeX etiketi eq:fofx, eq:lsf",
    teyit="okundu",
    not_="parametre sırası kaynaktaki gibi [b, m]",
)
def dogru_uydur(x: ArrayLike, y: ArrayLike, sigma_y: ArrayLike, *, orijinden: bool = False) -> DogrusalUyum:
    """y = m x + b doğrusunu uydurur.

    Parametreler
    ------------
    x, y : ndarray, biçim (N,)
    sigma_y : float ya da ndarray (N,)
        y belirsizlikleri.
    orijinden : bool
        Doğruysa kesişim sıfıra sabitlenir ve yalnızca eğim uydurulur.

    Döndürür
    --------
    DogrusalUyum
        ``parametreler`` sırası [b, m]; ``orijinden`` ise yalnızca [m].

    Kaynaklar
    ---------
    [HoggBovyLang2010] eq:fofx, eq:lsf.

    Doğrulama
    ---------
    tests/birim/test_uyum.py
    """
    xx = np.asarray(x, dtype=float)
    a = xx[:, None] if orijinden else np.column_stack([np.ones_like(xx), xx])
    return dogrusal_en_kucuk_kareler(a, y, sigma_y)
