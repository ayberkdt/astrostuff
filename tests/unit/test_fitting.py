import numpy as np
import pytest
from scipy.linalg import cholesky

from kozmos.statistics import fitting


def test_ki_kare_dogrudan_ters_ile():
    u = np.random.default_rng(11)
    a = u.normal(size=(40, 40))
    c = a @ a.T + 40 * np.eye(40)
    r = u.normal(size=40)
    assert fitting.chi_square(r, cholesky(c, lower=True)) == pytest.approx(r @ np.linalg.inv(c) @ r, rel=1e-11)


def test_dogru_uydur_polyfit_ile():
    u = np.random.default_rng(12)
    x = np.linspace(0, 10, 50)
    sigma = u.uniform(0.5, 2.0, 50)
    y = 2.5 * x - 1.0 + u.normal(0, sigma)
    sonuc = fitting.fit_line(x, y, sigma)
    coefficient, kov = np.polyfit(x, y, 1, w=1 / sigma, cov="unscaled")
    assert sonuc.parameters == pytest.approx(coefficient[::-1], rel=1e-10)  # bizde sıra [b, m]
    assert sonuc.covariance == pytest.approx(kov[::-1, ::-1], rel=1e-9)
    assert sonuc.degrees_of_freedom == 48
    assert np.all(np.abs(sonuc.parameters - [-1.0, 2.5]) < 4 * sonuc.uncertainties)


def test_tam_kovaryans_kosegen_ile_ayni():
    u = np.random.default_rng(13)
    x = np.linspace(-3, 3, 30)
    sigma = u.uniform(0.2, 1.0, 30)
    y = 0.7 * x + 3.0 + u.normal(0, sigma)
    a = np.column_stack([np.ones_like(x), x])
    kosegen = fitting.linear_least_squares(a, y, sigma)
    tam = fitting.linear_least_squares(a, y, np.diag(sigma**2))
    assert kosegen.parameters == pytest.approx(tam.parameters, rel=1e-12)
    assert kosegen.chi_square == pytest.approx(tam.chi_square, rel=1e-12)


def test_orijinden_gecen_dogru():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    sonuc = fitting.fit_line(x, 3.0 * x, 1.0, through_origin=True)
    assert sonuc.parameters == pytest.approx([3.0])
    assert sonuc.chi_square == pytest.approx(0.0, abs=1e-20)


def test_boyut_denetimi():
    with pytest.raises(ValueError):
        fitting.linear_least_squares(np.ones((5, 2)), np.ones(4), 1.0)
    with pytest.raises(ValueError):
        fitting.linear_least_squares(np.ones((5, 1)), np.ones(5), np.ones(3))
