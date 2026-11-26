import numpy as np
import pytest
from scipy.linalg import cholesky

from kozmos.statistics import fitting


def test_chi_square_against_direct_inverse():
    u = np.random.default_rng(11)
    a = u.normal(size=(40, 40))
    c = a @ a.T + 40 * np.eye(40)
    r = u.normal(size=40)
    assert fitting.chi_square(r, cholesky(c, lower=True)) == pytest.approx(r @ np.linalg.inv(c) @ r, rel=1e-11, abs=0)


def test_fit_line_against_polyfit():
    u = np.random.default_rng(12)
    x = np.linspace(0, 10, 50)
    sigma = u.uniform(0.5, 2.0, 50)
    y = 2.5 * x - 1.0 + u.normal(0, sigma)
    result = fitting.fit_line(x, y, sigma)
    coefficient, cov = np.polyfit(x, y, 1, w=1 / sigma, cov="unscaled")
    assert result.parameters == pytest.approx(coefficient[::-1], rel=1e-10, abs=0)  # our order is [b, m]
    assert result.covariance == pytest.approx(cov[::-1, ::-1], rel=1e-9, abs=0)
    assert result.degrees_of_freedom == 48
    assert np.all(np.abs(result.parameters - [-1.0, 2.5]) < 4 * result.uncertainties)


def test_full_covariance_same_as_diagonal():
    u = np.random.default_rng(13)
    x = np.linspace(-3, 3, 30)
    sigma = u.uniform(0.2, 1.0, 30)
    y = 0.7 * x + 3.0 + u.normal(0, sigma)
    a = np.column_stack([np.ones_like(x), x])
    diagonal = fitting.linear_least_squares(a, y, sigma)
    full = fitting.linear_least_squares(a, y, np.diag(sigma**2))
    assert diagonal.parameters == pytest.approx(full.parameters, rel=1e-12, abs=0)
    assert diagonal.chi_square == pytest.approx(full.chi_square, rel=1e-12, abs=0)


def test_line_through_origin():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = fitting.fit_line(x, 3.0 * x, 1.0, through_origin=True)
    assert result.parameters == pytest.approx([3.0], rel=1e-6, abs=0)
    assert result.chi_square == pytest.approx(0.0, abs=1e-20)


def test_shape_check():
    with pytest.raises(ValueError):
        fitting.linear_least_squares(np.ones((5, 2)), np.ones(4), 1.0)
    with pytest.raises(ValueError):
        fitting.linear_least_squares(np.ones((5, 1)), np.ones(5), np.ones(3))
