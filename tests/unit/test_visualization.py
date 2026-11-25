import numpy as np
import pytest

from kozmos.visualization import stretch


def test_asinh_limitleri():
    x = np.array([0.0, 1e-4, 1e-3])
    # Küçük sinyalde doğrusal: f ≈ α (x − m) ([Lupton2004]: asinh(x) ~ x, x << 1).
    assert stretch.asinh_stretch(x, black=0.0, alpha=2.0, q=8.0) == pytest.approx(2.0 * x, rel=1e-4, abs=1e-12)
    # Q → 0 doğrusal germe.
    assert stretch.asinh_stretch(5.0, black=1.0, alpha=0.5, q=0.0) == pytest.approx(2.0)
    # Tekdüze artan ve siyah altı sıfıra kırpılır.
    y = stretch.asinh_stretch(np.linspace(-5, 1e4, 500), black=0.0, alpha=0.1, q=10.0)
    assert np.all(np.diff(y) >= 0) and y[0] == 0.0
    # Parlak uçta logaritmik: girdinin 100 katı çıktıyı ln(100)/Q kadar artırır.
    b = stretch.asinh_stretch(np.array([1e5, 1e7]), black=0.0, alpha=1.0, q=5.0)
    assert b[1] - b[0] == pytest.approx(np.log(100.0) / 5.0, rel=1e-6)


def test_lupton_renk_korunur():
    u = np.random.default_rng(101)
    r, g, b = u.uniform(0.1, 50.0, (3, 30, 40))
    rgb = stretch.lupton_rgb(r, g, b, alpha=0.05, q=8.0)
    assert rgb.shape == (30, 40, 3) and rgb.min() >= 0.0 and rgb.max() <= 1.0 + 1e-12
    # Renk oranları parlaklıktan bağımsız korunur: R/G = r/g, G/B = g/b.
    assert rgb[..., 0] / rgb[..., 1] == pytest.approx(r / g, rel=1e-10)
    assert rgb[..., 1] / rgb[..., 2] == pytest.approx(g / b, rel=1e-10)
    # Aynı renkte 1000 kat parlak piksel doyar ama rengi aynı kalır (beyaza dönmez).
    parlak = stretch.lupton_rgb(np.array([[3e4]]), np.array([[1e4]]), np.array([[5e3]]), alpha=0.05, q=8.0)[0, 0]
    assert (
        parlak.max() == pytest.approx(1.0)
        and parlak[0] / parlak[1] == pytest.approx(3.0)
        and parlak[1] / parlak[2] == pytest.approx(2.0)
    )
    # Sıfır yeğinlik siyahtır.
    assert np.array_equal(
        stretch.lupton_rgb(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), alpha=1.0), np.zeros((2, 2, 3))
    )


def test_yuzdelik_olcekleme():
    x = np.arange(1001, dtype=float)
    y = stretch.percentile_scale(x, 10.0, 90.0)
    assert y.min() == 0.0 and y.max() == 1.0 and y[500] == pytest.approx(0.5)
    kirli = x.copy()
    kirli[3] = np.nan
    assert np.isfinite(stretch.percentile_scale(kirli)).all()
    assert np.array_equal(stretch.percentile_scale(np.ones((4, 4))), np.zeros((4, 4)))
