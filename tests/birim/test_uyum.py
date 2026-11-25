import numpy as np
import pytest
from scipy.linalg import cholesky

from kozmos.istatistik import uyum


def test_ki_kare_dogrudan_ters_ile():
    u = np.random.default_rng(11)
    a = u.normal(size=(40, 40))
    c = a @ a.T + 40 * np.eye(40)
    r = u.normal(size=40)
    assert uyum.ki_kare(r, cholesky(c, lower=True)) == pytest.approx(r @ np.linalg.inv(c) @ r, rel=1e-11)


def test_dogru_uydur_polyfit_ile():
    u = np.random.default_rng(12)
    x = np.linspace(0, 10, 50)
    sigma = u.uniform(0.5, 2.0, 50)
    y = 2.5 * x - 1.0 + u.normal(0, sigma)
    sonuc = uyum.dogru_uydur(x, y, sigma)
    katsayi, kov = np.polyfit(x, y, 1, w=1 / sigma, cov="unscaled")
    assert sonuc.parametreler == pytest.approx(katsayi[::-1], rel=1e-10)  # bizde sıra [b, m]
    assert sonuc.kovaryans == pytest.approx(kov[::-1, ::-1], rel=1e-9)
    assert sonuc.serbestlik_derecesi == 48
    assert np.all(np.abs(sonuc.parametreler - [-1.0, 2.5]) < 4 * sonuc.belirsizlikler)


def test_tam_kovaryans_kosegen_ile_ayni():
    u = np.random.default_rng(13)
    x = np.linspace(-3, 3, 30)
    sigma = u.uniform(0.2, 1.0, 30)
    y = 0.7 * x + 3.0 + u.normal(0, sigma)
    a = np.column_stack([np.ones_like(x), x])
    kosegen = uyum.dogrusal_en_kucuk_kareler(a, y, sigma)
    tam = uyum.dogrusal_en_kucuk_kareler(a, y, np.diag(sigma**2))
    assert kosegen.parametreler == pytest.approx(tam.parametreler, rel=1e-12)
    assert kosegen.ki_kare == pytest.approx(tam.ki_kare, rel=1e-12)


def test_orijinden_gecen_dogru():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    sonuc = uyum.dogru_uydur(x, 3.0 * x, 1.0, orijinden=True)
    assert sonuc.parametreler == pytest.approx([3.0])
    assert sonuc.ki_kare == pytest.approx(0.0, abs=1e-20)


def test_boyut_denetimi():
    with pytest.raises(ValueError):
        uyum.dogrusal_en_kucuk_kareler(np.ones((5, 2)), np.ones(4), 1.0)
    with pytest.raises(ValueError):
        uyum.dogrusal_en_kucuk_kareler(np.ones((5, 1)), np.ones(5), np.ones(3))
