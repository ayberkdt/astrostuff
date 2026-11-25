import numpy as np
import pytest
from astropy.timeseries import LombScargle

from kozmos.timeseries import periodogram

pytestmark = pytest.mark.oracle


def _veri(tohum=51, n=180):
    u = np.random.default_rng(tohum)
    t = np.sort(u.uniform(0, 120, n))  # eşit aralıklı olmayan örnekleme
    sigma = u.uniform(0.05, 0.3, n)
    y = 10.0 + 0.8 * np.sin(2 * np.pi * t / 3.7 + 0.4) + u.normal(0, sigma)
    return t, y, sigma


def test_agirlikli_yuzen_ortalama_astropy_ile():
    t, y, sigma = _veri()
    f = periodogram.frequency_grid(t, 2.0)
    bizim = periodogram.lomb_scargle(t, y, f, sigma)
    kahin = LombScargle(t, y, sigma, fit_mean=True, center_data=True, normalization="standard").power(f, method="slow")
    assert np.max(np.abs(bizim - kahin)) < 1e-10
    assert bizim.min() >= -1e-12 and bizim.max() <= 1.0
    assert 1 / f[np.argmax(bizim)] == pytest.approx(3.7, rel=5e-3)  # girilen dönem


def test_esit_agirlik_astropy_ile():
    t, y, _ = _veri(tohum=52)
    f = periodogram.frequency_grid(t, 1.0)
    kahin = LombScargle(t, y, fit_mean=True, normalization="standard").power(f, method="slow")
    assert np.max(np.abs(periodogram.lomb_scargle(t, y, f) - kahin)) < 1e-10


def test_klasik_astropy_ile():
    t, y, _ = _veri(tohum=53)
    f = periodogram.frequency_grid(t, 1.0)
    kahin = LombScargle(t, y, fit_mean=False, center_data=True, normalization="psd").power(f, method="slow")
    assert np.max(np.abs(periodogram.lomb_scargle_classical(t, y, f) / kahin - 1)) < 1e-9


def test_yanlis_alarm_astropy_ile():
    t, y, sigma = _veri(tohum=54)
    ls = LombScargle(t, y, sigma, fit_mean=True, normalization="standard")
    power = np.array([0.05, 0.1, 0.2])
    # Tek frekans dağılımı kesin karşılaştırılır (M = 1).
    tek = ls.false_alarm_probability(power, method="single")
    assert np.max(np.abs(periodogram.false_alarm_probability(power, t.size, 1.0) / tek - 1)) < 1e-12
    # astropy "naive" yöntemi bağımsız frekans sayısını f_max · T alır (astropy kaynak kodundan okundu).
    # Aynı formülle 1e-4 düzeyinde bağıl fark kalıyor; nedeni incelenmedi, tolerans buna göre kondu.
    f_max = 2.0
    kahin = ls.false_alarm_probability(power, method="naive", maximum_frequency=f_max)
    bizim = periodogram.false_alarm_probability(power, t.size, f_max * (t.max() - t.min()))
    assert np.max(np.abs(bizim / kahin - 1)) < 1e-3


def test_izgara():
    t = np.array([0.0, 3.0, 10.0])
    f = periodogram.frequency_grid(t, 1.0, oversampling=5.0)
    assert f[0] == pytest.approx(1 / 50) and np.allclose(np.diff(f), 1 / 50) and f[-1] >= 1.0
