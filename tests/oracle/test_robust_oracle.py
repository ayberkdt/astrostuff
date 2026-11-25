import numpy as np
import pytest
from astropy import stats as astropy_stats

from kozmos.statistics import robust

pytestmark = pytest.mark.oracle


def _kirli(tohum, n=500):
    u = np.random.default_rng(tohum)
    x = u.normal(10.0, 2.0, n)
    x[:25] += u.uniform(20, 200, 25)  # aykırı değerler
    return x


@pytest.mark.parametrize("tohum", [71, 72, 73])
def test_konum_ve_olcek_astropy_ile(tohum):
    x = _kirli(tohum)
    assert float(robust.mad_scale(x)) == pytest.approx(astropy_stats.mad_std(x), rel=1e-13)
    assert robust.biweight_location(x) == pytest.approx(astropy_stats.biweight_location(x), rel=1e-12)
    assert robust.biweight_scale(x) == pytest.approx(astropy_stats.biweight_scale(x), rel=1e-12)
    # Aykırı değerlere karşın gerçek değerlere yakın kalırlar; ortalama ve standart sapma kalmaz.
    assert abs(robust.biweight_location(x) - 10.0) < 0.4 and abs(robust.biweight_scale(x) - 2.0) < 0.4
    assert abs(x.mean() - 10.0) > 1.0


@pytest.mark.parametrize("tohum", [74, 75])
def test_sigma_kirp_astropy_ile(tohum):
    x = _kirli(tohum)
    kahin = astropy_stats.sigma_clip(x, sigma=3.0, maxiters=5, cenfunc="median", stdfunc="std")
    assert np.array_equal(robust.sigma_clip(x, sigma=3.0, iterations=5), kahin.mask)
    kup = np.random.default_rng(tohum).normal(size=(9, 20, 30))
    kup[0, 3, 4] = 80.0
    kahin = astropy_stats.sigma_clip(kup, sigma=2.5, maxiters=5, cenfunc="median", stdfunc="std", axis=0)
    bizim = robust.sigma_clip(kup, sigma=2.5, iterations=5, axis=0)
    assert np.array_equal(bizim, kahin.mask) and bizim[0, 3, 4]


def test_mad_eksen_ve_sifir_olcek():
    x = np.random.default_rng(76).normal(size=(40, 7))
    assert robust.mad_scale(x, axis=0) == pytest.approx(astropy_stats.mad_std(x, axis=0), rel=1e-13)
    sabit = np.full(20, 3.0)
    assert robust.biweight_location(sabit) == 3.0 and robust.biweight_scale(sabit) == 0.0
