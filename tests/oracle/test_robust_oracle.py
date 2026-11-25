import numpy as np
import pytest
from astropy import stats as astropy_stats

from kozmos.statistics import robust

pytestmark = pytest.mark.oracle


def _contaminated(seed, n=500):
    u = np.random.default_rng(seed)
    x = u.normal(10.0, 2.0, n)
    x[:25] += u.uniform(20, 200, 25)  # outliers
    return x


@pytest.mark.parametrize("seed", [71, 72, 73])
def test_location_and_scale_against_astropy(seed):
    x = _contaminated(seed)
    assert float(robust.mad_scale(x)) == pytest.approx(astropy_stats.mad_std(x), rel=1e-13)
    assert robust.biweight_location(x) == pytest.approx(astropy_stats.biweight_location(x), rel=1e-12)
    assert robust.biweight_scale(x) == pytest.approx(astropy_stats.biweight_scale(x), rel=1e-12)
    # Despite the outliers they stay close to the true values; the mean and standard deviation do not.
    assert abs(robust.biweight_location(x) - 10.0) < 0.4 and abs(robust.biweight_scale(x) - 2.0) < 0.4
    assert abs(x.mean() - 10.0) > 1.0


@pytest.mark.parametrize("seed", [74, 75])
def test_sigma_clip_against_astropy(seed):
    x = _contaminated(seed)
    oracle = astropy_stats.sigma_clip(x, sigma=3.0, maxiters=5, cenfunc="median", stdfunc="std")
    assert np.array_equal(robust.sigma_clip(x, sigma=3.0, iterations=5), oracle.mask)
    cube = np.random.default_rng(seed).normal(size=(9, 20, 30))
    cube[0, 3, 4] = 80.0
    oracle = astropy_stats.sigma_clip(cube, sigma=2.5, maxiters=5, cenfunc="median", stdfunc="std", axis=0)
    ours = robust.sigma_clip(cube, sigma=2.5, iterations=5, axis=0)
    assert np.array_equal(ours, oracle.mask) and ours[0, 3, 4]


def test_mad_axis_and_zero_scale():
    x = np.random.default_rng(76).normal(size=(40, 7))
    assert robust.mad_scale(x, axis=0) == pytest.approx(astropy_stats.mad_std(x, axis=0), rel=1e-13)
    constant = np.full(20, 3.0)
    assert robust.biweight_location(constant) == 3.0 and robust.biweight_scale(constant) == 0.0
