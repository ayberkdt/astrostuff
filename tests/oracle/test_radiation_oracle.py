import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import BlackBody

from kozmos.radiation import magnitudes

pytestmark = pytest.mark.oracle


@pytest.mark.parametrize("temperature", [30.0, 3000.0, 5772.0, 2e4, 1e6])
def test_planck_against_astropy(temperature):
    frequency = np.logspace(9, 17, 200)
    astropy_value = BlackBody(temperature * u.K)(frequency * u.Hz).to_value(u.W / u.m**2 / u.Hz / u.sr)
    ours = magnitudes.planck_frequency(frequency, temperature)
    valid = astropy_value > 1e-250
    assert np.max(np.abs(ours[valid] / astropy_value[valid] - 1)) < 1e-7
