import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import BlackBody

from kozmos.radiation import magnitudes

pytestmark = pytest.mark.oracle


@pytest.mark.parametrize("sicaklik", [30.0, 3000.0, 5772.0, 2e4, 1e6])
def test_planck_astropy_ile(sicaklik):
    frequency = np.logspace(9, 17, 200)
    astropy_degeri = BlackBody(sicaklik * u.K)(frequency * u.Hz).to_value(u.W / u.m**2 / u.Hz / u.sr)
    bizim = magnitudes.planck_frequency(frequency, sicaklik)
    gecerli = astropy_degeri > 1e-250
    assert np.max(np.abs(bizim[gecerli] / astropy_degeri[gecerli] - 1)) < 1e-7
