import math

import numpy as np
import pytest

from kozmos.core import constants
from kozmos.ephemeris import kepler


def test_earth_orbital_period():
    # For a = 1 au and the nominal GM_Sun [Prsa2016] the period is one Gaussian year: about 365.2569 days.
    period_days = (
        float(kepler.orbital_period_s(constants.ASTRONOMICAL_UNIT.value, constants.GM_SUN.value))
        / constants.DAY_SECONDS.value
    )
    assert period_days == pytest.approx(365.2569, abs=1e-3)


def test_circular_orbit_speed():
    r, mu = 7.0e6, constants.GM_EARTH.value
    assert float(kepler.orbital_speed_m_s(r, r, mu)) == pytest.approx(math.sqrt(mu / r), rel=1e-15, abs=0)


def test_special_cases():
    assert float(kepler.eccentric_anomaly(0.0, 0.5)) == pytest.approx(0.0, abs=1e-15)
    assert float(kepler.eccentric_anomaly(math.pi, 0.5)) == pytest.approx(math.pi, rel=1e-14, abs=0)
    m = np.linspace(-10, 10, 41)
    assert np.allclose(kepler.eccentric_anomaly(m, 0.0), m)  # e = 0: E = M
    with pytest.raises(ValueError):
        kepler.eccentric_anomaly(1.0, 1.0)
