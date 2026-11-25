import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from kozmos.ephemeris import kepler

anomalies = st.floats(-20.0, 20.0)
eccentricities = st.floats(0.0, 0.999)


@given(anomalies, eccentricities)
@settings(max_examples=500)
def test_kepler_equation_residual(m, e):
    ecc_anomaly = float(kepler.eccentric_anomaly(m, e))
    assert abs(ecc_anomaly - e * math.sin(ecc_anomaly) - m) < 1e-12


@given(anomalies, eccentricities)
@settings(max_examples=300)
def test_orbit_equation_two_forms(m, e):
    # r/a = 1 − e cos E = (1 − e²) / (1 + e cos f)
    ecc_anomaly = float(kepler.eccentric_anomaly(m, e))
    f = float(kepler.true_anomaly(ecc_anomaly, e))
    lhs = 1 - e * math.cos(ecc_anomaly)
    rhs = (1 - e * e) / (1 + e * math.cos(f))
    assert abs(lhs - rhs) < 1e-9 * max(1.0, lhs)


def test_residual_on_grid():
    m = np.linspace(0, 2 * np.pi, 400)[:, None]
    e = np.linspace(0, 0.999, 300)[None, :]
    ecc_anomaly = kepler.eccentric_anomaly(m, e)
    assert np.max(np.abs(ecc_anomaly - e * np.sin(ecc_anomaly) - m)) < 1e-12
