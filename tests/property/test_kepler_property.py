import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from kozmos.ephemeris import kepler

anomaliler = st.floats(-20.0, 20.0)
dis_merkezlikler = st.floats(0.0, 0.999)


@given(anomaliler, dis_merkezlikler)
@settings(max_examples=500)
def test_kepler_denklemi_artigi(m, e):
    buyuk_e = float(kepler.eccentric_anomaly(m, e))
    assert abs(buyuk_e - e * math.sin(buyuk_e) - m) < 1e-12


@given(anomaliler, dis_merkezlikler)
@settings(max_examples=300)
def test_yorunge_denklemi_iki_bicimi(m, e):
    # r/a = 1 − e cos E = (1 − e²) / (1 + e cos f)
    buyuk_e = float(kepler.eccentric_anomaly(m, e))
    f = float(kepler.true_anomaly(buyuk_e, e))
    sol = 1 - e * math.cos(buyuk_e)
    sag = (1 - e * e) / (1 + e * math.cos(f))
    assert abs(sol - sag) < 1e-9 * max(1.0, sol)


def test_izgara_uzerinde_artik():
    m = np.linspace(0, 2 * np.pi, 400)[:, None]
    e = np.linspace(0, 0.999, 300)[None, :]
    buyuk_e = kepler.eccentric_anomaly(m, e)
    assert np.max(np.abs(buyuk_e - e * np.sin(buyuk_e) - m)) < 1e-12
