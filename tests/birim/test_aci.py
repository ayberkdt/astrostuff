import math

import numpy as np

from kozmos.temel import aci


def test_donusum_sabitleri():
    assert aci.derece_den_radyan(180.0) == math.pi
    assert np.isclose(aci.radyan_dan_derece(math.pi / 2), 90.0, rtol=0, atol=1e-13)
    assert np.isclose(aci.yaysaniye_den_radyan(648000.0), math.pi, rtol=1e-15)


def test_ayriklik_bilinen_degerler():
    assert aci.acisal_ayriklik(0.0, 0.0, 0.0, 0.0) == 0.0
    assert np.isclose(aci.acisal_ayriklik(0.0, 0.0, math.pi, 0.0), math.pi)
    assert np.isclose(aci.acisal_ayriklik(0.3, math.pi / 2, 2.0, 0.0), math.pi / 2)
    # Kosinüs formülünün duyarlılık kaybettiği çok küçük açı.
    kucuk = 1e-9
    assert np.isclose(aci.acisal_ayriklik(1.0, 0.0, 1.0 + kucuk, 0.0), kucuk, rtol=1e-6)
