import math

import numpy as np
import pytest

from kozmos.efemeris import kepler
from kozmos.temel import sabitler as s


def test_dunya_yorunge_donemi():
    # a = 1 au ve nominal GM_Güneş [Prsa2016] için dönem bir Gauss yılıdır: 365,2569 gün dolayı.
    donem_gun = float(kepler.yorunge_donemi_s(s.ASTRONOMIK_BIRIM.deger, s.GM_GUNES.deger)) / s.GUN_SANIYE.deger
    assert donem_gun == pytest.approx(365.2569, abs=1e-3)


def test_dairesel_yorunge_hizi():
    r, mu = 7.0e6, s.GM_YER.deger
    assert float(kepler.yorunge_hizi_m_s(r, r, mu)) == pytest.approx(math.sqrt(mu / r), rel=1e-15)


def test_ozel_durumlar():
    assert float(kepler.dis_merkezli_anomali(0.0, 0.5)) == pytest.approx(0.0, abs=1e-15)
    assert float(kepler.dis_merkezli_anomali(math.pi, 0.5)) == pytest.approx(math.pi, rel=1e-14)
    m = np.linspace(-10, 10, 41)
    assert np.allclose(kepler.dis_merkezli_anomali(m, 0.0), m)  # e = 0: E = M
    with pytest.raises(ValueError):
        kepler.dis_merkezli_anomali(1.0, 1.0)
