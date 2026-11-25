import warnings

import erfa
import numpy as np
import pytest

from kozmos.zaman import olcekler

pytestmark = pytest.mark.kahin


def test_tai_eksi_utc_erfa_ile():
    u = np.random.default_rng(61)
    for _ in range(2000):
        yil, ay, gun = int(u.integers(1960, 2027)), int(u.integers(1, 13)), int(u.integers(1, 29))
        kesir = float(u.uniform(0, 1))
        assert olcekler.tai_eksi_utc_s(yil, ay, gun, kesir) == pytest.approx(erfa.dat(yil, ay, gun, kesir), abs=1e-12)


def test_artik_saniye_sinirlari():
    # Tablodaki her değişiklik tarihinde ve bir gün öncesinde ERFA ile aynı değer.
    for yil, ay, _ in olcekler._DEGISIKLIKLER[14:]:
        assert olcekler.tai_eksi_utc_s(yil, ay, 1) == erfa.dat(yil, ay, 1, 0.0)
    assert olcekler.tai_eksi_utc_s(2016, 12, 31) == 36.0
    assert olcekler.tai_eksi_utc_s(2017, 1, 1) == 37.0
    assert olcekler.tt_eksi_utc_s(2017, 1, 1) == pytest.approx(69.184)


def test_gecerlilik_siniri():
    with pytest.raises(ValueError):
        olcekler.tai_eksi_utc_s(1959, 12, 31)
    with pytest.raises(ValueError):
        olcekler.tai_eksi_utc_s(2020, 1, 1, 1.5)
    with pytest.warns(UserWarning):
        olcekler.tai_eksi_utc_s(olcekler.TABLO_SURUM_YILI + 6, 1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        olcekler.tai_eksi_utc_s(olcekler.TABLO_SURUM_YILI + 5, 1, 1)
