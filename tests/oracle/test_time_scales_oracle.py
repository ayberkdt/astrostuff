import warnings

import erfa
import numpy as np
import pytest

from kozmos.time import scales

pytestmark = pytest.mark.oracle


def test_tai_minus_utc_vs_erfa():
    u = np.random.default_rng(61)
    for _ in range(2000):
        year, month, day = int(u.integers(1960, 2027)), int(u.integers(1, 13)), int(u.integers(1, 29))
        fraction = float(u.uniform(0, 1))
        assert scales.tai_minus_utc_s(year, month, day, fraction) == pytest.approx(
            erfa.dat(year, month, day, fraction), abs=1e-12
        )


def test_leap_second_boundaries():
    # On every change date in the table the value equals ERFA's.
    for year, month, _ in scales._CHANGES[14:]:
        assert scales.tai_minus_utc_s(year, month, 1) == erfa.dat(year, month, 1, 0.0)
    assert scales.tai_minus_utc_s(2016, 12, 31) == 36.0
    assert scales.tai_minus_utc_s(2017, 1, 1) == 37.0
    assert scales.tt_minus_utc_s(2017, 1, 1) == pytest.approx(69.184)


def test_validity_limit():
    with pytest.raises(ValueError):
        scales.tai_minus_utc_s(1959, 12, 31)
    with pytest.raises(ValueError):
        scales.tai_minus_utc_s(2020, 1, 1, 1.5)
    with pytest.warns(UserWarning):
        scales.tai_minus_utc_s(scales.TABLE_RELEASE_YEAR + 6, 1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scales.tai_minus_utc_s(scales.TABLE_RELEASE_YEAR + 5, 1, 1)
