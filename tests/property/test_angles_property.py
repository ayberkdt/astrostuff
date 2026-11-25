import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from kozmos.core import angles

angles_rad = st.floats(-1e4, 1e4, allow_nan=False)
longitudes = st.floats(0, 2 * math.pi, exclude_max=True)
latitudes = st.floats(-math.pi / 2 + 1e-9, math.pi / 2 - 1e-9)


@given(angles_rad)
def test_zero_to_two_pi_range(a):
    w = float(angles.wrap_0_2pi(a))
    assert 0.0 <= w < 2 * math.pi
    assert abs(math.remainder(w - a, 2 * math.pi)) < 1e-9


@given(angles_rad)
def test_minus_pi_to_plus_pi_range(a):
    assert -math.pi <= float(angles.wrap_pm_pi(a)) < math.pi


@given(longitudes, latitudes)
def test_spherical_round_trip(longitude, latitude):
    v = angles.spherical_to_unit_vector(longitude, latitude)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-14
    longitude2, latitude2 = angles.unit_vector_to_spherical(v)
    assert float(angles.angular_separation(longitude, latitude, longitude2, latitude2)) < 1e-12


@given(longitudes, latitudes, longitudes, latitudes)
def test_separation_symmetric_and_in_range(l1, b1, l2, b2):
    d = float(angles.angular_separation(l1, b1, l2, b2))
    assert 0.0 <= d <= math.pi + 1e-15
    assert abs(d - float(angles.angular_separation(l2, b2, l1, b1))) < 1e-14
