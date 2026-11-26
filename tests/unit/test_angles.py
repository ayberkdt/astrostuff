import math

import numpy as np

from kozmos.core import angles


def test_conversion_constants():
    assert angles.deg_to_rad(180.0) == math.pi
    assert np.isclose(angles.rad_to_deg(math.pi / 2), 90.0, rtol=0, atol=1e-13)
    assert np.isclose(angles.arcsec_to_rad(648000.0), math.pi, rtol=1e-15)


def test_separation_known_values():
    assert angles.angular_separation(0.0, 0.0, 0.0, 0.0) == 0.0
    assert np.isclose(angles.angular_separation(0.0, 0.0, math.pi, 0.0), math.pi)
    assert np.isclose(angles.angular_separation(0.3, math.pi / 2, 2.0, 0.0), math.pi / 2)
    # A very small angle at which the cosine formula loses precision.
    small = 1e-9
    assert np.isclose(angles.angular_separation(1.0, 0.0, 1.0 + small, 0.0), small, rtol=1e-6, atol=0.0)
