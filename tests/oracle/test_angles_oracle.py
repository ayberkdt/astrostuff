import erfa
import numpy as np
import pytest

from kozmos.core import angles

pytestmark = pytest.mark.oracle


def test_separation_vs_erfa():
    u = np.random.default_rng(1)
    l1, l2 = u.uniform(0, 2 * np.pi, (2, 2000))
    b1, b2 = np.arcsin(u.uniform(-1, 1, (2, 2000)))
    assert np.max(np.abs(angles.angular_separation(l1, b1, l2, b2) - erfa.seps(l1, b1, l2, b2))) < 1e-14


def test_wrapping_vs_erfa():
    a = np.random.default_rng(2).uniform(-50, 50, 2000)
    assert np.max(np.abs(angles.wrap_0_2pi(a) - erfa.anp(a))) < 1e-13
    assert np.max(np.abs(angles.wrap_pm_pi(a) - erfa.anpm(a))) < 1e-13


def test_spherical_vector_vs_erfa():
    u = np.random.default_rng(3)
    longitude, latitude = u.uniform(0, 2 * np.pi, 500), np.arcsin(u.uniform(-1, 1, 500))
    assert np.max(np.abs(angles.spherical_to_unit_vector(longitude, latitude) - erfa.s2c(longitude, latitude))) < 1e-15
