"""Comparison of the distance measures with astropy.cosmology.

Astropy includes radiation by default; it is switched off with ``Tcmb0=0``
because [Hogg1999] Eq. 14 has no radiation term.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, LambdaCDM

from kozmos.cosmology import distances

pytestmark = pytest.mark.oracle

Z = np.concatenate([np.logspace(-4, 1, 60), [0.01, 0.5, 1.0, 2.26]])

MODELS = [
    (distances.Cosmology.flat(70.0, 0.3), FlatLambdaCDM(70.0, 0.3, Tcmb0=0)),
    (distances.Cosmology.flat(67.4, 0.315), FlatLambdaCDM(67.4, 0.315, Tcmb0=0)),
    (distances.Cosmology(72.0, 0.3, 0.6), LambdaCDM(72.0, 0.3, 0.6, Tcmb0=0)),  # open
    (distances.Cosmology(72.0, 0.4, 0.8), LambdaCDM(72.0, 0.4, 0.8, Tcmb0=0)),  # closed
    (distances.Cosmology.flat(70.0, 0.3, w=-0.9), FlatwCDM(70.0, 0.3, w0=-0.9, Tcmb0=0)),
    (distances.Cosmology.flat(100.0, 1.0), FlatLambdaCDM(100.0, 1.0, Tcmb0=0)),  # Einstein-de Sitter
]


def _relative(a, b):
    return np.max(np.abs(a / b - 1.0))


@pytest.mark.parametrize("ours,astropy_model", MODELS)
def test_distances(ours, astropy_model):
    assert _relative(distances.e_function(Z, ours), astropy_model.efunc(Z)) < 1e-12
    assert _relative(distances.comoving_distance_mpc(Z, ours), astropy_model.comoving_distance(Z).value) < 1e-9
    assert (
        _relative(
            distances.transverse_comoving_distance_mpc(Z, ours), astropy_model.comoving_transverse_distance(Z).value
        )
        < 1e-9
    )
    assert (
        _relative(distances.angular_diameter_distance_mpc(Z, ours), astropy_model.angular_diameter_distance(Z).value)
        < 1e-9
    )
    assert _relative(distances.luminosity_distance_mpc(Z, ours), astropy_model.luminosity_distance(Z).value) < 1e-9
    assert np.max(np.abs(distances.distance_modulus(Z, ours) - astropy_model.distmod(Z).value)) < 1e-8


@pytest.mark.parametrize("ours,astropy_model", MODELS)
def test_time_and_volume(ours, astropy_model):
    assert _relative(distances.lookback_time_gyr(Z, ours), astropy_model.lookback_time(Z).to_value(u.Gyr)) < 1e-8
    volume = astropy_model.differential_comoving_volume(Z).to_value(u.Mpc**3 / u.sr)
    assert _relative(distances.comoving_volume_element_mpc3_sr(Z, ours), volume) < 1e-8
    assert abs(distances.hubble_distance_mpc(ours) / astropy_model.hubble_distance.value - 1) < 1e-12
    assert abs(distances.hubble_time_gyr(ours) / astropy_model.hubble_time.to_value(u.Gyr) - 1) < 1e-9


@pytest.mark.parametrize("ours,astropy_model", MODELS)
def test_volume_and_two_redshifts(ours, astropy_model):
    volume = astropy_model.comoving_volume(Z).to_value(u.Mpc**3)
    # At small z the curved branch is the difference of two close terms; 1e-6 for z > 0.05, 1e-3 for the rest.
    relative = np.abs(distances.comoving_volume_mpc3(Z, ours) / volume - 1.0)
    assert np.max(relative[Z > 0.05]) < 1e-6 and np.max(relative) < 1e-3
    if ours.omega_k >= -1e-12:
        z1, z2 = np.array([0.1, 0.5, 1.0]), np.array([0.3, 2.0, 3.0])
        oracle = astropy_model.angular_diameter_distance_z1z2(z1, z2).value
        assert _relative(distances.angular_diameter_distance_z1_z2_mpc(z1, z2, ours), oracle) < 1e-9
    else:
        with pytest.raises(ValueError):
            distances.angular_diameter_distance_z1_z2_mpc(0.1, 0.5, ours)
