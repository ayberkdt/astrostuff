"""Uzaklık ölçülerinin astropy.cosmology ile karşılaştırması.

Astropy varsayılan olarak radyasyon içerir; ``Tcmb0=0`` ile kapatılır, çünkü
[Hogg1999] Denk. 14'te radyasyon terimi yoktur.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, LambdaCDM

from kozmos.cosmology import distances

pytestmark = pytest.mark.oracle

Z = np.concatenate([np.logspace(-4, 1, 60), [0.01, 0.5, 1.0, 2.26]])

MODELLER = [
    (distances.Cosmology.flat(70.0, 0.3), FlatLambdaCDM(70.0, 0.3, Tcmb0=0)),
    (distances.Cosmology.flat(67.4, 0.315), FlatLambdaCDM(67.4, 0.315, Tcmb0=0)),
    (distances.Cosmology(72.0, 0.3, 0.6), LambdaCDM(72.0, 0.3, 0.6, Tcmb0=0)),  # açık
    (distances.Cosmology(72.0, 0.4, 0.8), LambdaCDM(72.0, 0.4, 0.8, Tcmb0=0)),  # kapalı
    (distances.Cosmology.flat(70.0, 0.3, w=-0.9), FlatwCDM(70.0, 0.3, w0=-0.9, Tcmb0=0)),
    (distances.Cosmology.flat(100.0, 1.0), FlatLambdaCDM(100.0, 1.0, Tcmb0=0)),  # Einstein-de Sitter
]


def _bagil(a, b):
    return np.max(np.abs(a / b - 1.0))


@pytest.mark.parametrize("bizim,astropy_modeli", MODELLER)
def test_uzakliklar(bizim, astropy_modeli):
    assert _bagil(distances.e_function(Z, bizim), astropy_modeli.efunc(Z)) < 1e-12
    assert _bagil(distances.comoving_distance_mpc(Z, bizim), astropy_modeli.comoving_distance(Z).value) < 1e-9
    assert (
        _bagil(
            distances.transverse_comoving_distance_mpc(Z, bizim), astropy_modeli.comoving_transverse_distance(Z).value
        )
        < 1e-9
    )
    assert (
        _bagil(distances.angular_diameter_distance_mpc(Z, bizim), astropy_modeli.angular_diameter_distance(Z).value)
        < 1e-9
    )
    assert _bagil(distances.luminosity_distance_mpc(Z, bizim), astropy_modeli.luminosity_distance(Z).value) < 1e-9
    assert np.max(np.abs(distances.distance_modulus(Z, bizim) - astropy_modeli.distmod(Z).value)) < 1e-8


@pytest.mark.parametrize("bizim,astropy_modeli", MODELLER)
def test_zaman_ve_hacim(bizim, astropy_modeli):
    assert _bagil(distances.lookback_time_gyr(Z, bizim), astropy_modeli.lookback_time(Z).to_value(u.Gyr)) < 1e-8
    hacim = astropy_modeli.differential_comoving_volume(Z).to_value(u.Mpc**3 / u.sr)
    assert _bagil(distances.comoving_volume_element_mpc3(Z, bizim), hacim) < 1e-8
    assert abs(distances.hubble_distance_mpc(bizim) / astropy_modeli.hubble_distance.value - 1) < 1e-12
    assert abs(distances.hubble_time_gyr(bizim) / astropy_modeli.hubble_time.to_value(u.Gyr) - 1) < 1e-9


@pytest.mark.parametrize("bizim,astropy_modeli", MODELLER)
def test_hacim_ve_iki_kirmiziya_kayma(bizim, astropy_modeli):
    hacim = astropy_modeli.comoving_volume(Z).to_value(u.Mpc**3)
    # Küçük z'de eğrilikli dal iki yakın terimin farkıdır; z > 0,05 için 1e-6, geri kalanı 1e-3.
    bagil = np.abs(distances.comoving_volume_mpc3(Z, bizim) / hacim - 1.0)
    assert np.max(bagil[Z > 0.05]) < 1e-6 and np.max(bagil) < 1e-3
    if bizim.omega_k >= -1e-12:
        z1, z2 = np.array([0.1, 0.5, 1.0]), np.array([0.3, 2.0, 3.0])
        kahin = astropy_modeli.angular_diameter_distance_z1z2(z1, z2).value
        assert _bagil(distances.angular_diameter_distance_z1_z2_mpc(z1, z2, bizim), kahin) < 1e-9
    else:
        with pytest.raises(ValueError):
            distances.angular_diameter_distance_z1_z2_mpc(0.1, 0.5, bizim)
