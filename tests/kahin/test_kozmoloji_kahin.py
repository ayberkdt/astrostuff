"""Uzaklık ölçülerinin astropy.cosmology ile karşılaştırması.

Astropy varsayılan olarak radyasyon içerir; ``Tcmb0=0`` ile kapatılır, çünkü
[Hogg1999] Denk. 14'te radyasyon terimi yoktur.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, LambdaCDM

from kozmos.kozmoloji import uzaklik as uz

pytestmark = pytest.mark.kahin

Z = np.concatenate([np.logspace(-4, 1, 60), [0.01, 0.5, 1.0, 2.26]])

MODELLER = [
    (uz.Kozmoloji.duz(70.0, 0.3), FlatLambdaCDM(70.0, 0.3, Tcmb0=0)),
    (uz.Kozmoloji.duz(67.4, 0.315), FlatLambdaCDM(67.4, 0.315, Tcmb0=0)),
    (uz.Kozmoloji(72.0, 0.3, 0.6), LambdaCDM(72.0, 0.3, 0.6, Tcmb0=0)),  # açık
    (uz.Kozmoloji(72.0, 0.4, 0.8), LambdaCDM(72.0, 0.4, 0.8, Tcmb0=0)),  # kapalı
    (uz.Kozmoloji.duz(70.0, 0.3, w=-0.9), FlatwCDM(70.0, 0.3, w0=-0.9, Tcmb0=0)),
    (uz.Kozmoloji.duz(100.0, 1.0), FlatLambdaCDM(100.0, 1.0, Tcmb0=0)),  # Einstein-de Sitter
]


def _bagil(a, b):
    return np.max(np.abs(a / b - 1.0))


@pytest.mark.parametrize("bizim,astropy_modeli", MODELLER)
def test_uzakliklar(bizim, astropy_modeli):
    assert _bagil(uz.e_fonksiyonu(Z, bizim), astropy_modeli.efunc(Z)) < 1e-12
    assert _bagil(uz.esyuruyen_uzaklik_mpc(Z, bizim), astropy_modeli.comoving_distance(Z).value) < 1e-9
    assert _bagil(uz.enine_esyuruyen_uzaklik_mpc(Z, bizim), astropy_modeli.comoving_transverse_distance(Z).value) < 1e-9
    assert _bagil(uz.acisal_cap_uzakligi_mpc(Z, bizim), astropy_modeli.angular_diameter_distance(Z).value) < 1e-9
    assert _bagil(uz.isinim_gucu_uzakligi_mpc(Z, bizim), astropy_modeli.luminosity_distance(Z).value) < 1e-9
    assert np.max(np.abs(uz.uzaklik_modulu(Z, bizim) - astropy_modeli.distmod(Z).value)) < 1e-8


@pytest.mark.parametrize("bizim,astropy_modeli", MODELLER)
def test_zaman_ve_hacim(bizim, astropy_modeli):
    assert _bagil(uz.geriye_bakis_zamani_gyr(Z, bizim), astropy_modeli.lookback_time(Z).to_value(u.Gyr)) < 1e-8
    hacim = astropy_modeli.differential_comoving_volume(Z).to_value(u.Mpc**3 / u.sr)
    assert _bagil(uz.esyuruyen_hacim_elemani_mpc3(Z, bizim), hacim) < 1e-8
    assert abs(uz.hubble_uzakligi_mpc(bizim) / astropy_modeli.hubble_distance.value - 1) < 1e-12
    assert abs(uz.hubble_zamani_gyr(bizim) / astropy_modeli.hubble_time.to_value(u.Gyr) - 1) < 1e-9


@pytest.mark.parametrize("bizim,astropy_modeli", MODELLER)
def test_hacim_ve_iki_kirmiziya_kayma(bizim, astropy_modeli):
    hacim = astropy_modeli.comoving_volume(Z).to_value(u.Mpc**3)
    # Küçük z'de eğrilikli dal iki yakın terimin farkıdır; z > 0,05 için 1e-6, geri kalanı 1e-3.
    bagil = np.abs(uz.esyuruyen_hacim_mpc3(Z, bizim) / hacim - 1.0)
    assert np.max(bagil[Z > 0.05]) < 1e-6 and np.max(bagil) < 1e-3
    if bizim.omega_k >= -1e-12:
        z1, z2 = np.array([0.1, 0.5, 1.0]), np.array([0.3, 2.0, 3.0])
        kahin = astropy_modeli.angular_diameter_distance_z1z2(z1, z2).value
        assert _bagil(uz.acisal_cap_uzakligi_z1_z2_mpc(z1, z2, bizim), kahin) < 1e-9
    else:
        with pytest.raises(ValueError):
            uz.acisal_cap_uzakligi_z1_z2_mpc(0.1, 0.5, bizim)
