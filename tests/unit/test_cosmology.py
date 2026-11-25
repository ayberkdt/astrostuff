"""Numerical values and closed-form solutions from the source [Hogg1999]."""

import numpy as np
import pytest

from kozmos.cosmology import distances

EDS = distances.Cosmology.flat(100.0, 1.0)  # Einstein-de Sitter, h = 1


def test_hubble_distance_and_time_paper_values():
    # [Hogg1999] Eq. 4: D_H = 3000 h^-1 Mpc; Eq. 3: t_H = 9.78e9 h^-1 yr (three significant digits).
    assert round(distances.hubble_distance_mpc(EDS), -1) == 3000.0
    assert round(distances.hubble_time_gyr(EDS), 2) == 9.78


def test_einstein_de_sitter_closed_form():
    # Ωm = 1, ΩΛ = 0: E(z) = (1+z)^{3/2}; D_C = 2 D_H [1 − (1+z)^{-1/2}].
    z = np.array([0.0, 0.1, 1.0, 3.0, 10.0])
    expected = 2.0 * distances.hubble_distance_mpc(EDS) * (1.0 - (1.0 + z) ** -0.5)
    assert np.allclose(distances.comoving_distance_mpc(z, EDS), expected, rtol=1e-12, atol=1e-12)
    # Lookback time: t_L = (2/3) t_H [1 − (1+z)^{-3/2}].
    expected_t = (2.0 / 3.0) * distances.hubble_time_gyr(EDS) * (1.0 - (1.0 + z) ** -1.5)
    assert np.allclose(distances.lookback_time_gyr(z, EDS), expected_t, rtol=1e-12, atol=1e-14)


def test_hogg_equation_17_zero_lambda():
    # [Hogg1999] Eq. 17: closed form of D_M for ΩΛ = 0.
    om = 0.3
    k = distances.Cosmology(70.0, om, 0.0)
    z = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    dh = distances.hubble_distance_mpc(k)
    closed_form = dh * 2 * (2 - om * (1 - z) - (2 - om) * np.sqrt(1 + om * z)) / (om**2 * (1 + z))
    assert np.allclose(distances.transverse_comoving_distance_mpc(z, k), closed_form, rtol=1e-10)


def test_omega_k():
    assert distances.Cosmology.flat(70, 0.3).omega_k == pytest.approx(0.0, abs=1e-15)
    assert distances.Cosmology(70, 0.3, 0.6).omega_k == pytest.approx(0.1)


def test_invalid_input():
    with pytest.raises(ValueError):
        distances.Cosmology(-1.0, 0.3, 0.7)
    with pytest.raises(ValueError):
        distances.comoving_distance_mpc(-0.1, EDS)


def test_scalar_and_array():
    k = distances.Cosmology.flat(70.0, 0.3)
    assert np.ndim(distances.luminosity_distance_mpc(0.5, k)) == 0
    assert distances.luminosity_distance_mpc(np.zeros((2, 3)) + 0.5, k).shape == (2, 3)


def test_doppler_and_peculiar_velocity():
    from kozmos.core.constants import SPEED_OF_LIGHT_KM_S

    v = np.array([-3000.0, -300.0, 0.0, 300.0, 30000.0, 150000.0])
    z = distances.doppler_redshift(v)
    assert np.allclose(distances.doppler_velocity_km_s(z), v, rtol=1e-12, atol=1e-9)
    assert float(distances.doppler_redshift(30.0)) == pytest.approx(30.0 / SPEED_OF_LIGHT_KM_S, rel=1e-4)  # z ≈ v/c
    # The z_obs generated from the rule (1 + z_obs) = (1 + z_cos)(1 + v/c) gives back the same v at small velocity.
    z_cos, velocity = 0.05, 300.0
    z_obs = (1 + z_cos) * (1 + velocity / SPEED_OF_LIGHT_KM_S) - 1
    assert float(distances.peculiar_velocity_km_s(z_obs, z_cos)) == pytest.approx(velocity, rel=1e-12)
