"""Numerical values and closed-form solutions from the source [Hogg1999]."""

import numpy as np
import pytest
from scipy.integrate import quad

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
    assert float(distances.doppler_redshift(30.0)) == pytest.approx(
        30.0 / SPEED_OF_LIGHT_KM_S, rel=1e-4, abs=0
    )  # z ≈ v/c
    # The z_obs generated from the rule (1 + z_obs) = (1 + z_cos)(1 + v/c) gives back the same v at small velocity.
    z_cos, velocity = 0.05, 300.0
    z_obs = (1 + z_cos) * (1 + velocity / SPEED_OF_LIGHT_KM_S) - 1
    assert float(distances.peculiar_velocity_km_s(z_obs, z_cos)) == pytest.approx(velocity, rel=1e-12, abs=0)


def test_non_finite_redshifts_give_nan_without_aborting_the_call():
    # before the fix: OverflowError "cannot convert float infinity to integer" or ValueError "cannot convert float NaN
    # to integer" for z = inf or NaN; the old code returned NaN
    k = distances.Cosmology.flat(70.0, 0.3, omega_r=8.5e-5)
    z = np.array([0.5, np.nan, 2.0, np.inf, 1000.0])
    finite = np.isfinite(z)
    for function in (
        distances.comoving_distance_mpc,
        distances.lookback_time_gyr,
        distances.luminosity_distance_mpc,
        distances.angular_diameter_distance_mpc,
    ):
        out = function(z, k)
        assert out.shape == z.shape, function.__name__
        assert np.all(np.isnan(out[~finite])), function.__name__
        # the finite entries are those of the call without the non-finite ones (same panels: z <= 1000 sets them)
        assert np.array_equal(out[finite], function(z[finite], k)), function.__name__
        assert np.isnan(function(np.inf, k)) and np.isnan(function(np.nan, k)), function.__name__
    # a huge finite z next to inf: the panel count comes from the finite entries only
    big = distances.comoving_distance_mpc(np.array([1e30, np.inf]), k)
    assert big[0] == distances.comoving_distance_mpc(1e30, k) and np.isnan(big[1])
    # the scalar and 2-D shapes are kept
    assert np.ndim(distances.comoving_distance_mpc(np.inf, k)) == 0
    two_d = distances.comoving_distance_mpc(np.array([[0.5, np.nan], [np.inf, 1.0]]), k)
    assert two_d.shape == (2, 2) and np.isnan(two_d[0, 1]) and np.isnan(two_d[1, 0]) and np.isfinite(two_d[1, 1])


def test_panels_are_needed_at_extreme_redshift():
    # D_C(z)/D_H = int_0^{ln(1+z)} e^u / E(z(u)) du with radiation, to z = 1e30 (u = 69); scipy quad with a break
    # point at every e-fold is the reference. One 64-point Gauss-Legendre rule over the whole interval (what a panel
    # width of 1000 gives) is exact to 2e-15 up to z = 1e12 but 1.7e-9 at z = 1e30, where the integrand decays over
    # 69 e-folds; the panels of 2.5 in ln(1+z) give 1.1e-16.
    k = distances.Cosmology.flat(70.0, 0.3, omega_r=8.5e-5)
    z = 1e30
    top = np.log1p(z)
    reference = quad(
        lambda u: np.exp(u) / distances.e_function(np.expm1(u), k),
        0.0,
        top,
        epsabs=0.0,
        epsrel=1e-13,
        limit=2000,
        points=np.arange(1.0, int(top) + 1.0),
    )[0]
    ours = distances.comoving_distance_mpc(z, k) / distances.hubble_distance_mpc(k)
    nodes, weights = distances._NODES, distances._WEIGHTS
    zp = np.expm1(0.5 * top * (nodes + 1.0))
    single = 0.5 * top * np.sum(weights * (1.0 + zp) / distances.e_function(zp, k))
    assert abs(ours / reference - 1.0) < 1e-12  # measured 1.1e-16
    assert abs(single / reference - 1.0) > 1e-10  # measured 1.7e-9: the single-panel rule fails the same tolerance
