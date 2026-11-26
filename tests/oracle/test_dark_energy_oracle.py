"""Comparison of the w0-wa background, radiation and dark energy functions with astropy.cosmology.

What is compared, for every model: ``efunc`` (e_function), the comoving, transverse comoving, angular diameter
and luminosity distances, the lookback time, the age, Om(z), Ode(z), Ok(z) (curved models), Ogamma(z) + Onu(z)
(radiation models), H(z), the critical density, w(z) and de_density_scale(z). The deceleration parameter has no
astropy counterpart; it is compared with -1 - dlnE/dlna from central differences of astropy ``efunc``.

Radiation: astropy ``Tcmb0 = 2.7255 K`` with its default Neff = 3.04 massless neutrinos; the sum
``Ogamma0 + Onu0`` is passed as ``omega_r`` and astropy's own ``Ode0`` (flat models) is reproduced by
``Cosmology.flat``. Without radiation astropy is used with ``Tcmb0 = 0``.

Tolerances are the measured maxima rounded up by about one decade (measured values in the comments).
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM, FlatwCDM, w0waCDM

from kozmos.cosmology import dark_energy as de
from kozmos.cosmology import distances

pytestmark = pytest.mark.oracle

Z = np.concatenate([np.logspace(-4, 1, 60), [0.01, 0.5, 1.0, 2.26]])  # 1e-4 <= z <= 10
Z_HIGH = np.concatenate([np.logspace(-3, np.log10(1100.0), 60), [1100.0]])  # up to the last-scattering surface


def _radiation_model(model):
    return distances.Cosmology(
        model.H0.value,
        model.Om0,
        model.Ode0,
        w=getattr(model, "w0", -1.0),
        wa=getattr(model, "wa", 0.0),
        omega_r=model.Ogamma0 + model.Onu0,
    )


def _build():
    models = {}
    models["lcdm"] = (distances.Cosmology.flat(70.0, 0.3), FlatLambdaCDM(70.0, 0.3, Tcmb0=0))
    models["wcdm"] = (distances.Cosmology.flat(70.0, 0.3, w=-0.8), FlatwCDM(70.0, 0.3, w0=-0.8, Tcmb0=0))
    models["w0wa"] = (
        distances.Cosmology.flat(70.0, 0.3, w=-0.8, wa=0.5),
        Flatw0waCDM(70.0, 0.3, w0=-0.8, wa=0.5, Tcmb0=0),
    )
    models["w0wa_phantom"] = (
        distances.Cosmology.flat(67.4, 0.315, w=-1.1, wa=-0.4),
        Flatw0waCDM(67.4, 0.315, w0=-1.1, wa=-0.4, Tcmb0=0),
    )
    models["open"] = (
        distances.Cosmology(72.0, 0.3, 0.6, w=-0.9, wa=0.3),
        w0waCDM(72.0, 0.3, 0.6, w0=-0.9, wa=0.3, Tcmb0=0),
    )
    models["closed"] = (
        distances.Cosmology(72.0, 0.4, 0.8, w=-1.1, wa=-0.2),
        w0waCDM(72.0, 0.4, 0.8, w0=-1.1, wa=-0.2, Tcmb0=0),
    )
    for name, astropy_model in {
        "radiation_flat": Flatw0waCDM(67.7, 0.31, w0=-0.95, wa=0.2, Tcmb0=2.7255),
        "radiation_curved": w0waCDM(70.0, 0.3, 0.68, w0=-0.9, wa=0.1, Tcmb0=2.7255),
        "radiation_lcdm": FlatLambdaCDM(70.0, 0.3, Tcmb0=2.7255),
    }.items():
        models[name] = (_radiation_model(astropy_model), astropy_model)
    return models


MODELS = _build()
RADIATION = [name for name in MODELS if name.startswith("radiation")]


def _relative(a, b):
    return np.max(np.abs(np.asarray(a) / np.asarray(b) - 1.0))


@pytest.mark.parametrize("name", list(MODELS))
def test_background_functions(name):
    ours, astropy_model = MODELS[name]
    # measured maxima over all models: 2.2e-16 (efunc, H), 1.1e-15 (critical density), 6.7e-16 (others)
    assert _relative(distances.e_function(Z, ours), astropy_model.efunc(Z)) < 1e-14
    assert _relative(de.hubble_parameter_km_s_mpc(Z, ours), astropy_model.H(Z).value) < 1e-14
    assert _relative(de.omega_matter(Z, ours), astropy_model.Om(Z)) < 1e-14
    assert _relative(de.omega_de(Z, ours), astropy_model.Ode(Z)) < 1e-14
    assert _relative(de.de_rho_ratio(Z, ours), astropy_model.de_density_scale(Z)) < 1e-14
    assert np.max(np.abs(de.de_eos(Z, ours) - astropy_model.w(Z))) < 1e-15  # measured 0
    critical = astropy_model.critical_density(Z).to_value(u.kg / u.m**3)
    assert _relative(de.critical_density_kg_m3(Z, ours), critical) < 1e-14
    if abs(ours.omega_k) > 1e-12:
        assert _relative(de.omega_curvature(Z, ours), astropy_model.Ok(Z)) < 1e-14  # measured 2e-15
    if ours.omega_r > 0:
        assert _relative(de.omega_radiation(Z, ours), astropy_model.Ogamma(Z) + astropy_model.Onu(Z)) < 1e-14


@pytest.mark.parametrize("name", list(MODELS))
def test_distances_and_times(name):
    ours, astropy_model = MODELS[name]
    # measured maxima for 1e-4 <= z <= 10 over all models: dc 7.9e-12, dm/da/dl 7.9e-12, lookback 1.7e-12
    assert _relative(distances.comoving_distance_mpc(Z, ours), astropy_model.comoving_distance(Z).value) < 1e-10
    assert (
        _relative(
            distances.transverse_comoving_distance_mpc(Z, ours), astropy_model.comoving_transverse_distance(Z).value
        )
        < 1e-10
    )
    assert (
        _relative(distances.angular_diameter_distance_mpc(Z, ours), astropy_model.angular_diameter_distance(Z).value)
        < 1e-10
    )
    assert _relative(distances.luminosity_distance_mpc(Z, ours), astropy_model.luminosity_distance(Z).value) < 1e-10
    assert _relative(distances.lookback_time_gyr(Z, ours), astropy_model.lookback_time(Z).to_value(u.Gyr)) < 1e-10
    # measured age difference 7e-11 at most: astropy integrates with scipy quad; the project's own value is exact
    # to 1e-15 against a tight quad (tests/unit/test_dark_energy.py)
    assert _relative(de.age_gyr(Z, ours), astropy_model.age(Z).to_value(u.Gyr)) < 1e-9


@pytest.mark.parametrize("name", RADIATION)
def test_up_to_last_scattering(name):
    ours, astropy_model = MODELS[name]
    # measured maxima for z <= 1100: distances 2.5e-12, lookback 7.2e-14; the 64-point rule in ln(1+z) is not
    # the limit (the old rule in z gave 9e-4 for the same integral)
    assert (
        _relative(distances.comoving_distance_mpc(Z_HIGH, ours), astropy_model.comoving_distance(Z_HIGH).value) < 1e-10
    )
    assert (
        _relative(distances.luminosity_distance_mpc(Z_HIGH, ours), astropy_model.luminosity_distance(Z_HIGH).value)
        < 1e-10
    )
    assert (
        _relative(
            distances.angular_diameter_distance_mpc(Z_HIGH, ours), astropy_model.angular_diameter_distance(Z_HIGH).value
        )
        < 1e-10
    )
    assert (
        _relative(distances.lookback_time_gyr(Z_HIGH, ours), astropy_model.lookback_time(Z_HIGH).to_value(u.Gyr))
        < 1e-10
    )
    # astropy's age at z ~ 1000 differs from a tight quad by up to 9.2e-7 (tests/unit/test_dark_energy.py shows that
    # the project's value agrees with that quad to 1e-12): the tolerance here is astropy's, not ours
    assert _relative(de.age_gyr(Z_HIGH, ours), astropy_model.age(Z_HIGH).to_value(u.Gyr)) < 2e-6
    assert _relative(de.omega_radiation(Z_HIGH, ours), astropy_model.Ogamma(Z_HIGH) + astropy_model.Onu(Z_HIGH)) < 1e-14
    assert (
        _relative(
            de.critical_density_kg_m3(Z_HIGH, ours), astropy_model.critical_density(Z_HIGH).to_value(u.kg / u.m**3)
        )
        < 1e-14
    )


@pytest.mark.parametrize("name", list(MODELS))
def test_deceleration_parameter_against_finite_differences_of_astropy_efunc(name):
    ours, astropy_model = MODELS[name]
    z = np.array([0.0, 0.3, 1.0, 3.0, 10.0])
    h = 1e-4
    lna = -np.log1p(z)

    def ln_e(n):
        return np.log(astropy_model.efunc(np.expm1(-n)))

    q_astropy = -1.0 - (ln_e(lna + h) - ln_e(lna - h)) / (2.0 * h)
    # measured maximum difference over all models 5.4e-9 (truncation error of the central difference, h^2)
    assert np.max(np.abs(de.deceleration_parameter(z, ours) - q_astropy)) < 1e-7


def test_equality_and_onset_redshifts_against_astropy_density_parameters():
    ours, astropy_model = MODELS["lcdm"]
    z_eq = de.matter_de_equality_redshift(ours)
    assert astropy_model.Om(z_eq) == pytest.approx(astropy_model.Ode(z_eq), rel=1e-12, abs=0)
    # the acceleration onset: astropy q from -1 - dlnE/dlna vanishes
    z_acc = de.acceleration_onset_redshift(ours)
    h = 1e-4
    n = -np.log1p(z_acc)
    q = -1.0 - (np.log(astropy_model.efunc(np.expm1(-(n + h)))) - np.log(astropy_model.efunc(np.expm1(-(n - h))))) / (
        2 * h
    )
    assert abs(q) < 1e-7
