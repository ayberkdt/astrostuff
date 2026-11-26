"""The advanced modules checked against the mechanics foundation, and against each other.

Each test joins two packages that were written from different sources, so an error in either one
shows up as a disagreement:

- the geodesic equation (Christoffel symbols, Carroll) against the Euler-Lagrange equations of
  L = (1/2) g_mu_nu u^mu u^nu (Lagrangian package, Banados and Reyes);
- Killing constants against Noether charges of the same Lagrangian;
- the circular-orbit formulas of the Kerr module (Abramowicz and Fragile) against the geodesic
  equation of the Kerr metric;
- the Klein-Gordon equation of the scalar-field module (Copeland et al.) against the Lagrangian
  a^3 (phi_dot^2 / 2 - V);
- the cosmological redshift against a null geodesic of the FLRW metric;
- the symplectic integrators against the Lagrangian integrator.

Tolerances are about ten times the measured differences, which are given next to each one.
"""

from __future__ import annotations

import numpy as np
import pytest

from kozmos.black_holes import kerr
from kozmos.core.constants import GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT
from kozmos.cosmology import scalar_field
from kozmos.mechanics import integrators, lagrangian, noether
from kozmos.relativity import geodesics, metrics

MASS = 1.0  # geometric units: lengths in units of G M / c^2
SPIN = 0.7
X = np.array([0.3, 7.0, 1.1, 0.4])  # (t, r, theta, phi), off the equator
U = np.array([1.3, -0.11, 0.021, 0.035])


def _kerr(x):
    return metrics.kerr_metric(x, MASS, SPIN)


def _geodesic_lagrangian(q, v, t):
    return 0.5 * v @ _kerr(q) @ v


def test_geodesic_equation_is_the_euler_lagrange_equation_of_the_metric_lagrangian():
    from_christoffels = geodesics.geodesic_acceleration(_kerr, X, U)
    from_lagrangian = lagrangian.euler_lagrange_accelerations(_geodesic_lagrangian, X, U)
    assert np.allclose(from_lagrangian, from_christoffels, rtol=1e-9, atol=1e-12)  # measured 1.9e-11 relative


def test_killing_constants_are_noether_charges():
    energy, angular_momentum = geodesics.stationary_axisymmetric_constants(_kerr, X, U)
    # the source's charge is Q = K - p . delta q; with delta q = unit shift of t or phi and K = 0, Q = -p_t or -p_phi
    charge_t = noether.noether_charge(_geodesic_lagrangian, X, U, 0.0, lambda q, v, t: np.array([1.0, 0, 0, 0]))
    charge_phi = noether.noether_charge(_geodesic_lagrangian, X, U, 0.0, lambda q, v, t: np.array([0, 0, 0, 1.0]))
    assert charge_t == pytest.approx(energy, rel=1e-13, abs=0)
    assert charge_phi == pytest.approx(-angular_momentum, rel=1e-13, abs=0)
    # both shifts are symmetries (defect 0); a shift of r is not
    acceleration = geodesics.geodesic_acceleration(_kerr, X, U)
    for index, is_symmetry in ((0, True), (3, True), (1, False)):
        shift = np.eye(4)[index]
        defect = noether.symmetry_defect(_geodesic_lagrangian, X, U, 0.0, lambda q, v, t, s=shift: s, a=acceleration)
        assert (abs(defect) < 1e-14) == is_symmetry


def test_energy_function_of_the_metric_lagrangian_is_half_the_norm():
    # L is homogeneous of degree 2 in the velocities, so h = p.v - L = L
    h = lagrangian.jacobi_function(_geodesic_lagrangian, X, U)
    assert h == pytest.approx(0.5 * geodesics.tangent_norm_squared(_kerr, X, U), rel=1e-13, abs=0)


@pytest.mark.parametrize(("radius", "spin"), [(6.0, 0.0), (4.0, 0.7), (2.5, 0.95), (12.0, -0.7), (9.5, -0.95)])
def test_kerr_circular_orbit_formulas_satisfy_the_geodesic_equation(radius, spin):
    """Signed spin of the Kerr module (negative = retrograde orbit) against a metric with a > 0."""
    a = abs(spin)
    direction = 1.0 if spin >= 0 else -1.0
    mass_kg = 1.0e31
    time_unit_s = GRAVITATIONAL_CONSTANT.value * mass_kg / SPEED_OF_LIGHT.value**3
    omega = direction * float(kerr.kerr_circular_orbit_angular_velocity_rad_s(mass_kg, radius, spin)) * time_unit_s

    def metric(x):
        return metrics.kerr_metric(x, MASS, a)

    x = np.array([0.0, radius, np.pi / 2, 0.0])
    g = metric(x)
    u_t = 1.0 / np.sqrt(-(g[0, 0] + 2.0 * g[0, 3] * omega + g[3, 3] * omega**2))
    u = u_t * np.array([1.0, 0.0, 0.0, omega])
    assert geodesics.tangent_norm_squared(metric, x, u) == pytest.approx(-1.0, rel=1e-13, abs=0)
    # a circular orbit needs no radial (and no polar) acceleration: measured |a| <= 2.8e-17
    assert np.all(np.abs(geodesics.geodesic_acceleration(metric, x, u)) < 1e-13)
    energy, angular_momentum = geodesics.stationary_axisymmetric_constants(metric, x, u)
    assert energy == pytest.approx(float(kerr.kerr_circular_orbit_specific_energy(radius, spin)), rel=1e-12, abs=0)
    assert direction * angular_momentum == pytest.approx(
        float(kerr.kerr_circular_orbit_specific_angular_momentum_rg(radius, spin)), rel=1e-12, abs=0
    )


def test_klein_gordon_equation_follows_from_the_scalar_field_lagrangian():
    """L = a^3 (phi_dot^2/2 - V) with a = t^(2/3) gives phi_ddot = -3 H phi_dot - V'(phi), H = 2/(3t)."""

    def lagrangian_density(q, v, t):
        return t**2 * (0.5 * v[0] ** 2 - np.exp(-1.5 * q[0]) - 0.3 * q[0] ** 2)

    phi, phi_dot, t = 0.4, -0.25, 1.7
    expected = -3.0 * (2.0 / (3.0 * t)) * phi_dot - (-1.5 * np.exp(-1.5 * phi) + 0.6 * phi)
    result = lagrangian.euler_lagrange_accelerations(lagrangian_density, [phi], [phi_dot], t)
    assert result[0] == pytest.approx(expected, rel=1e-11, abs=0)  # measured 2.0e-14
    # the energy function is a^3 rho with the density of the scalar-field module, and it is not conserved
    potential = np.exp(-1.5 * phi) + 0.3 * phi**2
    h = lagrangian.jacobi_function(lagrangian_density, [phi], [phi_dot], t)
    assert h == pytest.approx(t**2 * float(scalar_field.scalar_rho(phi_dot, potential)), rel=1e-13, abs=0)


def test_cosmological_redshift_from_a_null_geodesic_of_the_flrw_metric():
    """Along a radial null geodesic the photon energy u^t falls as 1/a, i.e. 1 + z = a_obs / a_emit."""

    def metric(x):
        return metrics.flrw_metric(x, lambda t: t ** (2.0 / 3.0), 0.0)

    x0 = np.array([1.0, 0.5, np.pi / 2, 0.0])
    u0 = geodesics.normalize_tangent(metric, x0, np.array([1.0, 0.0, 0.0]), kind="null")
    x, u = geodesics.integrate_geodesic(metric, x0, u0, np.linspace(0.0, 3.0, 7))
    scale = x[:, 0] ** (2.0 / 3.0)
    assert x[-1, 0] > 2.0  # the photon has travelled over more than a doubling of cosmic time
    assert np.allclose(u[:, 0] * scale, u[0, 0] * scale[0], rtol=1e-9, atol=0)  # measured 2.0e-12
    assert np.max(np.abs([geodesics.tangent_norm_squared(metric, xi, ui) for xi, ui in zip(x, u, strict=True)])) < 1e-9


def test_symplectic_integrator_agrees_with_the_lagrangian_integrator():
    def pendulum(q, v, t):
        return 0.5 * v[0] ** 2 + np.cos(q[0])

    q0, v0, end = np.array([1.2]), np.array([0.3]), 5.0
    q_ref, v_ref = lagrangian.integrate_lagrangian(pendulum, q0, v0, [0.0, end], rtol=1e-12, atol=1e-14)
    errors = []
    for n in (500, 1000):
        _, q, v = integrators.integrate(lambda q, t: -np.sin(q), q0, v0, end / n, n, scheme="forest_ruth")
        errors.append(max(abs(q[-1, 0] - q_ref[-1, 0]), abs(v[-1, 0] - v_ref[-1, 0])))
    assert errors[1] < 1e-9  # measured 5.4e-12 (8.4e-11 with 500 steps)
    assert errors[0] / errors[1] == pytest.approx(16.0, rel=0.1, abs=0)  # fourth order; measured ratio 15.56


def _oscillation_frequency(times, signal):
    """Angular frequency from the upward zero crossings of ``signal`` (linear interpolation between samples)."""
    s = signal - 0.5 * (signal.max() + signal.min())
    k = np.where((s[:-1] < 0.0) & (s[1:] >= 0.0))[0]
    crossings = times[k] - s[k] * (times[k + 1] - times[k]) / (s[k + 1] - s[k])
    assert len(crossings) >= 3
    return 2.0 * np.pi / np.mean(np.diff(crossings))


@pytest.mark.parametrize(("radius", "spin"), [(8.0, 0.7), (10.0, -0.7), (9.0, 0.0)])
def test_kerr_epicyclic_frequencies_equal_the_oscillations_of_a_perturbed_geodesic(radius, spin):
    """A circular orbit pushed slightly in r and in theta oscillates at the epicyclic frequencies of the Kerr module.

    The module takes the frequencies from printed formulas; here they are measured on an integrated geodesic.
    Frequencies are per unit coordinate time, as seen from infinity.
    """
    a = abs(spin)
    direction = 1.0 if spin >= 0 else -1.0
    mass_kg = 1.0e31
    time_unit_s = GRAVITATIONAL_CONSTANT.value * mass_kg / SPEED_OF_LIGHT.value**3
    omega = direction / (radius**1.5 + spin)

    def metric(x):
        return metrics.kerr_metric(x, MASS, a)

    x0 = np.array([0.0, radius, np.pi / 2, 0.0])
    g = metric(x0)
    u_t = 1.0 / np.sqrt(-(g[0, 0] + 2.0 * g[0, 3] * omega + g[3, 3] * omega**2))
    kick = 1e-5
    u0 = geodesics.normalize_tangent(metric, x0, np.array([kick, kick / radius, u_t * omega]))
    x, _ = geodesics.integrate_geodesic(metric, x0, u0, np.linspace(0.0, 900.0, 3601))
    two_pi = 2.0 * np.pi
    radial = float(kerr.kerr_radial_epicyclic_frequency_hz(mass_kg, radius, spin)) * two_pi * time_unit_s
    vertical = float(kerr.kerr_vertical_epicyclic_frequency_hz(mass_kg, radius, spin)) * two_pi * time_unit_s
    # measured relative differences: at most 4e-9 (radial) and 7e-6 (vertical, set by the sampling of the crossings)
    assert _oscillation_frequency(x[:, 0], x[:, 1]) == pytest.approx(radial, rel=3e-5, abs=0)
    assert _oscillation_frequency(x[:, 0], x[:, 2]) == pytest.approx(vertical, rel=3e-5, abs=0)
