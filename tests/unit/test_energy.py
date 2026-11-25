"""Tests of kozmos.mechanics.energy.

Reference numbers come from closed forms of the two-body problem (written out in each test), conservation laws,
an independent numerical integral (binding energy of a uniform sphere) and the existing, separately tested
functions of kozmos.ephemeris.kepler. Tolerances were measured on 26 November 2026 and are stated in each test.

Every test that draws random numbers owns its generator (``np.random.default_rng(<own seed>)``). Its tolerance is a
round number at least 3 times the maximum of the error over seeds 0..299 (the maximum is stated in the test).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from kozmos.core.constants import GRAVITATIONAL_CONSTANT
from kozmos.ephemeris.kepler import orbital_period_s, orbital_speed_m_s
from kozmos.mechanics.energy import (
    accelerations_m_s2,
    barycenter_m,
    barycenter_velocity_m_s,
    eccentricity_from_vis_viva_constant,
    escape_speed_m_s,
    kinetic_energy_j,
    potential_energy_j,
    semi_major_axis_from_vis_viva_constant_m,
    specific_angular_momentum_m2_s,
    total_angular_momentum,
    total_energy_j,
    total_momentum,
    uniform_sphere_binding_energy_j,
    virial_ratio,
    vis_viva_constant_m2_s2,
)
from kozmos.mechanics.integrators import integrate, relative_drift

G = GRAVITATIONAL_CONSTANT.value
M1, M2 = 2.0e30, 6.0e24  # kg, a star and a planet
AU = 1.496e11  # m, only a convenient length scale here


def two_body(m1, m2, r, speed_rel, y_offset_zero=True):
    """Barycentric positions and velocities of two bodies at separation r (along x), relative speed along y."""
    m = m1 + m2
    x = np.array([[-m2 / m * r, 0.0, 0.0], [m1 / m * r, 0.0, 0.0]])
    v = np.array([[0.0, -m2 / m * speed_rel, 0.0], [0.0, m1 / m * speed_rel, 0.0]])
    return x, v


def circular():
    mu = G * (M1 + M2)
    return two_body(M1, M2, AU, np.sqrt(mu / AU)), mu


# --------------------------------------------------------------------------------------
# closed forms for a circular two-body orbit
# --------------------------------------------------------------------------------------


def test_circular_two_body_energies_and_virial():
    """T = (1/2) mu_red v^2, W = -G m1 m2 / r, E = W / 2, 2T/|W| = 1. Measured relative differences: at most 2.2e-16."""
    (x, v), mu = circular()
    m = np.array([M1, M2])
    mu_red = M1 * M2 / (M1 + M2)
    speed2 = mu / AU
    assert kinetic_energy_j(m, v) == pytest.approx(0.5 * mu_red * speed2, rel=1e-14)
    assert potential_energy_j(m, x) == pytest.approx(-G * M1 * M2 / AU, rel=1e-14)
    assert total_energy_j(m, x, v) == pytest.approx(-0.5 * G * M1 * M2 / AU, rel=1e-14)
    assert virial_ratio(m, x, v) == pytest.approx(1.0, rel=1e-14)


def test_circular_two_body_barycentre_momentum_and_angular_momentum():
    (x, v), mu = circular()
    m = np.array([M1, M2])
    assert np.allclose(barycenter_m(m, x), 0.0, atol=1e-3)  # m ~ 1e11 m, so 1e-14 of the scale
    assert np.allclose(barycenter_velocity_m_s(m, v), 0.0, atol=1e-9)  # speed scale 3e4 m/s
    scale = M2 * 3.0e4
    assert np.allclose(total_momentum(m, v), 0.0, atol=1e-12 * scale)
    mu_red = M1 * M2 / (M1 + M2)
    expected = mu_red * np.sqrt(mu * AU)  # |L| = mu_red * sqrt(G M r) on a circular orbit
    assert np.allclose(total_angular_momentum(m, x, v), [0.0, 0.0, expected], rtol=1e-14, atol=0.0)


def test_circular_accelerations_closed_form():
    (x, _), _ = circular()
    a = accelerations_m_s2([M1, M2], x)
    assert a[0, 0] == pytest.approx(G * M2 / AU**2, rel=1e-14)  # star pulled toward +x
    assert a[1, 0] == pytest.approx(-G * M1 / AU**2, rel=1e-14)
    assert np.allclose(a[:, 1:], 0.0)


def test_consistency_with_the_kepler_module():
    """Circular speed and period of kozmos.ephemeris.kepler against the state built here, and one revolution
    of the leapfrog returns to the start.

    Measured: relative position error after one period 3.3e-4 (500 steps), 8.3e-5 (1000), 2.07e-5 (2000): a 1/N^2
    law, as for a second-order scheme. Tolerance 1e-4 at N = 1000 (measured 8.3e-5 in position and velocity).
    """
    (x, v), mu = circular()
    assert np.linalg.norm(v[1] - v[0]) == pytest.approx(orbital_speed_m_s(AU, AU, mu), rel=1e-14)
    period = orbital_period_s(AU, mu)
    n = 1000
    _, q, vv = integrate(lambda q, t: accelerations_m_s2([M1, M2], q), x, v, period / n, n)
    speed = np.linalg.norm(v[1] - v[0])
    assert np.max(np.abs(q[-1] - x)) / AU < 1e-4
    assert np.max(np.abs(vv[-1] - v)) / speed < 1e-4


# --------------------------------------------------------------------------------------
# softening
# --------------------------------------------------------------------------------------


def test_softened_two_body_potential_and_force_closed_form():
    eps = 1.0e10
    x = np.array([[0.0, 0.0, 0.0], [3.0e10, 4.0e10, 0.0]])
    m = [M1, M2]
    r2 = 5.0e10**2
    assert potential_energy_j(m, x, eps) == pytest.approx(-G * M1 * M2 / np.sqrt(r2 + eps**2), rel=1e-14)
    a = accelerations_m_s2(m, x, eps)
    # a_0 = G m2 (x1 - x0) / (r^2 + eps^2)^(3/2)
    assert np.allclose(a[0], G * M2 * x[1] / (r2 + eps**2) ** 1.5, rtol=1e-14)
    assert np.allclose(M1 * a[0] + M2 * a[1], 0.0, atol=1e-14 * np.abs(M1 * a[0]).max())


def _gradient_defect(rng, softening):
    """Worst |-grad W - m a| / max |m a| over particles 0 and 5 of 40 random particles (step 1e6 m)."""
    m = rng.uniform(1e28, 1e30, 40)
    x = rng.normal(size=(40, 3)) * 1e11
    a = accelerations_m_s2(m, x, softening)
    h = 1e6
    worst = 0.0
    for i in (0, 5):
        grad = np.empty(3)
        for k in range(3):
            xp, xm = x.copy(), x.copy()
            xp[i, k] += h
            xm[i, k] -= h
            grad[k] = (potential_energy_j(m, xp, softening) - potential_energy_j(m, xm, softening)) / (2 * h)
        worst = max(worst, float(np.abs(-grad - m[i] * a[i]).max() / np.abs(m[i] * a[i]).max()))
    return worst


@pytest.mark.parametrize("softening", [0.0, 3.0e10])
def test_acceleration_is_minus_the_gradient_of_the_potential_energy(softening):
    """40 random particles; central differences of potential_energy_j, step 1e6 m (1e-5 of the spread).

    Maximum over seeds 0..299 of the worst relative difference of m a from -grad W: 3.2e-8 (eps = 0) and 3.8e-8
    (eps = 3e10) (median 1.1e-9 and 9.7e-10; the rounding of the difference quotient dominates at 1e6 m: it is
    1.2e-6 at 1e3 m and grows as h^2 for h >= 1e7 m). Tolerance 2e-7 (5 times the larger maximum). The earlier
    tolerance 1e-8 was exceeded by a large fraction of seeds.
    """
    assert _gradient_defect(np.random.default_rng(201), softening) <= 2e-7


@pytest.mark.parametrize("softening", [0.0, 3.0e10])
def test_direct_sum_conserves_momentum_sum_m_a_is_zero(softening):
    """Maximum over seeds 0..299 of |sum m a| / max |m a| for 40 random particles: 1.5e-15 (eps = 0) and 1.8e-15
    (eps = 3e10); tolerance 1e-14 (5.6 times the larger)."""
    rng = np.random.default_rng(202)
    m = rng.uniform(1e28, 1e30, 40)
    x = rng.normal(size=(40, 3)) * 1e11
    force = m[:, None] * accelerations_m_s2(m, x, softening)
    assert np.abs(force.sum(axis=0)).max() / np.abs(force).max() < 1e-14


def test_virial_ratio_needs_a_potential_energy():
    with pytest.raises(ValueError, match="at least two particles"):
        virial_ratio([1.0e30], [[0.0, 0.0, 0.0]], [[1.0e4, 0.0, 0.0]])
    # two particles at one point with a softening: W = 0, the ratio is undefined
    with pytest.raises(ValueError, match="W = 0"):
        virial_ratio([1.0, 1.0], np.zeros((2, 3)), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 1.0)


def test_kinetic_energy_enforces_the_documented_shape():
    assert kinetic_energy_j([2.0], [[3.0, 0.0, 4.0]]) == 25.0
    for bad in ([[1.0, 2.0], [3.0, 4.0]], np.zeros((2, 4)), np.zeros(3)):
        with pytest.raises(ValueError, match="shape"):
            kinetic_energy_j([1.0, 1.0], bad)
    with pytest.raises(ValueError, match="shape"):
        kinetic_energy_j([[1.0, 1.0]], np.zeros((2, 3)))


def test_peak_memory_coefficients_stated_in_the_docstrings():
    """tracemalloc peak of potential_energy_j and accelerations_m_s2 (eps > 0), N = 500: 56.0 N^2 and 48.3 N^2 bytes
    (N = 1000: 56.0005 and 48.07; N = 2000: 56.0001 and 48.02). The test allows 5 % above the stated coefficients."""
    import tracemalloc

    n = 500
    rng = np.random.default_rng(205)
    m, x = rng.uniform(1.0, 2.0, n), rng.normal(size=(n, 3))
    for function, coefficient in ((potential_energy_j, 56.0), (accelerations_m_s2, 48.0)):
        function(m, x, 0.1)  # warm up (imports, caches)
        tracemalloc.start()
        try:
            base = tracemalloc.get_traced_memory()[0]
            function(m, x, 0.1)
            peak = tracemalloc.get_traced_memory()[1] - base
        finally:
            tracemalloc.stop()
        assert peak <= 1.05 * coefficient * n * n, f"{function.__name__}: {peak / n**2:.2f} N^2 bytes"
        assert peak >= 0.9 * coefficient * n * n


def test_coincident_particles_without_softening_raise():
    x = np.zeros((2, 3))
    with pytest.raises(ValueError):
        accelerations_m_s2([1.0, 1.0], x)
    with pytest.raises(ValueError):
        potential_energy_j([1.0, 1.0], x)
    assert np.all(np.isfinite(accelerations_m_s2([1.0, 1.0], x, 1.0)))


def test_input_validation():
    with pytest.raises(ValueError):
        accelerations_m_s2([1.0, -1.0], np.eye(3)[:2])
    with pytest.raises(ValueError):
        accelerations_m_s2([1.0, 1.0], np.zeros((3, 3)))
    with pytest.raises(ValueError):
        accelerations_m_s2([1.0, 1.0], np.eye(3)[:2], -1.0)
    with pytest.raises(ValueError):
        escape_speed_m_s(1.0, -1.0)
    with pytest.raises(ValueError):
        uniform_sphere_binding_energy_j(1.0, 0.0)


# --------------------------------------------------------------------------------------
# three-body run with the leapfrog: energy, momentum, angular momentum
# --------------------------------------------------------------------------------------


def _hierarchical_triple():
    """A 2e30 kg star with a 1e29 kg planet at 1.5e11 m and a 1e27 kg outer body at 6e11 m, inclined 0.1 rad-ish."""
    masses = np.array([2.0e30, 1.0e29, 1.0e27])
    a_in, a_out = 1.5e11, 6.0e11
    mu_in = G * (masses[0] + masses[1])
    mu_out = G * masses.sum()
    v_in, v_out = np.sqrt(mu_in / a_in), np.sqrt(mu_out / a_out)
    pos = np.array([[0.0, 0.0, 0.0], [a_in, 0.0, 0.0], [0.0, a_out, 0.1 * a_out]])
    vel = np.array([[0.0, 0.0, 0.0], [0.0, v_in, 0.0], [-0.9 * v_out, 0.0, 0.1 * v_out]])
    vel = vel - total_momentum(masses, vel) / masses.sum()
    pos = pos - barycenter_m(masses, pos)
    period_in = 2.0 * np.pi * np.sqrt(a_in**3 / mu_in)
    period_out = 2.0 * np.pi * np.sqrt(a_out**3 / mu_out)
    return masses, pos, vel, period_in, period_out


def test_three_body_leapfrog_run_tracks_energy_momentum_and_angular_momentum():
    """KDK, dt = P_inner / 200, 10 outer periods (15996 steps), state recorded 500 times.

    Measured: max |dE/E| = 2.7e-7; max |dP| / sum(m |v|) = 1.1e-14; max |dL| / |L_0| = 1.0e-14; the barycentre
    speed at the end 2.4e-11 m/s (against speeds of 3e4 m/s). Tolerances: 1e-6, 1e-12, 1e-12, 1e-9 m/s.
    """
    masses, pos, vel, period_in, period_out = _hierarchical_triple()
    dt = period_in / 200.0
    n = int(round(10.0 * period_out / dt))
    _, q, v = integrate(lambda x, t: accelerations_m_s2(masses, x), pos, vel, dt, n, record_every=max(n // 500, 1))
    energy = np.array([total_energy_j(masses, q[i], v[i]) for i in range(len(q))])
    momentum = np.array([total_momentum(masses, v[i]) for i in range(len(q))])
    angular = np.array([total_angular_momentum(masses, q[i], v[i]) for i in range(len(q))])
    assert np.abs(relative_drift(energy)).max() < 1e-6
    assert np.abs(momentum - momentum[0]).max() / np.sum(masses * np.linalg.norm(v[0], axis=1)) < 1e-12
    assert np.abs(angular - angular[0]).max() / np.linalg.norm(angular[0]) < 1e-12
    assert np.abs(barycenter_velocity_m_s(masses, v[-1])).max() < 1e-9


# --------------------------------------------------------------------------------------
# virial ratio on an eccentric orbit
# --------------------------------------------------------------------------------------


def test_virial_ratio_at_periapsis_and_apoapsis_and_its_time_average():
    """For a Kepler orbit 2T/|W| = v^2 r / mu: 1 + e at periapsis, 1 - e at apoapsis (closed form).

    The virial theorem (time averages, 2<T> = <|W|>) holds over a full period: measured 2<T>/<|W|> - 1 = -1.1e-5
    for e = 0.5 with 4000 leapfrog steps per period (an O(dt^2) error; tolerance 5e-5); the ratio at apoapsis
    differs from 1 - e by -9.9e-6 (tolerance 1e-4) and at periapsis by 0. The mean of the instantaneous ratio is NOT 1
    (measured 0.875): only the ratio of the means is.
    """
    e = 0.5
    mu = G * (M1 + M2)
    rp = AU * (1.0 - e)
    vp = orbital_speed_m_s(rp, AU, mu)  # from the vis-viva module, independent of energy.py
    x, v = two_body(M1, M2, rp, vp)
    m = np.array([M1, M2])
    assert virial_ratio(m, x, v) == pytest.approx(1.0 + e, rel=1e-13)
    n = 4000
    period = orbital_period_s(AU, mu)
    _, q, vel = integrate(lambda y, t: accelerations_m_s2(m, y), x, v, period / n, n)
    ratio = np.array([virial_ratio(m, q[i], vel[i]) for i in range(n)])
    kinetic = np.array([kinetic_energy_j(m, vel[i]) for i in range(n)])
    potential = np.array([potential_energy_j(m, q[i]) for i in range(n)])
    # apoapsis is reached at half the period
    assert ratio[n // 2] == pytest.approx(1.0 - e, rel=1e-4)
    assert 2.0 * kinetic.mean() / np.abs(potential).mean() == pytest.approx(1.0, abs=5e-5)
    assert ratio.mean() != pytest.approx(1.0, abs=0.05)


def test_virial_ratio_equals_2T_over_potential_energy_without_softening():
    """sum m x.a = W for pure 1/r^2 gravity (the identity behind the use of W as the potential energy).

    Maximum over seeds 0..299 of the relative difference: 2.4e-15; tolerance 1e-14 (4 times)."""
    rng = np.random.default_rng(203)
    m = rng.uniform(1e28, 1e30, 25)
    x = rng.normal(size=(25, 3)) * 1e11
    v = rng.normal(size=(25, 3)) * 1e4
    expected = 2.0 * kinetic_energy_j(m, v) / abs(potential_energy_j(m, x))
    assert virial_ratio(m, x, v) == pytest.approx(expected, rel=1e-14)


# --------------------------------------------------------------------------------------
# two-body orbit relations
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("e", [0.1, 0.5, 0.9])
def test_orbit_elements_from_a_periapsis_state(e):
    """Periapsis speed from kepler.orbital_speed_m_s (vis viva); then C = -mu/(2a), h = sqrt(mu a (1 - e^2)) (eq22
    of the source), a and e recovered. Measured worst errors over e = 0.1, 0.5, 0.9: C and a 2.0e-15 (e = 0.9),
    e 4.4e-16, h 0; tolerance 1e-12 for C, h, a and 1e-13 for e. The circular case e = 0 is not tested here: the
    function cannot resolve e below about 1e-8 (see ``test_eccentricity_accuracy_limit_on_random_orbits``)."""
    mu = G * (M1 + M2)
    rp = AU * (1.0 - e)
    vp = orbital_speed_m_s(rp, AU, mu)
    rel_x = np.array([rp, 0.0, 0.0])
    rel_v = np.array([0.0, vp, 0.0])
    c = vis_viva_constant_m2_s2(rel_x, rel_v, mu)
    h = specific_angular_momentum_m2_s(rel_x, rel_v)
    assert c == pytest.approx(-mu / (2.0 * AU), rel=1e-12)
    assert np.linalg.norm(h) == pytest.approx(np.sqrt(mu * AU * (1.0 - e**2)), rel=1e-12)
    assert semi_major_axis_from_vis_viva_constant_m(c, mu) == pytest.approx(AU, rel=1e-12)
    assert eccentricity_from_vis_viva_constant(c, np.linalg.norm(h), mu) == pytest.approx(e, abs=1e-13)


def test_vis_viva_constant_is_the_total_energy_per_reduced_mass():
    mu = G * (M1 + M2)
    x, v = two_body(M1, M2, 0.7 * AU, 0.8 * np.sqrt(mu / (0.7 * AU)))
    c = vis_viva_constant_m2_s2(x[1] - x[0], v[1] - v[0], mu)
    assert total_energy_j([M1, M2], x, v) / (M1 * M2 / (M1 + M2)) == pytest.approx(c, rel=1e-12)


def test_vis_viva_constant_vectorised_and_hyperbolic_axis_is_negative():
    mu = 1.0
    c = vis_viva_constant_m2_s2([[1.0, 0, 0], [1.0, 0, 0]], [[0, 1.0, 0], [0, 2.0, 0]], mu)
    assert np.allclose(c, [-0.5, 1.0])  # 1/2 - 1 and 2 - 1
    a = semi_major_axis_from_vis_viva_constant_m(c, mu)
    assert np.allclose(a, [1.0, -0.5])
    # e^2 = 1 + 2 C h^2 / mu^2 = 0 exactly in floating point here; in general e is only resolved to sqrt(eps)
    assert eccentricity_from_vis_viva_constant(-0.5, 1.0, 1.0) == pytest.approx(0.0, abs=1e-7)
    with pytest.raises(ValueError):
        eccentricity_from_vis_viva_constant(-1.0, 1.0, 1.0)  # e^2 = 1 - 2 < 0


def _random_orbit_eccentricities(rng, n, circular):
    """(true e, computed e) for n random orbits built from the periapsis state (closed form, e known exactly)."""
    pairs = []
    for _ in range(n):
        mu = 10 ** rng.uniform(8, 21)
        a = 10 ** rng.uniform(6, 12)
        e = 0.0 if circular else rng.uniform(0.0, 0.95)
        rp = a * (1.0 - e)
        vp = np.sqrt(mu * (1.0 + e) / rp)
        frame = np.linalg.qr(rng.normal(size=(3, 3)))[0]  # random orientation of the orbit
        pos, vel = frame @ np.array([rp, 0.0, 0.0]), frame @ np.array([0.0, vp, 0.0])
        c = vis_viva_constant_m2_s2(pos, vel, mu)
        h = np.linalg.norm(specific_angular_momentum_m2_s(pos, vel))
        pairs.append((e, float(eccentricity_from_vis_viva_constant(c, h, mu))))
    return np.array(pairs)


def test_eccentricity_accuracy_limit_on_random_orbits():
    """e^2 = 1 + 2 C h^2 / mu^2 cancels for e -> 0, so e is resolved only to about sqrt(eps) = 1.5e-8.

    Maximum over seeds 0..299 (200 orbits each, mu 1e8..1e21, a 1e6..1e12 m, random orientation):
    circular orbits: computed e up to 3.2e-8 (that is e^2 = 1.0e-15, 4.5 eps) instead of 0 -> tolerance 1e-7;
    all orbits: |e_c^2 - e^2| up to 6.0e-15 -> tolerance 2e-14; orbits with e >= 0.1: |e_c - e| up to 3.9e-15 ->
    tolerance 2e-14. The earlier tests asserted e = 0 within 1e-12 only because the rounding happened to vanish."""
    rng = np.random.default_rng(204)
    circular = _random_orbit_eccentricities(rng, 200, circular=True)
    assert circular[:, 1].max() <= 1e-7
    general = _random_orbit_eccentricities(rng, 200, circular=False)
    assert np.abs(general[:, 1] ** 2 - general[:, 0] ** 2).max() <= 2e-14
    eccentric = general[general[:, 0] >= 0.1]
    assert len(eccentric) > 100
    assert np.abs(eccentric[:, 1] - eccentric[:, 0]).max() <= 2e-14


def test_parabolic_vis_viva_constant_gives_plus_infinity_for_both_zeros():
    for zero in (0.0, -0.0):
        assert semi_major_axis_from_vis_viva_constant_m(zero, 1.0) == np.inf
        assert semi_major_axis_from_vis_viva_constant_m(zero, 3.5e14) == np.inf
    a = semi_major_axis_from_vis_viva_constant_m([-0.5, 0.0, -0.0, 0.25], 1.0)
    assert a[0] == 1.0 and a[1] == np.inf and a[2] == np.inf and a[3] == -2.0
    # the limits on either side of the parabola keep their physical sign
    assert semi_major_axis_from_vis_viva_constant_m(-1e-300, 1.0) > 1e299
    assert semi_major_axis_from_vis_viva_constant_m(1e-300, 1.0) < -1e299


def test_escape_speed_against_the_kepler_module():
    """v_esc = sqrt(2) v_circ (kepler.orbital_speed_m_s with a = r) and equals the vis-viva speed of an orbit with
    1/a = 0 (parabola). Measured relative differences 0 and 0; C / (mu / r) = 0 at v_esc."""
    mu = G * M1
    v_esc = escape_speed_m_s(mu, AU)
    assert v_esc == pytest.approx(np.sqrt(2.0) * orbital_speed_m_s(AU, AU, mu), rel=1e-14)
    assert v_esc == pytest.approx(orbital_speed_m_s(AU, np.inf, mu), rel=1e-14)
    assert vis_viva_constant_m2_s2([AU, 0, 0], [0, v_esc, 0], mu) == pytest.approx(0.0, abs=1e-6 * mu / AU)
    assert np.allclose(escape_speed_m_s(mu, [AU, 4 * AU]), [v_esc, v_esc / 2.0], rtol=1e-14)


def test_escape_speed_rejects_a_negative_gm_and_accepts_zero():
    with pytest.raises(ValueError, match="gm_m3_s2"):
        escape_speed_m_s(-1.0, 1.0)
    with pytest.raises(ValueError, match="gm_m3_s2"):
        escape_speed_m_s([1.0, -1.0], 1.0)
    assert escape_speed_m_s(0.0, 1.0) == 0.0


# --------------------------------------------------------------------------------------
# binding energy of a uniform sphere
# --------------------------------------------------------------------------------------


def test_uniform_sphere_binding_energy_against_a_shell_integral():
    """Independent oracle: E_b = integral_0^R G M(r) dm / r with M(r) = M (r/R)^3 and dm = 3 M r^2 / R^3 dr, by
    scipy.integrate.quad. Measured relative difference 4e-16 (quad error estimate 2.6e27 J on E_b = 2.3e41 J)."""
    mass, radius = 2.0e30, 6.957e8
    value, _ = quad(lambda r: G * mass * (r / radius) ** 3 / r * 3.0 * mass * r**2 / radius**3, 0.0, radius)
    assert uniform_sphere_binding_energy_j(mass, radius) == pytest.approx(value, rel=1e-12)


def test_uniform_sphere_binding_energy_scaling():
    base = uniform_sphere_binding_energy_j(1.0e30, 1.0e9)
    assert uniform_sphere_binding_energy_j(2.0e30, 1.0e9) == pytest.approx(4.0 * base, rel=1e-14)
    assert uniform_sphere_binding_energy_j(1.0e30, 2.0e9) == pytest.approx(0.5 * base, rel=1e-14)
    assert base > 0.0
