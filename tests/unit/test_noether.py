"""Noether charges, symmetry defects and ready-made generators against closed forms.

Convention of the source (Banados and Reyes, arXiv:1601.03616): delta_s q = q'(t) - q(t), symmetry delta_s L = dK/dt,
charge Q = K - p . delta_s q. Consequences checked here: rotation Q = -alpha . L, time translation Q = +E,
Weyl scaling Q = -(printed Eq. QWeyl), and on a solution dQ/dt = -(delta_s L - dK/dt).

Oracles (none calls the code under test): angular momentum from np.cross, linear momentum and centre of mass from
sums, analytic motions (uniform force, cyclotron orbit, free particle), the closed-form defects written out below,
and the source's own printed charges.

Every tolerance is a round number (1, 2 or 5 times a power of ten) at least 3 times the maximum of the error over
the seeds 0..299 of the random draws (measured on Python 3.12, numpy 2.2, scipy 1.14 by running each test with
``np.random.default_rng`` replaced by a generator of the given seed; the largest ratio of maximum to tolerance is
0.27, 2.7e-15 against 1e-14); where the measured error is exactly zero the tolerance is
a floor (1e-14, or 1e-13 for symmetry defects that are built from finite-difference derivatives). Every test that
draws random numbers owns its generator with its own seed, so the order of the tests does not matter.
``pytest -s`` prints the measured value next to the tolerance (lines starting with MEASURED).
"""

from __future__ import annotations

import numpy as np
import pytest

from kozmos.mechanics import hamiltonian as hm
from kozmos.mechanics import lagrangian as lg
from kozmos.mechanics import noether as nt


def _check(name: str, error: float, tolerance: float) -> None:
    print(f"MEASURED {name}: {error:.3e} (tolerance {tolerance:.1e})")
    assert error <= tolerance, f"{name}: {error:.3e} > {tolerance:.1e}"


def _random_state(rng, n):
    return rng.normal(size=n), rng.normal(size=n), rng.normal(size=n), float(rng.uniform(-1.5, 1.5))


# --------------------------------------------------------------------------------------
# Lagrangians
# --------------------------------------------------------------------------------------

M1, M2, GRAV = 1.2, 0.8, 1.3


def central_3d(q, v, t):
    return 0.5 * M1 * np.sum(v * v) + 1.7 / np.sqrt(np.sum(q * q))


def two_body_2d(q, v, t):
    d = q[:2] - q[2:]
    return 0.5 * M1 * np.sum(v[:2] * v[:2]) + 0.5 * M2 * np.sum(v[2:] * v[2:]) + GRAV * M1 * M2 / np.sqrt(np.sum(d * d))


def two_body_3d(q, v, t):
    d = q[:3] - q[3:]
    return 0.5 * M1 * np.sum(v[:3] * v[:3]) + 0.5 * M2 * np.sum(v[3:] * v[3:]) + GRAV * M1 * M2 / np.sqrt(np.sum(d * d))


# --------------------------------------------------------------------------------------
# time translation = energy function
# --------------------------------------------------------------------------------------

FORCE_AMPLITUDE, DRIVE = 0.9, 1.7


def driven_oscillator(q, v, t):
    return 0.5 * v[0] ** 2 - 1.5 * q[0] ** 2 - FORCE_AMPLITUDE * np.cos(DRIVE * t) * q[0]


def test_time_translation_charge_is_the_energy_function():
    rng = np.random.default_rng(301)
    shift = nt.t_translation_generator
    for lagrangian, n in ((central_3d, 3), (two_body_2d, 4), (driven_oscillator, 1)):
        k = nt.t_translation_boundary_term(lagrangian)
        q, v, _, t = _random_state(rng, n)
        charge = nt.noether_charge(lagrangian, q, v, t, shift, k)
        _check(f"Q = h for {lagrangian.__name__}", abs(charge - lg.jacobi_function(lagrangian, q, v, t)), 1e-14)
    # source, line 251: Q_eps = eps E with E = (m/2) v^2 + V
    q, v, _, t = _random_state(rng, 3)
    energy = 0.5 * M1 * v @ v - 1.7 / np.sqrt(q @ q)
    charge = nt.noether_charge(central_3d, q, v, t, shift, nt.t_translation_boundary_term(central_3d))
    _check("Q = (m/2) v^2 + V (source line 251)", abs(charge - energy), 2e-14)  # max over 300 seeds: 4.4e-15


def test_time_translation_defect_is_the_explicit_time_derivative_of_L():
    rng = np.random.default_rng(302)
    shift = nt.t_translation_generator
    # not explicitly time dependent: a symmetry at EVERY (q, v, a, t), including off-shell accelerations
    worst = 0.0
    for lagrangian, n in ((central_3d, 3), (two_body_2d, 4)):
        k = nt.t_translation_boundary_term(lagrangian)
        for _ in range(5):
            q, v, a, t = _random_state(rng, n)
            worst = max(worst, abs(nt.symmetry_defect(lagrangian, q, v, t, shift, k, a=a)))
    _check("time-translation defect, t-independent L", worst, 1e-13)
    # explicitly time dependent: defect = dL/dt (partial) = F Omega sin(Omega t) q
    k = nt.t_translation_boundary_term(driven_oscillator)
    worst = 0.0
    for _ in range(5):
        q, v, a, t = _random_state(rng, 1)
        expected = FORCE_AMPLITUDE * DRIVE * np.sin(DRIVE * t) * q[0]
        worst = max(worst, abs(nt.symmetry_defect(driven_oscillator, q, v, t, shift, k, a=a) - expected))
    _check("time-translation defect = dL/dt, driven oscillator", worst, 1e-13)


def test_velocity_dependent_generator_needs_the_acceleration():
    k = nt.t_translation_boundary_term(two_body_2d)
    with pytest.raises(ValueError, match="acceleration"):
        nt.symmetry_defect(two_body_2d, np.array([0.5, 0.1, -0.4, 0.3]), np.ones(4), 0.0, nt.t_translation_generator, k)
    # the boundary term alone may depend on the velocity as well
    with pytest.raises(ValueError, match="acceleration"):
        nt.symmetry_defect(
            two_body_2d, np.array([0.5, 0.1, -0.4, 0.3]), np.ones(4), 0.0, nt.translation_generator([1.0, 0.0]), k
        )
    # a velocity-independent generator with K = None works without a; the pair potential is translation invariant
    # (measured defect exactly 0.0 at six (q, direction) combinations, so the floor 1e-13 of the sibling tests)
    defect = nt.symmetry_defect(
        two_body_2d, np.array([0.5, 0.1, -0.4, 0.3]), np.ones(4), 0.0, nt.translation_generator([1.0, 0.0])
    )
    _check("translation defect without a (pair potential)", abs(defect), 1e-13)


# --------------------------------------------------------------------------------------
# rotation and translation of N particles
# --------------------------------------------------------------------------------------


def test_rotation_charge_is_minus_alpha_dot_angular_momentum():
    rng = np.random.default_rng(303)
    axis = np.array([0.3, -0.5, 0.8])
    gen = nt.rotation_generator(axis)
    worst = 0.0
    for _ in range(5):
        q, v, a, t = _random_state(rng, 3)
        angular = M1 * np.cross(q, v)
        worst = max(worst, abs(nt.noether_charge(central_3d, q, v, t, gen) + axis @ angular))
        worst = max(worst, abs(nt.symmetry_defect(central_3d, q, v, t, gen, a=a)))
    _check("rotation: Q = -alpha.L (source Eq. angular), defect = 0, central force", worst, 1e-13)
    # two particles, simultaneous rotation: total angular momentum
    gen6 = nt.rotation_generator(axis)
    q, v, a, t = _random_state(rng, 6)
    total = M1 * np.cross(q[:3], v[:3]) + M2 * np.cross(q[3:], v[3:])
    _check(
        "two-body rotation charge = -alpha.L_total",
        abs(nt.noether_charge(two_body_3d, q, v, t, gen6) + axis @ total),
        1e-14,
    )
    _check("two-body rotation defect", abs(nt.symmetry_defect(two_body_3d, q, v, t, gen6, a=a)), 1e-13)


def test_planar_rotation_generator():
    rng = np.random.default_rng(304)
    gen = nt.planar_rotation_generator()
    q, v, a, t = _random_state(rng, 4)
    lz_total = M1 * (q[0] * v[1] - q[1] * v[0]) + M2 * (q[2] * v[3] - q[3] * v[2])
    _check("planar rotation charge = -Lz", abs(nt.noether_charge(two_body_2d, q, v, t, gen) + lz_total), 1e-14)
    _check("planar rotation defect", abs(nt.symmetry_defect(two_body_2d, q, v, t, gen, a=a)), 1e-13)
    # the same generator as the 3-D rotation about z restricted to the plane
    q3 = np.array([q[0], q[1], 0.0])
    z_rot = nt.rotation_generator([0.0, 0.0, 1.0])(q3, np.zeros(3), 0.0)
    assert np.allclose(z_rot[:2], gen(q[:2], np.zeros(2), 0.0)[:2], atol=0, rtol=0)


def test_translation_charge_is_minus_direction_dot_momentum_and_fixed_centre_breaks_it():
    rng = np.random.default_rng(305)
    direction = np.array([0.6, -0.8])
    gen = nt.translation_generator(direction)
    q, v, a, t = _random_state(rng, 4)
    momentum = M1 * v[:2] + M2 * v[2:]
    _check("translation Q = -n.P", abs(nt.noether_charge(two_body_2d, q, v, t, gen) + direction @ momentum), 1e-14)
    _check("translation defect, pair potential", abs(nt.symmetry_defect(two_body_2d, q, v, t, gen, a=a)), 1e-13)

    centre = np.array([0.4, -0.2])
    strength = 1.1

    def fixed_centre(q, v, t):
        d = q - centre
        return 0.5 * M1 * np.sum(v * v) + strength / np.sqrt(np.sum(d * d))

    r, w, _, _ = _random_state(rng, 2)
    d = r - centre
    expected = -strength * (d @ direction) / (d @ d) ** 1.5  # dL/dq . n
    measured = nt.symmetry_defect(fixed_centre, r, w, 0.0, gen)
    _check("fixed centre: defect = -grad V . n (source Eq. Ip)", abs(measured - expected), 1e-13)
    # non-triviality at a fixed point (not a random draw): the defect there is -grad V . n = 0.1 or larger
    fixed = nt.symmetry_defect(fixed_centre, np.array([1.0, 0.3]), np.array([0.2, -0.1]), 0.0, gen)
    assert abs(fixed) > 1e-2


# --------------------------------------------------------------------------------------
# Galilean boost (K != 0)
# --------------------------------------------------------------------------------------


def test_boost_needs_the_boundary_term_and_its_charge_is_the_centre_of_mass_law():
    rng = np.random.default_rng(306)
    masses = np.array([M1, M2])
    direction = np.array([0.6, 0.8])
    gen = nt.boost_generator(direction)
    k = nt.boost_boundary_term(masses, direction)
    worst_with, worst_without_error = 0.0, 0.0
    for _ in range(5):
        q, v, a, t = _random_state(rng, 4)
        worst_with = max(worst_with, abs(nt.symmetry_defect(two_body_2d, q, v, t, gen, k, a=a)))
        momentum = M1 * v[:2] + M2 * v[2:]
        without = nt.symmetry_defect(two_body_2d, q, v, t, gen, None, a=a)
        worst_without_error = max(worst_without_error, abs(without - direction @ momentum))
        charge = nt.noether_charge(two_body_2d, q, v, t, gen, k)
        centre = M1 * q[:2] + M2 * q[2:]
        _check("boost Q = n.(M R_cm - t P)", abs(charge - direction @ (centre - t * momentum)), 1e-14)
    # without K the "symmetry" fails visibly (fixed state, not a random draw): the defect is n.P = 0.6 M1 + 0.8 M2 ...
    q_fixed, v_fixed = np.array([0.5, 0.1, -0.4, 0.3]), np.array([1.0, 0.5, -0.2, 0.7])
    fixed_defect = nt.symmetry_defect(two_body_2d, q_fixed, v_fixed, 0.3, gen, None, a=np.zeros(4))
    assert abs(fixed_defect - direction @ (M1 * v_fixed[:2] + M2 * v_fixed[2:])) < 1e-13
    assert abs(fixed_defect) > 0.5
    _check("boost defect with K = sum m n.r (random q, v, a, t)", worst_with, 1e-13)
    _check("boost defect without K = n . P", worst_without_error, 1e-13)
    q, v, a, t = _random_state(rng, 4)
    _check(
        "symmetry_variation of the boost = n.P",
        abs(nt.symmetry_variation(two_body_2d, q, v, t, gen) - direction @ (M1 * v[:2] + M2 * v[2:])),
        1e-13,
    )


def test_boost_charge_is_conserved_along_an_integrated_two_body_orbit():
    masses = np.array([M1, M2])
    direction = np.array([0.6, 0.8])
    gen = nt.boost_generator(direction)
    k = nt.boost_boundary_term(masses, direction)
    q0 = np.array([1.0, 0.0, -1.2, 0.3])
    v0 = np.array([0.1, 0.7, 0.2, -0.6])
    times = np.linspace(0.0, 2.5, 11)
    q, v = lg.integrate_lagrangian(two_body_2d, q0, v0, times)
    charges = np.array([nt.noether_charge(two_body_2d, q[i], v[i], times[i], gen, k) for i in range(len(times))])
    _check("boost charge drift on the orbit", float(np.ptp(charges)), 5e-12)
    # linear momentum and angular momentum conserved as well
    translation = nt.translation_generator(direction)
    momentum = np.array([nt.noether_charge(two_body_2d, q[i], v[i], times[i], translation) for i in range(len(times))])
    _check("momentum charge drift on the orbit", float(np.ptp(momentum)), 5e-12)
    angular = np.array(
        [
            nt.noether_charge(two_body_2d, q[i], v[i], times[i], nt.planar_rotation_generator())
            for i in range(len(times))
        ]
    )
    _check("angular momentum charge drift on the orbit", float(np.ptp(angular)), 2e-9)
    # centre of mass moves uniformly: R(t) = R0 + P t / M
    total_mass = M1 + M2
    centre = (M1 * q[:, :2] + M2 * q[:, 2:]) / total_mass
    momentum_vec = M1 * v0[:2] + M2 * v0[2:]
    expected = (M1 * q0[:2] + M2 * q0[2:]) / total_mass + np.outer(times, momentum_vec / total_mass)
    _check("centre of mass R0 + P t / M", float(np.max(np.abs(centre - expected))), 2e-12)


# --------------------------------------------------------------------------------------
# conformal particle: Weyl symmetry
# --------------------------------------------------------------------------------------

MC, ALPHA = 1.3, 0.7


def conformal(q, v, t):
    return 0.5 * MC * v[0] ** 2 - ALPHA / q[0] ** 2


def test_weyl_symmetry_defect_and_charge_against_the_source():
    rng = np.random.default_rng(307)
    k = nt.weyl_boundary_term(conformal)
    worst, worst_charge, worst_hamiltonian = 0.0, 0.0, 0.0
    for _ in range(6):
        q, v, a, t = _random_state(rng, 1)
        q = np.abs(q) + 0.5
        worst = max(worst, abs(nt.symmetry_defect(conformal, q, v, t, nt.weyl_generator, k, a=a)))
        charge = nt.noether_charge(conformal, q, v, t, nt.weyl_generator, k)
        printed = 0.5 * MC * q[0] * v[0] - (0.5 * MC * t * v[0] ** 2 + ALPHA * t / q[0] ** 2)  # source Eq. (QWeyl)
        worst_charge = max(worst_charge, abs(charge + printed))
        _, hamiltonian_q, _ = hm.conformal_particle_charges(q[0], MC * v[0], t, MC, ALPHA)
        worst_hamiltonian = max(worst_hamiltonian, abs(charge + hamiltonian_q))
    _check("Weyl defect with K = -tL at random off-shell points", worst, 1e-13)
    _check("Weyl charge = -(printed Eq. QWeyl)", worst_charge, 1e-14)
    _check("Weyl charge = -Q of the Hamiltonian form (-tH + pq/2)", worst_hamiltonian, 1e-14)


def test_weyl_symmetry_requires_the_inverse_square_potential_and_the_time_dependent_boundary_term():
    def cubic(q, v, t):
        return 0.5 * MC * v[0] ** 2 - ALPHA / q[0] ** 3

    q, v, a, t = np.array([1.3]), np.array([0.4]), np.array([-0.2]), 0.8
    k_cubic = nt.weyl_boundary_term(cubic)
    assert abs(nt.symmetry_defect(cubic, q, v, t, nt.weyl_generator, k_cubic, a=a)) > 1e-2
    # without K the conformal particle's defect is -dK/dt = d(tL)/dt = L + t dL/dt: not zero
    assert abs(nt.symmetry_defect(conformal, q, v, t, nt.weyl_generator, None, a=a)) > 1e-2


def test_weyl_charge_and_energy_solve_the_conformal_particle():
    x0, v0 = 1.1, 0.35
    times = np.linspace(0.0, 3.0, 13)
    q, v = lg.integrate_lagrangian(conformal, [x0], [v0], times, rtol=1e-11, atol=1e-13)
    k = nt.weyl_boundary_term(conformal)
    charges = np.array(
        [nt.noether_charge(conformal, q[i], v[i], times[i], nt.weyl_generator, k) for i in range(len(times))]
    )
    energies = np.array([lg.jacobi_function(conformal, q[i], v[i], times[i]) for i in range(len(times))])
    _check("Weyl charge drift on the orbit", float(np.ptp(charges)), 2e-10)
    _check("energy drift on the orbit", float(np.ptp(energies)), 1e-10)
    # the source: two algebraic equations fix x(t); from m x xdot = 2 (Q_W + t E) one gets
    # x^2 = x0^2 + 4 Q_W t / m + 2 E t^2 / m with Q_W = -charge
    energy = 0.5 * MC * v0**2 + ALPHA / x0**2
    q_w = -charges[0]
    x_squared = x0**2 + 4.0 * q_w * times / MC + 2.0 * energy * times**2 / MC
    _check("x^2(t) from the two charges vs integration", float(np.max(np.abs(q[:, 0] ** 2 - x_squared))), 1e-9)


# --------------------------------------------------------------------------------------
# broken symmetry: the charge drifts, the defect is non-zero, and dQ/dt = -defect
# --------------------------------------------------------------------------------------


def test_uniform_force_breaks_rotation_dQ_dt_equals_minus_defect():
    mass, force = 1.4, np.array([0.5, -0.3, 0.8])  # the source's B vector (lines 319-364): L = m v^2/2 - B.r

    def uniform_force(q, v, t):
        return 0.5 * mass * np.sum(v * v) - force @ q

    axis = np.array([0.2, 0.9, -0.4])
    gen = nt.rotation_generator(axis)
    r0, v0 = np.array([0.3, -0.7, 1.1]), np.array([0.6, 0.2, -0.5])

    def orbit(t):  # m r'' = -B
        return r0 + v0 * t - force * t**2 / (2.0 * mass), v0 - force * t / mass

    worst_defect, worst_rate = 0.0, 0.0
    for t in (0.0, 0.4, 1.3):
        r, v = orbit(t)
        defect = nt.symmetry_defect(uniform_force, r, v, t, gen)
        _check("rotation defect = -alpha.(r x B) (source Eq. dL1)", abs(defect + axis @ np.cross(r, force)), 1e-13)
        assert abs(defect) > 1e-2
        h = 0.1  # Q(t) is a cubic polynomial on the exact orbit: the 5-point stencil is exact
        charges = [nt.noether_charge(uniform_force, *orbit(t + s * h), t + s * h, gen) for s in (-2, -1, 1, 2)]
        rate = (charges[0] - 8.0 * charges[1] + 8.0 * charges[2] - charges[3]) / (12.0 * h)
        worst_rate = max(worst_rate, abs(rate + defect))
        worst_defect = max(worst_defect, abs(defect))
    _check("dQ/dt + defect on the exact orbit", worst_rate, 1e-13)
    # the charge really drifts
    q_start = nt.noether_charge(uniform_force, *orbit(0.0), 0.0, gen)
    q_end = nt.noether_charge(uniform_force, *orbit(1.3), 1.3, gen)
    assert abs(q_end - q_start) > 0.1


def test_broken_rotation_integrated_orbit_matches_the_drift_predicted_by_the_defect():
    mass, force_x = 1.4, 0.6

    def plane_force(q, v, t):
        return 0.5 * mass * np.sum(v * v) - force_x * q[0]

    gen = nt.planar_rotation_generator()
    times = np.linspace(0.0, 2.0, 9)
    q, v = lg.integrate_lagrangian(plane_force, [0.3, -0.5], [0.4, 0.9], times)
    charges = np.array([nt.noether_charge(plane_force, q[i], v[i], times[i], gen) for i in range(len(times))])
    defects = np.array([nt.symmetry_defect(plane_force, q[i], v[i], times[i], gen) for i in range(len(times))])
    # defect = delta_s L = -F . delta_s q = force_x * y  (closed form), and Q(t) - Q(0) = -integral defect dt
    _check("defect = B y on the integrated orbit", float(np.max(np.abs(defects - force_x * q[:, 1]))), 1e-13)
    y_exact = -0.5 + 0.9 * times
    integral = force_x * (-0.5 * times + 0.9 * times**2 / 2.0)
    _check("Q(t) - Q(0) = -integral defect dt", float(np.max(np.abs((charges - charges[0]) + integral))), 1e-11)
    _check("y(t) of the integrated orbit", float(np.max(np.abs(q[:, 1] - y_exact))), 1e-11)


# --------------------------------------------------------------------------------------
# generalized coordinates, charged particle, inverse theorem
# --------------------------------------------------------------------------------------

G_ACC, LEN, BOB = 9.81, 1.3, 0.8


def spherical_pendulum(q, v, t):
    return 0.5 * BOB * LEN**2 * (v[0] ** 2 + np.sin(q[0]) ** 2 * v[1] ** 2) + BOB * G_ACC * LEN * np.cos(q[0])


def test_cyclic_angle_in_generalized_coordinates():
    rng = np.random.default_rng(308)

    def phi_shift(q, v, t):
        return np.array([0.0, 1.0])

    def theta_shift(q, v, t):
        return np.array([1.0, 0.0])

    q, v, a, t = _random_state(rng, 2)
    p_phi = BOB * LEN**2 * np.sin(q[0]) ** 2 * v[1]
    _check("Q = -p_phi", abs(nt.noether_charge(spherical_pendulum, q, v, t, phi_shift) + p_phi), 1e-14)
    _check("defect of the phi shift", abs(nt.symmetry_defect(spherical_pendulum, q, v, t, phi_shift, a=a)), 1e-13)
    expected = -BOB * G_ACC * LEN * np.sin(q[0]) + BOB * LEN**2 * np.sin(q[0]) * np.cos(q[0]) * v[1] ** 2
    _check(
        "defect of the theta shift = dL/dtheta",
        abs(nt.symmetry_defect(spherical_pendulum, q, v, t, theta_shift) - expected),
        1e-13,
    )


CHARGE, BFIELD, MQ = 0.8, 1.9, 1.3


def magnetic(q, v, t):
    return 0.5 * MQ * np.sum(v * v) + CHARGE * 0.5 * BFIELD * (-q[1] * v[0] + q[0] * v[1])


def _cyclotron(t, speed=0.9, x0=0.2, y0=-0.1):
    omega = CHARGE * BFIELD / MQ
    q = np.array([x0 + speed / omega * np.sin(omega * t), y0 + speed / omega * (np.cos(omega * t) - 1.0), 0.3 * t])
    v = np.array([speed * np.cos(omega * t), -speed * np.sin(omega * t), 0.3])
    return q, v


def test_charged_particle_rotation_and_magnetic_translation():
    rng = np.random.default_rng(309)
    rotation = nt.rotation_generator([0.0, 0.0, 1.0])
    q, v, a, t = _random_state(rng, 3)
    p = lg.generalized_momenta(magnetic, q, v)
    _check(
        "rotation about B: defect (symmetric gauge)", abs(nt.symmetry_defect(magnetic, q, v, t, rotation, a=a)), 1e-13
    )
    _check(
        "rotation about B: Q = -(x p_y - y p_x), canonical p",
        abs(nt.noether_charge(magnetic, q, v, t, rotation) + q[0] * p[1] - q[1] * p[0]),
        1e-14,
    )
    # conserved on the exact cyclotron orbit (closed form, no integrator)
    values = [nt.noether_charge(magnetic, *_cyclotron(s), s, rotation) for s in np.linspace(0.0, 6.0, 13)]
    _check("canonical angular momentum on the cyclotron orbit", float(np.ptp(values)), 1e-14)

    # translation along x: delta_s L = (e B / 2) vy = d/dt (e B y / 2), so K = e B y / 2 (magnetic translation)
    shift_x = nt.translation_generator([1.0, 0.0, 0.0])

    def k_term(q, v, t):
        return 0.5 * CHARGE * BFIELD * q[1]

    _check(
        "magnetic translation defect with K = e B y / 2",
        abs(nt.symmetry_defect(magnetic, q, v, t, shift_x, k_term, a=a)),
        1e-13,
    )
    # without K the translation is not a symmetry; evaluated at a fixed state (not a random draw): (e B / 2) vy
    q_fixed, v_fixed = np.array([0.4, -0.2, 0.1]), np.array([0.3, 0.7, 0.2])
    assert abs(nt.symmetry_defect(magnetic, q_fixed, v_fixed, 0.0, shift_x, None, a=np.zeros(3))) > 0.1
    values = [nt.noether_charge(magnetic, *_cyclotron(s), s, shift_x, k_term) for s in np.linspace(0.0, 6.0, 13)]
    _check("magnetic translation charge on the cyclotron orbit", float(np.ptp(values)), 1e-14)
    # Q = K - p_x = e B y - m vx
    qs, vs = _cyclotron(1.1)
    _check(
        "Q = e B y - m vx",
        abs(nt.noether_charge(magnetic, qs, vs, 1.1, shift_x, k_term) - (CHARGE * BFIELD * qs[1] - MQ * vs[0])),
        1e-14,
    )


def test_noether_inverse_theorem_sign():
    rng = np.random.default_rng(310)
    # source, Eq. (invNoe): Q conserved generates delta_s q = dQ/dp. For Q = H this is +qdot, whose boundary term is
    # K = +L; Eq. (qn) then gives Q_noether = L - p.qdot = -H (the prose of the source says +Q; see the docstring).
    q, v, a, t = _random_state(rng, 3)

    def plus_velocity(q, v, t):
        return v

    def plus_l(q, v, t):
        return central_3d(q, v, t)

    charge = nt.noether_charge(central_3d, q, v, t, plus_velocity, plus_l)
    _check("generator +qdot: Q = -h", abs(charge + lg.jacobi_function(central_3d, q, v, t)), 1e-14)
    _check(
        "generator +qdot, K = L: defect",
        abs(nt.symmetry_defect(central_3d, q, v, t, plus_velocity, plus_l, a=a)),
        1e-13,
    )
    # the generator of a rotation from Q = Lz through the Hamiltonian module is the Lagrangian rotation generator
    p = lg.generalized_momenta(central_3d, q, v)

    def lz(qq, pp, tt):
        return qq[0] * pp[1] - qq[1] * pp[0]

    dq, _ = hm.canonical_symmetry_variation(lz, q, p)
    assert np.allclose(dq, nt.rotation_generator([0.0, 0.0, 1.0])(q, v, t), atol=1e-14, rtol=0)


def test_input_validation():
    with pytest.raises(ValueError, match="three components"):
        nt.rotation_generator([1.0, 0.0])
    with pytest.raises(ValueError, match="multiple"):
        nt.translation_generator([1.0, 0.0, 0.0])(np.zeros(4), np.zeros(4), 0.0)
    with pytest.raises(ValueError, match="same length"):
        nt.symmetry_defect(central_3d, np.zeros(3) + 1, np.zeros(3), 0.0, nt.t_translation_generator, a=np.zeros(2))


# --------------------------------------------------------------------------------------
# mutation-testing survivors: the evaluation point of the Jacobian, the explicit time derivative of K, the
# velocity-dependence threshold and the size check of the rotation axis
# --------------------------------------------------------------------------------------


def _free_with_spring(q, v, t):
    return 0.5 * np.sum(v * v) - 0.5 * np.sum(q * q)


def _quadratic_generator(q, v, t):
    """delta_s q = (q0 q1, q1^2): NONLINEAR in q, so its Jacobian depends on the point where it is evaluated."""
    return np.array([q[0] * q[1], q[1] * q[1]])


@pytest.mark.parametrize("method", ["complex", "finite"])
def test_symmetry_variation_of_a_generator_nonlinear_in_q_has_a_closed_form(method):
    """At q = (2, 3), v = (5, 7) with L = v.v/2 - q.q/2: d(delta_s q)/dt = J(q) v = (v0 q1 + q0 v1, 2 q1 v1) = (29, 42),
    dL/dv . that = 5 * 29 + 7 * 42 = 439 and dL/dq . delta_s q = -(2 * 6 + 3 * 9) = -39, so delta_s L = 400 exactly.
    A Jacobian taken at the point v instead of q would give 1036 - 39 = 997."""
    q, v = np.array([2.0, 3.0]), np.array([5.0, 7.0])
    got = nt.symmetry_variation(_free_with_spring, q, v, 0.5, _quadratic_generator, method=method)
    assert got == pytest.approx(400.0, rel=1e-6 if method == "finite" else 1e-12, abs=0)
    assert abs(got - 997.0) > 500.0


def test_symmetry_defect_includes_the_explicit_time_derivative_of_the_boundary_term():
    """K(q, t) = q0 t^2 does not depend on v, so a=None is allowed. With a zero variation the defect is
    -dK/dt = -(t^2 v0 + 2 q0 t) = -(9 * 5 + 2 * 2 * 3) = -57 at q0 = 2, v0 = 5, t = 3; without the dK/dt term it would
    be -45."""

    def boundary(q, v, t):
        return q[0] * t**2

    def zero_generator(q, v, t):
        return np.zeros(2)

    defect = nt.symmetry_defect(
        _free_with_spring, [2.0, 3.0], [5.0, 7.0], 3.0, zero_generator, boundary_term=boundary, a=None
    )
    assert defect == pytest.approx(-57.0, rel=1e-9, abs=0)


def test_any_velocity_dependence_needs_the_acceleration():
    """A generator or boundary term whose velocity derivative is 1e-11 (above the documented 1e-12, below 1e-10)
    must demand the acceleration; one with 1e-13 (below the threshold) must not. The complex-step derivative of a
    linear function is exact, so the derivatives are the stated numbers."""
    q, v = np.array([1.0, 2.0]), np.array([0.3, -0.4])

    def weak_generator(q, v, t):
        return np.array([1e-11 * v[0], 0.0])

    def tiny_generator(q, v, t):
        return np.array([1e-13 * v[0], 0.0])

    def weak_boundary(q, v, t):
        return 1e-11 * v[0]

    def tiny_boundary(q, v, t):
        return 1e-13 * v[0]

    with pytest.raises(ValueError, match="depends on the velocity"):
        nt.symmetry_variation(_free_with_spring, q, v, 0.0, weak_generator)
    with pytest.raises(ValueError, match="depends on the velocity"):
        nt.symmetry_defect(_free_with_spring, q, v, 0.0, _quadratic_generator, boundary_term=weak_boundary)
    # a coefficient of 1e-13 is small only in its units: below an absolute threshold of 1e-12 the term with the
    # acceleration was dropped without a word, and the variation came out wrong by a factor of two
    with pytest.raises(ValueError, match="depends on the velocity"):
        nt.symmetry_variation(_free_with_spring, q, v, 0.0, tiny_generator)
    with pytest.raises(ValueError, match="depends on the velocity"):
        nt.symmetry_defect(_free_with_spring, q, v, 0.0, _quadratic_generator, boundary_term=tiny_boundary)


def test_rotation_generator_needs_exactly_three_axis_components():
    for axis in ([1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0] * 6):
        with pytest.raises(ValueError, match="three components"):
            nt.rotation_generator(axis)
    assert callable(nt.rotation_generator([0.0, 0.0, 1.0]))
