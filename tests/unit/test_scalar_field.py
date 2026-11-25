"""Quintessence: fixed points, stability, trajectories and the generic integrator.

Reference values are the table and text of [Copeland2006] (hepthv3.tex lines 3831-4059); the Jacobians are
computed here by finite differences of the right-hand side and never taken from the module under test.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from kozmos.core._differentiation import jacobian
from kozmos.cosmology import distances
from kozmos.cosmology import scalar_field as sf

S6 = np.sqrt(6.0)


def test_energy_density_pressure_and_equation_of_state():
    assert sf.scalar_rho(2.0, 3.0) == pytest.approx(5.0)
    assert sf.scalar_p(2.0, 3.0) == pytest.approx(-1.0)
    assert sf.scalar_eos(0.0, 1.7) == -1.0  # slow-roll limit [Copeland2006] line 2680
    assert sf.scalar_eos(1.3, 0.0) == 1.0  # stiff matter, line 2682
    phi_dot, v = np.array([0.1, 1.0, 5.0]), np.array([2.0, 1.0, 0.3])
    assert np.allclose(sf.scalar_eos(phi_dot, v), sf.scalar_p(phi_dot, v) / sf.scalar_rho(phi_dot, v), rtol=1e-15)
    assert np.all(np.abs(sf.scalar_eos(phi_dot, v)) <= 1.0)


@pytest.mark.parametrize(
    ("lam", "w_m", "names"),
    [
        (1.0, 0.0, ["a", "b1", "b2", "c"]),  # lambda^2 = 1 < 3 gamma: no scaling point
        (2.0, 0.0, ["a", "b1", "b2", "c", "d"]),  # 3 < 4 < 6
        (-2.0, 0.0, ["a", "b1", "b2", "c", "d"]),
        (2.5, 0.0, ["a", "b1", "b2", "d"]),  # lambda^2 = 6.25 > 6: (c) does not exist
        (4.0, 0.0, ["a", "b1", "b2", "d"]),
        (1.5, 1.0 / 3.0, ["a", "b1", "b2", "c"]),  # radiation: 3 gamma = 4 > 2.25
        (3.0, 1.0 / 3.0, ["a", "b1", "b2", "d"]),  # 4 < 9, (c) absent for lambda^2 = 9 > 6
    ],
)
def test_existence_conditions(lam, w_m, names):
    assert [p.name for p in sf.exponential_fixed_points(lam, w_m)] == names


CASES = [(lam, w_m) for lam in (-5.0, -1.0, 0.5, 1.0, 1.7, 2.05, 2.4, 3.0, 4.0, 7.0, 12.0) for w_m in (0.0, 1.0 / 3.0)]


@pytest.mark.parametrize(("lam", "w_m"), CASES)
def test_every_fixed_point_is_a_zero_of_the_right_hand_side(lam, w_m):
    for p in sf.exponential_fixed_points(lam, w_m):
        assert np.max(np.abs(sf.exponential_quintessence_rhs(0.0, [p.x, p.y], lam, w_m))) < 1e-14  # measured 4.4e-16
        # the constraint of the physical region: Omega_phi <= 1 (the matter fraction 1 - x^2 - y^2 is not negative)
        assert p.omega_phi <= 1.0 + 1e-14 and p.y >= 0.0


@pytest.mark.parametrize(("lam", "w_m"), CASES)
def test_eigenvalues_equal_the_finite_difference_jacobian(lam, w_m):
    for p in sf.exponential_fixed_points(lam, w_m):
        jac = jacobian(lambda s: sf.exponential_quintessence_rhs(0.0, s, lam, w_m), [p.x, p.y], method="finite")
        numerical = np.sort_complex(np.linalg.eigvals(jac))
        listed = np.sort_complex(np.array(p.eigenvalues))
        # measured maximum difference 7.0e-13 over all cases (error of the four-point difference)
        assert np.max(np.abs(numerical - listed)) < 1e-9, (lam, w_m, p.name)


def _table_stability(name, lam, gamma):
    """The 'Stability' column of TABLE I, written from the table (not from the module)."""
    if name == "a":
        return "saddle point"  # for 0 < gamma < 2
    if name == "b1":
        return "unstable node" if lam < S6 else "saddle point"
    if name == "b2":
        return "unstable node" if lam > -S6 else "saddle point"
    if name == "c":
        return "stable node" if lam**2 < 3.0 * gamma else "saddle point"  # exists for lambda^2 < 6
    if name == "d":
        if 9.0 * gamma - 2.0 <= 0.0 or lam**2 < 24.0 * gamma**2 / (9.0 * gamma - 2.0):
            return "stable node"
        return "stable spiral"
    raise AssertionError(name)


@pytest.mark.parametrize(("lam", "w_m"), CASES)
def test_stability_reproduces_the_table_and_the_numerical_eigenvalues(lam, w_m):
    gamma = 1.0 + w_m
    for p in sf.exponential_fixed_points(lam, w_m):
        assert p.stability == _table_stability(p.name, lam, gamma), (lam, w_m, p.name)
        jac = jacobian(lambda s: sf.exponential_quintessence_rhs(0.0, s, lam, w_m), [p.x, p.y], method="finite")
        mu = np.linalg.eigvals(jac)
        if p.stability == "stable node":
            assert np.all(mu.real < 0) and np.all(np.abs(mu.imag) < 1e-9)
        elif p.stability == "stable spiral":
            assert np.all(mu.real < 0) and np.all(np.abs(mu.imag) > 1e-6)
        elif p.stability == "unstable node":
            assert np.all(mu.real > 0)
        else:
            assert p.stability == "saddle point" and mu.real.min() < 0 < mu.real.max()


def test_table_values_of_omega_phi_and_w_phi():
    by_name = {p.name: p for p in sf.exponential_fixed_points(2.0, 0.0)}
    assert by_name["a"].omega_phi == 0.0 and np.isnan(by_name["a"].w_phi)
    assert by_name["b1"].omega_phi == 1.0 and by_name["b1"].w_phi == 1.0  # gamma_phi = 2
    assert by_name["b2"].x == -1.0 and by_name["b2"].w_phi == 1.0
    c = by_name["c"]
    assert (c.x, c.y) == pytest.approx((2.0 / S6, np.sqrt(1.0 - 4.0 / 6.0)), abs=1e-15)
    assert c.omega_phi == pytest.approx(1.0, abs=1e-15) and c.w_phi == pytest.approx(-1.0 + 4.0 / 3.0, abs=1e-14)
    d = by_name["d"]
    # [Copeland2006] lines 4021-4023 (figure caption): lambda = 2, gamma = 1: x = y = (3/8)^{1/2};
    # Omega_phi = 3 gamma/lambda^2
    assert d.x == pytest.approx(np.sqrt(3.0 / 8.0), abs=1e-15) and d.y == pytest.approx(np.sqrt(3.0 / 8.0), abs=1e-15)
    assert d.omega_phi == pytest.approx(0.75, abs=1e-15) and d.w_phi == pytest.approx(0.0, abs=1e-15)
    assert d.w_eff == pytest.approx(0.0, abs=1e-15)  # w_eff = w_m at the scaling solution
    # scalar-dominated point: w_eff = w_phi = -1 + lambda^2/3 (line 3999)
    assert c.w_eff == pytest.approx(c.w_phi, abs=1e-15)
    # saddle (c) of the figure caption: x = (2/3)^{1/2}, y = (1/3)^{1/2}
    assert (c.x, c.y) == pytest.approx((np.sqrt(2.0 / 3.0), np.sqrt(1.0 / 3.0)), abs=1e-15)


def test_invalid_background_is_rejected():
    for w_m in (-1.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            sf.exponential_fixed_points(1.0, w_m)
    with pytest.raises(ValueError):
        sf.integrate_exponential_quintessence(1.0, 1.0, 1.0, 0.0, (0.0, 1.0))  # outside the unit disc


def _end_state(x0, y0, lam, w_m, n_end=80.0):
    s = sf.integrate_exponential_quintessence(x0, y0, lam, w_m, (0.0, n_end))
    return s.x[-1], s.y[-1], s.omega_phi[-1], s.w_phi[-1], s.w_eff[-1]


def test_trajectory_from_the_fluid_dominated_saddle_to_the_scalar_dominated_point():
    # small lambda: lambda^2 = 1 < 3 gamma, (c) is the stable node with Omega_phi = 1 and w_phi = -1 + lambda^2/3
    for lam in (0.5, 1.0, 1.4):
        x, y, omega, w_phi, w_eff = _end_state(1e-3, 1e-3, lam, 0.0)
        assert omega == pytest.approx(1.0, abs=1e-9)  # measured 7.0e-11
        assert w_phi == pytest.approx(-1.0 + lam**2 / 3.0, abs=1e-9)
        assert w_eff == pytest.approx(-1.0 + lam**2 / 3.0, abs=1e-9)
        assert (x, y) == pytest.approx((lam / S6, np.sqrt(1.0 - lam**2 / 6.0)), abs=1e-9)


def test_trajectory_to_the_scaling_solution_for_large_lambda():
    # lambda^2 > 3 gamma: Omega_phi = 3 (1 + w_m)/lambda^2, w_phi = w_m
    for lam, w_m in ((4.0, 0.0), (6.0, 0.0), (5.0, 1.0 / 3.0), (3.0, 0.0)):
        x, y, omega, w_phi, w_eff = _end_state(1e-3, 1e-3, lam, w_m)
        assert omega == pytest.approx(3.0 * (1.0 + w_m) / lam**2, abs=1e-9)  # measured 2.0e-11
        assert w_phi == pytest.approx(w_m, abs=1e-9)
        assert w_eff == pytest.approx(w_m, abs=1e-9)


def test_trajectory_started_near_the_scalar_dominated_saddle_ends_on_the_scaling_point():
    # lambda = 2, gamma = 1: (c) is a saddle, (d) the stable spiral at x = y = (3/8)^{1/2} (figure caption)
    c = [p for p in sf.exponential_fixed_points(2.0, 0.0) if p.name == "c"][0]
    x, y, omega, w_phi, _ = _end_state(c.x - 1e-6, c.y - 1e-6, 2.0, 0.0)
    assert (x, y) == pytest.approx((np.sqrt(3.0 / 8.0),) * 2, abs=1e-9)
    assert omega == pytest.approx(0.75, abs=1e-9) and w_phi == pytest.approx(0.0, abs=1e-9)
    # started near the saddle (b1) for lambda > sqrt(6) the flow goes to (d) as well
    x, y, omega, _, _ = _end_state(1.0 - 1e-6, 1e-6, 3.0, 0.0)
    assert omega == pytest.approx(3.0 / 9.0, abs=1e-9)


def test_trajectory_is_continuous_backwards_and_reports_the_requested_grid():
    s = sf.integrate_exponential_quintessence(0.2, 0.3, 1.0, 0.0, (0.0, -3.0), n_eval=np.linspace(0.0, -3.0, 7))
    assert s.n.shape == (7,) and s.x[0] == pytest.approx(0.2) and s.y[0] == pytest.approx(0.3)
    assert np.allclose(s.omega_phi, s.x**2 + s.y**2, rtol=1e-15)


def _exponential(lam, v0):
    return lambda p: v0 * np.exp(-lam * p)


def test_generic_integrator_with_a_constant_potential_is_lcdm():
    # E(z)^2 = Omega_Lambda + Omega_m (1+z)^3 with v = Omega_Lambda (kappa^2 V = 3 H0^2 Omega_Lambda)
    k = distances.Cosmology.flat(70.0, 0.3)
    n = np.linspace(0.0, -np.log(11.0), 40)
    sol = sf.integrate_quintessence(lambda p: 0.7 + 0.0 * p, 0.0, 0.0, (0.0, n[-1]), 0.3, n_eval=n)
    z = np.exp(-sol.n) - 1.0
    assert np.max(np.abs(sol.e / distances.e_function(z, k) - 1.0)) < 1e-10  # measured 4.4e-16
    assert np.max(np.abs(sol.w_phi + 1.0)) < 1e-15 and np.all(sol.phi == 0.0)
    # with radiation as well
    k_r = distances.Cosmology.flat(70.0, 0.3, omega_r=8.5e-5)
    sol = sf.integrate_quintessence(
        lambda p: 1.0 - 0.3 - 8.5e-5 + 0.0 * p,
        0.0,
        0.0,
        (0.0, -np.log(1101.0)),
        0.3,
        8.5e-5,
        n_eval=-np.linspace(0, 7.0, 30),
    )
    z = np.exp(-sol.n) - 1.0
    assert np.max(np.abs(sol.e / distances.e_function(z, k_r) - 1.0)) < 1e-9  # measured 6.7e-16


def test_kinetic_energy_of_a_free_field_scales_as_a_to_the_minus_six():
    # constant potential, no matter: rho_kin = phi_dot^2/2 ∝ a^-6, i.e. E^2 phi'^2 / 6 e^{6N} is constant
    sol = sf.integrate_quintessence(lambda p: 1.0 + 0.0 * p, 0.0, 0.5, (0.0, 3.0), n_eval=np.linspace(0.0, 3.0, 25))
    invariant = sol.e**2 * sol.dphi_dn**2 / 6.0 * np.exp(6.0 * sol.n)
    assert np.max(np.abs(invariant / invariant[0] - 1.0)) < 1e-8  # measured 1.3e-9


@pytest.mark.parametrize(("lam", "x0", "y0"), [(1.0, 0.3, 0.6), (2.0, 0.1, 0.2), (4.0, 0.5, 0.3)])
def test_generic_integrator_agrees_with_the_autonomous_system(lam, x0, y0):
    # matter-only background (w_m = 0); closure at N = 0: v = y0^2 (E = 1), Omega_m(N = 0) = 1 - x0^2 - y0^2
    n = np.linspace(0.0, 6.0, 25)
    generic = sf.integrate_quintessence(
        _exponential(lam, y0**2), 0.0, S6 * x0, (0.0, 6.0), omega_matter_today=1.0 - x0**2 - y0**2, n_eval=n
    )
    auto = sf.integrate_exponential_quintessence(x0, y0, lam, 0.0, (0.0, 6.0), n_eval=n)
    x_generic = generic.dphi_dn / S6
    y_generic = np.sqrt(_exponential(lam, y0**2)(generic.phi)) / generic.e
    # measured maxima over the three cases: x 6.5e-10, y 2.7e-10, Omega_phi 3.6e-10, w_phi 1.2e-9
    assert np.max(np.abs(x_generic - auto.x)) < 1e-8
    assert np.max(np.abs(y_generic - auto.y)) < 1e-8
    assert np.max(np.abs(generic.omega_phi - auto.omega_phi)) < 1e-8
    assert np.max(np.abs(generic.w_phi - auto.w_phi)) < 1e-7


def test_generic_integrator_derivative_options_and_errors():
    n = np.linspace(0.0, 2.0, 5)
    default = sf.integrate_quintessence(_exponential(1.5, 0.5), 0.0, 0.4, (0.0, 2.0), 0.4, n_eval=n)
    given = sf.integrate_quintessence(
        _exponential(1.5, 0.5),
        0.0,
        0.4,
        (0.0, 2.0),
        0.4,
        potential_derivative=lambda p: -1.5 * 0.5 * np.exp(-1.5 * p),
        n_eval=n,
    )
    assert np.max(np.abs(default.phi - given.phi)) < 1e-10  # complex-step derivative is exact to rounding
    with pytest.raises(ValueError):
        sf.integrate_quintessence(_exponential(1.0, 1.0), 0.0, 3.0, (0.0, 1.0))
    with pytest.raises(ValueError):
        sf.integrate_quintessence(_exponential(1.0, 1.0), 0.0, 0.0, (0.0, 1.0), omega_matter_today=-0.1)


# --- boundaries of the classification --------------------------------------------------------------------------


def _point(lam, w_m, name):
    return {p.name: p for p in sf.exponential_fixed_points(lam, w_m)}.get(name)


def _names(lam, w_m):
    return [p.name for p in sf.exponential_fixed_points(lam, w_m)]


def test_boundary_lambda_squared_is_six():
    # before the fix: the eigenvalue 3 - (sqrt(6)/2) lambda = 4.4e-16 gave "unstable node" for b1 at lambda = sqrt(6)
    # (and "saddle point" for the neighbouring float), so the class depended on the rounding of sqrt(6)
    for lam in (S6, np.nextafter(S6, 0.0), np.nextafter(S6, 10.0), S6 * (1.0 + 1e-14)):
        assert _names(lam, 0.0) == ["a", "b1", "b2", "d"]  # (c) merges with (b1) and is not listed
        assert _point(lam, 0.0, "b1").stability == "non-hyperbolic", lam
        assert _point(lam, 0.0, "b2").stability == "unstable node"
        # measured 4.4e-16 for sqrt(6), 2.9e-14 for the 1e-14 shift
        assert abs(_point(lam, 0.0, "b1").eigenvalues[0]) < 1e-13
        assert _point(lam, 0.0, "d").stability == "stable spiral"  # lambda^2 = 6 > 24/7: discriminant -3
    # the mirror image lambda -> -lambda
    assert _names(-S6, 0.0) == ["a", "b1", "b2", "d"]
    assert _point(-S6, 0.0, "b2").stability == "non-hyperbolic"
    assert _point(-S6, 0.0, "b1").stability == "unstable node"
    assert _point(S6, 1.0 / 3.0, "b1").stability == "non-hyperbolic"
    # clearly on each side of the boundary the table applies
    assert _point(S6 * (1.0 - 1e-6), 0.0, "b1").stability == "unstable node"
    assert _point(S6 * (1.0 + 1e-6), 0.0, "b1").stability == "saddle point"
    assert "c" in _names(S6 * (1.0 - 1e-6), 0.0) and "c" not in _names(S6 * (1.0 + 1e-6), 0.0)


def test_boundary_lambda_squared_is_three_gamma():
    # w_m = 0: lambda = sqrt(3); before the fix mu_2 = lambda^2 - 3 = -4.4e-16 (or +) gave "stable node" or
    # "saddle point" for (c) and listed or dropped (d) depending on the rounding of sqrt(3)
    for lam in (
        np.sqrt(3.0),
        np.nextafter(np.sqrt(3.0), 0.0),
        np.nextafter(np.sqrt(3.0), 10.0),
        np.sqrt(3.0) * (1 + 1e-14),
    ):
        assert _names(lam, 0.0) == ["a", "b1", "b2", "c"], lam  # (d) merges with (c) and is not listed
        c = _point(lam, 0.0, "c")
        # eigenvalue measured 4.4e-16 for sqrt(3), 6.0e-14 for the 1e-14 shift
        assert c.stability == "non-hyperbolic" and abs(c.eigenvalues[1]) < 1e-13
        assert c.eigenvalues[0].real == pytest.approx(-1.5, abs=1e-12)
    # w_m = 1/3 (gamma = 4/3): lambda = 2 is exactly on the boundary
    assert _names(2.0, 1.0 / 3.0) == ["a", "b1", "b2", "c"]
    assert _point(2.0, 1.0 / 3.0, "c").stability == "non-hyperbolic"
    # clearly on each side
    assert _point(np.sqrt(3.0) * (1.0 - 1e-6), 0.0, "c").stability == "stable node"
    below, above = _names(np.sqrt(3.0) * (1.0 - 1e-6), 0.0), _names(np.sqrt(3.0) * (1.0 + 1e-6), 0.0)
    assert "d" not in below and "d" in above
    assert _point(np.sqrt(3.0) * (1.0 + 1e-6), 0.0, "c").stability == "saddle point"


@pytest.mark.parametrize("w_m", [0.0, 1.0 / 3.0])
def test_boundary_of_the_stable_spiral(w_m):
    # lambda^2 = 24 gamma^2 / (9 gamma - 2): the discriminant of (d) is zero, the eigenvalues are equal and real.
    # Before the fix the imaginary part of sqrt(rounding-sized discriminant) was ~1e-8, so the 1e-14 threshold was
    # ineffective: the class was "stable node" or "stable spiral" depending on the last bit of lambda
    gamma = 1.0 + w_m
    lam0 = np.sqrt(24.0 * gamma**2 / (9.0 * gamma - 2.0))
    for lam in (lam0, np.nextafter(lam0, 0.0), np.nextafter(lam0, 10.0), lam0 * (1.0 + 1e-14), lam0 * (1.0 - 1e-14)):
        d = _point(lam, w_m, "d")
        assert d.stability == "stable node", (lam, d.eigenvalues)
        assert d.eigenvalues[0].imag == 0.0 and d.eigenvalues[1].imag == 0.0
        assert d.eigenvalues[0].real == pytest.approx(-0.75 * (2.0 - gamma), abs=1e-14)
        assert d.eigenvalues[1].real == d.eigenvalues[0].real
    # the finite-difference Jacobian agrees to the conditioning of a double eigenvalue (sqrt of the 1e-13 error)
    d = _point(lam0, w_m, "d")
    jac = jacobian(lambda s: sf.exponential_quintessence_rhs(0.0, s, lam0, w_m), [d.x, d.y], method="finite")
    # measured 8.9e-7 (w_m = 0) and 3.9e-7 (w_m = 1/3)
    assert np.max(np.abs(np.linalg.eigvals(jac) - d.eigenvalues[0])) < 1e-5
    # outside the tolerance the discriminant decides: node below the boundary, spiral above, both with distinct
    # eigenvalues
    below = _point(lam0 * (1.0 - 1e-5), w_m, "d")
    above = _point(lam0 * (1.0 + 1e-5), w_m, "d")
    assert below.stability == "stable node" and below.eigenvalues[0] != below.eigenvalues[1]
    assert above.stability == "stable spiral" and above.eigenvalues[0].imag != 0.0
    assert above.eigenvalues[0] == above.eigenvalues[1].conjugate()


# --- generic integrator: independent integration in cosmic time ---------------------------------------------------


def _cosmic_time_reference(v, dv, om, orad, psi0, p0, n0, n_targets):
    """phi'' + 3 H phi' + V' = 0 with the Friedmann constraint, in tau = H0 t (no N, no H'/H anywhere).

    State (psi, psi_tau, a) with psi = kappa phi: E^2 = psi_tau^2/6 + v(psi) + om a^-3 + orad a^-4,
    psi_tauu = -3 E psi_tau - 3 v'(psi), a_tau = a E. Returns (psi, psi_N, E) at the e-folds ``n_targets``.
    """

    def e_of(psi, w, a):
        return np.sqrt(w**2 / 6.0 + v(psi) + om * a**-3 + orad * a**-4)

    def rhs(_tau, y):
        psi, w, a = y
        e = e_of(psi, w, a)
        return [w, -3.0 * e * w - 3.0 * dv(psi), a * e]

    a0 = np.exp(n0)
    e0 = np.sqrt((v(psi0) + om * a0**-3 + orad * a0**-4) / (1.0 - p0**2 / 6.0))  # the constraint solved for E
    sol = solve_ivp(
        rhs,
        (0.0, 50.0),
        [psi0, e0 * p0, a0],
        method="DOP853",
        rtol=1e-13,
        atol=1e-14,
        dense_output=True,
        events=lambda _t, y: y[2] - np.exp(n_targets[-1]),
    )
    out = []
    for n in n_targets:
        tau = brentq(lambda t, n=n: np.log(sol.sol(t)[2]) - n, 0.0, sol.t[-1], xtol=1e-15, rtol=1e-15)
        psi, w, a = sol.sol(tau)
        e = e_of(psi, w, a)
        out.append((psi, w / e, e))
    return np.array(out)


def test_radiation_term_of_the_background_against_cosmic_time_integration():
    # matter + radiation + a non-constant potential v = 0.55 exp(-1.2 psi); start at N = -3 (radiation and matter
    # comparable: rho_r = 1.6e3, rho_m = 2.4e3 in units of 3 H0^2/kappa^2) with psi' = 0.6, integrate to N = 0.
    # The reference never uses H'/H = -psi'^2/2 - (3/2) Omega_m - 2 Omega_r. With the radiation coefficient 2 replaced
    # by 1.5 (a scratch copy of the module) the differences become 3.6e-2 in psi, 2.6e-2 in psi_N and 8.4e-3 in E.
    def v(psi):
        return 0.55 * np.exp(-1.2 * psi)

    def dv(psi):
        return -1.2 * v(psi)

    n = np.array([-3.0, -2.5, -2.0, -1.0, -0.5, 0.0])
    sol = sf.integrate_quintessence(v, 0.3, 0.6, (-3.0, 0.0), 0.3, 0.01, potential_derivative=dv, n_eval=n)
    reference = _cosmic_time_reference(v, dv, 0.3, 0.01, 0.3, 0.6, -3.0, n)
    # measured maximum differences: psi 3.2e-12, psi_N 4.9e-12, E 6.2e-13 (relative)
    assert np.max(np.abs(sol.phi - reference[:, 0])) < 1e-9
    assert np.max(np.abs(sol.dphi_dn - reference[:, 1])) < 1e-9
    assert np.max(np.abs(sol.e / reference[:, 2] - 1.0)) < 1e-9
    # the field really moves: psi changes by 0.587 over the run, so the test is not a trivial identity
    assert abs(reference[-1, 0] - 0.3) > 0.5


def test_negative_total_energy_density_raises_as_documented():
    # before the fix: v = -0.5, Omega_m = 0.3 gave E = NaN with a numpy RuntimeWarning
    with pytest.raises(ValueError, match="not positive"):
        sf.integrate_quintessence(lambda p: -0.5 + 0.0 * p, 0.0, 0.0, (0.0, 1.0), omega_matter_today=0.3)
    # a negative potential is allowed while matter keeps the total positive (v = -0.2, Omega_m = 1.0 at N = 0)
    sol = sf.integrate_quintessence(lambda p: -0.2 + 0.0 * p, 0.0, 0.0, (0.0, 0.5), omega_matter_today=1.0)
    assert np.all(np.isfinite(sol.e)) and sol.e[0] == pytest.approx(np.sqrt(0.8), rel=1e-14)
    # ... but the total turns negative later when going forward (matter dilutes): the error names that moment
    with pytest.raises(ValueError, match="not positive"):
        sf.integrate_quintessence(lambda p: -0.5 + 0.0 * p, 0.0, 0.0, (0.0, 3.0), omega_matter_today=1.0)
