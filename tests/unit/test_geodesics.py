"""Tests of kozmos.relativity.metrics and kozmos.relativity.geodesics.

Units are geometric (G = c = 1) and M = 1 unless stated. Reference values are closed forms taken from the
sources (Carroll's printed Christoffel symbols, effective potential, Killing tensor; Abramowicz's Keplerian
angular velocity and frame-dragging frequency; Visser's metric components), from conservation laws, or from
independent oracles (scipy.integrate.quad, scipy.optimize.brentq). Tolerances are about ten times the value
measured in this file (the measured value stands next to every tolerance).

Helper functions return the measured quantity so that the numbers can be printed outside pytest.
"""

from __future__ import annotations

from functools import cache, partial

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq

from kozmos.relativity.geodesics import (
    christoffel_symbols,
    geodesic_acceleration,
    integrate_geodesic,
    killing_conserved_quantity,
    killing_equation_defect,
    normalize_tangent,
    stationary_axisymmetric_constants,
    tangent_norm_squared,
)
from kozmos.relativity.metrics import (
    kerr_eddington_finkelstein_metric,
    kerr_metric,
    metric_inverse,
    minkowski_metric,
    reissner_nordstrom_metric,
    schwarzschild_eddington_finkelstein_metric,
    schwarzschild_metric,
)


def sch(mass: float = 1.0):
    return partial(schwarzschild_metric, mass_geometric_m=mass)


def kerr(mass: float = 1.0, spin: float = 0.9):
    return partial(kerr_metric, mass_geometric_m=mass, spin_a_m=spin)


def flat_spherical(x):
    """Minkowski metric in spherical coordinates (t, r, theta, phi), complex safe (test-only)."""
    g = np.zeros((4, 4), dtype=np.result_type(x, float))
    g[0, 0] = -1.0
    g[1, 1] = 1.0
    g[2, 2] = x[1] ** 2
    g[3, 3] = x[1] ** 2 * np.sin(x[2]) ** 2
    return g


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------


def test_minkowski_metric_is_diagonal_with_signature_minus_plus_plus_plus():
    g = minkowski_metric(np.zeros(4))
    assert np.array_equal(g, np.diag([-1.0, 1.0, 1.0, 1.0]))
    assert minkowski_metric(np.zeros(4, dtype=complex)).dtype == complex


def test_metrics_are_symmetric_and_have_lorentzian_signature():
    points = [np.array([0.0, 7.0, 1.1, 0.4]), np.array([0.0, 25.0, 2.0, 5.0])]
    metrics = [
        sch(),
        kerr(),
        partial(reissner_nordstrom_metric, mass_geometric_m=1.0, charge_m=0.6),
        partial(schwarzschild_eddington_finkelstein_metric, mass_geometric_m=1.0),
        partial(kerr_eddington_finkelstein_metric, mass_geometric_m=1.0, spin_a_m=0.9),
    ]
    for metric in metrics:
        for x in points:
            g = metric(x)
            assert np.array_equal(g, g.T)
            eigenvalues = np.linalg.eigvalsh(g)
            assert np.sum(eigenvalues < 0) == 1 and np.sum(eigenvalues > 0) == 3


def test_metric_inverse_is_the_inverse():
    for metric, x in [(sch(), [0, 6.0, 1.0, 0.2]), (kerr(), [0, 6.0, 1.0, 0.2])]:
        g = metric(np.array(x))
        assert metric_inverse(g) @ g == pytest.approx(np.eye(4), abs=1e-14)
    with pytest.raises(ValueError):
        metric_inverse(np.eye(3))


def test_metrics_are_complex_safe_and_reject_wrong_shape():
    x = np.array([0.0, 7.0, 1.1, 0.4], dtype=complex)
    x[1] += 1e-30j
    for metric in (sch(), kerr(), partial(kerr_eddington_finkelstein_metric, mass_geometric_m=1.0, spin_a_m=0.5)):
        assert metric(x).dtype == complex
    with pytest.raises(ValueError):
        schwarzschild_metric(np.zeros(3), 1.0)


def test_kerr_components_agree_with_the_line_elements_of_carroll_and_visser():
    # Carroll Eq. 7.114: ds^2 = -dt^2 + rho^2/Delta dr^2 + rho^2 dtheta^2 + (r^2 + a^2) sin^2 dphi^2
    #                          + (2 M r / rho^2)(a sin^2 dphi - dt)^2
    # Visser (Boyer-Lindquist section): g_t_phi = -2 m r a sin^2 / rho^2 (coefficient -4 m r a sin^2/rho^2 of dt dphi)
    # and g_phi_phi = [r^2 + a^2 + 2 m r a^2 sin^2 / rho^2] sin^2.
    m, a = 1.3, 0.8
    for r, th in [(5.0, 0.7), (11.0, 1.5), (3.0, 2.5)]:
        sin2, cos2 = np.sin(th) ** 2, np.cos(th) ** 2
        rho2 = r**2 + a**2 * cos2
        delta = r**2 - 2 * m * r + a**2
        # build the quadratic form of Eq. 7.114 on the basis dt, dr, dtheta, dphi
        w = np.array([-1.0, 0.0, 0.0, a * sin2])  # (a sin^2 dphi - dt)
        carroll = np.zeros((4, 4))
        carroll[0, 0] = -1.0
        carroll[1, 1] = rho2 / delta
        carroll[2, 2] = rho2
        carroll[3, 3] = (r**2 + a**2) * sin2
        carroll += 2 * m * r / rho2 * np.outer(w, w)
        # measured maximum relative difference 7e-17 (rounding); tolerance 1e-15
        assert kerr_metric(np.array([0.0, r, th, 0.0]), m, a) == pytest.approx(carroll, rel=1e-15, abs=1e-15)
        assert carroll[0, 3] == pytest.approx(-2 * m * r * a * sin2 / rho2, rel=1e-15, abs=0)
        assert carroll[3, 3] == pytest.approx((r**2 + a**2 + 2 * m * r * a**2 * sin2 / rho2) * sin2, rel=1e-15, abs=0)


def test_kerr_with_zero_spin_is_schwarzschild_and_zero_mass_is_flat_oblate_spheroidal():
    x = np.array([0.0, 6.5, 1.2, 0.3])
    assert kerr_metric(x, 1.0, 0.0) == pytest.approx(schwarzschild_metric(x, 1.0), abs=1e-15)
    # m = 0: Carroll Eq. 7.117 / Visser: g_rr = (r^2 + a^2 cos^2)/(r^2 + a^2), g_thth = r^2 + a^2 cos^2,
    # g_phiphi = (r^2 + a^2) sin^2 (Visser's form; Carroll's printed 7.117 squares rho^2, see the report)
    r, th, a = 4.0, 0.9, 0.7
    g = kerr_metric(np.array([0.0, r, th, 0.0]), 0.0, a)
    assert g[1, 1] == pytest.approx((r**2 + a**2 * np.cos(th) ** 2) / (r**2 + a**2), rel=1e-14, abs=0)
    assert g[2, 2] == pytest.approx(r**2 + a**2 * np.cos(th) ** 2, rel=1e-14, abs=0)
    assert g[3, 3] == pytest.approx((r**2 + a**2) * np.sin(th) ** 2, rel=1e-14, abs=0)
    assert g[0, 3] == 0.0


def test_carroll_null_vectors_of_kerr_are_null_with_unit_product():
    # Eq. 7.122, 7.123: l^mu = (r^2 + a^2, Delta, 0, a)/Delta, n^mu = (r^2 + a^2, -Delta, 0, a)/(2 rho^2);
    # l.l = 0, n.n = 0, l.n = -1. This checks kerr_metric against the independently printed vectors.
    m, a = 1.0, 0.6
    for r, th in [(4.0, 0.8), (9.0, 2.0)]:
        delta = r**2 - 2 * m * r + a**2
        rho2 = r**2 + a**2 * np.cos(th) ** 2
        lvec = np.array([r**2 + a**2, delta, 0.0, a]) / delta
        n = np.array([r**2 + a**2, -delta, 0.0, a]) / (2 * rho2)
        g = kerr_metric(np.array([0.0, r, th, 0.0]), m, a)
        # measured maximum defect 2.2e-16; tolerance 2e-15
        assert lvec @ g @ lvec == pytest.approx(0.0, abs=2e-15)
        assert n @ g @ n == pytest.approx(0.0, abs=2e-15)
        assert lvec @ g @ n == pytest.approx(-1.0, abs=2e-15)


def test_frame_dragging_frequency_abramowicz_closed_form_equals_minus_gtphi_over_gphiphi():
    # Abramowicz Eq. dragg-velocity-kerr: omega = 2 a M r / ((a^2 + r^2) rho^2 + 2 a^2 M r sin^2 theta) and the text
    # defines omega = -g_tphi / g_phiphi.
    m, a = 1.0, 0.7
    for r, th in [(3.0, 0.5), (6.0, np.pi / 2), (12.0, 2.4)]:
        rho2 = r**2 + a**2 * np.cos(th) ** 2
        omega_printed = 2 * a * m * r / ((a**2 + r**2) * rho2 + 2 * a**2 * m * r * np.sin(th) ** 2)
        g = kerr_metric(np.array([0.0, r, th, 0.0]), m, a)
        assert -g[0, 3] / g[3, 3] == pytest.approx(omega_printed, rel=2e-15, abs=0)  # measured 2.2e-16


# --------------------------------------------------------------------------------------
# Christoffel symbols and acceleration
# --------------------------------------------------------------------------------------


def _schwarzschild_christoffel_printed(r: float, th: float, m: float = 1.0) -> np.ndarray:
    """The non-zero symbols printed in Carroll Eq. 7.33 (indices 0 1 2 3 = t r theta phi)."""
    gam = np.zeros((4, 4, 4))
    gam[1, 0, 0] = m * (r - 2 * m) / r**3
    gam[1, 1, 1] = -m / (r * (r - 2 * m))
    gam[0, 0, 1] = gam[0, 1, 0] = m / (r * (r - 2 * m))
    gam[2, 1, 2] = gam[2, 2, 1] = 1 / r
    gam[1, 2, 2] = -(r - 2 * m)
    gam[3, 1, 3] = gam[3, 3, 1] = 1 / r
    gam[1, 3, 3] = -(r - 2 * m) * np.sin(th) ** 2
    gam[2, 3, 3] = -np.sin(th) * np.cos(th)
    gam[3, 2, 3] = gam[3, 3, 2] = np.cos(th) / np.sin(th)
    return gam


def test_schwarzschild_christoffel_symbols_equal_carroll_eq_7_33():
    # Measured (r, theta) = (5, 0.4), (9, 1.3), (30, 2.7): complex step, maximum absolute difference from the printed
    # closed form 8.9e-16 (the largest symbol is 28, so 1e-16 relative: rounding of the closed form itself), tolerance
    # 1e-14; four-point finite method, maximum difference relative to the largest symbol 6.9e-13, 1.1e-13, 2.1e-12,
    # tolerance 2e-11.
    worst_complex = worst_finite = 0.0
    for r, th in [(5.0, 0.4), (9.0, 1.3), (30.0, 2.7)]:
        x = np.array([0.0, r, th, 0.9])
        ref = _schwarzschild_christoffel_printed(r, th)
        worst_complex = max(worst_complex, np.abs(christoffel_symbols(sch(), x) - ref).max())
        finite = christoffel_symbols(sch(), x, method="finite")
        worst_finite = max(worst_finite, np.abs(finite - ref).max() / np.abs(ref).max())
    assert worst_complex < 1e-14
    assert worst_finite < 2e-11


def test_christoffel_symbols_are_symmetric_in_the_lower_indices_and_flat_values():
    gamma = christoffel_symbols(kerr(), np.array([0.0, 6.0, 1.0, 0.3]))
    assert gamma == pytest.approx(np.swapaxes(gamma, 1, 2), abs=1e-15)
    # flat space in spherical coordinates: Gamma^r_thth = -r, Gamma^r_phph = -r sin^2, Gamma^th_rth = 1/r, ...
    r, th = 3.0, 0.8
    flat = christoffel_symbols(flat_spherical, np.array([0.0, r, th, 0.0]))
    assert flat[1, 2, 2] == pytest.approx(-r, rel=1e-14, abs=0)
    assert flat[1, 3, 3] == pytest.approx(-r * np.sin(th) ** 2, rel=1e-14, abs=0)
    assert flat[2, 1, 2] == pytest.approx(1 / r, rel=1e-14, abs=0)
    assert flat[2, 3, 3] == pytest.approx(-np.sin(th) * np.cos(th), rel=1e-14, abs=0)
    assert flat[3, 2, 3] == pytest.approx(np.cos(th) / np.sin(th), rel=1e-14, abs=0)
    assert np.count_nonzero(np.abs(flat) > 1e-14) == 9
    assert np.abs(christoffel_symbols(minkowski_metric, np.array([1.0, 2.0, 3.0, 4.0]))).max() == 0.0


def test_non_analytic_metric_is_detected_by_the_complex_method_and_works_with_the_finite_one():
    def bad(x):
        # np.abs of a complex number is its modulus: the imaginary part of g_11 vanishes and the complex-step
        # derivative would silently be zero; the helper compares it with a central difference and raises
        g = np.zeros((4, 4))
        g[0, 0] = -1.0
        g[1, 1] = 1.0 + 0.1 * np.abs(x[1])
        g[2, 2] = g[3, 3] = 1.0
        return g

    x = np.array([0.0, 3.0, 0.0, 0.0])
    gamma = christoffel_symbols(bad, x, method="finite")
    # Gamma^r_rr = (1/2) g^rr d_r g_rr = 0.5 * 0.1 / 1.3; measured relative error 4.9e-14, tolerance 1e-12
    assert gamma[1, 1, 1] == pytest.approx(0.5 * 0.1 / (1.0 + 0.3), rel=1e-12, abs=0)
    with pytest.raises(ValueError):
        christoffel_symbols(bad, x, method="complex")


def test_geodesic_acceleration_of_flat_space_and_of_a_schwarzschild_circular_orbit():
    # Cartesian flat space: no acceleration
    assert np.array_equal(
        geodesic_acceleration(minkowski_metric, np.zeros(4), np.array([1.0, 0.3, -0.2, 0.1])), np.zeros(4)
    )
    # Circular orbit r = 10: u^t = 1/sqrt(1 - 3M/r), u^phi = sqrt(M/r^3) u^t has zero radial acceleration
    r = 10.0
    ut = 1.0 / np.sqrt(1.0 - 3.0 / r)
    u = np.array([ut, 0.0, 0.0, np.sqrt(1.0 / r**3) * ut])
    x = np.array([0.0, r, np.pi / 2, 0.0])
    acc = geodesic_acceleration(sch(), x, u)
    assert np.abs(acc).max() < 1e-16  # measured 3.5e-18
    # a static observer (u^phi = 0) accelerates inwards: d2r = -Gamma^r_tt (u^t)^2 = -M f / r^2 / f = -M/r^2
    static_u = np.array([1.0 / np.sqrt(1.0 - 2.0 / r), 0.0, 0.0, 0.0])
    assert geodesic_acceleration(sch(), x, static_u)[1] == pytest.approx(
        -1.0 / r**2, rel=1e-14, abs=0
    )  # measured 4e-16


# --------------------------------------------------------------------------------------
# normalisation and conserved quantities
# --------------------------------------------------------------------------------------


def test_normalize_tangent_timelike_and_null_in_schwarzschild_and_kerr():
    x = np.array([0.0, 8.0, 1.2, 0.3])
    spatial = np.array([0.1, -0.05, 0.02])
    for metric in (sch(), kerr()):
        u = normalize_tangent(metric, x, spatial)
        assert u[0] > 0.0
        # measured norm defects 1.1e-16 (timelike) and 3.1e-17 / 1.7e-17 (null); tolerance 1e-15
        assert tangent_norm_squared(metric, x, u) == pytest.approx(-1.0, abs=1e-15)
        k = normalize_tangent(metric, x, spatial, kind="null")
        assert k[0] > 0.0
        assert abs(tangent_norm_squared(metric, x, k)) < 1e-15
    # static observer: u^t = 1/sqrt(f)
    f = 1.0 - 2.0 / 8.0
    assert normalize_tangent(sch(), x, np.zeros(3))[0] == pytest.approx(1.0 / np.sqrt(f), rel=1e-15, abs=0)
    # circular orbit at r = 10: u^phi = Omega u^t gives u^t = 1/sqrt(1 - 3M/r) (Eq. 7.45 with u^r = 0)
    r = 10.0
    uphi = np.sqrt(1.0 / r**3) / np.sqrt(1.0 - 3.0 / r)
    circ = normalize_tangent(sch(), np.array([0.0, r, np.pi / 2, 0.0]), np.array([0.0, 0.0, uphi]))
    assert circ[0] == pytest.approx(1.0 / np.sqrt(1.0 - 3.0 / r), rel=1e-14, abs=0)


def test_normalize_tangent_null_with_zero_spatial_part_raises():
    # A null vector with u^i = 0 would be the zero vector (u^t = 0): ValueError, in Schwarzschild and Kerr (outside the
    # ergosphere) and for the Minkowski metric.
    for metric, x in (
        (sch(), np.array([0.0, 8.0, 1.0, 0.0])),
        (kerr(), np.array([0.0, 8.0, 1.0, 0.0])),
        (minkowski_metric, np.array([0.0, 1.0, 2.0, 3.0])),
    ):
        with pytest.raises(ValueError, match="non-zero spatial part"):
            normalize_tangent(metric, x, np.zeros(3), kind="null")
        # the timelike case with zero spatial part stays valid (static observer)
        assert normalize_tangent(metric, x, np.zeros(3))[0] > 0.0


def test_normalize_tangent_rejects_unsupported_cases():
    with pytest.raises(ValueError):
        normalize_tangent(sch(), np.array([0.0, 8.0, 1.0, 0.0]), np.zeros(3), kind="spacelike")  # type: ignore[arg-type]
    # inside the ergosphere of Kerr (g_tt > 0): r = 1.5, theta = pi/2, a = 0.9: g_tt = -(1 - 2 / 1.5) > 0
    with pytest.raises(ValueError):
        normalize_tangent(kerr(), np.array([0.0, 1.5, np.pi / 2, 0.0]), np.array([0.0, 0.0, 0.1]))
    # g_uu = 0 on the horizon of the Eddington-Finkelstein form and g_ur u^r = 0 for a zero spatial part: u^u is undetermined
    ef = partial(schwarzschild_eddington_finkelstein_metric, mass_geometric_m=1.0)
    with pytest.raises(ValueError):
        normalize_tangent(ef, np.array([0.0, 2.0, np.pi / 2, 0.0]), np.zeros(3))
    # with u^r = -1 the normalisation is linear in u^u: 2 g_ur u^u u^r = -1 gives u^u = 1/2
    ingoing = normalize_tangent(ef, np.array([0.0, 2.0, np.pi / 2, 0.0]), np.array([-1.0, 0.0, 0.0]))
    assert ingoing[0] == 0.5
    assert tangent_norm_squared(ef, np.array([0.0, 2.0, np.pi / 2, 0.0]), ingoing) == pytest.approx(-1.0, abs=1e-15)


def test_killing_conserved_quantity_reproduces_carroll_eq_7_43_and_7_44():
    x = np.array([0.0, 7.0, np.pi / 2, 0.4])
    u = np.array([1.3, -0.2, 0.0, 0.05])
    f = 1.0 - 2.0 / 7.0
    e, l_z = stationary_axisymmetric_constants(sch(), x, u)
    assert e == pytest.approx(f * u[0], rel=1e-15, abs=0)  # Eq. 7.43
    assert l_z == pytest.approx(7.0**2 * u[3], rel=1e-15, abs=0)  # Eq. 7.44
    assert killing_conserved_quantity(sch(), x, u, np.array([1.0, 0, 0, 0])) == pytest.approx(-e, rel=1e-15, abs=0)
    assert killing_conserved_quantity(sch(), x, u, np.array([0, 0, 0, 1.0])) == pytest.approx(l_z, rel=1e-15, abs=0)


# --------------------------------------------------------------------------------------
# Killing equation
# --------------------------------------------------------------------------------------


def _killing(vector):
    v = np.asarray(vector, dtype=float)
    return lambda x: np.zeros(4, dtype=np.result_type(x, float)) + v


def test_killing_defect_kerr_and_schwarzschild():
    x = np.array([0.0, 6.0, 1.1, 0.5])
    for metric in (sch(), kerr()):
        for vector in ([1.0, 0, 0, 0], [0, 0, 0, 1.0], [1.0, 0, 0, 0.7]):
            defect = killing_equation_defect(metric, x, _killing(vector))
            assert np.array_equal(defect, np.zeros((4, 4)))  # exactly zero: g does not depend on t and phi
        # d/dr and d/dtheta are not Killing vectors: the defect is clearly non-zero (closed forms in the next test)
        for index in (1, 2):
            e = np.zeros(4)
            e[index] = 1.0
            assert np.abs(killing_equation_defect(metric, x, _killing(e))).max() > 1e-4


def test_killing_defect_along_d_r_and_d_theta_in_schwarzschild_against_hand_derivatives():
    # For a constant field V = d_r (or d_theta) the Lie derivative is L_V g_mn = d_V g_mn, the derivative of each metric
    # component, written out by hand for M = 1 at (r, theta) = (6, 1.1), f = 1 - 2/r:
    #   g_tt = -f              -> d_r g_tt = -2 / r^2
    #   g_rr = 1/f             -> d_r g_rr = -(2 / r^2) / f^2
    #   g_thth = r^2           -> d_r = 2 r
    #   g_phph = r^2 sin^2 th  -> d_r = 2 r sin^2 th,  d_th = 2 r^2 sin th cos th
    # All other components do not depend on r or theta. Measured largest difference 1.4e-17 (d_r) and 0 (d_theta);
    # tolerance 1e-12 (the helper's derivative routine is being rewritten, so the margin is wide).
    r, theta = 6.0, 1.1
    x = np.array([0.0, r, theta, 0.5])
    f = 1.0 - 2.0 / r
    d_r = np.diag([-2.0 / r**2, -(2.0 / r**2) / f**2, 2.0 * r, 2.0 * r * np.sin(theta) ** 2])
    d_theta = np.diag([0.0, 0.0, 0.0, 2.0 * r**2 * np.sin(theta) * np.cos(theta)])
    np.testing.assert_allclose(killing_equation_defect(sch(), x, _killing([0, 1.0, 0, 0])), d_r, rtol=0, atol=1e-12)
    np.testing.assert_allclose(killing_equation_defect(sch(), x, _killing([0, 0, 1.0, 0])), d_theta, rtol=0, atol=1e-12)


def test_killing_defect_flat_space_lorentz_generators_vanish_and_dilation_gives_two_eta():
    # Cartesian Minkowski: the six Lorentz generators (three rotations x^i d_j - x^j d_i, three boosts x^i d_t + t d_i)
    # and the translations are Killing (defect exactly 0, measured 0). The dilation V = x^mu d_mu has d_mu V^l =
    # delta^l_mu, so L_V eta_mn = eta_ln delta^l_m + eta_ml delta^l_n = 2 eta_mn (measured difference 0, tolerance 1e-15).
    p = np.array([0.3, 1.0, -2.0, 0.7])
    zero = lambda y: 0 * y[0]  # noqa: E731
    generators = {
        "rotation xy": lambda y: np.array([zero(y), -y[2], y[1], zero(y)]),
        "rotation xz": lambda y: np.array([zero(y), -y[3], zero(y), y[1]]),
        "rotation yz": lambda y: np.array([zero(y), zero(y), -y[3], y[2]]),
        "boost x": lambda y: np.array([y[1], y[0], zero(y), zero(y)]),
        "boost y": lambda y: np.array([y[2], zero(y), y[0], zero(y)]),
        "boost z": lambda y: np.array([y[3], zero(y), zero(y), y[0]]),
    }
    for name, generator in generators.items():
        assert np.abs(killing_equation_defect(minkowski_metric, p, generator)).max() == 0.0, name
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    dilation = killing_equation_defect(minkowski_metric, p, lambda y: y + 0)
    np.testing.assert_allclose(dilation, 2.0 * eta, rtol=0, atol=1e-15)


def test_killing_defect_spherical_rotation_in_schwarzschild_and_flat_cartesian_rotations():
    x = np.array([0.0, 6.0, 1.1, 0.5])

    def rotation_about_x(y):  # S^2 rotation generator: -sin(phi) d_theta - cot(theta) cos(phi) d_phi
        out = np.zeros(4, dtype=np.result_type(y, float))
        out[2] = -np.sin(y[3])
        out[3] = -np.cos(y[2]) / np.sin(y[2]) * np.cos(y[3])
        return out

    assert np.abs(killing_equation_defect(sch(), x, rotation_about_x)).max() < 2e-13  # measured 1.4e-14
    # not a Killing vector: the radial dilation r d_r
    assert np.abs(killing_equation_defect(sch(), x, lambda y: np.array([0, 1, 0, 0]) * y[1] + 0 * y)).max() > 1.0
    # Cartesian flat space: translations and rotations are Killing, the dilation x^mu d_mu is not (defect = 2 eta)
    p = np.array([0.3, 1.0, -2.0, 0.7])
    rot = lambda y: np.array([0 * y[0], -y[2], y[1], 0 * y[0]])  # noqa: E731
    boost = lambda y: np.array([y[1], y[0], 0 * y[0], 0 * y[0]])  # noqa: E731
    dil = lambda y: y + 0  # noqa: E731
    assert np.abs(killing_equation_defect(minkowski_metric, p, rot)).max() == 0.0
    assert np.abs(killing_equation_defect(minkowski_metric, p, boost)).max() == 0.0
    assert killing_equation_defect(minkowski_metric, p, dil) == pytest.approx(2 * np.diag([-1.0, 1, 1, 1]), abs=1e-15)


def test_killing_vectors_of_eddington_finkelstein_forms():
    x = np.array([0.0, 4.0, 1.0, 0.2])
    for metric in (
        partial(schwarzschild_eddington_finkelstein_metric, mass_geometric_m=1.0),
        partial(kerr_eddington_finkelstein_metric, mass_geometric_m=1.0, spin_a_m=0.8),
    ):
        for vector in ([1.0, 0, 0, 0], [0, 0, 0, 1.0]):  # Visser: U^a = (1,0,0,0), R^a = (0,0,0,1)
            assert np.array_equal(killing_equation_defect(metric, x, _killing(vector)), np.zeros((4, 4)))


# --------------------------------------------------------------------------------------
# integration: helpers
# --------------------------------------------------------------------------------------


def _drift(metric, xs, us):
    """Maximum deviation of E, L and the norm from their initial values along the solution."""
    e0, l0 = stationary_axisymmetric_constants(metric, xs[0], us[0])
    e_dev = l_dev = n_dev = 0.0
    for x, u in zip(xs, us, strict=True):
        e, l_z = stationary_axisymmetric_constants(metric, x, u)
        e_dev = max(e_dev, abs(e - e0))
        l_dev = max(l_dev, abs(l_z - l0))
        n_dev = max(n_dev, abs(tangent_norm_squared(metric, x, u) - tangent_norm_squared(metric, xs[0], us[0])))
    return e_dev, l_dev, n_dev


def _bound_orbit_start(r_peri: float, r_apo: float, mass: float = 1.0):
    """Specific energy and angular momentum of a Schwarzschild orbit with the given turning points.

    From Eq. 7.46 with epsilon = 1: E^2 = f(r) (1 + L^2 / r^2) at both turning points; the two equations are linear
    in E^2 and L^2.
    """
    f1, f2 = 1 - 2 * mass / r_peri, 1 - 2 * mass / r_apo
    l2 = (f2 - f1) / (f1 / r_peri**2 - f2 / r_apo**2)
    e2 = f1 * (1 + l2 / r_peri**2)
    return np.sqrt(e2), np.sqrt(l2)


def _periapsis_start(r_peri: float, r_apo: float):
    e, l_z = _bound_orbit_start(r_peri, r_apo)
    f = 1 - 2 / r_peri
    x0 = np.array([0.0, r_peri, np.pi / 2, 0.0])
    u0 = np.array([e / f, 0.0, 0.0, l_z / r_peri**2])
    return x0, u0, e, l_z


def _sign_change_roots(metric, lam, xs, us, component, direction):
    """Cubic Hermite roots of u^component along the solution; returns (lambda, cubic interpolants of x^mu)."""
    # the metric was checked for analyticity by the integration that produced xs; the check is skipped here
    acc = np.array([geodesic_acceleration(metric, x, u, check=False) for x, u in zip(xs, us, strict=True)])
    u_spline = CubicHermiteSpline(lam, us[:, component], acc[:, component])
    x_splines = [CubicHermiteSpline(lam, xs[:, i], us[:, i]) for i in range(4)]
    roots = []
    for i in range(len(lam) - 1):
        a, b = us[i, component], us[i + 1, component]
        if direction * a < 0.0 <= direction * b:
            roots.append(brentq(u_spline, lam[i], lam[i + 1], xtol=1e-12 * max(1.0, abs(lam[i])), rtol=1e-14))
    return np.array(roots), x_splines


@cache  # results are read-only here; the integrations are slow and shared between tests
def _periapsis_advance(p: float, e: float, orbits: int = 3, samples: int = 1500):
    """Advance of the periapsis per orbit, in radians, from the integrated geodesic (Schwarzschild, M = 1)."""
    r_peri, r_apo = p / (1 + e), p / (1 - e)
    x0, u0, _, _ = _periapsis_start(r_peri, r_apo)
    a = p / (1 - e**2)
    period = 2 * np.pi * a**1.5  # Newtonian radial period, enough to size the sampling
    lam = np.linspace(0.0, (orbits + 0.2) * period, int((orbits + 0.2) * samples) + 1)
    xs, us = integrate_geodesic(sch(), x0, u0, lam)
    roots, splines = _sign_change_roots(sch(), lam, xs, us, 1, +1)
    phi = np.array([splines[3](t) for t in roots])
    return np.diff(phi) - 2 * np.pi, roots


@cache  # results are read-only here; the integrations are slow and shared between tests
def _deflection(b: float, r0_over_b: float = 1e4, samples: int = 6001, e_photon: float = 1.0):
    """Deflection angle of a null geodesic with impact parameter b = L/E in Schwarzschild (M = 1)."""
    r0 = r0_over_b * b
    f0 = 1 - 2 / r0
    l_z = b * e_photon
    x0 = np.array([0.0, r0, np.pi / 2, 0.0])
    u0 = np.array([e_photon / f0, -np.sqrt(e_photon**2 - f0 * l_z**2 / r0**2), 0.0, l_z / r0**2])
    # grid dense near the closest approach; the end lies beyond 2 r0 because the Shapiro delay adds about 4 M ln(r0/b)
    s_start, s_end = -np.arcsinh(r0 / b), np.arcsinh((r0 + 0.5 * b) / b)
    lam = r0 + b * np.sinh(np.linspace(s_start, s_end, samples))
    lam[0] = 0.0
    xs, us = integrate_geodesic(sch(), x0, u0, lam, rtol=1e-13, atol=1e-15)
    imin = int(np.argmin(xs[:, 1]))
    r_spline = CubicHermiteSpline(lam, xs[:, 1] - r0, us[:, 1])
    phi_spline = CubicHermiteSpline(lam, xs[:, 3], us[:, 3])
    for i in range(imin, len(lam) - 1):
        if xs[i, 1] < r0 <= xs[i + 1, 1]:
            lam_end = brentq(r_spline, lam[i], lam[i + 1], xtol=1e-9, rtol=1e-14)
            break
    else:  # pragma: no cover
        raise AssertionError("the photon did not return to r0")
    swept = phi_spline(lam_end)
    return swept - (np.pi - 2 * np.arcsin(b / r0)), xs, us


# --------------------------------------------------------------------------------------
# integration: Schwarzschild
# --------------------------------------------------------------------------------------


def test_schwarzschild_circular_orbit_stays_circular_with_keplerian_phi_rate():
    r = 10.0
    ut = 1.0 / np.sqrt(1.0 - 3.0 / r)
    omega = np.sqrt(1.0 / r**3)
    x0 = np.array([0.0, r, np.pi / 2, 0.0])
    u0 = np.array([ut, 0.0, 0.0, omega * ut])
    lam = np.linspace(0.0, 10 * 2 * np.pi / (omega * ut), 2001)  # ten orbits
    xs, us = integrate_geodesic(sch(), x0, u0, lam)
    # Measured over ten orbits (2001 samples): |r - 10| < 3.8e-11 (tolerance 5e-10); |theta - pi/2| < 6.4e-10, which
    # is the integrator's absolute tolerance acting on a component whose exact value is constant (tolerance 1e-8);
    # relative error of phi / t against sqrt(M / r^3) 2.4e-15 (tolerance 3e-14); |u^t - u^t_0| < 1.1e-12 (1e-11);
    # drift of E, L, norm 5.6e-16, 3.1e-15, 1.6e-15 (tolerance 5e-14).
    assert np.abs(xs[:, 1] - r).max() < 5e-10
    assert np.abs(xs[:, 2] - np.pi / 2).max() < 1e-8
    assert xs[-1, 3] / xs[-1, 0] == pytest.approx(
        omega, rel=3e-14, abs=0
    )  # d phi / d t = sqrt(M / r^3), t and phi integrated
    assert np.abs(us[:, 0] - ut).max() < 1e-11
    e_dev, l_dev, n_dev = _drift(sch(), xs, us)
    assert max(e_dev, l_dev, n_dev) < 5e-14


def test_schwarzschild_eccentric_orbit_conserves_energy_angular_momentum_and_norm_over_20_orbits():
    r_peri, r_apo = 9.0, 17.0
    x0, u0, e, l_z = _periapsis_start(r_peri, r_apo)
    assert tangent_norm_squared(sch(), x0, u0) == pytest.approx(-1.0, abs=1e-14)  # the turning-point formula is right
    # The relativistic radial period of this orbit is 356.0 (measured), longer than the Newtonian 2 pi a^(3/2) = 294;
    # 7600 covers 21.3 radial periods, 19001 samples.
    lam = np.linspace(0.0, 7600.0, 19001)
    xs, us = integrate_geodesic(sch(), x0, u0, lam)
    e_dev, l_dev, n_dev = _drift(sch(), xs, us)
    # Measured over the 21 periods: drift of E 2.5e-12, of L 6.0e-11, of the norm 2.3e-12; tolerances 3e-11, 6e-10, 3e-11.
    assert e_dev < 3e-11
    assert l_dev < 6e-10
    assert n_dev < 3e-11
    # the orbit stays between its turning points (measured excursions 0 and 1.8e-11) and passes periapsis 21 times
    assert xs[:, 1].min() > r_peri - 1e-9
    assert xs[:, 1].max() < r_apo + 1e-9
    roots, _ = _sign_change_roots(sch(), lam, xs, us, 1, +1)
    assert len(roots) >= 20
    assert np.diff(roots) == pytest.approx(356.0, abs=0.01)  # constant radial period


@pytest.mark.parametrize("p", [500.0, 1000.0, 2000.0])
def test_periapsis_advance_next_order_coefficient_regression_pin(p):
    e = 0.1
    advance, _ = _periapsis_advance(p, e)
    expected = 6 * np.pi / p
    relative = advance / expected - 1.0
    # REGRESSION PIN, not an independent reference: the numbers below are this code's own output. The deviation is the
    # next post-Newtonian order, K / p. Measured K = 4.548, 4.525, 4.514 for p = 500, 1000, 2000 (the orbits of one run
    # agree to 1e-5); K - 4.50 halves with every doubling of p. Tolerance 0.002 on K. The independent check is the
    # Richardson test below (stability of the extrapolated limit).
    measured_k = {500.0: 4.548, 1000.0: 4.525, 2000.0: 4.514}[p]
    assert np.all(relative > 0.0)
    assert np.all(np.abs(relative * p - measured_k) < 0.002)


def test_periapsis_advance_deviation_scales_as_one_over_p():
    e = 0.1
    deviations = {}
    for p in (500.0, 1000.0, 2000.0):
        advance, _ = _periapsis_advance(p, e, orbits=2)
        deviations[p] = advance.mean() / (6 * np.pi / p) - 1.0
    # The deviation is proportional to 1 / p: measured ratios 2.0101 and 2.0050 (exactly 2 for a pure 1/p law; the
    # remainder is the order 1 / p^2). Tolerance 2%.
    assert deviations[500.0] / deviations[1000.0] == pytest.approx(2.0, rel=0.02, abs=0)
    assert deviations[1000.0] / deviations[2000.0] == pytest.approx(2.0, rel=0.02, abs=0)


@pytest.mark.parametrize("b", [500.0, 1000.0, 2000.0])
def test_light_deflection_next_order_coefficient_regression_pin(b):
    alpha, xs, us = _deflection(b)
    expected = 4.0 / b
    relative = alpha / expected - 1.0
    # REGRESSION PIN, not an independent reference: the numbers below are this code's own output. Next order: relative
    # deviation K M / b with K measured 2.9667, 2.9560, 2.9506 for b = 500, 1000, 2000; tolerance 0.002 on K. The
    # independent check is the Richardson test below.
    measured_k = {500.0: 2.9667, 1000.0: 2.9560, 2000.0: 2.9506}[b]
    assert relative * b == pytest.approx(measured_k, abs=0.002)
    # the null norm stays at zero along the path: measured 1.7e-13, 6.7e-14, 1.3e-12; tolerance 1e-11
    norms = [tangent_norm_squared(sch(), x, u) for x, u in zip(xs[::200], us[::200], strict=True)]
    assert np.abs(norms).max() < 1e-11


def test_light_deflection_deviation_scales_as_one_over_b():
    deviations = {b: _deflection(b, samples=4001)[0] / (4.0 / b) - 1.0 for b in (500.0, 1000.0, 2000.0)}
    # measured ratios 2.0073 and 2.0036 (a pure 1/b law gives 2); tolerance 2%
    assert deviations[500.0] / deviations[1000.0] == pytest.approx(2.0, rel=0.02, abs=0)
    assert deviations[1000.0] / deviations[2000.0] == pytest.approx(2.0, rel=0.02, abs=0)


def _richardson_pair(k_coarse, k_fine):
    """Extrapolate K(x) = K0 + K1 x (x halved between the arguments) to x -> 0: K0 = 2 K(x/2) - K(x)."""
    return 2.0 * k_fine - k_coarse


def test_periapsis_advance_coefficient_is_stable_under_richardson_extrapolation_in_m_over_p():
    # K(p) = (advance / (6 pi / p) - 1) p converges linearly in 1/p (K - K0 halves per doubling of p). Richardson with the
    # halving pairs (500, 1000) and (1000, 2000) removes the 1/p term. Measured (orbit-by-orbit values agree to 1e-5):
    # K = 4.54805, 4.52515, 4.51378; extrapolated 4.50225 and 4.50240 (mean over the orbits: the two pairs differ by 1.5e-4);
    # the raw K changes by 0.0229 and 0.0114 between the same pairs. Assertions: the two extrapolations agree within
    # 1.6e-3 (about ten times the measured difference), and the extrapolation changes at least 20 times less than the raw
    # coefficient (measured ratio 75). No literature value is asserted.
    k = {p: float(np.mean(_periapsis_advance(p, 0.1)[0] / (6 * np.pi / p) - 1.0) * p) for p in (500.0, 1000.0, 2000.0)}
    first = _richardson_pair(k[500.0], k[1000.0])
    second = _richardson_pair(k[1000.0], k[2000.0])
    assert abs(first - second) < 1.6e-3
    assert abs(first - second) < 0.05 * abs(k[1000.0] - k[2000.0])
    assert abs(k[500.0] - k[1000.0]) == pytest.approx(
        2.0 * abs(k[1000.0] - k[2000.0]), rel=0.05, abs=0
    )  # linear: measured 2.01


def test_light_deflection_coefficient_is_stable_under_richardson_extrapolation_in_m_over_b():
    # K(b) = (alpha / (4 / b) - 1) b converges linearly in 1/b. Measured K = 2.96675, 2.95595, 2.95059 for b = 500, 1000,
    # 2000; extrapolated 2.94516 and 2.94522 (differ by 6.5e-5); raw K changes by 0.0108 and 0.0054 between the pairs.
    # Assertions: the two extrapolations agree within 6.5e-4 (ten times the measured difference) and change at least 20
    # times less than the raw coefficient (measured ratio 83). No literature value is asserted.
    k = {b: float((_deflection(b)[0] / (4.0 / b) - 1.0) * b) for b in (500.0, 1000.0, 2000.0)}
    first = _richardson_pair(k[500.0], k[1000.0])
    second = _richardson_pair(k[1000.0], k[2000.0])
    assert abs(first - second) < 6.5e-4
    assert abs(first - second) < 0.05 * abs(k[1000.0] - k[2000.0])
    assert abs(k[500.0] - k[1000.0]) == pytest.approx(
        2.0 * abs(k[1000.0] - k[2000.0]), rel=0.05, abs=0
    )  # linear: measured 2.01


def test_photon_circular_orbit_at_3m_is_unstable_with_the_closed_form_growth_rate():
    # Eq. 7.51: r_c = 3M. From Eq. 7.46-7.48 with epsilon = 0: (dr/dl)^2 = E^2 - f L^2 / r^2 has its maximum at 3M
    # with L/E = 3 sqrt(3) M. Linearising d2(dr)/dl2 = -V''/2 dr gives the growth rate kappa = E / (sqrt(3) M) per
    # affine parameter, i.e. 1 / (3 sqrt(3) M) per unit coordinate time (dt/dl = E / f = 3 E).
    r = 3.0
    b_c = 3.0 * np.sqrt(3.0)
    f = 1.0 - 2.0 / r
    x0 = np.array([0.0, r, np.pi / 2, 0.0])
    u0 = np.array([1.0 / f, 0.0, 0.0, b_c / r**2])
    assert abs(tangent_norm_squared(sch(), x0, u0)) < 1e-15
    assert np.abs(geodesic_acceleration(sch(), x0, u0)).max() < 1e-15  # an exact circular orbit
    kappa = 1.0 / np.sqrt(3.0)
    delta = 1e-9
    # perturb the radius with the energy kept: u^r follows from the null condition only to second order, so put the
    # perturbation in r and renormalise u^t
    x1 = x0.copy()
    x1[1] = r + delta
    u1 = normalize_tangent(sch(), x1, np.array([0.0, 0.0, b_c / r**2]), kind="null")
    lam = np.linspace(0.0, 14.0 / kappa, 561)
    xs, us = integrate_geodesic(sch(), x1, u1, lam)
    dev = xs[:, 1] - r
    # the unperturbed exact orbit stays at r = 3 to rounding over the same range
    xs0, _ = integrate_geodesic(sch(), x0, u0, lam)
    assert np.abs(xs0[:, 1] - r).max() < 1e-9  # measured 1.1e-10: the rounding of the start grown by cosh(14)
    # growth: the perturbation grows by orders of magnitude (unstable), monotonically after the transient
    assert np.abs(dev[-1]) > 1e3 * delta
    # linear regime (20 delta < |dev| < 1e-3): the solution of d2(dev)/dl2 = kappa^2 dev with dev(0) = delta,
    # d(dev)/dl(0) = 0 is delta cosh(kappa l); the log slope is kappa. Measured: slope / kappa - 1 = -2.7e-5
    # (tolerance 3e-4), dev / (delta cosh(kappa l)) - 1 within 2e-4 (tolerance 2e-3).
    mask = (np.abs(dev) > 20 * delta) & (np.abs(dev) < 1e-3)
    slope = np.polyfit(lam[mask], np.log(np.abs(dev[mask])), 1)[0]
    assert slope == pytest.approx(kappa, rel=3e-4, abs=0)
    assert dev[mask] / (delta * np.cosh(kappa * lam[mask])) == pytest.approx(1.0, abs=2e-3)


def test_massive_circular_orbit_at_10m_is_stable_with_bounded_oscillation():
    r = 10.0
    ut = 1.0 / np.sqrt(1.0 - 3.0 / r)
    x0 = np.array([0.0, r + 1e-6, np.pi / 2, 0.0])
    u0 = normalize_tangent(sch(), x0, np.array([0.0, 0.0, np.sqrt(1.0 / r**3) * ut]))
    lam = np.linspace(0.0, 5000.0, 2001)
    xs, _ = integrate_geodesic(sch(), x0, u0, lam)
    dev = np.abs(xs[:, 1] - r)
    # Measured maximum 1.3e-5 (the start is circular for r, displaced by 1e-6, so the epicycle is of that order);
    # tolerance 1.3e-4. Bounded oscillation: the second half of the run does not exceed the first half.
    assert dev.max() < 1.3e-4
    assert dev[1000:].max() <= 1.001 * dev[:1000].max()


def test_radial_infall_proper_time_to_the_horizon_matches_the_cycloid_in_ingoing_eddington_finkelstein_coordinates():
    # Rest at r0: E^2 = 1 - 2M/r0, so (dr/dtau)^2 = 2M (1/r - 1/r0) (Eq. 7.46 with L = 0, epsilon = 1). Independent
    # oracle: scipy quad of dtau = -dr / sqrt(2M (1/r - 1/r0)); closed form: cycloid r = (r0/2)(1 + cos eta),
    # tau = r0^(3/2) (eta + sin eta) / (2 sqrt(2M)).
    r0 = 20.0

    def tau_of_r(r_end):
        value, _ = quad(lambda r: 1.0 / np.sqrt(2.0 * (1.0 / r - 1.0 / r0)), r_end, r0, limit=200, epsabs=1e-13)
        return value

    def tau_cycloid(r_end):
        eta = np.arccos(2.0 * r_end / r0 - 1.0)
        return r0**1.5 * (eta + np.sin(eta)) / (2.0 * np.sqrt(2.0))

    for r_end in (2.0, 1.0, 0.5):
        assert tau_cycloid(r_end) == pytest.approx(
            tau_of_r(r_end), rel=2e-11, abs=0
        )  # measured 1.0e-12, 1.6e-12, 3.6e-13
    metric = partial(schwarzschild_eddington_finkelstein_metric, mass_geometric_m=1.0)
    x0 = np.array([0.0, r0, 0.5, 0.0])
    u0 = np.array([1.0 / np.sqrt(1.0 - 2.0 / r0), 0.0, 0.0, 0.0])  # u~ = t + r*: at rest du~ = dt
    assert tangent_norm_squared(metric, x0, u0) == pytest.approx(-1.0, abs=1e-14)
    tau_horizon = tau_cycloid(2.0)
    tau_end = tau_cycloid(0.5)
    lam = np.linspace(0.0, tau_end, 1501)
    xs, us = integrate_geodesic(metric, x0, u0, lam)
    # r(tau) agrees with the closed form at every sample: invert the cycloid by root finding
    worst = 0.0
    for tau, x in zip(lam[::25], xs[::25], strict=True):
        eta = brentq(lambda h, tau=tau: r0**1.5 * (h + np.sin(h)) / (2.0 * np.sqrt(2.0)) - tau, 0.0, np.pi, xtol=1e-14)
        worst = max(worst, abs(x[1] - 0.5 * r0 * (1.0 + np.cos(eta))))
    assert worst < 2e-9  # measured 2.0e-10
    # the horizon is crossed at the closed-form proper time, with finite tangent vector
    r_spline = CubicHermiteSpline(lam, xs[:, 1] - 2.0, us[:, 1])
    tau_cross = brentq(r_spline, lam[0], lam[-1], xtol=1e-12)
    assert tau_cross == pytest.approx(tau_horizon, rel=1.5e-9, abs=0)  # measured 1.3e-10; tau_horizon = 97.970
    assert np.all(np.isfinite(us))
    assert np.abs(us).max() < 2.0  # measured 1.97: smooth through the horizon


def test_spatial_geodesics_of_flat_space_in_spherical_coordinates_are_straight_lines():
    # straight line in Cartesian coordinates X(l) = X0 + V l, converted to (r, theta, phi); u from the same line
    big_x0 = np.array([3.0, -2.0, 1.5])
    big_v = np.array([0.4, 0.9, -0.3])

    def spherical(vec):
        r = np.linalg.norm(vec)
        return np.array([r, np.arccos(vec[2] / r), np.arctan2(vec[1], vec[0])])

    def spherical_rate(vec, rate):
        r, th, _ = spherical(vec)
        rho = np.hypot(vec[0], vec[1])
        dr = vec @ rate / r
        dth = (vec[2] * (vec[0] * rate[0] + vec[1] * rate[1]) / rho - rho * rate[2]) / r**2
        dph = (vec[0] * rate[1] - vec[1] * rate[0]) / rho**2
        return np.array([dr, dth, dph])

    x0 = np.concatenate([[0.0], spherical(big_x0)])
    u0 = np.concatenate([[1.0], spherical_rate(big_x0, big_v)])
    lam = np.linspace(0.0, 4.0, 81)
    xs, _ = integrate_geodesic(flat_spherical, x0, u0, lam)
    expected = np.array([np.concatenate([[t], spherical(big_x0 + big_v * t)]) for t in lam])
    worst = np.abs(xs - expected).max()
    assert worst < 2e-10  # measured 1.7e-11 (four proper-time units, r from 3.9 to 6)
    # Cartesian Minkowski: exact straight line
    xs_c, _ = integrate_geodesic(minkowski_metric, np.zeros(4), np.array([1.0, 0.2, -0.1, 0.4]), lam)
    assert xs_c == pytest.approx(np.outer(lam, [1.0, 0.2, -0.1, 0.4]), abs=3e-14)  # measured 2.2e-15


def test_integrate_geodesic_input_checks():
    x0 = np.array([0.0, 10.0, np.pi / 2, 0.0])
    u0 = normalize_tangent(sch(), x0, np.zeros(3))
    with pytest.raises(ValueError):
        integrate_geodesic(sch(), x0, u0, [0.0])
    with pytest.raises(ValueError):
        integrate_geodesic(sch(), x0, u0, [0.0, 2.0, 1.0])
    with pytest.raises(ValueError):
        integrate_geodesic(sch(), x0[:3], u0, [0.0, 1.0])
    xs, us = integrate_geodesic(sch(), x0, u0, [0.0, 3.0, 5.0])
    assert xs.shape == (3, 4) and us.shape == (3, 4)
    assert xs[0] == pytest.approx(x0, abs=0.0)


# --------------------------------------------------------------------------------------
# integration: Kerr
# --------------------------------------------------------------------------------------


def _kerr_circular_start(r: float, spin: float, prograde: bool):
    """Equatorial circular orbit with the angular velocity of Abramowicz's Keplerian formula (M = 1)."""
    # Omega^2 = M / (r^(3/2) + a M^(1/2))^2 (Eq. keplerian-velocity-kerr); a < 0 stands for the counter-rotating orbit
    a_signed = spin if prograde else -spin
    omega = 1.0 / (r**1.5 + a_signed)
    omega = omega if prograde else -omega
    g = kerr_metric(np.array([0.0, r, np.pi / 2, 0.0]), 1.0, spin)
    norm = -(g[0, 0] + 2 * g[0, 3] * omega + g[3, 3] * omega**2)
    ut = 1.0 / np.sqrt(norm)
    return np.array([0.0, r, np.pi / 2, 0.0]), np.array([ut, 0.0, 0.0, omega * ut]), omega


@pytest.mark.parametrize(
    ("r", "prograde"),
    [(6.0, True), (10.0, True), (10.0, False), (12.0, False)],
    ids=["6-pro", "10-pro", "10-retro", "12-retro"],
)
def test_kerr_equatorial_circular_orbits_stay_circular_with_the_keplerian_angular_velocity(r, prograde):
    x0, u0, omega = _kerr_circular_start(r, 0.9, prograde)
    metric = kerr(spin=0.9)
    assert abs(tangent_norm_squared(metric, x0, u0) + 1.0) < 5e-15  # measured 4.4e-16
    assert abs(geodesic_acceleration(metric, x0, u0)[1]) < 1e-16  # no radial acceleration: measured < 5.2e-18
    lam = np.linspace(0.0, 5 * 2 * np.pi / abs(omega * u0[0]), 1501)
    xs, us = integrate_geodesic(metric, x0, u0, lam)
    # Measured over five orbits, 1501 samples: |r - r0| 1.9e-11, 9.9e-11, 3.0e-12, 1.3e-11 for the four cases
    # (tolerance 1e-9); relative error of phi / t against Omega at most 3.1e-14 (tolerance 3e-13); drift of E, L and the
    # norm at most 4.4e-15 (tolerance 5e-14). r = 6 retrograde is inside the retrograde marginally stable orbit and is
    # not used: there the rounding of the start grows (measured |r - 6| = 8.9e-6 after five orbits).
    assert np.abs(xs[:, 1] - r).max() < 1e-9
    assert xs[-1, 3] / xs[-1, 0] == pytest.approx(omega, rel=3e-13, abs=0)
    e_dev, l_dev, n_dev = _drift(metric, xs, us)
    assert max(e_dev, l_dev, n_dev) < 5e-14


def test_kerr_prograde_orbit_has_a_smaller_angular_velocity_magnitude_than_retrograde():
    _, _, omega_pro = _kerr_circular_start(8.0, 0.9, True)
    _, _, omega_retro = _kerr_circular_start(8.0, 0.9, False)
    assert omega_pro > 0.0 > omega_retro
    assert abs(omega_pro) < abs(omega_retro)  # |Omega| = 1/(r^1.5 + a) for prograde against 1/(r^1.5 - a)
    _, _, omega_zero = _kerr_circular_start(8.0, 0.0, True)
    assert omega_zero == pytest.approx(8.0**-1.5, rel=1e-15, abs=0)


def test_kerr_inclined_orbit_conserves_e_lz_the_norm_and_carrolls_killing_tensor():
    m, a = 1.0, 0.9
    metric = kerr(m, a)
    x0 = np.array([0.0, 12.0, 1.0, 0.0])
    u0 = normalize_tangent(metric, x0, np.array([-0.02, 0.012, 0.025]))
    # bound? E < 1 check; integrate over several radial periods
    e0, l0 = stationary_axisymmetric_constants(metric, x0, u0)
    assert e0 < 1.0
    lam = np.linspace(0.0, 3000.0, 3001)
    xs, us = integrate_geodesic(metric, x0, u0, lam)
    e_dev, l_dev, n_dev = _drift(metric, xs, us)
    # measured over 3001 samples (about 8 radial oscillations): E 1.6e-12, L 1.9e-10, norm 4.6e-12;
    # tolerances 2e-11, 2e-9, 5e-11
    assert e_dev < 2e-11
    assert l_dev < 2e-9
    assert n_dev < 5e-11
    assert xs[:, 2].min() < 0.85 and xs[:, 2].max() > 2.3  # the orbit really moves in theta: 0.81 to 2.33 measured

    def killing_tensor_value(x, u):
        r, th = x[1], x[2]
        delta = r**2 - 2 * m * r + a**2
        rho2 = r**2 + a**2 * np.cos(th) ** 2
        lvec = np.array([r**2 + a**2, delta, 0.0, a]) / delta  # Eq. 7.122
        n = np.array([r**2 + a**2, -delta, 0.0, a]) / (2 * rho2)
        g = metric(x)
        # xi_mn u^m u^n = 2 rho^2 (l.u)(n.u) + r^2 g(u, u)   (Eq. 7.121 with the symmetrised product)
        return 2 * rho2 * (lvec @ g @ u) * (n @ g @ u) + r**2 * (u @ g @ u)

    k0 = killing_tensor_value(xs[0], us[0])
    k_dev = max(abs(killing_tensor_value(x, u) - k0) for x, u in zip(xs, us, strict=True))
    # Eq. 7.120: the value is constant. K0 = 8.005; measured maximum deviation 4.9e-10 (6.1e-11 relative); tolerance 6e-10 relative.
    assert k_dev < 6e-10 * abs(k0)


def test_zero_angular_momentum_particle_in_kerr_has_the_frame_dragging_angular_velocity():
    # u_phi = 0 means d phi / d t = -g_tphi / g_phiphi (Abramowicz, text before Eq. ZAMO and Eq. dragg-velocity-kerr)
    m, a = 1.0, 0.9
    metric = kerr(m, a)
    r0, th0 = 6.0, np.pi / 2
    x0 = np.array([0.0, r0, th0, 0.0])
    g = metric(x0)
    omega = -g[0, 3] / g[3, 3]
    ut = 1.0 / np.sqrt(-(g[0, 0] + 2 * g[0, 3] * omega + g[3, 3] * omega**2))
    u0 = np.array([ut, 0.0, 0.0, omega * ut])
    e0, l0 = stationary_axisymmetric_constants(metric, x0, u0)
    assert abs(l0) < 1e-14
    lam = np.linspace(0.0, 12.0, 241)
    xs, us = integrate_geodesic(metric, x0, u0, lam)  # the particle falls in: r decreases
    assert xs[-1, 1] < r0 - 0.5
    worst = 0.0
    for x, u in zip(xs, us, strict=True):
        gx = metric(x)
        worst = max(worst, abs(u[3] / u[0] + gx[0, 3] / gx[3, 3]))
        assert abs(stationary_axisymmetric_constants(metric, x, u)[1]) < 2e-10  # measured 2.1e-11
    assert worst < 1e-11  # measured 8.0e-13; r falls from 6 to 3.68 in the 12 units of proper time


# --------------------------------------------------------------------------------------
# mutation-testing survivors: backward integration, a vanishing time component, the default atol
# --------------------------------------------------------------------------------------


def test_integrate_geodesic_backwards_retraces_the_forward_orbit():
    """The geodesic equation is autonomous in the affine parameter: starting from the END state of a forward run and
    integrating with a strictly DEcreasing affine parameter must return to the start. Eccentric Schwarzschild orbit
    (periapsis 7 M, apoapsis 15 M), 3 radial periods."""
    x0, u0, _, _ = _periapsis_start(7.0, 15.0)
    lam = np.linspace(0.0, 400.0, 161)
    xs, us = integrate_geodesic(sch(), x0, u0, lam)
    xb, ub = integrate_geodesic(sch(), xs[-1], us[-1], lam[::-1])
    assert xb.shape == xs.shape and np.array_equal(xb[0], xs[-1])
    # measured: 3.7e-10 (coordinates), 9.2e-12 (tangent), 2.9e-10 (return to the start); tolerances 2e-9, 5e-11, 2e-9
    assert np.abs(xb[::-1] - xs).max() < 2e-9
    assert np.abs(ub[::-1] - us).max() < 5e-11
    assert np.abs(xb[-1] - x0).max() < 2e-9
    assert xs[:, 1].min() == pytest.approx(7.0, rel=1e-12, abs=0) and xs[:, 1].max() > 14.99  # r spans 7 to 15


def test_normalize_tangent_refuses_a_zero_time_component_when_g_tt_vanishes():
    """Null coordinates: g = [[0, 1], [1, 0]] in the (t, x) block, so g_tt = 0, g_tx = 1, g_xx = 0. For the spatial
    part (1, 0, 0) the null condition 2 g_tx u^t u^x = 0 gives u^t = 0: no future-directed (u^t > 0) vector exists."""

    def null_coordinates(x):
        g = np.eye(4, dtype=np.result_type(x, float))
        g[0, 0], g[1, 1] = 0.0, 0.0
        g[0, 1] = g[1, 0] = 1.0
        return g

    with pytest.raises(ValueError, match="future-directed time component does not exist"):
        normalize_tangent(null_coordinates, np.zeros(4), [1.0, 0.0, 0.0], kind="null")
    # spatial (1, 1, 0): b = 1 and c = 1 give u^t = (0 - 1) / 2 = -1/2 < 0: also refused
    with pytest.raises(ValueError, match="future-directed time component does not exist"):
        normalize_tangent(null_coordinates, np.zeros(4), [1.0, 1.0, 0.0], kind="null")
    # a timelike vector with spatial part (-1, 0, 0): b = -1, c = 0, kappa = -1 give u^t = (-1 - 0) / (2 * -1) = 1/2 > 0
    u = normalize_tangent(null_coordinates, np.zeros(4), [-1.0, 0.0, 0.0], kind="timelike")
    assert np.array_equal(u, [0.5, -1.0, 0.0, 0.0])


def _y_dependent_metric(x):
    """g = diag(-1, 1 + 2y + y^2, 1, 1): independent of t and x, so p_x = g_xx u^x is conserved. Complex safe."""
    g = np.zeros((4, 4), dtype=np.result_type(x, float))
    g[0, 0] = -1.0
    g[1, 1] = 1.0 + 2.0 * x[2] + x[2] ** 2
    g[2, 2] = 1.0
    g[3, 3] = 1.0
    return g


def test_integrate_geodesic_default_atol_keeps_the_small_components_accurate():
    """All components of this orbit are small (positions and spatial velocities of order 0.1 to 3), so the absolute
    tolerance of the integrator controls the step: with the default atol = 1e-13 the drift of the normalisation is
    1.0e-14 and that of p_x is 4.1e-12 over 30 affine units; with atol = 1e-11 they are 2.8e-13 and 2.4e-11 (measured
    in this environment). The bounds 1e-13 and 1e-11 lie between the two sets."""
    x0 = np.zeros(4)
    u0 = normalize_tangent(_y_dependent_metric, x0, [0.1, 0.2, 0.0])
    lam = np.linspace(0.0, 30.0, 121)
    xs, us = integrate_geodesic(_y_dependent_metric, x0, u0, lam)  # default tolerances
    norm_drift = max(abs(tangent_norm_squared(_y_dependent_metric, x, u) + 1.0) for x, u in zip(xs, us, strict=True))
    px = np.array([_y_dependent_metric(x)[1, 1] * u[1] for x, u in zip(xs, us, strict=True)])
    print(f"MEASURED default atol: norm drift {norm_drift:.3e}, p_x drift {np.abs(px - px[0]).max():.3e}")
    assert norm_drift < 1e-13
    assert np.abs(px - px[0]).max() < 1e-11
    # the same call with a looser atol is visibly worse, which is what the bounds above separate
    xl, ul = integrate_geodesic(_y_dependent_metric, x0, u0, lam, atol=1e-11)
    loose = max(abs(tangent_norm_squared(_y_dependent_metric, x, u) + 1.0) for x, u in zip(xl, ul, strict=True))
    assert loose > 1e-13
