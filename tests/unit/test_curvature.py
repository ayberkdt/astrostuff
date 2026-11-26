"""Tests of kozmos.relativity.curvature.

Reference values: closed forms printed in the sources (Visser's Kerr quadratic invariant, which contains the
Schwarzschild value 48 M^2 / r^6 as a -> 0; Carroll's Ricci tensor and scalar of the Robertson-Walker metric,
Eq. 8.13 and 8.14; Carroll's Coulomb field and electromagnetic stress tensor, Eq. 7.107 and 7.112), symmetries of
the Riemann tensor (Eq. 3.78-3.81), the contracted Bianchi identity, and the Friedmann equations (Eq. 8.35, 8.36).

Numerical accuracy of the curvature code: second derivatives of the metric come from a four-point difference of
exact complex-step first derivatives, with step h = 1e-3 max(1, |x_i|). Every tolerance is about ten times the
maximum measured here and the measured value stands next to it.

Note on Carroll's Eq. 7.30: the TeX source prints R^{mnrs} R_{mnrs} = 12 G^2 M^2 / r^6 for Schwarzschild. The value
48 M^2 / r^6 (Visser's formula at a = 0, and what this code computes) is used; the factor 4 is a misprint of the
source.
"""

from __future__ import annotations

import warnings
from functools import partial

import numpy as np
import pytest

from kozmos.core._differentiation import jacobian_with_error
from kozmos.core.constants import GRAVITATIONAL_CONSTANT, MEGAPARSEC, SPEED_OF_LIGHT
from kozmos.cosmology import Cosmology
from kozmos.cosmology.dark_energy import critical_density_kg_m3, hubble_parameter_km_s_mpc
from kozmos.relativity import curvature
from kozmos.relativity.curvature import (
    einstein_tensor,
    kretschmann_scalar,
    ricci_scalar,
    ricci_tensor,
    riemann_tensor,
)
from kozmos.relativity.geodesics import christoffel_symbols
from kozmos.relativity.metrics import (
    flrw_metric,
    kerr_eddington_finkelstein_metric,
    kerr_metric,
    minkowski_metric,
    reissner_nordstrom_metric,
    schwarzschild_eddington_finkelstein_metric,
    schwarzschild_metric,
)

SCH = partial(schwarzschild_metric, mass_geometric_m=1.0)
SCH_EF = partial(schwarzschild_eddington_finkelstein_metric, mass_geometric_m=1.0)


def kerr(spin: float):
    return partial(kerr_metric, mass_geometric_m=1.0, spin_a_m=spin)


def kerr_ef(spin: float):
    return partial(kerr_eddington_finkelstein_metric, mass_geometric_m=1.0, spin_a_m=spin)


def kretschmann_visser(r: float, theta: float, spin: float, mass: float = 1.0) -> float:
    """Visser's quadratic curvature invariant of the Kerr spacetime (section 'Kerr's original coordinates')."""
    c2 = np.cos(theta) ** 2
    rho2 = r**2 + spin**2 * c2
    return 48 * mass**2 * (r**2 - spin**2 * c2) * (rho2**2 - 16 * r**2 * spin**2 * c2) / rho2**6


def flat_spherical(x):
    g = np.zeros((4, 4), dtype=np.result_type(x, float))
    g[0, 0] = -1.0
    g[1, 1] = 1.0
    g[2, 2] = x[1] ** 2
    g[3, 3] = x[1] ** 2 * np.sin(x[2]) ** 2
    return g


# --------------------------------------------------------------------------------------
# flat space
# --------------------------------------------------------------------------------------


def test_minkowski_cartesian_curvature_is_exactly_zero():
    x = np.array([0.3, 1.0, -2.0, 4.0])
    assert np.array_equal(riemann_tensor(minkowski_metric, x), np.zeros((4, 4, 4, 4)))
    assert np.array_equal(ricci_tensor(minkowski_metric, x), np.zeros((4, 4)))
    assert ricci_scalar(minkowski_metric, x) == 0.0
    assert kretschmann_scalar(minkowski_metric, x) == 0.0


def test_minkowski_in_spherical_coordinates_has_zero_curvature_to_the_numerical_accuracy():
    # nonzero Christoffel symbols cancel in the Riemann tensor. Measured at (r, theta) = (5, 1): Riemann 4.1e-12, Ricci
    # 4.1e-12, Ricci scalar 5.0e-13, Einstein 4.2e-12, Kretschmann 1.6e-25; tolerances 5e-11, 5e-11, 5e-12, 5e-11, 1e-23.
    x = np.array([0.0, 5.0, 1.0, 0.3])
    assert np.abs(riemann_tensor(flat_spherical, x)).max() < 5e-11
    assert np.abs(ricci_tensor(flat_spherical, x)).max() < 5e-11
    assert abs(ricci_scalar(flat_spherical, x)) < 5e-12
    assert np.abs(einstein_tensor(flat_spherical, x)).max() < 5e-11
    assert abs(kretschmann_scalar(flat_spherical, x)) < 1e-23


# --------------------------------------------------------------------------------------
# component-level check of the Riemann tensor: (2-sphere of radius R) x (flat plane)
# --------------------------------------------------------------------------------------

SPHERE_RADIUS = 1.7


def sphere_times_plane(x):
    """diag(-1, R^2, R^2 sin^2 theta, 1) in coordinates (t, theta, phi, z)."""
    g = np.zeros((4, 4), dtype=np.result_type(x, float))
    g[0, 0] = -1.0
    g[1, 1] = SPHERE_RADIUS**2
    g[2, 2] = SPHERE_RADIUS**2 * np.sin(x[1]) ** 2
    g[3, 3] = 1.0
    return g


@pytest.mark.parametrize("theta", [0.6, 1.0, 2.2])
def test_riemann_components_of_a_two_sphere_times_a_flat_plane(theta):
    """Hand derivation (coordinates 0 = t, 1 = theta, 2 = phi, 3 = z; array index [rho, sigma, mu, nu]).

    Non-zero Christoffel symbols: Gamma^theta_{phi phi} = -sin(theta) cos(theta), Gamma^phi_{theta phi} =
    cot(theta) (R drops out because g_{theta theta} and g_{phi phi} both scale with R^2; t and z are flat).
    R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma} + Gamma^rho_{mu l} Gamma^l_{nu sigma}
    - Gamma^rho_{nu l} Gamma^l_{mu sigma} (Eq. 3.67). Only d_theta of a Christoffel symbol is non-zero:
      R^theta_{phi theta phi} = d_theta Gamma^theta_{phi phi} - Gamma^theta_{phi phi} Gamma^phi_{theta phi}
        = -(cos^2 - sin^2) + sin cos cot = sin^2 theta,
      R^phi_{theta theta phi} = d_theta Gamma^phi_{phi theta} + Gamma^phi_{theta phi} Gamma^phi_{phi theta}
        = -1 / sin^2 + cot^2 = -1.
    With the antisymmetry in the last pair the complete list of non-zero components is
      [1, 2, 1, 2] = +sin^2 theta, [1, 2, 2, 1] = -sin^2 theta, [2, 1, 1, 2] = -1, [2, 1, 2, 1] = +1.
    Ricci R_{sigma nu} = R^rho_{sigma rho nu}: R_theta_theta = R^phi_{theta phi theta} = +1, R_phi_phi =
    R^theta_{phi theta phi} = sin^2 theta, all others 0 (that is diag(0, 1, sin^2 theta, 0)); scalar =
    g^{theta theta} + g^{phi phi} sin^2 theta = 2 / R^2; Kretschmann (2-sphere) = 4 / R^4.
    Measured at theta = 0.6, 1.0, 2.2: largest component error 8.5e-11, 4.2e-12, 6.9e-13; relative error of the
    scalar 4.2e-11, 2.3e-12, 2.1e-13 and of Kretschmann 8.5e-11, 4.5e-12, 4.1e-13; tolerance 1e-9 for all.
    """
    x = np.array([0.0, theta, 0.3, 0.0])
    s2 = np.sin(theta) ** 2
    expected = np.zeros((4, 4, 4, 4))
    expected[1, 2, 1, 2] = s2
    expected[1, 2, 2, 1] = -s2
    expected[2, 1, 1, 2] = -1.0
    expected[2, 1, 2, 1] = 1.0
    riemann = riemann_tensor(sphere_times_plane, x)
    assert riemann[1, 2, 1, 2] == pytest.approx(s2, abs=1e-9)
    assert riemann[2, 1, 1, 2] == pytest.approx(-1.0, abs=1e-9)
    assert np.abs(riemann - expected).max() < 1e-9  # every other component vanishes to the same level
    zero_mask = expected == 0.0
    assert np.abs(riemann[zero_mask]).max() < 1e-9
    assert np.abs(ricci_tensor(sphere_times_plane, x) - np.diag([0.0, 1.0, s2, 0.0])).max() < 1e-9
    assert ricci_scalar(sphere_times_plane, x) == pytest.approx(2.0 / SPHERE_RADIUS**2, rel=1e-9, abs=0)
    assert kretschmann_scalar(sphere_times_plane, x) == pytest.approx(4.0 / SPHERE_RADIUS**4, rel=1e-9, abs=0)


# --------------------------------------------------------------------------------------
# Kerr in Kerr's original coordinates: components against Visser's Eq. E:K1
# --------------------------------------------------------------------------------------


def visser_k1_line_element(x, dx, mass, spin):
    """Eq. E:K1 typed from the TeX source (Visser, 'Kerr's original coordinates'), coordinates (u, r, theta, phi):

    ds^2 = -[1 - 2 m r / (r^2 + a^2 cos^2 theta)] (du + a sin^2 theta dphi)^2
           + 2 (du + a sin^2 theta dphi)(dr + a sin^2 theta dphi) + (r^2 + a^2 cos^2 theta)(dtheta^2 + sin^2 theta dphi^2)
    """
    _, r, theta, _ = x
    du, dr, dtheta, dphi = dx
    s2 = np.sin(theta) ** 2
    rho2 = r**2 + spin**2 * np.cos(theta) ** 2
    first = du + spin * s2 * dphi
    second = dr + spin * s2 * dphi
    return -(1.0 - 2.0 * mass * r / rho2) * first**2 + 2.0 * first * second + rho2 * (dtheta**2 + s2 * dphi**2)


def test_kerr_eddington_finkelstein_metric_reproduces_the_line_element_of_eq_e_k1():
    # Quadratic form dx^T g dx against the typed line element at random points and random displacements (including
    # points inside the outer horizon). Measured worst |difference| / max(|quad|, |ref|, 1e-3) over 400 samples:
    # 2.2e-15; tolerance 1e-12.
    rng = np.random.default_rng(20261126)
    worst = 0.0
    for _ in range(400):
        mass, spin = rng.uniform(0.5, 2.0), rng.uniform(-0.95, 0.95) * 1.0
        x = np.array([rng.normal(), rng.uniform(0.3, 12.0), rng.uniform(0.1, np.pi - 0.1), rng.uniform(0.0, 6.0)])
        dx = rng.normal(size=4)
        quad = dx @ kerr_eddington_finkelstein_metric(x, mass_geometric_m=mass, spin_a_m=spin) @ dx
        ref = visser_k1_line_element(x, dx, mass, spin)
        scale = max(abs(quad), abs(ref), 1e-3)
        worst = max(worst, abs(quad - ref) / scale)
    assert worst < 1e-12


def test_kerr_eddington_finkelstein_azimuth_runs_opposite_to_boyer_lindquist():
    # From Eq. E:K1: g_{u phi} = a sin^2 theta [1 - (1 - 2 m r / rho^2)] = 2 m a r sin^2 theta / rho^2 > 0 for a > 0.
    # Boyer-Lindquist g_{t phi} = -2 m a r sin^2 theta / rho^2. The source's transformation (Eq. E:BL-change-2,
    # phi = -phi_BL - a int dr / Delta, t = t_BL + 2 m int r dr / Delta, u = t + r) leaves r, theta unchanged and flips
    # the sign of the azimuth, hence g_{u phi}(EF) = -g_{t phi}(BL) at the same (r, theta).
    mass, spin = 1.0, 0.7
    for r, theta in [(3.0, np.pi / 2), (5.5, np.pi / 2), (4.0, 1.1), (9.0, 2.0)]:
        ef = kerr_eddington_finkelstein_metric(np.array([0.0, r, theta, 0.0]), mass_geometric_m=mass, spin_a_m=spin)
        bl = kerr_metric(np.array([0.0, r, theta, 0.0]), mass_geometric_m=mass, spin_a_m=spin)
        rho2 = r**2 + spin**2 * np.cos(theta) ** 2
        closed = 2.0 * mass * spin * r * np.sin(theta) ** 2 / rho2
        assert ef[0, 3] > 0.0
        assert ef[0, 3] == pytest.approx(closed, rel=1e-14, abs=0)  # measured ~1e-16
        assert ef[0, 3] == pytest.approx(-bl[0, 3], rel=1e-14, abs=0)  # measured 1.2e-16 (absolute 5.6e-17)
        assert ef[3, 0] == ef[0, 3]


# --------------------------------------------------------------------------------------
# Schwarzschild
# --------------------------------------------------------------------------------------

# (r, theta, measured Ricci residual / sqrt(K)): residual in the largest Ricci component divided by sqrt(48 / r^6)
SCHWARZSCHILD_RICCI = [(4.0, 1.0, 7.7e-11), (10.0, 1.0, 5.8e-10), (30.0, 2.0, 1.2e-7), (3.0, 0.4, 3.8e-9)]


@pytest.mark.parametrize(("r", "theta", "measured"), SCHWARZSCHILD_RICCI)
def test_schwarzschild_is_ricci_flat_relative_to_the_curvature_scale(r, theta, measured):
    x = np.array([0.0, r, theta, 0.3])
    scale = np.sqrt(48.0 / r**6)
    assert np.abs(ricci_tensor(SCH, x)).max() < 10 * measured * scale
    assert abs(ricci_scalar(SCH, x)) < 10 * measured * scale
    assert np.abs(einstein_tensor(SCH, x)).max() < 10 * measured * scale


# measured relative error of the Kretschmann scalar against 48 M^2 / r^6: -3.8e-11, -1.1e-11, -1.7e-10, -7.1e-10
@pytest.mark.parametrize(
    ("r", "theta", "measured"), [(4.0, 1.0, 3.8e-11), (10.0, 1.0, 1.1e-11), (30.0, 2.0, 1.7e-10), (3.0, 0.4, 7.1e-10)]
)
def test_schwarzschild_kretschmann_scalar_is_48_m_squared_over_r_to_the_sixth(r, theta, measured):
    x = np.array([0.0, r, theta, 0.3])
    assert kretschmann_scalar(SCH, x) == pytest.approx(48.0 / r**6, rel=10 * measured, abs=0)
    # not the 12 M^2 / r^6 printed in Eq. 7.30 of the TeX source
    assert kretschmann_scalar(SCH, x) / (12.0 / r**6) == pytest.approx(4.0, rel=1e-8, abs=0)


def test_schwarzschild_curvature_scales_with_the_mass():
    # K = 48 M^2 / r^6: doubling M with r fixed multiplies K by 4
    x = np.array([0.0, 12.0, 1.0, 0.0])
    k1 = kretschmann_scalar(partial(schwarzschild_metric, mass_geometric_m=1.0), x)
    k2 = kretschmann_scalar(partial(schwarzschild_metric, mass_geometric_m=2.0), x)
    assert k2 / k1 == pytest.approx(4.0, rel=1e-8, abs=0)


def test_schwarzschild_eddington_finkelstein_curvature_is_regular_at_and_inside_the_horizon():
    # Kretschmann at r = 4, 2 (horizon), 1, 0.7: measured relative errors -9.7e-12, -9.5e-12, -9.5e-12, -3.7e-11;
    # Ricci residual / sqrt(K) at most 1.3e-10 (r = 0.7, where sqrt(K) = 20); tolerances 5e-10 and 2e-9.
    for r in (4.0, 2.0, 1.0, 0.7):
        x = np.array([0.0, r, 1.0, 0.3])
        assert kretschmann_scalar(SCH_EF, x) == pytest.approx(48.0 / r**6, rel=5e-10, abs=0)
        assert np.abs(ricci_tensor(SCH_EF, x)).max() < 2e-9 * np.sqrt(48.0 / r**6)
    assert kretschmann_scalar(SCH_EF, np.array([0.0, 2.0, 1.0, 0.3])) == pytest.approx(0.75, rel=5e-10, abs=0)


# --------------------------------------------------------------------------------------
# Kerr
# --------------------------------------------------------------------------------------

# (spin a, r, theta, measured K relative error, measured Ricci residual / sqrt(K))
KERR_POINTS = [
    (0.9, 6.0, 1.0, 1.2e-11, 1.5e-10),
    (0.9, 3.5, 2.0, 4.6e-11, 2.1e-10),
    (0.5, 10.0, 0.6, 1.5e-10, 1.3e-8),
    (0.99, 2.0, 1.2, 2.0e-10, 1.6e-10),
]


@pytest.mark.parametrize(("spin", "r", "theta", "k_error", "ricci_ratio"), KERR_POINTS)
def test_kerr_boyer_lindquist_is_ricci_flat_and_matches_vissers_quadratic_invariant(
    spin, r, theta, k_error, ricci_ratio
):
    x = np.array([0.0, r, theta, 0.3])
    expected = kretschmann_visser(r, theta, spin)
    assert kretschmann_scalar(kerr(spin), x) == pytest.approx(expected, rel=10 * k_error, abs=0)
    assert np.abs(ricci_tensor(kerr(spin), x)).max() < 10 * ricci_ratio * np.sqrt(abs(expected))


def test_kerr_with_zero_spin_has_the_schwarzschild_invariant_and_visser_formula_reduces_to_it():
    assert kretschmann_visser(7.0, 1.0, 0.0) == pytest.approx(48.0 / 7.0**6, rel=1e-15, abs=0)
    x = np.array([0.0, 7.0, 1.0, 0.0])
    assert kretschmann_scalar(kerr(0.0), x) == pytest.approx(48.0 / 7.0**6, rel=1e-9, abs=0)


def test_kerr_kretschmann_scalar_takes_negative_values_where_vissers_formula_does():
    # near the ring the factor (rho^4 - 16 r^2 a^2 cos^2 theta) of Visser's closed form turns negative while r^2 > a^2
    # cos^2 theta: K = -3.13 at (r, theta) = (1.0, 0.2) for a = 0.9 (inside the outer horizon, so Kerr's original
    # coordinates are used). Measured relative difference 2.1e-8 (the curvature scale is large here); tolerance 3e-7.
    r, theta, spin = 1.0, 0.2, 0.9
    expected = kretschmann_visser(r, theta, spin)
    assert expected < -3.0
    assert kretschmann_scalar(kerr_ef(spin), np.array([0.0, r, theta, 0.1])) == pytest.approx(expected, rel=3e-7, abs=0)


@pytest.mark.parametrize(
    ("r", "theta", "k_error", "ricci_ratio"),
    [(6.0, 1.0, 7.1e-12, 1.4e-10), (1.4, 1.0, 2.2e-11, 9.4e-12), (1.0, 0.8, 8.4e-12, 3.5e-12)],
)
def test_kerr_in_original_coordinates_is_ricci_flat_and_has_the_same_scalar_invariant(r, theta, k_error, ricci_ratio):
    # the Kretschmann scalar is a scalar: the same value at the same (r, theta) in Boyer-Lindquist and in Kerr's original
    # coordinates; the original coordinates also reach inside the outer horizon r_+ = 1.436 for a = 0.9
    x = np.array([0.0, r, theta, 0.3])
    expected = kretschmann_visser(r, theta, 0.9)
    assert kretschmann_scalar(kerr_ef(0.9), x) == pytest.approx(expected, rel=10 * k_error, abs=0)
    assert np.abs(ricci_tensor(kerr_ef(0.9), x)).max() < 10 * ricci_ratio * np.sqrt(abs(expected))


# --------------------------------------------------------------------------------------
# symmetries and the Bianchi identity
# --------------------------------------------------------------------------------------


def test_riemann_tensor_symmetries_in_kerr():
    # Eq. 3.78-3.81 for R_{rho sigma mu nu} = g_{rho lambda} R^lambda_{sigma mu nu} at (r, theta) = (5, 1), a = 0.9.
    # Measured relative to the largest component (7.18): antisymmetry in the last pair 1.2e-16 (exact by construction),
    # cyclic identity 1.2e-16, antisymmetry in the first pair 1.1e-11, pair exchange 1.1e-11 (numerical differentiation).
    metric = kerr(0.9)
    x = np.array([0.0, 5.0, 1.0, 0.3])
    g = metric(x)
    riemann = riemann_tensor(metric, x)
    low = np.einsum("rl,lsmn->rsmn", g, riemann)
    scale = np.abs(low).max()
    assert np.abs(low + low.transpose(0, 1, 3, 2)).max() / scale < 1e-15  # Eq. 3.64
    assert np.abs(low + low.transpose(0, 2, 3, 1) + low.transpose(0, 3, 1, 2)).max() / scale < 1e-15  # Eq. 3.80
    assert np.abs(low + low.transpose(1, 0, 2, 3)).max() / scale < 1e-10  # Eq. 3.78
    assert np.abs(low - low.transpose(2, 3, 0, 1)).max() / scale < 1e-10  # Eq. 3.79


def _contracted_bianchi_defect(metric, x):
    """g^{lm} nabla_l G_mn at x and the largest derivative of G (the scale)."""
    g = metric(x)
    ginv = np.linalg.inv(g)
    gam = christoffel_symbols(metric, x)
    G = einstein_tensor(metric, x)
    # dG[m, n, l] = d_l G_mn, with its error estimate left aside: the estimate of a difference of a tensor that
    # already contains finite differences carries a safety factor, and the test measures the true defect
    dG = jacobian_with_error(lambda y: einstein_tensor(metric, y), x, method="finite")[0]
    nabla = dG - np.einsum("slm,sn->mnl", gam, G) - np.einsum("sln,ms->mnl", gam, G)
    return np.abs(np.einsum("lm,mnl->n", ginv, nabla)).max(), np.abs(dG).max()


# measured defect relative to the largest derivative of G: Reissner-Nordstrom at r = 5, FLRW a = t^(2/3) at t = 2, r = 2
BIANCHI_MEASURED = (1.1e-10, 3.4e-11)


def test_contracted_bianchi_identity_in_reissner_nordstrom_and_flrw():
    # nabla^mu G_mu_nu = 0 (Eq. 3.96) for spacetimes with a non-zero Einstein tensor. The derivative of G is a
    # third derivative of the metric, taken as a finite difference of a tensor that already contains finite
    # differences; the tolerances are ten times the measured defects (see BIANCHI_MEASURED below).
    rn = partial(reissner_nordstrom_metric, mass_geometric_m=1.0, charge_m=0.6)
    defect, scale = _contracted_bianchi_defect(rn, np.array([0.0, 5.0, 1.0, 0.3]))
    assert scale > 1e-3  # the check is not empty
    assert defect < 10.0 * BIANCHI_MEASURED[0] * scale
    flrw = partial(flrw_metric, scale_factor=lambda t: t ** (2.0 / 3.0), spatial_curvature=0.0)
    defect, scale = _contracted_bianchi_defect(flrw, np.array([2.0, 2.0, 1.0, 0.3]))
    assert scale > 1e-2
    assert defect < 10.0 * BIANCHI_MEASURED[1] * scale


# --------------------------------------------------------------------------------------
# Reissner-Nordstrom
# --------------------------------------------------------------------------------------


def test_reissner_nordstrom_einstein_tensor_is_8_pi_times_the_coulomb_stress_tensor():
    # Carroll Eq. 7.112: F_tr = -q / r^2 (q = charge_m, p = 0); Eq. 7.107: T_mn = (F_mr F_n^r - g_mn F^2 / 4) / 4 pi; G = 8 pi T.
    # Measured |G - 8 pi T| relative to the largest component of G: 1.9e-9 at (3, 0.8), 2.1e-9 at (7, 2.0); Ricci scalar
    # relative to the same scale 5.1e-10 and 1.4e-10. Tolerances 2e-8 and 5e-9.
    q = 0.6
    metric = partial(reissner_nordstrom_metric, mass_geometric_m=1.0, charge_m=q)
    for r, theta in [(3.0, 0.8), (7.0, 2.0)]:
        x = np.array([0.0, r, theta, 0.3])
        g = metric(x)
        ginv = np.linalg.inv(g)
        field = np.zeros((4, 4))
        field[0, 1], field[1, 0] = -q / r**2, q / r**2
        f_squared = np.einsum("mr,ns,mn,rs->", field, field, ginv, ginv)  # F_{mr} F_{ns} g^{mn} g^{rs}
        stress = (np.einsum("mr,rs,ns->mn", field, ginv, field) - 0.25 * g * f_squared) / (4.0 * np.pi)
        einstein = einstein_tensor(metric, x)
        scale = np.abs(einstein).max()
        assert np.abs(einstein - 8.0 * np.pi * stress).max() < 2e-8 * scale
        assert abs(ricci_scalar(metric, x)) < 5e-9 * scale  # the stress tensor is traceless


def test_reissner_nordstrom_without_charge_is_schwarzschild_and_the_charge_changes_the_curvature():
    x = np.array([0.0, 6.0, 1.0, 0.2])
    assert reissner_nordstrom_metric(x, 1.0, 0.0) == pytest.approx(schwarzschild_metric(x, 1.0), abs=0.0)
    assert (
        np.abs(einstein_tensor(partial(reissner_nordstrom_metric, mass_geometric_m=1.0, charge_m=0.0), x)).max() < 1e-10
    )
    assert (
        np.abs(einstein_tensor(partial(reissner_nordstrom_metric, mass_geometric_m=1.0, charge_m=0.5), x)).max() > 1e-4
    )


# --------------------------------------------------------------------------------------
# Friedmann-Robertson-Walker
# --------------------------------------------------------------------------------------

POWER = (
    lambda t: t ** (2.0 / 3.0),
    lambda t: (2.0 / 3.0) * t ** (-1.0 / 3.0),
    lambda t: -(2.0 / 9.0) * t ** (-4.0 / 3.0),
)
HUBBLE = 0.3
EXPONENTIAL = (
    lambda t: np.exp(HUBBLE * t),
    lambda t: HUBBLE * np.exp(HUBBLE * t),
    lambda t: HUBBLE**2 * np.exp(HUBBLE * t),
)

# (name, a(t) with derivatives, k, (t, r, theta), measured errors: Ricci, Ricci scalar, G_tt, G against the printed
# Ricci and scalar, all relative); the points with r < 1 are less accurate because the finite step is 1e-3 absolute
FLRW_CASES = [
    ("t23-k0", POWER, 0.0, (2.0, 2.0, 1.0)),
    ("t23-k-1", POWER, -1.0, (2.0, 2.0, 1.0)),
    ("t23-k+1", POWER, 1.0, (2.0, 0.6, 1.0)),
    ("exp-k0", EXPONENTIAL, 0.0, (2.0, 2.0, 1.0)),
    ("exp-k-1", EXPONENTIAL, -1.0, (2.0, 2.0, 1.0)),
    ("exp-k+1", EXPONENTIAL, 1.0, (2.0, 0.6, 1.0)),
]


def _flrw_errors(case):
    _, (a_f, ad_f, add_f), k, (t0, r, theta) = case
    metric = partial(flrw_metric, scale_factor=a_f, spatial_curvature=k)
    x = np.array([t0, r, theta, 0.3])
    a, ad, add = a_f(t0), ad_f(t0), add_f(t0)
    ricci_printed = np.zeros((4, 4))  # Eq. 8.13
    ricci_printed[0, 0] = -3.0 * add / a
    ricci_printed[1, 1] = (a * add + 2 * ad**2 + 2 * k) / (1 - k * r**2)
    ricci_printed[2, 2] = r**2 * (a * add + 2 * ad**2 + 2 * k)
    ricci_printed[3, 3] = ricci_printed[2, 2] * np.sin(theta) ** 2
    scalar_printed = 6.0 / a**2 * (a * add + ad**2 + k)  # Eq. 8.14
    g = metric(x)
    einstein_printed = ricci_printed - 0.5 * scalar_printed * g
    ricci = ricci_tensor(metric, x)
    einstein = einstein_tensor(metric, x)
    return {
        "ricci": np.abs(ricci - ricci_printed).max() / np.abs(ricci_printed).max(),
        "scalar": abs(ricci_scalar(metric, x) / scalar_printed - 1.0),
        "gtt": abs(einstein[0, 0] / (3.0 * (ad**2 + k) / a**2) - 1.0),
        "einstein": np.abs(einstein - einstein_printed).max() / np.abs(einstein_printed).max(),
    }


# Measured maxima over the six cases: Ricci 4.5e-11, Ricci scalar 1.7e-11, G_tt 1.7e-11, Einstein tensor 5.1e-11 (the
# k = +1 cases with r = 0.6 are the least accurate); tolerances 5e-10, 2e-10, 2e-10, 5e-10.
@pytest.mark.parametrize("case", FLRW_CASES, ids=[c[0] for c in FLRW_CASES])
def test_flrw_ricci_and_einstein_tensors_equal_the_printed_expressions(case):
    errors = _flrw_errors(case)
    assert errors["ricci"] < 5e-10
    assert errors["scalar"] < 2e-10
    assert errors["gtt"] < 2e-10  # G_tt = 3 (adot^2 + k) / a^2, the first Friedmann equation Eq. 8.36 with 8 pi G rho
    assert errors["einstein"] < 5e-10


def test_flrw_dust_universe_obeys_the_friedmann_equations_with_zero_pressure():
    # a = t^(2/3), k = 0: 8 pi G rho = 3 H^2 = 4 / (3 t^2) (Eq. 8.36), and Eq. 8.35 with p = 0 gives
    # addot / a = -(1/2) H^2: G^i_j = 0 (zero pressure). G^t_t = -8 pi G rho.
    metric = partial(flrw_metric, scale_factor=POWER[0], spatial_curvature=0.0)
    t, r = 2.0, 2.0
    x = np.array([t, r, 1.0, 0.3])
    g = metric(x)
    mixed = np.linalg.inv(g) @ einstein_tensor(metric, x)
    # measured: G^t_t relative error 5.0e-12, largest spatial component 3.5e-12 of |G^t_t|; tolerances 5e-11 and 4e-11
    assert mixed[0, 0] == pytest.approx(-4.0 / (3.0 * t**2), rel=5e-11, abs=0)
    spatial = mixed[1:, 1:]
    assert np.abs(spatial).max() < 4e-11 * abs(mixed[0, 0])


def test_flrw_de_sitter_universe_has_a_cosmological_constant_einstein_tensor():
    # a = exp(H t), k = 0: Eq. 8.29 with T = 0 and 8 pi G rho = 3 H^2 gives G_mn = -Lambda g_mn with Lambda = 3 H^2.
    metric = partial(flrw_metric, scale_factor=EXPONENTIAL[0], spatial_curvature=0.0)
    x = np.array([2.0, 2.0, 1.0, 0.3])
    # measured maximum difference 1.3e-12 relative to the largest component (0.0035); tolerance 1.3e-11
    expected = -3.0 * HUBBLE**2 * metric(x)
    assert np.abs(einstein_tensor(metric, x) - expected).max() < 1.3e-11 * np.abs(expected).max()


def test_flrw_einstein_tensor_gives_the_critical_density_of_the_cosmology_package():
    # Einstein-de Sitter universe: Cosmology(h0=70, omega_m=1, omega_lambda=0), a(t) = (t / t0)^(2/3) with H = 2 / (3 t).
    # Geometric units: the coordinate t is c times the cosmic time and G_tt = 3 H^2 / c^2 (a curvature, m^-2).
    # Then rho_c = G_tt c^2 / (8 pi G) must equal critical_density_kg_m3 of the package (3 H^2 / (8 pi G)); the Hubble
    # parameter comes from the package (hubble_parameter_km_s_mpc), not from the metric.
    cosmology = Cosmology(h0=70.0, omega_m=1.0, omega_lambda=0.0)
    c = SPEED_OF_LIGHT.value
    z = 1.0
    h_si = hubble_parameter_km_s_mpc(z, cosmology) * 1e3 / MEGAPARSEC.value
    t_geo = c * 2.0 / (3.0 * h_si)  # H = 2 / (3 t)
    metric = partial(flrw_metric, scale_factor=lambda t: (t / t_geo) ** (2.0 / 3.0), spatial_curvature=0.0)
    # The comoving radius is given in the same length unit as t (a = 1 at t_geo), r = t_geo: with a dimensionless r of
    # order one the cancellation of the spatial Christoffel symbols (error about 1e-12 of 1/r^2) would swamp G_tt, which
    # is about 1e-51 m^-2.
    x = np.array([t_geo, t_geo, 1.0, 0.3])
    g_tt = einstein_tensor(metric, x)[0, 0]
    rho = g_tt * c**2 / (8.0 * np.pi * GRAVITATIONAL_CONSTANT.value)
    # Measured: G_tt = 4 / (3 t^2) to 7.9e-12 relative, rho against critical_density_kg_m3 to 7.9e-12 (z = 1);
    # tolerance 1e-10.
    assert g_tt == pytest.approx(4.0 / (3.0 * t_geo**2), rel=1e-10, abs=0)
    assert rho == pytest.approx(critical_density_kg_m3(z, cosmology), rel=1e-10, abs=0)


# --------------------------------------------------------------------------------------
# the precision guard of the Riemann tensor
# --------------------------------------------------------------------------------------


def test_curvature_that_is_not_known_to_rtol_is_refused():
    """Schwarzschild at r = 1e6 M: the curvature M / r^3 is 1e-18 against connection terms of 1 / r^2 = 1e-12.

    Measured: the error estimate is 6.1e-6 of the tensor, above the default limit 1e-6, so the default call raises;
    with rtol = 1e-4 the value is returned, and its true error is 1.1e-8.
    """
    metric = partial(schwarzschild_metric, mass_geometric_m=1.0)
    point = [0.0, 1e6, 1.0, 0.3]
    with pytest.raises(curvature.CurvaturePrecisionError, match="relative accuracy"):
        kretschmann_scalar(metric, point)
    with pytest.raises(curvature.CurvaturePrecisionError):
        riemann_tensor(metric, point)
    loose = kretschmann_scalar(metric, point, rtol=1e-4)
    assert loose == pytest.approx(48.0 / 1e36, rel=1e-6, abs=0)
    # well inside the limit nothing is raised and nothing is warned about
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        near = kretschmann_scalar(metric, [0.0, 1e4, 1.0, 0.3])
    assert near == pytest.approx(48.0 / 1e24, rel=1e-8, abs=0)  # measured 3.3e-10


def test_flat_space_in_spherical_coordinates_is_exactly_zero_with_a_warning():
    """Schwarzschild coordinates with M = 0: Christoffel terms of order 1 / r^2 cancel to nothing.

    The tensor vanishes within its uncertainty; it is returned as exactly zero (no rounding noise for scalars to
    square), and the warning says that this is a statement within an uncertainty.
    """
    flat = partial(schwarzschild_metric, mass_geometric_m=0.0)
    with pytest.warns(curvature.CurvaturePrecisionWarning, match="returned as zero"):
        riemann = riemann_tensor(flat, [0.0, 5.0, 1.0, 0.3])
    assert riemann.shape == (4, 4, 4, 4)
    assert np.all(riemann == 0.0)
    with pytest.warns(curvature.CurvaturePrecisionWarning):
        assert kretschmann_scalar(flat, [0.0, 5.0, 1.0, 0.3]) == 0.0
    # Cartesian flat space has no connection terms at all: zero without any warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.all(riemann_tensor(minkowski_metric, [0.0, 1.0, 2.0, 3.0]) == 0.0)


def test_unresolved_curvature_is_zero_with_a_warning_and_never_noise():
    """FLRW with a(t) = 1e-12 t^(2/3) at comoving r = 0.5: the true Kretschmann scalar is 12 [(a''/a)^2 + (a'/a)^4]
    = 0.185 at t = 2, but the spatial connection terms are 1 / (a r)^2 = 2.5e24, twenty-five orders above it.

    Before, the rounding noise of the cancellation was returned and its square appeared as a Kretschmann scalar.
    Now the result is zero and the warning carries the size below which nothing can be said.
    """

    def metric(x):
        return flrw_metric(x, lambda t: 1e-12 * t ** (2.0 / 3.0), 0.0)

    point = [2.0, 0.5, 1.0, 0.3]
    with pytest.warns(curvature.CurvaturePrecisionWarning, match="below about"):
        assert kretschmann_scalar(metric, point) == 0.0
    with pytest.warns(curvature.CurvaturePrecisionWarning):
        assert ricci_scalar(metric, point) == 0.0
    # the same spacetime with a scale factor of order one is resolved: K = 12 [(2/9 / 4)^2 + (1/3)^4]
    resolved = kretschmann_scalar(lambda x: flrw_metric(x, lambda t: t ** (2.0 / 3.0), 0.0), point)
    assert resolved == pytest.approx(12.0 * ((2.0 / 9.0 / 4.0) ** 2 + (1.0 / 3.0) ** 4), rel=1e-8, abs=0)
