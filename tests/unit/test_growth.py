"""Linear growth: closed forms, an independent form of the ODE, and the accuracy of Linder's growth-index fit.

References for the numbers:

* Einstein-de Sitter: delta ∝ a, hence D = a, g = 1, f = 1 (matter domination, [Frieman2008] sec2.tex line 314).
* Flat LCDM: D ∝ H(a) ∫_0^a da' / (a' H(a'))^3. This integral solution is a standard result that is NOT printed in
  the TeX sources read for this project ([Linder2005] only mentions "a single integral", line 391), so the test is
  an independent numerical identity: the integral is evaluated with scipy.integrate.quad in the test and compared
  with the ODE solution.
* The stated accuracies of the growth-index fit are those of [Linder2005] lines 372-389; the measured errors are
  recorded next to them in ``FIT_CASES``.
"""

import warnings

import numpy as np
import pytest
from scipy.integrate import IntegrationWarning, quad, solve_ivp

from kozmos.cosmology import dark_energy as de
from kozmos.cosmology import distances, growth

Z = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
EDS = distances.Cosmology.flat(100.0, 1.0)
LCDM = distances.Cosmology.flat(70.0, 0.3)
CPL = distances.Cosmology.flat(70.0, 0.3, w=-0.8, wa=0.5)


def test_einstein_de_sitter_growth_is_exactly_the_scale_factor():
    # measured: |D (1+z) - 1| < 2e-16, |f - 1| = 0, |g - 1| = 0 for 0 <= z <= 10
    assert np.max(np.abs(growth.growth_factor(Z, EDS) * (1.0 + Z) - 1.0)) < 1e-12
    assert np.max(np.abs(growth.growth_log_slope(Z, EDS) - 1.0)) < 1e-12
    assert np.max(np.abs(growth.growth_suppression(Z, EDS) - 1.0)) < 1e-12


def _lcdm_integral_growth(z, cosmology):
    def h(a):
        return distances.e_function(1.0 / a - 1.0, cosmology)

    def d_unnormalized(a):
        return h(a) * quad(lambda x: 1.0 / (x * h(x)) ** 3, 0.0, a, epsabs=0.0, epsrel=1e-13)[0]

    a = 1.0 / (1.0 + z)
    return d_unnormalized(a) / d_unnormalized(1.0)


def test_lcdm_matches_the_integral_solution():
    reference = np.array([_lcdm_integral_growth(z, LCDM) for z in Z])
    # measured maximum relative difference 6.1e-12 for 0 <= z <= 10
    assert np.max(np.abs(growth.growth_factor(Z, LCDM) / reference - 1.0)) < 1e-10


def _lcdm_integral_slope(z, cosmology=LCDM):
    # D = H I(a), I = int_0^a da'/(a' H)^3  ->  f = dlnD/dlna = dlnH/dlna + a / (I (a H)^3)
    a = 1.0 / (1.0 + z)

    def h(x):
        return distances.e_function(1.0 / x - 1.0, cosmology)

    step = 1e-5
    dlnh = (np.log(h(a * np.exp(step))) - np.log(h(a * np.exp(-step)))) / (2.0 * step)
    integral = quad(lambda x: 1.0 / (x * h(x)) ** 3, 0.0, a, epsabs=0.0, epsrel=1e-13)[0]
    return dlnh + a / (integral * (a * h(a)) ** 3)


def test_lcdm_growth_slope_matches_the_integral_solution():
    reference = np.array([_lcdm_integral_slope(z) for z in Z])
    # measured maximum relative difference 3.6e-11 (limited by the central difference in the reference)
    assert np.max(np.abs(growth.growth_log_slope(Z, LCDM) / reference - 1.0)) < 1e-9


def test_other_form_of_linders_equation_for_a_w0wa_model():
    # [Linder2005] Eq. grow (flat, no radiation), integrated here independently from Eq. groq of the module:
    # g'' + [5/2 - (3/2) w Omega_w] g' + (3/2)(1 - w) Omega_w g = 0
    def rhs(n, y):
        z = np.expm1(-n)
        w = float(de.de_eos(z, CPL))
        ow = float(de.omega_de(z, CPL))
        return [y[1], -(2.5 - 1.5 * w * ow) * y[1] - 1.5 * (1.0 - w) * ow * y[0]]

    solution = solve_ivp(
        rhs, (-np.log1p(1000.0), 0.0), [1.0, 0.0], method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True
    )
    other = solution.sol(-np.log1p(Z))[0]
    # measured maximum relative difference 1.4e-12
    assert np.max(np.abs(growth.growth_suppression(Z, CPL) / other - 1.0)) < 1e-9


def test_normalization_and_monotonicity():
    z = np.linspace(0.0, 20.0, 41)
    d = growth.growth_factor(z, CPL)
    assert d[0] == pytest.approx(1.0, abs=1e-15)
    assert np.all(np.diff(d) < 0)  # D falls towards the past
    g = growth.growth_suppression(z, CPL)
    assert np.all(np.diff(g) > 0)  # g = D/a rises towards the past, tends to 1
    assert growth.growth_suppression(1000.0, CPL) == pytest.approx(1.0, abs=1e-15)


def test_growth_slope_equals_the_logarithmic_derivative_of_d():
    z = np.array([0.2, 1.0, 3.0])
    h = 1e-4
    lna = -np.log1p(z)
    numerical = (
        np.log(growth.growth_factor(np.expm1(-(lna + h)), CPL))
        - np.log(growth.growth_factor(np.expm1(-(lna - h)), CPL))
    ) / (2 * h)
    # measured maximum difference 2e-9 (central-difference truncation)
    assert np.max(np.abs(growth.growth_log_slope(z, CPL) - numerical)) < 1e-7


def test_insensitivity_to_the_starting_redshift():
    # relative change of D (0 <= z <= 10) when z_start changes, reference z_start = 5000; measured values in comments
    reference = growth.growth_factor(Z, LCDM, z_start=5000.0)
    # LCDM: 5.9e-11 (200), 6.1e-11 (500), 4.5e-11 (1000), 1.9e-9 (100)
    for z_start in (200.0, 500.0, 1000.0):
        assert np.max(np.abs(growth.growth_factor(Z, LCDM, z_start=z_start) / reference - 1.0)) < 1e-9
    # w0wa (dark energy not negligible early, w0 + wa = -0.3): 3.4e-8 (500), 3.2e-9 (1000), 7.6e-7 (200)
    reference = growth.growth_factor(Z, CPL, z_start=5000.0)
    assert np.max(np.abs(growth.growth_factor(Z, CPL, z_start=1000.0) / reference - 1.0)) < 1e-8
    assert np.max(np.abs(growth.growth_factor(Z, CPL, z_start=200.0) / reference - 1.0)) < 5e-6
    # with radiation the initial conditions are only approximately matter dominated: 2.8e-6 (500), 1.0e-6 (1000)
    rad = distances.Cosmology.flat(70.0, 0.3, omega_r=8.5e-5)
    reference = growth.growth_factor(Z, rad, z_start=5000.0)
    assert np.max(np.abs(growth.growth_factor(Z, rad, z_start=1000.0) / reference - 1.0)) < 1e-5
    assert np.max(np.abs(growth.growth_factor(Z, rad, z_start=500.0) / reference - 1.0)) < 1e-5


def test_insensitivity_to_the_ode_tolerance():
    # reference rtol = 1e-13; measured: 1.3e-10 (rtol 1e-8), 6.3e-12 (1e-10) for LCDM; 2.8e-9 and 5.9e-11 with radiation
    reference = growth.growth_factor(Z, LCDM, rtol=1e-13)
    assert np.max(np.abs(growth.growth_factor(Z, LCDM, rtol=1e-10) / reference - 1.0)) < 1e-10
    assert np.max(np.abs(growth.growth_factor(Z, LCDM, rtol=1e-8) / reference - 1.0)) < 1e-8


def test_out_of_range_redshift_is_rejected():
    with pytest.raises(ValueError):
        growth.growth_factor(-0.1, LCDM)
    with pytest.raises(ValueError):
        growth.growth_factor(2000.0, LCDM)


def test_growth_index_values():
    assert growth.growth_index_gamma(LCDM) == pytest.approx(0.55, abs=1e-15)
    # w(z=1) = w0 + wa/2 = -0.8 + 0.25 = -0.55: gamma = 0.55 + 0.05 * 0.45
    assert growth.growth_index_gamma(CPL) == pytest.approx(0.55 + 0.05 * 0.45, abs=1e-15)
    phantom = distances.Cosmology.flat(70.0, 0.3, w=-1.2)
    assert growth.growth_index_gamma(phantom) == pytest.approx(0.55 - 0.05 * 0.2, abs=1e-15)
    assert growth.growth_index_gamma(phantom, phantom_variant=True) == pytest.approx(0.55 - 0.02 * 0.2, abs=1e-15)
    # the variant applies only to 1 + w(z=1) < 0
    assert growth.growth_index_gamma(CPL, phantom_variant=True) == growth.growth_index_gamma(CPL)


_ZFIT = np.concatenate([[0.0], np.logspace(-2, 1, 20), [30.0, 100.0]])

# (label, cosmology, phantom_variant, claim of [Linder2005] for g, measured maximum |g_fit/g_ode - 1| for 0 <= z <= 100,
#  bound used in the test = measured rounded up by ~10 percent)
FIT_CASES = [
    # claim: "better than 0.05% over Omega_m in [0.22, 1]" (line 375); measured 5.02e-4 at 0.22 (equal within rounding)
    ("LCDM 0.22", distances.Cosmology.flat(70.0, 0.22), False, 5e-4, 5.1e-4),
    ("LCDM 0.30", distances.Cosmology.flat(70.0, 0.30), False, 5e-4, 5.0e-4),
    ("LCDM 0.50", distances.Cosmology.flat(70.0, 0.50), False, 5e-4, 5.0e-4),
    ("LCDM 1.00", distances.Cosmology.flat(70.0, 1.00), False, 5e-4, 1e-12),
    # claim: "within 1% all the way down to Omega_m = 0.01" (line 376); measured 1.15% (exceeds the claim)
    ("LCDM 0.01", distances.Cosmology.flat(70.0, 0.01), False, 1e-2, 1.3e-2),
    # claim: w = -0.8: 0.2% for Omega_m in [0.2, 1] (line 379); measured 0.186%
    ("w=-0.8 0.20", distances.Cosmology.flat(70.0, 0.20, w=-0.8), False, 2e-3, 2.1e-3),
    ("w=-0.8 0.30", distances.Cosmology.flat(70.0, 0.30, w=-0.8), False, 2e-3, 2.1e-3),
    # claim: w = -0.5: 0.4%; measured 0.434% (exceeds the claim by 8 percent)
    ("w=-0.5 0.20", distances.Cosmology.flat(70.0, 0.20, w=-0.5), False, 4e-3, 4.8e-3),
    ("w=-0.5 0.30", distances.Cosmology.flat(70.0, 0.30, w=-0.5), False, 4e-3, 4.8e-3),
    # claim: w = -1.2: 0.3%, w = -1.5: 0.5% with Eq. gamma (lines 383-384); measured 0.259% and 0.509%
    ("w=-1.2 0.20", distances.Cosmology.flat(70.0, 0.20, w=-1.2), False, 3e-3, 2.9e-3),
    ("w=-1.5 0.20", distances.Cosmology.flat(70.0, 0.20, w=-1.5), False, 5e-3, 5.7e-3),
    # claim: 0.05% with Eq. gammam1 (line 388); measured 0.052% (w = -1.2) and 0.036% (w = -1.5)
    ("w=-1.2 0.20 variant", distances.Cosmology.flat(70.0, 0.20, w=-1.2), True, 5e-4, 5.8e-4),
    ("w=-1.5 0.20 variant", distances.Cosmology.flat(70.0, 0.20, w=-1.5), True, 5e-4, 4.0e-4),
    # claim: SUGRA w0 = -0.82, wa = 0.58 within 0.25% (line 381); the source's SUGRA background is the exact
    # supergravity solution, here the CPL form with those parameters is used, and Omega_m is not stated: the
    # measured 0.41% (Omega_m = 0.3) is for a different background and is not a test of the claim
    ("CPL(-0.82, 0.58) 0.30", distances.Cosmology.flat(70.0, 0.30, w=-0.82, wa=0.58), False, 2.5e-3, 4.5e-3),
]


@pytest.mark.parametrize(("label", "cosmology", "variant", "claim", "bound"), FIT_CASES, ids=[c[0] for c in FIT_CASES])
def test_growth_index_fit_accuracy(label, cosmology, variant, claim, bound):
    gamma = growth.growth_index_gamma(cosmology, phantom_variant=variant)
    exact = growth.growth_suppression(_ZFIT, cosmology)
    fit = growth.growth_suppression_from_index(_ZFIT, cosmology, gamma)
    error = np.max(np.abs(fit / exact - 1.0))
    assert error < bound, (label, error, claim)


def test_growth_slope_fit_is_less_accurate_than_the_growth_function_fit():
    # f = Omega_m^gamma: the paper states accuracies for g only. Measured for flat LCDM with Omega_m = 0.3:
    # max |f_fit/f_ode - 1| = 5.7e-3 over 0 <= z <= 100, against 4.6e-4 for g.
    exact = growth.growth_log_slope(_ZFIT, LCDM)
    fit = growth.growth_log_slope_from_index(_ZFIT, LCDM)
    error = np.max(np.abs(fit / exact - 1.0))
    assert 1e-3 < error < 6.5e-3


def test_fit_growth_equals_one_in_einstein_de_sitter():
    assert np.allclose(growth.growth_suppression_from_index(_ZFIT, EDS), 1.0, atol=1e-14)
    assert np.allclose(growth.growth_log_slope_from_index(_ZFIT, EDS), 1.0, atol=1e-14)


def test_fsigma8():
    assert np.allclose(growth.fsigma8(Z, EDS, 0.8), 0.8 / (1.0 + Z), rtol=1e-12)
    product = growth.growth_log_slope(Z, CPL) * 0.81 * growth.growth_factor(Z, CPL)
    assert np.allclose(growth.fsigma8(Z, CPL, 0.81), product, rtol=1e-14)
    # LCDM, Omega_m = 0.3, sigma8 = 0.8: independent f and D from the integral solution D ∝ H int da/(a H)^3 (scipy quad)
    reference = np.array([_lcdm_integral_slope(z) * 0.8 * _lcdm_integral_growth(z, LCDM) for z in Z])
    # measured maximum relative difference 3.7e-11 (the central difference in the reference slope)
    assert np.max(np.abs(growth.fsigma8(Z, LCDM, 0.8) / reference - 1.0)) < 1e-9
    assert growth.fsigma8(0.0, LCDM, 0.8) == pytest.approx(reference[0], rel=1e-9, abs=0)


# --- independent integration of the growth equation in the scale factor ------------------------------------------


def _e_squared_of_a(k, a):
    return (
        k.omega_m * a**-3
        + k.omega_r * a**-4
        + k.omega_k * a**-2
        + k.omega_lambda * a ** (-3.0 * (1.0 + k.w + k.wa)) * np.exp(-3.0 * k.wa * (1.0 - a))
    )


def _delta_reference(k, z_eval, z_start=1000.0):
    """delta'' + (3/a + dlnE/da) delta' - 1.5 Om0 a^-5 / E^2 delta = 0 integrated in a (written here, no ln a, no q).

    Starts with delta = a, delta' = 1 at z_start (the same matter-dominated condition as the module). Returns the
    growth factor D = delta/delta(1), the slope f = a delta'/delta and the suppression g = delta/a at z_eval.
    """

    def d_e2(a):  # d(E^2)/da, term by term
        de = k.omega_lambda * a ** (-3.0 * (1.0 + k.w + k.wa)) * np.exp(-3.0 * k.wa * (1.0 - a))
        return (
            -3.0 * k.omega_m * a**-4
            - 4.0 * k.omega_r * a**-5
            - 2.0 * k.omega_k * a**-3
            + de * (-3.0 * (1.0 + k.w + k.wa) / a + 3.0 * k.wa)
        )

    def rhs(a, y):
        e2 = _e_squared_of_a(k, a)
        return [y[1], -(3.0 / a + d_e2(a) / (2.0 * e2)) * y[1] + 1.5 * k.omega_m * a**-5 / e2 * y[0]]

    a0 = 1.0 / (1.0 + z_start)
    sol = solve_ivp(rhs, (a0, 1.0), [a0, 1.0], method="DOP853", rtol=1e-13, atol=1e-15, dense_output=True)
    a = 1.0 / (1.0 + np.asarray(z_eval, dtype=float))
    delta, d_delta = sol.sol(a)
    return delta / sol.sol(1.0)[0], a * d_delta / delta, delta / a


ZG = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0])
# (label, model, measured max |D/D_ref - 1| (g agrees within the same order), measured max |f/f_ref - 1|) for
# 0 <= z <= 100; the tests allow ten times the measured value
GROWTH_MODELS = [
    ("open", distances.Cosmology(70.0, 0.3, 0.5), 8.2e-11, 3.4e-10),
    ("closed", distances.Cosmology(70.0, 0.4, 0.8), 1.9e-11, 6.8e-11),
    ("phantom", distances.Cosmology.flat(70.0, 0.3, w=-1.2), 9.6e-12, 5.3e-11),
    ("curved w0wa radiation", distances.Cosmology(68.0, 0.28, 0.69, w=-0.9, wa=-0.3, omega_r=9e-5), 8.2e-11, 3.2e-10),
    ("flat radiation", distances.Cosmology.flat(70.0, 0.3, omega_r=8.5e-5), 5.8e-11, 1.8e-10),
]


@pytest.mark.parametrize(("label", "k", "d_err", "f_err"), GROWTH_MODELS, ids=[m[0] for m in GROWTH_MODELS])
def test_growth_against_an_independent_integration_in_the_scale_factor(label, k, d_err, f_err):
    d_ref, f_ref, g_ref = _delta_reference(k, ZG)
    assert np.max(np.abs(growth.growth_factor(ZG, k) / d_ref - 1.0)) < 10.0 * d_err, label
    assert np.max(np.abs(growth.growth_log_slope(ZG, k) / f_ref - 1.0)) < 10.0 * f_err, label
    assert np.max(np.abs(growth.growth_suppression(ZG, k) / g_ref - 1.0)) < 10.0 * d_err, label


# --- bouncing models ---------------------------------------------------------------------------------------------

# Cosmology(70, 0.1, 1.5): E^2 = 0.1 x^3 - 0.6 x^2 + 1.5 (x = 1 + z) is negative for 0.9166 < z < 4.4
BOUNCE = distances.Cosmology(70.0, 0.1, 1.5)


def test_growth_raises_when_e_squared_is_not_positive_in_the_integration_range():
    # before the fix: growth_factor([0, 1, 5]) = [1, 0.013, -9e-89], finite nonsense through the negative-E^2 window
    for function in (growth.growth_factor, growth.growth_log_slope, growth.growth_suppression):
        with pytest.raises(ValueError, match=r"E\(z\)\^2 <= 0"):
            function(np.array([0.0, 1.0, 5.0]), BOUNCE)
    with pytest.raises(ValueError, match=r"E\(z\)\^2 <= 0"):
        growth.fsigma8(0.0, BOUNCE, 0.8)
    # the window lies beyond z = 0.5: starting there is fine and finite
    d = growth.growth_factor(np.array([0.0, 0.2, 0.5]), BOUNCE, z_start=0.5)
    assert np.all(np.isfinite(d)) and d[0] == pytest.approx(1.0, abs=1e-15) and d[2] < d[1] < d[0]
    # the fit integrates Omega_m^gamma from a = 1e-4: NaN when that path crosses the window (z <= 4.4, at z = 0, 0.2
    # and 2), finite when it ends before it (z = 6 and 50)
    fit = growth.growth_suppression_from_index(np.array([0.0, 0.2, 2.0, 6.0, 50.0]), BOUNCE, 0.55)
    assert np.all(np.isnan(fit[:3])) and np.all(np.isfinite(fit[3:]))


# --- shapes ------------------------------------------------------------------------------------------------------


def test_growth_functions_accept_any_shape():
    # before the fix: "truth value of an array is ambiguous" (D, f, g, f sigma_8) and a broadcast error (fit) for 2-D z
    z2 = np.array([[0.0, 0.5, 1.0], [2.0, 5.0, 10.0]])
    z3 = np.array([[[0.1], [0.2]], [[3.0], [7.0]]])
    for z in (z2, z3):
        for function in (growth.growth_factor, growth.growth_log_slope, growth.growth_suppression):
            out = function(z, CPL)
            assert out.shape == z.shape
            assert np.array_equal(out, function(z.ravel(), CPL).reshape(z.shape))
        out = growth.fsigma8(z, CPL, 0.8)
        assert out.shape == z.shape
        assert np.allclose(out, growth.fsigma8(z.ravel(), CPL, 0.8).reshape(z.shape), rtol=1e-15, atol=0.0)
        fit = growth.growth_suppression_from_index(z, CPL)
        assert fit.shape == z.shape
        assert np.allclose(fit, growth.growth_suppression_from_index(z.ravel(), CPL).reshape(z.shape), rtol=1e-15)
        assert growth.growth_log_slope_from_index(z, CPL).shape == z.shape
    # scalars stay scalars
    assert np.ndim(growth.growth_factor(1.0, CPL)) == 0 and np.ndim(growth.growth_suppression_from_index(1.0, CPL)) == 0
    assert np.ndim(growth.growth_log_slope(1.0, CPL)) == 0 and np.ndim(growth.fsigma8(1.0, CPL, 0.8)) == 0


# --- the growth-index integral against scipy quad --------------------------------------------------------------


def _fit_reference(z, k, gamma):
    """exp int_{ln 1e-4}^{-ln(1+z)} [Omega_m(N)^gamma - 1] dN with Omega_m(a) from the own E^2(a) and scipy quad."""

    def integrand(n):
        a = np.exp(n)
        return (k.omega_m * a**-3 / _e_squared_of_a(k, a)) ** gamma - 1.0

    lo, hi = np.log(1e-4), -np.log1p(z)
    if hi == lo:
        return 1.0
    points = np.linspace(lo, hi, 12)[1:-1]
    # quad is asked for 1e-13 and warns about rounding at its limit; the value agrees with the module to 5e-15
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        return np.exp(quad(integrand, lo, hi, epsabs=0.0, epsrel=1e-13, limit=500, points=points)[0])


FIT_REFERENCE_MODELS = [
    distances.Cosmology.flat(70.0, 0.3),
    distances.Cosmology.flat(70.0, 0.01),
    distances.Cosmology.flat(70.0, 0.2, w=-1.5),
    distances.Cosmology.flat(70.0, 0.3, w=-1.0, wa=-3.0),
    distances.Cosmology(70.0, 0.3, 0.2),
    distances.Cosmology.flat(70.0, 0.3, omega_r=0.05),
]


@pytest.mark.parametrize("k", FIT_REFERENCE_MODELS)
def test_growth_index_integral_equals_scipy_quad(k):
    # one 64-point rule in ln a over at most 9.2 e-folds; measured maximum relative difference 4.9e-15 over these
    # six models (z up to 9999) for one panel, 3.3e-16 for the former ten panels: panels are not needed
    zs = np.array([0.0, 1.0, 10.0, 100.0, 1000.0, 9999.0])
    gamma = growth.growth_index_gamma(k)
    reference = np.array([_fit_reference(z, k, gamma) for z in zs])
    assert np.max(np.abs(growth.growth_suppression_from_index(zs, k) / reference - 1.0)) < 1e-13
