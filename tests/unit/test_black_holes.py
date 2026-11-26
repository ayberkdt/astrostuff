"""Tests of the black hole module against numerical values in the sources and independent calculations.

The numerical values were read from the following sources: [EHT2019] (M87* shadow diameter),
[Abramowicz2013] (Eddington luminosity, Hawking temperature footnote), [Inayoshi2020] (Table 3,
Salpeter time), [Komossa2015] (tidal radius). Independent calculations: numerical minimization,
numerical integral, the Δ = 0 identity of the Kerr metric and the first law of thermodynamics.
"""

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize_scalar

from kozmos.black_holes import accretion, basics
from kozmos.core.constants import (
    BOLTZMANN,
    GM_SUN,
    GRAVITATIONAL_CONSTANT,
    JULIAN_YEAR_SECONDS,
    PARSEC,
    PLANCK,
    PROTON_MASS,
    SPEED_OF_LIGHT,
    STEFAN_BOLTZMANN,
    THOMSON_CROSS_SECTION,
)

G = GRAVITATIONAL_CONSTANT.value
C = SPEED_OF_LIGHT.value
HBAR = PLANCK.value / (2.0 * np.pi)
MSUN = GM_SUN.value / G
YR = JULIAN_YEAR_SECONDS.value
MPC = 1e6 * PARSEC.value
PC = PARSEC.value
ERG_S_W = 1e-7  # 1 erg/s = 1e-7 W


# ----------------------------------------------------------------------------- Schwarzschild


def test_radii_for_sun():
    # r_G = GM/c² and R_S = 2 r_G [EHT2019 §1]; 1.477 km and 2.953 km for the Sun.
    assert basics.gravitational_radius_m(MSUN) == pytest.approx(1476.6e0, rel=1e-3, abs=0)
    assert basics.schwarzschild_radius_m(MSUN) == pytest.approx(2953.3e0, rel=1e-3, abs=0)
    assert basics.schwarzschild_radius_m(MSUN) == pytest.approx(2.0 * G * MSUN / C**2, rel=1e-14, abs=0)


def test_redshift_closed_form_and_limit():
    rs = basics.schwarzschild_radius_m(MSUN)
    # r = 3 R_S / 2: 1 − R_S/r = 1/3 ⇒ 1 + z = √3.
    assert basics.gravitational_redshift(1.5 * rs, MSUN) == pytest.approx(np.sqrt(3.0) - 1.0, rel=1e-14, abs=0)
    # Weak field: z ≈ R_S / (2 r); the relative error at first order is ~ 3 R_S/(4 r) = 7.5e-9 (r = 1e8 R_S).
    assert basics.gravitational_redshift(1e8 * rs, MSUN) == pytest.approx(0.5e-8, rel=1e-7, abs=0)
    with pytest.raises(ValueError):
        basics.gravitational_redshift(rs, MSUN)


def test_photon_sphere():
    assert basics.photon_sphere_radius_m(MSUN) == pytest.approx(
        1.5 * basics.schwarzschild_radius_m(MSUN), rel=1e-14, abs=0
    )
    # The a* = 0 value of the Kerr formula must give the same result.
    assert basics.kerr_photon_orbit_radius_rg(0.0) == pytest.approx(3.0, abs=1e-14)


def test_critical_impact_parameter_independent_minimum():
    # Independent route: in Schwarzschild, b(r) = r / √(1 − R_S/r) is the impact parameter
    # of the photon at its closest approach; the capture limit is the minimum of this function.
    rg = basics.gravitational_radius_m(MSUN)
    b = lambda x: x / np.sqrt(1.0 - 2.0 / x)  # noqa: E731  (x = r / r_G)
    result = minimize_scalar(b, bounds=(2.001, 20.0), method="bounded", options={"xatol": 1e-12})
    assert result.x == pytest.approx(3.0, rel=1e-6, abs=0)
    assert basics.critical_impact_parameter_m(MSUN) == pytest.approx(result.fun * rg, rel=1e-12, abs=0)
    # [EHT2019] §1: R_c = √27 r_g.
    assert basics.critical_impact_parameter_m(MSUN) == pytest.approx(np.sqrt(27.0) * rg, rel=1e-14, abs=0)


def test_m87_shadow_diameter_source_values():
    # [EHT2019] §2: for D = 16.8 Mpc the Schwarzschild shadow diameter is 37.6 µas with the stellar
    # dynamics mass (6.2e9 M☉) and 21.3 µas with the gas dynamics mass (3.5e9 M☉).
    # Tolerance 1%: the source rounds the masses to two and the diameters to three significant digits.
    d = 16.8 * MPC
    assert basics.shadow_angular_diameter_uas(6.2e9 * MSUN, d) == pytest.approx(37.6, rel=1e-2, abs=0)
    assert basics.shadow_angular_diameter_uas(3.5e9 * MSUN, d) == pytest.approx(21.3, rel=1e-2, abs=0)


def test_shadow_diameter_scaling():
    d = 10.0 * MPC
    assert basics.shadow_angular_diameter_uas(2e9 * MSUN, 2 * d) == pytest.approx(
        basics.shadow_angular_diameter_uas(1e9 * MSUN, d), rel=1e-14, abs=0
    )


def test_isco_schwarzschild():
    assert basics.schwarzschild_isco_radius_m(MSUN) == pytest.approx(
        3.0 * basics.schwarzschild_radius_m(MSUN), rel=1e-14, abs=0
    )
    assert basics.kerr_isco_radius_rg(0.0) == pytest.approx(6.0, abs=1e-13)


def test_isco_orbital_frequency():
    # Independent computation with Ω_K² = GM/r³; ≈ 2.2 kHz for one solar mass.
    r = 6.0 * G * MSUN / C**2
    expected = np.sqrt(G * MSUN / r**3) / (2.0 * np.pi)
    assert basics.isco_orbital_frequency_hz(MSUN) == pytest.approx(expected, rel=1e-13, abs=0)
    assert basics.isco_orbital_frequency_hz(MSUN) == pytest.approx(2.2e3, rel=1e-2, abs=0)
    assert basics.isco_orbital_frequency_hz(10 * MSUN) == pytest.approx(
        basics.isco_orbital_frequency_hz(MSUN) / 10, rel=1e-13, abs=0
    )


# ----------------------------------------------------------------------------- Kerr


def test_kerr_isco_limiting_values():
    # Known extreme values: prograde orbit r_G for a* = 1, retrograde orbit 9 r_G for a* = −1.
    assert basics.kerr_isco_radius_rg(1.0) == pytest.approx(1.0, abs=1e-12)
    assert basics.kerr_isco_radius_rg(-1.0) == pytest.approx(9.0, abs=1e-12)
    assert basics.kerr_isco_radius_rg(0.998) == pytest.approx(1.2370, abs=1e-4)


@pytest.mark.parametrize("a", np.linspace(-1.0, 1.0, 21))
def test_kerr_isco_radial_epicyclic_condition(a):
    # [Abramowicz2013] TeX label epicyclic-radial: (ω_r*)² ∝ 1 − 6/x + 8 a* x^{-3/2} − 3 a*² x^{-2};
    # the ISCO is the outermost radius at which this expression is zero (a* < 0: retrograde orbit). A check independent of the formula.
    x = basics.kerr_isco_radius_rg(a)
    assert 1.0 - 6.0 / x + 8.0 * a * x**-1.5 - 3.0 * a**2 / x**2 == pytest.approx(0.0, abs=1e-11)


@pytest.mark.parametrize("a", [-0.99, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 0.99])
def test_isco_specific_energy_circular_orbit_minimum(a):
    # Independent route: the circular-orbit energy E(x) = (x^{3/2} − 2 x^{1/2} + a) / (x^{3/4} √(x^{3/2} − 3 x^{1/2} + 2a))
    # (Bardeen-Press-Teukolsky, from memory) is smallest at the ISCO: dE/dx = 0.
    def energy(x):
        return (x**1.5 - 2 * x**0.5 + a) / (x**0.75 * np.sqrt(x**1.5 - 3 * x**0.5 + 2 * a))

    x0 = float(basics.kerr_isco_radius_rg(a))
    h = 1e-5 * x0
    derivative = (energy(x0 + h) - energy(x0 - h)) / (2 * h)
    assert abs(derivative) < 1e-8
    assert energy(1.02 * x0) > energy(x0)
    assert energy(x0) == pytest.approx(float(basics.kerr_isco_specific_energy(a)), rel=1e-9, abs=0)


def test_isco_energy_and_efficiency_values():
    assert basics.kerr_isco_specific_energy(0.0) == pytest.approx(np.sqrt(8.0 / 9.0), rel=1e-13, abs=0)
    assert basics.isco_radiative_efficiency(0.0) == pytest.approx(1.0 - np.sqrt(8.0 / 9.0), rel=1e-12, abs=0)
    # a* = 1: E = 1/√3, η = 1 − 1/√3 ≈ 0.4226.
    assert basics.isco_radiative_efficiency(1.0) == pytest.approx(1.0 - 1.0 / np.sqrt(3.0), rel=1e-10, abs=0)
    # [Inayoshi2020] §1: "~0.06" for a non-rotating black hole; the source gives one digit, tolerance 0.0035.
    assert abs(float(basics.isco_radiative_efficiency(0.0)) - 0.06) < 0.0035


@pytest.mark.parametrize("a", np.linspace(-1.0, 1.0, 9))
def test_horizon_kerr_metric_identity(a):
    # In the Kerr metric the horizon is the larger root of Δ = r² − 2 r + a² = 0 (in units of r_G).
    rp = basics.kerr_horizon_radius_rg(a)
    assert rp**2 - 2.0 * rp + a**2 == pytest.approx(0.0, abs=1e-14)


def test_horizon_and_ergosphere_values():
    assert basics.kerr_horizon_radius_rg(0.0) == 2.0
    assert basics.kerr_horizon_radius_rg(1.0) == 1.0
    assert basics.kerr_horizon_radius_rg(0.0) * basics.gravitational_radius_m(MSUN) == pytest.approx(
        basics.schwarzschild_radius_m(MSUN), rel=1e-14, abs=0
    )
    # Ergosphere: 2 r_G at the equator (for every spin), tangent to the horizon at the poles.
    assert basics.kerr_ergosphere_radius_rg(0.7) == pytest.approx(2.0, abs=1e-15)
    assert basics.kerr_ergosphere_radius_rg(0.7, 0.0) == pytest.approx(
        basics.kerr_horizon_radius_rg(0.7), rel=1e-14, abs=0
    )
    assert basics.kerr_ergosphere_radius_rg(0.0, 0.3) == pytest.approx(2.0, abs=1e-15)


def test_horizon_angular_velocity():
    m = 10.0 * MSUN
    # a* = 1: Ω_H = c³ / (2 G M).
    assert basics.horizon_angular_velocity_rad_s(m, 1.0) == pytest.approx(C**3 / (2 * G * m), rel=1e-14, abs=0)
    # Small a*: a* c³ / (4 G M).
    assert basics.horizon_angular_velocity_rad_s(m, 1e-5) == pytest.approx(1e-5 * C**3 / (4 * G * m), rel=1e-10, abs=0)
    # Independent route (Page2005 eq:3: Ω = 4π a / A, A = 4π (r_+² + a²)): Ω_H = a* c / r_G / (r_+² + a*²).
    a = 0.6
    rp = float(basics.kerr_horizon_radius_rg(a))
    expected = (C / basics.gravitational_radius_m(m)) * a / (rp**2 + a**2)
    assert basics.horizon_angular_velocity_rad_s(m, a) == pytest.approx(expected, rel=1e-13, abs=0)
    assert basics.horizon_angular_velocity_rad_s(m, -a) == pytest.approx(-expected, rel=1e-13, abs=0)


@pytest.mark.parametrize("a", np.linspace(-1.0, 1.0, 21))
def test_photon_orbit_cubic(a):
    # [Abramowicz2013]: root of the cubic y³ − 3y + 2a* = 0 for y = √(r/r_G) (Novikov-Thorne section).
    y = np.sqrt(basics.kerr_photon_orbit_radius_rg(a))
    assert y**3 - 3.0 * y + 2.0 * a == pytest.approx(0.0, abs=1e-12)


def test_photon_orbit_limiting_values():
    assert basics.kerr_photon_orbit_radius_rg(1.0) == pytest.approx(1.0, abs=1e-12)
    assert basics.kerr_photon_orbit_radius_rg(-1.0) == pytest.approx(4.0, abs=1e-12)


# ----------------------------------------------------------------------------- Hawking


def test_hawking_temperature():
    # [Abramowicz2013] §1 footnote: T_H ~ 1e-8 K for a stellar-mass black hole (order of magnitude).
    t = basics.hawking_temperature_k(MSUN)
    assert 1e-8 < t < 1e-7
    # Independent route 1: T = ħ κ / (2π c k_B), Schwarzschild surface gravity κ = c⁴ / (4 G M) [Page2005: κ = 1/(4M)].
    kappa = C**4 / (4.0 * G * MSUN)
    assert t == pytest.approx(HBAR * kappa / (2.0 * np.pi * C * BOLTZMANN.value), rel=1e-13, abs=0)
    # Independent route 2: T = ħ c / (4π k_B R_S).
    assert t == pytest.approx(
        HBAR * C / (4.0 * np.pi * BOLTZMANN.value * basics.schwarzschild_radius_m(MSUN)), rel=1e-13, abs=0
    )


def test_entropy_area_and_first_law():
    # S/k_B = A / (4 ℓ_P²), ℓ_P² = ħ G / c³, A = 4π R_S² [Page2005 eq:2: S = A/4].
    rs = basics.schwarzschild_radius_m(MSUN)
    s = basics.bekenstein_hawking_entropy(MSUN)
    assert s == pytest.approx(4.0 * np.pi * rs**2 / (4.0 * HBAR * G / C**3), rel=1e-13, abs=0)
    assert s == pytest.approx(1.05e77, rel=1e-2, abs=0)
    # First law: d(S k_B) = c² dM / T ⇒ d(S/k_B)/dM = c² / (k_B T). Central difference, relative step 1e-4.
    h = 1e-4 * MSUN
    derivative = (basics.bekenstein_hawking_entropy(MSUN + h) - basics.bekenstein_hawking_entropy(MSUN - h)) / (2 * h)
    assert derivative == pytest.approx(C**2 / (BOLTZMANN.value * basics.hawking_temperature_k(MSUN)), rel=1e-7, abs=0)


def test_evaporation_time_independent_integral():
    # dM/dt = −σ_SB T_H⁴ A / c²  ⇒  t = ∫_0^M c² dM' / (σ T⁴ A); independent numerical integral (in log space).
    def luminosity(m):
        t = basics.hawking_temperature_k(m)
        area = 4.0 * np.pi * basics.schwarzschild_radius_m(m) ** 2
        return STEFAN_BOLTZMANN.value * t**4 * area

    m0 = 1e12  # kg
    t, _ = quad(
        lambda u: np.exp(u) * C**2 / luminosity(np.exp(u)), np.log(1e-3 * m0), np.log(m0), epsabs=0, epsrel=1e-12
    )
    # Lower limit 1e-3 m0: the remaining contribution is of order (1e-3)³, a relative error of 1e-9.
    assert basics.hawking_evaporation_time_blackbody_s(m0) == pytest.approx(t, rel=2e-9, abs=0)
    # ≈ 2e67 yr for one solar mass.
    assert 1e67 < basics.hawking_evaporation_time_blackbody_s(MSUN) / YR < 3e67


# ----------------------------------------------------------------------------- binary


def test_chirp_mass_forms():
    m1, m2 = 36.0 * MSUN, 29.0 * MSUN
    m, mu, nu = m1 + m2, m1 * m2 / (m1 + m2), m1 * m2 / (m1 + m2) ** 2
    mc = basics.chirp_mass_kg(m1, m2)
    # [Sathyaprakash2009] eqn:chirpmassdef: 𝓜 = μ^{3/5} M^{2/5} = ν^{3/5} M.
    assert mc == pytest.approx(mu**0.6 * m**0.4, rel=1e-13, abs=0)
    assert mc == pytest.approx(nu**0.6 * m, rel=1e-13, abs=0)
    assert basics.chirp_mass_kg(m1, m1) == pytest.approx(m1 * 2.0**-0.2, rel=1e-14, abs=0)


def test_frequency_derivative_from_peters_equation():
    # Independent route: [Sathyaprakash2009] eqn:circularPM, Ṗ_b = −(192π/5) (2π 𝓜/P_b)^{5/3} (G = c = 1),
    # f = 2/P_b ⇒ ḟ = −2 Ṗ_b / P_b².
    mc = basics.chirp_mass_kg(1.4 * MSUN, 1.4 * MSUN)
    mc_s = G * mc / C**3
    for f in (10.0, 50.0, 300.0):
        pb = 2.0 / f
        pb_dot = -192.0 * np.pi / 5.0 * (2.0 * np.pi * mc_s / pb) ** (5.0 / 3.0)
        assert basics.gw_frequency_derivative_hz_s(mc, f) == pytest.approx(-2.0 * pb_dot / pb**2, rel=1e-12, abs=0)


def test_time_to_merger_integral_and_nu_form():
    m1, m2 = 1.4 * MSUN, 1.4 * MSUN
    mc = basics.chirp_mass_kg(m1, m2)
    f0 = 20.0
    # Independent route 1: τ = ∫_f0^∞ df / ḟ.
    t, _ = quad(lambda f: 1.0 / basics.gw_frequency_derivative_hz_s(mc, f), f0, np.inf, epsrel=1e-12)
    assert basics.time_to_merger_s(mc, f0) == pytest.approx(t, rel=1e-9, abs=0)
    # Independent route 2: the ν form in the source, τ0 = 5 / (256 π ν f) (π M f)^{-5/3}, G = c = 1.
    m_s = G * (m1 + m2) / C**3
    nu = m1 * m2 / (m1 + m2) ** 2
    assert basics.time_to_merger_s(mc, f0) == pytest.approx(
        5.0 / (256.0 * np.pi * nu * f0) * (np.pi * m_s * f0) ** (-5.0 / 3.0), rel=1e-12, abs=0
    )
    # Order-of-magnitude check: a 1.4 + 1.4 M☉ binary has ~158 s left until merger at 20 Hz at Newtonian order (from this calculation, not from the source).
    assert basics.time_to_merger_s(mc, f0) == pytest.approx(158.0, rel=2e-2, abs=0)


def test_frequency_from_remaining_time_round_trip():
    mc = basics.chirp_mass_kg(30.0 * MSUN, 30.0 * MSUN)
    f = np.array([10.0, 35.0, 100.0, 250.0])
    assert basics.gw_frequency_from_time_to_merger_hz(mc, basics.time_to_merger_s(mc, f)) == pytest.approx(
        f, rel=1e-13, abs=0
    )


def test_isco_wave_frequency():
    assert basics.gw_frequency_at_isco_hz(MSUN) == pytest.approx(
        2.0 * basics.isco_orbital_frequency_hz(MSUN), rel=1e-14, abs=0
    )
    # c³ / (6^{3/2} π G M☉) ≈ 4.4 kHz (a commonly quoted magnitude; not read from a source, 1%).
    assert basics.gw_frequency_at_isco_hz(MSUN) == pytest.approx(4.4e3, rel=1e-2, abs=0)
    assert basics.gw_frequency_at_isco_hz(20.0 * MSUN) == pytest.approx(
        basics.gw_frequency_at_isco_hz(MSUN) / 20.0, rel=1e-13, abs=0
    )


# ----------------------------------------------------------------------------- accretion


def test_eddington_luminosity():
    # [Abramowicz2013] §1 footnote: L_Edd ≡ 1.2e38 (M/M☉) erg/s. The source gives two digits; for hydrogen
    # 4π G M m_p c / σ_T = 1.26e38 erg/s, hence the tolerance of 6%.
    l_edd = accretion.eddington_luminosity_w(MSUN)
    assert l_edd / ERG_S_W == pytest.approx(1.2e38, rel=6e-2, abs=0)
    kappa = THOMSON_CROSS_SECTION.value / PROTON_MASS.value
    assert l_edd == pytest.approx(4.0 * np.pi * G * MSUN * C / kappa, rel=1e-13, abs=0)


def test_eddington_accretion_rate_table_3():
    # [Inayoshi2020] Table 3: Ṁ_Edd = L_Edd/(0.1 c²) ≈ 2.3e-5 M☉/yr (10³ M☉). 5% because κ_es is not given.
    mdot = accretion.eddington_accretion_rate_kg_s(1e3 * MSUN)
    assert mdot * YR / MSUN == pytest.approx(2.3e-5, rel=5e-2, abs=0)
    # L = η Ṁ c² round trip and 1/η scaling.
    assert accretion.accretion_luminosity_w(mdot, 0.1) == pytest.approx(
        accretion.eddington_luminosity_w(1e3 * MSUN), rel=1e-13, abs=0
    )
    assert accretion.eddington_accretion_rate_kg_s(MSUN, 0.2) == pytest.approx(
        accretion.eddington_accretion_rate_kg_s(MSUN, 0.1) / 2, rel=1e-14, abs=0
    )


def test_salpeter_time_coefficients_and_exponential_growth():
    # [Inayoshi2020]: coefficient 0.45 Gyr; t_Edd ≈ 50 Myr for η = 0.1 (1%: the source gives two digits).
    t_sal = accretion.salpeter_time_s(0.1)
    assert t_sal / (1e9 * YR) == pytest.approx(0.45 * 0.1 / 0.9, rel=1e-2, abs=0)
    assert t_sal / (1e6 * YR) == pytest.approx(50.0, rel=1e-2, abs=0)
    # Independent route: numerical solution of dM/dt = (1 − η) Ṁ_Edd(M); at t = t_Sal, M = e M0.
    eta = 0.1
    m0 = 100.0 * MSUN

    def rhs(t, m):
        return (1.0 - eta) * accretion.eddington_accretion_rate_kg_s(m, eta)

    solution = solve_ivp(rhs, (0.0, t_sal), [m0], rtol=1e-11, atol=0.0, method="DOP853")
    assert solution.y[0, -1] / m0 == pytest.approx(np.e, rel=1e-8, abs=0)


def test_bondi_table_3():
    # [Inayoshi2020] Table 3, T = 10³ K, n_H = 1e4 cm^-3, M = 10³ M☉. The source does not give the definition of c_s and ρ;
    # c_s² = k T / (μ m_p), ρ = μ m_p n_H, μ = 1.22 is an ASSUMPTION and was chosen to match the table: R_B ∝ μ,
    # Ṁ_B ∝ μ^{5/2}, so this test checks the prefactor (a wrong prefactor is caught at 5%) but does not check μ.
    mu = 1.22
    cs = np.sqrt(BOLTZMANN.value * 1e3 / (mu * PROTON_MASS.value))
    rho = mu * PROTON_MASS.value * 1e10  # m^-3
    assert accretion.bondi_radius_m(1e3 * MSUN, cs) / PC == pytest.approx(
        0.6, rel=0.10, abs=0
    )  # one digit in the table
    assert accretion.bondi_accretion_rate_kg_s(1e3 * MSUN, rho, cs) * YR / MSUN == pytest.approx(
        4.5e-3, rel=0.05, abs=0
    )
    # Identity: G²M²/c_s³ = R_B² c_s.
    rb = accretion.bondi_radius_m(1e3 * MSUN, cs)
    assert accretion.bondi_accretion_rate_kg_s(1e3 * MSUN, rho, cs) == pytest.approx(
        np.pi * np.exp(1.5) * rho * rb**2 * cs, rel=1e-13, abs=0
    )


R_INNER = 6.0 * G * 10 * MSUN / C**2  # 10 M☉ black hole, ISCO (Schwarzschild)
MDOT_DISK = 1e-8 * MSUN / YR


def test_thin_disk_profile_limits():
    m = 10 * MSUN
    assert accretion.thin_disk_effective_temperature_k(R_INNER, m, MDOT_DISK, R_INNER) == 0.0
    # r ≫ r_in: T ∝ r^{-3/4}; the correction 1 − √(r_in/r) is ~1e-3 at r = 1e6 r_in.
    t1 = accretion.thin_disk_effective_temperature_k(1e6 * R_INNER, m, MDOT_DISK, R_INNER)
    t2 = accretion.thin_disk_effective_temperature_k(1e7 * R_INNER, m, MDOT_DISK, R_INNER)
    assert t2 / t1 == pytest.approx(10.0**-0.75, rel=2e-3, abs=0)
    with pytest.raises(ValueError):
        accretion.thin_disk_effective_temperature_k(0.5 * R_INNER, m, MDOT_DISK, R_INNER)


def test_thin_disk_energy_conservation():
    # Independent check: the total power radiated from both faces must be
    # L = G M Ṁ / (2 r_in) in a Newtonian disk with zero torque at r_in (half of the Kepler energy is radiated).
    m = 10 * MSUN

    def flux(u):
        # Change to the variable u = √(r_in/r) ∈ (0, 1]: r = r_in/u², dr = 2 r_in/u³ du (the infinite range becomes finite).
        r = R_INNER / u**2
        emission = (
            2.0 * STEFAN_BOLTZMANN.value * accretion.thin_disk_effective_temperature_k(r, m, MDOT_DISK, R_INNER) ** 4
        )
        return emission * 2.0 * np.pi * r * 2.0 * R_INNER / u**3

    # Lower limit 1e-9: the contribution from u → 0 is ~1e-18 relative, from the ∝ u behavior of the integrand.
    total, _ = quad(flux, 1e-9, 1.0, epsrel=1e-12, limit=200)
    assert total == pytest.approx(G * m * MDOT_DISK / (2.0 * R_INNER), rel=1e-8, abs=0)


def test_thin_disk_peak_temperature():
    m = 10 * MSUN
    r_peak = accretion.thin_disk_peak_temperature_radius_m(R_INNER)
    assert r_peak == pytest.approx(49.0 / 36.0 * R_INNER, rel=1e-15, abs=0)
    # Independent route: numerical maximum of the profile.
    result = minimize_scalar(
        lambda r: -accretion.thin_disk_effective_temperature_k(r, m, MDOT_DISK, R_INNER),
        bounds=(1.0001 * R_INNER, 20.0 * R_INNER),
        method="bounded",
        options={"xatol": 1e-9 * R_INNER},
    )
    assert result.x == pytest.approx(r_peak, rel=1e-5, abs=0)
    assert accretion.thin_disk_peak_temperature_k(m, MDOT_DISK, R_INNER) == pytest.approx(-result.fun, rel=1e-10, abs=0)
    assert accretion.thin_disk_peak_temperature_k(m, MDOT_DISK, R_INNER) == pytest.approx(
        accretion.thin_disk_effective_temperature_k(r_peak, m, MDOT_DISK, R_INNER), rel=1e-13, abs=0
    )
    # Constant factor: (6^6 / 7^7)^{1/4} ≈ 0.488.
    t_ref = (3.0 * G * m * MDOT_DISK / (8.0 * np.pi * STEFAN_BOLTZMANN.value * R_INNER**3)) ** 0.25
    assert accretion.thin_disk_peak_temperature_k(m, MDOT_DISK, R_INNER) / t_ref == pytest.approx(0.488, abs=5e-4)


def test_tidal_radius_source_coefficient():
    # [Komossa2015] §1: r_t ≃ 7e12 cm (M/10⁶ M☉)^{1/3} (m*/M☉)^{-1/3} (r*/R☉); the source rounds after one
    # digit (6.96e12 → 7e12), tolerance 1%.
    r_sun = 6.957e8
    rt = accretion.tidal_disruption_radius_m(1e6 * MSUN, MSUN, r_sun)
    assert rt * 100.0 == pytest.approx(7e12, rel=1e-2, abs=0)
    # 8 times the mass → 2 times the radius.
    assert accretion.tidal_disruption_radius_m(8e6 * MSUN, MSUN, r_sun) == pytest.approx(2.0 * rt, rel=1e-13, abs=0)


def test_hills_mass():
    r_sun = 6.957e8
    mh = accretion.hills_mass_kg(MSUN, r_sun)
    # [Komossa2015] §1: a Sun-like star is swallowed whole above ≳ 1e8 M☉ (order of magnitude; within a factor of 2).
    assert 0.5e8 < mh / MSUN < 2e8
    # White dwarf (0.6 M☉, 0.01 R☉): the source says "disrupted below ≲ 1e5 M☉" (order of magnitude; between 0.5e5 and 3e5).
    assert 0.5e5 < accretion.hills_mass_kg(0.6 * MSUN, 0.01 * r_sun) / MSUN < 3e5
    # Identity: r_t(M_H) = R_S(M_H).
    assert accretion.tidal_disruption_radius_m(mh, MSUN, r_sun) == pytest.approx(
        basics.schwarzschild_radius_m(mh), rel=1e-12, abs=0
    )


# ----------------------------------------------------------------------------- input validation


@pytest.mark.parametrize(
    "call",
    [
        lambda: basics.gravitational_radius_m(-1.0),
        lambda: basics.kerr_isco_radius_rg(1.1),
        lambda: basics.kerr_horizon_radius_rg(-1.5),
        lambda: basics.chirp_mass_kg(0.0, 1.0),
        lambda: basics.time_to_merger_s(1.0, -1.0),
        lambda: accretion.salpeter_time_s(1.0),
        lambda: accretion.eddington_accretion_rate_kg_s(MSUN, 0.0),
        lambda: accretion.bondi_radius_m(MSUN, 0.0),
    ],
)
def test_invalid_input(call):
    with pytest.raises(ValueError):
        call()


def test_scalar_and_array():
    assert np.ndim(basics.kerr_isco_radius_rg(0.5)) == 0
    assert basics.kerr_isco_radius_rg(np.zeros((2, 3)) + 0.5).shape == (2, 3)
    assert basics.hawking_temperature_k(np.array([MSUN, 2 * MSUN])).shape == (2,)
    assert np.ndim(accretion.eddington_luminosity_w(MSUN)) == 0


def test_hawking_greybody_page_eq15():
    from scipy.integrate import quad

    # [Page2005] eq:15: coefficient 3.3638e-5 for photons only; blackbody estimate 1/(15360π) = 2.0723e-5.
    m0 = 1e12
    blackbody_rate = m0 / (3.0 * basics.hawking_evaporation_time_blackbody_s(m0))  # t = M/(3 |dM/dt|)
    photon_only = basics.hawking_mass_loss_rate_kg_s(m0, spin_one=1.0, spin_two=0.0)
    assert float(photon_only) / float(blackbody_rate) == pytest.approx(3.3638e-5 * 15360 * np.pi, rel=1e-12, abs=0)
    assert float(photon_only) / float(blackbody_rate) == pytest.approx(1.623, abs=2e-3)
    # M⁻² scaling and additivity of the species coefficients.
    assert float(basics.hawking_mass_loss_rate_kg_s(2 * m0)) == pytest.approx(
        float(basics.hawking_mass_loss_rate_kg_s(m0)) / 4, rel=1e-6, abs=0
    )
    total = basics.hawking_mass_loss_rate_kg_s(m0, spin_half=3.0)
    assert float(total) / float(basics.hawking_mass_loss_rate_kg_s(m0)) == pytest.approx(
        (3 * 8.1830 + 3.3638 + 0.3836) / (3.3638 + 0.3836), rel=1e-12, abs=0
    )
    # The time is the numerical integral of the rate: t = ∫ dM / |dM/dt|.
    t_numeric, _ = quad(lambda m: 1.0 / float(basics.hawking_mass_loss_rate_kg_s(m)), 0.0, m0)
    assert float(basics.hawking_evaporation_time_s(m0)) == pytest.approx(t_numeric, rel=1e-8, abs=0)
    # One solar mass, photon + graviton: about 1.16e67 yr; shorter than the blackbody estimate.
    assert float(basics.hawking_evaporation_time_s(MSUN)) / YR == pytest.approx(1.16e67, rel=0.02, abs=0)
    assert basics.hawking_evaporation_time_s(MSUN) < basics.hawking_evaporation_time_blackbody_s(MSUN)


# --------------------------------------------------------------------------------------
# Added after the audit of 26 November 2026: the tests above could not tell these cases apart.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("spin", [0.3, 0.9, -0.9, 1.0])
@pytest.mark.parametrize("theta", [0.4, 0.7, 1.2, 2.5])
def test_ergosphere_is_where_g_tt_of_the_kerr_metric_vanishes(spin, theta):
    # g_tt = -(1 - 2 r / (r^2 + a^2 cos^2 theta)) with M = 1, written here from the Boyer-Lindquist line element
    r = float(basics.kerr_ergosphere_radius_rg(spin, theta))
    assert 1.0 - 2.0 * r / (r**2 + spin**2 * np.cos(theta) ** 2) == pytest.approx(0.0, abs=1e-14)
    assert r >= float(basics.kerr_horizon_radius_rg(spin))  # the outer of the two roots


@pytest.mark.parametrize("spin", [0.1, 0.3, 0.6, 0.9, -0.3, -0.9])
def test_photon_orbit_is_the_outer_root_of_the_cubic(spin):
    # y^3 - 3 y + 2 a = 0 has two positive roots for 0 < a < 1 (a = 0.3: r = 0.041 and 2.630); the orbit is the outer one
    roots = np.roots([1.0, 0.0, -3.0, 2.0 * spin])
    outer = max(float(np.real(y)) for y in roots if abs(np.imag(y)) < 1e-12 and np.real(y) > 0.0)
    assert float(basics.kerr_photon_orbit_radius_rg(spin)) == pytest.approx(outer**2, rel=1e-12, abs=0)
    assert float(basics.kerr_photon_orbit_radius_rg(spin)) > float(basics.kerr_horizon_radius_rg(spin))


def test_eddington_luminosity_and_hills_mass_against_hand_numbers():
    # 4 pi G M m_p c / sigma_T with the CODATA 2022 and IAU 2015 numbers typed here, not imported:
    # G M_sun = 1.3271244e20 m^3 s^-2, m_p = 1.67262192595e-27 kg, c = 299792458 m/s, sigma_T = 6.6524587051e-29 m^2
    by_hand = 4.0 * np.pi * 1.3271244e20 * 1.67262192595e-27 * 299792458.0 / 6.6524587051e-29
    assert by_hand == pytest.approx(1.25707e31, rel=1e-5, abs=0)  # W per solar mass
    assert float(accretion.eddington_luminosity_w(MSUN)) == pytest.approx(by_hand, rel=1e-13, abs=0)
    # Hills mass of the Sun: (R c^2 / 2 G)^{3/2} / sqrt(M), R = 6.957e8 m: 1.1434e8 solar masses
    hills = float(accretion.hills_mass_kg(MSUN, 6.957e8)) / MSUN
    assert hills == pytest.approx((6.957e8 * 299792458.0**2 / (2.0 * 1.3271244e20)) ** 1.5, rel=1e-12, abs=0)
    assert hills == pytest.approx(1.1434e8, rel=1e-4, abs=0)


def test_hawking_species_counts_are_validated():
    for counts in ({"spin_one": -5, "spin_two": 0}, {"spin_half": 0, "spin_one": 0, "spin_two": 0}):
        with pytest.raises(ValueError, match="species"):
            basics.hawking_mass_loss_rate_kg_s(1e12, **counts)
        with pytest.raises(ValueError, match="species"):
            basics.hawking_evaporation_time_s(1e12, **counts)
