"""Numerical values from the sources and closed-form solutions for the star formation module.

Sources: [McKee2007] (arXiv:0707.3514), [Krumholz2015] (arXiv:1511.03457),
[Kennicutt1998] (astro-ph/9712213), [Kennicutt2012] (arXiv:1204.3552).
The sources give their numbers rounded, so the relative tolerances are
justified next to each comparison.
"""

import math

import numpy as np
import pytest

from kozmos.core.constants import BOLTZMANN, GM_SUN, GRAVITATIONAL_CONSTANT, JULIAN_YEAR_SECONDS, PARSEC, PROTON_MASS
from kozmos.star_formation import clouds, laws

G = GRAVITATIONAL_CONSTANT.value
M_H = PROTON_MASS.value
PC = PARSEC.value
M_SUN = GM_SUN.value / G
YR = JULIAN_YEAR_SECONDS.value
MYR = 1e6 * YR
# P/k_B = 3e5 cm^-3 K (typical GMC pressure that appears in both sources) -> Pa
P_TYPICAL = 3e5 * BOLTZMANN.value * 1e6


# --- gas basics ----------------------------------------------------------------------


def test_sound_speed_source_value():
    # [Krumholz2015] Section 4: 0.18 km/s for 10 K, μ = 2.33. The true value is 0.188 km/s;
    # the source rounds to two significant digits, so the tolerance is 5 %.
    cs = clouds.sound_speed_m_s(10.0)
    assert cs == pytest.approx(0.18e3, rel=0.05)
    # Exact relation: c_s² = k T / (μ m_H)
    assert cs**2 == pytest.approx(BOLTZMANN.value * 10.0 / (2.33 * M_H), rel=1e-13)


def test_sound_speed_scaling():
    assert clouds.sound_speed_m_s(40.0) / clouds.sound_speed_m_s(10.0) == pytest.approx(2.0, rel=1e-13)
    # If μ increases fourfold, c_s is halved.
    assert clouds.sound_speed_m_s(10.0, 4 * 2.33) / clouds.sound_speed_m_s(10.0, 2.33) == pytest.approx(0.5, rel=1e-13)


def test_mach_number():
    cs = clouds.sound_speed_m_s(10.0)
    assert clouds.mach_number(cs, cs) == pytest.approx(1.0, rel=1e-14)
    # [Krumholz2015] Section 4: the speed is ≈ 20 c_s for M ~ 20; only the definition is tested here.
    assert clouds.mach_number(20 * cs, cs) == pytest.approx(20.0, rel=1e-13)


def test_density_conversions_round_trip():
    n = np.array([1e6, 1e9, 1e12])
    assert np.allclose(clouds.number_density_m3(clouds.mass_density_kg_m3(n)), n, rtol=1e-13)
    s = np.array([1e20, 1e24, 1e26])
    assert np.allclose(clouds.column_density_m2(clouds.surface_density_kg_m2(s)), s, rtol=1e-13)
    # Definition ρ = μ m_H n
    assert clouds.mass_density_kg_m3(1.0, 2.3) == pytest.approx(2.3 * M_H, rel=1e-14)


def test_gmc_number_density_krumholz():
    # [Krumholz2015] Section 8: n = 23 cm⁻³ for Σ = 100 M☉/pc², M = 1e6 M☉ (μ = 2.3).
    # The source gives two significant digits (our value is 23.4); tolerance 3 %.
    m = 1e6 * M_SUN
    sigma = 100 * M_SUN / PC**2
    r = math.sqrt(m / (math.pi * sigma))
    rho = 3 * m / (4 * math.pi * r**3)
    n_cm3 = clouds.number_density_m3(rho, 2.3) * 1e-6
    assert n_cm3 == pytest.approx(23.0, rel=0.03)


# --- stability -----------------------------------------------------------------------


def test_jeans_length_krumholz_value():
    # [Krumholz2015] Section 6: λ_J = 3.4 pc for c_s = 0.2 km/s, ρ = 100 m_p.
    lam = clouds.jeans_length_m(0.2e3, 100 * M_H * 1e6) / PC  # 100 m_p cm^-3
    assert lam == pytest.approx(3.4, rel=0.02)  # the source gives 2 significant digits (3.44)


def test_jeans_length_identity():
    # λ_J² G ρ = π c_s² ([McKee2007] LJeans); exact equality for different inputs.
    cs = np.array([100.0, 300.0, 1e3])
    rho = np.array([1e-18, 1e-20, 1e-22])
    assert np.allclose(clouds.jeans_length_m(cs, rho) ** 2 * G * rho, math.pi * cs**2, rtol=1e-13)


def test_jeans_mass_two_definitions_and_bonnor_ebert():
    cs, rho = 190.0, 1e-18
    sphere = clouds.jeans_mass_kg(cs, rho, "sphere")
    cube = clouds.jeans_mass_kg(cs, rho, "cube")
    # Ratio of the two definitions (4π/3)/(1) ⇒ π/6 ÷ 1/8 = 4π/3.
    assert sphere / cube == pytest.approx(4 * math.pi / 3, rel=1e-13)
    # [McKee2007] after LJeans: M_J = 2.47 M_BE (P_0 = ρ_0 c_s²). The source has three digits; tolerance 0.5 %.
    m_be = clouds.bonnor_ebert_critical_mass_kg(cs, rho * cs**2)
    assert sphere / m_be == pytest.approx(2.47, rel=0.005)
    # Scaling relation: M_J = coefficient · c_s³ / sqrt(G³ρ)
    assert sphere == pytest.approx(math.pi**2.5 / 6 * cs**3 / math.sqrt(G**3 * rho), rel=1e-13)
    assert cube == pytest.approx(math.pi**1.5 / 8 * cs**3 / math.sqrt(G**3 * rho), rel=1e-13)


def test_jeans_mass_invalid_definition():
    with pytest.raises(ValueError):
        clouds.jeans_mass_kg(190.0, 1e-18, "cylinder")


def test_bonnor_ebert_source_values():
    # T = 10 K, P/k_B = 3e5 cm⁻³ K, μ = 2.33: [McKee2007] 0.66 M☉; [Krumholz2015] probsol2 0.67 M☉.
    # The sources round c_s and μ: tolerance 3 % for McKee, 1 % for Krumholz.
    m = clouds.bonnor_ebert_critical_mass_kg(clouds.sound_speed_m_s(10.0, 2.33), P_TYPICAL) / M_SUN
    assert m == pytest.approx(0.66, rel=0.03)
    assert m == pytest.approx(0.67, rel=0.01)


def test_bonnor_ebert_coefficients_from_ode():
    m_max, contrast, xi_s = clouds.bonnor_ebert_critical_coefficients()
    # [Krumholz2015] probsol2: m_max = 1.18; ρ_c/ρ_s = 14.0 (three significant digits, tolerance 0.5 %).
    assert m_max == pytest.approx(1.18, rel=0.005)
    assert contrast == pytest.approx(14.0, rel=0.005)
    # [McKee2007] MBE_def: 1.182; agreement with the ODE solution to 4 significant digits (tolerance 5e-4).
    assert m_max == pytest.approx(1.182, rel=5e-4)
    # [McKee2007]: R_BE = 0.486 c_s/sqrt(G ρ_0); in the ODE R/r_0 = ξ_s, r_0 = c_s/sqrt(4πGρ_c) ⇒ 0.486 = ξ_s/sqrt(4π ρ_c/ρ_s).
    assert xi_s / math.sqrt(4 * math.pi * contrast) == pytest.approx(0.486, rel=1e-3)


def test_bonnor_ebert_applied_coefficient_consistent_with_ode():
    # The constants 1.182 and 0.486 written in the code must agree with the ODE solution at the same rounding.
    m_max, contrast, xi_s = clouds.bonnor_ebert_critical_coefficients()
    cs, p = 200.0, 1e-12
    assert clouds.bonnor_ebert_critical_mass_kg(cs, p) / (m_max * cs**4 / math.sqrt(G**3 * p)) == pytest.approx(
        1.0, abs=5e-4
    )
    rho0 = 1e-18
    r_ode = xi_s / math.sqrt(4 * math.pi * contrast) * cs / math.sqrt(G * rho0)
    assert clouds.bonnor_ebert_radius_m(cs, rho0) / r_ode == pytest.approx(1.0, abs=1e-3)


def test_bonnor_ebert_characteristic_mass_ratio():
    # [McKee2007]: M_BE is "nearly equal" to M_G = σ³/sqrt(G³ρ) at the surface conditions (ratio 1.182).
    cs, rho0 = 200.0, 1e-18
    m_g = cs**3 / math.sqrt(G**3 * rho0)
    assert clouds.bonnor_ebert_critical_mass_kg(cs, rho0 * cs**2) / m_g == pytest.approx(1.182, rel=1e-13)


def test_virial_parameter_definition():
    m, r = 1e35, 3e17
    sigma = math.sqrt(G * m / (5 * r))
    assert clouds.virial_parameter(sigma, r, m) == pytest.approx(1.0, rel=1e-13)
    # The surface density form gives the same number for M = π R² Σ.
    s = m / (math.pi * r**2)
    assert clouds.virial_parameter_from_surface_density(sigma, r, s) == pytest.approx(1.0, rel=1e-13)
    # Larson's three laws: σ ∝ R^0.5, constant Σ ⇒ constant α_vir; if σ doubles, α_vir quadruples.
    assert clouds.virial_parameter(2 * sigma, r, m) == pytest.approx(4.0, rel=1e-13)


def test_virial_krumholz_numerical():
    # [Krumholz2015] Section 8: α_vir = 3.7 for σ = σ_pc R_pc^0.5 with σ_pc = 1 km/s, Σ = 100 M☉/pc².
    sigma_pc = 1e3
    r = 7.3 * PC  # Given Σ and σ_pc the result is independent of R; an arbitrary radius
    sigma = sigma_pc * math.sqrt(r / PC)
    alpha = clouds.virial_parameter_from_surface_density(sigma, r, 100 * M_SUN / PC**2)
    assert alpha == pytest.approx(3.7, rel=0.01)  # the source gives two digits (3.70)


def test_free_fall_time_mckee():
    # [McKee2007] eq:tff: 1.37e6 yr for n_H = 1e3 cm⁻³; He fraction 10 % (by number) ⇒ ρ = 1.4 m_H n_H.
    rho = 1.4 * M_H * 1e3 * 1e6
    assert clouds.free_fall_time_s(rho) / YR == pytest.approx(1.37e6, rel=0.01)  # 3 significant digits
    # t_ff ∝ ρ^-1/2
    assert clouds.free_fall_time_s(4 * rho) / clouds.free_fall_time_s(rho) == pytest.approx(0.5, rel=1e-13)


def test_timescales_krumholz_surface_density_form():
    # [Krumholz2015] Section 8: t_ff = π^{1/4}/sqrt(8G) (M/Σ³)^{1/4}, t_cr = 0.95/sqrt(α G) (M/Σ³)^{1/4}.
    m = 1e6 * M_SUN
    sigma_surf = 100 * M_SUN / PC**2
    r = math.sqrt(m / (math.pi * sigma_surf))
    rho = 3 * m / (4 * math.pi * r**3)
    tff = clouds.free_fall_time_s(rho)
    scale = (m / sigma_surf**3) ** 0.25
    assert tff == pytest.approx(math.pi**0.25 / math.sqrt(8 * G) * scale, rel=1e-12)
    assert tff / MYR == pytest.approx(7.0, rel=0.01)  # source "7.0 Myr"
    # Virial cloud (α = 1): t_cr = R/σ, σ² = G M/(5R)
    sigma = math.sqrt(G * m / (5 * r))
    tcr = clouds.crossing_time_s(r, sigma)
    assert tcr == pytest.approx(math.sqrt(5.0) * math.pi**-0.75 / math.sqrt(G) * scale, rel=1e-12)
    assert tcr / MYR == pytest.approx(14.0, rel=0.02)  # source "14 Myr" (ours 14.1)
    # Source: for α_vir = 1 the free-fall time is half the crossing time (exact ratio 2.01).
    assert tcr / tff == pytest.approx(2.0, rel=0.01)


def test_jeans_length_consistent_with_sound_crossing_and_free_fall():
    # λ_J / (c_s t_ff) = sqrt(π) / sqrt(3π/32) = sqrt(32/3): constant for every c_s, ρ.
    for cs, rho in [(190.0, 1e-18), (1e3, 1e-21), (50.0, 1e-15)]:
        ratio = clouds.jeans_length_m(cs, rho) / (cs * clouds.free_fall_time_s(rho))
        assert ratio == pytest.approx(math.sqrt(32.0 / 3.0), rel=1e-13)
    # The sound crossing time of half a Jeans length is sqrt(8/3) times t_ff.
    cs, rho = 190.0, 1e-18
    t_sound = clouds.crossing_time_s(clouds.jeans_length_m(cs, rho) / 2, cs)
    assert t_sound / clouds.free_fall_time_s(rho) == pytest.approx(math.sqrt(8.0 / 3.0), rel=1e-13)


# --- magnetic support ----------------------------------------------------------------


def test_magnetic_critical_mass_sheet_and_sphere_consistent():
    # [McKee2007]: c_Φ = 0.17 ≈ 1/(2π) = critical value of the infinite sheet G^{1/2}(Σ/B)_cr = 1/(2π).
    # Sphere of radius R with uniform B: Φ = π R² B, mass M = π R² Σ ⇒ M = M_Φ ⇔ Σ = Σ_cr.
    r, b = 3e16, 1e-9  # 1e-9 T = 10 µG
    flux = math.pi * r**2 * b
    m_cr = clouds.magnetic_critical_mass_kg(flux, c_phi=1.0 / (2 * math.pi))
    s_cr = m_cr / (math.pi * r**2)
    assert clouds.normalized_mass_to_flux_ratio(s_cr, b) == pytest.approx(1.0, rel=1e-12)
    # The default 0.17 is 7 % from 1/(2π) = 0.159; the source calls the two "essentially the same".
    assert clouds.magnetic_critical_mass_kg(flux) / m_cr == pytest.approx(1.0, abs=0.08)


def test_magnetic_critical_mass_with_virial_derivation():
    # [Krumholz2015] Section 6: for a uniform sphere B_energy = Φ²/(6π²R) (Gaussian) and W = −(3/5)GM²/R;
    # for c_Φ = sqrt(5/2)/(3π) the two are equal at M = M_Φ. Tested in SI.
    r, flux = 3e16, 2e8
    c = math.sqrt(2.5) / (3 * math.pi)
    assert c == pytest.approx(0.17, abs=0.005)  # source: "0.17"
    m = clouds.magnetic_critical_mass_kg(flux, c_phi=c)
    magnetic_energy_j = (flux * 1e8) ** 2 / (6 * math.pi**2 * (r * 100.0)) * 1e-7
    gravity_energy_j = 0.6 * G * m**2 / r
    assert magnetic_energy_j == pytest.approx(gravity_energy_j, rel=1e-12)


def test_magnetic_linear_and_invalid():
    assert clouds.magnetic_critical_mass_kg(3e8) / clouds.magnetic_critical_mass_kg(1e8) == pytest.approx(
        3.0, rel=1e-13
    )
    with pytest.raises(ValueError):
        clouds.magnetic_critical_mass_kg(-1.0)
    with pytest.raises(ValueError):
        clouds.magnetic_critical_mass_kg(1e8, c_phi=0.0)


# --- Larson, accretion, star formation rate ------------------------------------------


def test_linewidth_size():
    # [McKee2007] GMC_lws: δv = 0.9 L^0.56 km/s (default).
    assert clouds.linewidth_size_km_s(1.0) == pytest.approx(0.9, rel=1e-14)
    assert clouds.linewidth_size_km_s(10.0) == pytest.approx(0.9 * 10**0.56, rel=1e-13)
    # Solomon fit quoted in [Krumholz2015] Section 8: 0.72 R^0.5.
    assert clouds.linewidth_size_km_s(4.0, 0.72, 0.5) == pytest.approx(1.44, rel=1e-13)
    # [McKee2007]: the sonic length (δv = c_s) is about 0.03 pc ... with this normalization 0.19 km/s corresponds to 0.06 pc
    # (the source says "depends on the definition of size"; agreement within a factor of 2 is expected).
    l_sonic = (0.19 / 0.9) ** (1 / 0.56)
    assert 0.03 / 2 < l_sonic < 0.03 * 3


def test_shu_accretion_rate():
    # [McKee2007] eq:mdssis: 1.54e-6 M☉/yr for 10 K (μ = 2.33; the source has 3 significant digits).
    cs = clouds.sound_speed_m_s(10.0, 2.33)
    mdot = clouds.isothermal_collapse_accretion_rate_kg_s(cs) / M_SUN * YR
    assert mdot == pytest.approx(1.54e-6, rel=0.01)
    # T^{3/2} scaling
    ratio = clouds.isothermal_collapse_accretion_rate_kg_s(
        clouds.sound_speed_m_s(40.0)
    ) / clouds.isothermal_collapse_accretion_rate_kg_s(cs)
    assert ratio == pytest.approx(8.0, rel=1e-12)
    # [McKee2007] eq:mds: Ṁ ~ M_G / t_G = c_s³/G (M_G = σ³/sqrt(G³ρ), t_G = 1/sqrt(Gρ))
    rho = 1e-17
    m_g, t_g = cs**3 / math.sqrt(G**3 * rho), 1 / math.sqrt(G * rho)
    assert clouds.isothermal_collapse_accretion_rate_kg_s(cs, coefficient=1.0) == pytest.approx(m_g / t_g, rel=1e-12)


def test_cloud_sfr_and_efficiency_round_trip():
    m, rho = 1e5 * M_SUN, 1e-19
    mdot = clouds.cloud_star_formation_rate_kg_s(m, rho, 0.01)
    tff = clouds.free_fall_time_s(rho)
    # Depletion time t_dep = M/Ṁ = t_ff/ε_ff (definition in [Krumholz2015] Section 8)
    assert m / mdot == pytest.approx(tff / 0.01, rel=1e-13)
    assert clouds.free_fall_efficiency(tff, m / mdot) == pytest.approx(0.01, rel=1e-13)
    # ε_ff = 1 ⇒ Ṁ = M/t_ff
    assert clouds.cloud_star_formation_rate_kg_s(m, rho, 1.0) == pytest.approx(m / tff, rel=1e-13)


def test_invalid_inputs_clouds():
    with pytest.raises(ValueError):
        clouds.sound_speed_m_s(-5.0)
    with pytest.raises(ValueError):
        clouds.jeans_length_m(0.0, 1e-18)
    with pytest.raises(ValueError):
        clouds.free_fall_time_s(np.array([1e-18, -1e-18]))
    with pytest.raises(ValueError):
        clouds.cloud_star_formation_rate_kg_s(1e30, 1e-19, 1.5)
    with pytest.raises(ValueError):
        clouds.mach_number(-1.0, 100.0)


def test_scalar_and_array_shape():
    assert np.ndim(clouds.sound_speed_m_s(10.0)) == 0
    assert clouds.jeans_length_m(np.full((2, 3), 190.0), 1e-18).shape == (2, 3)
    assert np.ndim(clouds.bonnor_ebert_critical_mass_kg(190.0, 1e-12)) == 0
    assert np.ndim(clouds.virial_parameter(1e3, 1e17, 1e33)) == 0


# --- galactic laws -------------------------------------------------------------------


def test_kennicutt_schmidt_coefficients():
    # [Kennicutt1998] Eq. 4: Σ_SFR = 2.5e-4 (Σ_gas/1)^1.4. Amplitude at Σ_gas = 1.
    assert laws.kennicutt_schmidt_sfr_surface_density(1.0) == pytest.approx(2.5e-4, rel=1e-14)
    s = np.array([1.0, 10.0, 100.0, 1000.0])
    sfr = laws.kennicutt_schmidt_sfr_surface_density(s)
    slope = np.diff(np.log10(sfr)) / np.diff(np.log10(s))
    assert np.allclose(slope, 1.4, rtol=1e-12)
    # Specific value: 10^1.4 = 25.12 ⇒ Σ_SFR(10) = 6.28e-3
    assert laws.kennicutt_schmidt_sfr_surface_density(10.0) == pytest.approx(2.5e-4 * 10**1.4, rel=1e-13)


def test_kennicutt_orbital_law_efficiency_per_orbit():
    # [Kennicutt1998] Eq. 7: ≈ 10 % per orbit (0.017·2π = 0.107).
    omega = 25.0  # km/s/kpc
    om_yr = omega * 1e3 / (1e3 * PC) * YR
    t_orbit = 2 * math.pi / om_yr
    sigma = 10.0
    sfr = laws.kennicutt_orbital_sfr_surface_density(sigma, omega)
    yield_fraction = sfr * t_orbit / (sigma * 1e6)
    assert yield_fraction == pytest.approx(0.017 * 2 * math.pi, rel=1e-12)
    assert yield_fraction == pytest.approx(0.10, abs=0.01)  # source "~10 %"
    # Linear in Σ and Ω
    assert laws.kennicutt_orbital_sfr_surface_density(2 * sigma, 3 * omega) == pytest.approx(6 * sfr, rel=1e-13)
    # 1 km/s/kpc = 1.0227e-9 yr⁻¹: for Σ = 1 M☉/pc², Ω = 1: 0.017 · 1e6 · 1.0227e-9
    assert laws.kennicutt_orbital_sfr_surface_density(1.0, 1.0) == pytest.approx(0.017e6 * 1.0227e-9, rel=1e-3)


def test_depletion_time():
    # Definition: 10 M☉/pc² and 1e-2 M☉/yr/kpc² ⇒ 1e7/1e-2 = 1e9 yr.
    assert laws.depletion_time_yr(10.0, 1e-2) == pytest.approx(1e9, rel=1e-13)
    # Combined with Kennicutt-Schmidt, t_dep ∝ Σ^{1-N} = Σ^{-0.4}
    t1 = laws.depletion_time_yr(10.0, laws.kennicutt_schmidt_sfr_surface_density(10.0))
    t2 = laws.depletion_time_yr(100.0, laws.kennicutt_schmidt_sfr_surface_density(100.0))
    assert t2 / t1 == pytest.approx(10 ** (-0.4), rel=1e-12)
    # [Krumholz2015] Section 8: typical depletion times are of order Gyr; 1.6 Gyr for 10 M☉/pc².
    assert 1e9 < t1 < 3e9


def test_toomre_q_mckee_value():
    # [McKee2007] Qdef: Q = 1.4 for c_s = 7 km/s, κ = 36 km/s/kpc, Σ = 13 M☉/pc² (two digits; tolerance 3 %).
    assert laws.toomre_q_gas(36.0, 7.0, 13.0) == pytest.approx(1.4, rel=0.03)
    # Q ∝ κ c_s / Σ
    q = laws.toomre_q_gas(36.0, 7.0, 13.0)
    assert laws.toomre_q_gas(72.0, 7.0, 26.0) == pytest.approx(q, rel=1e-13)
    # Dimensions: κ c_s/(π G Σ) by hand in SI
    kappa = 36e3 / (1e3 * PC)
    assert q == pytest.approx(kappa * 7e3 / (math.pi * G * 13 * M_SUN / PC**2), rel=1e-13)


def test_two_dimensional_jeans_mass():
    # [McKee2007] MJ2D: 1e7 M☉ for c_s = 7 km/s, Σ = 13 M☉/pc² (tolerance 1 %; the source has one digit).
    assert laws.two_dimensional_jeans_mass_msun(7.0, 13.0) == pytest.approx(1e7, rel=0.01)
    # M ∝ c_s⁴/Σ
    assert laws.two_dimensional_jeans_mass_msun(14.0, 26.0) == pytest.approx(
        8 * laws.two_dimensional_jeans_mass_msun(7.0, 13.0), rel=1e-13
    )


def test_ff_and_life_cycle_laws_consistent():
    # [McKee2007]: if ε_GMC = ε_ff (t_GMC/t_ff) and f_GMC = t_GMC/t_lc, SFR-ff and SFR-lc give the same result.
    sigma, tff, eps_ff, t_gmc, t_lc = 12.0, 2e7, 0.01, 1e8, 3e8
    a = laws.free_fall_sfr_surface_density(sigma, tff, eps_ff, cloud_gas_fraction=t_gmc / t_lc)
    b = laws.life_cycle_sfr_surface_density(sigma, eps_ff * t_gmc / tff, t_lc)
    assert a == pytest.approx(b, rel=1e-13)
    # [McKee2007]: ε_GMC ≈ 0.05 and t_GMC/t_ff ≈ 5 ⇒ ε_ff ≈ 0.01 (with the function: t_ff / t_depletion = 1/100)
    assert float(clouds.free_fall_efficiency(1.0, 100.0)) == pytest.approx(0.01, rel=1e-12)


def test_sfr_calibration_table():
    # [Kennicutt2012] Table 1: log C_x. L = C_x erg/s ⇒ SFR = 1 M☉/yr.
    for band, logc in [
        ("FUV", 43.35),
        ("NUV", 43.17),
        ("Halpha", 41.27),
        ("TIR", 43.41),
        ("24um", 42.69),
        ("70um", 43.23),
        ("2-10keV", 39.77),
    ]:
        l_w = 10.0**logc * 1e-7
        assert laws.sfr_calibration_msun_yr(l_w, band) == pytest.approx(1.0, rel=1e-12)
    # 1.4 GHz: per unit frequency, its own function
    assert laws.sfr_radio_1p4ghz_msun_yr(10.0**28.20 * 1e-7) == pytest.approx(1.0, rel=1e-12)
    assert laws.sfr_radio_1p4ghz_msun_yr(3e22) == pytest.approx(3.0 * laws.sfr_radio_1p4ghz_msun_yr(1e22), rel=1e-13)
    with pytest.raises(ValueError):
        laws.sfr_calibration_msun_yr(1e22, "1.4GHz")
    # Hα: 1e42 erg/s = 1e35 W ⇒ SFR = 10^(42 − 41.27) = 5.37 M☉/yr
    assert laws.sfr_calibration_msun_yr(1e35, "Halpha") == pytest.approx(10 ** (42 - 41.27), rel=1e-12)
    # Linear
    assert laws.sfr_calibration_msun_yr(np.array([1e34, 1e35]), "TIR")[1] == pytest.approx(
        10 * laws.sfr_calibration_msun_yr(1e34, "TIR"), rel=1e-13
    )


def test_sfr_calibration_invalid():
    with pytest.raises(ValueError):
        laws.sfr_calibration_msun_yr(1e35, "radio")
    with pytest.raises(ValueError):
        laws.sfr_calibration_msun_yr(-1.0, "TIR")


def test_invalid_inputs_laws():
    with pytest.raises(ValueError):
        laws.kennicutt_schmidt_sfr_surface_density(-1.0)
    with pytest.raises(ValueError):
        laws.toomre_q_gas(36.0, 0.0, 13.0)
    with pytest.raises(ValueError):
        laws.free_fall_sfr_surface_density(10.0, 1e7, cloud_gas_fraction=1.5)
    with pytest.raises(ValueError):
        laws.life_cycle_sfr_surface_density(10.0, 1.5, 1e8)


def test_jeans_mass_cube_definition_and_molecular_weights():
    # [Krumholz2015] Section 6: M_J = ρ λ_J³ / 8; ratio to the sphere definition is 4π/3 ([McKee2007]).
    cs = clouds.sound_speed_m_s(10.0)
    rho = clouds.mass_density_kg_m3(1e10, clouds.MOLECULAR_WEIGHT_PER_H2)  # n_H2 = 1e4 cm^-3
    cube = float(clouds.jeans_mass_kg(cs, rho, definition="cube"))
    assert cube == pytest.approx(rho * float(clouds.jeans_length_m(cs, rho)) ** 3 / 8.0, rel=1e-12)
    assert float(clouds.jeans_mass_kg(cs, rho, definition="sphere")) / cube == pytest.approx(4 * np.pi / 3, rel=1e-12)
    # Same helium abundance: 2.33 · 1.2 = 2.8 (within rounding) and 2.8 / 2 = 1.4.
    assert clouds.MOLECULAR_WEIGHT_PER_H2 == pytest.approx(clouds.MEAN_MOLECULAR_WEIGHT * 1.2, rel=2e-3)
    assert clouds.MOLECULAR_WEIGHT_PER_H == pytest.approx(clouds.MOLECULAR_WEIGHT_PER_H2 / 2)
    # If n_H2 is given with the default μ (per particle), the density comes out 17 % too low.
    assert float(clouds.mass_density_kg_m3(1e10)) / rho == pytest.approx(2.33 / 2.8, rel=1e-12)
