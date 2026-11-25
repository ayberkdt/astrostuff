import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from kozmos.core.constants import GM_SUN, GRAVITATIONAL_CONSTANT, SOLAR_RADIUS
from kozmos.stars import imf, scaling, structure

# Güneş kütlesi, kütle parametresinden (yerel tanım).
M_GUNES = GM_SUN.value / GRAVITATIONAL_CONSTANT.value
R_GUNES = SOLAR_RADIUS.value

kutleler = st.floats(0.1, 50.0)  # M_⊙
yaricaplar = st.floats(0.05, 200.0)  # R_⊙
indisler = st.floats(0.0, 4.5)


@given(indisler, st.floats(0.05, 0.4))
@settings(max_examples=25, deadline=None)
def test_xi1_ve_merkez_yogunluk_orani_n_ile_artar(n, dn):
    # Daha büyük n daha merkezi yoğunlaşmış, daha geniş bir küre verir.
    assert structure.lane_emden_first_zero(n + dn) > structure.lane_emden_first_zero(n)
    assert structure.lane_emden_central_to_mean_density(n + dn) > structure.lane_emden_central_to_mean_density(n)
    assert structure.lane_emden_central_to_mean_density(n) >= 1.0 - 1e-12


@given(st.floats(0.0, 4.9))
@settings(max_examples=25, deadline=None)
def test_lane_emden_cozumu_tekduze_ve_sinirli(n):
    xi, theta, dtheta = structure.lane_emden_solve(n, 200)
    assert theta[0] == 1.0 and dtheta[0] == 0.0
    assert np.all(np.diff(theta) <= 0.0)
    assert np.all(theta >= -1e-9) and abs(theta[-1]) < 1e-8
    assert np.all(dtheta[1:] < 0.0)


@given(kutleler, yaricaplar, st.sampled_from([0.5, 1.0, 1.5, 2.0, 3.0, 4.0]))
@settings(max_examples=40, deadline=None)
def test_politrop_basinci_ve_enerji_isaretleri(m, r, n):
    mk, rm = m * M_GUNES, r * R_GUNES
    assert structure.polytrope_central_pressure_pa(mk, rm, n) > 0
    assert structure.polytrope_gravitational_energy_j(mk, rm, n) < 0
    # Merkez yoğunluk ortalamayı aşar.
    assert structure.polytrope_central_density_kg_m3(mk, rm, n) >= 3 * mk / (4 * math.pi * rm**3) * (1 - 1e-12)


@given(kutleler, yaricaplar, st.sampled_from([0.5, 1.5, 2.0, 4.0]))
@settings(max_examples=40, deadline=None)
def test_k_yaricap_gidis_donus(m, r, n):
    mk, rm = m * M_GUNES, r * R_GUNES
    k = structure.polytrope_k_coefficient_si(mk, rm, n)
    assert math.isclose(float(structure.polytrope_radius_m(mk, k, n)), rm, rel_tol=1e-9)


@given(yaricaplar, st.floats(2000.0, 50000.0))
@settings(max_examples=60, deadline=None)
def test_isinim_gucu_sicaklik_gidis_donus(r, t):
    rm = r * R_GUNES
    assert math.isclose(float(scaling.effective_temperature_k(scaling.luminosity_w(rm, t), rm)), t, rel_tol=1e-12)
    # L ∝ T⁴: sıcaklık iki katına çıkınca ışınım gücü 16 katına çıkar.
    assert math.isclose(float(scaling.luminosity_w(rm, 2 * t) / scaling.luminosity_w(rm, t)), 16.0, rel_tol=1e-12)


@given(st.floats(1e-12, 1e6), st.floats(1.01, 100.0))
@settings(max_examples=60, deadline=None)
def test_serbest_dusme_zamani_yogunlukla_azalir(rho, carpan):
    assert scaling.free_fall_time_s(rho * carpan) < scaling.free_fall_time_s(rho)
    assert math.isclose(
        float(scaling.free_fall_time_s(rho * carpan) / scaling.free_fall_time_s(rho)), carpan**-0.5, rel_tol=1e-12
    )


@given(st.floats(0.01, 200.0), st.floats(1.01, 5.0), st.floats(0.3, 1.0))
@settings(max_examples=60, deadline=None)
def test_eddington_beta_ve_isinim_monotonluk(mass, carpan, mu):
    m1, m2 = mass * M_GUNES, mass * carpan * M_GUNES
    b1, b2 = float(scaling.eddington_beta(m1, mu)), float(scaling.eddington_beta(m2, mu))
    assert 0.0 < b2 < b1 <= 1.0
    kappa = 0.034
    l1 = float(scaling.standard_model_luminosity_w(m1, mu, kappa))
    l2 = float(scaling.standard_model_luminosity_w(m2, mu, kappa))
    assert 0.0 < l1 < l2 < float(scaling.eddington_luminosity_w(m2, kappa))


@given(st.floats(0.5, 4.0), st.floats(1.01, 3.0))
@settings(max_examples=40, deadline=None)
def test_chandrasekhar_kutlesi_mu_e_ile_azalir(mu_e, carpan):
    m1 = scaling.chandrasekhar_mass_kg(mu_e)
    m2 = scaling.chandrasekhar_mass_kg(mu_e * carpan)
    assert m2 < m1
    assert math.isclose(m2 / m1, carpan**-2.0, rel_tol=1e-12)


alt_ust = st.tuples(st.floats(0.011, 0.5), st.floats(2.0, 150.0))


@given(alt_ust, st.floats(1.1, 3.5))
@settings(max_examples=60, deadline=None)
def test_salpeter_kutle_orani_monoton(aralik, alpha):
    a, b = aralik
    threshold = np.geomspace(a, b, 25)
    f = imf.salpeter_mass_fraction_above(threshold, a, b, alpha)
    assert np.all(np.diff(f) <= 1e-15) and np.all((f >= 0) & (f <= 1))
    assert math.isclose(float(imf.salpeter_mass_fraction_above(a, a, b, alpha)), 1.0, abs_tol=1e-12)
    assert abs(float(imf.salpeter_mass_fraction_above(b, a, b, alpha))) < 1e-12
    # Ortalama kütle aralık içinde kalır.
    assert a < imf.salpeter_mean_mass(a, b, alpha) < b


@given(alt_ust)
@settings(max_examples=40, deadline=None)
def test_kroupa_ve_chabrier_kutle_orani_monoton(aralik):
    a, b = aralik
    threshold = np.geomspace(a, b, 25)
    for f in (imf.kroupa_mass_fraction_above(threshold, a, b), imf.chabrier_mass_fraction_above(threshold, a, b)):
        assert np.all(np.diff(f) <= 1e-15) and np.all((f >= 0) & (f <= 1))
        assert math.isclose(float(f[0]), 1.0, abs_tol=1e-12) and abs(float(f[-1])) < 1e-12
    assert a < imf.kroupa_mean_mass(a, b) < b
    assert a < imf.chabrier_mean_mass(a, b) < b


@given(alt_ust, st.integers(0, 300), st.integers(0, 2**32 - 1))
@settings(max_examples=40, deadline=None)
def test_examples_aralik_icinde(aralik, n, tohum):
    a, b = aralik
    for sample in (imf.salpeter_sample, imf.kroupa_sample, imf.chabrier_sample):
        x = sample(np.random.default_rng(tohum), n, a, b)
        assert x.shape == (n,)
        assert np.all((x >= a * (1 - 1e-12)) & (x <= b * (1 + 1e-12)))
