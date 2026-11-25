"""Yıldız oluşumu modülü için özellik (hypothesis) testleri: tekdüzelik ve ölçekleme."""

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from kozmos.core.constants import GRAVITATIONAL_CONSTANT
from kozmos.star_formation import clouds, laws

G = GRAVITATIONAL_CONSTANT.value

sicakliklar = st.floats(5.0, 1e4)
hizlar = st.floats(50.0, 5e3)  # m/s
yogunluklar = st.floats(1e-24, 1e-12)  # kg/m^3
carpanlar = st.floats(1.05, 20.0)
yuzeyler = st.floats(0.1, 1e4)  # M☉ pc⁻²


@given(sicakliklar, carpanlar)
@settings(max_examples=50, deadline=None)
def test_ses_hizi_sicaklikla_artar_ve_karekok(t, k):
    assert clouds.sound_speed_m_s(k * t) > clouds.sound_speed_m_s(t)
    assert abs(clouds.sound_speed_m_s(k * t) / clouds.sound_speed_m_s(t) / math.sqrt(k) - 1) < 1e-12


@given(hizlar, yogunluklar, carpanlar)
@settings(max_examples=50, deadline=None)
def test_jeans_boyu_ve_kutlesi_tekduzelik(cs, rho, k):
    # λ_J: c_s ile artar, ρ ile azalır; M_J ∝ c_s³ ρ^-1/2.
    assert clouds.jeans_length_m(k * cs, rho) > clouds.jeans_length_m(cs, rho)
    assert clouds.jeans_length_m(cs, k * rho) < clouds.jeans_length_m(cs, rho)
    assert abs(clouds.jeans_mass_kg(k * cs, rho) / clouds.jeans_mass_kg(cs, rho) / k**3 - 1) < 1e-12
    assert abs(clouds.jeans_mass_kg(cs, k * rho) / clouds.jeans_mass_kg(cs, rho) * math.sqrt(k) - 1) < 1e-12


@given(hizlar, yogunluklar, carpanlar)
@settings(max_examples=50, deadline=None)
def test_jeans_ses_gecis_serbest_dusme_baginti(cs, rho, k):
    # λ_J = sqrt(32/3) c_s t_ff her koşulda (ikisi de aynı G ρ'ya bağlı).
    ratio = clouds.jeans_length_m(cs, rho) / (cs * clouds.free_fall_time_s(rho))
    assert abs(ratio / math.sqrt(32 / 3) - 1) < 1e-12
    assert clouds.free_fall_time_s(k * rho) < clouds.free_fall_time_s(rho)


@given(hizlar, st.floats(1e-14, 1e-8), carpanlar)
@settings(max_examples=50, deadline=None)
def test_bonnor_ebert_kutlesi_tekduzelik(cs, p, k):
    m = clouds.bonnor_ebert_critical_mass_kg(cs, p)
    assert clouds.bonnor_ebert_critical_mass_kg(k * cs, p) > m  # sıcak gaz daha çok kütle taşır
    assert clouds.bonnor_ebert_critical_mass_kg(cs, k * p) < m  # yüksek dış basınç kritik kütleyi küçültür
    assert abs(clouds.bonnor_ebert_critical_mass_kg(k * cs, p) / m / k**4 - 1) < 1e-12


@given(hizlar, st.floats(1e12, 1e20), st.floats(1e30, 1e38), carpanlar)
@settings(max_examples=50, deadline=None)
def test_viryal_parametresi_tekduzelik(sigma, r, m, k):
    a = clouds.virial_parameter(sigma, r, m)
    assert clouds.virial_parameter(k * sigma, r, m) > a
    assert clouds.virial_parameter(sigma, k * r, m) > a
    assert clouds.virial_parameter(sigma, r, k * m) < a
    # Yüzey yoğunluğu biçimiyle tutarlılık
    assert abs(clouds.virial_parameter_from_surface_density(sigma, r, m / (math.pi * r**2)) / a - 1) < 1e-12


@given(st.floats(1e3, 1e20))
@settings(max_examples=50, deadline=None)
def test_sayi_yogunlugu_gidis_donus(n):
    assert abs(clouds.number_density_m3(clouds.mass_density_kg_m3(n)) / n - 1) < 1e-12
    assert abs(clouds.column_density_m2(clouds.surface_density_kg_m2(n)) / n - 1) < 1e-12


@given(st.floats(1e-12, 1e-8), st.floats(1e-12, 1e-6), carpanlar)
@settings(max_examples=50, deadline=None)
def test_manyetik_oran_ve_kritik_kutle_dogrusal(b, s, k):
    # μ/μ_kr ∝ Σ/B; M_Φ ∝ Φ.
    o = clouds.normalized_mass_to_flux_ratio(s, b)
    assert abs(clouds.normalized_mass_to_flux_ratio(k * s, k * b) / o - 1) < 1e-12
    assert abs(clouds.normalized_mass_to_flux_ratio(k * s, b) / o / k - 1) < 1e-12
    assert abs(clouds.magnetic_critical_mass_kg(k * 1e8) / clouds.magnetic_critical_mass_kg(1e8) / k - 1) < 1e-12


@given(st.floats(0.1, 100.0), st.floats(0.1, 50.0), carpanlar)
@settings(max_examples=50, deadline=None)
def test_larson_tekduzelik(size, a, k):
    assert clouds.linewidth_size_km_s(k * size, a) > clouds.linewidth_size_km_s(size, a)
    assert (
        abs(clouds.linewidth_size_km_s(k * size, a, 0.5) / clouds.linewidth_size_km_s(size, a, 0.5) / math.sqrt(k) - 1)
        < 1e-12
    )


@given(yuzeyler, carpanlar)
@settings(max_examples=50, deadline=None)
def test_kennicutt_schmidt_tekduzelik_ve_tuketme(s, k):
    sfr = laws.kennicutt_schmidt_sfr_surface_density
    assert sfr(k * s) > sfr(s)
    # N > 1 olduğundan yoğun gaz daha çabuk tükenir: t_dep azalır.
    assert laws.depletion_time_yr(k * s, sfr(k * s)) < laws.depletion_time_yr(s, sfr(s))
    # Σ'ya k kat ⇒ SFR k^N kat
    assert abs(sfr(k * s) / sfr(s) / k**1.4 - 1) < 1e-12


@given(st.floats(1.0, 200.0), st.floats(1.0, 50.0), yuzeyler, carpanlar)
@settings(max_examples=50, deadline=None)
def test_toomre_q_ve_iki_boyutlu_jeans_olcekleme(kappa, cs, s, k):
    q = laws.toomre_q_gas(kappa, cs, s)
    assert abs(laws.toomre_q_gas(kappa, cs, k * s) / q * k - 1) < 1e-12  # Q ∝ 1/Σ
    assert abs(laws.toomre_q_gas(k * kappa, cs, s) / q / k - 1) < 1e-12
    m = laws.two_dimensional_jeans_mass_msun(cs, s)
    assert abs(laws.two_dimensional_jeans_mass_msun(cs, k * s) * k / m - 1) < 1e-12  # M ∝ 1/Σ


@given(st.floats(30.0, 46.0), carpanlar)
@settings(max_examples=50, deadline=None)
def test_sfr_kalibrasyonu_dogrusal_ve_bant_sirasi(log_l_erg, k):
    l_w = 10.0**log_l_erg * 1e-7
    for band in ("FUV", "NUV", "Halfa", "TIR", "24um", "70um", "2-10keV"):
        a = laws.sfr_calibration_msun_yr(l_w, band)
        assert abs(laws.sfr_calibration_msun_yr(k * l_w, band) / a / k - 1) < 1e-12
    # Hα'nın log C değeri (41,27) TIR'ınkinden (43,41) küçüktür ⇒ aynı ışınım gücü için daha büyük SFR çıkar.
    hs = laws.sfr_calibration_msun_yr(l_w, "Halfa")
    assert hs > laws.sfr_calibration_msun_yr(l_w, "TIR")
    assert np.isfinite(hs)
