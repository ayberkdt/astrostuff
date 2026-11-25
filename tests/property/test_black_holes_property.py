"""Kara delik modülünün özellik testleri: tekdüzelik, ölçekleme, sıralama ve gidiş-dönüş."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kozmos.black_holes import accretion, basics

kutleler = st.floats(1e-3, 1e10).map(lambda x: x * 1.988e30)  # 1e-3 ile 1e10 M☉
spinler = st.floats(-1.0, 1.0)
carpanlar = st.floats(0.1, 100.0)
AYAR = settings(max_examples=60, deadline=None)


@given(st.floats(0.0, 1.0), st.floats(0.0, 1.0))
@AYAR
def test_isco_spinle_tekduze(a1, a2):
    # Doğru yörüngede ISCO spinle küçülür, ters yörüngede büyür: r(−a) artan, r(+a) azalan.
    lo, hi = sorted((a1, a2))
    assert basics.kerr_isco_radius_rg(hi) <= basics.kerr_isco_radius_rg(lo) + 1e-12
    assert basics.kerr_isco_radius_rg(-hi) >= basics.kerr_isco_radius_rg(-lo) - 1e-12


@given(spinler)
@AYAR
def test_isco_sinirlari_ve_siralama(a):
    isco = basics.kerr_isco_radius_rg(a)
    foton = basics.kerr_photon_orbit_radius_rg(a)
    ufuk = basics.kerr_horizon_radius_rg(a)
    assert 1.0 - 1e-12 <= isco <= 9.0 + 1e-12
    # Ufuk ≤ foton yörüngesi ≤ ISCO; ergosfer ufuktan dışarıda.
    assert ufuk <= foton + 1e-9
    assert foton <= isco + 1e-9
    assert basics.kerr_ergosphere_radius_rg(a) >= ufuk - 1e-12
    # Doğru yörünge ters yörüngeden içeridedir.
    assert basics.kerr_isco_radius_rg(abs(a)) <= basics.kerr_isco_radius_rg(-abs(a)) + 1e-12


@given(st.floats(1e-6, 1e-3))
@AYAR
def test_kerr_isco_dusuk_spinde_surekli(eps):
    # a* → 0: ISCO her iki işaretten 6 r_G'ye sürekli gider (eğim ≈ 3,27 < 3,3).
    assert abs(basics.kerr_isco_radius_rg(eps) - 6.0) <= 3.3 * eps + 1e-9
    assert abs(basics.kerr_isco_radius_rg(-eps) - 6.0) <= 3.3 * eps + 1e-9


@given(st.floats(-1.0, 1.0), st.floats(-1.0, 1.0))
@AYAR
def test_verim_spinle_artar(a1, a2):
    lo, hi = sorted((a1, a2))
    assert basics.isco_radiative_efficiency(hi) >= basics.isco_radiative_efficiency(lo) - 1e-12
    assert 0.0 < basics.isco_radiative_efficiency(a1) < 0.4227


@given(spinler, st.floats(0.0, np.pi))
@AYAR
def test_ergosfer_ufuk_ile_ekvator_arasinda(a, teta):
    r = basics.kerr_ergosphere_radius_rg(a, teta)
    assert basics.kerr_horizon_radius_rg(a) - 1e-12 <= r <= 2.0 + 1e-12


@given(kutleler, carpanlar)
@AYAR
def test_geometri_dogrusal(m, k):
    for f in (
        basics.gravitational_radius_m,
        basics.schwarzschild_radius_m,
        basics.photon_sphere_radius_m,
        basics.critical_impact_parameter_m,
        basics.schwarzschild_isco_radius_m,
    ):
        assert f(k * m) == pytest.approx(k * f(m), rel=1e-12)
    for f in (basics.isco_orbital_frequency_hz, basics.gw_frequency_at_isco_hz, basics.hawking_temperature_k):
        assert f(k * m) == pytest.approx(f(m) / k, rel=1e-12)


@given(kutleler, st.floats(1e20, 1e26), carpanlar)
@AYAR
def test_golge_capi_kutle_uzaklik_oraninda(m, d, k):
    assert basics.shadow_angular_diameter_uas(k * m, k * d) == pytest.approx(
        basics.shadow_angular_diameter_uas(m, d), rel=1e-12
    )


@given(kutleler, st.floats(1.0001, 1e6), st.floats(1.0001, 1e6))
@AYAR
def test_kirmiziya_kayma_yaricapla_azalir(m, x1, x2):
    rs = basics.schwarzschild_radius_m(m)
    lo, hi = sorted((x1, x2))
    z_lo = basics.gravitational_redshift(lo * rs, m)
    z_hi = basics.gravitational_redshift(hi * rs, m)
    assert z_lo >= z_hi > 0.0


@given(kutleler, carpanlar)
@AYAR
def test_termodinamik_olcekleme(m, k):
    assert basics.bekenstein_hawking_entropy(k * m) == pytest.approx(
        k**2 * basics.bekenstein_hawking_entropy(m), rel=1e-12
    )
    assert basics.hawking_evaporation_time_s(k * m) == pytest.approx(
        k**3 * basics.hawking_evaporation_time_s(m), rel=1e-12
    )
    # T·M sabit: kütle büyüdükçe sıcaklık düşer.
    assert basics.hawking_temperature_k(k * m) * k * m == pytest.approx(basics.hawking_temperature_k(m) * m, rel=1e-12)


@given(kutleler, kutleler)
@AYAR
def test_cirp_kutlesi_simetri_ve_sinir(m1, m2):
    mc = basics.chirp_mass_kg(m1, m2)
    assert mc == pytest.approx(basics.chirp_mass_kg(m2, m1), rel=1e-13)
    # ν ≤ 1/4 ⇒ 𝓜 = ν^{3/5} M ≤ 2^{-6/5} M; eşitlik eşit kütlede.
    assert mc <= 2.0**-1.2 * (m1 + m2) * (1.0 + 1e-12)
    assert basics.chirp_mass_kg(2.0 * m1, 2.0 * m2) == pytest.approx(2.0 * mc, rel=1e-13)


@given(st.floats(1.0, 100.0), st.floats(1.0, 400.0), st.floats(1.0, 400.0))
@AYAR
def test_birlesme_suresi_frekansla_azalir_ve_gidis_donus(mc_gunes, f1, f2):
    mc = mc_gunes * 1.988e30
    lo, hi = sorted((f1, f2))
    assert basics.time_to_merger_s(mc, hi) <= basics.time_to_merger_s(mc, lo)
    assert basics.gw_frequency_derivative_hz_s(mc, hi) >= basics.gw_frequency_derivative_hz_s(mc, lo)
    tau = basics.time_to_merger_s(mc, lo)
    assert basics.gw_frequency_from_time_to_merger_hz(mc, tau) == pytest.approx(lo, rel=1e-12)
    # τ ∝ f^{-8/3}.
    assert basics.time_to_merger_s(mc, 2.0 * lo) == pytest.approx(
        basics.time_to_merger_s(mc, lo) * 2.0 ** (-8.0 / 3.0), rel=1e-12
    )


@given(kutleler, carpanlar, st.floats(0.01, 0.4))
@AYAR
def test_eddington_dogrusal_ve_verim_olcekleme(m, k, eta):
    assert accretion.eddington_luminosity_w(k * m) == pytest.approx(k * accretion.eddington_luminosity_w(m), rel=1e-12)
    # Işınım gücü verimden bağımsız: L = η Ṁ_Edd(η) c² = L_Edd.
    mdot = accretion.eddington_accretion_rate_kg_s(m, eta)
    assert accretion.accretion_luminosity_w(mdot, eta) == pytest.approx(accretion.eddington_luminosity_w(m), rel=1e-12)


@given(st.floats(0.01, 0.4), st.floats(0.01, 0.4))
@AYAR
def test_salpeter_zamani_verimle_artar(e1, e2):
    lo, hi = sorted((e1, e2))
    assert accretion.salpeter_time_s(lo) <= accretion.salpeter_time_s(hi)


@given(kutleler, st.floats(1e3, 1e6), st.floats(1e-25, 1e-15), carpanlar)
@AYAR
def test_bondi_olcekleme(m, cs, rho, k):
    assert accretion.bondi_radius_m(k * m, cs) == pytest.approx(k * accretion.bondi_radius_m(m, cs), rel=1e-12)
    assert accretion.bondi_radius_m(m, k * cs) == pytest.approx(accretion.bondi_radius_m(m, cs) / k**2, rel=1e-12)
    assert accretion.bondi_accretion_rate_kg_s(k * m, rho, cs) == pytest.approx(
        k**2 * accretion.bondi_accretion_rate_kg_s(m, rho, cs), rel=1e-12
    )
    assert accretion.bondi_accretion_rate_kg_s(m, k * rho, cs) == pytest.approx(
        k * accretion.bondi_accretion_rate_kg_s(m, rho, cs), rel=1e-12
    )
    assert accretion.bondi_accretion_rate_kg_s(m, rho, k * cs) == pytest.approx(
        accretion.bondi_accretion_rate_kg_s(m, rho, cs) / k**3, rel=1e-12
    )


@given(kutleler, st.floats(1e12, 1e18), st.floats(1e4, 1e8), carpanlar)
@AYAR
def test_ince_disk_olcekleme_ve_zarf(m, mdot, r_ic, k):
    # T_max ∝ (M Ṁ)^{1/4} r_in^{-3/4}.
    t0 = accretion.thin_disk_peak_temperature_k(m, mdot, r_ic)
    assert accretion.thin_disk_peak_temperature_k(k * m, mdot, r_ic) == pytest.approx(k**0.25 * t0, rel=1e-12)
    assert accretion.thin_disk_peak_temperature_k(m, k * mdot, r_ic) == pytest.approx(k**0.25 * t0, rel=1e-12)
    assert accretion.thin_disk_peak_temperature_k(m, mdot, k * r_ic) == pytest.approx(k**-0.75 * t0, rel=1e-12)
    # Profil hiçbir yarıçapta T_max'ı aşmaz.
    r = r_ic * np.linspace(1.0, 50.0, 400)
    assert np.all(accretion.thin_disk_effective_temperature_k(r, m, mdot, r_ic) <= t0 * (1.0 + 1e-12))


@given(kutleler, kutleler, st.floats(1e6, 1e10), carpanlar)
@AYAR
def test_gelgit_yaricapi_ve_hills_olcekleme(m_kd, m_y, r_y, k):
    rt = accretion.tidal_disruption_radius_m(m_kd, m_y, r_y)
    assert accretion.tidal_disruption_radius_m(k * m_kd, m_y, r_y) == pytest.approx(k ** (1.0 / 3.0) * rt, rel=1e-12)
    assert accretion.tidal_disruption_radius_m(m_kd, k * m_y, r_y) == pytest.approx(k ** (-1.0 / 3.0) * rt, rel=1e-12)
    # M_H'de r_t = R_S; daha büyük kara delikte r_t < R_S (yıldız parçalanmadan yutulur).
    mh = accretion.hills_mass_kg(m_y, r_y)
    assert accretion.tidal_disruption_radius_m(mh, m_y, r_y) == pytest.approx(
        basics.schwarzschild_radius_m(mh), rel=1e-12
    )
    assert accretion.tidal_disruption_radius_m(2.0 * mh, m_y, r_y) < basics.schwarzschild_radius_m(2.0 * mh)
