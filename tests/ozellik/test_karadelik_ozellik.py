"""Kara delik modülünün özellik testleri: tekdüzelik, ölçekleme, sıralama ve gidiş-dönüş."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kozmos.karadelik import temel as kd
from kozmos.karadelik import yigilma as yg

kutleler = st.floats(1e-3, 1e10).map(lambda x: x * 1.988e30)  # 1e-3 ile 1e10 M☉
spinler = st.floats(-1.0, 1.0)
carpanlar = st.floats(0.1, 100.0)
AYAR = settings(max_examples=60, deadline=None)


@given(st.floats(0.0, 1.0), st.floats(0.0, 1.0))
@AYAR
def test_isco_spinle_tekduze(a1, a2):
    # Doğru yörüngede ISCO spinle küçülür, ters yörüngede büyür: r(−a) artan, r(+a) azalan.
    lo, hi = sorted((a1, a2))
    assert kd.isco_yaricapi_kerr_rg(hi) <= kd.isco_yaricapi_kerr_rg(lo) + 1e-12
    assert kd.isco_yaricapi_kerr_rg(-hi) >= kd.isco_yaricapi_kerr_rg(-lo) - 1e-12


@given(spinler)
@AYAR
def test_isco_sinirlari_ve_siralama(a):
    isco = kd.isco_yaricapi_kerr_rg(a)
    foton = kd.foton_yorungesi_yaricapi_kerr_rg(a)
    ufuk = kd.ufuk_yaricapi_kerr_rg(a)
    assert 1.0 - 1e-12 <= isco <= 9.0 + 1e-12
    # Ufuk ≤ foton yörüngesi ≤ ISCO; ergosfer ufuktan dışarıda.
    assert ufuk <= foton + 1e-9
    assert foton <= isco + 1e-9
    assert kd.ergosfer_yaricapi_kerr_rg(a) >= ufuk - 1e-12
    # Doğru yörünge ters yörüngeden içeridedir.
    assert kd.isco_yaricapi_kerr_rg(abs(a)) <= kd.isco_yaricapi_kerr_rg(-abs(a)) + 1e-12


@given(st.floats(1e-6, 1e-3))
@AYAR
def test_kerr_isco_dusuk_spinde_surekli(eps):
    # a* → 0: ISCO her iki işaretten 6 r_G'ye sürekli gider (eğim ≈ 3,27 < 3,3).
    assert abs(kd.isco_yaricapi_kerr_rg(eps) - 6.0) <= 3.3 * eps + 1e-9
    assert abs(kd.isco_yaricapi_kerr_rg(-eps) - 6.0) <= 3.3 * eps + 1e-9


@given(st.floats(-1.0, 1.0), st.floats(-1.0, 1.0))
@AYAR
def test_verim_spinle_artar(a1, a2):
    lo, hi = sorted((a1, a2))
    assert kd.isco_isinim_verimliligi(hi) >= kd.isco_isinim_verimliligi(lo) - 1e-12
    assert 0.0 < kd.isco_isinim_verimliligi(a1) < 0.4227


@given(spinler, st.floats(0.0, np.pi))
@AYAR
def test_ergosfer_ufuk_ile_ekvator_arasinda(a, teta):
    r = kd.ergosfer_yaricapi_kerr_rg(a, teta)
    assert kd.ufuk_yaricapi_kerr_rg(a) - 1e-12 <= r <= 2.0 + 1e-12


@given(kutleler, carpanlar)
@AYAR
def test_geometri_dogrusal(m, k):
    for f in (
        kd.kutleceken_yaricapi_m,
        kd.schwarzschild_yaricapi_m,
        kd.foton_kuresi_yaricapi_m,
        kd.kritik_carpisma_parametresi_m,
        kd.isco_yaricapi_schwarzschild_m,
    ):
        assert f(k * m) == pytest.approx(k * f(m), rel=1e-12)
    for f in (kd.isco_yorunge_frekansi_hz, kd.gw_frekansi_isco_hz, kd.hawking_sicakligi_k):
        assert f(k * m) == pytest.approx(f(m) / k, rel=1e-12)


@given(kutleler, st.floats(1e20, 1e26), carpanlar)
@AYAR
def test_golge_capi_kutle_uzaklik_oraninda(m, d, k):
    assert kd.golge_acisal_capi_uas(k * m, k * d) == pytest.approx(kd.golge_acisal_capi_uas(m, d), rel=1e-12)


@given(kutleler, st.floats(1.0001, 1e6), st.floats(1.0001, 1e6))
@AYAR
def test_kirmiziya_kayma_yaricapla_azalir(m, x1, x2):
    rs = kd.schwarzschild_yaricapi_m(m)
    lo, hi = sorted((x1, x2))
    z_lo = kd.yercekimsel_kirmiziya_kayma(lo * rs, m)
    z_hi = kd.yercekimsel_kirmiziya_kayma(hi * rs, m)
    assert z_lo >= z_hi > 0.0


@given(kutleler, carpanlar)
@AYAR
def test_termodinamik_olcekleme(m, k):
    assert kd.bekenstein_hawking_entropisi(k * m) == pytest.approx(k**2 * kd.bekenstein_hawking_entropisi(m), rel=1e-12)
    assert kd.hawking_buharlasma_suresi_s(k * m) == pytest.approx(k**3 * kd.hawking_buharlasma_suresi_s(m), rel=1e-12)
    # T·M sabit: kütle büyüdükçe sıcaklık düşer.
    assert kd.hawking_sicakligi_k(k * m) * k * m == pytest.approx(kd.hawking_sicakligi_k(m) * m, rel=1e-12)


@given(kutleler, kutleler)
@AYAR
def test_cirp_kutlesi_simetri_ve_sinir(m1, m2):
    mc = kd.cirp_kutlesi_kg(m1, m2)
    assert mc == pytest.approx(kd.cirp_kutlesi_kg(m2, m1), rel=1e-13)
    # ν ≤ 1/4 ⇒ 𝓜 = ν^{3/5} M ≤ 2^{-6/5} M; eşitlik eşit kütlede.
    assert mc <= 2.0**-1.2 * (m1 + m2) * (1.0 + 1e-12)
    assert kd.cirp_kutlesi_kg(2.0 * m1, 2.0 * m2) == pytest.approx(2.0 * mc, rel=1e-13)


@given(st.floats(1.0, 100.0), st.floats(1.0, 400.0), st.floats(1.0, 400.0))
@AYAR
def test_birlesme_suresi_frekansla_azalir_ve_gidis_donus(mc_gunes, f1, f2):
    mc = mc_gunes * 1.988e30
    lo, hi = sorted((f1, f2))
    assert kd.birlesmeye_kalan_sure_s(mc, hi) <= kd.birlesmeye_kalan_sure_s(mc, lo)
    assert kd.gw_frekans_turevi_hz_s(mc, hi) >= kd.gw_frekans_turevi_hz_s(mc, lo)
    tau = kd.birlesmeye_kalan_sure_s(mc, lo)
    assert kd.kalan_sureden_gw_frekansi_hz(mc, tau) == pytest.approx(lo, rel=1e-12)
    # τ ∝ f^{-8/3}.
    assert kd.birlesmeye_kalan_sure_s(mc, 2.0 * lo) == pytest.approx(
        kd.birlesmeye_kalan_sure_s(mc, lo) * 2.0 ** (-8.0 / 3.0), rel=1e-12
    )


@given(kutleler, carpanlar, st.floats(0.01, 0.4))
@AYAR
def test_eddington_dogrusal_ve_verim_olcekleme(m, k, eta):
    assert yg.eddington_isinim_gucu_w(k * m) == pytest.approx(k * yg.eddington_isinim_gucu_w(m), rel=1e-12)
    # Işınım gücü verimden bağımsız: L = η Ṁ_Edd(η) c² = L_Edd.
    mdot = yg.eddington_kutle_akisi_kg_s(m, eta)
    assert yg.yigilma_isinim_gucu_w(mdot, eta) == pytest.approx(yg.eddington_isinim_gucu_w(m), rel=1e-12)


@given(st.floats(0.01, 0.4), st.floats(0.01, 0.4))
@AYAR
def test_salpeter_zamani_verimle_artar(e1, e2):
    lo, hi = sorted((e1, e2))
    assert yg.salpeter_zamani_s(lo) <= yg.salpeter_zamani_s(hi)


@given(kutleler, st.floats(1e3, 1e6), st.floats(1e-25, 1e-15), carpanlar)
@AYAR
def test_bondi_olcekleme(m, cs, rho, k):
    assert yg.bondi_yaricapi_m(k * m, cs) == pytest.approx(k * yg.bondi_yaricapi_m(m, cs), rel=1e-12)
    assert yg.bondi_yaricapi_m(m, k * cs) == pytest.approx(yg.bondi_yaricapi_m(m, cs) / k**2, rel=1e-12)
    assert yg.bondi_kutle_akisi_kg_s(k * m, rho, cs) == pytest.approx(
        k**2 * yg.bondi_kutle_akisi_kg_s(m, rho, cs), rel=1e-12
    )
    assert yg.bondi_kutle_akisi_kg_s(m, k * rho, cs) == pytest.approx(
        k * yg.bondi_kutle_akisi_kg_s(m, rho, cs), rel=1e-12
    )
    assert yg.bondi_kutle_akisi_kg_s(m, rho, k * cs) == pytest.approx(
        yg.bondi_kutle_akisi_kg_s(m, rho, cs) / k**3, rel=1e-12
    )


@given(kutleler, st.floats(1e12, 1e18), st.floats(1e4, 1e8), carpanlar)
@AYAR
def test_ince_disk_olcekleme_ve_zarf(m, mdot, r_ic, k):
    # T_max ∝ (M Ṁ)^{1/4} r_in^{-3/4}.
    t0 = yg.ince_disk_azami_sicakligi_k(m, mdot, r_ic)
    assert yg.ince_disk_azami_sicakligi_k(k * m, mdot, r_ic) == pytest.approx(k**0.25 * t0, rel=1e-12)
    assert yg.ince_disk_azami_sicakligi_k(m, k * mdot, r_ic) == pytest.approx(k**0.25 * t0, rel=1e-12)
    assert yg.ince_disk_azami_sicakligi_k(m, mdot, k * r_ic) == pytest.approx(k**-0.75 * t0, rel=1e-12)
    # Profil hiçbir yarıçapta T_max'ı aşmaz.
    r = r_ic * np.linspace(1.0, 50.0, 400)
    assert np.all(yg.ince_disk_etkin_sicakligi_k(r, m, mdot, r_ic) <= t0 * (1.0 + 1e-12))


@given(kutleler, kutleler, st.floats(1e6, 1e10), carpanlar)
@AYAR
def test_gelgit_yaricapi_ve_hills_olcekleme(m_kd, m_y, r_y, k):
    rt = yg.gelgit_parcalanma_yaricapi_m(m_kd, m_y, r_y)
    assert yg.gelgit_parcalanma_yaricapi_m(k * m_kd, m_y, r_y) == pytest.approx(k ** (1.0 / 3.0) * rt, rel=1e-12)
    assert yg.gelgit_parcalanma_yaricapi_m(m_kd, k * m_y, r_y) == pytest.approx(k ** (-1.0 / 3.0) * rt, rel=1e-12)
    # M_H'de r_t = R_S; daha büyük kara delikte r_t < R_S (yıldız parçalanmadan yutulur).
    mh = yg.hills_kutlesi_kg(m_y, r_y)
    assert yg.gelgit_parcalanma_yaricapi_m(mh, m_y, r_y) == pytest.approx(kd.schwarzschild_yaricapi_m(mh), rel=1e-12)
    assert yg.gelgit_parcalanma_yaricapi_m(2.0 * mh, m_y, r_y) < kd.schwarzschild_yaricapi_m(2.0 * mh)
