"""Yıldız oluşumu modülü için özellik (hypothesis) testleri: tekdüzelik ve ölçekleme."""

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from kozmos.temel.sabitler import CEKIM
from kozmos.yildizolusumu import bulut as bl
from kozmos.yildizolusumu import yasalar as ys

G = CEKIM.deger

sicakliklar = st.floats(5.0, 1e4)
hizlar = st.floats(50.0, 5e3)  # m/s
yogunluklar = st.floats(1e-24, 1e-12)  # kg/m^3
carpanlar = st.floats(1.05, 20.0)
yuzeyler = st.floats(0.1, 1e4)  # M☉ pc⁻²


@given(sicakliklar, carpanlar)
@settings(max_examples=50, deadline=None)
def test_ses_hizi_sicaklikla_artar_ve_karekok(t, k):
    assert bl.ses_hizi_m_s(k * t) > bl.ses_hizi_m_s(t)
    assert abs(bl.ses_hizi_m_s(k * t) / bl.ses_hizi_m_s(t) / math.sqrt(k) - 1) < 1e-12


@given(hizlar, yogunluklar, carpanlar)
@settings(max_examples=50, deadline=None)
def test_jeans_boyu_ve_kutlesi_tekduzelik(cs, rho, k):
    # λ_J: c_s ile artar, ρ ile azalır; M_J ∝ c_s³ ρ^-1/2.
    assert bl.jeans_boyu_m(k * cs, rho) > bl.jeans_boyu_m(cs, rho)
    assert bl.jeans_boyu_m(cs, k * rho) < bl.jeans_boyu_m(cs, rho)
    assert abs(bl.jeans_kutlesi_kg(k * cs, rho) / bl.jeans_kutlesi_kg(cs, rho) / k**3 - 1) < 1e-12
    assert abs(bl.jeans_kutlesi_kg(cs, k * rho) / bl.jeans_kutlesi_kg(cs, rho) * math.sqrt(k) - 1) < 1e-12


@given(hizlar, yogunluklar, carpanlar)
@settings(max_examples=50, deadline=None)
def test_jeans_ses_gecis_serbest_dusme_baginti(cs, rho, k):
    # λ_J = sqrt(32/3) c_s t_ff her koşulda (ikisi de aynı G ρ'ya bağlı).
    oran = bl.jeans_boyu_m(cs, rho) / (cs * bl.serbest_dusme_zamani_s(rho))
    assert abs(oran / math.sqrt(32 / 3) - 1) < 1e-12
    assert bl.serbest_dusme_zamani_s(k * rho) < bl.serbest_dusme_zamani_s(rho)


@given(hizlar, st.floats(1e-14, 1e-8), carpanlar)
@settings(max_examples=50, deadline=None)
def test_bonnor_ebert_kutlesi_tekduzelik(cs, p, k):
    m = bl.bonnor_ebert_kritik_kutle_kg(cs, p)
    assert bl.bonnor_ebert_kritik_kutle_kg(k * cs, p) > m  # sıcak gaz daha çok kütle taşır
    assert bl.bonnor_ebert_kritik_kutle_kg(cs, k * p) < m  # yüksek dış basınç kritik kütleyi küçültür
    assert abs(bl.bonnor_ebert_kritik_kutle_kg(k * cs, p) / m / k**4 - 1) < 1e-12


@given(hizlar, st.floats(1e12, 1e20), st.floats(1e30, 1e38), carpanlar)
@settings(max_examples=50, deadline=None)
def test_viryal_parametresi_tekduzelik(sigma, r, m, k):
    a = bl.viryal_parametresi(sigma, r, m)
    assert bl.viryal_parametresi(k * sigma, r, m) > a
    assert bl.viryal_parametresi(sigma, k * r, m) > a
    assert bl.viryal_parametresi(sigma, r, k * m) < a
    # Yüzey yoğunluğu biçimiyle tutarlılık
    assert abs(bl.viryal_parametresi_yuzeyden(sigma, r, m / (math.pi * r**2)) / a - 1) < 1e-12


@given(st.floats(1e3, 1e20))
@settings(max_examples=50, deadline=None)
def test_sayi_yogunlugu_gidis_donus(n):
    assert abs(bl.sayi_yogunlugu_m3(bl.kutle_yogunlugu_kg_m3(n)) / n - 1) < 1e-12
    assert abs(bl.sutun_yogunlugu_m2(bl.yuzey_yogunlugu_kg_m2(n)) / n - 1) < 1e-12


@given(st.floats(1e-12, 1e-8), st.floats(1e-12, 1e-6), carpanlar)
@settings(max_examples=50, deadline=None)
def test_manyetik_oran_ve_kritik_kutle_dogrusal(b, s, k):
    # μ/μ_kr ∝ Σ/B; M_Φ ∝ Φ.
    o = bl.kutle_akisi_orani_normlu(s, b)
    assert abs(bl.kutle_akisi_orani_normlu(k * s, k * b) / o - 1) < 1e-12
    assert abs(bl.kutle_akisi_orani_normlu(k * s, b) / o / k - 1) < 1e-12
    assert abs(bl.manyetik_kritik_kutle_kg(k * 1e8) / bl.manyetik_kritik_kutle_kg(1e8) / k - 1) < 1e-12


@given(st.floats(0.1, 100.0), st.floats(0.1, 50.0), carpanlar)
@settings(max_examples=50, deadline=None)
def test_larson_tekduzelik(boyut, a, k):
    assert bl.dogrusal_hiz_boyut_km_s(k * boyut, a) > bl.dogrusal_hiz_boyut_km_s(boyut, a)
    assert (
        abs(
            bl.dogrusal_hiz_boyut_km_s(k * boyut, a, 0.5) / bl.dogrusal_hiz_boyut_km_s(boyut, a, 0.5) / math.sqrt(k) - 1
        )
        < 1e-12
    )


@given(yuzeyler, carpanlar)
@settings(max_examples=50, deadline=None)
def test_kennicutt_schmidt_tekduzelik_ve_tuketme(s, k):
    sfr = ys.kennicutt_schmidt_sfr_yuzey_yogunlugu
    assert sfr(k * s) > sfr(s)
    # N > 1 olduğundan yoğun gaz daha çabuk tükenir: t_dep azalır.
    assert ys.tuketme_zamani_yil(k * s, sfr(k * s)) < ys.tuketme_zamani_yil(s, sfr(s))
    # Σ'ya k kat ⇒ SFR k^N kat
    assert abs(sfr(k * s) / sfr(s) / k**1.4 - 1) < 1e-12


@given(st.floats(1.0, 200.0), st.floats(1.0, 50.0), yuzeyler, carpanlar)
@settings(max_examples=50, deadline=None)
def test_toomre_q_ve_iki_boyutlu_jeans_olcekleme(kappa, cs, s, k):
    q = ys.toomre_q_gaz(kappa, cs, s)
    assert abs(ys.toomre_q_gaz(kappa, cs, k * s) / q * k - 1) < 1e-12  # Q ∝ 1/Σ
    assert abs(ys.toomre_q_gaz(k * kappa, cs, s) / q / k - 1) < 1e-12
    m = ys.iki_boyutlu_jeans_kutlesi_gunes_kutlesi(cs, s)
    assert abs(ys.iki_boyutlu_jeans_kutlesi_gunes_kutlesi(cs, k * s) * k / m - 1) < 1e-12  # M ∝ 1/Σ


@given(st.floats(30.0, 46.0), carpanlar)
@settings(max_examples=50, deadline=None)
def test_sfr_kalibrasyonu_dogrusal_ve_bant_sirasi(log_l_erg, k):
    l_w = 10.0**log_l_erg * 1e-7
    for bant in ("FUV", "NUV", "Halfa", "TIR", "24um", "70um", "2-10keV"):
        a = ys.sfr_kalibrasyonu_gunes_kutlesi_yil(l_w, bant)
        assert abs(ys.sfr_kalibrasyonu_gunes_kutlesi_yil(k * l_w, bant) / a / k - 1) < 1e-12
    # Aynı ışınım gücü için Hα'nın C değeri en küçük (41,27) ⇒ en büyük SFR çıkar.
    hs = ys.sfr_kalibrasyonu_gunes_kutlesi_yil(l_w, "Halfa")
    assert hs > ys.sfr_kalibrasyonu_gunes_kutlesi_yil(l_w, "TIR")
    assert np.isfinite(hs)
