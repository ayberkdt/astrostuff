"""Yıldız oluşumu modülünün kaynaktaki sayısal değerleri ve kapalı çözümleri.

Kaynaklar: [McKee2007] (arXiv:0707.3514), [Krumholz2015] (arXiv:1511.03457),
[Kennicutt1998] (astro-ph/9712213), [Kennicutt2012] (arXiv:1204.3552).
Kaynakların sayıları yuvarlanmış verildiği için bağıl toleranslar her
karşılaştırmanın yanında gerekçelendirilmiştir.
"""

import math

import numpy as np
import pytest

from kozmos.temel.sabitler import BOLTZMANN, CEKIM, GM_GUNES, JULYEN_YILI_SANIYE, PARSEK, PROTON_KUTLESI
from kozmos.yildizolusumu import bulut as bl
from kozmos.yildizolusumu import yasalar as ys

G = CEKIM.deger
M_H = PROTON_KUTLESI.deger
PC = PARSEK.deger
M_GUNES = GM_GUNES.deger / G
YIL = JULYEN_YILI_SANIYE.deger
MYR = 1e6 * YIL
# P/k_B = 3e5 cm^-3 K (iki kaynakta da geçen tipik GMC basıncı) -> Pa
P_TIPIK = 3e5 * BOLTZMANN.deger * 1e6


# --- gaz temelleri -------------------------------------------------------------------


def test_ses_hizi_kaynaktaki_deger():
    # [Krumholz2015] Bölüm 4: 10 K, μ = 2,33 için 0,18 km/s. Gerçek değer 0,188 km/s;
    # kaynak iki anlamlı basamağa yuvarlar, bu yüzden tolerans %5.
    cs = bl.ses_hizi_m_s(10.0)
    assert cs == pytest.approx(0.18e3, rel=0.05)
    # Tam bağıntı: c_s² = k T / (μ m_H)
    assert cs**2 == pytest.approx(BOLTZMANN.deger * 10.0 / (2.33 * M_H), rel=1e-13)


def test_ses_hizi_olceklemesi():
    assert bl.ses_hizi_m_s(40.0) / bl.ses_hizi_m_s(10.0) == pytest.approx(2.0, rel=1e-13)
    # μ dört katına çıkarsa c_s yarıya iner.
    assert bl.ses_hizi_m_s(10.0, 4 * 2.33) / bl.ses_hizi_m_s(10.0, 2.33) == pytest.approx(0.5, rel=1e-13)


def test_mach_sayisi():
    cs = bl.ses_hizi_m_s(10.0)
    assert bl.mach_sayisi(cs, cs) == pytest.approx(1.0, rel=1e-14)
    # [Krumholz2015] Bölüm 4: M ~ 20 için hız ≈ 20 c_s; burada yalnızca tanım sınanır.
    assert bl.mach_sayisi(20 * cs, cs) == pytest.approx(20.0, rel=1e-13)


def test_yogunluk_donusumleri_gidis_donus():
    n = np.array([1e6, 1e9, 1e12])
    assert np.allclose(bl.sayi_yogunlugu_m3(bl.kutle_yogunlugu_kg_m3(n)), n, rtol=1e-13)
    s = np.array([1e20, 1e24, 1e26])
    assert np.allclose(bl.sutun_yogunlugu_m2(bl.yuzey_yogunlugu_kg_m2(s)), s, rtol=1e-13)
    # ρ = μ m_H n tanımı
    assert bl.kutle_yogunlugu_kg_m3(1.0, 2.3) == pytest.approx(2.3 * M_H, rel=1e-14)


def test_gmc_sayi_yogunlugu_krumholz():
    # [Krumholz2015] Bölüm 8: Σ = 100 M☉/pc², M = 1e6 M☉ için n = 23 cm⁻³ (μ = 2,3).
    # Kaynak iki anlamlı basamak verir (bizim değer 23,4); tolerans %3.
    m = 1e6 * M_GUNES
    sigma = 100 * M_GUNES / PC**2
    r = math.sqrt(m / (math.pi * sigma))
    rho = 3 * m / (4 * math.pi * r**3)
    n_cm3 = bl.sayi_yogunlugu_m3(rho, 2.3) * 1e-6
    assert n_cm3 == pytest.approx(23.0, rel=0.03)


# --- kararlılık ----------------------------------------------------------------------


def test_jeans_boyu_krumholz_degeri():
    # [Krumholz2015] Bölüm 6: c_s = 0,2 km/s, ρ = 100 m_p için λ_J = 3,4 pc.
    lam = bl.jeans_boyu_m(0.2e3, 100 * M_H * 1e6) / PC  # 100 m_p cm^-3
    assert lam == pytest.approx(3.4, rel=0.02)  # kaynak 2 anlamlı basamak (3,44)


def test_jeans_boyu_esitligi():
    # λ_J² G ρ = π c_s² ([McKee2007] LJeans); farklı girdilerde tam eşitlik.
    cs = np.array([100.0, 300.0, 1e3])
    rho = np.array([1e-18, 1e-20, 1e-22])
    assert np.allclose(bl.jeans_boyu_m(cs, rho) ** 2 * G * rho, math.pi * cs**2, rtol=1e-13)


def test_jeans_kutlesi_iki_tanim_ve_bonnor_ebert():
    cs, rho = 190.0, 1e-18
    kure = bl.jeans_kutlesi_kg(cs, rho, "kure")
    kup = bl.jeans_kutlesi_kg(cs, rho, "kup")
    # İki tanımın oranı (4π/3)/(1) ⇒ π/6 ÷ 1/8 = 4π/3.
    assert kure / kup == pytest.approx(4 * math.pi / 3, rel=1e-13)
    # [McKee2007] LJeans sonrası: M_J = 2,47 M_BE (P_0 = ρ_0 c_s²). Kaynak üç basamak; tolerans %0,5.
    m_be = bl.bonnor_ebert_kritik_kutle_kg(cs, rho * cs**2)
    assert kure / m_be == pytest.approx(2.47, rel=0.005)
    # Ölçek bağıntısı: M_J = katsayı · c_s³ / sqrt(G³ρ)
    assert kure == pytest.approx(math.pi**2.5 / 6 * cs**3 / math.sqrt(G**3 * rho), rel=1e-13)
    assert kup == pytest.approx(math.pi**1.5 / 8 * cs**3 / math.sqrt(G**3 * rho), rel=1e-13)


def test_jeans_kutlesi_gecersiz_tanim():
    with pytest.raises(ValueError):
        bl.jeans_kutlesi_kg(190.0, 1e-18, "silindir")


def test_bonnor_ebert_kaynaktaki_degerler():
    # T = 10 K, P/k_B = 3e5 cm⁻³ K, μ = 2,33: [McKee2007] 0,66 M☉; [Krumholz2015] probsol2 0,67 M☉.
    # Kaynaklar c_s ve μ'yü yuvarlar: McKee için tolerans %3, Krumholz için %1.
    m = bl.bonnor_ebert_kritik_kutle_kg(bl.ses_hizi_m_s(10.0, 2.33), P_TIPIK) / M_GUNES
    assert m == pytest.approx(0.66, rel=0.03)
    assert m == pytest.approx(0.67, rel=0.01)


def test_bonnor_ebert_odeden_katsayilar():
    m_max, kontrast, xi_s = bl.bonnor_ebert_kritik_katsayilar()
    # [Krumholz2015] probsol2: m_max = 1,18; ρ_c/ρ_s = 14,0 (üç anlamlı basamak, tolerans %0,5).
    assert m_max == pytest.approx(1.18, rel=0.005)
    assert kontrast == pytest.approx(14.0, rel=0.005)
    # [McKee2007] MBE_def: 1,182; ODE çözümü ile 4 anlamlı basamakta uyum (tolerans 5e-4).
    assert m_max == pytest.approx(1.182, rel=5e-4)
    # [McKee2007]: R_BE = 0,486 c_s/sqrt(G ρ_0); ODE'de R/r_0 = ξ_s, r_0 = c_s/sqrt(4πGρ_c) ⇒ 0,486 = ξ_s/sqrt(4π ρ_c/ρ_s).
    assert xi_s / math.sqrt(4 * math.pi * kontrast) == pytest.approx(0.486, rel=1e-3)


def test_bonnor_ebert_uygulanan_katsayi_ode_ile_tutarli():
    # Kodda sabit yazılan 1,182 ve 0,486 değerleri ODE çözümüyle aynı yuvarlamada olmalı.
    m_max, kontrast, xi_s = bl.bonnor_ebert_kritik_katsayilar()
    cs, p = 200.0, 1e-12
    assert bl.bonnor_ebert_kritik_kutle_kg(cs, p) / (m_max * cs**4 / math.sqrt(G**3 * p)) == pytest.approx(
        1.0, abs=5e-4
    )
    rho0 = 1e-18
    r_ode = xi_s / math.sqrt(4 * math.pi * kontrast) * cs / math.sqrt(G * rho0)
    assert bl.bonnor_ebert_yaricapi_m(cs, rho0) / r_ode == pytest.approx(1.0, abs=1e-3)


def test_bonnor_ebert_karakteristik_kutle_orani():
    # [McKee2007]: M_BE, yüzey koşullarındaki M_G = σ³/sqrt(G³ρ) ile "neredeyse eşit" (oran 1,182).
    cs, rho0 = 200.0, 1e-18
    m_g = cs**3 / math.sqrt(G**3 * rho0)
    assert bl.bonnor_ebert_kritik_kutle_kg(cs, rho0 * cs**2) / m_g == pytest.approx(1.182, rel=1e-13)


def test_viryal_parametresi_tanimi():
    m, r = 1e35, 3e17
    sigma = math.sqrt(G * m / (5 * r))
    assert bl.viryal_parametresi(sigma, r, m) == pytest.approx(1.0, rel=1e-13)
    # Yüzey yoğunluğu biçimi M = π R² Σ ile aynı sayıyı verir.
    s = m / (math.pi * r**2)
    assert bl.viryal_parametresi_yuzeyden(sigma, r, s) == pytest.approx(1.0, rel=1e-13)
    # Üç Larson yasası: σ ∝ R^0,5, Σ sabit ⇒ α_vir sabit; σ iki katına çıkarsa dört katına çıkar.
    assert bl.viryal_parametresi(2 * sigma, r, m) == pytest.approx(4.0, rel=1e-13)


def test_viryal_krumholz_sayisal():
    # [Krumholz2015] Bölüm 8: σ = σ_pc R_pc^0,5 ve σ_pc = 1 km/s, Σ = 100 M☉/pc² için α_vir = 3,7.
    sigma_pc = 1e3
    r = 7.3 * PC  # Σ ve σ_pc verilince sonuç R'den bağımsızdır; rastgele bir yarıçap
    sigma = sigma_pc * math.sqrt(r / PC)
    alfa = bl.viryal_parametresi_yuzeyden(sigma, r, 100 * M_GUNES / PC**2)
    assert alfa == pytest.approx(3.7, rel=0.01)  # kaynak iki basamak (3,70)


def test_serbest_dusme_zamani_mckee():
    # [McKee2007] eq:tff: n_H = 1e3 cm⁻³ için 1,37e6 yıl; He oranı %10 (sayıca) ⇒ ρ = 1,4 m_H n_H.
    rho = 1.4 * M_H * 1e3 * 1e6
    assert bl.serbest_dusme_zamani_s(rho) / YIL == pytest.approx(1.37e6, rel=0.01)  # 3 anlamlı basamak
    # t_ff ∝ ρ^-1/2
    assert bl.serbest_dusme_zamani_s(4 * rho) / bl.serbest_dusme_zamani_s(rho) == pytest.approx(0.5, rel=1e-13)


def test_zaman_olcekleri_krumholz_yuzey_yogunlugu_bicimi():
    # [Krumholz2015] Bölüm 8: t_ff = π^{1/4}/sqrt(8G) (M/Σ³)^{1/4}, t_cr = 0,95/sqrt(α G) (M/Σ³)^{1/4}.
    m = 1e6 * M_GUNES
    sigma_yuz = 100 * M_GUNES / PC**2
    r = math.sqrt(m / (math.pi * sigma_yuz))
    rho = 3 * m / (4 * math.pi * r**3)
    tff = bl.serbest_dusme_zamani_s(rho)
    olcek = (m / sigma_yuz**3) ** 0.25
    assert tff == pytest.approx(math.pi**0.25 / math.sqrt(8 * G) * olcek, rel=1e-12)
    assert tff / MYR == pytest.approx(7.0, rel=0.01)  # kaynak "7,0 Myr"
    # Viryal bulut (α = 1): t_cr = R/σ, σ² = G M/(5R)
    sigma = math.sqrt(G * m / (5 * r))
    tcr = bl.gecis_zamani_s(r, sigma)
    assert tcr == pytest.approx(math.sqrt(5.0) * math.pi**-0.75 / math.sqrt(G) * olcek, rel=1e-12)
    assert tcr / MYR == pytest.approx(14.0, rel=0.02)  # kaynak "14 Myr" (bizim 14,1)
    # Kaynak: α_vir = 1 için serbest düşme zamanı geçiş zamanının yarısıdır (tam oran 2,01).
    assert tcr / tff == pytest.approx(2.0, rel=0.01)


def test_jeans_boyu_ses_gecis_ve_serbest_dusme_ile_tutarli():
    # λ_J / (c_s t_ff) = sqrt(π) / sqrt(3π/32) = sqrt(32/3): her c_s, ρ için sabit.
    for cs, rho in [(190.0, 1e-18), (1e3, 1e-21), (50.0, 1e-15)]:
        oran = bl.jeans_boyu_m(cs, rho) / (cs * bl.serbest_dusme_zamani_s(rho))
        assert oran == pytest.approx(math.sqrt(32.0 / 3.0), rel=1e-13)
    # Yarım Jeans boyunun ses geçiş zamanı t_ff'nin sqrt(8/3) katıdır.
    cs, rho = 190.0, 1e-18
    t_ses = bl.gecis_zamani_s(bl.jeans_boyu_m(cs, rho) / 2, cs)
    assert t_ses / bl.serbest_dusme_zamani_s(rho) == pytest.approx(math.sqrt(8.0 / 3.0), rel=1e-13)


# --- manyetik destek -----------------------------------------------------------------


def test_manyetik_kritik_kutle_levha_ve_kure_tutarli():
    # [McKee2007]: c_Φ = 0,17 ≈ 1/(2π) = sonsuz levhanın kritik değeri G^{1/2}(Σ/B)_kr = 1/(2π).
    # Düzgün B'li R yarıçaplı küre: Φ = π R² B, kütle M = π R² Σ ⇒ M = M_Φ ⇔ Σ = Σ_kr.
    r, b = 3e16, 1e-9  # 1e-9 T = 10 µG
    aki = math.pi * r**2 * b
    m_kr = bl.manyetik_kritik_kutle_kg(aki, c_phi=1.0 / (2 * math.pi))
    s_kr = m_kr / (math.pi * r**2)
    assert bl.kutle_akisi_orani_normlu(s_kr, b) == pytest.approx(1.0, rel=1e-12)
    # Varsayılan 0,17, 1/(2π) = 0,159'a %7 yakındır; kaynak ikisini "özünde aynı" der.
    assert bl.manyetik_kritik_kutle_kg(aki) / m_kr == pytest.approx(1.0, abs=0.08)


def test_manyetik_kritik_kutle_viryal_turetimi_ile():
    # [Krumholz2015] Bölüm 6: düzgün küre için B_enerji = Φ²/(6π²R) (Gauss) ve W = −(3/5)GM²/R;
    # c_Φ = sqrt(5/2)/(3π) için M = M_Φ'de ikisi eşit. SI'da sınanır.
    r, aki = 3e16, 2e8
    c = math.sqrt(2.5) / (3 * math.pi)
    assert c == pytest.approx(0.17, abs=0.005)  # kaynak: "0,17"
    m = bl.manyetik_kritik_kutle_kg(aki, c_phi=c)
    manyetik_enerji_j = (aki * 1e8) ** 2 / (6 * math.pi**2 * (r * 100.0)) * 1e-7
    cekim_enerjisi_j = 0.6 * G * m**2 / r
    assert manyetik_enerji_j == pytest.approx(cekim_enerjisi_j, rel=1e-12)


def test_manyetik_dogrusal_ve_gecersiz():
    assert bl.manyetik_kritik_kutle_kg(3e8) / bl.manyetik_kritik_kutle_kg(1e8) == pytest.approx(3.0, rel=1e-13)
    with pytest.raises(ValueError):
        bl.manyetik_kritik_kutle_kg(-1.0)
    with pytest.raises(ValueError):
        bl.manyetik_kritik_kutle_kg(1e8, c_phi=0.0)


# --- Larson, birikim, yıldız oluşum hızı ---------------------------------------------


def test_dogrusal_hiz_boyut():
    # [McKee2007] GMC_lws: δv = 0,9 L^0,56 km/s (varsayılan).
    assert bl.dogrusal_hiz_boyut_km_s(1.0) == pytest.approx(0.9, rel=1e-14)
    assert bl.dogrusal_hiz_boyut_km_s(10.0) == pytest.approx(0.9 * 10**0.56, rel=1e-13)
    # [Krumholz2015] Bölüm 8'de aktarılan Solomon uyumu: 0,72 R^0,5.
    assert bl.dogrusal_hiz_boyut_km_s(4.0, 0.72, 0.5) == pytest.approx(1.44, rel=1e-13)
    # [McKee2007]: sonik uzunluk (δv = c_s) yaklaşık 0,03 pc ... bu normalizasyonla 0,19 km/s'e 0,06 pc karşılık gelir
    # (kaynak "boyut tanımına göre değişir" der; 2 kat aralıkta uyum beklenir).
    l_sonik = (0.19 / 0.9) ** (1 / 0.56)
    assert 0.03 / 2 < l_sonik < 0.03 * 3


def test_birikim_hizi_shu():
    # [McKee2007] eq:mdssis: 10 K için 1,54e-6 M☉/yıl (μ = 2,33; kaynak 3 anlamlı basamak).
    cs = bl.ses_hizi_m_s(10.0, 2.33)
    mdot = bl.izotermal_cokus_birikim_hizi_kg_s(cs) / M_GUNES * YIL
    assert mdot == pytest.approx(1.54e-6, rel=0.01)
    # T^{3/2} ölçeklemesi
    oran = bl.izotermal_cokus_birikim_hizi_kg_s(bl.ses_hizi_m_s(40.0)) / bl.izotermal_cokus_birikim_hizi_kg_s(cs)
    assert oran == pytest.approx(8.0, rel=1e-12)
    # [McKee2007] eq:mds: Ṁ ~ M_G / t_G = c_s³/G (M_G = σ³/sqrt(G³ρ), t_G = 1/sqrt(Gρ))
    rho = 1e-17
    m_g, t_g = cs**3 / math.sqrt(G**3 * rho), 1 / math.sqrt(G * rho)
    assert bl.izotermal_cokus_birikim_hizi_kg_s(cs, katsayi=1.0) == pytest.approx(m_g / t_g, rel=1e-12)


def test_bulut_sfr_ve_verimlilik_gidis_donus():
    m, rho = 1e5 * M_GUNES, 1e-19
    mdot = bl.bulut_yildiz_olusum_hizi_kg_s(m, rho, 0.01)
    tff = bl.serbest_dusme_zamani_s(rho)
    # Tüketme zamanı t_dep = M/Ṁ = t_ff/ε_ff ([Krumholz2015] Bölüm 8 tanımı)
    assert m / mdot == pytest.approx(tff / 0.01, rel=1e-13)
    assert bl.serbest_dusme_verimliligi(tff, m / mdot) == pytest.approx(0.01, rel=1e-13)
    # ε_ff = 1 ⇒ Ṁ = M/t_ff
    assert bl.bulut_yildiz_olusum_hizi_kg_s(m, rho, 1.0) == pytest.approx(m / tff, rel=1e-13)


def test_gecersiz_girdiler_bulut():
    with pytest.raises(ValueError):
        bl.ses_hizi_m_s(-5.0)
    with pytest.raises(ValueError):
        bl.jeans_boyu_m(0.0, 1e-18)
    with pytest.raises(ValueError):
        bl.serbest_dusme_zamani_s(np.array([1e-18, -1e-18]))
    with pytest.raises(ValueError):
        bl.bulut_yildiz_olusum_hizi_kg_s(1e30, 1e-19, 1.5)
    with pytest.raises(ValueError):
        bl.mach_sayisi(-1.0, 100.0)


def test_skaler_ve_dizi_sekli():
    assert np.ndim(bl.ses_hizi_m_s(10.0)) == 0
    assert bl.jeans_boyu_m(np.full((2, 3), 190.0), 1e-18).shape == (2, 3)
    assert np.ndim(bl.bonnor_ebert_kritik_kutle_kg(190.0, 1e-12)) == 0
    assert np.ndim(bl.viryal_parametresi(1e3, 1e17, 1e33)) == 0


# --- galaktik yasalar ----------------------------------------------------------------


def test_kennicutt_schmidt_katsayilari():
    # [Kennicutt1998] Denk. 4: Σ_SFR = 2,5e-4 (Σ_gaz/1)^1,4. Σ_gaz = 1'de genlik.
    assert ys.kennicutt_schmidt_sfr_yuzey_yogunlugu(1.0) == pytest.approx(2.5e-4, rel=1e-14)
    s = np.array([1.0, 10.0, 100.0, 1000.0])
    sfr = ys.kennicutt_schmidt_sfr_yuzey_yogunlugu(s)
    egim = np.diff(np.log10(sfr)) / np.diff(np.log10(s))
    assert np.allclose(egim, 1.4, rtol=1e-12)
    # Özel değer: 10^1,4 = 25,12 ⇒ Σ_SFR(10) = 6,28e-3
    assert ys.kennicutt_schmidt_sfr_yuzey_yogunlugu(10.0) == pytest.approx(2.5e-4 * 10**1.4, rel=1e-13)


def test_kennicutt_yorunge_yasasi_yorunge_basina_verim():
    # [Kennicutt1998] Denk. 7: yörünge başına ≈ %10 (0,017·2π = 0,107).
    omega = 25.0  # km/s/kpc
    om_yil = omega * 1e3 / (1e3 * PC) * YIL
    t_yorunge = 2 * math.pi / om_yil
    sigma = 10.0
    sfr = ys.kennicutt_yorunge_sfr_yuzey_yogunlugu(sigma, omega)
    verim = sfr * t_yorunge / (sigma * 1e6)
    assert verim == pytest.approx(0.017 * 2 * math.pi, rel=1e-12)
    assert verim == pytest.approx(0.10, abs=0.01)  # kaynak "~%10"
    # Σ ve Ω'ya doğrusal
    assert ys.kennicutt_yorunge_sfr_yuzey_yogunlugu(2 * sigma, 3 * omega) == pytest.approx(6 * sfr, rel=1e-13)
    # 1 km/s/kpc = 1,0227e-9 yıl⁻¹: Σ = 1 M☉/pc², Ω = 1 için 0,017 · 1e6 · 1,0227e-9
    assert ys.kennicutt_yorunge_sfr_yuzey_yogunlugu(1.0, 1.0) == pytest.approx(0.017e6 * 1.0227e-9, rel=1e-3)


def test_tuketme_zamani():
    # Tanım: 10 M☉/pc² ve 1e-2 M☉/yıl/kpc² ⇒ 1e7/1e-2 = 1e9 yıl.
    assert ys.tuketme_zamani_yil(10.0, 1e-2) == pytest.approx(1e9, rel=1e-13)
    # Kennicutt-Schmidt ile birleşince t_dep ∝ Σ^{1-N} = Σ^{-0,4}
    t1 = ys.tuketme_zamani_yil(10.0, ys.kennicutt_schmidt_sfr_yuzey_yogunlugu(10.0))
    t2 = ys.tuketme_zamani_yil(100.0, ys.kennicutt_schmidt_sfr_yuzey_yogunlugu(100.0))
    assert t2 / t1 == pytest.approx(10 ** (-0.4), rel=1e-12)
    # [Krumholz2015] Bölüm 8: tipik tüketme zamanları Gyr mertebesinde; 10 M☉/pc² için 1,6 Gyr.
    assert 1e9 < t1 < 3e9


def test_toomre_q_mckee_degeri():
    # [McKee2007] Qdef: c_s = 7 km/s, κ = 36 km/s/kpc, Σ = 13 M☉/pc² için Q = 1,4 (iki basamak; tolerans %3).
    assert ys.toomre_q_gaz(36.0, 7.0, 13.0) == pytest.approx(1.4, rel=0.03)
    # Q ∝ κ c_s / Σ
    q = ys.toomre_q_gaz(36.0, 7.0, 13.0)
    assert ys.toomre_q_gaz(72.0, 7.0, 26.0) == pytest.approx(q, rel=1e-13)
    # Boyut: κ c_s/(π G Σ) SI'da elle
    kappa = 36e3 / (1e3 * PC)
    assert q == pytest.approx(kappa * 7e3 / (math.pi * G * 13 * M_GUNES / PC**2), rel=1e-13)


def test_iki_boyutlu_jeans_kutlesi():
    # [McKee2007] MJ2D: c_s = 7 km/s, Σ = 13 M☉/pc² için 1e7 M☉ (tolerans %1; kaynak tek basamak).
    assert ys.iki_boyutlu_jeans_kutlesi_gunes_kutlesi(7.0, 13.0) == pytest.approx(1e7, rel=0.01)
    # M ∝ c_s⁴/Σ
    assert ys.iki_boyutlu_jeans_kutlesi_gunes_kutlesi(14.0, 26.0) == pytest.approx(
        8 * ys.iki_boyutlu_jeans_kutlesi_gunes_kutlesi(7.0, 13.0), rel=1e-13
    )


def test_ff_ve_yasam_dongusu_yasalari_tutarli():
    # [McKee2007]: ε_GMC = ε_ff (t_GMC/t_ff) ve f_GMC = t_GMC/t_lc ise SFR-ff ve SFR-lc aynı sonucu verir.
    sigma, tff, eps_ff, t_gmc, t_lc = 12.0, 2e7, 0.01, 1e8, 3e8
    a = ys.serbest_dusme_sfr_yuzey_yogunlugu(sigma, tff, eps_ff, bulut_gazi_orani=t_gmc / t_lc)
    b = ys.yasam_dongusu_sfr_yuzey_yogunlugu(sigma, eps_ff * t_gmc / tff, t_lc)
    assert a == pytest.approx(b, rel=1e-13)
    # [McKee2007]: ε_GMC ≈ 0,05 ve t_GMC/t_ff ≈ 5 ⇒ ε_ff ≈ 0,01 (fonksiyonla: t_ff / t_tüketme = 1/100)
    assert float(bl.serbest_dusme_verimliligi(1.0, 100.0)) == pytest.approx(0.01, rel=1e-12)


def test_sfr_kalibrasyonu_tablo():
    # [Kennicutt2012] Tablo 1: log C_x. L = C_x erg/s ⇒ SFR = 1 M☉/yıl.
    for bant, logc in [
        ("FUV", 43.35),
        ("NUV", 43.17),
        ("Halfa", 41.27),
        ("TIR", 43.41),
        ("24um", 42.69),
        ("70um", 43.23),
        ("1.4GHz", 28.20),
        ("2-10keV", 39.77),
    ]:
        l_w = 10.0**logc * 1e-7
        assert ys.sfr_kalibrasyonu_gunes_kutlesi_yil(l_w, bant) == pytest.approx(1.0, rel=1e-12)
    # Hα: 1e42 erg/s = 1e35 W ⇒ SFR = 10^(42 − 41,27) = 5,37 M☉/yıl
    assert ys.sfr_kalibrasyonu_gunes_kutlesi_yil(1e35, "Halfa") == pytest.approx(10 ** (42 - 41.27), rel=1e-12)
    # Doğrusal
    assert ys.sfr_kalibrasyonu_gunes_kutlesi_yil(np.array([1e34, 1e35]), "TIR")[1] == pytest.approx(
        10 * ys.sfr_kalibrasyonu_gunes_kutlesi_yil(1e34, "TIR"), rel=1e-13
    )


def test_sfr_kalibrasyonu_gecersiz():
    with pytest.raises(ValueError):
        ys.sfr_kalibrasyonu_gunes_kutlesi_yil(1e35, "radyo")
    with pytest.raises(ValueError):
        ys.sfr_kalibrasyonu_gunes_kutlesi_yil(-1.0, "TIR")


def test_gecersiz_girdiler_yasalar():
    with pytest.raises(ValueError):
        ys.kennicutt_schmidt_sfr_yuzey_yogunlugu(-1.0)
    with pytest.raises(ValueError):
        ys.toomre_q_gaz(36.0, 0.0, 13.0)
    with pytest.raises(ValueError):
        ys.serbest_dusme_sfr_yuzey_yogunlugu(10.0, 1e7, bulut_gazi_orani=1.5)
    with pytest.raises(ValueError):
        ys.yasam_dongusu_sfr_yuzey_yogunlugu(10.0, 1.5, 1e8)


def test_jeans_kutlesi_kup_tanimi_ve_molekul_agirliklari():
    # [Krumholz2015] Bölüm 6: M_J = ρ λ_J³ / 8; küre tanımıyla oranı 4π/3 ([McKee2007]).
    cs = bl.ses_hizi_m_s(10.0)
    rho = bl.kutle_yogunlugu_kg_m3(1e10, bl.H2_BASINA_MOLEKUL_AGIRLIGI)  # n_H2 = 1e4 cm^-3
    kup = float(bl.jeans_kutlesi_kg(cs, rho, tanim="kup"))
    assert kup == pytest.approx(rho * float(bl.jeans_boyu_m(cs, rho)) ** 3 / 8.0, rel=1e-12)
    assert float(bl.jeans_kutlesi_kg(cs, rho, tanim="kure")) / kup == pytest.approx(4 * np.pi / 3, rel=1e-12)
    # Aynı helyum bolluğu: 2,33 · 1,2 = 2,8 (yuvarlama içinde) ve 2,8 / 2 = 1,4.
    assert bl.H2_BASINA_MOLEKUL_AGIRLIGI == pytest.approx(bl.ORTALAMA_MOLEKUL_AGIRLIGI * 1.2, rel=2e-3)
    assert bl.H_BASINA_MOLEKUL_AGIRLIGI == pytest.approx(bl.H2_BASINA_MOLEKUL_AGIRLIGI / 2)
    # n_H2 varsayılan (parçacık başına) μ ile verilirse yoğunluk %17 düşük çıkar.
    assert float(bl.kutle_yogunlugu_kg_m3(1e10)) / rho == pytest.approx(2.33 / 2.8, rel=1e-12)
