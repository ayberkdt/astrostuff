"""Yıldız modülü: kapalı çözümler, kaynak metinlerindeki sayılar ve örnekleyici dağılım sınamaları.

Kaynaklar ve sayıların okunduğu yerler:
[Bludman2010] arXiv:1008.4281 (Tablo I, II; Sec. II); [Liu1995] arXiv:astro-ph/9512061;
[Araujo2011] arXiv:1102.2393; [Prsa2016] arXiv:1605.09788; [McKee2007] arXiv:0707.3514;
[Zinnecker2007] arXiv:0707.1279; [Kroupa2001] astro-ph/0009005; [Chabrier2003] astro-ph/0304382.
"""

import math

import numpy as np
import pytest
from scipy import integrate, stats

from kozmos.core.constants import (
    BOLTZMANN,
    ELECTRON_MASS,
    GM_SUN,
    GRAVITATIONAL_CONSTANT,
    JULIAN_YEAR_SECONDS,
    PROTON_MASS,
    SOLAR_EFFECTIVE_TEMPERATURE,
    SOLAR_LUMINOSITY,
    SOLAR_RADIUS,
    SPEED_OF_LIGHT,
)
from kozmos.stars import imf, scaling, structure

# Güneş kütlesi, kütle parametresinden: GM_⊙ / G (yerel tanım; sabitler modülüne sabit eklenmez).
M_GUNES = GM_SUN.value / GRAVITATIONAL_CONSTANT.value
R_GUNES = SOLAR_RADIUS.value
YIL = JULIAN_YEAR_SECONDS.value


# ---------------------------------------------------------------- Lane-Emden


def test_kapali_cozumler_sayisal_cozumle_uyusur():
    # n = 0: θ = 1 - ξ²/6, ξ_1 = √6; n = 1: θ = sin ξ/ξ, ξ_1 = π ([Bludman2010] Tablo II).
    for n, xi1 in ((0, math.sqrt(6.0)), (1, math.pi)):
        xi, theta, dtheta = structure.lane_emden_solve(n, 600)
        assert structure.lane_emden_first_zero(n) == pytest.approx(xi1, rel=1e-10)
        assert xi[-1] == pytest.approx(xi1, rel=1e-10)
        # Çözücü rtol = 1e-12; ξ_1'e kadar biriken hata 1e-9'dan küçük olmalı.
        assert np.allclose(theta, structure.lane_emden_analytic(xi, n), rtol=0, atol=1e-9)
    # n = 1: θ = sin ξ / ξ, dθ/dξ = cos ξ / ξ − sin ξ / ξ² (kapalı türev).
    assert np.allclose(dtheta[1:], np.cos(xi[1:]) / xi[1:] - np.sin(xi[1:]) / xi[1:] ** 2, rtol=0, atol=1e-9)


@pytest.mark.parametrize("n", [0, 1, 5])
def test_kapali_cozum_denklemi_saglar(n):
    # θ'' + 2θ'/ξ + θ^n = 0; merkezi farklarla kalıntı (adım 1e-3, hata ~ h² = 1e-6).
    xi = np.linspace(0.3, 2.2, 40)
    h = 1e-3
    f = lambda x: structure.lane_emden_analytic(x, n)  # noqa: E731
    d1 = (f(xi + h) - f(xi - h)) / (2 * h)
    d2 = (f(xi + h) - 2 * f(xi) + f(xi - h)) / h**2
    assert np.max(np.abs(d2 + 2 * d1 / xi + f(xi) ** n)) < 5e-6


def test_n5_yaricap_sonsuz():
    with pytest.raises(ValueError):
        structure.lane_emden_first_zero(5.0)
    with pytest.raises(ValueError):
        structure.lane_emden_solve(-0.1)
    with pytest.raises(ValueError):
        structure.lane_emden_analytic(1.0, 2)
    assert structure.lane_emden_analytic(0.0, 5) == 1.0


# [Bludman2010] Tablo I: (n, ξ_1, ρ_c/ρ̄, ondalık basamak sayısı). Tabloya yuvarlanmış basamak kadar yazılır;
# tolerans, son basamağın yarısının 1,2 katıdır.
TABLO_I = [
    (0, 2.449, 1.0, 3),
    (1, 3.142, 3.290, 3),
    (1.5, 3.654, 5.991, 3),
    (2, 4.353, 11.403, 3),
    (3, 6.897, 54.183, 3),
    (4, 14.972, 622.408, 3),
    (4.5, 31.836, 6189.47, 2),
]


@pytest.mark.parametrize(("n", "xi1", "ratio", "ond"), TABLO_I)
def test_bludman_tablo_i(n, xi1, ratio, ond):
    tol = 0.6 * 10**-ond
    assert structure.lane_emden_first_zero(n) == pytest.approx(xi1, abs=tol)
    assert structure.lane_emden_central_to_mean_density(n) == pytest.approx(ratio, abs=tol)


def test_kutle_katsayisi_ozdeslik_ve_n3():
    # M = 4π ρ_c (R/ξ_1)³ ω ve ρ̄ = 3M/4πR³  ⇒  ω = ξ_1³ / (3 ρ_c/ρ̄).
    for n in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        xi1 = structure.lane_emden_first_zero(n)
        ratio = structure.lane_emden_central_to_mean_density(n)
        assert structure.lane_emden_mass_coefficient(n) == pytest.approx(xi1**3 / (3 * ratio), rel=1e-10)
    # [Bludman2010] Tablo I: 0ω_3 = 2,018.
    assert structure.lane_emden_mass_coefficient(3.0) == pytest.approx(2.018, abs=6e-4)
    # n = 1: ω = π ve ρ_c/ρ̄ = π²/3 (kapalı çözüm).
    assert structure.lane_emden_mass_coefficient(1.0) == pytest.approx(math.pi, rel=1e-9)
    assert structure.lane_emden_central_to_mean_density(1.0) == pytest.approx(math.pi**2 / 3, rel=1e-9)


def test_kesirli_n_icin_sifir_bulunur():
    # 0 < n < 1'de θ^n türevi yüzeyde sonsuzdur; ξ_1 yine de θ(ξ_1) = 0 verir ve n = 0 ile n = 1 arasındadır.
    xi1 = structure.lane_emden_first_zero(0.5)
    assert math.sqrt(6.0) < xi1 < math.pi
    # Bağımsız bir katı (Radau) çözücüyle bulunan değer: 2,75269805 (25 Kasım 2026 incelemesi).
    assert xi1 == pytest.approx(2.75269805, rel=1e-8)
    xi, theta, _ = structure.lane_emden_solve(0.5, 300)
    assert abs(theta[-1]) < 1e-9 and np.all(np.diff(theta) < 0)


# ---------------------------------------------------------------- politrop nicelikleri


def test_gunes_n3_modeli_kaynak_degerleri():
    # [Liu1995]: M = 1,989e33 g, R = 6,96e10 cm için Kippenhahn-Weigert sayısal değerleri:
    # ρ_c = 76,39 g cm^-3, P_c = 1,24e17 dyn cm^-2, T_c = 1,2e7 K (μ = 0,62).
    # Kaynağın kendi girdileri kullanılır. Tam n = 3 politropu bu girdilerle 76,31 verir; kaynaktaki 76,39
    # ile fark %0,1'dir (kaynak değerinin hangi sabitlerle hesaplandığı bilinmiyor), tolerans buna göre %0,2.
    m, r = 1.989e30, 6.96e8
    assert structure.polytrope_central_density_kg_m3(m, r, 3.0) / 1e3 == pytest.approx(76.39, rel=2e-3)
    assert structure.polytrope_central_pressure_pa(m, r, 3.0) * 10.0 == pytest.approx(1.24e17, rel=1e-2)
    assert structure.polytrope_central_temperature_k(m, r, 3.0, 0.62) == pytest.approx(1.2e7, rel=5e-2)
    # [Araujo2011]: 1 M_⊙, 1 R_⊙, n = 3 için K = 3,84e14 dyn cm^2 g^{-4/3} (üç anlamlı basamak).
    # cgs -> SI: K_SI = 0,1 · (10^3)^{-4/3} K_cgs = 1e-5 K_cgs.
    assert structure.polytrope_k_coefficient_si(m, r, 3.0) / 1e-5 == pytest.approx(3.84e14, rel=2e-3)


def test_duzgun_kure_n0():
    # n = 0: P_c = (2π/3) G ρ² R², E_g = -(3/5) G M²/R (düzgün yoğunluklu küre).
    m, r = 2.0e30, 7.0e8
    rho = 3 * m / (4 * math.pi * r**3)
    assert structure.polytrope_central_density_kg_m3(m, r, 0.0) == pytest.approx(rho, rel=1e-12)
    assert structure.polytrope_central_pressure_pa(m, r, 0.0) == pytest.approx(
        2 * math.pi / 3 * GRAVITATIONAL_CONSTANT.value * rho**2 * r**2, rel=1e-9
    )
    gm2_r = GRAVITATIONAL_CONSTANT.value * m**2 / r
    assert structure.polytrope_gravitational_energy_j(m, r, 0.0) == pytest.approx(-0.6 * gm2_r, rel=1e-12)
    assert structure.polytrope_gravitational_energy_j(m, r, 1.0) == pytest.approx(-0.75 * gm2_r, rel=1e-12)
    assert structure.polytrope_gravitational_energy_j(m, r, 3.0) == pytest.approx(-1.5 * gm2_r, rel=1e-12)
    # Bağımsız yol: E_g = −∫ G m(r) dm / r, Lane-Emden profilinden. m(ξ) ∝ −ξ² θ', dm ∝ ξ² θⁿ dξ ⇒
    # E_g R / (G M²) = −ξ_1 ∫ (−ξ² θ') ξ θⁿ dξ / (ξ_1² θ'(ξ_1))²; n = 3 için −3/2 olmalı.
    xi, teta, dteta = structure.lane_emden_solve(3.0, 4000)
    tumlev = integrate.trapezoid((-(xi**2) * dteta) * xi * np.clip(teta, 0.0, None) ** 3, xi)
    assert -xi[-1] * tumlev / (xi[-1] ** 2 * dteta[-1]) ** 2 == pytest.approx(-1.5, rel=1e-5)


def test_k_ile_basinc_yolu_tutarli():
    # P_c = K ρ_c^{(n+1)/n} ([Liu1995] Denk. 41) ile kapalı biçim aynı sayıyı vermeli.
    m, r = 1.5 * M_GUNES, 1.2 * R_GUNES
    for n in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        k = structure.polytrope_k_coefficient_si(m, r, n)
        rho_c = structure.polytrope_central_density_kg_m3(m, r, n)
        assert structure.polytrope_central_pressure_pa(m, r, n) == pytest.approx(k * rho_c ** ((n + 1) / n), rel=1e-12)


def test_yaricap_gidis_donus_ve_olcekleme():
    m, r = 0.8 * M_GUNES, 0.01 * R_GUNES
    for n in (0.5, 1.5, 2.0, 4.0):
        k = structure.polytrope_k_coefficient_si(m, r, n)
        assert structure.polytrope_radius_m(m, k, n) == pytest.approx(r, rel=1e-10)
    # n = 3/2: R ∝ M^{-1/3} ([Bludman2010] Tablo I): kütle 8 kat artınca yarıçap yarıya iner.
    k = structure.polytrope_k_coefficient_si(m, r, 1.5)
    assert structure.polytrope_radius_m(8 * m, k, 1.5) == pytest.approx(0.5 * r, rel=1e-12)
    # M ∝ R^{(n-3)/(n-1)} → R ∝ M^{(n-1)/(n-3)}; n = 2'de R ∝ M^{-1}.
    k2 = structure.polytrope_k_coefficient_si(m, r, 2.0)
    assert structure.polytrope_radius_m(2 * m, k2, 2.0) == pytest.approx(0.5 * r, rel=1e-12)


def test_n4_yaricap_us_yasasi():
    # n = 4: R ∝ M^{(n-1)/(n-3)} = M^3.
    m, r = 0.8 * M_GUNES, 3.0 * R_GUNES
    k = structure.polytrope_k_coefficient_si(m, r, 4.0)
    assert structure.polytrope_radius_m(1.1 * m, k, 4.0) == pytest.approx(r * 1.1**3, rel=1e-12)


def test_gecersiz_girdiler():
    with pytest.raises(ValueError):
        structure.polytrope_k_coefficient_si(1e30, 1e9, 0.0)
    with pytest.raises(ValueError):
        structure.polytrope_radius_m(1e30, 1e10, 3.0)
    with pytest.raises(ValueError):
        structure.polytrope_radius_m(1e30, -1.0, 1.5)
    with pytest.raises(ValueError):
        structure.virial_mean_temperature_k(1e30, 1e9, 0.6, n=5.0)


def test_viriyal_sicaklik():
    m, r, mu = M_GUNES, R_GUNES, 0.62
    t = structure.virial_mean_temperature_k(m, r, mu)
    # 2T + W = 0 ile T = (3/2)(k/μ m_p) M T̄: 3 k M T̄/(μ m_p) = -E_g.
    sol = 3 * BOLTZMANN.value * m * t / (mu * PROTON_MASS.value)
    assert sol == pytest.approx(-structure.polytrope_gravitational_energy_j(m, r, 3.0), rel=1e-12)
    # Kütle ağırlıklı ortalama, merkez sıcaklığının altında kalır (T azalan profil).
    assert t < structure.polytrope_central_temperature_k(m, r, 3.0, mu)
    assert t == pytest.approx(7.2e6, rel=2e-2)
    # Düzgün küre: T̄ = G M μ m_p / (5 k R).
    assert structure.virial_mean_temperature_k(m, r, mu, n=0.0) == pytest.approx(
        GRAVITATIONAL_CONSTANT.value * m * mu * PROTON_MASS.value / (5 * BOLTZMANN.value * r), rel=1e-12
    )


# ---------------------------------------------------------------- yıldız ölçekleri


def test_stefan_boltzmann_nominal_gunes():
    # [Prsa2016]: nominal R, L, T_eff güncel σ ile birbirine tutarlıdır (L'nin belirsizliği σ'dan gelir).
    lum = scaling.luminosity_w(R_GUNES, SOLAR_EFFECTIVE_TEMPERATURE.value)
    assert lum == pytest.approx(SOLAR_LUMINOSITY.value, rel=1e-3)
    assert scaling.effective_temperature_k(SOLAR_LUMINOSITY.value, R_GUNES) == pytest.approx(
        SOLAR_EFFECTIVE_TEMPERATURE.value, rel=1e-3
    )
    assert scaling.effective_temperature_k(lum, R_GUNES) == pytest.approx(SOLAR_EFFECTIVE_TEMPERATURE.value, rel=1e-12)


def test_yuzey_cekimi_ve_log_g():
    # [Prsa2016] Tablo 'formu': log g_⊙ = 4,438068... (cgs). Altı ondalık verilmiş; tolerans 2e-6.
    assert scaling.log_surface_gravity_cgs(M_GUNES, R_GUNES) == pytest.approx(4.438068, abs=2e-6)
    assert scaling.surface_gravity_m_s2(M_GUNES, R_GUNES) == pytest.approx(GM_SUN.value / R_GUNES**2, rel=1e-12)
    # log M - 2 log R ölçeklenmesi: kütle 10 kat, yarıçap 10 kat -> log g 1 azalır.
    a = scaling.log_surface_gravity_cgs(M_GUNES, R_GUNES)
    assert scaling.log_surface_gravity_cgs(10 * M_GUNES, 10 * R_GUNES) == pytest.approx(a - 1.0, abs=1e-12)


def test_ortalama_yogunluk_gunes():
    # Güneş'in ortalama yoğunluğu yaklaşık 1,41 g cm^-3.
    assert scaling.mean_density_kg_m3(M_GUNES, R_GUNES) / 1e3 == pytest.approx(1.41, rel=5e-3)


def test_serbest_dusme_zamani_mckee():
    # [McKee2007] eq:tff: n_H = 1e3 cm^-3 için 1,37e6 yıl (He payı sayıca yüzde 10: ρ = 1,4 n_H m_H).
    m_h = PROTON_MASS.value + ELECTRON_MASS.value
    rho = 1.4 * 1e3 * 1e6 * m_h
    assert scaling.free_fall_time_s(rho) / YIL == pytest.approx(1.37e6, rel=1e-2)
    assert scaling.free_fall_time_s(4 * rho) == pytest.approx(0.5 * scaling.free_fall_time_s(rho), rel=1e-12)
    with pytest.raises(ValueError):
        scaling.free_fall_time_s(0.0)


def test_kelvin_helmholtz_gunes():
    # [Zinnecker2007]: güneş tipi yıldızlar için t_KH ~ 1e7 yıl (mertebe); G M²/(R L) ≈ 3,1e7 yıl.
    t = scaling.kelvin_helmholtz_timescale_s(M_GUNES, R_GUNES, SOLAR_LUMINOSITY.value) / YIL
    assert 1e7 < t < 1e8
    assert scaling.kelvin_helmholtz_timescale_s(M_GUNES, R_GUNES, 2 * SOLAR_LUMINOSITY.value) == pytest.approx(
        0.5 * scaling.kelvin_helmholtz_timescale_s(M_GUNES, R_GUNES, SOLAR_LUMINOSITY.value), rel=1e-12
    )
    with pytest.raises(ValueError):
        scaling.kelvin_helmholtz_timescale_s(M_GUNES, R_GUNES, 0.0)


def test_nukleer_zaman_olcegi():
    # t = ε f M c² / L; ε = 0,007 ve f = 0,1 kaynağın Güneş için verdiği değerlerdir, sonuç ≈ 10^10 yıl [Pols2011].
    t = scaling.nuclear_timescale_yr(M_GUNES, SOLAR_LUMINOSITY.value, 0.007, 0.1)
    beklenen = 0.007 * 0.1 * M_GUNES * SPEED_OF_LIGHT.value**2 / SOLAR_LUMINOSITY.value / YIL
    assert t == pytest.approx(beklenen, rel=1e-12)
    assert 1e10 < t < 1.2e10
    assert scaling.nuclear_timescale_yr(2 * M_GUNES, SOLAR_LUMINOSITY.value, 0.007, 0.1) == pytest.approx(
        2 * t, rel=1e-12
    )
    with pytest.raises(ValueError):
        scaling.nuclear_timescale_yr(M_GUNES, 1e26, 1.5, 0.1)


def test_eddington_isinim_gucu():
    kappa = 0.034  # m² kg^-1 (Thomson saçılması, X = 0,7: 0,2(1+X) cm² g^-1)
    beklenen = 4 * math.pi * SPEED_OF_LIGHT.value * GRAVITATIONAL_CONSTANT.value * M_GUNES / kappa
    assert scaling.eddington_luminosity_w(M_GUNES, kappa) == pytest.approx(beklenen, rel=1e-12)
    assert scaling.eddington_luminosity_w(2 * M_GUNES, kappa) == pytest.approx(2 * beklenen, rel=1e-12)
    assert scaling.eddington_luminosity_w(M_GUNES, 2 * kappa) == pytest.approx(0.5 * beklenen, rel=1e-12)
    with pytest.raises(ValueError):
        scaling.eddington_luminosity_w(M_GUNES, 0.0)


def test_kritik_acisal_hiz_sinirlari():
    kappa = 0.034
    kepler = math.sqrt(GM_SUN.value / R_GUNES**3)
    assert scaling.critical_angular_velocity_rad_s(M_GUNES, R_GUNES, 0.0, kappa) == pytest.approx(kepler, rel=1e-12)
    l_edd = scaling.eddington_luminosity_w(M_GUNES, kappa)
    assert scaling.critical_angular_velocity_rad_s(M_GUNES, R_GUNES, l_edd, kappa) == pytest.approx(0.0, abs=1e-30)
    assert scaling.critical_angular_velocity_rad_s(M_GUNES, R_GUNES, 0.75 * l_edd, kappa) == pytest.approx(
        0.5 * kepler, rel=1e-12
    )
    with pytest.raises(ValueError):
        scaling.critical_angular_velocity_rad_s(M_GUNES, R_GUNES, 1.01 * l_edd, kappa)


def test_eddington_olcek_kutlesi_ve_beta():
    # [Bludman2010]: M_* = 18,3 M_⊙ (üç anlamlı basamak, yarım birim 0,05 → bağıl 0,3 %).
    assert scaling.eddington_scale_mass_kg() / M_GUNES == pytest.approx(18.3, rel=3e-3)
    mu = 0.6
    for mass in (0.05, 0.5, 1.0, 10.0, 100.0):
        m = mass * M_GUNES
        b = float(scaling.eddington_beta(m, mu))
        y = m * mu**2 / scaling.eddington_scale_mass_kg()
        assert 0.0 < b <= 1.0
        assert (1 - b) / b**4 == pytest.approx(y**2, rel=1e-10)
    assert float(scaling.eddington_beta(1e-6 * M_GUNES, mu)) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):
        scaling.eddington_beta(M_GUNES, -1.0)


def test_standart_model_isinim_gucu():
    mu, kappa = 0.6, 0.034
    for mass in (0.3, 3.0, 30.0):
        m = mass * M_GUNES
        ratio = scaling.standard_model_luminosity_w(m, mu, kappa) / scaling.eddington_luminosity_w(m, kappa)
        assert ratio == pytest.approx(1.0 - scaling.eddington_beta(m, mu), rel=1e-12)
    # Küçük kütlede 1 - β ≈ y², yani L ∝ M³ ([Bludman2010]: L ∝ μ⁴ β⁴ M³).
    kucuk = 1e-2 * M_GUNES
    assert scaling.standard_model_luminosity_w(2 * kucuk, mu, kappa) / scaling.standard_model_luminosity_w(
        kucuk, mu, kappa
    ) == pytest.approx(8.0, rel=1e-3)
    # Büyük kütlede L → L_Edd.
    cok = 1e12 * M_GUNES  # β ≈ y^{-1/2} ≈ 7e-6; 1 - β = L/L_Edd 1e-4 içinde 1
    assert scaling.standard_model_luminosity_w(cok, mu, kappa) == pytest.approx(
        scaling.eddington_luminosity_w(cok, kappa), rel=1e-4
    )
    # [Bludman2010]'daki 0,003 katsayısı, (M_⊙/M_*)² = (1/18,3)² = 0,00299'a karşılık gelir.
    assert (M_GUNES / scaling.eddington_scale_mass_kg()) ** 2 == pytest.approx(0.003, rel=1e-2)


def test_chandrasekhar_kutlesi():
    # [Bludman2010]: M_Ch = 5,83 M_⊙/μ_e² ve μ_e = 2 için 1,46 M_⊙.
    assert scaling.chandrasekhar_mass_kg(1.0) / M_GUNES == pytest.approx(5.83, rel=2e-3)
    assert scaling.chandrasekhar_mass_kg(2.0) / M_GUNES == pytest.approx(1.46, abs=5e-3)
    assert scaling.chandrasekhar_mass_kg(2.0) == pytest.approx(0.25 * scaling.chandrasekhar_mass_kg(1.0), rel=1e-12)
    # K_WD ∝ μ_e^{-4/3}.
    assert scaling.white_dwarf_k_coefficient_si(4.0) == pytest.approx(
        scaling.white_dwarf_k_coefficient_si(2.0) * 2.0 ** (-4.0 / 3.0), rel=1e-12
    )
    with pytest.raises(ValueError):
        scaling.chandrasekhar_mass_kg(0.0)


def test_olcekleme_skaler_ve_dizi():
    assert np.ndim(scaling.luminosity_w(R_GUNES, 5772.0)) == 0
    assert scaling.luminosity_w(np.full((2, 3), R_GUNES), 5772.0).shape == (2, 3)
    assert scaling.eddington_beta(np.array([1.0, 2.0]) * M_GUNES, 0.6).shape == (2,)
    assert np.ndim(scaling.eddington_beta(M_GUNES, 0.6)) == 0


# ---------------------------------------------------------------- IMF


def _norm(f, low, high, noktalar=()):
    return integrate.quad(
        lambda m: float(f(m)), low, high, points=list(noktalar) or None, limit=400, epsabs=0, epsrel=1e-11
    )[0]


def _kumulatif(f, low, high):
    x = np.geomspace(low, high, 400_001)
    y = f(x)
    c = np.concatenate(([0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))))
    c /= c[-1]
    return lambda m: np.interp(m, x, c)


def test_salpeter_kapali_bicim():
    a, b, al = 0.1, 100.0, 2.35
    c = (1 - al) / (b ** (1 - al) - a ** (1 - al))
    assert _norm(imf.salpeter_dn_dm, a, b) == pytest.approx(1.0, rel=1e-9)
    assert imf.salpeter_dn_dm(1.0) == pytest.approx(c, rel=1e-12)
    # log-log eğimi -α.
    assert math.log(imf.salpeter_dn_dm(2.0) / imf.salpeter_dn_dm(1.0)) / math.log(2.0) == pytest.approx(-al, rel=1e-12)
    # ⟨m⟩ = c (b^{2-α} - a^{2-α}) / (2-α)
    ort = c * (b ** (2 - al) - a ** (2 - al)) / (2 - al)
    assert imf.salpeter_mean_mass() == pytest.approx(ort, rel=1e-12)
    # Kütle payı: (b^{2-α} - e^{2-α}) / (b^{2-α} - a^{2-α})
    e = 1.0
    pay = (b ** (2 - al) - e ** (2 - al)) / (b ** (2 - al) - a ** (2 - al))
    assert imf.salpeter_mass_fraction_above(e) == pytest.approx(pay, rel=1e-12)
    assert imf.salpeter_dn_dm(0.05) == 0.0 and imf.salpeter_dn_dm(101.0) == 0.0


@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_salpeter_ozel_us_degerleri(alpha):
    # α = 1 ve α = 2'de integraller logaritmiktir (us = -1 kolu).
    a, b = 0.2, 20.0
    assert _norm(lambda m: imf.salpeter_dn_dm(m, a, b, alpha), a, b) == pytest.approx(1.0, rel=1e-9)
    ort = _norm(lambda m: m * imf.salpeter_dn_dm(m, a, b, alpha), a, b)
    assert imf.salpeter_mean_mass(a, b, alpha) == pytest.approx(ort, rel=1e-9)
    rng = np.random.default_rng(3)
    x = imf.salpeter_sample(rng, 2000, a, b, alpha)
    assert x.min() >= a and x.max() <= b
    assert stats.kstest(x, _kumulatif(lambda m: imf.salpeter_dn_dm(m, a, b, alpha), a, b)).pvalue > 0.01


def test_kroupa_makaledeki_sayilar():
    # [Kroupa2001]: 0,01-50 M_⊙ için ⟨m⟩ = 0,36 M_⊙; sayı payları BC %37, M cücesi %48, K cücesi %8,9,
    # ara kütle (1-8) %5,7, O yıldızları (> 8) %0,37; kütle payları sırasıyla %4,3, 28, 17, 34, 17.
    assert imf.kroupa_mean_mass() == pytest.approx(0.36, abs=5e-3)
    sinirlar = [0.01, 0.08, 0.5, 1.0, 8.0, 50.0]
    number = [0.37, 0.48, 0.089, 0.057, 0.0037]
    sayi_tol = [5e-3, 5e-3, 6e-4, 6e-4, 6e-5]
    mass = [0.043, 0.28, 0.17, 0.34, 0.17]
    kutle_tol = [6e-4, 5e-3, 5e-3, 5e-3, 5e-3]
    for i in range(5):
        a, b = sinirlar[i], sinirlar[i + 1]
        s = _norm(imf.kroupa_dn_dm, a, b)
        k = float(imf.kroupa_mass_fraction_above(a)) - float(imf.kroupa_mass_fraction_above(b))
        assert s == pytest.approx(number[i], abs=sayi_tol[i]), (a, b)
        assert k == pytest.approx(mass[i], abs=kutle_tol[i]), (a, b)
    # Kütlenin yarısı 1 M_⊙'in altında (makale).
    assert float(imf.kroupa_mass_fraction_above(1.0)) == pytest.approx(0.5, abs=1e-2)


def test_kroupa_normalizasyon_ve_surekli():
    assert _norm(imf.kroupa_dn_dm, 0.01, 50.0, (0.08, 0.5, 1.0)) == pytest.approx(1.0, rel=1e-9)
    for b in (0.08, 0.5, 1.0):
        sol, sag = imf.kroupa_dn_dm(b * (1 - 1e-12)), imf.kroupa_dn_dm(b * (1 + 1e-12))
        assert sag == pytest.approx(sol, rel=1e-9)
    # Eğimler α_i.
    for m, alpha in ((0.03, 0.3), (0.2, 1.3), (0.7, 2.3), (5.0, 2.3)):
        egim = math.log(imf.kroupa_dn_dm(m * 1.01) / imf.kroupa_dn_dm(m)) / math.log(1.01)
        assert egim == pytest.approx(-alpha, abs=1e-9)
    with pytest.raises(ValueError):
        imf.kroupa_dn_dm(0.5, m_low=0.005)


def test_chabrier_formu_ve_surekliligi():
    n = _norm(imf.chabrier_dn_dm, 0.01, 50.0, (1.0,))
    assert n == pytest.approx(1.0, rel=1e-9)
    # ξ(log m) = N^{-1} ... : dN/dm · m ln10 · N_toplam, kaynaktaki ξ(log m)'e eşit olmalı ([Chabrier2003] Tablo 1).
    toplam = integrate.quad(
        lambda m: (
            (
                0.158 * math.exp(-((math.log10(m) - math.log10(0.079)) ** 2) / (2 * 0.69**2))
                if m <= 1
                else 4.43e-2 * m**-1.3
            )
            / (m * math.log(10))
        ),
        0.01,
        50.0,
        points=[1.0],
        limit=400,
        epsabs=0,
        epsrel=1e-11,
    )[0]
    for m in (0.02, 0.079, 0.3, 0.9):
        xi = 0.158 * math.exp(-((math.log10(m) - math.log10(0.079)) ** 2) / (2 * 0.69**2))
        assert imf.chabrier_dn_dm(m) * toplam == pytest.approx(xi / (m * math.log(10)), rel=1e-12)
    for m in (1.5, 10.0):
        assert imf.chabrier_dn_dm(m) * toplam == pytest.approx(4.43e-2 * m**-1.3 / (m * math.log(10)), rel=1e-12)
    # Tablodaki iki katsayı m = 1'de yaklaşık %0,5 süreksizlik bırakır.
    sol, sag = imf.chabrier_dn_dm(1.0 - 1e-12), imf.chabrier_dn_dm(1.0 + 1e-12)
    assert sag == pytest.approx(sol, rel=1e-2)
    # dN/dlog m, log-normal parçada m_c = 0,079'da tepe yapar.
    m = np.geomspace(0.01, 1.0, 20001)
    assert m[np.argmax(m * imf.chabrier_dn_dm(m))] == pytest.approx(0.079, rel=2e-3)


def test_chabrier_ortalama_ve_kutle_orani_sayisal():
    for low, high in ((0.01, 50.0), (0.1, 100.0), (0.02, 0.9)):
        ort = _norm(
            lambda m, low=low, high=high: m * imf.chabrier_dn_dm(m, low, high), low, high, (1.0,) if high > 1 else ()
        )
        assert imf.chabrier_mean_mass(low, high) == pytest.approx(ort, rel=1e-9)
    ust_k = _norm(lambda m: m * imf.chabrier_dn_dm(m), 1.0, 50.0)
    tum_k = _norm(lambda m: m * imf.chabrier_dn_dm(m), 0.01, 50.0, (1.0,))
    assert imf.chabrier_mass_fraction_above(1.0) == pytest.approx(ust_k / tum_k, rel=1e-9)
    # Mantıklı aralık: disk IMF'inin ortalama kütlesi Kroupa'ya yakın ama biraz düşüktür.
    assert 0.30 < imf.chabrier_mean_mass() < imf.kroupa_mean_mass()


def test_kroupa_ortalama_sayisal_ve_kirpma():
    ort = _norm(lambda m: m * imf.kroupa_dn_dm(m, 0.05, 20.0), 0.05, 20.0, (0.08, 0.5, 1.0))
    assert imf.kroupa_mean_mass(0.05, 20.0) == pytest.approx(ort, rel=1e-9)
    # Eşik aralığın dışındaysa 1 ya da 0.
    assert imf.kroupa_mass_fraction_above(0.001) == pytest.approx(1.0, abs=1e-15)
    assert imf.kroupa_mass_fraction_above(500.0) == pytest.approx(0.0, abs=1e-15)
    assert imf.kroupa_mass_fraction_above(np.array([0.1, 1.0, 10.0])).shape == (3,)
    assert np.ndim(imf.kroupa_dn_dm(0.3)) == 0
    with pytest.raises(ValueError):
        imf.chabrier_dn_dm(1.0, m_low=2.0, m_high=1.0)


# Örnekleyici sınamaları: sabit tohumla Kolmogorov-Smirnov. Doğru bir örnekleyicide p-değeri (0,1)'de düzgündür;
# tohum sabit olduğu için sonuç yinelenebilirdir ve eşik 0,01 seçilmiştir. Sınama, CDF'i herkese açık
# dN/dm'nin sayısal birikimli integralinden kurar, örnekleyicinin kendi kodundan bağımsızdır.
ORNEKLEYICILER = [
    ("salpeter", imf.salpeter_sample, imf.salpeter_dn_dm, imf.salpeter_mean_mass, 0.1, 100.0),
    ("kroupa", imf.kroupa_sample, imf.kroupa_dn_dm, imf.kroupa_mean_mass, 0.01, 50.0),
    ("chabrier", imf.chabrier_sample, imf.chabrier_dn_dm, imf.chabrier_mean_mass, 0.01, 50.0),
]


@pytest.mark.parametrize(("name", "sample", "pdf", "ortalama", "low", "high"), ORNEKLEYICILER)
def test_ornekleyici_dagilimi(name, sample, pdf, ortalama, low, high):
    rng = np.random.default_rng(20261125)
    x = sample(rng, 4000)
    assert x.shape == (4000,) and x.min() >= low and x.max() <= high
    assert stats.kstest(x, _kumulatif(pdf, low, high)).pvalue > 0.01, name
    # Örnek ortalaması, kendi standart hatasının 4 katı içinde beklenen değerdedir.
    y = sample(np.random.default_rng(7), 200_000)
    assert abs(y.mean() - ortalama()) < 4 * y.std() / math.sqrt(y.size), name


def test_ornekleyici_yinelenebilir_ve_bos():
    a = imf.kroupa_sample(np.random.default_rng(1), 50)
    b = imf.kroupa_sample(np.random.default_rng(1), 50)
    assert np.array_equal(a, b)
    assert imf.chabrier_sample(np.random.default_rng(1), 0).shape == (0,)
    with pytest.raises(ValueError):
        imf.salpeter_sample(np.random.default_rng(1), -1)
