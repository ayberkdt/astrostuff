"""Kara delik modülünün kaynaktaki sayısal değerlere ve bağımsız hesaplara karşı sınanması.

Sayısal değerler şu kaynaklardan okunmuştur: [EHT2019] (M87* gölge çapı), [Abramowicz2013]
(Eddington gücü, Hawking sıcaklığı dipnotu), [Inayoshi2020] (Tablo 3, Salpeter süresi),
[Komossa2015] (gelgit yarıçapı). Bağımsız hesaplar: sayısal en küçükleme, sayısal tümlev,
Kerr metriğinin Δ = 0 özdeşliği ve termodinamiğin birinci yasası.
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
YIL = JULIAN_YEAR_SECONDS.value
MPC = 1e6 * PARSEC.value
PC = PARSEC.value
ERG_S_W = 1e-7  # 1 erg/s = 1e-7 W


# ----------------------------------------------------------------------------- Schwarzschild


def test_yaricaplar_gunes_icin():
    # r_G = GM/c² ve R_S = 2 r_G [EHT2019 §1]; Güneş için 1,477 km ve 2,953 km.
    assert basics.gravitational_radius_m(MSUN) == pytest.approx(1476.6e0, rel=1e-3)
    assert basics.schwarzschild_radius_m(MSUN) == pytest.approx(2953.3e0, rel=1e-3)
    assert basics.schwarzschild_radius_m(MSUN) == pytest.approx(2.0 * G * MSUN / C**2, rel=1e-14)


def test_kirmiziya_kayma_kapali_ve_sinir():
    rs = basics.schwarzschild_radius_m(MSUN)
    # r = 3 R_S / 2: 1 − R_S/r = 1/3 ⇒ 1 + z = √3.
    assert basics.gravitational_redshift(1.5 * rs, MSUN) == pytest.approx(np.sqrt(3.0) - 1.0, rel=1e-14)
    # Zayıf alan: z ≈ R_S / (2 r); birinci düzeyde göreli hata ~ 3 R_S/(4 r) = 7,5e-9 (r = 1e8 R_S).
    assert basics.gravitational_redshift(1e8 * rs, MSUN) == pytest.approx(0.5e-8, rel=1e-7)
    with pytest.raises(ValueError):
        basics.gravitational_redshift(rs, MSUN)


def test_foton_kuresi():
    assert basics.photon_sphere_radius_m(MSUN) == pytest.approx(1.5 * basics.schwarzschild_radius_m(MSUN), rel=1e-14)
    # Kerr formülünün a* = 0 değeri aynı sonucu vermeli.
    assert basics.kerr_photon_orbit_radius_rg(0.0) == pytest.approx(3.0, abs=1e-14)


def test_kritik_carpisma_parametresi_bagimsiz_en_kucuk():
    # Bağımsız yol: Schwarzschild'de b(r) = r / √(1 − R_S/r) fotonun en yakın yaklaşımındaki
    # çarpışma parametresidir; yakalanma sınırı bu fonksiyonun en küçük değeridir.
    rg = basics.gravitational_radius_m(MSUN)
    b = lambda x: x / np.sqrt(1.0 - 2.0 / x)  # noqa: E731  (x = r / r_G)
    sonuc = minimize_scalar(b, bounds=(2.001, 20.0), method="bounded", options={"xatol": 1e-12})
    assert sonuc.x == pytest.approx(3.0, rel=1e-6)
    assert basics.critical_impact_parameter_m(MSUN) == pytest.approx(sonuc.fun * rg, rel=1e-12)
    # [EHT2019] §1: R_c = √27 r_g.
    assert basics.critical_impact_parameter_m(MSUN) == pytest.approx(np.sqrt(27.0) * rg, rel=1e-14)


def test_m87_golge_capi_kaynaktaki_degerler():
    # [EHT2019] §2: D = 16,8 Mpc için Schwarzschild gölge çapı, yıldız dinamiği kütlesi
    # (6,2e9 M☉) ile 37,6 µas, gaz dinamiği kütlesi (3,5e9 M☉) ile 21,3 µas.
    # Tolerans %1: kaynak kütleleri iki, çapları üç anlamlı basamakla yuvarlar.
    d = 16.8 * MPC
    assert basics.shadow_angular_diameter_uas(6.2e9 * MSUN, d) == pytest.approx(37.6, rel=1e-2)
    assert basics.shadow_angular_diameter_uas(3.5e9 * MSUN, d) == pytest.approx(21.3, rel=1e-2)


def test_golge_capi_olceklemesi():
    d = 10.0 * MPC
    assert basics.shadow_angular_diameter_uas(2e9 * MSUN, 2 * d) == pytest.approx(
        basics.shadow_angular_diameter_uas(1e9 * MSUN, d), rel=1e-14
    )


def test_isco_schwarzschild():
    assert basics.schwarzschild_isco_radius_m(MSUN) == pytest.approx(
        3.0 * basics.schwarzschild_radius_m(MSUN), rel=1e-14
    )
    assert basics.kerr_isco_radius_rg(0.0) == pytest.approx(6.0, abs=1e-13)


def test_isco_yorunge_frekansi():
    # Ω_K² = GM/r³ ile bağımsız hesap; Güneş kütlesi için ≈ 2,2 kHz.
    r = 6.0 * G * MSUN / C**2
    beklenen = np.sqrt(G * MSUN / r**3) / (2.0 * np.pi)
    assert basics.isco_orbital_frequency_hz(MSUN) == pytest.approx(beklenen, rel=1e-13)
    assert basics.isco_orbital_frequency_hz(MSUN) == pytest.approx(2.2e3, rel=1e-2)
    assert basics.isco_orbital_frequency_hz(10 * MSUN) == pytest.approx(
        basics.isco_orbital_frequency_hz(MSUN) / 10, rel=1e-13
    )


# ----------------------------------------------------------------------------- Kerr


def test_kerr_isco_sinir_degerleri():
    # Bilinen uç değerler: a* = 1 doğru yörünge r_G, a* = −1 ters yörünge 9 r_G.
    assert basics.kerr_isco_radius_rg(1.0) == pytest.approx(1.0, abs=1e-12)
    assert basics.kerr_isco_radius_rg(-1.0) == pytest.approx(9.0, abs=1e-12)
    assert basics.kerr_isco_radius_rg(0.998) == pytest.approx(1.2370, abs=1e-4)


@pytest.mark.parametrize("a", np.linspace(-1.0, 1.0, 21))
def test_kerr_isco_radyal_yariyayma_kosulu(a):
    # [Abramowicz2013] TeX etiketi epicyclic-radial: (ω_r*)² ∝ 1 − 6/x + 8 a* x^{-3/2} − 3 a*² x^{-2};
    # ISCO bu ifadenin sıfır olduğu en dış yarıçaptır (a* < 0: ters yörünge). Formülden bağımsız sınav.
    x = basics.kerr_isco_radius_rg(a)
    assert 1.0 - 6.0 / x + 8.0 * a * x**-1.5 - 3.0 * a**2 / x**2 == pytest.approx(0.0, abs=1e-11)


@pytest.mark.parametrize("a", [-0.99, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 0.99])
def test_isco_ozgul_enerji_dairesel_yorunge_minimumu(a):
    # Bağımsız yol: dairesel yörünge enerjisi E(x) = (x^{3/2} − 2 x^{1/2} + a) / (x^{3/4} √(x^{3/2} − 3 x^{1/2} + 2a))
    # (Bardeen-Press-Teukolsky, bellekten) ISCO'da en küçüktür: dE/dx = 0.
    def enerji(x):
        return (x**1.5 - 2 * x**0.5 + a) / (x**0.75 * np.sqrt(x**1.5 - 3 * x**0.5 + 2 * a))

    x0 = float(basics.kerr_isco_radius_rg(a))
    h = 1e-5 * x0
    turev = (enerji(x0 + h) - enerji(x0 - h)) / (2 * h)
    assert abs(turev) < 1e-8
    assert enerji(1.02 * x0) > enerji(x0)
    assert enerji(x0) == pytest.approx(float(basics.kerr_isco_specific_energy(a)), rel=1e-9)


def test_isco_enerji_ve_verim_degerleri():
    assert basics.kerr_isco_specific_energy(0.0) == pytest.approx(np.sqrt(8.0 / 9.0), rel=1e-13)
    assert basics.isco_radiative_efficiency(0.0) == pytest.approx(1.0 - np.sqrt(8.0 / 9.0), rel=1e-12)
    # a* = 1: E = 1/√3, η = 1 − 1/√3 ≈ 0,4226.
    assert basics.isco_radiative_efficiency(1.0) == pytest.approx(1.0 - 1.0 / np.sqrt(3.0), rel=1e-10)
    # [Inayoshi2020] §1: dönmeyen kara delik için "~0,06"; kaynak tek basamak verir, tolerans 0,0035.
    assert abs(float(basics.isco_radiative_efficiency(0.0)) - 0.06) < 0.0035


@pytest.mark.parametrize("a", np.linspace(-1.0, 1.0, 9))
def test_ufuk_kerr_metrik_ozdesligi(a):
    # Kerr metriğinde ufuk Δ = r² − 2 r + a² = 0 (r_G biriminde) denkleminin büyük köküdür.
    rp = basics.kerr_horizon_radius_rg(a)
    assert rp**2 - 2.0 * rp + a**2 == pytest.approx(0.0, abs=1e-14)


def test_ufuk_ve_ergosfer_degerleri():
    assert basics.kerr_horizon_radius_rg(0.0) == 2.0
    assert basics.kerr_horizon_radius_rg(1.0) == 1.0
    assert basics.kerr_horizon_radius_rg(0.0) * basics.gravitational_radius_m(MSUN) == pytest.approx(
        basics.schwarzschild_radius_m(MSUN), rel=1e-14
    )
    # Ergosfer: ekvatorda 2 r_G (her spin için), kutuplarda ufka teğet.
    assert basics.kerr_ergosphere_radius_rg(0.7) == pytest.approx(2.0, abs=1e-15)
    assert basics.kerr_ergosphere_radius_rg(0.7, 0.0) == pytest.approx(basics.kerr_horizon_radius_rg(0.7), rel=1e-14)
    assert basics.kerr_ergosphere_radius_rg(0.0, 0.3) == pytest.approx(2.0, abs=1e-15)


def test_ufuk_acisal_hizi():
    m = 10.0 * MSUN
    # a* = 1: Ω_H = c³ / (2 G M).
    assert basics.horizon_angular_velocity_rad_s(m, 1.0) == pytest.approx(C**3 / (2 * G * m), rel=1e-14)
    # Küçük a*: a* c³ / (4 G M).
    assert basics.horizon_angular_velocity_rad_s(m, 1e-5) == pytest.approx(1e-5 * C**3 / (4 * G * m), rel=1e-10)
    # Bağımsız yol (Page2005 eq:3: Ω = 4π a / A, A = 4π (r_+² + a²)): Ω_H = a* c / r_G / (r_+² + a*²).
    a = 0.6
    rp = float(basics.kerr_horizon_radius_rg(a))
    beklenen = (C / basics.gravitational_radius_m(m)) * a / (rp**2 + a**2)
    assert basics.horizon_angular_velocity_rad_s(m, a) == pytest.approx(beklenen, rel=1e-13)
    assert basics.horizon_angular_velocity_rad_s(m, -a) == pytest.approx(-beklenen, rel=1e-13)


@pytest.mark.parametrize("a", np.linspace(-1.0, 1.0, 21))
def test_foton_yorungesi_kubik(a):
    # [Abramowicz2013]: y = √(r/r_G) için y³ − 3y + 2a* = 0 kübiğinin (Novikov-Thorne bölümü) kökü.
    y = np.sqrt(basics.kerr_photon_orbit_radius_rg(a))
    assert y**3 - 3.0 * y + 2.0 * a == pytest.approx(0.0, abs=1e-12)


def test_foton_yorungesi_sinir_degerleri():
    assert basics.kerr_photon_orbit_radius_rg(1.0) == pytest.approx(1.0, abs=1e-12)
    assert basics.kerr_photon_orbit_radius_rg(-1.0) == pytest.approx(4.0, abs=1e-12)


# ----------------------------------------------------------------------------- Hawking


def test_hawking_sicakligi():
    # [Abramowicz2013] §1 dipnotu: yıldız kütleli kara delik için T_H ~ 1e-8 K (mertebe).
    t = basics.hawking_temperature_k(MSUN)
    assert 1e-8 < t < 1e-7
    # Bağımsız yol 1: T = ħ κ / (2π c k_B), Schwarzschild yüzey çekimi κ = c⁴ / (4 G M) [Page2005: κ = 1/(4M)].
    kappa = C**4 / (4.0 * G * MSUN)
    assert t == pytest.approx(HBAR * kappa / (2.0 * np.pi * C * BOLTZMANN.value), rel=1e-13)
    # Bağımsız yol 2: T = ħ c / (4π k_B R_S).
    assert t == pytest.approx(
        HBAR * C / (4.0 * np.pi * BOLTZMANN.value * basics.schwarzschild_radius_m(MSUN)), rel=1e-13
    )


def test_entropi_alan_ve_birinci_yasa():
    # S/k_B = A / (4 ℓ_P²), ℓ_P² = ħ G / c³, A = 4π R_S² [Page2005 eq:2: S = A/4].
    rs = basics.schwarzschild_radius_m(MSUN)
    s = basics.bekenstein_hawking_entropy(MSUN)
    assert s == pytest.approx(4.0 * np.pi * rs**2 / (4.0 * HBAR * G / C**3), rel=1e-13)
    assert s == pytest.approx(1.05e77, rel=1e-2)
    # Birinci yasa: d(S k_B) = c² dM / T ⇒ d(S/k_B)/dM = c² / (k_B T). Merkezi fark, bağıl adım 1e-4.
    h = 1e-4 * MSUN
    turev = (basics.bekenstein_hawking_entropy(MSUN + h) - basics.bekenstein_hawking_entropy(MSUN - h)) / (2 * h)
    assert turev == pytest.approx(C**2 / (BOLTZMANN.value * basics.hawking_temperature_k(MSUN)), rel=1e-7)


def test_buharlasma_suresi_bagimsiz_tumlev():
    # dM/dt = −σ_SB T_H⁴ A / c²  ⇒  t = ∫_0^M c² dM' / (σ T⁴ A); bağımsız sayısal tümlev (log uzayında).
    def isinim_gucu(m):
        t = basics.hawking_temperature_k(m)
        alan = 4.0 * np.pi * basics.schwarzschild_radius_m(m) ** 2
        return STEFAN_BOLTZMANN.value * t**4 * alan

    m0 = 1e12  # kg
    t, _ = quad(
        lambda u: np.exp(u) * C**2 / isinim_gucu(np.exp(u)), np.log(1e-3 * m0), np.log(m0), epsabs=0, epsrel=1e-12
    )
    # Alt sınır 1e-3 m0: kalan katkı (1e-3)³ oranında, 1e-9 bağıl hata.
    assert basics.hawking_evaporation_time_blackbody_s(m0) == pytest.approx(t, rel=2e-9)
    # Güneş kütlesi için ≈ 2e67 yıl.
    assert 1e67 < basics.hawking_evaporation_time_blackbody_s(MSUN) / YIL < 3e67


# ----------------------------------------------------------------------------- ikili


def test_cirp_kutlesi_bicimleri():
    m1, m2 = 36.0 * MSUN, 29.0 * MSUN
    m, mu, nu = m1 + m2, m1 * m2 / (m1 + m2), m1 * m2 / (m1 + m2) ** 2
    mc = basics.chirp_mass_kg(m1, m2)
    # [Sathyaprakash2009] eqn:chirpmassdef: 𝓜 = μ^{3/5} M^{2/5} = ν^{3/5} M.
    assert mc == pytest.approx(mu**0.6 * m**0.4, rel=1e-13)
    assert mc == pytest.approx(nu**0.6 * m, rel=1e-13)
    assert basics.chirp_mass_kg(m1, m1) == pytest.approx(m1 * 2.0**-0.2, rel=1e-14)


def test_frekans_turevi_peters_denkleminden():
    # Bağımsız yol: [Sathyaprakash2009] eqn:circularPM, Ṗ_b = −(192π/5) (2π 𝓜/P_b)^{5/3} (G = c = 1),
    # f = 2/P_b ⇒ ḟ = −2 Ṗ_b / P_b².
    mc = basics.chirp_mass_kg(1.4 * MSUN, 1.4 * MSUN)
    mc_s = G * mc / C**3
    for f in (10.0, 50.0, 300.0):
        pb = 2.0 / f
        pb_nokta = -192.0 * np.pi / 5.0 * (2.0 * np.pi * mc_s / pb) ** (5.0 / 3.0)
        assert basics.gw_frequency_derivative_hz_s(mc, f) == pytest.approx(-2.0 * pb_nokta / pb**2, rel=1e-12)


def test_birlesmeye_kalan_sure_tumlev_ve_nu_bicimi():
    m1, m2 = 1.4 * MSUN, 1.4 * MSUN
    mc = basics.chirp_mass_kg(m1, m2)
    f0 = 20.0
    # Bağımsız yol 1: τ = ∫_f0^∞ df / ḟ.
    t, _ = quad(lambda f: 1.0 / basics.gw_frequency_derivative_hz_s(mc, f), f0, np.inf, epsrel=1e-12)
    assert basics.time_to_merger_s(mc, f0) == pytest.approx(t, rel=1e-9)
    # Bağımsız yol 2: kaynaktaki ν biçimi τ0 = 5 / (256 π ν f) (π M f)^{-5/3}, G = c = 1.
    m_s = G * (m1 + m2) / C**3
    nu = m1 * m2 / (m1 + m2) ** 2
    assert basics.time_to_merger_s(mc, f0) == pytest.approx(
        5.0 / (256.0 * np.pi * nu * f0) * (np.pi * m_s * f0) ** (-5.0 / 3.0), rel=1e-12
    )
    # Büyüklük sınaması: 1,4 + 1,4 M☉ ikili 20 Hz'de birleşmeye Newton mertebesinde ~158 s kalır (kaynaktan değil, bu hesaptan).
    assert basics.time_to_merger_s(mc, f0) == pytest.approx(158.0, rel=2e-2)


def test_kalan_sureden_frekans_gidis_donus():
    mc = basics.chirp_mass_kg(30.0 * MSUN, 30.0 * MSUN)
    f = np.array([10.0, 35.0, 100.0, 250.0])
    assert basics.gw_frequency_from_time_to_merger_hz(mc, basics.time_to_merger_s(mc, f)) == pytest.approx(f, rel=1e-13)


def test_isco_dalga_frekansi():
    assert basics.gw_frequency_at_isco_hz(MSUN) == pytest.approx(
        2.0 * basics.isco_orbital_frequency_hz(MSUN), rel=1e-14
    )
    # c³ / (6^{3/2} π G M☉) ≈ 4,4 kHz (yaygın büyüklük; kaynaktan okunmadı, %1).
    assert basics.gw_frequency_at_isco_hz(MSUN) == pytest.approx(4.4e3, rel=1e-2)
    assert basics.gw_frequency_at_isco_hz(20.0 * MSUN) == pytest.approx(
        basics.gw_frequency_at_isco_hz(MSUN) / 20.0, rel=1e-13
    )


# ----------------------------------------------------------------------------- birikim


def test_eddington_isinim_gucu():
    # [Abramowicz2013] §1 dipnotu: L_Edd ≡ 1,2e38 (M/M☉) erg/s. Kaynak iki basamak verir; hidrojen için
    # 4π G M m_p c / σ_T = 1,26e38 erg/s olduğundan tolerans %6.
    l_edd = accretion.eddington_luminosity_w(MSUN)
    assert l_edd / ERG_S_W == pytest.approx(1.2e38, rel=6e-2)
    kapa = THOMSON_CROSS_SECTION.value / PROTON_MASS.value
    assert l_edd == pytest.approx(4.0 * np.pi * G * MSUN * C / kapa, rel=1e-13)


def test_eddington_kutle_akisi_tablo_3():
    # [Inayoshi2020] Tablo 3: Ṁ_Edd = L_Edd/(0,1 c²) ≈ 2,3e-5 M☉/yıl (10³ M☉). κ_es verilmediğinden %5.
    mdot = accretion.eddington_accretion_rate_kg_s(1e3 * MSUN)
    assert mdot * YIL / MSUN == pytest.approx(2.3e-5, rel=5e-2)
    # L = η Ṁ c² gidiş-dönüş ve 1/η ölçeklemesi.
    assert accretion.accretion_luminosity_w(mdot, 0.1) == pytest.approx(
        accretion.eddington_luminosity_w(1e3 * MSUN), rel=1e-13
    )
    assert accretion.eddington_accretion_rate_kg_s(MSUN, 0.2) == pytest.approx(
        accretion.eddington_accretion_rate_kg_s(MSUN, 0.1) / 2, rel=1e-14
    )


def test_salpeter_zamani_katsayilar_ve_ustel_buyume():
    # [Inayoshi2020]: katsayı 0,45 Gyr; η = 0,1 için t_Edd ≈ 50 Myr (%1: kaynak iki basamak verir).
    t_sal = accretion.salpeter_time_s(0.1)
    assert t_sal / (1e9 * YIL) == pytest.approx(0.45 * 0.1 / 0.9, rel=1e-2)
    assert t_sal / (1e6 * YIL) == pytest.approx(50.0, rel=1e-2)
    # Bağımsız yol: dM/dt = (1 − η) Ṁ_Edd(M) sayısal çözümü, t = t_Sal sonunda M = e M0.
    eta = 0.1
    m0 = 100.0 * MSUN

    def sag(t, m):
        return (1.0 - eta) * accretion.eddington_accretion_rate_kg_s(m, eta)

    cozum = solve_ivp(sag, (0.0, t_sal), [m0], rtol=1e-11, atol=0.0, method="DOP853")
    assert cozum.y[0, -1] / m0 == pytest.approx(np.e, rel=1e-8)


def test_bondi_tablo_3():
    # [Inayoshi2020] Tablo 3, T = 10³ K, n_H = 1e4 cm^-3, M = 10³ M☉. Kaynak c_s ve ρ tanımını vermez;
    # c_s² = k T / (μ m_p), ρ = μ m_p n_H, μ = 1,22 bir VARSAYIMDIR ve tabloya göre seçilmiştir: R_B ∝ μ,
    # Ṁ_B ∝ μ^{5/2} olduğundan bu test önçarpanı sınar (yanlış önçarpan %5'te yakalanır) ama μ'yü sınamaz.
    mu = 1.22
    cs = np.sqrt(BOLTZMANN.value * 1e3 / (mu * PROTON_MASS.value))
    rho = mu * PROTON_MASS.value * 1e10  # m^-3
    assert accretion.bondi_radius_m(1e3 * MSUN, cs) / PC == pytest.approx(0.6, rel=0.10)  # tabloda tek basamak
    assert accretion.bondi_accretion_rate_kg_s(1e3 * MSUN, rho, cs) * YIL / MSUN == pytest.approx(4.5e-3, rel=0.05)
    # Özdeşlik: G²M²/c_s³ = R_B² c_s.
    rb = accretion.bondi_radius_m(1e3 * MSUN, cs)
    assert accretion.bondi_accretion_rate_kg_s(1e3 * MSUN, rho, cs) == pytest.approx(
        np.pi * np.exp(1.5) * rho * rb**2 * cs, rel=1e-13
    )


R_IC = 6.0 * G * 10 * MSUN / C**2  # 10 M☉ kara delik, ISCO (Schwarzschild)
MDOT_DISK = 1e-8 * MSUN / YIL


def test_ince_disk_profil_sinirlari():
    m = 10 * MSUN
    assert accretion.thin_disk_effective_temperature_k(R_IC, m, MDOT_DISK, R_IC) == 0.0
    # r ≫ r_in: T ∝ r^{-3/4}; 1 − √(r_in/r) düzeltmesi r = 1e6 r_in'de ~1e-3.
    t1 = accretion.thin_disk_effective_temperature_k(1e6 * R_IC, m, MDOT_DISK, R_IC)
    t2 = accretion.thin_disk_effective_temperature_k(1e7 * R_IC, m, MDOT_DISK, R_IC)
    assert t2 / t1 == pytest.approx(10.0**-0.75, rel=2e-3)
    with pytest.raises(ValueError):
        accretion.thin_disk_effective_temperature_k(0.5 * R_IC, m, MDOT_DISK, R_IC)


def test_ince_disk_enerji_korunumu():
    # Bağımsız sınav: iki yüzden yayılan toplam güç, Newton diskinde r_in'de sıfır torkla
    # L = G M Ṁ / (2 r_in) olmalıdır (Kepler enerjisinin yarısı yayılır).
    m = 10 * MSUN

    def akı(u):
        # u = √(r_in/r) ∈ (0, 1] değişkenine geçiş: r = r_in/u², dr = 2 r_in/u³ du (sonsuz aralık sonlu olur).
        r = R_IC / u**2
        yayim = 2.0 * STEFAN_BOLTZMANN.value * accretion.thin_disk_effective_temperature_k(r, m, MDOT_DISK, R_IC) ** 4
        return yayim * 2.0 * np.pi * r * 2.0 * R_IC / u**3

    # Alt sınır 1e-9: u → 0 katkısı integrandın ∝ u davranışından ~1e-18 bağıl.
    toplam, _ = quad(akı, 1e-9, 1.0, epsrel=1e-12, limit=200)
    assert toplam == pytest.approx(G * m * MDOT_DISK / (2.0 * R_IC), rel=1e-8)


def test_ince_disk_azami_sicaklik():
    m = 10 * MSUN
    r_az = accretion.thin_disk_peak_temperature_radius_m(R_IC)
    assert r_az == pytest.approx(49.0 / 36.0 * R_IC, rel=1e-15)
    # Bağımsız yol: profilin sayısal en büyüğü.
    sonuc = minimize_scalar(
        lambda r: -accretion.thin_disk_effective_temperature_k(r, m, MDOT_DISK, R_IC),
        bounds=(1.0001 * R_IC, 20.0 * R_IC),
        method="bounded",
        options={"xatol": 1e-9 * R_IC},
    )
    assert sonuc.x == pytest.approx(r_az, rel=1e-5)
    assert accretion.thin_disk_peak_temperature_k(m, MDOT_DISK, R_IC) == pytest.approx(-sonuc.fun, rel=1e-10)
    assert accretion.thin_disk_peak_temperature_k(m, MDOT_DISK, R_IC) == pytest.approx(
        accretion.thin_disk_effective_temperature_k(r_az, m, MDOT_DISK, R_IC), rel=1e-13
    )
    # Sabit çarpan: (6^6 / 7^7)^{1/4} ≈ 0,488.
    t_ref = (3.0 * G * m * MDOT_DISK / (8.0 * np.pi * STEFAN_BOLTZMANN.value * R_IC**3)) ** 0.25
    assert accretion.thin_disk_peak_temperature_k(m, MDOT_DISK, R_IC) / t_ref == pytest.approx(0.488, abs=5e-4)


def test_gelgit_yaricapi_kaynak_katsayisi():
    # [Komossa2015] §1: r_t ≃ 7e12 cm (M/10⁶ M☉)^{1/3} (m*/M☉)^{-1/3} (r*/R☉); kaynak bir basamak sonrası
    # yuvarlar (6,96e12 → 7e12), tolerans %1.
    r_gunes = 6.957e8
    rt = accretion.tidal_disruption_radius_m(1e6 * MSUN, MSUN, r_gunes)
    assert rt * 100.0 == pytest.approx(7e12, rel=1e-2)
    # 8 kat kütle → 2 kat yarıçap.
    assert accretion.tidal_disruption_radius_m(8e6 * MSUN, MSUN, r_gunes) == pytest.approx(2.0 * rt, rel=1e-13)


def test_hills_kutlesi():
    r_gunes = 6.957e8
    mh = accretion.hills_mass_kg(MSUN, r_gunes)
    # [Komossa2015] §1: Güneş türü yıldız ≳ 1e8 M☉'de bütün yutulur (mertebe; faktör 2 içinde).
    assert 0.5e8 < mh / MSUN < 2e8
    # Beyaz cüce (0,6 M☉, 0,01 R☉): kaynak "≲ 1e5 M☉'de parçalanır" (mertebe; 0,5e5 ile 3e5 arası).
    assert 0.5e5 < accretion.hills_mass_kg(0.6 * MSUN, 0.01 * r_gunes) / MSUN < 3e5
    # Özdeşlik: r_t(M_H) = R_S(M_H).
    assert accretion.tidal_disruption_radius_m(mh, MSUN, r_gunes) == pytest.approx(
        basics.schwarzschild_radius_m(mh), rel=1e-12
    )


# ----------------------------------------------------------------------------- girdi denetimi


@pytest.mark.parametrize(
    "cagri",
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
def test_gecersiz_girdi(cagri):
    with pytest.raises(ValueError):
        cagri()


def test_skaler_ve_dizi():
    assert np.ndim(basics.kerr_isco_radius_rg(0.5)) == 0
    assert basics.kerr_isco_radius_rg(np.zeros((2, 3)) + 0.5).shape == (2, 3)
    assert basics.hawking_temperature_k(np.array([MSUN, 2 * MSUN])).shape == (2,)
    assert np.ndim(accretion.eddington_luminosity_w(MSUN)) == 0


def test_hawking_greybody_page_eq15():
    from scipy.integrate import quad

    # [Page2005] eq:15: yalnız foton için katsayı 3,3638e-5; kara cisim kestirimi 1/(15360π) = 2,0723e-5.
    m0 = 1e12
    kara_cisim_hizi = m0 / (3.0 * basics.hawking_evaporation_time_blackbody_s(m0))  # t = M/(3 |dM/dt|)
    yalniz_foton = basics.hawking_mass_loss_rate_kg_s(m0, spin_one=1.0, spin_two=0.0)
    assert float(yalniz_foton) / float(kara_cisim_hizi) == pytest.approx(3.3638e-5 * 15360 * np.pi, rel=1e-12)
    assert float(yalniz_foton) / float(kara_cisim_hizi) == pytest.approx(1.623, abs=2e-3)
    # M⁻² ölçeklemesi ve tür katsayılarının toplanırlığı.
    assert float(basics.hawking_mass_loss_rate_kg_s(2 * m0)) == pytest.approx(
        float(basics.hawking_mass_loss_rate_kg_s(m0)) / 4
    )
    toplam = basics.hawking_mass_loss_rate_kg_s(m0, spin_half=3.0)
    assert float(toplam) / float(basics.hawking_mass_loss_rate_kg_s(m0)) == pytest.approx(
        (3 * 8.1830 + 3.3638 + 0.3836) / (3.3638 + 0.3836), rel=1e-12
    )
    # Süre, hızın sayısal tümlevidir: t = ∫ dM / |dM/dt|.
    t_sayisal, _ = quad(lambda m: 1.0 / float(basics.hawking_mass_loss_rate_kg_s(m)), 0.0, m0)
    assert float(basics.hawking_evaporation_time_s(m0)) == pytest.approx(t_sayisal, rel=1e-8)
    # Güneş kütlesi, foton + graviton: yaklaşık 1,16e67 yıl; kara cisim kestiriminden kısa.
    assert float(basics.hawking_evaporation_time_s(MSUN)) / YIL == pytest.approx(1.16e67, rel=0.02)
    assert basics.hawking_evaporation_time_s(MSUN) < basics.hawking_evaporation_time_blackbody_s(MSUN)
