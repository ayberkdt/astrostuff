"""Yıldız ölçekleri: ışınım, çekim, zaman ölçekleri, Eddington sınırı ve Chandrasekhar kütlesi.

Kaynakların metinleri 25 Kasım 2026'da arXiv TeX dosyalarından okunmuştur:

- Stefan-Boltzmann ve yüzey çekimi biçimleri: IAU 2015 B3 makalesi [Prsa2016].
- Serbest düşme zamanı: [McKee2007]; Kelvin-Helmholtz zamanı: [Zinnecker2007].
- Eddington ışınım gücü ve kritik açısal hız: [Paxton2013].
- Eddington standart modeli, M_*, ve beyaz cüce K sabiti ile Chandrasekhar kütlesi: [Bludman2010].

Nükleer zaman ölçeği için okunmuş bir kaynak bulunamadı; o fonksiyon ``teyit="bekliyor"`` taşır.
Bütün işlevler SI birimlerindedir; zamanlar yıl olarak veriliyorsa Jülyen yılıdır.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from kozmos._sources import source
from kozmos.core.constants import (
    ATOMIC_MASS_UNIT,
    GRAVITATIONAL_CONSTANT,
    JULIAN_YEAR_SECONDS,
    PLANCK,
    SPEED_OF_LIGHT,
    STEFAN_BOLTZMANN,
)
from kozmos.stars.structure import lane_emden_mass_coefficient

Array = NDArray[np.float64]


@source(
    "Prsa2016",
    where="Tablo 'formu', M_Bol satırı (4πσR² terimi); metin: T_eff Stefan-Boltzmann yasasından",
    status="read",
)
def luminosity_w(radius_m: ArrayLike, effective_temperature_k: ArrayLike) -> Array:
    """Küresel kara cisim yıldızın ışınım gücünü hesaplar: L = 4π R² σ T_eff⁴.

    Parameters
    ----------
    yaricap_m : float ya da ndarray
        Yarıçap [m].
    etkin_sicaklik_k : float ya da ndarray
        Etkin sıcaklık [K].

    Returns
    -------
    float ya da ndarray
        Işınım gücü [W].

    References
    ----------
    [Prsa2016] Tablo 'formu' (M_Bol satırındaki 4πσR²) ve T_eff'in Stefan-Boltzmann yasasından türetilmesi.

    Verification
    ------------
    tests/unit/test_stars.py: nominal Güneş değerleri 3,828e26 W ile yüzde 0,1 içinde uyuşur
    ([Prsa2016]: nominal değerler güncel σ ile birbirine tutarlıdır).
    """
    r = np.asarray(radius_m, dtype=float)
    t = np.asarray(effective_temperature_k, dtype=float)
    return 4.0 * np.pi * r**2 * STEFAN_BOLTZMANN.value * t**4


@source("Prsa2016", where="metin: T_eff, Stefan-Boltzmann yasasından; Tablo 'formu', M_Bol satırı", status="read")
def effective_temperature_k(luminosity_w: ArrayLike, radius_m: ArrayLike) -> Array:
    """Işınım gücü ve yarıçaptan etkin sıcaklığı hesaplar: T_eff = (L / 4π R² σ)^{1/4}.

    Parameters
    ----------
    isinim_gucu_w : float ya da ndarray
        Işınım gücü [W].
    yaricap_m : float ya da ndarray
        Yarıçap [m].

    Returns
    -------
    float ya da ndarray
        Etkin sıcaklık [K].

    References
    ----------
    [Prsa2016] Stefan-Boltzmann yasasından T_eff (metin) ve Tablo 'formu'.

    Verification
    ------------
    tests/unit/test_stars.py: nominal Güneş için 5772 K'e yüzde 0,1 içinde; ``isinim_gucu_w`` ile gidiş-dönüş.
    """
    isinim = np.asarray(luminosity_w, dtype=float)
    r = np.asarray(radius_m, dtype=float)
    return (isinim / (4.0 * np.pi * r**2 * STEFAN_BOLTZMANN.value)) ** 0.25


@source(
    "Prsa2016",
    where="Tablo 'formu', 'log of surface gravity' satırı: log g = log(10^6 GM) - 2 log(100 R) [cgs]",
    status="read",
)
def surface_gravity_m_s2(mass_kg: ArrayLike, radius_m: ArrayLike) -> Array:
    """Yüzey çekim ivmesini hesaplar: g = G M / R².

    Parameters
    ----------
    kutle_kg, yaricap_m : float ya da ndarray
        Kütle [kg] ve yarıçap [m].

    Returns
    -------
    float ya da ndarray
        Yüzey çekimi [m s^-2].

    References
    ----------
    [Prsa2016] Tablo 'formu' (log g satırı, g = GM/R²'nin cgs logaritması).

    Verification
    ------------
    tests/unit/test_stars.py: nominal Güneş için log g = 4,438068 ([Prsa2016]).
    """
    m = np.asarray(mass_kg, dtype=float)
    r = np.asarray(radius_m, dtype=float)
    return GRAVITATIONAL_CONSTANT.value * m / r**2


@source(
    "Prsa2016",
    where="Tablo 'formu', 'log of surface gravity' satırı: 4,438068... + log M - 2 log R",
    status="read",
)
def log_surface_gravity_cgs(mass_kg: ArrayLike, radius_m: ArrayLike) -> Array:
    """Yüzey çekiminin cgs logaritmasını hesaplar: log₁₀ g [cm s^-2].

    Parameters
    ----------
    kutle_kg, yaricap_m : float ya da ndarray
        Kütle [kg] ve yarıçap [m].

    Returns
    -------
    float ya da ndarray
        log g (cgs); Güneş için 4,438.

    References
    ----------
    [Prsa2016] Tablo 'formu', log g satırı.

    Verification
    ------------
    tests/unit/test_stars.py: nominal Güneş için 4,438068 (altı ondalık, [Prsa2016]).
    """
    return np.log10(100.0 * surface_gravity_m_s2(mass_kg, radius_m))


@source("Bludman2010", where="Sec. I, Tablo I öncesi metin: ρ̄_n = 3M/4πR³", status="read")
def mean_density_kg_m3(mass_kg: ArrayLike, radius_m: ArrayLike) -> Array:
    """Küresel cismin ortalama yoğunluğunu hesaplar: ρ̄ = 3M / 4π R³.

    Parameters
    ----------
    kutle_kg, yaricap_m : float ya da ndarray
        Kütle [kg] ve yarıçap [m].

    Returns
    -------
    float ya da ndarray
        Ortalama yoğunluk [kg m^-3].

    References
    ----------
    [Bludman2010] ρ̄_n tanımı.

    Verification
    ------------
    tests/unit/test_stars.py: nominal Güneş için yaklaşık 1,41 g cm^-3.
    """
    m = np.asarray(mass_kg, dtype=float)
    r = np.asarray(radius_m, dtype=float)
    return 3.0 * m / (4.0 * np.pi * r**3)


@source("McKee2007", where="TeX etiketi eq:tff", status="read")
def free_fall_time_s(density_kg_m3: ArrayLike) -> Array:
    """Basınçsız homojen kürenin serbest düşme (dinamik) zamanını hesaplar.

    t_ff = ( 3π / (32 G ρ̄) )^{1/2}

    Parameters
    ----------
    yogunluk_kg_m3 : float ya da ndarray
        Ortalama yoğunluk [kg m^-3], pozitif.

    Returns
    -------
    float ya da ndarray
        Serbest düşme zamanı [s].

    References
    ----------
    [McKee2007] eq:tff.

    Validity
    --------
    Yıldız için dinamik zaman ölçeği olarak ρ̄ ile kullanılır; basınç ve yoğunluk
    dağılımı yoksayılır.

    Verification
    ------------
    tests/unit/test_stars.py: n_H = 10³ cm^-3 için 1,37e6 yıl ([McKee2007] metnindeki sayı).
    """
    rho = np.asarray(density_kg_m3, dtype=float)
    if np.any(rho <= 0):
        raise ValueError("yoğunluk pozitif olmalı")
    return np.sqrt(3.0 * np.pi / (32.0 * GRAVITATIONAL_CONSTANT.value * rho))


@source(
    "Zinnecker2007",
    where="Sec. 'The Accretion Phase / Formation of the Hydrostatic Core' metni ve GLOSSARY: t_KH = G M²/(R L)",
    status="read",
    note="makale τ_KH ~ G M²/(R L) biçiminde mertebe olarak yazar; öncül 1 alınmıştır",
)
def kelvin_helmholtz_timescale_s(mass_kg: ArrayLike, radius_m: ArrayLike, luminosity_w: ArrayLike) -> Array:
    """Kelvin-Helmholtz (ısıl) zaman ölçeğini hesaplar: t_KH = G M² / (R L).

    Parameters
    ----------
    kutle_kg, yaricap_m : float ya da ndarray
        Kütle [kg] ve yarıçap [m].
    isinim_gucu_w : float ya da ndarray
        Işınım gücü [W], pozitif.

    Returns
    -------
    float ya da ndarray
        Zaman ölçeği [s].

    References
    ----------
    [Zinnecker2007] t_KH tanımı.

    Validity
    --------
    Mertebe tahminidir; öncül (G M²/R) yerçekimi enerjisinin tek başına ölçeğidir.

    Verification
    ------------
    tests/unit/test_stars.py: Güneş için 10⁷ ile 10⁸ yıl arasında ([Zinnecker2007]: güneş tipi yıldızlar için ~10⁷ yıl).
    """
    m = np.asarray(mass_kg, dtype=float)
    r = np.asarray(radius_m, dtype=float)
    isinim = np.asarray(luminosity_w, dtype=float)
    if np.any(isinim <= 0):
        raise ValueError("ışınım gücü pozitif olmalı")
    return GRAVITATIONAL_CONSTANT.value * m**2 / (r * isinim)


@source(
    "Pols2011",
    where="Bölüm 2.4.2, Denk. 2.37",
    status="read",
    note="τ_nuc = φ f_nuc M c² / L ≈ 10^10 (M/M_⊙)(L_⊙/L) yıl. Kaynak hidrojen yanması için φ ≈ 0,007 ve Güneş benzeri "
    "yıldızda f_nuc ≈ 0,1 (kütlenin %70'i hidrojen, yanma içteki ~%10'da) verir",
)
def nuclear_timescale_yr(
    mass_kg: ArrayLike, luminosity_w: ArrayLike, yield_fraction: float, fuel_fraction: float
) -> Array:
    """Nükleer zaman ölçeğini ve ana kol ömrü tahminini hesaplar.

    t_nuc = ε f M c² / L

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Yıldız kütlesi [kg].
    isinim_gucu_w : float ya da ndarray
        Ortalama ışınım gücü [W], pozitif.
    verim : float
        ε, yakıtın durgun kütlesinin ışınıma dönüşen payı (0 ile 1 arası); değer kullanıcıdan alınır.
    yakit_orani : float
        f, yanan çekirdeğin toplam kütleye oranı (0 ile 1 arası).

    Returns
    -------
    float ya da ndarray
        Zaman ölçeği [yıl].

    References
    ----------
    [Pols2011] Bölüm 2.4.2, Denk. 2.37.

    Validity
    --------
    ε ve f kullanıcıdan alınır. Kaynağın değerleri: hidrojen yanmasında ε ≈ 0,007, Güneş benzeri
    yıldızda f ≈ 0,1; bunlarla Güneş için yaklaşık 10^10 yıl çıkar. Helyum ve daha ağır
    yakıtlarda ε en az on kat küçüktür.

    Verification
    ------------
    tests/unit/test_stars.py: kütleyle doğrusal, ışınım gücüyle ters orantı; tanım gereği kapalı değer;
    kaynaktaki Güneş değeri (yaklaşık 10^10 yıl).
    """
    if not (0.0 < yield_fraction < 1.0 and 0.0 < fuel_fraction <= 1.0):
        raise ValueError("verim (0,1) ve yakıt_orani (0,1] aralığında olmalı")
    m = np.asarray(mass_kg, dtype=float)
    isinim = np.asarray(luminosity_w, dtype=float)
    if np.any(isinim <= 0):
        raise ValueError("ışınım gücü pozitif olmalı")
    return yield_fraction * fuel_fraction * m * SPEED_OF_LIGHT.value**2 / isinim / JULIAN_YEAR_SECONDS.value


@source(
    "Paxton2013",
    where="Sec. 'Rotationally-Enhanced Mass Loss' (TeX etiketi s.rotation-mass-loss): L_Edd = 4π c G M/κ",
    status="read",
)
@source("Bludman2010", where="Sec. II.B, Eddington standart modeli: L_Edd := 4π c G M/κ_p", status="read")
def eddington_luminosity_w(mass_kg: ArrayLike, opacity_m2_kg: ArrayLike) -> Array:
    """Eddington ışınım gücü sınırını hesaplar: L_Edd = 4π c G M / κ.

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Kütle [kg].
    opaklik_m2_kg : float ya da ndarray
        κ, kütle başına opaklık [m² kg^-1], pozitif.

    Returns
    -------
    float ya da ndarray
        L_Edd [W].

    References
    ----------
    [Paxton2013] s.rotation-mass-loss; [Bludman2010] Eddington standart modeli.

    Verification
    ------------
    tests/unit/test_stars.py: kütleyle doğrusal, κ ile ters orantı; kapalı değer.
    """
    m = np.asarray(mass_kg, dtype=float)
    kappa = np.asarray(opacity_m2_kg, dtype=float)
    if np.any(kappa <= 0):
        raise ValueError("opaklık pozitif olmalı")
    return 4.0 * np.pi * SPEED_OF_LIGHT.value * GRAVITATIONAL_CONSTANT.value * m / kappa


@source(
    "Paxton2013", where="mdotomega etiketli denklemi izleyen cümle: Ω_crit² = (1 - L/L_Edd) G M / R³", status="read"
)
def critical_angular_velocity_rad_s(
    mass_kg: ArrayLike, radius_m: ArrayLike, luminosity_w: ArrayLike, opacity_m2_kg: ArrayLike
) -> Array:
    """Işınım basıncıyla azaltılmış kritik (kopma) açısal hızı hesaplar.

    Ω_crit² = (1 - L / L_Edd) G M / R³

    Parameters
    ----------
    kutle_kg, yaricap_m : float ya da ndarray
        Kütle [kg] ve yarıçap [m].
    isinim_gucu_w : float ya da ndarray
        Işınım gücü [W]; L_Edd'i aşamaz.
    opaklik_m2_kg : float ya da ndarray
        Yüzey opaklığı κ [m² kg^-1].

    Returns
    -------
    float ya da ndarray
        Ω_crit [rad s^-1]. L = 0 için Kepler açısal hızı, L → L_Edd için sıfır.

    References
    ----------
    [Paxton2013] Ω_crit tanımı.

    Validity
    --------
    L > L_Edd için tanım yoktur; ``ValueError`` verilir.

    Verification
    ------------
    tests/unit/test_stars.py: L = 0 ve L = L_Edd sınır değerleri.
    """
    m = np.asarray(mass_kg, dtype=float)
    r = np.asarray(radius_m, dtype=float)
    ratio = np.asarray(luminosity_w, dtype=float) / eddington_luminosity_w(m, opacity_m2_kg)
    if np.any(ratio > 1.0):
        raise ValueError("L, Eddington sınırını aşıyor; kritik açısal hız tanımsız")
    return np.sqrt((1.0 - ratio) * GRAVITATIONAL_CONSTANT.value * m / r**3)


@source(
    "Bludman2010",
    where="Sec. II.B: M_* := 3√10 0ω_3/π³ (hc/(G m_H^{4/3}))^{3/2} = 18,3 M_⊙",
    status="read",
    note="m_H, metinde atomik kütle birimi olarak tanımlanır; 0ω_3 sayısal çözümden gelir",
)
def eddington_scale_mass_kg() -> float:
    """Eddington dördüncü derece denkleminin ölçek kütlesi M_*'ı hesaplar.

    M_* = 3√10 ω_3 / π³ · ( h c / (G m_H^{4/3}) )^{3/2}

    Returns
    -------
    float
        M_* [kg]; yaklaşık 18,3 Güneş kütlesi.

    References
    ----------
    [Bludman2010] Sec. II.B, M_* tanımı.

    Verification
    ------------
    tests/unit/test_stars.py: 18,3 M_⊙ ([Bludman2010]) yüzde 0,5 içinde.
    """
    omega3 = lane_emden_mass_coefficient(3.0)
    mh = ATOMIC_MASS_UNIT.value
    return float(
        3.0
        * np.sqrt(10.0)
        * omega3
        / np.pi**3
        * (PLANCK.value * SPEED_OF_LIGHT.value / (GRAVITATIONAL_CONSTANT.value * mh ** (4.0 / 3.0))) ** 1.5
    )


@source("Bludman2010", where="Sec. II.B, Eddington dördüncü derece denklemi: (1-β)/β⁴ = (M μ²/M_*)²", status="read")
def eddington_beta(mass_kg: ArrayLike, mean_molecular_weight: float) -> Array:
    """Eddington standart modelinde gaz basıncının toplam basınca oranı β'yı çözer.

    (1 - β) / β⁴ = ( M μ² / M_* )²

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Yıldız kütlesi [kg], pozitif.
    ortalama_molekul_agirligi : float
        μ (m_H biriminde), pozitif.

    Returns
    -------
    float ya da ndarray
        β ∈ (0, 1]; küçük kütlede 1'e, büyük kütlede 0'a gider.

    References
    ----------
    [Bludman2010] Eddington dördüncü derece denklemi.

    Validity
    --------
    Sabit radyasyon entropili (Eddington standart modeli, n = 3) yıldız varsayılır.

    Verification
    ------------
    tests/unit/test_stars.py: denklemin kalıntısı; β'nın kütleyle monoton azalması.
    """
    if mean_molecular_weight <= 0:
        raise ValueError("μ pozitif olmalı")
    y = np.asarray(mass_kg, dtype=float) * mean_molecular_weight**2 / eddington_scale_mass_kg()
    if np.any(y < 0):
        raise ValueError("kütle eksi olamaz")

    def kok(y_tek: float) -> float:
        return float(brentq(lambda b: 1.0 - b - y_tek**2 * b**4, 0.0, 1.0, xtol=1e-15, rtol=1e-15))

    return np.asarray(np.vectorize(kok, otypes=[float])(y))[()]


@source("Bludman2010", where="Sec. II.B: L = L_Edd [1 - β(M)]", status="read")
def standard_model_luminosity_w(mass_kg: ArrayLike, mean_molecular_weight: float, opacity_m2_kg: float) -> Array:
    """Eddington standart modelinin ışınım gücünü hesaplar: L = L_Edd (1 - β).

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Yıldız kütlesi [kg], pozitif.
    ortalama_molekul_agirligi : float
        μ (m_H biriminde).
    opaklik_m2_kg : float
        Fotosfer opaklığı κ [m² kg^-1].

    Returns
    -------
    float ya da ndarray
        Işınım gücü [W]; küçük kütlede M³ ile artar, büyük kütlede L_Edd'e yaklaşır.

    References
    ----------
    [Bludman2010] L = L_Edd [1 - β(M)].

    Verification
    ------------
    tests/unit/test_stars.py: L/L_Edd = 1 - β; küçük kütlede M³ yasası ve 0,003 ≈ (1/18,3)² katsayısı.
    """
    beta = eddington_beta(mass_kg, mean_molecular_weight)
    return eddington_luminosity_w(mass_kg, opacity_m2_kg) * (1.0 - beta)


@source(
    "Bludman2010",
    where="Sec. II.A (göreli dejenere yıldızlar): K_WD = (hc/8)(3/π)^{1/3}(m_H μ_e)^{-4/3}",
    status="read",
    note="m_H, metinde atomik kütle birimi olarak tanımlanır",
)
def white_dwarf_k_coefficient_si(molecular_weight_per_electron: float) -> float:
    """Göreli dejenere elektron gazının politrop sabitini hesaplar: P = K ρ^{4/3}.

    K_WD = (h c / 8) (3/π)^{1/3} (m_H μ_e)^{-4/3}

    Parameters
    ----------
    elektron_basina_molekul_agirligi : float
        μ_e, elektron başına ortalama kütle (m_H biriminde); He ve C/O için 2.

    Returns
    -------
    float
        K [Pa m⁴ kg^{-4/3}].

    References
    ----------
    [Bludman2010] K_WD ifadesi.

    Verification
    ------------
    tests/unit/test_stars.py: μ_e^{-4/3} ölçeklenmesi; Chandrasekhar kütlesiyle birlikte 5,83/μ_e² M_⊙.
    """
    if molecular_weight_per_electron <= 0:
        raise ValueError("μ_e pozitif olmalı")
    return float(
        PLANCK.value
        * SPEED_OF_LIGHT.value
        / 8.0
        * (3.0 / np.pi) ** (1.0 / 3.0)
        * (ATOMIC_MASS_UNIT.value * molecular_weight_per_electron) ** (-4.0 / 3.0)
    )


@source(
    "Bludman2010",
    where="Sec. II (giriş) M = 4π 0ω_3 (K/πG)^{3/2}; Sec. II.A M_Ch = 5,83 M_⊙/μ_e² = 1,46 M_⊙",
    status="read",
    note="0ω_3 = -ξ_1² θ'(ξ_1) (n = 3) sayısal çözümden gelir; metindeki 5,83 ile karşılaştırılır",
)
def chandrasekhar_mass_kg(molecular_weight_per_electron: float) -> float:
    """Chandrasekhar kütlesini hesaplar: n = 3 politropun kütlesi yalnızca K'ye bağlıdır.

    M_Ch = 4π ω_3 ( K_WD / (π G) )^{3/2}

    Parameters
    ----------
    elektron_basina_molekul_agirligi : float
        μ_e (m_H biriminde); He ve C/O için 2.

    Returns
    -------
    float
        M_Ch [kg]; μ_e = 2 için yaklaşık 1,46 Güneş kütlesi.

    References
    ----------
    [Bludman2010] Sec. II ve II.A.

    Validity
    --------
    Göreli olmayan düzeltmeler, Coulomb etkileri ve genel görelilik yoksayılır.

    Verification
    ------------
    tests/unit/test_stars.py: 5,83/μ_e² M_⊙ ve μ_e = 2 için 1,46 M_⊙ ([Bludman2010]).
    """
    k = white_dwarf_k_coefficient_si(molecular_weight_per_electron)
    omega3 = lane_emden_mass_coefficient(3.0)
    return float(4.0 * np.pi * omega3 * (k / (np.pi * GRAVITATIONAL_CONSTANT.value)) ** 1.5)
