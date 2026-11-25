"""Kara delik birikimi: Eddington sınırı, Bondi akışı, ince disk sıcaklığı ve gelgit parçalanması.

Denklemler şu kaynak metinlerinden 25 Kasım 2026'da arXiv TeX dosyalarından okunmuştur:
Inayoshi, Visbal ve Haiman [Inayoshi2020] (arXiv:1911.05791), Armitage [Armitage2011]
(arXiv:1011.1496), Komossa [Komossa2015] (arXiv:1505.01093) ve Abramowicz ve Fragile
[Abramowicz2013] (arXiv:1104.5499).

[Inayoshi2020] elektron saçılma opaklığı κ_es'in değerini vermez. Burada tam iyonize
saf hidrojen için κ_es = σ_T / m_p alınmıştır (0,40 cm² g⁻¹); bu seçim kaynağa değil
bu modüle aittir ve kaynağın sayısal değerleriyle karşılaştırılmıştır: 0,4505 Gyr (kaynak 0,45) ve 2,22e-5 M☉/yıl (kaynak 2,3e-5; %3,5 düşük).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.constants import (
    GM_SUN,
    GRAVITATIONAL_CONSTANT,
    PROTON_MASS,
    SPEED_OF_LIGHT,
    STEFAN_BOLTZMANN,
    THOMSON_CROSS_SECTION,
)

Array = NDArray[np.float64]

_G = GRAVITATIONAL_CONSTANT.value
_C = SPEED_OF_LIGHT.value
_M_P = PROTON_MASS.value
_SIGMA_T = THOMSON_CROSS_SECTION.value
_SIGMA_SB = STEFAN_BOLTZMANN.value
# Güneş kütlesi [kg]: nominal GM_güneş / G (IAU 2015 B3 ve CODATA 2022 sabitlerinden).
_SOLAR_MASS_KG = GM_SUN.value / _G
_KAPPA_ES = _SIGMA_T / _M_P  # saf hidrojen, tam iyonize [m² kg⁻¹]


def _positive(value: ArrayLike, name: str) -> Array:
    """Değeri diziye çevirir ve pozitif olduğunu denetler."""
    x = np.asarray(value, dtype=float)
    if np.any(~(x > 0.0)):
        raise ValueError(f"{name} pozitif olmalı")
    return x


def _efficiency(efficiency: ArrayLike) -> Array:
    """Işınım verimini diziye çevirir ve 0 < η < 1 olduğunu denetler."""
    e = np.asarray(efficiency, dtype=float)
    if np.any(~((e > 0.0) & (e < 1.0))):
        raise ValueError("ışınım verimi 0 ile 1 arasında olmalı")
    return e


@source(
    "Inayoshi2020",
    where="§'Growing BHs by Accretion: Is There an Eddington Limit?', L_Edd = 4π c G M / κ_es",
    status="read",
    note="κ_es kaynakta tanımlı değil; σ_T/m_p alındı (modül belgesine bakın). Abramowicz2013 §1 dipnotu "
    "L_Edd ≡ 1,2e38 (M/M☉) erg/s verir; bu çıktıdan yaklaşık %5 küçüktür (1,26e38).",
)
def eddington_luminosity_w(mass_kg: ArrayLike) -> Array:
    """Eddington ışınım gücünü hesaplar: L_Edd = 4π G M m_p c / σ_T.

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].

    Returns
    -------
    float ya da ndarray
        L_Edd [W]; 1 M☉ için ≈ 1,26e31 W (1,26e38 erg/s).

    References
    ----------
    [Inayoshi2020] §'Growing BHs by Accretion' (L_Edd = 4π c G M/κ_es).

    Validity
    --------
    Küresel, tam iyonize hidrojen, yalnızca Thomson saçılması. Disk geometrisinde ışınım
    dikey kaçabildiği için sınır aşılabilir (Abramowicz2013, ince disk bölümü).

    Verification
    ------------
    tests/unit/test_black_holes.py: 1,2e38 erg/s (M/M☉) [Abramowicz2013 §1] ile %6 içinde (kaynak iki basamağa yuvarlar).
    """
    return 4.0 * np.pi * _C * _G * _positive(mass_kg, "kütle") / _KAPPA_ES


@source(
    "Abramowicz2013",
    where="§1 (Giriş), verim tanımı η ≡ L/(Ṁ c²)",
    status="read",
    note="L = η Ṁ c² doğrudan tanımdır.",
)
def accretion_luminosity_w(accretion_rate_kg_s: ArrayLike, efficiency: ArrayLike) -> Array:
    """Birikim ışınım gücünü hesaplar: L = η Ṁ c².

    Parameters
    ----------
    kutle_akisi_kg_s : float ya da ndarray
        Birikim hızı Ṁ [kg s^-1], pozitif.
    verimlilik : float ya da ndarray
        Işınım verimi η, 0 < η < 1.

    Returns
    -------
    float ya da ndarray
        L [W].

    References
    ----------
    [Abramowicz2013] §1, η ≡ L/(Ṁ c²).

    Verification
    ------------
    tests/unit/test_black_holes.py: L_Edd üzerinden Eddington akışıyla gidiş-dönüş.
    """
    return _efficiency(efficiency) * _positive(accretion_rate_kg_s, "kütle akışı") * _C**2


@source(
    "Inayoshi2020",
    where="§'Growing BHs by Accretion' ve Tablo 3, Ṁ_Edd ≡ L_Edd / (0,1 c²) (dipnotta L_Edd/c² tanımına da değinilir)",
    status="read",
    note="Kaynak η = 0,1 sabitler; burada η parametredir ve η = 0,1 kaynağın tanımıdır.",
)
def eddington_accretion_rate_kg_s(mass_kg: ArrayLike, efficiency: ArrayLike = 0.1) -> Array:
    """Eddington birikim hızını hesaplar: Ṁ_Edd = L_Edd / (η c²).

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].
    verimlilik : float ya da ndarray, isteğe bağlı
        Işınım verimi η, 0 < η < 1; varsayılan 0,1 (kaynağın tanımı).

    Returns
    -------
    float ya da ndarray
        Ṁ_Edd [kg s^-1]. 10³ M☉ için 2,22e-5 M☉/yıl (η = 0,1); [Inayoshi2020] Tablo 3 değeri 2,3e-5.

    References
    ----------
    [Inayoshi2020] Tablo 3 ve §'Growing BHs by Accretion'.

    Verification
    ------------
    tests/unit/test_black_holes.py: Tablo 3'teki 2,3e-5 M☉/yıl değeriyle %5 içinde uyum; 1/η ölçeklemesi.
    """
    return eddington_luminosity_w(mass_kg) / (_efficiency(efficiency) * _C**2)


@source(
    "Inayoshi2020",
    where="§'Growing BHs by Accretion', t_grow ≈ 0,45 ε/(1−ε) f_duty^{-1} ln(M/M_tohum) Gyr; §1 t_Edd ≈ 50 Myr (ε = 0,1)",
    status="read",
    note="Kaynak yalnızca 0,45 Gyr katsayısını verir; burada katsayı σ_T c/(4π G m_p) olarak hesaplanır ve "
    "testte 0,45 Gyr ile karşılaştırılır. e-katlanma süresi, t_grow ifadesinin ln çarpanından önceki kısmıdır.",
)
def salpeter_time_s(efficiency: ArrayLike = 0.1) -> Array:
    """Eddington sınırlı büyümenin e-katlanma (Salpeter) süresini hesaplar.

    t_Sal = [η / (1 − η)] σ_T c / (4π G m_p)

    Kütle M(t) = M0 exp(t / t_Sal). η = 0,1 için ≈ 50 Myr.

    Parameters
    ----------
    verimlilik : float ya da ndarray, isteğe bağlı
        Işınım verimi η, 0 < η < 1; varsayılan 0,1.

    Returns
    -------
    float ya da ndarray
        t_Sal [s].

    References
    ----------
    [Inayoshi2020] t_grow ifadesi ve §1'deki t_Edd ≈ 50 Myr.

    Validity
    --------
    Sürekli Eddington sınırlı birikim, görev döngüsü 1; kütleye düşen enerjinin (1 − η)
    oranı kara deliğe eklenir.

    Verification
    ------------
    tests/unit/test_black_holes.py: 0,45 Gyr katsayısı ve 50 Myr; M(t) üstel büyümesi.
    """
    e = _efficiency(efficiency)
    return e / (1.0 - e) * _SIGMA_T * _C / (4.0 * np.pi * _G * _M_P)


@source(
    "Inayoshi2020",
    where="§'Inflow from large scales' (R_B ≡ G M / c_s²) ve Tablo 3 (TeX etiketi tab:phys)",
    status="read",
)
def bondi_radius_m(mass_kg: ArrayLike, sound_speed_m_s: ArrayLike) -> Array:
    """Bondi yarıçapını hesaplar: R_B = G M / c_s².

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].
    ses_hizi_m_s : float ya da ndarray
        Gazın ses hızı [m s^-1], pozitif.

    Returns
    -------
    float ya da ndarray
        R_B [m].

    References
    ----------
    [Inayoshi2020] R_B tanımı ve Tablo 3.

    Verification
    ------------
    tests/unit/test_black_holes.py: Tablo 3 değeri 0,6 pc (10³ M☉, T = 10³ K); M ve c_s ölçeklemesi.
    """
    return _G * _positive(mass_kg, "kütle") / _positive(sound_speed_m_s, "ses hızı") ** 2


@source("Inayoshi2020", where="Tablo 3, Ṁ_B ≡ π e^{3/2} ρ G² M² / c_s³", status="read")
def bondi_accretion_rate_kg_s(mass_kg: ArrayLike, density_kg_m3: ArrayLike, sound_speed_m_s: ArrayLike) -> Array:
    """Bondi birikim hızını hesaplar: Ṁ_B = π e^{3/2} ρ G² M² / c_s³.

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].
    yogunluk_kg_m3 : float ya da ndarray
        Uzaktaki gazın kütle yoğunluğu ρ [kg m^-3].
    ses_hizi_m_s : float ya da ndarray
        Gazın ses hızı [m s^-1].

    Returns
    -------
    float ya da ndarray
        Ṁ_B [kg s^-1].

    References
    ----------
    [Inayoshi2020] Tablo 3.

    Validity
    --------
    Kaynaktaki tanım eşsıcaklıklı (izotermal) küresel akış içindir; açısal momentum,
    ışınım geri tepmesi ve göreli etkiler yok sayılır.

    Verification
    ------------
    tests/unit/test_black_holes.py: Tablo 3 değeri 4,5e-3 M☉/yıl (n_H = 1e4 cm^-3, T = 10³ K, 10³ M☉);
    Ṁ_B = π e^{3/2} ρ R_B² c_s özdeşliği.
    """
    m = _positive(mass_kg, "kütle")
    rho = _positive(density_kg_m3, "yoğunluk")
    cs = _positive(sound_speed_m_s, "ses hızı")
    return np.pi * np.exp(1.5) * rho * (_G * m) ** 2 / cs**3


@source(
    "Armitage2011",
    where="§2 'Accretion Disk Evolution', T_eff⁴ = 3 G M Ṁ / (8π σ r³) [1 − √(r_in/r)] (TeX etiketi eq:2_nusigma ile aynı eqnarray, ikinci satır)",
    status="read",
)
def thin_disk_effective_temperature_k(
    radius_m: ArrayLike, mass_kg: ArrayLike, accretion_rate_kg_s: ArrayLike, inner_radius_m: ArrayLike
) -> Array:
    """Kararlı, ince, Newton diskinin etkin sıcaklık profilini hesaplar.

    T_eff(r) = { 3 G M Ṁ / (8π σ_SB r³) · [1 − √(r_in / r)] }^{1/4}

    Parameters
    ----------
    yaricap_m : float ya da ndarray
        Disk yarıçapı r [m]; r ≥ r_in olmalı.
    kutle_kg : float ya da ndarray
        Merkezi kütle [kg].
    kutle_akisi_kg_s : float ya da ndarray
        Birikim hızı Ṁ [kg s^-1].
    ic_yaricap_m : float ya da ndarray
        İç yarıçap r_in [m] (burada tork sıfırdır); kara delik için tipik olarak ISCO.

    Returns
    -------
    float ya da ndarray
        T_eff [K]; r = r_in'de 0.

    References
    ----------
    [Armitage2011] §2 'Accretion Disk Evolution', kararlı viskoz disk denklemleri.

    Validity
    --------
    Newton çekimi (göreli düzeltme çarpanları yok), r_in'de sıfır tork, her halkada yerel
    ışınım dengesi, ince ve optik olarak kalın disk. Novikov-Thorne göreli düzeltmeleri
    için bkz. Abramowicz2013; burada yer almaz. Kaynak protogezegen diskleri için
    yazılmıştır; profil viskozite yasasından (α) bağımsızdır, yalnızca enerji ve açısal momentum
    korunumuna dayanır; kara delik çevresinde yalnızca Newton limitinde (r ≫ r_G) kullanılabilir.

    Verification
    ------------
    tests/unit/test_black_holes.py: r ≫ r_in'de r^{-3/4}; Ṁ^{1/4} ölçeklemesi; azami değer.
    """
    r = _positive(radius_m, "yarıçap")
    r_ic = _positive(inner_radius_m, "iç yarıçap")
    if np.any(r < r_ic):
        raise ValueError("yarıçap iç yarıçaptan küçük olamaz")
    m = _positive(mass_kg, "kütle")
    mdot = _positive(accretion_rate_kg_s, "kütle akışı")
    t4 = 3.0 * _G * m * mdot / (8.0 * np.pi * _SIGMA_SB * r**3) * (1.0 - np.sqrt(r_ic / r))
    return t4**0.25


@source(
    "Armitage2011",
    where="§2 'Accretion Disk Evolution', T_eff profili; azami konum tarafımdan türetildi",
    status="read",
    note="Profil denklemi okundu; dT/dr = 0 çözümü u = √(r_in/r) = 6/7 benim türetmem, testte sayısal "
    "en büyükleme ile doğrulanır.",
)
def thin_disk_peak_temperature_radius_m(inner_radius_m: ArrayLike) -> Array:
    """İnce diskte etkin sıcaklığın en yüksek olduğu yarıçapı hesaplar: r = (49/36) r_in.

    Parameters
    ----------
    ic_yaricap_m : float ya da ndarray
        İç yarıçap r_in [m].

    Returns
    -------
    float ya da ndarray
        T_eff'in tepe yaptığı r [m].

    References
    ----------
    [Armitage2011] §2, T_eff profili (maksimizasyon benim türetmem).

    Validity
    --------
    Newton diski, r_in'de sıfır tork; [Armitage2011] profilinin geçerlilik koşulları.

    Verification
    ------------
    tests/unit/test_black_holes.py: scipy ile sayısal en büyükleme.
    """
    return 49.0 / 36.0 * _positive(inner_radius_m, "iç yarıçap")


@source(
    "Armitage2011",
    where="§2 'Accretion Disk Evolution', T_eff profili; azami değer tarafımdan türetildi",
    status="read",
    note="T_max⁴ = [3 G M Ṁ / (8π σ r_in³)] (6/7)⁶ (1/7): u = √(r_in/r) için u⁶(1−u) en büyüğü u = 6/7'de; "
    "benim türetmem, testte profil fonksiyonunun sayısal en büyükleriyle doğrulanır.",
)
def thin_disk_peak_temperature_k(
    mass_kg: ArrayLike, accretion_rate_kg_s: ArrayLike, inner_radius_m: ArrayLike
) -> Array:
    """İnce diskin etkin sıcaklığının azami değerini hesaplar.

    T_max = { 3 G M Ṁ / (8π σ_SB r_in³) · (6/7)⁶ / 7 }^{1/4} ≈ 0,488 (3 G M Ṁ / 8π σ r_in³)^{1/4}

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Merkezi kütle [kg].
    kutle_akisi_kg_s : float ya da ndarray
        Birikim hızı Ṁ [kg s^-1].
    ic_yaricap_m : float ya da ndarray
        İç yarıçap r_in [m].

    Returns
    -------
    float ya da ndarray
        T_max [K].

    References
    ----------
    [Armitage2011] §2, T_eff profili (maksimizasyon benim türetmem).

    Validity
    --------
    [Armitage2011] profilinin koşulları: Newton, r_in'de sıfır tork.

    Verification
    ------------
    tests/unit/test_black_holes.py: profil fonksiyonunun r = 49/36 r_in'deki değeri ve sayısal en büyükleme;
    tests/property/test_black_holes_property.py: M^{1/4} Ṁ^{1/4} r_in^{-3/4} ölçeklemesi.
    """
    m = _positive(mass_kg, "kütle")
    mdot = _positive(accretion_rate_kg_s, "kütle akışı")
    r_ic = _positive(inner_radius_m, "iç yarıçap")
    t4 = 3.0 * _G * m * mdot / (8.0 * np.pi * _SIGMA_SB * r_ic**3) * (6.0 / 7.0) ** 6 / 7.0
    return t4**0.25


@source(
    "Komossa2015",
    where="§1 (Giriş), gelgit yarıçapı r_t denklemi (etiketsiz): r_t ≃ 7e12 (M/10⁶ M☉)^{1/3} (m*/M☉)^{-1/3} (r*/R☉) cm",
    status="read",
    note="Kaynak yalnızca sayısal ölçeklemeyi verir. Burada r_t = r* (M/m*)^{1/3} yazıldı; katsayı "
    "R☉ (10⁶)^{1/3} = 6,96e12 cm kaynaktaki 7e12 cm ile %1 içinde uyuşur ve testte denetlenir.",
)
def tidal_disruption_radius_m(
    black_hole_mass_kg: ArrayLike, stellar_mass_kg: ArrayLike, stellar_radius_m: ArrayLike
) -> Array:
    """Yıldızın gelgit kuvvetleriyle parçalandığı yarıçapı hesaplar: r_t = r* (M / m*)^{1/3}.

    Parameters
    ----------
    kara_delik_kutlesi_kg : float ya da ndarray
        Kara delik kütlesi M [kg].
    yildiz_kutlesi_kg : float ya da ndarray
        Yıldız kütlesi m* [kg].
    yildiz_yaricapi_m : float ya da ndarray
        Yıldız yarıçapı r* [m].

    Returns
    -------
    float ya da ndarray
        r_t [m].

    References
    ----------
    [Komossa2015] §1, gelgit yarıçapı denklemi.

    Validity
    --------
    Mertebe kestirimi (O(1) çarpan yıldızın yapısına bağlıdır); Newton gelgit kuvveti,
    dönmeyen kara delik. Dönme parçalanma eşiğini değiştirebilir (kaynak §1).

    Verification
    ------------
    tests/unit/test_black_holes.py: 7e12 cm katsayısı (kaynak); M^{1/3} ölçeklemesi.
    """
    m_kd = _positive(black_hole_mass_kg, "kara delik kütlesi")
    m_y = _positive(stellar_mass_kg, "yıldız kütlesi")
    r_y = _positive(stellar_radius_m, "yıldız yarıçapı")
    return r_y * (m_kd / m_y) ** (1.0 / 3.0)


@source(
    "Komossa2015",
    where="§1, r_t denklemi ve devamındaki cümle: Güneş türü yıldızlar ≳ 10⁸ M☉'de bütün yutulur, beyaz cüceler ≲ 10⁵ M☉'de parçalanır",
    status="read",
    note="r_t = R_S koşulundan M çözümü tarafımdan türetildi: M = (r* c² / 2G)^{3/2} / √m*. Kaynağın yalnızca "
    "mertebe olarak verdiği 10⁸ ve 10⁵ M☉ değerleri testte karşılaştırılır.",
)
def hills_mass_kg(stellar_mass_kg: ArrayLike, stellar_radius_m: ArrayLike) -> Array:
    """Gelgit yarıçapının Schwarzschild yarıçapına eşit olduğu kara delik kütlesini (Hills kütlesi) hesaplar.

    r* (M / m*)^{1/3} = 2 G M / c²  ⇒  M_H = (r* c² / 2G)^{3/2} / √m*

    Bu kütlenin üstünde yıldız parlamadan bütün yutulur.

    Parameters
    ----------
    yildiz_kutlesi_kg : float ya da ndarray
        Yıldız kütlesi m* [kg].
    yildiz_yaricapi_m : float ya da ndarray
        Yıldız yarıçapı r* [m].

    Returns
    -------
    float ya da ndarray
        M_H [kg]. Güneş için ≈ 1,2e8 M☉.

    References
    ----------
    [Komossa2015] §1 (r_t denklemi ve yutulma eşiği).

    Validity
    --------
    Dönmeyen kara delik, Newton gelgit yarıçapı ve ufuk için Schwarzschild yarıçapı; Kerr
    ve göreli gelgit düzeltmeleri bu eşiği değiştirir.

    Verification
    ------------
    tests/unit/test_black_holes.py: Güneş ≈ 1e8 M☉, beyaz cüce ~1e5 M☉; r_t(M_H) = R_S özdeşliği.
    """
    m_y = _positive(stellar_mass_kg, "yıldız kütlesi")
    r_y = _positive(stellar_radius_m, "yıldız yarıçapı")
    return (r_y * _C**2 / (2.0 * _G)) ** 1.5 / np.sqrt(m_y)
