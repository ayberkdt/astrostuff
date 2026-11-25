"""Moleküler bulutların kararlılığı, zaman ölçekleri ve yıldız oluşum hızı.

Denklemler üç açık kaynaktan arXiv TeX dosyaları okunarak alınmıştır
(25 Kasım 2026): McKee ve Ostriker'in ARA&A derlemesi (arXiv:0707.3514,
[McKee2007]) ve Krumholz'un açık ders notları (arXiv:1511.03457,
[Krumholz2015]). Derlemenin TeX kaynağında denklem numarası yoktur; ``yer``
alanında TeX etiketi (``\\label``) verilmiştir.

Yazarlar Jeans niceliklerinde farklı sayısal çarpanlar kullanır; bu modülde
iki tanım da ``tanim`` bağımsız değişkeniyle seçilebilir ve ayrım ilgili
fonksiyonun belge dizgisinde açıkça belirtilmiştir.

Birimler SI'dır (kg, m, s, K, Pa); manyetik niceliklerde Gauss birimlerinden
SI'ya dönüşüm kesin çarpanlarla (1 T = 10⁴ G, 1 Wb = 10⁸ Mx) yapılır.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from kozmos._sources import source
from kozmos.core.constants import BOLTZMANN, GRAVITATIONAL_CONSTANT, PROTON_MASS

Array = NDArray[np.float64]

# Hidrojen atomunun kütlesi m_H yerine proton kütlesi kullanılır; fark yaklaşık
# %0,15'tir (elektron kütlesi eksi bağlanma enerjisi), kaynakların belirttiği
# μ değerlerinin belirsizliğinin çok altındadır.
_M_H = PROTON_MASS.value

# Ortalama molekül ağırlığı: H2 ve He (10 H başına 1 He) karışımı, PARÇACIK başına [Krumholz2015].
# Yoğunluk fonksiyonlarının varsayılanı budur: n, bütün parçacıkların (H2 + He) sayı yoğunluğudur.
MEAN_MOLECULAR_WEIGHT = 2.33
# n yalnızca H2 moleküllerinin sayı yoğunluğuysa (gözlemlerde yaygın olan n_H2) molekül başına kütle 2,8 m_H,
# n hidrojen çekirdeklerinin yoğunluğuysa (n_H) çekirdek başına kütle 1,4 m_H olur. Aynı helyum bolluğundan
# türetilir (2,33 · 1,2 = 2,8; 2,8 / 2 = 1,4); n_H için 1,4 değeri [McKee2007] eq:tff sayısal örneğinde geçer.
MOLECULAR_WEIGHT_PER_H2 = 2.8
MOLECULAR_WEIGHT_PER_H = 1.4


def _positive(**degerler: ArrayLike) -> list[Array]:
    """Bağımsız değişkenlerin kesin pozitif olduğunu sınar; dizilere çevirip döndürür."""
    cikti = []
    for name, d in degerler.items():
        a = np.asarray(d, dtype=float)
        if np.any(~(a > 0.0)):
            raise ValueError(f"{name} kesin pozitif olmalı")
        cikti.append(a)
    return cikti


@source(
    "Krumholz2015",
    where="Bölüm 4 (Gas Flows and Turbulence), Reynolds ve Mach sayıları alt bölümündeki c_s ifadesi",
    status="read",
    note="Kaynak μ = 2,33 alır ve 10 K için 0,18 km/s verir; μ bağımsız değişkendir",
)
def sound_speed_m_s(temperature_k: ArrayLike, mean_molecular_weight: float = MEAN_MOLECULAR_WEIGHT) -> Array:
    """Eşısıl ses hızını hesaplar: c_s = sqrt(k_B T / (μ m_H)).

    Parameters
    ----------
    sicaklik_k : float ya da ndarray
        Gaz sıcaklığı [K], pozitif.
    ortalama_molekul_agirligi : float
        Parçacık başına ortalama kütle, m_H biriminde; H2 ve He için 2,33.

    Returns
    -------
    float ya da ndarray
        Ses hızı [m/s].

    References
    ----------
    [Krumholz2015] Bölüm 4.

    Validity
    --------
    Eşısıl (γ = 1) gaz; yoğunluk arttıkça gaz adyabatik olur ve geçersizleşir.

    Verification
    ------------
    tests/unit/test_star_formation.py: 10 K için 0,18 km/s (kaynaktaki değer).
    """
    (t,) = _positive(temperature_k=temperature_k)
    (mu,) = _positive(mean_molecular_weight=mean_molecular_weight)
    return np.sqrt(BOLTZMANN.value * t / (mu * _M_H))


@source(
    "Krumholz2015",
    where="Bölüm 4 (Gas Flows and Turbulence), Reynolds ve Mach sayıları alt bölümü: M ~ V/c_s",
    status="read",
)
def mach_number(velocity_m_s: ArrayLike, sound_speed_m_s: ArrayLike) -> Array:
    """Mach sayısını hesaplar: M = V / c_s.

    M ≫ 1 ise basınç terimi akışın davranışında önemsizdir; kaynak, moleküler
    bulut için M ~ 20 bulur. Isıl ve türbülanslı desteğin göreli ağırlığını verir.

    Parameters
    ----------
    hiz_m_s : float ya da ndarray
        Tipik akış hızı ya da türbülans hız dağılımı [m/s], eksi olmayan.
    ses_hizi_m_s : float ya da ndarray
        Ses hızı [m/s], pozitif.

    Returns
    -------
    float ya da ndarray
        Boyutsuz Mach sayısı.

    References
    ----------
    [Krumholz2015] Bölüm 4.

    Verification
    ------------
    tests/unit/test_star_formation.py: ses hızına eşit hız için M = 1.
    """
    v = np.asarray(velocity_m_s, dtype=float)
    if np.any(v < 0):
        raise ValueError("hız eksi olamaz")
    (cs,) = _positive(ses_hizi=sound_speed_m_s)
    return v / cs


@source(
    "Krumholz2015",
    where="Bölüm 8 (Giant Molecular Clouds), Scaling Relations: n = 3M / (4π R³ μ m_H)",
    status="read",
    note="Kaynak μ = 2,3 (H2 molekülü başına) kullanır; burada μ bağımsız değişkendir",
)
def mass_density_kg_m3(number_density_m3: ArrayLike, mean_molecular_weight: float = MEAN_MOLECULAR_WEIGHT) -> Array:
    """Sayı yoğunluğundan kütle yoğunluğunu hesaplar: ρ = μ m_H n.

    Parameters
    ----------
    sayi_yogunlugu_m3 : float ya da ndarray
        Parçacık sayı yoğunluğu [m⁻³], eksi olmayan.
    ortalama_molekul_agirligi : float
        Parçacık başına ortalama kütle, m_H biriminde.

    Returns
    -------
    float ya da ndarray
        Kütle yoğunluğu [kg/m³].

    References
    ----------
    [Krumholz2015] Bölüm 8.

    Verification
    ------------
    tests/unit/test_star_formation.py: sayı yoğunluğu ile gidiş-dönüş.
    """
    n = np.asarray(number_density_m3, dtype=float)
    if np.any(n < 0):
        raise ValueError("sayı yoğunluğu eksi olamaz")
    (mu,) = _positive(mean_molecular_weight=mean_molecular_weight)
    return n * mu * _M_H


@source(
    "Krumholz2015",
    where="Bölüm 8 (Giant Molecular Clouds), Scaling Relations: n = 3M / (4π R³ μ m_H)",
    status="read",
)
def number_density_m3(mass_density_kg_m3: ArrayLike, mean_molecular_weight: float = MEAN_MOLECULAR_WEIGHT) -> Array:
    """Kütle yoğunluğundan sayı yoğunluğunu hesaplar: n = ρ / (μ m_H).

    Parameters
    ----------
    kutle_yogunlugu_kg_m3 : float ya da ndarray
        Kütle yoğunluğu [kg/m³], eksi olmayan.
    ortalama_molekul_agirligi : float
        Parçacık başına ortalama kütle, m_H biriminde.

    Returns
    -------
    float ya da ndarray
        Sayı yoğunluğu [m⁻³].

    References
    ----------
    [Krumholz2015] Bölüm 8.

    Verification
    ------------
    tests/unit/test_star_formation.py: ``kutle_yogunlugu_kg_m3`` ile gidiş-dönüş.
    """
    rho = np.asarray(mass_density_kg_m3, dtype=float)
    if np.any(rho < 0):
        raise ValueError("yoğunluk eksi olamaz")
    (mu,) = _positive(mean_molecular_weight=mean_molecular_weight)
    return rho / (mu * _M_H)


@source(
    "Krumholz2015",
    where="Bölüm 8 (Giant Molecular Clouds), Mass Measurement, Optically Thick Lines: N = M / (μ m_H π R²)",
    status="read",
    note="Kaynak N ile Σ arasındaki bağıntıyı ortalama sütun yoğunluğu için yazar; Σ = μ m_H N yeniden düzenlemedir",
)
def surface_density_kg_m2(column_density_m2: ArrayLike, mean_molecular_weight: float = MEAN_MOLECULAR_WEIGHT) -> Array:
    """Sütun yoğunluğundan yüzey yoğunluğunu hesaplar: Σ = μ m_H N.

    Parameters
    ----------
    sutun_yogunlugu_m2 : float ya da ndarray
        Parçacık sütun yoğunluğu [m⁻²], eksi olmayan.
    ortalama_molekul_agirligi : float
        Parçacık başına ortalama kütle, m_H biriminde.

    Returns
    -------
    float ya da ndarray
        Yüzey yoğunluğu [kg/m²].

    References
    ----------
    [Krumholz2015] Bölüm 8.

    Verification
    ------------
    tests/unit/test_star_formation.py: ``sutun_yogunlugu_m2`` ile gidiş-dönüş.
    """
    n = np.asarray(column_density_m2, dtype=float)
    if np.any(n < 0):
        raise ValueError("sütun yoğunluğu eksi olamaz")
    (mu,) = _positive(mean_molecular_weight=mean_molecular_weight)
    return n * mu * _M_H


@source(
    "Krumholz2015",
    where="Bölüm 8 (Giant Molecular Clouds), Mass Measurement, Optically Thick Lines: N = M / (μ m_H π R²)",
    status="read",
)
def column_density_m2(surface_density_kg_m2: ArrayLike, mean_molecular_weight: float = MEAN_MOLECULAR_WEIGHT) -> Array:
    """Yüzey yoğunluğundan sütun yoğunluğunu hesaplar: N = Σ / (μ m_H).

    Parameters
    ----------
    yuzey_yogunlugu_kg_m2 : float ya da ndarray
        Yüzey yoğunluğu [kg/m²], eksi olmayan.
    ortalama_molekul_agirligi : float
        Parçacık başına ortalama kütle, m_H biriminde.

    Returns
    -------
    float ya da ndarray
        Parçacık sütun yoğunluğu [m⁻²].

    References
    ----------
    [Krumholz2015] Bölüm 8.

    Verification
    ------------
    tests/unit/test_star_formation.py: ``yuzey_yogunlugu_kg_m2`` ile gidiş-dönüş.
    """
    s = np.asarray(surface_density_kg_m2, dtype=float)
    if np.any(s < 0):
        raise ValueError("yüzey yoğunluğu eksi olamaz")
    (mu,) = _positive(mean_molecular_weight=mean_molecular_weight)
    return s / (mu * _M_H)


@source("McKee2007", where="TeX etiketi LJeans", status="read")
@source("Krumholz2015", where="Bölüm 6 (Gravitational Instability and Collapse), λ_J = 2π/k_J", status="read")
def jeans_length_m(sound_speed_m_s: ArrayLike, density_kg_m3: ArrayLike) -> Array:
    """Jeans uzunluğunu hesaplar: λ_J = sqrt(π c_s² / (G ρ)).

    Parameters
    ----------
    ses_hizi_m_s : float ya da ndarray
        Eşısıl ses hızı [m/s], pozitif.
    yogunluk_kg_m3 : float ya da ndarray
        Düzgün ortamın yoğunluğu [kg/m³], pozitif.

    Returns
    -------
    float ya da ndarray
        Jeans uzunluğu [m]: kütleçekimsel parçalanmanın en küçük dalga boyu.

    References
    ----------
    [McKee2007] LJeans; [Krumholz2015] Bölüm 6 (doğrusal analizden türetir).
    İki kaynak aynı ifadeyi verir.

    Validity
    --------
    Varsayılan μ = 2,33 parçacık başınadır: ``sayi_yogunlugu_m3`` bütün parçacıkların (H2 + He)
    yoğunluğu olmalıdır. Yalnızca H2 yoğunluğu (n_H2) veriliyorsa ``H2_BASINA_MOLEKUL_AGIRLIGI`` (2,8),
    hidrojen çekirdeği yoğunluğu (n_H) veriliyorsa ``H_BASINA_MOLEKUL_AGIRLIGI`` (1,4) kullanılmalıdır;
    n_H2 varsayılanla verilirse yoğunluk yaklaşık %17 düşük, serbest düşme süresi %9 uzun çıkar.
    Düzgün, sonsuz, eşısıl ortam ("Jeans hilesi"); yalnızca basınç desteği.

    Verification
    ------------
    tests/unit/test_star_formation.py: c_s = 0,2 km/s, ρ = 100 m_p için 3,4 pc.
    """
    cs, rho = _positive(ses_hizi=sound_speed_m_s, density=density_kg_m3)
    return np.sqrt(np.pi * cs**2 / (GRAVITATIONAL_CONSTANT.value * rho))


@source(
    "McKee2007",
    where="TeX etiketi LJeans, ardından gelen paragraf (M_J = 2,47 M_BE)",
    status="read",
    note="tanim='kure': M_J = (4π/3)(λ_J/2)³ ρ (McKee-Ostriker, Binney-Tremaine 1987 tanımı)",
)
@source(
    "Krumholz2015",
    where="Bölüm 6, Jeans kütlesi tanımı ve dipnotu",
    status="read",
    note="tanim='kup': M_J = ρ λ_J³ / 8 (kenarı λ_J/2 olan küp); aradaki oran 4π/3'tür",
)
def jeans_mass_kg(sound_speed_m_s: ArrayLike, density_kg_m3: ArrayLike, definition: str = "kure") -> Array:
    """Jeans kütlesini hesaplar; çarpan yazara göre değişir.

    ``tanim='kure'``: M_J = (4π/3)(λ_J/2)³ ρ = (π^{5/2}/6) c_s³ / sqrt(G³ ρ)
    ([McKee2007]; Bonnor-Ebert kütlesinin 2,47 katı).
    ``tanim='kup'``: M_J = ρ λ_J³ / 8 = (π^{3/2}/8) c_s³ / sqrt(G³ ρ) ([Krumholz2015]).
    Kaynaklar bu çarpanın belirsiz olduğunu açıkça söyler; ölçek bağıntısı
    c_s³ / sqrt(G³ ρ) her iki tanımda aynıdır.

    Parameters
    ----------
    ses_hizi_m_s : float ya da ndarray
        Eşısıl ses hızı [m/s], pozitif.
    yogunluk_kg_m3 : float ya da ndarray
        Yoğunluk [kg/m³], pozitif.
    tanim : {"kure", "kup"}
        Çarpan tanımı.

    Returns
    -------
    float ya da ndarray
        Jeans kütlesi [kg].

    References
    ----------
    [McKee2007] LJeans sonrası paragraf; [Krumholz2015] Bölüm 6.

    Validity
    --------
    ``jeans_boyu_m`` ile aynı.

    Verification
    ------------
    tests/unit/test_star_formation.py: iki tanımın oranı 4π/3; M_J = 2,47 M_BE.
    """
    cs, rho = _positive(ses_hizi=sound_speed_m_s, density=density_kg_m3)
    lam = np.sqrt(np.pi * cs**2 / (GRAVITATIONAL_CONSTANT.value * rho))
    if definition == "kure":
        return (4.0 * np.pi / 3.0) * (lam / 2.0) ** 3 * rho
    if definition == "kup":
        return rho * lam**3 / 8.0
    raise ValueError("tanim 'kure' ya da 'kup' olmalı")


@source(
    "McKee2007",
    where="TeX etiketi MBE_def",
    status="read",
    note="Katsayı 1,182 (Krumholz2015 alıştırma çözümü 1,18 verir)",
)
@source("Krumholz2015", where="problems/probsol2.tex, Bonnor-Ebert küresi alıştırmasının (h) şıkkı", status="read")
def bonnor_ebert_critical_mass_kg(sound_speed_m_s: ArrayLike, surface_pressure_pa: ArrayLike) -> Array:
    """Bonnor-Ebert kütlesini hesaplar: M_BE = 1,182 c_s⁴ / sqrt(G³ P_0).

    Verilen yüzey basıncında hidrostatik dengede olabilen eşısıl, manyetik
    alansız kürenin en büyük kütlesidir (kararlı ya da kararsız).

    Parameters
    ----------
    ses_hizi_m_s : float ya da ndarray
        Eşısıl ses hızı [m/s], pozitif.
    yuzey_basinci_pa : float ya da ndarray
        Yüzeydeki dış basınç P_0 = ρ_0 c_s² [Pa], pozitif.

    Returns
    -------
    float ya da ndarray
        Kritik kütle [kg].

    References
    ----------
    [McKee2007] MBE_def; [Krumholz2015] probsol2 (alan: m_max = 1,18).

    Validity
    --------
    Eşısıl, dönmeyen, manyetik alansız, türbülanssız küre.

    Verification
    ------------
    tests/unit/test_star_formation.py: T = 10 K, P/k_B = 3e5 cm⁻³ K için
    kaynaklardaki 0,66 ve 0,67 güneş kütlesi; ODE'den türetilen katsayıyla uyum.
    """
    cs, p = _positive(ses_hizi=sound_speed_m_s, yuzey_basinci=surface_pressure_pa)
    return 1.182 * cs**4 / np.sqrt(GRAVITATIONAL_CONSTANT.value**3 * p)


@source(
    "McKee2007",
    where="TeX etiketi MBE_def, sonraki paragraf: R_BE = 0,486 c_s / sqrt(G ρ_0)",
    status="read",
)
def bonnor_ebert_radius_m(sound_speed_m_s: ArrayLike, surface_value_density_kg_m3: ArrayLike) -> Array:
    """Kritik Bonnor-Ebert küresinin yarıçapını hesaplar: R_BE = 0,486 c_s / sqrt(G ρ_0).

    Parameters
    ----------
    ses_hizi_m_s : float ya da ndarray
        Eşısıl ses hızı [m/s], pozitif.
    yuzeydeki_yogunluk_kg_m3 : float ya da ndarray
        Kürenin yüzeyindeki yoğunluk ρ_0 [kg/m³], pozitif.

    Returns
    -------
    float ya da ndarray
        Yarıçap [m].

    References
    ----------
    [McKee2007] MBE_def sonrası paragraf.

    Validity
    --------
    ``bonnor_ebert_kritik_kutle_kg`` ile aynı.

    Verification
    ------------
    tests/unit/test_star_formation.py: ODE'den türetilen ξ ve yoğunluk
    karşıtlığıyla uyum; R_BE ≈ Jeans uzunluğunun 0,486/√π katı.
    """
    cs, rho = _positive(ses_hizi=sound_speed_m_s, yuzeydeki_yogunluk=surface_value_density_kg_m3)
    return 0.486 * cs / np.sqrt(GRAVITATIONAL_CONSTANT.value * rho)


def _be_ode(xi: float, y: list[float]) -> list[float]:
    """Boyutsuz eşısıl küre denklemi: ψ'' + 2ψ'/ξ = exp(−ψ)."""
    psi, dpsi = y
    return [dpsi, -2.0 * dpsi / xi + np.exp(-psi)]


@lru_cache(maxsize=1)
def _be_coefficients() -> tuple[float, float, float]:
    """ODE'yi çözüp (m_max, ρ_c/ρ_s, ξ_s) üçlüsünü hesaplar.

    m(ξ) = (4π)^{-1/2} ξ² ψ' exp(−ψ/2); kaynaktaki kütle integrali
    ∫ ξ² e^{−ψ} dξ = ξ² ψ' özdeşliğiyle ve P_s = ρ_s c_s² ile elde edilir.
    """
    xi0 = 1e-4
    cozum = solve_ivp(_be_ode, (xi0, 30.0), [xi0**2 / 6.0, xi0 / 3.0], rtol=1e-11, atol=1e-13, dense_output=True)

    def eksi_m(xi: float) -> float:
        psi, dpsi = cozum.sol(xi)
        return float(-(xi**2) * dpsi * np.exp(-psi / 2.0) / np.sqrt(4.0 * np.pi))

    izgara = np.linspace(2.0, 15.0, 2000)
    en_iyi = izgara[int(np.argmin([eksi_m(x) for x in izgara]))]
    r = minimize_scalar(eksi_m, bounds=(en_iyi - 0.05, en_iyi + 0.05), method="bounded", options={"xatol": 1e-10})
    xi_s = float(r.x)
    psi_s = float(cozum.sol(xi_s)[0])
    return float(-r.fun), float(np.exp(psi_s)), xi_s


@source(
    "Krumholz2015",
    where="problems/probset2.tex (e) şıkkı (kütle ifadesi); probsol2.tex (c)-(d) şıkları (ODE), (f) şıkkı (m_max = 1,18, ρ_c/ρ_s = 14,0)",
    status="read",
    note="ODE ve m(ξ) = (4π)^{-1/2} ξ² ψ' e^{-ψ/2} ifadesi kaynaktan; yalnızca ξ_s ≈ 6,45 değeri kaynakta yoktur, sayısal çözümden gelir",
)
def bonnor_ebert_critical_coefficients() -> tuple[float, float, float]:
    """Eşısıl küre denkleminden Bonnor-Ebert kritik katsayılarını hesaplar.

    ψ'' + 2ψ'/ξ = e^{−ψ} denklemi (ψ(0) = ψ'(0) = 0) sayısal çözülür;
    boyutsuz kütle m = M sqrt(P_s) G^{3/2} / c_s⁴ en büyük değerine ulaştığında
    küre kritiktir.

    Returns
    -------
    tuple of float
        (m_max, ρ_c/ρ_s, ξ_s): en büyük boyutsuz kütle (≈ 1,18), kritik
        merkez/yüzey yoğunluk karşıtlığı (≈ 14,0) ve boyutsuz yüzey yarıçapı
        ξ_s = R/r_0 (≈ 6,45).

    References
    ----------
    [Krumholz2015] probsol2.

    Validity
    --------
    ξ_s > ξ_kritik olan küreler kararsızdır; ρ_c/ρ_s > 14 kararsız demektir.

    Verification
    ------------
    tests/unit/test_star_formation.py: kaynaktaki 1,18 ve 14,0 ile; McKee2007'deki
    1,182 ve R_BE = 0,486 c_s/sqrt(G ρ_0) ile tutarlılık.
    """
    return _be_coefficients()


@source("McKee2007", where="TeX etiketi alpha_vir_def", status="read")
@source(
    "Krumholz2015", where="Bölüm 8 (Giant Molecular Clouds), Larson'un üçüncü yasası: α_vir = 5σ²R/(GM)", status="read"
)
def virial_parameter(velocity_dispersion_m_s: ArrayLike, radius_m: ArrayLike, mass_kg: ArrayLike) -> Array:
    """Viryal parametreyi hesaplar: α_vir = 5 σ² R / (G M).

    Katsayı, viryal dengedeki düzgün, manyetik alansız küre için α_vir = 1
    olacak biçimde seçilmiştir (bu küre hidrostatik dengede değildir).
    α_vir ≲ 1 bağlı, ≫ 1 dış basınçla tutulan bulutlara karşılık gelir.

    Parameters
    ----------
    hiz_dagilimi_m_s : float ya da ndarray
        Tek boyutlu toplam (ısıl + türbülanslı) hız dağılımı σ [m/s], pozitif.
    yaricap_m : float ya da ndarray
        Bulut yarıçapı [m], pozitif.
    kutle_kg : float ya da ndarray
        Bulut kütlesi [kg], pozitif.

    Returns
    -------
    float ya da ndarray
        Boyutsuz viryal parametre.

    References
    ----------
    [McKee2007] alpha_vir_def; [Krumholz2015] Bölüm 8.

    Validity
    --------
    Manyetik terim ve yüzey basıncı hesaba katılmaz.

    Verification
    ------------
    tests/unit/test_star_formation.py: σ² = GM/(5R) için α_vir = 1; yüzey
    yoğunluğu biçimiyle tutarlılık.
    """
    sigma, r, m = _positive(hiz_dagilimi=velocity_dispersion_m_s, radius=radius_m, mass=mass_kg)
    return 5.0 * sigma**2 * r / (GRAVITATIONAL_CONSTANT.value * m)


@source(
    "McKee2007",
    where="TeX etiketi alpha_vir_def",
    status="read",
    note="M = π R² Σ yerine koymasıyla yazıldı; Krumholz2015 Bölüm 8'deki (5/π pc) σ_pc²/(GΣ) biçimiyle aynıdır",
)
def virial_parameter_from_surface_density(
    velocity_dispersion_m_s: ArrayLike, radius_m: ArrayLike, surface_density_kg_m2: ArrayLike
) -> Array:
    """Viryal parametreyi yüzey yoğunluğundan hesaplar: α_vir = 5 σ² / (π G Σ R).

    Parameters
    ----------
    hiz_dagilimi_m_s : float ya da ndarray
        Tek boyutlu hız dağılımı [m/s], pozitif.
    yaricap_m : float ya da ndarray
        Bulut yarıçapı [m], pozitif.
    yuzey_yogunlugu_kg_m2 : float ya da ndarray
        Ortalama yüzey yoğunluğu Σ = M/(π R²) [kg/m²], pozitif.

    Returns
    -------
    float ya da ndarray
        Boyutsuz viryal parametre.

    References
    ----------
    [McKee2007] alpha_vir_def.

    Validity
    --------
    ``viryal_parametresi`` ile aynı; yüzey yoğunluğu izdüşüm dairesine göre alınır.

    Verification
    ------------
    tests/unit/test_star_formation.py: M = π R² Σ ile ``viryal_parametresi`` eşitliği.
    """
    sigma, r, s = _positive(
        hiz_dagilimi=velocity_dispersion_m_s, radius=radius_m, yuzey_yogunlugu=surface_density_kg_m2
    )
    return 5.0 * sigma**2 / (np.pi * GRAVITATIONAL_CONSTANT.value * s * r)


@source(
    "McKee2007", where="TeX etiketi eq:tff", status="read", note="Sayısal değer n_H için He oranı %10 alınarak verilir"
)
@source("Krumholz2015", where="Bölüm 6, Jeans analizinden sonra t_ff tanımı; Bölüm 8 zaman ölçekleri", status="read")
def free_fall_time_s(density_kg_m3: ArrayLike) -> Array:
    """Serbest düşme zamanını hesaplar: t_ff = sqrt(3π / (32 G ρ)).

    Basınçsız, düzgün bir kürenin kendi çekimiyle bir noktaya çökme süresidir.

    Parameters
    ----------
    yogunluk_kg_m3 : float ya da ndarray
        Ortalama kütle yoğunluğu [kg/m³], pozitif.

    Returns
    -------
    float ya da ndarray
        Serbest düşme zamanı [s].

    References
    ----------
    [McKee2007] eq:tff; [Krumholz2015] Bölüm 6 ve 8.

    Validity
    --------
    Basınç, türbülans ve manyetik destek yok sayılır; gerçek bulutlar daha yavaş çöker.

    Verification
    ------------
    tests/unit/test_star_formation.py: n_H = 10³ cm⁻³ için 1,37 milyon yıl.
    """
    (rho,) = _positive(density=density_kg_m3)
    return np.sqrt(3.0 * np.pi / (32.0 * GRAVITATIONAL_CONSTANT.value * rho))


@source("Krumholz2015", where="Bölüm 8 (Giant Molecular Clouds), Molecular Cloud Timescales: t_cr = R/σ", status="read")
def crossing_time_s(radius_m: ArrayLike, velocity_dispersion_m_s: ArrayLike) -> Array:
    """Geçiş (crossing) zamanını hesaplar: t_cr = R / σ.

    σ olarak ses hızı verilirse ses geçiş zamanı, türbülans hız dağılımı
    verilirse türbülans geçiş zamanı elde edilir.

    Parameters
    ----------
    yaricap_m : float ya da ndarray
        Bulut yarıçapı [m], pozitif.
    hiz_dagilimi_m_s : float ya da ndarray
        Hız dağılımı ya da ses hızı [m/s], pozitif.

    Returns
    -------
    float ya da ndarray
        Geçiş zamanı [s].

    References
    ----------
    [Krumholz2015] Bölüm 8.

    Verification
    ------------
    tests/unit/test_star_formation.py: viryal bulutta t_cr ≈ 2 t_ff (kaynak: α_vir = 1
    için serbest düşme zamanı geçiş zamanının yarısıdır).
    """
    r, sigma = _positive(radius=radius_m, hiz_dagilimi=velocity_dispersion_m_s)
    return r / sigma


@source(
    "McKee2007",
    where="TeX etiketi eq:mphi",
    status="read",
    note="c_Φ = 0,17 Tomisaka (1988) değeridir (kaynakta aktarılmış, orijinali okunmadı)",
)
@source(
    "Krumholz2015",
    where="Bölüm 6, Magnetic Support and the Magnetic Critical Mass",
    status="read",
    note="Kaynak düzgün küre için sqrt(5/2)/(3π) = 0,17 türetir ve Tomisaka (1998) için 0,12 verir; McKee2007 ise 0,17 için başka bir makaleyi (Tomisaka, Ikeuchi ve Nakamura 1988) anar; iki özgün makale de okunmadı",
)
def magnetic_critical_mass_kg(magnetic_flux_wb: ArrayLike, c_phi: float = 0.17) -> Array:
    """Manyetik kritik kütleyi hesaplar: M_Φ = c_Φ Φ / G^{1/2} (Gauss birimlerinde).

    M > M_Φ ise bulut manyetik olarak süperkritiktir ve alan çöküşü
    durduramaz; M < M_Φ ise çökemez. Φ = π R² B düzgün alan için akıdır.
    SI'ya dönüşüm: Φ[Mx] = 10⁸ Φ[Wb], G_cgs = 10³ G_SI, kg = 10⁻³ g.

    Parameters
    ----------
    manyetik_aki_wb : float ya da ndarray
        Buluttan geçen manyetik akı [Wb], pozitif.
    c_phi : float
        İç yoğunluk ve alan dağılımına bağlı katsayı; 0,17 (kutupsal alan,
        sabit kütle/akı oranı), düzgün küre için sqrt(5/2)/(3π) ≈ 0,168.

    Returns
    -------
    float ya da ndarray
        Kritik kütle [kg].

    References
    ----------
    [McKee2007] eq:mphi; [Krumholz2015] Bölüm 6.

    Validity
    --------
    Soğuk bulut, ideal MHD (akı donmuş); çift kutuplu ya da rastgele alanda
    toplam akı kütleyi belirlemez.

    Verification
    ------------
    tests/unit/test_star_formation.py: levha eşdeğeriyle (Σ/B)_kr = 1/(2π√G)
    tutarlılık; Φ ile doğrusal ölçekleme.
    """
    (phi,) = _positive(manyetik_aki=magnetic_flux_wb)
    if not c_phi > 0:
        raise ValueError("c_phi pozitif olmalı")
    g_cgs = GRAVITATIONAL_CONSTANT.value * 1e3
    return c_phi * (phi * 1e8) / np.sqrt(g_cgs) * 1e-3


@source(
    "McKee2007",
    where="TeX etiketi eq:mphi sonrası: G^{1/2}(Σ/B)_cr = 1/(2π) (sonsuz soğuk levha)",
    status="read",
    note="Kaynak değeri Nakano-Nakamura (1978) için aktarır; orijinali okunmadı. Normlu oran 2π√G Σ/B olarak yazıldı",
)
def normalized_mass_to_flux_ratio(surface_density_kg_m2: ArrayLike, magnetic_field_t: ArrayLike) -> Array:
    """Kütle/akı oranının kritik değere oranını hesaplar: μ/μ_kr = 2π √G Σ / B.

    Kritik değer G^{1/2}(Σ/B)_kr = 1/(2π) (Gauss birimlerinde). Oran 1'den
    büyükse bulut manyetik olarak süperkritiktir.

    Parameters
    ----------
    yuzey_yogunlugu_kg_m2 : float ya da ndarray
        Yüzey yoğunluğu [kg/m²], pozitif.
    manyetik_alan_t : float ya da ndarray
        Yüzeye dik alan bileşeni [T], pozitif. (1 T = 10⁴ G; Σ[g/cm²] = 0,1 Σ[kg/m²].)

    Returns
    -------
    float ya da ndarray
        Boyutsuz normlu kütle/akı oranı.

    References
    ----------
    [McKee2007] eq:mphi sonrası paragraf.

    Validity
    --------
    Sonsuz soğuk levha; küresel bulutlar için c_Φ ≈ 0,17 ≈ 1/(2π) ile yaklaşık eşdeğer.

    Verification
    ------------
    tests/unit/test_star_formation.py: Σ = B/(2π√G) için oran 1; küre M_Φ ile tutarlılık.
    """
    s, b = _positive(yuzey_yogunlugu=surface_density_kg_m2, manyetik_alan=magnetic_field_t)
    g_cgs = GRAVITATIONAL_CONSTANT.value * 1e3
    return 2.0 * np.pi * np.sqrt(g_cgs) * (s * 0.1) / (b * 1e4)


@source(
    "McKee2007",
    where="TeX etiketi GMC_lws (Heyer ve Brunt 2004 PCA uyumu)",
    status="read",
    note="Varsayılan 0,9 km/s ve 0,56, McKee2007'de aktarılan PCA uyumudur; L_pca bulut çapının yaklaşık dörtte biridir",
)
@source(
    "Krumholz2015",
    where="Bölüm 8 (Giant Molecular Clouds), Scaling Relations: Larson'un ikinci yasası",
    status="read",
    note="Kaynakta aktarılan alternatifler: 0,72 ve 0,5 (Solomon 1987, Samanyolu), 0,44 ve 0,60 (Bolatto 2008); orijinalleri okunmadı",
)
def linewidth_size_km_s(size_pc: ArrayLike, coefficient_km_s: float = 0.9, exponent: float = 0.56) -> Array:
    """Larson doğrusal genişlik-boyut bağıntısını hesaplar: δv = a (L / 1 pc)^q.

    Parameters
    ----------
    boyut_pc : float ya da ndarray
        Ölçek ya da bulut yarıçapı [pc], pozitif.
    katsayi_km_s : float
        1 pc'teki hız dağılımı a [km/s].
    us : float
        Güç yasası üssü q (türbülans için ≈ 0,5; Kolmogorov 1/3).

    Returns
    -------
    float ya da ndarray
        Hız dağılımı [km/s].

    References
    ----------
    [McKee2007] GMC_lws; [Krumholz2015] Bölüm 8.

    Validity
    --------
    Samanyolu'ndaki moleküler bulutlar ve içlerindeki 0,03-30 pc ölçekleri; boyut
    tanımı (PCA ölçeği ya da bulut yarıçapı) normalizasyonu değiştirir.

    Verification
    ------------
    tests/unit/test_star_formation.py: 1 pc'te a; üs ölçeklemesi; kaynaktaki alternatifler.
    """
    (size,) = _positive(size=size_pc)
    return coefficient_km_s * size**exponent


@source(
    "McKee2007",
    where="TeX etiketi eq:mdssis (Shu genişleme dalgası çözümü)",
    status="read",
    note="Kaynak genel biçimi Φ c_s³/G (TeX eq:mds) yazar; Shu çözümü için Φ = 0,975",
)
def isothermal_collapse_accretion_rate_kg_s(sound_speed_m_s: ArrayLike, coefficient: float = 0.975) -> Array:
    """Eşısıl çekirdeğin çöküşünde protoyıldıza kütle akış hızını hesaplar: Ṁ = 0,975 c_s³ / G.

    Shu genişleme dalgası çözümünde hız zamanla sabittir; dalganın içindeki
    kütle 2Ṁt'dir ve yaklaşık yarısı protoyıldızdadır.

    Parameters
    ----------
    ses_hizi_m_s : float ya da ndarray
        Eşısıl ses hızı [m/s], pozitif.
    katsayi : float
        Boyutsuz çarpan Φ; Shu için 0,975, Larson-Penston-Hunter çözümü daha büyüktür.

    Returns
    -------
    float ya da ndarray
        Kütle akış hızı [kg/s].

    References
    ----------
    [McKee2007] eq:mdssis.

    Validity
    --------
    Dönmesiz, manyetik alansız, eşısıl çekirdek; protoyıldız çıkışları hızı bir
    ε_core < 1 çarpanıyla azaltır.

    Verification
    ------------
    tests/unit/test_star_formation.py: 10 K için 1,54e-6 güneş kütlesi/yıl.
    """
    (cs,) = _positive(ses_hizi=sound_speed_m_s)
    if not coefficient > 0:
        raise ValueError("katsayı pozitif olmalı")
    return coefficient * cs**3 / GRAVITATIONAL_CONSTANT.value


@source("McKee2007", where="TeX etiketi eq:epsff", status="read")
@source("Krumholz2015", where="Bölüm 8 (Giant Molecular Clouds), Depletion Time: ε_ff = t_ff / t_dep", status="read")
def free_fall_efficiency(free_fall_time_s: ArrayLike, depletion_time_s: ArrayLike) -> Array:
    """Serbest düşme zamanı başına yıldız oluşum verimliliğini hesaplar: ε_ff = t_ff / t_dep.

    Bir bulutun kütlesinin bir serbest düşme zamanında yıldıza dönüşen oranıdır.

    Parameters
    ----------
    serbest_dusme_zamani_s : float ya da ndarray
        Serbest düşme zamanı [s], pozitif.
    tuketme_zamani_s : float ya da ndarray
        Gaz tüketme zamanı M/Ṁ_* [s], pozitif.

    Returns
    -------
    float ya da ndarray
        Boyutsuz verimlilik.

    References
    ----------
    [McKee2007] eq:epsff; [Krumholz2015] Bölüm 8. Kaynaklar gözlenen tipik
    değerin ε_ff ~ 0,01 olduğunu belirtir.

    Verification
    ------------
    tests/unit/test_star_formation.py: ``bulut_yildiz_olusum_hizi_kg_s`` ile gidiş-dönüş.
    """
    tff, tdep = _positive(serbest_dusme_zamani=free_fall_time_s, tuketme_zamani=depletion_time_s)
    return tff / tdep


@source(
    "McKee2007",
    where="TeX etiketi eq:epsff",
    status="read",
    note="Ṁ_* = ε_ff M / t_ff; ε_ff için kaynakta tipik değer ~0,01 (Krumholz2015 Bölüm 8-10)",
)
def cloud_star_formation_rate_kg_s(
    cloud_mass_kg: ArrayLike, mean_density_kg_m3: ArrayLike, ff_efficiency: float = 0.01
) -> Array:
    """Bulutun yıldız oluşum hızını hesaplar: Ṁ_* = ε_ff M / t_ff.

    Parameters
    ----------
    bulut_kutlesi_kg : float ya da ndarray
        Gaz kütlesi [kg], pozitif.
    ortalama_yogunluk_kg_m3 : float ya da ndarray
        Bulutun ortalama yoğunluğu [kg/m³]; t_ff bu yoğunlukla hesaplanır.
    ff_verimliligi : float
        Serbest düşme zamanı başına verimlilik ε_ff, [0, 1] aralığında;
        gözlenen tipik değer ≈ 0,01.

    Returns
    -------
    float ya da ndarray
        Yıldız oluşum hızı [kg/s].

    References
    ----------
    [McKee2007] eq:epsff.

    Validity
    --------
    ε_ff ölçülen ya da varsayılan bir sabittir; bu fonksiyon onu öngörmez.

    Verification
    ------------
    tests/unit/test_star_formation.py: M/Ṁ_* = t_ff/ε_ff; ε_ff ile doğrusal ölçekleme.
    """
    m, rho = _positive(mass=cloud_mass_kg, density=mean_density_kg_m3)
    if not 0.0 <= ff_efficiency <= 1.0:
        raise ValueError("ε_ff [0, 1] aralığında olmalı")
    return ff_efficiency * m / free_fall_time_s(rho)
