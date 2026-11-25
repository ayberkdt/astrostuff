"""Kara delik geometrisi, Kerr yörüngeleri, Hawking termodinamiği ve ikili kara delikler.

Denklemler şu kaynak metinlerinden 25 Kasım 2026'da okunmuştur: Abramowicz ve Fragile
[Abramowicz2013] (arXiv:1104.5499, TeX), Page [Page2005] (arXiv:hep-th/0409024, TeX),
Sathyaprakash ve Schutz [Sathyaprakash2009] (arXiv:0903.0338, TeX) ve Event Horizon
Telescope Paper I [EHT2019] (arXiv:1906.11238, yalnızca PDF; TeX kaynağı yok, bu yüzden
denklem numarası yerine bölüm ve sayfa gösterilir).

Okunmayan her şey ``teyit="bekliyor"`` taşır ve fonksiyonun ``not_`` alanında neyin
okunup neyin okunmadığı yazılır.

Kerr kara delik için ``spin`` boyutsuz dönme a_* = a/M = cJ/(G M²) değeridir;
pozitif değer yörüngenin kara deliğin dönüşüyle aynı yönde (doğru, prograde),
negatif değer ters yönde (retrograde) olduğunu gösterir. Yarıçaplar ``_rg`` ekiyle
r_G = G M / c² birimindedir.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.constants import BOLTZMANN, GRAVITATIONAL_CONSTANT, PLANCK, SPEED_OF_LIGHT

Array = NDArray[np.float64]

_G = GRAVITATIONAL_CONSTANT.value
_C = SPEED_OF_LIGHT.value
_HBAR = PLANCK.value / (2.0 * np.pi)  # indirgenmiş Planck sabiti, ħ = h / 2π
_RAD_UAS = 180.0 / np.pi * 3600.0 * 1e6  # radyan → mikroyay saniyesi


def _positive(value: ArrayLike, name: str) -> Array:
    """Değeri diziye çevirir ve pozitif olduğunu denetler."""
    x = np.asarray(value, dtype=float)
    if np.any(~(x > 0.0)):
        raise ValueError(f"{name} pozitif olmalı")
    return x


def _spin(spin: ArrayLike) -> Array:
    """Boyutsuz dönmeyi diziye çevirir ve |a_*| ≤ 1 olduğunu denetler."""
    a = np.asarray(spin, dtype=float)
    if np.any(~(np.abs(a) <= 1.0)):
        raise ValueError("boyutsuz dönme |a*| ≤ 1 olmalı")
    return a


# --------------------------------------------------------------------------------------
# Schwarzschild geometrisi
# --------------------------------------------------------------------------------------


@source("EHT2019", where="§1 (Giriş), r_g ≡ GM/c² tanımı, sayfa L1", status="read")
def gravitational_radius_m(mass_kg: ArrayLike) -> Array:
    """Kütle çekimsel yarıçapı hesaplar: r_G = G M / c².

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg], pozitif.

    Returns
    -------
    float ya da ndarray
        r_G [m].

    References
    ----------
    [EHT2019] §1, "r_g ≡ GM/c² ... karakteristik uzunluk ölçeği".

    Verification
    ------------
    tests/unit/test_black_holes.py: Güneş kütlesi için 1,477 km; kütleyle doğrusal.
    """
    return _G * _positive(mass_kg, "kütle") / _C**2


@source("EHT2019", where="§1 (Giriş), R_S = 2 r_g, sayfa L1", status="read")
def schwarzschild_radius_m(mass_kg: ArrayLike) -> Array:
    """Schwarzschild yarıçapını (dönmeyen kara deliğin ufku) hesaplar: R_S = 2 G M / c².

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg], pozitif.

    Returns
    -------
    float ya da ndarray
        R_S [m].

    References
    ----------
    [EHT2019] §1, "R_S = 2 r_g".

    Verification
    ------------
    tests/unit/test_black_holes.py: Güneş kütlesi için 2,95 km; Kerr ufkunun a* = 0 limitiyle aynı.
    """
    return 2.0 * gravitational_radius_m(mass_kg)


@source(
    "Carroll1997",
    where="Bölüm 7, Denk. 7.58-7.61",
    status="read",
    note="sabit konumdaki iki gözlemci için ω2/ω1 = [(1 − 2GM/r1)/(1 − 2GM/r2)]^{1/2} (G = c = 1); "
    "burada alıcı sonsuzda (r2 → ∞): 1 + z = (1 − R_S/r)^{−1/2}",
)
def gravitational_redshift(radius_m: ArrayLike, mass_kg: ArrayLike) -> Array:
    """Statik yayıcının sonsuzdaki gözlemciye göre kütle çekimsel kırmızıya kaymasını hesaplar.

    1 + z = (1 − R_S / r)^(−1/2)

    Parameters
    ----------
    yaricap_m : float ya da ndarray
        Yayıcının Schwarzschild yarıçap koordinatı [m]; R_S değerinden büyük olmalı.
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].

    Returns
    -------
    float ya da ndarray
        Kırmızıya kayma z (boyutsuz).

    References
    ----------
    [Carroll1997] Denk. 7.58-7.61.

    Validity
    --------
    Yalnızca durağan (yörüngede olmayan) yayıcı; r > R_S. Ufka yaklaşırken z → ∞.

    Verification
    ------------
    tests/unit/test_black_holes.py: r = 3 R_S/2 için z = √3 − 1; r → ∞ için z ≈ R_S/(2r).
    """
    r = _positive(radius_m, "yarıçap")
    rs = schwarzschild_radius_m(mass_kg)
    if np.any(r <= rs):
        raise ValueError("yarıçap Schwarzschild yarıçapından büyük olmalı")
    return (1.0 - rs / r) ** -0.5 - 1.0


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies' (TeX etiketi schwarzschild-frequencies), x_ph = 3",
    status="read",
    note="Kerr formülünün (TeX etiketi photon-radius-kerr) a* = 0 değeri de 3 r_G verir.",
)
def photon_sphere_radius_m(mass_kg: ArrayLike) -> Array:
    """Dönmeyen kara deliğin foton küresinin yarıçapını hesaplar: r_ph = 3 G M / c².

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].

    Returns
    -------
    float ya da ndarray
        r_ph [m].

    References
    ----------
    [Abramowicz2013] Schwarzschild için x_ph = r_ph/r_G = 3.

    Verification
    ------------
    tests/unit/test_black_holes.py: Kerr formülünün a* = 0 değeriyle ve R_S oranıyla (3/2).
    """
    return 3.0 * gravitational_radius_m(mass_kg)


@source(
    "EHT2019",
    where="§1 (Giriş), 'foton yakalama yarıçapı' R_c = √27 r_g, sayfa L1",
    status="read",
    note="PDF'ten okundu; metin çıkarımı √ simgesini düşürüyor, sayfa görüntüsünde R_c = √27 r_g doğrulandı.",
)
def critical_impact_parameter_m(mass_kg: ArrayLike) -> Array:
    """Dönmeyen kara deliğin kritik çarpışma parametresini hesaplar: b_c = √27 G M / c².

    b < b_c ile gelen fotonlar yakalanır, b > b_c olanlar kaçar; b = b_c kararsız
    dairesel foton yörüngesidir (foton halkası).

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].

    Returns
    -------
    float ya da ndarray
        b_c [m].

    References
    ----------
    [EHT2019] §1, "R_c = √27 r_g".

    Verification
    ------------
    tests/unit/test_black_holes.py: foton küresi yarıçapı ve Schwarzschild metriğiyle
    bağımsız hesap b_c = r_ph / √(1 − R_S/r_ph).
    """
    return np.sqrt(27.0) * gravitational_radius_m(mass_kg)


@source(
    "EHT2019",
    where="§1 R_c = √27 r_g; §7 'Model Comparison and Parameter Estimation' θ_g ≡ GM/(c² D)",
    status="read",
    note="Çap = 2 R_c / D küçük açı yaklaşımıdır (geometri, kaynakta yazılı değil). EHT'nin Schwarzschild için "
    "verdiği 37,6 ve 21,3 µas değerleri testte yeniden üretilir.",
)
def shadow_angular_diameter_uas(mass_kg: ArrayLike, distance_m: ArrayLike) -> Array:
    """Dönmeyen kara deliğin gölgesinin gökyüzündeki açısal çapını hesaplar.

    θ_gölge = 2 √27 θ_g,  θ_g = G M / (c² D)

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].
    uzaklik_m : float ya da ndarray
        Gözlemciye uzaklık [m]; D ≫ r_G varsayılır.

    Returns
    -------
    float ya da ndarray
        Açısal çap [µas, mikroyay saniyesi].

    References
    ----------
    [EHT2019] §1 (R_c) ve §7 (θ_g tanımı; oradaki d = α θ_g ışıma halkasının çapıdır, gölgenin değil).

    Validity
    --------
    Dönmeyen kara delik, küçük açı yaklaşımı. Dönme gölge boyutunu yaklaşık %4 değiştirir
    (kaynak, §1); Kerr için α spin ve eğikliğe bağlıdır ve burada hesaplanmaz.

    Verification
    ------------
    tests/unit/test_black_holes.py: M87* için 37,6 µas (yıldız kütlesi) ve 21,3 µas (gaz kütlesi).
    """
    d = _positive(distance_m, "uzaklık")
    return 2.0 * critical_impact_parameter_m(mass_kg) / d * _RAD_UAS


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies' (TeX etiketi schwarzschild-frequencies), x_ms = 6",
    status="read",
)
def schwarzschild_isco_radius_m(mass_kg: ArrayLike) -> Array:
    """Dönmeyen kara deliğin en içteki kararlı dairesel yörüngesinin (ISCO) yarıçapı: r_ISCO = 6 G M / c².

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].

    Returns
    -------
    float ya da ndarray
        r_ISCO [m].

    References
    ----------
    [Abramowicz2013] Schwarzschild için x_ms = r_ms/r_G = 6.

    Verification
    ------------
    tests/unit/test_black_holes.py: Kerr ISCO formülünün a* = 0 değeri; 3 R_S.
    """
    return 6.0 * gravitational_radius_m(mass_kg)


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies', Ω_K² = GM/r³ (TeX etiketi schwarzschild-frequencies), x_ms = 6",
    status="read",
    note="f = Ω_K / (2π) dönüşümü tanımdır.",
)
def isco_orbital_frequency_hz(mass_kg: ArrayLike) -> Array:
    """ISCO'daki Kepler yörünge frekansını hesaplar (dönmeyen kara delik).

    Ω_K² = G M / r³,  r = 6 G M / c²  ⇒  f_yör = c³ / (2π 6^{3/2} G M)

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].

    Returns
    -------
    float ya da ndarray
        Yörünge frekansı [Hz], sonsuzdaki gözlemciye göre.

    References
    ----------
    [Abramowicz2013] Schwarzschild için Ω_K² = GM/r³ ve x_ms = 6.

    Validity
    --------
    r Schwarzschild yarıçap koordinatıdır; Ω_K sonsuzdaki gözlemcinin zamanına göre ölçülür.

    Verification
    ------------
    tests/unit/test_black_holes.py: Ω_K² = GM/r³ ile bağımsız hesap; 1/M ölçeklemesi.
    """
    m = _positive(mass_kg, "kütle")
    r = schwarzschild_isco_radius_m(m)
    return np.sqrt(_G * m / r**3) / (2.0 * np.pi)


# --------------------------------------------------------------------------------------
# Kerr geometrisi
# --------------------------------------------------------------------------------------


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies', TeX etiketi stable-radius-kerr (Bardeen vd. formülü)",
    status="read",
    note="Kaynağın formülü doğru (prograde) yörünge içindir; Z1 ve Z2 a* çiftidir. Her iki işaret de "
    "Abramowicz2013'ün epicyclic-radial denkleminin (ω_r*)² = 0 koşuluyla testte bağımsız doğrulanır.",
)
@source(
    "Reynolds2021",
    where="Bölüm 2: r_isco denklemi ve Z1, Z2 tanımları",
    status="read",
    note="r_isco = (3 + Z2 ∓ [(3 − Z1)(3 + Z1 + 2Z2)]^{1/2}) r_g; ∓ işareti sırasıyla ileri ve geri yönlü "
    "yörüngeler içindir. Aynı ifade özgün kaynakta Denk. 2.21'dir [Bardeen1972].",
)
def kerr_isco_radius_rg(spin: ArrayLike) -> Array:
    """Kerr kara deliğin ISCO yarıçapını hesaplar (Bardeen, Press, Teukolsky formülü).

    r_ISCO / r_G = 3 + Z2 ∓ √[(3 − Z1)(3 + Z1 + 2 Z2)]

    Z1 = 1 + (1 − a*²)^{1/3} [(1 + a*)^{1/3} + (1 − a*)^{1/3}],  Z2 = √(3 a*² + Z1²)

    Eksi işaret doğru (a* ≥ 0), artı işaret ters yörünge (a* < 0) içindir.

    Parameters
    ----------
    spin : float ya da ndarray
        Boyutsuz dönme a_* ∈ [−1, 1]; negatif değer retrograde yörünge.

    Returns
    -------
    float ya da ndarray
        r_ISCO / r_G (boyutsuz): a* = 0 için 6, a* = 1 için 1, a* = −1 için 9.

    References
    ----------
    [Abramowicz2013] stable-radius-kerr; [Reynolds2021] Bölüm 2.

    Validity
    --------
    Kerr metriği, ekvator düzlemi, test parçacığı.

    Verification
    ------------
    tests/unit/test_black_holes.py: sınır değerler (1, 6, 9) ve (ω_r*)² = 0 koşulu;
    tests/property/test_black_holes_property.py: spin ile tekdüzelik.
    """
    a_isaretli = _spin(spin)
    a = np.abs(a_isaretli)
    z1 = 1.0 + np.cbrt(1.0 - a**2) * (np.cbrt(1.0 + a) + np.cbrt(1.0 - a))
    z2 = np.sqrt(3.0 * a**2 + z1**2)
    kok = np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    isaret = np.where(a_isaretli >= 0.0, 1.0, -1.0)
    return 3.0 + z2 - isaret * kok


def _isco_energy(spin: ArrayLike) -> Array:
    """ISCO'daki birim kütle başına enerji, E = √(1 − 2/(3 x_ISCO)); yardımcı."""
    x = kerr_isco_radius_rg(spin)
    return np.sqrt(1.0 - 2.0 / (3.0 * x))


@source(
    "Bardeen1972",
    where="Denk. 2.20: kararlılık koşulu 1 − (E/μ)² ≥ (2/3)(M/r), eşitlik r = r_ms'de",
    status="read",
    note="ADS taramasından okundu. Koşulun r_ms'deki eşitliği Ẽ_ms = (1 − 2/(3 r̃_ms))^{1/2} verir. Testte "
    "Ẽ(r) dairesel yörünge enerjisinin ISCO'da minimum olması (dẼ/dr = 0) ve Schwarzschild değeri √(8/9) ile sınanır.",
)
def kerr_isco_specific_energy(spin: ArrayLike) -> Array:
    """ISCO'daki dairesel yörüngenin birim durgun kütle başına enerjisini hesaplar.

    E_ISCO = √(1 − 2 / (3 x_ISCO)),  x_ISCO = r_ISCO / r_G

    Parameters
    ----------
    spin : float ya da ndarray
        Boyutsuz dönme a_* ∈ [−1, 1]; negatif değer retrograde yörünge.

    Returns
    -------
    float ya da ndarray
        E_ISCO / (m c²): a* = 0 için √(8/9) ≈ 0,9428.

    References
    ----------
    [Bardeen1972] Denk. 2.20.

    Validity
    --------
    Kerr metriği, ekvator düzlemi, test parçacığı.

    Verification
    ------------
    tests/unit/test_black_holes.py: √(8/9) ve bağımsız dairesel yörünge enerjisinin minimumu.
    """
    return _isco_energy(spin)


@source(
    "Abramowicz2013",
    where="§1 (Giriş), verim tanımı η ≡ L/(Ṁ c²)",
    status="read",
    note="Inayoshi2020 dönmeyen kara delik için '~0,06' der (testte 0,057 ile uyuşur).",
)
@source(
    "Reynolds2021",
    where="Bölüm 5: Novikov-Thorne diski için η = 1 − e_isco/c²",
    status="read",
    note="e_isco: ISCO'da dairesel yörüngedeki birim kütlenin toplam enerjisi; e_isco'nun kapalı biçimi "
    "isco_ozgul_enerji_kerr içindedir [Bardeen1972, Denk. 2.20].",
)
def isco_radiative_efficiency(spin: ArrayLike) -> Array:
    """İnce disk için ışınım verimini hesaplar: η = 1 − E_ISCO.

    η ≡ L / (Ṁ c²). Dönmeyen kara delik için 1 − √(8/9) ≈ 0,0572; a* = 1 için ≈ 0,42.

    Parameters
    ----------
    spin : float ya da ndarray
        Boyutsuz dönme a_* ∈ [−1, 1]; negatif değer retrograde disk.

    Returns
    -------
    float ya da ndarray
        Verim η (boyutsuz).

    References
    ----------
    [Abramowicz2013] §1; [Reynolds2021] Bölüm 5.

    Validity
    --------
    Geometrik olarak ince, ışınımca verimli disk; ISCO içinden gelen ışınım ve ufka düşen
    ışığın geri yakalanması yok sayılır.

    Verification
    ------------
    tests/unit/test_black_holes.py: 0,057 (Inayoshi2020 '~0,06'); tests/property/test_black_holes_property.py:
    spinle artış.
    """
    return 1.0 - _isco_energy(spin)


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies', r_H (aynı eqnarray, etiketsiz satır)",
    status="read",
)
@source("Page2005", where="TeX etiketi eq:3, r_+ (Q = 0)", status="read")
def kerr_horizon_radius_rg(spin: ArrayLike) -> Array:
    """Kerr kara deliğin olay ufkunun yarıçapını hesaplar: r_+ = r_G (1 + √(1 − a*²)).

    Parameters
    ----------
    spin : float ya da ndarray
        Boyutsuz dönme a_* ∈ [−1, 1].

    Returns
    -------
    float ya da ndarray
        r_+ / r_G: a* = 0 için 2, a* = ±1 için 1.

    References
    ----------
    [Abramowicz2013] r_H; [Page2005] TeX etiketi eq:3 (yükü sıfır alınarak).

    Verification
    ------------
    tests/unit/test_black_holes.py: a* = 0 ve 1 değerleri; R_S/2r_G ile uyum.
    """
    a = _spin(spin)
    return 1.0 + np.sqrt(1.0 - a**2)


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies', r_0 (ergosfer; aynı eqnarray, etiketsiz satır)",
    status="read",
)
def kerr_ergosphere_radius_rg(spin: ArrayLike, polar_angle_rad: ArrayLike = np.pi / 2) -> Array:
    """Kerr kara deliğin ergosfer yarıçapını hesaplar: r_0 = r_G (1 + √(1 − a*² cos²θ)).

    Parameters
    ----------
    spin : float ya da ndarray
        Boyutsuz dönme a_* ∈ [−1, 1].
    kutup_acisi_rad : float ya da ndarray, isteğe bağlı
        Kutup açısı θ [rad]; varsayılan ekvator (π/2), burada ergosfer en geniştir.

    Returns
    -------
    float ya da ndarray
        r_0 / r_G: ekvatorda 2, kutuplarda ufka teğet (r_+).

    References
    ----------
    [Abramowicz2013] r_0.

    Verification
    ------------
    tests/unit/test_black_holes.py: ekvatorda 2 r_G; kutupta ufuk yarıçapı;
    tests/property/test_black_holes_property.py: ufuk ≤ ergosfer.
    """
    a = _spin(spin)
    teta = np.asarray(polar_angle_rad, dtype=float)
    return 1.0 + np.sqrt(1.0 - (a * np.cos(teta)) ** 2)


@source(
    "Page2005",
    where="TeX etiketi eq:3, Ω = 4π a / A (Q = 0)",
    status="read",
    note="Kaynak G = c = 1 birimindedir; SI'ya Ω_H = c³ Ω̃ / (G M) ile çevrildi.",
)
def horizon_angular_velocity_rad_s(mass_kg: ArrayLike, spin: ArrayLike) -> Array:
    """Kerr ufkunun açısal hızını hesaplar.

    Ω_H = (c³ / G M) · a* / [2 (1 + √(1 − a*²))]

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].
    spin : float ya da ndarray
        Boyutsuz dönme a_* ∈ [−1, 1].

    Returns
    -------
    float ya da ndarray
        Ω_H [rad s^-1]; işareti spinin işaretiyle aynı.

    References
    ----------
    [Page2005] TeX etiketi eq:3 (Ω, yükü sıfır alınarak).

    Verification
    ------------
    tests/unit/test_black_holes.py: a* = 1 için Ω_H = c³/(2 G M); küçük a* için a* c³/(4 G M).
    """
    m = _positive(mass_kg, "kütle")
    a = _spin(spin)
    return _C**3 / (_G * m) * a / (2.0 * (1.0 + np.sqrt(1.0 - a**2)))


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies', TeX etiketi photon-radius-kerr",
    status="read",
    note="Kaynakta yazıldığı gibi 2 r_G{1 + cos[(2/3) arccos(a*)]} formülü a* = 1 için 4 r_G verir, bu retrograde "
    "yörüngedir; doğru yörünge a* = 1'de r_G olmalı. Bu yüzden burada doğru yörünge için arccos(−a*) kullanıldı: "
    "kaynaktaki a*'ın işareti retrograde için pozitif sayılmış görünüyor. Sonuç aynı kaynaktaki "
    "y³ − 3y + 2a* = 0 kübiğinin (y = √(r/r_G)) kökleriyle testte doğrulanır.",
)
def kerr_photon_orbit_radius_rg(spin: ArrayLike) -> Array:
    """Kerr kara deliğin ekvatoral dairesel foton yörüngesinin yarıçapını hesaplar.

    r_ph / r_G = 2 {1 + cos[(2/3) arccos(−a*)]}

    Parameters
    ----------
    spin : float ya da ndarray
        Boyutsuz dönme a_* ∈ [−1, 1]; pozitif değer foton doğru yönde, negatif ters yönde döner.

    Returns
    -------
    float ya da ndarray
        r_ph / r_G: a* = 0 için 3, a* = 1 için 1, a* = −1 için 4.

    References
    ----------
    [Abramowicz2013] TeX etiketi photon-radius-kerr (işaret düzeltmesi not_ alanında).

    Validity
    --------
    Kerr metriği, ekvator düzlemi. Bu, gölgenin kenarı değil, yörüngenin yarıçapıdır.

    Verification
    ------------
    tests/unit/test_black_holes.py: sınır değerler ve y³ − 3y + 2a* = 0 kübiği.
    """
    a = _spin(spin)
    return 2.0 * (1.0 + np.cos((2.0 / 3.0) * np.arccos(-a)))


# --------------------------------------------------------------------------------------
# Hawking termodinamiği
# --------------------------------------------------------------------------------------


@source(
    "Abramowicz2013",
    where="§1 (Giriş), Hawking sıcaklığı dipnotu: T_H = ħc³/(8π G M k_B)",
    status="read",
)
@source(
    "Page2005",
    where="§1; T = κ/(2π) ve TeX etiketi eq:3 (Schwarzschild: κ = 1/(4M), Planck birimlerinde)",
    status="read",
)
def hawking_temperature_k(mass_kg: ArrayLike) -> Array:
    """Dönmeyen kara deliğin Hawking sıcaklığını hesaplar: T_H = ħ c³ / (8π G M k_B).

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].

    Returns
    -------
    float ya da ndarray
        T_H [K].

    References
    ----------
    [Abramowicz2013] §1 dipnotu (açık SI biçimi); [Page2005] T = κ/(2π), κ = 1/(4M).

    Validity
    --------
    Yarı klasik sınır: kütle Planck kütlesinden çok büyük. Arka plan ışınımına göre
    emilim ya da kütle kaybının geri etkisi yok sayılır.

    Verification
    ------------
    tests/unit/test_black_holes.py: yıldız kütleli delik için ~1e-8 K (kaynak dipnotu);
    κ/(2π) ile bağımsız hesap; T·M sabit.
    """
    m = _positive(mass_kg, "kütle")
    return _HBAR * _C**3 / (8.0 * np.pi * _G * m * BOLTZMANN.value)


@source(
    "Page2005",
    where="TeX etiketi eq:2, S_bh = S_BH ≡ A/4 (Planck birimleri); A = 16π M² (Schwarzschild, eq:3 sonrası paragraf)",
    status="read",
    note="Planck birimlerinden SI'ya çeviri (S/k_B = A c³ / (4 G ħ)) tarafımdan yapıldı; kaynakta yazılı değil.",
)
def bekenstein_hawking_entropy(mass_kg: ArrayLike) -> Array:
    """Dönmeyen kara deliğin Bekenstein-Hawking entropisini Boltzmann sabiti biriminde hesaplar.

    S / k_B = A c³ / (4 G ħ) = 4π G M² / (ħ c),  A = 16π G² M² / c⁴

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].

    Returns
    -------
    float ya da ndarray
        S / k_B (boyutsuz). Güneş kütlesi için ~1e77.

    References
    ----------
    [Page2005] TeX etiketi eq:2 (S = A/4) ve Schwarzschild alanı A = 16π M².

    Verification
    ------------
    tests/unit/test_black_holes.py: A/4 ile bağımsız hesap (ufuk alanından); M² ölçeklemesi.
    """
    m = _positive(mass_kg, "kütle")
    alan = 16.0 * np.pi * (_G * m / _C**2) ** 2
    return alan * _C**3 / (4.0 * _G * _HBAR)


@source(
    "Page2005",
    where="TeX etiketi eq:15",
    status="read",
    note="−dM/dt = 1e-5 M⁻² (8,1830 n_1/2 + 3,3638 n_1 + 0,3836 n_2), Planck birimlerinde; n_s iki kutuplanmalı "
    "kütlesiz spin-s tür sayısı. SI'ya çevrim (ħ c⁴/G² çarpanı) bu projede",
)
def hawking_mass_loss_rate_kg_s(
    mass_kg: ArrayLike, *, spin_half: float = 0.0, spin_one: float = 1.0, spin_two: float = 1.0
) -> Array:
    """Dönmeyen kara deliğin Hawking ışınımıyla kütle kaybı hızını hesaplar.

    −dM/dt = α ħ c⁴ / (G² M²),  α = 1e-5 (8,1830 n_½ + 3,3638 n_1 + 0,3836 n_2)

    Katsayılar sayısal greybody hesaplarından gelir. Varsayılan türler foton
    (n_1 = 1) ve gravitondur (n_2 = 1).

    Parameters
    ----------
    kutle_kg : float ya da ndarray
    spin_yarim_tur, spin_bir_tur, spin_iki_tur : float
        İki kutuplanmalı kütlesiz spin-½, spin-1 ve spin-2 tür sayıları.

    Returns
    -------
    float ya da ndarray
        Kütle kaybı hızı −dM/dt [kg s⁻¹], pozitif.

    References
    ----------
    [Page2005] eq:15.

    Validity
    --------
    Yalnızca kütlesiz türler ve dönmeyen, yüksüz kara delik. Kütleli bir tür,
    Hawking sıcaklığı o türün durgun kütle enerjisine yaklaştığında devreye
    girer; elektronlar için bu yaklaşık 1e13-1e14 kg altındadır (mertebe
    kestirimi, k T_H ~ m_e c²). Nötrinoların hangi kütlede "kütlesiz" sayılacağı
    kütlelerine bağlıdır; varsayılanda sayılmazlar.

    Verification
    ------------
    tests/unit/test_black_holes.py: yalnız foton için kara cisim kestiriminin 1,62 katı; M⁻² ölçeklemesi.
    """
    m = _positive(mass_kg, "kütle")
    alpha = 1e-5 * (8.1830 * spin_half + 3.3638 * spin_one + 0.3836 * spin_two)
    return alpha * _HBAR * _C**4 / (_G**2 * m**2)


@source(
    "Page2005",
    where="TeX etiketi eq:15",
    status="read",
    note="eq:15'in M'ye göre integrali (türetim bu projede): t = G² M³ / (3 α ħ c⁴); α sabit tür sayıları için",
)
def hawking_evaporation_time_s(
    mass_kg: ArrayLike, *, spin_half: float = 0.0, spin_one: float = 1.0, spin_two: float = 1.0
) -> Array:
    """Hawking ışınımıyla tam buharlaşma süresini greybody katsayılarıyla hesaplar.

    t = G² M³ / (3 α ħ c⁴),  α = 1e-5 (8,1830 n_½ + 3,3638 n_1 + 0,3836 n_2)

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Başlangıç kütlesi [kg].
    spin_yarim_tur, spin_bir_tur, spin_iki_tur : float
        Kütlesiz tür sayıları; varsayılan foton ve graviton.

    Returns
    -------
    float ya da ndarray
        Buharlaşma süresi [s]. Güneş kütlesi için varsayılan türlerle yaklaşık 1,2e67 yıl.

    References
    ----------
    [Page2005] eq:15.

    Validity
    --------
    Tür sayıları buharlaşma boyunca sabit alınır. Gerçekte kütle azaldıkça
    sıcaklık artar ve kütleli türler devreye girer; bu yüzden küçük başlangıç
    kütlelerinde (yaklaşık 1e14 kg altı) gerçek süre buradakinden kısadır ve
    sonuç yalnızca üst sınır niteliğindedir.

    Verification
    ------------
    tests/unit/test_black_holes.py: ``hawking_kutle_kaybi_hizi_kg_s`` ile sayısal tümlev; M³ ölçeklemesi.
    """
    m = _positive(mass_kg, "kütle")
    alpha = 1e-5 * (8.1830 * spin_half + 3.3638 * spin_one + 0.3836 * spin_two)
    return _G**2 * m**3 / (3.0 * alpha * _HBAR * _C**4)


@source(
    "Page2005",
    where="TeX etiketi eq:3 (A, κ) ve §1 (T = κ/2π)",
    status="read",
    note="L = σ T_H⁴ A kara cisim varsayımı ve t = 5120π G² M³ / (ħ c⁴) integrali bu projenin türetimidir; "
    "kaynak yalnızca T_H ve A'yı verir. Kaynağın eq:15 greybody katsayısı yalnız foton için bu kestirimin 1,62 katıdır",
)
def hawking_evaporation_time_blackbody_s(mass_kg: ArrayLike) -> Array:
    """Buharlaşma süresinin ders kitabı kara cisim kestirimini hesaplar.

    L = σ_SB T_H⁴ A = ħ c⁶ / (15360π G² M²),  t = 5120π G² M³ / (ħ c⁴)

    Returns
    -------
    float ya da ndarray
        Süre [s]. Güneş kütlesi için yaklaşık 2,1e67 yıl.

    References
    ----------
    [Page2005] eq:3, §1.

    Validity
    --------
    Yalnızca foton, greybody çarpanı 1. Bu varsayım ışıma hızını yalnız foton
    için bile 1,62 kat küçük verir ve süreyi olduğundan uzun gösterir;
    karşılaştırma ve öğretim amaçlıdır. Hesap için ``hawking_buharlasma_suresi_s`` kullanılmalıdır.

    Verification
    ------------
    tests/unit/test_black_holes.py: L = σ T⁴ A ile bağımsız tümlev ve M³ ölçeklemesi.
    """
    m = _positive(mass_kg, "kütle")
    return 5120.0 * np.pi * _G**2 * m**3 / (_HBAR * _C**4)


# --------------------------------------------------------------------------------------
# İkili kara delikler
# --------------------------------------------------------------------------------------


@source("Sathyaprakash2009", where="TeX etiketi eqn:chirpmassdef, 𝓜 = μ^{3/5} M^{2/5} = ν^{3/5} M", status="read")
def chirp_mass_kg(mass1_kg: ArrayLike, mass2_kg: ArrayLike) -> Array:
    """İkili sistemin cıvıltı (chirp) kütlesini hesaplar.

    𝓜 = (m1 m2)^{3/5} / (m1 + m2)^{1/5}

    Parameters
    ----------
    kutle1_kg, kutle2_kg : float ya da ndarray
        Bileşen kütleleri [kg], pozitif.

    Returns
    -------
    float ya da ndarray
        Cıvıltı kütlesi [kg]; eşit kütle m için m·2^{-1/5}.

    References
    ----------
    [Sathyaprakash2009] TeX etiketi eqn:chirpmassdef.

    Verification
    ------------
    tests/unit/test_black_holes.py: kaynaktaki μ^{3/5}M^{2/5} ve ν^{3/5}M biçimleriyle uyum;
    tests/property/test_black_holes_property.py: simetri ve üst sınır.
    """
    m1 = _positive(mass1_kg, "kütle 1")
    m2 = _positive(mass2_kg, "kütle 2")
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


@source(
    "Sathyaprakash2009",
    where="TeX etiketi eqn:circularPM (Ṗ_b, Peters-Mathews 1963'e atfen), f_gw = 2/P_b",
    status="read",
    note="Kaynak G = c = 1 birimindedir. f_gw = 2 f_yör (paragraf 'Waveform polarizations.' sonrası metin: ışınım yörünge "
    "frekansının iki katında yayılır); ḟ = −2 Ṗ_b / P_b² dönüşümünü ben yaptım ve testte Ṗ_b denklemiyle sınadım.",
)
def gw_frequency_derivative_hz_s(chirp_mass_kg: ArrayLike, frequency_hz: ArrayLike) -> Array:
    """Dairesel ikilinin kütle çekimsel dalga frekansının zamanla artış hızını hesaplar.

    ḟ = (96/5) π^{8/3} (G 𝓜 / c³)^{5/3} f^{11/3}

    Parameters
    ----------
    cirp_kutlesi_kg : float ya da ndarray
        Cıvıltı kütlesi 𝓜 [kg].
    frekans_hz : float ya da ndarray
        Kütle çekimsel dalga frekansı f [Hz] (yörünge frekansının iki katı).

    Returns
    -------
    float ya da ndarray
        df/dt [Hz s^-1].

    References
    ----------
    [Sathyaprakash2009] TeX etiketi eqn:circularPM.

    Validity
    --------
    Dairesel yörünge, Newton mertebesinde dördül ışınımı; ISCO'ya yaklaşırken
    yüksek mertebe PN düzeltmeleri önemlidir.

    Verification
    ------------
    tests/unit/test_black_holes.py: Ṗ_b denkleminden bağımsız türetim; 𝓜 ve f ölçeklemesi.
    """
    mc = _positive(chirp_mass_kg, "cıvıltı kütlesi")
    f = _positive(frequency_hz, "frekans")
    return 96.0 / 5.0 * np.pi ** (8.0 / 3.0) * (_G * mc / _C**3) ** (5.0 / 3.0) * f ** (11.0 / 3.0)


@source(
    "Sathyaprakash2009",
    where="§'Coalescing binaries' paragrafı, Newton mertebesi cıvıltı süresi τ0 (etiketsiz denklem)",
    status="read",
    note="Kaynaktaki τ0 = 5/(256 π ν f_L) (π M f_L)^{-5/3} (G = c = 1) 𝓜^{5/3} = ν M^{5/3} ile bu biçime çevrildi. "
    "Peters1964'ün kendi metni okunmadı.",
)
def time_to_merger_s(chirp_mass_kg: ArrayLike, frequency_hz: ArrayLike) -> Array:
    """Dalga frekansı f iken dairesel ikilinin birleşmeye kalan süresini hesaplar.

    τ = (5/256) (π f)^{−8/3} (G 𝓜 / c³)^{−5/3}

    Parameters
    ----------
    cirp_kutlesi_kg : float ya da ndarray
        Cıvıltı kütlesi 𝓜 [kg].
    frekans_hz : float ya da ndarray
        Kütle çekimsel dalga frekansı f [Hz].

    Returns
    -------
    float ya da ndarray
        Kalan süre [s]: frekansın biçimsel olarak ıraksadığı ana kadar.

    References
    ----------
    [Sathyaprakash2009] Newton mertebesi τ0 (etiketsiz denklem, 'Coalescing binaries').

    Validity
    --------
    Newton mertebesi, dairesel yörünge; 10 M☉ ve üzeri sistemlerde ISCO'ya yakın
    frekanslarda PN düzeltmeleri süreyi değiştirir.

    Verification
    ------------
    tests/unit/test_black_holes.py: ḟ'nin ters tümleviyle ve kaynağın ν biçimiyle uyum.
    """
    mc = _positive(chirp_mass_kg, "cıvıltı kütlesi")
    f = _positive(frequency_hz, "frekans")
    return 5.0 / 256.0 * (np.pi * f) ** (-8.0 / 3.0) * (_G * mc / _C**3) ** (-5.0 / 3.0)


@source(
    "Sathyaprakash2009",
    where="§'Coalescing binaries' paragrafı, τ0 (etiketsiz denklem); tersi tarafımdan alındı",
    status="read",
    note="τ0 denklemi okundu; f(τ) onun cebirsel tersidir ve testte gidiş-dönüşle sınanır.",
)
def gw_frequency_from_time_to_merger_hz(chirp_mass_kg: ArrayLike, remaining_time_s: ArrayLike) -> Array:
    """Birleşmeye kalan süre τ iken dalga frekansını hesaplar.

    f = (1/π) (5 / 256 τ)^{3/8} (G 𝓜 / c³)^{−5/8}

    Parameters
    ----------
    cirp_kutlesi_kg : float ya da ndarray
        Cıvıltı kütlesi 𝓜 [kg].
    kalan_sure_s : float ya da ndarray
        Birleşmeye kalan süre [s], pozitif.

    Returns
    -------
    float ya da ndarray
        Kütle çekimsel dalga frekansı [Hz].

    References
    ----------
    [Sathyaprakash2009] Newton mertebesi τ0'ın tersi.

    Validity
    --------
    Newton mertebesi, dairesel yörünge; ISCO frekansını aşan değerler fiziksel değildir.

    Verification
    ------------
    tests/unit/test_black_holes.py: birlesmeye_kalan_sure_s ile gidiş-dönüş;
    tests/property/test_black_holes_property.py: τ azaldıkça f artar.
    """
    mc = _positive(chirp_mass_kg, "cıvıltı kütlesi")
    tau = _positive(remaining_time_s, "kalan süre")
    return (5.0 / (256.0 * tau)) ** (3.0 / 8.0) * (_G * mc / _C**3) ** (-5.0 / 8.0) / np.pi


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies' (TeX etiketi schwarzschild-frequencies), Ω_K² = GM/r³, x_ms = 6",
    status="read",
)
@source(
    "Sathyaprakash2009",
    where="paragraf 'Waveform polarizations.' sonrası metin: ışınım en düşük mertebede yörünge frekansının iki katında yayılır",
    status="read",
    note="Yörünge frekansı için test parçacığı ISCO'su (6 G M/c²) toplam kütleye uygulanır; eş kütleli ikili için "
    "bu bir sezgisel sınırdır, kaynakta ikili ISCO'su verilmez.",
)
def gw_frequency_at_isco_hz(total_mass_kg: ArrayLike) -> Array:
    """İkili ISCO'sundaki kütle çekimsel dalga frekansını hesaplar (test parçacığı sınırı).

    f_gw = 2 f_yör = c³ / (6^{3/2} π G M)

    Parameters
    ----------
    toplam_kutle_kg : float ya da ndarray
        İkilinin toplam kütlesi M [kg].

    Returns
    -------
    float ya da ndarray
        f_gw [Hz]; 1 M☉ için yaklaşık 4,4 kHz, M ile ters orantılı.

    References
    ----------
    [Abramowicz2013] Ω_K² = GM/r³ ve x_ms = 6; [Sathyaprakash2009] f_gw = 2 f_yör.

    Validity
    --------
    Test parçacığı (ν → 0) sınırı; eş kütleli ikilide gerçek birleşme frekansı bundan farklıdır.

    Verification
    ------------
    tests/unit/test_black_holes.py: isco_yorunge_frekansi_hz'in iki katı; 4,4 kHz (M☉/M).
    """
    return 2.0 * isco_orbital_frequency_hz(total_mass_kg)
