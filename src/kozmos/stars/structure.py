"""Politroplar: Lane-Emden denklemi ve politrop kürelerin yapı nicelikleri.

Politrop hâl denklemi P = K ρ^{1+1/n} ve Lane-Emden denklemi [Liu1995]
(arXiv:astro-ph/9512061) metninden, ξ_1 ve ρ_c/ρ̄ değerleri ile n = 0, 1, 5 için
kapalı çözümler [Bludman2010] (arXiv:1008.4281) metninden 25 Kasım 2026'da
okunmuştur. Kaynakların numaralı denklemleri ``Denk.`` ile, numarasız olanlar
bölüm ya da tablo adıyla anılır.

Boyutsuz değişkenler [Liu1995] Denk. 3 ile aynıdır: ξ = A r,
A² = 4πG ρ_c^{(n-1)/n} / ((n+1) K) ve θ = (ρ/ρ_c)^{1/n}. Bu modüldeki
fonksiyonlar SI birimlerindedir; n dizi değil tek sayıdır, çünkü her n için ayrı
bir diferansiyel denklem çözülür (sonuç önbelleğe alınır).

Ortalama molekül ağırlığı μ bu dosyada proton kütlesiyle çarpılır; [Liu1995] Denk. 42 atom hidrojeninin
kütlesini kullanır (fark %0,05). ``olcekleme`` modülündeki Eddington fonksiyonları ise kaynakları
[Bludman2010] gibi atomik kütle birimini kullanır; proton kütlesiyle farkı %0,73'tür. Aynı μ ile iki
modülün sonuçları bu kadar ayrışır (Güneş, n = 3, μ = 0,62: merkez sıcaklığı 1,2241e7 ve 1,2151e7 K).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from kozmos._sources import source
from kozmos.core.constants import BOLTZMANN, GRAVITATIONAL_CONSTANT, PROTON_MASS

Array = NDArray[np.float64]

# Merkezden ilk adım: Taylor serisi [Liu1995] Denk. 5, [Bludman2010] Tablo II.
_XI_START = 1e-5
_XI_END = 1e5


def _valid_n(n: float) -> float:
    n = float(n)
    if not 0.0 <= n < 5.0:
        raise ValueError("politrop indisi 0 <= n < 5 olmalı (n >= 5 için yarıçap sonsuzdur)")
    return n


@lru_cache(maxsize=64)
def _solve(n: float):  # type: ignore[no-untyped-def]
    """Lane-Emden denklemini ilk sıfıra kadar yüksek doğrulukla çözer; (çözüm, ξ_1, θ'(ξ_1))."""

    def sag(x, y):  # type: ignore[no-untyped-def]
        return [y[1], -(max(y[0], 0.0) ** n) - 2.0 * y[1] / x]

    def sifir(x, y):  # type: ignore[no-untyped-def]
        return y[0]

    sifir.terminal = True  # type: ignore[attr-defined]
    sifir.direction = -1  # type: ignore[attr-defined]
    x0 = _XI_START
    y0 = [1.0 - x0**2 / 6.0 + n * x0**4 / 120.0, -x0 / 3.0 + n * x0**3 / 30.0]
    c = solve_ivp(sag, (x0, _XI_END), y0, method="DOP853", rtol=1e-12, atol=1e-14, events=sifir, dense_output=True)
    if c.t_events[0].size == 0:
        raise ValueError(f"n = {n} için θ ilk sıfırına ulaşmadı")
    return c.sol, float(c.t_events[0][0]), float(c.y_events[0][0][1])


@source("Liu1995", where="Denk. 2, 4, 5 (Lane-Emden denklemi, merkez koşulları, kuvvet serisi)", status="read")
@source("Bludman2010", where="Tablo II (Taylor serisi 1 - ξ²/6 + n ξ⁴/120 ...)", status="read")
def lane_emden_solve(n: float, n_points: int = 400) -> tuple[Array, Array, Array]:
    """Lane-Emden denklemini merkezden ilk sıfıra kadar çözer.

    θ'' + (2/ξ) θ' = -θ^n,   θ(0) = 1,  θ'(0) = 0

    Parameters
    ----------
    n : float
        Politrop indisi, 0 <= n < 5.
    nokta_sayisi : int
        [0, ξ_1] üzerindeki eşit aralıklı çıktı noktası sayısı (en az 2).

    Returns
    -------
    xi, theta, dtheta_dxi : ndarray
        Boyutsuz yarıçap, θ ve türevi. Son nokta θ = 0 yüzeyidir.

    References
    ----------
    [Liu1995] Denk. 2, 4, 5; [Bludman2010] Tablo II.

    Validity
    --------
    n >= 5 için θ sıfıra ulaşmaz (n = 5'te sonsuz yarıçap). 0 < n < 1 için θ^n türevi
    yüzeyde sonsuzdur; ξ_1 yine de bulunur (n = 0,5 için bağımsız bir çözücünün değeriyle 1e-8 içinde uyuşur).

    Verification
    ------------
    tests/unit/test_stars.py: n = 0, 1, 5 kapalı çözümleri ve [Bludman2010] Tablo I ξ_1 değerleri.
    """
    if n_points < 2:
        raise ValueError("nokta_sayisi en az 2 olmalı")
    n = _valid_n(n)
    cozum, xi1, _ = _solve(n)
    xi = np.linspace(0.0, xi1, int(n_points))
    theta = np.ones_like(xi)
    dtheta = np.zeros_like(xi)
    iceride = xi >= _XI_START
    y = cozum(xi[iceride])
    theta[iceride], dtheta[iceride] = y[0], y[1]
    kucuk = ~iceride
    theta[kucuk] = 1.0 - xi[kucuk] ** 2 / 6.0 + n * xi[kucuk] ** 4 / 120.0
    dtheta[kucuk] = -xi[kucuk] / 3.0 + n * xi[kucuk] ** 3 / 30.0
    return xi, theta, dtheta


@source("Bludman2010", where="Tablo I, ξ_1n sütunu", status="read")
@source("Liu1995", where="Denk. 2 (θ'nın ilk sıfırı yıldızın yüzeyidir)", status="read")
def lane_emden_first_zero(n: float) -> float:
    """Lane-Emden fonksiyonunun ilk sıfırını bulur: θ(ξ_1) = 0.

    Parameters
    ----------
    n : float
        Politrop indisi, 0 <= n < 5.

    Returns
    -------
    float
        ξ_1, yıldızın boyutsuz yarıçapı (R = ξ_1 / A).

    References
    ----------
    [Bludman2010] Tablo I; [Liu1995] Denk. 2.

    Validity
    --------
    n >= 5 için ξ_1 sonsuzdur; ``ValueError`` verilir.

    Verification
    ------------
    tests/unit/test_stars.py: n = 0, 1, 1,5, 2, 3, 4, 4,5 için Tablo I değerleri (dört anlamlı basamak).
    """
    return _solve(_valid_n(n))[1]


@source("Liu1995", where="Denk. 38 (m = 4π ρ_c r³ (-θ'/ξ))", status="read")
@source(
    "Bludman2010",
    where="Tablo I (0ω_3 = 2,018) ve Sec. II, M = 4π 0ω_3 (K/πG)^{3/2}",
    status="read",
    note="Tablodaki 0ω_n sütunu yalnızca n = 3 için -ξ_1² θ'(ξ_1) ile aynıdır; diğer n için farklı normalleştirme kullanır",
)
def lane_emden_mass_coefficient(n: float) -> float:
    """Boyutsuz toplam kütleyi hesaplar: ω_n = -ξ_1² θ'(ξ_1).

    M = 4π ρ_c (R/ξ_1)³ ω_n   ([Liu1995] Denk. 38, r = R'de)

    Parameters
    ----------
    n : float
        Politrop indisi, 0 <= n < 5.

    Returns
    -------
    float
        ω_n; n = 3 için 2,018.

    References
    ----------
    [Liu1995] Denk. 38; [Bludman2010] Tablo I ve Sec. II.

    Verification
    ------------
    tests/unit/test_stars.py: n = 3 için 2,018 ve ξ_1³ / (3 ρ_c/ρ̄) özdeşliği.
    """
    _, xi1, dtheta1 = _solve(_valid_n(n))
    return -(xi1**2) * dtheta1


@source("Bludman2010", where="Tablo I, ρ_cn/ρ̄_n sütunu; ρ̄_n = 3M/4πR³ tanımı", status="read")
@source("Liu1995", where="Denk. 38 (r = R'de m = M)", status="read")
def lane_emden_central_to_mean_density(n: float) -> float:
    """Merkez yoğunluğun ortalama yoğunluğa oranını hesaplar.

    ρ_c / ρ̄ = ξ_1 / (-3 θ'(ξ_1))

    Parameters
    ----------
    n : float
        Politrop indisi, 0 <= n < 5.

    Returns
    -------
    float
        ρ_c/ρ̄ >= 1; n = 0'da 1, n = 3'te 54,18.

    References
    ----------
    [Bludman2010] Tablo I; [Liu1995] Denk. 38.

    Verification
    ------------
    tests/unit/test_stars.py: n = 0 (1), n = 1 (π²/3) ve Tablo I değerleri.
    """
    _, xi1, dtheta1 = _solve(_valid_n(n))
    return xi1 / (-3.0 * dtheta1)


@source("Bludman2010", where="Tablo II (n = 0, 1, 5 satırları)", status="read")
def lane_emden_analytic(xi: ArrayLike, n: int) -> Array:
    """Lane-Emden denkleminin kapalı çözümlerini hesaplar: n = 0, 1, 5.

    n = 0:  θ = 1 - ξ²/6
    n = 1:  θ = sin ξ / ξ
    n = 5:  θ = (1 + ξ²/3)^{-1/2}

    Parameters
    ----------
    xi : float ya da ndarray
        Boyutsuz yarıçap, ξ >= 0.
    n : {0, 1, 5}
        Politrop indisi.

    Returns
    -------
    float ya da ndarray
        θ(ξ). n = 0 ve 1 için ξ_1'den sonrası fiziksel değildir (θ < 0).

    References
    ----------
    [Bludman2010] Tablo II.

    Verification
    ------------
    tests/unit/test_stars.py: n = 0 ve n = 1 için sayısal çözümle 1e-9 içinde uyum; n = 5 için denklemin kalıntısı
    (n = 5 sayısal olarak çözülmez: yarıçap sonsuzdur).
    """
    x = np.asarray(xi, dtype=float)
    if np.any(x < 0):
        raise ValueError("ξ eksi olamaz")
    if n == 0:
        return 1.0 - x**2 / 6.0
    if n == 1:
        return np.sinc(x / np.pi)
    if n == 5:
        return (1.0 + x**2 / 3.0) ** -0.5
    raise ValueError("kapalı çözüm yalnızca n = 0, 1, 5 için vardır")


@source("Liu1995", where="Denk. 38 (r = R'de), ρ̄ tanımı için [Bludman2010] Tablo I", status="read")
def polytrope_central_density_kg_m3(mass_kg: ArrayLike, radius_m: ArrayLike, n: float) -> Array:
    """Verilen kütle ve yarıçapa sahip politropun merkez yoğunluğunu hesaplar.

    ρ_c = (ρ_c/ρ̄) · 3M / (4π R³)

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Toplam kütle [kg].
    yaricap_m : float ya da ndarray
        Yarıçap [m].
    n : float
        Politrop indisi, 0 <= n < 5.

    Returns
    -------
    float ya da ndarray
        Merkez yoğunluk [kg m^-3].

    References
    ----------
    [Liu1995] Denk. 38.

    Verification
    ------------
    tests/unit/test_stars.py: n = 3 Güneş modeli için 76,39 g cm^-3 ([Liu1995] sayısal değeri).
    """
    m = np.asarray(mass_kg, dtype=float)
    r = np.asarray(radius_m, dtype=float)
    return lane_emden_central_to_mean_density(n) * 3.0 * m / (4.0 * np.pi * r**3)


@source("Liu1995", where="Denk. 3 (A² tanımı, A = ξ_1/R)", status="read")
def polytrope_k_coefficient_si(mass_kg: ArrayLike, radius_m: ArrayLike, n: float) -> Array:
    """Verilen kütle ve yarıçaptan politrop sabitini hesaplar: P = K ρ^{1+1/n}.

    K = 4π G ρ_c^{(n-1)/n} R² / ((n+1) ξ_1²)

    Parameters
    ----------
    kutle_kg, yaricap_m : float ya da ndarray
        Kütle [kg] ve yarıçap [m].
    n : float
        Politrop indisi, 0 < n < 5 (n = 0'da K tanımsızdır).

    Returns
    -------
    float ya da ndarray
        K [Pa m^{3(1+1/n)} kg^{-(1+1/n)}], yani P'yi Pa, ρ'yu kg m^-3 verir.

    References
    ----------
    [Liu1995] Denk. 3.

    Validity
    --------
    Sabit K'yi kütle ve yarıçap birlikte belirler; n = 3'te K yalnızca kütleye bağlıdır.

    Verification
    ------------
    tests/unit/test_stars.py: n = 3 Güneş modeli için 3,84e14 dyn cm^-2 g^{-4/3} ([Araujo2011]).
    """
    n = _valid_n(n)
    if n == 0.0:
        raise ValueError("n = 0 için K tanımsızdır (hâl denklemi P = K ρ^∞)")
    xi1 = lane_emden_first_zero(n)
    r = np.asarray(radius_m, dtype=float)
    rho_c = polytrope_central_density_kg_m3(mass_kg, radius_m, n)
    return 4.0 * np.pi * GRAVITATIONAL_CONSTANT.value * rho_c ** ((n - 1.0) / n) * r**2 / ((n + 1.0) * xi1**2)


@source(
    "Liu1995",
    where="Denk. 41 (P = K ρ_c^{(n+1)/n} ω^{n+1}, merkezde ω = 1) ve Denk. 3",
    status="read",
    note="Denk. 41 ile Denk. 3 birleştirilince P_c = 4π G R² ρ_c² / ((n+1) ξ_1²); n = 0'ı da kapsar",
)
def polytrope_central_pressure_pa(mass_kg: ArrayLike, radius_m: ArrayLike, n: float) -> Array:
    """Verilen kütle ve yarıçaptaki politropun merkez basıncını hesaplar.

    P_c = K ρ_c^{(n+1)/n} = 4π G R² ρ_c² / ((n+1) ξ_1²)

    Parameters
    ----------
    kutle_kg, yaricap_m : float ya da ndarray
        Kütle [kg] ve yarıçap [m].
    n : float
        Politrop indisi, 0 <= n < 5.

    Returns
    -------
    float ya da ndarray
        Merkez basıncı [Pa].

    References
    ----------
    [Liu1995] Denk. 41 ve Denk. 3.

    Verification
    ------------
    tests/unit/test_stars.py: n = 3 Güneş için 1,24e17 dyn cm^-2 ([Liu1995]'in aktardığı sayısal değer);
    n = 0 için düzgün küre kapalı çözümü (2π/3) G ρ² R².
    """
    n = _valid_n(n)
    xi1 = lane_emden_first_zero(n)
    r = np.asarray(radius_m, dtype=float)
    rho_c = polytrope_central_density_kg_m3(mass_kg, radius_m, n)
    return 4.0 * np.pi * GRAVITATIONAL_CONSTANT.value * r**2 * rho_c**2 / ((n + 1.0) * xi1**2)


@source("Liu1995", where="Denk. 3 ve Denk. 38", status="read")
@source(
    "Bludman2010",
    where="Tablo I, son sütun (M ~ R^{(n-3)/(n-1)})",
    status="read",
    note="üs, tablodaki 1 - ω̃_n ile n = 0, 1,5, 2, 4 için tek tek sınandı",
)
def polytrope_radius_m(mass_kg: ArrayLike, k_si: float, n: float) -> Array:
    """Verilen kütle ve politrop sabiti için yıldızın yarıçapını çözer.

    M = 4π (ω_n/ξ_1³) B^{n/(n-1)} R^{(n-3)/(n-1)},   B = (n+1) K ξ_1² / (4π G)

    Parameters
    ----------
    kutle_kg : float ya da ndarray
        Toplam kütle [kg].
    k_si : float
        Politrop sabiti K (``politrop_k_katsayisi_si`` ile aynı birim).
    n : float
        Politrop indisi, 0 < n < 5 ve n ∉ {1, 3} (n = 1 ve 3'te yarıçap kütleden bağımsızdır).

    Returns
    -------
    float ya da ndarray
        Yarıçap [m]. n = 3/2 için R ∝ M^{-1/3} (göreli olmayan dejenere gaz).

    References
    ----------
    [Liu1995] Denk. 3 ve 38; [Bludman2010] Tablo I.

    Verification
    ------------
    tests/unit/test_stars.py: K'dan yarıçapa gidiş-dönüş; n = 3/2'de kütle sekiz kat artınca yarıçap yarıya iner.
    """
    n = _valid_n(n)
    if n <= 0.0 or n in (1.0, 3.0):
        raise ValueError("yarıçap bu yolla yalnızca 0 < n < 5, n ∉ {1, 3} için bulunur")
    if k_si <= 0:
        raise ValueError("K pozitif olmalı")
    xi1 = lane_emden_first_zero(n)
    omega = lane_emden_mass_coefficient(n)
    b = (n + 1.0) * k_si * xi1**2 / (4.0 * np.pi * GRAVITATIONAL_CONSTANT.value)
    m = np.asarray(mass_kg, dtype=float)
    ic = m * xi1**3 / (4.0 * np.pi * omega * b ** (n / (n - 1.0)))
    return ic ** ((n - 1.0) / (n - 3.0))


@source("Liu1995", where="Denk. 42 (T = μ m_p K ρ_c^{1/n} ω / k, merkezde ω = 1)", status="read")
def polytrope_central_temperature_k(
    mass_kg: ArrayLike, radius_m: ArrayLike, n: float, mean_molecular_weight: float
) -> Array:
    """İdeal gaz politropunun merkez sıcaklığını hesaplar.

    T_c = μ m_p P_c / (k ρ_c)

    Parameters
    ----------
    kutle_kg, yaricap_m : float ya da ndarray
        Kütle [kg] ve yarıçap [m].
    n : float
        Politrop indisi, 0 <= n < 5.
    ortalama_molekul_agirligi : float
        μ, parçacık başına ortalama kütle (m_p biriminde).

    Returns
    -------
    float ya da ndarray
        Merkez sıcaklığı [K].

    References
    ----------
    [Liu1995] Denk. 42.

    Validity
    --------
    Yalnızca radyasyon basıncı ve dejenerelik yoksayılabiliyorsa ideal gaz hâl denklemi geçerlidir.

    Verification
    ------------
    tests/unit/test_stars.py: n = 3, μ = 0,62 Güneş için 1,2e7 K ([Liu1995] metni).
    """
    pc = polytrope_central_pressure_pa(mass_kg, radius_m, n)
    rho_c = polytrope_central_density_kg_m3(mass_kg, radius_m, n)
    return mean_molecular_weight * PROTON_MASS.value * pc / (BOLTZMANN.value * rho_c)


@source(
    "Liu1995",
    where="Denk. 44 ve 45 (E_g; yüzeyde Φ = 0 ve P = 0)",
    status="read",
    note="Denk. 44'te r = R, m = M alınır; ilk terim kalır: -3/(5-n) G M²/R",
)
def polytrope_gravitational_energy_j(mass_kg: ArrayLike, radius_m: ArrayLike, n: float) -> Array:
    """Politropun kendi çekim bağlanma enerjisini hesaplar.

    E_g = -3/(5-n) · G M² / R

    Parameters
    ----------
    kutle_kg, yaricap_m : float ya da ndarray
        Kütle [kg] ve yarıçap [m].
    n : float
        Politrop indisi, 0 <= n < 5.

    Returns
    -------
    float ya da ndarray
        E_g [J], eksi işaretli.

    References
    ----------
    [Liu1995] Denk. 44, 45.

    Verification
    ------------
    tests/unit/test_stars.py: n = 0 için -3/5, n = 1 için -3/4 GM²/R (kapalı çözümler).
    """
    n = _valid_n(n)
    m = np.asarray(mass_kg, dtype=float)
    r = np.asarray(radius_m, dtype=float)
    return -3.0 / (5.0 - n) * GRAVITATIONAL_CONSTANT.value * m**2 / r


@source("McKee2007", where="TeX etiketleri eq:iddot, eq:calt (viriyal teoremi)", status="read")
@source(
    "Liu1995",
    where="Denk. 44 ve 45",
    status="read",
    note="Denk. 44 üç terimlidir; yüzeyde P = 0 ve Denk. 45 ile E_g = -3/(5-n) G M²/R olur. Durağan, yüzey terimsiz, manyetik alansız küre için 2T + W = 0; T = (3/2)(k/μ m_p) M T̄ kütle ağırlıklı ortalamadır",
)
def virial_mean_temperature_k(
    mass_kg: ArrayLike, radius_m: ArrayLike, mean_molecular_weight: float, n: float = 3.0
) -> Array:
    """Viriyal teoreminden kütle ağırlıklı ortalama sıcaklığı hesaplar.

    2T + W = 0,  T = (3/2) (k / μ m_p) M T̄,  W = -3/(5-n) G M²/R
    ⇒  T̄ = G M μ m_p / ((5-n) k R)

    Parameters
    ----------
    kutle_kg, yaricap_m : float ya da ndarray
        Kütle [kg] ve yarıçap [m].
    ortalama_molekul_agirligi : float
        μ (m_p biriminde).
    n : float
        W'nin hesaplandığı politrop indisi; varsayılan 3 (Eddington standart modeli).

    Returns
    -------
    float ya da ndarray
        Kütle ağırlıklı ortalama sıcaklık [K].

    References
    ----------
    [McKee2007] viriyal teoremi (eq:iddot, eq:calt); [Liu1995] Denk. 44.

    Validity
    --------
    Radyasyon basıncı yoksayılır; ideal gaz ve durağan denge varsayılır.

    Verification
    ------------
    tests/unit/test_stars.py: 2T + W = 0 özdeşliği; n = 3 Güneş için merkez sıcaklığının altında kalması.
    """
    n = _valid_n(n)
    m = np.asarray(mass_kg, dtype=float)
    r = np.asarray(radius_m, dtype=float)
    return (
        GRAVITATIONAL_CONSTANT.value * m * mean_molecular_weight * PROTON_MASS.value / ((5.0 - n) * BOLTZMANN.value * r)
    )
