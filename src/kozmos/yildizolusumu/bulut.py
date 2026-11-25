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

from kozmos._kaynak import kaynak
from kozmos.temel.sabitler import BOLTZMANN, CEKIM, PROTON_KUTLESI

Dizi = NDArray[np.float64]

# Hidrojen atomunun kütlesi m_H yerine proton kütlesi kullanılır; fark yaklaşık
# %0,15'tir (elektron kütlesi eksi bağlanma enerjisi), kaynakların belirttiği
# μ değerlerinin belirsizliğinin çok altındadır.
_M_H = PROTON_KUTLESI.deger

# Ortalama molekül ağırlığı: H2 ve He (10 H başına 1 He) karışımı, PARÇACIK başına [Krumholz2015].
# Yoğunluk fonksiyonlarının varsayılanı budur: n, bütün parçacıkların (H2 + He) sayı yoğunluğudur.
ORTALAMA_MOLEKUL_AGIRLIGI = 2.33
# n yalnızca H2 moleküllerinin sayı yoğunluğuysa (gözlemlerde yaygın olan n_H2) molekül başına kütle 2,8 m_H,
# n hidrojen çekirdeklerinin yoğunluğuysa (n_H) çekirdek başına kütle 1,4 m_H olur. Aynı helyum bolluğundan
# türetilir (2,33 · 1,2 = 2,8; 2,8 / 2 = 1,4); n_H için 1,4 değeri [McKee2007] eq:tff sayısal örneğinde geçer.
H2_BASINA_MOLEKUL_AGIRLIGI = 2.8
H_BASINA_MOLEKUL_AGIRLIGI = 1.4


def _pozitif(**degerler: ArrayLike) -> list[Dizi]:
    """Bağımsız değişkenlerin kesin pozitif olduğunu sınar; dizilere çevirip döndürür."""
    cikti = []
    for ad, d in degerler.items():
        a = np.asarray(d, dtype=float)
        if np.any(~(a > 0.0)):
            raise ValueError(f"{ad} kesin pozitif olmalı")
        cikti.append(a)
    return cikti


@kaynak(
    "Krumholz2015",
    yer="Bölüm 4 (Gas Flows and Turbulence), Reynolds ve Mach sayıları alt bölümündeki c_s ifadesi",
    teyit="okundu",
    not_="Kaynak μ = 2,33 alır ve 10 K için 0,18 km/s verir; μ bağımsız değişkendir",
)
def ses_hizi_m_s(sicaklik_k: ArrayLike, ortalama_molekul_agirligi: float = ORTALAMA_MOLEKUL_AGIRLIGI) -> Dizi:
    """Eşısıl ses hızını hesaplar: c_s = sqrt(k_B T / (μ m_H)).

    Parametreler
    ------------
    sicaklik_k : float ya da ndarray
        Gaz sıcaklığı [K], pozitif.
    ortalama_molekul_agirligi : float
        Parçacık başına ortalama kütle, m_H biriminde; H2 ve He için 2,33.

    Döndürür
    --------
    float ya da ndarray
        Ses hızı [m/s].

    Kaynaklar
    ---------
    [Krumholz2015] Bölüm 4.

    Geçerlilik
    ----------
    Eşısıl (γ = 1) gaz; yoğunluk arttıkça gaz adyabatik olur ve geçersizleşir.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: 10 K için 0,18 km/s (kaynaktaki değer).
    """
    (t,) = _pozitif(sicaklik_k=sicaklik_k)
    (mu,) = _pozitif(ortalama_molekul_agirligi=ortalama_molekul_agirligi)
    return np.sqrt(BOLTZMANN.deger * t / (mu * _M_H))


@kaynak(
    "Krumholz2015",
    yer="Bölüm 4 (Gas Flows and Turbulence), Reynolds ve Mach sayıları alt bölümü: M ~ V/c_s",
    teyit="okundu",
)
def mach_sayisi(hiz_m_s: ArrayLike, ses_hizi_m_s: ArrayLike) -> Dizi:
    """Mach sayısını hesaplar: M = V / c_s.

    M ≫ 1 ise basınç terimi akışın davranışında önemsizdir; kaynak, moleküler
    bulut için M ~ 20 bulur. Isıl ve türbülanslı desteğin göreli ağırlığını verir.

    Parametreler
    ------------
    hiz_m_s : float ya da ndarray
        Tipik akış hızı ya da türbülans hız dağılımı [m/s], eksi olmayan.
    ses_hizi_m_s : float ya da ndarray
        Ses hızı [m/s], pozitif.

    Döndürür
    --------
    float ya da ndarray
        Boyutsuz Mach sayısı.

    Kaynaklar
    ---------
    [Krumholz2015] Bölüm 4.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: ses hızına eşit hız için M = 1.
    """
    v = np.asarray(hiz_m_s, dtype=float)
    if np.any(v < 0):
        raise ValueError("hız eksi olamaz")
    (cs,) = _pozitif(ses_hizi=ses_hizi_m_s)
    return v / cs


@kaynak(
    "Krumholz2015",
    yer="Bölüm 8 (Giant Molecular Clouds), Scaling Relations: n = 3M / (4π R³ μ m_H)",
    teyit="okundu",
    not_="Kaynak μ = 2,3 (H2 molekülü başına) kullanır; burada μ bağımsız değişkendir",
)
def kutle_yogunlugu_kg_m3(
    sayi_yogunlugu_m3: ArrayLike, ortalama_molekul_agirligi: float = ORTALAMA_MOLEKUL_AGIRLIGI
) -> Dizi:
    """Sayı yoğunluğundan kütle yoğunluğunu hesaplar: ρ = μ m_H n.

    Parametreler
    ------------
    sayi_yogunlugu_m3 : float ya da ndarray
        Parçacık sayı yoğunluğu [m⁻³], eksi olmayan.
    ortalama_molekul_agirligi : float
        Parçacık başına ortalama kütle, m_H biriminde.

    Döndürür
    --------
    float ya da ndarray
        Kütle yoğunluğu [kg/m³].

    Kaynaklar
    ---------
    [Krumholz2015] Bölüm 8.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: sayı yoğunluğu ile gidiş-dönüş.
    """
    n = np.asarray(sayi_yogunlugu_m3, dtype=float)
    if np.any(n < 0):
        raise ValueError("sayı yoğunluğu eksi olamaz")
    (mu,) = _pozitif(ortalama_molekul_agirligi=ortalama_molekul_agirligi)
    return n * mu * _M_H


@kaynak(
    "Krumholz2015",
    yer="Bölüm 8 (Giant Molecular Clouds), Scaling Relations: n = 3M / (4π R³ μ m_H)",
    teyit="okundu",
)
def sayi_yogunlugu_m3(
    kutle_yogunlugu_kg_m3: ArrayLike, ortalama_molekul_agirligi: float = ORTALAMA_MOLEKUL_AGIRLIGI
) -> Dizi:
    """Kütle yoğunluğundan sayı yoğunluğunu hesaplar: n = ρ / (μ m_H).

    Parametreler
    ------------
    kutle_yogunlugu_kg_m3 : float ya da ndarray
        Kütle yoğunluğu [kg/m³], eksi olmayan.
    ortalama_molekul_agirligi : float
        Parçacık başına ortalama kütle, m_H biriminde.

    Döndürür
    --------
    float ya da ndarray
        Sayı yoğunluğu [m⁻³].

    Kaynaklar
    ---------
    [Krumholz2015] Bölüm 8.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: ``kutle_yogunlugu_kg_m3`` ile gidiş-dönüş.
    """
    rho = np.asarray(kutle_yogunlugu_kg_m3, dtype=float)
    if np.any(rho < 0):
        raise ValueError("yoğunluk eksi olamaz")
    (mu,) = _pozitif(ortalama_molekul_agirligi=ortalama_molekul_agirligi)
    return rho / (mu * _M_H)


@kaynak(
    "Krumholz2015",
    yer="Bölüm 8 (Giant Molecular Clouds), Mass Measurement, Optically Thick Lines: N = M / (μ m_H π R²)",
    teyit="okundu",
    not_="Kaynak N ile Σ arasındaki bağıntıyı ortalama sütun yoğunluğu için yazar; Σ = μ m_H N yeniden düzenlemedir",
)
def yuzey_yogunlugu_kg_m2(
    sutun_yogunlugu_m2: ArrayLike, ortalama_molekul_agirligi: float = ORTALAMA_MOLEKUL_AGIRLIGI
) -> Dizi:
    """Sütun yoğunluğundan yüzey yoğunluğunu hesaplar: Σ = μ m_H N.

    Parametreler
    ------------
    sutun_yogunlugu_m2 : float ya da ndarray
        Parçacık sütun yoğunluğu [m⁻²], eksi olmayan.
    ortalama_molekul_agirligi : float
        Parçacık başına ortalama kütle, m_H biriminde.

    Döndürür
    --------
    float ya da ndarray
        Yüzey yoğunluğu [kg/m²].

    Kaynaklar
    ---------
    [Krumholz2015] Bölüm 8.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: ``sutun_yogunlugu_m2`` ile gidiş-dönüş.
    """
    n = np.asarray(sutun_yogunlugu_m2, dtype=float)
    if np.any(n < 0):
        raise ValueError("sütun yoğunluğu eksi olamaz")
    (mu,) = _pozitif(ortalama_molekul_agirligi=ortalama_molekul_agirligi)
    return n * mu * _M_H


@kaynak(
    "Krumholz2015",
    yer="Bölüm 8 (Giant Molecular Clouds), Mass Measurement, Optically Thick Lines: N = M / (μ m_H π R²)",
    teyit="okundu",
)
def sutun_yogunlugu_m2(
    yuzey_yogunlugu_kg_m2: ArrayLike, ortalama_molekul_agirligi: float = ORTALAMA_MOLEKUL_AGIRLIGI
) -> Dizi:
    """Yüzey yoğunluğundan sütun yoğunluğunu hesaplar: N = Σ / (μ m_H).

    Parametreler
    ------------
    yuzey_yogunlugu_kg_m2 : float ya da ndarray
        Yüzey yoğunluğu [kg/m²], eksi olmayan.
    ortalama_molekul_agirligi : float
        Parçacık başına ortalama kütle, m_H biriminde.

    Döndürür
    --------
    float ya da ndarray
        Parçacık sütun yoğunluğu [m⁻²].

    Kaynaklar
    ---------
    [Krumholz2015] Bölüm 8.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: ``yuzey_yogunlugu_kg_m2`` ile gidiş-dönüş.
    """
    s = np.asarray(yuzey_yogunlugu_kg_m2, dtype=float)
    if np.any(s < 0):
        raise ValueError("yüzey yoğunluğu eksi olamaz")
    (mu,) = _pozitif(ortalama_molekul_agirligi=ortalama_molekul_agirligi)
    return s / (mu * _M_H)


@kaynak("McKee2007", yer="TeX etiketi LJeans", teyit="okundu")
@kaynak("Krumholz2015", yer="Bölüm 6 (Gravitational Instability and Collapse), λ_J = 2π/k_J", teyit="okundu")
def jeans_boyu_m(ses_hizi_m_s: ArrayLike, yogunluk_kg_m3: ArrayLike) -> Dizi:
    """Jeans uzunluğunu hesaplar: λ_J = sqrt(π c_s² / (G ρ)).

    Parametreler
    ------------
    ses_hizi_m_s : float ya da ndarray
        Eşısıl ses hızı [m/s], pozitif.
    yogunluk_kg_m3 : float ya da ndarray
        Düzgün ortamın yoğunluğu [kg/m³], pozitif.

    Döndürür
    --------
    float ya da ndarray
        Jeans uzunluğu [m]: kütleçekimsel parçalanmanın en küçük dalga boyu.

    Kaynaklar
    ---------
    [McKee2007] LJeans; [Krumholz2015] Bölüm 6 (doğrusal analizden türetir).
    İki kaynak aynı ifadeyi verir.

    Geçerlilik
    ----------
    Varsayılan μ = 2,33 parçacık başınadır: ``sayi_yogunlugu_m3`` bütün parçacıkların (H2 + He)
    yoğunluğu olmalıdır. Yalnızca H2 yoğunluğu (n_H2) veriliyorsa ``H2_BASINA_MOLEKUL_AGIRLIGI`` (2,8),
    hidrojen çekirdeği yoğunluğu (n_H) veriliyorsa ``H_BASINA_MOLEKUL_AGIRLIGI`` (1,4) kullanılmalıdır;
    n_H2 varsayılanla verilirse yoğunluk yaklaşık %17 düşük, serbest düşme süresi %9 uzun çıkar.
    Düzgün, sonsuz, eşısıl ortam ("Jeans hilesi"); yalnızca basınç desteği.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: c_s = 0,2 km/s, ρ = 100 m_p için 3,4 pc.
    """
    cs, rho = _pozitif(ses_hizi=ses_hizi_m_s, yogunluk=yogunluk_kg_m3)
    return np.sqrt(np.pi * cs**2 / (CEKIM.deger * rho))


@kaynak(
    "McKee2007",
    yer="TeX etiketi LJeans, ardından gelen paragraf (M_J = 2,47 M_BE)",
    teyit="okundu",
    not_="tanim='kure': M_J = (4π/3)(λ_J/2)³ ρ (McKee-Ostriker, Binney-Tremaine 1987 tanımı)",
)
@kaynak(
    "Krumholz2015",
    yer="Bölüm 6, Jeans kütlesi tanımı ve dipnotu",
    teyit="okundu",
    not_="tanim='kup': M_J = ρ λ_J³ / 8 (kenarı λ_J/2 olan küp); aradaki oran 4π/3'tür",
)
def jeans_kutlesi_kg(ses_hizi_m_s: ArrayLike, yogunluk_kg_m3: ArrayLike, tanim: str = "kure") -> Dizi:
    """Jeans kütlesini hesaplar; çarpan yazara göre değişir.

    ``tanim='kure'``: M_J = (4π/3)(λ_J/2)³ ρ = (π^{5/2}/6) c_s³ / sqrt(G³ ρ)
    ([McKee2007]; Bonnor-Ebert kütlesinin 2,47 katı).
    ``tanim='kup'``: M_J = ρ λ_J³ / 8 = (π^{3/2}/8) c_s³ / sqrt(G³ ρ) ([Krumholz2015]).
    Kaynaklar bu çarpanın belirsiz olduğunu açıkça söyler; ölçek bağıntısı
    c_s³ / sqrt(G³ ρ) her iki tanımda aynıdır.

    Parametreler
    ------------
    ses_hizi_m_s : float ya da ndarray
        Eşısıl ses hızı [m/s], pozitif.
    yogunluk_kg_m3 : float ya da ndarray
        Yoğunluk [kg/m³], pozitif.
    tanim : {"kure", "kup"}
        Çarpan tanımı.

    Döndürür
    --------
    float ya da ndarray
        Jeans kütlesi [kg].

    Kaynaklar
    ---------
    [McKee2007] LJeans sonrası paragraf; [Krumholz2015] Bölüm 6.

    Geçerlilik
    ----------
    ``jeans_boyu_m`` ile aynı.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: iki tanımın oranı 4π/3; M_J = 2,47 M_BE.
    """
    cs, rho = _pozitif(ses_hizi=ses_hizi_m_s, yogunluk=yogunluk_kg_m3)
    lam = np.sqrt(np.pi * cs**2 / (CEKIM.deger * rho))
    if tanim == "kure":
        return (4.0 * np.pi / 3.0) * (lam / 2.0) ** 3 * rho
    if tanim == "kup":
        return rho * lam**3 / 8.0
    raise ValueError("tanim 'kure' ya da 'kup' olmalı")


@kaynak(
    "McKee2007",
    yer="TeX etiketi MBE_def",
    teyit="okundu",
    not_="Katsayı 1,182 (Krumholz2015 alıştırma çözümü 1,18 verir)",
)
@kaynak("Krumholz2015", yer="problems/probsol2.tex, Bonnor-Ebert küresi alıştırmasının (h) şıkkı", teyit="okundu")
def bonnor_ebert_kritik_kutle_kg(ses_hizi_m_s: ArrayLike, yuzey_basinci_pa: ArrayLike) -> Dizi:
    """Bonnor-Ebert kütlesini hesaplar: M_BE = 1,182 c_s⁴ / sqrt(G³ P_0).

    Verilen yüzey basıncında hidrostatik dengede olabilen eşısıl, manyetik
    alansız kürenin en büyük kütlesidir (kararlı ya da kararsız).

    Parametreler
    ------------
    ses_hizi_m_s : float ya da ndarray
        Eşısıl ses hızı [m/s], pozitif.
    yuzey_basinci_pa : float ya da ndarray
        Yüzeydeki dış basınç P_0 = ρ_0 c_s² [Pa], pozitif.

    Döndürür
    --------
    float ya da ndarray
        Kritik kütle [kg].

    Kaynaklar
    ---------
    [McKee2007] MBE_def; [Krumholz2015] probsol2 (alan: m_max = 1,18).

    Geçerlilik
    ----------
    Eşısıl, dönmeyen, manyetik alansız, türbülanssız küre.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: T = 10 K, P/k_B = 3e5 cm⁻³ K için
    kaynaklardaki 0,66 ve 0,67 güneş kütlesi; ODE'den türetilen katsayıyla uyum.
    """
    cs, p = _pozitif(ses_hizi=ses_hizi_m_s, yuzey_basinci=yuzey_basinci_pa)
    return 1.182 * cs**4 / np.sqrt(CEKIM.deger**3 * p)


@kaynak(
    "McKee2007",
    yer="TeX etiketi MBE_def, sonraki paragraf: R_BE = 0,486 c_s / sqrt(G ρ_0)",
    teyit="okundu",
)
def bonnor_ebert_yaricapi_m(ses_hizi_m_s: ArrayLike, yuzeydeki_yogunluk_kg_m3: ArrayLike) -> Dizi:
    """Kritik Bonnor-Ebert küresinin yarıçapını hesaplar: R_BE = 0,486 c_s / sqrt(G ρ_0).

    Parametreler
    ------------
    ses_hizi_m_s : float ya da ndarray
        Eşısıl ses hızı [m/s], pozitif.
    yuzeydeki_yogunluk_kg_m3 : float ya da ndarray
        Kürenin yüzeyindeki yoğunluk ρ_0 [kg/m³], pozitif.

    Döndürür
    --------
    float ya da ndarray
        Yarıçap [m].

    Kaynaklar
    ---------
    [McKee2007] MBE_def sonrası paragraf.

    Geçerlilik
    ----------
    ``bonnor_ebert_kritik_kutle_kg`` ile aynı.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: ODE'den türetilen ξ ve yoğunluk
    karşıtlığıyla uyum; R_BE ≈ Jeans uzunluğunun 0,486/√π katı.
    """
    cs, rho = _pozitif(ses_hizi=ses_hizi_m_s, yuzeydeki_yogunluk=yuzeydeki_yogunluk_kg_m3)
    return 0.486 * cs / np.sqrt(CEKIM.deger * rho)


def _be_ode(xi: float, y: list[float]) -> list[float]:
    """Boyutsuz eşısıl küre denklemi: ψ'' + 2ψ'/ξ = exp(−ψ)."""
    psi, dpsi = y
    return [dpsi, -2.0 * dpsi / xi + np.exp(-psi)]


@lru_cache(maxsize=1)
def _be_katsayilari() -> tuple[float, float, float]:
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


@kaynak(
    "Krumholz2015",
    yer="problems/probset2.tex (e) şıkkı (kütle ifadesi); probsol2.tex (c)-(d) şıkları (ODE), (f) şıkkı (m_max = 1,18, ρ_c/ρ_s = 14,0)",
    teyit="okundu",
    not_="ODE ve m(ξ) = (4π)^{-1/2} ξ² ψ' e^{-ψ/2} ifadesi kaynaktan; yalnızca ξ_s ≈ 6,45 değeri kaynakta yoktur, sayısal çözümden gelir",
)
def bonnor_ebert_kritik_katsayilar() -> tuple[float, float, float]:
    """Eşısıl küre denkleminden Bonnor-Ebert kritik katsayılarını hesaplar.

    ψ'' + 2ψ'/ξ = e^{−ψ} denklemi (ψ(0) = ψ'(0) = 0) sayısal çözülür;
    boyutsuz kütle m = M sqrt(P_s) G^{3/2} / c_s⁴ en büyük değerine ulaştığında
    küre kritiktir.

    Döndürür
    --------
    tuple of float
        (m_max, ρ_c/ρ_s, ξ_s): en büyük boyutsuz kütle (≈ 1,18), kritik
        merkez/yüzey yoğunluk karşıtlığı (≈ 14,0) ve boyutsuz yüzey yarıçapı
        ξ_s = R/r_0 (≈ 6,45).

    Kaynaklar
    ---------
    [Krumholz2015] probsol2.

    Geçerlilik
    ----------
    ξ_s > ξ_kritik olan küreler kararsızdır; ρ_c/ρ_s > 14 kararsız demektir.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: kaynaktaki 1,18 ve 14,0 ile; McKee2007'deki
    1,182 ve R_BE = 0,486 c_s/sqrt(G ρ_0) ile tutarlılık.
    """
    return _be_katsayilari()


@kaynak("McKee2007", yer="TeX etiketi alpha_vir_def", teyit="okundu")
@kaynak(
    "Krumholz2015", yer="Bölüm 8 (Giant Molecular Clouds), Larson'un üçüncü yasası: α_vir = 5σ²R/(GM)", teyit="okundu"
)
def viryal_parametresi(hiz_dagilimi_m_s: ArrayLike, yaricap_m: ArrayLike, kutle_kg: ArrayLike) -> Dizi:
    """Viryal parametreyi hesaplar: α_vir = 5 σ² R / (G M).

    Katsayı, viryal dengedeki düzgün, manyetik alansız küre için α_vir = 1
    olacak biçimde seçilmiştir (bu küre hidrostatik dengede değildir).
    α_vir ≲ 1 bağlı, ≫ 1 dış basınçla tutulan bulutlara karşılık gelir.

    Parametreler
    ------------
    hiz_dagilimi_m_s : float ya da ndarray
        Tek boyutlu toplam (ısıl + türbülanslı) hız dağılımı σ [m/s], pozitif.
    yaricap_m : float ya da ndarray
        Bulut yarıçapı [m], pozitif.
    kutle_kg : float ya da ndarray
        Bulut kütlesi [kg], pozitif.

    Döndürür
    --------
    float ya da ndarray
        Boyutsuz viryal parametre.

    Kaynaklar
    ---------
    [McKee2007] alpha_vir_def; [Krumholz2015] Bölüm 8.

    Geçerlilik
    ----------
    Manyetik terim ve yüzey basıncı hesaba katılmaz.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: σ² = GM/(5R) için α_vir = 1; yüzey
    yoğunluğu biçimiyle tutarlılık.
    """
    sigma, r, m = _pozitif(hiz_dagilimi=hiz_dagilimi_m_s, yaricap=yaricap_m, kutle=kutle_kg)
    return 5.0 * sigma**2 * r / (CEKIM.deger * m)


@kaynak(
    "McKee2007",
    yer="TeX etiketi alpha_vir_def",
    teyit="okundu",
    not_="M = π R² Σ yerine koymasıyla yazıldı; Krumholz2015 Bölüm 8'deki (5/π pc) σ_pc²/(GΣ) biçimiyle aynıdır",
)
def viryal_parametresi_yuzeyden(
    hiz_dagilimi_m_s: ArrayLike, yaricap_m: ArrayLike, yuzey_yogunlugu_kg_m2: ArrayLike
) -> Dizi:
    """Viryal parametreyi yüzey yoğunluğundan hesaplar: α_vir = 5 σ² / (π G Σ R).

    Parametreler
    ------------
    hiz_dagilimi_m_s : float ya da ndarray
        Tek boyutlu hız dağılımı [m/s], pozitif.
    yaricap_m : float ya da ndarray
        Bulut yarıçapı [m], pozitif.
    yuzey_yogunlugu_kg_m2 : float ya da ndarray
        Ortalama yüzey yoğunluğu Σ = M/(π R²) [kg/m²], pozitif.

    Döndürür
    --------
    float ya da ndarray
        Boyutsuz viryal parametre.

    Kaynaklar
    ---------
    [McKee2007] alpha_vir_def.

    Geçerlilik
    ----------
    ``viryal_parametresi`` ile aynı; yüzey yoğunluğu izdüşüm dairesine göre alınır.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: M = π R² Σ ile ``viryal_parametresi`` eşitliği.
    """
    sigma, r, s = _pozitif(hiz_dagilimi=hiz_dagilimi_m_s, yaricap=yaricap_m, yuzey_yogunlugu=yuzey_yogunlugu_kg_m2)
    return 5.0 * sigma**2 / (np.pi * CEKIM.deger * s * r)


@kaynak(
    "McKee2007", yer="TeX etiketi eq:tff", teyit="okundu", not_="Sayısal değer n_H için He oranı %10 alınarak verilir"
)
@kaynak("Krumholz2015", yer="Bölüm 6, Jeans analizinden sonra t_ff tanımı; Bölüm 8 zaman ölçekleri", teyit="okundu")
def serbest_dusme_zamani_s(yogunluk_kg_m3: ArrayLike) -> Dizi:
    """Serbest düşme zamanını hesaplar: t_ff = sqrt(3π / (32 G ρ)).

    Basınçsız, düzgün bir kürenin kendi çekimiyle bir noktaya çökme süresidir.

    Parametreler
    ------------
    yogunluk_kg_m3 : float ya da ndarray
        Ortalama kütle yoğunluğu [kg/m³], pozitif.

    Döndürür
    --------
    float ya da ndarray
        Serbest düşme zamanı [s].

    Kaynaklar
    ---------
    [McKee2007] eq:tff; [Krumholz2015] Bölüm 6 ve 8.

    Geçerlilik
    ----------
    Basınç, türbülans ve manyetik destek yok sayılır; gerçek bulutlar daha yavaş çöker.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: n_H = 10³ cm⁻³ için 1,37 milyon yıl.
    """
    (rho,) = _pozitif(yogunluk=yogunluk_kg_m3)
    return np.sqrt(3.0 * np.pi / (32.0 * CEKIM.deger * rho))


@kaynak("Krumholz2015", yer="Bölüm 8 (Giant Molecular Clouds), Molecular Cloud Timescales: t_cr = R/σ", teyit="okundu")
def gecis_zamani_s(yaricap_m: ArrayLike, hiz_dagilimi_m_s: ArrayLike) -> Dizi:
    """Geçiş (crossing) zamanını hesaplar: t_cr = R / σ.

    σ olarak ses hızı verilirse ses geçiş zamanı, türbülans hız dağılımı
    verilirse türbülans geçiş zamanı elde edilir.

    Parametreler
    ------------
    yaricap_m : float ya da ndarray
        Bulut yarıçapı [m], pozitif.
    hiz_dagilimi_m_s : float ya da ndarray
        Hız dağılımı ya da ses hızı [m/s], pozitif.

    Döndürür
    --------
    float ya da ndarray
        Geçiş zamanı [s].

    Kaynaklar
    ---------
    [Krumholz2015] Bölüm 8.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: viryal bulutta t_cr ≈ 2 t_ff (kaynak: α_vir = 1
    için serbest düşme zamanı geçiş zamanının yarısıdır).
    """
    r, sigma = _pozitif(yaricap=yaricap_m, hiz_dagilimi=hiz_dagilimi_m_s)
    return r / sigma


@kaynak(
    "McKee2007",
    yer="TeX etiketi eq:mphi",
    teyit="okundu",
    not_="c_Φ = 0,17 Tomisaka (1988) değeridir (kaynakta aktarılmış, orijinali okunmadı)",
)
@kaynak(
    "Krumholz2015",
    yer="Bölüm 6, Magnetic Support and the Magnetic Critical Mass",
    teyit="okundu",
    not_="Kaynak düzgün küre için sqrt(5/2)/(3π) = 0,17 türetir ve Tomisaka (1998) için 0,12 verir; McKee2007 ise 0,17 için başka bir makaleyi (Tomisaka, Ikeuchi ve Nakamura 1988) anar; iki özgün makale de okunmadı",
)
def manyetik_kritik_kutle_kg(manyetik_aki_wb: ArrayLike, c_phi: float = 0.17) -> Dizi:
    """Manyetik kritik kütleyi hesaplar: M_Φ = c_Φ Φ / G^{1/2} (Gauss birimlerinde).

    M > M_Φ ise bulut manyetik olarak süperkritiktir ve alan çöküşü
    durduramaz; M < M_Φ ise çökemez. Φ = π R² B düzgün alan için akıdır.
    SI'ya dönüşüm: Φ[Mx] = 10⁸ Φ[Wb], G_cgs = 10³ G_SI, kg = 10⁻³ g.

    Parametreler
    ------------
    manyetik_aki_wb : float ya da ndarray
        Buluttan geçen manyetik akı [Wb], pozitif.
    c_phi : float
        İç yoğunluk ve alan dağılımına bağlı katsayı; 0,17 (kutupsal alan,
        sabit kütle/akı oranı), düzgün küre için sqrt(5/2)/(3π) ≈ 0,168.

    Döndürür
    --------
    float ya da ndarray
        Kritik kütle [kg].

    Kaynaklar
    ---------
    [McKee2007] eq:mphi; [Krumholz2015] Bölüm 6.

    Geçerlilik
    ----------
    Soğuk bulut, ideal MHD (akı donmuş); çift kutuplu ya da rastgele alanda
    toplam akı kütleyi belirlemez.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: levha eşdeğeriyle (Σ/B)_kr = 1/(2π√G)
    tutarlılık; Φ ile doğrusal ölçekleme.
    """
    (phi,) = _pozitif(manyetik_aki=manyetik_aki_wb)
    if not c_phi > 0:
        raise ValueError("c_phi pozitif olmalı")
    g_cgs = CEKIM.deger * 1e3
    return c_phi * (phi * 1e8) / np.sqrt(g_cgs) * 1e-3


@kaynak(
    "McKee2007",
    yer="TeX etiketi eq:mphi sonrası: G^{1/2}(Σ/B)_cr = 1/(2π) (sonsuz soğuk levha)",
    teyit="okundu",
    not_="Kaynak değeri Nakano-Nakamura (1978) için aktarır; orijinali okunmadı. Normlu oran 2π√G Σ/B olarak yazıldı",
)
def kutle_akisi_orani_normlu(yuzey_yogunlugu_kg_m2: ArrayLike, manyetik_alan_t: ArrayLike) -> Dizi:
    """Kütle/akı oranının kritik değere oranını hesaplar: μ/μ_kr = 2π √G Σ / B.

    Kritik değer G^{1/2}(Σ/B)_kr = 1/(2π) (Gauss birimlerinde). Oran 1'den
    büyükse bulut manyetik olarak süperkritiktir.

    Parametreler
    ------------
    yuzey_yogunlugu_kg_m2 : float ya da ndarray
        Yüzey yoğunluğu [kg/m²], pozitif.
    manyetik_alan_t : float ya da ndarray
        Yüzeye dik alan bileşeni [T], pozitif. (1 T = 10⁴ G; Σ[g/cm²] = 0,1 Σ[kg/m²].)

    Döndürür
    --------
    float ya da ndarray
        Boyutsuz normlu kütle/akı oranı.

    Kaynaklar
    ---------
    [McKee2007] eq:mphi sonrası paragraf.

    Geçerlilik
    ----------
    Sonsuz soğuk levha; küresel bulutlar için c_Φ ≈ 0,17 ≈ 1/(2π) ile yaklaşık eşdeğer.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: Σ = B/(2π√G) için oran 1; küre M_Φ ile tutarlılık.
    """
    s, b = _pozitif(yuzey_yogunlugu=yuzey_yogunlugu_kg_m2, manyetik_alan=manyetik_alan_t)
    g_cgs = CEKIM.deger * 1e3
    return 2.0 * np.pi * np.sqrt(g_cgs) * (s * 0.1) / (b * 1e4)


@kaynak(
    "McKee2007",
    yer="TeX etiketi GMC_lws (Heyer ve Brunt 2004 PCA uyumu)",
    teyit="okundu",
    not_="Varsayılan 0,9 km/s ve 0,56, McKee2007'de aktarılan PCA uyumudur; L_pca bulut çapının yaklaşık dörtte biridir",
)
@kaynak(
    "Krumholz2015",
    yer="Bölüm 8 (Giant Molecular Clouds), Scaling Relations: Larson'un ikinci yasası",
    teyit="okundu",
    not_="Kaynakta aktarılan alternatifler: 0,72 ve 0,5 (Solomon 1987, Samanyolu), 0,44 ve 0,60 (Bolatto 2008); orijinalleri okunmadı",
)
def dogrusal_hiz_boyut_km_s(boyut_pc: ArrayLike, katsayi_km_s: float = 0.9, us: float = 0.56) -> Dizi:
    """Larson doğrusal genişlik-boyut bağıntısını hesaplar: δv = a (L / 1 pc)^q.

    Parametreler
    ------------
    boyut_pc : float ya da ndarray
        Ölçek ya da bulut yarıçapı [pc], pozitif.
    katsayi_km_s : float
        1 pc'teki hız dağılımı a [km/s].
    us : float
        Güç yasası üssü q (türbülans için ≈ 0,5; Kolmogorov 1/3).

    Döndürür
    --------
    float ya da ndarray
        Hız dağılımı [km/s].

    Kaynaklar
    ---------
    [McKee2007] GMC_lws; [Krumholz2015] Bölüm 8.

    Geçerlilik
    ----------
    Samanyolu'ndaki moleküler bulutlar ve içlerindeki 0,03-30 pc ölçekleri; boyut
    tanımı (PCA ölçeği ya da bulut yarıçapı) normalizasyonu değiştirir.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: 1 pc'te a; üs ölçeklemesi; kaynaktaki alternatifler.
    """
    (boyut,) = _pozitif(boyut=boyut_pc)
    return katsayi_km_s * boyut**us


@kaynak(
    "McKee2007",
    yer="TeX etiketi eq:mdssis (Shu genişleme dalgası çözümü)",
    teyit="okundu",
    not_="Kaynak genel biçimi Φ c_s³/G (TeX eq:mds) yazar; Shu çözümü için Φ = 0,975",
)
def izotermal_cokus_birikim_hizi_kg_s(ses_hizi_m_s: ArrayLike, katsayi: float = 0.975) -> Dizi:
    """Eşısıl çekirdeğin çöküşünde protoyıldıza kütle akış hızını hesaplar: Ṁ = 0,975 c_s³ / G.

    Shu genişleme dalgası çözümünde hız zamanla sabittir; dalganın içindeki
    kütle 2Ṁt'dir ve yaklaşık yarısı protoyıldızdadır.

    Parametreler
    ------------
    ses_hizi_m_s : float ya da ndarray
        Eşısıl ses hızı [m/s], pozitif.
    katsayi : float
        Boyutsuz çarpan Φ; Shu için 0,975, Larson-Penston-Hunter çözümü daha büyüktür.

    Döndürür
    --------
    float ya da ndarray
        Kütle akış hızı [kg/s].

    Kaynaklar
    ---------
    [McKee2007] eq:mdssis.

    Geçerlilik
    ----------
    Dönmesiz, manyetik alansız, eşısıl çekirdek; protoyıldız çıkışları hızı bir
    ε_core < 1 çarpanıyla azaltır.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: 10 K için 1,54e-6 güneş kütlesi/yıl.
    """
    (cs,) = _pozitif(ses_hizi=ses_hizi_m_s)
    if not katsayi > 0:
        raise ValueError("katsayı pozitif olmalı")
    return katsayi * cs**3 / CEKIM.deger


@kaynak("McKee2007", yer="TeX etiketi eq:epsff", teyit="okundu")
@kaynak("Krumholz2015", yer="Bölüm 8 (Giant Molecular Clouds), Depletion Time: ε_ff = t_ff / t_dep", teyit="okundu")
def serbest_dusme_verimliligi(serbest_dusme_zamani_s: ArrayLike, tuketme_zamani_s: ArrayLike) -> Dizi:
    """Serbest düşme zamanı başına yıldız oluşum verimliliğini hesaplar: ε_ff = t_ff / t_dep.

    Bir bulutun kütlesinin bir serbest düşme zamanında yıldıza dönüşen oranıdır.

    Parametreler
    ------------
    serbest_dusme_zamani_s : float ya da ndarray
        Serbest düşme zamanı [s], pozitif.
    tuketme_zamani_s : float ya da ndarray
        Gaz tüketme zamanı M/Ṁ_* [s], pozitif.

    Döndürür
    --------
    float ya da ndarray
        Boyutsuz verimlilik.

    Kaynaklar
    ---------
    [McKee2007] eq:epsff; [Krumholz2015] Bölüm 8. Kaynaklar gözlenen tipik
    değerin ε_ff ~ 0,01 olduğunu belirtir.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: ``bulut_yildiz_olusum_hizi_kg_s`` ile gidiş-dönüş.
    """
    tff, tdep = _pozitif(serbest_dusme_zamani=serbest_dusme_zamani_s, tuketme_zamani=tuketme_zamani_s)
    return tff / tdep


@kaynak(
    "McKee2007",
    yer="TeX etiketi eq:epsff",
    teyit="okundu",
    not_="Ṁ_* = ε_ff M / t_ff; ε_ff için kaynakta tipik değer ~0,01 (Krumholz2015 Bölüm 8-10)",
)
def bulut_yildiz_olusum_hizi_kg_s(
    bulut_kutlesi_kg: ArrayLike, ortalama_yogunluk_kg_m3: ArrayLike, ff_verimliligi: float = 0.01
) -> Dizi:
    """Bulutun yıldız oluşum hızını hesaplar: Ṁ_* = ε_ff M / t_ff.

    Parametreler
    ------------
    bulut_kutlesi_kg : float ya da ndarray
        Gaz kütlesi [kg], pozitif.
    ortalama_yogunluk_kg_m3 : float ya da ndarray
        Bulutun ortalama yoğunluğu [kg/m³]; t_ff bu yoğunlukla hesaplanır.
    ff_verimliligi : float
        Serbest düşme zamanı başına verimlilik ε_ff, [0, 1] aralığında;
        gözlenen tipik değer ≈ 0,01.

    Döndürür
    --------
    float ya da ndarray
        Yıldız oluşum hızı [kg/s].

    Kaynaklar
    ---------
    [McKee2007] eq:epsff.

    Geçerlilik
    ----------
    ε_ff ölçülen ya da varsayılan bir sabittir; bu fonksiyon onu öngörmez.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: M/Ṁ_* = t_ff/ε_ff; ε_ff ile doğrusal ölçekleme.
    """
    m, rho = _pozitif(kutle=bulut_kutlesi_kg, yogunluk=ortalama_yogunluk_kg_m3)
    if not 0.0 <= ff_verimliligi <= 1.0:
        raise ValueError("ε_ff [0, 1] aralığında olmalı")
    return ff_verimliligi * m / serbest_dusme_zamani_s(rho)
