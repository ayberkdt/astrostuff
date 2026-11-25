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

from kozmos._kaynak import kaynak
from kozmos.temel.sabitler import BOLTZMANN, CEKIM, ISIK_HIZI, PLANCK

Dizi = NDArray[np.float64]

_G = CEKIM.deger
_C = ISIK_HIZI.deger
_HBAR = PLANCK.deger / (2.0 * np.pi)  # indirgenmiş Planck sabiti, ħ = h / 2π
_RAD_UAS = 180.0 / np.pi * 3600.0 * 1e6  # radyan → mikroyay saniyesi


def _pozitif(deger: ArrayLike, ad: str) -> Dizi:
    """Değeri diziye çevirir ve pozitif olduğunu denetler."""
    x = np.asarray(deger, dtype=float)
    if np.any(~(x > 0.0)):
        raise ValueError(f"{ad} pozitif olmalı")
    return x


def _spin(spin: ArrayLike) -> Dizi:
    """Boyutsuz dönmeyi diziye çevirir ve |a_*| ≤ 1 olduğunu denetler."""
    a = np.asarray(spin, dtype=float)
    if np.any(~(np.abs(a) <= 1.0)):
        raise ValueError("boyutsuz dönme |a*| ≤ 1 olmalı")
    return a


# --------------------------------------------------------------------------------------
# Schwarzschild geometrisi
# --------------------------------------------------------------------------------------


@kaynak("EHT2019", yer="§1 (Giriş), r_g ≡ GM/c² tanımı, sayfa L1", teyit="okundu")
def kutleceken_yaricapi_m(kutle_kg: ArrayLike) -> Dizi:
    """Kütle çekimsel yarıçapı hesaplar: r_G = G M / c².

    Parametreler
    ------------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg], pozitif.

    Döndürür
    --------
    float ya da ndarray
        r_G [m].

    Kaynaklar
    ---------
    [EHT2019] §1, "r_g ≡ GM/c² ... karakteristik uzunluk ölçeği".

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: Güneş kütlesi için 1,477 km; kütleyle doğrusal.
    """
    return _G * _pozitif(kutle_kg, "kütle") / _C**2


@kaynak("EHT2019", yer="§1 (Giriş), R_S = 2 r_g, sayfa L1", teyit="okundu")
def schwarzschild_yaricapi_m(kutle_kg: ArrayLike) -> Dizi:
    """Schwarzschild yarıçapını (dönmeyen kara deliğin ufku) hesaplar: R_S = 2 G M / c².

    Parametreler
    ------------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg], pozitif.

    Döndürür
    --------
    float ya da ndarray
        R_S [m].

    Kaynaklar
    ---------
    [EHT2019] §1, "R_S = 2 r_g".

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: Güneş kütlesi için 2,95 km; Kerr ufkunun a* = 0 limitiyle aynı.
    """
    return 2.0 * kutleceken_yaricapi_m(kutle_kg)


@kaynak(
    "Carroll1997",
    yer="Bölüm 7, Denk. 7.58-7.61",
    teyit="okundu",
    not_="sabit konumdaki iki gözlemci için ω2/ω1 = [(1 − 2GM/r1)/(1 − 2GM/r2)]^{1/2} (G = c = 1); "
    "burada alıcı sonsuzda (r2 → ∞): 1 + z = (1 − R_S/r)^{−1/2}",
)
def yercekimsel_kirmiziya_kayma(yaricap_m: ArrayLike, kutle_kg: ArrayLike) -> Dizi:
    """Statik yayıcının sonsuzdaki gözlemciye göre kütle çekimsel kırmızıya kaymasını hesaplar.

    1 + z = (1 − R_S / r)^(−1/2)

    Parametreler
    ------------
    yaricap_m : float ya da ndarray
        Yayıcının Schwarzschild yarıçap koordinatı [m]; R_S değerinden büyük olmalı.
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].

    Döndürür
    --------
    float ya da ndarray
        Kırmızıya kayma z (boyutsuz).

    Kaynaklar
    ---------
    [Carroll1997] Denk. 7.58-7.61.

    Geçerlilik
    ----------
    Yalnızca durağan (yörüngede olmayan) yayıcı; r > R_S. Ufka yaklaşırken z → ∞.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: r = 3 R_S/2 için z = √3 − 1; r → ∞ için z ≈ R_S/(2r).
    """
    r = _pozitif(yaricap_m, "yarıçap")
    rs = schwarzschild_yaricapi_m(kutle_kg)
    if np.any(r <= rs):
        raise ValueError("yarıçap Schwarzschild yarıçapından büyük olmalı")
    return (1.0 - rs / r) ** -0.5 - 1.0


@kaynak(
    "Abramowicz2013",
    yer="'Summary: characteristic radii and frequencies' (TeX etiketi schwarzschild-frequencies), x_ph = 3",
    teyit="okundu",
    not_="Kerr formülünün (TeX etiketi photon-radius-kerr) a* = 0 değeri de 3 r_G verir.",
)
def foton_kuresi_yaricapi_m(kutle_kg: ArrayLike) -> Dizi:
    """Dönmeyen kara deliğin foton küresinin yarıçapını hesaplar: r_ph = 3 G M / c².

    Parametreler
    ------------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].

    Döndürür
    --------
    float ya da ndarray
        r_ph [m].

    Kaynaklar
    ---------
    [Abramowicz2013] Schwarzschild için x_ph = r_ph/r_G = 3.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: Kerr formülünün a* = 0 değeriyle ve R_S oranıyla (3/2).
    """
    return 3.0 * kutleceken_yaricapi_m(kutle_kg)


@kaynak(
    "EHT2019",
    yer="§1 (Giriş), 'foton yakalama yarıçapı' R_c = √27 r_g, sayfa L1",
    teyit="okundu",
    not_="PDF'ten okundu; metin çıkarımı √ simgesini düşürüyor, sayfa görüntüsünde R_c = √27 r_g doğrulandı.",
)
def kritik_carpisma_parametresi_m(kutle_kg: ArrayLike) -> Dizi:
    """Dönmeyen kara deliğin kritik çarpışma parametresini hesaplar: b_c = √27 G M / c².

    b < b_c ile gelen fotonlar yakalanır, b > b_c olanlar kaçar; b = b_c kararsız
    dairesel foton yörüngesidir (foton halkası).

    Parametreler
    ------------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].

    Döndürür
    --------
    float ya da ndarray
        b_c [m].

    Kaynaklar
    ---------
    [EHT2019] §1, "R_c = √27 r_g".

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: foton küresi yarıçapı ve Schwarzschild metriğiyle
    bağımsız hesap b_c = r_ph / √(1 − R_S/r_ph).
    """
    return np.sqrt(27.0) * kutleceken_yaricapi_m(kutle_kg)


@kaynak(
    "EHT2019",
    yer="§1 R_c = √27 r_g; §7 'Model Comparison and Parameter Estimation' θ_g ≡ GM/(c² D)",
    teyit="okundu",
    not_="Çap = 2 R_c / D küçük açı yaklaşımıdır (geometri, kaynakta yazılı değil). EHT'nin Schwarzschild için "
    "verdiği 37,6 ve 21,3 µas değerleri testte yeniden üretilir.",
)
def golge_acisal_capi_uas(kutle_kg: ArrayLike, uzaklik_m: ArrayLike) -> Dizi:
    """Dönmeyen kara deliğin gölgesinin gökyüzündeki açısal çapını hesaplar.

    θ_gölge = 2 √27 θ_g,  θ_g = G M / (c² D)

    Parametreler
    ------------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].
    uzaklik_m : float ya da ndarray
        Gözlemciye uzaklık [m]; D ≫ r_G varsayılır.

    Döndürür
    --------
    float ya da ndarray
        Açısal çap [µas, mikroyay saniyesi].

    Kaynaklar
    ---------
    [EHT2019] §1 (R_c) ve §7 (θ_g tanımı; oradaki d = α θ_g ışıma halkasının çapıdır, gölgenin değil).

    Geçerlilik
    ----------
    Dönmeyen kara delik, küçük açı yaklaşımı. Dönme gölge boyutunu yaklaşık %4 değiştirir
    (kaynak, §1); Kerr için α spin ve eğikliğe bağlıdır ve burada hesaplanmaz.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: M87* için 37,6 µas (yıldız kütlesi) ve 21,3 µas (gaz kütlesi).
    """
    d = _pozitif(uzaklik_m, "uzaklık")
    return 2.0 * kritik_carpisma_parametresi_m(kutle_kg) / d * _RAD_UAS


@kaynak(
    "Abramowicz2013",
    yer="'Summary: characteristic radii and frequencies' (TeX etiketi schwarzschild-frequencies), x_ms = 6",
    teyit="okundu",
)
def isco_yaricapi_schwarzschild_m(kutle_kg: ArrayLike) -> Dizi:
    """Dönmeyen kara deliğin en içteki kararlı dairesel yörüngesinin (ISCO) yarıçapı: r_ISCO = 6 G M / c².

    Parametreler
    ------------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].

    Döndürür
    --------
    float ya da ndarray
        r_ISCO [m].

    Kaynaklar
    ---------
    [Abramowicz2013] Schwarzschild için x_ms = r_ms/r_G = 6.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: Kerr ISCO formülünün a* = 0 değeri; 3 R_S.
    """
    return 6.0 * kutleceken_yaricapi_m(kutle_kg)


@kaynak(
    "Abramowicz2013",
    yer="'Summary: characteristic radii and frequencies', Ω_K² = GM/r³ (TeX etiketi schwarzschild-frequencies), x_ms = 6",
    teyit="okundu",
    not_="f = Ω_K / (2π) dönüşümü tanımdır.",
)
def isco_yorunge_frekansi_hz(kutle_kg: ArrayLike) -> Dizi:
    """ISCO'daki Kepler yörünge frekansını hesaplar (dönmeyen kara delik).

    Ω_K² = G M / r³,  r = 6 G M / c²  ⇒  f_yör = c³ / (2π 6^{3/2} G M)

    Parametreler
    ------------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].

    Döndürür
    --------
    float ya da ndarray
        Yörünge frekansı [Hz], sonsuzdaki gözlemciye göre.

    Kaynaklar
    ---------
    [Abramowicz2013] Schwarzschild için Ω_K² = GM/r³ ve x_ms = 6.

    Geçerlilik
    ----------
    r Schwarzschild yarıçap koordinatıdır; Ω_K sonsuzdaki gözlemcinin zamanına göre ölçülür.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: Ω_K² = GM/r³ ile bağımsız hesap; 1/M ölçeklemesi.
    """
    m = _pozitif(kutle_kg, "kütle")
    r = isco_yaricapi_schwarzschild_m(m)
    return np.sqrt(_G * m / r**3) / (2.0 * np.pi)


# --------------------------------------------------------------------------------------
# Kerr geometrisi
# --------------------------------------------------------------------------------------


@kaynak(
    "Abramowicz2013",
    yer="'Summary: characteristic radii and frequencies', TeX etiketi stable-radius-kerr (Bardeen vd. formülü)",
    teyit="okundu",
    not_="Kaynağın formülü doğru (prograde) yörünge içindir; Z1 ve Z2 a* çiftidir. Her iki işaret de "
    "Abramowicz2013'ün epicyclic-radial denkleminin (ω_r*)² = 0 koşuluyla testte bağımsız doğrulanır.",
)
@kaynak(
    "Reynolds2021",
    yer="Bölüm 2: r_isco denklemi ve Z1, Z2 tanımları",
    teyit="okundu",
    not_="r_isco = (3 + Z2 ∓ [(3 − Z1)(3 + Z1 + 2Z2)]^{1/2}) r_g; ∓ işareti sırasıyla ileri ve geri yönlü "
    "yörüngeler içindir. Aynı ifade özgün kaynakta Denk. 2.21'dir [Bardeen1972].",
)
def isco_yaricapi_kerr_rg(spin: ArrayLike) -> Dizi:
    """Kerr kara deliğin ISCO yarıçapını hesaplar (Bardeen, Press, Teukolsky formülü).

    r_ISCO / r_G = 3 + Z2 ∓ √[(3 − Z1)(3 + Z1 + 2 Z2)]

    Z1 = 1 + (1 − a*²)^{1/3} [(1 + a*)^{1/3} + (1 − a*)^{1/3}],  Z2 = √(3 a*² + Z1²)

    Eksi işaret doğru (a* ≥ 0), artı işaret ters yörünge (a* < 0) içindir.

    Parametreler
    ------------
    spin : float ya da ndarray
        Boyutsuz dönme a_* ∈ [−1, 1]; negatif değer retrograde yörünge.

    Döndürür
    --------
    float ya da ndarray
        r_ISCO / r_G (boyutsuz): a* = 0 için 6, a* = 1 için 1, a* = −1 için 9.

    Kaynaklar
    ---------
    [Abramowicz2013] stable-radius-kerr; [Reynolds2021] Bölüm 2.

    Geçerlilik
    ----------
    Kerr metriği, ekvator düzlemi, test parçacığı.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: sınır değerler (1, 6, 9) ve (ω_r*)² = 0 koşulu;
    tests/ozellik/test_karadelik_ozellik.py: spin ile tekdüzelik.
    """
    a_isaretli = _spin(spin)
    a = np.abs(a_isaretli)
    z1 = 1.0 + np.cbrt(1.0 - a**2) * (np.cbrt(1.0 + a) + np.cbrt(1.0 - a))
    z2 = np.sqrt(3.0 * a**2 + z1**2)
    kok = np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    isaret = np.where(a_isaretli >= 0.0, 1.0, -1.0)
    return 3.0 + z2 - isaret * kok


def _isco_enerjisi(spin: ArrayLike) -> Dizi:
    """ISCO'daki birim kütle başına enerji, E = √(1 − 2/(3 x_ISCO)); yardımcı."""
    x = isco_yaricapi_kerr_rg(spin)
    return np.sqrt(1.0 - 2.0 / (3.0 * x))


@kaynak(
    "Bardeen1972",
    yer="Denk. 2.20: kararlılık koşulu 1 − (E/μ)² ≥ (2/3)(M/r), eşitlik r = r_ms'de",
    teyit="okundu",
    not_="ADS taramasından okundu. Koşulun r_ms'deki eşitliği Ẽ_ms = (1 − 2/(3 r̃_ms))^{1/2} verir. Testte "
    "Ẽ(r) dairesel yörünge enerjisinin ISCO'da minimum olması (dẼ/dr = 0) ve Schwarzschild değeri √(8/9) ile sınanır.",
)
def isco_ozgul_enerji_kerr(spin: ArrayLike) -> Dizi:
    """ISCO'daki dairesel yörüngenin birim durgun kütle başına enerjisini hesaplar.

    E_ISCO = √(1 − 2 / (3 x_ISCO)),  x_ISCO = r_ISCO / r_G

    Parametreler
    ------------
    spin : float ya da ndarray
        Boyutsuz dönme a_* ∈ [−1, 1]; negatif değer retrograde yörünge.

    Döndürür
    --------
    float ya da ndarray
        E_ISCO / (m c²): a* = 0 için √(8/9) ≈ 0,9428.

    Kaynaklar
    ---------
    [Bardeen1972] Denk. 2.20.

    Geçerlilik
    ----------
    Kerr metriği, ekvator düzlemi, test parçacığı.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: √(8/9) ve bağımsız dairesel yörünge enerjisinin minimumu.
    """
    return _isco_enerjisi(spin)


@kaynak(
    "Abramowicz2013",
    yer="§1 (Giriş), verim tanımı η ≡ L/(Ṁ c²)",
    teyit="okundu",
    not_="Inayoshi2020 dönmeyen kara delik için '~0,06' der (testte 0,057 ile uyuşur).",
)
@kaynak(
    "Reynolds2021",
    yer="Bölüm 5: Novikov-Thorne diski için η = 1 − e_isco/c²",
    teyit="okundu",
    not_="e_isco: ISCO'da dairesel yörüngedeki birim kütlenin toplam enerjisi; e_isco'nun kapalı biçimi "
    "isco_ozgul_enerji_kerr içindedir [Bardeen1972, Denk. 2.20].",
)
def isco_isinim_verimliligi(spin: ArrayLike) -> Dizi:
    """İnce disk için ışınım verimini hesaplar: η = 1 − E_ISCO.

    η ≡ L / (Ṁ c²). Dönmeyen kara delik için 1 − √(8/9) ≈ 0,0572; a* = 1 için ≈ 0,42.

    Parametreler
    ------------
    spin : float ya da ndarray
        Boyutsuz dönme a_* ∈ [−1, 1]; negatif değer retrograde disk.

    Döndürür
    --------
    float ya da ndarray
        Verim η (boyutsuz).

    Kaynaklar
    ---------
    [Abramowicz2013] §1; [Reynolds2021] Bölüm 5.

    Geçerlilik
    ----------
    Geometrik olarak ince, ışınımca verimli disk; ISCO içinden gelen ışınım ve ufka düşen
    ışığın geri yakalanması yok sayılır.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: 0,057 (Inayoshi2020 '~0,06'); tests/ozellik/test_karadelik_ozellik.py:
    spinle artış.
    """
    return 1.0 - _isco_enerjisi(spin)


@kaynak(
    "Abramowicz2013",
    yer="'Summary: characteristic radii and frequencies', r_H (aynı eqnarray, etiketsiz satır)",
    teyit="okundu",
)
@kaynak("Page2005", yer="TeX etiketi eq:3, r_+ (Q = 0)", teyit="okundu")
def ufuk_yaricapi_kerr_rg(spin: ArrayLike) -> Dizi:
    """Kerr kara deliğin olay ufkunun yarıçapını hesaplar: r_+ = r_G (1 + √(1 − a*²)).

    Parametreler
    ------------
    spin : float ya da ndarray
        Boyutsuz dönme a_* ∈ [−1, 1].

    Döndürür
    --------
    float ya da ndarray
        r_+ / r_G: a* = 0 için 2, a* = ±1 için 1.

    Kaynaklar
    ---------
    [Abramowicz2013] r_H; [Page2005] TeX etiketi eq:3 (yükü sıfır alınarak).

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: a* = 0 ve 1 değerleri; R_S/2r_G ile uyum.
    """
    a = _spin(spin)
    return 1.0 + np.sqrt(1.0 - a**2)


@kaynak(
    "Abramowicz2013",
    yer="'Summary: characteristic radii and frequencies', r_0 (ergosfer; aynı eqnarray, etiketsiz satır)",
    teyit="okundu",
)
def ergosfer_yaricapi_kerr_rg(spin: ArrayLike, kutup_acisi_rad: ArrayLike = np.pi / 2) -> Dizi:
    """Kerr kara deliğin ergosfer yarıçapını hesaplar: r_0 = r_G (1 + √(1 − a*² cos²θ)).

    Parametreler
    ------------
    spin : float ya da ndarray
        Boyutsuz dönme a_* ∈ [−1, 1].
    kutup_acisi_rad : float ya da ndarray, isteğe bağlı
        Kutup açısı θ [rad]; varsayılan ekvator (π/2), burada ergosfer en geniştir.

    Döndürür
    --------
    float ya da ndarray
        r_0 / r_G: ekvatorda 2, kutuplarda ufka teğet (r_+).

    Kaynaklar
    ---------
    [Abramowicz2013] r_0.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: ekvatorda 2 r_G; kutupta ufuk yarıçapı;
    tests/ozellik/test_karadelik_ozellik.py: ufuk ≤ ergosfer.
    """
    a = _spin(spin)
    teta = np.asarray(kutup_acisi_rad, dtype=float)
    return 1.0 + np.sqrt(1.0 - (a * np.cos(teta)) ** 2)


@kaynak(
    "Page2005",
    yer="TeX etiketi eq:3, Ω = 4π a / A (Q = 0)",
    teyit="okundu",
    not_="Kaynak G = c = 1 birimindedir; SI'ya Ω_H = c³ Ω̃ / (G M) ile çevrildi.",
)
def ufuk_acisal_hizi_rad_s(kutle_kg: ArrayLike, spin: ArrayLike) -> Dizi:
    """Kerr ufkunun açısal hızını hesaplar.

    Ω_H = (c³ / G M) · a* / [2 (1 + √(1 − a*²))]

    Parametreler
    ------------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].
    spin : float ya da ndarray
        Boyutsuz dönme a_* ∈ [−1, 1].

    Döndürür
    --------
    float ya da ndarray
        Ω_H [rad s^-1]; işareti spinin işaretiyle aynı.

    Kaynaklar
    ---------
    [Page2005] TeX etiketi eq:3 (Ω, yükü sıfır alınarak).

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: a* = 1 için Ω_H = c³/(2 G M); küçük a* için a* c³/(4 G M).
    """
    m = _pozitif(kutle_kg, "kütle")
    a = _spin(spin)
    return _C**3 / (_G * m) * a / (2.0 * (1.0 + np.sqrt(1.0 - a**2)))


@kaynak(
    "Abramowicz2013",
    yer="'Summary: characteristic radii and frequencies', TeX etiketi photon-radius-kerr",
    teyit="okundu",
    not_="Kaynakta yazıldığı gibi 2 r_G{1 + cos[(2/3) arccos(a*)]} formülü a* = 1 için 4 r_G verir, bu retrograde "
    "yörüngedir; doğru yörünge a* = 1'de r_G olmalı. Bu yüzden burada doğru yörünge için arccos(−a*) kullanıldı: "
    "kaynaktaki a*'ın işareti retrograde için pozitif sayılmış görünüyor. Sonuç aynı kaynaktaki "
    "y³ − 3y + 2a* = 0 kübiğinin (y = √(r/r_G)) kökleriyle testte doğrulanır.",
)
def foton_yorungesi_yaricapi_kerr_rg(spin: ArrayLike) -> Dizi:
    """Kerr kara deliğin ekvatoral dairesel foton yörüngesinin yarıçapını hesaplar.

    r_ph / r_G = 2 {1 + cos[(2/3) arccos(−a*)]}

    Parametreler
    ------------
    spin : float ya da ndarray
        Boyutsuz dönme a_* ∈ [−1, 1]; pozitif değer foton doğru yönde, negatif ters yönde döner.

    Döndürür
    --------
    float ya da ndarray
        r_ph / r_G: a* = 0 için 3, a* = 1 için 1, a* = −1 için 4.

    Kaynaklar
    ---------
    [Abramowicz2013] TeX etiketi photon-radius-kerr (işaret düzeltmesi not_ alanında).

    Geçerlilik
    ----------
    Kerr metriği, ekvator düzlemi. Bu, gölgenin kenarı değil, yörüngenin yarıçapıdır.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: sınır değerler ve y³ − 3y + 2a* = 0 kübiği.
    """
    a = _spin(spin)
    return 2.0 * (1.0 + np.cos((2.0 / 3.0) * np.arccos(-a)))


# --------------------------------------------------------------------------------------
# Hawking termodinamiği
# --------------------------------------------------------------------------------------


@kaynak(
    "Abramowicz2013",
    yer="§1 (Giriş), Hawking sıcaklığı dipnotu: T_H = ħc³/(8π G M k_B)",
    teyit="okundu",
)
@kaynak(
    "Page2005",
    yer="§1; T = κ/(2π) ve TeX etiketi eq:3 (Schwarzschild: κ = 1/(4M), Planck birimlerinde)",
    teyit="okundu",
)
def hawking_sicakligi_k(kutle_kg: ArrayLike) -> Dizi:
    """Dönmeyen kara deliğin Hawking sıcaklığını hesaplar: T_H = ħ c³ / (8π G M k_B).

    Parametreler
    ------------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].

    Döndürür
    --------
    float ya da ndarray
        T_H [K].

    Kaynaklar
    ---------
    [Abramowicz2013] §1 dipnotu (açık SI biçimi); [Page2005] T = κ/(2π), κ = 1/(4M).

    Geçerlilik
    ----------
    Yarı klasik sınır: kütle Planck kütlesinden çok büyük. Arka plan ışınımına göre
    emilim ya da kütle kaybının geri etkisi yok sayılır.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: yıldız kütleli delik için ~1e-8 K (kaynak dipnotu);
    κ/(2π) ile bağımsız hesap; T·M sabit.
    """
    m = _pozitif(kutle_kg, "kütle")
    return _HBAR * _C**3 / (8.0 * np.pi * _G * m * BOLTZMANN.deger)


@kaynak(
    "Page2005",
    yer="TeX etiketi eq:2, S_bh = S_BH ≡ A/4 (Planck birimleri); A = 16π M² (Schwarzschild, eq:3 sonrası paragraf)",
    teyit="okundu",
    not_="Planck birimlerinden SI'ya çeviri (S/k_B = A c³ / (4 G ħ)) tarafımdan yapıldı; kaynakta yazılı değil.",
)
def bekenstein_hawking_entropisi(kutle_kg: ArrayLike) -> Dizi:
    """Dönmeyen kara deliğin Bekenstein-Hawking entropisini Boltzmann sabiti biriminde hesaplar.

    S / k_B = A c³ / (4 G ħ) = 4π G M² / (ħ c),  A = 16π G² M² / c⁴

    Parametreler
    ------------
    kutle_kg : float ya da ndarray
        Kara delik kütlesi [kg].

    Döndürür
    --------
    float ya da ndarray
        S / k_B (boyutsuz). Güneş kütlesi için ~1e77.

    Kaynaklar
    ---------
    [Page2005] TeX etiketi eq:2 (S = A/4) ve Schwarzschild alanı A = 16π M².

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: A/4 ile bağımsız hesap (ufuk alanından); M² ölçeklemesi.
    """
    m = _pozitif(kutle_kg, "kütle")
    alan = 16.0 * np.pi * (_G * m / _C**2) ** 2
    return alan * _C**3 / (4.0 * _G * _HBAR)


@kaynak(
    "Page2005",
    yer="TeX etiketi eq:15",
    teyit="okundu",
    not_="−dM/dt = 1e-5 M⁻² (8,1830 n_1/2 + 3,3638 n_1 + 0,3836 n_2), Planck birimlerinde; n_s iki kutuplanmalı "
    "kütlesiz spin-s tür sayısı. SI'ya çevrim (ħ c⁴/G² çarpanı) bu projede",
)
def hawking_kutle_kaybi_hizi_kg_s(
    kutle_kg: ArrayLike, *, spin_yarim_tur: float = 0.0, spin_bir_tur: float = 1.0, spin_iki_tur: float = 1.0
) -> Dizi:
    """Dönmeyen kara deliğin Hawking ışınımıyla kütle kaybı hızını hesaplar.

    −dM/dt = α ħ c⁴ / (G² M²),  α = 1e-5 (8,1830 n_½ + 3,3638 n_1 + 0,3836 n_2)

    Katsayılar sayısal greybody hesaplarından gelir. Varsayılan türler foton
    (n_1 = 1) ve gravitondur (n_2 = 1).

    Parametreler
    ------------
    kutle_kg : float ya da ndarray
    spin_yarim_tur, spin_bir_tur, spin_iki_tur : float
        İki kutuplanmalı kütlesiz spin-½, spin-1 ve spin-2 tür sayıları.

    Döndürür
    --------
    float ya da ndarray
        Kütle kaybı hızı −dM/dt [kg s⁻¹], pozitif.

    Kaynaklar
    ---------
    [Page2005] eq:15.

    Geçerlilik
    ----------
    Yalnızca kütlesiz türler ve dönmeyen, yüksüz kara delik. Kütleli bir tür,
    Hawking sıcaklığı o türün durgun kütle enerjisine yaklaştığında devreye
    girer; elektronlar için bu yaklaşık 1e13-1e14 kg altındadır (mertebe
    kestirimi, k T_H ~ m_e c²). Nötrinoların hangi kütlede "kütlesiz" sayılacağı
    kütlelerine bağlıdır; varsayılanda sayılmazlar.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: yalnız foton için kara cisim kestiriminin 1,62 katı; M⁻² ölçeklemesi.
    """
    m = _pozitif(kutle_kg, "kütle")
    alfa = 1e-5 * (8.1830 * spin_yarim_tur + 3.3638 * spin_bir_tur + 0.3836 * spin_iki_tur)
    return alfa * _HBAR * _C**4 / (_G**2 * m**2)


@kaynak(
    "Page2005",
    yer="TeX etiketi eq:15",
    teyit="okundu",
    not_="eq:15'in M'ye göre integrali (türetim bu projede): t = G² M³ / (3 α ħ c⁴); α sabit tür sayıları için",
)
def hawking_buharlasma_suresi_s(
    kutle_kg: ArrayLike, *, spin_yarim_tur: float = 0.0, spin_bir_tur: float = 1.0, spin_iki_tur: float = 1.0
) -> Dizi:
    """Hawking ışınımıyla tam buharlaşma süresini greybody katsayılarıyla hesaplar.

    t = G² M³ / (3 α ħ c⁴),  α = 1e-5 (8,1830 n_½ + 3,3638 n_1 + 0,3836 n_2)

    Parametreler
    ------------
    kutle_kg : float ya da ndarray
        Başlangıç kütlesi [kg].
    spin_yarim_tur, spin_bir_tur, spin_iki_tur : float
        Kütlesiz tür sayıları; varsayılan foton ve graviton.

    Döndürür
    --------
    float ya da ndarray
        Buharlaşma süresi [s]. Güneş kütlesi için varsayılan türlerle yaklaşık 1,2e67 yıl.

    Kaynaklar
    ---------
    [Page2005] eq:15.

    Geçerlilik
    ----------
    Tür sayıları buharlaşma boyunca sabit alınır. Gerçekte kütle azaldıkça
    sıcaklık artar ve kütleli türler devreye girer; bu yüzden küçük başlangıç
    kütlelerinde (yaklaşık 1e14 kg altı) gerçek süre buradakinden kısadır ve
    sonuç yalnızca üst sınır niteliğindedir.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: ``hawking_kutle_kaybi_hizi_kg_s`` ile sayısal tümlev; M³ ölçeklemesi.
    """
    m = _pozitif(kutle_kg, "kütle")
    alfa = 1e-5 * (8.1830 * spin_yarim_tur + 3.3638 * spin_bir_tur + 0.3836 * spin_iki_tur)
    return _G**2 * m**3 / (3.0 * alfa * _HBAR * _C**4)


@kaynak(
    "Page2005",
    yer="TeX etiketi eq:3 (A, κ) ve §1 (T = κ/2π)",
    teyit="okundu",
    not_="L = σ T_H⁴ A kara cisim varsayımı ve t = 5120π G² M³ / (ħ c⁴) integrali bu projenin türetimidir; "
    "kaynak yalnızca T_H ve A'yı verir. Kaynağın eq:15 greybody katsayısı yalnız foton için bu kestirimin 1,62 katıdır",
)
def hawking_buharlasma_suresi_kara_cisim_s(kutle_kg: ArrayLike) -> Dizi:
    """Buharlaşma süresinin ders kitabı kara cisim kestirimini hesaplar.

    L = σ_SB T_H⁴ A = ħ c⁶ / (15360π G² M²),  t = 5120π G² M³ / (ħ c⁴)

    Döndürür
    --------
    float ya da ndarray
        Süre [s]. Güneş kütlesi için yaklaşık 2,1e67 yıl.

    Kaynaklar
    ---------
    [Page2005] eq:3, §1.

    Geçerlilik
    ----------
    Yalnızca foton, greybody çarpanı 1. Bu varsayım ışıma hızını yalnız foton
    için bile 1,62 kat küçük verir ve süreyi olduğundan uzun gösterir;
    karşılaştırma ve öğretim amaçlıdır. Hesap için ``hawking_buharlasma_suresi_s`` kullanılmalıdır.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: L = σ T⁴ A ile bağımsız tümlev ve M³ ölçeklemesi.
    """
    m = _pozitif(kutle_kg, "kütle")
    return 5120.0 * np.pi * _G**2 * m**3 / (_HBAR * _C**4)


# --------------------------------------------------------------------------------------
# İkili kara delikler
# --------------------------------------------------------------------------------------


@kaynak("Sathyaprakash2009", yer="TeX etiketi eqn:chirpmassdef, 𝓜 = μ^{3/5} M^{2/5} = ν^{3/5} M", teyit="okundu")
def cirp_kutlesi_kg(kutle1_kg: ArrayLike, kutle2_kg: ArrayLike) -> Dizi:
    """İkili sistemin cıvıltı (chirp) kütlesini hesaplar.

    𝓜 = (m1 m2)^{3/5} / (m1 + m2)^{1/5}

    Parametreler
    ------------
    kutle1_kg, kutle2_kg : float ya da ndarray
        Bileşen kütleleri [kg], pozitif.

    Döndürür
    --------
    float ya da ndarray
        Cıvıltı kütlesi [kg]; eşit kütle m için m·2^{-1/5}.

    Kaynaklar
    ---------
    [Sathyaprakash2009] TeX etiketi eqn:chirpmassdef.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: kaynaktaki μ^{3/5}M^{2/5} ve ν^{3/5}M biçimleriyle uyum;
    tests/ozellik/test_karadelik_ozellik.py: simetri ve üst sınır.
    """
    m1 = _pozitif(kutle1_kg, "kütle 1")
    m2 = _pozitif(kutle2_kg, "kütle 2")
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


@kaynak(
    "Sathyaprakash2009",
    yer="TeX etiketi eqn:circularPM (Ṗ_b, Peters-Mathews 1963'e atfen), f_gw = 2/P_b",
    teyit="okundu",
    not_="Kaynak G = c = 1 birimindedir. f_gw = 2 f_yör (paragraf 'Waveform polarizations.' sonrası metin: ışınım yörünge "
    "frekansının iki katında yayılır); ḟ = −2 Ṗ_b / P_b² dönüşümünü ben yaptım ve testte Ṗ_b denklemiyle sınadım.",
)
def gw_frekans_turevi_hz_s(cirp_kutlesi_kg: ArrayLike, frekans_hz: ArrayLike) -> Dizi:
    """Dairesel ikilinin kütle çekimsel dalga frekansının zamanla artış hızını hesaplar.

    ḟ = (96/5) π^{8/3} (G 𝓜 / c³)^{5/3} f^{11/3}

    Parametreler
    ------------
    cirp_kutlesi_kg : float ya da ndarray
        Cıvıltı kütlesi 𝓜 [kg].
    frekans_hz : float ya da ndarray
        Kütle çekimsel dalga frekansı f [Hz] (yörünge frekansının iki katı).

    Döndürür
    --------
    float ya da ndarray
        df/dt [Hz s^-1].

    Kaynaklar
    ---------
    [Sathyaprakash2009] TeX etiketi eqn:circularPM.

    Geçerlilik
    ----------
    Dairesel yörünge, Newton mertebesinde dördül ışınımı; ISCO'ya yaklaşırken
    yüksek mertebe PN düzeltmeleri önemlidir.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: Ṗ_b denkleminden bağımsız türetim; 𝓜 ve f ölçeklemesi.
    """
    mc = _pozitif(cirp_kutlesi_kg, "cıvıltı kütlesi")
    f = _pozitif(frekans_hz, "frekans")
    return 96.0 / 5.0 * np.pi ** (8.0 / 3.0) * (_G * mc / _C**3) ** (5.0 / 3.0) * f ** (11.0 / 3.0)


@kaynak(
    "Sathyaprakash2009",
    yer="§'Coalescing binaries' paragrafı, Newton mertebesi cıvıltı süresi τ0 (etiketsiz denklem)",
    teyit="okundu",
    not_="Kaynaktaki τ0 = 5/(256 π ν f_L) (π M f_L)^{-5/3} (G = c = 1) 𝓜^{5/3} = ν M^{5/3} ile bu biçime çevrildi. "
    "Peters1964'ün kendi metni okunmadı.",
)
def birlesmeye_kalan_sure_s(cirp_kutlesi_kg: ArrayLike, frekans_hz: ArrayLike) -> Dizi:
    """Dalga frekansı f iken dairesel ikilinin birleşmeye kalan süresini hesaplar.

    τ = (5/256) (π f)^{−8/3} (G 𝓜 / c³)^{−5/3}

    Parametreler
    ------------
    cirp_kutlesi_kg : float ya da ndarray
        Cıvıltı kütlesi 𝓜 [kg].
    frekans_hz : float ya da ndarray
        Kütle çekimsel dalga frekansı f [Hz].

    Döndürür
    --------
    float ya da ndarray
        Kalan süre [s]: frekansın biçimsel olarak ıraksadığı ana kadar.

    Kaynaklar
    ---------
    [Sathyaprakash2009] Newton mertebesi τ0 (etiketsiz denklem, 'Coalescing binaries').

    Geçerlilik
    ----------
    Newton mertebesi, dairesel yörünge; 10 M☉ ve üzeri sistemlerde ISCO'ya yakın
    frekanslarda PN düzeltmeleri süreyi değiştirir.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: ḟ'nin ters tümleviyle ve kaynağın ν biçimiyle uyum.
    """
    mc = _pozitif(cirp_kutlesi_kg, "cıvıltı kütlesi")
    f = _pozitif(frekans_hz, "frekans")
    return 5.0 / 256.0 * (np.pi * f) ** (-8.0 / 3.0) * (_G * mc / _C**3) ** (-5.0 / 3.0)


@kaynak(
    "Sathyaprakash2009",
    yer="§'Coalescing binaries' paragrafı, τ0 (etiketsiz denklem); tersi tarafımdan alındı",
    teyit="okundu",
    not_="τ0 denklemi okundu; f(τ) onun cebirsel tersidir ve testte gidiş-dönüşle sınanır.",
)
def kalan_sureden_gw_frekansi_hz(cirp_kutlesi_kg: ArrayLike, kalan_sure_s: ArrayLike) -> Dizi:
    """Birleşmeye kalan süre τ iken dalga frekansını hesaplar.

    f = (1/π) (5 / 256 τ)^{3/8} (G 𝓜 / c³)^{−5/8}

    Parametreler
    ------------
    cirp_kutlesi_kg : float ya da ndarray
        Cıvıltı kütlesi 𝓜 [kg].
    kalan_sure_s : float ya da ndarray
        Birleşmeye kalan süre [s], pozitif.

    Döndürür
    --------
    float ya da ndarray
        Kütle çekimsel dalga frekansı [Hz].

    Kaynaklar
    ---------
    [Sathyaprakash2009] Newton mertebesi τ0'ın tersi.

    Geçerlilik
    ----------
    Newton mertebesi, dairesel yörünge; ISCO frekansını aşan değerler fiziksel değildir.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: birlesmeye_kalan_sure_s ile gidiş-dönüş;
    tests/ozellik/test_karadelik_ozellik.py: τ azaldıkça f artar.
    """
    mc = _pozitif(cirp_kutlesi_kg, "cıvıltı kütlesi")
    tau = _pozitif(kalan_sure_s, "kalan süre")
    return (5.0 / (256.0 * tau)) ** (3.0 / 8.0) * (_G * mc / _C**3) ** (-5.0 / 8.0) / np.pi


@kaynak(
    "Abramowicz2013",
    yer="'Summary: characteristic radii and frequencies' (TeX etiketi schwarzschild-frequencies), Ω_K² = GM/r³, x_ms = 6",
    teyit="okundu",
)
@kaynak(
    "Sathyaprakash2009",
    yer="paragraf 'Waveform polarizations.' sonrası metin: ışınım en düşük mertebede yörünge frekansının iki katında yayılır",
    teyit="okundu",
    not_="Yörünge frekansı için test parçacığı ISCO'su (6 G M/c²) toplam kütleye uygulanır; eş kütleli ikili için "
    "bu bir sezgisel sınırdır, kaynakta ikili ISCO'su verilmez.",
)
def gw_frekansi_isco_hz(toplam_kutle_kg: ArrayLike) -> Dizi:
    """İkili ISCO'sundaki kütle çekimsel dalga frekansını hesaplar (test parçacığı sınırı).

    f_gw = 2 f_yör = c³ / (6^{3/2} π G M)

    Parametreler
    ------------
    toplam_kutle_kg : float ya da ndarray
        İkilinin toplam kütlesi M [kg].

    Döndürür
    --------
    float ya da ndarray
        f_gw [Hz]; 1 M☉ için yaklaşık 4,4 kHz, M ile ters orantılı.

    Kaynaklar
    ---------
    [Abramowicz2013] Ω_K² = GM/r³ ve x_ms = 6; [Sathyaprakash2009] f_gw = 2 f_yör.

    Geçerlilik
    ----------
    Test parçacığı (ν → 0) sınırı; eş kütleli ikilide gerçek birleşme frekansı bundan farklıdır.

    Doğrulama
    ---------
    tests/birim/test_karadelik.py: isco_yorunge_frekansi_hz'in iki katı; 4,4 kHz (M☉/M).
    """
    return 2.0 * isco_yorunge_frekansi_hz(toplam_kutle_kg)
