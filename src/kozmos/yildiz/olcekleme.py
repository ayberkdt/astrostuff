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

from kozmos._kaynak import kaynak
from kozmos.temel.sabitler import (
    ATOMIK_KUTLE_BIRIMI,
    CEKIM,
    ISIK_HIZI,
    JULYEN_YILI_SANIYE,
    PLANCK,
    STEFAN_BOLTZMANN,
)
from kozmos.yildiz.yapi import lane_emden_kutle_katsayisi

Dizi = NDArray[np.float64]


@kaynak(
    "Prsa2016",
    yer="Tablo 'formu', M_Bol satırı (4πσR² terimi); metin: T_eff Stefan-Boltzmann yasasından",
    teyit="okundu",
)
def isinim_gucu_w(yaricap_m: ArrayLike, etkin_sicaklik_k: ArrayLike) -> Dizi:
    """Küresel kara cisim yıldızın ışınım gücünü hesaplar: L = 4π R² σ T_eff⁴.

    Parametreler
    ------------
    yaricap_m : float ya da ndarray
        Yarıçap [m].
    etkin_sicaklik_k : float ya da ndarray
        Etkin sıcaklık [K].

    Döndürür
    --------
    float ya da ndarray
        Işınım gücü [W].

    Kaynaklar
    ---------
    [Prsa2016] Tablo 'formu' (M_Bol satırındaki 4πσR²) ve T_eff'in Stefan-Boltzmann yasasından türetilmesi.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: nominal Güneş değerleri 3,828e26 W ile yüzde 0,1 içinde uyuşur
    ([Prsa2016]: nominal değerler güncel σ ile birbirine tutarlıdır).
    """
    r = np.asarray(yaricap_m, dtype=float)
    t = np.asarray(etkin_sicaklik_k, dtype=float)
    return 4.0 * np.pi * r**2 * STEFAN_BOLTZMANN.deger * t**4


@kaynak("Prsa2016", yer="metin: T_eff, Stefan-Boltzmann yasasından; Tablo 'formu', M_Bol satırı", teyit="okundu")
def etkin_sicaklik_k(isinim_gucu_w: ArrayLike, yaricap_m: ArrayLike) -> Dizi:
    """Işınım gücü ve yarıçaptan etkin sıcaklığı hesaplar: T_eff = (L / 4π R² σ)^{1/4}.

    Parametreler
    ------------
    isinim_gucu_w : float ya da ndarray
        Işınım gücü [W].
    yaricap_m : float ya da ndarray
        Yarıçap [m].

    Döndürür
    --------
    float ya da ndarray
        Etkin sıcaklık [K].

    Kaynaklar
    ---------
    [Prsa2016] Stefan-Boltzmann yasasından T_eff (metin) ve Tablo 'formu'.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: nominal Güneş için 5772 K'e yüzde 0,1 içinde; ``isinim_gucu_w`` ile gidiş-dönüş.
    """
    isinim = np.asarray(isinim_gucu_w, dtype=float)
    r = np.asarray(yaricap_m, dtype=float)
    return (isinim / (4.0 * np.pi * r**2 * STEFAN_BOLTZMANN.deger)) ** 0.25


@kaynak(
    "Prsa2016",
    yer="Tablo 'formu', 'log of surface gravity' satırı: log g = log(10^6 GM) - 2 log(100 R) [cgs]",
    teyit="okundu",
)
def yuzey_cekimi_m_s2(kutle_kg: ArrayLike, yaricap_m: ArrayLike) -> Dizi:
    """Yüzey çekim ivmesini hesaplar: g = G M / R².

    Parametreler
    ------------
    kutle_kg, yaricap_m : float ya da ndarray
        Kütle [kg] ve yarıçap [m].

    Döndürür
    --------
    float ya da ndarray
        Yüzey çekimi [m s^-2].

    Kaynaklar
    ---------
    [Prsa2016] Tablo 'formu' (log g satırı, g = GM/R²'nin cgs logaritması).

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: nominal Güneş için log g = 4,438068 ([Prsa2016]).
    """
    m = np.asarray(kutle_kg, dtype=float)
    r = np.asarray(yaricap_m, dtype=float)
    return CEKIM.deger * m / r**2


@kaynak(
    "Prsa2016",
    yer="Tablo 'formu', 'log of surface gravity' satırı: 4,438068... + log M - 2 log R",
    teyit="okundu",
)
def log_yuzey_cekimi_cgs(kutle_kg: ArrayLike, yaricap_m: ArrayLike) -> Dizi:
    """Yüzey çekiminin cgs logaritmasını hesaplar: log₁₀ g [cm s^-2].

    Parametreler
    ------------
    kutle_kg, yaricap_m : float ya da ndarray
        Kütle [kg] ve yarıçap [m].

    Döndürür
    --------
    float ya da ndarray
        log g (cgs); Güneş için 4,438.

    Kaynaklar
    ---------
    [Prsa2016] Tablo 'formu', log g satırı.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: nominal Güneş için 4,438068 (altı ondalık, [Prsa2016]).
    """
    return np.log10(100.0 * yuzey_cekimi_m_s2(kutle_kg, yaricap_m))


@kaynak("Bludman2010", yer="Sec. I, Tablo I öncesi metin: ρ̄_n = 3M/4πR³", teyit="okundu")
def ortalama_yogunluk_kg_m3(kutle_kg: ArrayLike, yaricap_m: ArrayLike) -> Dizi:
    """Küresel cismin ortalama yoğunluğunu hesaplar: ρ̄ = 3M / 4π R³.

    Parametreler
    ------------
    kutle_kg, yaricap_m : float ya da ndarray
        Kütle [kg] ve yarıçap [m].

    Döndürür
    --------
    float ya da ndarray
        Ortalama yoğunluk [kg m^-3].

    Kaynaklar
    ---------
    [Bludman2010] ρ̄_n tanımı.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: nominal Güneş için yaklaşık 1,41 g cm^-3.
    """
    m = np.asarray(kutle_kg, dtype=float)
    r = np.asarray(yaricap_m, dtype=float)
    return 3.0 * m / (4.0 * np.pi * r**3)


@kaynak("McKee2007", yer="TeX etiketi eq:tff", teyit="okundu")
def serbest_dusme_zamani_s(yogunluk_kg_m3: ArrayLike) -> Dizi:
    """Basınçsız homojen kürenin serbest düşme (dinamik) zamanını hesaplar.

    t_ff = ( 3π / (32 G ρ̄) )^{1/2}

    Parametreler
    ------------
    yogunluk_kg_m3 : float ya da ndarray
        Ortalama yoğunluk [kg m^-3], pozitif.

    Döndürür
    --------
    float ya da ndarray
        Serbest düşme zamanı [s].

    Kaynaklar
    ---------
    [McKee2007] eq:tff.

    Geçerlilik
    ----------
    Yıldız için dinamik zaman ölçeği olarak ρ̄ ile kullanılır; basınç ve yoğunluk
    dağılımı yoksayılır.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: n_H = 10³ cm^-3 için 1,37e6 yıl ([McKee2007] metnindeki sayı).
    """
    rho = np.asarray(yogunluk_kg_m3, dtype=float)
    if np.any(rho <= 0):
        raise ValueError("yoğunluk pozitif olmalı")
    return np.sqrt(3.0 * np.pi / (32.0 * CEKIM.deger * rho))


@kaynak(
    "Zinnecker2007",
    yer="Sec. 'The Accretion Phase / Formation of the Hydrostatic Core' metni ve GLOSSARY: t_KH = G M²/(R L)",
    teyit="okundu",
    not_="makale τ_KH ~ G M²/(R L) biçiminde mertebe olarak yazar; öncül 1 alınmıştır",
)
def kelvin_helmholtz_zaman_olcegi_s(kutle_kg: ArrayLike, yaricap_m: ArrayLike, isinim_gucu_w: ArrayLike) -> Dizi:
    """Kelvin-Helmholtz (ısıl) zaman ölçeğini hesaplar: t_KH = G M² / (R L).

    Parametreler
    ------------
    kutle_kg, yaricap_m : float ya da ndarray
        Kütle [kg] ve yarıçap [m].
    isinim_gucu_w : float ya da ndarray
        Işınım gücü [W], pozitif.

    Döndürür
    --------
    float ya da ndarray
        Zaman ölçeği [s].

    Kaynaklar
    ---------
    [Zinnecker2007] t_KH tanımı.

    Geçerlilik
    ----------
    Mertebe tahminidir; öncül (G M²/R) yerçekimi enerjisinin tek başına ölçeğidir.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: Güneş için 10⁷ ile 10⁸ yıl arasında ([Zinnecker2007]: güneş tipi yıldızlar için ~10⁷ yıl).
    """
    m = np.asarray(kutle_kg, dtype=float)
    r = np.asarray(yaricap_m, dtype=float)
    isinim = np.asarray(isinim_gucu_w, dtype=float)
    if np.any(isinim <= 0):
        raise ValueError("ışınım gücü pozitif olmalı")
    return CEKIM.deger * m**2 / (r * isinim)


@kaynak(
    "Kippenhahn2012",
    yer="nükleer zaman ölçeği tanımı (bölüm ve denklem numarasına bakılmadı)",
    teyit="bekliyor",
    not_="kitap bu görevde açılmadı; t = verim · yakıt payı · M c² / L genel tanımdır ve ana kol ömrü tahmini için kullanılır",
)
def nukleer_zaman_olcegi_yil(kutle_kg: ArrayLike, isinim_gucu_w: ArrayLike, verim: float, yakit_orani: float) -> Dizi:
    """Nükleer zaman ölçeğini ve ana kol ömrü tahminini hesaplar.

    t_nuc = ε f M c² / L

    Parametreler
    ------------
    kutle_kg : float ya da ndarray
        Yıldız kütlesi [kg].
    isinim_gucu_w : float ya da ndarray
        Ortalama ışınım gücü [W], pozitif.
    verim : float
        ε, yakıtın durgun kütlesinin ışınıma dönüşen payı (0 ile 1 arası); değer kullanıcıdan alınır.
    yakit_orani : float
        f, yanan çekirdeğin toplam kütleye oranı (0 ile 1 arası).

    Döndürür
    --------
    float ya da ndarray
        Zaman ölçeği [yıl].

    Kaynaklar
    ---------
    [Kippenhahn2012] (teyit bekliyor: bu görevde kitap açılmadı).

    Geçerlilik
    ----------
    ε ve f için varsayılan sayı verilmemiştir, çünkü okunmuş kaynağı yoktur.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: kütleyle doğrusal, ışınım gücüyle ters orantı; tanım gereği kapalı değer.
    """
    if not (0.0 < verim < 1.0 and 0.0 < yakit_orani <= 1.0):
        raise ValueError("verim (0,1) ve yakıt_orani (0,1] aralığında olmalı")
    m = np.asarray(kutle_kg, dtype=float)
    isinim = np.asarray(isinim_gucu_w, dtype=float)
    if np.any(isinim <= 0):
        raise ValueError("ışınım gücü pozitif olmalı")
    return verim * yakit_orani * m * ISIK_HIZI.deger**2 / isinim / JULYEN_YILI_SANIYE.deger


@kaynak(
    "Paxton2013",
    yer="Sec. 'Rotationally-Enhanced Mass Loss' (TeX etiketi s.rotation-mass-loss): L_Edd = 4π c G M/κ",
    teyit="okundu",
)
@kaynak("Bludman2010", yer="Sec. II.B, Eddington standart modeli: L_Edd := 4π c G M/κ_p", teyit="okundu")
def eddington_isinim_gucu_w(kutle_kg: ArrayLike, opaklik_m2_kg: ArrayLike) -> Dizi:
    """Eddington ışınım gücü sınırını hesaplar: L_Edd = 4π c G M / κ.

    Parametreler
    ------------
    kutle_kg : float ya da ndarray
        Kütle [kg].
    opaklik_m2_kg : float ya da ndarray
        κ, kütle başına opaklık [m² kg^-1], pozitif.

    Döndürür
    --------
    float ya da ndarray
        L_Edd [W].

    Kaynaklar
    ---------
    [Paxton2013] s.rotation-mass-loss; [Bludman2010] Eddington standart modeli.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: kütleyle doğrusal, κ ile ters orantı; kapalı değer.
    """
    m = np.asarray(kutle_kg, dtype=float)
    kappa = np.asarray(opaklik_m2_kg, dtype=float)
    if np.any(kappa <= 0):
        raise ValueError("opaklık pozitif olmalı")
    return 4.0 * np.pi * ISIK_HIZI.deger * CEKIM.deger * m / kappa


@kaynak("Paxton2013", yer="mdotomega etiketli denklemi izleyen cümle: Ω_crit² = (1 - L/L_Edd) G M / R³", teyit="okundu")
def kritik_acisal_hiz_rad_s(
    kutle_kg: ArrayLike, yaricap_m: ArrayLike, isinim_gucu_w: ArrayLike, opaklik_m2_kg: ArrayLike
) -> Dizi:
    """Işınım basıncıyla azaltılmış kritik (kopma) açısal hızı hesaplar.

    Ω_crit² = (1 - L / L_Edd) G M / R³

    Parametreler
    ------------
    kutle_kg, yaricap_m : float ya da ndarray
        Kütle [kg] ve yarıçap [m].
    isinim_gucu_w : float ya da ndarray
        Işınım gücü [W]; L_Edd'i aşamaz.
    opaklik_m2_kg : float ya da ndarray
        Yüzey opaklığı κ [m² kg^-1].

    Döndürür
    --------
    float ya da ndarray
        Ω_crit [rad s^-1]. L = 0 için Kepler açısal hızı, L → L_Edd için sıfır.

    Kaynaklar
    ---------
    [Paxton2013] Ω_crit tanımı.

    Geçerlilik
    ----------
    L > L_Edd için tanım yoktur; ``ValueError`` verilir.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: L = 0 ve L = L_Edd sınır değerleri.
    """
    m = np.asarray(kutle_kg, dtype=float)
    r = np.asarray(yaricap_m, dtype=float)
    oran = np.asarray(isinim_gucu_w, dtype=float) / eddington_isinim_gucu_w(m, opaklik_m2_kg)
    if np.any(oran > 1.0):
        raise ValueError("L, Eddington sınırını aşıyor; kritik açısal hız tanımsız")
    return np.sqrt((1.0 - oran) * CEKIM.deger * m / r**3)


@kaynak(
    "Bludman2010",
    yer="Sec. II.B: M_* := 3√10 0ω_3/π³ (hc/(G m_H^{4/3}))^{3/2} = 18,3 M_⊙",
    teyit="okundu",
    not_="m_H, metinde atomik kütle birimi olarak tanımlanır; 0ω_3 sayısal çözümden gelir",
)
def eddington_olcek_kutlesi_kg() -> float:
    """Eddington dördüncü derece denkleminin ölçek kütlesi M_*'ı hesaplar.

    M_* = 3√10 ω_3 / π³ · ( h c / (G m_H^{4/3}) )^{3/2}

    Döndürür
    --------
    float
        M_* [kg]; yaklaşık 18,3 Güneş kütlesi.

    Kaynaklar
    ---------
    [Bludman2010] Sec. II.B, M_* tanımı.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: 18,3 M_⊙ ([Bludman2010]) yüzde 0,5 içinde.
    """
    omega3 = lane_emden_kutle_katsayisi(3.0)
    mh = ATOMIK_KUTLE_BIRIMI.deger
    return float(
        3.0
        * np.sqrt(10.0)
        * omega3
        / np.pi**3
        * (PLANCK.deger * ISIK_HIZI.deger / (CEKIM.deger * mh ** (4.0 / 3.0))) ** 1.5
    )


@kaynak("Bludman2010", yer="Sec. II.B, Eddington dördüncü derece denklemi: (1-β)/β⁴ = (M μ²/M_*)²", teyit="okundu")
def eddington_beta(kutle_kg: ArrayLike, ortalama_molekul_agirligi: float) -> Dizi:
    """Eddington standart modelinde gaz basıncının toplam basınca oranı β'yı çözer.

    (1 - β) / β⁴ = ( M μ² / M_* )²

    Parametreler
    ------------
    kutle_kg : float ya da ndarray
        Yıldız kütlesi [kg], pozitif.
    ortalama_molekul_agirligi : float
        μ (m_H biriminde), pozitif.

    Döndürür
    --------
    float ya da ndarray
        β ∈ (0, 1]; küçük kütlede 1'e, büyük kütlede 0'a gider.

    Kaynaklar
    ---------
    [Bludman2010] Eddington dördüncü derece denklemi.

    Geçerlilik
    ----------
    Sabit radyasyon entropili (Eddington standart modeli, n = 3) yıldız varsayılır.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: denklemin kalıntısı; β'nın kütleyle monoton azalması.
    """
    if ortalama_molekul_agirligi <= 0:
        raise ValueError("μ pozitif olmalı")
    y = np.asarray(kutle_kg, dtype=float) * ortalama_molekul_agirligi**2 / eddington_olcek_kutlesi_kg()
    if np.any(y < 0):
        raise ValueError("kütle eksi olamaz")

    def kok(y_tek: float) -> float:
        return float(brentq(lambda b: 1.0 - b - y_tek**2 * b**4, 0.0, 1.0, xtol=1e-15, rtol=1e-15))

    return np.asarray(np.vectorize(kok, otypes=[float])(y))[()]


@kaynak("Bludman2010", yer="Sec. II.B: L = L_Edd [1 - β(M)]", teyit="okundu")
def standart_model_isinim_gucu_w(kutle_kg: ArrayLike, ortalama_molekul_agirligi: float, opaklik_m2_kg: float) -> Dizi:
    """Eddington standart modelinin ışınım gücünü hesaplar: L = L_Edd (1 - β).

    Parametreler
    ------------
    kutle_kg : float ya da ndarray
        Yıldız kütlesi [kg], pozitif.
    ortalama_molekul_agirligi : float
        μ (m_H biriminde).
    opaklik_m2_kg : float
        Fotosfer opaklığı κ [m² kg^-1].

    Döndürür
    --------
    float ya da ndarray
        Işınım gücü [W]; küçük kütlede M³ ile artar, büyük kütlede L_Edd'e yaklaşır.

    Kaynaklar
    ---------
    [Bludman2010] L = L_Edd [1 - β(M)].

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: L/L_Edd = 1 - β; küçük kütlede M³ yasası ve 0,003 ≈ (1/18,3)² katsayısı.
    """
    beta = eddington_beta(kutle_kg, ortalama_molekul_agirligi)
    return eddington_isinim_gucu_w(kutle_kg, opaklik_m2_kg) * (1.0 - beta)


@kaynak(
    "Bludman2010",
    yer="Sec. II.A (göreli dejenere yıldızlar): K_WD = (hc/8)(3/π)^{1/3}(m_H μ_e)^{-4/3}",
    teyit="okundu",
    not_="m_H, metinde atomik kütle birimi olarak tanımlanır",
)
def beyaz_cuce_k_katsayisi_si(elektron_basina_molekul_agirligi: float) -> float:
    """Göreli dejenere elektron gazının politrop sabitini hesaplar: P = K ρ^{4/3}.

    K_WD = (h c / 8) (3/π)^{1/3} (m_H μ_e)^{-4/3}

    Parametreler
    ------------
    elektron_basina_molekul_agirligi : float
        μ_e, elektron başına ortalama kütle (m_H biriminde); He ve C/O için 2.

    Döndürür
    --------
    float
        K [Pa m⁴ kg^{-4/3}].

    Kaynaklar
    ---------
    [Bludman2010] K_WD ifadesi.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: μ_e^{-4/3} ölçeklenmesi; Chandrasekhar kütlesiyle birlikte 5,83/μ_e² M_⊙.
    """
    if elektron_basina_molekul_agirligi <= 0:
        raise ValueError("μ_e pozitif olmalı")
    return float(
        PLANCK.deger
        * ISIK_HIZI.deger
        / 8.0
        * (3.0 / np.pi) ** (1.0 / 3.0)
        * (ATOMIK_KUTLE_BIRIMI.deger * elektron_basina_molekul_agirligi) ** (-4.0 / 3.0)
    )


@kaynak(
    "Bludman2010",
    yer="Sec. II (giriş) M = 4π 0ω_3 (K/πG)^{3/2}; Sec. II.A M_Ch = 5,83 M_⊙/μ_e² = 1,46 M_⊙",
    teyit="okundu",
    not_="0ω_3 = -ξ_1² θ'(ξ_1) (n = 3) sayısal çözümden gelir; metindeki 5,83 ile karşılaştırılır",
)
def chandrasekhar_kutlesi_kg(elektron_basina_molekul_agirligi: float) -> float:
    """Chandrasekhar kütlesini hesaplar: n = 3 politropun kütlesi yalnızca K'ye bağlıdır.

    M_Ch = 4π ω_3 ( K_WD / (π G) )^{3/2}

    Parametreler
    ------------
    elektron_basina_molekul_agirligi : float
        μ_e (m_H biriminde); He ve C/O için 2.

    Döndürür
    --------
    float
        M_Ch [kg]; μ_e = 2 için yaklaşık 1,46 Güneş kütlesi.

    Kaynaklar
    ---------
    [Bludman2010] Sec. II ve II.A.

    Geçerlilik
    ----------
    Göreli olmayan düzeltmeler, Coulomb etkileri ve genel görelilik yoksayılır.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: 5,83/μ_e² M_⊙ ve μ_e = 2 için 1,46 M_⊙ ([Bludman2010]).
    """
    k = beyaz_cuce_k_katsayisi_si(elektron_basina_molekul_agirligi)
    omega3 = lane_emden_kutle_katsayisi(3.0)
    return float(4.0 * np.pi * omega3 * (k / (np.pi * CEKIM.deger)) ** 1.5)
