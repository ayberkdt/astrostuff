"""Galaktik ölçekte yıldız oluşum yasaları ve ışınım gücünden hız kalibrasyonları.

Denklemler arXiv TeX kaynaklarından okunmuştur (25 Kasım 2026): Kennicutt
(1998, astro-ph/9712213; denklem numaraları makale metnindeki sıradır),
Kennicutt ve Evans (2012, arXiv:1204.3552, Tablo 1) ve McKee ve Ostriker
(2007, arXiv:0707.3514; TeX etiketleri).

Birimler gökbilimcilerin gerçek birimleridir ve bağımsız değişken adlarında
yazılıdır: yüzey yoğunluğu M☉ pc⁻², yıldız oluşum yüzey yoğunluğu
M☉ yıl⁻¹ kpc⁻², açısal hız km s⁻¹ kpc⁻¹. Işınım gücü W alınır (1 erg/s = 10⁻⁷ W
tam çarpandır).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._kaynak import kaynak
from kozmos.temel.sabitler import CEKIM, GM_GUNES, JULYEN_YILI_SANIYE, PARSEK

Dizi = NDArray[np.float64]

# Güneş kütlesi: GM_☉ / G; GM_☉ nominal IAU değeridir, G CODATA'dır.
_GUNES_KUTLESI_KG = GM_GUNES.deger / CEKIM.deger
_PC_M = PARSEK.deger
_YIL_S = JULYEN_YILI_SANIYE.deger

# Kennicutt ve Evans (2012) Tablo 1: log C_x (Kroupa IMF, Starburst99, güneş metal bolluğu).
# L_x birimleri: FUV, NUV, TIR, 24 µm, 70 µm, 2-10 keV için erg/s; Hα için erg/s;
# 1,4 GHz için erg/s/Hz.
_LOG_C = {
    "FUV": 43.35,
    "NUV": 43.17,
    "Halfa": 41.27,
    "TIR": 43.41,
    "24um": 42.69,
    "70um": 43.23,
    "1.4GHz": 28.20,
    "2-10keV": 39.77,
}


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
    "Kennicutt1998",
    yer="Denk. 4 (metindeki sıra; 'equation (4)' olarak anılır)",
    teyit="okundu",
    not_="Kaynak: Σ_SFR = (2,5 ± 0,7)e-4 (Σ_gaz / 1 M☉ pc⁻²)^{1,4 ± 0,15} M☉ yıl⁻¹ kpc⁻²; "
    "McKee2007 TeX etiketi eq:ks aynı değeri aktarır. Kennicutt2012'ye göre bu makale Salpeter IMF (0,1-100 M☉) varsayar",
)
@kaynak("McKee2007", yer="TeX etiketi eq:ks", teyit="okundu")
def kennicutt_schmidt_sfr_yuzey_yogunlugu(
    gaz_yuzey_yogunlugu_gunes_kutlesi_pc2: ArrayLike, indis: float = 1.4, genlik: float = 2.5e-4
) -> Dizi:
    """Kennicutt-Schmidt yasasını hesaplar: Σ_SFR = A (Σ_gaz / 1 M☉ pc⁻²)^N.

    Parametreler
    ------------
    gaz_yuzey_yogunlugu_gunes_kutlesi_pc2 : float ya da ndarray
        Toplam (HI + H2) gaz yüzey yoğunluğu [M☉ pc⁻²], pozitif.
    indis : float
        Güç yasası indisi N; kaynakta 1,4 ± 0,15.
    genlik : float
        Σ_gaz = 1 M☉ pc⁻² için Σ_SFR [M☉ yıl⁻¹ kpc⁻²]; kaynakta (2,5 ± 0,7)e-4.

    Döndürür
    --------
    float ya da ndarray
        Yıldız oluşum hızı yüzey yoğunluğu [M☉ yıl⁻¹ kpc⁻²].

    Kaynaklar
    ---------
    [Kennicutt1998] Denk. 4; [McKee2007] eq:ks.

    Geçerlilik
    ----------
    Kaynak, gaz bakımından fakir spiral disklerden en parlak yıldız patlaması
    çekirdeklerine kadar tek güç yasasının küresel hızı iyi temsil ettiğini söyler.
    Tek bir yıldız oluşum bölgesi için değil, gökada ölçeğinde ortalama içindir.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: A ve N'nin geri kazanılması; ölçekleme sınaması.
    """
    (s,) = _pozitif(gaz_yuzey_yogunlugu=gaz_yuzey_yogunlugu_gunes_kutlesi_pc2)
    if not (genlik > 0 and indis > 0):
        raise ValueError("genlik ve indis pozitif olmalı")
    return genlik * s**indis


@kaynak(
    "Kennicutt1998",
    yer="Denk. 7 (metindeki sıra; 'equation (7)' olarak anılır)",
    teyit="okundu",
    not_="Kaynak Σ_SFR ≃ 0,017 Σ_gaz Ω_gaz yazar; Ω_gaz diskin gaz dönme hızıdır. "
    "McKee2007 TeX etiketi eq:ks aynı katsayıyı aktarır",
)
@kaynak("McKee2007", yer="TeX etiketi eq:ks", teyit="okundu")
def kennicutt_yorunge_sfr_yuzey_yogunlugu(
    gaz_yuzey_yogunlugu_gunes_kutlesi_pc2: ArrayLike, omega_km_s_kpc: ArrayLike, katsayi: float = 0.017
) -> Dizi:
    """Yörünge zamanlı yıldız oluşum yasasını hesaplar: Σ_SFR = 0,017 Σ_gaz Ω.

    Gaz kütlesinin yörünge başına yaklaşık %10'u (0,017·2π ≈ 0,107) yıldıza döner.

    Parametreler
    ------------
    gaz_yuzey_yogunlugu_gunes_kutlesi_pc2 : float ya da ndarray
        Gaz yüzey yoğunluğu [M☉ pc⁻²], pozitif.
    omega_km_s_kpc : float ya da ndarray
        Gazın açısal dönme hızı Ω [km s⁻¹ kpc⁻¹], pozitif.
    katsayi : float
        Boyutsuz verimlilik katsayısı; kaynakta 0,017.

    Döndürür
    --------
    float ya da ndarray
        Yıldız oluşum hızı yüzey yoğunluğu [M☉ yıl⁻¹ kpc⁻²].

    Kaynaklar
    ---------
    [Kennicutt1998] Denk. 7; [McKee2007] eq:ks.

    Geçerlilik
    ----------
    Kaynak, dış kenarda yörünge başına %21'lik verimi, diskin içindeki ortalama
    yörünge süresi kenarın yaklaşık yarısı olduğundan ≈ %10'a indirir; 0,017 bu
    yerel biçimin katsayısıdır.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: yörünge başına ~%10 verimlilik.
    """
    s, om = _pozitif(gaz_yuzey_yogunlugu=gaz_yuzey_yogunlugu_gunes_kutlesi_pc2, omega=omega_km_s_kpc)
    if not katsayi > 0:
        raise ValueError("katsayı pozitif olmalı")
    om_yil = om * 1e3 / (1e3 * _PC_M) * _YIL_S  # km s^-1 kpc^-1 -> yıl^-1
    return katsayi * (s * 1e6) * om_yil


@kaynak(
    "Krumholz2015",
    yer="Bölüm 8 (Giant Molecular Clouds), Depletion Time: t_dep = Σ_gaz / Σ̇_*",
    teyit="okundu",
    not_="McKee2007 aynı niceliği t_{S*} = M_S / Ṁ_* olarak tanımlar (TeX etiketi eq:epsff öncesi)",
)
def tuketme_zamani_yil(
    gaz_yuzey_yogunlugu_gunes_kutlesi_pc2: ArrayLike, sfr_yuzey_yogunlugu_gunes_kutlesi_yil_kpc2: ArrayLike
) -> Dizi:
    """Gaz tüketme zamanını hesaplar: t_dep = Σ_gaz / Σ_SFR.

    Parametreler
    ------------
    gaz_yuzey_yogunlugu_gunes_kutlesi_pc2 : float ya da ndarray
        Gaz yüzey yoğunluğu [M☉ pc⁻²], pozitif.
    sfr_yuzey_yogunlugu_gunes_kutlesi_yil_kpc2 : float ya da ndarray
        Yıldız oluşum yüzey yoğunluğu [M☉ yıl⁻¹ kpc⁻²], pozitif.

    Döndürür
    --------
    float ya da ndarray
        Tüketme zamanı [yıl].

    Kaynaklar
    ---------
    [Krumholz2015] Bölüm 8. Kaynak, Samanyolu için ≈ 1 Gyr, yakın gökada
    molekül gazı için ≈ 2 Gyr tipik değerini verir.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: Kennicutt-Schmidt yasasıyla Σ^{-0,4} ölçeklemesi.
    """
    s, sfr = _pozitif(
        gaz_yuzey_yogunlugu=gaz_yuzey_yogunlugu_gunes_kutlesi_pc2,
        sfr_yuzey_yogunlugu=sfr_yuzey_yogunlugu_gunes_kutlesi_yil_kpc2,
    )
    return s * 1e6 / sfr


@kaynak("McKee2007", yer="TeX etiketi Qdef", teyit="okundu", not_="Kaynak sıvı (gaz) diski için π'li biçimi kullanır")
def toomre_q_gaz(
    kappa_km_s_kpc: ArrayLike, ses_hizi_km_s: ArrayLike, yuzey_yogunlugu_gunes_kutlesi_pc2: ArrayLike
) -> Dizi:
    """Gaz diski için Toomre parametresini hesaplar: Q = κ c_s / (π G Σ).

    Parametreler
    ------------
    kappa_km_s_kpc : float ya da ndarray
        Epiciklik frekans κ [km s⁻¹ kpc⁻¹], pozitif.
    ses_hizi_km_s : float ya da ndarray
        Gazın ortalama ses hızı (ya da hız dağılımı) [km s⁻¹], pozitif.
    yuzey_yogunlugu_gunes_kutlesi_pc2 : float ya da ndarray
        Gaz yüzey yoğunluğu [M☉ pc⁻²], pozitif.

    Döndürür
    --------
    float ya da ndarray
        Boyutsuz Q. Kaynak, gerçekçi kalınlıklı modellerde bağlı bulutların
        Q < Q_kr ≈ 1,5 iken oluştuğunu aktarır.

    Kaynaklar
    ---------
    [McKee2007] Qdef.

    Geçerlilik
    ----------
    Tek bileşenli, sıvı, ince disk; kalınlık, manyetik alan ve yıldız bileşeni ihmal edilir.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: c_s = 7 km/s, κ = 36 km/s/kpc, Σ = 13 M☉/pc² için Q = 1,4.
    """
    kappa, cs, s = _pozitif(
        kappa=kappa_km_s_kpc, ses_hizi=ses_hizi_km_s, yuzey_yogunlugu=yuzey_yogunlugu_gunes_kutlesi_pc2
    )
    kappa_si = kappa * 1e3 / (1e3 * _PC_M)  # s^-1
    cs_si = cs * 1e3
    s_si = s * _GUNES_KUTLESI_KG / _PC_M**2
    return kappa_si * cs_si / (np.pi * CEKIM.deger * s_si)


@kaynak("McKee2007", yer="TeX etiketi MJ2D (MJ2Ddef)", teyit="okundu")
def iki_boyutlu_jeans_kutlesi_gunes_kutlesi(
    ses_hizi_km_s: ArrayLike, yuzey_yogunlugu_gunes_kutlesi_pc2: ArrayLike
) -> Dizi:
    """İki boyutlu Jeans kütlesini hesaplar: M_J,2D = c_s⁴ / (G² Σ).

    Düzgün yüzey yoğunluklu gaz disklerinde kütleçekimiyle oluşan bulutların
    kütlesi, kaynağa göre bunun birkaç ila on katıdır.

    Parametreler
    ------------
    ses_hizi_km_s : float ya da ndarray
        Gazın ortalama ses hızı [km s⁻¹], pozitif.
    yuzey_yogunlugu_gunes_kutlesi_pc2 : float ya da ndarray
        Gaz yüzey yoğunluğu [M☉ pc⁻²], pozitif.

    Döndürür
    --------
    float ya da ndarray
        Kütle [M☉].

    Kaynaklar
    ---------
    [McKee2007] MJ2D.

    Geçerlilik
    ----------
    Kaynaktaki tanım; disk kalınlığı ve alan etkileri katsayıyı değiştirir.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: c_s = 7 km/s, Σ = 13 M☉/pc² için 1e7 M☉.
    """
    cs, s = _pozitif(ses_hizi=ses_hizi_km_s, yuzey_yogunlugu=yuzey_yogunlugu_gunes_kutlesi_pc2)
    s_si = s * _GUNES_KUTLESI_KG / _PC_M**2
    return (cs * 1e3) ** 4 / (CEKIM.deger**2 * s_si) / _GUNES_KUTLESI_KG


@kaynak("McKee2007", yer="TeX etiketi SFR-ff", teyit="okundu")
def serbest_dusme_sfr_yuzey_yogunlugu(
    gaz_yuzey_yogunlugu_gunes_kutlesi_pc2: ArrayLike,
    serbest_dusme_zamani_yil: ArrayLike,
    ff_verimliligi: float = 0.01,
    bulut_gazi_orani: float = 1.0,
) -> Dizi:
    """Yıldız oluşum hızı yüzey yoğunluğunu bulut serbest düşme zamanından hesaplar.

    Σ_SFR = ε_ff,GMC Σ_gaz f_GMC / t_ff,GMC.

    Parametreler
    ------------
    gaz_yuzey_yogunlugu_gunes_kutlesi_pc2 : float ya da ndarray
        Büyük bir bulut topluluğu üzerinden ortalanmış gaz yüzey yoğunluğu [M☉ pc⁻²], pozitif.
    serbest_dusme_zamani_yil : float ya da ndarray
        Dev moleküler bulutların ortalama yoğunluğundan t_ff [yıl], pozitif.
    ff_verimliligi : float
        ε_ff,GMC, [0, 1]; gözlenen tipik değer ≈ 0,01.
    bulut_gazi_orani : float
        Gazın bulutlarda bulunan kesri f_GMC, (0, 1].

    Döndürür
    --------
    float ya da ndarray
        Yıldız oluşum hızı yüzey yoğunluğu [M☉ yıl⁻¹ kpc⁻²].

    Kaynaklar
    ---------
    [McKee2007] SFR-ff.

    Geçerlilik
    ----------
    Yüzey yoğunluğu bir bulut topluluğunun ortalamasıdır; tek bulutun hızı çok değişkendir.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: ε_ff ve f_GMC ile doğrusal ölçekleme; t_ff^-1 ile ters orantı.
    """
    s, tff = _pozitif(
        gaz_yuzey_yogunlugu=gaz_yuzey_yogunlugu_gunes_kutlesi_pc2, serbest_dusme_zamani=serbest_dusme_zamani_yil
    )
    if not 0.0 <= ff_verimliligi <= 1.0:
        raise ValueError("ε_ff [0, 1] aralığında olmalı")
    if not 0.0 < bulut_gazi_orani <= 1.0:
        raise ValueError("f_GMC (0, 1] aralığında olmalı")
    return ff_verimliligi * (s * 1e6) * bulut_gazi_orani / tff


@kaynak(
    "McKee2007",
    yer="TeX etiketi SFR-lc",
    teyit="okundu",
    not_="Kaynak, Samanyolu'nda gözlenen ortalama ε_GMC ≈ 0,05 verir (t_GMC/t_ff ≈ 5)",
)
def yasam_dongusu_sfr_yuzey_yogunlugu(
    gaz_yuzey_yogunlugu_gunes_kutlesi_pc2: ArrayLike, bulut_omru_verimliligi: float, yasam_dongusu_zamani_yil: ArrayLike
) -> Dizi:
    """Yıldız oluşum hızı yüzey yoğunluğunu gazın yaşam döngüsü süresinden hesaplar.

    Σ_SFR = ε_GMC Σ_gaz / t_lc; burada t_lc = t_yayılık + t_GMC gazın bulutlara
    dönüşüp yıldız oluşturarak dağılmasına kadar geçen süredir.

    Parametreler
    ------------
    gaz_yuzey_yogunlugu_gunes_kutlesi_pc2 : float ya da ndarray
        Gaz yüzey yoğunluğu [M☉ pc⁻²], pozitif.
    bulut_omru_verimliligi : float
        Bulutun ömrü boyunca yıldıza dönüşen kütle oranı ε_GMC, [0, 1].
    yasam_dongusu_zamani_yil : float ya da ndarray
        Yaşam döngüsü süresi t_lc [yıl], pozitif.

    Döndürür
    --------
    float ya da ndarray
        Yıldız oluşum hızı yüzey yoğunluğu [M☉ yıl⁻¹ kpc⁻²].

    Kaynaklar
    ---------
    [McKee2007] SFR-lc.

    Geçerlilik
    ----------
    Kararlı hâl (oluşum ve yıkım hızları eşit) varsayımı.

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: ε_GMC ve Σ_gaz ile doğrusal, t_lc ile ters orantılı.
    """
    s, tlc = _pozitif(gaz_yuzey_yogunlugu=gaz_yuzey_yogunlugu_gunes_kutlesi_pc2, yasam_dongusu=yasam_dongusu_zamani_yil)
    if not 0.0 <= bulut_omru_verimliligi <= 1.0:
        raise ValueError("ε_GMC [0, 1] aralığında olmalı")
    return bulut_omru_verimliligi * (s * 1e6) / tlc


@kaynak(
    "Kennicutt2012",
    yer="Tablo 1 (TeX etiketi sfrequations) ve 'An Updated Compendium of Integrated SFR Calibrations' alt bölümündeki log SFR = log L_x − log C_x denklemi",
    teyit="okundu",
    not_="Kroupa IMF (Kroupa ve Weidner 2003), Starburst99, güneş metal bolluğu. IR bantları sürekli yıldız oluşumu "
    "(0-100 Myr) ve tam sönümleme için; Hα için K98'e göre 0,68, TIR için 0,86 çarpanı. "
    "Birimler: FUV, NUV, Hα, TIR, 24 µm, 70 µm, 2-10 keV için erg/s; 1,4 GHz için erg/s/Hz",
)
def sfr_kalibrasyonu_gunes_kutlesi_yil(isinim_gucu_w: ArrayLike, bant: str) -> Dizi:
    """Işınım gücünden yıldız oluşum hızını hesaplar: log SFR = log L_x − log C_x.

    Parametreler
    ------------
    isinim_gucu_w : float ya da ndarray
        Işınım gücü [W] (1,4 GHz bandı için W/Hz), pozitif. Kaynağın cgs
        birimlerine 1 erg/s = 10⁻⁷ W ile dönüştürülür.
    bant : str
        "FUV", "NUV", "Halfa", "TIR", "24um", "70um", "1.4GHz" ya da "2-10keV".
        FUV ve NUV için L_x = ν L_ν, TIR için 3-1100 µm integrali kullanılır.

    Döndürür
    --------
    float ya da ndarray
        Yıldız oluşum hızı [M☉ yıl⁻¹].

    Kaynaklar
    ---------
    [Kennicutt2012] Tablo 1.

    Geçerlilik
    ----------
    Bütün katsayılar güneş metal bolluğu içindir ve metal bolluğuna duyarlıdır;
    IR bantları sönümlemenin tam olduğu yıldız patlaması sınırındadır ve sönük
    kaynaklarda kopar. Her bandın yaş aralığı farklıdır (Tablo 1, ikinci sütun).

    Doğrulama
    ---------
    tests/birim/test_yildizolusumu.py: Tablo 1 değerleriyle; Hα için SFR = L / 10^41,27.
    """
    if bant not in _LOG_C:
        raise ValueError(f"bilinmeyen bant {bant!r}; geçerli: {sorted(_LOG_C)}")
    (l_w,) = _pozitif(isinim_gucu=isinim_gucu_w)
    l_erg = l_w * 1e7
    return l_erg / 10.0 ** _LOG_C[bant]
