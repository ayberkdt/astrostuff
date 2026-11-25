"""Kadir, uzaklık modülü ve kara cisim ışınımı.

Bolometrik kadir sıfır noktası IAU 2015 B2 kararından [Mamajek2015], uzaklık
modülü tanımı [Hogg1999] Denk. 25'ten okunmuştur. Planck fonksiyonu
astropy'nin ``BlackBody`` modelinin kaynak kodundan okunmuştur [Astropy2022]. Sabitler CODATA 2022 değerleridir [CODATA2022].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._kaynak import kaynak
from kozmos.temel.sabitler import (
    BOLOMETRIK_SIFIR_ISINIM_GUCU,
    BOLTZMANN,
    ISIK_HIZI,
    PLANCK,
    STEFAN_BOLTZMANN,
)

Dizi = NDArray[np.float64]


@kaynak("Mamajek2015", yer="M_Bol tanımı; L_0 = 3,0128e28 W", teyit="okundu", not_="M_Bol = −2,5 log10(L / L_0)")
def mutlak_bolometrik_kadir(isinim_gucu_w: ArrayLike) -> Dizi:
    """Işınım gücünden mutlak bolometrik kadiri hesaplar.

    M_Bol = −2,5 log10(L / L_0), L_0 = 3,0128 × 10²⁸ W.

    Parametreler
    ------------
    isinim_gucu_w : float ya da ndarray
        Işınım gücü [W], pozitif.

    Kaynaklar
    ---------
    [Mamajek2015] L_0 tanımı.

    Doğrulama
    ---------
    tests/birim/test_kadir.py: nominal Güneş ışınım gücü için 4,74 mag (kararda bildirilen değer).
    """
    return -2.5 * np.log10(np.asarray(isinim_gucu_w, dtype=float) / BOLOMETRIK_SIFIR_ISINIM_GUCU.deger)


@kaynak("Mamajek2015", yer="M_Bol tanımı; L_0 = 3,0128e28 W", teyit="okundu")
def bolometrik_kadirden_isinim_gucu_w(mutlak_bolometrik_kadir_mag: ArrayLike) -> Dizi:
    """Mutlak bolometrik kadirden ışınım gücünü hesaplar: L = L_0 · 10^(−0,4 M_Bol).

    Kaynaklar
    ---------
    [Mamajek2015] L_0 tanımı.

    Doğrulama
    ---------
    tests/ozellik/test_kadir_ozellik.py: ``mutlak_bolometrik_kadir`` ile gidiş-dönüş.
    """
    m = np.asarray(mutlak_bolometrik_kadir_mag, dtype=float)
    return BOLOMETRIK_SIFIR_ISINIM_GUCU.deger * 10.0 ** (-0.4 * m)


@kaynak("Hogg1999", yer="Denk. 25", teyit="okundu", not_="DM = 5 log10(D_L / 10 pc)")
def uzaklik_modulu_pc(uzaklik_pc: ArrayLike) -> Dizi:
    """Parsek cinsinden uzaklıktan uzaklık modülünü hesaplar.

    Kaynaklar
    ---------
    [Hogg1999] Denk. 25.

    Geçerlilik
    ----------
    Sönümleme ve K düzeltmesi içermez.

    Doğrulama
    ---------
    tests/birim/test_kadir.py: 10 pc için 0 mag.
    """
    return 5.0 * np.log10(np.asarray(uzaklik_pc, dtype=float) / 10.0)


@kaynak("Hogg1999", yer="Denk. 25", teyit="okundu")
def uzaklik_modulunden_pc(uzaklik_modulu_mag: ArrayLike) -> Dizi:
    """Uzaklık modülünden parsek cinsinden uzaklığı hesaplar: d = 10^(μ/5 + 1).

    Kaynaklar
    ---------
    [Hogg1999] Denk. 25.

    Doğrulama
    ---------
    tests/ozellik/test_kadir_ozellik.py: ``uzaklik_modulu_pc`` ile gidiş-dönüş.
    """
    return 10.0 ** (np.asarray(uzaklik_modulu_mag, dtype=float) / 5.0 + 1.0)


@kaynak(
    "Astropy2022",
    yer="modeling/physical_models.py, BlackBody.evaluate",
    teyit="okundu",
    not_="bb_nu = 2 h ν³ / (c² expm1(hν/kT)); astropy 8.0.0 kaynak kodu. Ders kitabı künyesi [RybickiLightman1979] okunmadı",
)
def planck_frekans(frekans_hz: ArrayLike, sicaklik_k: ArrayLike) -> Dizi:
    """Kara cismin özgül yeğinliğini frekans başına hesaplar.

    B_ν(T) = (2 h ν³ / c²) / (exp(hν / kT) − 1)

    Döndürür
    --------
    float ya da ndarray
        B_ν [W m⁻² Hz⁻¹ sr⁻¹].

    Kaynaklar
    ---------
    [Astropy2022] BlackBody.evaluate.

    Geçerlilik
    ----------
    ν > 0, T > 0. Küçük hν/kT için ``expm1`` kullanılır; büyük değerlerde üs
    taşması sıfıra iner.

    Doğrulama
    ---------
    tests/birim/test_kadir.py: frekans integrali σT⁴/π; tests/kahin/test_isinim_kahin.py: astropy BlackBody.
    """
    nu = np.asarray(frekans_hz, dtype=float)
    t = np.asarray(sicaklik_k, dtype=float)
    x = PLANCK.deger * nu / (BOLTZMANN.deger * t)
    with np.errstate(over="ignore"):
        return 2.0 * PLANCK.deger * nu**3 / ISIK_HIZI.deger**2 / np.expm1(x)


@kaynak(
    "Astropy2022",
    yer="modeling/physical_models.py, BlackBody.evaluate",
    teyit="okundu",
    not_="B_λ = B_ν c/λ² dönüşümüyle (türetim bu projede): B_λ = (2hc²/λ⁵) / (exp(hc/λkT) − 1)",
)
def planck_dalgaboyu(dalga_boyu_m: ArrayLike, sicaklik_k: ArrayLike) -> Dizi:
    """Kara cismin özgül yeğinliğini dalga boyu başına hesaplar.

    Döndürür
    --------
    float ya da ndarray
        B_λ [W m⁻³ sr⁻¹].

    Kaynaklar
    ---------
    [Astropy2022] BlackBody.evaluate.

    Doğrulama
    ---------
    tests/birim/test_kadir.py: tepe dalga boyu Wien sabitiyle uyuşur [CODATA2022].
    """
    lam = np.asarray(dalga_boyu_m, dtype=float)
    t = np.asarray(sicaklik_k, dtype=float)
    x = PLANCK.deger * ISIK_HIZI.deger / (lam * BOLTZMANN.deger * t)
    with np.errstate(over="ignore"):
        return 2.0 * PLANCK.deger * ISIK_HIZI.deger**2 / lam**5 / np.expm1(x)


@kaynak(
    "CODATA2022",
    yer="Stefan-Boltzmann constant",
    teyit="okundu",
    not_="F = σ T⁴; yasanın kendisi için ders kitabı künyesi bekliyor",
)
def kara_cisim_akisi(sicaklik_k: ArrayLike) -> Dizi:
    """Kara cisim yüzeyinden çıkan toplam akıyı hesaplar: F = σ T⁴ [W m⁻²].

    Kaynaklar
    ---------
    [CODATA2022] Stefan-Boltzmann sabiti.

    Doğrulama
    ---------
    tests/birim/test_kadir.py: nominal Güneş yarıçapı ve etkin sıcaklığından nominal ışınım gücü [Prsa2016].
    """
    return STEFAN_BOLTZMANN.deger * np.asarray(sicaklik_k, dtype=float) ** 4
