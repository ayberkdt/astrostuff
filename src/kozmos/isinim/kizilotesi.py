"""Kızılötesi fotometri: 2MASS ve WISE Vega sıfır noktaları, kadir-akı dönüşümleri.

Sıfır kadir akı yoğunlukları kaynak makalelerin arXiv TeX dosyalarından
okunmuştur: 2MASS için [Cohen2003] Tablo 2 (F_ν(iso) ve λ(iso)), WISE için
[Wright2010] Bölüm 4.4 (F_ν° ve F_λ°). Wien kayma sabiti CODATA 2022
değeridir [CODATA2022].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._kaynak import altyapi, kaynak
from kozmos.temel.sabitler import WIEN_DALGABOYU

Dizi = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Bant:
    """Bir fotometrik bandın Vega sistemindeki sıfır noktası."""

    ad: str
    dalga_boyu_um: float
    sifir_aki_jy: float
    sifir_aki_belirsizligi_jy: float
    kaynak: str


# [Cohen2003] Tablo 2: λ(iso) [µm], F_ν(iso) [Jy] ve belirsizliği.
# [Wright2010] Bölüm 4.4: F_ν° = 306.681, 170.663, 29.0448, 8.2839 Jy (W1..W4); genel sistematik belirsizlik ±%1,5.
BANTLAR: dict[str, Bant] = {
    "J": Bant("2MASS J", 1.235, 1594.0, 27.80, "Cohen2003"),
    "H": Bant("2MASS H", 1.662, 1024.0, 19.95, "Cohen2003"),
    "Ks": Bant("2MASS Ks", 2.159, 666.7, 12.55, "Cohen2003"),
    "W1": Bant("WISE W1", 3.4, 306.681, 0.015 * 306.681, "Wright2010"),
    "W2": Bant("WISE W2", 4.6, 170.663, 0.015 * 170.663, "Wright2010"),
    "W3": Bant("WISE W3", 12.0, 29.0448, 0.015 * 29.0448, "Wright2010"),
    "W4": Bant("WISE W4", 22.0, 8.2839, 0.015 * 8.2839, "Wright2010"),
}


@altyapi("bant sözlüğünü sorgular; denklem uygulamaz")
def bant(ad: str) -> Bant:
    """Bant adından (``"J"``, ``"H"``, ``"Ks"``, ``"W1"`` ... ``"W4"``) sıfır noktası kaydını döndürür."""
    try:
        return BANTLAR[ad]
    except KeyError:
        raise ValueError(f"bilinmeyen bant {ad!r}; bilinenler: {', '.join(BANTLAR)}") from None


@kaynak("Cohen2003", yer="Tablo 2: F_ν(iso)", teyit="okundu", not_="2MASS J, H, Ks: 1594, 1024, 666.7 Jy")
@kaynak("Wright2010", yer="Bölüm 4.4: F_ν°", teyit="okundu", not_="WISE W1..W4: 306.681, 170.663, 29.0448, 8.2839 Jy")
def vega_kadirinden_aki_jy(kadir: ArrayLike, bant_adi: str) -> Dizi:
    """Vega sistemindeki kadirden akı yoğunluğunu hesaplar: F_ν = F_ν° · 10^(−0,4 m).

    Parametreler
    ------------
    kadir : float ya da ndarray
        Vega kadiri.
    bant_adi : str
        ``"J"``, ``"H"``, ``"Ks"``, ``"W1"``, ``"W2"``, ``"W3"``, ``"W4"``.

    Döndürür
    --------
    float ya da ndarray
        Akı yoğunluğu [Jy].

    Kaynaklar
    ---------
    [Cohen2003] Tablo 2; [Wright2010] Bölüm 4.4.

    Geçerlilik
    ----------
    Sıfır noktaları belirli bir tayf biçimi için tanımlıdır; çok kırmızı ya da
    çok mavi kaynaklarda renk düzeltmesi gerekir. [Wright2010] W3 ve W4 için
    kırmızı kaynaklarda %8 ve %4 düzeyinde düzeltmeden söz eder; burada uygulanmaz.

    Doğrulama
    ---------
    tests/birim/test_kizilotesi.py: sıfır kadirde tablo değeri; 2,5 kadirde onda biri.
    """
    return bant(bant_adi).sifir_aki_jy * 10.0 ** (-0.4 * np.asarray(kadir, dtype=float))


@kaynak("Cohen2003", yer="Tablo 2: F_ν(iso)", teyit="okundu")
@kaynak("Wright2010", yer="Bölüm 4.4: F_ν°", teyit="okundu")
def akidan_vega_kadiri(aki_jy: ArrayLike, bant_adi: str) -> Dizi:
    """Akı yoğunluğundan Vega kadirini hesaplar: m = −2,5 log10(F_ν / F_ν°).

    Kaynaklar
    ---------
    [Cohen2003] Tablo 2; [Wright2010] Bölüm 4.4.

    Doğrulama
    ---------
    tests/birim/test_kizilotesi.py: ``vega_kadirinden_aki_jy`` ile gidiş-dönüş.
    """
    return -2.5 * np.log10(np.asarray(aki_jy, dtype=float) / bant(bant_adi).sifir_aki_jy)


@kaynak(
    "Cohen2003",
    yer="Tablo 2: F_ν(iso)",
    teyit="okundu",
    not_="m_AB = −2,5 log10(F_ν / 3631 Jy) tanımıyla fark: m_AB − m_Vega = 2,5 log10(3631 / F_ν°). "
    "3631 Jy sabitinin künyesi [OkeGunn1983] okunmadı",
)
@kaynak("Wright2010", yer="Bölüm 4.4: F_ν°", teyit="okundu")
def vega_ab_farki(bant_adi: str) -> float:
    """Bir bantta AB ve Vega kadirleri arasındaki farkı hesaplar: m_AB − m_Vega.

    Kaynaklar
    ---------
    [Cohen2003] Tablo 2; [Wright2010] Bölüm 4.4.

    Geçerlilik
    ----------
    AB sıfır noktası 3631 Jy alınır; bu değerin kaynağı bu çalışmada okunmamıştır.

    Doğrulama
    ---------
    tests/birim/test_kizilotesi.py: Ks bandında yaklaşık 1,84 mag (tablo değerinden hesaplanan).
    """
    return float(2.5 * np.log10(3631.0 / bant(bant_adi).sifir_aki_jy))


@kaynak("CODATA2022", yer="Wien wavelength displacement law constant", teyit="okundu", not_="λ_tepe · T = b")
def tepe_dalgaboyu_um(sicaklik_k: ArrayLike) -> Dizi:
    """Kara cisim ışınımının (B_λ) tepe dalga boyunu hesaplar: λ_tepe = b / T.

    Döndürür
    --------
    float ya da ndarray
        Tepe dalga boyu [µm].

    Kaynaklar
    ---------
    [CODATA2022] Wien kayma sabiti.

    Geçerlilik
    ----------
    Tepe, B_λ için tanımlıdır; B_ν'nün tepesi başka bir dalga boyundadır.

    Doğrulama
    ---------
    tests/birim/test_kizilotesi.py: 5772 K için 0,502 µm; 30 K toz için yaklaşık 97 µm.
    """
    return WIEN_DALGABOYU.deger / np.asarray(sicaklik_k, dtype=float) * 1e6


@kaynak("CODATA2022", yer="Wien wavelength displacement law constant", teyit="okundu", not_="T = b / λ_tepe")
def tepe_sicakligi_k(dalga_boyu_um: ArrayLike) -> Dizi:
    """Tepe dalga boyundan kara cisim sıcaklığını hesaplar: T = b / λ_tepe.

    Bir bandın en duyarlı olduğu ısıl kaynak sıcaklığının mertebesini verir:
    2MASS Ks için yaklaşık 1340 K, WISE W4 için yaklaşık 130 K.

    Kaynaklar
    ---------
    [CODATA2022] Wien kayma sabiti.

    Doğrulama
    ---------
    tests/birim/test_kizilotesi.py: ``tepe_dalgaboyu_um`` ile gidiş-dönüş.
    """
    return WIEN_DALGABOYU.deger / (np.asarray(dalga_boyu_um, dtype=float) * 1e-6)
