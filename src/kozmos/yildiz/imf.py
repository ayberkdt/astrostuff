"""Başlangıç kütle fonksiyonları (IMF): Salpeter, Kroupa ve Chabrier.

Biçimler ve parametreler, ilgili makalelerin arXiv TeX kaynağından 25 Kasım 2026'da
okunmuştur:

- Kroupa: çok parçalı kuvvet yasası, ξ(m) ∝ m^{-α_i} [Kroupa2001] (astro-ph/0009005).
- Chabrier: m ≤ 1 M_⊙ için log-normal, m > 1 M_⊙ için kuvvet yasası, Tablo 1'deki
  "IMF" satırı [Chabrier2003] (astro-ph/0304382).
- Salpeter: tek kuvvet yasası, α = 2,35. Salpeter'in kendi makalesi arXiv'de yoktur;
  eğim değeri [Chabrier2003] ve [Kroupa2001] metinlerinden okunmuştur, Salpeter'in
  özgün makalesi açılmadığı için [Salpeter1955] kaydı ``teyit="bekliyor"`` taşır.

Her IMF için dört işlev vardır: normalleştirilmiş sayı yoğunluğu dN/dm, ortalama kütle,
bir eşiğin üzerindeki kütle oranı ve ters dönüşümle rastgele örnekleme. Kütleler Güneş
kütlesi birimindedir. Yoğunluk, [m_alt, m_ust] aralığında ∫ dN/dm dm = 1 olacak biçimde
normalleştirilir ve aralığın dışında sıfırdır. Bütün integraller kapalı çözümlüdür
(kuvvet yasası parçası: basit kuvvet integrali; log-normal parça: normal dağılım
fonksiyonu); sayısal integrasyon yalnızca testlerde yapılır.

Kroupa parçaları kırılma kütlelerinde sürekli olacak biçimde birleştirilir. Makalede
yalnızca orantılılık yazılıdır; süreklilik varsayımı, makaledeki toplu sayıların
(ortalama kütle, BC ve M cücesi payları) yeniden üretilmesiyle testte sınanır.
Chabrier'in log-normal ve kuvvet yasası katsayıları tablodaki değerlerle verilir ve
m = 1'de yaklaşık yüzde 0,5 süreksizlik taşır (testte ölçülür).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri

from kozmos._kaynak import kaynak

Dizi = NDArray[np.float64]

_LN10 = math.log(10.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

# [Kroupa2001] eq:imf: kırılma kütleleri ve eğimler α_i (m^{-α_i}).
_KROUPA_KIRILMA = (0.01, 0.08, 0.5, 1.0, math.inf)
_KROUPA_ALFA = (0.3, 1.3, 2.3, 2.3)

# [Chabrier2003] Tablo 1 (TeX etiketi table.diskIMF), "IMF" satırı: ξ(log m) = A exp{...} (m <= 1)
# ve ξ(log m) = A m^{-x} (m > 1).
_CHABRIER_A_LOGNORMAL = 0.158
_CHABRIER_MC = 0.079
_CHABRIER_SIGMA = 0.69
_CHABRIER_A_KUVVET = 4.43e-2
_CHABRIER_X = 1.3


def _kuvvet_integrali(katsayi: float, us: float, alt: ArrayLike, ust: ArrayLike) -> Dizi:
    """∫ katsayi · m^us dm, alt'tan ust'e; us = -1 için logaritma."""
    lo = np.asarray(alt, dtype=float)
    hi = np.asarray(ust, dtype=float)
    if abs(us + 1.0) < 1e-12:
        return katsayi * np.log(hi / lo)
    return katsayi * (hi ** (us + 1.0) - lo ** (us + 1.0)) / (us + 1.0)


@dataclass(frozen=True, slots=True)
class _KuvvetParcasi:
    """dN/dm = katsayi · m^{-alfa}, [alt, ust) üzerinde."""

    alt: float
    ust: float
    katsayi: float
    alfa: float

    def yogunluk(self, m: Dizi) -> Dizi:
        return self.katsayi * m**-self.alfa

    def sayi(self, lo: ArrayLike, hi: ArrayLike) -> Dizi:
        return _kuvvet_integrali(self.katsayi, -self.alfa, lo, hi)

    def kutle(self, lo: ArrayLike, hi: ArrayLike) -> Dizi:
        return _kuvvet_integrali(self.katsayi, 1.0 - self.alfa, lo, hi)

    def ters(self, u: Dizi, lo: float, hi: float) -> Dizi:
        q = 1.0 - self.alfa
        if abs(q) < 1e-12:
            return lo * (hi / lo) ** u
        return (lo**q + u * (hi**q - lo**q)) ** (1.0 / q)


@dataclass(frozen=True, slots=True)
class _LogNormalParca:
    """dN/dlog₁₀m = genlik · exp{-(log m - log_mc)² / 2σ²}, [alt, ust) üzerinde."""

    alt: float
    ust: float
    genlik: float
    log_mc: float
    sigma: float

    def _z(self, m: ArrayLike) -> Dizi:
        return (np.log10(np.asarray(m, dtype=float)) - self.log_mc) / self.sigma

    def yogunluk(self, m: Dizi) -> Dizi:
        return self.genlik * np.exp(-0.5 * self._z(m) ** 2) / (m * _LN10)

    def sayi(self, lo: ArrayLike, hi: ArrayLike) -> Dizi:
        return self.genlik * self.sigma * _SQRT_2PI * (ndtr(self._z(hi)) - ndtr(self._z(lo)))

    def kutle(self, lo: ArrayLike, hi: ArrayLike) -> Dizi:
        # ∫ 10^x exp{-(x-xc)²/2σ²} dx: kareyi tamamlayınca ortalaması xc + σ² ln10 olan normal çıkar.
        kayma = self.sigma * _LN10
        carpan = math.exp(_LN10 * self.log_mc + 0.5 * kayma**2)
        return self.genlik * self.sigma * _SQRT_2PI * carpan * (ndtr(self._z(hi) - kayma) - ndtr(self._z(lo) - kayma))

    def ters(self, u: Dizi, lo: float, hi: float) -> Dizi:
        p_lo, p_hi = float(ndtr(self._z(lo))), float(ndtr(self._z(hi)))
        z = ndtri(p_lo + u * (p_hi - p_lo))
        return 10.0 ** (self.log_mc + self.sigma * z)


class _Imf:
    """Parçalı bir IMF'nin [m_alt, m_ust] aralığındaki normalleştirilmiş işlemleri."""

    def __init__(self, parcalar: tuple[_KuvvetParcasi | _LogNormalParca, ...], m_alt: float, m_ust: float):
        if not 0.0 < m_alt < m_ust:
            raise ValueError("0 < m_alt < m_ust olmalı")
        self.m_alt, self.m_ust = float(m_alt), float(m_ust)
        self.dilimler = [
            (p, max(p.alt, m_alt), min(p.ust, m_ust)) for p in parcalar if min(p.ust, m_ust) > max(p.alt, m_alt)
        ]
        self.sayilar = np.array([float(p.sayi(lo, hi)) for p, lo, hi in self.dilimler])
        self.toplam_sayi = float(self.sayilar.sum())

    def yogunluk(self, m: ArrayLike) -> Dizi:
        x = np.asarray(m, dtype=float)
        cikti = np.zeros_like(x)
        for i, (p, lo, hi) in enumerate(self.dilimler):
            son = i == len(self.dilimler) - 1
            maske = (x >= lo) & ((x <= hi) if son else (x < hi))
            if np.any(maske):
                cikti[maske] = p.yogunluk(x[maske])
        return np.asarray(cikti / self.toplam_sayi)[()]

    def _kutle_toplami(self, alt: ArrayLike) -> Dizi:
        a = np.clip(np.asarray(alt, dtype=float), self.m_alt, self.m_ust)
        return sum(p.kutle(np.clip(a, lo, hi), hi) for p, lo, hi in self.dilimler)

    def ortalama_kutle(self) -> float:
        return float(self._kutle_toplami(self.m_alt)) / self.toplam_sayi

    def kutle_orani_ustunde(self, esik: ArrayLike) -> Dizi:
        return np.asarray(self._kutle_toplami(esik) / self._kutle_toplami(self.m_alt))[()]

    def ornekle(self, rng: np.random.Generator, n: int) -> Dizi:
        if n < 0:
            raise ValueError("örnek sayısı eksi olamaz")
        u = rng.random(int(n))
        birikim = np.cumsum(self.sayilar) / self.toplam_sayi
        alt_sinir = np.concatenate(([0.0], birikim[:-1]))
        idx = np.minimum(np.searchsorted(birikim, u, side="right"), len(self.dilimler) - 1)
        yerel = (u - alt_sinir[idx]) / (birikim[idx] - alt_sinir[idx])
        m = np.empty_like(u)
        for i, (p, lo, hi) in enumerate(self.dilimler):
            maske = idx == i
            if np.any(maske):
                m[maske] = p.ters(yerel[maske], lo, hi)
        return m


def _salpeter(alfa: float, m_alt: float, m_ust: float) -> _Imf:
    if alfa <= 0:
        raise ValueError("α pozitif olmalı")
    return _Imf((_KuvvetParcasi(0.0, math.inf, 1.0, alfa),), m_alt, m_ust)


def _kroupa(m_alt: float, m_ust: float) -> _Imf:
    if m_alt < _KROUPA_KIRILMA[0]:
        raise ValueError("Kroupa IMF'i 0,01 M_⊙'in altında tanımlı değildir")
    parcalar = []
    katsayi = 1.0
    for i, alfa in enumerate(_KROUPA_ALFA):
        if i > 0:
            katsayi *= _KROUPA_KIRILMA[i] ** (alfa - _KROUPA_ALFA[i - 1])  # kırılmada süreklilik
        parcalar.append(_KuvvetParcasi(_KROUPA_KIRILMA[i], _KROUPA_KIRILMA[i + 1], katsayi, alfa))
    return _Imf(tuple(parcalar), m_alt, m_ust)


def _chabrier(m_alt: float, m_ust: float) -> _Imf:
    parcalar = (
        _LogNormalParca(0.0, 1.0, _CHABRIER_A_LOGNORMAL, math.log10(_CHABRIER_MC), _CHABRIER_SIGMA),
        # ξ(log m) = A m^{-x}  ⇒  dN/dm = A/ln10 · m^{-(x+1)}
        _KuvvetParcasi(1.0, math.inf, _CHABRIER_A_KUVVET / _LN10, _CHABRIER_X + 1.0),
    )
    return _Imf(parcalar, m_alt, m_ust)


@kaynak(
    "Chabrier2003",
    yer="Sec. 'Mass function' ve 'Functional forms' (TeX etiketi IMF1): α = 2,35, x = 1,35",
    teyit="okundu",
)
@kaynak("Kroupa2001", yer="Sec. '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2,35", teyit="okundu")
@kaynak(
    "Salpeter1955",
    yer="özgün makale (eğim değeri)",
    teyit="bekliyor",
    not_="makale arXiv'de yok ve açılmadı; α = 2,35 [Chabrier2003] ve [Kroupa2001] metinlerinden okundu",
)
def salpeter_dn_dm(kutle_gunes: ArrayLike, m_alt: float = 0.1, m_ust: float = 100.0, alfa: float = 2.35) -> Dizi:
    """Salpeter IMF'inin normalleştirilmiş sayı yoğunluğunu hesaplar.

    dN/dm = C m^{-α},   ∫_{m_alt}^{m_ust} dN/dm dm = 1

    Parametreler
    ------------
    kutle_gunes : float ya da ndarray
        Kütle [M_⊙].
    m_alt, m_ust : float
        Normalleştirme aralığı [M_⊙]; varsayılanlar bu işleve özgü bir seçimdir.
    alfa : float
        Eğim; Salpeter değeri 2,35 (x = α - 1 = 1,35).

    Döndürür
    --------
    float ya da ndarray
        dN/dm [M_⊙^-1]; aralığın dışında 0.

    Kaynaklar
    ---------
    [Chabrier2003] IMF1; [Kroupa2001] Salpeter eğimi; [Salpeter1955] (teyit bekliyor).

    Geçerlilik
    ----------
    [Kroupa2001]: Salpeter 0,4 < m < 10 M_⊙ için α = 2,35 bulmuştur; aralığı aşmak ekstrapolasyondur.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: normalleştirme integrali ve log-log eğimi; tests/ozellik/test_yildiz_ozellik.py.
    """
    return _salpeter(alfa, m_alt, m_ust).yogunluk(kutle_gunes)


@kaynak("Chabrier2003", yer="Sec. 'Mass function' ve 'Functional forms' (TeX etiketi IMF1)", teyit="okundu")
@kaynak("Kroupa2001", yer="Sec. '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2,35", teyit="okundu")
@kaynak("Salpeter1955", yer="özgün makale (eğim değeri)", teyit="bekliyor", not_="makale açılmadı")
def salpeter_ortalama_kutle(m_alt: float = 0.1, m_ust: float = 100.0, alfa: float = 2.35) -> float:
    """Salpeter IMF'inin sayı ağırlıklı ortalama kütlesini hesaplar.

    ⟨m⟩ = ∫ m dN/dm dm / ∫ dN/dm dm

    Parametreler
    ------------
    m_alt, m_ust : float
        Aralık [M_⊙].
    alfa : float
        Eğim.

    Döndürür
    --------
    float
        Ortalama kütle [M_⊙].

    Kaynaklar
    ---------
    [Chabrier2003] IMF1; [Kroupa2001]; [Salpeter1955] (teyit bekliyor).

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: kapalı biçimdeki kuvvet integrali oranıyla; sayısal integralle.
    """
    return _salpeter(alfa, m_alt, m_ust).ortalama_kutle()


@kaynak("Chabrier2003", yer="Sec. 'Mass function' ve 'Functional forms' (TeX etiketi IMF1)", teyit="okundu")
@kaynak("Kroupa2001", yer="Sec. '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2,35", teyit="okundu")
@kaynak("Salpeter1955", yer="özgün makale (eğim değeri)", teyit="bekliyor", not_="makale açılmadı")
def salpeter_kutle_orani_ustunde(
    m_esik: ArrayLike, m_alt: float = 0.1, m_ust: float = 100.0, alfa: float = 2.35
) -> Dizi:
    """Eşiğin üzerindeki yıldızların toplam kütledeki payını hesaplar (Salpeter).

    f(>m_esik) = ∫_{m_esik}^{m_ust} m dN/dm dm / ∫_{m_alt}^{m_ust} m dN/dm dm

    Parametreler
    ------------
    m_esik : float ya da ndarray
        Eşik kütle [M_⊙]; aralığa kırpılır.
    m_alt, m_ust : float
        Aralık [M_⊙].
    alfa : float
        Eğim.

    Döndürür
    --------
    float ya da ndarray
        Kütle oranı ∈ [0, 1]; eşik arttıkça azalır.

    Kaynaklar
    ---------
    [Chabrier2003] IMF1; [Kroupa2001]; [Salpeter1955] (teyit bekliyor).

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: kapalı biçim; tests/ozellik/test_yildiz_ozellik.py: monotonluk.
    """
    return _salpeter(alfa, m_alt, m_ust).kutle_orani_ustunde(m_esik)


@kaynak("Chabrier2003", yer="Sec. 'Mass function' ve 'Functional forms' (TeX etiketi IMF1)", teyit="okundu")
@kaynak("Kroupa2001", yer="Sec. '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2,35", teyit="okundu")
@kaynak("Salpeter1955", yer="özgün makale (eğim değeri)", teyit="bekliyor", not_="makale açılmadı")
def salpeter_ornekle(
    rng: np.random.Generator, n: int, m_alt: float = 0.1, m_ust: float = 100.0, alfa: float = 2.35
) -> Dizi:
    """Salpeter IMF'inden ters dönüşümle rastgele kütle çeker.

    m = [m_alt^{1-α} + u (m_ust^{1-α} - m_alt^{1-α})]^{1/(1-α)},   u ~ U[0, 1)

    Parametreler
    ------------
    rng : numpy.random.Generator
        Rastgele sayı üreteci.
    n : int
        Örnek sayısı.
    m_alt, m_ust : float
        Aralık [M_⊙].
    alfa : float
        Eğim.

    Döndürür
    --------
    ndarray, şekil (n,)
        Kütleler [M_⊙], [m_alt, m_ust] içinde.

    Kaynaklar
    ---------
    [Chabrier2003] IMF1; [Kroupa2001]; [Salpeter1955] (teyit bekliyor).

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: Kolmogorov-Smirnov sınaması (sabit tohum); ortalama kütle uyumu.
    """
    return _salpeter(alfa, m_alt, m_ust).ornekle(rng, n)


@kaynak(
    "Kroupa2001",
    yer="Sec. 'The universal IMF', TeX etiketleri eq:xi ve eq:imf",
    teyit="okundu",
    not_="parçalar kırılmalarda sürekli birleştirilmiştir (makale yalnızca orantılılık yazar); makaledeki toplu sayılarla sınandı",
)
def kroupa_dn_dm(kutle_gunes: ArrayLike, m_alt: float = 0.01, m_ust: float = 50.0) -> Dizi:
    """Kroupa (2001) IMF'inin normalleştirilmiş sayı yoğunluğunu hesaplar.

    dN/dm ∝ m^{-α_i};  α = 0,3 (0,01-0,08), 1,3 (0,08-0,5), 2,3 (0,5-1), 2,3 (≥ 1 M_⊙)

    Parametreler
    ------------
    kutle_gunes : float ya da ndarray
        Kütle [M_⊙].
    m_alt, m_ust : float
        Normalleştirme aralığı [M_⊙]; m_alt >= 0,01. Varsayılan üst sınır makalenin
        ortalama kütle için kullandığı 50 M_⊙'dir.

    Döndürür
    --------
    float ya da ndarray
        dN/dm [M_⊙^-1], ∫ = 1; aralığın dışında 0.

    Kaynaklar
    ---------
    [Kroupa2001] eq:xi, eq:imf.

    Geçerlilik
    ----------
    Makale, 0,08 M_⊙ altında güven aralığının iyi belirlenmediğini söyler (α_0 = 0,3 ± 0,7).
    Tek yıldızlar içindir; çift sistemler için düzeltme içermez.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: ortalama kütle 0,36 M_⊙ ve makaledeki sayı/kütle payları.
    """
    return _kroupa(m_alt, m_ust).yogunluk(kutle_gunes)


@kaynak("Kroupa2001", yer="Sec. 'The universal IMF', TeX etiketi eq:imf ve ortalama kütle cümlesi", teyit="okundu")
def kroupa_ortalama_kutle(m_alt: float = 0.01, m_ust: float = 50.0) -> float:
    """Kroupa IMF'inin sayı ağırlıklı ortalama kütlesini hesaplar.

    Parametreler
    ------------
    m_alt, m_ust : float
        Aralık [M_⊙]; m_alt >= 0,01.

    Döndürür
    --------
    float
        Ortalama kütle [M_⊙]; 0,01-50 M_⊙ için 0,36.

    Kaynaklar
    ---------
    [Kroupa2001] eq:imf ve "⟨m⟩ = 0,36 M_⊙" cümlesi.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: makaledeki 0,36 M_⊙ (iki anlamlı basamak).
    """
    return _kroupa(m_alt, m_ust).ortalama_kutle()


@kaynak("Kroupa2001", yer="Sec. 'The universal IMF', TeX etiketi eq:imf ve kütle payları cümleleri", teyit="okundu")
def kroupa_kutle_orani_ustunde(m_esik: ArrayLike, m_alt: float = 0.01, m_ust: float = 50.0) -> Dizi:
    """Eşiğin üzerindeki yıldızların toplam kütledeki payını hesaplar (Kroupa).

    Parametreler
    ------------
    m_esik : float ya da ndarray
        Eşik kütle [M_⊙]; aralığa kırpılır.
    m_alt, m_ust : float
        Aralık [M_⊙]; m_alt >= 0,01.

    Döndürür
    --------
    float ya da ndarray
        Kütle oranı ∈ [0, 1]; 1 M_⊙ eşiği için yaklaşık 0,5 (makale: kütlenin yarısı 1 M_⊙'in altında).

    Kaynaklar
    ---------
    [Kroupa2001] eq:imf ve kütle payları cümleleri.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: yüzde 50 ve 8 M_⊙ üzeri için yüzde 17 ([Kroupa2001]).
    """
    return _kroupa(m_alt, m_ust).kutle_orani_ustunde(m_esik)


@kaynak("Kroupa2001", yer="Sec. 'The universal IMF', TeX etiketi eq:imf", teyit="okundu")
def kroupa_ornekle(rng: np.random.Generator, n: int, m_alt: float = 0.01, m_ust: float = 50.0) -> Dizi:
    """Kroupa IMF'inden parça parça ters dönüşümle rastgele kütle çeker.

    Tek bir düzgün sayı u, birikimli sayı payına göre bir parçayı ve parça içindeki konumu seçer.

    Parametreler
    ------------
    rng : numpy.random.Generator
        Rastgele sayı üreteci.
    n : int
        Örnek sayısı.
    m_alt, m_ust : float
        Aralık [M_⊙]; m_alt >= 0,01.

    Döndürür
    --------
    ndarray, şekil (n,)
        Kütleler [M_⊙].

    Kaynaklar
    ---------
    [Kroupa2001] eq:imf.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: Kolmogorov-Smirnov sınaması ve örnek ortalaması.
    """
    return _kroupa(m_alt, m_ust).ornekle(rng, n)


@kaynak(
    "Chabrier2003",
    yer="Sec. 'The disk stellar mass function', TeX etiketi IMFdisk ve Tablo 1 (table.diskIMF), 'IMF' satırı",
    teyit="okundu",
    not_="m <= 1: A = 0,158, m_c = 0,079, σ = 0,69; m > 1: A = 4,43e-2, x = 1,3; ξ(log m) tanımıdır, dN/dm = ξ/(m ln10)",
)
def chabrier_dn_dm(kutle_gunes: ArrayLike, m_alt: float = 0.01, m_ust: float = 50.0) -> Dizi:
    """Chabrier (2003) disk IMF'inin normalleştirilmiş sayı yoğunluğunu hesaplar.

    ξ(log m) = 0,158 exp{-(log m - log 0,079)² / (2 · 0,69²)}   (m ≤ 1 M_⊙)
    ξ(log m) = 4,43e-2 m^{-1,3}                                 (m > 1 M_⊙)
    dN/dm = ξ(log m) / (m ln 10)

    Parametreler
    ------------
    kutle_gunes : float ya da ndarray
        Kütle [M_⊙].
    m_alt, m_ust : float
        Normalleştirme aralığı [M_⊙]; varsayılanlar Kroupa ile karşılaştırma için seçilmiştir.

    Döndürür
    --------
    float ya da ndarray
        dN/dm [M_⊙^-1], ∫ = 1; aralığın dışında 0.

    Kaynaklar
    ---------
    [Chabrier2003] IMFdisk ve Tablo 1.

    Geçerlilik
    ----------
    Tek nesneler için disk IMF'idir; çift sistemlerin katsayıları farklıdır (makalenin IMFsys denklemi,
    burada uygulanmadı). Parametre belirsizlikleri makalede verilir (örn. A = 0,158 +0,051/-0,046).

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: normalleştirme, m = 1'de süreklilik (yüzde 1) ve log-normal tepe noktası.
    """
    return _chabrier(m_alt, m_ust).yogunluk(kutle_gunes)


@kaynak("Chabrier2003", yer="Sec. 'The disk stellar mass function', TeX etiketi IMFdisk ve Tablo 1", teyit="okundu")
def chabrier_ortalama_kutle(m_alt: float = 0.01, m_ust: float = 50.0) -> float:
    """Chabrier disk IMF'inin sayı ağırlıklı ortalama kütlesini hesaplar.

    Parametreler
    ------------
    m_alt, m_ust : float
        Aralık [M_⊙].

    Döndürür
    --------
    float
        Ortalama kütle [M_⊙].

    Kaynaklar
    ---------
    [Chabrier2003] IMFdisk ve Tablo 1.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: sayısal integralle (scipy.integrate.quad) 1e-8 bağıl uyum.
    """
    return _chabrier(m_alt, m_ust).ortalama_kutle()


@kaynak("Chabrier2003", yer="Sec. 'The disk stellar mass function', TeX etiketi IMFdisk ve Tablo 1", teyit="okundu")
def chabrier_kutle_orani_ustunde(m_esik: ArrayLike, m_alt: float = 0.01, m_ust: float = 50.0) -> Dizi:
    """Eşiğin üzerindeki yıldızların toplam kütledeki payını hesaplar (Chabrier).

    Parametreler
    ------------
    m_esik : float ya da ndarray
        Eşik kütle [M_⊙]; aralığa kırpılır.
    m_alt, m_ust : float
        Aralık [M_⊙].

    Döndürür
    --------
    float ya da ndarray
        Kütle oranı ∈ [0, 1].

    Kaynaklar
    ---------
    [Chabrier2003] IMFdisk ve Tablo 1.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: sayısal integralle uyum; tests/ozellik/test_yildiz_ozellik.py: monotonluk.
    """
    return _chabrier(m_alt, m_ust).kutle_orani_ustunde(m_esik)


@kaynak("Chabrier2003", yer="Sec. 'The disk stellar mass function', TeX etiketi IMFdisk ve Tablo 1", teyit="okundu")
def chabrier_ornekle(rng: np.random.Generator, n: int, m_alt: float = 0.01, m_ust: float = 50.0) -> Dizi:
    """Chabrier disk IMF'inden ters dönüşümle rastgele kütle çeker.

    Log-normal parçada normal dağılımın tersi (``ndtri``), kuvvet yasası parçasında kapalı ters kullanılır.

    Parametreler
    ------------
    rng : numpy.random.Generator
        Rastgele sayı üreteci.
    n : int
        Örnek sayısı.
    m_alt, m_ust : float
        Aralık [M_⊙].

    Döndürür
    --------
    ndarray, şekil (n,)
        Kütleler [M_⊙].

    Kaynaklar
    ---------
    [Chabrier2003] IMFdisk ve Tablo 1.

    Doğrulama
    ---------
    tests/birim/test_yildiz.py: Kolmogorov-Smirnov sınaması (sabit tohum).
    """
    return _chabrier(m_alt, m_ust).ornekle(rng, n)
