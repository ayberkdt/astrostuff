"""Başlangıç kütle fonksiyonları (IMF): Salpeter, Kroupa ve Chabrier.

Biçimler ve parametreler, ilgili makalelerin arXiv TeX kaynağından 25 Kasım 2026'da
okunmuştur:

- Kroupa: çok parçalı kuvvet yasası, ξ(m) ∝ m^{-α_i} [Kroupa2001] (astro-ph/0009005).
- Chabrier: m ≤ 1 M_⊙ için log-normal, m > 1 M_⊙ için kuvvet yasası, Tablo 1'deki
  "IMF" satırı [Chabrier2003] (astro-ph/0304382).
- Salpeter: tek kuvvet yasası, α = 2,35. Özgün makale ADS taramasından okunmuştur
  [Salpeter1955, Denk. 5]: ξ ≈ 0,03 (M/M_⊙)^−1,35 logaritmik kütle aralığı başınadır
  ve log(M/M_⊙) −0,4 ile +1,0 arasında verilmiştir; bu aralığın dışı dışdeğerlemedir.

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

İki sınır notu. Chabrier fonksiyonunda tam m = 1 noktası kuvvet yasası dalına düşer ([Chabrier2003]
Tablo 1 m <= 1'i log-normal dala verir); o tek noktada yoğunluk %0,47 farklıdır, tümlevler etkilenmez.
[Kroupa2001] ayrıca çözülmemiş çiftler için düzeltilmiş bir biçim verir (eq:imfr: α1 = 1,8, α2 = 2,7);
burada makalenin tek yıldızlar için verdiği, çift yanlılığı düzeltilmemiş biçimi (eq:imf) uygulanmıştır.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri

from kozmos._sources import source

Array = NDArray[np.float64]

_LN10 = math.log(10.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

# [Kroupa2001] eq:imf: kırılma kütleleri ve eğimler α_i (m^{-α_i}).
_KROUPA_BREAKS = (0.01, 0.08, 0.5, 1.0, math.inf)
_KROUPA_ALPHA = (0.3, 1.3, 2.3, 2.3)

# [Chabrier2003] Tablo 1 (TeX etiketi table.diskIMF), "IMF" satırı: ξ(log m) = A exp{...} (m <= 1)
# ve ξ(log m) = A m^{-x} (m > 1).
_CHABRIER_A_LOGNORMAL = 0.158
_CHABRIER_MC = 0.079
_CHABRIER_SIGMA = 0.69
_CHABRIER_A_POWER_LAW = 4.43e-2
_CHABRIER_X = 1.3


def _power_law_integral(coefficient: float, exponent: float, low: ArrayLike, high: ArrayLike) -> Array:
    """∫ katsayi · m^us dm, alt'tan ust'e; us = -1 için logaritma."""
    lo = np.asarray(low, dtype=float)
    hi = np.asarray(high, dtype=float)
    if abs(exponent + 1.0) < 1e-12:
        return coefficient * np.log(hi / lo)
    return coefficient * (hi ** (exponent + 1.0) - lo ** (exponent + 1.0)) / (exponent + 1.0)


@dataclass(frozen=True, slots=True)
class _PowerLawSegment:
    """dN/dm = katsayi · m^{-alfa}, [alt, ust) üzerinde."""

    low: float
    high: float
    coefficient: float
    alpha: float

    def density(self, m: Array) -> Array:
        return self.coefficient * m**-self.alpha

    def number(self, lo: ArrayLike, hi: ArrayLike) -> Array:
        return _power_law_integral(self.coefficient, -self.alpha, lo, hi)

    def mass(self, lo: ArrayLike, hi: ArrayLike) -> Array:
        return _power_law_integral(self.coefficient, 1.0 - self.alpha, lo, hi)

    def inverse(self, u: Array, lo: float, hi: float) -> Array:
        q = 1.0 - self.alpha
        if abs(q) < 1e-12:
            return lo * (hi / lo) ** u
        return (lo**q + u * (hi**q - lo**q)) ** (1.0 / q)


@dataclass(frozen=True, slots=True)
class _LogNormalSegment:
    """dN/dlog₁₀m = genlik · exp{-(log m - log_mc)² / 2σ²}, [alt, ust) üzerinde."""

    low: float
    high: float
    amplitude: float
    log_mc: float
    sigma: float

    def _z(self, m: ArrayLike) -> Array:
        return (np.log10(np.asarray(m, dtype=float)) - self.log_mc) / self.sigma

    def density(self, m: Array) -> Array:
        return self.amplitude * np.exp(-0.5 * self._z(m) ** 2) / (m * _LN10)

    def number(self, lo: ArrayLike, hi: ArrayLike) -> Array:
        return self.amplitude * self.sigma * _SQRT_2PI * (ndtr(self._z(hi)) - ndtr(self._z(lo)))

    def mass(self, lo: ArrayLike, hi: ArrayLike) -> Array:
        # ∫ 10^x exp{-(x-xc)²/2σ²} dx: kareyi tamamlayınca ortalaması xc + σ² ln10 olan normal çıkar.
        kayma = self.sigma * _LN10
        carpan = math.exp(_LN10 * self.log_mc + 0.5 * kayma**2)
        return (
            self.amplitude * self.sigma * _SQRT_2PI * carpan * (ndtr(self._z(hi) - kayma) - ndtr(self._z(lo) - kayma))
        )

    def inverse(self, u: Array, lo: float, hi: float) -> Array:
        p_lo, p_hi = float(ndtr(self._z(lo))), float(ndtr(self._z(hi)))
        z = ndtri(p_lo + u * (p_hi - p_lo))
        return 10.0 ** (self.log_mc + self.sigma * z)


class _Imf:
    """Parçalı bir IMF'nin [m_alt, m_ust] aralığındaki normalleştirilmiş işlemleri."""

    def __init__(self, parts: tuple[_PowerLawSegment | _LogNormalSegment, ...], m_low: float, m_high: float):
        if not 0.0 < m_low < m_high:
            raise ValueError("0 < m_alt < m_ust olmalı")
        self.m_low, self.m_high = float(m_low), float(m_high)
        self.dilimler = [
            (p, max(p.low, m_low), min(p.high, m_high)) for p in parts if min(p.high, m_high) > max(p.low, m_low)
        ]
        self.sayilar = np.array([float(p.number(lo, hi)) for p, lo, hi in self.dilimler])
        self.toplam_sayi = float(self.sayilar.sum())

    def density(self, m: ArrayLike) -> Array:
        x = np.asarray(m, dtype=float)
        cikti = np.zeros_like(x)
        for i, (p, lo, hi) in enumerate(self.dilimler):
            son = i == len(self.dilimler) - 1
            mask = (x >= lo) & ((x <= hi) if son else (x < hi))
            if np.any(mask):
                cikti[mask] = p.density(x[mask])
        return np.asarray(cikti / self.toplam_sayi)[()]

    def _total_mass(self, low: ArrayLike) -> Array:
        a = np.clip(np.asarray(low, dtype=float), self.m_low, self.m_high)
        return sum(p.mass(np.clip(a, lo, hi), hi) for p, lo, hi in self.dilimler)

    def mean_mass(self) -> float:
        return float(self._total_mass(self.m_low)) / self.toplam_sayi

    def mass_fraction_above(self, threshold: ArrayLike) -> Array:
        return np.asarray(self._total_mass(threshold) / self._total_mass(self.m_low))[()]

    def sample(self, rng: np.random.Generator, n: int) -> Array:
        if n < 0:
            raise ValueError("örnek sayısı eksi olamaz")
        u = rng.random(int(n))
        birikim = np.cumsum(self.sayilar) / self.toplam_sayi
        alt_sinir = np.concatenate(([0.0], birikim[:-1]))
        idx = np.minimum(np.searchsorted(birikim, u, side="right"), len(self.dilimler) - 1)
        yerel = (u - alt_sinir[idx]) / (birikim[idx] - alt_sinir[idx])
        m = np.empty_like(u)
        for i, (p, lo, hi) in enumerate(self.dilimler):
            mask = idx == i
            if np.any(mask):
                m[mask] = p.inverse(yerel[mask], lo, hi)
        return m


def _salpeter(alpha: float, m_low: float, m_high: float) -> _Imf:
    if alpha <= 0:
        raise ValueError("α pozitif olmalı")
    return _Imf((_PowerLawSegment(0.0, math.inf, 1.0, alpha),), m_low, m_high)


def _kroupa(m_low: float, m_high: float) -> _Imf:
    if m_low < _KROUPA_BREAKS[0]:
        raise ValueError("Kroupa IMF'i 0,01 M_⊙'in altında tanımlı değildir")
    parts = []
    coefficient = 1.0
    for i, alpha in enumerate(_KROUPA_ALPHA):
        if i > 0:
            coefficient *= _KROUPA_BREAKS[i] ** (alpha - _KROUPA_ALPHA[i - 1])  # kırılmada süreklilik
        parts.append(_PowerLawSegment(_KROUPA_BREAKS[i], _KROUPA_BREAKS[i + 1], coefficient, alpha))
    return _Imf(tuple(parts), m_low, m_high)


def _chabrier(m_low: float, m_high: float) -> _Imf:
    parts = (
        _LogNormalSegment(0.0, 1.0, _CHABRIER_A_LOGNORMAL, math.log10(_CHABRIER_MC), _CHABRIER_SIGMA),
        # ξ(log m) = A m^{-x}  ⇒  dN/dm = A/ln10 · m^{-(x+1)}
        _PowerLawSegment(1.0, math.inf, _CHABRIER_A_POWER_LAW / _LN10, _CHABRIER_X + 1.0),
    )
    return _Imf(parts, m_low, m_high)


@source(
    "Chabrier2003",
    where="Sec. 'Mass function' ve 'Functional forms' (TeX etiketi IMF1): α = 2,35, x = 1,35",
    status="read",
)
@source("Kroupa2001", where="Sec. '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2,35", status="read")
@source(
    "Salpeter1955",
    where="Denk. 2 (tanım) ve Denk. 5",
    status="read",
    note="ADS taramasından okundu: dN = ξ(M) d(log10 M) dt/T0 ve ξ(M) ≈ 0,03 (M/M_⊙)^−1,35, log(M/M_⊙) −0,4 ile "
    "+1,0 arasında. ξ logaritmik kütle aralığı başınadır; dN/dM ∝ M^−2,35",
)
def salpeter_dn_dm(mass_msun: ArrayLike, m_low: float = 0.1, m_high: float = 100.0, alpha: float = 2.35) -> Array:
    """Salpeter IMF'inin normalleştirilmiş sayı yoğunluğunu hesaplar.

    dN/dm = C m^{-α},   ∫_{m_alt}^{m_ust} dN/dm dm = 1

    Parameters
    ----------
    kutle_gunes : float ya da ndarray
        Kütle [M_⊙].
    m_alt, m_ust : float
        Normalleştirme aralığı [M_⊙]; varsayılanlar bu işleve özgü bir seçimdir.
    alfa : float
        Eğim; Salpeter değeri 2,35 (x = α - 1 = 1,35).

    Returns
    -------
    float ya da ndarray
        dN/dm [M_⊙^-1]; aralığın dışında 0.

    References
    ----------
    [Chabrier2003] IMF1; [Kroupa2001] Salpeter eğimi; [Salpeter1955] Denk. 5.

    Validity
    --------
    [Kroupa2001]: Salpeter 0,4 < m < 10 M_⊙ için α = 2,35 bulmuştur; aralığı aşmak ekstrapolasyondur.

    Verification
    ------------
    tests/unit/test_stars.py: normalleştirme integrali ve log-log eğimi; tests/property/test_stars_property.py.
    """
    return _salpeter(alpha, m_low, m_high).density(mass_msun)


@source("Chabrier2003", where="Sec. 'Mass function' ve 'Functional forms' (TeX etiketi IMF1)", status="read")
@source("Kroupa2001", where="Sec. '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2,35", status="read")
@source(
    "Salpeter1955",
    where="Denk. 2 (tanım) ve Denk. 5",
    status="read",
    note="ADS taramasından okundu: dN = ξ(M) d(log10 M) dt/T0 ve ξ(M) ≈ 0,03 (M/M_⊙)^−1,35, log(M/M_⊙) −0,4 ile "
    "+1,0 arasında. ξ logaritmik kütle aralığı başınadır; dN/dM ∝ M^−2,35",
)
def salpeter_mean_mass(m_low: float = 0.1, m_high: float = 100.0, alpha: float = 2.35) -> float:
    """Salpeter IMF'inin sayı ağırlıklı ortalama kütlesini hesaplar.

    ⟨m⟩ = ∫ m dN/dm dm / ∫ dN/dm dm

    Parameters
    ----------
    m_alt, m_ust : float
        Aralık [M_⊙].
    alfa : float
        Eğim.

    Returns
    -------
    float
        Ortalama kütle [M_⊙].

    References
    ----------
    [Chabrier2003] IMF1; [Kroupa2001]; [Salpeter1955] Denk. 5.

    Verification
    ------------
    tests/unit/test_stars.py: kapalı biçimdeki kuvvet integrali oranıyla; sayısal integralle.
    """
    return _salpeter(alpha, m_low, m_high).mean_mass()


@source("Chabrier2003", where="Sec. 'Mass function' ve 'Functional forms' (TeX etiketi IMF1)", status="read")
@source("Kroupa2001", where="Sec. '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2,35", status="read")
@source(
    "Salpeter1955",
    where="Denk. 2 (tanım) ve Denk. 5",
    status="read",
    note="ADS taramasından okundu: dN = ξ(M) d(log10 M) dt/T0 ve ξ(M) ≈ 0,03 (M/M_⊙)^−1,35, log(M/M_⊙) −0,4 ile "
    "+1,0 arasında. ξ logaritmik kütle aralığı başınadır; dN/dM ∝ M^−2,35",
)
def salpeter_mass_fraction_above(
    m_threshold: ArrayLike, m_low: float = 0.1, m_high: float = 100.0, alpha: float = 2.35
) -> Array:
    """Eşiğin üzerindeki yıldızların toplam kütledeki payını hesaplar (Salpeter).

    f(>m_esik) = ∫_{m_esik}^{m_ust} m dN/dm dm / ∫_{m_alt}^{m_ust} m dN/dm dm

    Parameters
    ----------
    m_esik : float ya da ndarray
        Eşik kütle [M_⊙]; aralığa kırpılır.
    m_alt, m_ust : float
        Aralık [M_⊙].
    alfa : float
        Eğim.

    Returns
    -------
    float ya da ndarray
        Kütle oranı ∈ [0, 1]; eşik arttıkça azalır.

    References
    ----------
    [Chabrier2003] IMF1; [Kroupa2001]; [Salpeter1955] Denk. 5.

    Verification
    ------------
    tests/unit/test_stars.py: kapalı biçim; tests/property/test_stars_property.py: monotonluk.
    """
    return _salpeter(alpha, m_low, m_high).mass_fraction_above(m_threshold)


@source("Chabrier2003", where="Sec. 'Mass function' ve 'Functional forms' (TeX etiketi IMF1)", status="read")
@source("Kroupa2001", where="Sec. '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2,35", status="read")
@source(
    "Salpeter1955",
    where="Denk. 2 (tanım) ve Denk. 5",
    status="read",
    note="ADS taramasından okundu: dN = ξ(M) d(log10 M) dt/T0 ve ξ(M) ≈ 0,03 (M/M_⊙)^−1,35, log(M/M_⊙) −0,4 ile "
    "+1,0 arasında. ξ logaritmik kütle aralığı başınadır; dN/dM ∝ M^−2,35",
)
def salpeter_sample(
    rng: np.random.Generator, n: int, m_low: float = 0.1, m_high: float = 100.0, alpha: float = 2.35
) -> Array:
    """Salpeter IMF'inden ters dönüşümle rastgele kütle çeker.

    m = [m_alt^{1-α} + u (m_ust^{1-α} - m_alt^{1-α})]^{1/(1-α)},   u ~ U[0, 1)

    Parameters
    ----------
    rng : numpy.random.Generator
        Rastgele sayı üreteci.
    n : int
        Örnek sayısı.
    m_alt, m_ust : float
        Aralık [M_⊙].
    alfa : float
        Eğim.

    Returns
    -------
    ndarray, şekil (n,)
        Kütleler [M_⊙], [m_alt, m_ust] içinde.

    References
    ----------
    [Chabrier2003] IMF1; [Kroupa2001]; [Salpeter1955] Denk. 5.

    Verification
    ------------
    tests/unit/test_stars.py: Kolmogorov-Smirnov sınaması (sabit tohum); ortalama kütle uyumu.
    """
    return _salpeter(alpha, m_low, m_high).sample(rng, n)


@source(
    "Kroupa2001",
    where="Sec. 'The universal IMF', TeX etiketleri eq:xi ve eq:imf",
    status="read",
    note="parçalar kırılmalarda sürekli birleştirilmiştir (makale yalnızca orantılılık yazar); makaledeki toplu sayılarla sınandı",
)
def kroupa_dn_dm(mass_msun: ArrayLike, m_low: float = 0.01, m_high: float = 50.0) -> Array:
    """Kroupa (2001) IMF'inin normalleştirilmiş sayı yoğunluğunu hesaplar.

    dN/dm ∝ m^{-α_i};  α = 0,3 (0,01-0,08), 1,3 (0,08-0,5), 2,3 (0,5-1), 2,3 (≥ 1 M_⊙)

    Parameters
    ----------
    kutle_gunes : float ya da ndarray
        Kütle [M_⊙].
    m_alt, m_ust : float
        Normalleştirme aralığı [M_⊙]; m_alt >= 0,01. Varsayılan üst sınır makalenin
        ortalama kütle için kullandığı 50 M_⊙'dir.

    Returns
    -------
    float ya da ndarray
        dN/dm [M_⊙^-1], ∫ = 1; aralığın dışında 0.

    References
    ----------
    [Kroupa2001] eq:xi, eq:imf.

    Validity
    --------
    Makale, 0,08 M_⊙ altında güven aralığının iyi belirlenmediğini söyler (α_0 = 0,3 ± 0,7).
    Tek yıldızlar içindir; çift sistemler için düzeltme içermez.

    Verification
    ------------
    tests/unit/test_stars.py: ortalama kütle 0,36 M_⊙ ve makaledeki sayı/kütle payları.
    """
    return _kroupa(m_low, m_high).density(mass_msun)


@source("Kroupa2001", where="Sec. 'The universal IMF', TeX etiketi eq:imf ve ortalama kütle cümlesi", status="read")
def kroupa_mean_mass(m_low: float = 0.01, m_high: float = 50.0) -> float:
    """Kroupa IMF'inin sayı ağırlıklı ortalama kütlesini hesaplar.

    Parameters
    ----------
    m_alt, m_ust : float
        Aralık [M_⊙]; m_alt >= 0,01.

    Returns
    -------
    float
        Ortalama kütle [M_⊙]; 0,01-50 M_⊙ için 0,36.

    References
    ----------
    [Kroupa2001] eq:imf ve "⟨m⟩ = 0,36 M_⊙" cümlesi.

    Verification
    ------------
    tests/unit/test_stars.py: makaledeki 0,36 M_⊙ (iki anlamlı basamak).
    """
    return _kroupa(m_low, m_high).mean_mass()


@source("Kroupa2001", where="Sec. 'The universal IMF', TeX etiketi eq:imf ve kütle payları cümleleri", status="read")
def kroupa_mass_fraction_above(m_threshold: ArrayLike, m_low: float = 0.01, m_high: float = 50.0) -> Array:
    """Eşiğin üzerindeki yıldızların toplam kütledeki payını hesaplar (Kroupa).

    Parameters
    ----------
    m_esik : float ya da ndarray
        Eşik kütle [M_⊙]; aralığa kırpılır.
    m_alt, m_ust : float
        Aralık [M_⊙]; m_alt >= 0,01.

    Returns
    -------
    float ya da ndarray
        Kütle oranı ∈ [0, 1]; 1 M_⊙ eşiği için yaklaşık 0,5 (makale: kütlenin yarısı 1 M_⊙'in altında).

    References
    ----------
    [Kroupa2001] eq:imf ve kütle payları cümleleri.

    Verification
    ------------
    tests/unit/test_stars.py: yüzde 50 ve 8 M_⊙ üzeri için yüzde 17 ([Kroupa2001]).
    """
    return _kroupa(m_low, m_high).mass_fraction_above(m_threshold)


@source("Kroupa2001", where="Sec. 'The universal IMF', TeX etiketi eq:imf", status="read")
def kroupa_sample(rng: np.random.Generator, n: int, m_low: float = 0.01, m_high: float = 50.0) -> Array:
    """Kroupa IMF'inden parça parça ters dönüşümle rastgele kütle çeker.

    Tek bir düzgün sayı u, birikimli sayı payına göre bir parçayı ve parça içindeki konumu seçer.

    Parameters
    ----------
    rng : numpy.random.Generator
        Rastgele sayı üreteci.
    n : int
        Örnek sayısı.
    m_alt, m_ust : float
        Aralık [M_⊙]; m_alt >= 0,01.

    Returns
    -------
    ndarray, şekil (n,)
        Kütleler [M_⊙].

    References
    ----------
    [Kroupa2001] eq:imf.

    Verification
    ------------
    tests/unit/test_stars.py: Kolmogorov-Smirnov sınaması ve örnek ortalaması.
    """
    return _kroupa(m_low, m_high).sample(rng, n)


@source(
    "Chabrier2003",
    where="Sec. 'The disk stellar mass function', TeX etiketi IMFdisk ve Tablo 1 (table.diskIMF), 'IMF' satırı",
    status="read",
    note="m <= 1: A = 0,158, m_c = 0,079, σ = 0,69; m > 1: A = 4,43e-2, x = 1,3; ξ(log m) tanımıdır, dN/dm = ξ/(m ln10)",
)
def chabrier_dn_dm(mass_msun: ArrayLike, m_low: float = 0.01, m_high: float = 50.0) -> Array:
    """Chabrier (2003) disk IMF'inin normalleştirilmiş sayı yoğunluğunu hesaplar.

    ξ(log m) = 0,158 exp{-(log m - log 0,079)² / (2 · 0,69²)}   (m ≤ 1 M_⊙)
    ξ(log m) = 4,43e-2 m^{-1,3}                                 (m > 1 M_⊙)
    dN/dm = ξ(log m) / (m ln 10)

    Parameters
    ----------
    kutle_gunes : float ya da ndarray
        Kütle [M_⊙].
    m_alt, m_ust : float
        Normalleştirme aralığı [M_⊙]; varsayılanlar Kroupa ile karşılaştırma için seçilmiştir.

    Returns
    -------
    float ya da ndarray
        dN/dm [M_⊙^-1], ∫ = 1; aralığın dışında 0.

    References
    ----------
    [Chabrier2003] IMFdisk ve Tablo 1.

    Validity
    --------
    Tek nesneler için disk IMF'idir; çift sistemlerin katsayıları farklıdır (makalenin IMFsys denklemi,
    burada uygulanmadı). Parametre belirsizlikleri makalede verilir (örn. A = 0,158 +0,051/-0,046).

    Verification
    ------------
    tests/unit/test_stars.py: normalleştirme, m = 1'de süreklilik (yüzde 1) ve log-normal tepe noktası.
    """
    return _chabrier(m_low, m_high).density(mass_msun)


@source("Chabrier2003", where="Sec. 'The disk stellar mass function', TeX etiketi IMFdisk ve Tablo 1", status="read")
def chabrier_mean_mass(m_low: float = 0.01, m_high: float = 50.0) -> float:
    """Chabrier disk IMF'inin sayı ağırlıklı ortalama kütlesini hesaplar.

    Parameters
    ----------
    m_alt, m_ust : float
        Aralık [M_⊙].

    Returns
    -------
    float
        Ortalama kütle [M_⊙].

    References
    ----------
    [Chabrier2003] IMFdisk ve Tablo 1.

    Verification
    ------------
    tests/unit/test_stars.py: sayısal integralle (scipy.integrate.quad) 1e-8 bağıl uyum.
    """
    return _chabrier(m_low, m_high).mean_mass()


@source("Chabrier2003", where="Sec. 'The disk stellar mass function', TeX etiketi IMFdisk ve Tablo 1", status="read")
def chabrier_mass_fraction_above(m_threshold: ArrayLike, m_low: float = 0.01, m_high: float = 50.0) -> Array:
    """Eşiğin üzerindeki yıldızların toplam kütledeki payını hesaplar (Chabrier).

    Parameters
    ----------
    m_esik : float ya da ndarray
        Eşik kütle [M_⊙]; aralığa kırpılır.
    m_alt, m_ust : float
        Aralık [M_⊙].

    Returns
    -------
    float ya da ndarray
        Kütle oranı ∈ [0, 1].

    References
    ----------
    [Chabrier2003] IMFdisk ve Tablo 1.

    Verification
    ------------
    tests/unit/test_stars.py: sayısal integralle uyum; tests/property/test_stars_property.py: monotonluk.
    """
    return _chabrier(m_low, m_high).mass_fraction_above(m_threshold)


@source("Chabrier2003", where="Sec. 'The disk stellar mass function', TeX etiketi IMFdisk ve Tablo 1", status="read")
def chabrier_sample(rng: np.random.Generator, n: int, m_low: float = 0.01, m_high: float = 50.0) -> Array:
    """Chabrier disk IMF'inden ters dönüşümle rastgele kütle çeker.

    Log-normal parçada normal dağılımın tersi (``ndtri``), kuvvet yasası parçasında kapalı ters kullanılır.

    Parameters
    ----------
    rng : numpy.random.Generator
        Rastgele sayı üreteci.
    n : int
        Örnek sayısı.
    m_alt, m_ust : float
        Aralık [M_⊙].

    Returns
    -------
    ndarray, şekil (n,)
        Kütleler [M_⊙].

    References
    ----------
    [Chabrier2003] IMFdisk ve Tablo 1.

    Verification
    ------------
    tests/unit/test_stars.py: Kolmogorov-Smirnov sınaması (sabit tohum).
    """
    return _chabrier(m_low, m_high).sample(rng, n)
