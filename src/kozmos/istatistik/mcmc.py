"""Afin değişmez topluluk örnekleyici ve zincir tanıları.

Gerdirme hamlesi ve paralel güncelleme şeması emcee makalesinin arXiv TeX
kaynağından (1202.3665) okunmuştur [ForemanMackey2013]; yöntemin özgün
kaynağı [GoodmanWeare2010] makalesidir ve bu çalışmada okunmamıştır.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._kaynak import kaynak

Dizi = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Zincir:
    """Topluluk örnekleyicinin çıktısı.

    Öznitelikler
    ------------
    ornekler : ndarray, biçim (adım, yürüyücü, boyut)
    log_olasilik : ndarray, biçim (adım, yürüyücü)
    kabul_orani : ndarray, biçim (yürüyücü,)
    """

    ornekler: Dizi
    log_olasilik: Dizi
    kabul_orani: Dizi

    def duzlestir(self, atilacak: int = 0, seyreltme: int = 1) -> Dizi:
        """Isınma adımlarını atar ve yürüyücüleri tek örnek dizisinde birleştirir."""
        return self.ornekler[atilacak::seyreltme].reshape(-1, self.ornekler.shape[-1])


@kaynak(
    "ForemanMackey2013",
    yer="Denk. goodmanprop (g(z) ∝ 1/√z, z ∈ [1/a, a])",
    teyit="okundu",
    not_="ters dönüşüm örneklemesi: z = ((a−1)u + 1)² / a, u ~ U(0,1); türetim bu projede",
)
def gerdirme_carpani(uretec: np.random.Generator, boyut: int | tuple[int, ...], a: float = 2.0) -> Dizi:
    """Gerdirme hamlesinin ölçek çarpanını g(z) ∝ 1/√z dağılımından örnekler.

    Birikimli dağılım F(z) = (√z − 1/√a) / (√a − 1/√a) ters çevrilir:
    z = ((a − 1) u + 1)² / a.

    Kaynaklar
    ---------
    [ForemanMackey2013] Denk. goodmanprop.

    Geçerlilik
    ----------
    a > 1.

    Doğrulama
    ---------
    tests/birim/test_mcmc.py: örneklerin aralığı ve birikimli dağılımı (KS testi).
    """
    if a <= 1.0:
        raise ValueError("a 1'den büyük olmalı")
    u = uretec.random(boyut)
    return ((a - 1.0) * u + 1.0) ** 2 / a


@kaynak(
    "ForemanMackey2013",
    yer="Denk. proposal, acceptance; Algoritma 3 (paralel gerdirme hamlesi)",
    teyit="okundu",
    not_="Y = X_j + Z (X_k − X_j); kabul q = min(1, Z^{N−1} p(Y)/p(X_k)); topluluk iki yarıya bölünür",
)
@kaynak("GoodmanWeare2010", yer="gerdirme hamlesi", teyit="bekliyor")
def topluluk_ornekle(
    log_olasilik: Callable[[Dizi], float],
    baslangic: ArrayLike,
    adim_sayisi: int,
    uretec: np.random.Generator,
    *,
    a: float = 2.0,
) -> Zincir:
    """Afin değişmez topluluk örnekleyiciyi (gerdirme hamlesi) koşturur.

    Her adımda topluluk iki yarıya ayrılır; bir yarıdaki her yürüyücü öbür
    yarıdan rastgele seçilen bir yürüyücüye göre güncellenir.

    Parametreler
    ------------
    log_olasilik : callable
        Parametre vektörünü alıp log sonsal yoğunluğu döndürür; tanım kümesi
        dışında ``-inf``.
    baslangic : ndarray, biçim (yürüyücü, boyut)
        Başlangıç konumları. Yürüyücü sayısı çift ve boyutun en az iki katı olmalı.
    adim_sayisi : int
    uretec : numpy.random.Generator
    a : float
        Gerdirme ölçeği; kaynakta önerilen değer 2.

    Kaynaklar
    ---------
    [ForemanMackey2013] Denk. proposal, acceptance, Algoritma 3; [GoodmanWeare2010].

    Geçerlilik
    ----------
    Sürekli, tek tepeli ya da hafif çok tepeli dağılımlar. Yüksek boyutta
    (onlarca parametre üstü) verimi düşer.

    Doğrulama
    ---------
    tests/birim/test_mcmc.py: ilintili Gauss hedefte ortalama ve kovaryansın geri kazanımı.
    """
    x = np.array(baslangic, dtype=float)
    if x.ndim != 2:
        raise ValueError("başlangıç (yürüyücü, boyut) biçiminde olmalı")
    k, n = x.shape
    if k % 2 or k < 2 * n:
        raise ValueError("yürüyücü sayısı çift ve boyutun en az iki katı olmalı")
    lp = np.array([log_olasilik(v) for v in x])
    if not np.all(np.isfinite(lp)):
        raise ValueError("başlangıç konumlarının bir kısmında log olasılık sonlu değil")
    ornekler = np.empty((adim_sayisi, k, n))
    lp_kayit = np.empty((adim_sayisi, k))
    kabul = np.zeros(k)
    yari = k // 2
    yarilar = (np.arange(yari), np.arange(yari, k))
    for t in range(adim_sayisi):
        for hedef, tamamlayici in (yarilar, yarilar[::-1]):
            z = gerdirme_carpani(uretec, yari, a)
            es = tamamlayici[uretec.integers(0, yari, yari)]
            oneri = x[es] + z[:, None] * (x[hedef] - x[es])
            lp_oneri = np.array([log_olasilik(v) for v in oneri])
            log_q = (n - 1) * np.log(z) + lp_oneri - lp[hedef]
            kabul_mu = np.log(uretec.random(yari)) < log_q
            secili = hedef[kabul_mu]
            x[secili] = oneri[kabul_mu]
            lp[secili] = lp_oneri[kabul_mu]
            kabul[secili] += 1
        ornekler[t] = x
        lp_kayit[t] = lp
    return Zincir(ornekler, lp_kayit, kabul / adim_sayisi)


@kaynak(
    "ForemanMackey2013",
    yer="Bölüm 3 (öz ilinti süresi)",
    teyit="bekliyor",
    not_="Sokal pencereleme: en küçük M, M >= c·τ(M); uygulama ayrıntısı kaynaktan teyit edilmedi",
)
def oz_ilinti_suresi(zincir: ArrayLike, c: float = 5.0) -> Dizi:
    """Tümleşik öz ilinti süresini parametre başına kestirir.

    Parametreler
    ------------
    zincir : ndarray, biçim (adım, yürüyücü, boyut)
    c : float
        Pencere katsayısı.

    Döndürür
    --------
    ndarray, biçim (boyut,)
        Adım cinsinden τ. Zincir uzunluğu 50 τ'dan kısaysa kestirim güvenilmezdir.

    Kaynaklar
    ---------
    [ForemanMackey2013] Bölüm 3.

    Doğrulama
    ---------
    tests/birim/test_mcmc.py: AR(1) sürecinde kapalı çözüm (1+ρ)/(1−ρ).
    """
    x = np.asarray(zincir, dtype=float)
    adim, k, n = x.shape
    tau = np.empty(n)
    boy = 1 << int(np.ceil(np.log2(2 * adim)))
    for d in range(n):
        acf = np.zeros(adim)
        for w in range(k):
            y = x[:, w, d] - x[:, w, d].mean()
            f = np.fft.rfft(y, boy)
            g = np.fft.irfft(f * np.conjugate(f))[:adim]
            acf += g / g[0] if g[0] > 0 else 0.0
        acf /= k
        taus = 2.0 * np.cumsum(acf) - 1.0
        pencere = np.arange(adim) >= c * taus
        m = int(np.argmax(pencere)) if np.any(pencere) else adim - 1
        tau[d] = taus[m]
    return tau


@kaynak(
    "GelmanRubin1992",
    yer="potansiyel ölçek indirgeme çarpanı",
    teyit="bekliyor",
    not_="bölünmüş R̂: her yürüyücü zinciri ikiye bölünür; sıra normalleştirmesi yok",
)
def bolunmus_rhat(zincir: ArrayLike) -> Dizi:
    """Bölünmüş R̂ yakınsama tanısını parametre başına hesaplar.

    Parametreler
    ------------
    zincir : ndarray, biçim (adım, yürüyücü, boyut)
        Isınma atıldıktan sonraki örnekler.

    Döndürür
    --------
    ndarray, biçim (boyut,)
        R̂; 1'e yakın değer zincirlerin aynı dağılımı örneklediğiyle tutarlıdır.

    Kaynaklar
    ---------
    [GelmanRubin1992].

    Geçerlilik
    ----------
    Topluluk örnekleyicinin yürüyücüleri bağımsız zincirler değildir; bu
    yüzden R̂ burada iyimser olabilir ve öz ilinti süresiyle birlikte okunmalıdır.

    Doğrulama
    ---------
    tests/birim/test_mcmc.py: bağımsız Gauss örneklerde R̂ ≈ 1, kaydırılmış zincirlerde R̂ > 1,1.
    """
    x = np.asarray(zincir, dtype=float)
    yari = x.shape[0] // 2
    parcalar = np.concatenate([x[:yari], x[yari : 2 * yari]], axis=1)  # (yari, 2k, n)
    ortalama = parcalar.mean(axis=0)
    w = parcalar.var(axis=0, ddof=1).mean(axis=0)
    b = yari * ortalama.var(axis=0, ddof=1)
    var_arti = (yari - 1) / yari * w + b / yari
    return np.sqrt(var_arti / w)
