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
from scipy.special import ndtri
from scipy.stats import rankdata

from kozmos._sources import source

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Chain:
    """Topluluk örnekleyicinin çıktısı.

    Attributes
    ----------
    examples : ndarray, biçim (adım, yürüyücü, boyut)
    log_olasilik : ndarray, biçim (adım, yürüyücü)
    kabul_orani : ndarray, biçim (yürüyücü,)
    """

    samples: Array
    log_probability: Array
    acceptance_fraction: Array

    def flatten(self, discard: int = 0, thin: int = 1) -> Array:
        """Isınma adımlarını atar ve yürüyücüleri tek örnek dizisinde birleştirir."""
        return self.samples[discard::thin].reshape(-1, self.samples.shape[-1])


@source(
    "ForemanMackey2013",
    where="Denk. goodmanprop (g(z) ∝ 1/√z, z ∈ [1/a, a])",
    status="read",
    note="ters dönüşüm örneklemesi: z = ((a−1)u + 1)² / a, u ~ U(0,1); türetim bu projede",
)
def stretch_factor(rng: np.random.Generator, size: int | tuple[int, ...], a: float = 2.0) -> Array:
    """Gerdirme hamlesinin ölçek çarpanını g(z) ∝ 1/√z dağılımından örnekler.

    Birikimli dağılım F(z) = (√z − 1/√a) / (√a − 1/√a) ters çevrilir:
    z = ((a − 1) u + 1)² / a.

    References
    ----------
    [ForemanMackey2013] Denk. goodmanprop.

    Validity
    --------
    a > 1.

    Verification
    ------------
    tests/unit/test_mcmc.py: örneklerin aralığı ve birikimli dağılımı (KS testi).
    """
    if a <= 1.0:
        raise ValueError("a 1'den büyük olmalı")
    u = rng.random(size)
    return ((a - 1.0) * u + 1.0) ** 2 / a


@source(
    "ForemanMackey2013",
    where="Denk. proposal, acceptance; Algoritma 3 (paralel gerdirme hamlesi)",
    status="read",
    note="Y = X_j + Z (X_k − X_j); kabul q = min(1, Z^{N−1} p(Y)/p(X_k)); topluluk iki yarıya bölünür",
)
def ensemble_sample(
    log_probability: Callable[[Array], float],
    initial: ArrayLike,
    n_steps: int,
    rng: np.random.Generator,
    *,
    a: float = 2.0,
) -> Chain:
    """Afin değişmez topluluk örnekleyiciyi (gerdirme hamlesi) koşturur.

    Her adımda topluluk iki yarıya ayrılır; bir yarıdaki her yürüyücü öbür
    yarıdan rastgele seçilen bir yürüyücüye göre güncellenir.

    Parameters
    ----------
    log_olasilik : callable
        Parametre vektörünü alıp log sonsal yoğunluğu döndürür; tanım kümesi
        dışında ``-inf``.
    baslangic : ndarray, biçim (yürüyücü, boyut)
        Başlangıç konumları. Yürüyücü sayısı çift ve boyutun en az iki katı olmalı.
    adim_sayisi : int
    uretec : numpy.random.Generator
    a : float
        Gerdirme ölçeği; kaynakta önerilen değer 2.

    References
    ----------
    [ForemanMackey2013] Denk. proposal, acceptance, Algoritma 3.

    Validity
    --------
    Sürekli, tek tepeli ya da hafif çok tepeli dağılımlar. Yüksek boyutta
    (onlarca parametre üstü) verimi düşer.

    Verification
    ------------
    tests/unit/test_mcmc.py: ilintili Gauss hedefte ortalama ve kovaryansın geri kazanımı.
    """
    x = np.array(initial, dtype=float)
    if x.ndim != 2:
        raise ValueError("başlangıç (yürüyücü, boyut) biçiminde olmalı")
    k, n = x.shape
    if k % 2 or k < 2 * n:
        raise ValueError("yürüyücü sayısı çift ve boyutun en az iki katı olmalı")
    lp = np.array([log_probability(v) for v in x])
    if not np.all(np.isfinite(lp)):
        raise ValueError("başlangıç konumlarının bir kısmında log olasılık sonlu değil")
    samples = np.empty((n_steps, k, n))
    lp_kayit = np.empty((n_steps, k))
    kabul = np.zeros(k)
    yari = k // 2
    yarilar = (np.arange(yari), np.arange(yari, k))
    for t in range(n_steps):
        for target, tamamlayici in (yarilar, yarilar[::-1]):
            z = stretch_factor(rng, yari, a)
            es = tamamlayici[rng.integers(0, yari, yari)]
            oneri = x[es] + z[:, None] * (x[target] - x[es])
            lp_oneri = np.array([log_probability(v) for v in oneri])
            log_q = (n - 1) * np.log(z) + lp_oneri - lp[target]
            kabul_mu = np.log(rng.random(yari)) < log_q
            secili = target[kabul_mu]
            x[secili] = oneri[kabul_mu]
            lp[secili] = lp_oneri[kabul_mu]
            kabul[secili] += 1
        samples[t] = x
        lp_kayit[t] = lp
    return Chain(samples, lp_kayit, kabul / n_steps)


@source(
    "ForemanMackey2013",
    where="Bölüm 3, 'Autocorrelation time': τ_f ve C_f(T) denklemleri",
    status="read",
    note="τ = 1 + 2 Σ C(T)/C(0). Toplamın kesildiği pencere kuralı (en küçük M, M >= c·τ(M), c = 5) makalede yoktur; "
    "emcee figuresindeki uygulamadan hatırlanmıştır ve teyit edilmemiştir",
)
def autocorrelation_time(chain: ArrayLike, c: float = 5.0) -> Array:
    """Tümleşik öz ilinti süresini parametre başına kestirir.

    Parameters
    ----------
    zincir : ndarray, biçim (adım, yürüyücü, boyut)
    c : float
        Pencere katsayısı.

    Returns
    -------
    ndarray, biçim (boyut,)
        Adım cinsinden τ. Zincir uzunluğu 50 τ'dan kısaysa kestirim güvenilmezdir.

    References
    ----------
    [ForemanMackey2013] Bölüm 3.

    Verification
    ------------
    tests/unit/test_mcmc.py: AR(1) sürecinde kapalı çözüm (1+ρ)/(1−ρ).
    """
    x = np.asarray(chain, dtype=float)
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


@source(
    "Vehtari2021",
    where="Bölüm 3.1: B, W, var⁺ ve R̂ denklemleri",
    status="read",
    note="B = N/(M−1) Σ(θ̄_m − θ̄)², W = ortalama s_m², var⁺ = (N−1)/N·W + B/N, R̂ = sqrt(var⁺/W); ikiye bölünür",
)
def split_rhat(chain: ArrayLike) -> Array:
    """Bölünmüş R̂ yakınsama tanısını parametre başına hesaplar.

    Parameters
    ----------
    zincir : ndarray, biçim (adım, yürüyücü, boyut)
        Isınma atıldıktan sonraki örnekler.

    Returns
    -------
    ndarray, biçim (boyut,)
        R̂; 1'e yakın değer zincirlerin aynı dağılımı örneklediğiyle tutarlıdır.

    References
    ----------
    [Vehtari2021] Bölüm 3.1.

    Validity
    --------
    Topluluk örnekleyicinin yürüyücüleri bağımsız zincirler değildir; bu
    yüzden R̂ burada iyimser olabilir ve öz ilinti süresiyle birlikte okunmalıdır.

    Verification
    ------------
    tests/unit/test_mcmc.py: bağımsız Gauss örneklerde R̂ ≈ 1, kaydırılmış zincirlerde R̂ > 1,1.
    """
    x = np.asarray(chain, dtype=float)
    yari = x.shape[0] // 2
    parts = np.concatenate([x[:yari], x[yari : 2 * yari]], axis=1)  # (yari, 2k, n)
    ortalama = parts.mean(axis=0)
    w = parts.var(axis=0, ddof=1).mean(axis=0)
    b = yari * ortalama.var(axis=0, ddof=1)
    var_arti = (yari - 1) / yari * w + b / yari
    return np.sqrt(var_arti / w)


@source(
    "Vehtari2021",
    where="Bölüm 4.1-4.2: sıra normalleştirme ve katlama denklemleri",
    status="read",
    note="z = Φ⁻¹((r − 3/8)/(S + 1/4)); katlama ζ = |θ − ortanca(θ)|; ikisinin büyüğü raporlanır",
)
def rank_normalized_rhat(chain: ArrayLike) -> Array:
    """Sıra normalleştirilmiş bölünmüş R̂ ile katlanmış bölünmüş R̂'nin büyüğünü hesaplar.

    Her parametre için örnekler havuzdaki sıralarıyla değiştirilir (eşitlikte
    ortalama sıra) ve z = Φ⁻¹((r − 3/8)/(S + 1/4)) ile normal puanlara
    çevrilir; bölünmüş R̂ bu puanlardan hesaplanır. Aynı işlem ortancadan
    mutlak sapmalara (katlanmış örnekler) uygulanır; kuyruklardaki
    yakınsamayı bu ölçer. Sonuç ikisinin büyüğüdür.

    Parameters
    ----------
    zincir : ndarray, biçim (adım, yürüyücü, boyut)

    Returns
    -------
    ndarray, biçim (boyut,)

    References
    ----------
    [Vehtari2021] Bölüm 4.1-4.2.

    Validity
    --------
    Tekdüze dönüşümlere değişmezdir ve sonsuz varyanslı dağılımlarda da
    tanımlıdır. Topluluk örnekleyicide yürüyücüler bağımsız zincir değildir;
    değer iyimser olabilir.

    Verification
    ------------
    tests/unit/test_mcmc.py: konumu kaymış ve ölçeği farklı zincirlerin yakalanması; Cauchy örneklerde tanımlılık.
    """
    x = np.asarray(chain, dtype=float)
    adim, k, n = x.shape

    def puan(d: Array) -> Array:
        sira = rankdata(d.reshape(-1, n), axis=0)  # eşitlikte ortalama sıra
        return ndtri((sira - 0.375) / (adim * k + 0.25)).reshape(adim, k, n)

    katlanmis = np.abs(x - np.median(x.reshape(-1, n), axis=0))
    return np.maximum(split_rhat(puan(x)), split_rhat(puan(katlanmis)))
