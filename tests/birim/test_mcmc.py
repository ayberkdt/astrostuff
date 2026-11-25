import numpy as np
import pytest
from scipy import stats

from kozmos.istatistik import mcmc


def test_gerdirme_carpani_dagilimi():
    # g(z) ∝ 1/√z, z ∈ [1/a, a] için birikimli dağılım: F(z) = (√z − 1/√a) / (√a − 1/√a).
    a = 2.0
    z = mcmc.gerdirme_carpani(np.random.default_rng(21), 20000, a)
    assert z.min() >= 1 / a and z.max() <= a
    birikimli = lambda x: (np.sqrt(x) - a**-0.5) / (a**0.5 - a**-0.5)  # noqa: E731
    assert stats.kstest(z, birikimli).pvalue > 1e-3
    with pytest.raises(ValueError):
        mcmc.gerdirme_carpani(np.random.default_rng(0), 1, 1.0)


def test_ilintili_gauss_geri_kazanimi():
    ortalama = np.array([1.0, -2.0, 0.5])
    kov = np.array([[1.0, 0.8, 0.0], [0.8, 2.0, -0.3], [0.0, -0.3, 0.5]])
    ters = np.linalg.inv(kov)
    log_p = lambda x: -0.5 * (x - ortalama) @ ters @ (x - ortalama)  # noqa: E731
    u = np.random.default_rng(22)
    zincir = mcmc.topluluk_ornekle(log_p, ortalama + u.normal(0, 0.1, (24, 3)), 3000, u)
    tau = mcmc.oz_ilinti_suresi(zincir.ornekler)
    assert 3000 > 50 * tau.max()
    ornek = zincir.duzlestir(atilacak=500)
    etkin = ornek.shape[0] / tau.max()
    assert np.all(np.abs(ornek.mean(axis=0) - ortalama) < 5 * np.sqrt(np.diag(kov) / etkin))
    assert np.cov(ornek.T) == pytest.approx(kov, abs=0.15)
    assert np.all(mcmc.bolunmus_rhat(zincir.ornekler[500:]) < 1.05)
    assert 0.2 < zincir.kabul_orani.mean() < 0.9


def test_baslangic_denetimi():
    u = np.random.default_rng(23)
    log_p = lambda x: -0.5 * float(x @ x)  # noqa: E731
    with pytest.raises(ValueError):
        mcmc.topluluk_ornekle(log_p, u.normal(size=(3, 2)), 10, u)  # tek sayıda yürüyücü
    with pytest.raises(ValueError):
        mcmc.topluluk_ornekle(lambda x: -np.inf, u.normal(size=(4, 2)), 10, u)


def test_oz_ilinti_suresi_ar1():
    # AR(1) sürecinde tümleşik öz ilinti süresi (1+ρ)/(1−ρ).
    rho, adim, k = 0.9, 40000, 8
    u = np.random.default_rng(24)
    x = np.zeros((adim, k, 1))
    gurultu = u.normal(size=(adim, k))
    for t in range(1, adim):
        x[t, :, 0] = rho * x[t - 1, :, 0] + gurultu[t]
    assert mcmc.oz_ilinti_suresi(x)[0] == pytest.approx((1 + rho) / (1 - rho), rel=0.15)


def test_rhat_ayrik_zincirleri_yakalar():
    u = np.random.default_rng(25)
    iyi = u.normal(size=(1000, 8, 2))
    assert np.all(np.abs(mcmc.bolunmus_rhat(iyi) - 1) < 0.02)
    kotu = iyi.copy()
    kotu[:, :4, 0] += 3.0  # yürüyücülerin yarısı başka bir tepede
    assert mcmc.bolunmus_rhat(kotu)[0] > 1.1
