"""Dedektör modeli ve indirgeme hattının sentetik veriyle sınanması.

Gerçeği bilinen bir sahne üretilir, ham karelere çevrilir, indirgenir ve
sonuç girilen sahneyle karşılaştırılır.
"""

import numpy as np
import pytest

from kozmos.gozlem import dedektor as dd
from kozmos.indirgeme import kalibrasyon as kal

BICIM = (120, 160)
DEDEKTOR = dd.Dedektor(kazanc_e_adu=1.7, okuma_gurultusu_e=6.0, karanlik_akim_e_s=0.2, bias_adu=1000.0)


def _duz_alan(u):
    yy, xx = np.mgrid[: BICIM[0], : BICIM[1]]
    vinyet = 1.0 - 0.25 * (((xx - 80) / 110) ** 2 + ((yy - 60) / 110) ** 2)
    duz = vinyet * u.normal(1.0, 0.02, BICIM)
    return duz / duz.mean()


def _sahne(u, n_yildiz=25):
    x, y = u.uniform(10, BICIM[1] - 10, n_yildiz), u.uniform(10, BICIM[0] - 10, n_yildiz)
    aki = 10 ** u.uniform(2.0, 3.5, n_yildiz)
    return 20.0 + dd.yildiz_alani(BICIM, x, y, aki, fwhm_piksel=3.2)  # gökyüzü 20 e/s/piksel


def test_yildiz_alani_akiyi_korur():
    sahne = dd.yildiz_alani((61, 61), [30.3], [29.6], [1000.0], fwhm_piksel=2.5)
    assert sahne.sum() == pytest.approx(1000.0, rel=1e-9)
    # Dar PSF'te de: piksel merkezinde örnekleme akıyı korumazdı, piksel integrali korur.
    assert dd.yildiz_alani((41, 41), [20.4], [20.1], [500.0], fwhm_piksel=0.8).sum() == pytest.approx(500.0, rel=1e-9)


def test_kazanc_ve_okuma_gurultusu_geri_kazanimi():
    u = np.random.default_rng(81)
    duz = _duz_alan(u)
    bos = np.zeros(BICIM)
    aydinlik = np.full(BICIM, 8000.0)  # e/s; 1 s poz
    f1, f2 = (dd.ham_kare_uret(aydinlik, 1.0, DEDEKTOR, u, duz_alan=duz) for _ in range(2))
    b1, b2 = (dd.ham_kare_uret(bos, 0.0, DEDEKTOR, u) for _ in range(2))
    kazanc, okuma = kal.kazanc_ve_okuma_gurultusu(f1, f2, b1, b2)
    assert kazanc == pytest.approx(DEDEKTOR.kazanc_e_adu, rel=0.03)
    assert okuma == pytest.approx(DEDEKTOR.okuma_gurultusu_e, rel=0.03)


def test_indirgeme_sahneyi_geri_verir():
    u = np.random.default_rng(82)
    duz = _duz_alan(u)
    sahne = _sahne(u)
    bos = np.zeros(BICIM)
    poz = 60.0

    bias, _ = kal.birlestir([dd.ham_kare_uret(bos, 0.0, DEDEKTOR, u) for _ in range(25)])
    karanlik_adu, _ = kal.birlestir([dd.ham_kare_uret(bos, 300.0, DEDEKTOR, u) for _ in range(15)])
    karanlik_e_s = (karanlik_adu - bias) * DEDEKTOR.kazanc_e_adu / 300.0
    duzler, _ = kal.birlestir(
        [dd.ham_kare_uret(np.full(BICIM, 3000.0), 5.0, DEDEKTOR, u, duz_alan=duz) for _ in range(15)]
    )
    ana_duz = (duzler - bias) * DEDEKTOR.kazanc_e_adu - karanlik_e_s * 5.0
    ana_duz /= ana_duz.mean()
    assert np.std(ana_duz / duz) < 0.004  # düz alan deseni binde birkaç düzeyinde geri kazanılır

    ham = dd.ham_kare_uret(sahne, poz, DEDEKTOR, u, duz_alan=duz)
    sonuc = kal.kare_indirge(
        ham,
        bias,
        karanlik_e_s,
        ana_duz,
        kazanc_e_adu=DEDEKTOR.kazanc_e_adu,
        okuma_gurultusu_e=DEDEKTOR.okuma_gurultusu_e,
        poz_s=poz,
    )
    artik = (sonuc.veri - sahne) / np.sqrt(sonuc.varyans)
    # Yansız (ortalama artık sıfır dolayında) ve varyans düzlemi gerçek saçılımı öngörüyor (χ²/N ≈ 1).
    assert abs(artik.mean()) < 0.05
    assert np.mean(artik**2) == pytest.approx(1.0, abs=0.08)
    # Düz alan düzeltmesi yapılmasaydı vinyet artığı kalırdı.
    duzeltmesiz = kal.kare_indirge(ham, bias, karanlik_e_s, 1.0, kazanc_e_adu=1.7, okuma_gurultusu_e=6.0, poz_s=poz)
    assert np.mean(((duzeltmesiz.veri - sahne) / np.sqrt(duzeltmesiz.varyans)) ** 2) > 3.0


def test_birlestirme_kozmik_isini_ayiklar():
    u = np.random.default_rng(83)
    kareler = u.normal(100.0, 5.0, (9, 40, 50))
    temiz_ortalama = kareler.mean(axis=0)
    vurulan = u.integers(0, 9, 30), u.integers(0, 40, 30), u.integers(0, 50, 30)
    kareler[vurulan] += 5000.0
    kirpmasiz, _ = kal.birlestir(kareler, sigma=None)
    kirpmali, kullanilan = kal.birlestir(kareler, sigma=2.5)
    assert np.abs(kirpmasiz - temiz_ortalama).max() > 400
    assert np.abs(kirpmali - temiz_ortalama).max() < 6.0
    assert kullanilan.min() >= 6 and kullanilan[vurulan[1], vurulan[2]].max() <= 8
    ortanca, _ = kal.birlestir(kareler, yontem="ortanca", sigma=None)
    assert np.abs(ortanca - 100.0).max() < 12
    with pytest.raises(ValueError):
        kal.birlestir(kareler[0])
    with pytest.raises(ValueError):
        kal.birlestir(kareler, yontem="kip")


def test_ccd_denklemi_limitleri():
    # Kaynak baskın: S/N → sqrt(N*).
    assert float(dd.sinyal_gurultu_orani(1e6, 0.0, 0.0, 0.0, 10)) == pytest.approx(1e3)
    # Gökyüzü baskın: S/N → N* / sqrt(n_pix · N_S).
    assert float(dd.sinyal_gurultu_orani(100.0, 1e5, 0.0, 5.0, 50)) == pytest.approx(100 / np.sqrt(50 * 1e5), rel=1e-3)
    # Arka plan az pikselden kestirilirse gürültü artar.
    assert dd.sinyal_gurultu_orani(1e3, 200.0, 1.0, 5.0, 50, 100) < dd.sinyal_gurultu_orani(1e3, 200.0, 1.0, 5.0, 50)


def test_ccd_denklemi_sentetik_fotometriyle():
    # Tek yıldızın sabit açıklıktaki akısının kareden kareye saçılımı CCD denkleminin öngördüğü kadardır.
    u = np.random.default_rng(84)
    bicim, poz, gokyuzu, aki = (41, 41), 30.0, 15.0, 400.0
    sahne = gokyuzu + dd.yildiz_alani(bicim, [20.0], [20.0], [aki], fwhm_piksel=3.0)
    yy, xx = np.mgrid[:41, :41]
    aciklik = (xx - 20) ** 2 + (yy - 20) ** 2 <= 6.0**2
    d = dd.Dedektor(kazanc_e_adu=1.0, okuma_gurultusu_e=8.0, karanlik_akim_e_s=0.0, bias_adu=0.0)
    olcum = []
    for _ in range(600):
        kare = dd.ham_kare_uret(sahne, poz, d, u)
        olcum.append(kare[aciklik].sum() - aciklik.sum() * gokyuzu * poz)
    olcum = np.array(olcum)
    kaynak_e = (sahne - gokyuzu)[aciklik].sum() * poz
    beklenen = float(dd.sinyal_gurultu_orani(kaynak_e, gokyuzu * poz, 0.0, 8.0, aciklik.sum()))
    assert olcum.mean() == pytest.approx(kaynak_e, rel=0.01)
    assert olcum.mean() / olcum.std(ddof=1) == pytest.approx(beklenen, rel=0.08)
