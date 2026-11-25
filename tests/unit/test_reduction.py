"""Dedektör modeli ve indirgeme hattının sentetik veriyle sınanması.

Gerçeği bilinen bir sahne üretilir, ham karelere çevrilir, indirgenir ve
sonuç girilen sahneyle karşılaştırılır.
"""

import numpy as np
import pytest

from kozmos.observing import detector
from kozmos.reduction import calibration

BICIM = (120, 160)
DEDEKTOR = detector.Detector(gain_e_adu=1.7, read_noise_e=6.0, dark_current_e_s=0.2, bias_adu=1000.0)


def _duz_alan(u):
    yy, xx = np.mgrid[: BICIM[0], : BICIM[1]]
    vinyet = 1.0 - 0.25 * (((xx - 80) / 110) ** 2 + ((yy - 60) / 110) ** 2)
    flat = vinyet * u.normal(1.0, 0.02, BICIM)
    return flat / flat.mean()


def _sahne(u, n_yildiz=25):
    x, y = u.uniform(10, BICIM[1] - 10, n_yildiz), u.uniform(10, BICIM[0] - 10, n_yildiz)
    flux = 10 ** u.uniform(2.0, 3.5, n_yildiz)
    return 20.0 + detector.star_field(BICIM, x, y, flux, fwhm_pix=3.2)  # gökyüzü 20 e/s/piksel


def test_yildiz_alani_akiyi_korur():
    sahne = detector.star_field((61, 61), [30.3], [29.6], [1000.0], fwhm_pix=2.5)
    assert sahne.sum() == pytest.approx(1000.0, rel=1e-9)
    # Dar PSF'te de: piksel merkezinde örnekleme akıyı korumazdı, piksel integrali korur.
    assert detector.star_field((41, 41), [20.4], [20.1], [500.0], fwhm_pix=0.8).sum() == pytest.approx(500.0, rel=1e-9)


def test_kazanc_ve_okuma_gurultusu_geri_kazanimi():
    u = np.random.default_rng(81)
    flat = _duz_alan(u)
    bos = np.zeros(BICIM)
    aydinlik = np.full(BICIM, 8000.0)  # e/s; 1 s poz
    f1, f2 = (detector.simulate_raw_frame(aydinlik, 1.0, DEDEKTOR, u, flat_field=flat) for _ in range(2))
    b1, b2 = (detector.simulate_raw_frame(bos, 0.0, DEDEKTOR, u) for _ in range(2))
    kazanc, okuma = calibration.gain_and_read_noise(f1, f2, b1, b2)
    assert kazanc == pytest.approx(DEDEKTOR.gain_e_adu, rel=0.03)
    assert okuma == pytest.approx(DEDEKTOR.read_noise_e, rel=0.03)


def test_indirgeme_sahneyi_geri_verir():
    u = np.random.default_rng(82)
    flat = _duz_alan(u)
    sahne = _sahne(u)
    bos = np.zeros(BICIM)
    poz = 60.0

    bias, _ = calibration.combine([detector.simulate_raw_frame(bos, 0.0, DEDEKTOR, u) for _ in range(25)])
    karanlik_adu, _ = calibration.combine([detector.simulate_raw_frame(bos, 300.0, DEDEKTOR, u) for _ in range(15)])
    karanlik_e_s = (karanlik_adu - bias) * DEDEKTOR.gain_e_adu / 300.0
    duzler, _ = calibration.combine(
        [detector.simulate_raw_frame(np.full(BICIM, 3000.0), 5.0, DEDEKTOR, u, flat_field=flat) for _ in range(15)]
    )
    master_flat = (duzler - bias) * DEDEKTOR.gain_e_adu - karanlik_e_s * 5.0
    master_flat /= master_flat.mean()
    assert np.std(master_flat / flat) < 0.004  # düz alan deseni binde birkaç düzeyinde geri kazanılır

    raw = detector.simulate_raw_frame(sahne, poz, DEDEKTOR, u, flat_field=flat)
    sonuc = calibration.reduce_frame(
        raw,
        bias,
        karanlik_e_s,
        master_flat,
        gain_e_adu=DEDEKTOR.gain_e_adu,
        read_noise_e=DEDEKTOR.read_noise_e,
        exposure_s=poz,
    )
    residual = (sonuc.data - sahne) / np.sqrt(sonuc.variance)
    # Yansız (ortalama artık sıfır dolayında) ve varyans düzlemi gerçek saçılımı öngörüyor (χ²/N ≈ 1).
    assert abs(residual.mean()) < 0.05
    assert np.mean(residual**2) == pytest.approx(1.0, abs=0.08)
    # Düz alan düzeltmesi yapılmasaydı vinyet artığı kalırdı.
    duzeltmesiz = calibration.reduce_frame(
        raw, bias, karanlik_e_s, 1.0, gain_e_adu=1.7, read_noise_e=6.0, exposure_s=poz
    )
    assert np.mean(((duzeltmesiz.data - sahne) / np.sqrt(duzeltmesiz.variance)) ** 2) > 3.0


def test_birlestirme_kozmik_isini_ayiklar():
    u = np.random.default_rng(83)
    frames = u.normal(100.0, 5.0, (9, 40, 50))
    temiz_ortalama = frames.mean(axis=0)
    vurulan = u.integers(0, 9, 30), u.integers(0, 40, 30), u.integers(0, 50, 30)
    frames[vurulan] += 5000.0
    kirpmasiz, _ = calibration.combine(frames, sigma=None)
    kirpmali, kullanilan = calibration.combine(frames, sigma=2.5)
    assert np.abs(kirpmasiz - temiz_ortalama).max() > 400
    assert np.abs(kirpmali - temiz_ortalama).max() < 6.0
    assert kullanilan.min() >= 6 and kullanilan[vurulan[1], vurulan[2]].max() <= 8
    ortanca, _ = calibration.combine(frames, method="ortanca", sigma=None)
    assert np.abs(ortanca - 100.0).max() < 12
    with pytest.raises(ValueError):
        calibration.combine(frames[0])
    with pytest.raises(ValueError):
        calibration.combine(frames, method="kip")


def test_ccd_denklemi_limitleri():
    # Kaynak baskın: S/N → sqrt(N*).
    assert float(detector.signal_to_noise(1e6, 0.0, 0.0, 0.0, 10)) == pytest.approx(1e3)
    # Gökyüzü baskın: S/N → N* / sqrt(n_pix · N_S).
    assert float(detector.signal_to_noise(100.0, 1e5, 0.0, 5.0, 50)) == pytest.approx(100 / np.sqrt(50 * 1e5), rel=1e-3)
    # Arka plan az pikselden kestirilirse gürültü artar.
    assert detector.signal_to_noise(1e3, 200.0, 1.0, 5.0, 50, 100) < detector.signal_to_noise(1e3, 200.0, 1.0, 5.0, 50)


def test_ccd_denklemi_sentetik_fotometriyle():
    # Tek yıldızın sabit açıklıktaki akısının kareden kareye saçılımı CCD denkleminin öngördüğü kadardır.
    u = np.random.default_rng(84)
    shape, poz, gokyuzu, flux = (41, 41), 30.0, 15.0, 400.0
    sahne = gokyuzu + detector.star_field(shape, [20.0], [20.0], [flux], fwhm_pix=3.0)
    yy, xx = np.mgrid[:41, :41]
    aciklik = (xx - 20) ** 2 + (yy - 20) ** 2 <= 6.0**2
    d = detector.Detector(gain_e_adu=1.0, read_noise_e=8.0, dark_current_e_s=0.0, bias_adu=0.0)
    olcum = []
    for _ in range(600):
        kare = detector.simulate_raw_frame(sahne, poz, d, u)
        olcum.append(kare[aciklik].sum() - aciklik.sum() * gokyuzu * poz)
    olcum = np.array(olcum)
    source_e = (sahne - gokyuzu)[aciklik].sum() * poz
    beklenen = float(detector.signal_to_noise(source_e, gokyuzu * poz, 0.0, 8.0, aciklik.sum()))
    assert olcum.mean() == pytest.approx(source_e, rel=0.01)
    assert olcum.mean() / olcum.std(ddof=1) == pytest.approx(beklenen, rel=0.08)


def test_arka_plan_kestirimi_terimi_monte_carlo():
    # (1 + n_pix/n_B) çarpanı bu projede türetildi; burada doğrudan benzetimle sınanır.
    # Kaynak yok (N* -> 0 limiti): açıklık toplamından n_B pikselin ortalamasıyla kestirilen arka plan çıkarılır.
    u = np.random.default_rng(191)
    n_pix, n_b, gokyuzu, okuma = 50, 100, 200.0, 5.0
    deneme = 40000
    aciklik = u.poisson(gokyuzu, (deneme, n_pix)) + u.normal(0, okuma, (deneme, n_pix))
    annulus = u.poisson(gokyuzu, (deneme, n_b)) + u.normal(0, okuma, (deneme, n_b))
    olculen = (aciklik.sum(axis=1) - n_pix * annulus.mean(axis=1)).std()
    # S/N = N*/gürültü olduğundan N* = 1e-9 ile 1/(S/N) gürültünün kendisidir (kaynak Poisson terimi ihmal).
    beklenen = 1e-9 / float(detector.signal_to_noise(1e-9, gokyuzu, 0.0, okuma, n_pix, n_b))
    terimsiz = 1e-9 / float(detector.signal_to_noise(1e-9, gokyuzu, 0.0, okuma, n_pix))
    assert olculen == pytest.approx(beklenen, rel=0.015)
    assert beklenen / terimsiz == pytest.approx(np.sqrt(1.5), rel=1e-9) and abs(olculen / terimsiz - 1) > 0.15
