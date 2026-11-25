"""Kozmik ışın saptamanın sentetik veriyle sınanması."""

import numpy as np

from kozmos.gozlem import dedektor as dd
from kozmos.indirgeme import kozmik_isin as ki

BICIM = (160, 160)
DEDEKTOR = dd.Dedektor(kazanc_e_adu=2.0, okuma_gurultusu_e=5.0, karanlik_akim_e_s=0.0, bias_adu=0.0)


def _kare(tohum, fwhm=3.0, n_yildiz=30, n_isin=60):
    u = np.random.default_rng(tohum)
    x, y = u.uniform(8, BICIM[1] - 8, n_yildiz), u.uniform(8, BICIM[0] - 8, n_yildiz)
    sahne = 50.0 + dd.yildiz_alani(BICIM, x, y, 10 ** u.uniform(2.5, 4.3, n_yildiz), fwhm)  # e/s
    temiz = dd.ham_kare_uret(sahne, 20.0, DEDEKTOR, u)  # ADU
    kirli = temiz.copy()
    gercek = np.zeros(BICIM, dtype=bool)
    # Tek piksellik vuruşlar ve 2-4 piksellik kısa izler; her piksel gürültünün en az ~10 katı.
    for _ in range(n_isin):
        y0, x0 = u.integers(3, BICIM[0] - 3), u.integers(3, BICIM[1] - 3)
        uzunluk = u.integers(1, 5)
        dy, dx = (0, 1) if u.random() < 0.5 else (1, 0)
        for k in range(uzunluk):
            yy, xx = y0 + k * dy, x0 + k * dx
            if 0 <= yy < BICIM[0] and 0 <= xx < BICIM[1]:
                kirli[yy, xx] += u.uniform(400.0, 4000.0)
                gercek[yy, xx] = True
    return temiz, kirli, gercek, (x, y)


def test_kozmik_isinlar_bulunur_ve_yildizlar_korunur():
    temiz, kirli, gercek, (x, y) = _kare(131)
    sonuc, maske = ki.kozmik_isin_bul(kirli, kazanc_e_adu=2.0, okuma_gurultusu_e=5.0)
    tamlik = (maske & gercek).sum() / gercek.sum()
    yanlis = (maske & ~gercek).sum()
    assert tamlik > 0.95
    assert yanlis <= 0.002 * maske.size  # yanlış işaretlenen piksel binde ikinin altında
    # Yıldız merkezleri işaretlenmez (iyi örneklenmiş PSF, FWHM = 3 piksel).
    merkez = np.zeros(BICIM, dtype=bool)
    merkez[np.round(y).astype(int), np.round(x).astype(int)] = True
    assert not (maske & merkez & ~gercek).any()
    # Temizlenen görüntü vuruşlu piksellerde temiz kareye, kirli kareden çok daha yakındır.
    assert np.abs(sonuc - temiz)[gercek].mean() < 0.05 * np.abs(kirli - temiz)[gercek].mean()


def test_temiz_karede_yanlis_alarm_az():
    temiz, _, _, _ = _kare(132, n_isin=0)
    _, maske = ki.kozmik_isin_bul(temiz, kazanc_e_adu=2.0, okuma_gurultusu_e=5.0)
    assert maske.sum() <= 0.001 * maske.size


def test_esik_buyudukce_daha_az_piksel():
    _, kirli, _, _ = _kare(133)
    _, gevsek = ki.kozmik_isin_bul(kirli, kazanc_e_adu=2.0, okuma_gurultusu_e=5.0, sigma_lim=4.0)
    _, siki = ki.kozmik_isin_bul(kirli, kazanc_e_adu=2.0, okuma_gurultusu_e=5.0, sigma_lim=12.0)
    assert siki.sum() <= gevsek.sum()
