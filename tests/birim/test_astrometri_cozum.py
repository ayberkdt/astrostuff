"""Görüntüden astrometrik çözümün bilinen WCS'e karşı sınanması."""

import numpy as np
import pytest

from kozmos.astrometri import cozum
from kozmos.astrometri.wcs import Wcs, piksel_den_gok

OLCEK = 1.0 / 3600.0  # derece / piksel


def _gercek(aci_derece=17.0):
    a = np.radians(aci_derece)
    cd = OLCEK * np.array([[-np.cos(a), np.sin(a)], [np.sin(a), np.cos(a)]])
    return Wcs(crpix=np.array([512.3, 498.7]), crval=np.array([201.3, 41.2]), cd=cd)


def _alan(tohum, n=250, gurultu=0.08):
    u = np.random.default_rng(tohum)
    gercek = _gercek()
    x, y = u.uniform(0, 1024, n), u.uniform(0, 1024, n)
    ra, dec = piksel_den_gok(gercek, x, y)
    return gercek, x + u.normal(0, gurultu, n), y + u.normal(0, gurultu, n), ra, dec, u


def _en_buyuk_fark_yaysaniye(w1, w2):
    kx, ky = np.meshgrid([0.0, 512.0, 1023.0], [0.0, 512.0, 1023.0])
    r1, d1 = piksel_den_gok(w1, kx, ky)
    r2, d2 = piksel_den_gok(w2, kx, ky)
    return float(np.max(np.hypot((r1 - r2) * np.cos(np.radians(d1)), d1 - d2)) * 3600.0)


def test_wcs_uydur_bilinen_cozumu_bulur():
    gercek, x, y, ra, dec, _ = _alan(201)
    c = cozum.wcs_uydur(x, y, ra, dec, gercek.crval)
    assert np.max(np.abs(c.wcs.cd / gercek.cd - 1)) < 2e-4
    assert np.max(np.abs(c.wcs.crpix - gercek.crpix)) < 0.03
    # 0,08 piksel gürültü iki eksende: konum artığı rms ≈ 0,08 · √2 piksel = 0,11 yay saniyesi.
    assert c.artik_rms_yaysaniye == pytest.approx(0.08 * np.sqrt(2.0), rel=0.15)
    assert _en_buyuk_fark_yaysaniye(c.wcs, gercek) < 0.05
    # Teğet noktası verilmezse yıldızların ortalama yönü alınır; gökteki dönüşüm aynı kalır.
    serbest = cozum.wcs_uydur(x, y, ra, dec)
    assert _en_buyuk_fark_yaysaniye(serbest.wcs, gercek) < 0.05
    with pytest.raises(ValueError):
        cozum.wcs_uydur(x[:2], y[:2], ra[:2], dec[:2])


def test_gurultusuz_veride_tam_cozum():
    gercek, _, _, _, _, u = _alan(202)
    x, y = u.uniform(0, 1024, 50), u.uniform(0, 1024, 50)
    ra, dec = piksel_den_gok(gercek, x, y)
    c = cozum.wcs_uydur(x, y, ra, dec, gercek.crval)
    assert np.max(np.abs(c.wcs.cd - gercek.cd)) < 1e-13 and np.max(np.abs(c.wcs.crpix - gercek.crpix)) < 1e-7
    assert c.artik_rms_yaysaniye < 1e-6


def test_kaydirma_oylamayla_bulunur():
    gercek, x, y, ra, dec, u = _alan(203)
    # Saptama listesi: yıldızların %70'i + 80 sahte kaynak; katalog: yıldızların %80'i.
    sapt = u.random(x.size) < 0.7
    xs = np.concatenate([x[sapt], u.uniform(0, 1024, 80)])
    ys = np.concatenate([y[sapt], u.uniform(0, 1024, 80)])
    kat = u.random(x.size) < 0.8
    kayik = Wcs(crpix=gercek.crpix + np.array([-37.4, 21.8]), crval=gercek.crval, cd=gercek.cd)
    duzeltilmis, oy = cozum.kaydirma_bul(kayik, xs, ys, ra[kat], dec[kat])
    assert np.max(np.abs(duzeltilmis.crpix - gercek.crpix)) < 0.3
    assert oy > 0.4 * (sapt & kat).sum()  # ortak yıldızların çoğu tepe kutusunda


def test_iyilestirme_bozuk_baslangictan_yakinsar():
    gercek, x, y, ra, dec, u = _alan(204)
    xs = np.concatenate([x[:200], u.uniform(0, 1024, 60)])
    ys = np.concatenate([y[:200], u.uniform(0, 1024, 60)])
    # Başlangıç: 0,15° dönük, ölçeği %0,3 büyük, 12 piksel kayık (kenarlarda birkaç yay saniyesi hata).
    a = np.radians(0.15)
    don = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    bozuk = Wcs(crpix=gercek.crpix + np.array([12.0, -9.0]), crval=gercek.crval, cd=1.003 * don @ gercek.cd)
    assert _en_buyuk_fark_yaysaniye(bozuk, gercek) > 10.0
    kaymasiz, _ = cozum.kaydirma_bul(bozuk, xs, ys, ra, dec, en_cok_piksel=40.0)
    c = cozum.wcs_iyilestir(kaymasiz, xs, ys, ra, dec, yaricap_yaysaniye=6.0)
    assert _en_buyuk_fark_yaysaniye(c.wcs, gercek) < 0.1
    assert c.yildiz_sayisi >= 190 and c.artik_rms_yaysaniye < 0.2
    with pytest.raises(ValueError):
        cozum.wcs_iyilestir(gercek, xs[:2], ys[:2], ra + 5.0, dec, yaricap_yaysaniye=1.0)
