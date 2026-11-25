"""Arka plan, kaynak saptama ve açıklık fotometrisinin sentetik veriyle sınanması."""

import numpy as np
import pytest

from kozmos.fotometri import kaynak as fk
from kozmos.gozlem import dedektor as dd

BICIM = (256, 256)
FWHM = 3.0


def _alan(tohum, n_yildiz=40, gokyuzu=200.0, egim=0.0, gurultu=5.0):
    u = np.random.default_rng(tohum)
    # Kaynaklar birbirinden en az 12 piksel uzak olsun: örtüşme bu testin konusu değil.
    konumlar = []
    while len(konumlar) < n_yildiz:
        aday = u.uniform(15, BICIM[0] - 15, 2)
        if all(np.hypot(*(aday - k)) > 12 for k in konumlar):
            konumlar.append(aday)
    x, y = np.array(konumlar).T
    aki = 10 ** u.uniform(3.0, 4.5, n_yildiz)
    yy, xx = np.mgrid[: BICIM[0], : BICIM[1]]
    zemin = gokyuzu + egim * (xx + yy)
    goruntu = zemin + dd.yildiz_alani(BICIM, x, y, aki, FWHM) + u.normal(0, gurultu, BICIM)
    return goruntu, zemin, x, y, aki


def test_arka_plan_duzeyi_yansiz():
    goruntu, _, *_ = _alan(111)
    duzey, rms = fk.arka_plan_kestir(goruntu)
    assert duzey == pytest.approx(200.0, abs=0.3)  # yıldızlara karşın; düz ortalama yanlı olurdu
    assert rms == pytest.approx(5.0, rel=0.1)
    assert goruntu.mean() - 200.0 > 1.0


def test_arka_plan_haritasi_egimi_izler():
    goruntu, zemin, *_ = _alan(112, egim=0.05)
    harita, rms = fk.arka_plan_haritasi(goruntu, hucre=32)
    ic = (slice(32, -32), slice(32, -32))
    assert np.abs(harita[ic] / zemin[ic] - 1).max() < 0.01
    assert np.median(rms) == pytest.approx(5.0, rel=0.15)
    # Dik eğimde hücre içi değişim gürültüye eklenir: rms haritası gerçek gürültüden büyük çıkar.
    _, rms_dik = fk.arka_plan_haritasi(_alan(112, egim=0.3)[0], hucre=32)
    assert np.median(rms_dik) > 5.5


def test_kaynak_saptama_tamlik_ve_konum():
    goruntu, _, x, y, aki = _alan(113)
    harita, rms = fk.arka_plan_haritasi(goruntu, hucre=64)
    bulunan = fk.kaynak_bul(goruntu, harita, rms, esik_sigma=4.0, en_az_piksel=5)
    # Her gerçek yıldızın 1 piksel içinde bir saptaması var; fazladan saptama en çok 1.
    uzaklik = np.hypot(bulunan.x[None, :] - x[:, None], bulunan.y[None, :] - y[:, None])
    assert np.all(uzaklik.min(axis=1) < 1.0)
    assert len(bulunan) - len(x) <= 1
    assert np.all(np.diff(bulunan.aki) <= 0)  # akıya göre azalan sıra
    # Parlak yıldızlarda merkez hatası 0,1 pikselin altında.
    parlak = aki > 1e4
    assert np.all(uzaklik.min(axis=1)[parlak] < 0.1)


def test_yalniz_gurultude_yanlis_saptama_az():
    u = np.random.default_rng(114)
    gurultu = u.normal(0, 1, (512, 512))
    assert len(fk.kaynak_bul(gurultu, 0.0, 1.0, esik_sigma=3.0, en_az_piksel=5)) <= 2
    # En az piksel koşulu olmasaydı 3σ üstünde yüzlerce tek piksel çıkardı.
    assert len(fk.kaynak_bul(gurultu, 0.0, 1.0, esik_sigma=3.0, en_az_piksel=1)) > 200
    assert len(fk.kaynak_bul(np.zeros((8, 8)), 0.0, 1.0)) == 0


def test_yumusatma_soluk_kaynagi_cikarir():
    u = np.random.default_rng(115)
    sahne = dd.yildiz_alani((64, 64), [32.0], [32.0], [60.0], 4.0)  # tepe ≈ 3,3; gürültü 1
    goruntu = sahne + u.normal(0, 1, (64, 64))
    yumusak = fk.kaynak_bul(goruntu, 0.0, 1.0, esik_sigma=5.0, en_az_piksel=3, yumusatma_fwhm=4.0)
    assert len(fk.kaynak_bul(goruntu, 0.0, 1.0, esik_sigma=5.0, en_az_piksel=3)) == 0
    assert len(yumusak) == 1 and abs(yumusak.x[0] - 32) < 1.5 and abs(yumusak.y[0] - 32) < 1.5


def test_aciklik_alani_ve_cevrelenen_aki():
    aki, alan = fk.aciklik_fotometrisi(np.ones((60, 60)), [30.2], [29.7], 7.5, alt_piksel=9)
    assert alan[0] == pytest.approx(np.pi * 7.5**2, rel=2e-3) and aki[0] == pytest.approx(alan[0])
    # Gauss PSF'te r yarıçapı içindeki akı kesri 1 − exp(−r²/2σ²).
    sigma = FWHM / 2.3548200450309493
    sahne = 50.0 + dd.yildiz_alani((81, 81), [40.3], [39.8], [1e4], FWHM)

    def kesir(r):
        return fk.aciklik_fotometrisi(sahne, [40.3], [39.8], r, halka=(15.0, 20.0), alt_piksel=9)[0][0] / 1e4

    for r in (5.0, 7.0, 9.0):
        assert kesir(r) == pytest.approx(1 - np.exp(-(r**2) / (2 * sigma**2)), abs=5e-3)
    # Dar açıklıkta yöntemin sınırı: piksel içinde akı düzgün sayıldığı için PSF'in dik olduğu
    # yarıçaplarda kesir eksik çıkar (r = 2, FWHM = 3 için yaklaşık %4).
    assert 0.02 < (1 - np.exp(-(2.0**2) / (2 * sigma**2))) - kesir(2.0) < 0.06


def test_aciklik_fotometrisi_gurultulu_alanda_yansiz():
    goruntu, _, x, y, aki = _alan(116, n_yildiz=30)
    olculen, _ = fk.aciklik_fotometrisi(goruntu, x, y, 6.0, halka=(9.0, 12.0))
    sigma = FWHM / 2.3548200450309493
    beklenen = aki * (1 - np.exp(-(6.0**2) / (2 * sigma**2)))
    # Açıklıktaki gürültü: sqrt(π r²) · 5 ≈ 53; 30 yıldızın ortalama bağıl sapması yüzde birin altında.
    assert np.mean(olculen / beklenen - 1) == pytest.approx(0.0, abs=0.01)
    assert np.std((olculen - beklenen) / 60.0) < 2.0


def test_yakin_ciftler_ayristirilir():
    u = np.random.default_rng(171)
    bicim = (200, 200)
    # 16 çift: aralık 2 FWHM (aralarında eyer var), akı oranı 1 ile 2 arasında; çiftler birbirinden uzak.
    cx, cy = (a.ravel() for a in np.meshgrid(np.arange(25, 200, 50.0), np.arange(25, 200, 50.0)))
    aci = u.uniform(0, 2 * np.pi, 16)
    fwhm, aralik = 3.0, 6.0
    x = np.concatenate([cx - aralik / 2 * np.cos(aci), cx + aralik / 2 * np.cos(aci)])
    y = np.concatenate([cy - aralik / 2 * np.sin(aci), cy + aralik / 2 * np.sin(aci)])
    aki = np.concatenate([np.full(16, 4000.0), 4000.0 / u.uniform(1, 2, 16)])
    goruntu = 100.0 + dd.yildiz_alani(bicim, x, y, aki, fwhm) + u.normal(0, 2.0, bicim)
    tek = fk.kaynak_bul(goruntu, 100.0, 2.0, esik_sigma=4.0, yumusatma_fwhm=fwhm)
    ayri = fk.kaynak_bul(goruntu, 100.0, 2.0, esik_sigma=4.0, yumusatma_fwhm=fwhm, ayristirma_duzeyi=32)
    assert len(tek) == 16  # ayrıştırma olmadan her çift tek kaynak
    # 200 gerçeklemeli ölçümde (rastgele yön, alt piksel konum) bu ayar oran 1-2 çiftlerinin %92-98'ini ayırır.
    assert 29 <= len(ayri) <= 32
    uzaklik = np.hypot(ayri.x[None, :] - x[:, None], ayri.y[None, :] - y[:, None]).min(axis=1)
    assert np.sum(uzaklik < 1.0) >= 29  # ayrışan her bileşen bir pikselin içinde bulunur
    # Toplam akı korunur: ayrıştırma pikselleri paylaştırır, eklemez ya da silmez.
    assert ayri.aki.sum() == pytest.approx(tek.aki.sum(), rel=1e-12)
    assert ayri.piksel_sayisi.sum() == tek.piksel_sayisi.sum()


def test_ayristirma_siniri_sig_eyer():
    # Aynı aralıkta akı oranı büyüdükçe eyer sığlaşır. 200 gerçeklemeli ölçüm (2 FWHM aralık, rastgele yön):
    # 32 düzey ve 0,005 karşıtlık oran 3'te çiftlerin %14'ünü, oran 4'te %1'ini ayırır;
    # 128 düzey ve 0,001 karşıtlık oran 3'te %99'unu, oran 4'te %59'unu ayırır.
    u = np.random.default_rng(173)
    ortak = {"esik_sigma": 4.0, "yumusatma_fwhm": 3.0}
    kaba = ince = 0
    for _ in range(20):
        goruntu = 100.0 + dd.yildiz_alani((60, 60), [27.0, 33.0], [30.0, 30.0], [4000.0, 4000.0 / 3.0], 3.0)
        goruntu = goruntu + u.normal(0, 2.0, (60, 60))
        kaba += len(fk.kaynak_bul(goruntu, 100.0, 2.0, ayristirma_duzeyi=32, **ortak)) == 2
        ince += len(fk.kaynak_bul(goruntu, 100.0, 2.0, ayristirma_duzeyi=128, karsitlik=0.001, **ortak)) == 2
    assert kaba <= 5 and ince >= 17


def test_ayristirma_yalniz_yildizi_bolmez():
    u = np.random.default_rng(172)
    bicim = (200, 200)
    x, y = (a.ravel().astype(float) for a in np.meshgrid(np.arange(20, 200, 40), np.arange(20, 200, 40)))
    goruntu = 100.0 + dd.yildiz_alani(bicim, x, y, 10 ** u.uniform(2.6, 4.5, x.size), 3.0) + u.normal(0, 2.0, bicim)
    tek = fk.kaynak_bul(goruntu, 100.0, 2.0, esik_sigma=4.0, yumusatma_fwhm=3.0)
    ayri = fk.kaynak_bul(goruntu, 100.0, 2.0, esik_sigma=4.0, yumusatma_fwhm=3.0, ayristirma_duzeyi=32)
    assert len(tek) == x.size and len(ayri) == x.size


def test_arti_pikseli_olmayan_bolutun_merkezi_tanimli():
    # Saptama görüntüsü ile ölçüm görüntüsü ayrışınca (yumuşatma) bölütün artı ağırlığı sıfır olabilir.
    goruntu = np.zeros((40, 40))
    goruntu[18:23, 18:23] = 5.0
    liste = fk.kaynak_bul(goruntu, 6.0, 1.0, esik_sigma=-2.0, en_az_piksel=5)  # arka plan 6: bütün pikseller eksi
    assert len(liste) >= 1 and np.all(np.isfinite(liste.x)) and np.all(np.isfinite(liste.y))
