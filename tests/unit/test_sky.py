"""Kaydırmalı karelerden gökyüzü çıkarmanın sentetik yakın kızılötesi dizisiyle sınanması."""

import numpy as np
import pytest

from kozmos.observing import detector
from kozmos.photometry import sources
from kozmos.reduction import sky

BICIM = (96, 96)
GURULTU = 3.0


def _dizi(tohum, kare_sayisi=12, bicim_degisir=False):
    """Parlak ve yapılı gökyüzü + her karede kaydırılmış aynı yıldız alanı."""
    u = np.random.default_rng(tohum)
    yy, xx = np.mgrid[: BICIM[0], : BICIM[1]]
    desen = 1.0 + 0.08 * np.sin(xx / 15.0) + 0.05 * (yy / BICIM[0])  # dedektör/optik kaynaklı sabit yapı
    egim = (xx - BICIM[1] / 2) / BICIM[1]
    x0, y0 = u.uniform(20, 76, 12), u.uniform(20, 76, 12)
    flux = 10 ** u.uniform(2.8, 3.6, 12)
    times = np.arange(kare_sayisi, dtype=float)
    kaydirma = u.uniform(-12, 12, (kare_sayisi, 2))
    frames, yildizlar = [], []
    for k in range(kare_sayisi):
        duzey = 2000.0 * (1.0 + 0.15 * np.sin(times[k] / 2.5))  # gökyüzü dakikalar içinde %15 oynar
        shape = desen * (1.0 + (0.06 * (times[k] - kare_sayisi / 2) / kare_sayisi * egim if bicim_degisir else 0.0))
        yildiz = detector.star_field(BICIM, x0 + kaydirma[k, 0], y0 + kaydirma[k, 1], flux, 3.0)
        frames.append(duzey * shape + yildiz + u.normal(0, GURULTU, BICIM))
        yildizlar.append(yildiz)
    return np.array(frames), np.array(yildizlar), times, (x0, y0, flux, kaydirma)


def test_duragan_model_gokyuzunu_cikarir_yildizi_korur():
    frames, yildizlar, _, (x0, y0, flux, kaydirma) = _dizi(141)
    cikti, model = sky.subtract_static_sky(frames)
    residual = cikti - yildizlar
    # Gökyüzü 2000 düzeyinden gürültü düzeyine iner. Kalan düzey sıfır değildir: en yüksek değeri atmak
    # ortancayı aşağı çeker ve artığı gürültünün onda biri kadar artıya kaydırır.
    assert np.abs(np.median(residual, axis=(1, 2))).max() < 0.3 * GURULTU
    assert residual.std() < 1.25 * GURULTU
    assert np.median(model) == pytest.approx(1.0, abs=0.01)
    # Yıldız akısı: aynı açıklıklarla gürültüsüz, gökyüzüsüz yıldız karesinde ölçülen akıyla karşılaştırılır.
    # Alan kalabalıktır (12 yıldız, ±12 piksel kaydırma): maskesiz model yıldız ışığını içerir ve akıyı eksiltir.
    # Nesne maskesiyle kurulan model akıyı %3 içinde korur.
    k = 5
    xk, yk = x0 + kaydirma[k, 0], y0 + kaydirma[k, 1]

    def ratio(image):
        olculen, _ = sources.aperture_photometry(image, xk, yk, 7.0, annulus=(10.0, 13.0))
        gercek, _ = sources.aperture_photometry(yildizlar[k], xk, yk, 7.0, annulus=(10.0, 13.0))
        return float(np.median(olculen / gercek))

    maskeli, _ = sky.subtract_static_sky(frames, masks=yildizlar > GURULTU)
    print(f"akı oranı: maskesiz {ratio(cikti[k]):.3f}, maskeli {ratio(maskeli[k]):.3f}")
    assert ratio(maskeli[k]) == pytest.approx(1.0, abs=0.03)
    assert ratio(cikti[k]) < ratio(maskeli[k])
    assert np.all(flux > 0)


def test_en_yuksek_pikseli_atmak_yildiz_kalintisini_azaltir():
    frames, yildizlar, _, _ = _dizi(142)
    atmadan, model_atmadan = sky.subtract_static_sky(frames, reject_highest=0)
    atarak, model_atarak = sky.subtract_static_sky(frames, reject_highest=1)
    # Bir yıldızın başka karelerde üstüne düştüğü, bu karede ise boş olan pikseller: maskesiz ortanca burada
    # yıldız ışığıyla yukarı çekilir ve çıkarma sonrası eksi bir çukur bırakır. En yükseği atmak çukuru sığlaştırır.
    for k in range(frames.shape[0]):
        golge = (np.delete(yildizlar, k, axis=0).max(axis=0) > 200.0) & (yildizlar[k] < 1.0)
        if golge.sum() > 30:
            break
    cukur_atmadan = np.mean((atmadan - yildizlar)[k][golge])
    cukur_atarak = np.mean((atarak - yildizlar)[k][golge])
    assert cukur_atmadan < 0.0 and cukur_atarak > cukur_atmadan
    # Bedeli: yıldız görmeyen piksellerde model aşağı kayar, artık artıya geçer (gürültünün ~onda biri).
    bos = yildizlar.max(axis=0) < 1.0
    kayma = np.mean((atarak - yildizlar)[:, bos]) - np.mean((atmadan - yildizlar)[:, bos])
    assert 0.0 < kayma < 0.25 * GURULTU


def test_dinamik_model_degisen_bicimde_daha_iyi():
    frames, yildizlar, times, _ = _dizi(143, kare_sayisi=24, bicim_degisir=True)
    duragan, _ = sky.subtract_static_sky(frames)
    dinamik = sky.subtract_dynamic_sky(frames, times, n_neighbors=6)

    # Büyük ölçekli artık: 8×8 kutularda ortalama alınıp gürültü bastırılır, kalan gökyüzü yapısı ölçülür.
    def yapi(residual):
        kutu = residual[:, : 96 // 8 * 8, : 96 // 8 * 8].reshape(residual.shape[0], 12, 8, 12, 8).mean(axis=(2, 4))
        return kutu.std()

    assert yapi(dinamik - yildizlar) < 0.8 * yapi(duragan - yildizlar)


def test_girdi_denetimi():
    frames = np.ones((5, 8, 8))
    with pytest.raises(ValueError):
        sky.subtract_static_sky(frames[:2])
    with pytest.raises(ValueError):
        sky.subtract_dynamic_sky(frames, np.arange(4.0))
    with pytest.raises(ValueError):
        sky.subtract_dynamic_sky(frames, np.arange(5.0), n_neighbors=5)


def test_her_karede_maskeli_piksel_modelde_bosluk_birakmaz():
    # Gerçek SOFI dizisinde görülen durum: parlak bir nesnenin maskesi bir pikseli bütün karelerde örter.
    frames, yildizlar, times, _ = _dizi(144)
    masks = yildizlar > GURULTU
    masks[:, 40:44, 40:44] = True
    cikti, model = sky.subtract_static_sky(frames, masks=masks)
    assert np.all(np.isfinite(model)) and np.all(np.isfinite(cikti))
    assert np.all(np.isfinite(sky.subtract_dynamic_sky(frames, times, n_neighbors=4, masks=masks)))
