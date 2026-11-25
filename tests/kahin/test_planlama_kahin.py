import math

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from kozmos.gozlem import planlama

pytestmark = pytest.mark.kahin

ENLEM, BOYLAM = math.radians(36.82), math.radians(30.34)  # 36,82° K, 30,34° D


def _astropy(ra, dec, jd, **atmosfer):
    konum = EarthLocation(lat=ENLEM * u.rad, lon=BOYLAM * u.rad, height=0 * u.m)
    cerceve = AltAz(obstime=Time(jd, format="jd", scale="utc"), location=konum, **atmosfer)
    return SkyCoord(ra * u.rad, dec * u.rad, frame="icrs").transform_to(cerceve)


def _ayriklik_yaydakika(az1, alt1, az2, alt2):
    c = np.sin(alt1) * np.sin(alt2) + np.cos(alt1) * np.cos(alt2) * np.cos(az1 - az2)
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 60.0


def test_gorunurluk_astropy_ile():
    ra, dec = math.radians(85.42), math.radians(-1.90)
    jd = 2461370.5 + np.linspace(0, 1, 49)  # 2026-11-25, yarım saat aralıklarla
    bizim = planlama.gorunurluk(ra, dec, ENLEM, BOYLAM, jd)
    kahin = _astropy(ra, dec, jd, pressure=0 * u.hPa)
    # Yanlılık + presesyon uygulanır; kalan fark nutasyon, aberasyon, UT1 − UTC ve kutup hareketidir.
    fark = _ayriklik_yaydakika(bizim.azimut_rad, bizim.yukseklik_rad, kahin.az.rad, kahin.alt.rad)
    print(f"astropy ile en büyük yön farkı: {fark.max():.3f} yay dakikası")
    assert fark.max() < 1.0
    assert np.all(np.isnan(bizim.hava_kutlesi[bizim.yukseklik_rad <= 0]))
    assert np.nanmin(bizim.hava_kutlesi) >= 1.0
    # Presesyonsuz hesap aynı ölçüte göre onlarca kat kötüdür (26 yılda yaklaşık 0,36°).
    ham = planlama.gorunurluk(ra, dec, ENLEM, BOYLAM, jd, tarihe_tasi=False)
    assert _ayriklik_yaydakika(ham.azimut_rad, ham.yukseklik_rad, kahin.az.rad, kahin.alt.rad).max() > 10.0


def test_gorunurluk_kirilmayla_astropy_ile():
    from kozmos.gozlem import kirilma

    ra, dec = math.radians(85.42), math.radians(-1.90)
    jd = 2461370.5 + np.linspace(0, 1, 97)
    sabitler = kirilma.kirilma_sabitleri(1013.25, 10.0, 0.5, 0.55)
    bizim = planlama.gorunurluk(ra, dec, ENLEM, BOYLAM, jd, kirilma=sabitler)
    kahin = _astropy(
        ra, dec, jd, pressure=1013.25 * u.hPa, temperature=10 * u.deg_C, relative_humidity=0.5, obswl=0.55 * u.micron
    )
    ust = kahin.alt.deg > 10
    fark = _ayriklik_yaydakika(bizim.azimut_rad[ust], bizim.yukseklik_rad[ust], kahin.az.rad[ust], kahin.alt.rad[ust])
    assert ust.sum() > 20 and fark.max() < 1.0
    # Kırılma cismi yükseltir: 10° yükseklikte yaklaşık 5 yay dakikası.
    kirilmasiz = planlama.gorunurluk(ra, dec, ENLEM, BOYLAM, jd)
    kazanc = np.degrees(bizim.yukseklik_rad - kirilmasiz.yukseklik_rad)[ust] * 60.0
    assert np.all(kazanc > 0) and 3.0 < kazanc.max() < 6.5
    # Modelin geçersiz olduğu ufuk yakınında nan.
    assert np.all(np.isnan(bizim.yukseklik_rad[kirilmasiz.yukseklik_rad < math.radians(3.0)]))


def test_meridyen_gecisi():
    from kozmos.koordinat.presesyon import icrs_den_tarihin_ortalamasina

    ra, dec = math.radians(85.42), math.radians(-1.90)
    gecis = planlama.meridyen_gecisi_jd(ra, BOYLAM, 2461370.5, dik_acikllik_rad=dec)
    assert 2461370.5 <= gecis < 2461371.5
    an = planlama.gorunurluk(ra, dec, ENLEM, BOYLAM, gecis)
    assert abs(float(an.saat_acisi_rad)) < 1e-6  # bir günde presesyon 0,14 yay saniyesi: 6,6e-7 rad
    # Meridyende yükseklik 90° − |φ − δ(tarih)| ve bir saat önce ya da sonra daha düşük.
    dec_tarih = float(icrs_den_tarihin_ortalamasina(ra, dec, gecis)[1])
    assert float(an.yukseklik_rad) == pytest.approx(math.pi / 2 - abs(ENLEM - dec_tarih), abs=1e-9)
    komsu = planlama.gorunurluk(ra, dec, ENLEM, BOYLAM, gecis + np.array([-1, 1]) / 24)
    assert np.all(komsu.yukseklik_rad < an.yukseklik_rad)
    # Verildiği gibi kullanılan koordinatla saat açısı sıfır (GMST hızının sabit sayılmasından 6e-7 rad kalır).
    ham = planlama.meridyen_gecisi_jd(ra, BOYLAM, 2461370.5)
    assert abs(float(planlama.gorunurluk(ra, dec, ENLEM, BOYLAM, ham, tarihe_tasi=False).saat_acisi_rad)) < 1e-6


def test_ufuk_ustu_suresi():
    dec = np.radians([-60.0, -20.0, 0.0, 40.0])
    assert planlama.ufuk_ustu_suresi_saat(dec, 0.0) == pytest.approx(12.0)  # ekvatorda her yıldız yarım gün
    assert float(planlama.ufuk_ustu_suresi_saat(math.radians(80.0), ENLEM)) == 24.0  # batmayan yıldız
    assert float(planlama.ufuk_ustu_suresi_saat(math.radians(-70.0), ENLEM)) == 0.0  # doğmayan yıldız
    assert float(planlama.ufuk_ustu_suresi_saat(0.0, ENLEM)) == pytest.approx(12.0)  # gök ekvatoru
    # Sınır yükseldikçe süre kısalır.
    assert planlama.ufuk_ustu_suresi_saat(0.0, ENLEM, math.radians(30)) < planlama.ufuk_ustu_suresi_saat(0.0, ENLEM)
