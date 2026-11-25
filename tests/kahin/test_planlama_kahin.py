import math

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from kozmos.gozlem import planlama

pytestmark = pytest.mark.kahin

ENLEM, BOYLAM = math.radians(36.82), math.radians(30.34)  # 36,82° K, 30,34° D


def test_gorunurluk_astropy_ile():
    ra, dec = math.radians(85.42), math.radians(-1.90)
    jd = 2461370.5 + np.linspace(0, 1, 49)  # 2026-11-25, yarım saat aralıklarla
    bizim = planlama.gorunurluk(ra, dec, ENLEM, BOYLAM, jd)
    konum = EarthLocation(lat=ENLEM * u.rad, lon=BOYLAM * u.rad, height=0 * u.m)
    cerceve = AltAz(obstime=Time(jd, format="jd", scale="utc"), location=konum, pressure=0 * u.hPa)
    kahin = SkyCoord(ra * u.rad, dec * u.rad, frame="icrs").transform_to(cerceve)
    # Presesyon (26 yılda yaklaşık 0,36°) ve nutasyon/aberasyon uygulanmadığı için tolerans 0,6°.
    assert np.max(np.abs(np.degrees(bizim.yukseklik_rad) - kahin.alt.deg)) < 0.6
    ust = kahin.alt.deg > 10
    fark = np.degrees(np.angle(np.exp(1j * (bizim.azimut_rad - kahin.az.rad))))
    assert np.max(np.abs(fark[ust])) < 0.8
    assert np.all(np.isnan(bizim.hava_kutlesi[bizim.yukseklik_rad <= 0]))
    assert np.nanmin(bizim.hava_kutlesi) >= 1.0


def test_meridyen_gecisi():
    ra, dec = math.radians(85.42), math.radians(-1.90)
    gecis = planlama.meridyen_gecisi_jd(ra, BOYLAM, 2461370.5)
    assert 2461370.5 <= gecis < 2461371.5
    an = planlama.gorunurluk(ra, dec, ENLEM, BOYLAM, gecis)
    assert abs(float(an.saat_acisi_rad)) < 1e-6
    # Meridyende yükseklik 90° − |φ − δ| ve bir saat önce ya da sonra daha düşük.
    assert float(an.yukseklik_rad) == pytest.approx(math.pi / 2 - abs(ENLEM - dec), abs=1e-9)
    komsu = planlama.gorunurluk(ra, dec, ENLEM, BOYLAM, gecis + np.array([-1, 1]) / 24)
    assert np.all(komsu.yukseklik_rad < an.yukseklik_rad)


def test_ufuk_ustu_suresi():
    dec = np.radians([-60.0, -20.0, 0.0, 40.0])
    assert planlama.ufuk_ustu_suresi_saat(dec, 0.0) == pytest.approx(12.0)  # ekvatorda her yıldız yarım gün
    assert float(planlama.ufuk_ustu_suresi_saat(math.radians(80.0), ENLEM)) == 24.0  # batmayan yıldız
    assert float(planlama.ufuk_ustu_suresi_saat(math.radians(-70.0), ENLEM)) == 0.0  # doğmayan yıldız
    assert float(planlama.ufuk_ustu_suresi_saat(0.0, ENLEM)) == pytest.approx(12.0)  # gök ekvatoru
    # Sınır yükseldikçe süre kısalır.
    assert planlama.ufuk_ustu_suresi_saat(0.0, ENLEM, math.radians(30)) < planlama.ufuk_ustu_suresi_saat(0.0, ENLEM)
