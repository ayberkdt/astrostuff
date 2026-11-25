import math

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from kozmos.observing import planning

pytestmark = pytest.mark.oracle

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
    bizim = planning.visibility(ra, dec, ENLEM, BOYLAM, jd)
    kahin = _astropy(ra, dec, jd, pressure=0 * u.hPa)
    # Yanlılık + presesyon uygulanır; kalan fark nutasyon, aberasyon, UT1 − UTC ve kutup hareketidir.
    fark = _ayriklik_yaydakika(bizim.azimuth_rad, bizim.altitude_rad, kahin.az.rad, kahin.alt.rad)
    print(f"astropy ile en büyük yön farkı: {fark.max():.3f} yay dakikası")
    assert fark.max() < 1.0
    assert np.all(np.isnan(bizim.airmass[bizim.altitude_rad <= 0]))
    assert np.nanmin(bizim.airmass) >= 1.0
    # Presesyonsuz hesap aynı ölçüte göre onlarca kat kötüdür (26 yılda yaklaşık 0,36°).
    raw = planning.visibility(ra, dec, ENLEM, BOYLAM, jd, to_date=False)
    assert _ayriklik_yaydakika(raw.azimuth_rad, raw.altitude_rad, kahin.az.rad, kahin.alt.rad).max() > 10.0


def test_gorunurluk_kirilmayla_astropy_ile():
    from kozmos.observing import refraction

    ra, dec = math.radians(85.42), math.radians(-1.90)
    jd = 2461370.5 + np.linspace(0, 1, 97)
    sabitler = refraction.refraction_constants(1013.25, 10.0, 0.5, 0.55)
    bizim = planning.visibility(ra, dec, ENLEM, BOYLAM, jd, refraction=sabitler)
    kahin = _astropy(
        ra, dec, jd, pressure=1013.25 * u.hPa, temperature=10 * u.deg_C, relative_humidity=0.5, obswl=0.55 * u.micron
    )
    high = kahin.alt.deg > 10
    fark = _ayriklik_yaydakika(
        bizim.azimuth_rad[high], bizim.altitude_rad[high], kahin.az.rad[high], kahin.alt.rad[high]
    )
    assert high.sum() > 20 and fark.max() < 1.0
    # Kırılma cismi yükseltir: 10° yükseklikte yaklaşık 5 yay dakikası.
    kirilmasiz = planning.visibility(ra, dec, ENLEM, BOYLAM, jd)
    kazanc = np.degrees(bizim.altitude_rad - kirilmasiz.altitude_rad)[high] * 60.0
    assert np.all(kazanc > 0) and 3.0 < kazanc.max() < 6.5
    # Modelin geçersiz olduğu ufuk yakınında nan.
    assert np.all(np.isnan(bizim.altitude_rad[kirilmasiz.altitude_rad < math.radians(3.0)]))


def test_meridyen_gecisi():
    from kozmos.coordinates.precession import icrs_to_mean_of_date

    ra, dec = math.radians(85.42), math.radians(-1.90)
    gecis = planning.transit_jd(ra, BOYLAM, 2461370.5, dec_rad=dec)
    assert 2461370.5 <= gecis < 2461371.5
    an = planning.visibility(ra, dec, ENLEM, BOYLAM, gecis)
    assert abs(float(an.hour_angle_rad)) < 1e-6  # bir günde presesyon 0,14 yay saniyesi: 6,6e-7 rad
    # Meridyende yükseklik 90° − |φ − δ(tarih)| ve bir saat önce ya da sonra daha düşük.
    dec_tarih = float(icrs_to_mean_of_date(ra, dec, gecis)[1])
    assert float(an.altitude_rad) == pytest.approx(math.pi / 2 - abs(ENLEM - dec_tarih), abs=1e-9)
    komsu = planning.visibility(ra, dec, ENLEM, BOYLAM, gecis + np.array([-1, 1]) / 24)
    assert np.all(komsu.altitude_rad < an.altitude_rad)
    # Verildiği gibi kullanılan koordinatla saat açısı sıfır (GMST hızının sabit sayılmasından 6e-7 rad kalır).
    raw = planning.transit_jd(ra, BOYLAM, 2461370.5)
    assert abs(float(planning.visibility(ra, dec, ENLEM, BOYLAM, raw, to_date=False).hour_angle_rad)) < 1e-6


def test_ufuk_ustu_suresi():
    dec = np.radians([-60.0, -20.0, 0.0, 40.0])
    assert planning.time_above_horizon_hr(dec, 0.0) == pytest.approx(12.0)  # ekvatorda her yıldız yarım gün
    assert float(planning.time_above_horizon_hr(math.radians(80.0), ENLEM)) == 24.0  # batmayan yıldız
    assert float(planning.time_above_horizon_hr(math.radians(-70.0), ENLEM)) == 0.0  # doğmayan yıldız
    assert float(planning.time_above_horizon_hr(0.0, ENLEM)) == pytest.approx(12.0)  # gök ekvatoru
    # Sınır yükseldikçe süre kısalır.
    assert planning.time_above_horizon_hr(0.0, ENLEM, math.radians(30)) < planning.time_above_horizon_hr(0.0, ENLEM)
