"""Presesyon ve kırılmanın ERFA ve astropy ile karşılaştırılması."""

import astropy.units as u
import erfa
import numpy as np
import pytest
from astropy.coordinates import PrecessedGeocentric, SkyCoord
from astropy.time import Time

from kozmos.gozlem import kirilma
from kozmos.koordinat import presesyon

pytestmark = pytest.mark.kahin

TARIHLER = 2451545.0 + np.array([-36525.0, -3652.5, 0.0, 9826.5, 18262.5, 36525.0])  # 1900-2100


def test_fukushima_williams_ve_matris_erfa_ile():
    for jd in TARIHLER:
        bizim = presesyon.fukushima_williams_acilari(jd, 0.25)
        assert np.allclose(bizim, erfa.pfw06(jd, 0.25), rtol=0, atol=1e-15)
        r = presesyon.yanlilik_presesyon_matrisi(jd, 0.25)
        assert np.max(np.abs(r - erfa.pmat06(jd, 0.25))) < 1e-14
        assert np.max(np.abs(r @ r.T - np.eye(3))) < 1e-14
    # Dizi girdisi: matrisler yığılır.
    assert presesyon.yanlilik_presesyon_matrisi(TARIHLER).shape == (TARIHLER.size, 3, 3)
    # J2000'de matris yalnızca çerçeve yanlılığıdır: birimden farkı 1e-7 düzeyinde (yaklaşık 23 miliyay saniyesi).
    j2000 = presesyon.yanlilik_presesyon_matrisi(2451545.0)
    assert 1e-8 < np.max(np.abs(j2000 - np.eye(3))) < 2e-7


def test_tarihin_ortalamasi_erfa_ve_astropy_ile():
    rs = np.random.default_rng(181)
    ra, dec = rs.uniform(0, 2 * np.pi, 200), np.arcsin(rs.uniform(-1, 1, 200))
    jd = 2461370.5
    a, d = presesyon.icrs_den_tarihin_ortalamasina(ra, dec, jd)
    a_k, d_k = erfa.c2s(erfa.rxp(erfa.pmat06(jd, 0.0), erfa.s2c(ra, dec)))
    assert np.max(np.abs(np.angle(np.exp(1j * (a - a_k))))) < 1e-13 and np.max(np.abs(d - d_k)) < 1e-13
    # 26 yılda presesyon yaklaşık 0,36°: taşınan yön özgün yönden bu kadar ayrılır.
    ayriklik = np.degrees(erfa.seps(ra, dec, a, d))
    assert 0.3 < np.median(ayriklik) < 0.4
    # Astropy'nin tarihin ortalama ekvatoru çerçevesi: fark aberasyonla sınırlı (20,5 yay saniyesi).
    kahin = SkyCoord(ra * u.rad, dec * u.rad, frame="icrs").transform_to(
        PrecessedGeocentric(equinox=Time(jd, format="jd", scale="tt"), obstime=Time(jd, format="jd", scale="tt"))
    )
    fark = np.degrees(erfa.seps(a, d, kahin.ra.rad, kahin.dec.rad)) * 3600.0
    assert fark.max() < 25.0


def test_kirilma_sabitleri_erfa_ile():
    for kosul in [
        (1013.25, 10.0, 0.5, 0.55),
        (780.0, -5.0, 0.2, 2.2),
        (1005.0, 7.0, 0.8, 0.574),
        (600.0, 0.0, 0.0, 10.0),
    ]:
        assert kirilma.kirilma_sabitleri(*kosul) == pytest.approx(erfa.refco(*kosul), rel=1e-13)
    assert kirilma.kirilma_sabitleri(0.0, 10.0, 0.5) == (0.0, 0.0)
    with pytest.raises(ValueError):
        kirilma.kirilma_sabitleri(1013.25, 10.0, 0.5, 1e4)


def test_kirilma_kaynak_tablosu():
    # refco.c açıklamasındaki tablo: 1005 mb, 280,15 K, %80 nem, 5740 Å; "eraRefco" sütunu [yay saniyesi].
    # Hesap erfa.refco ile tam eşittir ama tablodan en çok 0,015 yay saniyesi ayrılır (76° ve 80°'de);
    # farkın nedeni kaynakta açıklanmamıştır.
    a, b = kirilma.kirilma_sabitleri(1005.0, 7.0, 0.8, 0.574)
    tablo = {
        10: 10.27,
        20: 21.20,
        30: 33.61,
        40: 48.83,
        45: 58.18,
        50: 69.30,
        60: 100.54,
        70: 158.68,
        76: 229.43,
        80: 318.55,
    }
    for zd, beklenen in tablo.items():
        t = np.tan(np.radians(zd))
        assert np.degrees(a * t + b * t**3) * 3600.0 == pytest.approx(beklenen, abs=0.02)


def test_gozlenen_yukseklik_gidis_donus():
    a, b = kirilma.kirilma_sabitleri(1013.25, 10.0, 0.5)
    gozlenen = np.radians(np.linspace(10.0, 89.0, 80))
    t = np.tan(np.pi / 2 - gozlenen)
    toposantrik = gozlenen - (a * t + b * t**3)  # ileri model: Z_topo = Z_göz + dZ
    assert np.max(np.abs(kirilma.gozlenen_yukseklik(toposantrik, a, b) - gozlenen)) < 1e-11
    assert np.isnan(kirilma.gozlenen_yukseklik(np.radians(2.0), a, b))
    assert float(kirilma.gozlenen_yukseklik(np.pi / 2, a, b)) == pytest.approx(np.pi / 2)
