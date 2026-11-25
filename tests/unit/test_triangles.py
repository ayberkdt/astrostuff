"""Üçgen eşleme ve kör astrometrik çözümün sınanması."""

import numpy as np
import pytest

from kozmos.astrometry import triangles
from kozmos.astrometry.wcs import Wcs, pixel_to_sky


def test_bilinen_ucgenin_degismezleri():
    # 3-4-5 dik üçgeni: en kısa kenar 3, en uzun 5; R = 5/3. Köşe 1 en kısa ve en uzun kenarın ortak ucudur;
    # oradaki açının kosinüsü komşu kenar / hipotenüs = 3/5.
    u = triangles.build_triangles([0.0, 3.0, 3.0], [0.0, 0.0, 4.0])
    assert u.ratio[0] == pytest.approx(5.0 / 3.0) and u.cosine[0] == pytest.approx(0.6)
    assert u.log_perimeter[0] == pytest.approx(np.log(12.0))
    assert list(u.vertices[0]) == [0, 1, 2]  # köşe 1 = (0,0), köşe 2 = (3,0), köşe 3 = (3,4)
    # Oran sınırını aşan ince üçgen atılır.
    assert triangles.build_triangles([0.0, 1.0, 30.0], [0.0, 0.0, 0.1]).ratio.size == 0


def test_degismezler_donusum_altinda_korunur():
    rs = np.random.default_rng(211)
    x, y = rs.uniform(0, 100, 12), rs.uniform(0, 100, 12)
    a = np.radians(73.0)
    x2 = 2.7 * (np.cos(a) * x - np.sin(a) * y) + 40.0
    y2 = 2.7 * (np.sin(a) * x + np.cos(a) * y) - 15.0
    u, v = triangles.build_triangles(x, y), triangles.build_triangles(x2, y2)
    assert np.array_equal(u.vertices, v.vertices) and np.allclose(u.ratio, v.ratio) and np.allclose(u.cosine, v.cosine)
    assert np.array_equal(u.orientation, v.orientation) and np.allclose(v.log_perimeter - u.log_perimeter, np.log(2.7))
    # Ayna: R ve C aynı kalır, dolanım yönü ters döner.
    w = triangles.build_triangles(-x2, y2)
    assert (
        np.allclose(u.ratio, w.ratio)
        and np.allclose(u.cosine, w.cosine)
        and np.array_equal(u.orientation, -w.orientation)
    )


@pytest.mark.parametrize("ayna", [False, True])
def test_ucgen_esleme_karsiliklari_bulur(ayna):
    rs = np.random.default_rng(212)
    x, y = rs.uniform(0, 1000, 22), rs.uniform(0, 1000, 22)
    a = np.radians(211.0)
    x2 = 0.37 * (np.cos(a) * x - np.sin(a) * y) + 5.0 + rs.normal(0, 0.02, 22)
    y2 = 0.37 * (np.sin(a) * x + np.cos(a) * y) - 9.0 + rs.normal(0, 0.02, 22)
    if ayna:
        x2 = -x2
    # İkinci liste karıştırılır, 4 nokta çıkarılır, 4 sahte nokta eklenir.
    kalan = rs.permutation(22)[:18]
    x2 = np.concatenate([x2[kalan], rs.uniform(x2.min(), x2.max(), 4)])
    y2 = np.concatenate([y2[kalan], rs.uniform(y2.min(), y2.max(), 4)])
    i1, i2, oy, orientation = triangles.match_triangles(x, y, x2, y2)
    assert orientation == (-1 if ayna else 1)
    assert i1.size >= 12
    assert all(i2[k] < 18 and kalan[i2[k]] == i1[k] for k in range(i1.size))  # atanan her çift doğru
    assert np.all(np.diff(oy) <= 0)


def test_ilgisiz_listeler_eslesmez():
    rs = np.random.default_rng(213)
    i1, _, _, _ = triangles.match_triangles(*rs.uniform(0, 1, (2, 15)), *rs.uniform(0, 1, (2, 15)))
    assert i1.size <= 3  # rastlantısal oylar yarıya düşme kuralıyla kesilir


def test_kor_cozum_baslangic_bilgisi_olmadan():
    rs = np.random.default_rng(214)
    a = np.radians(128.0)
    olcek = 0.6 / 3600.0
    cd = olcek * np.array([[np.cos(a), np.sin(a)], [np.sin(a), -np.cos(a)]])  # dönük ve aynalı
    gercek = Wcs(crpix=np.array([400.0, 700.0]), crval=np.array([311.2, -37.4]), cd=cd)
    x, y = rs.uniform(0, 1500, 120), rs.uniform(0, 1500, 120)
    flux = 10 ** rs.uniform(2, 5, 120)
    ra, dec = pixel_to_sky(gercek, x, y)
    magnitude = -2.5 * np.log10(flux) + rs.normal(0, 0.05, 120)
    # Saptama: %85'i + 25 sahte; katalog: %85'i; konum gürültüsü 0,1 piksel.
    s = rs.random(120) < 0.85
    xs = np.concatenate([x[s] + rs.normal(0, 0.1, s.sum()), rs.uniform(0, 1500, 25)])
    ys = np.concatenate([y[s] + rs.normal(0, 0.1, s.sum()), rs.uniform(0, 1500, 25)])
    akis = np.concatenate([flux[s], 10 ** rs.uniform(2, 3, 25)])
    k = rs.random(120) < 0.85
    cozum, cift = triangles.blind_solve(xs, ys, akis, ra[k], dec[k], magnitude[k])
    kx, ky = np.meshgrid([0.0, 750.0, 1499.0], [0.0, 750.0, 1499.0])
    r1, d1 = pixel_to_sky(cozum.wcs, kx, ky)
    r2, d2 = pixel_to_sky(gercek, kx, ky)
    assert np.max(np.hypot((r1 - r2) * np.cos(np.radians(d2)), d1 - d2)) * 3600.0 < 0.1
    assert cift >= 8 and cozum.n_stars > 60 and cozum.residual_rms_arcsec < 0.2
    with pytest.raises(ValueError):
        triangles.blind_solve(xs, ys, akis, ra[k] + 20.0 * rs.random(k.sum()), dec[k], rs.permutation(magnitude[k]))


def test_kor_cozum_yanlis_cozumu_reddeder():
    # Katalog görüntünün alanını örtmüyorsa (burada: başka bir bölgenin yoğun kataloğu) rastlantısal üçgen
    # eşleşmelerinden çözüm kurulabilir; doğrulama adımı bunu reddetmelidir.
    rs = np.random.default_rng(215)
    for _ in range(5):
        x, y = rs.uniform(0, 1000, 60), rs.uniform(0, 1000, 60)
        ra, dec = 150.0 + rs.uniform(-0.1, 0.1, 400), 20.0 + rs.uniform(-0.1, 0.1, 400)
        with pytest.raises(ValueError):
            triangles.blind_solve(x, y, rs.uniform(1, 100, 60), ra, dec, rs.uniform(8, 16, 400))
