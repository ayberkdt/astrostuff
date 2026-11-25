"""Arşiv erişimi: CSV çözümü ağsız; gerçek sorgular ``-m ag`` ile."""

import numpy as np
import pytest

from kozmos.archives import cache, gaia, skyview, tap
from kozmos.radiation.magnitudes import distance_modulus_from_pc


def test_csv_coz(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text(
        "source_id,ra,bp_rp,ad\n1926461164913660160,355.48,3.53,x\n762815470562110464,165.83,,y\n", encoding="utf-8"
    )
    s = tap.read_csv(path)
    assert s["source_id"].dtype.kind == "U" and s["source_id"][0] == "1926461164913660160"  # 2^53 üstü: dizgi
    assert s["ra"].dtype == float and np.isnan(s["bp_rp"][1]) and list(s["ad"]) == ["x", "y"]
    path.write_text("<html>hata</html>", encoding="utf-8")
    with pytest.raises(ValueError):
        tap.read_csv(path)


def test_mutlak_kadir():
    # 10 pc (ϖ = 100 mas) uzaklıkta mutlak kadir görünen kadire eşittir.
    assert float(gaia.absolute_magnitude(7.3, 100.0)) == pytest.approx(7.3, abs=1e-12)
    paralaks = np.array([2.0, 20.0, 250.0])
    # Uzaklık modülü tanımıyla tutarlılık: m − M = 5 log10(d / 10 pc), d = 1000/ϖ pc.
    assert 10.0 - gaia.absolute_magnitude(10.0, paralaks) == pytest.approx(
        distance_modulus_from_pc(1000.0 / paralaks), abs=1e-12
    )


def test_onbellek_anahtari_kararli():
    assert cache.key("a", "b") == cache.key("a", "b") != cache.key("ab", "")
    assert len(cache.key("x")) == 16


@pytest.mark.network
def test_gaia_sorgusu(tmp_path):
    s = gaia.nearby_stars(tmp_path, parallax_min_mas=200.0)
    assert 20 < len(s["ra"]) < 200 and np.all(s["parallax"] > 200.0)
    assert (tmp_path / "KAYNAK_KAYDI.json").exists()


@pytest.mark.network
def test_skyview_kesiti(tmp_path):
    image, header = skyview.download_cutout(85.42, -1.90, "2MASS-K", tmp_path, size_deg=0.1, pixels=64)
    assert image.shape == (64, 64) and np.isfinite(image).all() and header["NAXIS"] == 2


def test_heasarc_boru_metni_cozulur(tmp_path):
    from kozmos.archives import heasarc

    path = tmp_path / "t.txt"
    path.write_text(
        "name   |ra     |dec    |vmag\nHR 2491|101.287|-16.716|-1.46\nHR 7001|279.234| 38.784| 0.03\nHR 1   |1.0    |2.0    |\n",
        encoding="utf-8",
    )
    tablo = heasarc.read_pipe_text(path)
    assert list(tablo) == ["name", "ra", "dec", "vmag"]
    assert list(tablo["name"]) == ["HR 2491", "HR 7001", "HR 1"]
    assert tablo["vmag"][0] == -1.46 and np.isnan(tablo["vmag"][2]) and tablo["dec"][1] == 38.784
    path.write_text("hata", encoding="utf-8")
    with pytest.raises(ValueError):
        heasarc.read_pipe_text(path)


@pytest.mark.network
def test_parlak_yildiz_katalogu_sirius(tmp_path):
    from kozmos.archives import heasarc

    tablo = heasarc.bright_stars(tmp_path)
    v = tablo["vmag"]
    # Katalogdaki en parlak yıldız Sirius'tur (HR 2491); V = -1,46 [HoffleitWarren1991].
    i = int(np.nanargmin(v))
    assert v[i] == pytest.approx(-1.46, abs=0.01) and int(tablo["hr"][i]) == 2491
    assert 9000 < len(v) < 9200


@pytest.mark.network
def test_2mass_koni_sorgusu(tmp_path):
    from kozmos.archives import twomass
    from kozmos.core.angles import angular_separation

    k = twomass.point_sources(85.42, -1.90, 0.05, tmp_path)
    assert 50 < len(k["ra"]) < 400 and k["ph_qual"].dtype.kind == "U"
    # Bütün kaynaklar koninin içindedir.
    ayriklik = angular_separation(np.radians(85.42), np.radians(-1.90), np.radians(k["ra"]), np.radians(k["dec"]))
    assert np.degrees(ayriklik).max() <= 0.05 + 1e-9
    assert 4.0 < np.nanmin(k["k_m"]) < np.nanmax(k["k_m"]) < 17.5


@pytest.mark.network
def test_gaia_koni_sorgusu_ve_donem_tasimasi(tmp_path):
    from kozmos.core.angles import angular_separation

    raw = gaia.cone_search(194.2425, 22.3912, 0.05, tmp_path)
    eski = gaia.cone_search(194.2425, 22.3912, 0.05, tmp_path, epoch_yr=2012.2)
    assert len(raw["ra"]) > 20 and np.all(raw["ref_epoch"] == 2016.0)
    ayriklik = angular_separation(
        np.radians(194.2425), np.radians(22.3912), np.radians(raw["ra"]), np.radians(raw["dec"])
    )
    assert np.degrees(ayriklik).max() <= 0.05 + 1e-9
    # 3,8 yıllık taşıma: kayma = toplam öz hareket × 3,8 yıl (mas → yay saniyesi).
    var = np.isfinite(raw["pmra"])
    kayma = (
        np.degrees(
            angular_separation(*np.radians([raw["ra"][var], raw["dec"][var], eski["ra"][var], eski["dec"][var]]))
        )
        * 3600.0
    )
    beklenen = np.hypot(raw["pmra"][var], raw["pmdec"][var]) * 3.8 / 1000.0
    assert np.allclose(kayma, beklenen, rtol=1e-4, atol=1e-6)
    assert np.array_equal(raw["ra"][~var], eski["ra"][~var])


def test_indirme_gecici_hatada_yeniden_dener(tmp_path, monkeypatch):
    import io
    import urllib.error

    cagrilar = []

    class Yanit(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def sahte(istek, timeout):
        cagrilar.append(istek.full_url)
        if len(cagrilar) < 3:
            raise urllib.error.URLError("geçici hata") if len(cagrilar) == 1 else TimeoutError("zaman aşımı")
        return Yanit(b"veri")

    monkeypatch.setattr(cache.urllib.request, "urlopen", sahte)
    path = cache.download("https://ornek.test/dosya", tmp_path / "d.bin", wait_s=0.0)
    assert path.read_bytes() == b"veri" and len(cagrilar) == 3
    # Kalıcı hata: bütün denemeler tükenince hata deneme sayısıyla yükselir; hedef dosya oluşmaz.
    cagrilar.clear()
    monkeypatch.setattr(
        cache.urllib.request, "urlopen", lambda istek, timeout: (_ for _ in ()).throw(TimeoutError("x"))
    )
    with pytest.raises(OSError, match="3 denemede"):
        cache.download("https://ornek.test/yok", tmp_path / "y.bin", wait_s=0.0)
    assert not (tmp_path / "y.bin").exists()
    # 4xx yanıtı yeniden denenmez.
    sayac = []

    def dort_yuz_dort(istek, timeout):
        sayac.append(1)
        raise urllib.error.HTTPError(istek.full_url, 404, "yok", None, None)

    monkeypatch.setattr(cache.urllib.request, "urlopen", dort_yuz_dort)
    with pytest.raises(urllib.error.HTTPError):
        cache.download("https://ornek.test/404", tmp_path / "z.bin", wait_s=0.0)
    assert len(sayac) == 1


def test_skyview_bozuk_yaniti_yeniden_dener(tmp_path, monkeypatch):
    from astropy.io import fits as astropy_fits

    gecerli = tmp_path / "gecerli.fits"
    astropy_fits.PrimaryHDU(np.arange(16, dtype=np.float32).reshape(4, 4)).writeto(gecerli)
    cagrilar = []

    def sahte_indir(url, target, **secenekler):
        cagrilar.append(secenekler["refresh"])
        target.write_bytes(
            b"<html>Service temporarily unavailable</html>" if len(cagrilar) < 3 else gecerli.read_bytes()
        )
        return target

    monkeypatch.setattr(skyview.cache, "download", sahte_indir)
    cache_dir = tmp_path / "o"
    cache_dir.mkdir()
    image, _ = skyview.download_cutout(10.0, 20.0, "2MASS-K", cache_dir, pixels=4)
    assert image.shape == (4, 4) and cagrilar == [False, True, True]
    assert len(list(cache_dir.glob("*.hatali*"))) == 2  # bozuk yanıtlar kanıt olarak saklanır
    # Hep bozuk yanıt: hata, yanıtın başını taşır ve önbellekte geçerli sayılacak dosya kalmaz.
    cagrilar.clear()
    monkeypatch.setattr(
        skyview.cache, "indir", lambda url, target, **s: (target.write_bytes(b"<html>hata</html>"), target)[1]
    )
    with pytest.raises(ValueError, match="geçerli FITS döndürmedi"):
        skyview.download_cutout(11.0, 20.0, "2MASS-K", cache_dir, pixels=4)
    assert not list(cache_dir.glob("skyview_*.fits"))[1:]  # yalnızca ilk çağrının geçerli dosyası durur
