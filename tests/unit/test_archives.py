"""Archive access: CSV parsing without the network; real queries with ``-m network``."""

import numpy as np
import pytest

from kozmos.archives import cache, gaia, skyview, tap
from kozmos.radiation.magnitudes import distance_modulus_from_pc


def test_parse_csv(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text(
        "source_id,ra,bp_rp,name\n1926461164913660160,355.48,3.53,x\n762815470562110464,165.83,,y\n", encoding="utf-8"
    )
    s = tap.read_csv(path)
    assert s["source_id"].dtype.kind == "U" and s["source_id"][0] == "1926461164913660160"  # above 2^53: string
    assert s["ra"].dtype == float and np.isnan(s["bp_rp"][1]) and list(s["name"]) == ["x", "y"]
    path.write_text("<html>error</html>", encoding="utf-8")
    with pytest.raises(ValueError):
        tap.read_csv(path)


def test_absolute_magnitude():
    # At 10 pc (ϖ = 100 mas) the absolute magnitude equals the apparent magnitude.
    assert float(gaia.absolute_magnitude(7.3, 100.0)) == pytest.approx(7.3, abs=1e-12)
    parallax = np.array([2.0, 20.0, 250.0])
    # Consistency with the distance modulus definition: m − M = 5 log10(d / 10 pc), d = 1000/ϖ pc.
    assert 10.0 - gaia.absolute_magnitude(10.0, parallax) == pytest.approx(
        distance_modulus_from_pc(1000.0 / parallax), abs=1e-12
    )


def test_cache_key_is_stable():
    assert cache.key("a", "b") == cache.key("a", "b") != cache.key("ab", "")
    assert len(cache.key("x")) == 16


@pytest.mark.network
def test_gaia_query(tmp_path):
    s = gaia.nearby_stars(tmp_path, parallax_min_mas=200.0)
    assert 20 < len(s["ra"]) < 200 and np.all(s["parallax"] > 200.0)
    assert (tmp_path / "SOURCE_RECORD.json").exists()


@pytest.mark.network
def test_skyview_cutout(tmp_path):
    image, header = skyview.download_cutout(85.42, -1.90, "2MASS-K", tmp_path, size_deg=0.1, pixels=64)
    assert image.shape == (64, 64) and np.isfinite(image).all() and header["NAXIS"] == 2


def test_heasarc_pipe_text_is_parsed(tmp_path):
    from kozmos.archives import heasarc

    path = tmp_path / "t.txt"
    path.write_text(
        "name   |ra     |dec    |vmag\nHR 2491|101.287|-16.716|-1.46\nHR 7001|279.234| 38.784| 0.03\nHR 1   |1.0    |2.0    |\n",
        encoding="utf-8",
    )
    table = heasarc.read_pipe_text(path)
    assert list(table) == ["name", "ra", "dec", "vmag"]
    assert list(table["name"]) == ["HR 2491", "HR 7001", "HR 1"]
    assert table["vmag"][0] == -1.46 and np.isnan(table["vmag"][2]) and table["dec"][1] == 38.784
    path.write_text("error", encoding="utf-8")
    with pytest.raises(ValueError):
        heasarc.read_pipe_text(path)


@pytest.mark.network
def test_bright_star_catalog_sirius(tmp_path):
    from kozmos.archives import heasarc

    table = heasarc.bright_stars(tmp_path)
    v = table["vmag"]
    # The brightest star in the catalog is Sirius (HR 2491); V = -1.46 [HoffleitWarren1991].
    i = int(np.nanargmin(v))
    assert v[i] == pytest.approx(-1.46, abs=0.01) and int(table["hr"][i]) == 2491
    assert 9000 < len(v) < 9200


@pytest.mark.network
def test_2mass_cone_query(tmp_path):
    from kozmos.archives import twomass
    from kozmos.core.angles import angular_separation

    k = twomass.point_sources(85.42, -1.90, 0.05, tmp_path)
    assert 50 < len(k["ra"]) < 400 and k["ph_qual"].dtype.kind == "U"
    # All sources are inside the cone.
    separation = angular_separation(np.radians(85.42), np.radians(-1.90), np.radians(k["ra"]), np.radians(k["dec"]))
    assert np.degrees(separation).max() <= 0.05 + 1e-9
    assert 4.0 < np.nanmin(k["k_m"]) < np.nanmax(k["k_m"]) < 17.5


@pytest.mark.network
def test_gaia_cone_query_and_epoch_shift(tmp_path):
    from kozmos.core.angles import angular_separation

    raw = gaia.cone_search(194.2425, 22.3912, 0.05, tmp_path)
    earlier = gaia.cone_search(194.2425, 22.3912, 0.05, tmp_path, epoch_yr=2012.2)
    assert len(raw["ra"]) > 20 and np.all(raw["ref_epoch"] == 2016.0)
    separation = angular_separation(
        np.radians(194.2425), np.radians(22.3912), np.radians(raw["ra"]), np.radians(raw["dec"])
    )
    assert np.degrees(separation).max() <= 0.05 + 1e-9
    # 3.8-year shift: displacement = total proper motion × 3.8 yr (mas → arcsec).
    has_pm = np.isfinite(raw["pmra"])
    shift = (
        np.degrees(
            angular_separation(
                *np.radians([raw["ra"][has_pm], raw["dec"][has_pm], earlier["ra"][has_pm], earlier["dec"][has_pm]])
            )
        )
        * 3600.0
    )
    expected = np.hypot(raw["pmra"][has_pm], raw["pmdec"][has_pm]) * 3.8 / 1000.0
    assert np.allclose(shift, expected, rtol=1e-4, atol=1e-6)
    assert np.array_equal(raw["ra"][~has_pm], earlier["ra"][~has_pm])


def test_download_retries_on_transient_error(tmp_path, monkeypatch):
    import io
    import urllib.error

    calls = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) < 3:
            raise urllib.error.URLError("transient error") if len(calls) == 1 else TimeoutError("timeout")
        return Response(b"data")

    monkeypatch.setattr(cache.urllib.request, "urlopen", fake_urlopen)
    path = cache.download("https://example.test/file", tmp_path / "d.bin", wait_s=0.0)
    assert path.read_bytes() == b"data" and len(calls) == 3
    # Permanent error: when all attempts are used up the error is raised with the attempt count; no target file.
    calls.clear()
    monkeypatch.setattr(
        cache.urllib.request, "urlopen", lambda request, timeout: (_ for _ in ()).throw(TimeoutError("x"))
    )
    with pytest.raises(OSError, match="3 attempts"):
        cache.download("https://example.test/missing", tmp_path / "y.bin", wait_s=0.0)
    assert not (tmp_path / "y.bin").exists()
    # A 4xx response is not retried.
    counter = []

    def not_found(request, timeout):
        counter.append(1)
        raise urllib.error.HTTPError(request.full_url, 404, "not found", None, None)

    monkeypatch.setattr(cache.urllib.request, "urlopen", not_found)
    with pytest.raises(urllib.error.HTTPError):
        cache.download("https://example.test/404", tmp_path / "z.bin", wait_s=0.0)
    assert len(counter) == 1


def test_skyview_retries_invalid_response(tmp_path, monkeypatch):
    from astropy.io import fits as astropy_fits

    valid = tmp_path / "valid.fits"
    astropy_fits.PrimaryHDU(np.arange(16, dtype=np.float32).reshape(4, 4)).writeto(valid)
    calls = []

    def fake_download(url, target, **options):
        calls.append(options["refresh"])
        target.write_bytes(b"<html>Service temporarily unavailable</html>" if len(calls) < 3 else valid.read_bytes())
        return target

    monkeypatch.setattr(skyview.cache, "download", fake_download)
    cache_dir = tmp_path / "o"
    cache_dir.mkdir()
    image, _ = skyview.download_cutout(10.0, 20.0, "2MASS-K", cache_dir, pixels=4)
    assert image.shape == (4, 4) and calls == [False, True, True]
    assert len(list(cache_dir.glob("*.invalid*"))) == 2  # invalid responses are kept as evidence
    # Always an invalid response: the error carries the start of the response and no file that would count as valid stays in the cache.
    calls.clear()
    monkeypatch.setattr(
        skyview.cache, "download", lambda url, target, **s: (target.write_bytes(b"<html>error</html>"), target)[1]
    )
    with pytest.raises(ValueError, match="did not return valid FITS"):
        skyview.download_cutout(11.0, 20.0, "2MASS-K", cache_dir, pixels=4)
    assert not list(cache_dir.glob("skyview_*.fits"))[1:]  # only the valid file of the first call remains
