"""Arşiv erişimi: CSV çözümü ağsız; gerçek sorgular ``-m ag`` ile."""

import numpy as np
import pytest

from kozmos.arsiv import gaia, onbellek, skyview, tap
from kozmos.isinim.kadir import uzaklik_modulu_pc


def test_csv_coz(tmp_path):
    yol = tmp_path / "a.csv"
    yol.write_text(
        "source_id,ra,bp_rp,ad\n1926461164913660160,355.48,3.53,x\n762815470562110464,165.83,,y\n", encoding="utf-8"
    )
    s = tap.csv_oku(yol)
    assert s["source_id"].dtype.kind == "U" and s["source_id"][0] == "1926461164913660160"  # 2^53 üstü: dizgi
    assert s["ra"].dtype == float and np.isnan(s["bp_rp"][1]) and list(s["ad"]) == ["x", "y"]
    yol.write_text("<html>hata</html>", encoding="utf-8")
    with pytest.raises(ValueError):
        tap.csv_oku(yol)


def test_mutlak_kadir():
    # 10 pc (ϖ = 100 mas) uzaklıkta mutlak kadir görünen kadire eşittir.
    assert float(gaia.mutlak_kadir(7.3, 100.0)) == pytest.approx(7.3, abs=1e-12)
    paralaks = np.array([2.0, 20.0, 250.0])
    # Uzaklık modülü tanımıyla tutarlılık: m − M = 5 log10(d / 10 pc), d = 1000/ϖ pc.
    assert 10.0 - gaia.mutlak_kadir(10.0, paralaks) == pytest.approx(uzaklik_modulu_pc(1000.0 / paralaks), abs=1e-12)


def test_onbellek_anahtari_kararli():
    assert onbellek.anahtar("a", "b") == onbellek.anahtar("a", "b") != onbellek.anahtar("ab", "")
    assert len(onbellek.anahtar("x")) == 16


@pytest.mark.ag
def test_gaia_sorgusu(tmp_path):
    s = gaia.yakin_yildizlar(tmp_path, paralaks_alt_mas=200.0)
    assert 20 < len(s["ra"]) < 200 and np.all(s["parallax"] > 200.0)
    assert (tmp_path / "KAYNAK_KAYDI.json").exists()


@pytest.mark.ag
def test_skyview_kesiti(tmp_path):
    goruntu, baslik = skyview.kesit_indir(85.42, -1.90, "2MASS-K", tmp_path, alan_derece=0.1, piksel=64)
    assert goruntu.shape == (64, 64) and np.isfinite(goruntu).all() and baslik["NAXIS"] == 2


def test_heasarc_boru_metni_cozulur(tmp_path):
    from kozmos.arsiv import heasarc

    yol = tmp_path / "t.txt"
    yol.write_text(
        "name   |ra     |dec    |vmag\nHR 2491|101.287|-16.716|-1.46\nHR 7001|279.234| 38.784| 0.03\nHR 1   |1.0    |2.0    |\n",
        encoding="utf-8",
    )
    tablo = heasarc.boru_metni_oku(yol)
    assert list(tablo) == ["name", "ra", "dec", "vmag"]
    assert list(tablo["name"]) == ["HR 2491", "HR 7001", "HR 1"]
    assert tablo["vmag"][0] == -1.46 and np.isnan(tablo["vmag"][2]) and tablo["dec"][1] == 38.784
    yol.write_text("hata", encoding="utf-8")
    with pytest.raises(ValueError):
        heasarc.boru_metni_oku(yol)


@pytest.mark.ag
def test_parlak_yildiz_katalogu_sirius(tmp_path):
    from kozmos.arsiv import heasarc

    tablo = heasarc.parlak_yildizlar(tmp_path)
    v = tablo["vmag"]
    # Katalogdaki en parlak yıldız Sirius'tur (HR 2491); V = -1,46 [HoffleitWarren1991].
    i = int(np.nanargmin(v))
    assert v[i] == pytest.approx(-1.46, abs=0.01) and int(tablo["hr"][i]) == 2491
    assert 9000 < len(v) < 9200


@pytest.mark.ag
def test_2mass_koni_sorgusu(tmp_path):
    from kozmos.arsiv import ikimass
    from kozmos.temel.aci import acisal_ayriklik

    k = ikimass.nokta_kaynaklari(85.42, -1.90, 0.05, tmp_path)
    assert 50 < len(k["ra"]) < 400 and k["ph_qual"].dtype.kind == "U"
    # Bütün kaynaklar koninin içindedir.
    ayriklik = acisal_ayriklik(np.radians(85.42), np.radians(-1.90), np.radians(k["ra"]), np.radians(k["dec"]))
    assert np.degrees(ayriklik).max() <= 0.05 + 1e-9
    assert 4.0 < np.nanmin(k["k_m"]) < np.nanmax(k["k_m"]) < 17.5
