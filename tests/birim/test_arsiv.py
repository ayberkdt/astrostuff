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
