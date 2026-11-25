"""Pantheon+SH0ES veri erişimi.

Biçim testleri küçük sahte dosyalarla ağsız koşar. Gerçek veri testi yerel
önbellek varsa koşar; yoksa atlanır (indirmek için ``-m ag``).
"""

from pathlib import Path

import numpy as np
import pytest

from kozmos.arsiv import pantheon

ONBELLEK = Path(__file__).resolve().parents[2] / "veri" / "ham" / "pantheon"


def test_kovaryans_bicimi(tmp_path):
    yol = tmp_path / "ornek.cov"
    yol.write_text("2\n1.0\n0.5\n0.5\n2.0\n", encoding="utf-8")
    assert np.array_equal(pantheon.kovaryans_oku(yol), [[1.0, 0.5], [0.5, 2.0]])
    yol.write_text("2\n1.0\n0.5\n0.5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        pantheon.kovaryans_oku(yol)


def test_tablo_bicimi(tmp_path):
    yol = tmp_path / "ornek.dat"
    yol.write_text("CID zHD IS_CALIBRATOR\n2011fe 0.00122 1\n1999aa 0.01522 0\n", encoding="utf-8")
    sutunlar = pantheon.tablo_oku(yol)
    assert list(sutunlar["CID"]) == ["2011fe", "1999aa"]
    assert sutunlar["zHD"].dtype == float and sutunlar["IS_CALIBRATOR"].sum() == 1


@pytest.mark.ag
def test_indir(tmp_path):
    yollar = pantheon.indir(tmp_path)
    assert yollar["tablo"].stat().st_size == 579283
    assert (tmp_path / "KAYNAK_KAYDI.json").exists()


@pytest.mark.skipif(not (ONBELLEK / "Pantheon+SH0ES.dat").exists(), reason="yerel önbellekte veri yok")
def test_gercek_veri_ozellikleri():
    veri = pantheon.yukle(ONBELLEK)
    # Veri deposu README: N = 1701; makale: 1701 ışık eğrisi [Brout2022].
    assert len(veri) == 1701 and len(veri.sutunlar) == 47
    assert len(set(veri.sutunlar["CID"])) == 1543
    assert int(veri.sutunlar["IS_CALIBRATOR"].sum()) == 77
    simetrik = 0.5 * (veri.kovaryans + veri.kovaryans.T)
    assert np.abs(veri.kovaryans - simetrik).max() < 1e-6
    np.linalg.cholesky(simetrik)  # pozitif tanımlı değilse hata verir
