"""Pantheon+SH0ES veri erişimi.

Biçim testleri küçük sahte dosyalarla ağsız koşar. Gerçek veri testi yerel
önbellek varsa koşar; yoksa atlanır (indirmek için ``-m ag``).
"""

from pathlib import Path

import numpy as np
import pytest

from kozmos.archives import pantheon

ONBELLEK = Path(__file__).resolve().parents[2] / "data" / "raw" / "pantheon"


def test_kovaryans_bicimi(tmp_path):
    path = tmp_path / "ornek.cov"
    path.write_text("2\n1.0\n0.5\n0.5\n2.0\n", encoding="utf-8")
    assert np.array_equal(pantheon.read_covariance(path), [[1.0, 0.5], [0.5, 2.0]])
    path.write_text("2\n1.0\n0.5\n0.5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        pantheon.read_covariance(path)


def test_tablo_bicimi(tmp_path):
    path = tmp_path / "ornek.dat"
    path.write_text("CID zHD IS_CALIBRATOR\n2011fe 0.00122 1\n1999aa 0.01522 0\n", encoding="utf-8")
    columns = pantheon.read_table(path)
    assert list(columns["CID"]) == ["2011fe", "1999aa"]
    assert columns["zHD"].dtype == float and columns["IS_CALIBRATOR"].sum() == 1


@pytest.mark.network
def test_indir(tmp_path):
    yollar = pantheon.download(tmp_path)
    assert yollar["tablo"].stat().st_size == 579283
    assert (tmp_path / "KAYNAK_KAYDI.json").exists()


@pytest.mark.skipif(not (ONBELLEK / "Pantheon+SH0ES.dat").exists(), reason="yerel önbellekte veri yok")
def test_gercek_veri_ozellikleri():
    data = pantheon.load(ONBELLEK)
    # Veri deposu README: N = 1701; makale: 1701 ışık eğrisi [Brout2022].
    assert len(data) == 1701 and len(data.columns) == 47
    assert len(set(data.columns["CID"])) == 1543
    assert int(data.columns["IS_CALIBRATOR"].sum()) == 77
    simetrik = 0.5 * (data.covariance + data.covariance.T)
    assert np.abs(data.covariance - simetrik).max() < 1e-6
    np.linalg.cholesky(simetrik)  # pozitif tanımlı değilse hata verir
