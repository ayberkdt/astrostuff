"""Pantheon+SH0ES data access.

Format tests run without the network, with small fake files. The real-data test
runs if a local cache exists; otherwise it is skipped (use ``-m network`` to download).
"""

from pathlib import Path

import numpy as np
import pytest

from kozmos.archives import pantheon

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "pantheon"


def test_covariance_format(tmp_path):
    path = tmp_path / "sample.cov"
    path.write_text("2\n1.0\n0.5\n0.5\n2.0\n", encoding="utf-8")
    assert np.array_equal(pantheon.read_covariance(path), [[1.0, 0.5], [0.5, 2.0]])
    path.write_text("2\n1.0\n0.5\n0.5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        pantheon.read_covariance(path)


def test_table_format(tmp_path):
    path = tmp_path / "sample.dat"
    path.write_text("CID zHD IS_CALIBRATOR\n2011fe 0.00122 1\n1999aa 0.01522 0\n", encoding="utf-8")
    columns = pantheon.read_table(path)
    assert list(columns["CID"]) == ["2011fe", "1999aa"]
    assert columns["zHD"].dtype == float and columns["IS_CALIBRATOR"].sum() == 1


@pytest.mark.network
def test_download(tmp_path):
    paths = pantheon.download(tmp_path)
    assert paths["table"].stat().st_size == 579283
    assert (tmp_path / "SOURCE_RECORD.json").exists()


@pytest.mark.skipif(not (CACHE_DIR / "Pantheon+SH0ES.dat").exists(), reason="no data in the local cache")
def test_real_data_properties():
    data = pantheon.load(CACHE_DIR)
    # Data repository README: N = 1701; paper: 1701 light curves [Brout2022].
    assert len(data) == 1701 and len(data.columns) == 47
    assert len(set(data.columns["CID"])) == 1543
    assert int(data.columns["IS_CALIBRATOR"].sum()) == 77
    symmetric = 0.5 * (data.covariance + data.covariance.T)
    assert np.abs(data.covariance - symmetric).max() < 1e-6
    np.linalg.cholesky(symmetric)  # raises an error if not positive definite
