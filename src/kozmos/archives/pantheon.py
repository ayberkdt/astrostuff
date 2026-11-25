"""Access to the Pantheon+SH0ES data release.

The data are not redistributed; they are downloaded from the official GitHub
repository and kept in a local cache together with their SHA-256 digest. The
file format and column definitions were read from the repository ``README``
file on 25 November 2026 [Scolnic2022; Brout2022].
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kozmos._sources import source

Array = NDArray[np.float64]

BASE_URL = (
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
)
FILES = {
    "table": "Pantheon+SH0ES.dat",
    "covariance": "Pantheon+SH0ES_STAT+SYS.cov",
    "covariance_stat_only": "Pantheon+SH0ES_STATONLY.cov",
}
ATTRIBUTION = "Scolnic et al. 2022 [Scolnic2022]; Brout et al. 2022 [Brout2022]; Riess et al. 2022 [Riess2022]"


@dataclass(frozen=True, slots=True)
class PantheonData:
    """Pantheon+SH0ES table and covariance.

    Attributes
    ----------
    columns : dict[str, ndarray]
        Column name → values (1701 rows). Names are as in the data file.
    covariance : ndarray, shape (1701, 1701)
        Statistical + systematic covariance [mag²].
    """

    columns: dict[str, np.ndarray]
    covariance: Array

    def __len__(self) -> int:
        return int(self.covariance.shape[0])


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@source(
    "Scolnic2022",
    where="data repository, 4_DISTANCES_AND_COVAR",
    status="read",
    note="download address and file names were read from the GitHub API",
)
def download(cache: Path, *, refresh: bool = False) -> dict[str, Path]:
    """Download the data files to the cache and write the provenance record.

    Parameters
    ----------
    cache : Path
        Directory the files are written to (kept out of version control).
    refresh : bool
        If true, existing files are downloaded again.

    Returns
    -------
    dict[str, Path]
        ``"table"``, ``"covariance"``, ``"covariance_stat_only"`` → local path.

    References
    ----------
    [Scolnic2022] data repository.

    Validity
    --------
    The repository has no license file; the data are used locally only and are not redistributed.

    Verification
    ------------
    tests/unit/test_pantheon.py (needs the network, ``-m network``).
    """
    cache.mkdir(parents=True, exist_ok=True)
    record_path = cache / "SOURCE_RECORD.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    paths: dict[str, Path] = {}
    for name, file_name in FILES.items():
        target = cache / file_name
        if refresh or not target.exists():
            url = BASE_URL + file_name.replace("+", "%2B")
            partial = target.with_suffix(target.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as f:  # noqa: S310
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    f.write(chunk)
            partial.replace(target)
            record[file_name] = {
                "url": url,
                "sha256": _digest(target),
                "bytes": target.stat().st_size,
                "downloaded_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "attribution": ATTRIBUTION,
            }
            record_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        paths[name] = target
    return paths


@source(
    "Brout2022",
    where="data repository README: .cov format",
    status="read",
    note="first line is N, then one element per line, in order, N×N",
)
def read_covariance(path: Path) -> Array:
    """Read a ``.cov`` file into an N×N matrix.

    References
    ----------
    [Brout2022] data repository README.

    Verification
    ------------
    tests/unit/test_pantheon.py: symmetry and positive definiteness.
    """
    flat = np.loadtxt(path)
    n = int(flat[0])
    if flat.size != n * n + 1:
        raise ValueError(f"expected {n * n + 1} numbers, read {flat.size}")
    return flat[1:].reshape(n, n)


@source("Brout2022", where="data repository README: column definitions", status="read")
def read_table(path: Path) -> dict[str, np.ndarray]:
    """Read the ``Pantheon+SH0ES.dat`` table into a column dictionary.

    ``CID`` is read as a string, the other columns as ``float64``.

    References
    ----------
    [Brout2022] data repository README.

    Verification
    ------------
    tests/unit/test_pantheon.py: 1701 rows, 47 columns.
    """
    with path.open(encoding="utf-8") as f:
        headers = f.readline().split()
        rows = [s.split() for s in f if s.strip()]
    columns: dict[str, np.ndarray] = {}
    for i, name in enumerate(headers):
        values = [row[i] for row in rows]
        columns[name] = np.array(values) if name == "CID" else np.array(values, dtype=float)
    return columns


@source("Brout2022", where="data repository README", status="read")
def load(cache: Path, *, stat_only: bool = False) -> PantheonData:
    """Read the data, downloading them if needed.

    Parameters
    ----------
    cache : Path
    stat_only : bool
        If true, only the statistical covariance is read.

    References
    ----------
    [Brout2022] data repository README.

    Verification
    ------------
    tests/unit/test_pantheon.py
    """
    paths = download(cache)
    npy = cache / ("covariance_stat_only.npy" if stat_only else "covariance.npy")
    if npy.exists():
        cov = np.load(npy)
    else:
        cov = read_covariance(paths["covariance_stat_only" if stat_only else "covariance"])
        np.save(npy, cov)
    return PantheonData(read_table(paths["table"]), cov)
