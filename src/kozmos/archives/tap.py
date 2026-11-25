"""Synchronous client for the Virtual Observatory Table Access Protocol (TAP).

The endpoint and parameter names were read from the IVOA TAP 1.1 specification [TAP2019]:
the ``/sync`` resource under the base address, and the ``LANG`` and ``QUERY`` parameters
(Section 2.1, 2.7). The result is requested as CSV and converted to a column dictionary.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

from kozmos._sources import source
from kozmos.archives import cache


@source(
    "TAP2019",
    where="Section 2.1 (TAP-sync), 2.7.1 (LANG), 2.7.2 (QUERY), 2.7.3 (FORMAT)",
    status="read",
    note="LANG=ADQL, QUERY=<ADQL>, FORMAT=csv; the REQUEST parameter was removed in version 1.1",
)
def query(endpoint: str, adql: str, cache_dir: Path, *, attribution: str = "", refresh: bool = False) -> Path:
    """Run an ADQL query synchronously and write the result to the cache as a CSV file.

    Parameters
    ----------
    endpoint : str
        The ``/sync`` address of the service.
    adql : str
        Query text. The data release of the archive must be written explicitly in the table name.
    cache_dir : Path
        Directory the result file and the provenance record are written to.
    attribution : str
        Attribution text of the data; recorded in the provenance record.
    refresh : bool
        Download again even if the cached file exists.

    Returns
    -------
    Path
        Path of the CSV file. The file name is derived from a digest of the endpoint and the query;
        the same query does not go to the network a second time.

    References
    ----------
    [TAP2019] Section 2.1, 2.7.

    Validity
    --------
    Synchronous queries are subject to the time and row limits of the service; large queries
    need the asynchronous (``/async``) interface, which is not implemented here.

    Verification
    ------------
    tests/unit/test_archives.py (needs the network, ``-m network``); CSV parsing is tested without the network.
    """
    target = Path(cache_dir) / f"tap_{cache.key(endpoint, adql)}.csv"
    return cache.download(
        endpoint,
        target,
        parameters={"LANG": "ADQL", "FORMAT": "csv", "QUERY": adql},
        attribution=attribution,
        refresh=refresh,
    )


@source(
    "TAP2019",
    where="Section 2.7.3 (FORMAT): text/csv output",
    status="read",
    note="first line is the column names; an empty field is a missing value and is read as nan",
)
def read_csv(path: Path) -> dict[str, np.ndarray]:
    """Read a CSV file from TAP into a column dictionary.

    Columns whose values can all be converted to numbers become ``float64``
    (empty field is ``nan``); the others become string arrays. Integer
    identifiers larger than 2⁵³ (Gaia ``source_id``) are kept as strings so
    that no precision is lost.

    References
    ----------
    [TAP2019] Section 2.7.3.

    Verification
    ------------
    tests/unit/test_archives.py
    """
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("<"):
        raise ValueError(
            "the service returned XML/HTML instead of CSV; the query may be wrong or the endpoint may have changed"
        )
    rows = list(csv.reader(io.StringIO(text)))
    headers, body = rows[0], [s for s in rows[1:] if s]
    columns: dict[str, np.ndarray] = {}
    for i, name in enumerate(headers):
        raw = [row[i] for row in body]
        try:
            if any(len(value.lstrip("-")) > 15 and value.lstrip("-").isdigit() for value in raw):
                raise ValueError
            columns[name] = np.array([float(value) if value != "" else np.nan for value in raw])
        except ValueError:
            columns[name] = np.array(raw)
    return columns
