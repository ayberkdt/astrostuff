"""Bright Star Catalog (BSC5P): stars visible to the naked eye.

The catalog is taken from the ``bsc5p`` table of the HEASARC TAP service. The
table is derived from the 5th revised (preliminary) edition of the Bright Star
Catalog by Hoffleit and Warren and contains stars brighter than about 6.5
magnitude; the description was read from the HEASARC table page [HoffleitWarren1991].

Because Gaia saturates on very bright stars, this catalog complements the
Gaia sample: stars such as Sirius, Canopus and Arcturus are here.

The HEASARC TAP service does not give CSV (on 25 November 2026 ``FORMAT=csv``
returned an "Unsupported format" error); with ``FORMAT=text`` the result is
pipe-separated text.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kozmos._sources import source
from kozmos.archives import cache

TAP = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync"
ATTRIBUTION = (
    "Bright Star Catalog, 5th Revised Edition (Preliminary), Hoffleit and Warren 1991; "
    "HEASARC bsc5p table (NASA Goddard Space Flight Center)"
)
COLUMNS = "name, alt_name, hr, hd, ra, dec, vmag, bv_color, spect_type, parallax, pmra, pmdec, radvel"


@source(
    "TAP2019",
    where="Section 2.7.3 (FORMAT)",
    status="read",
    note="HEASARC 'text' format: first line is the column names, fields are separated by '|' and padded with "
    "spaces (the format was observed in the service response; not read from its documentation)",
)
def read_pipe_text(path: Path) -> dict[str, np.ndarray]:
    """Read a pipe-separated text table into a column dictionary.

    Columns whose values can all be converted to numbers become ``float64``
    (empty field is ``nan``); the others become string arrays.

    References
    ----------
    [TAP2019] Section 2.7.3.

    Verification
    ------------
    tests/unit/test_archives.py
    """
    lines = [s for s in Path(path).read_text(encoding="utf-8").splitlines() if "|" in s]
    if not lines:
        raise ValueError("no table line found; the service may have returned an error")
    headers = [b.strip() for b in lines[0].split("|")]
    body = [[a.strip() for a in s.split("|")] for s in lines[1:]]
    body = [row for row in body if len(row) == len(headers)]
    columns: dict[str, np.ndarray] = {}
    for i, name in enumerate(headers):
        raw = [row[i] for row in body]
        try:
            columns[name] = np.array([float(value) if value not in ("", "null") else np.nan for value in raw])
        except ValueError:
            columns[name] = np.array(raw)
    return columns


@source(
    "HoffleitWarren1991",
    where="HEASARC bsc5p table page: Overview, References, Provenance",
    status="read",
    note="the catalog itself (book/file documentation) was not read; the description and column explanations are "
    "from the HEASARC page and the VOTable field definitions of the service",
)
def bright_stars(cache_dir: Path, *, refresh: bool = False) -> dict[str, np.ndarray]:
    """Download the Bright Star Catalog (about 9100 records).

    Returned columns: ``name`` (HR number), ``alt_name`` (Bayer or
    Flamsteed name), ``hr``, ``hd``, ``ra`` and ``dec`` (J2000, degrees), ``vmag``,
    ``bv_color``, ``spect_type``, ``parallax`` (arcseconds), ``pmra``,
    ``pmdec``, ``radvel``.

    References
    ----------
    [HoffleitWarren1991] HEASARC bsc5p.

    Validity
    --------
    The catalog dates from 1991: parallaxes are ground-based measurements and are
    missing or uncertain for most stars; for work that needs distances, Hipparcos
    or Gaia should be used. 14 records that received an HR number are not stars
    (clusters, novae etc.; HEASARC page).

    Verification
    ------------
    tests/unit/test_archives.py (needs the network, ``-m network``): position and magnitude of Sirius.
    """
    adql = f"SELECT {COLUMNS} FROM bsc5p ORDER BY vmag"
    target = Path(cache_dir) / f"bsc5p_{cache.key(TAP, adql)}.txt"
    path = cache.download(
        TAP,
        target,
        parameters={"LANG": "ADQL", "FORMAT": "text", "MAXREC": "20000", "QUERY": adql},
        attribution=ATTRIBUTION,
        refresh=refresh,
    )
    return read_pipe_text(path)
