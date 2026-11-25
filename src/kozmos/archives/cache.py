"""Local cache of downloaded data and its provenance record.

Each download is recorded in a ``SOURCE_RECORD.json`` file next to the data file,
with its URL, SHA-256 digest, size, time and attribution text. The same request
never goes to the network a second time. The principles are in
``plan/05_data_science/05_data_management_and_reproducibility.md`` [Wilson2017].
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from kozmos._sources import infrastructure

USER_AGENT = "kozmos/0.2 (python-urllib)"


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@infrastructure("network and file infrastructure; applies no equation")
def download(
    url: str,
    target: Path,
    *,
    parameters: dict[str, str] | None = None,
    attribution: str = "",
    refresh: bool = False,
    timeout_s: float = 300.0,
    attempts: int = 3,
    wait_s: float = 5.0,
) -> Path:
    """Download a URL (optionally with query parameters) to a file and write its record.

    If the file exists and ``refresh`` is not set, the network is not used. The
    download goes to a temporary file first; an interrupted download does not
    damage the target. Transient network errors (dropped connection, timeout,
    5xx response) are retried ``attempts`` times, waiting ``wait_s`` seconds
    between tries; 4xx responses are not retried. If the last attempt also
    fails, the error is raised together with the number of attempts made.
    """
    target = Path(target)
    if target.exists() and not refresh:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    full_url = url + ("?" + urllib.parse.urlencode(parameters) if parameters else "")
    request = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT})
    partial = target.with_name(target.name + ".partial")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response, partial.open("wb") as f:  # noqa: S310
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    f.write(chunk)
            break
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == attempts:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == attempts:
                raise OSError(f"download failed after {attempts} attempts: {full_url} ({error})") from error
        time.sleep(wait_s)
    partial.replace(target)
    record_path = target.parent / "SOURCE_RECORD.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    record[target.name] = {
        "url": full_url,
        "sha256": _digest(target),
        "bytes": target.stat().st_size,
        "downloaded_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "attribution": attribution,
    }
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return target


@infrastructure("builds a stable file name for a request; applies no equation")
def key(*parts: str) -> str:
    """Build a stable 16-character digest from request parts (for the cache file name)."""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
