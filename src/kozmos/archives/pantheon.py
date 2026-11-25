"""Pantheon+SH0ES veri sürümüne erişim.

Veri yeniden dağıtılmaz; GitHub'daki resmî depodan indirilir ve yerel
önbellekte SHA-256 özetiyle birlikte tutulur. Dosya biçimi ve sütun
tanımları deponun ``README`` dosyasından 25 Kasım 2026'da okunmuştur
[Scolnic2022; Brout2022].
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
    "tablo": "Pantheon+SH0ES.dat",
    "kovaryans": "Pantheon+SH0ES_STAT+SYS.cov",
    "kovaryans_istatistik": "Pantheon+SH0ES_STATONLY.cov",
}
ATTRIBUTION = "Scolnic et al. 2022 [Scolnic2022]; Brout et al. 2022 [Brout2022]; Riess et al. 2022 [Riess2022]"


@dataclass(frozen=True, slots=True)
class PantheonData:
    """Pantheon+SH0ES tablosu ve kovaryansı.

    Attributes
    ----------
    sutunlar : dict[str, ndarray]
        Sütun adı → değerler (1701 satır). Adlar veri dosyasındaki gibidir.
    kovaryans : ndarray, biçim (1701, 1701)
        İstatistik + sistematik kovaryans [mag²].
    """

    columns: dict[str, np.ndarray]
    covariance: Array

    def __len__(self) -> int:
        return int(self.covariance.shape[0])


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


@source(
    "Scolnic2022",
    where="veri deposu, 4_DISTANCES_AND_COVAR",
    status="read",
    note="indirme adresi ve dosya adları GitHub API'sinden okundu",
)
def download(cache: Path, *, refresh: bool = False) -> dict[str, Path]:
    """Veri dosyalarını önbelleğe indirir ve kaynak kaydını yazar.

    Parameters
    ----------
    onbellek : Path
        Dosyaların yazılacağı dizin (sürüm denetimi dışında tutulur).
    yenile : bool
        Doğruysa var olan dosyalar yeniden indirilir.

    Returns
    -------
    dict[str, Path]
        ``"tablo"``, ``"kovaryans"``, ``"kovaryans_istatistik"`` → yerel yol.

    References
    ----------
    [Scolnic2022] veri deposu.

    Validity
    --------
    Depoda lisans dosyası yoktur; veri yalnızca yerelde kullanılır, yeniden dağıtılmaz.

    Verification
    ------------
    tests/unit/test_pantheon.py (ağ gerektirir, ``-m ag``).
    """
    cache.mkdir(parents=True, exist_ok=True)
    kayit_yolu = cache / "KAYNAK_KAYDI.json"
    kayit = json.loads(kayit_yolu.read_text(encoding="utf-8")) if kayit_yolu.exists() else {}
    yollar: dict[str, Path] = {}
    for name, dosya in FILES.items():
        target = cache / dosya
        if refresh or not target.exists():
            url = BASE_URL + dosya.replace("+", "%2B")
            gecici = target.with_suffix(target.suffix + ".kismi")
            with urllib.request.urlopen(url, timeout=120) as yanit, gecici.open("wb") as f:  # noqa: S310
                for parca in iter(lambda: yanit.read(1 << 20), b""):
                    f.write(parca)
            gecici.replace(target)
            kayit[dosya] = {
                "url": url,
                "sha256": _digest(target),
                "bayt": target.stat().st_size,
                "indirme_zamani": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "atif": ATTRIBUTION,
            }
            kayit_yolu.write_text(json.dumps(kayit, ensure_ascii=False, indent=1), encoding="utf-8")
        yollar[name] = target
    return yollar


@source(
    "Brout2022",
    where="veri deposu README: .cov biçimi",
    status="read",
    note="ilk satır N, ardından satır başına bir eleman, sırayla N×N",
)
def read_covariance(path: Path) -> Array:
    """``.cov`` dosyasını N×N matrise okur.

    References
    ----------
    [Brout2022] veri deposu README.

    Verification
    ------------
    tests/unit/test_pantheon.py: simetri ve pozitif tanımlılık.
    """
    flat = np.loadtxt(path)
    n = int(flat[0])
    if flat.size != n * n + 1:
        raise ValueError(f"beklenen {n * n + 1} sayı, okunan {flat.size}")
    return flat[1:].reshape(n, n)


@source("Brout2022", where="veri deposu README: sütun tanımları", status="read")
def read_table(path: Path) -> dict[str, np.ndarray]:
    """``Pantheon+SH0ES.dat`` tablosunu sütun sözlüğüne okur.

    ``CID`` dizgi, diğer sütunlar ``float64`` okunur.

    References
    ----------
    [Brout2022] veri deposu README.

    Verification
    ------------
    tests/unit/test_pantheon.py: 1701 satır, 47 sütun.
    """
    with path.open(encoding="utf-8") as f:
        basliklar = f.readline().split()
        satirlar = [s.split() for s in f if s.strip()]
    columns: dict[str, np.ndarray] = {}
    for i, name in enumerate(basliklar):
        degerler = [s[i] for s in satirlar]
        columns[name] = np.array(degerler) if name == "CID" else np.array(degerler, dtype=float)
    return columns


@source("Brout2022", where="veri deposu README", status="read")
def load(cache: Path, *, stat_only: bool = False) -> PantheonData:
    """Veriyi (gerekirse indirerek) okur.

    Parameters
    ----------
    onbellek : Path
    yalniz_istatistik : bool
        Doğruysa yalnızca istatistik kovaryans okunur.

    References
    ----------
    [Brout2022] veri deposu README.

    Verification
    ------------
    tests/unit/test_pantheon.py
    """
    yollar = download(cache)
    npy = cache / ("kovaryans_istatistik.npy" if stat_only else "kovaryans.npy")
    if npy.exists():
        kov = np.load(npy)
    else:
        kov = read_covariance(yollar["kovaryans_istatistik" if stat_only else "kovaryans"])
        np.save(npy, kov)
    return PantheonData(read_table(yollar["tablo"]), kov)
