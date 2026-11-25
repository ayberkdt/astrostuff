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

from kozmos._kaynak import kaynak

Dizi = NDArray[np.float64]

TABAN_URL = (
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
)
DOSYALAR = {
    "tablo": "Pantheon+SH0ES.dat",
    "kovaryans": "Pantheon+SH0ES_STAT+SYS.cov",
    "kovaryans_istatistik": "Pantheon+SH0ES_STATONLY.cov",
}
ATIF = "Scolnic et al. 2022 [Scolnic2022]; Brout et al. 2022 [Brout2022]; Riess et al. 2022 [Riess2022]"


@dataclass(frozen=True, slots=True)
class PantheonVerisi:
    """Pantheon+SH0ES tablosu ve kovaryansı.

    Öznitelikler
    ------------
    sutunlar : dict[str, ndarray]
        Sütun adı → değerler (1701 satır). Adlar veri dosyasındaki gibidir.
    kovaryans : ndarray, biçim (1701, 1701)
        İstatistik + sistematik kovaryans [mag²].
    """

    sutunlar: dict[str, np.ndarray]
    kovaryans: Dizi

    def __len__(self) -> int:
        return int(self.kovaryans.shape[0])


def _ozet(yol: Path) -> str:
    h = hashlib.sha256()
    with yol.open("rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


@kaynak(
    "Scolnic2022",
    yer="veri deposu, 4_DISTANCES_AND_COVAR",
    teyit="okundu",
    not_="indirme adresi ve dosya adları GitHub API'sinden okundu",
)
def indir(onbellek: Path, *, yenile: bool = False) -> dict[str, Path]:
    """Veri dosyalarını önbelleğe indirir ve kaynak kaydını yazar.

    Parametreler
    ------------
    onbellek : Path
        Dosyaların yazılacağı dizin (sürüm denetimi dışında tutulur).
    yenile : bool
        Doğruysa var olan dosyalar yeniden indirilir.

    Döndürür
    --------
    dict[str, Path]
        ``"tablo"``, ``"kovaryans"``, ``"kovaryans_istatistik"`` → yerel yol.

    Kaynaklar
    ---------
    [Scolnic2022] veri deposu.

    Geçerlilik
    ----------
    Depoda lisans dosyası yoktur; veri yalnızca yerelde kullanılır, yeniden dağıtılmaz.

    Doğrulama
    ---------
    tests/birim/test_pantheon.py (ağ gerektirir, ``-m ag``).
    """
    onbellek.mkdir(parents=True, exist_ok=True)
    kayit_yolu = onbellek / "KAYNAK_KAYDI.json"
    kayit = json.loads(kayit_yolu.read_text(encoding="utf-8")) if kayit_yolu.exists() else {}
    yollar: dict[str, Path] = {}
    for ad, dosya in DOSYALAR.items():
        hedef = onbellek / dosya
        if yenile or not hedef.exists():
            url = TABAN_URL + dosya.replace("+", "%2B")
            gecici = hedef.with_suffix(hedef.suffix + ".kismi")
            with urllib.request.urlopen(url, timeout=120) as yanit, gecici.open("wb") as f:  # noqa: S310
                for parca in iter(lambda: yanit.read(1 << 20), b""):
                    f.write(parca)
            gecici.replace(hedef)
            kayit[dosya] = {
                "url": url,
                "sha256": _ozet(hedef),
                "bayt": hedef.stat().st_size,
                "indirme_zamani": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "atif": ATIF,
            }
            kayit_yolu.write_text(json.dumps(kayit, ensure_ascii=False, indent=1), encoding="utf-8")
        yollar[ad] = hedef
    return yollar


@kaynak(
    "Brout2022",
    yer="veri deposu README: .cov biçimi",
    teyit="okundu",
    not_="ilk satır N, ardından satır başına bir eleman, sırayla N×N",
)
def kovaryans_oku(yol: Path) -> Dizi:
    """``.cov`` dosyasını N×N matrise okur.

    Kaynaklar
    ---------
    [Brout2022] veri deposu README.

    Doğrulama
    ---------
    tests/birim/test_pantheon.py: simetri ve pozitif tanımlılık.
    """
    duz = np.loadtxt(yol)
    n = int(duz[0])
    if duz.size != n * n + 1:
        raise ValueError(f"beklenen {n * n + 1} sayı, okunan {duz.size}")
    return duz[1:].reshape(n, n)


@kaynak("Brout2022", yer="veri deposu README: sütun tanımları", teyit="okundu")
def tablo_oku(yol: Path) -> dict[str, np.ndarray]:
    """``Pantheon+SH0ES.dat`` tablosunu sütun sözlüğüne okur.

    ``CID`` dizgi, diğer sütunlar ``float64`` okunur.

    Kaynaklar
    ---------
    [Brout2022] veri deposu README.

    Doğrulama
    ---------
    tests/birim/test_pantheon.py: 1701 satır, 47 sütun.
    """
    with yol.open(encoding="utf-8") as f:
        basliklar = f.readline().split()
        satirlar = [s.split() for s in f if s.strip()]
    sutunlar: dict[str, np.ndarray] = {}
    for i, ad in enumerate(basliklar):
        degerler = [s[i] for s in satirlar]
        sutunlar[ad] = np.array(degerler) if ad == "CID" else np.array(degerler, dtype=float)
    return sutunlar


@kaynak("Brout2022", yer="veri deposu README", teyit="okundu")
def yukle(onbellek: Path, *, yalniz_istatistik: bool = False) -> PantheonVerisi:
    """Veriyi (gerekirse indirerek) okur.

    Parametreler
    ------------
    onbellek : Path
    yalniz_istatistik : bool
        Doğruysa yalnızca istatistik kovaryans okunur.

    Kaynaklar
    ---------
    [Brout2022] veri deposu README.

    Doğrulama
    ---------
    tests/birim/test_pantheon.py
    """
    yollar = indir(onbellek)
    npy = onbellek / ("kovaryans_istatistik.npy" if yalniz_istatistik else "kovaryans.npy")
    if npy.exists():
        kov = np.load(npy)
    else:
        kov = kovaryans_oku(yollar["kovaryans_istatistik" if yalniz_istatistik else "kovaryans"])
        np.save(npy, kov)
    return PantheonVerisi(tablo_oku(yollar["tablo"]), kov)
