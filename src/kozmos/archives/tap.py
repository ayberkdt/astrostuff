"""Sanal Gözlemevi Tablo Erişim Protokolü (TAP) için eşzamanlı istemci.

Uç nokta ve parametre adları IVOA TAP 1.1 belirtiminden okunmuştur [TAP2019]:
taban adresin altındaki ``/sync`` kaynağı, ``LANG`` ve ``QUERY`` parametreleri
(Bölüm 2.1, 2.7). Sonuç CSV olarak istenir ve sütun sözlüğüne çevrilir.
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
    where="Bölüm 2.1 (TAP-sync), 2.7.1 (LANG), 2.7.2 (QUERY), 2.7.3 (FORMAT)",
    status="read",
    note="LANG=ADQL, QUERY=<ADQL>, FORMAT=csv; 1.1 sürümünde REQUEST parametresi kaldırılmıştır",
)
def query(endpoint: str, adql: str, cache_dir: Path, *, attribution: str = "", refresh: bool = False) -> Path:
    """Bir ADQL sorgusunu eşzamanlı çalıştırır ve sonucu CSV dosyası olarak önbelleğe yazar.

    Parameters
    ----------
    uc_nokta : str
        Servisin ``/sync`` adresi.
    adql : str
        Sorgu metni. Arşivin veri sürümü tablo adında açıkça yazılmalıdır.
    onbellek_dizini : Path
        Sonuç dosyası ve kaynak kaydının yazılacağı dizin.
    atif : str
        Verinin atıf metni; kaynak kaydına işlenir.

    Returns
    -------
    Path
        CSV dosyasının yolu. Dosya adı uç nokta ve sorgunun özetinden türetilir;
        aynı sorgu ikinci kez ağa çıkmaz.

    References
    ----------
    [TAP2019] Bölüm 2.1, 2.7.

    Validity
    --------
    Eşzamanlı sorgular servisin süre ve satır sınırına tabidir; büyük sorgular
    için eşzamansız (``/async``) arayüz gerekir ve burada uygulanmamıştır.

    Verification
    ------------
    tests/unit/test_archives.py (ağ gerektirir, ``-m ag``); CSV çözümü ağsız sınanır.
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
    where="Bölüm 2.7.3 (FORMAT): text/csv çıktısı",
    status="read",
    note="ilk satır sütun adları; boş alan eksik değerdir ve nan okunur",
)
def read_csv(path: Path) -> dict[str, np.ndarray]:
    """TAP'tan gelen CSV dosyasını sütun sözlüğüne okur.

    Bütün değerleri sayıya çevrilebilen sütunlar ``float64`` (boş alan ``nan``),
    diğerleri dizgi dizisi olur. 2⁵³'ten büyük tamsayı kimlikler (Gaia
    ``source_id``) duyarlık kaybetmemek için dizgi olarak bırakılır.

    References
    ----------
    [TAP2019] Bölüm 2.7.3.

    Verification
    ------------
    tests/unit/test_archives.py
    """
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("<"):
        raise ValueError("servis CSV yerine XML/HTML döndürdü; sorgu hatalı ya da uç nokta değişmiş olabilir")
    satirlar = list(csv.reader(io.StringIO(text)))
    basliklar, govde = satirlar[0], [s for s in satirlar[1:] if s]
    columns: dict[str, np.ndarray] = {}
    for i, name in enumerate(basliklar):
        raw = [s[i] for s in govde]
        try:
            if any(len(h.lstrip("-")) > 15 and h.lstrip("-").isdigit() for h in raw):
                raise ValueError
            columns[name] = np.array([float(h) if h != "" else np.nan for h in raw])
        except ValueError:
            columns[name] = np.array(raw)
    return columns
