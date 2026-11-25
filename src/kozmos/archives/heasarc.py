"""Parlak Yıldız Kataloğu (BSC5P): çıplak gözle görülen yıldızlar.

Katalog HEASARC'ın TAP servisindeki ``bsc5p`` tablosundan alınır. Tablo,
Hoffleit ve Warren'ın Bright Star Catalog 5. gözden geçirilmiş (ön) sürümünden
türetilmiştir ve yaklaşık 6,5 kadirden parlak yıldızları içerir; tanım
HEASARC'ın tablo sayfasından okunmuştur [HoffleitWarren1991].

Gaia çok parlak yıldızlarda doyduğu için bu katalog Gaia örneklemini
tamamlar: Sirius, Canopus, Arcturus gibi yıldızlar buradadır.

HEASARC TAP servisi CSV vermez (25 Kasım 2026'da ``FORMAT=csv`` "Unsupported
format" hatası döndürdü); ``FORMAT=text`` ile dikey çizgiyle ayrılmış metin alınır.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kozmos._sources import source
from kozmos.archives import cache

TAP = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync"
ATTRIBUTION = (
    "Bright Star Catalog, 5th Revised Edition (Preliminary), Hoffleit ve Warren 1991; "
    "HEASARC bsc5p tablosu (NASA Goddard Space Flight Center)"
)
COLUMNS = "name, alt_name, hr, hd, ra, dec, vmag, bv_color, spect_type, parallax, pmra, pmdec, radvel"


@source(
    "TAP2019",
    where="Bölüm 2.7.3 (FORMAT)",
    status="read",
    note="HEASARC'ın 'text' biçimi: ilk satır sütun adları, alanlar '|' ile ayrılır, boşlukla hizalanır "
    "(biçim servisin yanıtından gözlendi; belgesinden okunmadı)",
)
def read_pipe_text(path: Path) -> dict[str, np.ndarray]:
    """Dikey çizgiyle ayrılmış metin tablosunu sütun sözlüğüne okur.

    Bütün değerleri sayıya çevrilebilen sütunlar ``float64`` (boş alan ``nan``),
    diğerleri dizgi dizisi olur.

    References
    ----------
    [TAP2019] Bölüm 2.7.3.

    Verification
    ------------
    tests/unit/test_archives.py
    """
    satirlar = [s for s in Path(path).read_text(encoding="utf-8").splitlines() if "|" in s]
    if not satirlar:
        raise ValueError("tablo satırı bulunamadı; servis hata döndürmüş olabilir")
    basliklar = [b.strip() for b in satirlar[0].split("|")]
    govde = [[a.strip() for a in s.split("|")] for s in satirlar[1:]]
    govde = [g for g in govde if len(g) == len(basliklar)]
    columns: dict[str, np.ndarray] = {}
    for i, name in enumerate(basliklar):
        raw = [g[i] for g in govde]
        try:
            columns[name] = np.array([float(h) if h not in ("", "null") else np.nan for h in raw])
        except ValueError:
            columns[name] = np.array(raw)
    return columns


@source(
    "HoffleitWarren1991",
    where="HEASARC bsc5p tablo sayfası: Overview, References, Provenance",
    status="read",
    note="katalogun kendisi (kitap/dosya belgesi) okunmadı; tanım ve sütun açıklamaları HEASARC sayfasından ve "
    "servisin VOTable alan tanımlarından",
)
def bright_stars(cache_dir: Path, *, refresh: bool = False) -> dict[str, np.ndarray]:
    """Parlak Yıldız Kataloğu'nu (yaklaşık 9100 kayıt) indirir.

    Döndürülen sütunlar: ``name`` (HR numarası), ``alt_name`` (Bayer ya da
    Flamsteed adı), ``hr``, ``hd``, ``ra`` ve ``dec`` (J2000, derece), ``vmag``,
    ``bv_color``, ``spect_type``, ``parallax`` (yay saniyesi), ``pmra``,
    ``pmdec``, ``radvel``.

    References
    ----------
    [HoffleitWarren1991] HEASARC bsc5p.

    Validity
    --------
    Katalog 1991 tarihlidir: paralakslar yer tabanlı ölçümlerdir ve çoğu yıldız
    için yoktur ya da belirsizdir; uzaklık gerektiren işlerde Hipparcos ya da
    Gaia kullanılmalıdır. HR numarası almış 14 kayıt yıldız değildir (küme, nova
    vb.; HEASARC sayfası).

    Verification
    ------------
    tests/unit/test_archives.py (ağ gerektirir, ``-m ag``): Sirius'un konumu ve kadiri.
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
