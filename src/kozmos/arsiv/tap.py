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

from kozmos._kaynak import kaynak
from kozmos.arsiv import onbellek


@kaynak(
    "TAP2019",
    yer="Bölüm 2.1 (TAP-sync), 2.7.1 (LANG), 2.7.2 (QUERY), 2.7.3 (FORMAT)",
    teyit="okundu",
    not_="LANG=ADQL, QUERY=<ADQL>, FORMAT=csv; 1.1 sürümünde REQUEST parametresi kaldırılmıştır",
)
def sorgula(uc_nokta: str, adql: str, onbellek_dizini: Path, *, atif: str = "", yenile: bool = False) -> Path:
    """Bir ADQL sorgusunu eşzamanlı çalıştırır ve sonucu CSV dosyası olarak önbelleğe yazar.

    Parametreler
    ------------
    uc_nokta : str
        Servisin ``/sync`` adresi.
    adql : str
        Sorgu metni. Arşivin veri sürümü tablo adında açıkça yazılmalıdır.
    onbellek_dizini : Path
        Sonuç dosyası ve kaynak kaydının yazılacağı dizin.
    atif : str
        Verinin atıf metni; kaynak kaydına işlenir.

    Döndürür
    --------
    Path
        CSV dosyasının yolu. Dosya adı uç nokta ve sorgunun özetinden türetilir;
        aynı sorgu ikinci kez ağa çıkmaz.

    Kaynaklar
    ---------
    [TAP2019] Bölüm 2.1, 2.7.

    Geçerlilik
    ----------
    Eşzamanlı sorgular servisin süre ve satır sınırına tabidir; büyük sorgular
    için eşzamansız (``/async``) arayüz gerekir ve burada uygulanmamıştır.

    Doğrulama
    ---------
    tests/birim/test_arsiv.py (ağ gerektirir, ``-m ag``); CSV çözümü ağsız sınanır.
    """
    hedef = Path(onbellek_dizini) / f"tap_{onbellek.anahtar(uc_nokta, adql)}.csv"
    return onbellek.indir(
        uc_nokta, hedef, parametreler={"LANG": "ADQL", "FORMAT": "csv", "QUERY": adql}, atif=atif, yenile=yenile
    )


@kaynak(
    "TAP2019",
    yer="Bölüm 2.7.3 (FORMAT): text/csv çıktısı",
    teyit="okundu",
    not_="ilk satır sütun adları; boş alan eksik değerdir ve nan okunur",
)
def csv_oku(yol: Path) -> dict[str, np.ndarray]:
    """TAP'tan gelen CSV dosyasını sütun sözlüğüne okur.

    Bütün değerleri sayıya çevrilebilen sütunlar ``float64`` (boş alan ``nan``),
    diğerleri dizgi dizisi olur. 2⁵³'ten büyük tamsayı kimlikler (Gaia
    ``source_id``) duyarlık kaybetmemek için dizgi olarak bırakılır.

    Kaynaklar
    ---------
    [TAP2019] Bölüm 2.7.3.

    Doğrulama
    ---------
    tests/birim/test_arsiv.py
    """
    metin = Path(yol).read_text(encoding="utf-8")
    if metin.lstrip().startswith("<"):
        raise ValueError("servis CSV yerine XML/HTML döndürdü; sorgu hatalı ya da uç nokta değişmiş olabilir")
    satirlar = list(csv.reader(io.StringIO(metin)))
    basliklar, govde = satirlar[0], [s for s in satirlar[1:] if s]
    sutunlar: dict[str, np.ndarray] = {}
    for i, ad in enumerate(basliklar):
        ham = [s[i] for s in govde]
        try:
            if any(len(h.lstrip("-")) > 15 and h.lstrip("-").isdigit() for h in ham):
                raise ValueError
            sutunlar[ad] = np.array([float(h) if h != "" else np.nan for h in ham])
        except ValueError:
            sutunlar[ad] = np.array(ham)
    return sutunlar
