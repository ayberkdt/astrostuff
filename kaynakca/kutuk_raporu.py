"""Uygulama durumunu kaynak kütüğünden üretir: ``plan/DURUM.md``.

Rapor elle yazılmaz; ``kozmos`` paketindeki ``@kaynak`` kayıtlarından ve
``Sabit`` tanımlarından türetilir. Böylece belgedeki "uygulandı" bilgisi
koddan ayrışamaz.

Kullanım:
    python kaynakca/kutuk_raporu.py
"""

from __future__ import annotations

import importlib
import pkgutil
from collections import Counter
from pathlib import Path

import kozmos
from kozmos._kaynak import altyapi_kutugu, kutuk, ters_dizin
from kozmos.temel.sabitler import tum_sabitler

KOK = Path(__file__).resolve().parents[1]


def main() -> None:
    for bilgi in pkgutil.walk_packages(kozmos.__path__, "kozmos."):
        importlib.import_module(bilgi.name)
    kayitlar = kutuk()
    sabitler = tum_sabitler()
    teyit = Counter(k.teyit for ks in kayitlar.values() for k in ks)
    tam_okundu = sum(all(k.teyit == "okundu" for k in ks) for ks in kayitlar.values())
    test_sayisi = sum(
        1
        for p in (KOK / "tests").rglob("test_*.py")
        for s in p.read_text(encoding="utf-8").splitlines()
        if s.startswith("def test_")
    )

    s = ["# Uygulama durumu", ""]
    s.append("Bu dosya `python kaynakca/kutuk_raporu.py` ile kaynak kütüğünden üretilir; elle düzenlenmez.")
    s.append("")
    s.append(f"- Kaynağa bağlı açık fonksiyon: {len(kayitlar)}")
    s.append(f"- Bütün kayıtları kaynak metninden okunmuş fonksiyon: {tam_okundu}")
    s.append(f"- Kaynak kaydı: {sum(teyit.values())} (okundu: {teyit['okundu']}, bekliyor: {teyit['bekliyor']})")
    s.append(
        f"- Kaynaklı sabit: {len(sabitler)} (teyit bekleyen: {sum(x.teyit != 'okundu' for x in sabitler.values())})"
    )
    s.append(f"- Test fonksiyonu: {test_sayisi} (parametreli ve özellik testleri tek sayılır)")
    s.append(f"- Kaynak kuralından muaf altyapı fonksiyonu: {len(altyapi_kutugu())}")
    s.append("")
    s.append("`okundu`: denklem kod yazılırken kaynağın metninden okundu. `bekliyor`: denklem hatırlanan biçimiyle")
    s.append("yazıldı, testlerle sınandı, ama kaynağın metni henüz açılmadı.")
    s.append("")

    s.append("## Fonksiyonlar")
    moduller: dict[str, list[str]] = {}
    for ad in sorted(kayitlar):
        moduller.setdefault(ad.rsplit(".", 1)[0], []).append(ad)
    for modul, adlar in moduller.items():
        s += ["", f"### `{modul}`", ""]
        for ad in adlar:
            parcalar = "; ".join(f"[{k.anahtar}] {k.yer} ({k.teyit})" for k in kayitlar[ad])
            s.append(f"- `{ad.rsplit('.', 1)[1]}`: {parcalar}")

    s += ["", "## Sabitler", ""]
    for ad, x in sabitler.items():
        belirsizlik = "kesin" if x.belirsizlik == 0 else f"± {x.belirsizlik:.2g}"
        ek = f"; türetim: {x.turetim}" if x.turetim else ""
        s.append(f"- `{ad}` = {x.deger:.12g} {x.birim} ({belirsizlik}) [{x.kaynak}] {x.yer} ({x.teyit}{ek})")

    s += ["", "## Kaynak başına fonksiyon sayısı", ""]
    for anahtar, adlar in ters_dizin().items():
        s.append(f"- [{anahtar}]: {len(adlar)}")

    bekleyen = sorted({ad for ad, ks in kayitlar.items() if any(k.teyit == "bekliyor" for k in ks)})
    s += ["", "## Kaynağı okunmayı bekleyenler", ""]
    for ad in bekleyen:
        s.append(f"- `{ad}`: " + "; ".join(f"[{k.anahtar}] {k.yer}" for k in kayitlar[ad] if k.teyit == "bekliyor"))
    s.append("")
    (KOK / "plan" / "DURUM.md").write_text("\n".join(s), encoding="utf-8")
    print(
        f"{len(kayitlar)} fonksiyon, {sum(teyit.values())} kayıt (okundu {teyit['okundu']}, bekliyor {teyit['bekliyor']}), {len(sabitler)} sabit, {test_sayisi} test"
    )


if __name__ == "__main__":
    main()
