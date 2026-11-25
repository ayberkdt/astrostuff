"""Uygulama durumunu kaynak kütüğünden üretir: ``plan/STATUS.md``.

Rapor elle yazılmaz; ``kozmos`` paketindeki ``@kaynak`` kayıtlarından ve
``Sabit`` tanımlarından türetilir. Böylece belgedeki "uygulandı" bilgisi
koddan ayrışamaz.

Kullanım:
    python bibliography/registry_report.py
"""

from __future__ import annotations

import importlib
import pkgutil
from collections import Counter
from pathlib import Path

import kozmos
from kozmos._sources import infrastructure_registry, registry, reverse_index
from kozmos.core.constants import all_constants

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for bilgi in pkgutil.walk_packages(kozmos.__path__, "kozmos."):
        importlib.import_module(bilgi.name)
    kayitlar = registry()
    sabitler = all_constants()
    status = Counter(k.status for ks in kayitlar.values() for k in ks)
    tam_okundu = sum(all(k.status == "read" for k in ks) for ks in kayitlar.values())
    test_sayisi = sum(
        1
        for p in (ROOT / "tests").rglob("test_*.py")
        for s in p.read_text(encoding="utf-8").splitlines()
        if s.startswith("def test_")
    )

    s = ["# Uygulama durumu", ""]
    s.append("Bu dosya `python bibliography/registry_report.py` ile kaynak kütüğünden üretilir; elle düzenlenmez.")
    s.append("")
    s.append(f"- Kaynağa bağlı açık fonksiyon: {len(kayitlar)}")
    s.append(f"- Bütün kayıtları kaynak metninden okunmuş fonksiyon: {tam_okundu}")
    s.append(f"- Kaynak kaydı: {sum(status.values())} (okundu: {status['read']}, bekliyor: {status['pending']})")
    s.append(
        f"- Kaynaklı sabit: {len(sabitler)} (teyit bekleyen: {sum(x.status != 'read' for x in sabitler.values())})"
    )
    s.append(f"- Test fonksiyonu: {test_sayisi} (parametreli ve özellik testleri tek sayılır)")
    s.append(f"- Kaynak kuralından muaf altyapı fonksiyonu: {len(infrastructure_registry())}")
    s.append("")
    s.append("`okundu`: denklem kod yazılırken kaynağın metninden okundu. `bekliyor`: denklem hatırlanan biçimiyle")
    s.append("yazıldı, testlerle sınandı, ama kaynağın metni henüz açılmadı.")
    s.append("")

    s.append("## Fonksiyonlar")
    moduller: dict[str, list[str]] = {}
    for name in sorted(kayitlar):
        moduller.setdefault(name.rsplit(".", 1)[0], []).append(name)
    for modul, adlar in moduller.items():
        s += ["", f"### `{modul}`", ""]
        for name in adlar:
            parts = "; ".join(f"[{k.key}] {k.where} ({k.status})" for k in kayitlar[name])
            s.append(f"- `{name.rsplit('.', 1)[1]}`: {parts}")

    s += ["", "## Sabitler", ""]
    for name, x in sabitler.items():
        uncertainty = "kesin" if x.uncertainty == 0 else f"± {x.uncertainty:.2g}"
        ek = f"; türetim: {x.derivation}" if x.derivation else ""
        s.append(f"- `{name}` = {x.value:.12g} {x.unit} ({uncertainty}) [{x.source}] {x.where} ({x.status}{ek})")

    s += ["", "## Kaynak başına fonksiyon sayısı", ""]
    for key, adlar in reverse_index().items():
        s.append(f"- [{key}]: {len(adlar)}")

    bekleyen = sorted({name for name, ks in kayitlar.items() if any(k.status == "pending" for k in ks)})
    s += ["", "## Kaynağı okunmayı bekleyenler", ""]
    for name in bekleyen:
        s.append(f"- `{name}`: " + "; ".join(f"[{k.key}] {k.where}" for k in kayitlar[name] if k.status == "pending"))
    s.append("")
    (ROOT / "plan" / "DURUM.md").write_text("\n".join(s), encoding="utf-8")
    print(
        f"{len(kayitlar)} fonksiyon, {sum(status.values())} kayıt (okundu {status['read']}, bekliyor {status['pending']}), {len(sabitler)} sabit, {test_sayisi} test"
    )


if __name__ == "__main__":
    main()
