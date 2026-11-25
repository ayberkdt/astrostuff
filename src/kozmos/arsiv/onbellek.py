"""İndirilen verinin yerel önbelleği ve kaynak kaydı.

Her indirme, dosyanın yanındaki ``KAYNAK_KAYDI.json`` dosyasına adresi,
SHA-256 özeti, boyutu, zamanı ve atıf metniyle işlenir. Aynı istek ikinci kez
ağa çıkmaz. İlkeler ``plan/05_veri_bilimi/05_veri_yonetimi_ve_tekrarlanabilirlik.md``
belgesindedir [Wilson2017].
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

from kozmos._kaynak import altyapi

ISTEMCI = "kozmos/0.2 (python-urllib)"


def _ozet(yol: Path) -> str:
    h = hashlib.sha256()
    with yol.open("rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


@altyapi("ağ ve dosya altyapısı; denklem uygulamaz")
def indir(
    url: str,
    hedef: Path,
    *,
    parametreler: dict[str, str] | None = None,
    atif: str = "",
    yenile: bool = False,
    zaman_asimi_s: float = 300.0,
    deneme_sayisi: int = 3,
    bekleme_s: float = 5.0,
) -> Path:
    """Bir adresi (isteğe bağlı sorgu parametreleriyle) dosyaya indirir ve kaydını yazar.

    Dosya varsa ve ``yenile`` verilmemişse ağa çıkılmaz. İndirme önce geçici
    bir dosyaya yapılır; yarıda kesilen indirme hedefi bozmaz. Geçici ağ hataları
    (bağlantı kopması, zaman aşımı, 5xx yanıtı) ``deneme_sayisi`` kez, aralarında
    ``bekleme_s`` saniye beklenerek yeniden denenir; 4xx yanıtları yeniden denenmez.
    Son deneme de başarısızsa hata, kaç deneme yapıldığıyla birlikte yükseltilir.
    """
    hedef = Path(hedef)
    if hedef.exists() and not yenile:
        return hedef
    hedef.parent.mkdir(parents=True, exist_ok=True)
    tam_url = url + ("?" + urllib.parse.urlencode(parametreler) if parametreler else "")
    istek = urllib.request.Request(tam_url, headers={"User-Agent": ISTEMCI})
    gecici = hedef.with_name(hedef.name + ".kismi")
    for deneme in range(1, deneme_sayisi + 1):
        try:
            with urllib.request.urlopen(istek, timeout=zaman_asimi_s) as yanit, gecici.open("wb") as f:  # noqa: S310
                for parca in iter(lambda: yanit.read(1 << 20), b""):
                    f.write(parca)
            break
        except urllib.error.HTTPError as hata:
            if hata.code < 500 or deneme == deneme_sayisi:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as hata:
            if deneme == deneme_sayisi:
                raise OSError(f"{deneme_sayisi} denemede indirilemedi: {tam_url} ({hata})") from hata
        time.sleep(bekleme_s)
    gecici.replace(hedef)
    kayit_yolu = hedef.parent / "KAYNAK_KAYDI.json"
    kayit = json.loads(kayit_yolu.read_text(encoding="utf-8")) if kayit_yolu.exists() else {}
    kayit[hedef.name] = {
        "url": tam_url,
        "sha256": _ozet(hedef),
        "bayt": hedef.stat().st_size,
        "indirme_zamani": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "atif": atif,
    }
    kayit_yolu.write_text(json.dumps(kayit, ensure_ascii=False, indent=1), encoding="utf-8")
    return hedef


@altyapi("istek için kararlı dosya adı üretir; denklem uygulamaz")
def anahtar(*parcalar: str) -> str:
    """İstek parçalarından 16 haneli kararlı bir özet üretir (önbellek dosya adı için)."""
    return hashlib.sha256("\x1f".join(parcalar).encode("utf-8")).hexdigest()[:16]
