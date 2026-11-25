"""İndirilen verinin yerel önbelleği ve kaynak kaydı.

Her indirme, dosyanın yanındaki ``KAYNAK_KAYDI.json`` dosyasına adresi,
SHA-256 özeti, boyutu, zamanı ve atıf metniyle işlenir. Aynı istek ikinci kez
ağa çıkmaz. İlkeler ``plan/05_data_science/05_data_management_and_reproducibility.md``
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

from kozmos._sources import infrastructure

USER_AGENT = "kozmos/0.2 (python-urllib)"


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


@infrastructure("ağ ve dosya altyapısı; denklem uygulamaz")
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
    """Bir adresi (isteğe bağlı sorgu parametreleriyle) dosyaya indirir ve kaydını yazar.

    Dosya varsa ve ``yenile`` verilmemişse ağa çıkılmaz. İndirme önce geçici
    bir dosyaya yapılır; yarıda kesilen indirme hedefi bozmaz. Geçici ağ hataları
    (bağlantı kopması, zaman aşımı, 5xx yanıtı) ``deneme_sayisi`` kez, aralarında
    ``bekleme_s`` saniye beklenerek yeniden denenir; 4xx yanıtları yeniden denenmez.
    Son deneme de başarısızsa hata, kaç deneme yapıldığıyla birlikte yükseltilir.
    """
    target = Path(target)
    if target.exists() and not refresh:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tam_url = url + ("?" + urllib.parse.urlencode(parameters) if parameters else "")
    istek = urllib.request.Request(tam_url, headers={"User-Agent": USER_AGENT})
    gecici = target.with_name(target.name + ".kismi")
    for deneme in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(istek, timeout=timeout_s) as yanit, gecici.open("wb") as f:  # noqa: S310
                for parca in iter(lambda: yanit.read(1 << 20), b""):
                    f.write(parca)
            break
        except urllib.error.HTTPError as hata:
            if hata.code < 500 or deneme == attempts:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as hata:
            if deneme == attempts:
                raise OSError(f"{attempts} denemede indirilemedi: {tam_url} ({hata})") from hata
        time.sleep(wait_s)
    gecici.replace(target)
    kayit_yolu = target.parent / "KAYNAK_KAYDI.json"
    kayit = json.loads(kayit_yolu.read_text(encoding="utf-8")) if kayit_yolu.exists() else {}
    kayit[target.name] = {
        "url": tam_url,
        "sha256": _digest(target),
        "bayt": target.stat().st_size,
        "indirme_zamani": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "atif": attribution,
    }
    kayit_yolu.write_text(json.dumps(kayit, ensure_ascii=False, indent=1), encoding="utf-8")
    return target


@infrastructure("istek için kararlı dosya adı üretir; denklem uygulamaz")
def key(*parts: str) -> str:
    """İstek parçalarından 16 haneli kararlı bir özet üretir (önbellek dosya adı için)."""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
