"""Kaynakça ve paket envanterini kaynağından doğrular; kanıtı dosyaya yazar.

Üç denetim yapar ve sonucu ``dogrulama_kaydi.json`` ile ``DOGRULAMA_OZETI.md``
dosyalarına kaydeder:

1. ``kaynaklar.bib`` içindeki her giriş için künyeyi dış kayıtla karşılaştırır:
   DOI -> Crossref (bulunamazsa DataCite), eprint -> arXiv API, ISBN -> Open Library.
   Başlık benzerliği ve yıl farkı ölçülür.
2. ``paketler.txt`` içindeki her paket için PyPI JSON API'sinden son sürümü ve
   yükleme tarihini okur.
3. ``plan/`` altındaki Markdown dosyalarında geçen ``[Anahtar]`` atıflarının ve
   ``src/`` altındaki ``@kaynak`` kayıtlarının ``kaynaklar.bib`` içinde tanımlı
   olduğunu denetler.

Kullanım:
    python dogrula.py            # üç denetim
    python dogrula.py --yalniz atif   # yalnızca atıf denetimi (ağ gerekmez)

Çıkış kodu: tanımsız atıf ya da uyuşmayan künye varsa 1, yoksa 0.
Dış servislere yalnızca DOI, arXiv kimliği, ISBN ve paket adı gönderilir.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

DIZIN = Path(__file__).resolve().parent
KOK = DIZIN.parent
PLAN = KOK / "plan"
ISTEMCI = "kaynakca-dogrulama/1.0 (python-urllib)"
BASLIK_ESIGI = 0.80
YIL_TOLERANSI = 1  # çevrimiçi ve basılı yayın yılı bir yıl ayrışabilir


def getir(url: str, deneme: int = 3) -> tuple[int, bytes]:
    """URL'yi okur; (HTTP durumu, gövde) döndürür. Ağ hatasında durum 0'dır."""
    istek = urllib.request.Request(url, headers={"User-Agent": ISTEMCI, "Accept": "application/json, */*"})
    for i in range(deneme):
        try:
            with urllib.request.urlopen(istek, timeout=30) as yanit:
                return yanit.status, yanit.read()
        except urllib.error.HTTPError as hata:
            if hata.code in (429, 500, 502, 503, 504) and i < deneme - 1:
                time.sleep(2 * (i + 1))
                continue
            return hata.code, b""
        except (urllib.error.URLError, TimeoutError, OSError):
            if i < deneme - 1:
                time.sleep(2 * (i + 1))
                continue
            return 0, b""
    return 0, b""


def sadelestir(metin: str) -> str:
    """Başlığı karşılaştırma için küçük harfli ASCII sözcük dizisine indirger."""
    metin = re.sub(r"<[^>]+>", " ", metin)
    metin = unicodedata.normalize("NFKD", metin).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", metin.lower()).split())


def benzerlik(a: str, b: str) -> float:
    a, b = sadelestir(a), sadelestir(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return round(SequenceMatcher(None, a, b).ratio(), 3)


def bib_oku(yol: Path) -> dict[str, dict[str, str]]:
    """Tek satırlık girişlerden oluşan .bib dosyasını sözlüğe çevirir."""
    girisler: dict[str, dict[str, str]] = {}
    for satir in yol.read_text(encoding="utf-8").splitlines():
        eslesme = re.match(r"@(\w+)\{([^,]+),\s*(.*)\}\s*$", satir.strip())
        if not eslesme:
            continue
        tur, anahtar, govde = eslesme.groups()
        alanlar = {"tur": tur}
        for ad, deger in re.findall(r"(\w+)=\{((?:[^{}]|\{[^{}]*\})*)\}", govde):
            alanlar[ad] = deger.replace("{", "").replace("}", "")
        girisler[anahtar] = alanlar
    return girisler


def crossref(doi: str) -> dict | None:
    durum, govde = getir("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/"))
    if durum != 200:
        return None
    m = json.loads(govde)["message"]
    yil = None
    for alan in ("published-print", "issued", "published-online", "published"):
        parca = m.get(alan, {}).get("date-parts", [[None]])[0]
        if parca and parca[0]:
            yil = parca[0]
            break
    yazar = m.get("author", [{}])
    ilk = yazar[0].get("family") or yazar[0].get("name") if yazar else None
    return {
        "kaynak": "Crossref",
        "baslik": (m.get("title") or [""])[0],
        "yil": yil,
        "ilk_yazar": ilk,
        "yayin": (m.get("container-title") or [m.get("publisher", "")])[0],
        "cilt": m.get("volume"),
        "sayfa": m.get("page") or m.get("article-number"),
    }


def datacite(doi: str) -> dict | None:
    durum, govde = getir("https://api.datacite.org/dois/" + urllib.parse.quote(doi, safe="/"))
    if durum != 200:
        return None
    a = json.loads(govde)["data"]["attributes"]
    yaratan = a.get("creators") or [{}]
    return {
        "kaynak": "DataCite",
        "baslik": (a.get("titles") or [{}])[0].get("title", ""),
        "yil": a.get("publicationYear"),
        "ilk_yazar": yaratan[0].get("familyName") or yaratan[0].get("name"),
        "yayin": a.get("publisher") if isinstance(a.get("publisher"), str) else str(a.get("publisher")),
    }


def arxiv(kimlik: str) -> dict | None:
    durum, govde = getir("https://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(kimlik, safe="/"))
    if durum != 200:
        return None
    ad = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    giris = ET.fromstring(govde).find("a:entry", ad)
    if giris is None or giris.find("a:title", ad) is None or giris.find("a:published", ad) is None:
        return None
    yazar = giris.find("a:author/a:name", ad)
    dergi = giris.find("x:journal_ref", ad)
    doi = giris.find("x:doi", ad)
    return {
        "kaynak": "arXiv",
        "baslik": " ".join((giris.findtext("a:title", "", ad)).split()),
        "yil": int(giris.findtext("a:published", "0000", ad)[:4]),
        "ilk_yazar": yazar.text.split()[-1] if yazar is not None and yazar.text else None,
        "yayin": dergi.text if dergi is not None else None,
        "doi": doi.text if doi is not None else None,
    }


def openlibrary(isbn: str) -> dict | None:
    durum, govde = getir(f"https://openlibrary.org/isbn/{isbn}.json")
    if durum != 200:
        return None
    k = json.loads(govde)
    yil = re.search(r"(1[89]|20)\d\d", k.get("publish_date", ""))
    baslik = k.get("title", "")
    if k.get("subtitle"):
        baslik += ": " + k["subtitle"]
    return {
        "kaynak": "Open Library",
        "baslik": baslik,
        "yil": int(yil.group()) if yil else None,
        "ilk_yazar": None,
        "yayin": ", ".join(k.get("publishers", [])),
    }


def kunye_denetle(anahtar: str, g: dict[str, str]) -> dict:
    kayit: dict = {"anahtar": anahtar, "bib_baslik": g.get("title"), "bib_yil": g.get("year")}
    dis = None
    if "doi" in g:
        kayit["kimlik"] = "doi:" + g["doi"]
        dis = crossref(g["doi"]) or datacite(g["doi"])
    elif "eprint" in g:
        kayit["kimlik"] = "arXiv:" + g["eprint"]
        dis = arxiv(g["eprint"])
        time.sleep(3)  # arXiv API kullanım kuralı
    elif "isbn" in g:
        kayit["kimlik"] = "isbn:" + g["isbn"]
        dis = openlibrary(g["isbn"])
    else:
        kayit.update(kimlik=g.get("url"), durum="ELLE", aciklama="DOI, arXiv ya da ISBN yok; elle doğrulanacak")
        return kayit
    if dis is None:
        kayit.update(durum="BULUNAMADI", aciklama="dış kayıt okunamadı")
        return kayit
    kayit["dis"] = dis
    b = benzerlik(g.get("title", ""), dis["baslik"])
    kayit["baslik_benzerligi"] = b
    yil_farki = None
    if dis.get("yil") and g.get("year", "").isdigit():
        yil_farki = abs(int(dis["yil"]) - int(g["year"]))
    kayit["yil_farki"] = yil_farki
    sorun = []
    if b < BASLIK_ESIGI:
        sorun.append(f"başlık benzerliği {b}")
    # arXiv yılı ön baskı yılıdır; kitapta baskı yılı değişir. Yıl yalnızca DOI kaydında bağlayıcıdır.
    if "doi" in g and yil_farki is not None and yil_farki > YIL_TOLERANSI:
        sorun.append(f"yıl farkı {yil_farki}")
    kayit["durum"] = "UYUSMUYOR" if sorun else "DOGRULANDI"
    if sorun:
        kayit["aciklama"] = "; ".join(sorun)
    return kayit


def pypi(ad: str) -> dict:
    durum, govde = getir(f"https://pypi.org/pypi/{ad}/json")
    if durum != 200:
        return {"paket": ad, "durum": "BULUNAMADI", "http": durum}
    v = json.loads(govde)
    surum = v["info"]["version"]
    dosyalar = v.get("urls") or []
    tarih = min((d["upload_time_iso_8601"][:10] for d in dosyalar), default=None)
    tum = [d["upload_time_iso_8601"][:10] for ds in v.get("releases", {}).values() for d in ds]
    return {
        "paket": ad,
        "durum": "OKUNDU",
        "surum": surum,
        "surum_tarihi": tarih,
        "son_yukleme": max(tum, default=None),
        "python": v["info"].get("requires_python"),
        "ozet": v["info"].get("summary"),
    }


def atif_denetle(anahtarlar: set[str]) -> dict:
    """plan/ altındaki .md dosyalarında [Anahtar] atıflarını bulur."""
    # [Anahtar], [Anahtar, Böl. 3] ve [Anahtar1; Anahtar2, Denk. 5] biçimleri; Markdown bağlantıları hariç.
    ayrac = re.compile(r"\[([^\[\]]+)\](?!\()")
    anahtar = re.compile(r"^([A-Za-z]+(?:19|20)\d\d[A-Za-z0-9]*)(?=$|,)")
    kullanim: dict[str, list[str]] = {}
    for dosya in sorted(PLAN.rglob("*.md")):
        if dosya.name == "DOGRULAMA_OZETI.md":
            continue
        goreli = dosya.relative_to(PLAN).as_posix()
        for grup in ayrac.findall(dosya.read_text(encoding="utf-8")):
            for parca in grup.split(";"):
                eslesme = anahtar.match(parca.strip())
                if eslesme and goreli not in kullanim.setdefault(eslesme.group(1), []):
                    kullanim[eslesme.group(1)].append(goreli)
    # Kaynak kodundaki @kaynak("Anahtar", ...) kayıtları ve Sabit(..., "Anahtar", ...) tanımları.
    for dosya in sorted((KOK / "src").rglob("*.py")):
        metin = dosya.read_text(encoding="utf-8")
        goreli = dosya.relative_to(KOK).as_posix()
        for a in re.findall(r'kaynak\(\s*"([A-Za-z]+(?:19|20)\d\d[A-Za-z0-9]*)"', metin):
            if goreli not in kullanim.setdefault(a, []):
                kullanim[a].append(goreli)
    return {
        "tanimsiz": {a: d for a, d in sorted(kullanim.items()) if a not in anahtarlar},
        "kullanilmayan": sorted(anahtarlar - set(kullanim)),
        "kullanilan_sayisi": len([a for a in kullanim if a in anahtarlar]),
    }


def ozet_yaz(kayit: dict) -> str:
    s = [f"# Doğrulama özeti\n\nÜretim: `dogrula.py`, {kayit['zaman']}. Ayrıntı: `dogrulama_kaydi.json`.\n"]
    if "kunyeler" in kayit:
        say: dict[str, int] = {}
        for k in kayit["kunyeler"]:
            say[k["durum"]] = say.get(k["durum"], 0) + 1
        s.append("## Künyeler\n")
        s.append(", ".join(f"{d}: {n}" for d, n in sorted(say.items())) + f" (toplam {len(kayit['kunyeler'])}).\n")
        for durum, baslik in (
            ("UYUSMUYOR", "Uyuşmayan"),
            ("BULUNAMADI", "Dış kaydı okunamayan"),
            ("ELLE", "Elle doğrulanacak"),
        ):
            secili = [k for k in kayit["kunyeler"] if k["durum"] == durum]
            if secili:
                s.append(f"### {baslik}\n")
                for k in secili:
                    dis = k.get("dis", {})
                    ek = f" → dış kayıt: «{dis.get('baslik')}» ({dis.get('yil')})" if dis else ""
                    s.append(f"- `{k['anahtar']}` ({k.get('kimlik')}): {k.get('aciklama', '')}{ek}")
                s.append("")
        s.append("### Doğrulanan\n")
        for k in kayit["kunyeler"]:
            if k["durum"] == "DOGRULANDI":
                d = k["dis"]
                yer = ", ".join(str(x) for x in (d.get("yayin"), d.get("cilt"), d.get("sayfa")) if x)
                s.append(
                    f"- `{k['anahtar']}`: {d.get('ilk_yazar') or ''} {d.get('yil')}, «{d['baslik']}». {yer} [{d['kaynak']}, {k['kimlik']}]"
                )
        s.append("")
    if "paketler" in kayit:
        s.append("## Paketler (PyPI)\n")
        for p in kayit["paketler"]:
            if p["durum"] == "OKUNDU":
                s.append(f"- `{p['paket']}` {p['surum']} ({p['surum_tarihi']}); Python {p['python']}")
            else:
                s.append(f"- `{p['paket']}`: PyPI'da bulunamadı (HTTP {p['http']})")
        s.append("")
    if "atiflar" in kayit:
        a = kayit["atiflar"]
        s.append("## Atıf denetimi\n")
        s.append(f"Plan belgelerinde kullanılan tanımlı anahtar: {a['kullanilan_sayisi']}.\n")
        if a["tanimsiz"]:
            s.append("Tanımsız anahtarlar:\n")
            s += [f"- `{k}`: {', '.join(v)}" for k, v in a["tanimsiz"].items()]
        else:
            s.append("Tanımsız anahtar yok.")
        s.append(f"\nKaynakçada olup planda anılmayan: {len(a['kullanilmayan'])}.")
    return "\n".join(s) + "\n"


def main() -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ayristirici.add_argument("--yalniz", choices=["kunye", "paket", "atif"], help="tek denetimi koş")
    secim = ayristirici.parse_args().yalniz
    bib = bib_oku(DIZIN / "kaynaklar.bib")
    kayit_yolu = DIZIN / "dogrulama_kaydi.json"
    # Kısmi koşu önceki koşunun diğer bölümlerini korur.
    kayit: dict = json.loads(kayit_yolu.read_text(encoding="utf-8")) if kayit_yolu.exists() and secim else {}
    kayit["zaman"] = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")

    if secim in (None, "kunye"):
        kayit["kunyeler"] = []
        for anahtar, g in bib.items():
            sonuc = kunye_denetle(anahtar, g)
            kayit["kunyeler"].append(sonuc)
            print(f"{sonuc['durum']:<11} {anahtar}", flush=True)
            time.sleep(0.15)
    if secim in (None, "paket"):
        adlar = [s.strip() for s in (DIZIN / "paketler.txt").read_text(encoding="utf-8").splitlines()]
        kayit["paketler"] = [pypi(a) for a in adlar if a and not a.startswith("#")]
        for p in kayit["paketler"]:
            print(f"{p['durum']:<11} {p['paket']} {p.get('surum', '')}", flush=True)
    if secim in (None, "atif"):
        kayit["atiflar"] = atif_denetle(set(bib))
        print(f"tanımsız atıf: {len(kayit['atiflar']['tanimsiz'])}")

    kayit_yolu.write_text(json.dumps(kayit, ensure_ascii=False, indent=1), encoding="utf-8")
    (DIZIN / "DOGRULAMA_OZETI.md").write_text(ozet_yaz(kayit), encoding="utf-8")
    hatali = [k for k in kayit.get("kunyeler", []) if k["durum"] == "UYUSMUYOR"]
    return 1 if hatali or kayit.get("atiflar", {}).get("tanimsiz") else 0


if __name__ == "__main__":
    sys.exit(main())
