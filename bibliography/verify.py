"""Verify the bibliography and the package inventory against their sources; write the evidence to files.

Three checks are made and the result is saved to ``verification_record.json``
and ``VERIFICATION_SUMMARY.md``:

1. For every entry in the ``references*.bib`` files, compare the bibliographic data
   with an external record: DOI -> Crossref (DataCite if not found), eprint -> arXiv API,
   ISBN -> Open Library. Title similarity and year difference are measured.
2. For every package in ``packages.txt``, read the latest version and upload date
   from the PyPI JSON API.
3. Check that the ``[Key]`` citations in the Markdown files under ``plan/`` and the
   ``@source`` records under ``src/`` are defined in the ``references*.bib`` files.

Usage:
    python bibliography/verify.py                  # all three checks
    python bibliography/verify.py --only citations # citation check only (no network needed)

Exit code: 1 if there is an undefined citation or a mismatching entry, 0 otherwise.
Only the DOI, arXiv identifier, ISBN and package name are sent to external services.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLAN = ROOT / "plan"
USER_AGENT = "bibliography-verification/1.0 (python-urllib)"
TITLE_THRESHOLD = 0.80
YEAR_TOLERANCE = 1  # the online and print publication years can differ by one year


def fetch(url: str, attempts: int = 3) -> tuple[int, bytes]:
    """Read the URL; return (HTTP status, body). The status is 0 on a network error."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json, */*"})
    for i in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            if error.code in (429, 500, 502, 503, 504) and i < attempts - 1:
                time.sleep(2 * (i + 1))
                continue
            return error.code, b""
        except (urllib.error.URLError, TimeoutError, OSError):
            if i < attempts - 1:
                time.sleep(2 * (i + 1))
                continue
            return 0, b""
    return 0, b""


def simplify(text: str) -> str:
    """Reduce a title to a lowercase ASCII word string for comparison."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def similarity(a: str, b: str) -> float:
    a, b = simplify(a), simplify(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return round(SequenceMatcher(None, a, b).ratio(), 3)


def read_bib(path: Path) -> dict[str, dict[str, str]]:
    """Convert a .bib file made of one-line entries into a dictionary."""
    entries: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"@(\w+)\{([^,]+),\s*(.*)\}\s*$", line.strip())
        if not match:
            continue
        kind, key, body = match.groups()
        fields = {"type": kind}
        for name, value in re.findall(r"(\w+)=\{((?:[^{}]|\{[^{}]*\})*)\}", body):
            fields[name] = value.replace("{", "").replace("}", "")
        entries[key] = fields
    return entries


def crossref(doi: str) -> dict | None:
    status, body = fetch("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/"))
    if status != 200:
        return None
    m = json.loads(body)["message"]
    year = None
    for field in ("published-print", "issued", "published-online", "published"):
        part = m.get(field, {}).get("date-parts", [[None]])[0]
        if part and part[0]:
            year = part[0]
            break
    authors = m.get("author", [{}])
    first = authors[0].get("family") or authors[0].get("name") if authors else None
    return {
        "source": "Crossref",
        "title": (m.get("title") or [""])[0],
        "year": year,
        "first_author": first,
        "publication": (m.get("container-title") or [m.get("publisher", "")])[0],
        "volume": m.get("volume"),
        "page": m.get("page") or m.get("article-number"),
    }


def datacite(doi: str) -> dict | None:
    status, body = fetch("https://api.datacite.org/dois/" + urllib.parse.quote(doi, safe="/"))
    if status != 200:
        return None
    a = json.loads(body)["data"]["attributes"]
    creators = a.get("creators") or [{}]
    return {
        "source": "DataCite",
        "title": (a.get("titles") or [{}])[0].get("title", ""),
        "year": a.get("publicationYear"),
        "first_author": creators[0].get("familyName") or creators[0].get("name"),
        "publication": a.get("publisher") if isinstance(a.get("publisher"), str) else str(a.get("publisher")),
    }


def arxiv(identifier: str) -> dict | None:
    status, body = fetch("https://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(identifier, safe="/"))
    if status != 200:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    entry = ET.fromstring(body).find("a:entry", ns)
    if entry is None or entry.find("a:title", ns) is None or entry.find("a:published", ns) is None:
        return None
    author = entry.find("a:author/a:name", ns)
    journal = entry.find("x:journal_ref", ns)
    doi = entry.find("x:doi", ns)
    return {
        "source": "arXiv",
        "title": " ".join((entry.findtext("a:title", "", ns)).split()),
        "year": int(entry.findtext("a:published", "0000", ns)[:4]),
        "first_author": author.text.split()[-1] if author is not None and author.text else None,
        "publication": journal.text if journal is not None else None,
        "doi": doi.text if doi is not None else None,
    }


def openlibrary(isbn: str) -> dict | None:
    status, body = fetch(f"https://openlibrary.org/isbn/{isbn}.json")
    if status != 200:
        return None
    k = json.loads(body)
    year = re.search(r"(1[89]|20)\d\d", k.get("publish_date", ""))
    title = k.get("title", "")
    if k.get("subtitle"):
        title += ": " + k["subtitle"]
    return {
        "source": "Open Library",
        "title": title,
        "year": int(year.group()) if year else None,
        "first_author": None,
        "publication": ", ".join(k.get("publishers", [])),
    }


def check_entry(key: str, g: dict[str, str]) -> dict:
    record: dict = {"key": key, "bib_title": g.get("title"), "bib_year": g.get("year")}
    external = None
    if "doi" in g:
        record["identifier"] = "doi:" + g["doi"]
        external = crossref(g["doi"]) or datacite(g["doi"])
    elif "eprint" in g:
        record["identifier"] = "arXiv:" + g["eprint"]
        external = arxiv(g["eprint"])
        time.sleep(3)  # arXiv API usage rule
    elif "isbn" in g:
        record["identifier"] = "isbn:" + g["isbn"]
        external = openlibrary(g["isbn"])
    else:
        record.update(
            identifier=g.get("url"),
            status="MANUAL",
            description="no DOI, arXiv identifier or ISBN; to be verified by hand",
        )
        return record
    if external is None:
        record.update(status="NOT_FOUND", description="external record could not be read")
        return record
    record["external"] = external
    b = similarity(g.get("title", ""), external["title"])
    record["title_similarity"] = b
    year_difference = None
    if external.get("year") and g.get("year", "").isdigit():
        year_difference = abs(int(external["year"]) - int(g["year"]))
    record["year_difference"] = year_difference
    problems = []
    if b < TITLE_THRESHOLD:
        problems.append(f"title similarity {b}")
    # The arXiv year is the preprint year; the edition year of a book varies. The year is binding only for a DOI record.
    if "doi" in g and year_difference is not None and year_difference > YEAR_TOLERANCE:
        problems.append(f"year difference {year_difference}")
    record["status"] = "MISMATCH" if problems else "VERIFIED"
    if problems:
        record["description"] = "; ".join(problems)
    return record


def pypi(name: str) -> dict:
    status, body = fetch(f"https://pypi.org/pypi/{name}/json")
    if status != 200:
        return {"package": name, "status": "NOT_FOUND", "http": status}
    v = json.loads(body)
    version = v["info"]["version"]
    files = v.get("urls") or []
    date = min((d["upload_time_iso_8601"][:10] for d in files), default=None)
    all_dates = [d["upload_time_iso_8601"][:10] for ds in v.get("releases", {}).values() for d in ds]
    return {
        "package": name,
        "status": "READ",
        "version": version,
        "version_date": date,
        "last_upload": max(all_dates, default=None),
        "python": v["info"].get("requires_python"),
        "summary": v["info"].get("summary"),
    }


def check_citations(keys: set[str]) -> dict:
    """Find the [Key] citations in the .md files under plan/ and the @source records in src/."""
    # Forms [Key], [Key, Section 3] and [Key1; Key2, Eq. 5]; Markdown links are excluded.
    bracket = re.compile(r"\[([^\[\]]+)\](?!\()")
    key_pattern = re.compile(r"^([A-Za-z]+(?:19|20)\d\d[A-Za-z0-9]*)(?=$|,)")
    usage: dict[str, list[str]] = {}
    for file in sorted(PLAN.rglob("*.md")):
        if file.name == "VERIFICATION_SUMMARY.md":
            continue
        relative = file.relative_to(PLAN).as_posix()
        for group in bracket.findall(file.read_text(encoding="utf-8")):
            for part in group.split(";"):
                match = key_pattern.match(part.strip())
                if match and relative not in usage.setdefault(match.group(1), []):
                    usage[match.group(1)].append(relative)
    # The @source("Key", ...) records in the source code.
    for file in sorted((ROOT / "src").rglob("*.py")):
        text = file.read_text(encoding="utf-8")
        relative = file.relative_to(ROOT).as_posix()
        for a in re.findall(r'\bsource\(\s*"([A-Za-z]+(?:19|20)\d\d[A-Za-z0-9]*)"', text):
            if relative not in usage.setdefault(a, []):
                usage[a].append(relative)
    return {
        "undefined": {a: d for a, d in sorted(usage.items()) if a not in keys},
        "unused": sorted(keys - set(usage)),
        "used_count": len([a for a in usage if a in keys]),
    }


def write_summary(record: dict) -> str:
    s = [
        f"# Verification summary\n\nGenerated by `verify.py`, {record['time']}. Details: `verification_record.json`.\n"
    ]
    if "entries" in record:
        counts: dict[str, int] = {}
        for k in record["entries"]:
            counts[k["status"]] = counts.get(k["status"], 0) + 1
        s.append("## Entries\n")
        s.append(", ".join(f"{d}: {n}" for d, n in sorted(counts.items())) + f" (total {len(record['entries'])}).\n")
        for status, heading in (
            ("MISMATCH", "Mismatching"),
            ("NOT_FOUND", "External record could not be read"),
            ("MANUAL", "To be verified by hand"),
        ):
            selected = [k for k in record["entries"] if k["status"] == status]
            if selected:
                s.append(f"### {heading}\n")
                for k in selected:
                    external = k.get("external", {})
                    extra = (
                        f" → external record: «{external.get('title')}» ({external.get('year')})" if external else ""
                    )
                    s.append(f"- `{k['key']}` ({k.get('identifier')}): {k.get('description', '')}{extra}")
                s.append("")
        s.append("### Verified\n")
        for k in record["entries"]:
            if k["status"] == "VERIFIED":
                d = k["external"]
                where = ", ".join(str(x) for x in (d.get("publication"), d.get("volume"), d.get("page")) if x)
                s.append(
                    f"- `{k['key']}`: {d.get('first_author') or ''} {d.get('year')}, «{d['title']}». {where} [{d['source']}, {k['identifier']}]"
                )
        s.append("")
    if "packages" in record:
        s.append("## Packages (PyPI)\n")
        for p in record["packages"]:
            if p["status"] == "READ":
                s.append(f"- `{p['package']}` {p['version']} ({p['version_date']}); Python {p['python']}")
            else:
                s.append(f"- `{p['package']}`: not found on PyPI (HTTP {p['http']})")
        s.append("")
    if "citations" in record:
        a = record["citations"]
        s.append("## Citation check\n")
        s.append(f"Defined keys used in the plan and the source code: {a['used_count']}.\n")
        if a["undefined"]:
            s.append("Undefined keys:\n")
            s += [f"- `{k}`: {', '.join(v)}" for k, v in a["undefined"].items()]
        else:
            s.append("No undefined keys.")
        s.append(f"\nIn the bibliography but not cited in the plan or the code: {len(a['unused'])}.")
    return "\n".join(s) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["entries", "packages", "citations"], help="run a single check")
    selection = parser.parse_args().only
    bib: dict[str, dict[str, str]] = {}
    for file in sorted(HERE.glob("*.bib")):
        bib.update(read_bib(file))
    record_path = HERE / "verification_record.json"
    # A partial run keeps the other sections of the previous run.
    record: dict = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() and selection else {}
    record["time"] = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")

    if selection in (None, "entries"):
        record["entries"] = []
        for key, g in bib.items():
            result = check_entry(key, g)
            record["entries"].append(result)
            print(f"{result['status']:<11} {key}", flush=True)
            time.sleep(0.15)
    if selection in (None, "packages"):
        names = [s.strip() for s in (HERE / "packages.txt").read_text(encoding="utf-8").splitlines()]
        record["packages"] = [pypi(a) for a in names if a and not a.startswith("#")]
        for p in record["packages"]:
            print(f"{p['status']:<11} {p['package']} {p.get('version', '')}", flush=True)
    if selection in (None, "citations"):
        record["citations"] = check_citations(set(bib))
        print(f"undefined citations: {len(record['citations']['undefined'])}")

    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "VERIFICATION_SUMMARY.md").write_text(write_summary(record), encoding="utf-8")
    mismatched = [k for k in record.get("entries", []) if k["status"] == "MISMATCH"]
    return 1 if mismatched or record.get("citations", {}).get("undefined") else 0


if __name__ == "__main__":
    sys.exit(main())
