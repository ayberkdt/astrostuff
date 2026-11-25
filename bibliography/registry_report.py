"""Generate the implementation status from the source registry: ``plan/STATUS.md``.

The report is not written by hand; it is derived from the ``@source`` records
and the ``Constant`` definitions in the ``kozmos`` package. This way the
"implemented" information in the documentation cannot drift away from the code.

Usage:
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
    for info in pkgutil.walk_packages(kozmos.__path__, "kozmos."):
        importlib.import_module(info.name)
    records = registry()
    constants = all_constants()
    status = Counter(k.status for ks in records.values() for k in ks)
    fully_read = sum(all(k.status == "read" for k in ks) for ks in records.values())
    test_count = sum(
        1
        for p in (ROOT / "tests").rglob("test_*.py")
        for s in p.read_text(encoding="utf-8").splitlines()
        if s.startswith("def test_")
    )

    s = ["# Implementation status", ""]
    s.append(
        "This file is generated from the source registry by `python bibliography/registry_report.py`; do not edit it by hand."
    )
    s.append("")
    s.append(f"- Public functions tied to a source: {len(records)}")
    s.append(f"- Functions whose records were all read from the source text: {fully_read}")
    s.append(f"- Source records: {sum(status.values())} (read: {status['read']}, pending: {status['pending']})")
    s.append(
        f"- Constants with a source: {len(constants)} (awaiting verification: {sum(x.status != 'read' for x in constants.values())})"
    )
    s.append(f"- Test functions: {test_count} (parametrized and property tests count as one)")
    s.append(f"- Infrastructure functions exempt from the source rule: {len(infrastructure_registry())}")
    s.append("")
    s.append(
        "`read`: the equation was read from the text of the source while the code was written. `pending`: the equation"
    )
    s.append("was written as remembered and tested, but the text of the source has not been opened yet.")
    s.append("")

    s.append("## Functions")
    modules: dict[str, list[str]] = {}
    for name in sorted(records):
        modules.setdefault(name.rsplit(".", 1)[0], []).append(name)
    for module, names in modules.items():
        s += ["", f"### `{module}`", ""]
        for name in names:
            parts = "; ".join(f"[{k.key}] {k.where} ({k.status})" for k in records[name])
            s.append(f"- `{name.rsplit('.', 1)[1]}`: {parts}")

    s += ["", "## Constants", ""]
    for name, x in constants.items():
        uncertainty = "exact" if x.uncertainty == 0 else f"± {x.uncertainty:.2g}"
        derivation = f"; derivation: {x.derivation}" if x.derivation else ""
        s.append(
            f"- `{name}` = {x.value:.12g} {x.unit} ({uncertainty}) [{x.source}] {x.where} ({x.status}{derivation})"
        )

    s += ["", "## Number of functions per source", ""]
    for key, names in reverse_index().items():
        s.append(f"- [{key}]: {len(names)}")

    pending = sorted({name for name, ks in records.items() if any(k.status == "pending" for k in ks)})
    s += ["", "## Functions awaiting source reading", ""]
    for name in pending:
        s.append(f"- `{name}`: " + "; ".join(f"[{k.key}] {k.where}" for k in records[name] if k.status == "pending"))
    s.append("")
    (ROOT / "plan" / "STATUS.md").write_text("\n".join(s), encoding="utf-8")
    print(
        f"{len(records)} functions, {sum(status.values())} records (read {status['read']}, pending {status['pending']}), {len(constants)} constants, {test_count} tests"
    )


if __name__ == "__main__":
    main()
