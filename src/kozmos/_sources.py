"""Source-citation machinery.

Every public function is tied with ``@source`` to the published source of the
equation it implements. The decorator does not change the function; it adds a
record to a registry. The registry is checked by ``tests/test_source_rule.py``
and is used to list the publications an analysis rests on.

The ``status`` field carries the evidential status of a record:

``"read"``
    The equation was read from the text of the source while the code was written.
``"pending"``
    The equation was written from memory; the text of the source has not been
    opened. The function may be tested, but its link to the source is unverified.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

F = TypeVar("F", bound=Callable[..., object])
Status = Literal["read", "pending"]


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Link between one function and one source."""

    key: str
    where: str
    status: Status
    note: str | None = None


_REGISTRY: dict[str, list[SourceRecord]] = {}
_INFRASTRUCTURE: dict[str, str] = {}


def infrastructure(reason: str) -> Callable[[F], F]:
    """Exempt a public function from the source rule and record why.

    Only for helpers that implement no physical or mathematical equation
    (registry queries, format conversion). The rule test skips these functions
    and checks that the reason is not empty.
    """
    if not reason.strip():
        raise ValueError("the infrastructure reason must not be empty")

    def wrap(f: F) -> F:
        _INFRASTRUCTURE[f"{f.__module__}.{f.__qualname__}"] = reason
        return f

    return wrap


def infrastructure_registry() -> dict[str, str]:
    """Functions exempt from the source rule, with their reasons."""
    return dict(_INFRASTRUCTURE)


def source(key: str, *, where: str, status: Status, note: str | None = None) -> Callable[[F], F]:
    """Tie a function to an entry in the bibliography.

    Parameters
    ----------
    key : str
        BibTeX key in one of the ``bibliography/*.bib`` files.
    where : str
        Location inside the source: equation number, section, table or file name.
    status : {"read", "pending"}
        Whether the equation was read from the text of the source.
    note : str, optional
        Symbol mapping, generalisation or restriction.
    """
    if not where.strip():
        raise ValueError("the location inside the source must not be empty")
    if status not in ("read", "pending"):
        raise ValueError(f"invalid status: {status!r}")

    def wrap(f: F) -> F:
        name = f"{f.__module__}.{f.__qualname__}"
        _REGISTRY.setdefault(name, []).append(SourceRecord(key, where, status, note))
        return f

    return wrap


def sources_of(f: Callable[..., object]) -> tuple[SourceRecord, ...]:
    """Return the source records of a function."""
    return tuple(_REGISTRY.get(f"{f.__module__}.{f.__qualname__}", ()))


def registry() -> dict[str, tuple[SourceRecord, ...]]:
    """Return a copy of every record: qualified name -> records."""
    return {name: tuple(records) for name, records in _REGISTRY.items()}


def reverse_index() -> dict[str, list[str]]:
    """Source key -> names of the functions that rest on that source."""
    index: dict[str, list[str]] = {}
    for name, records in _REGISTRY.items():
        for record in records:
            index.setdefault(record.key, []).append(name)
    return {key: sorted(set(names)) for key, names in sorted(index.items())}


def bib_keys(path: Path) -> set[str]:
    """Read the keys of one ``.bib`` file, or of every ``.bib`` file in a directory."""
    files = sorted(path.glob("*.bib")) if path.is_dir() else [path]
    keys: set[str] = set()
    for file in files:
        keys |= set(re.findall(r"^@\w+\{([^,\s]+),", file.read_text(encoding="utf-8"), flags=re.M))
    return keys


def bibliography_report(*functions: Callable[..., object]) -> str:
    """List, line by line, the sources the given functions rest on.

    With no functions, the whole registry is reported.
    """
    if functions:
        selected = {f"{f.__module__}.{f.__qualname__}": sources_of(f) for f in functions}
    else:
        selected = registry()
    lines = []
    for name in sorted(selected):
        for record in selected[name]:
            extra = f" ({record.note})" if record.note else ""
            lines.append(f"{name}: [{record.key}] {record.where} · {record.status}{extra}")
    return "\n".join(lines)
