"""Machine check of the source-citation rule.

Rule: every public function under kozmos carries at least one ``@source``
record, the key of the record is defined in the bibliography, the
``References`` section of the docstring names the same key, and the test file
named in the ``Verification`` section exists. Docstrings are written in English.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import pytest

import kozmos
from kozmos._sources import bib_keys, infrastructure, infrastructure_registry, registry, source, sources_of
from kozmos.core.constants import all_constants

ROOT = Path(__file__).resolve().parents[1]
BIB = bib_keys(ROOT / "bibliography")
TURKISH_LETTERS = re.compile("[çğıöşüÇĞİÖŞÜ]")


def _public_functions():
    for info in pkgutil.walk_packages(kozmos.__path__, "kozmos."):
        if info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        module = importlib.import_module(info.name)
        for name, obj in vars(module).items():
            if name.startswith("_") or not inspect.isfunction(obj) or obj.__module__ != module.__name__:
                continue
            if f"{module.__name__}.{name}" in infrastructure_registry():
                continue
            yield pytest.param(obj, id=f"{module.__name__}.{name}")


FUNCTIONS = list(_public_functions())


def test_functions_found():
    assert len(FUNCTIONS) > 30


@pytest.mark.parametrize("f", FUNCTIONS)
def test_source_record_exists_and_is_defined(f):
    records = sources_of(f)
    assert records, "public function carries no @source"
    for record in records:
        assert record.key in BIB, f"{record.key} is not in the bibliography"
        assert record.where.strip()
        assert record.status in ("read", "pending")


@pytest.mark.parametrize("f", FUNCTIONS)
def test_docstring_names_the_source(f):
    doc = inspect.getdoc(f) or ""
    assert "References" in doc, "docstring has no References section"
    section = doc.split("References", 1)[1]
    for record in sources_of(f):
        assert f"[{record.key}]" in section, f"References section does not name [{record.key}]"


@pytest.mark.parametrize("f", FUNCTIONS)
def test_verification_file_exists(f):
    doc = inspect.getdoc(f) or ""
    assert "Verification" in doc, "docstring has no Verification section"
    paths = re.findall(r"tests/[\w/]+\.py", doc.split("Verification", 1)[1])
    assert paths, "Verification section names no test file"
    for path in paths:
        assert (ROOT / path).exists(), f"{path} does not exist"


@pytest.mark.parametrize("f", FUNCTIONS)
def test_docstring_and_records_are_english(f):
    doc = inspect.getdoc(f) or ""
    assert not TURKISH_LETTERS.search(doc), "docstring contains Turkish letters"
    for record in sources_of(f):
        assert not TURKISH_LETTERS.search(record.where + (record.note or "")), "source record contains Turkish letters"


def test_constant_sources_are_defined():
    constants = all_constants()
    assert len(constants) > 25
    for name, constant in constants.items():
        assert constant.source in BIB, f"{name}: {constant.source} is not in the bibliography"
        assert constant.where.strip() and constant.status in ("read", "pending")


def test_every_registry_key_is_defined():
    for name, records in registry().items():
        for record in records:
            assert record.key in BIB, f"{name}: {record.key}"


def test_decorator_rejects_invalid_input():
    with pytest.raises(ValueError):
        source("Hogg1999", where=" ", status="read")
    with pytest.raises(ValueError):
        source("Hogg1999", where="Eq. 1", status="maybe")  # type: ignore[arg-type]


def test_infrastructure_exemptions_are_justified_and_few():
    exempt = infrastructure_registry()
    assert all(reason.strip() for reason in exempt.values())
    assert len(exempt) <= 5, "exemption must stay exceptional; a function that implements an equation carries @source"
    with pytest.raises(ValueError):
        infrastructure("  ")
