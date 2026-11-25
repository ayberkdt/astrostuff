"""Kaynak gösterme kuralının makine denetimi.

Kural: kozmos altındaki her açık fonksiyon en az bir ``@kaynak`` kaydı taşır,
kayıttaki anahtar kaynakçada tanımlıdır, belge dizgisinin ``Kaynaklar`` bölümü
aynı anahtarı anar ve ``Doğrulama`` bölümünde adı geçen test dosyası vardır.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import pytest

import kozmos
from kozmos._kaynak import altyapi, altyapi_kutugu, bib_anahtarlari, kaynak, kaynaklar, kutuk
from kozmos.temel.sabitler import tum_sabitler

KOK = Path(__file__).resolve().parents[1]
BIB = bib_anahtarlari(KOK / "kaynakca" / "kaynaklar.bib")


def _acik_fonksiyonlar():
    for bilgi in pkgutil.walk_packages(kozmos.__path__, "kozmos."):
        if bilgi.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        modul = importlib.import_module(bilgi.name)
        for ad, nesne in vars(modul).items():
            if ad.startswith("_") or not inspect.isfunction(nesne) or nesne.__module__ != modul.__name__:
                continue
            if f"{modul.__name__}.{ad}" in altyapi_kutugu():
                continue
            yield pytest.param(nesne, id=f"{modul.__name__}.{ad}")


FONKSIYONLAR = list(_acik_fonksiyonlar())


def test_fonksiyon_bulundu():
    assert len(FONKSIYONLAR) > 30


@pytest.mark.parametrize("f", FONKSIYONLAR)
def test_kaynak_kaydi_var_ve_tanimli(f):
    kayitlar = kaynaklar(f)
    assert kayitlar, "açık fonksiyon @kaynak taşımıyor"
    for k in kayitlar:
        assert k.anahtar in BIB, f"{k.anahtar} kaynakçada yok"
        assert k.yer.strip()
        assert k.teyit in ("okundu", "bekliyor")


@pytest.mark.parametrize("f", FONKSIYONLAR)
def test_belge_dizgisi_kaynagi_anar(f):
    belge = inspect.getdoc(f) or ""
    assert "Kaynaklar" in belge, "belge dizgisinde Kaynaklar bölümü yok"
    bolum = belge.split("Kaynaklar", 1)[1]
    for k in kaynaklar(f):
        assert f"[{k.anahtar}]" in bolum, f"Kaynaklar bölümü [{k.anahtar}] anahtarını anmıyor"


@pytest.mark.parametrize("f", FONKSIYONLAR)
def test_dogrulama_dosyasi_var(f):
    belge = inspect.getdoc(f) or ""
    assert "Doğrulama" in belge, "belge dizgisinde Doğrulama bölümü yok"
    yollar = re.findall(r"tests/[\w/]+\.py", belge.split("Doğrulama", 1)[1])
    assert yollar, "Doğrulama bölümü bir test dosyası göstermiyor"
    for yol in yollar:
        assert (KOK / yol).exists(), f"{yol} yok"


def test_sabitlerin_kaynagi_tanimli():
    sabitler = tum_sabitler()
    assert len(sabitler) > 25
    for ad, s in sabitler.items():
        assert s.kaynak in BIB, f"{ad}: {s.kaynak} kaynakçada yok"
        assert s.yer.strip() and s.teyit in ("okundu", "bekliyor")


def test_kutukteki_her_anahtar_tanimli():
    for ad, kayitlar in kutuk().items():
        for k in kayitlar:
            assert k.anahtar in BIB, f"{ad}: {k.anahtar}"


def test_dekorator_gecersiz_girdiyi_reddeder():
    with pytest.raises(ValueError):
        kaynak("Hogg1999", yer=" ", teyit="okundu")
    with pytest.raises(ValueError):
        kaynak("Hogg1999", yer="Denk. 1", teyit="belki")  # type: ignore[arg-type]


def test_altyapi_muafiyeti_gerekceli_ve_az():
    muaf = altyapi_kutugu()
    assert all(g.strip() for g in muaf.values())
    assert len(muaf) <= 5, "muafiyet istisna olmalı; denklem uygulayan fonksiyon @kaynak taşır"
    with pytest.raises(ValueError):
        altyapi("  ")
