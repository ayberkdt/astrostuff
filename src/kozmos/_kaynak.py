"""Kaynak gösterme altyapısı.

Her açık fonksiyon ``@kaynak`` ile uyguladığı denklemin yayımlanmış kaynağına
bağlanır. Dekoratör fonksiyonu değiştirmez; kaydı bir kütüğe ekler. Kütük,
``tests/test_kaynak_kurali.py`` tarafından denetlenir ve bir analizin dayandığı
yayınların listesini üretmek için kullanılır.

``teyit`` alanı kaydın kanıt statüsünü taşır:

``"okundu"``
    Denklem, kod yazılırken kaynağın metninden okunmuştur.
``"bekliyor"``
    Denklem hatırlanan biçimiyle yazılmıştır; kaynağın metni henüz açılmamıştır.
    Fonksiyon testlerle sınanmış olabilir, ama kaynağa bağı doğrulanmamıştır.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

F = TypeVar("F", bound=Callable[..., object])
Teyit = Literal["okundu", "bekliyor"]


@dataclass(frozen=True, slots=True)
class KaynakKaydi:
    """Bir fonksiyonun bir kaynağa bağı."""

    anahtar: str
    yer: str
    teyit: Teyit
    not_: str | None = None


_KUTUK: dict[str, list[KaynakKaydi]] = {}
_ALTYAPI: dict[str, str] = {}


def altyapi(gerekce: str) -> Callable[[F], F]:
    """Bir açık fonksiyonu kaynak kuralının dışında tutar ve gerekçesini kaydeder.

    Yalnızca fiziksel ya da matematiksel bir denklem uygulamayan yardımcılar
    içindir (kütük sorgusu, biçim dönüşümü). Denetim testi bu fonksiyonları
    atlar ve gerekçenin boş olmadığını sınar.
    """
    if not gerekce.strip():
        raise ValueError("altyapı gerekçesi boş bırakılamaz")

    def sarmala(f: F) -> F:
        _ALTYAPI[f"{f.__module__}.{f.__qualname__}"] = gerekce
        return f

    return sarmala


def altyapi_kutugu() -> dict[str, str]:
    """Kaynak kuralının dışında tutulan fonksiyonlar ve gerekçeleri."""
    return dict(_ALTYAPI)


def kaynak(anahtar: str, *, yer: str, teyit: Teyit, not_: str | None = None) -> Callable[[F], F]:
    """Fonksiyonu ``kaynaklar.bib`` içindeki bir girişe bağlar.

    Parametreler
    ------------
    anahtar : str
        ``kaynakca/kaynaklar.bib`` dosyasındaki BibTeX anahtarı.
    yer : str
        Kaynağın içindeki yer: denklem numarası, bölüm, tablo ya da dosya adı.
    teyit : {"okundu", "bekliyor"}
        Denklemin kaynağın metninden okunup okunmadığı.
    not_ : str, isteğe bağlı
        Sembol eşlemesi, genelleme ya da kısıt notu.
    """
    if not yer.strip():
        raise ValueError("kaynağın içindeki yer boş bırakılamaz")
    if teyit not in ("okundu", "bekliyor"):
        raise ValueError(f"geçersiz teyit durumu: {teyit!r}")

    def sarmala(f: F) -> F:
        ad = f"{f.__module__}.{f.__qualname__}"
        _KUTUK.setdefault(ad, []).append(KaynakKaydi(anahtar, yer, teyit, not_))
        return f

    return sarmala


def kaynaklar(f: Callable[..., object]) -> tuple[KaynakKaydi, ...]:
    """Bir fonksiyonun kaynak kayıtlarını döndürür."""
    return tuple(_KUTUK.get(f"{f.__module__}.{f.__qualname__}", ()))


def kutuk() -> dict[str, tuple[KaynakKaydi, ...]]:
    """Bütün kayıtların kopyasını döndürür: tam ad → kayıtlar."""
    return {ad: tuple(k) for ad, k in _KUTUK.items()}


def ters_dizin() -> dict[str, list[str]]:
    """Kaynak anahtarı → o kaynağa dayanan fonksiyonların adları."""
    dizin: dict[str, list[str]] = {}
    for ad, kayitlar in _KUTUK.items():
        for k in kayitlar:
            dizin.setdefault(k.anahtar, []).append(ad)
    return {a: sorted(set(v)) for a, v in sorted(dizin.items())}


def bib_anahtarlari(yol: Path) -> set[str]:
    """Bir ``.bib`` dosyasındaki ya da bir klasördeki bütün ``.bib`` dosyalarındaki anahtarları okur."""
    dosyalar = sorted(yol.glob("*.bib")) if yol.is_dir() else [yol]
    anahtarlar: set[str] = set()
    for dosya in dosyalar:
        anahtarlar |= set(re.findall(r"^@\w+\{([^,\s]+),", dosya.read_text(encoding="utf-8"), flags=re.M))
    return anahtarlar


def kaynakca_raporu(*fonksiyonlar: Callable[..., object]) -> str:
    """Verilen fonksiyonların dayandığı kaynakları satır satır listeler.

    Fonksiyon verilmezse kütüğün tamamı raporlanır.
    """
    if fonksiyonlar:
        secili = {f"{f.__module__}.{f.__qualname__}": kaynaklar(f) for f in fonksiyonlar}
    else:
        secili = kutuk()
    satirlar = []
    for ad in sorted(secili):
        for k in secili[ad]:
            ek = f" ({k.not_})" if k.not_ else ""
            satirlar.append(f"{ad}: [{k.anahtar}] {k.yer} · {k.teyit}{ek}")
    return "\n".join(satirlar)
