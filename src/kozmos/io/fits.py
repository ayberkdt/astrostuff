"""FITS dosyalarının birincil görüntü biriminin okunması ve yazılması.

Biçim kuralları FITS standardının 4.0 sürümünden (NASA FITS destek bürosunun
PDF belgesi) okunmuştur [FITS2018]: 2880 baytlık bloklar, 80 karakterlik
anahtar kayıtları, ``BITPIX`` yorumu (Tablo 8), ölçekleme (Denk. 3) ve büyük
uçlu bayt sırası (Bölüm 5).

Kapsam: birincil HDU ve ``IMAGE`` uzantılarındaki görüntü dizileri. Tablo
uzantıları listelenir ama içerikleri çözülmez; sıkıştırılmış görüntüler ve
``CONTINUE`` kayıtları okunmaz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._kaynak import kaynak

BLOK = 2880  # [FITS2018] "FITS block": 2880 sekiz bitlik bayt
KAYIT = 80  # anahtar kaydı uzunluğu
_BITPIX_TIPI = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}  # [FITS2018] Tablo 8

Baslik = dict[str, bool | int | float | str]


def _deger_coz(metin: str) -> bool | int | float | str:
    """Değer alanını (10. karakterden sonrası, yorum ayrılmış) Python değerine çevirir."""
    m = metin.strip()
    if m.startswith("'"):
        son = 1
        while True:  # dizgi içindeki '' tek tırnak demektir
            son = m.index("'", son)
            if m[son : son + 2] == "''":
                son += 2
                continue
            break
        return m[1:son].replace("''", "'").rstrip()
    m = m.split("/", 1)[0].strip()
    if m in ("T", "F"):
        return m == "T"
    try:
        return int(m)
    except ValueError:
        return float(m.replace("D", "E"))


@kaynak(
    "FITS2018",
    yer="Bölüm 3.3.1 ve 4.1: başlık blokları, anahtar kaydı bileşenleri; END anahtarı",
    teyit="okundu",
    not_="anahtar adı 1-8. baytlar; değer göstergesi '= ' 9-10. baytlar; başlık 2880 baytlık bloklarda, END ile biter",
)
def baslik_coz(ham: bytes) -> tuple[Baslik, int]:
    """Bir FITS başlığını anahtar-değer sözlüğüne çözer.

    Parametreler
    ------------
    ham : bytes
        Dosyanın başından itibaren baytlar (en az başlığın tamamı).

    Döndürür
    --------
    (başlık, veri_başlangıcı) : anahtar → değer sözlüğü ve veri biriminin bayt konumu.

    Kaynaklar
    ---------
    [FITS2018] Bölüm 3.3.1, 4.1.

    Geçerlilik
    ----------
    ``COMMENT``, ``HISTORY`` ve değer göstergesi taşımayan kayıtlar atlanır;
    ``CONTINUE`` ile uzatılmış dizgiler birleştirilmez. Aynı anahtar yinelenirse son değer kalır.

    Doğrulama
    ---------
    tests/birim/test_fits.py: astropy.io.fits ile başlık değerlerinin karşılaştırılması.
    """
    baslik: Baslik = {}
    konum = 0
    while True:
        if konum + KAYIT > len(ham):
            raise ValueError("END anahtarı bulunamadı; başlık eksik")
        kayit = ham[konum : konum + KAYIT].decode("ascii", errors="replace")
        konum += KAYIT
        anahtar = kayit[:8].rstrip()
        if anahtar == "END":
            break
        if kayit[8:10] == "= " and anahtar:
            baslik[anahtar] = _deger_coz(kayit[10:])
    veri_baslangici = -(-konum // BLOK) * BLOK
    return baslik, veri_baslangici


def _veri_bayt(baslik: Baslik) -> int:
    """Bir HDU'nun veri biriminin dolgusuz bayt sayısı ([FITS2018] Denk. 1 ve 2)."""
    eksen = int(baslik.get("NAXIS", 0))
    if eksen == 0:
        return 0
    eleman = 1
    for i in range(1, eksen + 1):
        eleman *= int(baslik[f"NAXIS{i}"])
    return abs(int(baslik["BITPIX"])) // 8 * int(baslik.get("GCOUNT", 1)) * (int(baslik.get("PCOUNT", 0)) + eleman)


@kaynak(
    "FITS2018",
    yer="Denk. 1 ve 2 (veri boyutu), Bölüm 3.3-3.4 (HDU sırası), XTENSION anahtarı",
    teyit="okundu",
    not_="N_bit = |BITPIX| × GCOUNT × (PCOUNT + NAXIS1 × ... × NAXISm); her HDU 2880 baytın katında başlar",
)
def hdu_listesi(yol: Path | str) -> list[tuple[str, Baslik, int]]:
    """Dosyadaki bütün HDU'ları sırayla listeler.

    Döndürür
    --------
    list[(tür, başlık, veri_başlangıcı)]
        Tür birincil HDU için ``"PRIMARY"``, uzantılar için ``XTENSION`` değeridir
        (``"IMAGE"``, ``"BINTABLE"``, ``"TABLE"``).

    Kaynaklar
    ---------
    [FITS2018] Denk. 1, 2; Bölüm 3.3-3.4.

    Geçerlilik
    ----------
    Dosyanın tamamı belleğe okunur. Rastgele gruplar (``GROUPS = T``) desteklenmez.

    Doğrulama
    ---------
    tests/birim/test_fits.py: astropy.io.fits ile aynı HDU sayısı, türleri ve boyutları.
    """
    ham = Path(yol).read_bytes()
    liste: list[tuple[str, Baslik, int]] = []
    konum = 0
    while konum < len(ham) and ham[konum : konum + KAYIT].strip():
        baslik, uzunluk = baslik_coz(ham[konum:])
        if not liste and baslik.get("SIMPLE") is not True:
            raise ValueError("SIMPLE = T değil; standart FITS dosyası değil")
        tur = "PRIMARY" if not liste else str(baslik.get("XTENSION", "?")).strip()
        veri_baslangici = konum + uzunluk
        liste.append((tur, baslik, veri_baslangici))
        konum = veri_baslangici + -(-_veri_bayt(baslik) // BLOK) * BLOK
    return liste


@kaynak(
    "FITS2018",
    yer="Tablo 8 (BITPIX), Denk. 3 (BZERO, BSCALE), Bölüm 5 (büyük uçlu bayt sırası), Şekil 1 (eksen sırası)",
    teyit="okundu",
    not_="fiziksel değer = BZERO + BSCALE × dizi değeri; 1. eksenin indisi en hızlı değişir, bu yüzden dizi "
    "(NAXISm, ..., NAXIS1) biçimindedir",
)
def fits_oku(yol: Path | str, hdu: int = 0) -> tuple[NDArray[np.float64], Baslik]:
    """FITS dosyasındaki bir görüntü dizisini ve başlığını okur.

    Parametreler
    ------------
    yol : Path
    hdu : int
        Okunacak HDU'nun sırası; 0 birincil HDU'dur. ``IMAGE`` uzantıları da okunur.

    Döndürür
    --------
    (veri, başlık) : veri ``float64`` dizisidir; biçimi ``(NAXISn, ..., NAXIS1)``, yani
    iki boyutlu görüntüde ``veri[y, x]``. Ölçekleme (``BSCALE``, ``BZERO``) uygulanmıştır.

    Kaynaklar
    ---------
    [FITS2018] Tablo 8, Denk. 3, Bölüm 5, Şekil 1.

    Geçerlilik
    ----------
    Yalnızca görüntü dizileri: veri taşımayan HDU (``NAXIS = 0``) ya da tablo
    uzantısı ``ValueError`` verir. ``BLANK`` ile işaretlenmiş tamsayı pikseller ``nan`` yapılır.

    Doğrulama
    ---------
    tests/birim/test_fits.py: astropy.io.fits.getdata ile tam eşitlik; yaz-oku gidiş-dönüşü; çok uzantılı dosya.
    """
    ham = Path(yol).read_bytes()
    liste = hdu_listesi(yol)
    if not 0 <= hdu < len(liste):
        raise ValueError(f"dosyada {len(liste)} HDU var; {hdu} yok")
    tur, baslik, baslangic = liste[hdu]
    if tur not in ("PRIMARY", "IMAGE"):
        raise ValueError(f"HDU {hdu} bir görüntü değil ({tur}); tablo uzantıları çözülmez")
    eksen = int(baslik.get("NAXIS", 0))
    if eksen == 0:
        raise ValueError(f"HDU {hdu} veri içermiyor (NAXIS = 0)")
    bicim = tuple(int(baslik[f"NAXIS{i}"]) for i in range(eksen, 0, -1))
    tip = np.dtype(_BITPIX_TIPI[int(baslik["BITPIX"])])
    n = int(np.prod(bicim))
    dizi = np.frombuffer(ham, dtype=tip, count=n, offset=baslangic).reshape(bicim)
    veri = dizi.astype(np.float64)
    if "BLANK" in baslik and tip.kind in "iu":
        veri[dizi == int(baslik["BLANK"])] = np.nan
    veri = float(baslik.get("BZERO", 0.0)) + float(baslik.get("BSCALE", 1.0)) * veri
    return veri, baslik


def _kayit(anahtar: str, deger: bool | int | float | str) -> str:
    if isinstance(deger, bool):
        alan = f"{'T' if deger else 'F':>20}"
    elif isinstance(deger, int):
        alan = f"{deger:>20d}"
    elif isinstance(deger, float):
        alan = f"{deger:>20.14G}"
    else:
        alan = "'" + f"{str(deger).replace(chr(39), chr(39) * 2):<8}" + "'"
    return f"{anahtar:<8}= {alan}".ljust(KAYIT)[:KAYIT]


@kaynak(
    "FITS2018",
    yer="Bölüm 3.3.1, 4.4.1 (zorunlu anahtarlar), Tablo 8, Bölüm 5",
    teyit="okundu",
    not_="SIMPLE, BITPIX, NAXIS, NAXISn sırayla; başlık boşlukla, veri sıfırla 2880 baytın katına doldurulur",
)
def fits_yaz(yol: Path | str, veri: ArrayLike, baslik: Baslik | None = None) -> None:
    """Bir diziyi birincil HDU olarak FITS dosyasına yazar.

    Parametreler
    ------------
    yol : Path
    veri : ndarray
        ``float64`` ya da ``float32`` olarak (BITPIX −64 / −32) yazılır; tamsayı diziler
        ``int16``/``int32`` olarak yazılır.
    baslik : dict, isteğe bağlı
        Ek anahtarlar (ad en çok 8 karakter, büyük harf). Zorunlu anahtarlar veriden üretilir.

    Kaynaklar
    ---------
    [FITS2018] Bölüm 3.3.1, 4.4.1, Tablo 8, Bölüm 5.

    Geçerlilik
    ----------
    68 karakterden uzun dizgi değerleri kesilir; yorum alanı yazılmaz.

    Doğrulama
    ---------
    tests/birim/test_fits.py: yazılan dosyanın astropy.io.fits ile okunması ve doğrulanması.
    """
    dizi = np.asarray(veri)
    bitpix = {"f": {4: -32, 8: -64}, "i": {2: 16, 4: 32, 8: 64}, "u": {1: 8}}.get(dizi.dtype.kind, {}).get(
        dizi.dtype.itemsize
    )
    if bitpix is None:
        dizi, bitpix = dizi.astype(np.float64), -64
    kayitlar = [_kayit("SIMPLE", True), _kayit("BITPIX", bitpix), _kayit("NAXIS", dizi.ndim)]
    kayitlar += [_kayit(f"NAXIS{i + 1}", n) for i, n in enumerate(dizi.shape[::-1])]
    zorunlu = {"SIMPLE", "BITPIX", "NAXIS", "END"} | {f"NAXIS{i + 1}" for i in range(dizi.ndim)}
    for anahtar, deger in (baslik or {}).items():
        if anahtar.upper() not in zorunlu:
            kayitlar.append(_kayit(anahtar.upper()[:8], deger))
    kayitlar.append("END".ljust(KAYIT))
    bas = "".join(kayitlar).encode("ascii")
    bas += b" " * (-len(bas) % BLOK)
    govde = dizi.astype(_BITPIX_TIPI[bitpix]).tobytes()
    govde += b"\x00" * (-len(govde) % BLOK)
    Path(yol).write_bytes(bas + govde)
