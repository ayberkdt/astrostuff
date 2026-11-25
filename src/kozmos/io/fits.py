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

from kozmos._sources import source

BLOCK = 2880  # [FITS2018] "FITS block": 2880 sekiz bitlik bayt
CARD = 80  # anahtar kaydı uzunluğu
_BITPIX_DTYPE = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}  # [FITS2018] Tablo 8

Header = dict[str, bool | int | float | str]


def _parse_value(text: str) -> bool | int | float | str:
    """Değer alanını (10. karakterden sonrası, yorum ayrılmış) Python değerine çevirir."""
    m = text.strip()
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


@source(
    "FITS2018",
    where="Bölüm 3.3.1 ve 4.1: başlık blokları, anahtar kaydı bileşenleri; END anahtarı",
    status="read",
    note="anahtar adı 1-8. baytlar; değer göstergesi '= ' 9-10. baytlar; başlık 2880 baytlık bloklarda, END ile biter",
)
@source(
    "Hierarch2009",
    where="FITS kayıtlı kurallar sayfası: 'The HIERARCH Keyword Convention', Description",
    status="read",
    note="HIERARCH kaydının 10-80. baytları boşlukla ayrılmış sınıflandırma sözcükleri, ardından '=' ve serbest "
    "biçimli değer taşır. Kuralın tam belgesi (hierarch.pdf) okunmadı; yalnızca kayıt sayfasındaki tanım uygulandı",
)
def parse_header(raw: bytes) -> tuple[Header, int]:
    """Bir FITS başlığını anahtar-değer sözlüğüne çözer.

    Parameters
    ----------
    ham : bytes
        Dosyanın başından itibaren baytlar (en az başlığın tamamı).

    Returns
    -------
    (başlık, veri_başlangıcı) : anahtar → değer sözlüğü ve veri biriminin bayt konumu.

    References
    ----------
    [FITS2018] Bölüm 3.3.1, 4.1; [Hierarch2009] Description.

    Validity
    --------
    ESO'nun ``HIERARCH`` kayıtları, sözcükleri tek boşlukla birleştirilmiş anahtarla saklanır
    (``HIERARCH ESO DET DIT = 6.0`` → ``baslik["ESO DET DIT"]``). ``COMMENT``, ``HISTORY`` ve
    değer göstergesi taşımayan kayıtlar atlanır;
    ``CONTINUE`` ile uzatılmış dizgiler birleştirilmez. Aynı anahtar yinelenirse son değer kalır.

    Verification
    ------------
    tests/unit/test_fits.py: astropy.io.fits ile başlık değerlerinin karşılaştırılması; HIERARCH kartları.
    """
    header: Header = {}
    konum = 0
    while True:
        if konum + CARD > len(raw):
            raise ValueError("END anahtarı bulunamadı; başlık eksik")
        kayit = raw[konum : konum + CARD].decode("ascii", errors="replace")
        konum += CARD
        key = kayit[:8].rstrip()
        if key == "END":
            break
        if kayit[8:10] == "= " and key:
            header[key] = _parse_value(kayit[10:])
        elif key == "HIERARCH" and "=" in kayit[9:]:
            name, _, value = kayit[9:].partition("=")
            header[" ".join(name.split())] = _parse_value(value)
    veri_baslangici = -(-konum // BLOCK) * BLOCK
    return header, veri_baslangici


def _data_bytes(header: Header) -> int:
    """Bir HDU'nun veri biriminin dolgusuz bayt sayısı ([FITS2018] Denk. 1 ve 2)."""
    axis = int(header.get("NAXIS", 0))
    if axis == 0:
        return 0
    eleman = 1
    for i in range(1, axis + 1):
        eleman *= int(header[f"NAXIS{i}"])
    return abs(int(header["BITPIX"])) // 8 * int(header.get("GCOUNT", 1)) * (int(header.get("PCOUNT", 0)) + eleman)


@source(
    "FITS2018",
    where="Denk. 1 ve 2 (veri boyutu), Bölüm 3.3-3.4 (HDU sırası), XTENSION anahtarı",
    status="read",
    note="N_bit = |BITPIX| × GCOUNT × (PCOUNT + NAXIS1 × ... × NAXISm); her HDU 2880 baytın katında başlar",
)
def list_hdus(path: Path | str) -> list[tuple[str, Header, int]]:
    """Dosyadaki bütün HDU'ları sırayla listeler.

    Returns
    -------
    list[(tür, başlık, veri_başlangıcı)]
        Tür birincil HDU için ``"PRIMARY"``, uzantılar için ``XTENSION`` değeridir
        (``"IMAGE"``, ``"BINTABLE"``, ``"TABLE"``).

    References
    ----------
    [FITS2018] Denk. 1, 2; Bölüm 3.3-3.4.

    Validity
    --------
    Dosyanın tamamı belleğe okunur. Rastgele gruplar (``GROUPS = T``) desteklenmez.

    Verification
    ------------
    tests/unit/test_fits.py: astropy.io.fits ile aynı HDU sayısı, türleri ve boyutları.
    """
    raw = Path(path).read_bytes()
    liste: list[tuple[str, Header, int]] = []
    konum = 0
    while konum < len(raw) and raw[konum : konum + CARD].strip():
        header, uzunluk = parse_header(raw[konum:])
        if not liste and header.get("SIMPLE") is not True:
            raise ValueError("SIMPLE = T değil; standart FITS dosyası değil")
        tur = "PRIMARY" if not liste else str(header.get("XTENSION", "?")).strip()
        veri_baslangici = konum + uzunluk
        liste.append((tur, header, veri_baslangici))
        konum = veri_baslangici + -(-_data_bytes(header) // BLOCK) * BLOCK
    return liste


@source(
    "FITS2018",
    where="Tablo 8 (BITPIX), Denk. 3 (BZERO, BSCALE), Bölüm 5 (büyük uçlu bayt sırası), Şekil 1 (eksen sırası)",
    status="read",
    note="fiziksel değer = BZERO + BSCALE × dizi değeri; 1. eksenin indisi en hızlı değişir, bu yüzden dizi "
    "(NAXISm, ..., NAXIS1) biçimindedir",
)
def read_fits(path: Path | str, hdu: int = 0) -> tuple[NDArray[np.float64], Header]:
    """FITS dosyasındaki bir görüntü dizisini ve başlığını okur.

    Parameters
    ----------
    yol : Path
    hdu : int
        Okunacak HDU'nun sırası; 0 birincil HDU'dur. ``IMAGE`` uzantıları da okunur.

    Returns
    -------
    (veri, başlık) : veri ``float64`` dizisidir; biçimi ``(NAXISn, ..., NAXIS1)``, yani
    iki boyutlu görüntüde ``veri[y, x]``. Ölçekleme (``BSCALE``, ``BZERO``) uygulanmıştır.

    References
    ----------
    [FITS2018] Tablo 8, Denk. 3, Bölüm 5, Şekil 1.

    Validity
    --------
    Yalnızca görüntü dizileri: veri taşımayan HDU (``NAXIS = 0``) ya da tablo
    uzantısı ``ValueError`` verir. ``BLANK`` ile işaretlenmiş tamsayı pikseller ``nan`` yapılır.

    Verification
    ------------
    tests/unit/test_fits.py: astropy.io.fits.getdata ile tam eşitlik; yaz-oku gidiş-dönüşü; çok uzantılı dosya.
    """
    raw = Path(path).read_bytes()
    liste = list_hdus(path)
    if not 0 <= hdu < len(liste):
        raise ValueError(f"dosyada {len(liste)} HDU var; {hdu} yok")
    tur, header, initial = liste[hdu]
    if tur not in ("PRIMARY", "IMAGE"):
        raise ValueError(f"HDU {hdu} bir görüntü değil ({tur}); tablo uzantıları çözülmez")
    axis = int(header.get("NAXIS", 0))
    if axis == 0:
        raise ValueError(f"HDU {hdu} veri içermiyor (NAXIS = 0)")
    shape = tuple(int(header[f"NAXIS{i}"]) for i in range(axis, 0, -1))
    tip = np.dtype(_BITPIX_DTYPE[int(header["BITPIX"])])
    n = int(np.prod(shape))
    dizi = np.frombuffer(raw, dtype=tip, count=n, offset=initial).reshape(shape)
    data = dizi.astype(np.float64)
    if "BLANK" in header and tip.kind in "iu":
        data[dizi == int(header["BLANK"])] = np.nan
    data = float(header.get("BZERO", 0.0)) + float(header.get("BSCALE", 1.0)) * data
    return data, header


def _card(key: str, value: bool | int | float | str) -> str:
    if isinstance(value, bool):
        alan = f"{'T' if value else 'F':>20}"
    elif isinstance(value, int):
        alan = f"{value:>20d}"
    elif isinstance(value, float):
        alan = f"{value:>20.14G}"
    else:
        alan = "'" + f"{str(value).replace(chr(39), chr(39) * 2):<8}" + "'"
    return f"{key:<8}= {alan}".ljust(CARD)[:CARD]


@source(
    "FITS2018",
    where="Bölüm 3.3.1, 4.4.1 (zorunlu anahtarlar), Tablo 8, Bölüm 5",
    status="read",
    note="SIMPLE, BITPIX, NAXIS, NAXISn sırayla; başlık boşlukla, veri sıfırla 2880 baytın katına doldurulur",
)
def write_fits(path: Path | str, data: ArrayLike, header: Header | None = None) -> None:
    """Bir diziyi birincil HDU olarak FITS dosyasına yazar.

    Parameters
    ----------
    yol : Path
    veri : ndarray
        ``float64`` ya da ``float32`` olarak (BITPIX −64 / −32) yazılır; tamsayı diziler
        ``int16``/``int32`` olarak yazılır.
    baslik : dict, isteğe bağlı
        Ek anahtarlar (ad en çok 8 karakter, büyük harf). Zorunlu anahtarlar veriden üretilir.

    References
    ----------
    [FITS2018] Bölüm 3.3.1, 4.4.1, Tablo 8, Bölüm 5.

    Validity
    --------
    68 karakterden uzun dizgi değerleri kesilir; yorum alanı yazılmaz.

    Verification
    ------------
    tests/unit/test_fits.py: yazılan dosyanın astropy.io.fits ile okunması ve doğrulanması.
    """
    dizi = np.asarray(data)
    bitpix = {"f": {4: -32, 8: -64}, "i": {2: 16, 4: 32, 8: 64}, "u": {1: 8}}.get(dizi.dtype.kind, {}).get(
        dizi.dtype.itemsize
    )
    if bitpix is None:
        dizi, bitpix = dizi.astype(np.float64), -64
    kayitlar = [_card("SIMPLE", True), _card("BITPIX", bitpix), _card("NAXIS", dizi.ndim)]
    kayitlar += [_card(f"NAXIS{i + 1}", n) for i, n in enumerate(dizi.shape[::-1])]
    zorunlu = {"SIMPLE", "BITPIX", "NAXIS", "END"} | {f"NAXIS{i + 1}" for i in range(dizi.ndim)}
    for key, value in (header or {}).items():
        if key.upper() not in zorunlu:
            kayitlar.append(_card(key.upper()[:8], value))
    kayitlar.append("END".ljust(CARD))
    bas = "".join(kayitlar).encode("ascii")
    bas += b" " * (-len(bas) % BLOCK)
    govde = dizi.astype(_BITPIX_DTYPE[bitpix]).tobytes()
    govde += b"\x00" * (-len(govde) % BLOCK)
    Path(path).write_bytes(bas + govde)
