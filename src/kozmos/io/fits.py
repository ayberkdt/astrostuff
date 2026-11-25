"""Reading and writing the primary image unit of FITS files.

The format rules were read from version 4.0 of the FITS standard (the PDF
document of the NASA FITS support office) [FITS2018]: 2880-byte blocks,
80-character header cards, the ``BITPIX`` interpretation (Table 8), scaling
(Eq. 3) and big-endian byte order (Section 5).

Scope: image arrays in the primary HDU and in ``IMAGE`` extensions. Table
extensions are listed but their contents are not decoded; compressed images and
``CONTINUE`` cards are not read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source

BLOCK = 2880  # [FITS2018] "FITS block": 2880 eight-bit bytes
CARD = 80  # length of a header card
_BITPIX_DTYPE = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}  # [FITS2018] Table 8

Header = dict[str, bool | int | float | str]


def _parse_value(text: str) -> bool | int | float | str:
    """Convert a value field (after the 10th character, comment separated) to a Python value."""
    m = text.strip()
    if m.startswith("'"):
        end = 1
        while True:  # '' inside a string means a single quote
            end = m.index("'", end)
            if m[end : end + 2] == "''":
                end += 2
                continue
            break
        return m[1:end].replace("''", "'").rstrip()
    m = m.split("/", 1)[0].strip()
    if m in ("T", "F"):
        return m == "T"
    try:
        return int(m)
    except ValueError:
        return float(m.replace("D", "E"))


@source(
    "FITS2018",
    where="Section 3.3.1 and 4.1: header blocks, header card components; END keyword",
    status="read",
    note="keyword name is bytes 1-8; value indicator '= ' is bytes 9-10; the header is in 2880-byte blocks and ends with END",
)
@source(
    "Hierarch2009",
    where="FITS registered conventions page: 'The HIERARCH Keyword Convention', Description",
    status="read",
    note="bytes 10-80 of a HIERARCH card hold space-separated classification words, then '=' and a free-format "
    "value. The full document of the convention (hierarch.pdf) was not read; only the definition on the registry page was applied",
)
def parse_header(raw: bytes) -> tuple[Header, int]:
    """Parse a FITS header into a key-value dictionary.

    Parameters
    ----------
    raw : bytes
        Bytes from the start of the file (at least the whole header).

    Returns
    -------
    (header, data_start) : key → value dictionary and the byte position of the data unit.

    References
    ----------
    [FITS2018] Section 3.3.1, 4.1; [Hierarch2009] Description.

    Validity
    --------
    ESO ``HIERARCH`` cards are stored with a key whose words are joined by single spaces
    (``HIERARCH ESO DET DIT = 6.0`` → ``header["ESO DET DIT"]``). ``COMMENT``, ``HISTORY`` and
    cards without a value indicator are skipped;
    strings extended with ``CONTINUE`` are not joined. If the same key is repeated, the last value is kept.

    Verification
    ------------
    tests/unit/test_fits.py: comparison of header values with astropy.io.fits; HIERARCH cards.
    """
    header: Header = {}
    position = 0
    while True:
        if position + CARD > len(raw):
            raise ValueError("END keyword not found; header is incomplete")
        card = raw[position : position + CARD].decode("ascii", errors="replace")
        position += CARD
        key = card[:8].rstrip()
        if key == "END":
            break
        if card[8:10] == "= " and key:
            header[key] = _parse_value(card[10:])
        elif key == "HIERARCH" and "=" in card[9:]:
            name, _, value = card[9:].partition("=")
            header[" ".join(name.split())] = _parse_value(value)
    data_start = -(-position // BLOCK) * BLOCK
    return header, data_start


def _data_bytes(header: Header) -> int:
    """Number of bytes of the data unit of an HDU, without padding ([FITS2018] Eq. 1 and 2)."""
    axis = int(header.get("NAXIS", 0))
    if axis == 0:
        return 0
    elements = 1
    for i in range(1, axis + 1):
        elements *= int(header[f"NAXIS{i}"])
    return abs(int(header["BITPIX"])) // 8 * int(header.get("GCOUNT", 1)) * (int(header.get("PCOUNT", 0)) + elements)


@source(
    "FITS2018",
    where="Eq. 1 and 2 (data size), Section 3.3-3.4 (HDU order), XTENSION keyword",
    status="read",
    note="N_bit = |BITPIX| × GCOUNT × (PCOUNT + NAXIS1 × ... × NAXISm); each HDU starts at a multiple of 2880 bytes",
)
def list_hdus(path: Path | str) -> list[tuple[str, Header, int]]:
    """List all HDUs in the file, in order.

    Returns
    -------
    list[(kind, header, data_start)]
        The kind is ``"PRIMARY"`` for the primary HDU and the ``XTENSION`` value for extensions
        (``"IMAGE"``, ``"BINTABLE"``, ``"TABLE"``).

    References
    ----------
    [FITS2018] Eq. 1, 2; Section 3.3-3.4.

    Validity
    --------
    The whole file is read into memory. Random groups (``GROUPS = T``) are not supported.

    Verification
    ------------
    tests/unit/test_fits.py: same number, kinds and sizes of HDUs as astropy.io.fits.
    """
    raw = Path(path).read_bytes()
    hdus: list[tuple[str, Header, int]] = []
    position = 0
    while position < len(raw) and raw[position : position + CARD].strip():
        header, length = parse_header(raw[position:])
        if not hdus and header.get("SIMPLE") is not True:
            raise ValueError("SIMPLE is not T; not a standard FITS file")
        kind = "PRIMARY" if not hdus else str(header.get("XTENSION", "?")).strip()
        data_start = position + length
        hdus.append((kind, header, data_start))
        position = data_start + -(-_data_bytes(header) // BLOCK) * BLOCK
    return hdus


@source(
    "FITS2018",
    where="Table 8 (BITPIX), Eq. 3 (BZERO, BSCALE), Section 5 (big-endian byte order), Fig. 1 (axis order)",
    status="read",
    note="physical value = BZERO + BSCALE × array value; the index of axis 1 varies fastest, so the array has "
    "shape (NAXISm, ..., NAXIS1)",
)
def read_fits(path: Path | str, hdu: int = 0) -> tuple[NDArray[np.float64], Header]:
    """Read an image array and its header from a FITS file.

    Parameters
    ----------
    path : Path
    hdu : int
        Index of the HDU to read; 0 is the primary HDU. ``IMAGE`` extensions are also read.

    Returns
    -------
    (data, header) : data is a ``float64`` array of shape ``(NAXISn, ..., NAXIS1)``, that is
    ``data[y, x]`` for a two-dimensional image. Scaling (``BSCALE``, ``BZERO``) has been applied.

    References
    ----------
    [FITS2018] Table 8, Eq. 3, Section 5, Fig. 1.

    Validity
    --------
    Image arrays only: an HDU without data (``NAXIS = 0``) or a table
    extension raises ``ValueError``. Integer pixels marked with ``BLANK`` become ``nan``.

    Verification
    ------------
    tests/unit/test_fits.py: exact equality with astropy.io.fits.getdata; write-read round trip; multi-extension file.
    """
    raw = Path(path).read_bytes()
    hdus = list_hdus(path)
    if not 0 <= hdu < len(hdus):
        raise ValueError(f"the file has {len(hdus)} HDUs; there is no HDU {hdu}")
    kind, header, initial = hdus[hdu]
    if kind not in ("PRIMARY", "IMAGE"):
        raise ValueError(f"HDU {hdu} is not an image ({kind}); table extensions are not decoded")
    axis = int(header.get("NAXIS", 0))
    if axis == 0:
        raise ValueError(f"HDU {hdu} contains no data (NAXIS = 0)")
    shape = tuple(int(header[f"NAXIS{i}"]) for i in range(axis, 0, -1))
    dtype = np.dtype(_BITPIX_DTYPE[int(header["BITPIX"])])
    n = int(np.prod(shape))
    array = np.frombuffer(raw, dtype=dtype, count=n, offset=initial).reshape(shape)
    data = array.astype(np.float64)
    if "BLANK" in header and dtype.kind in "iu":
        data[array == int(header["BLANK"])] = np.nan
    data = float(header.get("BZERO", 0.0)) + float(header.get("BSCALE", 1.0)) * data
    return data, header


def _card(key: str, value: bool | int | float | str) -> str:
    if isinstance(value, bool):
        field = f"{'T' if value else 'F':>20}"
    elif isinstance(value, int):
        field = f"{value:>20d}"
    elif isinstance(value, float):
        field = f"{value:>20.14G}"
    else:
        field = "'" + f"{str(value).replace(chr(39), chr(39) * 2):<8}" + "'"
    return f"{key:<8}= {field}".ljust(CARD)[:CARD]


@source(
    "FITS2018",
    where="Section 3.3.1, 4.4.1 (mandatory keywords), Table 8, Section 5",
    status="read",
    note="SIMPLE, BITPIX, NAXIS, NAXISn in order; the header is padded with spaces and the data with zeros to a multiple of 2880 bytes",
)
def write_fits(path: Path | str, data: ArrayLike, header: Header | None = None) -> None:
    """Write an array to a FITS file as the primary HDU.

    Parameters
    ----------
    path : Path
    data : ndarray
        Written as ``float64`` or ``float32`` (BITPIX −64 / −32); integer arrays are
        written as ``int16``/``int32``.
    header : dict, optional
        Extra keywords (name at most 8 characters, upper case). The mandatory keywords are produced from the data.

    References
    ----------
    [FITS2018] Section 3.3.1, 4.4.1, Table 8, Section 5.

    Validity
    --------
    String values longer than 68 characters are truncated; the comment field is not written.

    Verification
    ------------
    tests/unit/test_fits.py: reading the written file with astropy.io.fits and checking it.
    """
    array = np.asarray(data)
    bitpix = {"f": {4: -32, 8: -64}, "i": {2: 16, 4: 32, 8: 64}, "u": {1: 8}}.get(array.dtype.kind, {}).get(
        array.dtype.itemsize
    )
    if bitpix is None:
        array, bitpix = array.astype(np.float64), -64
    cards = [_card("SIMPLE", True), _card("BITPIX", bitpix), _card("NAXIS", array.ndim)]
    cards += [_card(f"NAXIS{i + 1}", n) for i, n in enumerate(array.shape[::-1])]
    mandatory = {"SIMPLE", "BITPIX", "NAXIS", "END"} | {f"NAXIS{i + 1}" for i in range(array.ndim)}
    for key, value in (header or {}).items():
        if key.upper() not in mandatory:
            cards.append(_card(key.upper()[:8], value))
    cards.append("END".ljust(CARD))
    header_bytes = "".join(cards).encode("ascii")
    header_bytes += b" " * (-len(header_bytes) % BLOCK)
    body = array.astype(_BITPIX_DTYPE[bitpix]).tobytes()
    body += b"\x00" * (-len(body) % BLOCK)
    Path(path).write_bytes(header_bytes + body)
