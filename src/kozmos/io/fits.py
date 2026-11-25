"""Reading and writing the primary image unit of FITS files.

The format rules were read from version 4.0 of the FITS standard (the PDF
document of the NASA FITS support office) [FITS2018]: 2880-byte blocks,
80-character header cards, the ``BITPIX`` interpretation (Table 8), scaling
(Eq. 3) and big-endian byte order (Section 5).

Scope: image arrays in the primary HDU and in ``IMAGE`` extensions
(``read_fits``) and ``BINTABLE`` extensions (``read_table``, Section 7.3 of
[FITS2018]). ``TABLE`` (ASCII) extensions are listed but their contents are not
decoded; compressed images and ``CONTINUE`` cards are not read.
"""

from __future__ import annotations

import re
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
    extension raises ``ValueError`` (binary tables are read by ``read_table``). Integer pixels marked with ``BLANK`` become ``nan``.

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
        raise ValueError(f"HDU {hdu} is not an image ({kind}); binary tables are read by read_table")
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


_NUMERIC_DTYPE = {"B": ">u1", "I": ">i2", "J": ">i4", "K": ">i8", "E": ">f4", "D": ">f8", "C": ">c8", "M": ">c16"}
_ELEMENT_BYTES = {"L": 1, "A": 1, "B": 1, "I": 2, "J": 4, "K": 8, "E": 4, "D": 8, "C": 8, "M": 16}  # Table 18
_DESCRIPTOR = {"P": ">i4", "Q": ">i8"}  # two 32-bit or two 64-bit signed integers (Sect. 7.3.5)
# Table 19: code -> (TZERO, unsigned storage dtype, sign-bit mask, physical dtype)
_UNSIGNED_CONVENTION = {
    "B": (-128.0, np.uint8, 0x80, np.int8),
    "I": (2.0**15, np.uint16, 0x8000, np.uint16),
    "J": (2.0**31, np.uint32, 0x80000000, np.uint32),
    "K": (2.0**63, np.uint64, 1 << 63, np.uint64),
}
_TFORM = re.compile(r"^\s*(\d*)\s*([A-Z])(.*)$")
_TDIM = re.compile(r"^\s*\(\s*(\d+(?:\s*,\s*\d+)*)\s*\)\s*$")
_DATA_TYPES = frozenset(_ELEMENT_BYTES) | {"X"}


def _field_bytes(code: str, count: int) -> int:
    """Number of bytes of a field of `count` elements of data type `code` (Table 18)."""
    return -(-count // 8) if code == "X" else count * _ELEMENT_BYTES[code]


def _decode_strings(chars: NDArray[np.uint8]) -> NDArray[np.str_]:
    """Rows of ASCII bytes to strings: cut at the first NUL, drop trailing blanks (Sect. 7.3.3.1, Character)."""
    n, width = chars.shape
    if width == 0:
        return np.full(n, "", dtype="U1")
    chars = np.where(np.cumsum(chars == 0, axis=1) > 0, 0, chars).astype(np.uint8)
    text = np.ascontiguousarray(chars).view(f"S{width}").reshape(n)
    return np.char.rstrip(np.char.decode(text, "ascii"), " ")


def _convert_elements(
    buf: NDArray[np.uint8], code: str, count: int, tscal: float, tzero: float, tnull: int | None
) -> NDArray:
    """Convert big-endian field bytes (n rows by bytes) to values: shape (n, count), or (n,) for ``A``."""
    n = buf.shape[0]
    flat = np.ascontiguousarray(buf).reshape(-1)
    if code == "A":
        return _decode_strings(flat.reshape(n, count))
    if code == "L":
        return flat.reshape(n, count) == ord("T")  # 'T' true, 'F' false; a zero byte (null) is returned as False
    if code == "X":
        return np.unpackbits(flat.reshape(n, _field_bytes("X", count)), axis=1)[:, :count].astype(bool)
    stored = flat.view(_NUMERIC_DTYPE[code]).reshape(n, count)
    native = stored.astype(stored.dtype.newbyteorder("="))
    if code in _UNSIGNED_CONVENTION and tscal == 1.0 and tzero == _UNSIGNED_CONVENTION[code][0]:
        _, storage, mask, physical = _UNSIGNED_CONVENTION[code]
        return (native.view(storage) ^ storage(mask)).view(physical)  # adding TZERO = flipping the sign bit
    if tscal == 1.0 and tzero == 0.0:
        return native
    values = native.astype(np.complex128 if code in "CM" else np.float64)
    if tscal != 1.0:
        values = values * tscal
    if tzero != 0.0:
        values = values + tzero  # Eq. 7
    if tnull is not None and code in "BIJK":
        values[native == tnull] = np.nan
    return values


def _column_layout(header: Header) -> tuple[list[dict], int]:
    """Describe the fields of a binary table from TFORMn, TTYPEn, TSCALn, TZEROn, TNULLn, TDIMn; and the row width."""
    layout = []
    offset = 0
    for n in range(1, int(header["TFIELDS"]) + 1):
        tform = header.get(f"TFORM{n}")
        match = _TFORM.match(tform) if isinstance(tform, str) else None
        if match is None:
            raise ValueError(f"TFORM{n} is missing or malformed: {tform!r}")
        repeat = int(match.group(1)) if match.group(1) else 1
        code, rest = match.group(2), match.group(3)
        descriptor = code if code in _DESCRIPTOR else None
        if descriptor:
            if repeat not in (0, 1) or not rest or rest[0] not in _DATA_TYPES:
                raise ValueError(f"TFORM{n} = {tform!r}: P/Q needs a repeat count of 0 or 1 and an element type")
            element, nbytes = rest[0], repeat * (8 if code == "P" else 16)
        else:
            if code not in _DATA_TYPES:
                raise ValueError(f"TFORM{n} = {tform!r}: unknown data type {code!r}")
            if code == "A" and rest.strip():
                raise ValueError(f"TFORM{n} = {tform!r}: a width suffix on a character column is not supported")
            element, nbytes = code, _field_bytes(code, repeat)
        dims = None
        if f"TDIM{n}" in header and not descriptor:
            dim_match = _TDIM.match(str(header[f"TDIM{n}"]))
            if dim_match is None:
                raise ValueError(f"TDIM{n} is malformed: {header[f'TDIM{n}']!r}")
            dims = tuple(int(v) for v in dim_match.group(1).split(","))
            if int(np.prod(dims)) > repeat:
                raise ValueError(f"TDIM{n} = {dims} has more elements than the repeat count {repeat}")
        numeric = element in _NUMERIC_DTYPE
        tnull = header.get(f"TNULL{n}") if element in "BIJK" else None
        layout.append(
            {
                "name": str(header.get(f"TTYPE{n}", f"col{n}")).strip(),
                "repeat": repeat,
                "descriptor": descriptor,
                "element": element,
                "offset": offset,
                "nbytes": nbytes,
                "tscal": float(header.get(f"TSCAL{n}", 1.0)) if numeric else 1.0,
                "tzero": float(header.get(f"TZERO{n}", 0.0)) if numeric else 0.0,
                "tnull": None if tnull is None else int(tnull),
                "dims": dims,
            }
        )
        offset += nbytes
    return layout, offset


@source(
    "FITS2018",
    where="Section 7.3.1 (mandatory keywords, Eq. 8), 7.3.2 (TTYPEn, TSCALn, TZEROn, TNULLn, TDIMn, Table 19; "
    "Eq. 7 is printed in Section 7.2.2), 7.3.3 (Table 18, data types, main table), 7.3.5 (variable-length arrays, THEAP)",
    status="read",
    note="physical = TZEROn + TSCALn x field (Eq. 7); the first TDIMn dimension varies fastest so the array shape is "
    "reversed; P and Q descriptors are (element count, byte offset from the heap start). This project's choices: "
    "a null logical (zero byte) is returned as False; TNULLn becomes NaN only where the output is floating point; "
    "TDIMn is not applied to variable-length columns; TSCALn/TZEROn on A, L, X columns are ignored; "
    "an 'rAw' width suffix on a character column is rejected (the standard leaves the extra characters undefined)",
)
def read_table(path: Path | str, hdu: int = 1) -> tuple[dict[str, NDArray], Header]:
    """Read a binary-table extension into a dictionary of column arrays.

    physical value = TZEROn + TSCALn x field value, with the unsigned-integer convention kept exact.

    Parameters
    ----------
    path : Path
    hdu : int
        Index of the HDU to read (0 is the primary HDU; the first extension is 1). It must be a ``BINTABLE``.

    Returns
    -------
    (columns, header) : ``columns`` maps ``TTYPEn`` (``col<n>`` if absent) to an array with one entry per row.
    Fixed-length columns give shape ``(rows,)`` for a repeat count of 1 (always ``(rows, bits)`` for ``X``) and
    ``(rows, r)`` otherwise; ``TDIMn = (l, m, n)`` gives ``(rows, n, m, l)``. Types: ``L`` and ``X`` give ``bool`` (one entry per bit for ``X``),
    ``B`` gives ``uint8``, ``I`` ``int16``, ``J`` ``int32``, ``K`` ``int64``, ``E`` ``float32``, ``D`` ``float64``,
    ``C`` ``complex64``, ``M`` ``complex128`` (native byte order); ``A`` gives ``str`` (trailing blanks removed; the
    text ends at the first NUL; a ``TDIMn`` whose first dimension is the string length gives arrays of strings). A
    column with ``TSCALn``/``TZEROn`` other than 1 and 0 is returned as ``float64`` (``complex128``) with ``TNULLn``
    entries set to NaN. ``TSCALn = 1`` with ``TZEROn`` of 32768, 2**31, 2**63 (types I, J, K) or -128 (type B) gives
    exact ``uint16``, ``uint32``, ``uint64`` or ``int8``. ``P``/``Q`` columns give an object array holding one array
    (or one ``str`` for element type ``A``) per row, read from the heap.

    References
    ----------
    [FITS2018] Section 7.3.1-7.3.3, 7.3.5, Tables 17-19, Eq. 7 and 8.

    Validity
    --------
    Supported: ``BINTABLE`` with the data types L, X, B, I, J, K, A, E, D, C, M, P and Q, scaling, nulls, TDIMn,
    the heap with ``THEAP``. Not supported: ASCII tables (``XTENSION = 'TABLE'``, ``ValueError``); decompression of
    tile-compressed images stored in a binary table (the tile table itself is read as it is, with its
    ``COMPRESSED_DATA`` descriptors); random groups; ``TDIMn`` of variable-length columns (arrays stay flat);
    duplicate column names (``ValueError``). ``TNULLn`` is not applied to unscaled or unsigned integer columns (an
    integer cannot hold NaN; read ``header["TNULLn"]``); a null logical (zero byte) becomes ``False``; ``TSCALn`` and
    ``TZEROn`` on A, L and X columns are ignored. When ``TDIMn`` is smaller than the repeat count the trailing
    elements are dropped. The whole file is read into memory; measured on this machine, 10**6 rows of three fixed columns take about 1 s and
    2 x 10**4 variable-length rows about 0.2 s (one Python loop iteration per variable-length row). A row width that differs from the sum of Eq. 8, a
    descriptor outside the heap, or a truncated file raises ``ValueError``.

    Verification
    ------------
    tests/unit/test_fits.py: column-by-column exact equality with astropy.io.fits for every data type, vector and
    TDIM columns, scaled, unsigned and null columns, strings, variable-length arrays, a table after image
    extensions, a zero-row table, error paths.
    """
    raw = Path(path).read_bytes()
    hdus = list_hdus(path)
    if not 0 <= hdu < len(hdus):
        raise ValueError(f"the file has {len(hdus)} HDUs; there is no HDU {hdu}")
    kind, header, start = hdus[hdu]
    if kind == "TABLE":
        raise ValueError(f"HDU {hdu} is an ASCII table (XTENSION = 'TABLE'); only binary tables are supported")
    if kind != "BINTABLE":
        raise ValueError(f"HDU {hdu} is not a binary table ({kind}); read images with read_fits")
    if header.get("BITPIX") != 8 or header.get("NAXIS") != 2 or header.get("GCOUNT", 1) != 1:
        raise ValueError("mandatory keywords of a binary table are violated (BITPIX = 8, NAXIS = 2, GCOUNT = 1)")
    row_bytes, rows, pcount = int(header["NAXIS1"]), int(header["NAXIS2"]), int(header.get("PCOUNT", 0))
    main_bytes = row_bytes * rows
    layout, width = _column_layout(header)
    if width != row_bytes:
        raise ValueError(f"the fields add up to {width} bytes per row but NAXIS1 = {row_bytes} (Eq. 8)")
    names = [field["name"] for field in layout]
    if len(set(names)) != len(names):
        raise ValueError("duplicate column names")
    if start + main_bytes + pcount > len(raw):
        raise ValueError("the file is truncated: the data unit extends past its end")
    if main_bytes:
        table = np.frombuffer(raw, dtype=np.uint8, count=main_bytes, offset=start).reshape(rows, row_bytes)
    else:
        table = np.zeros((rows, row_bytes), dtype=np.uint8)
    theap = int(header.get("THEAP", main_bytes))
    if not main_bytes <= theap <= main_bytes + pcount:
        raise ValueError(f"THEAP = {theap} lies outside the data unit")
    heap_start, heap_bytes = start + theap, main_bytes + pcount - theap

    columns: dict[str, NDArray] = {}
    for field in layout:
        element, repeat, dims = field["element"], field["repeat"], field["dims"]
        block = table[:, field["offset"] : field["offset"] + field["nbytes"]]
        scaling = (field["tscal"], field["tzero"], field["tnull"])
        if field["descriptor"] is None:
            if element == "A":
                size, strings = (dims[0], int(np.prod(dims[1:]))) if dims else (repeat, 1)
                values = _convert_elements(
                    block[:, : size * strings].reshape(rows * strings, size), "A", size, 1, 0, None
                )
                shape = (rows, *dims[:0:-1]) if dims else (rows,)
            else:
                count = int(np.prod(dims)) if dims else repeat
                values = _convert_elements(block[:, : _field_bytes(element, count)], element, count, *scaling)
                shape = (rows, *dims[::-1]) if dims else ((rows,) if repeat == 1 and element != "X" else (rows, repeat))
            columns[field["name"]] = values.reshape(shape)
            continue
        descriptors = np.zeros((rows, 2), dtype=np.int64)
        if repeat:
            descriptors = np.ascontiguousarray(block).reshape(-1).view(_DESCRIPTOR[field["descriptor"]])
            descriptors = descriptors.reshape(rows, 2).astype(np.int64)
        arrays = np.empty(rows, dtype=object)
        for i, (length, heap_offset) in enumerate(descriptors):
            size = _field_bytes(element, int(length))
            if length < 0 or heap_offset < 0 or (length and heap_offset + size > heap_bytes):
                raise ValueError(
                    f"column {field['name']!r}, row {i}: descriptor ({length}, {heap_offset}) is outside the heap"
                )
            if size:
                data = np.frombuffer(raw, dtype=np.uint8, count=size, offset=heap_start + int(heap_offset))
            else:
                data = np.zeros(0, dtype=np.uint8)
            values = _convert_elements(data.reshape(1, size), element, int(length), *scaling)
            arrays[i] = str(values[0]) if element == "A" else values[0]
        columns[field["name"]] = arrays
    return columns, header


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
