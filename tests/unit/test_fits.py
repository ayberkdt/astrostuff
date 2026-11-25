"""Comparison of the FITS reader and writer with astropy.io.fits."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits as astropy_fits

from kozmos.io import fits


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int16, np.int32, np.uint8])
def test_astropy_reads_what_we_write(tmp_path, dtype):
    rng = np.random.default_rng(91)
    data = (rng.normal(100, 30, (37, 53))).astype(dtype)
    path = tmp_path / "sample.fits"
    fits.write_fits(path, data, {"OBJECT": "test field", "EXPTIME": 12.5, "NCOMBINE": 3, "FLAG": True})
    with astropy_fits.open(path) as hdul:
        hdul.verify("exception")  # raises an error if the standard is violated
        assert np.array_equal(hdul[0].data, data)
        primary_header = hdul[0].header
        assert (
            primary_header["OBJECT"] == "test field"
            and primary_header["EXPTIME"] == 12.5
            and primary_header["NCOMBINE"] == 3
            and primary_header["FLAG"] is True
        )
    assert path.stat().st_size % fits.BLOCK == 0


def test_we_read_what_astropy_writes(tmp_path):
    rng = np.random.default_rng(92)
    data = rng.normal(size=(20, 31)).astype(">f4")
    astropy_header = astropy_fits.Header()
    astropy_header["TELESCOP"] = "T'est"  # single quote inside a string
    astropy_header["AIRMASS"] = (1.23, "comment / with slash")
    astropy_header["HISTORY"] = "card that must be skipped"
    path = tmp_path / "a.fits"
    astropy_fits.writeto(path, data, astropy_header)
    read_data, header = fits.read_fits(path)
    assert np.array_equal(read_data, data.astype(float)) and read_data.shape == (20, 31)
    assert header["TELESCOP"] == "T'est" and header["AIRMASS"] == 1.23 and header["NAXIS1"] == 31
    assert "HISTORY" not in header


def test_scaling_and_unsigned_integer(tmp_path):
    # Unsigned 16 bit: BITPIX = 16, BZERO = 32768 ([FITS2018] BZERO description).
    data = np.array([[0, 1, 65535], [32768, 40000, 123]], dtype=np.uint16)
    path = tmp_path / "u16.fits"
    astropy_fits.writeto(path, data)
    read_data, header = fits.read_fits(path)
    assert header["BITPIX"] == 16 and header["BZERO"] == 32768
    assert np.array_equal(read_data, data.astype(float))


def test_round_trip_and_errors(tmp_path):
    data = np.arange(24, dtype=float).reshape(2, 3, 4)
    path = tmp_path / "cube.fits"
    fits.write_fits(path, data)
    read_data, header = fits.read_fits(path)
    assert np.array_equal(read_data, data) and (header["NAXIS1"], header["NAXIS2"], header["NAXIS3"]) == (4, 3, 2)
    with pytest.raises(ValueError):
        fits.parse_header(b"SIMPLE  =                    T".ljust(80))  # no END
    empty = tmp_path / "empty.fits"
    astropy_fits.PrimaryHDU().writeto(empty)
    with pytest.raises(ValueError):
        fits.read_fits(empty)
    with pytest.raises(ValueError):
        fits.read_fits(empty, hdu=3)


def test_multi_extension_file(tmp_path):
    # Primary HDU without data + two image extensions + a binary table + one more image (HST/JWST style).
    rng = np.random.default_rng(93)
    science = rng.normal(size=(17, 23)).astype(np.float32)
    error = np.abs(rng.normal(size=(17, 23))).astype(np.float64)
    quality = rng.integers(0, 4, (17, 23)).astype(np.int16)
    table = astropy_fits.BinTableHDU.from_columns(
        [astropy_fits.Column(name="x", format="E", array=np.arange(5, dtype=np.float32))], name="TAB"
    )
    hdul = astropy_fits.HDUList(
        [
            astropy_fits.PrimaryHDU(),
            astropy_fits.ImageHDU(science, name="SCI"),
            astropy_fits.ImageHDU(error, name="ERR"),
            table,
            astropy_fits.ImageHDU(quality, name="DQ"),
        ]
    )
    path = tmp_path / "mef.fits"
    hdul.writeto(path)
    hdus = fits.list_hdus(path)
    assert [kind for kind, _, _ in hdus] == ["PRIMARY", "IMAGE", "IMAGE", "BINTABLE", "IMAGE"]
    assert [b.get("EXTNAME") for _, b, _ in hdus[1:]] == ["SCI", "ERR", "TAB", "DQ"]
    for i, expected in ((1, science), (2, error), (4, quality)):  # the extension AFTER the table must also be found
        data, header = fits.read_fits(path, hdu=i)
        assert np.array_equal(data, expected.astype(float)) and header["XTENSION"] == "IMAGE"
    with pytest.raises(ValueError):
        fits.read_fits(path, hdu=3)  # table
    with pytest.raises(ValueError):
        fits.read_fits(path, hdu=0)  # no data


def test_hierarch_cards(tmp_path):
    # ESO HIERARCH convention: astropy writes keywords longer than 8 characters or containing spaces in this form.
    hdu = astropy_fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32))
    hdu.header["HIERARCH ESO DET DIT"] = (6.0, "Integration Time")
    hdu.header["HIERARCH ESO DET NDIT"] = 15
    hdu.header["HIERARCH ESO INS FILT1 NAME"] = "Ks"
    hdu.header["HIERARCH ESO SEQ CUMOFFSETX"] = -208.333333
    hdu.header["HIERARCH ESO TEL FOCU LEN"] = True
    path = tmp_path / "h.fits"
    hdu.writeto(path)
    _, header = fits.read_fits(path)
    oracle = astropy_fits.getheader(path)
    for key in ("ESO DET DIT", "ESO DET NDIT", "ESO INS FILT1 NAME", "ESO SEQ CUMOFFSETX", "ESO TEL FOCU LEN"):
        assert header[key] == oracle[key] and type(header[key]) is type(oracle[key])
    assert header["NAXIS1"] == 4  # ordinary cards are not affected


# ---------------------------------------------------------------------------------------------------------------------
# read_table: binary-table extensions ([FITS2018] Section 7.3). astropy.io.fits writes most files and is the oracle;
# a few files are assembled byte by byte so that the expected values are known without astropy.
# ---------------------------------------------------------------------------------------------------------------------

Column = astropy_fits.Column


def _write_tables(path, columns, patches=None, before=(), name="TAB"):
    """Write [primary, *before, table] with astropy; `patches` edit the table header afterwards (TSCALn, TZEROn, ...).

    The stored integers stay unchanged, so scaling is applied only by the reader under test and by the oracle.
    Returns the index of the table HDU.
    """
    table = astropy_fits.BinTableHDU.from_columns(columns, name=name)
    astropy_fits.HDUList([astropy_fits.PrimaryHDU(), *before, table]).writeto(path, overwrite=True)
    if patches:
        with astropy_fits.open(path, mode="update") as hdul:
            for key, value in patches.items():
                hdul[1 + len(before)].header[key] = value
    return 1 + len(before)


def _oracle(path, ext):
    """Columns as astropy reads them, by position (a variable-length column becomes a list of arrays)."""
    out = []
    with astropy_fits.open(path) as hdul:
        data = hdul[ext].data
        for i in range(len(hdul[ext].columns)):
            field = data.field(i)
            out.append([np.array(v) for v in field] if field.dtype == object else np.array(field))
    return out


def _same(mine, oracle):
    """Exact equality; floating-point and complex columns compare bit by bit (NaN included)."""
    if isinstance(oracle, list):  # variable-length column
        assert mine.dtype == object and len(mine) == len(oracle)
        for a, b in zip(mine, oracle, strict=True):
            assert a.dtype.isnative
            _same(a, b)
        return
    assert mine.shape == oracle.shape, (mine.shape, oracle.shape)
    if mine.dtype.kind in "fc":
        assert mine.tobytes() == oracle.astype(mine.dtype).tobytes()
    else:
        assert np.array_equal(mine, oracle)


def _compare(path, ext):
    mine, header = fits.read_table(path, ext)
    oracle = _oracle(path, ext)
    assert len(mine) == len(oracle) == header["TFIELDS"]
    for values, expected in zip(mine.values(), oracle, strict=True):
        _same(values, expected)
    return mine, header


def _all_columns(n=7):
    rng = np.random.default_rng(401)
    text = ["a", "bc", "", "12345678", " lead", "tail  ", "x y"][:n]
    return [
        Column(name="flag", format="L", array=rng.random(n) > 0.5),
        Column(name="flags", format="3L", array=rng.random((n, 3)) > 0.5),
        Column(name="bit1", format="1X", array=rng.random((n, 1)) > 0.5),
        Column(name="bits", format="11X", array=rng.random((n, 11)) > 0.5),
        Column(name="byte", format="B", array=rng.integers(0, 256, n).astype(np.uint8)),
        Column(name="short", format="I", array=rng.integers(-32768, 32768, n).astype(np.int16)),
        Column(name="int", format="J", array=rng.integers(-(2**31), 2**31, n).astype(np.int32)),
        Column(name="long", format="K", array=rng.integers(-(2**62), 2**62, n).astype(np.int64) * 2),
        Column(name="single", format="E", array=rng.normal(size=n).astype(np.float32)),
        Column(name="double", format="D", array=rng.normal(size=n) * 1e300),
        Column(name="text", format="8A", array=np.array(text)),
        Column(name="csingle", format="C", array=(rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)),
        Column(name="cdouble", format="2M", array=rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))),
        Column(name="vec", format="5J", array=rng.integers(-9, 9, (n, 5)).astype(np.int32)),
    ]


def _raw_bintable(columns, rows, heap=b"", extra=None, naxis1=None):
    """Bytes of a file with an empty primary HDU and one BINTABLE assembled by hand (no astropy).

    `columns` is a list of (name, TFORM or None, bytes of that field for all rows in row order).
    `extra` adds header cards; a GCOUNT entry replaces the default 1.
    """
    widths = [len(data) // rows if rows else 0 for _, _, data in columns]
    body = b"".join(
        b"".join(data[row * width : (row + 1) * width] for (_, _, data), width in zip(columns, widths, strict=True))
        for row in range(rows)
    )

    def card(key, value):
        if isinstance(value, str):
            text = f"{key:<8}= '{value:<8}'"
        elif isinstance(value, bool):
            text = f"{key:<8}= {'T' if value else 'F':>20}"
        else:
            text = f"{key:<8}= {value:>20}"
        return text.ljust(80)

    extra = dict(extra or {})
    primary = card("SIMPLE", True) + card("BITPIX", 8) + card("NAXIS", 0) + "END".ljust(80)
    cards = [
        card("XTENSION", "BINTABLE"),
        card("BITPIX", 8),
        card("NAXIS", 2),
        card("NAXIS1", sum(widths) if naxis1 is None else naxis1),
        card("NAXIS2", rows),
        card("PCOUNT", len(heap)),
        card("GCOUNT", extra.pop("GCOUNT", 1)),
        card("TFIELDS", len(columns)),
    ]
    for i, (name, tform, _) in enumerate(columns, start=1):
        cards.append(card(f"TTYPE{i}", name))
        if tform is not None:
            cards.append(card(f"TFORM{i}", tform))
    cards += [card(key, value) for key, value in extra.items()]
    cards.append("END".ljust(80))
    header = "".join(cards).encode("ascii")
    header = header.ljust(-(-len(header) // fits.BLOCK) * fits.BLOCK, b" ")
    data = (body + heap).ljust(-(-len(body + heap) // fits.BLOCK) * fits.BLOCK, b"\x00")
    return primary.encode("ascii").ljust(fits.BLOCK, b" ") + header + data


def _descriptors(*pairs, dtype=">i4"):
    return b"".join(np.array(pair, dtype=dtype).tobytes() for pair in pairs)


def test_every_fixed_type_matches_astropy(tmp_path):
    path = tmp_path / "all.fits"
    columns = _all_columns()
    ext = _write_tables(path, columns)
    mine, header = _compare(path, ext)
    assert list(mine) == [c.name for c in columns]
    kinds = ["bool", "bool", "bool", "bool", "uint8", "int16", "int32", "int64", "float32", "float64"]
    kinds += ["str", "complex64", "complex128", "int32"]
    assert [("str" if v.dtype.kind == "U" else str(v.dtype)) for v in mine.values()] == kinds
    assert mine["bits"].shape == (7, 11) and mine["bit1"].shape == (7, 1) and mine["flag"].shape == (7,)
    assert mine["vec"].shape == (7, 5) and mine["cdouble"].shape == (7, 2)
    assert all(v.dtype.isnative for v in mine.values())
    assert header["NAXIS2"] == 7 and header["EXTNAME"] == "TAB"
    assert mine["text"].tolist() == ["a", "bc", "", "12345678", " lead", "tail", "x y"]  # trailing blanks only


def test_float_special_values_are_bit_exact(tmp_path):
    single = np.array([np.nan, np.inf, -np.inf, -0.0, 1e-45, 3.4028235e38], dtype=np.float32)
    double = np.array([np.nan, np.inf, -np.inf, -0.0, 5e-324, 1.7976931348623157e308])
    path = tmp_path / "special.fits"
    ext = _write_tables(path, [Column(name="s", format="E", array=single), Column(name="d", format="D", array=double)])
    mine, _ = _compare(path, ext)
    assert mine["s"].tobytes() == single.tobytes() and mine["d"].tobytes() == double.tobytes()


def test_columns_without_ttype_are_named_col_n(tmp_path):
    columns = [
        Column(name="a", format="J", array=np.arange(3, dtype=np.int32)),
        Column(name="b", format="E", array=np.ones(3, np.float32)),
    ]
    path = tmp_path / "noname.fits"
    ext = _write_tables(path, columns)
    with astropy_fits.open(path, mode="update") as hdul:
        del hdul[ext].header["TTYPE1"]
        hdul[ext].header["TTYPE2"] = "kept"
    mine, _ = fits.read_table(path, ext)
    assert list(mine) == ["col1", "kept"] and mine["col1"].tolist() == [0, 1, 2]


def test_tdim_reshapes_with_first_dimension_fastest(tmp_path):
    # TDIM1 = (5,3,2): index of the first dimension varies fastest ([FITS2018] Sect. 7.3.2, TDIMn),
    # so a row of 30 values becomes an array of shape (2, 3, 5).
    rng = np.random.default_rng(402)
    cube = rng.integers(0, 1000, (4, 2, 3, 5)).astype(np.int16)
    bits = rng.random((4, 12)) > 0.5
    strings = np.array([[["ab", "cd", "ef"], ["gh", "ij", "kl"]]] * 4)  # TDIM (2,3,2): strings of 2 characters
    path = tmp_path / "tdim.fits"
    ext = _write_tables(
        path,
        [
            Column(name="cube", format="30I", dim="(5,3,2)", array=cube),
            Column(name="grid", format="12A", dim="(2,3,2)", array=strings),
        ],
    )
    mine, _ = fits.read_table(path, ext)
    oracle = _oracle(path, ext)
    assert mine["cube"].shape == (4, 2, 3, 5) and np.array_equal(mine["cube"], cube)
    assert np.array_equal(mine["cube"], oracle[0])
    assert mine["grid"].shape == (4, 2, 3) and mine["grid"][0, 1, 2] == "kl" and mine["grid"][3, 0, 1] == "cd"
    assert np.array_equal(mine["grid"], oracle[1])
    # bit array with TDIM (4,3), assembled by hand (astropy cannot write it): bit k of the row is element
    # k % 4 of row k // 4 of a (3, 4) array
    raw = np.packbits(np.concatenate([bits, np.zeros((4, 4), dtype=bool)], axis=1), axis=1).tobytes()
    path = tmp_path / "tdim_bits.fits"
    path.write_bytes(_raw_bintable([("bits", "12X", raw)], rows=4, extra={"TDIM1": "(4,3)"}))
    bit_columns, _ = fits.read_table(path, 1)
    assert bit_columns["bits"].shape == (4, 3, 4) and np.array_equal(bit_columns["bits"], bits.reshape(4, 3, 4))


def test_tdim_smaller_than_repeat_count_keeps_leading_elements(tmp_path):
    values = np.arange(24, dtype=np.int32).reshape(4, 6)
    path = tmp_path / "short_dim.fits"
    ext = _write_tables(path, [Column(name="v", format="6J", array=values)], patches={"TDIM1": "(2,2)"})
    mine, _ = fits.read_table(path, ext)
    assert np.array_equal(mine["v"], values[:, :4].reshape(4, 2, 2))  # Sect. 7.3.2: TDIMn size <= repeat count


def test_string_nul_termination_and_blanks(tmp_path):
    # Sect. 7.3.3.1 (Character): a string ends at the first NUL; characters after it are undefined.
    cells = [b"abc\x00junk", b"ab  \x00   ", b"\x00xxxxxxx", b"  lead  ", b"12345678"]
    path = tmp_path / "nul.fits"
    path.write_bytes(_raw_bintable([("s", "8A", b"".join(cells))], rows=len(cells)))
    mine, _ = fits.read_table(path, 1)
    assert mine["s"].dtype.kind == "U" and mine["s"].tolist() == ["abc", "ab", "", "  lead", "12345678"]


def test_logical_null_is_false_and_bit_order_is_msb_first(tmp_path):
    # L: 'T' true, 'F' false, zero byte null; X: the most significant bit comes first (Sect. 7.3.3.1).
    rows = [b"TF\x00" + bytes([0b10100000, 0b01000000]), b"FTT" + bytes([0b00000001, 0b11000000])]
    path = tmp_path / "bits.fits"
    path.write_bytes(
        _raw_bintable(
            [("flag", "3L", rows[0][:3] + rows[1][:3]), ("bits", "10X", rows[0][3:] + rows[1][3:])],
            rows=2,
        )
    )
    mine, _ = fits.read_table(path, 1)
    assert mine["flag"].tolist() == [[True, False, False], [False, True, True]]
    assert mine["bits"].astype(int).tolist() == [[1, 0, 1, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]


def test_scaled_and_unsigned_columns(tmp_path):
    rng = np.random.default_rng(403)
    n = 6
    columns = [
        Column(name="u16", format="I", array=np.array([-32768, -32767, 32767, 0, -1, 7232], dtype=np.int16)),
        Column(name="u32", format="J", array=np.array([-(2**31), -(2**31) + 1, 2**31 - 1, 0, -1, 5], dtype=np.int32)),
        Column(name="u64", format="K", array=np.array([-(2**63), -(2**63) + 1, 2**63 - 1, 0, -1, 5], dtype=np.int64)),
        Column(name="s8", format="B", array=np.array([0, 127, 128, 255, 1, 133], dtype=np.uint8)),
        Column(name="lin_i", format="I", array=rng.integers(-100, 100, n).astype(np.int16)),
        Column(name="lin_j", format="J", array=np.array([2**30 + 1, -(2**30), 3, 4, 5, 6], dtype=np.int32)),
        Column(name="lin_e", format="E", array=rng.normal(size=n).astype(np.float32)),
        Column(name="lin_d", format="3D", array=rng.normal(size=(n, 3))),
        Column(name="lin_d2", format="D", array=rng.normal(size=n) * 1e6),
        Column(name="offset_only", format="J", array=np.arange(n, dtype=np.int32)),
        Column(name="scale_only", format="B", array=np.arange(n, dtype=np.uint8)),
        Column(name="scaled_int", format="I", array=np.arange(n, dtype=np.int16)),
    ]
    patches = {
        "TZERO1": 32768,
        "TZERO2": 2147483648,
        "TZERO3": 9223372036854775808,
        "TZERO4": -128,
        "TSCAL5": 0.25,
        "TZERO5": 10.5,
        "TSCAL6": 1.5,
        "TZERO6": -3.0,
        "TSCAL7": 2.0,
        "TZERO7": 1.0,
        "TSCAL8": 0.1,
        "TZERO8": 100.0,
        "TSCAL9": 3.0,
        "TZERO9": 2.0,
        "TZERO10": 5,
        "TSCAL11": 2.5,
        "TSCAL12": 2.0,
        "TZERO12": 7.0,
    }
    path = tmp_path / "scaled.fits"
    ext = _write_tables(path, columns, patches)
    mine, _ = _compare(path, ext)
    assert mine["u16"].dtype == np.uint16 and mine["u16"].tolist() == [0, 1, 65535, 32768, 32767, 40000]
    assert mine["u32"].dtype == np.uint32 and mine["u32"].tolist() == [0, 1, 2**32 - 1, 2**31, 2**31 - 1, 2**31 + 5]
    assert mine["u64"].dtype == np.uint64
    assert mine["u64"].tolist() == [0, 1, 2**64 - 1, 2**63, 2**63 - 1, 2**63 + 5]  # exact, beyond float64 precision
    assert mine["s8"].dtype == np.int8 and mine["s8"].tolist() == [-128, -1, 0, 127, -127, 5]
    assert mine["lin_i"].dtype == np.float64 and mine["lin_e"].dtype == np.float64
    assert mine["lin_j"].tolist() == [1.5 * v - 3.0 for v in (2**30 + 1, -(2**30), 3, 4, 5, 6)]  # Eq. 7
    assert mine["scaled_int"].dtype == np.float64 and mine["scaled_int"].tolist() == [7.0 + 2.0 * i for i in range(6)]


def test_tnull_becomes_nan_only_for_floating_output(tmp_path):
    columns = [
        Column(name="scaled", format="I", array=np.array([1, -9, 3, -9, 5], dtype=np.int16)),
        Column(name="byte", format="B", array=np.array([255, 1, 255, 3, 4], dtype=np.uint8)),
        Column(name="long", format="K", array=np.array([-(2**63), 1, 2, 3, -(2**63)], dtype=np.int64)),
        Column(name="plain", format="J", array=np.array([-1, 0, 1, -1, 2], dtype=np.int32)),
        Column(name="unsigned", format="I", array=np.array([-32768, 5, -32768, 7, 8], dtype=np.int16)),
        Column(name="vec", format="2I", array=np.array([[1, -9], [-9, 2], [3, 4], [-9, -9], [5, 6]], dtype=np.int16)),
    ]
    patches = {
        "TNULL1": -9,
        "TSCAL1": 0.5,
        "TZERO1": 3.0,
        "TNULL2": 255,
        "TZERO2": 100.0,
        "TNULL3": -(2**63),
        "TSCAL3": 2.0,
        "TNULL4": -1,  # unscaled: the integer column keeps the pattern
        "TNULL5": -32768,  # unsigned convention: stays an exact integer
        "TZERO5": 32768,
        "TNULL6": -9,
        "TSCAL6": 2.0,
    }
    path = tmp_path / "nulls.fits"
    ext = _write_tables(path, columns, patches)
    mine, _ = fits.read_table(path, ext)
    oracle = _oracle(path, ext)
    # astropy does not map TNULLn to NaN: it returns the scaled null pattern. Everywhere else the two must agree.
    for i in (0, 1, 2, 5):
        expected = np.where(columns[i].array == int(patches[f"TNULL{i + 1}"]), np.nan, oracle[i])
        _same(mine[columns[i].name], expected)
    _same(mine["plain"], oracle[3])
    _same(mine["unsigned"], oracle[4])
    assert np.isnan(mine["scaled"]).tolist() == [False, True, False, True, False]
    assert mine["scaled"][[0, 2, 4]].tolist() == [3.5, 4.5, 5.5]
    assert np.isnan(mine["byte"]).tolist() == [True, False, True, False, False]
    assert np.isnan(mine["long"]).tolist() == [True, False, False, False, True]
    assert mine["plain"].dtype == np.int32 and mine["plain"].tolist() == [-1, 0, 1, -1, 2]
    assert mine["unsigned"].dtype == np.uint16 and mine["unsigned"][0] == 0
    assert np.isnan(mine["vec"]).tolist() == [
        [False, True],
        [True, False],
        [False, False],
        [True, True],
        [False, False],
    ]


def _variable_columns():
    rng = np.random.default_rng(404)
    lengths = [3, 0, 5, 1, 2, 0, 4]

    def ragged(make):
        return np.array([make(m) for m in lengths], dtype=object)

    return [
        Column(name="pj", format="PJ()", array=ragged(lambda m: (rng.normal(size=m) * 1000).astype(np.int32))),
        Column(name="qd", format="QD()", array=ragged(lambda m: rng.normal(size=m))),
        Column(name="pe", format="1PE(9)", array=ragged(lambda m: rng.normal(size=m).astype(np.float32))),
        Column(name="pb", format="PB()", array=ragged(lambda m: np.arange(m, dtype=np.uint8) * 7)),
        Column(name="qi", format="QI()", array=ragged(lambda m: np.arange(m, dtype=np.int16) - 2)),
        Column(name="pk", format="PK()", array=ragged(lambda m: np.arange(m, dtype=np.int64) * 2**40)),
        Column(name="pm", format="PM()", array=ragged(lambda m: np.arange(m) * (1 + 2j))),
        Column(name="fixed", format="J", array=np.arange(len(lengths), dtype=np.int32)),
    ]


def test_variable_length_arrays_match_astropy(tmp_path):
    path = tmp_path / "vla.fits"
    ext = _write_tables(path, _variable_columns())
    mine, header = _compare(path, ext)
    assert header["PCOUNT"] > 0
    for name in ("pj", "qd", "pe", "pb", "qi", "pk", "pm"):
        assert mine[name].dtype == object and len(mine[name]) == 7
    assert [len(a) for a in mine["pj"]] == [3, 0, 5, 1, 2, 0, 4]
    assert (
        mine["pj"][0].dtype == np.int32 and mine["qd"][2].dtype == np.float64 and mine["pm"][2].dtype == np.complex128
    )
    assert mine["pk"][2].tolist() == [i * 2**40 for i in range(5)]
    assert mine["fixed"].tolist() == list(range(7))


def test_variable_length_scaling_is_applied_to_heap_values(tmp_path):
    # Sect. 7.3.2 (TSCALn) and 7.3.5: scaling acts on the heap array, not on the descriptor.
    rows = [np.array([1, -9, 3], dtype=np.int16), np.array([], dtype=np.int16), np.array([-9, 4], dtype=np.int16)]
    unsigned = [np.array([-32768, 0], dtype=np.int16), np.array([], dtype=np.int16), np.array([32767], dtype=np.int16)]
    path = tmp_path / "vla_scaled.fits"
    ext = _write_tables(
        path,
        [
            Column(name="a", format="PI()", array=np.array(rows, dtype=object)),
            Column(name="b", format="QI()", array=np.array(unsigned, dtype=object)),
        ],
        patches={"TSCAL1": 2.0, "TZERO1": 1.0, "TNULL1": -9, "TZERO2": 32768},
    )
    mine, _ = fits.read_table(path, ext)
    assert mine["a"][0][0] == 3.0 and np.isnan(mine["a"][0][1]) and mine["a"][0][2] == 7.0
    assert mine["a"][1].shape == (0,) and np.isnan(mine["a"][2][0]) and mine["a"][2][1] == 9.0
    assert mine["b"][0].dtype == np.uint16 and mine["b"][0].tolist() == [0, 32768] and mine["b"][2].tolist() == [65535]


def test_heap_gap_aliasing_and_out_of_order_descriptors(tmp_path):
    # Hand-made file ([FITS2018] Sect. 7.3.5): THEAP leaves a gap after the main table, descriptors share bytes
    # and are not in increasing order of offset. Descriptor = (length, byte offset from the heap start).
    heap = np.arange(10, dtype=">i4").tobytes()  # the 32-bit integers 0..9
    rows = _descriptors([4, 16], [2, 0], [0, 0], [3, 20], [10, 0])
    gap = 24
    path = tmp_path / "heap.fits"
    path.write_bytes(
        _raw_bintable([("v", "1PJ(10)", rows)], rows=5, heap=b"\xee" * gap + heap, extra={"THEAP": 5 * 8 + gap})
    )
    mine, header = fits.read_table(path, 1)
    assert header["PCOUNT"] == gap + 40
    assert [a.tolist() for a in mine["v"]] == [[4, 5, 6, 7], [0, 1], [], [5, 6, 7], list(range(10))]
    assert all(a.dtype == np.int32 for a in mine["v"])


def test_q_descriptors_are_64_bit(tmp_path):
    heap = np.array([1.5, -2.5, 3.5], dtype=">f8").tobytes()
    rows = _descriptors([2, 8], [1, 0], dtype=">i8")
    path = tmp_path / "q.fits"
    path.write_bytes(_raw_bintable([("v", "1QD(2)", rows)], rows=2, heap=heap))
    mine, _ = fits.read_table(path, 1)
    assert [a.tolist() for a in mine["v"]] == [[-2.5, 3.5], [1.5]]


def test_variable_length_descriptors_outside_the_heap_raise(tmp_path):
    heap = np.arange(4, dtype=">i4").tobytes()
    for pair in ([5, 0], [2, 12], [-1, 0], [1, -4]):
        path = tmp_path / f"bad_{pair[0]}_{pair[1]}.fits".replace("-", "m")
        path.write_bytes(_raw_bintable([("v", "1PJ(5)", _descriptors(pair))], rows=1, heap=heap))
        with pytest.raises(ValueError, match="outside the heap"):
            fits.read_table(path, 1)


def test_variable_length_with_zero_repeat_count_is_all_empty(tmp_path):
    path = tmp_path / "p0.fits"
    path.write_bytes(_raw_bintable([("v", "0PJ(0)", b""), ("n", "J", np.arange(3, dtype=">i4").tobytes())], rows=3))
    mine, _ = fits.read_table(path, 1)
    assert [len(a) for a in mine["v"]] == [0, 0, 0] and mine["n"].tolist() == [0, 1, 2]


def test_variable_length_character_logical_and_bit_arrays(tmp_path):
    # Element types A, L, X of heap arrays; for X the element count is the number of bits.
    heap = b"hello world  " + b"TFT" + bytes([0b10110000, 0b01000000])  # characters 0-12, logicals 13-15, bits 16-17
    path = tmp_path / "vla_types.fits"
    path.write_bytes(
        _raw_bintable(
            [
                ("s", "1PA(8)", _descriptors([5, 0], [8, 0], [0, 0])),
                ("t", "1PL(3)", _descriptors([0, 0], [3, 13], [2, 13])),
                ("x", "1PX(10)", _descriptors([4, 16], [10, 16], [0, 0])),
            ],
            rows=3,
            heap=heap,
        )
    )
    mine, _ = fits.read_table(path, 1)
    assert mine["s"].tolist() == ["hello", "hello wo", ""]
    assert [a.tolist() for a in mine["t"]] == [[], [True, False, True], [True, False]]
    assert [a.astype(int).tolist() for a in mine["x"]] == [[1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0, 0, 1], []]


def test_table_after_image_extensions_and_hdu_selection(tmp_path):
    rng = np.random.default_rng(405)
    images = [astropy_fits.ImageHDU(rng.normal(size=(9, 11)).astype(np.float32), name=f"IM{i}") for i in range(2)]
    path = tmp_path / "mef_table.fits"
    ext = _write_tables(path, _all_columns(4), before=images)
    assert ext == 3
    assert [kind for kind, _, _ in fits.list_hdus(path)] == ["PRIMARY", "IMAGE", "IMAGE", "BINTABLE"]
    mine, header = _compare(path, ext)
    assert header["EXTNAME"] == "TAB" and mine["flag"].shape == (4,)
    with pytest.raises(ValueError, match="read_fits"):
        fits.read_table(path, 1)


def test_two_tables_are_read_independently(tmp_path):
    first = astropy_fits.BinTableHDU.from_columns(
        [Column(name="a", format="J", array=np.arange(3, dtype=np.int32))], name="ONE"
    )
    second = astropy_fits.BinTableHDU.from_columns(
        [Column(name="a", format="PD()", array=np.array([np.arange(2.0), np.arange(3.0)], dtype=object))], name="TWO"
    )
    path = tmp_path / "two.fits"
    astropy_fits.HDUList([astropy_fits.PrimaryHDU(), first, second]).writeto(path)
    one, _ = fits.read_table(path, 1)
    two, header = fits.read_table(path, 2)
    assert one["a"].tolist() == [0, 1, 2] and header["EXTNAME"] == "TWO"
    assert [a.tolist() for a in two["a"]] == [[0.0, 1.0], [0.0, 1.0, 2.0]]


def test_zero_row_table_and_zero_repeat_column(tmp_path):
    columns = [
        Column(name="a", format="J", array=np.zeros(0, dtype=np.int32)),
        Column(name="b", format="3E", array=np.zeros((0, 3), dtype=np.float32)),
        Column(name="c", format="4A", array=np.array([], dtype="U4")),
        Column(name="d", format="PJ()", array=np.array([], dtype=object)),
        Column(name="e", format="2X", array=np.zeros((0, 2), dtype=bool)),
    ]
    path = tmp_path / "empty_rows.fits"
    ext = _write_tables(path, columns, patches={"TZERO1": 2147483648})
    assert astropy_fits.getheader(path, ext)["NAXIS2"] == 0
    mine, header = fits.read_table(path, ext)
    assert header["NAXIS2"] == 0 and list(mine) == ["a", "b", "c", "d", "e"]
    assert mine["a"].shape == (0,) and mine["a"].dtype == np.uint32
    assert mine["b"].shape == (0, 3) and mine["b"].dtype == np.float32
    assert mine["c"].shape == (0,) and mine["c"].dtype.kind == "U"
    assert mine["d"].shape == (0,) and mine["d"].dtype == object
    assert mine["e"].shape == (0, 2) and mine["e"].dtype == bool
    no_fields = tmp_path / "no_fields.fits"
    no_fields.write_bytes(_raw_bintable([], rows=0))
    assert fits.read_table(no_fields, 1)[0] == {}


def test_zero_repeat_count_is_an_empty_field(tmp_path):
    # Sect. 7.3.1 (TFORMn): a zero element count is permitted and occupies no bytes.
    path = tmp_path / "zero_repeat.fits"
    path.write_bytes(
        _raw_bintable(
            [("n", "J", np.arange(3, dtype=">i4").tobytes()), ("z", "0E", b""), ("zs", "0A", b"")],
            rows=3,
            naxis1=4,
        )
    )
    mine, _ = fits.read_table(path, 1)
    assert mine["z"].shape == (3, 0) and mine["z"].dtype == np.float32
    assert mine["zs"].tolist() == ["", "", ""] and mine["n"].tolist() == [0, 1, 2]


def test_file_without_final_padding_is_read(tmp_path):
    path = tmp_path / "unpadded.fits"
    ext = _write_tables(path, [Column(name="a", format="J", array=np.arange(5, dtype=np.int32))])
    start = fits.list_hdus(path)[ext][2]
    path.write_bytes(path.read_bytes()[: start + 5 * 4])
    assert fits.read_table(path, ext)[0]["a"].tolist() == [0, 1, 2, 3, 4]


def test_error_paths(tmp_path):
    ascii_table = astropy_fits.TableHDU.from_columns([Column(name="a", format="I4", array=np.arange(3))])
    path = tmp_path / "ascii.fits"
    astropy_fits.HDUList([astropy_fits.PrimaryHDU(), ascii_table]).writeto(path)
    assert fits.list_hdus(path)[1][0] == "TABLE"
    with pytest.raises(ValueError, match="ASCII table"):
        fits.read_table(path, 1)
    with pytest.raises(ValueError, match="read_fits"):
        fits.read_table(path, 0)  # the primary HDU
    with pytest.raises(ValueError, match="no HDU 5"):
        fits.read_table(path, 5)
    with pytest.raises(ValueError, match="no HDU -1"):
        fits.read_table(path, -1)

    good = tmp_path / "good.fits"
    _write_tables(good, [Column(name="a", format="J", array=np.arange(5, dtype=np.int32))])
    with pytest.raises(ValueError, match="read_table"):  # read_fits points to the table reader
        fits.read_fits(good, 1)
    truncated = tmp_path / "truncated.fits"
    truncated.write_bytes(good.read_bytes()[: fits.list_hdus(good)[1][2] + 10])
    with pytest.raises(ValueError, match="truncated"):
        fits.read_table(truncated, 1)

    four = b"\0" * 4
    cases = {
        "bad_width": (_raw_bintable([("a", "J", four * 2)], rows=2, naxis1=5), "Eq. 8"),
        "lower_case": (_raw_bintable([("a", "j", four * 2)], rows=2), "malformed"),
        "unknown_type": (_raw_bintable([("a", "1Z", four * 2)], rows=2), "unknown"),
        "no_tform": (_raw_bintable([("a", None, b"")], rows=0), "TFORM1"),
        "string_width": (_raw_bintable([("a", "8A10", b"\0" * 16)], rows=2), "width suffix"),
        "p_repeat": (_raw_bintable([("a", "2PJ(3)", b"\0" * 16)], rows=1), "repeat count of 0 or 1"),
        "p_no_type": (_raw_bintable([("a", "1P", b"\0" * 8)], rows=1), "element type"),
        "dim_too_big": (_raw_bintable([("a", "4J", four * 4)], rows=1, extra={"TDIM1": "(3,2)"}), "more elements"),
        "dim_malformed": (_raw_bintable([("a", "4J", four * 4)], rows=1, extra={"TDIM1": "3,2"}), "malformed"),
        "duplicate": (_raw_bintable([("a", "J", four), ("a", "J", four)], rows=1), "duplicate"),
        "theap": (_raw_bintable([("a", "J", four)], rows=1, extra={"THEAP": 100}), "THEAP"),
        "gcount": (_raw_bintable([("a", "J", four)], rows=1, extra={"GCOUNT": 2}), "mandatory"),
    }
    for label, (data, message) in cases.items():
        bad = tmp_path / f"{label}.fits"
        bad.write_bytes(data)
        with pytest.raises(ValueError, match=message):
            fits.read_table(bad, 1)


def test_non_ascii_text_raises(tmp_path):
    path = tmp_path / "latin.fits"
    path.write_bytes(_raw_bintable([("s", "4A", "caf\xe9".encode("latin-1"))], rows=1))
    with pytest.raises(ValueError):
        fits.read_table(path, 1)


def test_real_binary_tables_in_data_raw_match_astropy():
    root = Path(__file__).resolve().parents[2] / "data" / "raw"
    found = []
    for candidate in sorted(root.rglob("*")) if root.is_dir() else []:
        if candidate.is_file() and candidate.suffix.lower() in (".fits", ".fit", ".fts"):
            kinds = [kind for kind, _, _ in fits.list_hdus(candidate)]
            found += [(candidate, i) for i, kind in enumerate(kinds) if kind == "BINTABLE"]
    if not found:
        pytest.skip("no FITS file with a BINTABLE extension under data/raw")
    for candidate, ext in found:
        _compare(candidate, ext)


def test_scaled_64_bit_integers_are_converted_to_float64(tmp_path):
    # Hand-made file (astropy mishandles scaled K columns): physical = TZERO + TSCAL x field (Eq. 7) in float64.
    stored = np.array([2**62 + 1, -(2**62), 3, -(2**63)], dtype=">i8")
    path = tmp_path / "k.fits"
    path.write_bytes(
        _raw_bintable([("k", "K", stored.tobytes())], rows=4, extra={"TSCAL1": 1.5, "TZERO1": -3.0, "TNULL1": -(2**63)})
    )
    mine, _ = fits.read_table(path, 1)
    assert mine["k"].dtype == np.float64
    assert mine["k"][:3].tolist() == [1.5 * float(2**62 + 1) - 3.0, 1.5 * -float(2**62) - 3.0, 1.5 * 3.0 - 3.0]
    assert np.isnan(mine["k"][3])


def test_scaling_of_complex_and_of_the_unsigned_offset_with_a_scale_factor(tmp_path):
    # Hand-made file (astropy discards the imaginary part of scaled complex columns and fails on TSCALn != 1 with
    # TZEROn = 32768). [FITS2018] Sect. 7.3.2: for C and M, TSCALn and TZEROn belong to the real part and the
    # imaginary part of the scale factor is zero, so the real factor multiplies both parts and TZEROn is added to the
    # real part only.
    complex_stored = np.array([1 + 2j, -3.5 + 0.5j, 0j], dtype=">c16")
    short_stored = np.array([-32768, 0, 5], dtype=">i2")
    path = tmp_path / "complex_scaled.fits"
    path.write_bytes(
        _raw_bintable(
            [("z", "M", complex_stored.tobytes()), ("u", "I", short_stored.tobytes())],
            rows=3,
            extra={"TSCAL1": 2.0, "TZERO1": 10.0, "TSCAL2": 2.0, "TZERO2": 32768},
        )
    )
    mine, _ = fits.read_table(path, 1)
    assert mine["z"].dtype == np.complex128 and mine["z"].tolist() == [12 + 4j, 3 + 1j, 10 + 0j]
    assert mine["u"].dtype == np.float64 and mine["u"].tolist() == [32768.0 - 65536.0, 32768.0, 32768.0 + 10.0]
