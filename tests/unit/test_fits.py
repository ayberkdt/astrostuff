"""Comparison of the FITS reader and writer with astropy.io.fits."""

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
