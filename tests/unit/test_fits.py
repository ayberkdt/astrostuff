"""FITS okuyucu ve yazıcının astropy.io.fits ile karşılaştırılması."""

import numpy as np
import pytest
from astropy.io import fits as astropy_fits

from kozmos.io import fits


@pytest.mark.parametrize("tip", [np.float64, np.float32, np.int16, np.int32, np.uint8])
def test_yaz_astropy_okur(tmp_path, tip):
    u = np.random.default_rng(91)
    data = (u.normal(100, 30, (37, 53))).astype(tip)
    path = tmp_path / "ornek.fits"
    fits.write_fits(path, data, {"OBJECT": "deneme alani", "EXPTIME": 12.5, "NCOMBINE": 3, "FLAG": True})
    with astropy_fits.open(path) as hdul:
        hdul.verify("exception")  # standarda aykırılık varsa hata verir
        assert np.array_equal(hdul[0].data, data)
        bas = hdul[0].header
        assert (
            bas["OBJECT"] == "deneme alani" and bas["EXPTIME"] == 12.5 and bas["NCOMBINE"] == 3 and bas["FLAG"] is True
        )
    assert path.stat().st_size % fits.BLOCK == 0


def test_astropy_yazar_biz_okuruz(tmp_path):
    u = np.random.default_rng(92)
    data = u.normal(size=(20, 31)).astype(">f4")
    bas = astropy_fits.Header()
    bas["TELESCOP"] = "T'est"  # dizgi içinde tek tırnak
    bas["AIRMASS"] = (1.23, "yorum / bolu isareti")
    bas["HISTORY"] = "atlanmasi gereken kayit"
    path = tmp_path / "a.fits"
    astropy_fits.writeto(path, data, bas)
    okunan, header = fits.read_fits(path)
    assert np.array_equal(okunan, data.astype(float)) and okunan.shape == (20, 31)
    assert header["TELESCOP"] == "T'est" and header["AIRMASS"] == 1.23 and header["NAXIS1"] == 31
    assert "HISTORY" not in header


def test_olcekleme_ve_isaretsiz_tamsayi(tmp_path):
    # İşaretsiz 16 bit: BITPIX = 16, BZERO = 32768 ([FITS2018] BZERO açıklaması).
    data = np.array([[0, 1, 65535], [32768, 40000, 123]], dtype=np.uint16)
    path = tmp_path / "u16.fits"
    astropy_fits.writeto(path, data)
    okunan, header = fits.read_fits(path)
    assert header["BITPIX"] == 16 and header["BZERO"] == 32768
    assert np.array_equal(okunan, data.astype(float))


def test_gidis_donus_ve_hatalar(tmp_path):
    data = np.arange(24, dtype=float).reshape(2, 3, 4)
    path = tmp_path / "kup.fits"
    fits.write_fits(path, data)
    okunan, header = fits.read_fits(path)
    assert np.array_equal(okunan, data) and (header["NAXIS1"], header["NAXIS2"], header["NAXIS3"]) == (4, 3, 2)
    with pytest.raises(ValueError):
        fits.parse_header(b"SIMPLE  =                    T".ljust(80))  # END yok
    bos = tmp_path / "bos.fits"
    astropy_fits.PrimaryHDU().writeto(bos)
    with pytest.raises(ValueError):
        fits.read_fits(bos)
    with pytest.raises(ValueError):
        fits.read_fits(bos, hdu=3)


def test_cok_uzantili_dosya(tmp_path):
    # Veri taşımayan birincil HDU + iki görüntü uzantısı + bir ikili tablo + bir görüntü daha (HST/JWST tarzı).
    u = np.random.default_rng(93)
    bilim = u.normal(size=(17, 23)).astype(np.float32)
    hata = np.abs(u.normal(size=(17, 23))).astype(np.float64)
    kalite = u.integers(0, 4, (17, 23)).astype(np.int16)
    tablo = astropy_fits.BinTableHDU.from_columns(
        [astropy_fits.Column(name="x", format="E", array=np.arange(5, dtype=np.float32))], name="TAB"
    )
    hdul = astropy_fits.HDUList(
        [
            astropy_fits.PrimaryHDU(),
            astropy_fits.ImageHDU(bilim, name="SCI"),
            astropy_fits.ImageHDU(hata, name="ERR"),
            tablo,
            astropy_fits.ImageHDU(kalite, name="DQ"),
        ]
    )
    path = tmp_path / "mef.fits"
    hdul.writeto(path)
    liste = fits.list_hdus(path)
    assert [tur for tur, _, _ in liste] == ["PRIMARY", "IMAGE", "IMAGE", "BINTABLE", "IMAGE"]
    assert [b.get("EXTNAME") for _, b, _ in liste[1:]] == ["SCI", "ERR", "TAB", "DQ"]
    for i, beklenen in ((1, bilim), (2, hata), (4, kalite)):  # tablodan SONRAKİ uzantı da doğru yerde bulunmalı
        data, header = fits.read_fits(path, hdu=i)
        assert np.array_equal(data, beklenen.astype(float)) and header["XTENSION"] == "IMAGE"
    with pytest.raises(ValueError):
        fits.read_fits(path, hdu=3)  # tablo
    with pytest.raises(ValueError):
        fits.read_fits(path, hdu=0)  # veri yok


def test_hierarch_kartlari(tmp_path):
    # ESO HIERARCH kuralı: astropy 8 karakterden uzun ya da boşluklu anahtarları bu biçimde yazar.
    hdu = astropy_fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32))
    hdu.header["HIERARCH ESO DET DIT"] = (6.0, "Integration Time")
    hdu.header["HIERARCH ESO DET NDIT"] = 15
    hdu.header["HIERARCH ESO INS FILT1 NAME"] = "Ks"
    hdu.header["HIERARCH ESO SEQ CUMOFFSETX"] = -208.333333
    hdu.header["HIERARCH ESO TEL FOCU LEN"] = True
    path = tmp_path / "h.fits"
    hdu.writeto(path)
    _, header = fits.read_fits(path)
    kahin = astropy_fits.getheader(path)
    for key in ("ESO DET DIT", "ESO DET NDIT", "ESO INS FILT1 NAME", "ESO SEQ CUMOFFSETX", "ESO TEL FOCU LEN"):
        assert header[key] == kahin[key] and type(header[key]) is type(kahin[key])
    assert header["NAXIS1"] == 4  # olağan kartlar etkilenmez
