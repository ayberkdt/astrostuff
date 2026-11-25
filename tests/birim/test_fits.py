"""FITS okuyucu ve yazıcının astropy.io.fits ile karşılaştırılması."""

import numpy as np
import pytest
from astropy.io import fits as astropy_fits

from kozmos.io import fits


@pytest.mark.parametrize("tip", [np.float64, np.float32, np.int16, np.int32, np.uint8])
def test_yaz_astropy_okur(tmp_path, tip):
    u = np.random.default_rng(91)
    veri = (u.normal(100, 30, (37, 53))).astype(tip)
    yol = tmp_path / "ornek.fits"
    fits.fits_yaz(yol, veri, {"OBJECT": "deneme alani", "EXPTIME": 12.5, "NCOMBINE": 3, "FLAG": True})
    with astropy_fits.open(yol) as hdul:
        hdul.verify("exception")  # standarda aykırılık varsa hata verir
        assert np.array_equal(hdul[0].data, veri)
        bas = hdul[0].header
        assert (
            bas["OBJECT"] == "deneme alani" and bas["EXPTIME"] == 12.5 and bas["NCOMBINE"] == 3 and bas["FLAG"] is True
        )
    assert yol.stat().st_size % fits.BLOK == 0


def test_astropy_yazar_biz_okuruz(tmp_path):
    u = np.random.default_rng(92)
    veri = u.normal(size=(20, 31)).astype(">f4")
    bas = astropy_fits.Header()
    bas["TELESCOP"] = "T'est"  # dizgi içinde tek tırnak
    bas["AIRMASS"] = (1.23, "yorum / bolu isareti")
    bas["HISTORY"] = "atlanmasi gereken kayit"
    yol = tmp_path / "a.fits"
    astropy_fits.writeto(yol, veri, bas)
    okunan, baslik = fits.fits_oku(yol)
    assert np.array_equal(okunan, veri.astype(float)) and okunan.shape == (20, 31)
    assert baslik["TELESCOP"] == "T'est" and baslik["AIRMASS"] == 1.23 and baslik["NAXIS1"] == 31
    assert "HISTORY" not in baslik


def test_olcekleme_ve_isaretsiz_tamsayi(tmp_path):
    # İşaretsiz 16 bit: BITPIX = 16, BZERO = 32768 ([FITS2018] BZERO açıklaması).
    veri = np.array([[0, 1, 65535], [32768, 40000, 123]], dtype=np.uint16)
    yol = tmp_path / "u16.fits"
    astropy_fits.writeto(yol, veri)
    okunan, baslik = fits.fits_oku(yol)
    assert baslik["BITPIX"] == 16 and baslik["BZERO"] == 32768
    assert np.array_equal(okunan, veri.astype(float))


def test_gidis_donus_ve_hatalar(tmp_path):
    veri = np.arange(24, dtype=float).reshape(2, 3, 4)
    yol = tmp_path / "kup.fits"
    fits.fits_yaz(yol, veri)
    okunan, baslik = fits.fits_oku(yol)
    assert np.array_equal(okunan, veri) and (baslik["NAXIS1"], baslik["NAXIS2"], baslik["NAXIS3"]) == (4, 3, 2)
    with pytest.raises(ValueError):
        fits.baslik_coz(b"SIMPLE  =                    T".ljust(80))  # END yok
    bos = tmp_path / "bos.fits"
    astropy_fits.PrimaryHDU().writeto(bos)
    with pytest.raises(ValueError):
        fits.fits_oku(bos)
    with pytest.raises(ValueError):
        fits.fits_oku(bos, hdu=3)


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
    yol = tmp_path / "mef.fits"
    hdul.writeto(yol)
    liste = fits.hdu_listesi(yol)
    assert [tur for tur, _, _ in liste] == ["PRIMARY", "IMAGE", "IMAGE", "BINTABLE", "IMAGE"]
    assert [b.get("EXTNAME") for _, b, _ in liste[1:]] == ["SCI", "ERR", "TAB", "DQ"]
    for i, beklenen in ((1, bilim), (2, hata), (4, kalite)):  # tablodan SONRAKİ uzantı da doğru yerde bulunmalı
        veri, baslik = fits.fits_oku(yol, hdu=i)
        assert np.array_equal(veri, beklenen.astype(float)) and baslik["XTENSION"] == "IMAGE"
    with pytest.raises(ValueError):
        fits.fits_oku(yol, hdu=3)  # tablo
    with pytest.raises(ValueError):
        fits.fits_oku(yol, hdu=0)  # veri yok
