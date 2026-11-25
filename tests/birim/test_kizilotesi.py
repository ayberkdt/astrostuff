import numpy as np
import pytest

from kozmos.isinim import kizilotesi as ko


def test_sifir_noktalari_tablo_degerleri():
    # [Cohen2003] Tablo 2 ve [Wright2010] Bölüm 4.4'ten okunan değerler.
    beklenen = {"J": 1594.0, "H": 1024.0, "Ks": 666.7, "W1": 306.681, "W2": 170.663, "W3": 29.0448, "W4": 8.2839}
    for ad, aki in beklenen.items():
        assert float(ko.vega_kadirinden_aki_jy(0.0, ad)) == aki
        assert float(ko.vega_kadirinden_aki_jy(2.5, ad)) == pytest.approx(aki / 10.0)
    with pytest.raises(ValueError):
        ko.bant("Z")


def test_gidis_donus_ve_ab_farki():
    kadir = np.array([-1.0, 5.0, 14.3])
    for ad in ko.BANTLAR:
        assert ko.akidan_vega_kadiri(ko.vega_kadirinden_aki_jy(kadir, ad), ad) == pytest.approx(kadir, abs=1e-12)
    # Ks: 2,5 log10(3631 / 666,7) = 1,840 mag.
    assert ko.vega_ab_farki("Ks") == pytest.approx(1.840, abs=1e-3)
    # Dalga boyu arttıkça Vega'nın akısı düşer, AB − Vega farkı büyür.
    farklar = [ko.vega_ab_farki(a) for a in ("J", "H", "Ks", "W1", "W2", "W3", "W4")]
    assert all(b > a for a, b in zip(farklar, farklar[1:], strict=False))


def test_wien():
    assert float(ko.tepe_dalgaboyu_um(5772.0)) == pytest.approx(0.50204, abs=1e-4)
    assert float(ko.tepe_dalgaboyu_um(30.0)) == pytest.approx(96.6, abs=0.1)
    um = np.array([0.5, 2.159, 22.0])
    assert ko.tepe_dalgaboyu_um(ko.tepe_sicakligi_k(um)) == pytest.approx(um, rel=1e-14)
    assert float(ko.tepe_sicakligi_k(2.159)) == pytest.approx(1342.0, abs=1.0)
