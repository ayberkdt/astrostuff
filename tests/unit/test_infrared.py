import numpy as np
import pytest

from kozmos.radiation import infrared


def test_zero_points_table_values():
    # Values read from [Cohen2003] Table 2 and [Wright2010] Section 4.4.
    expected = {"J": 1594.0, "H": 1024.0, "Ks": 666.7, "W1": 306.681, "W2": 170.663, "W3": 29.0448, "W4": 8.2839}
    for name, flux in expected.items():
        assert float(infrared.flux_jy_from_vega_magnitude(0.0, name)) == flux
        assert float(infrared.flux_jy_from_vega_magnitude(2.5, name)) == pytest.approx(flux / 10.0)
    with pytest.raises(ValueError):
        infrared.band("Z")


def test_round_trip_and_ab_offset():
    magnitude = np.array([-1.0, 5.0, 14.3])
    for name in infrared.BANDS:
        assert infrared.vega_magnitude_from_flux(
            infrared.flux_jy_from_vega_magnitude(magnitude, name), name
        ) == pytest.approx(magnitude, abs=1e-12)
    # Ks: 2.5 log10(3631 / 666.7) = 1.840 mag.
    assert infrared.vega_ab_offset("Ks") == pytest.approx(1.840, abs=1e-3)
    # As the wavelength increases, Vega's flux drops and the AB − Vega offset grows.
    offsets = [infrared.vega_ab_offset(a) for a in ("J", "H", "Ks", "W1", "W2", "W3", "W4")]
    assert all(b > a for a, b in zip(offsets, offsets[1:], strict=False))


def test_wien():
    assert float(infrared.peak_wavelength_um(5772.0)) == pytest.approx(0.50204, abs=1e-4)
    assert float(infrared.peak_wavelength_um(30.0)) == pytest.approx(96.6, abs=0.1)
    um = np.array([0.5, 2.159, 22.0])
    assert infrared.peak_wavelength_um(infrared.peak_temperature_k(um)) == pytest.approx(um, rel=1e-14, abs=0)
    assert float(infrared.peak_temperature_k(2.159)) == pytest.approx(1342.0, abs=1.0)
