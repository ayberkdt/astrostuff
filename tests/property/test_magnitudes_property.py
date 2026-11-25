from hypothesis import given
from hypothesis import strategies as st

from kozmos.radiation import magnitudes


@given(st.floats(-15, 20))
def test_bolometric_round_trip(m):
    luminosity = magnitudes.luminosity_from_bolometric_magnitude_w(m)
    assert abs(float(magnitudes.absolute_bolometric_magnitude(luminosity)) - m) < 1e-12


@given(st.floats(-5, 50))
def test_distance_modulus_round_trip(mu):
    assert abs(float(magnitudes.distance_modulus_from_pc(magnitudes.pc_from_distance_modulus(mu))) - mu) < 1e-11


@given(st.floats(1.0, 1e9), st.floats(1.01, 100.0))
def test_distance_factor_modulus_difference(d, factor):
    # A tenfold distance increases the modulus by 5 mag; in general by 5 log10(factor).
    import math

    difference = float(magnitudes.distance_modulus_from_pc(d * factor)) - float(magnitudes.distance_modulus_from_pc(d))
    assert abs(difference - 5 * math.log10(factor)) < 1e-10
