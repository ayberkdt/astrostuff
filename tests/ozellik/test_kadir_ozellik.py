from hypothesis import given
from hypothesis import strategies as st

from kozmos.isinim import kadir


@given(st.floats(-15, 20))
def test_bolometrik_gidis_donus(m):
    isinim_gucu = kadir.bolometrik_kadirden_isinim_gucu_w(m)
    assert abs(float(kadir.mutlak_bolometrik_kadir(isinim_gucu)) - m) < 1e-12


@given(st.floats(-5, 50))
def test_uzaklik_modulu_gidis_donus(mu):
    assert abs(float(kadir.uzaklik_modulu_pc(kadir.uzaklik_modulunden_pc(mu))) - mu) < 1e-11


@given(st.floats(1.0, 1e9), st.floats(1.01, 100.0))
def test_uzaklik_carpani_modul_farki(d, carpan):
    # Uzaklık 10 katına çıkınca modül 5 mag artar; genel olarak 5 log10(çarpan).
    import math

    fark = float(kadir.uzaklik_modulu_pc(d * carpan)) - float(kadir.uzaklik_modulu_pc(d))
    assert abs(fark - 5 * math.log10(carpan)) < 1e-10
