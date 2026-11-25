import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from kozmos.core import angles

acilar = st.floats(-1e4, 1e4, allow_nan=False)
boylamlar = st.floats(0, 2 * math.pi, exclude_max=True)
enlemler = st.floats(-math.pi / 2 + 1e-9, math.pi / 2 - 1e-9)


@given(acilar)
def test_sifir_iki_pi_araligi(a):
    w = float(angles.wrap_0_2pi(a))
    assert 0.0 <= w < 2 * math.pi
    assert abs(math.remainder(w - a, 2 * math.pi)) < 1e-9


@given(acilar)
def test_eksi_pi_arti_pi_araligi(a):
    assert -math.pi <= float(angles.wrap_pm_pi(a)) < math.pi


@given(boylamlar, enlemler)
def test_kuresel_gidis_donus(boylam, enlem):
    v = angles.spherical_to_unit_vector(boylam, enlem)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-14
    boylam2, enlem2 = angles.unit_vector_to_spherical(v)
    assert float(angles.angular_separation(boylam, enlem, boylam2, enlem2)) < 1e-12


@given(boylamlar, enlemler, boylamlar, enlemler)
def test_ayriklik_simetrik_ve_aralikta(l1, b1, l2, b2):
    d = float(angles.angular_separation(l1, b1, l2, b2))
    assert 0.0 <= d <= math.pi + 1e-15
    assert abs(d - float(angles.angular_separation(l2, b2, l1, b1))) < 1e-14
