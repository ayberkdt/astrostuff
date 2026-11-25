import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from kozmos.core.constants import SPEED_OF_LIGHT_KM_S
from kozmos.cosmology import distances

h0lar = st.floats(50, 90)
omega_mler = st.floats(0.05, 0.95)


@given(h0lar, omega_mler)
@settings(max_examples=50, deadline=None)
def test_dusuk_z_limiti_hubble_yasasi(h0, om):
    # z → 0: D_L → c z / H0 ([Hogg1999] Denk. 11 ile tutarlı).
    k = distances.Cosmology.flat(h0, om)
    z = 1e-6
    assert abs(distances.luminosity_distance_mpc(z, k) / (SPEED_OF_LIGHT_KM_S * z / h0) - 1) < 1e-5


@given(h0lar, omega_mler, st.floats(0.0, 0.9))
@settings(max_examples=50, deadline=None)
def test_uzaklik_z_ile_artar(h0, om, ol):
    k = distances.Cosmology(h0, om, ol)
    z = np.linspace(0.01, 3.0, 40)
    assert np.all(np.diff(distances.comoving_distance_mpc(z, k)) > 0)
    assert np.all(np.diff(distances.luminosity_distance_mpc(z, k)) > 0)


@given(h0lar, omega_mler)
@settings(max_examples=50, deadline=None)
def test_egrilik_dallari_duz_limitte_surekli(h0, om):
    # Ωk → 0± için sinh ve sin dalları düz evren değerine yakınsar.
    z = np.array([0.5, 2.0])
    flat = distances.transverse_comoving_distance_mpc(z, distances.Cosmology.flat(h0, om))
    for eps in (1e-7, -1e-7):
        egri = distances.transverse_comoving_distance_mpc(z, distances.Cosmology(h0, om, 1.0 - om - eps))
        assert np.max(np.abs(egri / flat - 1)) < 1e-6


@given(h0lar, omega_mler, st.floats(0.01, 5.0))
@settings(max_examples=50, deadline=None)
def test_uzaklik_bagintilari(h0, om, z):
    # [Hogg1999] Denk. 21: D_L = (1+z) D_M = (1+z)² D_A.
    k = distances.Cosmology.flat(h0, om)
    dl = distances.luminosity_distance_mpc(z, k)
    assert abs(dl / ((1 + z) ** 2 * distances.angular_diameter_distance_mpc(z, k)) - 1) < 1e-13
    # H0 ölçeklemesi: uzaklık 1/H0 ile orantılı.
    assert abs(distances.luminosity_distance_mpc(z, distances.Cosmology.flat(2 * h0, om)) * 2 / dl - 1) < 1e-13
