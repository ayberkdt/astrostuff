import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from kozmos.kozmoloji import uzaklik as uz
from kozmos.temel.sabitler import ISIK_HIZI_KM_S

h0lar = st.floats(50, 90)
omega_mler = st.floats(0.05, 0.95)


@given(h0lar, omega_mler)
@settings(max_examples=50, deadline=None)
def test_dusuk_z_limiti_hubble_yasasi(h0, om):
    # z → 0: D_L → c z / H0 ([Hogg1999] Denk. 11 ile tutarlı).
    k = uz.Kozmoloji.duz(h0, om)
    z = 1e-6
    assert abs(uz.isinim_gucu_uzakligi_mpc(z, k) / (ISIK_HIZI_KM_S * z / h0) - 1) < 1e-5


@given(h0lar, omega_mler, st.floats(0.0, 0.9))
@settings(max_examples=50, deadline=None)
def test_uzaklik_z_ile_artar(h0, om, ol):
    k = uz.Kozmoloji(h0, om, ol)
    z = np.linspace(0.01, 3.0, 40)
    assert np.all(np.diff(uz.esyuruyen_uzaklik_mpc(z, k)) > 0)
    assert np.all(np.diff(uz.isinim_gucu_uzakligi_mpc(z, k)) > 0)


@given(h0lar, omega_mler)
@settings(max_examples=50, deadline=None)
def test_egrilik_dallari_duz_limitte_surekli(h0, om):
    # Ωk → 0± için sinh ve sin dalları düz evren değerine yakınsar.
    z = np.array([0.5, 2.0])
    duz = uz.enine_esyuruyen_uzaklik_mpc(z, uz.Kozmoloji.duz(h0, om))
    for eps in (1e-7, -1e-7):
        egri = uz.enine_esyuruyen_uzaklik_mpc(z, uz.Kozmoloji(h0, om, 1.0 - om - eps))
        assert np.max(np.abs(egri / duz - 1)) < 1e-6


@given(h0lar, omega_mler, st.floats(0.01, 5.0))
@settings(max_examples=50, deadline=None)
def test_uzaklik_bagintilari(h0, om, z):
    # [Hogg1999] Denk. 21: D_L = (1+z) D_M = (1+z)² D_A.
    k = uz.Kozmoloji.duz(h0, om)
    dl = uz.isinim_gucu_uzakligi_mpc(z, k)
    assert abs(dl / ((1 + z) ** 2 * uz.acisal_cap_uzakligi_mpc(z, k)) - 1) < 1e-13
    # H0 ölçeklemesi: uzaklık 1/H0 ile orantılı.
    assert abs(uz.isinim_gucu_uzakligi_mpc(z, uz.Kozmoloji.duz(2 * h0, om)) * 2 / dl - 1) < 1e-13
