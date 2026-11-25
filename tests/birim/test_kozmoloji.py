"""Kaynaktaki sayısal değerler ve kapalı çözümler [Hogg1999]."""

import numpy as np
import pytest

from kozmos.kozmoloji import uzaklik as uz

EDS = uz.Kozmoloji.duz(100.0, 1.0)  # Einstein-de Sitter, h = 1


def test_hubble_uzakligi_ve_zamani_makaledeki_degerler():
    # [Hogg1999] Denk. 4: D_H = 3000 h^-1 Mpc; Denk. 3: t_H = 9,78e9 h^-1 yıl (üç anlamlı basamak).
    assert round(uz.hubble_uzakligi_mpc(EDS), -1) == 3000.0
    assert round(uz.hubble_zamani_gyr(EDS), 2) == 9.78


def test_einstein_de_sitter_kapali_cozum():
    # Ωm = 1, ΩΛ = 0: E(z) = (1+z)^{3/2}; D_C = 2 D_H [1 − (1+z)^{-1/2}].
    z = np.array([0.0, 0.1, 1.0, 3.0, 10.0])
    beklenen = 2.0 * uz.hubble_uzakligi_mpc(EDS) * (1.0 - (1.0 + z) ** -0.5)
    assert np.allclose(uz.esyuruyen_uzaklik_mpc(z, EDS), beklenen, rtol=1e-12, atol=1e-12)
    # Geriye bakış zamanı: t_L = (2/3) t_H [1 − (1+z)^{-3/2}].
    beklenen_t = (2.0 / 3.0) * uz.hubble_zamani_gyr(EDS) * (1.0 - (1.0 + z) ** -1.5)
    assert np.allclose(uz.geriye_bakis_zamani_gyr(z, EDS), beklenen_t, rtol=1e-12, atol=1e-14)


def test_hogg_denklem_17_lambda_sifir():
    # [Hogg1999] Denk. 17: ΩΛ = 0 için D_M'nin kapalı biçimi.
    om = 0.3
    k = uz.Kozmoloji(70.0, om, 0.0)
    z = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    dh = uz.hubble_uzakligi_mpc(k)
    kapali = dh * 2 * (2 - om * (1 - z) - (2 - om) * np.sqrt(1 + om * z)) / (om**2 * (1 + z))
    assert np.allclose(uz.enine_esyuruyen_uzaklik_mpc(z, k), kapali, rtol=1e-10)


def test_omega_k():
    assert uz.Kozmoloji.duz(70, 0.3).omega_k == pytest.approx(0.0, abs=1e-15)
    assert uz.Kozmoloji(70, 0.3, 0.6).omega_k == pytest.approx(0.1)


def test_gecersiz_girdi():
    with pytest.raises(ValueError):
        uz.Kozmoloji(-1.0, 0.3, 0.7)
    with pytest.raises(ValueError):
        uz.esyuruyen_uzaklik_mpc(-0.1, EDS)


def test_skaler_ve_dizi():
    k = uz.Kozmoloji.duz(70.0, 0.3)
    assert np.ndim(uz.isinim_gucu_uzakligi_mpc(0.5, k)) == 0
    assert uz.isinim_gucu_uzakligi_mpc(np.zeros((2, 3)) + 0.5, k).shape == (2, 3)


def test_doppler_ve_ozel_hiz():
    from kozmos.temel.sabitler import ISIK_HIZI_KM_S

    v = np.array([-3000.0, -300.0, 0.0, 300.0, 30000.0, 150000.0])
    z = uz.doppler_kirmiziya_kayma(v)
    assert np.allclose(uz.doppler_hizi_km_s(z), v, rtol=1e-12, atol=1e-9)
    assert float(uz.doppler_kirmiziya_kayma(30.0)) == pytest.approx(30.0 / ISIK_HIZI_KM_S, rel=1e-4)  # z ≈ v/c
    # (1 + z_göz) = (1 + z_koz)(1 + v/c) kuralından üretilen z_göz, küçük hızda aynı v'yi geri verir.
    z_koz, hiz = 0.05, 300.0
    z_goz = (1 + z_koz) * (1 + hiz / ISIK_HIZI_KM_S) - 1
    assert float(uz.ozel_hiz_km_s(z_goz, z_koz)) == pytest.approx(hiz, rel=1e-12)
