import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from kozmos.temel.sabitler import CEKIM, GM_GUNES, GUNES_YARICAPI
from kozmos.yildiz import imf, yapi
from kozmos.yildiz import olcekleme as ol

# Güneş kütlesi, kütle parametresinden (yerel tanım).
M_GUNES = GM_GUNES.deger / CEKIM.deger
R_GUNES = GUNES_YARICAPI.deger

kutleler = st.floats(0.1, 50.0)  # M_⊙
yaricaplar = st.floats(0.05, 200.0)  # R_⊙
indisler = st.floats(0.0, 4.5)


@given(indisler, st.floats(0.05, 0.4))
@settings(max_examples=25, deadline=None)
def test_xi1_ve_merkez_yogunluk_orani_n_ile_artar(n, dn):
    # Daha büyük n daha merkezi yoğunlaşmış, daha geniş bir küre verir.
    assert yapi.lane_emden_ilk_sifir(n + dn) > yapi.lane_emden_ilk_sifir(n)
    assert yapi.lane_emden_merkez_ortalama_yogunluk_orani(n + dn) > yapi.lane_emden_merkez_ortalama_yogunluk_orani(n)
    assert yapi.lane_emden_merkez_ortalama_yogunluk_orani(n) >= 1.0 - 1e-12


@given(st.floats(0.0, 4.9))
@settings(max_examples=25, deadline=None)
def test_lane_emden_cozumu_tekduze_ve_sinirli(n):
    xi, theta, dtheta = yapi.lane_emden_coz(n, 200)
    assert theta[0] == 1.0 and dtheta[0] == 0.0
    assert np.all(np.diff(theta) <= 0.0)
    assert np.all(theta >= -1e-9) and abs(theta[-1]) < 1e-8
    assert np.all(dtheta[1:] < 0.0)


@given(kutleler, yaricaplar, st.sampled_from([0.5, 1.0, 1.5, 2.0, 3.0, 4.0]))
@settings(max_examples=40, deadline=None)
def test_politrop_basinci_ve_enerji_isaretleri(m, r, n):
    mk, rm = m * M_GUNES, r * R_GUNES
    assert yapi.politrop_merkez_basinci_pa(mk, rm, n) > 0
    assert yapi.politrop_yercekimi_enerjisi_j(mk, rm, n) < 0
    # Merkez yoğunluk ortalamayı aşar.
    assert yapi.politrop_merkez_yogunlugu_kg_m3(mk, rm, n) >= 3 * mk / (4 * math.pi * rm**3) * (1 - 1e-12)


@given(kutleler, yaricaplar, st.sampled_from([0.5, 1.5, 2.0, 4.0]))
@settings(max_examples=40, deadline=None)
def test_k_yaricap_gidis_donus(m, r, n):
    mk, rm = m * M_GUNES, r * R_GUNES
    k = yapi.politrop_k_katsayisi_si(mk, rm, n)
    assert math.isclose(float(yapi.politrop_yaricap_m(mk, k, n)), rm, rel_tol=1e-9)


@given(yaricaplar, st.floats(2000.0, 50000.0))
@settings(max_examples=60, deadline=None)
def test_isinim_gucu_sicaklik_gidis_donus(r, t):
    rm = r * R_GUNES
    assert math.isclose(float(ol.etkin_sicaklik_k(ol.isinim_gucu_w(rm, t), rm)), t, rel_tol=1e-12)
    # L ∝ T⁴: sıcaklık iki katına çıkınca ışınım gücü 16 katına çıkar.
    assert math.isclose(float(ol.isinim_gucu_w(rm, 2 * t) / ol.isinim_gucu_w(rm, t)), 16.0, rel_tol=1e-12)


@given(st.floats(1e-12, 1e6), st.floats(1.01, 100.0))
@settings(max_examples=60, deadline=None)
def test_serbest_dusme_zamani_yogunlukla_azalir(rho, carpan):
    assert ol.serbest_dusme_zamani_s(rho * carpan) < ol.serbest_dusme_zamani_s(rho)
    assert math.isclose(
        float(ol.serbest_dusme_zamani_s(rho * carpan) / ol.serbest_dusme_zamani_s(rho)), carpan**-0.5, rel_tol=1e-12
    )


@given(st.floats(0.01, 200.0), st.floats(1.01, 5.0), st.floats(0.3, 1.0))
@settings(max_examples=60, deadline=None)
def test_eddington_beta_ve_isinim_monotonluk(kutle, carpan, mu):
    m1, m2 = kutle * M_GUNES, kutle * carpan * M_GUNES
    b1, b2 = float(ol.eddington_beta(m1, mu)), float(ol.eddington_beta(m2, mu))
    assert 0.0 < b2 < b1 <= 1.0
    kappa = 0.034
    l1 = float(ol.standart_model_isinim_gucu_w(m1, mu, kappa))
    l2 = float(ol.standart_model_isinim_gucu_w(m2, mu, kappa))
    assert 0.0 < l1 < l2 < float(ol.eddington_isinim_gucu_w(m2, kappa))


@given(st.floats(0.5, 4.0), st.floats(1.01, 3.0))
@settings(max_examples=40, deadline=None)
def test_chandrasekhar_kutlesi_mu_e_ile_azalir(mu_e, carpan):
    m1 = ol.chandrasekhar_kutlesi_kg(mu_e)
    m2 = ol.chandrasekhar_kutlesi_kg(mu_e * carpan)
    assert m2 < m1
    assert math.isclose(m2 / m1, carpan**-2.0, rel_tol=1e-12)


alt_ust = st.tuples(st.floats(0.011, 0.5), st.floats(2.0, 150.0))


@given(alt_ust, st.floats(1.1, 3.5))
@settings(max_examples=60, deadline=None)
def test_salpeter_kutle_orani_monoton(aralik, alfa):
    a, b = aralik
    esik = np.geomspace(a, b, 25)
    f = imf.salpeter_kutle_orani_ustunde(esik, a, b, alfa)
    assert np.all(np.diff(f) <= 1e-15) and np.all((f >= 0) & (f <= 1))
    assert math.isclose(float(imf.salpeter_kutle_orani_ustunde(a, a, b, alfa)), 1.0, abs_tol=1e-12)
    assert abs(float(imf.salpeter_kutle_orani_ustunde(b, a, b, alfa))) < 1e-12
    # Ortalama kütle aralık içinde kalır.
    assert a < imf.salpeter_ortalama_kutle(a, b, alfa) < b


@given(alt_ust)
@settings(max_examples=40, deadline=None)
def test_kroupa_ve_chabrier_kutle_orani_monoton(aralik):
    a, b = aralik
    esik = np.geomspace(a, b, 25)
    for f in (imf.kroupa_kutle_orani_ustunde(esik, a, b), imf.chabrier_kutle_orani_ustunde(esik, a, b)):
        assert np.all(np.diff(f) <= 1e-15) and np.all((f >= 0) & (f <= 1))
        assert math.isclose(float(f[0]), 1.0, abs_tol=1e-12) and abs(float(f[-1])) < 1e-12
    assert a < imf.kroupa_ortalama_kutle(a, b) < b
    assert a < imf.chabrier_ortalama_kutle(a, b) < b


@given(alt_ust, st.integers(0, 300), st.integers(0, 2**32 - 1))
@settings(max_examples=40, deadline=None)
def test_ornekler_aralik_icinde(aralik, n, tohum):
    a, b = aralik
    for ornekle in (imf.salpeter_ornekle, imf.kroupa_ornekle, imf.chabrier_ornekle):
        x = ornekle(np.random.default_rng(tohum), n, a, b)
        assert x.shape == (n,)
        assert np.all((x >= a * (1 - 1e-12)) & (x <= b * (1 + 1e-12)))
