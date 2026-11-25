"""Hubble diyagramı olabilirliğinin sentetik veriyle sınanması.

Bilinen (H0, Ωm, M) ile üretilen sahte bir süpernova kataloğunda olabilirlik
girilen parametreleri geri vermelidir.
"""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from kozmos.cosmology import hubble_diagram
from kozmos.cosmology.distances import Cosmology, distance_modulus

GERCEK = {"h0": 72.0, "omega_m": 0.31, "m": -19.3}


def _sentetik(tohum=31, n_akis=400, n_kal=30):
    u = np.random.default_rng(tohum)
    cosmology = Cosmology.flat(GERCEK["h0"], GERCEK["omega_m"])
    z_akis = np.sort(u.uniform(0.012, 1.5, n_akis))
    z_kal = u.uniform(0.001, 0.009, n_kal)
    z_yakin = u.uniform(0.002, 0.009, 20)  # ne kalibratör ne akış: atılmalı
    z = np.concatenate([z_kal, z_yakin, z_akis])
    n = z.size
    kal = np.zeros(n, bool)
    kal[:n_kal] = True
    sefeid = np.where(kal, u.uniform(29, 33, n), -9.0)
    mu = np.where(kal, sefeid, distance_modulus(z, cosmology))
    a = u.normal(0, 0.02, (n, 5))
    kov = np.diag(u.uniform(0.1, 0.2, n) ** 2) + a @ a.T  # köşegen + düşük ranklı ilintili terim
    magnitude = mu + GERCEK["m"] + u.multivariate_normal(np.zeros(n), kov)
    columns = {"zHD": z, "IS_CALIBRATOR": kal.astype(float), "m_b_corr": magnitude, "CEPH_DIST": sefeid}
    return columns, kov


def test_secim():
    columns, kov = _sentetik()
    yalniz_sn = hubble_diagram.prepare_data(columns, kov, calibrators=False)
    sefeidli = hubble_diagram.prepare_data(columns, kov, calibrators=True)
    assert len(yalniz_sn) == 400 and not yalniz_sn.calibrator.any()
    assert len(sefeidli) == 430 and sefeidli.calibrator.sum() == 30
    assert np.all(yalniz_sn.z > 0.01)


def test_kapali_mutlak_kadir_sayisal_en_kucukleme_ile():
    columns, kov = _sentetik()
    data = hubble_diagram.prepare_data(columns, kov, calibrators=True)
    cosmology = Cosmology.flat(GERCEK["h0"], GERCEK["omega_m"])
    m, sigma, ki2 = hubble_diagram.best_absolute_magnitude(data, cosmology)
    sayisal = minimize_scalar(
        lambda x: -hubble_diagram.log_likelihood(data, cosmology, x),
        bounds=(-20, -18.5),
        method="bounded",
        options={"xatol": 1e-10},
    )
    assert m == pytest.approx(sayisal.x, abs=1e-7)
    assert ki2 == pytest.approx(2 * sayisal.fun, rel=1e-9)
    assert abs(m - GERCEK["m"]) < 4 * sigma
    # Δχ² = 1 tanımı: M̂ ± σ_M noktasında χ² bir artar.
    assert -2 * hubble_diagram.log_likelihood(data, cosmology, m + sigma) - ki2 == pytest.approx(1.0, rel=1e-6)


def test_parametre_geri_kazanimi():
    columns, kov = _sentetik()
    data = hubble_diagram.prepare_data(columns, kov, calibrators=True)

    def profil(om):
        # Her Ωm için H0 ve M üzerinden en küçük χ²: iç içe tek boyutlu en küçükleme.
        ic = minimize_scalar(
            lambda h0: hubble_diagram.best_absolute_magnitude(data, Cosmology.flat(h0, om))[2],
            bounds=(60, 85),
            method="bounded",
        )
        return ic.fun, ic.x

    izgara = np.linspace(0.2, 0.45, 26)
    sonuc = [profil(om) for om in izgara]
    ki2 = np.array([s[0] for s in sonuc])
    en_iyi = int(np.argmin(ki2))
    icinde = izgara[ki2 - ki2.min() < 4.0]  # yaklaşık 2σ aralığı
    assert icinde.min() <= GERCEK["omega_m"] <= icinde.max()
    assert abs(sonuc[en_iyi][1] - GERCEK["h0"]) < 3.0
    assert ki2.min() / (len(data) - 3) == pytest.approx(1.0, abs=0.2)


def test_h0_ve_m_dejenerasyonu_sefeidsiz():
    # Kalibratör yokken H0 değişimi M kaymasıyla tam telafi edilir: χ²_min H0'dan bağımsızdır.
    columns, kov = _sentetik()
    data = hubble_diagram.prepare_data(columns, kov, calibrators=False)
    m1, _, ki1 = hubble_diagram.best_absolute_magnitude(data, Cosmology.flat(60.0, 0.31))
    m2, _, ki2 = hubble_diagram.best_absolute_magnitude(data, Cosmology.flat(80.0, 0.31))
    assert ki1 == pytest.approx(ki2, rel=1e-9)
    # H0 büyüyünce model uzaklığı küçülür, μ azalır; M = m − μ aynı miktarda artar.
    assert m2 - m1 == pytest.approx(5 * np.log10(80.0 / 60.0), abs=1e-8)
