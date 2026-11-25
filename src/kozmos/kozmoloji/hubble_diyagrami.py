"""Süpernova Hubble diyagramı olabilirliği.

Olabilirlik tanımları Pantheon+ kozmoloji makalesinin arXiv TeX kaynağından
(2202.04077) okunmuştur [Brout2022]; denklemler oradaki TeX etiketleriyle
anılır. Veri sütunlarının anlamı veri deposunun ``README`` dosyasından gelir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cholesky, solve_triangular

from kozmos._kaynak import kaynak
from kozmos.kozmoloji.uzaklik import Kozmoloji, uzaklik_modulu

Dizi = NDArray[np.float64]
Maske = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class HubbleVerisi:
    """Olabilirlik için hazırlanmış süpernova verisi.

    Öznitelikler
    ------------
    z : ndarray
        Hubble diyagramı kırmızıya kayması (``zHD``).
    kadir : ndarray
        Standartlaştırılmış kadir (``m_b_corr``).
    kalibrator : ndarray[bool]
        Sefeid uzaklığı olan konaklardaki süpernovalar (``IS_CALIBRATOR``).
    sefeid_modulu : ndarray
        Sefeid konak uzaklık modülü (``CEPH_DIST``); kalibratör olmayanlarda kullanılmaz.
    cholesky_alt : ndarray
        Seçilen satırların kovaryansının alt üçgen Cholesky çarpanı.
    """

    z: Dizi
    kadir: Dizi
    kalibrator: Maske
    sefeid_modulu: Dizi
    cholesky_alt: Dizi

    def __len__(self) -> int:
        return int(self.z.shape[0])


@kaynak(
    "Brout2022",
    yer="Bölüm 4: z < ~0,01 Hubble diyagramı kullanılmaz; eq:dmuprime",
    teyit="okundu",
    not_="eşik 0,01 makalede '~0.01' diye geçer; kesin değer bu projenin seçimidir",
)
def veri_hazirla(
    sutunlar: dict[str, np.ndarray],
    kovaryans: ArrayLike,
    *,
    z_alt: float = 0.01,
    kalibratorler: bool,
) -> HubbleVerisi:
    """Satırları seçer ve kovaryans alt matrisinin Cholesky çarpanını hesaplar.

    Parametreler
    ------------
    sutunlar : dict
        ``kozmos.arsiv.pantheon.tablo_oku`` çıktısı.
    kovaryans : ndarray (N, N)
    z_alt : float
        Kalibratör olmayan süpernovalar için alt kırmızıya kayma sınırı.
    kalibratorler : bool
        Doğruysa Sefeid konaklarındaki süpernovalar (kırmızıya kaymalarından
        bağımsız olarak) eklenir; yanlışsa bütünüyle dışarıda bırakılır.

    Kaynaklar
    ---------
    [Brout2022] Bölüm 4, eq:dmuprime.

    Geçerlilik
    ----------
    Özel hızların baskın olduğu yakın süpernovalar yalnızca Sefeid çapası
    olarak kullanılır; Hubble akışına katılmaz.

    Doğrulama
    ---------
    tests/birim/test_hubble_diyagrami.py (sentetik veri).
    """
    z = np.asarray(sutunlar["zHD"], dtype=float)
    kal = np.asarray(sutunlar["IS_CALIBRATOR"], dtype=float) == 1.0
    secim = (z > z_alt) & ~kal
    if kalibratorler:
        secim = secim | kal
    c = np.asarray(kovaryans, dtype=float)[np.ix_(secim, secim)]
    return HubbleVerisi(
        z=z[secim],
        kadir=np.asarray(sutunlar["m_b_corr"], dtype=float)[secim],
        kalibrator=kal[secim],
        sefeid_modulu=np.asarray(sutunlar["CEPH_DIST"], dtype=float)[secim],
        cholesky_alt=cholesky(c, lower=True),
    )


@kaynak(
    "Brout2022",
    yer="eq:dmu, eq:dmuprime, eq:dl",
    teyit="okundu",
    not_="μ_i = m_b_corr,i − M; kalibratörde model yerine Sefeid uzaklık modülü",
)
def artik_vektoru(veri: HubbleVerisi, kozmoloji: Kozmoloji, mutlak_kadir: float) -> Dizi:
    """Veri eksi model artığını hesaplar.

    Kalibratör: ΔD_i = (m_i − M) − μ_Sefeid,i. Diğerleri: ΔD_i = (m_i − M) − μ_model(z_i).

    Kaynaklar
    ---------
    [Brout2022] eq:dmu, eq:dmuprime.

    Geçerlilik
    ----------
    Model uzaklığı [Brout2022] eq:dl'deki gibi tek kırmızıya kaymayla (``zHD``)
    hesaplanır; güneş merkezli kırmızıya kaymayla (1 + z_hel) çarpanı düzeltmesi
    uygulanmaz.

    Doğrulama
    ---------
    tests/birim/test_hubble_diyagrami.py
    """
    mu = veri.kadir - mutlak_kadir
    model = np.where(veri.kalibrator, veri.sefeid_modulu, 0.0)
    akis = ~veri.kalibrator
    model[akis] = uzaklik_modulu(veri.z[akis], kozmoloji)
    return mu - model


@kaynak(
    "Brout2022",
    yer="eq:likelihood, eq:sh0eslikelihood",
    teyit="okundu",
    not_="−2 ln L = ΔDᵀ C⁻¹ ΔD; C veri sürümündeki STAT+SYS matrisi",
)
def log_olabilirlik(veri: HubbleVerisi, kozmoloji: Kozmoloji, mutlak_kadir: float) -> float:
    """Gauss log olabilirliğini (sabit terim hariç) hesaplar: ln L = −χ²/2.

    Kaynaklar
    ---------
    [Brout2022] eq:likelihood, eq:sh0eslikelihood.

    Doğrulama
    ---------
    tests/birim/test_hubble_diyagrami.py: sentetik veride girilen parametrelerin geri kazanımı.
    """
    y = solve_triangular(veri.cholesky_alt, artik_vektoru(veri, kozmoloji, mutlak_kadir), lower=True)
    return -0.5 * float(y @ y)


@kaynak(
    "Brout2022",
    yer="eq:likelihood",
    teyit="okundu",
    not_="M doğrusal parametredir; sabit kozmolojide en iyi M ağırlıklı ortalamadır (türetim bu projede)",
)
def en_iyi_mutlak_kadir(veri: HubbleVerisi, kozmoloji: Kozmoloji) -> tuple[float, float, float]:
    """Sabit kozmolojide χ²'yi en küçükleyen M değerini kapalı biçimde bulur.

    ΔD = r − M·1 için M̂ = (1ᵀC⁻¹r)/(1ᵀC⁻¹1), σ_M² = 1/(1ᵀC⁻¹1).

    Döndürür
    --------
    (M̂, σ_M, χ²_min)

    Kaynaklar
    ---------
    [Brout2022] eq:likelihood.

    Doğrulama
    ---------
    tests/birim/test_hubble_diyagrami.py: sayısal en küçükleme ile karşılaştırma.
    """
    r = solve_triangular(veri.cholesky_alt, artik_vektoru(veri, kozmoloji, 0.0), lower=True)
    bir = solve_triangular(veri.cholesky_alt, np.ones(len(veri)), lower=True)
    payda = float(bir @ bir)
    m = float(bir @ r) / payda
    kalan = r - m * bir
    return m, payda**-0.5, float(kalan @ kalan)
