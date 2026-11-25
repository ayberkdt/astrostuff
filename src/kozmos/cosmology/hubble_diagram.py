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

from kozmos._sources import source
from kozmos.cosmology.distances import Cosmology, distance_modulus

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class HubbleData:
    """Olabilirlik için hazırlanmış süpernova verisi.

    Attributes
    ----------
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

    z: Array
    magnitude: Array
    calibrator: Mask
    cepheid_modulus: Array
    cholesky_lower: Array

    def __len__(self) -> int:
        return int(self.z.shape[0])


@source(
    "Brout2022",
    where="Bölüm 4: z < ~0,01 Hubble diyagramı kullanılmaz; eq:dmuprime",
    status="read",
    note="eşik 0,01 makalede '~0.01' diye geçer; kesin değer bu projenin seçimidir",
)
def prepare_data(
    columns: dict[str, np.ndarray],
    covariance: ArrayLike,
    *,
    z_min: float = 0.01,
    calibrators: bool,
) -> HubbleData:
    """Satırları seçer ve kovaryans alt matrisinin Cholesky çarpanını hesaplar.

    Parameters
    ----------
    sutunlar : dict
        ``kozmos.archives.pantheon.tablo_oku`` çıktısı.
    kovaryans : ndarray (N, N)
    z_alt : float
        Kalibratör olmayan süpernovalar için alt kırmızıya kayma sınırı.
    kalibratorler : bool
        Doğruysa Sefeid konaklarındaki süpernovalar (kırmızıya kaymalarından
        bağımsız olarak) eklenir; yanlışsa bütünüyle dışarıda bırakılır.

    References
    ----------
    [Brout2022] Bölüm 4, eq:dmuprime.

    Validity
    --------
    Özel hızların baskın olduğu yakın süpernovalar yalnızca Sefeid çapası
    olarak kullanılır; Hubble akışına katılmaz.

    Verification
    ------------
    tests/unit/test_hubble_diagram.py (sentetik veri).
    """
    z = np.asarray(columns["zHD"], dtype=float)
    kal = np.asarray(columns["IS_CALIBRATOR"], dtype=float) == 1.0
    secim = (z > z_min) & ~kal
    if calibrators:
        secim = secim | kal
    c = np.asarray(covariance, dtype=float)[np.ix_(secim, secim)]
    return HubbleData(
        z=z[secim],
        magnitude=np.asarray(columns["m_b_corr"], dtype=float)[secim],
        calibrator=kal[secim],
        cepheid_modulus=np.asarray(columns["CEPH_DIST"], dtype=float)[secim],
        cholesky_lower=cholesky(c, lower=True),
    )


@source(
    "Brout2022",
    where="eq:dmu, eq:dmuprime, eq:dl",
    status="read",
    note="μ_i = m_b_corr,i − M; kalibratörde model yerine Sefeid uzaklık modülü",
)
def residual_vector(data: HubbleData, cosmology: Cosmology, absolute_magnitude: float) -> Array:
    """Veri eksi model artığını hesaplar.

    Kalibratör: ΔD_i = (m_i − M) − μ_Sefeid,i. Diğerleri: ΔD_i = (m_i − M) − μ_model(z_i).

    References
    ----------
    [Brout2022] eq:dmu, eq:dmuprime.

    Validity
    --------
    Model uzaklığı [Brout2022] eq:dl'deki gibi tek kırmızıya kaymayla (``zHD``)
    hesaplanır; güneş merkezli kırmızıya kaymayla (1 + z_hel) çarpanı düzeltmesi
    uygulanmaz.

    Verification
    ------------
    tests/unit/test_hubble_diagram.py
    """
    mu = data.magnitude - absolute_magnitude
    model = np.where(data.calibrator, data.cepheid_modulus, 0.0)
    akis = ~data.calibrator
    model[akis] = distance_modulus(data.z[akis], cosmology)
    return mu - model


@source(
    "Brout2022",
    where="eq:likelihood, eq:sh0eslikelihood",
    status="read",
    note="−2 ln L = ΔDᵀ C⁻¹ ΔD; C veri sürümündeki STAT+SYS matrisi",
)
def log_likelihood(data: HubbleData, cosmology: Cosmology, absolute_magnitude: float) -> float:
    """Gauss log olabilirliğini (sabit terim hariç) hesaplar: ln L = −χ²/2.

    References
    ----------
    [Brout2022] eq:likelihood, eq:sh0eslikelihood.

    Verification
    ------------
    tests/unit/test_hubble_diagram.py: sentetik veride girilen parametrelerin geri kazanımı.
    """
    y = solve_triangular(data.cholesky_lower, residual_vector(data, cosmology, absolute_magnitude), lower=True)
    return -0.5 * float(y @ y)


@source(
    "Brout2022",
    where="eq:likelihood",
    status="read",
    note="M doğrusal parametredir; sabit kozmolojide en iyi M ağırlıklı ortalamadır (türetim bu projede)",
)
def best_absolute_magnitude(data: HubbleData, cosmology: Cosmology) -> tuple[float, float, float]:
    """Sabit kozmolojide χ²'yi en küçükleyen M değerini kapalı biçimde bulur.

    ΔD = r − M·1 için M̂ = (1ᵀC⁻¹r)/(1ᵀC⁻¹1), σ_M² = 1/(1ᵀC⁻¹1).

    Returns
    -------
    (M̂, σ_M, χ²_min)

    References
    ----------
    [Brout2022] eq:likelihood.

    Verification
    ------------
    tests/unit/test_hubble_diagram.py: sayısal en küçükleme ile karşılaştırma.
    """
    r = solve_triangular(data.cholesky_lower, residual_vector(data, cosmology, 0.0), lower=True)
    bir = solve_triangular(data.cholesky_lower, np.ones(len(data)), lower=True)
    payda = float(bir @ bir)
    m = float(bir @ r) / payda
    kalan = r - m * bir
    return m, payda**-0.5, float(kalan @ kalan)
