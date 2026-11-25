"""Kozmolojik uzaklık ölçüleri.

Tanımlar Hogg'un derlemesini izler [Hogg1999]; denklemler makalenin arXiv TeX
kaynağından (astro-ph/9905116) 25 Kasım 2026'da okunmuş ve numaraları oradaki
sıraya göre verilmiştir. Karanlık enerji durum denklemi parametresi ``w`` için
genelleme Pantheon+ analizindeki H(z) ifadesinden alınmıştır [Brout2022].

Radyasyon terimi yoktur: [Hogg1999] Denk. 14 yalnızca madde, eğrilik ve
kozmolojik sabit içerir. Bu yüzden sonuçlar ``z`` birkaç yüzü aştığında
fiziksel evrenden sapar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.constants import JULIAN_YEAR_SECONDS, MEGAPARSEC, SPEED_OF_LIGHT_KM_S

Array = NDArray[np.float64]

# Gauss-Legendre düğümleri: [0, z] üzerinde 1/E(z') integrali için.
_DUGUM, _AGIRLIK = np.polynomial.legendre.leggauss(64)


@dataclass(frozen=True, slots=True)
class Cosmology:
    """Friedmann-Lemaître-Robertson-Walker evreninin parametreleri.

    Attributes
    ----------
    h0 : float
        Hubble sabiti [km s^-1 Mpc^-1].
    omega_m : float
        Madde yoğunluk parametresi.
    omega_lambda : float
        Karanlık enerji yoğunluk parametresi.
    w : float
        Karanlık enerji durum denklemi parametresi; −1 kozmolojik sabittir.
    """

    h0: float
    omega_m: float
    omega_lambda: float
    w: float = -1.0

    def __post_init__(self) -> None:
        if self.h0 <= 0:
            raise ValueError("H0 pozitif olmalı")
        if self.omega_m < 0:
            raise ValueError("Ωm eksi olamaz")

    @property
    def omega_k(self) -> float:
        """Eğrilik yoğunluk parametresi: Ωk = 1 − Ωm − ΩΛ ([Hogg1999] Denk. 7)."""
        return 1.0 - self.omega_m - self.omega_lambda

    @classmethod
    def flat(cls, h0: float, omega_m: float, w: float = -1.0) -> Cosmology:
        """Düz evren kurar: ΩΛ = 1 − Ωm."""
        return cls(h0=h0, omega_m=omega_m, omega_lambda=1.0 - omega_m, w=w)


@source("Hogg1999", where="Denk. 4", status="read")
def hubble_distance_mpc(cosmology: Cosmology) -> float:
    """Hubble uzaklığını hesaplar: D_H = c / H0.

    Returns
    -------
    float
        Hubble uzaklığı [Mpc].

    References
    ----------
    [Hogg1999] Denk. 4.

    Verification
    ------------
    tests/unit/test_cosmology.py: h = 1 için 3000 Mpc dolayı (makaledeki değer).
    """
    return SPEED_OF_LIGHT_KM_S / cosmology.h0


@source("Hogg1999", where="Denk. 3", status="read")
def hubble_time_gyr(cosmology: Cosmology) -> float:
    """Hubble zamanını hesaplar: t_H = 1 / H0.

    Returns
    -------
    float
        Hubble zamanı [Gyr, Jülyen yılı].

    References
    ----------
    [Hogg1999] Denk. 3.

    Verification
    ------------
    tests/unit/test_cosmology.py: h = 1 için 9,78 Gyr (makaledeki değer).
    """
    h0_si = cosmology.h0 * 1e3 / MEGAPARSEC.value  # s^-1
    return 1.0 / h0_si / JULIAN_YEAR_SECONDS.value / 1e9


@source("Hogg1999", where="Denk. 14", status="read")
@source(
    "Brout2022",
    where="H(z) denklemi (TeX etiketi eq:hz)",
    status="read",
    note="ΩΛ terimi (1+z)^{3(1+w)} ile çarpılır; Brout düz evren için yazar, eğrilik terimi Hogg'dan",
)
def e_function(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Boyutsuz Hubble parametresini hesaplar: E(z) = H(z)/H0.

    E(z) = sqrt( Ωm (1+z)³ + Ωk (1+z)² + ΩΛ (1+z)^{3(1+w)} )

    Parameters
    ----------
    z : float ya da ndarray
        Kırmızıya kayma, z > −1.
    kozmoloji : Kozmoloji

    References
    ----------
    [Hogg1999] Denk. 14; [Brout2022] H(z) denklemi.

    Validity
    --------
    Radyasyon terimi yoktur.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py: astropy.cosmology efunc ile bağıl fark < 1e-12.
    """
    zp1 = 1.0 + np.asarray(z, dtype=float)
    k = cosmology
    return np.sqrt(k.omega_m * zp1**3 + k.omega_k * zp1**2 + k.omega_lambda * zp1 ** (3.0 * (1.0 + k.w)))


def _integral_from_zero(z: Array, integrand) -> Array:  # type: ignore[no-untyped-def]
    """[0, z] üzerinde 64 noktalı Gauss-Legendre integrali; z dizisi için tek geçiş."""
    yari = 0.5 * z[..., None]
    dugum = yari * (_DUGUM + 1.0)
    return np.sum(_AGIRLIK * integrand(dugum), axis=-1) * yari[..., 0]


@source("Hogg1999", where="Denk. 15", status="read")
def comoving_distance_mpc(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Bakış doğrultusundaki eşyürüyen uzaklığı hesaplar.

    D_C = D_H ∫₀ᶻ dz′ / E(z′)

    Returns
    -------
    float ya da ndarray
        Eşyürüyen uzaklık [Mpc].

    References
    ----------
    [Hogg1999] Denk. 15.

    Validity
    --------
    0 <= z <= 10 aralığında 64 noktalı Gauss-Legendre integrali astropy ile
    1e-9 bağıl farkın altındadır; daha büyük z sınanmamıştır.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py; tests/unit/test_cosmology.py (Einstein-de Sitter kapalı çözümü).
    """
    zz = np.asarray(z, dtype=float)
    if np.any(zz < 0):
        raise ValueError("z eksi olamaz")
    return hubble_distance_mpc(cosmology) * _integral_from_zero(zz, lambda x: 1.0 / e_function(x, cosmology))


@source("Hogg1999", where="Denk. 16", status="read")
def transverse_comoving_distance_mpc(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Enine eşyürüyen uzaklığı hesaplar.

    Ωk > 0: D_M = D_H/√Ωk · sinh(√Ωk D_C/D_H); Ωk = 0: D_M = D_C;
    Ωk < 0: D_M = D_H/√|Ωk| · sin(√|Ωk| D_C/D_H).

    References
    ----------
    [Hogg1999] Denk. 16.

    Validity
    --------
    |Ωk| < 1e-12 düz evren sayılır; dallar bu sınırda süreklidir.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py; tests/property/test_cosmology_property.py (Ωk → 0 sürekliliği).
    """
    dc = comoving_distance_mpc(z, cosmology)
    ok = cosmology.omega_k
    if abs(ok) < 1e-12:
        return dc
    dh = hubble_distance_mpc(cosmology)
    kok = np.sqrt(abs(ok))
    if ok > 0:
        return dh / kok * np.sinh(kok * dc / dh)
    return dh / kok * np.sin(kok * dc / dh)


@source("Hogg1999", where="Denk. 18", status="read")
def angular_diameter_distance_mpc(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Açısal çap uzaklığını hesaplar: D_A = D_M / (1 + z).

    References
    ----------
    [Hogg1999] Denk. 18.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py
    """
    return transverse_comoving_distance_mpc(z, cosmology) / (1.0 + np.asarray(z, dtype=float))


@source("Hogg1999", where="Denk. 21", status="read")
def luminosity_distance_mpc(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Işınım gücü uzaklığını hesaplar: D_L = (1 + z) D_M.

    References
    ----------
    [Hogg1999] Denk. 21.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py; tests/property/test_cosmology_property.py (z → 0 limitinde cz/H0).
    """
    return (1.0 + np.asarray(z, dtype=float)) * transverse_comoving_distance_mpc(z, cosmology)


@source("Hogg1999", where="Denk. 25", status="read", note="D_L Mpc ise μ = 5 log10(D_L) + 25")
def distance_modulus(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Uzaklık modülünü hesaplar: μ = 5 log10(D_L / 10 pc).

    Returns
    -------
    float ya da ndarray
        Uzaklık modülü [mag].

    References
    ----------
    [Hogg1999] Denk. 25.

    Validity
    --------
    z > 0; z = 0 için −∞ döner.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py: astropy distmod ile fark < 1e-9 mag.
    """
    with np.errstate(divide="ignore"):
        return 5.0 * np.log10(luminosity_distance_mpc(z, cosmology)) + 25.0


@source("Hogg1999", where="Denk. 30", status="read")
def lookback_time_gyr(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Geriye bakış zamanını hesaplar: t_L = t_H ∫₀ᶻ dz′ / [(1+z′) E(z′)].

    Returns
    -------
    float ya da ndarray
        Geriye bakış zamanı [Gyr].

    References
    ----------
    [Hogg1999] Denk. 30.

    Validity
    --------
    0 <= z <= 10 aralığında sınanmıştır.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py: astropy lookback_time ile bağıl fark < 1e-8.
    """
    zz = np.asarray(z, dtype=float)
    if np.any(zz < 0):
        raise ValueError("z eksi olamaz")
    return hubble_time_gyr(cosmology) * _integral_from_zero(zz, lambda x: 1.0 / ((1.0 + x) * e_function(x, cosmology)))


@source("Hogg1999", where="Denk. 28", status="read")
def comoving_volume_element_mpc3(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Birim katı açı ve birim z başına eşyürüyen hacmi hesaplar.

    dV_C / (dΩ dz) = D_H (1+z)² D_A² / E(z)

    Returns
    -------
    float ya da ndarray
        Hacim elemanı [Mpc³ sr⁻¹].

    References
    ----------
    [Hogg1999] Denk. 28.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py: astropy differential_comoving_volume.
    """
    zz = np.asarray(z, dtype=float)
    da = angular_diameter_distance_mpc(zz, cosmology)
    return hubble_distance_mpc(cosmology) * (1.0 + zz) ** 2 * da**2 / e_function(zz, cosmology)


@source("Hogg1999", where="Denk. 19", status="read", note="kaynak, formülün Ωk < 0 için geçerli olmadığını belirtir")
def angular_diameter_distance_z1_z2_mpc(z1: ArrayLike, z2: ArrayLike, cosmology: Cosmology) -> Array:
    """z1'deki gözlemcinin z2'deki cisme ölçtüğü açısal çap uzaklığını hesaplar.

    D_A12 = [D_M2 sqrt(1 + Ωk D_M1²/D_H²) − D_M1 sqrt(1 + Ωk D_M2²/D_H²)] / (1 + z2)

    References
    ----------
    [Hogg1999] Denk. 19.

    Validity
    --------
    Ωk >= 0 ve z1 <= z2. Çekimsel merceklenmede mercek-kaynak uzaklığı için kullanılır.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py: astropy angular_diameter_distance_z1z2.
    """
    if cosmology.omega_k < -1e-12:
        raise ValueError("Denk. 19 kapalı evrende (Ωk < 0) geçerli değil")
    ok = max(cosmology.omega_k, 0.0)
    dh = hubble_distance_mpc(cosmology)
    dm1 = transverse_comoving_distance_mpc(z1, cosmology)
    dm2 = transverse_comoving_distance_mpc(z2, cosmology)
    ic = dm2 * np.sqrt(1.0 + ok * dm1**2 / dh**2) - dm1 * np.sqrt(1.0 + ok * dm2**2 / dh**2)
    return ic / (1.0 + np.asarray(z2, dtype=float))


@source("Hogg1999", where="Denk. 29", status="read", note="tüm gökyüzü hacmi; üç eğrilik dalı")
def comoving_volume_mpc3(z: ArrayLike, cosmology: Cosmology) -> Array:
    """z'ye kadar tüm gökyüzünün eşyürüyen hacmini hesaplar.

    Düz evrende V_C = (4π/3) D_M³; eğrilikli evrende kaynaktaki kapalı biçim.

    Returns
    -------
    float ya da ndarray
        Eşyürüyen hacim [Mpc³].

    References
    ----------
    [Hogg1999] Denk. 29.

    Validity
    --------
    |Ωk| < 1e-8 düz evren sayılır: eğrilikli dalda iki büyük terimin farkı
    küçük |Ωk| için duyarlılık kaybeder.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py: astropy comoving_volume.
    """
    dm = transverse_comoving_distance_mpc(z, cosmology)
    ok = cosmology.omega_k
    if abs(ok) < 1e-8:
        return 4.0 * np.pi / 3.0 * dm**3
    dh = hubble_distance_mpc(cosmology)
    x = dm / dh
    kok = np.sqrt(abs(ok))
    inverse = np.arcsinh(kok * x) if ok > 0 else np.arcsin(kok * x)
    return 4.0 * np.pi * dh**3 / (2.0 * ok) * (x * np.sqrt(1.0 + ok * x**2) - inverse / kok)


@source("Hogg1999", where="Denk. 9", status="read")
def doppler_redshift(velocity_km_s: ArrayLike) -> Array:
    """Bakış doğrultusundaki hızdan görelilikli Doppler kırmızıya kaymasını hesaplar.

    1 + z = sqrt((1 + v/c) / (1 − v/c))

    References
    ----------
    [Hogg1999] Denk. 9.

    Validity
    --------
    |v| < c. Kozmolojik kırmızıya kayma bu formülle hıza çevrilmez; kaynak
    bunun yalnızca özel hızlar için anlamlı olduğunu vurgular.

    Verification
    ------------
    tests/unit/test_cosmology.py: küçük hız limiti z ≈ v/c ve ters fonksiyonla gidiş-dönüş.
    """
    beta = np.asarray(velocity_km_s, dtype=float) / SPEED_OF_LIGHT_KM_S
    return np.sqrt((1.0 + beta) / (1.0 - beta)) - 1.0


@source("Hogg1999", where="Denk. 9", status="read", note="Denk. 9'un v için çözümü: β = ((1+z)² − 1)/((1+z)² + 1)")
def doppler_velocity_km_s(z: ArrayLike) -> Array:
    """Doppler kırmızıya kaymasından bakış doğrultusundaki hızı hesaplar.

    References
    ----------
    [Hogg1999] Denk. 9.

    Verification
    ------------
    tests/unit/test_cosmology.py: ``doppler_kirmiziya_kayma`` ile gidiş-dönüş.
    """
    kare = (1.0 + np.asarray(z, dtype=float)) ** 2
    return SPEED_OF_LIGHT_KM_S * (kare - 1.0) / (kare + 1.0)


@source("Hogg1999", where="Denk. 10", status="read", note="v_pec = c (z_obs − z_cos)/(1 + z); küçük hız yaklaşımı")
def peculiar_velocity_km_s(z_observed: ArrayLike, z_cosmological: ArrayLike) -> Array:
    """Gözlenen ve kozmolojik kırmızıya kaymadan özel hızı hesaplar.

    v_pec = c (z_göz − z_koz) / (1 + z)

    References
    ----------
    [Hogg1999] Denk. 10.

    Validity
    --------
    v_pec << c. Paydadaki z için kozmolojik kırmızıya kayma kullanılır;
    kaynak hangisinin kullanılacağını belirtmez, fark ikinci mertebedendir.

    Verification
    ------------
    tests/unit/test_cosmology.py: (1 + z_göz) = (1 + z_koz)(1 + v/c) kuralıyla tutarlılık.
    """
    zg = np.asarray(z_observed, dtype=float)
    zk = np.asarray(z_cosmological, dtype=float)
    return SPEED_OF_LIGHT_KM_S * (zg - zk) / (1.0 + zk)
