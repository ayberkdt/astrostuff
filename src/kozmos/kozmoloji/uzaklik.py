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

from kozmos._kaynak import kaynak
from kozmos.temel.sabitler import ISIK_HIZI_KM_S, JULYEN_YILI_SANIYE, MEGAPARSEK

Dizi = NDArray[np.float64]

# Gauss-Legendre düğümleri: [0, z] üzerinde 1/E(z') integrali için.
_DUGUM, _AGIRLIK = np.polynomial.legendre.leggauss(64)


@dataclass(frozen=True, slots=True)
class Kozmoloji:
    """Friedmann-Lemaître-Robertson-Walker evreninin parametreleri.

    Öznitelikler
    ------------
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
    def duz(cls, h0: float, omega_m: float, w: float = -1.0) -> Kozmoloji:
        """Düz evren kurar: ΩΛ = 1 − Ωm."""
        return cls(h0=h0, omega_m=omega_m, omega_lambda=1.0 - omega_m, w=w)


@kaynak("Hogg1999", yer="Denk. 4", teyit="okundu")
def hubble_uzakligi_mpc(kozmoloji: Kozmoloji) -> float:
    """Hubble uzaklığını hesaplar: D_H = c / H0.

    Döndürür
    --------
    float
        Hubble uzaklığı [Mpc].

    Kaynaklar
    ---------
    [Hogg1999] Denk. 4.

    Doğrulama
    ---------
    tests/birim/test_kozmoloji.py: h = 1 için 3000 Mpc dolayı (makaledeki değer).
    """
    return ISIK_HIZI_KM_S / kozmoloji.h0


@kaynak("Hogg1999", yer="Denk. 3", teyit="okundu")
def hubble_zamani_gyr(kozmoloji: Kozmoloji) -> float:
    """Hubble zamanını hesaplar: t_H = 1 / H0.

    Döndürür
    --------
    float
        Hubble zamanı [Gyr, Jülyen yılı].

    Kaynaklar
    ---------
    [Hogg1999] Denk. 3.

    Doğrulama
    ---------
    tests/birim/test_kozmoloji.py: h = 1 için 9,78 Gyr (makaledeki değer).
    """
    h0_si = kozmoloji.h0 * 1e3 / MEGAPARSEK.deger  # s^-1
    return 1.0 / h0_si / JULYEN_YILI_SANIYE.deger / 1e9


@kaynak("Hogg1999", yer="Denk. 14", teyit="okundu")
@kaynak(
    "Brout2022",
    yer="H(z) denklemi (TeX etiketi eq:hz)",
    teyit="okundu",
    not_="ΩΛ terimi (1+z)^{3(1+w)} ile çarpılır; Brout düz evren için yazar, eğrilik terimi Hogg'dan",
)
def e_fonksiyonu(z: ArrayLike, kozmoloji: Kozmoloji) -> Dizi:
    """Boyutsuz Hubble parametresini hesaplar: E(z) = H(z)/H0.

    E(z) = sqrt( Ωm (1+z)³ + Ωk (1+z)² + ΩΛ (1+z)^{3(1+w)} )

    Parametreler
    ------------
    z : float ya da ndarray
        Kırmızıya kayma, z > −1.
    kozmoloji : Kozmoloji

    Kaynaklar
    ---------
    [Hogg1999] Denk. 14; [Brout2022] H(z) denklemi.

    Geçerlilik
    ----------
    Radyasyon terimi yoktur.

    Doğrulama
    ---------
    tests/kahin/test_kozmoloji_kahin.py: astropy.cosmology efunc ile bağıl fark < 1e-12.
    """
    zp1 = 1.0 + np.asarray(z, dtype=float)
    k = kozmoloji
    return np.sqrt(k.omega_m * zp1**3 + k.omega_k * zp1**2 + k.omega_lambda * zp1 ** (3.0 * (1.0 + k.w)))


def _integral_sifirdan_z(z: Dizi, integrand) -> Dizi:  # type: ignore[no-untyped-def]
    """[0, z] üzerinde 64 noktalı Gauss-Legendre integrali; z dizisi için tek geçiş."""
    yari = 0.5 * z[..., None]
    dugum = yari * (_DUGUM + 1.0)
    return np.sum(_AGIRLIK * integrand(dugum), axis=-1) * yari[..., 0]


@kaynak("Hogg1999", yer="Denk. 15", teyit="okundu")
def esyuruyen_uzaklik_mpc(z: ArrayLike, kozmoloji: Kozmoloji) -> Dizi:
    """Bakış doğrultusundaki eşyürüyen uzaklığı hesaplar.

    D_C = D_H ∫₀ᶻ dz′ / E(z′)

    Döndürür
    --------
    float ya da ndarray
        Eşyürüyen uzaklık [Mpc].

    Kaynaklar
    ---------
    [Hogg1999] Denk. 15.

    Geçerlilik
    ----------
    0 <= z <= 10 aralığında 64 noktalı Gauss-Legendre integrali astropy ile
    1e-9 bağıl farkın altındadır; daha büyük z sınanmamıştır.

    Doğrulama
    ---------
    tests/kahin/test_kozmoloji_kahin.py; tests/birim/test_kozmoloji.py (Einstein-de Sitter kapalı çözümü).
    """
    zz = np.asarray(z, dtype=float)
    if np.any(zz < 0):
        raise ValueError("z eksi olamaz")
    return hubble_uzakligi_mpc(kozmoloji) * _integral_sifirdan_z(zz, lambda x: 1.0 / e_fonksiyonu(x, kozmoloji))


@kaynak("Hogg1999", yer="Denk. 16", teyit="okundu")
def enine_esyuruyen_uzaklik_mpc(z: ArrayLike, kozmoloji: Kozmoloji) -> Dizi:
    """Enine eşyürüyen uzaklığı hesaplar.

    Ωk > 0: D_M = D_H/√Ωk · sinh(√Ωk D_C/D_H); Ωk = 0: D_M = D_C;
    Ωk < 0: D_M = D_H/√|Ωk| · sin(√|Ωk| D_C/D_H).

    Kaynaklar
    ---------
    [Hogg1999] Denk. 16.

    Geçerlilik
    ----------
    |Ωk| < 1e-12 düz evren sayılır; dallar bu sınırda süreklidir.

    Doğrulama
    ---------
    tests/kahin/test_kozmoloji_kahin.py; tests/ozellik/test_kozmoloji_ozellik.py (Ωk → 0 sürekliliği).
    """
    dc = esyuruyen_uzaklik_mpc(z, kozmoloji)
    ok = kozmoloji.omega_k
    if abs(ok) < 1e-12:
        return dc
    dh = hubble_uzakligi_mpc(kozmoloji)
    kok = np.sqrt(abs(ok))
    if ok > 0:
        return dh / kok * np.sinh(kok * dc / dh)
    return dh / kok * np.sin(kok * dc / dh)


@kaynak("Hogg1999", yer="Denk. 18", teyit="okundu")
def acisal_cap_uzakligi_mpc(z: ArrayLike, kozmoloji: Kozmoloji) -> Dizi:
    """Açısal çap uzaklığını hesaplar: D_A = D_M / (1 + z).

    Kaynaklar
    ---------
    [Hogg1999] Denk. 18.

    Doğrulama
    ---------
    tests/kahin/test_kozmoloji_kahin.py
    """
    return enine_esyuruyen_uzaklik_mpc(z, kozmoloji) / (1.0 + np.asarray(z, dtype=float))


@kaynak("Hogg1999", yer="Denk. 21", teyit="okundu")
def isinim_gucu_uzakligi_mpc(z: ArrayLike, kozmoloji: Kozmoloji) -> Dizi:
    """Işınım gücü uzaklığını hesaplar: D_L = (1 + z) D_M.

    Kaynaklar
    ---------
    [Hogg1999] Denk. 21.

    Doğrulama
    ---------
    tests/kahin/test_kozmoloji_kahin.py; tests/ozellik/test_kozmoloji_ozellik.py (z → 0 limitinde cz/H0).
    """
    return (1.0 + np.asarray(z, dtype=float)) * enine_esyuruyen_uzaklik_mpc(z, kozmoloji)


@kaynak("Hogg1999", yer="Denk. 25", teyit="okundu", not_="D_L Mpc ise μ = 5 log10(D_L) + 25")
def uzaklik_modulu(z: ArrayLike, kozmoloji: Kozmoloji) -> Dizi:
    """Uzaklık modülünü hesaplar: μ = 5 log10(D_L / 10 pc).

    Döndürür
    --------
    float ya da ndarray
        Uzaklık modülü [mag].

    Kaynaklar
    ---------
    [Hogg1999] Denk. 25.

    Geçerlilik
    ----------
    z > 0; z = 0 için −∞ döner.

    Doğrulama
    ---------
    tests/kahin/test_kozmoloji_kahin.py: astropy distmod ile fark < 1e-9 mag.
    """
    with np.errstate(divide="ignore"):
        return 5.0 * np.log10(isinim_gucu_uzakligi_mpc(z, kozmoloji)) + 25.0


@kaynak("Hogg1999", yer="Denk. 30", teyit="okundu")
def geriye_bakis_zamani_gyr(z: ArrayLike, kozmoloji: Kozmoloji) -> Dizi:
    """Geriye bakış zamanını hesaplar: t_L = t_H ∫₀ᶻ dz′ / [(1+z′) E(z′)].

    Döndürür
    --------
    float ya da ndarray
        Geriye bakış zamanı [Gyr].

    Kaynaklar
    ---------
    [Hogg1999] Denk. 30.

    Geçerlilik
    ----------
    0 <= z <= 10 aralığında sınanmıştır.

    Doğrulama
    ---------
    tests/kahin/test_kozmoloji_kahin.py: astropy lookback_time ile bağıl fark < 1e-8.
    """
    zz = np.asarray(z, dtype=float)
    if np.any(zz < 0):
        raise ValueError("z eksi olamaz")
    return hubble_zamani_gyr(kozmoloji) * _integral_sifirdan_z(
        zz, lambda x: 1.0 / ((1.0 + x) * e_fonksiyonu(x, kozmoloji))
    )


@kaynak("Hogg1999", yer="Denk. 28", teyit="okundu")
def esyuruyen_hacim_elemani_mpc3(z: ArrayLike, kozmoloji: Kozmoloji) -> Dizi:
    """Birim katı açı ve birim z başına eşyürüyen hacmi hesaplar.

    dV_C / (dΩ dz) = D_H (1+z)² D_A² / E(z)

    Döndürür
    --------
    float ya da ndarray
        Hacim elemanı [Mpc³ sr⁻¹].

    Kaynaklar
    ---------
    [Hogg1999] Denk. 28.

    Doğrulama
    ---------
    tests/kahin/test_kozmoloji_kahin.py: astropy differential_comoving_volume.
    """
    zz = np.asarray(z, dtype=float)
    da = acisal_cap_uzakligi_mpc(zz, kozmoloji)
    return hubble_uzakligi_mpc(kozmoloji) * (1.0 + zz) ** 2 * da**2 / e_fonksiyonu(zz, kozmoloji)


@kaynak("Hogg1999", yer="Denk. 19", teyit="okundu", not_="kaynak, formülün Ωk < 0 için geçerli olmadığını belirtir")
def acisal_cap_uzakligi_z1_z2_mpc(z1: ArrayLike, z2: ArrayLike, kozmoloji: Kozmoloji) -> Dizi:
    """z1'deki gözlemcinin z2'deki cisme ölçtüğü açısal çap uzaklığını hesaplar.

    D_A12 = [D_M2 sqrt(1 + Ωk D_M1²/D_H²) − D_M1 sqrt(1 + Ωk D_M2²/D_H²)] / (1 + z2)

    Kaynaklar
    ---------
    [Hogg1999] Denk. 19.

    Geçerlilik
    ----------
    Ωk >= 0 ve z1 <= z2. Çekimsel merceklenmede mercek-kaynak uzaklığı için kullanılır.

    Doğrulama
    ---------
    tests/kahin/test_kozmoloji_kahin.py: astropy angular_diameter_distance_z1z2.
    """
    if kozmoloji.omega_k < -1e-12:
        raise ValueError("Denk. 19 kapalı evrende (Ωk < 0) geçerli değil")
    ok = max(kozmoloji.omega_k, 0.0)
    dh = hubble_uzakligi_mpc(kozmoloji)
    dm1 = enine_esyuruyen_uzaklik_mpc(z1, kozmoloji)
    dm2 = enine_esyuruyen_uzaklik_mpc(z2, kozmoloji)
    ic = dm2 * np.sqrt(1.0 + ok * dm1**2 / dh**2) - dm1 * np.sqrt(1.0 + ok * dm2**2 / dh**2)
    return ic / (1.0 + np.asarray(z2, dtype=float))


@kaynak("Hogg1999", yer="Denk. 29", teyit="okundu", not_="tüm gökyüzü hacmi; üç eğrilik dalı")
def esyuruyen_hacim_mpc3(z: ArrayLike, kozmoloji: Kozmoloji) -> Dizi:
    """z'ye kadar tüm gökyüzünün eşyürüyen hacmini hesaplar.

    Düz evrende V_C = (4π/3) D_M³; eğrilikli evrende kaynaktaki kapalı biçim.

    Döndürür
    --------
    float ya da ndarray
        Eşyürüyen hacim [Mpc³].

    Kaynaklar
    ---------
    [Hogg1999] Denk. 29.

    Geçerlilik
    ----------
    |Ωk| < 1e-8 düz evren sayılır: eğrilikli dalda iki büyük terimin farkı
    küçük |Ωk| için duyarlılık kaybeder.

    Doğrulama
    ---------
    tests/kahin/test_kozmoloji_kahin.py: astropy comoving_volume.
    """
    dm = enine_esyuruyen_uzaklik_mpc(z, kozmoloji)
    ok = kozmoloji.omega_k
    if abs(ok) < 1e-8:
        return 4.0 * np.pi / 3.0 * dm**3
    dh = hubble_uzakligi_mpc(kozmoloji)
    x = dm / dh
    kok = np.sqrt(abs(ok))
    ters = np.arcsinh(kok * x) if ok > 0 else np.arcsin(kok * x)
    return 4.0 * np.pi * dh**3 / (2.0 * ok) * (x * np.sqrt(1.0 + ok * x**2) - ters / kok)


@kaynak("Hogg1999", yer="Denk. 9", teyit="okundu")
def doppler_kirmiziya_kayma(hiz_km_s: ArrayLike) -> Dizi:
    """Bakış doğrultusundaki hızdan görelilikli Doppler kırmızıya kaymasını hesaplar.

    1 + z = sqrt((1 + v/c) / (1 − v/c))

    Kaynaklar
    ---------
    [Hogg1999] Denk. 9.

    Geçerlilik
    ----------
    |v| < c. Kozmolojik kırmızıya kayma bu formülle hıza çevrilmez; kaynak
    bunun yalnızca özel hızlar için anlamlı olduğunu vurgular.

    Doğrulama
    ---------
    tests/birim/test_kozmoloji.py: küçük hız limiti z ≈ v/c ve ters fonksiyonla gidiş-dönüş.
    """
    beta = np.asarray(hiz_km_s, dtype=float) / ISIK_HIZI_KM_S
    return np.sqrt((1.0 + beta) / (1.0 - beta)) - 1.0


@kaynak("Hogg1999", yer="Denk. 9", teyit="okundu", not_="Denk. 9'un v için çözümü: β = ((1+z)² − 1)/((1+z)² + 1)")
def doppler_hizi_km_s(z: ArrayLike) -> Dizi:
    """Doppler kırmızıya kaymasından bakış doğrultusundaki hızı hesaplar.

    Kaynaklar
    ---------
    [Hogg1999] Denk. 9.

    Doğrulama
    ---------
    tests/birim/test_kozmoloji.py: ``doppler_kirmiziya_kayma`` ile gidiş-dönüş.
    """
    kare = (1.0 + np.asarray(z, dtype=float)) ** 2
    return ISIK_HIZI_KM_S * (kare - 1.0) / (kare + 1.0)


@kaynak("Hogg1999", yer="Denk. 10", teyit="okundu", not_="v_pec = c (z_obs − z_cos)/(1 + z); küçük hız yaklaşımı")
def ozel_hiz_km_s(z_gozlenen: ArrayLike, z_kozmolojik: ArrayLike) -> Dizi:
    """Gözlenen ve kozmolojik kırmızıya kaymadan özel hızı hesaplar.

    v_pec = c (z_göz − z_koz) / (1 + z)

    Kaynaklar
    ---------
    [Hogg1999] Denk. 10.

    Geçerlilik
    ----------
    v_pec << c. Paydadaki z için kozmolojik kırmızıya kayma kullanılır;
    kaynak hangisinin kullanılacağını belirtmez, fark ikinci mertebedendir.

    Doğrulama
    ---------
    tests/birim/test_kozmoloji.py: (1 + z_göz) = (1 + z_koz)(1 + v/c) kuralıyla tutarlılık.
    """
    zg = np.asarray(z_gozlenen, dtype=float)
    zk = np.asarray(z_kozmolojik, dtype=float)
    return ISIK_HIZI_KM_S * (zg - zk) / (1.0 + zk)
