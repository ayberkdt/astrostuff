"""Atmosfer kırılması: A tan Z + B tan³ Z modeli.

Model ve sabitlerin hesabı ERFA kaynak kodundan (``refco.c``) okunmuştur
[ERFA2023]. Z kırılmadan etkilenmiş ("gözlenen") zenit uzaklığıdır;
dZ = A tan Z + B tan³ Z, boşluktaki ("toposantrik") zenit uzaklığını bulmak
için Z'ye eklenir. Kırılma cismi ufka göre yükseltir.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source

Array = NDArray[np.float64]


@source(
    "ERFA2023",
    where="refco.c",
    status="read",
    note="su buharı basıncı, kırılma indisi (γ = n − 1) ve β (ölçek yüksekliği / yer merkezi uzaklığı) "
    "refco.c'deki ifadelerle; A = γ(1 − β), B = −γ(β − γ/2). ERFA bu ifadeleri Gill (1982), Crane (1976), "
    "Hohenkerk ve Sinclair (1985), Rueger (2002), Stone (1996) ve Green (1987)'ye dayandırır; "
    "o kaynaklar okunmadı. Yalnızca optik/kızılötesi dalı (λ <= 100 µm) uygulandı",
)
def refraction_constants(
    pressure_hpa: float, temperature_c: float, relative_humidity: float, wavelength_um: float = 0.55
) -> tuple[float, float]:
    """Kırılma modelinin A ve B sabitlerini hesaplar.

    Parameters
    ----------
    basinc_hpa : float
        Gözlemcideki basınç [hPa].
    sicaklik_c : float
        Ortam sıcaklığı [°C].
    bagil_nem : float
        Bağıl nem, 0-1.
    dalgaboyu_um : float
        Dalga boyu [µm]; 0,1-100 aralığı.

    Returns
    -------
    (A, B) [rad]

    References
    ----------
    [ERFA2023] refco.c.

    Validity
    --------
    Kaynak, ışın izlemeye göre doğruluğu optik/kızılötesinde en kötü 62, karekök ortalama
    8 miliyay saniyesi verir (zenit uzaklığı 15°-75°). Deniz düzeyinden yükseklik yalnızca
    basınç üzerinden girer. Radyo dalı uygulanmadı; λ > 100 µm ``ValueError`` verir.
    Girdiler kaynaktaki güvenli aralıklara kırpılır.

    Verification
    ------------
    tests/oracle/test_precession_oracle.py: erfa.refco ile eşitlik; refco.c'deki tabloyla 0,02 yay saniyesi içinde uyum.
    """
    if wavelength_um > 100.0:
        raise ValueError("yalnızca optik/kızılötesi dal uygulanmıştır (dalga boyu <= 100 µm)")
    t = min(max(temperature_c, -150.0), 200.0)
    p = min(max(pressure_hpa, 0.0), 10000.0)
    r = min(max(relative_humidity, 0.0), 1.0)
    w = max(wavelength_um, 0.1)
    if p > 0.0:
        ps = 10.0 ** ((0.7859 + 0.03477 * t) / (1.0 + 0.00412 * t)) * (1.0 + p * (4.5e-6 + 6e-10 * t * t))
        pw = r * ps / (1.0 - (1.0 - r) * ps / p)
    else:
        pw = 0.0
    tk = t + 273.15
    wlsq = w * w
    gamma = ((77.53484e-6 + (4.39108e-7 + 3.666e-9 / wlsq) / wlsq) * p - 11.2684e-6 * pw) / tk
    beta = 4.4474e-6 * tk
    return gamma * (1.0 - beta), -gamma * (beta - gamma / 2.0)


@source(
    "ERFA2023",
    where="refco.c: 'dZ = A tan Z + B tan^3 Z', Z gözlenen zenit uzaklığı",
    status="read",
    note="toposantrik → gözlenen yön için denklem Z_gözlenen = Z_topo − dZ(Z_gözlenen) sabit nokta yinelemesiyle "
    "çözülür (yineleme bu projenin seçimi)",
)
def observed_altitude(altitude_rad: ArrayLike, a_rad: float, b_rad: float, iterations: int = 4) -> Array:
    """Boşluktaki (toposantrik) yükseklikten kırılmış (gözlenen) yüksekliği hesaplar.

    Parameters
    ----------
    yukseklik_rad : float ya da ndarray
        Kırılmasız yükseklik [rad].
    a_rad, b_rad : float
        ``kirilma_sabitleri`` çıktıları.

    Returns
    -------
    ndarray
        Gözlenen yükseklik [rad]; her zaman girdiden büyük ya da ona eşit.

    References
    ----------
    [ERFA2023] refco.c.

    Validity
    --------
    tan Z modeli ufka yakın geçersizdir: kaynak tablosu 80° zenit uzaklığında ışın izlemeden
    0,6 yay saniyesi sapar ve fark ufka doğru hızla büyür. Gözlenen zenit uzaklığı 85°'yi
    aşan yönlerde (yükseklik yaklaşık 5°'nin altı) sonuç ``nan`` döner.

    Verification
    ------------
    tests/oracle/test_precession_oracle.py: ileri modelle gidiş-dönüş; refco.c tablosu.
    """
    z_topo = np.pi / 2.0 - np.asarray(altitude_rad, dtype=float)
    z = z_topo.copy()
    for _ in range(iterations):
        tan_z = np.tan(np.clip(z, 0.0, np.radians(85.0)))
        z = z_topo - (a_rad * tan_z + b_rad * tan_z**3)
    return np.where(z <= np.radians(85.0), np.pi / 2.0 - z, np.nan)
