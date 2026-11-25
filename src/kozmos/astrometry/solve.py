"""Görüntüden astrometrik çözüm: saptanan kaynakları katalogla eşleyip WCS uydurma.

FITS WCS'in doğrusal bölümü piksel koordinatlarını teğet düzlem koordinatlarına
bağlar: (ξ, η) = CD · (p − CRPIX) [CalabrettaGreisen2002, TeX etiketi eq:px].
Buradaki işlevler bu bağıntıyı ters yönde kullanır: eşlenmiş yıldızların piksel
ve gök konumlarından CD ve CRPIX en küçük kareler ile bulunur.

Yöntemin kendisi (oylamayla kaydırma bulma, eşle-uydur yinelemesi) bu projenin
seçimidir; bir makaleden alınmamıştır. Kapsamı sınırlıdır: yaklaşık bir WCS
(ölçek ve yönelim yüzde birkaç içinde, kaydırma onlarca piksel) gerekir. Hiçbir
ön bilgi olmadan desen eşleme ("kör" çözüm) yoktur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.astrometry.matching import cross_match
from kozmos.astrometry.wcs import Wcs, pixel_to_sky, sky_to_pixel

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class WcsSolution:
    """Uydurulan WCS ve uyumun niteliği.

    Attributes
    ----------
    wcs : Wcs
    yildiz_sayisi : int
        Uyuma giren eşleşmiş yıldız sayısı.
    artik_rms_yaysaniye : float
        Eşleşmiş yıldızlarda gökteki konum artığının karekök ortalaması.
    """

    wcs: Wcs
    n_stars: int
    residual_rms_arcsec: float


def _tangent_plane(ra_deg: Array, dec_deg: Array, crval: Array) -> tuple[Array, Array]:
    """Gök konumlarının crval çevresindeki teğet düzlem koordinatları (ξ, η) [derece]."""
    unit = Wcs(crpix=np.array([1.0, 1.0]), crval=np.asarray(crval, dtype=float), cd=np.eye(2))
    return sky_to_pixel(unit, ra_deg, dec_deg)


@source(
    "CalabrettaGreisen2002",
    where="TeX etiketi eq:px (doğrusal piksel → ara koordinat bağıntısı)",
    status="read",
    note="(ξ, η) = CD (p − CRPIX) bağıntısı ξ = a0 + a1 x + a2 y, η = b0 + b1 x + b2 y biçiminde doğrusaldır; "
    "altı katsayı en küçük kareler ile bulunur, CRPIX = −CD⁻¹ (a0, b0). Uydurma yöntemi bu projenin seçimidir",
)
def fit_wcs(
    x: ArrayLike,
    y: ArrayLike,
    ra_deg: ArrayLike,
    dec_deg: ArrayLike,
    crval: ArrayLike | None = None,
) -> WcsSolution:
    """Eşlenmiş yıldızların piksel ve gök konumlarından TAN izdüşümlü WCS uydurur.

    Parameters
    ----------
    x, y : ndarray
        Sıfır tabanlı piksel koordinatları.
    sag_acikllik_derece, dik_acikllik_derece : ndarray
        Aynı yıldızların katalog konumları [derece].
    crval : (2,), isteğe bağlı
        Teğet noktası [derece]; verilmezse yıldızların ortalama yönü alınır.

    Returns
    -------
    WcsCozumu

    References
    ----------
    [CalabrettaGreisen2002] eq:px.

    Validity
    --------
    En az üç yıldız gerekir; aynı doğru üzerindeki yıldızlar çözümü belirlemez.
    Yalnızca doğrusal terimler uydurulur (ölçek, dönme, eğiklik, kaydırma); optik
    bozulma modellenmez. Aykırı eşleşmeler ayıklanmaz.

    Verification
    ------------
    tests/unit/test_astrometry_solve.py: bilinen WCS'ten üretilen yıldızlarda CD ve CRPIX'in geri bulunması.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ra, dec = np.asarray(ra_deg, dtype=float), np.asarray(dec_deg, dtype=float)
    if x.size < 3:
        raise ValueError("WCS uydurmak için en az üç eşleşmiş yıldız gerekir")
    if crval is None:
        v = np.array(
            [
                np.mean(np.cos(np.radians(dec)) * np.cos(np.radians(ra))),
                np.mean(np.cos(np.radians(dec)) * np.sin(np.radians(ra))),
                np.mean(np.sin(np.radians(dec))),
            ]
        )
        crval = np.array(
            [np.degrees(np.arctan2(v[1], v[0])) % 360.0, np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))]
        )
    crval = np.asarray(crval, dtype=float)
    xi, eta = _tangent_plane(ra, dec, crval)
    design = np.column_stack([np.ones(x.size), x + 1.0, y + 1.0])  # FITS pikseli = sıfır tabanlı + 1
    a, *_ = np.linalg.lstsq(design, xi, rcond=None)
    b, *_ = np.linalg.lstsq(design, eta, rcond=None)
    cd = np.array([[a[1], a[2]], [b[1], b[2]]])
    crpix = -np.linalg.solve(cd, np.array([a[0], b[0]]))
    wcs = Wcs(crpix=crpix, crval=crval, cd=cd)
    ra_m, dec_m = pixel_to_sky(wcs, x, y)
    d_ra = (ra_m - ra + 180.0) % 360.0 - 180.0
    residual = np.hypot(d_ra * np.cos(np.radians(dec)), dec_m - dec) * 3600.0
    return WcsSolution(wcs, int(x.size), float(np.sqrt(np.mean(residual**2))))


@source(
    "CalabrettaGreisen2002",
    where="TeX etiketi eq:px: CRPIX referans noktanın piksel konumudur",
    status="read",
    note="kaydırma, kataloğun yaklaşık WCS ile hesaplanan piksel konumları ile saptanan konumlar arasındaki bütün "
    "ikili farkların iki boyutlu histogramının tepesinden bulunur ve CRPIX'e eklenir. Oylama yöntemi bu projenin seçimidir",
)
def find_shift(
    wcs: Wcs,
    x: ArrayLike,
    y: ArrayLike,
    ra_deg: ArrayLike,
    dec_deg: ArrayLike,
    *,
    max_shift_pix: float = 60.0,
    bin_pix: float = 2.0,
) -> tuple[Wcs, int]:
    """Yaklaşık bir WCS'in piksel kaydırmasını oylamayla bulur ve düzeltilmiş WCS'i döndürür.

    Parameters
    ----------
    wcs : Wcs
        Ölçeği ve yönelimi yaklaşık doğru, sıfır noktası kaymış WCS.
    x, y : ndarray
        Saptanan kaynakların sıfır tabanlı piksel konumları.
    sag_acikllik_derece, dik_acikllik_derece : ndarray
        Katalog konumları [derece].
    en_cok_piksel : float
        Aranan en büyük kaydırma (her eksende).
    kutu_piksel : float
        Oylama histogramının kutu genişliği.

    Returns
    -------
    (düzeltilmiş WCS, tepe kutusundaki oy sayısı)

    References
    ----------
    [CalabrettaGreisen2002] eq:px.

    Validity
    --------
    Dönme ya da ölçek hatası, alanın kenarında kutu genişliğini aşan bir kaymaya yol
    açıyorsa oylar dağılır. Tepe oyu rastlantısal düzeyden (kaynak sayısı × katalog
    sayısı × kutu alanı / arama alanı) belirgin yüksek değilse çözüme güvenilmez.

    Verification
    ------------
    tests/unit/test_astrometry_solve.py: bilinen kaydırmanın, sahte saptamalar ve eksik katalog yıldızları varken bulunması.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    kx, ky = sky_to_pixel(wcs, ra_deg, dec_deg)
    gecerli = np.isfinite(kx) & np.isfinite(ky)
    dx = (x[:, None] - kx[gecerli][None, :]).ravel()
    dy = (y[:, None] - ky[gecerli][None, :]).ravel()
    yakin = (np.abs(dx) <= max_shift_pix) & (np.abs(dy) <= max_shift_pix)
    kenarlar = np.arange(-max_shift_pix, max_shift_pix + bin_pix, bin_pix)
    oy, _, _ = np.histogram2d(dx[yakin], dy[yakin], bins=(kenarlar, kenarlar))
    # Tepe iki kutunun sınırına düşebilir: komşu kutularla toplanmış haritada aranır.
    toplu = sum(np.roll(np.roll(oy, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1))
    i, j = np.unravel_index(np.argmax(toplu), toplu.shape)
    sec = yakin & (np.abs(dx - (kenarlar[i] + bin_pix / 2)) <= 1.5 * bin_pix)
    sec &= np.abs(dy - (kenarlar[j] + bin_pix / 2)) <= 1.5 * bin_pix
    kayma = np.array([np.median(dx[sec]), np.median(dy[sec])])
    return Wcs(crpix=wcs.crpix + kayma, crval=wcs.crval, cd=wcs.cd, lonpole=wcs.lonpole), int(toplu[i, j])


@source(
    "CalabrettaGreisen2002",
    where="TeX etiketi eq:px",
    status="read",
    note="eşle (capraz_esle) ve uydur (wcs_uydur) adımları, eşleme yarıçapı daraltılarak yinelenir. "
    "Yineleme düzeni bu projenin seçimidir",
)
def refine_wcs(
    wcs: Wcs,
    x: ArrayLike,
    y: ArrayLike,
    ra_deg: ArrayLike,
    dec_deg: ArrayLike,
    *,
    radius_arcsec: float = 5.0,
    final_radius_arcsec: float = 1.5,
    iterations: int = 6,
) -> WcsSolution:
    """Yaklaşık bir WCS'i katalogla eşle-uydur yinelemesiyle iyileştirir.

    Parameters
    ----------
    wcs : Wcs
        Başlangıç WCS'i; yıldızların çoğunu ``yaricap_yaysaniye`` içinde doğru yıldıza düşürmelidir
        (gerekirse önce ``kaydirma_bul``).
    yaricap_yaysaniye, son_yaricap_yaysaniye : float
        İlk ve son eşleme yarıçapı; arada geometrik olarak daraltılır.

    Returns
    -------
    WcsCozumu
        Son yinelemenin çözümü.

    References
    ----------
    [CalabrettaGreisen2002] eq:px.

    Validity
    --------
    Başlangıç hatası eşleme yarıçapını aşan yıldızlar ilk yinelemede eşleşmez; hata alanın
    büyük bölümünde yarıçapı aşıyorsa yineleme yanlış çözüme yakınsayabilir. Öz hareket
    uygulanmaz.

    Verification
    ------------
    tests/unit/test_astrometry_solve.py: döndürülmüş, ölçeği bozulmuş ve kaydırılmış başlangıçtan gerçek WCS'e yakınsama.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ra_k, dec_k = np.asarray(ra_deg, dtype=float), np.asarray(dec_deg, dtype=float)
    yaricaplar = np.geomspace(radius_arcsec, final_radius_arcsec, iterations)
    cozum = None
    for radius in yaricaplar:
        ra, dec = pixel_to_sky(wcs, x, y)
        e = cross_match(ra, dec, ra_k, dec_k, float(radius))
        if e.matched.sum() < 3:
            raise ValueError(f"{radius:.1f} yay saniyesi içinde üçten az eşleşme var; başlangıç WCS'i yetersiz")
        cozum = fit_wcs(x[e.matched], y[e.matched], ra_k[e.index[e.matched]], dec_k[e.index[e.matched]], wcs.crval)
        wcs = cozum.wcs
    return cozum
