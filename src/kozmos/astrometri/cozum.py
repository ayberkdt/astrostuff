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

from kozmos._kaynak import kaynak
from kozmos.astrometri.eslestirme import capraz_esle
from kozmos.astrometri.wcs import Wcs, gok_den_piksel, piksel_den_gok

Dizi = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class WcsCozumu:
    """Uydurulan WCS ve uyumun niteliği.

    Öznitelikler
    ------------
    wcs : Wcs
    yildiz_sayisi : int
        Uyuma giren eşleşmiş yıldız sayısı.
    artik_rms_yaysaniye : float
        Eşleşmiş yıldızlarda gökteki konum artığının karekök ortalaması.
    """

    wcs: Wcs
    yildiz_sayisi: int
    artik_rms_yaysaniye: float


def _teget_duzlem(sag_acikllik_derece: Dizi, dik_acikllik_derece: Dizi, crval: Dizi) -> tuple[Dizi, Dizi]:
    """Gök konumlarının crval çevresindeki teğet düzlem koordinatları (ξ, η) [derece]."""
    birim = Wcs(crpix=np.array([1.0, 1.0]), crval=np.asarray(crval, dtype=float), cd=np.eye(2))
    return gok_den_piksel(birim, sag_acikllik_derece, dik_acikllik_derece)


@kaynak(
    "CalabrettaGreisen2002",
    yer="TeX etiketi eq:px (doğrusal piksel → ara koordinat bağıntısı)",
    teyit="okundu",
    not_="(ξ, η) = CD (p − CRPIX) bağıntısı ξ = a0 + a1 x + a2 y, η = b0 + b1 x + b2 y biçiminde doğrusaldır; "
    "altı katsayı en küçük kareler ile bulunur, CRPIX = −CD⁻¹ (a0, b0). Uydurma yöntemi bu projenin seçimidir",
)
def wcs_uydur(
    x: ArrayLike,
    y: ArrayLike,
    sag_acikllik_derece: ArrayLike,
    dik_acikllik_derece: ArrayLike,
    crval: ArrayLike | None = None,
) -> WcsCozumu:
    """Eşlenmiş yıldızların piksel ve gök konumlarından TAN izdüşümlü WCS uydurur.

    Parametreler
    ------------
    x, y : ndarray
        Sıfır tabanlı piksel koordinatları.
    sag_acikllik_derece, dik_acikllik_derece : ndarray
        Aynı yıldızların katalog konumları [derece].
    crval : (2,), isteğe bağlı
        Teğet noktası [derece]; verilmezse yıldızların ortalama yönü alınır.

    Döndürür
    --------
    WcsCozumu

    Kaynaklar
    ---------
    [CalabrettaGreisen2002] eq:px.

    Geçerlilik
    ----------
    En az üç yıldız gerekir; aynı doğru üzerindeki yıldızlar çözümü belirlemez.
    Yalnızca doğrusal terimler uydurulur (ölçek, dönme, eğiklik, kaydırma); optik
    bozulma modellenmez. Aykırı eşleşmeler ayıklanmaz.

    Doğrulama
    ---------
    tests/birim/test_astrometri_cozum.py: bilinen WCS'ten üretilen yıldızlarda CD ve CRPIX'in geri bulunması.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ra, dec = np.asarray(sag_acikllik_derece, dtype=float), np.asarray(dik_acikllik_derece, dtype=float)
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
    xi, eta = _teget_duzlem(ra, dec, crval)
    tasarim = np.column_stack([np.ones(x.size), x + 1.0, y + 1.0])  # FITS pikseli = sıfır tabanlı + 1
    a, *_ = np.linalg.lstsq(tasarim, xi, rcond=None)
    b, *_ = np.linalg.lstsq(tasarim, eta, rcond=None)
    cd = np.array([[a[1], a[2]], [b[1], b[2]]])
    crpix = -np.linalg.solve(cd, np.array([a[0], b[0]]))
    wcs = Wcs(crpix=crpix, crval=crval, cd=cd)
    ra_m, dec_m = piksel_den_gok(wcs, x, y)
    d_ra = (ra_m - ra + 180.0) % 360.0 - 180.0
    artik = np.hypot(d_ra * np.cos(np.radians(dec)), dec_m - dec) * 3600.0
    return WcsCozumu(wcs, int(x.size), float(np.sqrt(np.mean(artik**2))))


@kaynak(
    "CalabrettaGreisen2002",
    yer="TeX etiketi eq:px: CRPIX referans noktanın piksel konumudur",
    teyit="okundu",
    not_="kaydırma, kataloğun yaklaşık WCS ile hesaplanan piksel konumları ile saptanan konumlar arasındaki bütün "
    "ikili farkların iki boyutlu histogramının tepesinden bulunur ve CRPIX'e eklenir. Oylama yöntemi bu projenin seçimidir",
)
def kaydirma_bul(
    wcs: Wcs,
    x: ArrayLike,
    y: ArrayLike,
    sag_acikllik_derece: ArrayLike,
    dik_acikllik_derece: ArrayLike,
    *,
    en_cok_piksel: float = 60.0,
    kutu_piksel: float = 2.0,
) -> tuple[Wcs, int]:
    """Yaklaşık bir WCS'in piksel kaydırmasını oylamayla bulur ve düzeltilmiş WCS'i döndürür.

    Parametreler
    ------------
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

    Döndürür
    --------
    (düzeltilmiş WCS, tepe kutusundaki oy sayısı)

    Kaynaklar
    ---------
    [CalabrettaGreisen2002] eq:px.

    Geçerlilik
    ----------
    Dönme ya da ölçek hatası, alanın kenarında kutu genişliğini aşan bir kaymaya yol
    açıyorsa oylar dağılır. Tepe oyu rastlantısal düzeyden (kaynak sayısı × katalog
    sayısı × kutu alanı / arama alanı) belirgin yüksek değilse çözüme güvenilmez.

    Doğrulama
    ---------
    tests/birim/test_astrometri_cozum.py: bilinen kaydırmanın, sahte saptamalar ve eksik katalog yıldızları varken bulunması.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    kx, ky = gok_den_piksel(wcs, sag_acikllik_derece, dik_acikllik_derece)
    gecerli = np.isfinite(kx) & np.isfinite(ky)
    dx = (x[:, None] - kx[gecerli][None, :]).ravel()
    dy = (y[:, None] - ky[gecerli][None, :]).ravel()
    yakin = (np.abs(dx) <= en_cok_piksel) & (np.abs(dy) <= en_cok_piksel)
    kenarlar = np.arange(-en_cok_piksel, en_cok_piksel + kutu_piksel, kutu_piksel)
    oy, _, _ = np.histogram2d(dx[yakin], dy[yakin], bins=(kenarlar, kenarlar))
    # Tepe iki kutunun sınırına düşebilir: komşu kutularla toplanmış haritada aranır.
    toplu = sum(np.roll(np.roll(oy, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1))
    i, j = np.unravel_index(np.argmax(toplu), toplu.shape)
    sec = yakin & (np.abs(dx - (kenarlar[i] + kutu_piksel / 2)) <= 1.5 * kutu_piksel)
    sec &= np.abs(dy - (kenarlar[j] + kutu_piksel / 2)) <= 1.5 * kutu_piksel
    kayma = np.array([np.median(dx[sec]), np.median(dy[sec])])
    return Wcs(crpix=wcs.crpix + kayma, crval=wcs.crval, cd=wcs.cd, lonpole=wcs.lonpole), int(toplu[i, j])


@kaynak(
    "CalabrettaGreisen2002",
    yer="TeX etiketi eq:px",
    teyit="okundu",
    not_="eşle (capraz_esle) ve uydur (wcs_uydur) adımları, eşleme yarıçapı daraltılarak yinelenir. "
    "Yineleme düzeni bu projenin seçimidir",
)
def wcs_iyilestir(
    wcs: Wcs,
    x: ArrayLike,
    y: ArrayLike,
    sag_acikllik_derece: ArrayLike,
    dik_acikllik_derece: ArrayLike,
    *,
    yaricap_yaysaniye: float = 5.0,
    son_yaricap_yaysaniye: float = 1.5,
    yineleme: int = 6,
) -> WcsCozumu:
    """Yaklaşık bir WCS'i katalogla eşle-uydur yinelemesiyle iyileştirir.

    Parametreler
    ------------
    wcs : Wcs
        Başlangıç WCS'i; yıldızların çoğunu ``yaricap_yaysaniye`` içinde doğru yıldıza düşürmelidir
        (gerekirse önce ``kaydirma_bul``).
    yaricap_yaysaniye, son_yaricap_yaysaniye : float
        İlk ve son eşleme yarıçapı; arada geometrik olarak daraltılır.

    Döndürür
    --------
    WcsCozumu
        Son yinelemenin çözümü.

    Kaynaklar
    ---------
    [CalabrettaGreisen2002] eq:px.

    Geçerlilik
    ----------
    Başlangıç hatası eşleme yarıçapını aşan yıldızlar ilk yinelemede eşleşmez; hata alanın
    büyük bölümünde yarıçapı aşıyorsa yineleme yanlış çözüme yakınsayabilir. Öz hareket
    uygulanmaz.

    Doğrulama
    ---------
    tests/birim/test_astrometri_cozum.py: döndürülmüş, ölçeği bozulmuş ve kaydırılmış başlangıçtan gerçek WCS'e yakınsama.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ra_k, dec_k = np.asarray(sag_acikllik_derece, dtype=float), np.asarray(dik_acikllik_derece, dtype=float)
    yaricaplar = np.geomspace(yaricap_yaysaniye, son_yaricap_yaysaniye, yineleme)
    cozum = None
    for yaricap in yaricaplar:
        ra, dec = piksel_den_gok(wcs, x, y)
        e = capraz_esle(ra, dec, ra_k, dec_k, float(yaricap))
        if e.eslesti.sum() < 3:
            raise ValueError(f"{yaricap:.1f} yay saniyesi içinde üçten az eşleşme var; başlangıç WCS'i yetersiz")
        cozum = wcs_uydur(x[e.eslesti], y[e.eslesti], ra_k[e.indis[e.eslesti]], dec_k[e.indis[e.eslesti]], wcs.crval)
        wcs = cozum.wcs
    return cozum
