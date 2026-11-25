"""FITS dünya koordinat sistemi: piksel ↔ gök dönüşümü (gnomonik TAN izdüşümü).

Denklemler FITS WCS makalelerinin ikincisinin arXiv TeX kaynağından
(astro-ph/0207413, ``ccs.ltx``) okunmuştur [CalabrettaGreisen2002] ve oradaki
TeX etiketleriyle anılır:

- ``eq:px``: piksel → ara dünya koordinatları, x_i = Σ_j (s_i m_ij)(p_j − r_j)
- ``eq:azphi``, ``eq:azrt``, ``eq:TANtheta``: düzlem → doğal küresel koordinatlar
- ``eq:nat2std``, ``eq:std2nat``: doğal ↔ gök küresel dönmesi

Kapsam: iki eksenli, ``RA---TAN`` / ``DEC--TAN`` izdüşümü; ``CD`` matrisi ya da
``CDELT`` + ``PC`` (ya da eski ``CROTA2``). Bozulma polinomları (SIP) yoktur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._kaynak import kaynak

Dizi = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Wcs:
    """İki eksenli TAN izdüşümlü dünya koordinat sistemi.

    Öznitelikler
    ------------
    crpix : (2,) ndarray
        Referans pikselin FITS koordinatları (ilk piksel 1'dir).
    crval : (2,) ndarray
        Referans noktanın sağ açıklığı ve dik açıklığı [derece].
    cd : (2, 2) ndarray
        Piksel → ara dünya koordinatları matrisi [derece/piksel].
    lonpole : float
        Gök kutbunun doğal boylamı φ_p [derece]; TAN için varsayılan 180.
    """

    crpix: Dizi
    crval: Dizi
    cd: Dizi
    lonpole: float = 180.0


@kaynak(
    "CalabrettaGreisen2002",
    yer="TeX etiketi eq:px ve LONPOLE anahtarının tanımı",
    teyit="okundu",
    not_="CD_ij = s_i m_ij (eq:px'in ikinci biçimi). LONPOLE varsayılanı δ_0 < θ_0 için 180°; TAN'da θ_0 = 90°. "
    "CROTA2'den matris kurma kuralı makalede okunmadı; yaygın kural uygulandı ve astropy ile sınandı",
)
def basliktan_wcs(baslik: dict) -> Wcs:
    """FITS başlığından TAN izdüşümlü WCS kurar.

    Öncelik sırası: ``CDi_j`` varsa o; yoksa ``CDELTi`` ile ``PCi_j``; o da yoksa
    ``CDELTi`` ile ``CROTA2`` (eski kural).

    Kaynaklar
    ---------
    [CalabrettaGreisen2002] eq:px, LONPOLE tanımı.

    Geçerlilik
    ----------
    Yalnızca ``CTYPE1 = 'RA---TAN'``, ``CTYPE2 = 'DEC--TAN'``; başka izdüşüm ya da
    eksen sırası ``ValueError`` verir. SIP ve ``PV`` bozulma terimleri yok sayılır.

    Doğrulama
    ---------
    tests/kahin/test_wcs_kahin.py: astropy.wcs ile aynı başlıklardan aynı dönüşüm.
    """
    if not (
        str(baslik.get("CTYPE1", "")).startswith("RA---TAN") and str(baslik.get("CTYPE2", "")).startswith("DEC--TAN")
    ):
        raise ValueError(
            f"yalnızca RA---TAN / DEC--TAN desteklenir: {baslik.get('CTYPE1')!r}, {baslik.get('CTYPE2')!r}"
        )
    if "CD1_1" in baslik:
        cd = np.array(
            [[baslik["CD1_1"], baslik.get("CD1_2", 0.0)], [baslik.get("CD2_1", 0.0), baslik["CD2_2"]]], dtype=float
        )
    else:
        olcek = np.array([baslik["CDELT1"], baslik["CDELT2"]], dtype=float)
        if "PC1_1" in baslik:
            pc = np.array(
                [[baslik["PC1_1"], baslik.get("PC1_2", 0.0)], [baslik.get("PC2_1", 0.0), baslik.get("PC2_2", 1.0)]],
                dtype=float,
            )
            cd = olcek[:, None] * pc
        else:
            rho = np.radians(float(baslik.get("CROTA2", 0.0)))
            cd = np.array(
                [[olcek[0] * np.cos(rho), -olcek[1] * np.sin(rho)], [olcek[0] * np.sin(rho), olcek[1] * np.cos(rho)]]
            )
    return Wcs(
        crpix=np.array([baslik["CRPIX1"], baslik["CRPIX2"]], dtype=float),
        crval=np.array([baslik["CRVAL1"], baslik["CRVAL2"]], dtype=float),
        cd=cd,
        lonpole=float(baslik.get("LONPOLE", 180.0)),
    )


@kaynak(
    "CalabrettaGreisen2002",
    yer="TeX etiketleri eq:px, eq:azphi, eq:azrt, eq:TANtheta, eq:nat2std",
    teyit="okundu",
    not_="φ = arg(−y, x), R = sqrt(x² + y²), θ = atan(180°/(π R)); ardından doğal → gök dönmesi (α_p, δ_p) = CRVAL",
)
def piksel_den_gok(wcs: Wcs, x: ArrayLike, y: ArrayLike) -> tuple[Dizi, Dizi]:
    """Piksel koordinatlarını sağ açıklık ve dik açıklığa çevirir.

    Parametreler
    ------------
    wcs : Wcs
    x, y : float ya da ndarray
        Sıfır tabanlı piksel koordinatları (``goruntu[y, x]``; ilk pikselin merkezi 0).

    Döndürür
    --------
    (sağ açıklık, dik açıklık) [derece]; sağ açıklık [0, 360).

    Kaynaklar
    ---------
    [CalabrettaGreisen2002] eq:px, eq:azphi, eq:azrt, eq:TANtheta, eq:nat2std.

    Geçerlilik
    ----------
    TAN izdüşümü teğet noktadan 90° uzakta ıraksar; geniş alanlarda bozulma büyür.

    Doğrulama
    ---------
    tests/kahin/test_wcs_kahin.py: astropy.wcs.all_pix2world ile fark < 1e-9 derece.
    """
    p = np.stack(np.broadcast_arrays(np.asarray(x, dtype=float) + 1.0, np.asarray(y, dtype=float) + 1.0), axis=-1)
    ara = (p - wcs.crpix) @ wcs.cd.T  # eq:px [derece]
    xi, eta = ara[..., 0], ara[..., 1]
    phi = np.arctan2(xi, -eta)  # arg(−y, x)
    r = np.hypot(xi, eta)
    teta = np.arctan2(180.0 / np.pi, r)  # atan(180°/(π R)); R = 0'da 90°
    ap, dp, pp = np.radians(wcs.crval[0]), np.radians(wcs.crval[1]), np.radians(wcs.lonpole)
    ra = ap + np.arctan2(
        -np.cos(teta) * np.sin(phi - pp),
        np.sin(teta) * np.cos(dp) - np.cos(teta) * np.sin(dp) * np.cos(phi - pp),
    )
    dec = np.arcsin(np.sin(teta) * np.sin(dp) + np.cos(teta) * np.cos(dp) * np.cos(phi - pp))
    return np.degrees(ra) % 360.0, np.degrees(dec)


@kaynak(
    "CalabrettaGreisen2002",
    yer="TeX etiketleri eq:std2nat, eq:TANRt, eq:azx, eq:azy, eq:px (ters)",
    teyit="okundu",
    not_="R = (180°/π) cot θ; x = R sin φ, y = −R cos φ; piksel = CD⁻¹ (x, y) + CRPIX",
)
def gok_den_piksel(wcs: Wcs, sag_acikllik_derece: ArrayLike, dik_acikllik_derece: ArrayLike) -> tuple[Dizi, Dizi]:
    """Sağ açıklık ve dik açıklığı sıfır tabanlı piksel koordinatlarına çevirir.

    Kaynaklar
    ---------
    [CalabrettaGreisen2002] eq:std2nat, eq:TANRt, eq:azx, eq:azy, eq:px.

    Geçerlilik
    ----------
    Teğet noktanın karşı yarıküresindeki yönler (θ <= 0) için ``nan`` döner.

    Doğrulama
    ---------
    tests/kahin/test_wcs_kahin.py: astropy.wcs.all_world2pix ile fark < 1e-7 piksel; gidiş-dönüş.
    """
    a = np.radians(np.asarray(sag_acikllik_derece, dtype=float))
    d = np.radians(np.asarray(dik_acikllik_derece, dtype=float))
    ap, dp, pp = np.radians(wcs.crval[0]), np.radians(wcs.crval[1]), np.radians(wcs.lonpole)
    # Doğal küresel birim vektörün bileşenleri: (cos θ sin(φ−φ_p), cos θ cos(φ−φ_p), sin θ).
    bx = -np.cos(d) * np.sin(a - ap)
    by = np.sin(d) * np.cos(dp) - np.cos(d) * np.sin(dp) * np.cos(a - ap)
    sin_teta = np.sin(d) * np.sin(dp) + np.cos(d) * np.cos(dp) * np.cos(a - ap)
    phi = pp + np.arctan2(bx, by)
    # R = (180°/π) cot θ. θ, arcsin ile bulunmaz: teğet nokta yakınında (θ ≈ 90°) arcsin basamak yitirir.
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(sin_teta > 0, 180.0 / np.pi * np.hypot(bx, by) / sin_teta, np.nan)
    ara = np.stack([r * np.sin(phi), -r * np.cos(phi)], axis=-1)
    p = ara @ np.linalg.inv(wcs.cd).T + wcs.crpix
    return p[..., 0] - 1.0, p[..., 1] - 1.0
