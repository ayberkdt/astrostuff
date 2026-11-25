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

from kozmos._sources import source

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Wcs:
    """İki eksenli TAN izdüşümlü dünya koordinat sistemi.

    Attributes
    ----------
    crpix : (2,) ndarray
        Referans pikselin FITS koordinatları (ilk piksel 1'dir).
    crval : (2,) ndarray
        Referans noktanın sağ açıklığı ve dik açıklığı [derece].
    cd : (2, 2) ndarray
        Piksel → ara dünya koordinatları matrisi [derece/piksel].
    lonpole : float
        Gök kutbunun doğal boylamı φ_p [derece]; TAN için varsayılan 180.
    """

    crpix: Array
    crval: Array
    cd: Array
    lonpole: float = 180.0


@source(
    "CalabrettaGreisen2002",
    where="TeX etiketi eq:px ve LONPOLE anahtarının tanımı",
    status="read",
    note="CD_ij = s_i m_ij (eq:px'in ikinci biçimi). LONPOLE varsayılanı δ_0 < θ_0 için 180°; TAN'da θ_0 = 90°. "
    "CROTA2'den matris kurma kuralı makalede okunmadı; yaygın kural uygulandı ve astropy ile sınandı",
)
def wcs_from_header(header: dict) -> Wcs:
    """FITS başlığından TAN izdüşümlü WCS kurar.

    Öncelik sırası: ``CDi_j`` varsa o; yoksa ``CDELTi`` ile ``PCi_j``; o da yoksa
    ``CDELTi`` ile ``CROTA2`` (eski kural).

    References
    ----------
    [CalabrettaGreisen2002] eq:px, LONPOLE tanımı.

    Validity
    --------
    Yalnızca ``CTYPE1 = 'RA---TAN'``, ``CTYPE2 = 'DEC--TAN'``; başka izdüşüm ya da
    eksen sırası ``ValueError`` verir. SIP ve ``PV`` bozulma terimleri yok sayılır.

    Verification
    ------------
    tests/oracle/test_wcs_oracle.py: astropy.wcs ile aynı başlıklardan aynı dönüşüm.
    """
    if not (
        str(header.get("CTYPE1", "")).startswith("RA---TAN") and str(header.get("CTYPE2", "")).startswith("DEC--TAN")
    ):
        raise ValueError(
            f"yalnızca RA---TAN / DEC--TAN desteklenir: {header.get('CTYPE1')!r}, {header.get('CTYPE2')!r}"
        )
    if "CD1_1" in header:
        cd = np.array(
            [[header["CD1_1"], header.get("CD1_2", 0.0)], [header.get("CD2_1", 0.0), header["CD2_2"]]], dtype=float
        )
    else:
        olcek = np.array([header["CDELT1"], header["CDELT2"]], dtype=float)
        if "PC1_1" in header:
            pc = np.array(
                [[header["PC1_1"], header.get("PC1_2", 0.0)], [header.get("PC2_1", 0.0), header.get("PC2_2", 1.0)]],
                dtype=float,
            )
            cd = olcek[:, None] * pc
        else:
            rho = np.radians(float(header.get("CROTA2", 0.0)))
            cd = np.array(
                [[olcek[0] * np.cos(rho), -olcek[1] * np.sin(rho)], [olcek[0] * np.sin(rho), olcek[1] * np.cos(rho)]]
            )
    return Wcs(
        crpix=np.array([header["CRPIX1"], header["CRPIX2"]], dtype=float),
        crval=np.array([header["CRVAL1"], header["CRVAL2"]], dtype=float),
        cd=cd,
        lonpole=float(header.get("LONPOLE", 180.0)),
    )


@source(
    "CalabrettaGreisen2002",
    where="TeX etiketleri eq:px, eq:azphi, eq:azrt, eq:TANtheta, eq:nat2std",
    status="read",
    note="φ = arg(−y, x), R = sqrt(x² + y²), θ = atan(180°/(π R)); ardından doğal → gök dönmesi (α_p, δ_p) = CRVAL",
)
def pixel_to_sky(wcs: Wcs, x: ArrayLike, y: ArrayLike) -> tuple[Array, Array]:
    """Piksel koordinatlarını sağ açıklık ve dik açıklığa çevirir.

    Parameters
    ----------
    wcs : Wcs
    x, y : float ya da ndarray
        Sıfır tabanlı piksel koordinatları (``goruntu[y, x]``; ilk pikselin merkezi 0).

    Returns
    -------
    (sağ açıklık, dik açıklık) [derece]; sağ açıklık [0, 360).

    References
    ----------
    [CalabrettaGreisen2002] eq:px, eq:azphi, eq:azrt, eq:TANtheta, eq:nat2std.

    Validity
    --------
    TAN izdüşümü teğet noktadan 90° uzakta ıraksar; geniş alanlarda bozulma büyür.

    Verification
    ------------
    tests/oracle/test_wcs_oracle.py: astropy.wcs.all_pix2world ile fark < 1e-9 derece.
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


@source(
    "CalabrettaGreisen2002",
    where="TeX etiketleri eq:std2nat, eq:TANRt, eq:azx, eq:azy, eq:px (ters)",
    status="read",
    note="R = (180°/π) cot θ; x = R sin φ, y = −R cos φ; piksel = CD⁻¹ (x, y) + CRPIX",
)
def sky_to_pixel(wcs: Wcs, ra_deg: ArrayLike, dec_deg: ArrayLike) -> tuple[Array, Array]:
    """Sağ açıklık ve dik açıklığı sıfır tabanlı piksel koordinatlarına çevirir.

    References
    ----------
    [CalabrettaGreisen2002] eq:std2nat, eq:TANRt, eq:azx, eq:azy, eq:px.

    Validity
    --------
    Teğet noktanın karşı yarıküresindeki yönler (θ <= 0) için ``nan`` döner.

    Verification
    ------------
    tests/oracle/test_wcs_oracle.py: astropy.wcs.all_world2pix ile fark < 1e-7 piksel; gidiş-dönüş.
    """
    a = np.radians(np.asarray(ra_deg, dtype=float))
    d = np.radians(np.asarray(dec_deg, dtype=float))
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
