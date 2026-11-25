"""FITS world coordinate system: pixel ↔ sky transformation (gnomonic TAN projection).

The equations were read from the arXiv TeX source of the second of the FITS WCS papers
(astro-ph/0207413, ``ccs.ltx``) [CalabrettaGreisen2002] and are cited with the
TeX labels used there:

- ``eq:px``: pixel → intermediate world coordinates, x_i = Σ_j (s_i m_ij)(p_j − r_j)
- ``eq:azphi``, ``eq:azrt``, ``eq:TANtheta``: plane → native spherical coordinates
- ``eq:nat2std``, ``eq:std2nat``: native ↔ celestial spherical rotation

Scope: two axes, ``RA---TAN`` / ``DEC--TAN`` projection; ``CD`` matrix or
``CDELT`` + ``PC`` (or the old ``CROTA2``). There are no distortion polynomials (SIP).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Wcs:
    """World coordinate system with two axes and a TAN projection.

    Attributes
    ----------
    crpix : (2,) ndarray
        FITS coordinates of the reference pixel (the first pixel is 1).
    crval : (2,) ndarray
        Right ascension and declination of the reference point [deg].
    cd : (2, 2) ndarray
        Pixel → intermediate world coordinate matrix [deg/pixel].
    lonpole : float
        Native longitude φ_p of the celestial pole [deg]; default 180 for TAN.
    """

    crpix: Array
    crval: Array
    cd: Array
    lonpole: float = 180.0


@source(
    "CalabrettaGreisen2002",
    where="TeX label eq:px and the definition of the LONPOLE keyword",
    status="read",
    note="CD_ij = s_i m_ij (second form of eq:px). The LONPOLE default is 180° for δ_0 < θ_0; θ_0 = 90° for TAN. "
    "The rule for building the matrix from CROTA2 was not read in the paper; the common convention was applied "
    "and tested against astropy",
)
def wcs_from_header(header: dict) -> Wcs:
    """Build a WCS with TAN projection from a FITS header.

    Order of precedence: ``CDi_j`` if present; otherwise ``CDELTi`` with ``PCi_j``; otherwise
    ``CDELTi`` with ``CROTA2`` (old convention).

    References
    ----------
    [CalabrettaGreisen2002] eq:px, definition of LONPOLE.

    Validity
    --------
    Only ``CTYPE1 = 'RA---TAN'``, ``CTYPE2 = 'DEC--TAN'``; any other projection or
    axis order raises ``ValueError``. The SIP and ``PV`` distortion terms are ignored.

    Verification
    ------------
    tests/oracle/test_wcs_oracle.py: same transformation as astropy.wcs from the same headers.
    """
    if not (
        str(header.get("CTYPE1", "")).startswith("RA---TAN") and str(header.get("CTYPE2", "")).startswith("DEC--TAN")
    ):
        raise ValueError(f"only RA---TAN / DEC--TAN is supported: {header.get('CTYPE1')!r}, {header.get('CTYPE2')!r}")
    if "CD1_1" in header:
        cd = np.array(
            [[header["CD1_1"], header.get("CD1_2", 0.0)], [header.get("CD2_1", 0.0), header["CD2_2"]]], dtype=float
        )
    else:
        scale = np.array([header["CDELT1"], header["CDELT2"]], dtype=float)
        if "PC1_1" in header:
            pc = np.array(
                [[header["PC1_1"], header.get("PC1_2", 0.0)], [header.get("PC2_1", 0.0), header.get("PC2_2", 1.0)]],
                dtype=float,
            )
            cd = scale[:, None] * pc
        else:
            rho = np.radians(float(header.get("CROTA2", 0.0)))
            cd = np.array(
                [[scale[0] * np.cos(rho), -scale[1] * np.sin(rho)], [scale[0] * np.sin(rho), scale[1] * np.cos(rho)]]
            )
    return Wcs(
        crpix=np.array([header["CRPIX1"], header["CRPIX2"]], dtype=float),
        crval=np.array([header["CRVAL1"], header["CRVAL2"]], dtype=float),
        cd=cd,
        lonpole=float(header.get("LONPOLE", 180.0)),
    )


@source(
    "CalabrettaGreisen2002",
    where="TeX labels eq:px, eq:azphi, eq:azrt, eq:TANtheta, eq:nat2std",
    status="read",
    note="φ = arg(−y, x), R = sqrt(x² + y²), θ = atan(180°/(π R)); then native → celestial rotation (α_p, δ_p) = CRVAL",
)
def pixel_to_sky(wcs: Wcs, x: ArrayLike, y: ArrayLike) -> tuple[Array, Array]:
    """Convert pixel coordinates to right ascension and declination.

    Parameters
    ----------
    wcs : Wcs
    x, y : float or ndarray
        Zero-based pixel coordinates (``image[y, x]``; the center of the first pixel is 0).

    Returns
    -------
    (right ascension, declination) [deg]; right ascension in [0, 360).

    References
    ----------
    [CalabrettaGreisen2002] eq:px, eq:azphi, eq:azrt, eq:TANtheta, eq:nat2std.

    Validity
    --------
    The TAN projection diverges 90° from the tangent point; distortion grows over wide fields.

    Verification
    ------------
    tests/oracle/test_wcs_oracle.py: difference from astropy.wcs.all_pix2world < 1e-9 deg.
    """
    p = np.stack(np.broadcast_arrays(np.asarray(x, dtype=float) + 1.0, np.asarray(y, dtype=float) + 1.0), axis=-1)
    intermediate = (p - wcs.crpix) @ wcs.cd.T  # eq:px [deg]
    xi, eta = intermediate[..., 0], intermediate[..., 1]
    phi = np.arctan2(xi, -eta)  # arg(−y, x)
    r = np.hypot(xi, eta)
    theta = np.arctan2(180.0 / np.pi, r)  # atan(180°/(π R)); 90° at R = 0
    ap, dp, pp = np.radians(wcs.crval[0]), np.radians(wcs.crval[1]), np.radians(wcs.lonpole)
    ra = ap + np.arctan2(
        -np.cos(theta) * np.sin(phi - pp),
        np.sin(theta) * np.cos(dp) - np.cos(theta) * np.sin(dp) * np.cos(phi - pp),
    )
    dec = np.arcsin(np.sin(theta) * np.sin(dp) + np.cos(theta) * np.cos(dp) * np.cos(phi - pp))
    return np.degrees(ra) % 360.0, np.degrees(dec)


@source(
    "CalabrettaGreisen2002",
    where="TeX labels eq:std2nat, eq:TANRt, eq:azx, eq:azy, eq:px (inverse)",
    status="read",
    note="R = (180°/π) cot θ; x = R sin φ, y = −R cos φ; pixel = CD⁻¹ (x, y) + CRPIX",
)
def sky_to_pixel(wcs: Wcs, ra_deg: ArrayLike, dec_deg: ArrayLike) -> tuple[Array, Array]:
    """Convert right ascension and declination to zero-based pixel coordinates.

    References
    ----------
    [CalabrettaGreisen2002] eq:std2nat, eq:TANRt, eq:azx, eq:azy, eq:px.

    Validity
    --------
    Returns ``nan`` for directions in the hemisphere opposite the tangent point (θ <= 0).

    Verification
    ------------
    tests/oracle/test_wcs_oracle.py: difference from astropy.wcs.all_world2pix < 1e-7 pixel; round trip.
    """
    a = np.radians(np.asarray(ra_deg, dtype=float))
    d = np.radians(np.asarray(dec_deg, dtype=float))
    ap, dp, pp = np.radians(wcs.crval[0]), np.radians(wcs.crval[1]), np.radians(wcs.lonpole)
    # Components of the native spherical unit vector: (cos θ sin(φ−φ_p), cos θ cos(φ−φ_p), sin θ).
    bx = -np.cos(d) * np.sin(a - ap)
    by = np.sin(d) * np.cos(dp) - np.cos(d) * np.sin(dp) * np.cos(a - ap)
    sin_theta = np.sin(d) * np.sin(dp) + np.cos(d) * np.cos(dp) * np.cos(a - ap)
    phi = pp + np.arctan2(bx, by)
    # R = (180°/π) cot θ. θ is not found with arcsin: near the tangent point (θ ≈ 90°) arcsin loses digits.
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(sin_theta > 0, 180.0 / np.pi * np.hypot(bx, by) / sin_theta, np.nan)
    intermediate = np.stack([r * np.sin(phi), -r * np.cos(phi)], axis=-1)
    p = intermediate @ np.linalg.inv(wcs.cd).T + wcs.crpix
    return p[..., 0] - 1.0, p[..., 1] - 1.0
