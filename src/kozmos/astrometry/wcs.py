"""FITS world coordinate system: pixel ↔ sky transformation (gnomonic TAN projection).

The equations were read from the arXiv TeX source of the second of the FITS WCS papers
(astro-ph/0207413, ``ccs.ltx``) [CalabrettaGreisen2002] and are cited with the
TeX labels used there:

- ``eq:px``: pixel → intermediate world coordinates, x_i = Σ_j (s_i m_ij)(p_j − r_j)
- ``eq:azphi``, ``eq:azrt``, ``eq:TANtheta``: plane → native spherical coordinates
- ``eq:nat2std``, ``eq:std2nat``: native ↔ celestial spherical rotation

The SIP distortion convention [Shupe2005] was read from the ADS scan of the 2005 paper
(``2005ASPC..347..491S``) and from the registered-convention document ``SIP_distortion_v1_0.pdf``
(version 1.0, 16 September 2008); the equation numbers below are the same in both:

- Eq. 1: (x, y) = CD (u + f(u, v), v + g(u, v)), with u, v relative to CRPIX
- Eqs. 2, 3: f = sum A_p_q u^p v^q (p + q <= A_ORDER), g = sum B_p_q u^p v^q (p + q <= B_ORDER)
- Eq. 4: (U, V) = CD^-1 (x, y)
- Eqs. 5, 6: u = U + sum AP_p_q U^p V^q, v = V + sum BP_p_q U^p V^q

Scope: two axes, ``RA---TAN`` / ``DEC--TAN`` projection, optionally with the ``-SIP`` suffix; ``CD``
matrix or ``CDELT`` + ``PC`` (or the old ``CROTA2``). ``PV`` terms and distortion tables are not supported.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.polynomial import polynomial as _poly_mod
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source

Array = NDArray[np.float64]


# Iteration controls of the numerical SIP inverse (this project's choice; see ``sky_to_pixel``).
_NEWTON_TOL_PIX = 1e-10
_NEWTON_MAX_ITER = 50


def _check_coefficients(c: Array | None, name: str) -> None:
    """Validate one SIP coefficient array: square, indexed [p, q], no entry with p + q above the order."""
    if c is None:
        return
    c = np.asarray(c)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] < 2:
        raise ValueError(f"{name} must be a square (order+1, order+1) array indexed [p, q]; got shape {c.shape}")
    p, q = np.indices(c.shape)
    if np.any(c[p + q > c.shape[0] - 1] != 0.0):
        raise ValueError(f"{name} has nonzero entries with p + q above its order {c.shape[0] - 1}")


@dataclass(frozen=True, slots=True)
class Wcs:
    """World coordinate system with two axes and a TAN projection, optionally with SIP distortion.

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
    sip_a, sip_b : (n+1, n+1) ndarray or None
        Forward SIP coefficients A_p_q, B_p_q [pixel / pixel^(p+q)] stored at index ``[p, q]``; the polynomial
        order is ``n = shape[0] - 1`` (A_ORDER, B_ORDER). Both or neither. ``None`` means no distortion.
    sip_ap, sip_bp : (m+1, m+1) ndarray or None
        Reverse SIP coefficients AP_p_q, BP_p_q, same layout. Both or neither; only meaningful with ``sip_a``.
        When ``None`` and ``sip_a`` is set, the inverse transformation is found numerically.
    """

    crpix: Array
    crval: Array
    cd: Array
    lonpole: float = 180.0
    sip_a: Array | None = None
    sip_b: Array | None = None
    sip_ap: Array | None = None
    sip_bp: Array | None = None

    def __post_init__(self) -> None:
        if (self.sip_a is None) != (self.sip_b is None):
            raise ValueError("sip_a and sip_b must be given together")
        if (self.sip_ap is None) != (self.sip_bp is None):
            raise ValueError("sip_ap and sip_bp must be given together")
        if self.sip_a is None and self.sip_ap is not None:
            raise ValueError("reverse SIP coefficients need forward SIP coefficients")
        for name in ("sip_a", "sip_b", "sip_ap", "sip_bp"):
            _check_coefficients(getattr(self, name), name)


def _poly2d(c: Array, u: Array, v: Array) -> Array:
    """Evaluate sum c[p, q] u^p v^q."""
    return _poly_mod.polyval2d(u, v, c)


def _read_sip_array(header: dict, prefix: str, order: int) -> Array:
    """Collect ``<prefix>_p_q`` keywords with p + q <= order into an array; absent coefficients are 0."""
    c = np.zeros((order + 1, order + 1))
    for p in range(order + 1):
        for q in range(order + 1 - p):
            c[p, q] = float(header.get(f"{prefix}_{p}_{q}", 0.0))
    return c


def _read_order(header: dict, key: str, lowest: int) -> int:
    value = header[key]
    order = int(value)
    if order != value or not lowest <= order <= 9:
        raise ValueError(f"{key} must be an integer from {lowest} to 9; got {value!r}")
    return order


@source(
    "CalabrettaGreisen2002",
    where="TeX label eq:px and the definition of the LONPOLE keyword",
    status="read",
    note="CD_ij = s_i m_ij (second form of eq:px). The LONPOLE default is 180° for δ_0 < θ_0; θ_0 = 90° for TAN. "
    "The rule for building the matrix from CROTA2 was not read in the paper; the common convention was applied "
    "and tested against astropy",
)
@source(
    "Shupe2005",
    where="Section 2, Eqs. 2, 3 (A_p_q, B_p_q, A_ORDER, B_ORDER), Eqs. 5, 6 (AP_p_q, BP_p_q), '-SIP' suffix of CTYPE",
    status="read",
    note="Read in the ADS scan 2005ASPC..347..491S and in SIP_distortion_v1_0.pdf (Version 1.0, 2008). The range "
    "2..9 of A_ORDER and B_ORDER and the default 0.0 of unspecified coefficients are stated only in the 2008 "
    "document. The allowed range 1..9 for AP_ORDER and BP_ORDER, and the errors raised for inconsistent keyword "
    "sets, are this project's choice",
)
def wcs_from_header(header: dict) -> Wcs:
    """Build a WCS with TAN projection, with SIP distortion if the header has it, from a FITS header.

    Order of precedence: ``CDi_j`` if present; otherwise ``CDELTi`` with ``PCi_j``; otherwise
    ``CDELTi`` with ``CROTA2`` (old convention).

    SIP is read when both ``CTYPE1`` and ``CTYPE2`` end in ``-SIP``: ``A_ORDER``, ``B_ORDER``,
    ``A_p_q``, ``B_p_q`` (p + q <= order), and, when present, ``AP_ORDER``, ``BP_ORDER``, ``AP_p_q``,
    ``BP_p_q``. Unspecified coefficients are 0.

    References
    ----------
    [CalabrettaGreisen2002] eq:px, definition of LONPOLE.
    [Shupe2005] Section 2, Eqs. 2, 3, 5, 6.

    Validity
    --------
    Only ``CTYPE1 = 'RA---TAN'`` / ``'RA---TAN-SIP'`` and ``CTYPE2 = 'DEC--TAN'`` / ``'DEC--TAN-SIP'``; any other
    projection or axis order raises ``ValueError``. ``PV`` terms are ignored. A header with SIP keywords but
    without the ``-SIP`` suffix is read as plain TAN (the keywords are ignored); astropy applies them in that
    case with an informational message, so the two differ there. Exactly one ``-SIP`` suffix, ``A_ORDER``
    without ``B_ORDER`` (or the reverse pair), or a ``-SIP`` header without ``A_ORDER`` and ``B_ORDER`` raises
    ``ValueError``.

    Verification
    ------------
    tests/oracle/test_wcs_oracle.py: same transformation as astropy.wcs from the same headers.
    tests/oracle/test_sip_oracle.py: SIP headers read the same coefficients as astropy.wcs.
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
    sip: dict[str, Array] = {}
    flags = (str(header["CTYPE1"]).strip().endswith("-SIP"), str(header["CTYPE2"]).strip().endswith("-SIP"))
    if flags[0] != flags[1]:
        raise ValueError("only one of CTYPE1 and CTYPE2 ends in '-SIP'")
    if all(flags):
        if "A_ORDER" not in header or "B_ORDER" not in header:
            raise ValueError("CTYPE ends in '-SIP' but A_ORDER and B_ORDER are not both present")
        sip["sip_a"] = _read_sip_array(header, "A", _read_order(header, "A_ORDER", 2))
        sip["sip_b"] = _read_sip_array(header, "B", _read_order(header, "B_ORDER", 2))
        if ("AP_ORDER" in header) != ("BP_ORDER" in header):
            raise ValueError("AP_ORDER and BP_ORDER must be given together")
        if "AP_ORDER" in header:
            sip["sip_ap"] = _read_sip_array(header, "AP", _read_order(header, "AP_ORDER", 1))
            sip["sip_bp"] = _read_sip_array(header, "BP", _read_order(header, "BP_ORDER", 1))
    return Wcs(
        crpix=np.array([header["CRPIX1"], header["CRPIX2"]], dtype=float),
        crval=np.array([header["CRVAL1"], header["CRVAL2"]], dtype=float),
        cd=cd,
        lonpole=float(header.get("LONPOLE", 180.0)),
        **sip,
    )


@source(
    "CalabrettaGreisen2002",
    where="TeX labels eq:px, eq:azphi, eq:azrt, eq:TANtheta, eq:nat2std",
    status="read",
    note="φ = arg(−y, x), R = sqrt(x² + y²), θ = atan(180°/(π R)); then native → celestial rotation (α_p, δ_p) = CRVAL",
)
@source(
    "Shupe2005",
    where="Eq. 1 with Eqs. 2, 3",
    status="read",
    note="(x, y) = CD (u + f(u, v), v + g(u, v)); u, v are FITS pixel coordinates minus CRPIX. Without sip_a and "
    "sip_b the distortion terms are zero and the result is the plain TAN transformation",
)
def pixel_to_sky(wcs: Wcs, x: ArrayLike, y: ArrayLike) -> tuple[Array, Array]:
    """Convert pixel coordinates to right ascension and declination.

    With SIP coefficients the relative pixel coordinates (u, v) are replaced by
    (u + f(u, v), v + g(u, v)) before the CD matrix is applied.

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
    [Shupe2005] Eqs. 1, 2, 3.

    Validity
    --------
    The TAN projection diverges 90° from the tangent point; distortion grows over wide fields. The SIP
    polynomial is evaluated as written, with no check that the pixel lies inside the region the coefficients were
    fitted over; outside it a polynomial of order 3 or more can grow without bound.

    Verification
    ------------
    tests/oracle/test_wcs_oracle.py: difference from astropy.wcs.all_pix2world < 1e-9 deg.
    tests/oracle/test_sip_oracle.py: with SIP, difference from astropy.wcs.all_pix2world < 1e-9 deg.
    """
    p = np.stack(np.broadcast_arrays(np.asarray(x, dtype=float) + 1.0, np.asarray(y, dtype=float) + 1.0), axis=-1)
    relative = p - wcs.crpix
    if wcs.sip_a is not None and wcs.sip_b is not None:
        u, v = relative[..., 0], relative[..., 1]
        relative = np.stack([u + _poly2d(wcs.sip_a, u, v), v + _poly2d(wcs.sip_b, u, v)], axis=-1)  # Eq. 1
    intermediate = relative @ wcs.cd.T  # eq:px [deg]
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


def _invert_sip(
    a: Array, b: Array, uc: Array, vc: Array, tol: float = _NEWTON_TOL_PIX, max_iter: int = _NEWTON_MAX_ITER
) -> tuple[Array, Array, int, Array]:
    """Solve u + f(u, v) = U, v + g(u, v) = V by Newton iteration from the start (u, v) = (U, V).

    Returns u, v, the number of iterations used, and a boolean array that is True where the last Newton step
    was shorter than ``tol`` [pixel]. Entries that did not converge, or whose input is not finite, are ``nan``.
    """
    a_u, a_v = _poly_mod.polyder(a, axis=0), _poly_mod.polyder(a, axis=1)
    b_u, b_v = _poly_mod.polyder(b, axis=0), _poly_mod.polyder(b, axis=1)
    u, v = uc.copy(), vc.copy()
    step = np.full(uc.shape, np.inf)
    step[~(np.isfinite(uc) & np.isfinite(vc))] = 0.0  # nan input stays nan and does not hold the loop
    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        ru = u + _poly2d(a, u, v) - uc
        rv = v + _poly2d(b, u, v) - vc
        j11, j12 = 1.0 + _poly2d(a_u, u, v), _poly2d(a_v, u, v)
        j21, j22 = _poly2d(b_u, u, v), 1.0 + _poly2d(b_v, u, v)
        det = j11 * j22 - j12 * j21
        with np.errstate(divide="ignore", invalid="ignore"):
            du = (j22 * ru - j12 * rv) / det
            dv = (j11 * rv - j21 * ru) / det
        u, v = u - du, v - dv
        step = np.where(np.isfinite(uc) & np.isfinite(vc), np.hypot(du, dv), 0.0)
        if np.all(step < tol):
            break
    converged = step < tol
    u = np.where(converged, u, np.nan)
    v = np.where(converged, v, np.nan)
    return u, v, iterations, converged


@source(
    "CalabrettaGreisen2002",
    where="TeX labels eq:std2nat, eq:TANRt, eq:azx, eq:azy, eq:px (inverse)",
    status="read",
    note="R = (180°/π) cot θ; x = R sin φ, y = −R cos φ; pixel = CD⁻¹ (x, y) + CRPIX",
)
@source(
    "Shupe2005",
    where="Eqs. 4, 5, 6 (reverse polynomials); Eq. 1 for the numerical inverse",
    status="read",
    note="With AP/BP: (U, V) = CD⁻¹ (x, y), u = U + AP(U, V), v = V + BP(U, V) (Eqs. 4-6). Without them the "
    "forward relation of Eq. 1, U = u + f(u, v), V = v + g(u, v), is solved by Newton iteration with the analytic "
    "Jacobian, started at (u, v) = (U, V). The Newton scheme, the step tolerance 1e-10 pixel and the limit of 50 "
    "iterations are this project's choice; the paper does not prescribe an inversion method",
)
def sky_to_pixel(wcs: Wcs, ra_deg: ArrayLike, dec_deg: ArrayLike) -> tuple[Array, Array]:
    """Convert right ascension and declination to zero-based pixel coordinates.

    With SIP coefficients the intermediate coordinates are first mapped to corrected pixel coordinates
    (U, V) = CD⁻¹ (x, y). Then, if the reverse coefficients are present, u = U + AP(U, V), v = V + BP(U, V);
    otherwise (u, v) is found by Newton iteration on u + f(u, v) = U, v + g(u, v) = V.

    References
    ----------
    [CalabrettaGreisen2002] eq:std2nat, eq:TANRt, eq:azx, eq:azy, eq:px.
    [Shupe2005] Eqs. 4, 5, 6, 1.

    Validity
    --------
    Returns ``nan`` for directions in the hemisphere opposite the tangent point (θ <= 0), and ``nan`` where
    the Newton iteration does not converge (for example where the Jacobian of the forward polynomial is
    singular). The reverse polynomials of the header are used as given and are only as accurate as the header
    makes them; they are not refined. Measured accuracy of the Newton inverse
    (order-3 SIP with random coefficients, 2048 x 2048 field, 100000 random pixels, 0.7 arcsec/pixel, step
    tolerance 1e-10 pixel; distortion = largest |f|, |g| over the field): for 1, 3, 10 and 30 pixel the iteration
    count (including the final confirming step) is 3, 3, 4, 4 and all pixels converge; the maximum error of the
    inverse alone is 1.1e-13 to 1.6e-13 pixel; the maximum round-trip error pixel → sky → pixel is 1.9e-10 pixel
    for every case, which is set by the double-precision RA/Dec in degrees, not by the iteration. For
    comparison, a plain fixed-point iteration u = U − f(u, v) needed 6, 7, 9 and 13 iterations to the same
    tolerance (exploratory run, not part of the test suite). The reverse polynomials of a header are used as
    given: their accuracy is that of the header (an AP/BP set fitted here by least squares to order 4 to 6 had
    maximum errors of 4e-5 to 1.8e-3 pixel).

    Verification
    ------------
    tests/oracle/test_wcs_oracle.py: difference from astropy.wcs.all_world2pix < 1e-7 pixel; round trip.
    tests/oracle/test_sip_oracle.py: reverse-polynomial path against astropy's SIP reverse polynomials; Newton
    path against astropy.wcs.all_world2pix and by round trip.
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
    p = intermediate @ np.linalg.inv(wcs.cd).T  # Eq. 4: (U, V), relative to CRPIX
    if wcs.sip_a is not None and wcs.sip_b is not None:
        uc, vc = p[..., 0], p[..., 1]
        if wcs.sip_ap is not None and wcs.sip_bp is not None:
            p = np.stack([uc + _poly2d(wcs.sip_ap, uc, vc), vc + _poly2d(wcs.sip_bp, uc, vc)], axis=-1)  # Eqs. 5, 6
        else:
            u, v, _, _ = _invert_sip(wcs.sip_a, wcs.sip_b, uc, vc)
            p = np.stack([u, v], axis=-1)
    p = p + wcs.crpix
    return p[..., 0] - 1.0, p[..., 1] - 1.0


def _angular_separation_arcsec(ra1: Array, dec1: Array, ra2: Array, dec2: Array) -> Array:
    """Great-circle separation [arcsec] by the Vincenty form, accurate for small and large angles alike."""
    r1, d1, r2, d2 = (np.radians(t) for t in (ra1, dec1, ra2, dec2))
    dr = r2 - r1
    num = np.hypot(np.cos(d2) * np.sin(dr), np.cos(d1) * np.sin(d2) - np.sin(d1) * np.cos(d2) * np.cos(dr))
    den = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(dr)
    return np.degrees(np.arctan2(num, den)) * 3600.0


@source(
    "Shupe2005",
    where="Eq. 1 with Eqs. 2, 3 (definition of the polynomial)",
    status="read",
    note="The fit is this project's choice, not the paper's: with the linear WCS held fixed, the corrected pixel "
    "coordinates (U, V) = CD⁻¹ (x, y) of each star come from its catalog position by the inverse TAN projection, "
    "and f = U − u, g = V − v are fitted for A_p_q, B_p_q (2 <= p + q <= order) by ordinary linear least squares "
    "on monomials scaled by the largest |u|, |v|. All noise is assumed to sit in (U, V)",
)
def fit_sip(
    wcs: Wcs, x: ArrayLike, y: ArrayLike, ra_deg: ArrayLike, dec_deg: ArrayLike, order: int
) -> tuple[Wcs, float]:
    """Fit forward SIP coefficients to matched stars with the linear WCS held fixed.

    For each star, u, v are its measured zero-based pixel position (as FITS coordinates minus CRPIX) and
    (U, V) = CD⁻¹ (x, y) is the position the catalog direction has under the plain TAN solution. Eq. 1 then
    reads U = u + f(u, v), V = v + g(u, v), linear in the coefficients A_p_q, B_p_q with 2 <= p + q <= order.

    Parameters
    ----------
    wcs : Wcs
        Linear WCS (``crpix``, ``crval``, ``cd``, ``lonpole``) that stays fixed. Any SIP coefficients it
        carries are ignored.
    x, y : ndarray
        Measured zero-based pixel positions of the stars.
    ra_deg, dec_deg : ndarray
        Catalog right ascension and declination of the same stars [deg].
    order : int
        Polynomial order, A_ORDER = B_ORDER, from 2 to 9.

    Returns
    -------
    (Wcs, float)
        New WCS with ``sip_a`` and ``sip_b`` set (no reverse coefficients, so ``sky_to_pixel`` inverts
        numerically), and the root-mean-square separation [arcsec] between ``pixel_to_sky`` of the new WCS and the
        catalog positions of the same stars. The rms is not corrected for the fitted degrees of freedom.

    References
    ----------
    [Shupe2005] Eq. 1, Eqs. 2, 3.

    Validity
    --------
    Needs more stars than coefficients and stars that are not all on one curve of degree ``order`` (otherwise
    ``ValueError``). The fit is only as good as the stars' coverage: the polynomial is unconstrained and can
    grow where there are no stars. The linear part of the WCS is not refitted, so a wrong CRVAL or CD is
    absorbed only to the extent that quadratic and higher terms can mimic it. Measured recovery
    (exploratory run of 20 noise realisations, not part of the test suite; 2048-pixel field, 0.7 arcsec/pixel,
    order-3 truth with 3 pixel distortion at the corners, 0.05 pixel Gaussian noise on both measured pixel
    coordinates): 300 stars give a residual rms median of 0.0488 arcsec (the noise alone predicts
    0.05 x 0.7 x sqrt 2 = 0.0495 arcsec), the largest coefficient error (each A_p_q times 1024^(p+q)) 0.038
    pixel and the largest error of the distortion over the field 0.063 pixel; 1000 stars give 0.020 and 0.032
    pixel. A fit of order 4 to an order-4 truth with 300 stars gives 0.104 and 0.124 pixel. The test suite
    repeats the experiment with fixed seeds (tests/oracle/test_sip_oracle.py::test_fit_sip_recovers_known_distortion).

    Verification
    ------------
    tests/oracle/test_sip_oracle.py: stars generated through a known SIP WCS with 0.05 pixel noise; the
    corrected coordinates are recovered and the residual rms is reported.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    ra = np.atleast_1d(np.asarray(ra_deg, dtype=float))
    dec = np.atleast_1d(np.asarray(dec_deg, dtype=float))
    if not (x.shape == y.shape == ra.shape == dec.shape) or x.ndim != 1:
        raise ValueError("x, y, ra_deg and dec_deg must be one-dimensional arrays of the same length")
    if int(order) != order or not 2 <= order <= 9:
        raise ValueError(f"order must be an integer from 2 to 9; got {order!r}")
    order = int(order)
    linear = replace(wcs, sip_a=None, sip_b=None, sip_ap=None, sip_bp=None)
    xl, yl = sky_to_pixel(linear, ra, dec)  # where the linear WCS puts each catalog star
    if not (np.all(np.isfinite(xl)) and np.all(np.isfinite(yl)) and np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("every star must be finite and on the tangent-point hemisphere")
    u, v = x + 1.0 - wcs.crpix[0], y + 1.0 - wcs.crpix[1]
    big_u, big_v = xl + 1.0 - wcs.crpix[0], yl + 1.0 - wcs.crpix[1]
    terms = [(p, q) for total in range(2, order + 1) for p in range(total + 1) for q in (total - p,)]
    if x.size <= len(terms):
        raise ValueError(f"{x.size} stars are not enough for {len(terms)} coefficients per axis")
    scale = float(max(np.max(np.abs(u)), np.max(np.abs(v))))
    if scale == 0.0:
        raise ValueError("all stars are at the reference pixel")
    design = np.stack([(u / scale) ** p * (v / scale) ** q for p, q in terms], axis=1)
    coef_f, _, rank, _ = np.linalg.lstsq(design, big_u - u, rcond=None)
    coef_g = np.linalg.lstsq(design, big_v - v, rcond=None)[0]
    if rank < len(terms):
        raise ValueError(f"the star positions do not constrain all {len(terms)} coefficients (rank {rank})")
    sip_a, sip_b = np.zeros((order + 1, order + 1)), np.zeros((order + 1, order + 1))
    for (p, q), cf, cg in zip(terms, coef_f, coef_g, strict=True):
        sip_a[p, q] = cf / scale ** (p + q)
        sip_b[p, q] = cg / scale ** (p + q)
    fitted = replace(linear, sip_a=sip_a, sip_b=sip_b)
    ra_fit, dec_fit = pixel_to_sky(fitted, x, y)
    rms_arcsec = float(np.sqrt(np.mean(_angular_separation_arcsec(ra_fit, dec_fit, ra, dec) ** 2)))
    return fitted, rms_arcsec
