"""SIP distortion: comparison with astropy.wcs and measurements of the numerical inverse and of ``fit_sip``.

Reference definitions: [Shupe2005] Eqs. 1-6 (ADS scan 2005ASPC..347..491S and SIP_distortion_v1_0.pdf).

Tolerances (measured values are the ones recorded in the docstrings of ``sky_to_pixel`` and ``fit_sip``):

- forward direction, ``kozmos`` vs ``astropy.wcs.all_pix2world``: both evaluate the same polynomial and the same
  TAN projection in double precision; measured 5e-14 deg, tolerance 1e-11 deg (0.036 micro-arcsec).
- reverse polynomial, ``kozmos.sky_to_pixel`` vs ``astropy`` ``Sip.foc2pix`` (same AP/BP): measured 2e-10 pixel,
  tolerance 1e-7 pixel.
- numerical inverse, round trip: limited by the double-precision RA/Dec in degrees (about 1e-13 deg, which is
  about 2e-10 pixel at 0.72 arcsec/pixel); tolerance 1e-8 pixel.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from kozmos.astrometry import wcs as kw

pytestmark = pytest.mark.oracle

HALF = 1024.0  # half of the 2048-pixel field


def _header(a_order, b_order, rotation_deg, seed, reverse_order=None):
    """TAN-SIP header; each coefficient term contributes about 1 pixel at 1024 pixels from CRPIX."""
    r = np.random.default_rng(seed)
    th, s = np.radians(rotation_deg), 2.0e-4
    h = {
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "CRPIX1": 1000.5,
        "CRPIX2": 1030.0,
        "CRVAL1": 150.0,
        "CRVAL2": -30.0,
        "CD1_1": -s * np.cos(th),
        "CD1_2": s * np.sin(th) * 1.02,
        "CD2_1": s * np.sin(th),
        "CD2_2": s * np.cos(th) * 0.98,
        "A_ORDER": a_order,
        "B_ORDER": b_order,
    }
    for name, order in (("A", a_order), ("B", b_order)):
        for p in range(order + 1):
            for q in range(order + 1 - p):
                if p + q >= 2:
                    h[f"{name}_{p}_{q}"] = float(r.normal() / HALF ** (p + q))
    if reverse_order is not None:
        h.update(_reverse_keywords(h, reverse_order))
    return h


def _reverse_keywords(header, order):
    """AP/BP keywords fitted (least squares on a 41 x 41 grid) so that they invert the forward polynomial."""
    k = kw.wcs_from_header(header)
    g = np.linspace(-1.1 * HALF, 1.1 * HALF, 41)
    u0, v0 = (t.ravel() for t in np.meshgrid(g, g))
    big_u = u0 + kw._poly2d(k.sip_a, u0, v0)
    big_v = v0 + kw._poly2d(k.sip_b, u0, v0)
    terms = [(p, t - p) for t in range(1, order + 1) for p in range(t + 1)]
    scale = 1.1 * HALF
    design = np.stack([(big_u / scale) ** p * (big_v / scale) ** q for p, q in terms], axis=1)
    out = {"AP_ORDER": order, "BP_ORDER": order}
    for name, target in (("AP", u0 - big_u), ("BP", v0 - big_v)):
        coef = np.linalg.lstsq(design, target, rcond=None)[0]
        for (p, q), c in zip(terms, coef, strict=True):
            out[f"{name}_{p}_{q}"] = float(c / scale ** (p + q))
    return out


# (A_ORDER, B_ORDER, rotation [deg], seed, reverse polynomial order or None)
CASES = [
    pytest.param(2, 2, 0.0, 11, None, id="order2-norot-forward-only"),
    pytest.param(3, 3, 30.0, 12, None, id="order3-rot30-forward-only"),
    pytest.param(3, 3, 30.0, 12, 5, id="order3-rot30-with-reverse"),
    pytest.param(4, 4, -70.0, 13, 6, id="order4-rot-70-with-reverse"),
    pytest.param(4, 4, 115.0, 14, None, id="order4-rot115-forward-only"),
    pytest.param(3, 2, 200.0, 15, 4, id="orders-3-and-2-rot200-with-reverse"),
]


def _field_points(n=500, seed=21):
    r = np.random.default_rng(seed)
    return r.uniform(0, 2047, n), r.uniform(0, 2047, n)


def _same(ours, theirs):
    """astropy goes through the FITS card text, which keeps 16 significant digits: compare to 1e-12 relative."""
    return ours.shape == theirs.shape and np.allclose(ours, theirs, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize(("a_order", "b_order", "rotation", "seed", "reverse"), CASES)
def test_header_reads_the_same_coefficients_as_astropy(a_order, b_order, rotation, seed, reverse):
    h = _header(a_order, b_order, rotation, seed, reverse)
    ours, theirs = kw.wcs_from_header(h), WCS(h).sip
    assert _same(ours.sip_a, theirs.a) and _same(ours.sip_b, theirs.b)
    if reverse is None:
        assert ours.sip_ap is None and ours.sip_bp is None and theirs.ap is None
    else:
        assert _same(ours.sip_ap, theirs.ap) and _same(ours.sip_bp, theirs.bp)


@pytest.mark.parametrize(("a_order", "b_order", "rotation", "seed", "reverse"), CASES)
def test_pixel_to_sky_against_astropy(a_order, b_order, rotation, seed, reverse):
    h = _header(a_order, b_order, rotation, seed, reverse)
    ours, theirs = kw.wcs_from_header(h), WCS(h)
    x, y = _field_points()
    ra, dec = kw.pixel_to_sky(ours, x, y)
    ra_a, dec_a = theirs.all_pix2world(x, y, 0)
    separation = SkyCoord(ra * u.deg, dec * u.deg).separation(SkyCoord(ra_a * u.deg, dec_a * u.deg)).deg
    assert separation.max() < 1e-11
    # The distortion must be large enough for the comparison to mean something (a few pixels).
    plain = kw.Wcs(crpix=ours.crpix, crval=ours.crval, cd=ours.cd)
    ra0, dec0 = kw.pixel_to_sky(plain, x, y)
    shift_pixel = SkyCoord(ra * u.deg, dec * u.deg).separation(SkyCoord(ra0 * u.deg, dec0 * u.deg)).deg / 2.0e-4
    assert shift_pixel.max() > 1.0


@pytest.mark.parametrize(("a_order", "b_order", "rotation", "seed", "reverse"), CASES)
def test_sky_to_pixel_against_astropy(a_order, b_order, rotation, seed, reverse):
    h = _header(a_order, b_order, rotation, seed, reverse)
    ours, theirs = kw.wcs_from_header(h), WCS(h)
    x, y = _field_points()
    ra, dec = theirs.all_pix2world(x, y, 0)
    xo, yo = kw.sky_to_pixel(ours, ra, dec)
    if reverse is None:
        # astropy inverts iteratively; compare with a tight tolerance and by round trip with the true pixels.
        xa, ya = theirs.all_world2pix(ra, dec, 0, tolerance=1e-12, maxiter=100, quiet=True)
        assert np.max(np.abs(xo - xa)) < 1e-8 and np.max(np.abs(yo - ya)) < 1e-8
        assert np.max(np.abs(xo - x)) < 1e-8 and np.max(np.abs(yo - y)) < 1e-8
    else:
        # Same AP/BP polynomials on both sides: (U, V) relative to CRPIX through astropy's Sip.foc2pix
        # (origin 1: the result is the one-based pixel).
        linear = np.array(theirs.wcs_world2pix(ra, dec, 1)).T
        foc = theirs.sip.foc2pix(linear - theirs.wcs.crpix, 1) - 1.0
        assert np.max(np.abs(xo - foc[:, 0])) < 1e-7 and np.max(np.abs(yo - foc[:, 1])) < 1e-7
        # The fitted reverse polynomial is an approximation of the true inverse: measured 4e-5 to 1.8e-3 pixel.
        assert np.max(np.abs(xo - x)) < 1e-2 and np.max(np.abs(yo - y)) < 1e-2


def test_header_without_reverse_coefficients_ignores_them_only_when_absent():
    """With AP/BP present the reverse polynomial is used, not the Newton inverse: the two differ measurably."""
    h = _header(3, 3, 30.0, 12, 3)  # a deliberately low-order reverse fit
    with_reverse = kw.wcs_from_header(h)
    forward_only = kw.wcs_from_header({k: v for k, v in h.items() if not k.startswith(("AP_", "BP_"))})
    x, y = _field_points(200)
    ra, dec = kw.pixel_to_sky(forward_only, x, y)
    xr, yr = kw.sky_to_pixel(with_reverse, ra, dec)
    xn, yn = kw.sky_to_pixel(forward_only, ra, dec)
    assert np.max(np.abs(xn - x)) < 1e-8
    assert np.max(np.abs(xr - xn)) > 1e-4  # the order-3 reverse fit is not exact


def _distortion_wcs(corner_pixel, order=3, seed=7):
    """Random SIP of the given order, scaled so that the largest |f|, |g| over the field is ``corner_pixel``."""
    r = np.random.default_rng(seed)
    a, b = np.zeros((order + 1, order + 1)), np.zeros((order + 1, order + 1))
    for p in range(order + 1):
        for q in range(order + 1 - p):
            if p + q >= 2:
                a[p, q], b[p, q] = r.normal(), r.normal()
    g = np.linspace(-HALF, HALF, 33)
    gu, gv = np.meshgrid(g, g)
    peak = max(np.abs(kw._poly2d(a, gu, gv)).max(), np.abs(kw._poly2d(b, gu, gv)).max())
    th, s = np.radians(25.0), 0.7 / 3600.0
    cd = s * np.array([[-np.cos(th), np.sin(th)], [np.sin(th), np.cos(th)]])
    return kw.Wcs(
        crpix=np.array([1024.5, 1024.5]),
        crval=np.array([210.0, 40.0]),
        cd=cd,
        sip_a=a * corner_pixel / peak,
        sip_b=b * corner_pixel / peak,
    )


@pytest.mark.parametrize("corner_pixel", [1.0, 3.0, 10.0, 30.0])
def test_numerical_inverse_accuracy(corner_pixel):
    """Newton inverse over a 2048-pixel field (100000 random pixels), distortion up to ``corner_pixel`` pixel.

    Measured (order 3, 0.7 arcsec/pixel): 3 iterations for 1 and 3 pixel, 4 iterations for 10 and 30 pixel;
    maximum error of the inverse itself 1.1e-13 to 1.6e-13 pixel; maximum round-trip error through RA/Dec
    (pixel -> sky -> pixel) 1.9e-10 pixel, set by the double-precision sky coordinates in degrees.
    """
    w = _distortion_wcs(corner_pixel)
    x, y = np.random.default_rng(3).uniform(0, 2047, (2, 100000))
    # Inverse alone, without the sky conversion.
    rel_u, rel_v = x + 1.0 - w.crpix[0], y + 1.0 - w.crpix[1]
    big_u = rel_u + kw._poly2d(w.sip_a, rel_u, rel_v)
    big_v = rel_v + kw._poly2d(w.sip_b, rel_u, rel_v)
    assert max(np.abs(kw._poly2d(w.sip_a, rel_u, rel_v)).max(), np.abs(kw._poly2d(w.sip_b, rel_u, rel_v)).max()) == (
        pytest.approx(corner_pixel, rel=0.05)
    )
    uu, vv, iterations, converged = kw._invert_sip(w.sip_a, w.sip_b, big_u, big_v)
    assert converged.all() and iterations <= 5
    assert np.hypot(uu - rel_u, vv - rel_v).max() < 1e-11
    # Through the sky.
    ra, dec = kw.pixel_to_sky(w, x, y)
    xb, yb = kw.sky_to_pixel(w, ra, dec)
    assert np.hypot(xb - x, yb - y).max() < 1e-8


def test_newton_reports_nonconvergence_as_nan():
    """A target with no real preimage gives nan, not a wrong number.

    f = u^2 / 100 makes U = u + u^2 / 100, which has the minimum U = -25 at u = -50. U = -30 has no real
    solution; U = -20 and U = 40 do.
    """
    a = np.zeros((3, 3))
    a[2, 0] = 1.0 / 100.0
    b = np.zeros((3, 3))
    big_u = np.array([-30.0, -20.0, 40.0])
    uu, vv, _, converged = kw._invert_sip(a, b, big_u, np.zeros(3))
    assert list(converged) == [False, True, True]
    assert np.isnan(uu[0]) and np.isnan(vv[0])
    assert np.all(np.abs(uu[1:] + uu[1:] ** 2 / 100.0 - big_u[1:]) < 1e-9)


def test_plain_tan_is_unchanged_and_zero_sip_matches_it():
    plain = kw.Wcs(crpix=np.array([10.0, 20.0]), crval=np.array([5.0, 6.0]), cd=np.array([[-1e-4, 0.0], [0.0, 1e-4]]))
    assert plain.sip_a is None and plain.sip_b is None and plain.sip_ap is None and plain.sip_bp is None
    zero = kw.Wcs(plain.crpix, plain.crval, plain.cd, sip_a=np.zeros((3, 3)), sip_b=np.zeros((3, 3)))
    x, y = _field_points(50)
    for f, g in zip(kw.pixel_to_sky(plain, x, y), kw.pixel_to_sky(zero, x, y), strict=True):
        assert np.array_equal(f, g)
    ra, dec = kw.pixel_to_sky(plain, x, y)
    for f, g in zip(kw.sky_to_pixel(plain, ra, dec), kw.sky_to_pixel(zero, ra, dec), strict=True):
        assert np.max(np.abs(f - g)) < 1e-10
    # Scalars and nan pass through the SIP path.
    w = _distortion_wcs(3.0)
    ra_s, dec_s = kw.pixel_to_sky(w, 100.0, 200.0)
    x_s, y_s = kw.sky_to_pixel(w, ra_s, dec_s)
    assert float(x_s) == pytest.approx(100.0, abs=1e-8) and float(y_s) == pytest.approx(200.0, abs=1e-8)
    assert np.isnan(kw.sky_to_pixel(w, np.array([np.nan, 210.0]), np.array([40.0, 40.0]))[0][0])
    assert np.isfinite(kw.sky_to_pixel(w, np.array([np.nan, 210.0]), np.array([40.0, 40.0]))[0][1])


def test_header_errors_and_plain_tan_with_stray_sip_keywords():
    h = _header(3, 3, 30.0, 12)
    for bad in (
        {**h, "CTYPE2": "DEC--TAN"},  # only one axis has -SIP
        {k: v for k, v in h.items() if k != "B_ORDER"},
        {k: v for k, v in h.items() if k not in ("A_ORDER", "B_ORDER")},
        {**h, "A_ORDER": 10},
        {**h, "B_ORDER": 1},
        {**h, "AP_ORDER": 2},  # AP without BP
    ):
        with pytest.raises(ValueError):
            kw.wcs_from_header(bad)
    stray = {**h, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN"}
    assert kw.wcs_from_header(stray).sip_a is None  # documented difference from astropy
    with pytest.raises(ValueError):
        kw.Wcs(np.ones(2), np.ones(2), np.eye(2), sip_a=np.zeros((3, 3)))
    with pytest.raises(ValueError):
        bad_coef = np.zeros((3, 3))
        bad_coef[2, 2] = 1.0  # p + q = 4 above order 2
        kw.Wcs(np.ones(2), np.ones(2), np.eye(2), sip_a=bad_coef, sip_b=np.zeros((3, 3)))


# --- fit_sip -------------------------------------------------------------------------------------------------------


def _stars(w, n, seed, noise_pixel):
    """Stars at random pixels; catalog positions through ``w``; measured pixels with Gaussian noise."""
    r = np.random.default_rng(seed)
    x, y = r.uniform(0, 2047, n), r.uniform(0, 2047, n)
    ra, dec = kw.pixel_to_sky(w, x, y)
    return x + r.normal(0, noise_pixel, n), y + r.normal(0, noise_pixel, n), ra, dec


def _coefficient_error_pixel(fit, truth, order):
    """Largest |error| of A_p_q and B_p_q, each expressed as a displacement at 1024 pixels from CRPIX [pixel]."""
    weight = HALF ** np.add.outer(np.arange(order + 1), np.arange(order + 1))
    padded = [np.zeros((order + 1, order + 1)) for _ in range(2)]
    for target, source in zip(padded, (truth.sip_a, truth.sip_b), strict=True):
        target[: source.shape[0], : source.shape[1]] = source
    return max(np.abs((fit.sip_a - padded[0]) * weight).max(), np.abs((fit.sip_b - padded[1]) * weight).max())


def _field_error_pixel(fit, truth):
    """Largest difference between fitted and true pixel -> sky maps over a 41 x 41 grid, in pixels."""
    g = np.linspace(0, 2047, 41)
    gx, gy = (t.ravel() for t in np.meshgrid(g, g))
    ra_f, dec_f = kw.pixel_to_sky(fit, gx, gy)
    ra_t, dec_t = kw.pixel_to_sky(truth, gx, gy)
    return float((kw._angular_separation_arcsec(ra_f, dec_f, ra_t, dec_t) / 0.7).max())


@pytest.mark.parametrize("order", [2, 3, 4])
def test_fit_sip_noiseless_recovers_coefficients_exactly(order):
    truth = _distortion_wcs(3.0, order=order, seed=30 + order)
    x, y, ra, dec = _stars(truth, 200, 5, 0.0)
    fit, rms = kw.fit_sip(kw.Wcs(truth.crpix, truth.crval, truth.cd), x, y, ra, dec, order)
    # Limited by the double-precision sky coordinates (about 2e-10 pixel), amplified by the monomial conditioning.
    assert _coefficient_error_pixel(fit, truth, order) < 1e-5
    assert rms < 1e-6  # arcsec
    assert fit.sip_ap is None and fit.sip_bp is None


@pytest.mark.parametrize(("order", "n_stars", "seed"), [(3, 400, 1), (2, 400, 2), (4, 400, 3)])
def test_fit_sip_recovers_known_distortion(order, n_stars, seed):
    """Stars through a known SIP WCS, 0.05 pixel noise on the measured pixels.

    Measured in a separate exploratory run of 20 noise realisations (not part of this file; 3 pixel distortion at the corners, 2048-pixel field, 0.7 arcsec/pixel):
    order 3 with 300 stars: residual rms median 0.0488 arcsec (noise 0.05 pixel x 0.7 x sqrt 2 = 0.0495),
    coefficient error at 1024 pixels at most 0.038 pixel, field error at most 0.063 pixel; with 1000 stars
    0.020 and 0.032 pixel. Order 4 with 300 stars: 0.104 and 0.124 pixel. The tolerances below are about twice
    the worst of the 20 runs.
    """
    truth = _distortion_wcs(3.0, order=order, seed=40 + order)
    noise = 0.05
    x, y, ra, dec = _stars(truth, n_stars, seed, noise)
    fit, rms = kw.fit_sip(kw.Wcs(truth.crpix, truth.crval, truth.cd), x, y, ra, dec, order)
    expected_rms = noise * 0.7 * np.sqrt(2.0)  # arcsec; the two axes each carry the pixel noise
    assert 0.8 * expected_rms < rms < 1.1 * expected_rms
    tolerance = {2: 0.05, 3: 0.1, 4: 0.25}[order]
    assert _coefficient_error_pixel(fit, truth, order) < tolerance
    assert _field_error_pixel(fit, truth) < 1.5 * tolerance
    # The undistorted WCS is far worse, so the fit did something.
    plain = kw.Wcs(truth.crpix, truth.crval, truth.cd)
    assert _field_error_pixel(plain, truth) > 10 * _field_error_pixel(fit, truth)
    # The fitted WCS inverts numerically.
    xb, yb = kw.sky_to_pixel(fit, ra, dec)
    assert np.hypot(xb - x, yb - y).max() < 6 * noise  # measured pixels carry the noise
    print(
        f"order {order}: rms={rms:.4f} arcsec, coefficient error {_coefficient_error_pixel(fit, truth, order):.3f} px"
    )


def test_fit_sip_ignores_sip_in_the_input_and_rejects_bad_input():
    truth = _distortion_wcs(3.0)
    x, y, ra, dec = _stars(truth, 100, 9, 0.0)
    fit_from_truth, _ = kw.fit_sip(truth, x, y, ra, dec, 3)  # truth carries SIP; only its linear part is used
    fit_from_plain, _ = kw.fit_sip(kw.Wcs(truth.crpix, truth.crval, truth.cd), x, y, ra, dec, 3)
    assert np.array_equal(fit_from_truth.sip_a, fit_from_plain.sip_a)
    linear = kw.Wcs(truth.crpix, truth.crval, truth.cd)
    with pytest.raises(ValueError):
        kw.fit_sip(linear, x, y, ra, dec, 1)
    with pytest.raises(ValueError):
        kw.fit_sip(linear, x[:5], y[:5], ra[:5], dec[:5], 3)  # fewer stars than coefficients
    with pytest.raises(ValueError):
        kw.fit_sip(linear, x[:50], y[:49], ra[:50], dec[:50], 3)
    line = np.linspace(0, 2000, 60)
    ra_l, dec_l = kw.pixel_to_sky(linear, line, line)
    with pytest.raises(ValueError):
        kw.fit_sip(linear, line, line, ra_l, dec_l, 3)  # all stars on one line: coefficients not constrained
