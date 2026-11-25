"""Test of PSF models and PSF-fitting photometry with synthetic data whose truth is known.

The Monte Carlo tests use fixed seeds; a tolerance is written as a multiple of the standard error of the
quantity measured, so a failure means a real change of behaviour, not bad luck.
"""

import numpy as np
import pytest
from scipy import integrate

from kozmos import photometry
from kozmos.observing import detector
from kozmos.photometry import psf

FWHM = 3.0
SIGMA = FWHM * psf.FWHM_TO_SIGMA
SKY = 100.0
NOISE = 5.0
SHAPE = (41, 41)
# Flux uncertainty of a single known-position star in white noise: noise * sqrt(4 pi sigma^2) = flux / (S/N).
UNIT = NOISE * np.sqrt(4 * np.pi * SIGMA**2)


def _exact_flux_error():
    """Flux uncertainty of a known-position star in white noise: NOISE / sqrt(sum of squared pixel responses).

    The pixel integration widens the profile (variance + 1/12 pixel^2), so this is 2.5% above UNIT.
    """
    yy, xx = np.mgrid[:41, :41].astype(float)
    return NOISE / np.sqrt(np.sum(psf.gaussian_prf(xx, yy, 20.0, 20.0, FWHM) ** 2))


def _enclosed(radius):
    return 1.0 - np.exp(-(radius**2) / (2 * SIGMA**2))


def _single(rng, snr, noise=NOISE):
    x = 20 + rng.uniform(-0.5, 0.5)
    y = 20 + rng.uniform(-0.5, 0.5)
    flux = snr * UNIT
    image = SKY + detector.star_field(SHAPE, [x], [y], [flux], FWHM) + rng.normal(0, noise, SHAPE)
    return image, x, y, flux


# ----------------------------------------------------------------------------------------------- Gaussian PRF


def test_gaussian_prf_conserves_flux_and_matches_star_field():
    yy, xx = np.mgrid[:41, :41].astype(float)
    value = psf.gaussian_prf(xx, yy, 20.3, 19.6, FWHM, flux=1234.0)
    assert value.sum() == pytest.approx(1234.0, rel=1e-12)  # the Gaussian is 1e-30 of its peak at the edge
    scene = detector.star_field((41, 41), [20.3], [19.6], [1234.0], FWHM)
    np.testing.assert_allclose(value, scene, rtol=0, atol=1e-12)  # same erf formula; equal to rounding


@pytest.mark.parametrize("offset", [(0.0, 0.0), (0.37, -0.21)])
def test_gaussian_prf_equals_numerical_pixel_integral(offset):
    x0, y0 = 10.0 + offset[0], 10.0 + offset[1]
    for px, py in [(10, 10), (11, 9), (8, 12), (13, 10)]:
        expected, _ = integrate.dblquad(
            lambda v, u: np.exp(-((u - x0) ** 2 + (v - y0) ** 2) / (2 * SIGMA**2)) / (2 * np.pi * SIGMA**2),
            px - 0.5,
            px + 0.5,
            py - 0.5,
            py + 0.5,
            epsabs=1e-13,
            epsrel=1e-12,
        )
        got = psf.gaussian_prf(px, py, x0, y0, FWHM)
        assert got == pytest.approx(expected, rel=1e-9)  # quadrature tolerance 1e-12 relative; erf is exact


@pytest.mark.parametrize("model,beta", [("gaussian", 0.0), ("moffat", 2.5)])
def test_analytic_derivatives_match_finite_differences(model, beta):
    yy, xx = np.mgrid[:21, :21].astype(float)
    x0, y0, h = 10.2, 9.7, 1e-5
    value, dx, dy = psf._profile(model, xx, yy, x0, y0, FWHM, beta, 9)
    plus = psf._profile(model, xx, yy, x0 + h, y0, FWHM, beta, 9)[0]
    minus = psf._profile(model, xx, yy, x0 - h, y0, FWHM, beta, 9)[0]
    np.testing.assert_allclose(dx, (plus - minus) / (2 * h), rtol=0, atol=1e-9)  # central difference error h^2
    plus = psf._profile(model, xx, yy, x0, y0 + h, FWHM, beta, 9)[0]
    minus = psf._profile(model, xx, yy, x0, y0 - h, FWHM, beta, 9)[0]
    np.testing.assert_allclose(dy, (plus - minus) / (2 * h), rtol=0, atol=1e-9)
    assert value.max() > 0.05


# ----------------------------------------------------------------------------------------------- Moffat PRF


def test_moffat_total_flux_matches_numerical_integral():
    beta = 2.5
    n = 30
    yy, xx = np.mgrid[-n : n + 1, -n : n + 1].astype(float)
    pixel_sum = psf.moffat_prf(xx, yy, 0.0, 0.0, FWHM, beta).sum()
    radius = psf._moffat_alpha(FWHM, beta)
    expected, _ = integrate.dblquad(
        lambda v, u: (beta - 1) / (np.pi * radius**2) * (1 + (u * u + v * v) / radius**2) ** (-beta),
        -n - 0.5,
        n + 0.5,
        -n - 0.5,
        n + 0.5,
        epsabs=1e-12,
        epsrel=1e-10,
    )
    assert pixel_sum == pytest.approx(expected, rel=1e-8)  # the sum is the quadrature rule; error 1e-9 at 5 nodes
    assert 0.999 < expected < 1.0  # measured 0.9995; the wings carry the missing flux, (1 + r^2/R^2)^(1-beta) outside r


def test_moffat_fwhm_definition():
    # For a profile much wider than a pixel the pixel integral equals the point value to O(1/FWHM^2).
    beta, wide = 3.0, 60.0
    center = psf.moffat_prf(0.0, 0.0, 0.0, 0.0, wide, beta)
    half = psf.moffat_prf(wide / 2.0, 0.0, 0.0, 0.0, wide, beta)
    assert half / center == pytest.approx(0.5, rel=1e-3)


@pytest.mark.parametrize("beta", [1.5, 2.5, 5.0])
def test_moffat_pixel_integral_against_adaptive_quadrature(beta):
    x0, y0 = 5.3, 5.2
    for px, py in [(5, 5), (6, 4), (3, 7)]:
        expected, _ = integrate.dblquad(
            lambda v, u: (
                (beta - 1)
                / (np.pi * psf._moffat_alpha(FWHM, beta) ** 2)
                * (1 + ((u - x0) ** 2 + (v - y0) ** 2) / psf._moffat_alpha(FWHM, beta) ** 2) ** (-beta)
            ),
            px - 0.5,
            px + 0.5,
            py - 0.5,
            py + 0.5,
            epsabs=1e-14,
            epsrel=1e-12,
        )
        assert psf.moffat_prf(px, py, x0, y0, FWHM, beta) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("beta", [1.5, 2.5, 5.0])
def test_moffat_quadrature_accuracy(beta):
    # Error of the n x n Gauss-Legendre rule relative to the 64 x 64 rule, as a fraction of the central pixel value.
    # Measured (offset 0.3, 0.2 pixel; 21 x 21 grid): n=5 gives <= 2.1e-6 for FWHM >= 1.5 pixel and <= 6e-5 for FWHM = 1.
    yy, xx = np.mgrid[-10:11, -10:11].astype(float)
    limits = {1.5: 3e-6, 3.0: 1e-8}
    for fwhm, limit in limits.items():
        ref = psf.moffat_prf(xx, yy, 0.3, 0.2, fwhm, beta, n_nodes=64)
        fine = psf.moffat_prf(xx, yy, 0.3, 0.2, fwhm, beta, n_nodes=96)
        assert np.abs(ref - fine).max() / ref.max() < 1e-12  # the reference itself has converged
        coarse = psf.moffat_prf(xx, yy, 0.3, 0.2, fwhm, beta, n_nodes=5)
        assert np.abs(coarse - ref).max() / ref.max() < limit
    narrow_ref = psf.moffat_prf(xx, yy, 0.3, 0.2, 1.0, beta, n_nodes=64)
    narrow = psf.moffat_prf(xx, yy, 0.3, 0.2, 1.0, beta, n_nodes=5)
    assert np.abs(narrow - narrow_ref).max() / narrow_ref.max() < 1e-4
    # Accuracy improves rapidly with the number of nodes.
    errors = [
        np.abs(
            psf.moffat_prf(xx, yy, 0.3, 0.2, 1.5, beta, n_nodes=n)
            - psf.moffat_prf(xx, yy, 0.3, 0.2, 1.5, beta, n_nodes=64)
        ).max()
        for n in (2, 3, 5, 7)
    ]
    assert errors == sorted(errors, reverse=True)


def test_moffat_large_beta_is_gaussian():
    yy, xx = np.mgrid[-8:9, -8:9].astype(float)
    moffat = psf.moffat_prf(xx, yy, 0.2, -0.1, FWHM, beta=1e6, n_nodes=7)
    gauss = psf.gaussian_prf(xx, yy, 0.2, -0.1, FWHM)
    assert np.abs(moffat - gauss).max() / gauss.max() < 1e-5  # (1 + rho)^(-beta) -> exp(-beta rho), error O(1/beta)


def test_moffat_rejects_invalid_beta():
    with pytest.raises(ValueError):
        psf.moffat_prf(0.0, 0.0, 0.0, 0.0, FWHM, beta=1.0)


# ----------------------------------------------------------------------------------------------- isolated stars


def _monte_carlo_isolated(snr, fit_positions, n_trials=300, seed=7, noise_given=True):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_trials):
        image, x, y, flux = _single(rng, snr)
        start = (20.0, 20.0) if fit_positions else (x, y)
        r = psf.psf_photometry(
            image,
            [start[0]],
            [start[1]],
            FWHM,
            background=SKY,
            noise=NOISE if noise_given else None,
            fit_positions=fit_positions,
        )
        ap, _ = photometry.aperture_photometry(image, [x], [y], 1.2 * FWHM, annulus=(2.5 * FWHM, 4 * FWHM))
        ap_known, _ = photometry.aperture_photometry(image - SKY, [x], [y], 1.2 * FWHM)
        corr = _enclosed(1.2 * FWHM)
        out.append(
            (
                r.flux[0] - flux,
                r.flux_err[0],
                r.x[0] - x,
                r.y[0] - y,
                r.x_err[0],
                ap[0] / corr - flux,
                ap_known[0] / corr - flux,
            )
        )
    return np.array(out)


@pytest.mark.parametrize("snr", [10, 50, 200])
def test_isolated_flux_unbiased_and_scatter_equals_formal_error(snr):
    o = _monte_carlo_isolated(snr, fit_positions=True)
    n = len(o)
    std = o[:, 0].std()
    # Bias within 4 standard errors of the mean (measured: 1.6, 0.7, 1.3 flux units against a standard error of 1.2).
    assert abs(o[:, 0].mean()) < 4 * std / np.sqrt(n)
    # The scatter of the fitted flux equals the formal uncertainty; the standard error of a standard deviation is
    # 1/sqrt(2 n) = 4%, the window is +-3 standard errors. Measured ratio 0.97 to 1.0.
    assert std / o[:, 1].mean() == pytest.approx(1.0, abs=0.12)
    assert o[:, 1].mean() == pytest.approx(_exact_flux_error(), rel=0.005)  # NOISE / sqrt(sum P^2), derived below


@pytest.mark.parametrize("snr", [3, 5])
def test_low_snr_fixed_positions_unbiased_and_fitted_positions_bias_small(snr):
    fixed = _monte_carlo_isolated(snr, fit_positions=False)
    assert abs(fixed[:, 0].mean()) < 4 * fixed[:, 0].std() / np.sqrt(len(fixed))  # linear problem: exactly unbiased
    fitted = _monte_carlo_isolated(snr, fit_positions=True)
    # Fitting the position at low S/N follows noise peaks and biases the flux high: measured +14% at S/N 3 and +5.6%
    # at S/N 5 of the true flux. Documented in the Validity section; here it is bounded, and it must be positive or zero.
    bias = fitted[:, 0].mean() / (snr * UNIT)
    assert -0.03 < bias < 0.20


def test_psf_fit_is_more_precise_than_aperture_photometry():
    # Sky-noise-limited stars of FWHM 3 pixels. Aperture: radius 1.2 FWHM, divided by the enclosed fraction of the
    # Gaussian, sky from an annulus (2.5 to 4 FWHM) or the known sky. Measured ratios of aperture scatter to PSF
    # scatter: 1.46 to 1.60 (annulus) and 1.26 to 1.45 (known sky) from S/N 3 to 200.
    for snr in (5, 20):
        o = _monte_carlo_isolated(snr, fit_positions=True, seed=21)
        psf_scatter = o[:, 0].std()
        assert o[:, 5].std() / psf_scatter > 1.25
        assert o[:, 6].std() / psf_scatter > 1.10
        # Known-position, known-sky linear fit is the minimum-variance estimator: nothing beats it by more than noise.
        assert o[:, 6].std() / o[:, 1].mean() > 1.05


@pytest.mark.parametrize("snr", [20, 200])
def test_position_error_scales_with_snr_and_matches_formal_error(snr):
    o = _monte_carlo_isolated(snr, fit_positions=True, seed=33)
    position_std = np.std(np.r_[o[:, 2], o[:, 3]])
    # Formal x error from the covariance; measured ratio 0.99 to 1.05.
    assert position_std / o[:, 4].mean() == pytest.approx(1.0, abs=0.12)
    # Closed form for a Gaussian PRF: sigma_x = sqrt(2) sigma_psf / (S/N) for S/N defined by flux / (noise sqrt(4 pi s^2));
    # the pixel integration adds a few percent; measured 0.0946 at S/N 20 (formula 0.0899 -> ratio 1.05).
    assert position_std == pytest.approx(np.sqrt(2) * SIGMA / snr, rel=0.12)


def test_flux_error_without_noise_map_is_scaled_by_the_residual_scatter():
    o = _monte_carlo_isolated(20, fit_positions=True, seed=44, noise_given=False)
    # With no noise map the weights are equal and the error comes from the residual scatter (this project's choice).
    assert o[:, 0].std() / o[:, 1].mean() == pytest.approx(1.0, abs=0.15)
    assert o[:, 1].mean() == pytest.approx(_exact_flux_error(), rel=0.05)


def test_chi2_is_unity_for_the_true_model_and_large_for_the_wrong_one():
    rng = np.random.default_rng(55)
    chi_true, chi_wrong = [], []
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]].astype(float)
    for _ in range(100):
        image, *_ = _single(rng, 100)
        r = psf.psf_photometry(image, [20.0], [20.0], FWHM, background=SKY, noise=NOISE)
        chi_true.append(r.chi2_reduced[0])
        moffat = SKY + psf.moffat_prf(xx, yy, 20.3, 19.8, 3.0, 3.0, 100 * UNIT, n_nodes=7) + rng.normal(0, NOISE, SHAPE)
        r = psf.psf_photometry(moffat, [20.0], [20.0], FWHM, background=SKY, noise=NOISE)
        chi_wrong.append(r.chi2_reduced[0])
    # Reduced chi-square of about 30 pixels has a relative error 1/sqrt(n)=0.18 per star; the mean of 100 stars 0.02.
    assert np.mean(chi_true) == pytest.approx(1.0, abs=0.08)  # measured 0.98 to 1.02
    assert np.mean(chi_wrong) > 2.0  # measured 2.65 for a Moffat star (beta 3) fitted with a Gaussian


# ----------------------------------------------------------------------------------------------- blends


def _blend_monte_carlo(separation_fwhm, flux_ratio, fit_positions, n_trials=150, seed=3, snr_bright=50):
    rng = np.random.default_rng(seed)
    f1 = snr_bright * UNIT
    f2 = f1 / flux_ratio
    rows = []
    for _ in range(n_trials):
        xc, yc = 20 + rng.uniform(-0.5, 0.5), 20 + rng.uniform(-0.5, 0.5)
        angle = rng.uniform(0, 2 * np.pi)
        d = separation_fwhm * FWHM / 2
        xs = np.array([xc + d * np.cos(angle), xc - d * np.cos(angle)])
        ys = np.array([yc + d * np.sin(angle), yc - d * np.sin(angle)])
        image = SKY + detector.star_field(SHAPE, xs, ys, [f1, f2], FWHM) + rng.normal(0, NOISE, SHAPE)
        start_x = xs + rng.normal(0, 0.2, 2) if fit_positions else xs
        start_y = ys + rng.normal(0, 0.2, 2) if fit_positions else ys
        r = psf.psf_photometry(image, start_x, start_y, FWHM, background=SKY, noise=NOISE, fit_positions=fit_positions)
        rows.append((r.flux[0] - f1, r.flux[1] - f2, r.flux_err[0], r.flux_err[1], r.group_size[0], r.converged.all()))
    return f1, f2, np.array(rows, dtype=float)


@pytest.mark.parametrize("fit_positions", [False, True])
@pytest.mark.parametrize("flux_ratio", [1, 4])
@pytest.mark.parametrize("separation", [1.0, 1.5, 2.0])
def test_blended_pair_recovers_both_fluxes(separation, flux_ratio, fit_positions):
    f1, f2, o = _blend_monte_carlo(separation, flux_ratio, fit_positions)
    n = len(o)
    assert np.all(o[:, 4] == 2)  # fitted as one group
    assert o[:, 5].all()
    for k, flux in ((0, f1), (1, f2)):
        std = o[:, k].std()
        assert abs(o[:, k].mean()) < 4 * std / np.sqrt(n)  # unbiased within 4 standard errors (measured <= 2.0)
        # The scatter agrees with the formal uncertainty, including the correlation between the two fluxes;
        # window +-3 standard errors of a standard deviation from 150 trials (1/sqrt(2n)=5.8%) plus the
        # non-linearity of the position fit: measured 0.94 to 1.04 (fixed positions), 0.93 to 1.00 (fitted).
        assert std / o[:, 2 + k].mean() == pytest.approx(1.0, abs=0.20)
        assert std / flux < 0.12  # the faint star at flux ratio 4 and S/N 12.5 has 8 to 10%, the bright one 2 to 3%


def test_blended_pair_error_grows_when_the_pair_gets_closer():
    # Fitted positions, equal fluxes: measured flux scatter 2.0% (2 FWHM), 2.1% (1.5), 2.6% (1) against 2.0% isolated.
    spread = [_blend_monte_carlo(s, 1, True, n_trials=100, seed=5)[2][:, 0].std() for s in (2.0, 1.5, 1.0)]
    assert spread[0] < spread[2]
    assert spread[2] / spread[0] == pytest.approx(1.3, abs=0.25)


@pytest.mark.parametrize(
    "separation,flux_ratio,expected_bias",
    [(1.0, 1, 0.267), (1.5, 1, 0.051), (2.0, 1, 0.005), (1.5, 4, 0.204)],
)
def test_ungrouped_fits_are_biased_by_the_neighbour(separation, flux_ratio, expected_bias):
    # Noiseless: with the group radius set below the separation each star is fitted alone, ignoring its neighbour.
    # Measured bias of the star that is the fainter one (flux ratio 4) or either (ratio 1) listed in the first row of
    # the Validity table; grouped fits are exact to 2e-15.
    rng = np.random.default_rng(9)
    biases = []
    for _ in range(40):
        xc, yc, angle = 20 + rng.uniform(-0.5, 0.5), 20 + rng.uniform(-0.5, 0.5), rng.uniform(0, 2 * np.pi)
        d = separation * FWHM / 2
        xs = [xc + d * np.cos(angle), xc - d * np.cos(angle)]
        ys = [yc + d * np.sin(angle), yc - d * np.sin(angle)]
        f1, f2 = 50 * UNIT, 50 * UNIT / flux_ratio
        image = SKY + detector.star_field(SHAPE, xs, ys, [f1, f2], FWHM)
        alone = psf.psf_photometry(
            image, xs, ys, FWHM, background=SKY, noise=1.0, fit_positions=False, group_radius_pix=0.1
        )
        grouped = psf.psf_photometry(image, xs, ys, FWHM, background=SKY, noise=1.0, fit_positions=False)
        np.testing.assert_allclose(grouped.flux, [f1, f2], rtol=1e-10)  # exact: linear model, no noise
        assert np.all(alone.group_size == 1)
        biases.append(alone.flux[1] / f2 - 1 if flux_ratio == 4 else alone.flux[0] / f1 - 1)
    assert np.mean(biases) == pytest.approx(expected_bias, rel=0.1, abs=0.003)


# ----------------------------------------------------------------------------------------------- wrong width


@pytest.mark.parametrize("ratio,expected", [(0.9, -0.0993), (1.1, 0.0908)])
def test_wrong_fwhm_biases_flux_in_proportion(ratio, expected):
    # Noiseless: the fitted flux is the true flux times about the ratio of assumed to true width (the model peak is
    # matched to the data). Measured -9.93% and +9.08% for a 10% error of the width; with noise at S/N 50 -9.8% and +9.4%.
    x, y = 20.3, 19.8
    image = SKY + detector.star_field(SHAPE, [x], [y], [1000.0], FWHM)
    r = psf.psf_photometry(image, [x], [y], FWHM * ratio, background=SKY, noise=1.0, fit_positions=False)
    assert r.flux[0] / 1000.0 - 1 == pytest.approx(expected, abs=0.002)
    # The fit radius scales with the assumed width, so the radius is not the cause: fix it and the result is the same.
    r = psf.psf_photometry(
        image, [x], [y], FWHM * ratio, background=SKY, noise=1.0, fit_positions=False, fit_radius_pix=6.0
    )
    assert r.flux[0] / 1000.0 - 1 == pytest.approx(expected, abs=0.002)


# ----------------------------------------------------------------------------------------------- Moffat model


def test_moffat_model_recovers_moffat_stars_and_gaussian_model_does_not():
    beta, flux = 3.0, 3000.0
    yy, xx = np.mgrid[:61, :61].astype(float)
    image = SKY + psf.moffat_prf(xx, yy, 30.3, 29.8, FWHM, beta, flux, n_nodes=9)
    r = psf.psf_photometry(image, [30.0], [30.0], FWHM, model="moffat", beta=beta, background=SKY, noise=1.0)
    assert r.flux[0] == pytest.approx(flux, rel=1e-6)  # noiseless and the right model: exact to the quadrature (1e-9)
    assert r.x[0] == pytest.approx(30.3, abs=1e-4)
    assert r.y[0] == pytest.approx(29.8, abs=1e-4)
    assert r.converged[0]
    wrong = psf.psf_photometry(image, [30.0], [30.0], FWHM, model="gaussian", background=SKY, noise=1.0)
    # The Gaussian misses the wings: measured 0.771 of the true flux for a fit radius of 1.5 and of 3 FWHM.
    assert 0.74 < wrong.flux[0] / flux < 0.80


# ----------------------------------------------------------------------------------------------- width estimate


def _width_field(rng, fwhm, snr, n=12, size=120):
    xs = rng.uniform(10, size - 10, n)
    ys = rng.uniform(10, size - 10, n)
    flux = snr * NOISE * np.sqrt(4 * np.pi * (fwhm * psf.FWHM_TO_SIGMA) ** 2) * 10 ** rng.uniform(0, 0.7, n)
    image = SKY + detector.star_field((size, size), xs, ys, flux, fwhm) + rng.normal(0, NOISE, (size, size))
    return image, xs, ys


@pytest.mark.parametrize("fwhm", [2.0, 3.0, 4.5])
@pytest.mark.parametrize("snr", [20, 100])
def test_fit_psf_width_recovers_input(fwhm, snr):
    rng = np.random.default_rng(int(fwhm * 10) + snr)
    errors, estimates = [], []
    for _ in range(10):
        image, xs, ys = _width_field(rng, fwhm, snr)
        estimate, error, used = psf.fit_psf_width(image, xs, ys, noise=NOISE, initial_fwhm_pix=3.0)
        assert used >= 1
        estimates.append(estimate)
        errors.append(error)
    mean_estimate = np.mean(estimates)
    # Measured mean estimate 1.994, 2.002 (FWHM 2); 2.980, 2.998 (3); 4.494, 4.501 (4.5) at S/N 20 and 100.
    # The standard error of the mean of 10 fields is std/sqrt(10); allow 4 of them plus 0.3% for the
    # finite-sample non-linearity of the fit.
    assert abs(mean_estimate - fwhm) < 4 * np.std(estimates) / np.sqrt(10) + 0.003 * fwhm
    # The stated error is the scatter of one field's median: compare with the realized scatter (within a factor of 2).
    assert 0.5 < np.mean(errors) / np.std(estimates) < 2.5


def test_fit_psf_width_ignores_blended_stars_and_raises_without_isolated_ones():
    rng = np.random.default_rng(66)
    xs = np.array([20.0, 60.0, 100.0, 40.0, 42.0, 80.0, 83.0])
    ys = np.array([20.0, 60.0, 20.0, 90.0, 90.0, 90.0, 90.0])
    flux = np.array([6000.0, 5000.0, 4000.0, 6000.0, 6000.0, 6000.0, 6000.0])
    image = SKY + detector.star_field((120, 120), xs, ys, flux, 4.0) + rng.normal(0, NOISE, (120, 120))
    # The two pairs (2 and 3 pixels apart) would give a width that is too large if they were used.
    estimate, error, used = psf.fit_psf_width(image, xs, ys, noise=NOISE, bright_fraction=1.0, min_stars=3)
    assert used == 3
    assert estimate == pytest.approx(4.0, abs=4 * error + 0.01)
    with pytest.raises(ValueError):
        psf.fit_psf_width(image, xs[3:5], ys[3:5], noise=NOISE)


# ----------------------------------------------------------------------------------------------- bookkeeping


def test_grouping_chains_neighbours_and_splits_large_groups():
    flux = 3000.0
    # Chain of stars 8 pixels apart plus two distant stars; default fit radius 4.5 and critical separation 10.5 pixels.
    xs = np.r_[10.0 + 8.0 * np.arange(6), 10.0, 90.0]
    ys = np.r_[np.full(6, 20.0), 80.0, 80.0]
    image = SKY + detector.star_field((100, 100), xs, ys, np.full(xs.size, flux), FWHM)
    r = psf.psf_photometry(image, xs, ys, FWHM, background=SKY, noise=1.0, fit_positions=False)
    assert len(set(r.group_id[:6])) == 1  # the chain links through its neighbours (friends of friends)
    assert np.all(r.group_size[:6] == 6)
    assert r.group_size[6] == 1 and r.group_size[7] == 1
    np.testing.assert_allclose(r.flux, flux, rtol=1e-10)
    # Group larger than max_group_size: regrouped with half the separation (5.25 pixels), the chain falls apart.
    small = psf.psf_photometry(image, xs, ys, FWHM, background=SKY, noise=1.0, fit_positions=False, max_group_size=3)
    assert np.all(small.group_size == 1)
    # Stars closer together than the halved separation cannot be split: the group stays whole.
    close = psf.psf_photometry(
        SKY
        + detector.star_field(
            (100, 100), xs[:3] * 0.0 + [30.0, 33.0, 36.0], xs[:3] * 0.0 + 50.0, np.full(3, flux), FWHM
        ),
        [30.0, 33.0, 36.0],
        [50.0, 50.0, 50.0],
        FWHM,
        background=SKY,
        noise=1.0,
        fit_positions=False,
        max_group_size=2,
    )
    assert np.all(close.group_size == 3)
    np.testing.assert_allclose(close.flux, flux, rtol=1e-9)


def test_edge_star_outside_star_and_masked_pixels():
    flux = 3000.0
    near_edge = SKY + detector.star_field(SHAPE, [1.3], [20.2], [flux], FWHM)
    r = psf.psf_photometry(near_edge, [1.0], [20.0], FWHM, background=SKY, noise=1.0)
    assert r.flux[0] == pytest.approx(flux, rel=1e-6)  # only the pixels inside the image are fitted; truth is exact
    assert r.x[0] == pytest.approx(1.3, abs=1e-4)
    assert 0 < r.n_pixels[0] < np.pi * (1.5 * FWHM) ** 2
    outside = psf.psf_photometry(near_edge, [-30.0], [20.0], FWHM, background=SKY, noise=1.0)
    assert np.isnan(outside.flux[0]) and not outside.converged[0]
    masked = SKY + detector.star_field(SHAPE, [20.3], [19.8], [flux], FWHM)
    masked[19:22, 19:22] = np.nan  # the core is missing; the wings still define the flux exactly
    r = psf.psf_photometry(masked, [20.0], [20.0], FWHM, background=SKY, noise=1.0)
    assert r.flux[0] == pytest.approx(flux, rel=1e-6)
    assert r.n_pixels[0] < np.pi * (1.5 * FWHM) ** 2 - 8  # the 3 x 3 block is excluded


def test_background_map_equals_constant_and_wrong_background_shifts_flux_as_derived():
    flux = 3000.0
    image = SKY + detector.star_field(SHAPE, [20.0], [20.0], [flux], FWHM)
    constant = psf.psf_photometry(image, [20.0], [20.0], FWHM, background=SKY, noise=1.0, fit_positions=False)
    as_map = psf.psf_photometry(
        image, [20.0], [20.0], FWHM, background=np.full(SHAPE, SKY), noise=np.full(SHAPE, 1.0), fit_positions=False
    )
    assert as_map.flux[0] == constant.flux[0]
    # Background error db: flux error = db * sum(P) / sum(P^2) over the fit pixels (linear least squares with one star).
    yy, xx = np.mgrid[:41, :41]
    mask = (xx - 20.0) ** 2 + (yy - 20.0) ** 2 <= (1.5 * FWHM) ** 2
    p = psf.gaussian_prf(xx, yy, 20.0, 20.0, FWHM)[mask]
    biased = psf.psf_photometry(image, [20.0], [20.0], FWHM, background=SKY - 1.0, noise=1.0, fit_positions=False)
    assert biased.flux[0] - flux == pytest.approx(p.sum() / (p**2).sum(), rel=1e-9)  # exact algebra, rounding only


def test_input_validation():
    image = np.zeros((20, 20))
    with pytest.raises(ValueError):
        psf.psf_photometry(image, [5.0], [5.0], 0.0)
    with pytest.raises(ValueError):
        psf.psf_photometry(image, [5.0, 6.0], [5.0], 3.0)
    with pytest.raises(ValueError):
        psf.psf_photometry(image, [5.0], [5.0], 3.0, model="airy")
    with pytest.raises(ValueError):
        psf.psf_photometry(image, [5.0], [5.0], 3.0, noise=np.ones((5, 5)))
    with pytest.raises(ValueError):
        psf.psf_photometry(np.zeros(5), [1.0], [1.0], 3.0)
    empty = psf.psf_photometry(image, [], [], 3.0)
    assert empty.flux.size == 0


def test_multi_star_field_with_detect_sources_matches_truth():
    rng = np.random.default_rng(88)
    n = 25
    xs, ys = rng.uniform(10, 150, n), rng.uniform(10, 150, n)
    flux = 10 ** rng.uniform(3.3, 4.3, n)
    shape = (160, 160)
    image = SKY + detector.star_field(shape, xs, ys, flux, FWHM) + rng.normal(0, NOISE, shape)
    bkg, rms = photometry.background_map(image, cell=40)
    found = photometry.detect_sources(image, bkg, rms, threshold_sigma=5.0, min_pixels=5)
    r = psf.psf_photometry(image, found.x, found.y, FWHM, background=bkg, noise=rms)
    matched = 0
    for i in range(len(found)):
        d = np.hypot(xs - found.x[i], ys - found.y[i])
        j = int(np.argmin(d))
        if d[j] < 1.0 and r.group_size[i] == 1:
            matched += 1
            # The background map estimated from stars-containing cells carries a small error; allow 5 formal
            # sigma plus 0.5% of the flux for it.
            assert abs(r.flux[i] - flux[j]) < 5 * r.flux_err[i] + 0.005 * flux[j]
    assert matched >= 15
