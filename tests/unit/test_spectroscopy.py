"""kozmos.spectroscopy: source worked examples, closed forms and synthetic data with known truth.

Sources: Morton 1991 (Eq. 3 and its air/vacuum table), Horne 1986 (Table I, Eqs. 8-14),
Tonry & Davis 1979 (Eqs. 1-5, 20-24, 34-35, Sect. IV), Vollmann & Eversberg 2006, SciPy documentation.
Monte Carlo tests use fixed seeds, so they are deterministic; the tolerances are set from the measured values
(quoted in the comments) with room for the statistical error of the number of trials.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from kozmos.core.constants import SPEED_OF_LIGHT_KM_S as C_KM_S
from kozmos.spectroscopy import extraction, lines, velocity, wavelengths

# --------------------------------------------------------------------------------------------- wavelengths

# Air and vacuum wavelengths printed in the Na I multiplets of Table 2 of [Morton1991] (ADS scan, p. 134).
MORTON_PAIRS = [
    (5891.9409, 5893.5736),
    (5895.9243, 5897.5581),
    (5889.9512, 5891.5833),
    (3302.572, 3303.522),
    (3302.978, 3303.929),
    (3302.368, 3303.319),
    (2852.878, 2853.716),
    (2853.012, 2853.850),
    (2852.811, 2853.649),
    (2680.372, 2681.168),
    (2680.434, 2681.230),
    (2680.341, 2681.137),
]


def test_morton_table_pairs():
    # The table prints 3 or 4 decimals, so the largest allowed difference is the rounding of two printed numbers
    # (up to 1e-3); the measured largest difference is 0.91e-3 (3302.572 / 3303.522, printed with 3 decimals).
    air = np.array([p[0] for p in MORTON_PAIRS])
    vac = np.array([p[1] for p in MORTON_PAIRS])
    assert np.max(np.abs(wavelengths.vacuum_to_air(vac) - air)) < 1.0e-3
    assert np.max(np.abs(wavelengths.air_to_vacuum(air) - vac)) < 1.0e-3
    # The four-decimal Na D pairs agree better: rounding 1e-4, measured about 2.4e-4.
    assert np.max(np.abs(wavelengths.air_to_vacuum(air[:3]) - vac[:3])) < 3.0e-4


def test_morton_text_pair_at_2000_angstrom():
    # Sect. 2 of [Morton1991]: lambda_air = 2000 A corresponds to lambda_vac = 2000.648 A.
    assert abs(wavelengths.air_to_vacuum(2000.0) - 2000.648) < 5.0e-4
    assert abs(wavelengths.vacuum_to_air(2000.648) - 2000.0) < 1.0e-4


def test_refractive_index_matches_table_ratio():
    # n = lambda_vac / lambda_air for the Na D2 line of the table (rounding 1e-4 / 5890 = 2e-8).
    n_table = 5891.5833 / 5889.9512
    assert abs(wavelengths.air_refractive_index(5891.5833) - n_table) < 5.0e-8


def test_air_vacuum_round_trip_is_exact():
    lam = np.linspace(2001.0, 30000.0, 5001)  # vacuum 2001 A is air 2000.35 A, inside the accepted range
    back = wavelengths.air_to_vacuum(wavelengths.vacuum_to_air(lam))
    assert np.max(np.abs(back - lam)) < 1.0e-9  # measured about 1e-12 (rounding)
    back2 = wavelengths.vacuum_to_air(wavelengths.air_to_vacuum(lam))
    assert np.max(np.abs(back2 - lam)) < 1.0e-9


def test_alternative_sigma_convention_difference():
    # sigma from the air wavelength instead of the vacuum wavelength: measured differences in the air wavelength
    # 9.8e-5 A at 2000 A, 1.8e-5 A at 5000 A, 4.2e-6 A at 20000 A.
    def n1(sigma):
        return 6.4328e-5 + 2.94981e-2 / (146 - sigma**2) + 2.5540e-4 / (41 - sigma**2)

    for lam_vac, expected in ((2000.0, 9.8e-5), (5000.0, 1.8e-5), (20000.0, 4.2e-6)):
        air_b = lam_vac
        for _ in range(30):
            air_b = lam_vac / (1 + n1(1e4 / air_b))
        diff = abs(float(wavelengths.vacuum_to_air(lam_vac)) - air_b)
        assert diff == pytest.approx(expected, rel=0.05, abs=0)


def test_air_vacuum_range_guard():
    with pytest.raises(ValueError):
        wavelengths.vacuum_to_air(1500.0)
    with pytest.raises(ValueError):
        wavelengths.air_to_vacuum([3000.0, 1999.0])
    with pytest.raises(ValueError):
        wavelengths.air_refractive_index(np.nan)


def test_doppler_relativistic_and_first_order():
    rest = 6562.8
    # 1 + z = sqrt(3) at beta = 0.5 ([Hogg1999] Eq. 9).
    assert wavelengths.doppler_shifted_wavelength(1.0, 0.5 * C_KM_S) == pytest.approx(np.sqrt(3.0), rel=1e-14, abs=0)
    v = np.array([-300.0, -30.0, 0.0, 30.0, 300.0])
    rel = wavelengths.doppler_shifted_wavelength(rest, v, True)
    nonrel = wavelengths.doppler_shifted_wavelength(rest, v, False)
    beta = v / C_KM_S
    assert np.allclose(nonrel, rest * (1 + beta), rtol=0, atol=1e-12)
    # The two forms differ by rest * beta^2 / 2 to third order in beta.
    assert np.allclose((rel - nonrel) / rest, beta**2 / 2, rtol=1e-2, atol=1e-16)
    assert rel[2] == rest and rel[4] > rel[3] > rest > rel[1]


@pytest.mark.parametrize("relativistic", [True, False])
def test_radial_velocity_round_trip(relativistic):
    v = np.linspace(-0.4 * C_KM_S, 0.4 * C_KM_S, 41) if relativistic else np.linspace(-1000.0, 1000.0, 41)
    obs = wavelengths.doppler_shifted_wavelength(5000.0, v, relativistic)
    back = wavelengths.radial_velocity_from_wavelengths(obs, 5000.0, relativistic)
    assert np.allclose(back, v, rtol=1e-12, atol=1e-8)


def test_resolving_power_tonry_davis_example():
    # [TonryDavis1979] Sect. IV: rms resolution about 2 A, given as 120 km per second at 5000 A.
    r = wavelengths.resolving_power(5000.0, 2.0)
    assert r == 2500.0
    dv = wavelengths.velocity_resolution_km_s(r)
    assert dv == pytest.approx(119.917, abs=1e-3)
    assert round(float(dv), -1) == 120.0
    assert wavelengths.resolving_power_from_velocity(dv) == pytest.approx(2500.0, rel=1e-14, abs=0)


def test_velocity_resolution_first_order_error():
    # c / R versus the relativistic velocity of a shift lambda / R: relative difference about 1 / (2 R).
    for r in (100.0, 1000.0, 10000.0):
        dv_first = wavelengths.velocity_resolution_km_s(r)
        dv_rel = wavelengths.radial_velocity_from_wavelengths(1.0 + 1.0 / r, 1.0, True)
        assert (dv_first - dv_rel) / dv_rel == pytest.approx(0.5 / r, rel=2e-2, abs=0)
    # measured: 0.05 percent at R = 1000.
    dv_rel = wavelengths.radial_velocity_from_wavelengths(1.001, 1.0, True)
    assert abs(1 - dv_rel / wavelengths.velocity_resolution_km_s(1000.0)) == pytest.approx(5.0e-4, rel=0.02, abs=0)


# --------------------------------------------------------------------------------------------------- lines


def _gauss_line(lam, mu, sigma, depth, continuum=1.0):
    return continuum * (1.0 - depth * np.exp(-0.5 * ((lam - mu) / sigma) ** 2))


def test_equivalent_width_noiseless_gaussian_area():
    lam = np.arange(5980.0, 6020.0, 0.3)
    flux = _gauss_line(lam, 6000.07, 1.5, 0.3)
    result = lines.equivalent_width(lam, flux, (5988.0, 6012.0), flux_sigma=0.01)
    exact = 0.3 * 1.5 * np.sqrt(2.0 * np.pi)  # the +-8 sigma window holds all but 1e-15 of the area
    assert result.value_angstrom == pytest.approx(exact, rel=1e-6, abs=0)
    assert result.window_width_angstrom == pytest.approx(24.0, abs=1e-12)


def test_equivalent_width_partial_pixels_and_box_line():
    lam = np.arange(0.0, 40.0, 0.5)  # pixel edges at -0.25, 0.25, ...
    flux = np.ones_like(lam)
    flux[(lam >= 10.0) & (lam < 20.0)] = 0.5  # pixels centred 10 ... 19.5 cover edges 9.75 ... 19.75
    r = lines.equivalent_width(lam, flux, (5.0, 25.0), flux_sigma=0.1)
    assert r.value_angstrom == pytest.approx(0.5 * 10.0, abs=1e-12)
    # A window edge inside a pixel counts that pixel in proportion: the pixel centred at 10 spans 9.75-10.25, so the
    # window (9.875, 25.0) holds 0.375 A of it; the pixels 10.5 ... 19.5 hold 0.5 A each of the line (flux 0.5).
    weights = lines._pixel_weights(lam, 9.875, 25.0)
    assert weights.sum() == pytest.approx(25.0 - 9.875, abs=1e-12)
    assert weights[20] == pytest.approx(0.375, abs=1e-12)
    r = lines.equivalent_width(lam, flux, (9.875, 25.0), flux_sigma=0.1)
    assert r.value_angstrom == pytest.approx(0.5 * (0.375 + 19.0 * 0.5), abs=1e-12)


def test_equivalent_width_uncertainty_formulas():
    lam = np.arange(0.0, 40.0, 0.5)
    flux = np.ones_like(lam)
    # Pixel width h = 0.5; the window edges 10.0 and 20.0 lie at pixel centres, so the two end pixels count half
    # (weight 0.25) and 19 pixels count fully: sigma_W^2 = sum (w sigma / Fc)^2 = (19 * 0.25 + 2 * 0.0625) (0.02/2)^2.
    r = lines.equivalent_width(lam, flux, (10.0, 20.0), flux_sigma=0.02, continuum=2.0 * np.ones_like(lam))
    assert r.sigma_angstrom == pytest.approx(np.sqrt(19 * 0.25 + 2 * 0.0625) * 0.02 / 2.0, rel=1e-12, abs=0)
    # window on pixel edges (10.25 ... 19.75): 19 pixels of weight 0.5, sigma_W = h sqrt(M) sigma / Fc
    r = lines.equivalent_width(lam, flux, (10.25, 19.75), flux_sigma=0.02, continuum=2.0 * np.ones_like(lam))
    assert r.sigma_angstrom == pytest.approx(0.5 * np.sqrt(19.0) * 0.02 / 2.0, rel=1e-12, abs=0)
    # Continuum-level term: (window - W) sigma_c / Fc with W = 0 for flux = continuum.
    r = lines.equivalent_width(lam, 2.0 * flux, (10.0, 20.0), flux_sigma=0.0, continuum=2.0, continuum_sigma=0.1)
    assert r.value_angstrom == pytest.approx(0.0, abs=1e-12)
    assert r.sigma_angstrom == pytest.approx(10.0 * 0.1 / 2.0, rel=1e-12, abs=0)
    with pytest.raises(ValueError):
        lines.equivalent_width(lam, flux, (10.0, 50.0), flux_sigma=0.1)
    with pytest.raises(ValueError):
        lines.equivalent_width(lam, flux, (20.0, 10.0), flux_sigma=0.1)


@pytest.mark.parametrize("snr", [10.0, 30.0, 100.0, 300.0])
def test_equivalent_width_scatter_matches_formal_uncertainty(snr):
    # Gaussian line, depth 0.3, sigma 1.5 A, pixel 0.3 A, window +-5 sigma; 1500 trials.
    # Measured scatter / formal at S/N 10, 30, 100, 300: 0.98, 1.03, 1.01, 1.01 (known continuum); 0.98, 1.00, 0.99,
    # 0.99 (continuum from 7 side pixels, error passed as continuum_sigma); bias below 0.07 formal sigma.
    rng = np.random.default_rng(1)
    lam = np.arange(5990.0, 6010.0, 0.3)
    truth = 0.3 * 1.5 * np.sqrt(2.0 * np.pi)
    profile = _gauss_line(lam, 6000.0, 1.5, 0.3)
    side = np.abs(lam - 6000.0) > 9.0
    known, formal, est, formal_est = [], [], [], []
    for _ in range(1500):
        flux = profile + rng.normal(0.0, 1.0 / snr, lam.size)
        a = lines.equivalent_width(lam, flux, (5992.5, 6007.5), flux_sigma=1.0 / snr)
        known.append(a.value_angstrom)
        formal.append(a.sigma_angstrom)
        cont = flux[side].mean()
        b = lines.equivalent_width(
            lam,
            flux,
            (5992.5, 6007.5),
            flux_sigma=1.0 / snr,
            continuum=cont,
            continuum_sigma=1.0 / snr / np.sqrt(side.sum()),
        )
        est.append(b.value_angstrom)
        formal_est.append(b.sigma_angstrom)
    for values, sigmas in ((known, formal), (est, formal_est)):
        values = np.asarray(values)
        assert abs(values.mean() - truth) / np.mean(sigmas) < 0.15
        assert 0.93 < values.std() / np.mean(sigmas) < 1.07


def test_vollmann_eversberg_weak_line_limit():
    # F = Fc: sigma(W) = sqrt(2) (window - W) / SNR (TeX label eq8 follows from eq4).
    value = lines.equivalent_width_sigma_vollmann_eversberg(10.0, 0.5, 50.0, 1.0, 1.0)
    assert value == pytest.approx(np.sqrt(2.0) * 9.5 / 50.0, rel=1e-14, abs=0)
    # Poisson low-flux line: sqrt(1 + Fc/F) with Fc/F = 4 gives sqrt(5).
    value = lines.equivalent_width_sigma_vollmann_eversberg(10.0, 7.5, 50.0, 0.25, 1.0)
    assert value == pytest.approx(np.sqrt(5.0) * 2.5 / 50.0, rel=1e-14, abs=0)


@pytest.mark.parametrize("depth", [0.3, 0.6])
def test_vollmann_eversberg_monte_carlo_in_the_papers_model(depth):
    # The paper's model: line mean and continuum mean each from M pixels, Poisson noise, per-pixel S/N = sqrt(N0).
    # Measured scatter / formula: 1.002 (depth 0.3, S/N 30), 1.002 (0.3, 100), 1.008 (0.6, 30), 1.000 (0.6, 100);
    # if the continuum comes from 20 M pixels the formula overestimates: scatter / formula 0.78 (0.3), 0.86 (0.6).
    rng = np.random.default_rng(2)
    m, trials = 40, 40000
    for snr_pixel in (30.0, 100.0):
        n0 = snr_pixel**2
        line = rng.poisson(n0 * (1 - depth), (trials, m)).mean(axis=1)
        cont = rng.poisson(n0, (trials, m)).mean(axis=1)
        w = m * (1.0 - line / cont)
        formula = lines.equivalent_width_sigma_vollmann_eversberg(m, m * depth, np.sqrt(m) * snr_pixel, 1 - depth, 1.0)
        assert abs(w.mean() - m * depth) < 0.01 * m * depth
        assert w.std() / formula == pytest.approx(1.0, abs=0.03)
        cont_many = rng.poisson(n0, (trials, 20 * m)).mean(axis=1)
        w_many = m * (1.0 - line / cont_many)
        expected = 0.78 if depth == 0.3 else 0.86
        assert w_many.std() / formula == pytest.approx(expected, abs=0.04)


def test_gaussian_fit_noiseless_recovery():
    lam = np.linspace(5985.0, 6015.0, 121)
    flux = 2.0 * (1.0 + 0.01 * (lam - 6000.0)) - 0.5 * np.exp(-0.5 * ((lam - 6000.3) / 1.8) ** 2)
    fit = lines.fit_gaussian_line(lam, flux, 0.01)
    assert fit.centre == pytest.approx(6000.3, abs=1e-6)
    assert fit.sigma == pytest.approx(1.8, abs=1e-6)
    assert fit.amplitude == pytest.approx(-0.5, abs=1e-6)
    assert fit.fwhm == pytest.approx(2.0 * np.sqrt(2.0 * np.log(2.0)) * 1.8, rel=1e-6, abs=0)
    continuum_at_centre = 2.0 * (1.0 + 0.01 * 0.3)
    assert fit.continuum_at_centre == pytest.approx(continuum_at_centre, rel=1e-6, abs=0)
    assert fit.depth == pytest.approx(0.5 / continuum_at_centre, rel=1e-6, abs=0)
    assert fit.equivalent_width == pytest.approx(
        0.5 * 1.8 * np.sqrt(2.0 * np.pi) / continuum_at_centre, rel=1e-6, abs=0
    )
    assert fit.reduced_chi2 < 1e-6


def test_gaussian_fit_emission_line_signs():
    lam = np.linspace(-10.0, 10.0, 81)
    flux = 1.0 + 0.8 * np.exp(-0.5 * ((lam - 0.5) / 1.2) ** 2)
    fit = lines.fit_gaussian_line(lam, flux, 0.01, continuum_order=0)
    assert fit.amplitude == pytest.approx(0.8, abs=1e-6)
    assert fit.depth < 0 and fit.equivalent_width < 0
    assert fit.equivalent_width == pytest.approx(-0.8 * 1.2 * np.sqrt(2.0 * np.pi), rel=1e-6, abs=0)


@pytest.mark.parametrize("snr", [30.0, 100.0, 300.0])
def test_gaussian_fit_scatter_matches_formal_errors(snr):
    # depth 0.3 line, sigma 2 pixels, 24 pixels (+-6 sigma), free linear continuum, 600 trials.
    # Measured scatter / formal for (centre, sigma, amplitude, EW, depth, FWHM): 0.94-1.00 at S/N 20, 30, 100, 300
    # (e.g. S/N 30: 0.99 0.95 0.94 0.97 0.94 0.95); bias / formal 0.06 (centre), 0.03 (sigma), 0.17 (amplitude),
    # 0.08 (EW), 0.16 (depth) at S/N 30, smaller at higher S/N. At S/N 10 (line significance 3.8) the fit has
    # heavy tails (scatter of the centre 2.6 pixels against a median formal error of 0.46) and is outside the
    # validity range.
    rng = np.random.default_rng(1)
    x = np.arange(-12.0, 12.0, 1.0)
    truth = np.array([0.3, 2.0, -0.3, 0.3 * 2.0 * np.sqrt(2.0 * np.pi), 0.3, 2.0 * 2.3548200450309493])
    values, errors, chi2 = [], [], []
    for _ in range(600):
        flux = (1.0 + 0.01 * x / 12.0) - 0.3 * np.exp(-0.5 * ((x - 0.3) / 2.0) ** 2) + rng.normal(0, 1 / snr, x.size)
        f = lines.fit_gaussian_line(x, flux, 1.0 / snr)
        values.append([f.centre, f.sigma, f.amplitude, f.equivalent_width, f.depth, f.fwhm])
        errors.append(
            [f.centre_sigma, f.sigma_sigma, f.amplitude_sigma, f.equivalent_width_sigma, f.depth_sigma, f.fwhm_sigma]
        )
        chi2.append(f.reduced_chi2)
    values, errors = np.array(values), np.array(errors)
    rms_error = np.sqrt((errors**2).mean(axis=0))
    assert np.all(np.abs(values.mean(axis=0) - truth) / rms_error < 0.25)
    ratio = values.std(axis=0) / rms_error
    assert np.all((ratio > 0.86) & (ratio < 1.14)), ratio
    assert 0.9 < np.mean(chi2) < 1.1


def test_gaussian_fit_without_sigma_scales_to_residuals():
    rng = np.random.default_rng(4)
    x = np.arange(-12.0, 12.0, 1.0)
    flux = 1.0 - 0.3 * np.exp(-0.5 * (x / 2.0) ** 2) + rng.normal(0, 0.02, x.size)
    with_sigma = lines.fit_gaussian_line(x, flux, 0.02)
    without = lines.fit_gaussian_line(x, flux)
    assert without.centre == pytest.approx(with_sigma.centre, abs=1e-8)
    # the residual rms is within sampling error of 0.02, so the scaled errors are close to the absolute ones
    assert without.centre_sigma == pytest.approx(with_sigma.centre_sigma, rel=0.35, abs=0)
    assert without.reduced_chi2 == 1.0
    with pytest.raises(ValueError):
        lines.fit_gaussian_line(x[:4], flux[:4])


def test_voigt_limits():
    x = np.linspace(-6.0, 6.0, 241)
    sigma = 0.8
    gauss = np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    assert np.allclose(lines.voigt_profile(x, sigma, 0.0), gauss, rtol=1e-14, atol=0)
    # gamma / sigma = 1e-6: the first-order Lorentzian correction is about 7 gamma / sigma relative at 2.5 sigma
    # (Dawson-function term of Re w), so the difference from the Gaussian is below 1e-5 there.
    near = np.abs(x) < 2.0
    assert np.allclose(lines.voigt_profile(x, sigma, 1e-6 * sigma)[near], gauss[near], rtol=1e-5)
    gamma = 0.7
    lorentz = gamma / (np.pi * (x**2 + gamma**2))
    assert np.allclose(lines.voigt_profile(x, 0.0, gamma), lorentz, rtol=1e-14, atol=0)
    assert np.allclose(lines.voigt_profile(x, 1e-6 * gamma, gamma), lorentz, rtol=2e-6)
    v = lines.voigt_profile(x, 0.6, 0.5)
    assert np.allclose(v, v[::-1], rtol=1e-14, atol=0)
    assert np.all(np.diff(v[120:]) < 0)  # monotone decrease from the centre
    pole = lines.voigt_profile(np.array([0.0, 1.0]), 0.0, 0.0)
    assert np.isinf(pole[0]) and pole[1] == 0.0
    with pytest.raises(ValueError):
        lines.voigt_profile(0.0, -1.0, 1.0)


@pytest.mark.parametrize("ratio", [1e-3, 0.1, 1.0, 10.0, 1e3])
def test_voigt_unit_area(ratio):
    # (sigma, gamma) = (1, ratio) and (ratio, 1): integral over the real line = 1 to better than 1e-9. The
    # substitution x = s sinh(u) (s the larger width) turns the 1/x^2 Lorentzian tail into an exponentially
    # decaying integrand; u runs over +-40 (the neglected tail mass is 1e-17).
    for sigma, gamma in ((1.0, ratio), (ratio, 1.0)):
        s = max(sigma, gamma)
        u_core = float(np.arcsinh(min(sigma, gamma) / s))
        cuts = np.array([-40.0, -10.0, -3.0, -1.0, -3 * u_core, -u_core, 0.0, u_core, 3 * u_core, 1.0, 3.0, 10.0, 40.0])
        cuts = np.unique(cuts)

        def integrand(u, s=s, sigma=sigma, gamma=gamma):
            return float(lines.voigt_profile(s * np.sinh(u), sigma, gamma)) * s * np.cosh(u)

        total = sum(
            quad(integrand, lo, hi, limit=400, epsabs=1e-13, epsrel=1e-13)[0]
            for lo, hi in zip(cuts[:-1], cuts[1:], strict=True)
        )
        assert total == pytest.approx(1.0, abs=1e-9)


def test_voigt_peak_value_and_half_width():
    # Peak value Re w(i gamma / (sqrt 2 sigma)) / (sigma sqrt(2 pi)) = erfcx(gamma / (sqrt 2 sigma)) / (sigma sqrt(2 pi)).
    from scipy.special import erfcx

    sigma, gamma = 0.9, 0.4
    peak = float(lines.voigt_profile(0.0, sigma, gamma))
    assert peak == pytest.approx(
        erfcx(gamma / (np.sqrt(2.0) * sigma)) / (sigma * np.sqrt(2.0 * np.pi)), rel=1e-13, abs=0
    )


# ----------------------------------------------------------------------------------------------- extraction

GAIN = 2.0
READ_NOISE = 5.0
NX = 31
SIGMA_PIX = 2.0
APERTURE = (9, 22)  # 13 pixels, +-3.25 sigma around x = 15


def _image(rng, flux_e, sky_e=100.0, trace=0.0, sigma=SIGMA_PIX, n_x=NX):
    """Data in ADU (Poisson in electrons plus Gaussian read noise), the true profile and the true flux in ADU."""
    n_l = flux_e.size
    x = np.arange(n_x)[:, None]
    centre = 15.0 + trace * np.linspace(-1.0, 1.0, n_l)[None, :]
    profile = np.exp(-0.5 * ((x - centre) / sigma) ** 2)
    profile /= profile.sum(axis=0)
    mean_e = flux_e[None, :] * profile + sky_e
    electrons = rng.poisson(mean_e).astype(float) + rng.normal(0.0, READ_NOISE, mean_e.shape)
    return electrons / GAIN, profile


def _truth(level, n_l=300):
    return level * (1.0 + 0.3 * np.sin(2.0 * np.pi * np.linspace(0.0, 1.0, n_l) * 2.0))


def _monte_carlo(level, *, known=False, order=3, aperture=APERTURE, trials=25, trace=0.0, cr_fraction=0.0, seed=7):
    rng = np.random.default_rng(seed)
    truth_e = _truth(level)
    truth_adu = truth_e / GAIN
    kw = {"gain_e_per_adu": GAIN, "read_noise_e": READ_NOISE, "sky": 100.0 / GAIN, "aperture": aperture}
    opt_err, box_err, opt_var, box_var, injected, hit, rejected = [], [], [], [], 0, 0, 0
    for _ in range(trials):
        data, profile = _image(rng, truth_e, trace=trace)
        cr = np.zeros(data.shape, dtype=bool)
        if cr_fraction > 0:
            for j in np.flatnonzero(rng.random(data.shape[1]) < cr_fraction):
                i = int(rng.integers(10, 21))
                data[i, j] += 1500.0  # 3000 electrons
                cr[i, j] = True
        opt = extraction.optimal_extract(data, profile=profile if known else None, profile_order=order, **kw)
        box = extraction.box_extract(data, **kw)
        opt_err.append(opt.flux - truth_adu)
        box_err.append(box.flux - truth_adu)
        opt_var.append(opt.variance)
        box_var.append(box.variance)
        injected += int(cr.sum())
        hit += int((opt.rejected & cr).sum())
        rejected += int(opt.rejected.sum())
    stats = {
        "opt_err": np.array(opt_err),
        "box_err": np.array(box_err),
        "opt_var": np.array(opt_var),
        "box_var": np.array(box_var),
        "injected": injected,
        "hit": hit,
        "rejected": rejected,
    }
    stats["opt_z"] = stats["opt_err"] / np.sqrt(stats["opt_var"])
    stats["box_z"] = stats["box_err"] / np.sqrt(stats["box_var"])
    stats["gain"] = stats["box_err"].std() / stats["opt_err"].std()
    return stats


def test_horne_variance_ratio_eq10_closed_form():
    # [Horne1986] Eq. 10: for noise independent of x, var(standard)/var(optimal) = (x2 - x1)/(2 sqrt(pi) sigma);
    # +-3 sigma limits give 3/sqrt(pi) = 1.69. Read-noise-only image (zero signal) with the exact Gaussian profile.
    n_l = 50
    x = np.arange(NX)[:, None]
    profile = np.exp(-0.5 * ((x - 15.0) / SIGMA_PIX) ** 2) * np.ones((1, n_l))
    profile /= profile.sum(axis=0)
    data = np.zeros((NX, n_l))
    ap = (9, 21)  # 12 pixels = 6 sigma, centred at 14.5 +- ... (pixel centres 9 ... 20)
    kw = {"gain_e_per_adu": 1.0, "read_noise_e": 4.0, "aperture": ap}
    opt = extraction.optimal_extract(data, profile=profile, **kw)
    box = extraction.box_extract(data, **kw)
    ratio = box.variance[0] / opt.variance[0]
    assert ratio == pytest.approx(12.0 / (2.0 * np.sqrt(np.pi) * SIGMA_PIX), rel=0.03, abs=0)  # 1.69 within sampling
    # In the continuous limit: exactly n_ap * sum(P^2) with sum(P^2) = 1/(2 sqrt(pi) sigma) for a well sampled Gaussian.
    p_ap = profile[ap[0] : ap[1], 0] / profile[ap[0] : ap[1], 0].sum()
    assert opt.variance[0] == pytest.approx(16.0 / np.sum(p_ap**2), rel=1e-12, abs=0)


def test_optimal_extract_matches_horne_equations_directly():
    # Recompute Eq. 8, Eq. 9 and Eq. 13 from the returned flux, profile and mask for a small example.
    rng = np.random.default_rng(3)
    flux_e = np.full(20, 600.0)
    data, profile = _image(rng, flux_e, n_x=15)
    res = extraction.optimal_extract(data, profile=profile, gain_e_per_adu=GAIN, read_noise_e=READ_NOISE, sky=50.0)
    d = data - 50.0
    p = res.profile
    v = (READ_NOISE / GAIN) ** 2 + np.abs(res.flux * p + 50.0) / GAIN  # Eq. 13 with V_0 and Q
    m = res.mask
    f = np.sum(m * p * d / v, axis=0) / np.sum(m * p**2 / v, axis=0)  # Eq. 8
    assert np.allclose(f, res.flux, rtol=1e-7)  # fixed point of f and V, converged to 1e-9
    assert np.allclose(1.0 / np.sum(m * p**2 / v, axis=0), res.variance, rtol=1e-7)  # Eq. 9
    assert np.allclose(p.sum(axis=0), 1.0, rtol=1e-12)


@pytest.mark.parametrize("level", [300.0])
def test_box_extraction_unbiased_and_variance_consistent(level):
    s = _monte_carlo(level)
    # box: measured mean z -0.02 +- 0.012 and std z 1.002 at flux 300 e; std z 1.01 at flux 30 e.
    assert abs(s["box_z"].mean()) < 0.05
    assert s["box_z"].std() == pytest.approx(1.0, abs=0.04)
    low = _monte_carlo(30.0)
    assert abs(low["box_z"].mean()) < 0.05
    assert low["box_z"].std() == pytest.approx(1.0, abs=0.04)


def test_optimal_extraction_unbiased_and_variance_consistent():
    # Known profile: measured bias_z -0.02 +- 0.009, std z 1.01 (flux 300 e, 25 trials of 300 columns).
    known = _monte_carlo(300.0, known=True)
    assert abs(known["opt_z"].mean()) < 0.05
    assert known["opt_z"].std() == pytest.approx(1.0, abs=0.04)
    # Estimated profile (cubic in wavelength), +-3.25 sigma aperture: measured bias_z -0.005, std z 1.01.
    est = _monte_carlo(300.0)
    assert abs(est["opt_z"].mean()) < 0.06
    assert est["opt_z"].std() == pytest.approx(1.0, abs=0.05)
    # Tilted trace (+-3 pixels across the spectrum) with a wider aperture: measured bias_z +0.06, std z 1.00.
    tilt = _monte_carlo(300.0, trace=3.0, aperture=(6, 25))
    assert abs(tilt["opt_z"].mean()) < 0.12
    assert tilt["opt_z"].std() == pytest.approx(1.0, abs=0.05)


def test_optimal_extraction_wide_aperture_bias_is_positive_and_small():
    # Positivity (Table I: P = MAX[P, 0]) adds profile mass in the wings of a noisy fit; with an aperture of
    # +-6 sigma (rows 3-27) and a cubic profile the measured bias is +0.17 standard errors; a constant (order 0)
    # profile reduces it to +0.07 and the known profile removes it (-0.01).
    cubic = _monte_carlo(300.0, aperture=(3, 28))
    assert 0.10 < cubic["opt_z"].mean() < 0.25
    flat = _monte_carlo(300.0, aperture=(3, 28), order=0)
    assert 0.03 < flat["opt_z"].mean() < 0.12
    known = _monte_carlo(300.0, aperture=(3, 28), known=True)
    assert abs(known["opt_z"].mean()) < 0.05


@pytest.mark.parametrize(
    ("level", "known", "order", "low", "high"),
    [
        (30.0, True, 3, 1.28, 1.44),  # measured 1.36
        (100.0, True, 3, 1.24, 1.38),  # measured 1.31 (variance ratio 1.71; Horne's 1.69 for +-3 sigma)
        (1000.0, True, 3, 1.08, 1.20),  # measured 1.14: the gain fades at high signal
        (30.0, False, 0, 1.12, 1.30),  # measured 1.21 with a constant profile estimated from the same data
        (30.0, False, 3, 1.05, 1.25),  # measured 1.145: a cubic profile is noisy at this signal
    ],
)
def test_signal_to_noise_gain_over_box_extraction(level, known, order, low, high):
    s = _monte_carlo(level, known=known, order=order, trials=25)
    assert low < s["gain"] < high
    assert s["opt_z"].std() == pytest.approx(1.0, abs=0.13)  # 1.10 for the cubic profile at the lowest signal


def test_cosmic_rays_are_rejected():
    # 5 percent of the columns get +1500 ADU in one pixel (about 8 sigma of the peak signal). Measured with the
    # default settings: 294 of 294 hits rejected and no other pixel rejected (60 trials); optimal rms error 17.8
    # ADU against 325 ADU for the box sum; std z 1.00.
    s = _monte_carlo(300.0, cr_fraction=0.05, trials=25)
    assert s["hit"] == s["injected"] > 100
    assert s["rejected"] <= s["injected"] + 5
    assert abs(s["opt_z"].mean()) < 0.08
    assert s["opt_z"].std() == pytest.approx(1.0, abs=0.05)
    assert s["box_err"].std() > 10.0 * s["opt_err"].std()


def test_table_one_variance_underestimates_when_pixels_are_masked():
    # Table I prints var f = sum(M P) / sum(M P^2/V); Eq. 9 gives 1 / sum(M P^2/V). The two differ by the factor
    # sum(M P) < 1 as soon as a pixel is rejected: for a hit on the peak pixel (P about 0.2) by about 20 percent.
    rng = np.random.default_rng(5)
    data, profile = _image(rng, np.full(40, 600.0))
    data[15, 10] += 2000.0
    kw = {"gain_e_per_adu": GAIN, "read_noise_e": READ_NOISE, "sky": 100.0 / GAIN, "aperture": APERTURE}
    eq9 = extraction.optimal_extract(data, profile=profile, variance_convention="eq9", **kw)
    t1 = extraction.optimal_extract(data, profile=profile, variance_convention="table1", **kw)
    assert eq9.rejected[15, 10] and eq9.rejected.sum() == 1
    ratio = t1.variance[10] / eq9.variance[10]
    p_sum = float(np.sum(eq9.mask[:, 10] * eq9.profile[:, 10]))
    assert ratio == pytest.approx(p_sum, rel=1e-12, abs=0)
    assert 0.7 < ratio < 0.9
    other = np.arange(40) != 10
    assert np.allclose(t1.variance[other], eq9.variance[other], rtol=1e-12)
    assert np.allclose(t1.flux, eq9.flux, rtol=0, atol=0)


def test_reject_negative_option_and_axis_handling():
    rng = np.random.default_rng(9)
    data, profile = _image(rng, np.full(40, 600.0))
    data[15, 7] -= 2000.0  # a cold pixel
    kw = {"gain_e_per_adu": GAIN, "read_noise_e": READ_NOISE, "sky": 100.0 / GAIN, "aperture": APERTURE}
    on = extraction.optimal_extract(data, profile=profile, **kw)
    off = extraction.optimal_extract(data, profile=profile, reject_negative=False, **kw)
    assert on.rejected[15, 7] and not off.rejected[15, 7]
    # spatial_axis = 1 on the transposed image gives identical results in the transposed layout.
    tr = extraction.optimal_extract(data.T, profile=profile.T, spatial_axis=1, **{**kw, "sky": 100.0 / GAIN})
    assert np.allclose(tr.flux, on.flux, rtol=1e-12, equal_nan=True)
    assert np.array_equal(tr.rejected, on.rejected.T)
    assert tr.profile.shape == data.T.shape
    box = extraction.box_extract(data, **kw)
    box_t = extraction.box_extract(data.T, spatial_axis=1, **kw)
    assert np.allclose(box.flux, box_t.flux, rtol=1e-12) and np.allclose(box.variance, box_t.variance, rtol=1e-12)


def test_extraction_input_checks():
    data = np.ones((10, 12))
    with pytest.raises(ValueError):
        extraction.box_extract(data, gain_e_per_adu=0.0, read_noise_e=1.0)
    with pytest.raises(ValueError):
        extraction.box_extract(data, gain_e_per_adu=1.0, read_noise_e=1.0, aperture=(5, 3))
    with pytest.raises(ValueError):
        extraction.optimal_extract(data, gain_e_per_adu=1.0, read_noise_e=1.0, variance_convention="x")
    with pytest.raises(ValueError):
        extraction.optimal_extract(np.ones(5), gain_e_per_adu=1.0, read_noise_e=1.0)
    with pytest.raises(ValueError):
        extraction.optimal_extract(data, gain_e_per_adu=1.0, read_noise_e=1.0, profile=-np.ones((10, 12)))
    with pytest.raises(ValueError):
        extraction.box_extract(data, gain_e_per_adu=1.0, read_noise_e=1.0, spatial_axis=2)


def test_box_extract_formulas():
    # [Horne1986] Eq. 2, 3, 12 by hand: D = 10 everywhere, S = 4, V_0 = (3/2)^2, Q = 2, aperture rows 2-4.
    data = np.full((8, 5), 10.0)
    res = extraction.box_extract(data, gain_e_per_adu=2.0, read_noise_e=3.0, sky=4.0, aperture=(2, 5))
    assert np.allclose(res.flux, 3 * (10.0 - 4.0))
    assert np.allclose(res.variance, 3 * ((3.0 / 2.0) ** 2 + 10.0 / 2.0))


def test_sky_fit_recovers_flat_and_tilted_sky_and_rejects_outliers():
    rng = np.random.default_rng(12)
    n_l = 300
    sky_pixels = (np.arange(NX) < 5) | (np.arange(NX) >= 26)
    # flat sky of 100 e (50 ADU): order-0 fit over the 10 sky pixels; with variances taken from the fit the bias is
    # below 0.1 expected sigma (it was -0.28 sigma = -1/Q ADU when the weights came from the noisy pixel values) and
    # the scatter over the Poisson-plus-read expectation sqrt(V / n) is within 5 percent of 1.
    ratios, biases = [], []
    for _ in range(20):
        data, _ = _image(rng, np.zeros(n_l))
        fit = extraction.fit_sky_background(
            data, sky_pixels, gain_e_per_adu=GAIN, read_noise_e=READ_NOISE, order=0, spatial_axis=0
        )
        err = fit[0] - 50.0
        expected = np.sqrt(((READ_NOISE / GAIN) ** 2 + 50.0 / GAIN) / sky_pixels.sum())
        ratios.append(err.std() / expected)
        biases.append(err.mean() / expected)
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.05)
    assert abs(np.mean(biases)) < 0.1
    # tilted sky, noiseless: linear fit reproduces the plane exactly at every x, also under the object.
    xx = np.arange(NX)[:, None] * np.ones((1, 10))
    tilted = 40.0 + 0.5 * xx
    fit = extraction.fit_sky_background(
        tilted, sky_pixels, gain_e_per_adu=GAIN, read_noise_e=READ_NOISE, order=1, spatial_axis=0
    )
    assert np.allclose(fit, tilted, rtol=0, atol=1e-9)
    # an outlier (cosmic ray) in the sky region is clipped
    dirty = tilted.copy()
    dirty[2, 3] += 500.0
    fit = extraction.fit_sky_background(
        dirty, sky_pixels, gain_e_per_adu=GAIN, read_noise_e=READ_NOISE, order=1, spatial_axis=0
    )
    assert np.allclose(fit, tilted, rtol=0, atol=1e-9)
    with pytest.raises(ValueError):
        extraction.fit_sky_background(tilted, np.zeros(NX, dtype=bool), gain_e_per_adu=GAIN, read_noise_e=READ_NOISE)


def test_extraction_with_fitted_sky_underestimates_the_error():
    # Eq. 3 of [Horne1986] neglects the sky-fit noise. With a sky fitted from 10 pixels and a 13-pixel aperture the
    # measured scatter is 1.24-1.27 times the estimate (two seeds, order 0 and 1); the extraction itself stays
    # unbiased (mean z 0.03-0.06).
    rng = np.random.default_rng(21)
    truth_e = _truth(300.0, 200)
    truth_adu = truth_e / GAIN
    sky_pixels = (np.arange(NX) < 5) | (np.arange(NX) >= 26)
    errs, vars_ = [], []
    for _ in range(15):
        data, _ = _image(rng, truth_e)
        sky = extraction.fit_sky_background(data, sky_pixels, gain_e_per_adu=GAIN, read_noise_e=READ_NOISE, order=1)
        res = extraction.optimal_extract(data, gain_e_per_adu=GAIN, read_noise_e=READ_NOISE, sky=sky, aperture=APERTURE)
        errs.append(res.flux - truth_adu)
        vars_.append(res.variance)
    z = np.array(errs) / np.sqrt(np.array(vars_))
    assert abs(z.mean()) < 0.15
    assert 1.15 < z.std() < 1.35


# ---------------------------------------------------------------------------------------------- velocity


def _line_list():
    rng = np.random.default_rng(11)
    return rng.uniform(4500.0, 6000.0, 60), rng.uniform(0.15, 0.6, 60)


_CENTRES, _DEPTHS = _line_list()


def _absorption_spectrum(wave, v_km_s, fwhm_km_s=100.0):
    z = wavelengths.doppler_shifted_wavelength(1.0, v_km_s) - 1.0
    flux = 1.0 + 1e-5 * (wave - 5000.0)
    for c, d in zip(_CENTRES, _DEPTHS, strict=True):
        centre = c * (1.0 + z)
        sig = centre * fwhm_km_s / C_KM_S / 2.3548200450309493
        flux = flux - d * np.exp(-0.5 * ((wave - centre) / sig) ** 2)
    return flux


WAVE = np.arange(4400.0, 6100.0, 0.25)
RANGE = (4600.0, 5900.0)
TEMPLATE = _absorption_spectrum(WAVE, 0.0)


def test_log_rebin_grid_and_example():
    lam = np.linspace(4000.0, 7000.0, 3001)
    out = velocity.rebin_log_wavelength(lam, 2.0 + 3e-3 * lam, n_bins=2048, wavelength_range_angstrom=(4550.0, 6200.0))
    steps = np.diff(np.log(out.wavelength_angstrom))
    assert np.allclose(steps, 1.0 / out.bins_per_ln_lambda, rtol=1e-9, atol=0)  # rounding of log of 4-digit numbers
    assert out.wavelength_angstrom[0] == pytest.approx(4550.0, rel=1e-6, abs=0) and out.wavelength_angstrom[
        -1
    ] == pytest.approx(6200.0, rel=1e-6, abs=0)
    assert out.bins_per_ln_lambda == pytest.approx(2047.0 / np.log(6200.0 / 4550.0), rel=1e-6, abs=0)
    # [TonryDavis1979] Sect. IV: 4550-6200 A in 2048 bins, a velocity of about 45 km per second shifts a feature by one bin.
    assert 44.5 < C_KM_S / out.bins_per_ln_lambda < 46.0
    # linear interpolation reproduces a linear function exactly
    assert np.allclose(out.flux, 2.0 + 3e-3 * out.wavelength_angstrom, rtol=1e-12)
    with pytest.raises(ValueError):
        velocity.rebin_log_wavelength(lam, lam, wavelength_range_angstrom=(3000.0, 6000.0))
    with pytest.raises(ValueError):
        velocity.rebin_log_wavelength(lam[::-1], lam)


def test_prepare_for_correlation_filters():
    n = 2048
    t = np.linspace(-1.0, 1.0, n)
    # a fourth-degree polynomial is removed exactly
    poly = 3.0 + 2.0 * t - 5.0 * t**2 + 1.5 * t**3 + 4.0 * t**4
    assert np.max(np.abs(velocity.prepare_for_correlation(poly))) < 1e-9
    n_idx = np.arange(n)

    def passband_gain(k):
        wave = np.sin(2.0 * np.pi * k * n_idx / n)
        out = velocity.prepare_for_correlation(wave)
        core = slice(n // 4, 3 * n // 4)  # away from the 5 percent end tapers
        return np.sqrt(np.mean(out[core] ** 2) / np.mean(wave[core] ** 2))

    assert passband_gain(3.0) < 0.02  # below wavenumber 10
    assert passband_gain(60.0) == pytest.approx(1.0, abs=0.02)  # in the flat part 20-100
    assert passband_gain(400.0) < 0.01  # above 200
    out = velocity.prepare_for_correlation(np.sin(2.0 * np.pi * 60.0 * n_idx / n) + 5.0)
    assert abs(out.mean()) < 1e-3
    assert abs(out[0]) < 1e-3 and abs(out[-1]) < 1e-3  # tapered ends (unit-amplitude sine; filter spreads the taper)
    with pytest.raises(ValueError):
        velocity.prepare_for_correlation(np.ones(8))
    with pytest.raises(ValueError):
        velocity.prepare_for_correlation(np.ones(64), bandpass_wavenumbers=(20.0, 10.0, 100.0, 200.0))


def test_cross_correlation_definition_and_sign():
    rng = np.random.default_rng(2)
    n = 256
    t = rng.normal(size=n)
    g = rng.normal(size=n)
    cc = velocity.cross_correlate(g, t)
    sg, st = np.sqrt(np.mean(g**2)), np.sqrt(np.mean(t**2))
    for lag in (-37, -1, 0, 5, 100):
        direct = sum(g[m] * t[(m - lag) % n] for m in range(n)) / (n * sg * st)  # Eq. 4, periodic
        assert cc.correlation[int(lag + n // 2)] == pytest.approx(direct, abs=1e-12)
    ident = velocity.cross_correlate(t, t)
    assert ident.correlation[n // 2] == pytest.approx(1.0, abs=1e-12)
    assert np.argmax(ident.correlation) == n // 2
    shifted = np.roll(t, 7)  # g(n) = t(n - 7): peak at +7
    peak_lag = velocity.cross_correlate(shifted, t).lag_bins[
        np.argmax(velocity.cross_correlate(shifted, t).correlation)
    ]
    assert peak_lag == 7.0
    with pytest.raises(ValueError):
        velocity.cross_correlate(np.zeros(8), np.ones(8))


def test_correlation_peak_fit_parabola_and_gaussian():
    lags = np.arange(-512.0, 512.0)
    # exact parabola, FWHM w: c = h (1 - 2 (n - delta)^2 / w^2) (Tonry & Davis text after Eq. 20)
    for delta in (-3.0, 0.37, 4.5):
        c = 0.8 * (1.0 - 2.0 * (lags - delta) ** 2 / 9.0**2)
        peak = velocity.fit_correlation_peak(
            velocity.CrossCorrelation(lags, c), lag_window_bins=(-20.0, 20.0), n_points=11
        )
        assert peak.centre_bins == pytest.approx(delta, abs=1e-9)
        assert peak.height == pytest.approx(0.8, rel=1e-12, abs=0)
        assert peak.fwhm_bins == pytest.approx(9.0, rel=1e-9, abs=0)

    # Gaussian peaks at fractional positions in +-2 bins: the largest vertex error of the parabola fit (measured)
    # is 0.074 bin for FWHM 9 and 11 points (the paper's choice), 0.012 bin with 5 points, 0.029 bin for FWHM 15
    # and 11 points. The error comes from the parabola not being a Gaussian over +-1.3 sigma.
    def worst(fwhm, n_points):
        sigma = fwhm / 2.3548200450309493
        largest = 0.0
        for delta in np.linspace(-2.0, 2.0, 41):
            c = 0.9 * np.exp(-0.5 * ((lags - delta) / sigma) ** 2)
            peak = velocity.fit_correlation_peak(
                velocity.CrossCorrelation(lags, c), lag_window_bins=(-20.0, 20.0), n_points=n_points
            )
            largest = max(largest, abs(peak.centre_bins - delta))
        return largest

    assert 0.05 < worst(9.0, 11) < 0.09
    assert worst(9.0, 5) < 0.02
    assert worst(15.0, 11) < 0.04
    # the window restricts the search: a larger peak outside it is ignored
    c = np.exp(-0.5 * ((lags - 100.0) / 5.0) ** 2) + 0.5 * np.exp(-0.5 * ((lags + 50.0) / 5.0) ** 2)
    peak = velocity.fit_correlation_peak(velocity.CrossCorrelation(lags, c), lag_window_bins=(-80.0, 80.0))
    assert peak.centre_bins == pytest.approx(-50.0, abs=0.05)
    with pytest.raises(ValueError):
        velocity.fit_correlation_peak(
            velocity.CrossCorrelation(lags, (lags / 10.0) ** 2), lag_window_bins=(100.0, 101.0)
        )


def test_r_statistic_closed_forms():
    n = 512
    lags = np.arange(n) - n // 2
    n_idx = np.arange(n)
    even = np.exp(-0.5 * (lags / 4.0) ** 2)
    # purely even about lag 0: sigma_a = 0 and r = inf
    assert np.isinf(velocity.correlation_r_statistic(velocity.CrossCorrelation(lags.astype(float), even), 0.0, 1.0))
    # even part plus an odd sine of amplitude A: sigma_a = A exactly (Eq. 20), r = h / (sqrt(2) A) (Eq. 23)
    amp = 0.05
    odd = amp * np.sin(2.0 * np.pi * 9.0 * (lags) / n)
    c = velocity.CrossCorrelation(lags.astype(float), even + odd)
    assert velocity.correlation_r_statistic(c, 0.0, 1.0) == pytest.approx(1.0 / (np.sqrt(2.0) * amp), rel=1e-9, abs=0)
    # the same even peak centred at a fractional position is still symmetric about it (band-limited Fourier shift)
    centred = np.exp(-0.5 * (((n_idx - n // 2 - 0.37) + n / 2) % n - n / 2) ** 2 / 4.0**2)
    r_big = velocity.correlation_r_statistic(velocity.CrossCorrelation(lags.astype(float), centred), 0.37, 1.0)
    assert r_big > 1e6


def test_half_amplitude_wavenumber_gaussian_correlation():
    # c(n) = exp(-n^2 / (2 s^2)): |C(k)| ~ exp(-(2 pi k s / N)^2 / 2), half amplitude at k = N sqrt(2 ln 2) / (2 pi s).
    n, s = 2048, 2.0
    lags = np.arange(n) - n // 2
    c = np.exp(-0.5 * (lags / s) ** 2)
    k_half = n * np.sqrt(2.0 * np.log(2.0)) / (2.0 * np.pi * s)
    est = velocity.half_amplitude_wavenumber(velocity.CrossCorrelation(lags.astype(float), c))
    assert est == pytest.approx(k_half, rel=0.02, abs=0)


def test_shift_error_paper_numbers():
    # Eq. 35 of [TonryDavis1979]: N = 2048 and 2B = 100 give (1/4)(2048/100) = 5.12, printed as 5 / (1 + r).
    assert velocity.shift_error_bins(2048, 0.0, 50.0) == pytest.approx(5.12, rel=1e-6, abs=0)
    assert velocity.shift_error_bins(2048, 4.0, 50.0) == pytest.approx(5.12 / 5.0, rel=1e-6, abs=0)
    assert velocity.shift_error_bins(2048, 4.0, 50.0, scale=2.0) == pytest.approx(2.0 * 5.12 / 5.0, rel=1e-6, abs=0)
    with pytest.raises(ValueError):
        velocity.shift_error_bins(2048, -1.0, 50.0)


@pytest.mark.parametrize("v", [-150.0, -37.3, 0.0, 12.9, 149.0])
def test_noise_free_velocity_is_recovered_with_correct_sign(v):
    # Measured noise-free error over +-150 km/s: largest 1.63 km/s, rms 0.95 km/s with the paper's 11-point
    # parabola (peak fit systematic; one log bin is 36 km/s here); 0.27 and 0.17 km/s with 5 points.
    obs = _absorption_spectrum(WAVE, v)
    res = velocity.radial_velocity_cross_correlation(WAVE, obs, WAVE, TEMPLATE, wavelength_range_angstrom=RANGE)
    assert res.velocity_km_s == pytest.approx(v, abs=1.9)
    res5 = velocity.radial_velocity_cross_correlation(
        WAVE, obs, WAVE, TEMPLATE, wavelength_range_angstrom=RANGE, peak_points=5
    )
    assert res5.velocity_km_s == pytest.approx(v, abs=0.4)
    assert (res.velocity_km_s > 0) == (v > 5.0) or abs(v) < 5.0
    assert res.height > 0.97 and res.r > 100.0


def test_velocity_scatter_vs_estimated_error():
    # 150 trials per S/N, velocities uniform in +-150 km/s, 60 lines of FWHM 100 km/s, white noise at the given S/N
    # per 0.25 A pixel, noise-free template, 2048 bins of 36.45 km/s. Measured (bias, scatter, mean estimated
    # error, scatter / rms estimate, median r) in km/s:
    #   S/N 200: -0.16, 0.85, 0.26, 3.3, 413   S/N 100: -0.17, 0.92, 0.46, 2.0, 233   S/N 50: -0.20, 1.14, 0.94, 1.2, 122
    #   S/N  20: -0.39, 2.08, 2.42, 0.84, 49   S/N  10: -0.63, 4.03, 4.93, 0.79, 25
    # At high S/N the scatter saturates near 0.9 km/s (peak-fit floor, not in the estimate); with peak_points=5 the
    # scatter is 0.24 km/s at S/N 200 and 3.80 km/s at S/N 10.
    expected = {  # snr: (scatter, scatter / estimate)
        200: (0.85, 3.3),
        100: (0.92, 2.0),
        50: (1.14, 1.2),
        20: (2.08, 0.84),
        10: (4.03, 0.79),
    }
    for snr, (scatter_ref, ratio_ref) in expected.items():
        rng = np.random.default_rng(3)
        true, found, err = [], [], []
        for _ in range(150):
            v = rng.uniform(-150.0, 150.0)
            obs = _absorption_spectrum(WAVE, v) + rng.normal(0.0, 1.0 / snr, WAVE.size)
            res = velocity.radial_velocity_cross_correlation(WAVE, obs, WAVE, TEMPLATE, wavelength_range_angstrom=RANGE)
            true.append(v)
            found.append(res.velocity_km_s)
            err.append(res.error_km_s)
        diff = np.array(found) - np.array(true)
        scatter = diff.std()
        assert abs(diff.mean()) < 3.5 * scatter / np.sqrt(diff.size) + 0.3
        assert scatter == pytest.approx(scatter_ref, rel=0.2, abs=0)
        assert scatter / np.sqrt(np.mean(np.square(err))) == pytest.approx(ratio_ref, rel=0.2, abs=0)
    # the estimated error decreases monotonically with S/N and the estimate is within a factor 1.3 of the scatter
    # for S/N <= 50


def test_radial_velocity_window_and_failures():
    obs = _absorption_spectrum(WAVE, 400.0)
    # a +-200 km/s window does not contain the true peak at +400 km/s: the largest value in the window is at its
    # edge and the parabola there is not concave, so the function refuses to return a velocity
    with pytest.raises(ValueError):
        velocity.radial_velocity_cross_correlation(
            WAVE, obs, WAVE, TEMPLATE, wavelength_range_angstrom=RANGE, velocity_window_km_s=(-200.0, 200.0)
        )
    res = velocity.radial_velocity_cross_correlation(
        WAVE, obs, WAVE, TEMPLATE, wavelength_range_angstrom=RANGE, velocity_window_km_s=(0.0, 800.0)
    )
    assert res.velocity_km_s == pytest.approx(400.0, abs=2.0)
