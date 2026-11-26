"""kozmos.spectroscopy against independent implementations: SciPy, astropy.

No independent air/vacuum implementation is installed (specutils is absent; astropy alone has none), so the
air-vacuum conversions have no oracle test here; they are checked against the printed table of Morton (1991)
in tests/unit/test_spectroscopy.py.
"""

import numpy as np
import pytest
import scipy.signal
import scipy.special
from astropy import units as u
from astropy.modeling import fitting, models

from kozmos.spectroscopy import lines, velocity, wavelengths

pytestmark = pytest.mark.oracle


def test_voigt_vs_scipy_voigt_profile():
    x = np.linspace(-40.0, 40.0, 4001)
    worst = 0.0
    for sigma in (0.01, 0.3, 1.0, 5.0):
        for gamma in (0.0, 1e-4, 0.1, 1.0, 20.0):
            ours = lines.voigt_profile(x, sigma, gamma)
            ref = scipy.special.voigt_profile(x, sigma, gamma)
            worst = max(worst, float(np.max(np.abs(ours - ref) / ref)))
    # Same Faddeeva routine underneath, so the agreement is rounding level; measured largest relative difference
    # 2.4e-14 (sigma = 0.01 ... 5, gamma = 0 ... 20, |x| <= 40). Tolerance 1e-12 leaves room for platforms.
    assert worst < 1e-12


def test_voigt_vs_astropy_voigt1d_shape():
    # astropy's Voigt1D has its own parametrisation (FWHMs and a Lorentzian peak amplitude); compare the shape,
    # V(x) / V(0), using its exact 'wofz' method. FWHM_G = 2 sqrt(2 ln 2) sigma, FWHM_L = 2 gamma.
    x = np.linspace(-8.0, 8.0, 321)
    for sigma, gamma in ((0.5, 0.3), (1.0, 1.0), (0.2, 2.0)):
        model = models.Voigt1D(
            x_0=0.0,
            amplitude_L=1.0,
            fwhm_L=2.0 * gamma,
            fwhm_G=2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma,
            method="wofz",
        )
        shape_astropy = model(x) / model(0.0)
        ours = lines.voigt_profile(x, sigma, gamma)
        assert np.allclose(ours / ours[160], shape_astropy, rtol=1e-10, atol=0)


def test_gaussian_fit_vs_astropy_levmar():
    rng = np.random.default_rng(8)
    x = np.arange(-12.0, 12.0, 0.5)
    noise = 0.02
    flux = 1.0 + 0.01 * x - 0.3 * np.exp(-0.5 * ((x - 0.4) / 1.9) ** 2) + rng.normal(0.0, noise, x.size)
    ours = lines.fit_gaussian_line(x, flux, noise, continuum_order=1)
    init = models.Gaussian1D(amplitude=-0.25, mean=0.2, stddev=1.5) + models.Linear1D(slope=0.0, intercept=1.0)
    fitter = fitting.LevMarLSQFitter(calc_uncertainties=True)
    best = fitter(init, x, flux, weights=np.full(x.size, 1.0 / noise))
    # Same model, same weights: the optimum agrees to the optimiser tolerances (measured relative differences:
    # centre 6.5e-7, sigma 9.8e-7, amplitude 3.0e-7; formal errors 1e-5 to 2e-5), so 1e-5 and 1e-3 are used.
    assert ours.centre == pytest.approx(best.mean_0.value, rel=1e-5, abs=0)
    assert ours.sigma == pytest.approx(abs(best.stddev_0.value), rel=1e-5, abs=0)
    assert ours.amplitude == pytest.approx(best.amplitude_0.value, rel=1e-5, abs=0)
    cov = fitter.fit_info["param_cov"]
    # astropy parameter order: amplitude_0, mean_0, stddev_0, slope_1, intercept_1; ours: amplitude, centre, sigma.
    assert ours.amplitude_sigma == pytest.approx(np.sqrt(cov[0, 0]), rel=1e-3, abs=0)
    assert ours.centre_sigma == pytest.approx(np.sqrt(cov[1, 1]), rel=1e-3, abs=0)
    assert ours.sigma_sigma == pytest.approx(np.sqrt(cov[2, 2]), rel=1e-3, abs=0)


@pytest.mark.parametrize("v_km_s", [-250.0, -12.0, 30.0, 4.0e4])
def test_doppler_vs_astropy_equivalencies(v_km_s):
    rest = 6562.8 * u.AA
    observed_rel = wavelengths.doppler_shifted_wavelength(6562.8, v_km_s, True)
    # astropy: relativistic velocity from wavelength (rest wavelength given to the equivalency)
    v_astropy = (observed_rel * u.AA).to(u.km / u.s, equivalencies=u.doppler_relativistic(rest))
    assert v_astropy.value == pytest.approx(v_km_s, rel=1e-10, abs=1e-8)
    assert float(wavelengths.radial_velocity_from_wavelengths(observed_rel, 6562.8, True)) == pytest.approx(
        v_astropy.value, rel=1e-10, abs=1e-8
    )
    observed_nr = wavelengths.doppler_shifted_wavelength(6562.8, v_km_s, False)
    v_optical = (observed_nr * u.AA).to(u.km / u.s, equivalencies=u.doppler_optical(rest))
    assert v_optical.value == pytest.approx(v_km_s, rel=1e-10, abs=1e-8)
    assert float(wavelengths.radial_velocity_from_wavelengths(observed_nr, 6562.8, False)) == pytest.approx(
        v_optical.value, rel=1e-10, abs=1e-8
    )


def test_cross_correlation_vs_scipy_correlate():
    # c(n) N sigma_g sigma_t = sum_m g(m + n) t(m) (periodic) = scipy.signal.correlate(g tiled twice, t, 'valid')[n].
    rng = np.random.default_rng(6)
    n = 300
    g, t = rng.normal(size=n), rng.normal(size=n)
    ours = velocity.cross_correlate(g, t)
    ref = scipy.signal.correlate(np.concatenate([g, g]), t, mode="valid")[:n]
    scale = n * np.sqrt(np.mean(g**2)) * np.sqrt(np.mean(t**2))
    # lags 0 ... N-1 of the periodic function: lag n >= N/2 is the same as lag n - N in the fftshifted layout
    got = np.array([ours.correlation[(lag if lag < n // 2 else lag - n) + n // 2] for lag in range(n)])
    assert np.allclose(got, ref / scale, rtol=0, atol=1e-12)
