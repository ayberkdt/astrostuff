# Time series analysis

Target module: `timeseries/`.

Astronomical time series are not evenly spaced (gaps from day, weather and season), their noise varies from point to point, and it is often correlated. Methods are chosen according to these three properties.

## 1. Data model and preprocessing

- Light curve container: time (scale and reference stated, preferably `BJD_TDB`), flux or magnitude, uncertainty, quality flag, band [Eastman2010].
- Outlier removal: sigma clipping relative to a moving median; one-sided clipping so as not to cut a real signal such as a transit.
- **Detrending:** moving median, biweight filter, Savitzky-Golay, spline, Gaussian process. The window width is chosen longer than the duration of the sought signal; otherwise the signal is suppressed. Oracle: `wotan`.
- Systematics removal: modeling common trends with principal components (in space telescope data); pixel-level regression.
- Binning and phase folding.

## 2. Period search

**Lomb-Scargle periodogram.** Interpreted as the `χ²` improvement of a sinusoid fit to unevenly spaced data [Lomb1976; Scargle1982]. Implementation details follow VanderPlas's review [VanderPlas2018]:

- Floating-mean (generalized) form and measurement error weights.
- Frequency grid: resolution as a fraction of `1/T` (oversampling factor of 5 or more); choice of the upper limit. For unevenly spaced data the Nyquist frequency is not well defined.
- Window function and aliases: daily (`±1 day⁻¹`), yearly and sampling-induced peaks. The window periodogram is plotted in every analysis.
- False alarm probability: analytic approximations and bootstrap.
- Multi-term (Fourier series) extension: for non-sinusoidal shapes.
- Fast algorithm: the `O(N log N)` approximation; the direct `O(N·N_f)` form is written first.

**Phase dispersion methods.** The period that minimizes the scatter of the folded light curve; strong for non-sinusoidal shapes (eclipsing binaries) [Stellingwerf1978].

**Refining the period and its uncertainty:** model fitting around the peak; the posterior distribution with MCMC. The width of the periodogram peak is not the uncertainty.

Oracle: `astropy.timeseries.LombScargle`.

## 3. Transit search

- **Box least squares (BLS):** search for a box-shaped dip over a grid of period, duration and phase; the signal residue and signal detection efficiency statistics [Kovacs2002].
- **Transit least squares (TLS):** a realistic limb-darkened transit shape instead of a box; more sensitive for small planets [HippkeHeller2019].
- Period grid: not uniform in frequency, optimal sampling that depends on the stellar density.
- Vetting tests: odd-even transit depth difference, secondary eclipse, centroid shift; removing eclipsing binary false positives.

Oracles: `astropy.timeseries.BoxLeastSquares`, `transitleastsquares`.

## 4. Model fitting

- Transit model: radius ratio, scaled semi-major axis, inclination, limb darkening coefficients, mid-transit time [MandelAgol2002; Winn2010]. Oracle: `batman` [Kreidberg2015].
- Integration of the model over the exposure time for long exposures.
- Pulsating variables: Fourier decomposition, amplitude and phase ratios; classification of Cepheids and RR Lyrae.
- Eclipsing binaries: times of minima and the O-C diagram; period change.
- Supernova light curves: template fitting through `sncosmo` [Guy2007].

## 5. Correlated noise

- In the presence of red noise, the white noise assumption makes the uncertainties look smaller.
- Diagnostic: whether the RMS of the binned residuals scales as `1/√N`.
- Modeling the noise and the signal together with a Gaussian process [RasmussenWilliams2006; ForemanMackey2017].
- Stochastic variability: structure function, damped random walk (active galactic nuclei) [Ivezic2020, Section 10].

## 6. Other methods

- Wavelet transform: time-varying periods.
- Autocorrelation function: stellar rotation period.
- Bayesian blocks: detecting outbursts and flares [Ivezic2020, Section 5].
- Transient classification: feature extraction and supervised learning (ZTF and Rubin alert streams) [Bellm2019].

## 7. Verification

- Synthetic light curve: known period, amplitude, realistic sampling and noise. The recovery rate is measured against S/N and the number of observations.
- Injection-recovery: an artificial transit is added to a real light curve; a detection efficiency map is produced.
- The period, depth and duration of a published TESS planet are recovered [Ricker2015].
- The catalog periods of OGLE or Gaia Cepheids are found again.
- Periodogram values agree with `astropy` on the same grid at the `1e-10` level.
