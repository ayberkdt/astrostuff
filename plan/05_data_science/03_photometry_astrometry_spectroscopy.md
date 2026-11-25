# Photometry, astrometry and spectroscopic measurement

Target modules: `photometry/`, `astrometry/`, `spectroscopy/`.

The input is a reduced image ([02_image_reduction_pipeline.md](02_image_reduction_pipeline.md)); the output is a calibrated catalog or spectrum.

## 1. Source detection

- Matched filter: convolving the image with the PSF gives the highest S/N for a point source; the threshold is a multiple of the background RMS.
- Star finder: local maxima in the convolved image, rejection of cosmic rays and extended sources with sharpness and roundness criteria [Stetson1987].
- Segmentation: groups of connected pixels above the threshold, a minimum area condition; multi-threshold deblending [BertinArnouts1996].
- Completeness and purity: measured against magnitude with an artificial star injection experiment.

Oracles: `photutils` (DAOStarFinder, segmentation), `sep` [Barbary2016].

## 2. Centroid and shape measurement

- Centroid: weighted first moment, Gaussian fit to the marginal sums, 2D PSF fit. The astrometric precision is of the order of `FWHM / (S/N)`; the coefficient and the derivation are written with their source [King1971].
- Second moments: FWHM, ellipticity, position angle; star-galaxy separation.

## 3. Aperture photometry

- Circular and elliptical apertures; exact area computation of the pixel-circle intersection (a geometric solution, not subpixel sampling).
- Local sky: sigma-clipped estimate inside the annulus.
- Uncertainty: the CCD equation [MerlineHowell1995].
- Aperture correction: carrying the flux of a small aperture to the total flux with the growth curve.
- Kron aperture: an adaptive elliptical aperture tied to the first-moment radius; for galaxies [Kron1980].
- Optimal (weighted) photometry: PSF-weighted sum; higher S/N than aperture photometry for faint sources [Naylor1998].

## 4. PSF photometry

- PSF model: analytic (Gaussian, Moffat) + empirical residual table; variation across the field [Stetson1987].
- Choice of PSF stars: bright, isolated, unsaturated.
- Fitting: nonlinear least squares for position and flux; simultaneous fitting of neighbor groups in a crowded field.
- Iteration: fit, subtract, search for new sources in the residual.
- Quality measures: `χ`, sharpness; inspection of the residual image.

## 5. Photometric calibration

- Instrumental magnitude: `m_inst = −2.5 log10(flux / t)`.
- **Absolute (all-sky) photometry:** `m_std = m_inst + ZP − k X + c (color)`; the zero point, extinction coefficient and color term are solved together with standard stars [Landolt1992; Bessell2005].
- **Differential photometry:** relative to comparison stars in the same frame; the first-order effect of the atmosphere drops out. Ensemble photometry: weighted combination of many comparison stars.
- **Catalog calibration:** matching the field stars to Gaia, Pan-STARRS or SDSS magnitudes; system transformation and color term with a robust linear fit [Vallenari2023; Fukugita1996].
- Systematics: flat field error, second-order extinction, shutter time, linearity; a diagnostic plot for each.

## 6. Astrometric solution

- **Blind solution:** geometric hashing of four-star asterisms and index lookup [Lang2010]. This algorithm could be rewritten, but the index files are large; in the first stage `astrometry.net` or `twirl` is used, and our own solver is written for the case where the approximate position is known.
- **Approximate position known:** triangle matching between the detected sources and the Gaia catalog in the TAN projection; then a linear (6-parameter) and polynomial distortion fit.
- Reference catalog: Gaia DR3; the stars are moved to the observation epoch with their proper motions [Vallenari2023].
- Solution quality: RMS of the residuals, pattern across the field; differential color refraction.
- Minor body astrometry: mid-exposure time and trail center for a moving source.

## 7. Spectral reduction

Common sequence for long slit and echelle:

1. Image reduction (bias, dark, flat field; in the spectral flat field the continuum of the lamp is normalized).
2. **Trace finding:** following the spatial center of the spectrum along the dispersion axis with a polynomial.
3. **Background:** a model per wavelength column from sky regions along the slit.
4. **Extraction:** boxcar sum and optimal extraction. Optimal extraction uses the spatial profile as the weight, increases the S/N and removes cosmic rays through profile mismatch [Horne1986].
5. **Wavelength calibration:** detecting arc lamp lines, matching them with the line list, a pixel → wavelength polynomial; the residual RMS is the quality measure. Zero-point correction with sky lines. The conversion between air and vacuum wavelengths relies on the refractive index formula of air [Ciddor1996].
6. **Flux calibration:** sensitivity function with a spectrophotometric standard star; correction for atmospheric extinction [OkeGunn1983].
7. **Telluric correction** and barycentric correction [WrightEastman2014].

Oracles: `specreduce`, `specutils`, PypeIt.

## 8. Spectral analysis

- Continuum normalization: iterative polynomial or spline, line masking.
- Line measurement: Gaussian, Lorentzian, Voigt fitting; equivalent width, flux, center, FWHM and their uncertainties.
- **Redshift:** emission line matching; cross-correlation with a template and peak fitting, confidence measure with the `r` statistic [TonryDavis1979]; template `χ²` scan.
- Radial velocity: cross-correlation on a log-wavelength grid.
- Spectral classification: line ratios and template matching.
- Combining and resampling spectra: conservation of flux and variance.

The redshift measurement gives the horizontal axis of the Hubble diagram; recovering the published values on SDSS spectra is the end-to-end test of this module [York2000].

## 9. Verification

- On a synthetic star field, aperture and PSF photometry return the input flux without bias; the residuals are consistent with the uncertainty estimate.
- Comparison of the source list and fluxes with `photutils` and `sep` on the same image.
- The astrometric solution is compared with the `astrometry.net` solution; the difference is below the Gaia residuals.
- The color-magnitude diagram of an open cluster is compared with published photometry.
