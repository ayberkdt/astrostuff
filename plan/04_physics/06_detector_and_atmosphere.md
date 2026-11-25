# Detector, telescope and atmosphere physics

Target modules: `observing/`, `reduction/` (model side).

In ground-based observation a photon passes through the atmosphere, the optics and the detector between leaving the source and becoming a count. Data reduction is the inversion of this chain; without modeling the chain, it is not clear what the reduction steps correct. The equations are read from the cited source before they are coded.

## 1. Atmosphere

- **Airmass:** plane-parallel approximation `X = sec z`; approximation formulas for a spherical atmosphere [KastenYoung1989; Young1994]. For `z > 80°` the formulas diverge; the validity range of each is written in its docstring.
- **Extinction:** Bouguer law `m_observed = m_0 + k_λ X`. `k_λ` consists of Rayleigh scattering (`∝ λ⁻⁴`), aerosol and ozone components; it is measured during the night from standard stars. A second-order color term is needed in wide bands.
- **Refraction and dispersion:** [../03_mathematics/01_spherical_astronomy_and_transforms.md](../03_mathematics/01_spherical_astronomy_and_transforms.md).
- **Seeing:** Kolmogorov turbulence, Fried parameter `r0 ∝ λ^(6/5)`, seeing disk `FWHM ≈ 0.98 λ/r0`; airmass dependence as `X^0.6` [Fried1966].
- **Sky brightness:** airglow, zodiacal light, the contribution of the Moon (phase and angular distance), light pollution; Moon brightness model [KrisciunasSchaefer1991].
- **Telluric absorption:** O₂ and H₂O bands; correction in the spectrum.

## 2. Telescope and optics

- Collecting area, focal ratio, plate scale: `scale [″/mm] = 206265 / f [mm]`; pixel scale.
- Diffraction: Airy pattern, `θ = 1.22 λ/D`; effect of the central obstruction.
- Point spread function models: Gaussian, Moffat, Airy; combination of seeing + optics + tracking [King1971; Moffat1969].
- Sampling: Nyquist criterion `FWHM ≳ 2 pixels`; the effect of undersampling on photometry and astrometry.
- Transmission chain: atmosphere × mirrors × filter × detector quantum efficiency; the system zero point is predicted from this chain and compared with the measured one.

## 3. Detector

Signal model of a CCD and CMOS detector [Howell2006, Section 3-4; Janesick2007]:

```
raw[ADU] = ( flux · t · QE · flat_field + dark_current · t ) / gain + bias + noise
```

- **Gain** [e⁻/ADU], **read noise** [e⁻], **dark current** [e⁻/s] and its temperature dependence.
- **Bias** level and structure; overscan region.
- **Flat field:** pixel-to-pixel sensitivity, vignetting, dust shadows; it depends on wavelength.
- **Linearity and saturation:** full well capacity, deviation from linearity curve, blooming.
- **Photon transfer curve:** gain from the slope of the variance-mean line, read noise from its intercept [Janesick2007].
- Defects: hot and dead pixels, bad columns, charge transfer inefficiency, fringing, residual image.
- CMOS differences: per-pixel gain and noise, fixed pattern noise, rolling shutter, amplifier glow.
- Cosmic rays: sharp-edged tracks narrower than the PSF [vanDokkum2001].

## 4. Signal-to-noise ratio

CCD equation [Howell2006, Section 4; MerlineHowell1995]:

```
S/N = N* / sqrt( N* + n_pix (1 + n_pix/n_B) (N_S + N_D + N_R² + G² σ_f²) )
```

`N*` is the source electrons, `n_pix` the number of pixels in the aperture, `n_B` the number of pixels used in the background estimate, `N_S` the sky per pixel, `N_D` the dark current, `N_R` the read noise, `G² σ_f²` the digitization noise.

- Regimes: source dominated (`S/N ∝ √t`), sky dominated, read noise dominated (`S/N ∝ t`).
- Optimal aperture radius: about `0.68 FWHM` for a Gaussian PSF and the sky-dominated case; the derivation is written in the project.
- Exposure time calculator: time for a target magnitude and S/N; saturation time; exposure splitting (balance of cosmic rays and read noise).
- For spectra, S/N per wavelength element.

## 5. Spectrograph

- Diffraction grating equation: `m λ = d (sin α + sin β)`; angular and linear dispersion, resolving power `R = λ/Δλ`, effect of the slit width.
- Echelle format, order overlap, blaze function.
- Instrument profile; line spread function.
- Calibration lamps and wavelength solution [../05_data_science/03_photometry_astrometry_spectroscopy.md](../05_data_science/03_photometry_astrometry_spectroscopy.md).

## 6. Observation planning

- Target visibility: altitude-time curve, airmass limit, meridian transit, twilight limits.
- Moon constraint: angular distance and illumination fraction.
- Constraint-based scheduling: priority, time window, slew time.
- Calibration plan: bias, dark, flat field (twilight or dome), standard stars (covering the airmass range), focus sequence.

Oracle: `astroplan` [Morris2018]. According to its PyPI entry, its latest release is dated August 2024; its maintenance status should be monitored.

## Test cases

- The synthetic image generator produces frames with a known gain and read noise; the photon transfer curve returns these values.
- The CCD equation reduces to the limits of the three regimes.
- The airmass formulas agree with `sec z` to one part in a thousand for `z < 60°`.
- The predicted zero point is compared with the published value of a real observatory.
