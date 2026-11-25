# Image reduction pipeline

Target module: `reduction/`. Physical model: [../04_physics/06_detector_and_atmosphere.md](../04_physics/06_detector_and_atmosphere.md).

Raw detector output is converted into a science image in the order below. Every step is a function: it takes data, uncertainty and mask, and returns the same triple together with a processing log.

## 1. Data model

- Image container: `data` (2D array), `variance`, `mask` (bit field: bad pixel, saturated, cosmic ray, edge, interpolated), `header`, `history`.
- The unit is tracked: ADU, electrons, electrons/second. The Poisson variance is not computed before the gain is applied.
- Variance propagation is written explicitly at every step: variances add in subtraction, relative variances add in division [Howell2006, Section 4].

## 2. Preparing the calibration frames

- **Master bias:** sigma-clipped mean or median of many zero-length frames. If there is an overscan region, the row-by-row level correction is done first.
- **Master dark:** combination of bias-subtracted dark frames; scaling to the exposure time is valid only if the dark current is linear. The hot pixel map comes from here.
- **Master flat field:** per filter; combination of bias- and dark-subtracted flat frames, each normalized by its own median. In twilight flats, stars are removed by dithering and the median.
- **Bad pixel mask:** from flat field ratios (two different illumination levels) and from dark frames.
- Combination methods: mean, median, sigma-clipped mean, rejection of the lowest/highest values; robust scale estimation [Beers1990]. For Gaussian noise the variance of the median is about `π/2` times that of the mean; the choice is made accordingly.
- The calibration quality is measured: the noise of the master frames adds to the noise of the science frame; the number of frames needed is computed from this.

## 3. Reducing the science frame

```
reduced = ( raw − overscan − bias − dark · (t/t_d) ) / flat_field
```

Order:

1. Overscan correction and trimming.
2. Linearity correction (if the instrument requires it) and masking of saturated pixels.
3. Bias subtraction.
4. Dark subtraction.
5. Applying the gain (conversion to electrons) and building the variance plane: `σ² = signal + read_noise²`.
6. Division by the flat field.
7. Fringe correction (in thin detectors in red bands).
8. Applying the bad pixel mask.

Measuring gain and read noise with the photon transfer curve is a separate tool and produces the instrument definition file [Janesick2007].

## 4. Cosmic ray cleaning

- **In a single frame:** Laplacian edge detection. The image is subsampled and convolved with the Laplacian kernel; pixels that are significant according to the noise model and sharp relative to their surroundings are flagged; they are separated from stars with a "fine structure" image [vanDokkum2001]. Oracle: `astroscrappy`.
- **In many frames:** sigma-clipped combination of the aligned frames; it is more reliable.
- Satellite and aircraft trails: Hough transform or linear structure detection; masking.

The cleaned pixels are interpolated and flagged in the mask; photometry knows these pixels when weighting.

## 5. Background estimation

- Global: sigma-clipped median or mode estimate (`mode ≈ 2.5 · median − 1.5 · mean`) [BertinArnouts1996].
- Local map: the image is divided into grid cells, a clipped estimate is made in each cell, followed by a median filter across cells and spline interpolation [BertinArnouts1996]. Oracles: `sep` [Barbary2016] and `photutils`.
- Source masking and iteration: sources are detected, masked, and the background is estimated again.
- The background noise map (RMS) is an input of the detection threshold and of the photometric uncertainty.
- Choice of cell size: larger than the sources, smaller than the scale of background variation. For extended sources (galaxies, nebulae) the risk of oversubtraction is documented.

## 6. Alignment and combination

- Shift estimation: star matching (triangle or asterism invariants), phase correlation, WCS-based.
- Resampling: interpolation kernels and flux conservation; noise correlation arises and is reflected in the variance map.
- Combination: weighted mean (`1/σ²` or S/N² weights), sigma clipping, equalizing scale and zero point.
- Reconstruction of undersampled data from dithered frames (drizzle) [FruchterHook2002].
- Image differencing: PSF matching kernel [AlardLupton1998] and the statistically optimal difference image [Zackay2016]. It is the basis of transient searches; advanced stage.

## 7. Quality control

An automatic report for every night: the evolution of the bias level and the read noise over time, flat field uniformity, seeing (FWHM) and ellipticity, sky brightness, zero point and transparency, limiting magnitude, astrometric residuals. Frames outside the threshold are flagged, not silently discarded.

## 8. Validation with synthetic data

The generator produces a raw frame with known parameters: star field (positions, fluxes), PSF, sky, Poisson noise, flat field pattern, dark current and hot pixels, bias structure, read noise, cosmic rays, gain. The reduction pipeline processes this frame; the result is compared with the known truth:

- The residual of the reduced image follows the distribution predicted by the noise model (reduced `χ²` about 1).
- The completeness and false positive rate of the cosmic ray detection are measured.
- The variance plane predicts the real scatter.

On real data the second test is to reduce the same raw frames with `ccdproc` and compare at the pixel level.
