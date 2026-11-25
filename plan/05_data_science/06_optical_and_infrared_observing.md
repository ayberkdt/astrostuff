# Optical and infrared observing: implemented pipeline and next steps

Modules: `archives/` (tap, gaia, skyview, cache), `io/fits.py`, `observing/` (detector, planning), `reduction/`, `photometry/`, `visualization/`, `radiation/infrared.py`.

## Implemented pipeline

The chain from archive to image and catalog was run end to end on 25 November 2026:

1. **Access.** The TAP client runs the ADQL query and writes the result to the cache as CSV [TAP2019]; the SkyView client downloads an image cutout at a given position [SkyView2026]. Every download is recorded with its address, SHA-256 digest and time.
2. **Reading.** The FITS reader decodes the primary image unit according to the standard [FITS2018]; it was compared bit by bit with `astropy.io.fits`.
3. **Background.** Clipped mode estimate on a cell grid, median filter and spline interpolation [SExtractorDocs2026].
4. **Detection.** Connected pixel groups above the background; optional Gaussian smoothing.
5. **Measurement.** Circular aperture photometry, local sky from the annulus.
6. **Visualization.** asinh stretch and color-preserving RGB composition [Lupton2004].

On the catalog side, 31 298 stars within 50 pc were downloaded from Gaia DR3 and the HR diagram was built [Babusiaux2018; Vallenari2023].

## Service status

Tried on 25 November 2026, and the results were observed:

- **IRSA TAP** (`irsa.ipac.caltech.edu/TAP/sync`): working; the `gaia_dr3_source` and 2MASS `fp_psc` tables were queried and CSV was returned.
- **NASA SkyView** (`skyview.gsfc.nasa.gov/current/cgi/runquery.pl`): working; DSS2, 2MASS and WISE cutouts were downloaded as FITS.
- **Gaia AIP mirror** (`gaia.aip.de/tap/sync`): working; it returns VOTable.
- **ESA Gaia archive** (`gea.esac.esa.int/tap-server/tap/sync`): returned the landing page (HTML) for the query; the endpoint may have changed.
- **CDS** (TAPVizieR, hips2fits): the connection could not be made or it returned 404; the cause was not investigated.

Bright star catalogs such as Hipparcos could not be downloaded because they reside in VizieR. The Gaia sample does not contain very bright stars; the catalog of named bright stars remained incomplete.

## Optical and infrared comparison

The 0.3-degree field around the Flame Nebula (NGC 2024) was processed in three bands (`examples/optical_infrared.py`). The numbers of sources found with the same detection settings (4σ, at least 6 pixels):

- DSS2 Red (0.64 µm): 17
- 2MASS K (2.16 µm): 132
- WISE 12 µm: 15

The numbers are the output of the same algorithm on three images; the completeness and the false detection rate were not measured in this field. Part of the bright nebular structure is also counted as a "source"; this effect is larger in the optical and 12 µm images. The completeness and positional accuracy of the method were tested on synthetic star fields (`tests/unit/test_photometry.py`).

The dark region in the middle of the 12 µm image consists of the lowest-valued pixels in the image; detector saturation is suspected, but this was not verified from the header or from the WISE documentation.

## Infrared-specific topics

Implemented:

- Vega zero points of the 2MASS J, H, Ks and WISE W1-W4 bands; magnitude-flux conversion and AB offset [Cohen2003; Wright2010].
- The temperature of the thermal source to which the band is most sensitive, from Wien's law [CODATA2022].

To do:

- **Sky subtraction (written).** `reduction/sky.py` builds a static and a dynamic sky model from dithered frames [Schirmer2013]. Measured on a synthetic series: in a crowded field the aperture flux comes out 6.8% too low without an object mask and 0.6% with the mask; rejecting the highest value in the stack shifts the residual in blank sky by one tenth of the noise to the positive side. On a real SOFI Ks series (16 raw frames) the large-scale residual fell from 151 ADU to 0.8 ADU; the record is `figures/infrared_reduction.json`.
- **Thermal background.** Beyond 3 µm the emission of the telescope and the atmosphere dominates; short exposures, fast readout and nodding are needed. The background photon noise can be computed with `observing.detector.signal_to_noise`; sky brightnesses per band must be added.
- **Detector.** Infrared arrays deviate from linearity and use non-destructive reads (ramp sampling); the linearity correction and the ramp fit are separate functions.
- **Extinction.** Estimating the visual extinction from the near-infrared color excess (`H − K`); an extinction law is required [CCM1989; Fitzpatrick1999].
- **Color correction.** The WISE W3 and W4 zero points need a correction of a few percent for red sources [Wright2010].
- **Original data.** The SkyView cutouts are resampled. For photometry, the 2MASS and WISE atlas images or catalogs should be taken from IRSA; the IRSA TAP client is ready.

## To do on the optical side

- PSF photometry [Stetson1987].
- An indexed astrometric solver that searches the whole sky. Written so far: a solution with a catalog from an approximate WCS (`astrometry/solve.py`) [CalabrettaGreisen2002] and, given the catalog of the region, a solution by triangle matching without an initial WCS (`astrometry/triangles.py`) [Groth1986]. The blind solution carries a verification step: on a real SOFI stack, a solution built from a few wrong pairs had matched 17 stars by chance in a dense catalog, and in a sparse field a solution with a collapsed scale had reduced all the sources to a single star.
- Crosstalk and illumination correction in SOFI; decompressing `.fits.Z`. The special dome flat field was implemented following the manual recipe; a point-source filter was added to the photometric zero point (the 2MASS Point Source Catalog also contains compact galaxies).
- Photometric calibration pipeline. The zero point was measured in the example script with 2MASS magnitudes (0.021 mag scatter on 40 stars); it is not a library function.
- For observation planning: nutation, aberration, Sun and Moon position, twilight limits. Precession (`coordinates/precession.py`) and refraction (`observing/refraction.py`) were written [ERFA2023].
- FITS tables and compressed images. Image extensions were written.

## Written on the optical side

- Cosmic ray cleaning: `reduction/cosmic_rays.py`, Laplacian edge detection [vanDokkum2001]. On a synthetic frame the completeness is above 95% and false flagging is below two per thousand (tests/unit/test_cosmic_rays.py).
- Source deblending: multi-threshold tree and saddle criterion, basins with priority-queue flooding [Photutils2026]. `scipy.ndimage.watershed_ift` was not used, because it expects a gradient image and so crosses the saddle on a brightness image (a split of 4 pixels out of 164 was observed).
- Measurement on a real image: in the 2MASS K cutout, 59 of the 61 catalog stars in the range K = 7-11 were found; deblending raised the recovery rate of bright artificial stars from 84% to 96% (figures/real_image_validation.json).

## Pending source records

The source texts have not been opened for source detection [BertinArnouts1996], aperture photometry and the Gaussian PSF model [Howell2006], the CCD signal-to-noise equation [Howell2006; MerlineHowell1995] and the Gaia column definitions [Vallenari2023]. These functions were tested with synthetic data; the records will be updated when the relevant books and papers are read.
