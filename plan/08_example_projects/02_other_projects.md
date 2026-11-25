# Other end-to-end projects

Each project is an acceptance test of a group of modules: it reproduces a known result from open data or from our own observations. The order follows the phases in the roadmap.

## P1. Observing night planner

- Input: observer location, date, target list.
- Output: twilight times, altitude and airmass curves of the targets, position and distance of the Moon, a proposed observing order.
- Modules: `time`, `coordinates`, `ephemeris` (Level 1), `observing/planning`.
- Acceptance: rise-set and meridian transit times agree with `astroplan` and Skyfield within 1 minute [Morris2018; Meeus1998].

## P2. Synthetic CCD and reduction pipeline

- Generating a raw frame with known parameters; a reduced image from bias, dark, flat-field and science frames.
- Modules: `observing/detector`, `reduction`, `io/fits`.
- Acceptance: the residuals are consistent with the noise model; the photon transfer curve returns the input gain and read noise; on real raw frames the result agrees with the `ccdproc` output at the pixel level [Howell2006; Janesick2007].

## P3. Open cluster color-magnitude diagram

- Cluster images in two bands (our own observation or from the archive), photometry, astrometric and photometric calibration with Gaia, membership by proper motion and parallax, isochrone fitting.
- Modules: `photometry`, `astrometry`, `archives/gaia`, `stars/isochrones`, `statistics`.
- Acceptance: the distance modulus, reddening and age agree with published values within the uncertainty; the distance is consistent with the mean Gaia parallax [Vallenari2023; Choi2016; Lindegren2021].

## P4. Gaia HR diagram and the local stellar population

- Selection of stars within 100 pc with ADQL, quality cuts, absolute magnitude and color; main sequence, giant branch and white dwarf sequence.
- Modules: `archives/tap`, `archives/gaia`, `stars/hr`.
- Acceptance: the structure of the HR diagram of the Gaia Collaboration is obtained again; the difference between `1/ϖ` and the posterior distance estimate is shown against the relative parallax error [BailerJones2015; Luri2018].

## P5. Exoplanet transit

- Downloading a TESS light curve, detrending, period search with BLS and TLS, transit model fitting, parameter uncertainties with MCMC.
- Modules: `archives` (MAST), `timeseries`, `ephemeris` (transit model), `statistics/mcmc`.
- Acceptance: the period, depth and duration agree with the NASA Exoplanet Archive values within the uncertainty; an injection-recovery map is produced [Kovacs2002; HippkeHeller2019; MandelAgol2002; Ricker2015].
- Extension: ground-based observation and differential photometry of the same transit.

## P6. Cepheid period-luminosity relation

- Light curves of Large Magellanic Cloud Cepheids (OGLE or Gaia), period with Lomb-Scargle, mean magnitude, Wesenheit magnitude, linear fit with errors on two axes.
- Modules: `timeseries/periodogram`, `radiation/extinction`, `statistics/fitting`.
- Acceptance: the slope and scatter agree with the published relation; the relation is applied to another galaxy and the distance modulus is estimated [LeavittPickering1912; Madore1982; Kelly2007]. This project connects to the distance ladder step of the Hubble project.

## P7. Redshift from a galaxy spectrum

- Downloading SDSS spectra, continuum normalization, detection of emission and absorption lines, template cross-correlation.
- Modules: `archives`, `spectroscopy`.
- Acceptance: the measured redshifts agree with the SDSS catalog values within the uncertainty [TonryDavis1979; York2000].
- Extension: reducing our own long-slit spectrum from the raw frame [Horne1986].

## P8. Galaxy rotation curve

- Decomposing a published rotation curve into disk, gas and NFW halo components; the degeneracy between the mass-to-light ratio and the halo parameters.
- Modules: `galaxies/potentials`, `galaxies/rotation_curves`, `statistics/mcmc`.
- Acceptance: the halo parameters are consistent with the published fit; a model without dark matter is incompatible with the data [NFW1997; BinneyTremaine2008].

## P9. Minor body astrometry and orbit

- Astrometry from a few nights of observation of an asteroid, an initial orbit with Gauss's method, differential correction.
- Modules: `astrometry`, `ephemeris`.
- Acceptance: the elements agree with the JPL values; the predicted position is verified with a later observation [Vallado2022; Lang2010].

## P10. Solar system integration

- Symplectic integration of the planets from DE440 initial conditions; the perihelion advance of Mercury (Newtonian and post-Newtonian).
- Modules: `ephemeris`, `core/numerical`.
- Acceptance: 100-year positions agree with DE440; the energy error stays bounded; agreement with REBOUND for the same initial conditions [Park2021; ReinTamayo2015; WisdomHolman1991].

## P11. Stellar structure: polytrope and a simple main-sequence model

- Lane-Emden solutions; solving a main-sequence star of homogeneous composition by the shooting method; mass-luminosity and mass-radius relations.
- Modules: `stars/structure`.
- Acceptance: the Lane-Emden closed-form solutions; the central conditions of a solar-mass model agree with textbook values in order of magnitude; the trends are consistent with MIST tracks [Kippenhahn2012; Choi2016].

## Advanced-phase candidates

Finding a gravitational-wave event with a matched filter (the open data of GW150914) [Abbott2016]; transient search by image differencing [Zackay2016]; spectral energy distribution fitting; the galaxy two-point correlation function and the BAO peak [Eisenstein2005]; `H0` from lens time delays. A field survey of its own is made before each is started.
