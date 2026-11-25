# Software ecosystem

Version and date information was read on 25 November 2026 from the PyPI JSON API with `bibliography/verify.py`; the full list is in [../../bibliography/VERIFICATION_SUMMARY.md](../../bibliography/VERIFICATION_SUMMARY.md). Maintenance status and citation information come from a six-branch web survey; what the survey read by opening the page is marked separately from what it saw only in a search summary. Information that this document attributes to "the survey" is the survey's own report and has not been verified separately.

Each package is given its role in the project:

- **oracle:** an independent implementation of the same computation; compared against in tests.
- **use:** not rewritten, used as a dependency.
- **study:** its design or algorithm is taken as a model.

## 1. Core

- `astropy` 8.0.3 (17 November 2026). The units, constants, coordinates, time, WCS, FITS, table, cosmology, modeling, statistics and time series subpackages ship in a single distribution. Citation: three project papers [Astropy2013; Astropy2018; Astropy2022]. Role: oracle for almost every module.
- `pyerfa` 2.0.1.5 (11 November 2024). Python wrapper of ERFA, the adaptation of the IAU SOFA algorithms. The current SOFA release is dated 11 October 2023 (iausofa.org, read in the survey) [SOFA2023]. Role: primary oracle for time and coordinates; the SOFA cookbooks are an algorithm source.
- `numpy` 2.5.5, `scipy` 1.18.1. Role: runtime dependency [Harris2020; Virtanen2020].

## 2. Archive access

- `astroquery` 0.4.11 (20 September 2025); queries to archives and services [Ginsburg2019]. The stable release has not been renewed for more than a year; according to the survey, development pre-releases continue.
- `pyvo` 1.10.0 (4 November 2026); TAP, SIA, SSA client.
- `astrocut` 1.5.1, `mocpy` 0.21.0, `astropy-healpix` 2.0.1, `healpy` 1.20.1, `lsdb` 0.11.2.

Role: oracle (for `archives/tap.py`) and fallback access path.

## 3. Positional astronomy, ephemerides, orbits

- `skyfield` 1.55 (7 August 2026), `jplephem` 2.24: position computation with the JPL DE ephemerides. Role: ephemeris oracle.
- `spiceypy` 8.4.0 (11 November 2026): wrapper of the NAIF SPICE toolkit. Role: Level 3 oracle.
- `sgp4` 2.27: satellite propagator. Role: use.
- `ephem` 4.2.1: legacy-generation library; releases are still being made.
- `rebound` 5.2.3 (18 November 2026), `reboundx` 5.1.0: N-body integrators [ReinLiu2012; ReinTamayo2015; ReinSpiegel2015]. Role: oracle and study.
- `poliastro` 0.17.0 (10 July 2022): according to the survey, its GitHub repository was archived on 14 October 2023. Its fork `hapsira` 0.18.0 (24 December 2023) has not published a new release on PyPI since that date. Neither counts as a maintained oracle; orbital mechanics tests rely on REBOUND, Skyfield and worked examples from sources.
- `galpy` 1.12.1 (2 November 2026), `gala` 1.12.0: galactic dynamics. Role: oracle for `galaxies`.
- JPL Horizons: independent ephemeris service. DE440/DE441 is the current planetary ephemeris [Park2021]; the survey saw no newer release name on the JPL page.

## 4. Image reduction, photometry, astrometry

- `ccdproc` 2.5.1 (5 July 2025): CCD reduction. According to the survey, the GitHub release title calls it the "last 2.x release"; the status of 3.x has not been verified.
- `astroscrappy` 1.3.0: implementation of L.A.Cosmic [vanDokkum2001].
- `photutils` 3.1.1 (16 October 2026): source detection, aperture and PSF photometry. The major version has moved to 3; interface changes relative to 2.x have not been examined.
- `sep` 1.4.1 (18 February 2025): library form of the SExtractor core [Barbary2016; BertinArnouts1996]. According to the survey, the project has changed hands (the original repository up to 1.2.1, current development under `sep-developers`).
- `astroalign` 2.6.2, `reproject` 0.21.0, `regions` 0.13.
- `twirl` 0.6.0 (26 October 2026): pure-Python astrometric solution with Gaia. `astrometry.net`: blind solver [Lang2010].
- `stdpipe` 0.5.0: transient source pipeline.
- SExtractor, SCAMP, SWarp: according to the survey, the latest releases on the official pages are dated 2006 and 2010; the successor is SourceXtractor++. For Python work, `sep` and `photutils` are preferred.
- IRAF: according to the survey, the community distribution is 2.18.2, and the community page recommends Astropy for new projects.

Role: `ccdproc`, `photutils`, `sep`, `astroscrappy` are oracles; `astrometry.net` and `twirl` are used in the first stage.

## 5. Spectroscopy

- `specutils` 2.4.0, `specreduce` 1.9.0: spectrum containers, extraction and wavelength calibration.
- `pypeit` 3.0.0 (9 November 2026): semi-automated spectral reduction pipeline.
- `synphot` 1.9.0, `speclite` 1.0.0: synthetic photometry and filters.
- `dust-extinction` 1.7, `extinction` 0.4.10, `dustmaps` 1.0.14: extinction laws and dust maps.
- `ppxf` 9.5.3 (24 November 2026): full spectrum fitting.

Role: oracle.

## 6. Time series and exoplanets

- `lightkurve` 2.6.0: Kepler and TESS data.
- `transitleastsquares` 2.3.0 (12 November 2026), `wotan` 2.1.1 (18 November 2026): transit search and detrending [HippkeHeller2019].
- `batman-package` 2.5.3: transit model [Kreidberg2015; MandelAgol2002].
- `exoplanet` 0.6.0 (April 2024), `starry` 1.2.0 (October 2021): no release for a long time.
- `radvel` 1.6.7, `stingray` 2.3.2, `celerite2` 0.3.3, `george` 0.4.4, `tinygp` 0.3.1.

Role: `astropy.timeseries`, `transitleastsquares`, `batman` are oracles.

## 7. Cosmology

- `astropy.cosmology`: distance computations. Role: primary oracle.
- `camb` 2.0.5, `classy` 3.4.3.0: Boltzmann codes. Role: use (input for the power spectrum and the sound horizon).
- `cobaya` 3.6.2, `getdist` 1.7.7: sampling framework and chain analysis.
- `sncosmo` 2.13.2 (23 November 2026): supernova light curve fitting [Guy2007; Kenworthy2021]. Role: use.
- `pyccl` 3.3.6, `colossus` 1.4.0: structure formation and halo computations.

## 8. Statistics and inference

- `emcee` 3.1.6 (19 April 2024) [ForemanMackey2013], `zeus-mcmc` 2.5.4 (January 2023).
- `dynesty` 3.2.0 (10 November 2026) [Speagle2020], `nautilus-sampler` 1.0.6, `ultranest` 4.6.3.
- `pymc` 6.4.0, `numpyro` 0.22.0, `blackjax` 1.7.1, `cmdstanpy` 1.3.2: gradient-based sampling.
- `arviz` 1.3.0, `corner` 2.3.0: diagnostics and visualization.
- `astroML` 1.0.2.post1 (March 2022): companion code of the book by Ivezić and colleagues [Ivezic2020]; no release for a long time, valuable as a source of methods.
- `scikit-learn` 1.9.1, `iminuit` 2.33.0, `lmfit` 1.3.4, `uncertainties` 3.2.3, `sbi` 0.27.0, `jax` 0.12.1.

Role: `emcee` and `dynesty` are oracles; `scikit-learn` is used; NumPyro or PyMC are used in a later stage.

## 9. Astrophysical domain codes

- Stellar: MESA (current release 26.4.1 according to the survey) [Paxton2011], MIST and PARSEC grids [Choi2016; Bressan2012], `fsps` 0.5.0, `astro-prospector` 1.4.1, `bagpipes` 1.4.1.
- Simulation analysis: `yt` 4.4.4, `pynbody` 2.8.0.
- High energy: `gammapy` 2.1, `sherpa` 4.19.0, `naima` 0.11.0.
- Gravitational waves: `gwpy` 4.1.0, `pycbc` 2.11.0, `bilby` 2.8.2 [Abbott2016].
- Pulsar timing: `pint-pulsar` 1.1.8.
- Solar and plasma: `sunpy` 8.0.1, `plasmapy` 2026.2.0.
- Relativity: `einsteinpy` 0.5.0.
- Spectral lines and radiation: `pyneb` 1.1.32, `radis` 0.17.1.
- `tardis-sn`: the 1.0.1 release on PyPI is dated 2015; the project may not be distributed through PyPI, and the current installation path has not been examined.

Role: study for most of the later stages; not rewritten.

## 10. Outside the survey's scope

The survey could not read or did not cover the following; this part of the inventory is incomplete: radio astronomy (CASA, `spectral-cube`, `pyuvdata`), radiative transfer codes (RADMC-3D, Cloudy, SKIRT), atomic and molecular databases (CHIANTI, HITRAN, ExoMol), hydrodynamics codes (GADGET-4, AREPO, Athena++), exoplanet atmosphere codes (petitRADTRANS, PICASO), spectral energy distribution fitting (CIGALE), the LSST Science Pipelines, DRAGONS. These areas must be surveyed separately when they enter the later stages of the roadmap.

## 11. Conclusion

- The Python astronomy ecosystem is mature: for almost every basic computation there is a maintained oracle. The project's contribution is not a new capability but a readable and verified implementation that ties every equation to its source.
- Packages whose maintenance has stopped or is uncertain: `poliastro`, `hapsira`, `astroML`, `starry`, `exoplanet`, `zeus-mcmc`, `astroplan`. Tests that rely on them must be backed by worked examples from sources.
- Packages that changed major version in 2026 (`astropy` 8, `photutils` 3, `pypeit` 3, `pymc` 6, `sunpy` 8): oracle tests are pinned to a specific version, and the version is written into the header of the reference output.
- The SOFA/ERFA pair is the de facto standard of positional astronomy; it is the first place to look, both as an algorithm source and as an oracle.
