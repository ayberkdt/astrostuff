# Roadmap

The phases are in dependency order: each phase rests only on the ones before it and closes with an acceptance project. No durations are given; progress is measured by the acceptance criteria. The projects are defined under [08_example_projects](08_example_projects/).

## Phase 0: Skeleton (done)

- Git repository, `pyproject.toml`, `src/kozmos` layout, `ruff`, `mypy`, `pytest`, `hypothesis`.
- `_sources.py`: the `@source` decorator, the registry, the bibliography reader.
- `tests/test_source_rule.py`: machine check of the source rule.
- The `bibliography/` folder is at the repository root; `verify.py` is tied to continuous checking.
- Decision: the language of identifiers (Turkish ASCII or English).

Acceptance: an example function that cites no source makes the test fail; one that cites a source passes. This behavior was observed during development: the `all_constants` function, left without a source, made the rule test fail, and the justified infrastructure exemption was added for this reason.

## Phase 1: Foundation (partial)

Written: `core/constants.py`, `core/angles.py`, `statistics/fitting.py`, `statistics/robust.py`. Remaining: `core/vectors.py`, `core/numerical/`, uncertainty propagation, the Hubble 1929 table.

- `core/constants.py` [CODATA2022; Prsa2016], `core/units.py`, `core/angles.py`, `core/vectors.py`.
- `core/numerical/`: root finding, integrals, ODE solvers, interpolation.
- `statistics/`: distributions, robust statistics, linear fitting, uncertainty propagation.
- `io/table.py`.

Acceptance: Hubble project Step 0 [Hubble1929].

## Phase 2: Time and coordinates (partial)

Written: `time/julian.py` (calendar, ERA, GMST, obliquity), `time/scales.py` (TAI − UTC), `coordinates/transforms.py` (galactic, horizontal), `ephemeris/kepler.py`. Remaining: the TDB transform, precession and nutation, refraction, Sun and Moon positions, the planner.

- `time/`: Julian date, scale transforms, sidereal time.
- `coordinates/`: spherical transforms, Level 1 precession and nutation, horizontal transform, refraction.
- `ephemeris/`: Kepler solver, elements, Level 1 Sun, Moon and planets.
- `observing/planning.py`, `observing/airmass.py`.

Acceptance: P1 (observing night planner). Agreement with ERFA within the Level 1 accuracy limits.

## Phase 3: Cosmology and inference (done)

Written: `cosmology/distances.py`, `cosmology/hubble_diagram.py`, `statistics/mcmc.py`, `archives/pantheon.py`, `projects/hubble/hubble_analysis.py`. The analysis gave Ωm = 0.332 ± 0.018 and H0 = 73.54 ± 1.02 km s⁻¹ Mpc⁻¹; the published values are 0.334 ± 0.018 and 73.6 ± 1.1 [Brout2022]. Record: `projects/hubble/result.json`.

- `cosmology/friedmann.py`, `cosmology/distances.py`.
- `statistics/mcmc.py`: Metropolis-Hastings, ensemble sampler, diagnostics.
- `archives/pantheon.py`, `archives/cache.py`.

Acceptance: Hubble project Steps 1-3. This phase gives the project's first scientific result and was moved forward because it needs no detector or image processing.

## Phase 4: Radiation and detector model (started)

Written: `radiation/magnitudes.py`, `radiation/infrared.py` (2MASS and WISE zero points), `observing/detector.py` (signal model, synthetic raw frame, star field, CCD equation), `observing/planning.py` (visibility, meridian transit), `io/fits.py` (primary image unit). Remaining: filters and synthetic photometry, extinction, FITS extensions.

- `radiation/`: blackbody, magnitude systems, filters, synthetic photometry, extinction.
- `observing/detector.py`, `observing/signal_to_noise.py`; synthetic image generator.
- `io/fits.py`: reader and writer.

Acceptance: synthetic photometry of Vega and the Sun; generation of a synthetic CCD frame.

## Phase 5: Image reduction and photometry (started)

Written: `reduction/calibration.py` (frame combining, bias/dark/flat-field correction, photon transfer), `photometry/` (`background.py`, `detection.py`, `aperture.py`) (background map, source detection, aperture photometry), `archives/tap.py`, `archives/gaia.py`, `archives/skyview.py`, `visualization/stretch.py`. The Gaia HR diagram (P4) and the optical and infrared comparison were produced. Added later: `reduction/cosmic_rays.py` (L.A.Cosmic), `reduction/sky.py` (static and dynamic sky model), source deblending (`detect_sources(..., deblend_levels=)`), `astrometry/wcs.py` (TAN projection), `astrometry/matching.py` (catalog cross-matching), `archives/twomass.py`, `archives/heasarc.py` and FITS image extensions. Detection and astrometry were measured against the catalog on a real 2MASS image (`examples/real_image_validation.py`). Then `astrometry/solve.py` (solution from an approximate WCS with a catalog) was written; cosmic-ray cleaning was tested on a real HST STIS frame, and sky subtraction and stacking on a real ESO SOFI sequence (`examples/cosmic_rays_real_data.py`, `examples/infrared_reduction.py`). Then the blind astrometric solution by triangle matching (`astrometry/triangles.py`) and the dome flat were added; the tests were extended to a second STIS field and a second SOFI sequence. Remaining: PSF photometry, an indexed (all-sky) astrometric solver, writing photometric calibration as a library function. Details: [05_data_science/06_optical_and_infrared_observing.md](05_data_science/06_optical_and_infrared_observing.md).

- `reduction/`: calibration, cosmic rays, background, combining.
- `photometry/`: source detection, aperture and PSF photometry, calibration.
- `astrometry/`: TAN projection, catalog matching, WCS fitting.
- `archives/tap.py`, `archives/gaia.py`, `io/votable.py`.

Acceptance: P2 (reduction pipeline), P3 (open cluster), P4 (Gaia HR diagram).

## Phase 6: Time series (started)

Written: `timeseries/periodogram.py` (classical and floating-mean Lomb-Scargle).

- `timeseries/`: periodogram, folding, detrending, BLS and TLS.
- Transit model; barycentric correction (raising the `time` module to Level 2).
- MAST access.

Acceptance: P5 (exoplanet transit), P6 (Cepheid period-luminosity relation).

## Phase 7: Spectroscopy

- `spectroscopy/`: trace finding, optimal extraction, wavelength and flux calibration, line measurement, redshift.
- SDSS access.

Acceptance: P7 (redshift measurement).

## Phase 8: High accuracy and dynamics

- `coordinates/` Levels 2 and 3: IAU 2006/2000A, the full reduction chain.
- `ephemeris/`: DE440 reader, N-body integrators, orbit determination.
- `galaxies/`: potentials, orbits, rotation curves.

Acceptance: P8, P9, P10. Agreement with ERFA to `1e-10` rad.

## Phase 9: Stellar astrophysics (started)

Written: `stars/` (polytropes, scaling relations, Chandrasekhar mass, initial mass functions), `star_formation/` (Jeans, Bonnor-Ebert, virial parameter, Kennicutt-Schmidt, Toomre Q) and `black_holes/` (Schwarzschild and Kerr radii, accretion, Hawking radiation, merging binaries). Details: [04_physics/08_black_holes_stars_and_star_formation.md](04_physics/08_black_holes_stars_and_star_formation.md).

- `stars/`: polytropes, simple structure model, reading and fitting isochrones, mass function.

Acceptance: P11; the likelihood-based version of the isochrone fit of P3.

## Phase 10: Advanced analyses

- Hubble project Steps 4-5: selection effects, combined probes.
- Nested sampling, hierarchical models, Gaussian processes.
- Image differencing and transients; gravitational waves; large-scale structure.

A field survey of its own is made before each topic is started.

## Rules between phases

- A phase does not close until its acceptance project reproduces a published result.
- Every module passes four tests ([06_architecture/03_testing_and_verification.md](06_architecture/03_testing_and_verification.md)).
- Before a function is written, the relevant section of the cited source is read ([02_field_survey/04_limits_of_the_survey.md](02_field_survey/04_limits_of_the_survey.md)).
- A new source is first added to `references.bib` and tested with `verify.py`.
- Scope growth is written to the advanced-phase list; it is not added to the running phase.

## First steps

The first four steps (the language of identifiers, the skeleton, reading the sources, transferring the constants) are done. The next ones:

1. Done: every source record is now in the `read` state (the list is in [STATUS.md](STATUS.md)). The last pending records (the CCD signal-to-noise equation, Kerr ISCO energy, Salpeter slope, nuclear timescale) were closed by reading the original papers and open lecture notes.
2. Transferring the Hubble 1929 table from the paper (Step 0).
3. The remainder of Phase 1: numerical methods, robust statistics, uncertainty propagation.
4. The remainder of Phase 2: UTC and TDB, precession and nutation, the observing planner (P1).
5. Done as far as open data allows: the `H0` uncertainty was checked with a closed-form solution and against the four released chains, which all give ±1.0; the published ±1.1 comes from a PolyChord run that was not released (see the README).
