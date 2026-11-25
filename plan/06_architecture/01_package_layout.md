# Package layout

The working name of the package is `kozmos`. The name was queried on PyPI on 25 November 2026 and found free (`https://pypi.org/pypi/kozmos/json` → 404); the names `astrokit` and `astrolab` are taken. The name is a suggestion and can be changed before the first code is written.

## Design principles

1. **The core consists of pure functions.** A calculation function does not modify its input, read files, reach the network or look at global state. Input/output and network access are in separate layers (`kozmos.io`, `kozmos.archives`).
2. **The only run-time dependencies are NumPy and SciPy.** Libraries such as Astropy, ERFA, Skyfield and photutils are used not to do calculations but as test oracles, and sit in the optional dependency group called `verification`. The aim is to rewrite the algorithms; a function that wraps `astropy` does not meet this aim.
3. **Every public function is tied to a source.** The rule and its checking are in [02_code_standards_and_source_citation.md](02_code_standards_and_source_citation.md).
4. **Units are visible in names and types.** Core functions take `float`/`ndarray` in SI or an explicitly named astronomical unit; the argument name carries the unit (`distance_pc`, `angle_rad`, `wavelength_m`). In the user-facing layer a light `Quantity` type does the unit conversion. Rationale: a unit object is slow in an inner loop, and an unnamed `float` is a source of errors; a unit embedded in the argument name is the middle way.
5. **Vectorization is the default.** Every function gives the same result for scalar and array input; the broadcasting rules are those of NumPy.
6. **Layers depend downward.** A module imports only modules of lower layers; circular imports are prevented by a test.

## Layers

```
Layer 0  core          constants, units, angle and vector helpers, numerical methods
Layer 1  time          time scales, Julian date, sidereal time
         coordinates   spherical transforms, precession-nutation, aberration, refraction
Layer 2  ephemeris     Kepler problem, orbital elements, planet positions
         radiation     blackbody, magnitude systems, filters, extinction
Layer 3  observing     planning, airmass, S/N, detector model
         reduction     bias/dark/flat, cosmic rays, background
         photometry    aperture and PSF photometry, calibration
         astrometry    source matching, WCS solution
         spectroscopy  extraction, wavelength calibration, line measurement
         timeseries    periodogram, folding, transit search
Layer 4  stars         stellar structure, HR diagram, isochrones
         galaxies      potentials, rotation curves, light profiles
         cosmology     Friedmann equations, distances, Hubble diagram
         statistics    likelihood, MCMC, fitting, uncertainty propagation
Layer 5  archives      TAP/ADQL, catalog and data download (network)
         io            FITS, VOTable, table reading and writing (disk)
         projects      end-to-end analyses
```

Although `statistics` is shown in Layer 4, it depends only on Layer 0, and every field module can use it.

## Directory layout

```
Astronomy-and-Astrophysics/
├── plan/                     this folder
├── src/kozmos/
│   ├── core/                 constants.py, units.py, angles.py, vectors.py, numerical/
│   ├── time/                 julian.py, scales.py, sidereal.py
│   ├── coordinates/          spherical.py, frames.py, precession.py, nutation.py,
│   │                         aberration.py, refraction.py, parallax.py
│   ├── ephemeris/            kepler.py, elements.py, sun.py, moon.py, planets.py
│   ├── radiation/            blackbody.py, magnitudes.py, filters.py, extinction.py
│   ├── observing/            planning.py, airmass.py, signal_to_noise.py, detector.py
│   ├── reduction/            calibration.py, cosmic_rays.py, background.py, combining.py
│   ├── photometry/           aperture.py, psf.py, source_detection.py, calibration.py
│   ├── astrometry/           matching.py, wcs.py, projections.py
│   ├── spectroscopy/         extraction.py, wavelength.py, lines.py, redshift.py
│   ├── timeseries/           periodogram.py, folding.py, transits.py
│   ├── stars/                structure.py, hr.py, isochrones.py, imf.py
│   ├── galaxies/             potentials.py, rotation_curves.py, profiles.py
│   ├── cosmology/            friedmann.py, distances.py, hubble_diagram.py, standard_candles.py
│   ├── statistics/           likelihood.py, mcmc.py, fitting.py, robust.py, propagation.py
│   ├── archives/             tap.py, gaia.py, pantheon.py, cache.py
│   ├── io/                   fits.py, votable.py, table.py
│   └── _sources.py           source-citation infrastructure
├── tests/
│   ├── unit/                 against the worked examples of the source
│   ├── oracle/               against the output of astropy/erfa/skyfield
│   └── property/             invariants with hypothesis
├── projects/                 hubble_diagram/, hr_diagram/, transit_curve/ ...
├── data/                     raw/ (outside version control), reference/ (small, under version control)
├── bibliography/             references.bib and the verification tools
└── pyproject.toml
```

Module and function names are English, in snake_case, and the unit is part of the name when the quantity has one. The first version was written with Turkish names reduced to ASCII, with no Turkish letters, and was later renamed to English.

## Data types

The core types are immutable data classes (`@dataclass(frozen=True, slots=True)`):

- `Time`: the Julian date held in two `float64` values (`jd1`, `jd2`) and a scale label (`"UTC"`, `"TT"`, `"TDB"` ...). The two-part representation brings the resolution of a single `float64`, about 40 microseconds near J2000, below a nanosecond; this is the representation SOFA uses [SOFA2023].
- `SkyPosition`: a frame label (`"ICRS"`, `"FK5"`, `"galactic"`, `"ecliptic"`), two angles (radians), and optional distance, proper motion and radial velocity.
- `Observer`: geodetic latitude, longitude, height and the reference ellipsoid.
- `OrbitalElements`: `a, e, i, Ω, ω, M0`, the epoch and the `GM` of the central body.
- `Cosmology`: `H0`, `Ωm`, `ΩΛ`, `Ωr`, `Ωk`, `w0`, `wa`.
- `Measurement`: value, uncertainty and an optional covariance pointer.

For images, spectra and light curves, the containers carry the data, the mask, the uncertainty and the metadata (header, processing history) together; every reduction step returns a new container and adds a record to the history.

## Dependency groups

- Core: `numpy`, `scipy`.
- `verification`: `astropy`, `pyerfa`, `skyfield`, `photutils`, `sep`, `emcee` and the other oracles.
- `dev`: `pytest`, `hypothesis`, `ruff`, `mypy`.
- `archives`: `httpx` or the standard library for network access; the decision is made when that module is written.

The version numbers are recorded, with the date they were read, in the inventory under [../02_field_survey](../02_field_survey/).
