# Testing and verification

A function counts as verified when it has passed four independent tests. The tests catch different kinds of errors; one does not take the place of another.

## The four tests

**1. Worked example of the source (`tests/unit`).** If the source has a numerical example, the test reproduces that example and cites the source with its page. The numbered examples in Meeus's book [Meeus1998], the example outputs in the SOFA cookbooks [SOFA2023] and the figures of Hogg [Hogg1999] are the main sources of this kind. The error it catches: the equation was transcribed wrongly.

**2. Oracle comparison (`tests/oracle`).** The same input is given to an independent implementation, and the difference must be below the tolerance. The oracles: ERFA and Astropy for coordinates and time, Skyfield and JPL Horizons for ephemerides, `astropy.cosmology` for cosmology, photutils and sep for photometry, `astropy.timeseries` for the periodogram, emcee for sampling. The inputs are random points with a fixed seed that scan the domain, and hand-picked boundary cases. The error it catches: deviation in regions the example does not show.

An oracle can also be wrong. When a difference appears, one first goes back to the source; if the oracle uses a different model or constant, the difference is documented and the tolerance is set accordingly.

**3. Property test (`tests/property`).** Invariants are tested on inputs generated with `hypothesis`:

- round trip: `equatorial_to_galactic(galactic_to_equatorial(x)) = x`
- conservation: rotation matrices are orthogonal and have determinant +1; in the two-body integration, energy and angular momentum are conserved
- limit: for `z → 0`, `D_L → cz/H0`; for `e → 0`, the eccentric anomaly goes to the mean anomaly
- symmetry and monotonicity: airmass increases with zenith angle; comoving distance increases with `z`
- dimension: the unit of the output is the unit expected from the units of the inputs

The error it catches: boundary cases such as the pole, the zero crossing and the 0/2π wrap.

**4. End-to-end science test (`projects`).** An analysis reproduces a known result: the published `H0` and `Ωm` range from the Pantheon+ data, the period and transit depth of a known exoplanet, the published distance modulus of an open cluster. The error it catches: individually correct functions combined wrongly.

## Tolerance

The tolerance is written separately for each test, and its rationale stays in a comment line. Starting values:

- Closed-form transforms (spherical trigonometry, unit conversion): relative `1e-12`.
- Models based on series expansions (precession, nutation): `1e-10` rad if the same model as the oracle is used; for a different model, the model difference reported by the sources.
- Numerical integrals and root finding: ten times the requested tolerance.
- Algorithms that work with noisy data (photometry, periodogram): on synthetic data against the truth, as a multiple of the expected statistical uncertainty.

These values are a starting proposal; when the first tests are written they are set again from the measured difference, and the change is recorded.

## Synthetic data

Next to every module that processes observational data there is a generator: an artificial CCD image with known parameters (star field, PSF, sky background, Poisson and read noise, bias, dark current, flat field, cosmic rays), an artificial spectrum and an artificial light curve. An algorithm is tested first on synthetic data where the truth is known, and then on real data. Injection-recovery experiments (adding a known signal to real data and finding it again) are the standard tool for measuring completeness and bias.

## Reference data

Small files under version control are kept in `data/reference`: oracle outputs (with the header stating which version of which library produced them), example tables transferred from sources, and small cutouts of real data. Large data is downloaded and kept under `data/raw`, outside version control, recorded with its SHA-256 digest.

## Continuous checking

On every change: `ruff`, `mypy --strict`, the unit and property tests, the source rule test. The oracle tests need the `verification` dependencies, so they are marked separately (`pytest -m oracle`) and run at least before merging. Tests that need the network are separated with `-m network` and do not run in the default run.
