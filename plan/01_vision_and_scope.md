# Vision and scope

## Purpose

To gather astronomy and astrophysics calculations, from the reduction of observational data to the estimation of cosmological parameters, in a Python library that we write ourselves. The distinguishing feature of the library is traceability: every function cites the published source of the equation it implements, with the equation number, and is tested against an independent implementation.

Mature libraries already do most of these calculations ([02_field_survey/01_software_ecosystem.md](02_field_survey/01_software_ecosystem.md)). There are three reasons for rewriting them:

1. **Understanding.** A person who writes every step of a reduction chain or a distance calculation knows how sensitive the result is to each assumption.
2. **Traceability.** In a ready-made library, finding out which equation, which constant and which approximation a number came from means digging through the source code. Here this information is part of the function's interface.
3. **Verified reproduction.** Mature libraries are used as test oracles; the agreement of two independent implementations is stronger evidence than the result of one.

## Scope

Four layers:

- **Calculation infrastructure:** constants, units, angles, time scales, coordinate transforms, numerical methods, statistics.
- **Observing:** planning, detector and atmosphere model, image reduction, photometry, astrometry, spectroscopy, time series. "Ground observation" is interpreted as ground-based (with a telescope, from the Earth's surface) astronomical observation; observing the Earth from a satellite (remote sensing) is not in scope.
- **Astrophysical modeling:** radiation, orbital mechanics, stellar structure and populations, galaxy dynamics, cosmology.
- **End-to-end analysis:** reproduction of published results from open data; the first target is the Hubble diagram and the Hubble constant.

## Out of scope

- Rewriting large codes: stellar evolution (MESA), Boltzmann solvers (CAMB, CLASS), hydrodynamic and cosmological simulation, radiative transfer codes, supernova light-curve models (SALT). These are used, or their outputs are read.
- General machine learning algorithms; `scikit-learn` is used.
- Telescope and camera control, real-time operation.
- A graphical user interface.

The criterion for the boundary: an algorithm is in scope if it can be written in a few hundred lines from published equations and tested against an independent implementation.

## Success criteria

- Every public function has a source, a validity range and a verification test; this rule is checked by machine.
- Every module agrees with an oracle within a documented tolerance.
- Every phase closes with a project that reproduces a published result.
- An analysis re-run from its run record gives the same result.

## Target user

The person who writes the library and the person who uses it are the same: a researcher who observes, reduces the data themselves and wants to compare the result with published values. The code must be readable enough to teach and correct enough to give scientific results; when the two conflict, correctness comes first, and readability is provided by the docstrings.

## Open decisions

- **Package name:** `kozmos` (free on PyPI).
- **Identifier language:** identifiers are English snake_case with unit suffixes (`luminosity_distance_mpc`); the same names are used in the plan documents. The first version was written with Turkish ASCII identifiers and was later renamed to English.
- **License:** the repository is published with all rights reserved (`LICENSE`). The time, angle and coordinate functions follow the algorithms in the ERFA source code; the copyright and licence notice of ERFA is in `THIRD_PARTY_NOTICES.md`, and the obligations of the derived parts must be assessed by the owner of the repository. The licence of SOFA itself has not been read in this work; nothing was transferred directly from the SOFA source code.
- **Acceleration:** Numba or JAX when NumPy is not enough; the decision is made by measurement.
