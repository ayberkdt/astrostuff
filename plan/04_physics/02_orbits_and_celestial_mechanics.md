# Orbits and celestial mechanics

Target module: `ephemeris/`.

The equations are at planning level; they are read from the cited source before they are coded.

## 1. Two-body problem

- Relative equation of motion: `r̈ = −μ r / r³`, `μ = G(m1 + m2)`.
- Conserved quantities: specific energy `ε = v²/2 − μ/r`, angular momentum `h = r × v`, eccentricity (Laplace-Runge-Lenz) vector [MurrayDermott1999, Section 2].
- Orbit equation: `r = a(1 − e²) / (1 + e cos f)`; vis-viva: `v² = μ(2/r − 1/a)`.
- Kepler's third law: `P² = 4π² a³ / μ`.
- The product `GM` is measured much more precisely than `G` and `M` separately, so the constants file keeps nominal `GM` values [Prsa2016].

## 2. Anomalies and Kepler's equation

- Mean anomaly `M = n(t − t0)`, eccentric anomaly `E`, true anomaly `f`.
- Kepler's equation `M = E − e sin E`; solvers are in [../03_mathematics/03_numerical_methods.md](../03_mathematics/03_numerical_methods.md) [Markley1995].
- `E → f`: `tan(f/2) = sqrt((1+e)/(1−e)) tan(E/2)`; the `atan2` form is preferred.
- Parabolic (Barker's equation) and hyperbolic orbits; universal variable form [Vallado2022, Section 2].

## 3. Orbital elements and state vector

- Classical elements: `a, e, i, Ω, ω, M0` and the epoch.
- Conversion between elements and state vector (`r, v`); rotation `R3(−Ω) R1(−i) R3(−ω)` from the perifocal frame [Vallado2022, Section 2].
- Singularities: for `e = 0` and `i = 0` some angles are undefined; equinoctial elements remove these cases.
- Distinction between mean and instantaneous (osculating) elements.

## 4. Positions of the Sun, Moon and planets

Three accuracy levels:

- **Level 1:** truncated series. About 0.01° for the Sun, about 10″ for the Moon, mean elements for the planets [Meeus1998, Section 25, 31-33, 47]. Sufficient for planning and for rise and set times.
- **Level 2:** term tables of the full analytical theories (VSOP87, ELP) [Bretagnon1988]; the reference entry for the ELP theory will be added.
- **Level 3:** reading the JPL DE440 numerical ephemeris: the SPK file format, evaluation of Chebyshev coefficients [Park2021]. DE440 covers the years 1550-2650; DE441 gives a longer span at lower accuracy. This span was read from the JPL `de440_de441` documentation page during the survey.

Oracles: Skyfield and `jplephem` (agreement close to the bit level is expected with the same DE file), JPL Horizons (an independent service).

## 5. Minor bodies and artificial satellites

- Ephemerides of asteroids and comets: two-body propagation from Minor Planet Center elements and numerical integration with planetary perturbations.
- Magnitude model: the `H, G` system.
- Artificial satellites: the TLE format and the SGP4 propagator [Vallado2022, Section 9]. SGP4 is not rewritten; the `sgp4` package is used, because TLEs are meaningful only with the definition of that model. It is needed to predict satellite trails in images.

## 6. Orbit determination

- Initial orbit from three observations: the Gauss and Laplace methods [Vallado2022, Section 7].
- Differential correction: weighted least squares, state transition matrix.
- Visual and spectroscopic binary orbits: Thiele-Innes constants, radial velocity curve `v_r = K[cos(ω + f) + e cos ω] + γ`, mass function.

## 7. N-body and perturbations

- Direct summation `O(N²)`; softening length.
- Integrators: leapfrog, Wisdom-Holman [WisdomHolman1991; ReinTamayo2015], high-order adaptive [ReinSpiegel2015]. Oracle: REBOUND [ReinLiu2012].
- Restricted three-body problem: Jacobi constant, Lagrange points, Hill sphere, Roche limit [MurrayDermott1999, Section 3].
- Secular perturbations, resonances, tidal evolution (advanced stage) [MurrayDermott1999, Section 6-8].
- Relativistic corrections: perihelion advance; the first post-Newtonian term.

## 8. Exoplanet observables

- Transit geometry: depth `(Rp/R*)²`, duration, impact parameter; stellar density from the light curve [SeagerMallenOrnelas2003; Winn2010].
- Transit light curve model: analytic solution on a limb-darkened disk [MandelAgol2002]. Oracle: `batman` [Kreidberg2015].
- Radial velocity semi-amplitude `K` and minimum mass `m sin i`.
- Transit timing variations (advanced stage).

## Test cases

- Closed-form solution: return to the initial state after one period on circular and elliptical orbits; conservation of energy and angular momentum.
- The worked examples of the sources [Meeus1998; Vallado2022] are reproduced numerically.
- Planet positions are compared with Skyfield + DE440, within the accuracy reported by the source at Level 1 and at the microarcsecond level at Level 3.
- N-body: orbit difference against REBOUND for the same initial conditions; stability of the figure-eight three-body solution.
