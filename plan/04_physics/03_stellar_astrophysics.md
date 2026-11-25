# Stellar astrophysics

Target module: `stars/`.

The equations are at planning level; they are read from the cited source before they are coded.

## 1. From observables to physical quantities

- Distance from parallax: `d = 1/ϖ` can be used only for a small relative error; in the general case a posterior distribution is needed [BailerJones2015; BailerJones2021]. The Gaia parallax zero-point correction depends on magnitude, color and position [Lindegren2021].
- Luminosity: `L = 4π R² σ T_eff⁴`; bolometric correction.
- Color-temperature relations and spectral type classification [CarrollOstlie2017, Section 8].
- Mass from binaries: visual, spectroscopic and eclipsing binaries [CarrollOstlie2017, Section 7].
- Mass-luminosity relation and main sequence lifetime.

## 2. Hertzsprung-Russell diagram

- Building the color-magnitude diagram from Gaia data: quality cuts, parallax error cut, extinction correction.
- Evolutionary regions: main sequence, subgiant branch, red giant branch and its tip (TRGB), horizontal branch, asymptotic giant branch, white dwarf sequence.
- The TRGB as a standard candle: the constant I-band luminosity of the helium flash [Freedman2019].

## 3. Stellar structure equations

For a spherically symmetric star in hydrostatic equilibrium [Kippenhahn2012, Section 1-10]:

```
dm/dr = 4π r² ρ                       mass continuity
dP/dr = −G m ρ / r²                   hydrostatic equilibrium
dL/dr = 4π r² ρ ε                     energy generation
dT/dr = −3 κ ρ L / (16π a c r² T³)    radiative transport
dT/dr = (1 − 1/γ) (T/P) dP/dr         adiabatic convection
```

Closure relations: the equation of state `P(ρ, T, X)`, the opacity `κ(ρ, T, X)`, the energy generation rate `ε(ρ, T, X)`. The convection criterion is the Schwarzschild criterion.

- **Polytropes:** `P = K ρ^(1+1/n)` and the Lane-Emden equation. Closed-form solutions exist for `n = 0, 1, 5` and are a test of the numerical solver; `n = 1.5` gives a fully convective star, `n = 3` gives the Eddington standard model [Kippenhahn2012, Section 19].
- **Equation of state:** ideal gas, radiation pressure, degenerate electron gas (nonrelativistic and extreme relativistic limits); the Chandrasekhar mass [CarrollOstlie2017, Section 16].
- **Opacity:** Kramers approximation, electron scattering; tabulated opacities at an advanced stage.
- **Nuclear reactions:** pp chain, CNO cycle, triple alpha; power-law approximations for the temperature sensitivity [Kippenhahn2012, Section 18].
- **Simple model:** solving the four equations by the shooting method for a zero-age main sequence star of homogeneous composition. A full evolution code is not written; MESA output is comparison data [Paxton2011].

## 4. Isochrones and stellar populations

- Evolutionary tracks and isochrone grids: MIST [Choi2016] and PARSEC [Bressan2012]. The grids are downloaded, read and interpolated in age-metallicity space.
- Cluster fitting: joint estimation of age, metallicity, distance modulus and reddening on the color-magnitude diagram; the likelihood is built from the distance to the isochrone and the weight of the initial mass function.
- Initial mass function: Salpeter power law [Salpeter1955], piecewise power law [Kroupa2001], log-normal form [Chabrier2003]; normalization and sampling.
- Simple stellar population synthesis: isochrone + mass function + spectral library (advanced stage).

## 5. Variable stars

- Pulsating variables: Cepheids, RR Lyrae, Mira; the instability strip.
- **Leavitt law** (period-luminosity relation): `M = α log10 P + β` [LeavittPickering1912]; the Wesenheit form removes reddening [Madore1982]; the metallicity dependence is one of the contested terms of the distance ladder [Riess2022].
- Eclipsing binaries: light curve geometry, Roche model (advanced stage).
- Period-finding methods are in [../05_data_science/04_time_series.md](../05_data_science/04_time_series.md).

## 6. Stellar atmospheres and spectra

- Gray atmosphere and the Eddington approximation: `T⁴(τ) = (3/4) T_eff⁴ (τ + 2/3)` [CarrollOstlie2017, Section 9].
- Line formation, equivalent width, curve of growth; the basis of abundance analysis.
- Radial velocity: cross-correlation [TonryDavis1979]; barycentric correction [WrightEastman2014].
- Model atmosphere grids and synthetic spectral libraries are not reproduced; they are read and interpolated.

## 7. Final stages

White dwarf mass-radius relation, neutron stars and the Tolman-Oppenheimer-Volkoff equation, supernova types [CarrollOstlie2017, Section 15-16]. The standardization of Type Ia supernovae is in [05_cosmology.md](05_cosmology.md).

## Test cases

- Lane-Emden: the closed-form solutions for `n = 0, 1, 5`; published table values of the first zero and of the central concentration for `n = 3` [Kippenhahn2012, Section 19].
- Order-of-magnitude estimates of hydrostatic equilibrium for the Sun [CarrollOstlie2017, Section 10].
- Mass function: the normalization integral is 1; KS test of the sampled distribution.
- Isochrone reading: recovery of one point of the MIST grid by interpolation.
