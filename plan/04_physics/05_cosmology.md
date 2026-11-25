# Cosmology

Target module: `cosmology/`. End-to-end application: [../08_example_projects/01_hubble_diagram.md](../08_example_projects/01_hubble_diagram.md).

The equations are at planning level; they are read from the cited source before they are coded.

## 1. The expanding universe

- Friedmann-Lemaître-Robertson-Walker metric, scale factor `a(t)`, redshift `1 + z = 1/a`.
- Friedmann equation and the dimensionless Hubble parameter [Hogg1999, Eq. 14; CarrollPressTurner1992]:

  `E(z) = H(z)/H0 = sqrt( Ωr(1+z)⁴ + Ωm(1+z)³ + Ωk(1+z)² + Ω_DE f(z) )`

  `Ωk = 1 − Ωm − Ωr − Ω_DE`. For a cosmological constant `f = 1`; for constant `w`, `f = (1+z)^{3(1+w)}`; for the parametrization `w(a) = w0 + wa(1 − a)`, `f = (1+z)^{3(1+w0+wa)} exp(−3 wa z/(1+z))` [ChevallierPolarski2001; Linder2003].
- Radiation density: photons from the CMB temperature, neutrinos from the effective number of species; massive neutrinos are a separate term and are taken as massless in the first stage.
- Hubble distance `D_H = c/H0` and Hubble time `t_H = 1/H0`.
- Common misinterpretations of redshift, recession velocity and horizon [DavisLineweaver2004].

## 2. Distance measures

All follow the definitions in Hogg's review [Hogg1999]:

- Comoving distance (line of sight): `D_C = D_H ∫₀ᶻ dz′/E(z′)` (Eq. 15)
- Transverse comoving distance `D_M`: `(D_H/√Ωk) sinh(√Ωk D_C/D_H)` for `Ωk > 0`, `D_C` for `Ωk = 0`, the `sin` form for `Ωk < 0` (Eq. 16)
- Angular diameter distance: `D_A = D_M/(1+z)` (Eq. 18)
- Luminosity distance: `D_L = (1+z) D_M` (Eq. 21)
- Distance modulus: `μ = 5 log10(D_L / 10 pc)` (Eq. 25)
- Comoving volume element (Eq. 28), lookback time (Eq. 30), age of the universe
- Angular diameter distance between two redshifts (for lensing, Eq. 19)

The equation numbers were recalled from memory from Hogg's arXiv version and will be verified during coding.

Low-`z` expansion: `D_L ≈ (c z/H0) [1 + (1 − q0) z/2 + …]`; cosmographic expansion with the deceleration parameter `q0` and the jerk `j0` [Riess2022, around Eq. 5]. The convergence of the numerical integral to this expansion in the `z → 0` limit is a property test.

Oracle: `astropy.cosmology`. Independent comparison: the method of Wright's calculator [Wright2006].

## 3. Ways of measuring the Hubble constant

**Local distance ladder.** Three rungs [Riess2022]:

1. Geometric anchors: Gaia parallaxes of Milky Way Cepheids, the eclipsing binary distance of the Large Magellanic Cloud, the maser distance of NGC 4258.
2. Primary indicator: distances to the host galaxies of Type Ia supernovae from the Cepheid period-luminosity relation; alternatively TRGB and JAGB [Freedman2019; Freedman2025].
3. Hubble flow: the magnitude-redshift relation of standardized Type Ia supernovae in the range `0.023 < z < 0.15`.

`H0` is fully degenerate with the absolute magnitude `M_B` of the supernovae: the Hubble diagram alone gives only the combination of `H0` and `M_B`, and the absolute scale comes from the first and second rungs. This degeneracy is shown explicitly in the project.

**Inverse distance ladder and the early universe.** `H0` derived from the CMB power spectrum under the ΛCDM assumption [Planck2020]; calibration of BAO measurements with the sound horizon [DESI2025; DESI2025DR2].

**Independent routes.** Gravitational wave standard sirens [Abbott2017], lens time delays, cosmic chronometers (`H(z)` directly) [Moresco2022].

**Current status.** Values read during the survey:

- SH0ES Cepheid ladder: `H0 = 73.04 ± 1.04 km s⁻¹ Mpc⁻¹` [Riess2022]. The value was read from the arXiv abstract page of the paper.
- CCHP, TRGB only: `70.39 ± 1.22 (stat.) ± 1.33 (sys.) ± 0.70 (σ_SN)` [Freedman2025]. The value was read from the abstract page of arXiv:2408.06153; the value changed between versions of the paper, so the version to be used must be recorded.
- The Planck 2018 ΛCDM value (about 67.4 ± 0.5) was not read from its source in this session [Planck2020].

The difference between the local and early-universe values is called the "Hubble tension"; a review of proposed solutions is [DiValentino2021]. The difference between Cepheid-based and TRGB-based ladders is also an open debate. A survey of the 2025-2026 publications (the CosmoVerse report, DESI DR2, Union3.1, the DES-SN5YR recalibration) is on the to-do list; during the survey only their search summaries were seen.

## 4. Standardization of Type Ia supernovae

- Luminosity-decline rate relation [Phillips1993] and color correction [Tripp1998].
- Tripp relation: `μ = m_B − M_B + α x1 − β c − δ_bias + δ_host`. `x1` is the stretch, `c` the color parameter; they are fitted from the SALT2 and SALT3 light curve models [Guy2007; Kenworthy2021].
- Host galaxy mass step and bias correction [KesslerScolnic2017; Brout2022].
- Peculiar velocity correction: correcting the redshift at low `z` to the CMB frame and with a flow model [Carr2022].
- Light curve fitting is not rewritten; `sncosmo` is used, and the published distance moduli of the Pantheon+ compilation are taken as input [Scolnic2022].

## 5. Likelihood

Gaussian likelihood with covariance: `χ² = Δμᵀ C⁻¹ Δμ`, `Δμ_i = μ_i − μ_model(z_i; θ)`; `C` is the sum of the statistical and systematic covariance [Brout2022].

- Analytic marginalization over `M_B` (equivalently `H0`): closed form with a flat prior [Conley2011, Appendix C]. The appendix number will be verified during coding.
- Including Cepheid-anchored supernovae breaks the degeneracy: these supernovae are flagged in the Pantheon+SH0ES data file and are compared with the Cepheid distance instead of the model value [Brout2022].
- Joint likelihood: SN + BAO + CMB compressed distance priors (advanced stage).

## 6. Early universe and structure formation (advanced stage)

- Thermal history, recombination, sound horizon `r_s` (fitting formula [EisensteinHu1998]).
- Linear growth factor `D(z)` and growth rate `f ≈ Ωm(z)^0.55`.
- Matter power spectrum and transfer function; Boltzmann codes (CAMB, CLASS) are not rewritten, they are an oracle and a source of input [Dodelson2020].

## Test cases

- Closed-form solutions in an Einstein-de Sitter universe: `D_C = 2 D_H [1 − (1+z)^(−1/2)]`, age `(2/3) t_H`.
- Continuity of the `sinh/sin` branches in the `Ωk → 0` limit for `Ωk ≠ 0`.
- Relative difference below `1e-8` against `astropy.cosmology` over `0 < z < 1100`.
- Reproduction of Hogg's figures [Hogg1999, Fig. 1-6].
- The fit of flat ΛCDM to Pantheon+ must give the published `Ωm` range [Brout2022]; the published value will be read from the paper during coding.
