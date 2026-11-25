# Project: the Hubble diagram and the Hubble constant

The project's first end-to-end science goal: to build the Hubble diagram from open data, estimate the cosmological parameters and reproduce a published result. Physics: [../04_physics/05_cosmology.md](../04_physics/05_cosmology.md). Data: [../02_field_survey/02_hubble_and_cosmology_data.md](../02_field_survey/02_hubble_and_cosmology_data.md).

The project is divided into steps; each step uses the code of the previous one and has its own acceptance criterion.

## Status

Steps 1, 2 and 3 are implemented and were run on 25 November 2026 (`projects/hubble/hubble_analysis.py`, record `projects/hubble/result.json`):

- Step 2, supernovae only (N = 1580, `zHD > 0.01`, calibrators excluded), flat ΛCDM: Ωm = 0.332 ± 0.018; χ² = 1387.1, degrees of freedom 1578. The published value is 0.334 ± 0.018 [Brout2022]; the difference is −0.07σ.
- Step 3, with Cepheid calibrators (N = 1657, 77 calibrator rows): H0 = 73.53 ± 1.00 km s⁻¹ Mpc⁻¹, Ωm = 0.331 ± 0.018, M = −19.244 ± 0.029. The published value is 73.6 ± 1.1 [Brout2022]; the difference is −0.05σ.
- Chain: 32 walkers, 2000 steps, autocorrelation time at most 36 steps, effective sample size about 1600, split R̂ at most 1.03.

The published values were read from the macros in the arXiv TeX source of the paper. The `H0` uncertainty found is smaller than the published one (1.00 and 1.1); the cause has not been investigated. Step 0 (Hubble 1929) and Steps 4-5 have not been written.

There are two simplifications in the implementation: the model distance is computed with a single redshift (`zHD`), as in the equation of [Brout2022], and the heliocentric redshift factor is not applied; the lower redshift threshold is taken as 0.01, and the paper gives it as "~0.01".

## Step 0: Hubble 1929

**Aim.** To establish the velocity-distance relation with the original data; to show the concepts of linear fitting, uncertainty and systematic error on the smallest dataset.

- Data: the table of distances and radial velocities in Hubble's paper [Hubble1929]. The table is transferred from the paper by hand; the transfer is checked by a second reading and recorded under `data/reference/` together with its source.
- Method: a linear fit through the origin and with a free intercept; the uncertainty of the slope (analytic and bootstrap).
- Discussion: the reason the slope found is several times today's value is the systematic error in the distance scale. The statistical uncertainty is small and the systematic error is large; this distinction is the main theme of the rest of the project.

**Acceptance.** The slope reported in the paper is obtained again; the value is read from the paper and written into the test.

Required modules: `statistics/fitting.py`, `io/table.py`.

## Step 1: Distance calculator

**Aim.** `cosmology/friedmann.py` and `cosmology/distances.py`.

- `E(z)`, `D_C`, `D_M`, `D_A`, `D_L`, distance modulus, lookback time, age [Hogg1999].
- Models: flat ΛCDM, curved ΛCDM, `wCDM`, `w0waCDM` [ChevallierPolarski2001; Linder2003].
- Vectorized integral: a single pass for 1701 redshifts; since the likelihood will be called thousands of times, the speed is measured.

**Acceptance.** Relative agreement of `1e-8` with `astropy.cosmology`; the Einstein-de Sitter closed-form solutions; reproduction of the figures of Hogg.

## Step 2: Flat ΛCDM with Pantheon+

**Aim.** Estimating `Ωm` from the supernova Hubble diagram.

- Data: `Pantheon+SH0ES.dat` and `Pantheon+SH0ES_STAT+SYS.cov` [Scolnic2022; Brout2022]. `archives/pantheon.py` downloads the files, records their digests, and reads the table and the 1701 × 1701 covariance.
- Selection: a low `z` cut only in the Hubble diagram analysis (the region where peculiar velocities dominate is left out). The threshold is read from the paper [Brout2022].
- Model: `μ_model(z) = 5 log10(D_L(z; Ωm, H0)/10 pc)`. Using the heliocentric and the CMB-frame redshifts together in the luminosity distance: `D_L = (1 + z_hel) D_M(z_HD)` [Carr2022].
- Likelihood: `χ² = Δᵀ C⁻¹ Δ`; the Cholesky factor is computed a single time. `M_B` and `H0` are degenerate: in this step the combined parameter is marginalized analytically [Conley2011].
- Inference: first a grid scan and the maximum likelihood, then MCMC with our own ensemble sampler [GoodmanWeare2010]; comparison with `emcee` [ForemanMackey2013].
- Diagnostics: `R̂`, effective sample size, chain traces [Vehtari2021]; a plot of the residuals against redshift; `χ²`/degrees of freedom [Andrae2010].

**Acceptance.** The estimated `Ωm` and its uncertainty agree within the uncertainty with the published flat ΛCDM result; the published value is read from the paper and written into the test [Brout2022]. The uncertainty of the fit made with the statistical covariance only comes out smaller than that of the fit with the full covariance; the difference shows the share of the systematics.

## Step 3: H0 with the Cepheid anchor

**Aim.** To break the degeneracy and estimate `H0`.

- For the calibrator supernovae (`IS_CALIBRATOR = 1`, 77 rows), the Cepheid host distance modulus (`CEPH_DIST`) is used instead of the model value; the residual becomes `Δ_i = m_b_corr,i − M_B − μ_Cepheid,i`. For the Hubble-flow supernovae, `Δ_i = m_b_corr,i − M_B − μ_model(z_i)` [Brout2022]. The definition will be verified from the `README` of the data repository and from the paper.
- Free parameters: `H0`, `Ωm`, `M_B`.
- Extensions: `w0CDM`, `w0waCDM`; curvature.

**Acceptance.** The published `H0` value is obtained again; the values reported by the survey are `73.04 ± 1.04` [Riess2022] and, together with Pantheon+, `73.5 ± 1.1` (`w0CDM`) [Brout2022]. The acceptance value is read from the papers while coding.

## Step 4: Systematics and selection effects

After the reproduction holds:

- The effect of the low `z` cut, the peculiar velocity correction and the host mass step on the result.
- Synthetic Hubble diagram: a flux limit is applied to data generated with a known cosmology; the effect of Malmquist bias on `H0` and `w` is measured [KesslerScolnic2017].
- Compilation dependence: repetition with a second supernova compilation.
- Cosmographic fit: model-independent `H0` with `q0` and `j0` [Riess2022].

## Step 5: Combined probes (advanced phase)

- Adding BAO distance measurements; the degeneracy of `H0` with the sound horizon `r_d` and the inverse distance ladder [DESI2025; DESI2025DR2; EisensteinHu1998].
- CMB compressed distance priors [Planck2020].
- `H(z)` with cosmic chronometers [Moresco2022].
- A single-figure comparison of local and early-universe values; the expression of the tension in `σ` and the assumptions of this expression [DiValentino2021].

## Connection with our own observations

When the ground-based observing pipeline is complete, the same project can be fed with raw data:

- Observing the light curve of a Cepheid, finding its period and estimating its distance with the Leavitt law [LeavittPickering1912; Madore1982].
- The multi-band light curve of a nearby Type Ia supernova, fitting with `sncosmo` and the distance modulus [Guy2007].
- Measuring redshifts from galaxy spectra and comparing with SDSS values [TonryDavis1979].

These measurements do not improve `H0`; they show that every link of the measurement chain is understood.

## Outputs

- The Hubble diagram and the residual plot; the `Ωm-H0` and `Ωm-w` posterior distributions; a table of comparison with the published result.
- Run record: data digests, parameters, chains, diagnostics.
- The list of publications the analysis rests on (`kozmos.bibliography_report()`).

## Risks

- Misreading the data definitions (which redshift, which magnitude column, the calibrator rule) is the most likely source of error; the precaution is reading the `README` and the paper and comparing Step 2 with the published value.
- Solving the 1701-dimensional covariance at every step is slow; the Cholesky factor is computed beforehand.
- The extra steps in the published analysis (bias corrections) are embedded in the ready-made columns; reproduction from the light-curve level is outside the scope of this project.
