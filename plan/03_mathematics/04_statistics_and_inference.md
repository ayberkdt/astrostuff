# Statistics and inference

Target module: `statistics/`.

Every step from observation to physical parameter is an inference problem: the Hubble constant, the depth of a transit or the distance of a cluster is obtained from the probability of the data under a model. The equations are read from the cited source before they are coded.

## 1. Probability basics

- Distributions: Gaussian, Poisson, binomial, `χ²`, Student-t, log-normal, exponential, Cauchy. For each: density, cumulative distribution, moments and sampling [Ivezic2020, Section 3].
- Poisson statistics: `σ = √N` in photon counting; the breakdown of the Gaussian approximation at low counts and the Cash statistic [Cash1979].
- The central limit theorem and the cases where it does not hold (heavy tails).
- Bayes' theorem: `p(θ|D) ∝ p(D|θ) p(θ)` [SiviaSkilling2006, Section 1].

## 2. Uncertainty propagation

- Linear propagation: `Σ_f = J Σ_x Jᵀ`. The Jacobian by analytic derivatives, finite differences or automatic differentiation.
- Bias in the nonlinear case: magnitude ↔ flux, parallax → distance. `1/ϖ` is not an unbiased distance estimate; as the fractional parallax error grows, full inference with a prior distribution is needed [BailerJones2015; Luri2018].
- Monte Carlo propagation and a consistency test against linear propagation.
- Keeping systematic and statistical uncertainty separate; adding systematic terms to the covariance matrix [Conley2011; Brout2022].

## 3. Robust statistics

- Location: median, trimmed mean, biweight location. Scale: MAD (`σ ≈ 1.4826 · MAD`), biweight scale [Beers1990].
- Sigma clipping: iterative; the center and scale estimators are selectable. It is the basic tool of image combination and background estimation.
- Modeling outliers instead of discarding them: a mixture model [HoggBovyLang2010, Section 3].

## 4. Model fitting

- Maximum likelihood and its equivalence to `χ²` under Gaussian noise.
- Linear least squares: weighted, with covariance; closed-form solution and parameter covariance [HoggBovyLang2010, Section 1].
- **Errors in both axes and intrinsic scatter:** the usual case in astronomy (Leavitt law, Tully-Fisher, `M-σ`). Methods: Akritas-Bershady BCES [AkritasBershady1996], the Bayesian hierarchical model [Kelly2007], the likelihood of the perpendicular distance to the line [HoggBovyLang2010, Section 7]. Ordinary least squares gives a biased slope here.
- Goodness of fit: `χ²`, reduced `χ²` and its limits (in a nonlinear model the number of degrees of freedom is uncertain) [Andrae2010]; residual analysis.
- Model comparison: likelihood ratio, AIC, BIC, Bayes evidence. Which one answers which question is documented [Ivezic2020, Section 5].

## 5. Bayesian sampling

- **Metropolis-Hastings:** proposal distribution, acceptance fraction, detailed balance [Metropolis1953; Hastings1970].
- **Affine-invariant ensemble sampler:** the stretch move [GoodmanWeare2010]; the algorithm of `emcee` [ForemanMackey2013]. It is rewritten, and its moments are compared with `emcee` on the same target distribution.
- **Hamiltonian Monte Carlo and NUTS:** requires gradients; efficient in high dimensions [HoffmanGelman2014]. Not written in the first stage; NumPyro or PyMC are used as the oracle and production tool.
- **Nested sampling:** evidence computation and multimodal posterior distributions [Skilling2006; Speagle2020].
- **Convergence diagnostics:** rank-normalized split `R̂`, effective sample size, autocorrelation time [GelmanRubin1992; Vehtari2021; ForemanMackey2013]. Chain traces and corner plots are produced in every analysis.
- Practical rules [HoggForemanMackey2018]: initialization, discarding the burn-in phase, writing the prior explicitly, the effect of parametrization.

## 6. Priors and hierarchical models

- There is no uninformative prior: for a scale parameter, a flat and a log-flat prior give different results [SiviaSkilling2006, Section 5].
- Hierarchical model: modeling the population distribution and the individual measurements together (intrinsic scatter, selection effect) [Gelman2013, Section 5].
- Posterior predictive check: comparing synthetic data generated from the model with the observed data [Gelman2013, Section 6].

## 7. Selection effects and biases

- **Malmquist bias:** in a flux-limited sample, distant sources are brighter than average; it reduces the distance in the Hubble diagram.
- **Eddington bias:** in a steep count distribution, sources that measurement error carries above the limit.
- **Lutz-Kelker bias:** in a sample selected by a parallax threshold [Luri2018].
- Ways of correcting: forward modeling (the selection function enters the likelihood), bias correction by simulation [KesslerScolnic2017], the likelihood of truncated and censored data [Ivezic2020, Section 4].

The selection effect is one of the dominant systematics of Hubble diagram analysis; the `cosmology` module takes it into the design from the start.

## 8. Nonparametric and resampling methods

- Bootstrap and jackknife [Efron1979]; block bootstrap for correlated data.
- Kernel density estimation, histogram bin width rules, Bayesian blocks [Ivezic2020, Section 5-6].
- Kolmogorov-Smirnov and Anderson-Darling tests and their misuses (against a distribution whose parameters were estimated from the data).

## 9. Gaussian processes

A model of correlated noise (stellar variability, instrumental systematics) [RasmussenWilliams2006, Section 2]. The general solution scales as `O(N³)`; for a one-dimensional time series, the `celerite` kernel family gives an `O(N)` solution [ForemanMackey2017]. In the first stage, a general Cholesky-based solution is written for small `N`.

## 10. Machine learning

Classification (star/galaxy separation, variable star types), regression (photometric redshift), clustering (open cluster membership: in the Gaia proper motion and parallax space), dimensionality reduction [Ivezic2020, Section 6-9]. These methods are not rewritten; `scikit-learn` is used. The project's contribution is feature extraction, setting up the training/test split correctly and carrying the uncertainty through.

Simulation-based inference (when the likelihood cannot be written) is a later-stage topic [Cranmer2020].

## Implementation order

Distributions and robust statistics → linear fitting (with covariance) → uncertainty propagation → Metropolis-Hastings → ensemble sampler and diagnostics → fitting with errors in both axes → likelihood with selection effects → nested sampling → Gaussian process.
