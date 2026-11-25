# Relativity, compact objects and high-energy astrophysics

Target modules (advanced stage): `relativity/`, `compact_objects/`, `gravitational_waves/`.

This section belongs to the last stages of the roadmap. The aim is not to rewrite the codes of these fields, but to collect the basic results that have a closed form or reduce to a one-dimensional integral as sourced and tested functions. The equations are at planning level; they are read from the cited source before they are coded.

## 1. Special relativity

- Lorentz transformation, velocity addition, Lorentz factor.
- Relativistic Doppler shift and its connection to [Hogg1999, Eq. 9]: `1 + z = sqrt((1 + v/c)/(1 − v/c))`. This equation was read from the TeX source of the paper.
- Aberration and beaming; Doppler factor `δ = 1/[γ(1 − β cos θ)]`; superluminal motion [RybickiLightman1979, Section 4].
- Combining redshifts: `(1 + z_obs) = (1 + z_cos)(1 + z_pec)`; at small velocities [Hogg1999, Eq. 10]. The distinction between heliocentric and CMB-frame redshifts in supernova analysis comes from here [Carr2022].

## 2. General relativity: spherically symmetric spacetime

- Schwarzschild metric, Schwarzschild radius `r_s = 2GM/c²`, gravitational redshift `1 + z = (1 − r_s/r)^(−1/2)` [Carroll2019, Section 5].
- Geodesics: effective potential, innermost stable circular orbit (`3 r_s`), photon sphere (`1.5 r_s`), angular size of the black hole shadow.
- Classical tests: perihelion advance `Δϖ = 6πGM/[c² a(1 − e²)]` (per period), light deflection `4GM/(c² b)`, Shapiro delay [Carroll2019, Section 5]. They are the source of the corrections in the time and coordinate modules.
- Kerr metric: spin parameter, ergosphere, dependence of the orbit radius on spin (closed forms only).

## 3. Compact objects

- White dwarfs: equation of state of the degenerate electron gas, mass-radius relation, Chandrasekhar mass [ShapiroTeukolsky1983, Section 3].
- Neutron stars: the Tolman-Oppenheimer-Volkoff equation; numerical integration of the mass-radius curve for a given equation of state [ShapiroTeukolsky1983, Section 9].
- Accretion: Eddington luminosity `L_Edd = 4πGMm_p c/σ_T`, accretion efficiency, thin disk temperature profile and multicolor blackbody spectrum [Longair2011, Section 14]. `σ_T` and `m_p` are defined in the constants module with the CODATA 2022 values [CODATA2022].
- Pulsars: magnetic dipole braking, characteristic age, dispersion measure and arrival time delay `Δt ∝ DM ν⁻²`.

## 4. Gravitational waves

- Quadrupole formula; power radiated by a two-body system and orbital decay [Peters1964]. The period decay of the Hulse-Taylor pulsar is a test of this formula.
- Chirp mass `𝓜 = (m1 m2)^(3/5)/(m1 + m2)^(1/5)`; frequency evolution `ḟ ∝ 𝓜^(5/3) f^(11/3)`; time to merger [Maggiore2007, Section 4].
- Waveform: amplitude and phase at Newtonian order; post-Newtonian corrections are not rewritten [Blanchet2014].
- Luminosity distance from the strain amplitude: the standard siren; combined with the redshift it gives `H0` [Abbott2017].
- Matched filter search: noise power spectrum, whitening, template bank, S/N time series [Allen2012].
- End-to-end goal: making the signal visible in the open GW150914 data with whitening and a band-pass filter, and the matched filter S/N with a single template [Abbott2016]. Oracles: `gwpy`, `pycbc`.

## 5. High-energy radiation processes

- Synchrotron: spectrum of a single electron, spectral index of a power-law electron distribution `α = (p − 1)/2`, self-absorption [RybickiLightman1979, Section 6].
- Inverse Compton scattering: Thomson and Klein-Nishina regimes; synchrotron self-Compton [RybickiLightman1979, Section 7].
- Thermal bremsstrahlung: emission coefficient, Gaunt factor; X-ray emission of galaxy clusters [RybickiLightman1979, Section 5].
- Sunyaev-Zeldovich effect: the Compton `y` parameter (advanced stage).
- Absorption of cosmic rays and gamma rays: pair production and the extragalactic background light (advanced stage) [Longair2011].

## 6. Counting statistics

High-energy data is a photon list; the counts per bin are low.

- Poisson likelihood and the Cash statistic [Cash1979].
- Significance in on-off measurements; uncertainty of the background estimate.
- Upper limits: Bayesian and frequentist constructions; coverage check.
- Instrument response: effective area, energy dispersion matrix, PSF; forward folding (convolving the model with the response and comparing with the counts).

Oracles: `gammapy`, `sherpa`.

## Verification

- Mercury's perihelion advance gives about 43 arcseconds per century (the source value will be read during coding).
- The light deflection at the solar limb is about 1.75 arcseconds.
- The Chandrasekhar mass and the limiting mass of the TOV solution are compared with the source values.
- The period decay given by Peters' formula for the Hulse-Taylor system is compared with the published measurement.
- The frequency evolution derived from the chirp mass is consistent with the published parameters of GW150914.
