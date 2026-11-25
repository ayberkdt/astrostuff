# Radiation and photometry physics

Target modules: `radiation/`, `photometry/calibration.py`.

The equations are at planning level; they are read from the cited source before they are coded.

## 1. Radiometric quantities

- Specific intensity `I_ν` [W m⁻² Hz⁻¹ sr⁻¹], flux density `F_ν`, total flux `F`, luminosity `L` [RybickiLightman1979, Section 1].
- Conversion between `F_ν` and `F_λ`: `F_ν = (λ²/c) F_λ`.
- Inverse square law: `F = L / (4π d²)`.
- Jansky: `1 Jy = 1e-26 W m⁻² Hz⁻¹`.
- Surface brightness: independent of distance in the nearby universe; at cosmological distances it dims as `(1+z)⁻⁴`.

## 2. Blackbody

- Planck function: `B_ν(T) = (2hν³/c²) / (exp(hν/kT) − 1)` and its `B_λ` form [RybickiLightman1979, Section 1].
- Wien displacement law, Stefan-Boltzmann law `F = σT⁴`, Rayleigh-Jeans and Wien limits.
- Numerical stability: `expm1` for small `x = hν/kT`, an overflow check for large `x`.
- Definitions of effective temperature, color temperature and brightness temperature [CarrollOstlie2017, Section 3].
- Constants: `h, c, k, σ` from the CODATA recommended values [CODATA2022].

## 3. Magnitude systems

- Pogson definition: `m1 − m2 = −2.5 log10(F1/F2)`.
- **AB system:** `m_AB = −2.5 log10(F_ν / 3631 Jy)`; a flat `F_ν` spectrum gives the same magnitude in every band [OkeGunn1983].
- **Vega system:** the zero point is the spectrum of Vega (or of its model); the AB-Vega offset is tabulated per band [BessellMurphy2012].
- **ST system:** a flat `F_λ` reference.
- Absolute magnitude and distance modulus: `m − M = 5 log10(d / 10 pc) + A + K`.
- Bolometric magnitude: the zero point is fixed by IAU 2015 Resolution B2; the value `M_bol = 0` corresponds to a luminosity of `L = 3.0128e28 W` [Mamajek2015]. The nominal solar values are from IAU 2015 B3 [Prsa2016].
- Magnitude uncertainty: `σ_m ≈ 1.0857 · σ_F / F`; at low S/N the asymmetry and the asinh magnitude apply.

## 4. Filters and synthetic photometry

- Bandpass curve `S(λ)`: the detector counts photons, so the integral is photon weighted:
  `⟨F_λ⟩ = ∫ F_λ S λ dλ / ∫ S λ dλ` [BessellMurphy2012, Appendix A].
- Confusing energy-response and photon-response filter curves is a common source of error; the type of each filter file is kept in its metadata.
- Definitions of effective, pivot and mean wavelength.
- Systems: Johnson-Cousins UBVRI [Bessell1990; Bessell2005], SDSS ugriz [Fukugita1996], Gaia G/BP/RP, 2MASS JHK [Skrutskie2006].
- Color transformations: empirical relations between systems; the color range over which they are valid is recorded.
- Standard stars: Landolt fields [Landolt1992].

## 5. Interstellar extinction and reddening

- `A_λ` extinction, `E(B−V) = A_B − A_V` color excess, `R_V = A_V / E(B−V)`; the Milky Way average is about 3.1.
- Extinction laws: the `A_λ/A_V` curve parametrized by `R_V` [CCM1989; Fitzpatrick1999].
- Dust maps: the all-sky `E(B−V)` map [SFD1998] and its recalibration [SchlaflyFinkbeiner2011]. The resolution of the map and its unreliability at low galactic latitude are documented.
- Wesenheit magnitude: a reddening-free combination `W = m − R (color)` [Madore1982]; used in the Cepheid distance ladder.

## 6. K correction

The flux of a redshifted source in the observed band comes from a different rest-frame wavelength range. The correction depends on the shape of the spectrum and on the band pair; definition and derivation in [Hogg2002; OkeSandage1968]. The `K` term in the distance modulus is this correction.

## 7. Radiative transfer

- Transfer equation: `dI_ν/dτ_ν = −I_ν + S_ν`; optical depth, source function [RybickiLightman1979, Section 1].
- Optically thin and thick limits; formal solution in a plane-parallel atmosphere.
- Limb darkening: linear, quadratic and nonlinear laws; it is an input of the transit light curve model [MandelAgol2002].
- Line profiles: Doppler (Gaussian), natural and pressure broadening (Lorentzian), Voigt profile; equivalent width and curve of growth.
- Saha and Boltzmann equations: ionization and excitation equilibrium [CarrollOstlie2017, Section 8].

## 8. Radiation processes (advanced stage)

Free-free emission, synchrotron, Compton scattering, dust emission (modified blackbody) [RybickiLightman1979, Section 5-7; Draine2011]. They are components of the spectral energy distribution fitting project.

## Test cases

- The integral of the Planck function must give `σT⁴/π`; the peak wavelength must agree with the Wien constant.
- The synthetic photometry of the Vega and solar spectra is compared with published magnitudes [BessellMurphy2012].
- Magnitude difference against `synphot` and `speclite` for the same spectrum and filter is below `1e-4` mag.
- The extinction laws are compared with the `dust_extinction` and `extinction` packages.
