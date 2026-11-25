# Galaxies and dynamics

Target module: `galaxies/`.

The equations are at planning level; they are read from the cited source before they are coded.

## 1. Gravitational potential theory

- Poisson equation: `∇²Φ = 4πGρ`; Newton's theorems for spherical systems; circular velocity `v_c² = r dΦ/dr`; escape velocity [BinneyTremaine2008, Section 2].
- Analytic potential-density pairs:
  - point mass, homogeneous sphere
  - Plummer sphere [Plummer1911]
  - Hernquist profile [Hernquist1990]
  - Navarro-Frenk-White profile: `ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]`; concentration parameter, definitions of the virial mass and radius [NFW1997]
  - singular and cored isothermal sphere, logarithmic potential
  - disk potentials: Miyamoto-Nagai, exponential disk (rotation curve with Bessel functions) [BinneyTremaine2008, Section 2]
- One interface for every profile: density, potential, force, enclosed mass, circular velocity. A composite potential is the sum over this interface.

Oracles: `galpy` and `gala`.

## 2. Orbits

- Orbit integration in spherical and axisymmetric potentials; symplectic integrator [../03_mathematics/03_numerical_methods.md](../03_mathematics/03_numerical_methods.md).
- Integrals of motion: energy, `L_z`; effective potential, epicyclic approximation, Oort constants [BinneyTremaine2008, Section 3].
- Galactocentric frame: the distance of the Sun from the center, its height and peculiar velocity; 6D position-velocity from ICRS + distance + proper motion + radial velocity.

## 3. Rotation curves and dark matter

- Decomposition of the observed rotation curve into components: `v_c² = v_disk² + v_gas² + v_bulge² + v_halo²`.
- Degeneracy between the mass-to-light ratio and the halo parameters.
- Tully-Fisher relation: a power law between rotation velocity and luminosity; a distance indicator [TullyFisher1977].
- Virial theorem and mass estimate: `M ≈ σ² R / G` in order of magnitude; projection and anisotropy coefficients [BinneyTremaine2008, Section 4].

## 4. Collisionless dynamics

- Collisionless Boltzmann equation and Jeans equations; mass profile from the velocity dispersion with the spherical Jeans equation [BinneyTremaine2008, Section 4].
- Distribution functions, Jeans theorem; generating initial conditions from analytic models.
- Relaxation time, dynamical friction (Chandrasekhar formula), tidal radius [BinneyTremaine2008, Section 7-8].

## 5. Light profiles and structure measurement

- Sérsic profile: `I(R) = I_e exp{−b_n [(R/R_e)^(1/n) − 1]}`; the coefficient `b_n`, total flux, Petrosian and Kron quantities [GrahamDriver2005; Kron1980].
- The exponential disk and the de Vaucouleurs profile are the `n = 1` and `n = 4` members of the Sérsic family.
- Two-dimensional profile fitting: PSF convolution, ellipticity and position angle; image modeling.
- Morphology measures: concentration, asymmetry, Gini coefficient (advanced stage).

## 6. Large-scale structure (advanced stage)

- Two-point correlation function and the Landy-Szalay estimator; power spectrum [LandySzalay1993].
- Baryon acoustic oscillations: the sound horizon scale in the correlation function [Eisenstein2005]; fitting formula for the transfer function [EisensteinHu1998].
- Halo mass function and halo model [MoBoschWhite2010, Section 7].

## 7. Gravitational lensing (advanced stage)

Lens equation, Einstein radius, point and singular isothermal sphere lenses, magnification; time delay and its connection to the Hubble constant [MoBoschWhite2010, Section 6].

## Test cases

- For every potential, `∇²Φ = 4πGρ` is verified with a numerical derivative; total mass in the `r → ∞` limit.
- A circular orbit closes after one period; the energy error stays bounded.
- Force and circular velocity agree with `galpy` for the same potential to a relative difference of `1e-10`.
- The numerical integral of the total Sérsic flux agrees with the closed form [GrahamDriver2005].
