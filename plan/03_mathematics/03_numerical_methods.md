# Numerical methods

Target module: `core/numerics/`.

SciPy is a runtime dependency, so general-purpose integration, root finding and optimization can be taken from it. The methods below are still written: the astronomy-specific ones (Kepler solver, symplectic integrator) are not in SciPy, and the basic ones are written for learning and tested against SciPy. Which function will be used in production is chosen explicitly in each module.

## 1. Floating point arithmetic

- IEEE 754 double precision: about 15.95 decimal digits, `ε ≈ 2.22e-16` [Goldberg1991].
- Cancellation error: the difference of nearby numbers. Examples: `1 − cos θ` at a small angle, distance differences at small `z`, holding JD in a single `float64`.
- Compensated summation: Kahan and Neumaier summation; for summing long series (nutation) [Higham2002, Section 4].
- Using `log1p`, `expm1`, `hypot`, `atan2` is written down as a rule.
- Condition number and backward stability: a measure of how many digits of a result can be trusted [Higham2002, Section 1].

## 2. Root finding

- Bisection, Newton-Raphson, secant, Brent's method [Press2007, Section 9].
- **Kepler's equation** `M = E − e sin E`: Newton and Halley iteration, choice of starting guess, convergence near `e → 1`. A solution without iteration: Markley's fifth-order corrected method [Markley1995]. The hyperbolic (`M = e sinh H − H`) and universal variable forms [Vallado2022, Section 2].
- Test: `|E − e sin E − M| < 1e-14` on the grid `[0, 2π) × [0, 1)`.

## 3. Numerical integration

- Trapezoid, Simpson, Romberg; Gauss-Legendre and adaptive Gauss-Kronrod [Press2007, Section 4].
- Cosmological distance integral `∫ dz / E(z)`: a smooth integrand, so fixed-order Gauss-Legendre is enough; a vectorized cumulative integral (a single pass for many `z`) is a separate function.
- Synthetic photometry integrals: the trapezoid rule over a tabulated spectrum and filter curve; the grid resampling error is measured.
- Change of variables for singular or oscillatory integrands.

## 4. Ordinary differential equations

- Explicit Runge-Kutta: RK4; step control with an embedded pair, Dormand-Prince 5(4) [DormandPrince1980; Hairer1993, Section II].
- **Symplectic methods:** leapfrog (kick-drift-kick), Yoshida's fourth-order composition [Yoshida1990]. In Hamiltonian systems the energy error stays bounded; in RK4 it accumulates. The comparison test measures the energy error over 10⁴ periods on a Kepler orbit.
- **Wisdom-Holman map:** the splitting of Kepler motion and the interaction between planets [WisdomHolman1991; ReinTamayo2015].
- **High-order adaptive integrator:** a 15th-order method with Gauss-Radau spacing, for close encounters [ReinSpiegel2015].
- Stiff systems (stellar structure, reaction networks): implicit methods, through SciPy `solve_ivp` (`Radau`, `BDF`).
- Two-point boundary value problem: shooting method and relaxation; for the Lane-Emden and stellar structure equations.

## 5. Interpolation and approximation

- Linear, cubic spline, Lagrange, Chebyshev [Press2007, Section 3 and 5].
- **Chebyshev ephemeris:** the JPL DE files store position as Chebyshev coefficients over time intervals; evaluation is done with Clenshaw recursion, and velocity comes from the derivative series [Park2021].
- Image resampling: nearest neighbor, bilinear, bicubic, Lanczos kernel; the effect on flux conservation and noise correlation.
- Spectrum resampling: flux-conserving rebinning.

## 6. Linear algebra

- Least squares: QR or SVD instead of the normal equations; the condition number is reported [Press2007, Section 15].
- `χ² = rᵀ C⁻¹ r` with a covariance matrix: the inverse is not taken; it is computed with a Cholesky decomposition and a triangular solve. The Pantheon+ covariance is 1701 × 1701 [Brout2022]; this path is both faster and more stable.
- Principal component analysis: through SVD.

## 7. Fourier methods

- Discrete Fourier transform, sampling theorem, Nyquist frequency, aliasing, leakage and window functions [Press2007, Section 12-13].
- Convolution and cross-correlation with the FFT: PSF convolution, image alignment (phase correlation), spectral redshift measurement [TonryDavis1979].
- Unevenly spaced sampling: Lomb-Scargle [VanderPlas2018]; details are in [../05_data_science/04_time_series.md](../05_data_science/04_time_series.md).

## 8. Optimization

- Nonlinear least squares: Gauss-Newton and Levenberg-Marquardt; computing the Jacobian analytically or by finite differences [Press2007, Section 15].
- Derivative-free method: Nelder-Mead. Quasi-Newton: BFGS.
- Parameter covariance: `(JᵀWJ)⁻¹` at the best point; valid only if the likelihood is approximately Gaussian.

## 9. Random numbers and Monte Carlo

- `numpy.random.Generator` (PCG64); the generator is passed as an argument.
- Inverse transform, rejection sampling, importance sampling [Press2007, Section 7].
- Monte Carlo for uncertainty propagation and comparison with linear propagation.

## 10. Special functions

Elliptic integrals (transit model [MandelAgol2002]), gamma and incomplete gamma (`χ²` probability), the error function, Bessel functions (Airy pattern), Legendre polynomials and spherical harmonics (gravity field, CMB). All are used through `scipy.special`; they are not rewritten.

## Verification

Each method is tested by its convergence order on a problem with a known closed-form solution: when the step is halved, the error must drop by a factor of `2^p`. The convergence order test shows not merely that the implementation works but that it converges at the rate the theory predicts.
