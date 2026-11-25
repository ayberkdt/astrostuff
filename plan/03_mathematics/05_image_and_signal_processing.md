# Mathematics of image and signal processing

Target modules: `reduction/`, `photometry/`, `spectroscopy/`, `timeseries/`.

A CCD frame is the sky convolved with the point spread function, sampled on a pixel grid, with noise added. Almost every step of reduction, photometry and spectral extraction rests on the mathematics of these three operations (convolution, sampling, noise). The equations are at planning level; they are read from the cited source before they are coded.

## 1. Convolution and linear filters

- Definition of convolution, commutativity and associativity; the convolution theorem: the Fourier transform of a convolution is the product of the transforms [VanderPlas2018, Section 2].
- Observed image: `G = (K ⊗ PSF) ⊗ pixel + noise`; the pixel response is also a convolution.
- Separable kernels: a two-dimensional Gaussian convolution splits into two one-dimensional convolutions; the cost drops from `O(N k²)` to `O(N k)`.
- Convolution with the FFT: zero padding, wrap-around effect, boundary conditions (reflect, constant, masked normalization).
- Masked convolution: bad pixels are convolved together with a weight image, and the result is divided by the convolution of the weights.

## 2. Sampling and interpolation

- Sampling theorem: a band-limited signal is fully recovered if it is sampled at twice its highest frequency [Shannon1949].
- Its counterpart in images: if the full width at half maximum of the PSF is narrower than about two pixels, the image is undersampled; shifting and interpolation lose information, and aliasing occurs.
- Interpolation kernels: nearest neighbor, bilinear, bicubic, Lanczos-`a`. Every kernel is a low-pass filter: there is a trade-off between sharpness, ringing and noise correlation.
- Flux conservation in resampling: the Jacobian factor; area-preserving (pixel overlap) methods.
- Combining undersampled frames with sub-pixel offsets (drizzle): pixel shrink factor and output grid; noise correlation [FruchterHook2002].
- Interpolation correlates the noise: the covariance of neighboring pixels becomes nonzero, and the uncertainty of aperture photometry comes out too small under a diagonal assumption. The correction factor is computed from the kernel.

## 3. Noise models

- The sum of Poisson counting noise and Gaussian read noise; variance `σ² = S/g + R²` (in ADU, `g` is the gain) [Howell2006, Section 4; Janesick2007].
- Bias in variance estimation: estimating the variance from the observed counts overweights low-count pixels and pulls the weighted mean down. The remedy: compute the variance from a model (from a smoothed image) [Horne1986].
- Variance-stabilizing transform: for Poisson data, the transform `2√(x + 3/8)` makes the variance approximately constant [Anscombe1948].
- In the low-count regime, the Poisson likelihood instead of `χ²` [Cash1979].
- Magnitudes at low S/N: the logarithm is undefined at negative flux; the asinh magnitude removes this problem [Lupton1999].

## 4. Detection theory

- Matched filter: for a signal of known shape in white Gaussian noise, the linear filter that gives the highest S/N is the signal itself. In point source detection, the kernel is the PSF [Stetson1987].
- Detection threshold and false positive rate: the expected number of false detections of a `kσ` threshold among `N` independent resolution elements; after convolution, the number of independent elements is smaller than the number of pixels.
- Completeness and purity curves; receiver operating characteristic.
- Matched filter in correlated noise: weighting by the noise power spectrum (whitening). It is the basic operation of gravitational wave searches [Allen2012].
- The optimal detection statistic in image differencing: a difference that takes the PSF and noise levels of the two images into account together [Zackay2016].

## 5. PSF modeling and fitting

- Analytic profiles: Gaussian, Moffat [Moffat1969], Airy; multi-component sums [King1971].
- Integral over the pixel: the integral of the profile over the pixel area instead of its value at the pixel center; for a narrow PSF the difference reaches the percent level.
- Empirical (pixel-table) PSF: built from stars on a sub-pixel grid; polynomial variation across the field [Stetson1987].
- Maximum likelihood estimation of flux and position; Cramér-Rao lower bound: astrometric precision is of order `FWHM/(S/N)` [King1971].
- PSF matching kernel: a linear least squares solution in the basis functions of the convolution kernel that equalizes the PSFs of two images [AlardLupton1998].

## 6. Inverse problems

- Deconvolution is ill-posed: noise grows at high frequencies.
- Richardson-Lucy iteration: the expectation-maximization solution of the Poisson likelihood; it conserves flux and positivity, and the number of iterations acts as regularization [Richardson1972; Lucy1974].
- Wiener filter: the minimum mean square error linear solution when the signal and noise power spectra are known.
- Forward modeling is preferred: instead of inverting the image, the model is convolved with the PSF and compared with the data.

## 7. Spectral analysis

- Discrete Fourier transform, power spectrum, Parseval's identity; frequency resolution `1/T`.
- Window functions and leakage: rectangular, Hann, Hamming, Blackman; the trade-off between main lobe width and side lobe level [Harris1978].
- Unevenly spaced sampling: the Fourier transform of the observing window is convolved with the true spectrum and produces aliases [VanderPlas2018, Section 3-4].
- The Lomb-Scargle periodogram and its least squares interpretation [VanderPlas2018, Section 5-6]; the floating-mean and Keplerian generalization [Zechmeister2009]; peak significance [Baluev2008]. The classical and floating-mean forms are implemented in `timeseries/periodogram.py`.
- Cross-correlation and phase correlation: image alignment and radial velocity measurement [TonryDavis1979].

## 8. Multiscale methods

- Wavelet transform (à trous algorithm): scale-by-scale decomposition; separating extended sources from background.
- Median and morphological filters: background map, cosmic ray and trail removal [vanDokkum2001].
- Harmonic analysis on the sphere: spherical harmonics and HEALPix pixelization [Gorski2005].

## 9. Setting up calibration as linear algebra

- Joint solution of multi-night photometry with overlapping fields: the stellar magnitudes and the per-night or per-frame zero points are the unknowns of a single sparse linear system [Padmanabhan2008].
- Astrometric block adjustment: the WCS parameters of all frames and the stellar positions are fitted together.
- These systems are sparse and large; they are solved with a sparse Cholesky or conjugate gradient.

## Verification

- Convolution: convolution with a delta function returns the kernel; total flux is conserved; the FFT and the direct sum agree.
- Interpolation: residual per kernel from a shift and shift back experiment; the covariance of the correlated noise agrees with the predicted one.
- Matched filter: on synthetic data the S/N gain reaches the theoretical value; the false positive rate is the rate predicted by the threshold.
- Richardson-Lucy: flux conservation and the convergence curve on a known scene.
- Periodogram: agreement with `astropy.timeseries.LombScargle` at the `1e-10` level (implemented, `tests/oracle/test_periodogram_oracle.py`).
