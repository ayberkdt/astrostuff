# Function inventory

The list of the functions to be written, module by module. The current status of the implemented ones is not maintained by hand; the file [STATUS.md](STATUS.md) is generated from the source registry. The "implemented" marks in this document belong to the first release of 25 November 2026.

Each function line gives the name, the input and output units, and the primary source. Names may change; units and sources are part of the contract.

## `core`

**`constants`** (implemented). CODATA 2022 physical constants [CODATA2022], IAU 2015 B3 nominal solar and planetary values [Prsa2016], IAU 2015 B2 bolometric zero point [Mamajek2015], time constants [ERFA2023]. To be added: the absolute magnitudes of the Sun band by band, all the planetary mass parameters [Park2021], the Earth ellipsoids (WGS84 and GRS80) [IERS2010].

**`angles`** (implemented). Conversions between degrees, arcseconds and radians, range reduction, unit vector conversions, angular separation [ERFA2023]. To be added:

- `sexagesimal_to_degrees`, `degrees_to_sexagesimal`: string parsing and formatting; the negative-zero case.
- `position_angle(l1, b1, l2, b2) → rad`: from north through east.
- `offset_point(l, b, position_angle, separation) → (l, b)`.

**`vectors`**. `rotation_1/2/3(angle) → 3×3`, matrix product and transpose, Rodrigues rotation, spherical velocity transform [SOFA2023].

**`numerical`**. `find_root_brent`, `gauss_legendre`, `romberg`, `rk4_step`, `dormand_prince` [DormandPrince1980], `leapfrog_step`, `yoshida4_step` [Yoshida1990], `chebyshev_evaluate` (Clenshaw), `cubic_spline`, `lanczos_kernel`, `kahan_sum` [Higham2002].

## `time`

**`julian`** (implemented). Calendar ↔ MJD/JD, TAI ↔ TT, Earth rotation angle, Greenwich and local mean sidereal time, mean obliquity [ERFA2023].

**`scales`** (partially implemented: `tai_minus_utc_s`, `tt_minus_utc_s` [ERFA2023]).

- `utc_to_tai(jd1, jd2) → (jd1, jd2)`: handling the leap-second day [IERS2010].
- `tt_to_tdb(jd1, jd2, ...) → (jd1, jd2)`: the single-term low-precision form and the full series [Kaplan2005].
- `utc_to_ut1(jd1, jd2, dut1_s)`: DUT1 is given from outside.
- `delta_t_s(year)`: historical polynomials [Meeus1998].

**`barycentric`**. `roemer_delay_s`, `bjd_tdb` [Eastman2010]; `barycentric_velocity_correction_m_s` [WrightEastman2014].

**`events`**. `rise_set_transit`, `twilight`, `moon_phase` [Meeus1998].

## `coordinates`

**`transforms`** (implemented). ICRS ↔ galactic, hour angle ↔ horizontal, plane-parallel airmass [ERFA2023].

To be added:

- `equatorial_to_ecliptic(ra, dec, obliquity) → (lam, bet)` and its inverse.
- `precession_matrix_iau2006(tt) → 3×3` [Capitaine2003]; `nutation_iau2000b(tt) → (dpsi, deps)` [IERS2010].
- `apply_proper_motion(ra, dec, pmra, pmdec, parallax, rv, epoch1, epoch2)` [SOFA2023].
- `annual_aberration`, `light_deflection`, `diurnal_aberration` [Kaplan2005].
- `refraction_bennett(altitude, pressure, temperature) → rad` [Bennett1982].
- `airmass_kasten_young(zenith) → X` [KastenYoung1989], `airmass_young(zenith)` [Young1994].
- `icrs_to_observed(...)`: the whole chain; compared with ERFA `atco13`.

**`projections`**. `tan_projection`, `tan_inverse`, `sin`, `stg`, `arc`, `zea`, `car`, `ait` [CalabrettaGreisen2002]; `pixel_to_sky`, `sky_to_pixel` (`CRPIX`, `CD`, SIP) [GreisenCalabretta2002; Shupe2005].

**`healpix`**. `angle_to_pixel`, `pixel_to_angle`, `neighbors`, ring ↔ nested [Gorski2005].

## `ephemeris`

**`kepler`** (implemented). Kepler's equation (elliptic), true anomaly, period, vis-viva; the source records are waiting to be read [MurrayDermott1999].

To be added:

- `kepler_markley(M, e) → E`: iteration-free solution [Markley1995].
- `hyperbolic_anomaly`, `universal_variable_propagate` [Vallado2022].
- `elements_to_state(a, e, i, Ω, ω, M, μ) → (r, v)` and its inverse [Vallado2022].
- `sun_position_low`, `moon_position_low`, `planet_position_low` [Meeus1998].
- `read_spk`, `spk_position_velocity`: DE440 [Park2021].
- `n_body_propagate(state, masses, dt, steps, method)`: leapfrog, Wisdom-Holman [WisdomHolman1991; ReinTamayo2015].
- `transit_light_curve(t, parameters)`: with limb darkening [MandelAgol2002].
- `radial_velocity_curve(t, K, P, e, ω, T0, γ)`.

## `radiation`

**`magnitudes`** (implemented). Bolometric magnitude ↔ luminosity [Mamajek2015], distance modulus [Hogg1999], Planck function (source record pending) [RybickiLightman1979], blackbody flux.

To be added:

- `ab_magnitude_from_flux(flux_jy)`, `flux_jy_from_ab_magnitude(ab_magnitude)` [OkeGunn1983].
- `asinh_magnitude(flux, softening)` [Lupton1999].
- `read_filter`, `synthetic_magnitude(spectrum, filter, system)` [BessellMurphy2012].
- `pivot_wavelength`, `effective_wavelength`.
- `extinction_ccm89(wavelength, rv)`, `extinction_f99` [CCM1989; Fitzpatrick1999].
- `wesenheit(magnitude, color, r)` [Madore1982].
- `k_correction(spectrum, z, filter_observed, filter_rest)` [Hogg2002].

## `observing`

**`detector`** (implemented; source records pending). `simulate_raw_frame`, `star_field`, `signal_to_noise` [Howell2006; MerlineHowell1995].

To be added:

- `target_visibility(target, observer, times) → (altitude, airmass)`.
- `moon_separation_and_illumination`; `moon_sky_brightness` [KrisciunasSchaefer1991].
- `exposure_time(target_snr, ...)`, `saturation_time`.
- `seeing_fwhm(r0, wavelength, airmass)` [Fried1966].
- To be added to the generator: cosmic rays, hot pixels, saturation, Moffat PSF [Moffat1969].
- `photon_transfer_curve(flat_pair_list)`: multi-level variance-mean fit [Janesick2007].

## `reduction`

**`calibration`** (implemented; source records pending). `combine`, `reduce_frame`, `gain_and_read_noise` [Howell2006; Janesick2007].

To be added:

- A mask plane, and adding the noise of the master calibration frames to the variance.
- `cosmic_ray_laplacian(image, ...)` [vanDokkum2001].
- `background_map(image, cell, filter_size)` [BertinArnouts1996].
- `align`, `resample`, `combine`; `drizzle` [FruchterHook2002].
- `psf_matching_kernel`, `difference_image` [AlardLupton1998; Zackay2016].

## `photometry` and `astrometry`

- `find_stars(image, fwhm, threshold)` [Stetson1987]; `segment`, `deblend` [BertinArnouts1996].
- `centroid_moment`, `centroid_gauss`; `second_moments`.
- `aperture_photometry(image, positions, r, annulus)`: exact pixel-circle intersection [MerlineHowell1995].
- `kron_radius` [Kron1980]; `optimal_photometry` [Naylor1998].
- `build_psf_model`, `psf_photometry` [Stetson1987].
- `fit_zero_point(instrumental_magnitude, catalog_magnitude, color, airmass)` [Landolt1992; Padmanabhan2008].
- `match_catalog(xy, sky, tolerance)`; `fit_wcs(xy, radec, degree)` [Lang2010; Shupe2005]. Written: `astrometry.wcs.wcs_from_header`, `pixel_to_sky`, `sky_to_pixel` (TAN) and `astrometry.matching.cross_match`; `fit_wcs` was not written (at the first release; it was written later, in `astrometry.solve`).

## `spectroscopy`

- `find_trace`, `subtract_sky`, `optimal_extraction` [Horne1986].
- `find_arc_lines`, `wavelength_solution`; `air_to_vacuum` [Ciddor1996].
- `sensitivity_function`, `flux_calibrate`.
- `normalize_continuum`, `fit_line_profile` (Gauss, Voigt), `equivalent_width`.
- `cross_correlation_redshift(spectrum, template)` [TonryDavis1979].

## `timeseries`

**`periodogram`** (implemented). Weighted Lomb-Scargle, classical and floating-mean, false alarm probability, frequency grid [VanderPlas2018].

To be added:

- `window_periodogram`, `multiterm_lomb_scargle` [VanderPlas2018]; `baluev_fap` [Baluev2008].
- `phase_fold`, `bin_data`; `phase_dispersion_minimization` [Stellingwerf1978].
- `detrend_median`, `detrend_biweight`.
- `box_least_squares(t, flux, periods, durations)` [Kovacs2002]; `transit_least_squares` [HippkeHeller2019].
- `gp_log_likelihood(kernel, t, y, sigma)` [RasmussenWilliams2006].

## `statistics`

**`fitting`** (implemented). χ² with covariance, linear least squares, line fitting [HoggBovyLang2010].

**`mcmc`** (implemented). Stretch move, ensemble sampler [ForemanMackey2013], split and rank-normalized R̂ [Vehtari2021], autocorrelation time (source record pending).

**`robust`** (implemented; source records pending). Sigma clipping, MAD scale, biweight location and scale [Beers1990].

To be added:

- `propagate_linear(f, x, cov)`, `propagate_monte_carlo`.
- `fit_line_two_axes(x, y, sx, sy, intrinsic_scatter)` [HoggBovyLang2010; Kelly2007].
- `mixture_model_outliers` [HoggBovyLang2010].
- `metropolis_hastings` [Metropolis1953; Hastings1970]; effective sample size [Vehtari2021]; comparison with classical R̂ [GelmanRubin1992].
- `nested_sample` [Skilling2006].
- `bootstrap`, `jackknife` [Efron1979].
- `distance_posterior_from_parallax(parallax, sigma, prior)` [BailerJones2015].
- `cash_statistic` [Cash1979].

## `cosmology`

**`distances`** (implemented). `E(z)`, comoving, transverse comoving, angular diameter and luminosity distances, distance modulus, lookback time, volume element and comoving volume, angular diameter distance between two redshifts, Doppler shift and peculiar velocity, Hubble distance and time [Hogg1999; Brout2022].

**`hubble_diagram`** (implemented). Data selection, residual vector, likelihood, closed-form best absolute magnitude [Brout2022].

To be added:

- Radiation and massive-neutrino terms; the `w0wa` parametrization [ChevallierPolarski2001; Linder2003].
- `age_of_universe_gyr` (including the radiation term).
- `cosmographic_distance(z, h0, q0, j0)` [Riess2022].
- `sound_horizon_eh98(omega_m_h2, omega_b_h2)` [EisensteinHu1998]; `bao_distances(z)` (`D_V`, `D_M/r_d`, `D_H/r_d`) [DESI2025].
- `growth_factor(z)` [Dodelson2020].
- `tripp_distance_modulus(mB, x1, c, alpha, beta, M)` [Tripp1998].

## `stars` and `galaxies`

- `lane_emden_solve(n) → (ξ, θ)` [Kippenhahn2012]; `white_dwarf_mass_radius` [ShapiroTeukolsky1983].
- `imf_salpeter`, `imf_kroupa`, `imf_chabrier`: density and sampling [Salpeter1955; Kroupa2001; Chabrier2003].
- `read_isochrone`, `interpolate_isochrone` [Choi2016].
- `leavitt_distance_modulus(period, magnitude, color)` [Madore1982].
- Potential interface: `density`, `potential`, `force`, `circular_velocity`; Plummer, Hernquist, NFW, Miyamoto-Nagai [Plummer1911; Hernquist1990; NFW1997].
- `sersic_profile`, `sersic_bn`, `sersic_total_flux` [GrahamDriver2005].
- `orbit_propagate(potential, state, dt, steps)`.

## `archives` and `io`

**`archives.pantheon`** (implemented). Download, digest record, reading the table and covariance [Scolnic2022; Brout2022].

To be added: `cache` (general download and digest record), `tap` (synchronous and asynchronous queries), `gaia`, `mast`, `horizons`; `io.fits` (header, image, binary table) [Pence2010; FITS2018], `io.votable`.

## Modules added in the second release

The modules written in the second working session of 25 November 2026 that appeared as "to be added" in the first form of this inventory, or did not appear at all:

- `black_holes.basics`, `black_holes.accretion` (implemented): radii, Kerr orbits, shadow, accretion, Hawking radiation, merging binaries [Abramowicz2013; EHT2019; Page2005; Inayoshi2020; Armitage2011; Komossa2015; Sathyaprakash2009].
- `stars.structure`, `stars.scaling`, `stars.imf` (implemented): polytropes, scaling relations, Chandrasekhar mass, mass functions [Liu1995; Bludman2010; Paxton2013; Kroupa2001; Chabrier2003].
- `star_formation.clouds`, `star_formation.laws` (implemented): Jeans, Bonnor-Ebert, virial parameter, Kennicutt-Schmidt, Toomre Q [McKee2007; Krumholz2015; Kennicutt1998; Kennicutt2012].
- `archives.cache`, `archives.tap`, `archives.gaia`, `archives.skyview` (implemented) [TAP2019; Babusiaux2018; SkyView2026].
- `io.fits` (implemented: primary image unit) [FITS2018].
- `photometry.background`, `photometry.detection`, `photometry.aperture` (implemented: background, detection, aperture photometry) [SExtractorDocs2026].
- `visualization.stretch` (implemented) [Lupton2004].
- `radiation.infrared` (implemented) [Cohen2003; Wright2010].
- `observing.planning` (implemented: visibility, meridian transit, time above the horizon) [ERFA2023].

The function list of these modules is in [STATUS.md](STATUS.md); the scope, limits and to-do items are in [04_physics/08_black_holes_stars_and_star_formation.md](04_physics/08_black_holes_stars_and_star_formation.md) and [05_data_science/06_optical_and_infrared_observing.md](05_data_science/06_optical_and_infrared_observing.md).

## Order

The phases in the roadmap give the reading order of this inventory ([07_roadmap.md](07_roadmap.md)). Before a function is written, its source is opened; if the source cannot be reached, the function is written with `status="pending"` and falls into the pending list of [STATUS.md](STATUS.md).
