# Implementation status

This file is generated from the source registry by `python bibliography/registry_report.py`; do not edit it by hand.

- Public functions tied to a source: 263
- Functions whose records were all read from the source text: 263
- Source records: 307 (read: 307, pending: 0)
- Constants with a source: 35 (awaiting verification: 0)
- Test functions: 512 (parametrized and property tests count as one)
- Infrastructure functions exempt from the source rule: 4

`read`: the equation was read from the text of the source while the code was written. `pending`: the equation
was written as remembered and tested, but the text of the source has not been opened yet.

## Functions

### `kozmos.archives.gaia`

- `absolute_magnitude`: [Babusiaux2018] Section 2: M_G = G + 5 + 5 log10(ϖ/1000), ϖ in milliarcseconds (read)
- `cone_search`: [ADQL2023] Section 4.2.11 (CIRCLE), 4.2.12 (CONTAINS), 4.2.18 (POINT) (read); [IRSA2026] TAP_SCHEMA.columns, table_name = 'gaia_dr3_source' (read)
- `nearby_stars`: [IRSA2026] TAP_SCHEMA.columns, table_name = 'gaia_dr3_source' (read); [Babusiaux2018] Appendix B: archive query and filters; Section 2 (read)

### `kozmos.archives.heasarc`

- `bright_stars`: [HoffleitWarren1991] HEASARC bsc5p table page: Overview, References, Provenance (read)
- `read_pipe_text`: [TAP2019] Section 2.7.3 (FORMAT) (read)

### `kozmos.archives.pantheon`

- `download`: [Scolnic2022] data repository, 4_DISTANCES_AND_COVAR (read)
- `load`: [Brout2022] data repository README (read)
- `read_covariance`: [Brout2022] data repository README: .cov format (read)
- `read_table`: [Brout2022] data repository README: column definitions (read)

### `kozmos.archives.skyview`

- `download_cutout`: [SkyView2026] Batch Page: runquery.pl examples (Position, Survey, Pixels, Size, Return) (read)

### `kozmos.archives.tap`

- `query`: [TAP2019] Section 2.1 (TAP-sync), 2.7.1 (LANG), 2.7.2 (QUERY), 2.7.3 (FORMAT) (read)
- `read_csv`: [TAP2019] Section 2.7.3 (FORMAT): text/csv output (read)

### `kozmos.archives.twomass`

- `point_sources`: [ADQL2023] Section 4.2.11 (CIRCLE), 4.2.12 (CONTAINS), 4.2.18 (POINT) (read); [IRSA2026] TAP_SCHEMA.columns, table_name = 'fp_psc' (read)

### `kozmos.astrometry.matching`

- `cross_match`: [ERFA2023] sepp.c, s2c.c (read)

### `kozmos.astrometry.solve`

- `find_shift`: [CalabrettaGreisen2002] TeX label eq:px: CRPIX is the pixel position of the reference point (read)
- `fit_wcs`: [CalabrettaGreisen2002] TeX label eq:px (linear pixel → intermediate coordinate relation) (read)
- `refine_wcs`: [CalabrettaGreisen2002] TeX label eq:px (read)

### `kozmos.astrometry.triangles`

- `blind_solve`: [Groth1986] Section II (algorithm) and Section III (example of matching a catalog and a plate list) (read)
- `build_triangles`: [Groth1986] Section II.b: vertex ordering; Eq. 1-3 (R) and Eq. 5 (C); ratio limit (read)
- `match_triangles`: [Groth1986] Section II.c (triangle matching, orientation), II.d (removing false matches), II.e (voting and assignment) (read)

### `kozmos.astrometry.wcs`

- `fit_sip`: [Shupe2005] Eq. 1 with Eqs. 2, 3 (definition of the polynomial) (read)
- `pixel_to_sky`: [Shupe2005] Eq. 1 with Eqs. 2, 3 (read); [CalabrettaGreisen2002] TeX labels eq:px, eq:azphi, eq:azrt, eq:TANtheta, eq:nat2std (read)
- `sky_to_pixel`: [Shupe2005] Eqs. 4, 5, 6 (reverse polynomials); Eq. 1 for the numerical inverse (read); [CalabrettaGreisen2002] TeX labels eq:std2nat, eq:TANRt, eq:azx, eq:azy, eq:px (inverse) (read)
- `wcs_from_header`: [Shupe2005] Section 2, Eqs. 2, 3 (A_p_q, B_p_q, A_ORDER, B_ORDER), Eqs. 5, 6 (AP_p_q, BP_p_q), '-SIP' suffix of CTYPE (read); [CalabrettaGreisen2002] TeX label eq:px and the definition of the LONPOLE keyword (read)

### `kozmos.black_holes.accretion`

- `accretion_luminosity_w`: [Abramowicz2013] §1 (Introduction), efficiency definition η ≡ L/(Ṁ c²) (read)
- `bondi_accretion_rate_kg_s`: [Inayoshi2020] Table 3, Ṁ_B ≡ π e^{3/2} ρ G² M² / c_s³ (read)
- `bondi_radius_m`: [Inayoshi2020] §'Inflow from large scales' (R_B ≡ G M / c_s²) and Table 3 (TeX label tab:phys) (read)
- `eddington_accretion_rate_kg_s`: [Inayoshi2020] §'Growing BHs by Accretion' and Table 3, Ṁ_Edd ≡ L_Edd / (0.1 c²) (a footnote also mentions the L_Edd/c² definition) (read)
- `eddington_luminosity_w`: [Inayoshi2020] §'Growing BHs by Accretion: Is There an Eddington Limit?', L_Edd = 4π c G M / κ_es (read)
- `hills_mass_kg`: [Komossa2015] §1, the r_t equation and the sentence after it: Sun-like stars are swallowed whole above ≳ 10⁸ M☉, white dwarfs are disrupted below ≲ 10⁵ M☉ (read)
- `salpeter_time_s`: [Inayoshi2020] §'Growing BHs by Accretion', t_grow ≈ 0.45 ε/(1−ε) f_duty^{-1} ln(M/M_seed) Gyr; §1 t_Edd ≈ 50 Myr (ε = 0.1) (read)
- `thin_disk_effective_temperature_k`: [Armitage2011] §2 'Accretion Disk Evolution', T_eff⁴ = 3 G M Ṁ / (8π σ r³) [1 − √(r_in/r)] (same eqnarray as TeX label eq:2_nusigma, second line) (read)
- `thin_disk_peak_temperature_k`: [Armitage2011] §2 'Accretion Disk Evolution', T_eff profile; the maximum value was derived by this project (read)
- `thin_disk_peak_temperature_radius_m`: [Armitage2011] §2 'Accretion Disk Evolution', T_eff profile; the location of the maximum was derived by this project (read)
- `tidal_disruption_radius_m`: [Komossa2015] §1 (Introduction), tidal radius equation for r_t (no label): r_t ≃ 7e12 (M/10⁶ M☉)^{1/3} (m*/M☉)^{-1/3} (r*/R☉) cm (read)

### `kozmos.black_holes.basics`

- `bekenstein_hawking_entropy`: [Page2005] TeX label eq:2, S_bh = S_BH ≡ A/4 (Planck units); A = 16π M² (Schwarzschild, paragraph after eq:3) (read)
- `chirp_mass_kg`: [Sathyaprakash2009] TeX label eqn:chirpmassdef, 𝓜 = μ^{3/5} M^{2/5} = ν^{3/5} M (read)
- `critical_impact_parameter_m`: [EHT2019] §1 (Introduction), 'photon capture radius' R_c = √27 r_g, page L1 (read)
- `gravitational_radius_m`: [EHT2019] §1 (Introduction), definition r_g ≡ GM/c², page L1 (read)
- `gravitational_redshift`: [Carroll1997] Section 7, Eq. 7.58-7.61 (read)
- `gw_frequency_at_isco_hz`: [Sathyaprakash2009] text after the paragraph 'Waveform polarizations.': at lowest order the radiation is emitted at twice the orbital frequency (read); [Abramowicz2013] 'Summary: characteristic radii and frequencies' (TeX label schwarzschild-frequencies), Ω_K² = GM/r³, x_ms = 6 (read)
- `gw_frequency_derivative_hz_s`: [Sathyaprakash2009] TeX label eqn:circularPM (Ṗ_b, attributed to Peters-Mathews 1963), f_gw = 2/P_b (read)
- `gw_frequency_from_time_to_merger_hz`: [Sathyaprakash2009] §'Coalescing binaries' paragraph, τ0 (equation without a label); its inverse was taken by this project (read)
- `hawking_evaporation_time_blackbody_s`: [Page2005] TeX label eq:3 (A, κ) and §1 (T = κ/2π) (read)
- `hawking_evaporation_time_s`: [Page2005] TeX label eq:15 (read)
- `hawking_mass_loss_rate_kg_s`: [Page2005] TeX label eq:15 (read)
- `hawking_temperature_k`: [Page2005] §1; T = κ/(2π) and TeX label eq:3 (Schwarzschild: κ = 1/(4M), in Planck units) (read); [Abramowicz2013] §1 (Introduction), Hawking temperature footnote: T_H = ħc³/(8π G M k_B) (read)
- `horizon_angular_velocity_rad_s`: [Page2005] TeX label eq:3, Ω = 4π a / A (Q = 0) (read)
- `isco_orbital_frequency_hz`: [Abramowicz2013] 'Summary: characteristic radii and frequencies', Ω_K² = GM/r³ (TeX label schwarzschild-frequencies), x_ms = 6 (read)
- `isco_radiative_efficiency`: [Reynolds2021] Section 5: η = 1 − e_isco/c² for a Novikov-Thorne disk (read); [Abramowicz2013] §1 (Introduction), efficiency definition η ≡ L/(Ṁ c²) (read)
- `kerr_ergosphere_radius_rg`: [Abramowicz2013] 'Summary: characteristic radii and frequencies', r_0 (ergosphere; same eqnarray, line without a label) (read)
- `kerr_horizon_radius_rg`: [Page2005] TeX label eq:3, r_+ (Q = 0) (read); [Abramowicz2013] 'Summary: characteristic radii and frequencies', r_H (same eqnarray, line without a label) (read)
- `kerr_isco_radius_rg`: [Reynolds2021] Section 2: r_isco equation and the definitions of Z1, Z2 (read); [Abramowicz2013] 'Summary: characteristic radii and frequencies', TeX label stable-radius-kerr (formula of Bardeen et al.) (read)
- `kerr_isco_specific_energy`: [Bardeen1972] Eq. 2.20: stability condition 1 − (E/μ)² ≥ (2/3)(M/r), equality at r = r_ms (read)
- `kerr_photon_orbit_radius_rg`: [Abramowicz2013] 'Summary: characteristic radii and frequencies', TeX label photon-radius-kerr (read)
- `photon_sphere_radius_m`: [Abramowicz2013] 'Summary: characteristic radii and frequencies' (TeX label schwarzschild-frequencies), x_ph = 3 (read)
- `schwarzschild_isco_radius_m`: [Abramowicz2013] 'Summary: characteristic radii and frequencies' (TeX label schwarzschild-frequencies), x_ms = 6 (read)
- `schwarzschild_radius_m`: [EHT2019] §1 (Introduction), R_S = 2 r_g, page L1 (read)
- `shadow_angular_diameter_uas`: [EHT2019] §1 R_c = √27 r_g; §7 'Model Comparison and Parameter Estimation' θ_g ≡ GM/(c² D) (read)
- `time_to_merger_s`: [Sathyaprakash2009] §'Coalescing binaries' paragraph, Newtonian-order chirp time τ0 (equation without a label) (read)

### `kozmos.coordinates.precession`

- `bias_precession_matrix`: [ERFA2023] pmat06.c, fw2m.c (read)
- `fukushima_williams_angles`: [ERFA2023] pfw06.c (read)
- `icrs_to_mean_of_date`: [ERFA2023] pmat06.c (matrix), s2c.c and c2s.c (spherical ↔ unit vector) (read)

### `kozmos.coordinates.transforms`

- `airmass_plane_parallel`: [ERFA2023] hd2ae.c (z component) (read)
- `galactic_to_icrs`: [ERFA2023] icrs2g.c (read)
- `horizontal_to_hour_angle`: [ERFA2023] ae2hd.c (read)
- `hour_angle_to_horizontal`: [ERFA2023] hd2ae.c (read)
- `icrs_to_galactic`: [ERFA2023] icrs2g.c (read)

### `kozmos.core.angles`

- `angular_separation`: [ERFA2023] sepp.c (read)
- `arcsec_to_rad`: [ERFA2023] erfam.h ERFA_DAS2R (read)
- `deg_to_rad`: [ERFA2023] erfam.h ERFA_DD2R (read)
- `rad_to_deg`: [ERFA2023] erfam.h ERFA_DD2R (read)
- `spherical_to_unit_vector`: [ERFA2023] s2c.c (read)
- `unit_vector_to_spherical`: [ERFA2023] c2s.c (read)
- `wrap_0_2pi`: [ERFA2023] anp.c (read)
- `wrap_pm_pi`: [ERFA2023] anpm.c (read)

### `kozmos.core.units`

- `unit_from_suffix`: [SIBrochure9] Table 7 (SI prefixes, p. 138), Table 8 (non-SI units, pp. 140-141), Section 5.2 (unit symbols, p. 142) (read)
- `with_units`: [SIBrochure9] Section 2.3.3 and Table 3 (dimensions of quantities, p. 132), Section 5.2 (p. 142) (read)

### `kozmos.cosmology.distances`

- `angular_diameter_distance_mpc`: [Hogg1999] Eq. 18 (read)
- `angular_diameter_distance_z1_z2_mpc`: [Hogg1999] Eq. 19 (read)
- `comoving_distance_mpc`: [Hogg1999] Eq. 15 (read)
- `comoving_volume_element_mpc3_sr`: [Hogg1999] Eq. 28 (read)
- `comoving_volume_mpc3`: [Hogg1999] Eq. 29 (read)
- `distance_modulus`: [Hogg1999] Eq. 25 (read)
- `doppler_redshift`: [Hogg1999] Eq. 9 (read)
- `doppler_velocity_km_s`: [Hogg1999] Eq. 9 (read)
- `e_function`: [Brout2022] H(z) equation (TeX label eq:hz) (read); [Hogg1999] Eq. 14 (read)
- `hubble_distance_mpc`: [Hogg1999] Eq. 4 (read)
- `hubble_time_gyr`: [Hogg1999] Eq. 3 (read)
- `lookback_time_gyr`: [Hogg1999] Eq. 30 (read)
- `luminosity_distance_mpc`: [Hogg1999] Eq. 21 (read)
- `peculiar_velocity_km_s`: [Hogg1999] Eq. 10 (read)
- `transverse_comoving_distance_mpc`: [Hogg1999] Eq. 16 (read)

### `kozmos.cosmology.hubble_diagram`

- `best_absolute_magnitude`: [Brout2022] eq:likelihood (read)
- `log_likelihood`: [Brout2022] eq:likelihood, eq:sh0eslikelihood (read)
- `prepare_data`: [Brout2022] Section 4: z < ~0.01 is not used in the Hubble diagram; eq:dmuprime (read)
- `residual_vector`: [Brout2022] eq:dmu, eq:dmuprime, eq:dl (read)

### `kozmos.ephemeris.kepler`

- `eccentric_anomaly`: [MurrayCorreia2010] TeX label eq37 (Kepler's equation), eq38-eq39 (Newton-Raphson) (read)
- `orbital_period_s`: [MurrayCorreia2010] TeX label eq19 (read)
- `orbital_speed_m_s`: [MurrayCorreia2010] TeX label eq30 (read)
- `true_anomaly`: [MurrayCorreia2010] TeX label eq17 and eq33 (read)

### `kozmos.io.fits`

- `list_hdus`: [FITS2018] Eq. 1 and 2 (data size), Section 3.3-3.4 (HDU order), XTENSION keyword (read)
- `parse_header`: [Hierarch2009] FITS registered conventions page: 'The HIERARCH Keyword Convention', Description (read); [FITS2018] Section 3.3.1 and 4.1: header blocks, header card components; END keyword (read)
- `read_fits`: [FITS2018] Table 8 (BITPIX), Eq. 3 (BZERO, BSCALE), Section 5 (big-endian byte order), Fig. 1 (axis order) (read)
- `read_table`: [FITS2018] Section 7.3.1 (mandatory keywords, Eq. 8), 7.3.2 (TTYPEn, TSCALn, TZEROn, TNULLn, TDIMn, Table 19; Eq. 7 is printed in Section 7.2.2), 7.3.3 (Table 18, data types, main table), 7.3.5 (variable-length arrays, THEAP) (read)
- `write_fits`: [FITS2018] Section 3.3.1, 4.4.1 (mandatory keywords), Table 8, Section 5 (read)

### `kozmos.observing.detector`

- `signal_to_noise`: [Howell1989] Eq. 1 ('CCD equation') (read)
- `simulate_raw_frame`: [CraigChambers2026] notebook 01-05 'Calibration overview': raw image equation (read)
- `star_field`: [Photutils2026] API: photutils.psf.GaussianPRF, equation for the integral over a pixel (read)

### `kozmos.observing.planning`

- `time_above_horizon_hr`: [ERFA2023] hd2ae.c (altitude component) (read)
- `transit_jd`: [ERFA2023] era00.c (rotation rate), hd2ae.c (read)
- `visibility`: [ERFA2023] gmst06.c, hd2ae.c, pmat06.c, refco.c (read)

### `kozmos.observing.refraction`

- `observed_altitude`: [ERFA2023] refco.c: 'dZ = A tan Z + B tan^3 Z', Z is the observed zenith distance (read)
- `refraction_constants`: [ERFA2023] refco.c (read)

### `kozmos.photometry.aperture`

- `aperture_photometry`: [Photutils2026] User guide: Aperture Photometry, 'subpixel' method (read)

### `kozmos.photometry.background`

- `background_map`: [SExtractorDocs2026] Modeling the background: Background map (read)
- `estimate_background`: [SExtractorDocs2026] Modeling the background: Background estimation, Eq. 4 (read)

### `kozmos.photometry.detection`

- `detect_sources`: [Photutils2026] User guide: Image Segmentation and Source Deblending (read)

### `kozmos.photometry.psf`

- `fit_psf_width`: [PhotutilsPSF2026] User guide: PSF Photometry, 'Estimating the FWHM of Sources' (fit_fwhm) and 'Terminology' (read)
- `gaussian_prf`: [PhotutilsPSF2026] User guide: PSF Photometry, 'Terminology' and 'Analytic PSF Models' (PRF models integrated over the pixels) (read)
- `moffat_prf`: [Moffat1969] p. 458, Eq. 7: I*(r) = I0 / {1 + (r/R)^2}^beta (scan of the paper, read) (read)
- `psf_photometry`: [Stetson1987] Section III.D.1 'The Basics' (grouping of stars, iterated linearized least squares, Eq. 8) and Section III.D.2 subsections a and b (sky held fixed; pixel weights, Eq. 10) (read)

### `kozmos.radiation.infrared`

- `flux_jy_from_vega_magnitude`: [Wright2010] Section 4.4: F_ν° (read); [Cohen2003] Table 2: F_ν(iso) (read)
- `peak_temperature_k`: [CODATA2022] Wien wavelength displacement law constant (read)
- `peak_wavelength_um`: [CODATA2022] Wien wavelength displacement law constant (read)
- `vega_ab_offset`: [Wright2010] Section 4.4: F_ν° (read); [Cohen2003] Table 2: F_ν(iso) (read)
- `vega_magnitude_from_flux`: [Wright2010] Section 4.4: F_ν° (read); [Cohen2003] Table 2: F_ν(iso) (read)

### `kozmos.radiation.magnitudes`

- `absolute_bolometric_magnitude`: [Mamajek2015] M_Bol definition; L_0 = 3.0128e28 W (read)
- `blackbody_flux`: [CODATA2022] Stefan-Boltzmann constant (read)
- `distance_modulus_from_pc`: [Hogg1999] Eq. 25 (read)
- `luminosity_from_bolometric_magnitude_w`: [Mamajek2015] M_Bol definition; L_0 = 3.0128e28 W (read)
- `pc_from_distance_modulus`: [Hogg1999] Eq. 25 (read)
- `planck_frequency`: [Astropy2022] modeling/physical_models.py, BlackBody.evaluate (read)
- `planck_wavelength`: [Astropy2022] modeling/physical_models.py, BlackBody.evaluate (read)

### `kozmos.reduction.calibration`

- `combine`: [CraigChambers2026] notebook 01-05 'Calibration overview': 'Every image has noise' (read)
- `gain_and_read_noise`: [IRAF2026] noao/obsutil/src/doc/findgain.hlp, ALGORITHM (read)
- `reduce_frame`: [CraigChambers2026] notebook 01-05 'Calibration overview': reduction equation (read)

### `kozmos.reduction.cosmic_rays`

- `detect_cosmic_rays`: [vanDokkum2001] TeX labels kernel.eq, noise.eq, sigmap.eq and the following equations (S′, F, selection criterion) (read)

### `kozmos.reduction.sky`

- `subtract_dynamic_sky`: [Schirmer2013] 'Static and dynamic background models' section: dynamic model (read)
- `subtract_static_sky`: [Schirmer2013] 'Static and dynamic background models' section: static model (read)

### `kozmos.spectroscopy.extraction`

- `box_extract`: [Horne1986] Eq. 2 and Eq. 3 (standard spectrum and its variance); Eq. 12 (pixel variance); Table I steps 2 and 4 (read)
- `fit_sky_background`: [Horne1986] Table I step 3 (fit sky background, FIT_x[D; V]); Sect. II.A (weighted polynomial per wavelength, weights from Step 2 variances); Sect. II.D (iterative sigma-clipping, threshold sigma_clip^2 about 16) (read)
- `optimal_extract`: [Horne1986] Table I steps 4-9; Eq. 8 and Eq. 9 (optimal spectrum and variance); Eq. 13 (revised variance); Eq. 14 (initial profile); Sect. II.D (profile fit); Sect. II.E (cosmic-ray rejection, threshold 25); Sect. II.F (iteration) (read)

### `kozmos.spectroscopy.lines`

- `equivalent_width`: [VollmannEversberg2006] Eq. 1 (definition; TeX label eq1) and the derivatives following Eq. 3, Eq. for sigma^2(W) (TeX label eq10) (read)
- `equivalent_width_sigma_vollmann_eversberg`: [VollmannEversberg2006] TeX label eq4 (Sect. 'The error of the equivalent width', low- and high-flux lines) (read)
- `fit_gaussian_line`: [SciPyDocs2026] scipy.optimize.curve_fit docstring: parameters sigma and absolute_sigma, covariance pcov (read)
- `voigt_profile`: [Zaghloul2011] Eq. (1): w(z) = exp(-z^2) erfc(-iz) and Re w(x + iy) = V(x, y), the real Voigt function (y > 0) (read); [SciPyDocs2026] scipy.special.voigt_profile Notes: V(x; sigma, gamma) = Re[w(z)] / (sigma sqrt(2 pi)), z = (x + i gamma) / (sqrt(2) sigma) (read)

### `kozmos.spectroscopy.velocity`

- `correlation_r_statistic`: [TonryDavis1979] Eq. 20 (antisymmetric rms sigma_a) and Eq. 23 (r = h / (sqrt(2) sigma_a)) (read)
- `cross_correlate`: [TonryDavis1979] Eqs. 3, 4 and 5 (rms sigma_g, sigma_t; normalised correlation c(n); Fourier form C(k)) (read)
- `fit_correlation_peak`: [TonryDavis1979] Sect. III.b (quadratic least-squares fit to the correlation peak); Sect. III.c.ii (polynomial fit to 11 points around the peak); Sect. IV (parabola fitted to the largest peak in -1000 < cZ < 20000 km/s); the parabola h0 (1 - 2 (n - n0)^2 / w0^2) in the text after Eq. 20 gives the FWHM w (read)
- `half_amplitude_wavenumber`: [TonryDavis1979] p. 1519, text before Eq. 34 (correlation functions have a half-amplitude point in Fourier space of about wavenumber 50); definition of the highest frequency B in Sect. III.c.i, text after Eq. 21 (read)
- `prepare_for_correlation`: [TonryDavis1979] Sect. IV (spectrum preparation: fourth-order least-squares polynomial subtracted, 5 percent cosine-bell end masking, bandpass filter rising from wavenumber 10, 1 at 20, falling from 100, 0 at 200 for N = 2048); Sect. III.a (spectra continuum subtracted, endmasked and filtered) (read)
- `radial_velocity_cross_correlation`: [TonryDavis1979] Eq. 34 (ln(1 + Z) = delta / A) and Eq. 35 (error of ln(1 + Z) = epsilon / A), p. 1519 (read)
- `rebin_log_wavelength`: [TonryDavis1979] Eq. 1 (n = A ln(lambda) + B) and Sect. IV (a section mapped to 2048 bins linear in ln lambda) (read)
- `shift_error_bins`: [TonryDavis1979] Eq. 24 (mean shift error eps = (1/4) (N / 2B) (1 / (1 + r))) and Eq. 35 (the same with numbers) (read)

### `kozmos.spectroscopy.wavelengths`

- `air_refractive_index`: [Morton1991] Eq. 3 (Sect. 2, IAU standard after Oosterhoff 1957 and Edlen 1953) (read)
- `air_to_vacuum`: [Morton1991] Eq. 3 (Sect. 2) (read)
- `doppler_shifted_wavelength`: [Hogg1999] Eq. 9 and the redshift definition z = lambda_o/lambda_e - 1 printed just before it (read)
- `radial_velocity_from_wavelengths`: [Hogg1999] Eq. 9 and the redshift definition z = lambda_o/lambda_e - 1 printed just before it (read)
- `resolving_power`: [TonryDavis1979] Sect. IV: rms resolution about 2 A, given as 120 km per second at 5000 A (read)
- `resolving_power_from_velocity`: [TonryDavis1979] Sect. IV: rms resolution about 2 A, given as 120 km per second at 5000 A (read)
- `vacuum_to_air`: [Morton1991] Eq. 3 (Sect. 2) (read)
- `velocity_resolution_km_s`: [TonryDavis1979] Sect. IV: rms resolution about 2 A, given as 120 km per second at 5000 A (read)

### `kozmos.star_formation.clouds`

- `bonnor_ebert_critical_coefficients`: [Krumholz2015] problems/probset2.tex part (e) (mass expression); probsol2.tex parts (c)-(d) (ODE), part (f) (m_max = 1.18, ρ_c/ρ_s = 14.0) (read)
- `bonnor_ebert_critical_mass_kg`: [Krumholz2015] problems/probsol2.tex, part (h) of the Bonnor-Ebert sphere exercise (read); [McKee2007] TeX label MBE_def (read)
- `bonnor_ebert_radius_m`: [McKee2007] TeX label MBE_def, the next paragraph: R_BE = 0.486 c_s / sqrt(G ρ_0) (read)
- `cloud_star_formation_rate_kg_s`: [McKee2007] TeX label eq:epsff (read)
- `column_density_m2`: [Krumholz2015] Section 8 (Giant Molecular Clouds), Mass Measurement, Optically Thick Lines: N = M / (μ m_H π R²) (read)
- `crossing_time_s`: [Krumholz2015] Section 8 (Giant Molecular Clouds), Molecular Cloud Timescales: t_cr = R/σ (read)
- `free_fall_efficiency`: [Krumholz2015] Section 8 (Giant Molecular Clouds), Depletion Time: ε_ff = t_ff / t_dep (read); [McKee2007] TeX label eq:epsff (read)
- `free_fall_time_s`: [Krumholz2015] Section 6, definition of t_ff after the Jeans analysis; Section 8 timescales (read); [McKee2007] TeX label eq:tff (read)
- `isothermal_collapse_accretion_rate_kg_s`: [McKee2007] TeX label eq:mdssis (Shu expansion wave solution) (read)
- `jeans_length_m`: [Krumholz2015] Section 6 (Gravitational Instability and Collapse), λ_J = 2π/k_J (read); [McKee2007] TeX label LJeans (read)
- `jeans_mass_kg`: [Krumholz2015] Section 6, definition of the Jeans mass and its footnote (read); [McKee2007] TeX label LJeans, the paragraph that follows (M_J = 2.47 M_BE) (read)
- `linewidth_size_km_s`: [Krumholz2015] Section 8 (Giant Molecular Clouds), Scaling Relations: Larson's second law (read); [McKee2007] TeX label GMC_lws (PCA fit of Heyer and Brunt 2004) (read)
- `mach_number`: [Krumholz2015] Section 4 (Gas Flows and Turbulence), Reynolds and Mach numbers subsection: M ~ V/c_s (read)
- `magnetic_critical_mass_kg`: [Krumholz2015] Section 6, Magnetic Support and the Magnetic Critical Mass (read); [McKee2007] TeX label eq:mphi (read)
- `mass_density_kg_m3`: [Krumholz2015] Section 8 (Giant Molecular Clouds), Scaling Relations: n = 3M / (4π R³ μ m_H) (read)
- `normalized_mass_to_flux_ratio`: [McKee2007] After TeX label eq:mphi: G^{1/2}(Σ/B)_cr = 1/(2π) (infinite cold sheet) (read)
- `number_density_m3`: [Krumholz2015] Section 8 (Giant Molecular Clouds), Scaling Relations: n = 3M / (4π R³ μ m_H) (read)
- `sound_speed_m_s`: [Krumholz2015] Section 4 (Gas Flows and Turbulence), the c_s expression in the Reynolds and Mach numbers subsection (read)
- `surface_density_kg_m2`: [Krumholz2015] Section 8 (Giant Molecular Clouds), Mass Measurement, Optically Thick Lines: N = M / (μ m_H π R²) (read)
- `virial_parameter`: [Krumholz2015] Section 8 (Giant Molecular Clouds), Larson's third law: α_vir = 5σ²R/(GM) (read); [McKee2007] TeX label alpha_vir_def (read)
- `virial_parameter_from_surface_density`: [McKee2007] TeX label alpha_vir_def (read)

### `kozmos.star_formation.laws`

- `depletion_time_yr`: [Krumholz2015] Section 8 (Giant Molecular Clouds), Depletion Time: t_dep = Σ_gas / Σ̇_* (read)
- `free_fall_sfr_surface_density`: [McKee2007] TeX label SFR-ff (read)
- `kennicutt_orbital_sfr_surface_density`: [McKee2007] TeX label eq:ks (read); [Kennicutt1998] Eq. 7 (order in the text; referred to as 'equation (7)') (read)
- `kennicutt_schmidt_sfr_surface_density`: [McKee2007] TeX label eq:ks (read); [Kennicutt1998] Eq. 4 (order in the text; referred to as 'equation (4)') (read)
- `life_cycle_sfr_surface_density`: [McKee2007] TeX label SFR-lc (read)
- `sfr_calibration_msun_yr`: [Kennicutt2012] Table 1 (TeX label sfrequations) and the equation log SFR = log L_x − log C_x in the subsection 'An Updated Compendium of Integrated SFR Calibrations' (read)
- `sfr_radio_1p4ghz_msun_yr`: [Kennicutt2012] Table 1 (TeX label sfrequations), 1.4 GHz row, and the equation log SFR = log L_x − log C_x (read)
- `toomre_q_gas`: [McKee2007] TeX label Qdef (read)
- `two_dimensional_jeans_mass_msun`: [McKee2007] TeX label MJ2D (MJ2Ddef) (read)

### `kozmos.stars.imf`

- `chabrier_dn_dm`: [Chabrier2003] Section 'The disk stellar mass function', TeX label IMFdisk and Table 1 (table.diskIMF), 'IMF' row (read)
- `chabrier_mass_fraction_above`: [Chabrier2003] Section 'The disk stellar mass function', TeX label IMFdisk and Table 1 (read)
- `chabrier_mean_mass`: [Chabrier2003] Section 'The disk stellar mass function', TeX label IMFdisk and Table 1 (read)
- `chabrier_sample`: [Chabrier2003] Section 'The disk stellar mass function', TeX label IMFdisk and Table 1 (read)
- `kroupa_dn_dm`: [Kroupa2001] Section 'The universal IMF', TeX labels eq:xi and eq:imf (read)
- `kroupa_mass_fraction_above`: [Kroupa2001] Section 'The universal IMF', TeX label eq:imf and the mass-fraction sentences (read)
- `kroupa_mean_mass`: [Kroupa2001] Section 'The universal IMF', TeX label eq:imf and the mean-mass sentence (read)
- `kroupa_sample`: [Kroupa2001] Section 'The universal IMF', TeX label eq:imf (read)
- `salpeter_dn_dm`: [Salpeter1955] Eq. 2 (definition) and Eq. 5 (read); [Kroupa2001] Section '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2.35 (read); [Chabrier2003] Section 'Mass function' and 'Functional forms' (TeX label IMF1): α = 2.35, x = 1.35 (read)
- `salpeter_mass_fraction_above`: [Salpeter1955] Eq. 2 (definition) and Eq. 5 (read); [Kroupa2001] Section '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2.35 (read); [Chabrier2003] Section 'Mass function' and 'Functional forms' (TeX label IMF1) (read)
- `salpeter_mean_mass`: [Salpeter1955] Eq. 2 (definition) and Eq. 5 (read); [Kroupa2001] Section '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2.35 (read); [Chabrier2003] Section 'Mass function' and 'Functional forms' (TeX label IMF1) (read)
- `salpeter_sample`: [Salpeter1955] Eq. 2 (definition) and Eq. 5 (read); [Kroupa2001] Section '0.1 < m < 1 M_⊙': Salpeter 1955, α = 2.35 (read); [Chabrier2003] Section 'Mass function' and 'Functional forms' (TeX label IMF1) (read)

### `kozmos.stars.scaling`

- `chandrasekhar_mass_kg`: [Bludman2010] Section II (introduction) M = 4π 0ω_3 (K/πG)^{3/2}; Section II.A M_Ch = 5.83 M_⊙/μ_e² = 1.46 M_⊙ (read)
- `critical_angular_velocity_rad_s`: [Paxton2013] sentence after the equation labeled mdotomega: Ω_crit² = (1 - L/L_Edd) G M / R³ (read)
- `eddington_beta`: [Bludman2010] Section II.B, Eddington quartic equation: (1-β)/β⁴ = (M μ²/M_*)² (read)
- `eddington_luminosity_w`: [Bludman2010] Section II.B, Eddington standard model: L_Edd := 4π c G M/κ_p (read); [Paxton2013] Section 'Rotationally-Enhanced Mass Loss' (TeX label s.rotation-mass-loss): L_Edd = 4π c G M/κ (read)
- `eddington_scale_mass_kg`: [Bludman2010] Section II.B: M_* := 3√10 0ω_3/π³ (hc/(G m_H^{4/3}))^{3/2} = 18.3 M_⊙ (read)
- `effective_temperature_k`: [Prsa2016] text: T_eff from the Stefan-Boltzmann law; Table 'formu', M_Bol row (read)
- `free_fall_time_s`: [McKee2007] TeX etiketi eq:tff (read)
- `kelvin_helmholtz_timescale_s`: [Zinnecker2007] Section 'The Accretion Phase / Formation of the Hydrostatic Core' text and GLOSSARY: t_KH = G M²/(R L) (read)
- `log_surface_gravity_cgs`: [Prsa2016] Table 'formu', 'log of surface gravity' row: 4.438068... + log M - 2 log R (read)
- `luminosity_w`: [Prsa2016] Table 'formu', M_Bol row (4πσR² term); text: T_eff from the Stefan-Boltzmann law (read)
- `mean_density_kg_m3`: [Bludman2010] Section I, text before Table I: ρ̄_n = 3M/4πR³ (read)
- `nuclear_timescale_yr`: [Pols2011] Section 2.4.2, Eq. 2.37 (read)
- `standard_model_luminosity_w`: [Bludman2010] Section II.B: L = L_Edd [1 - β(M)] (read)
- `surface_gravity_m_s2`: [Prsa2016] Table 'formu', 'log of surface gravity' row: log g = log(10^6 GM) - 2 log(100 R) [cgs] (read)
- `white_dwarf_k_coefficient_si`: [Bludman2010] Section II.A (relativistic degenerate stars): K_WD = (hc/8)(3/π)^{1/3}(m_H μ_e)^{-4/3} (read)

### `kozmos.stars.structure`

- `lane_emden_analytic`: [Bludman2010] Table II (rows n = 0, 1, 5) (read)
- `lane_emden_central_to_mean_density`: [Liu1995] Eq. 38 (m = M at r = R) (read); [Bludman2010] Table I, ρ_cn/ρ̄_n column; definition ρ̄_n = 3M/4πR³ (read)
- `lane_emden_first_zero`: [Liu1995] Eq. 2 (the first zero of θ is the surface of the star) (read); [Bludman2010] Table I, ξ_1n column (read)
- `lane_emden_mass_coefficient`: [Bludman2010] Table I (0ω_3 = 2.018) and Section II, M = 4π 0ω_3 (K/πG)^{3/2} (read); [Liu1995] Eq. 38 (m = 4π ρ_c r³ (-θ'/ξ)) (read)
- `lane_emden_solve`: [Bludman2010] Table II (Taylor series 1 - ξ²/6 + n ξ⁴/120 ...) (read); [Liu1995] Eq. 2, 4, 5 (Lane-Emden equation, central conditions, power series) (read)
- `polytrope_central_density_kg_m3`: [Liu1995] Eq. 38 (at r = R), [Bludman2010] Table I for the definition of ρ̄ (read)
- `polytrope_central_pressure_pa`: [Liu1995] Eq. 41 (P = K ρ_c^{(n+1)/n} ω^{n+1}, ω = 1 at the center) and Eq. 3 (read)
- `polytrope_central_temperature_k`: [Liu1995] Eq. 42 (T = μ m_p K ρ_c^{1/n} ω / k, ω = 1 at the center) (read)
- `polytrope_gravitational_energy_j`: [Liu1995] Eq. 44 and 45 (E_g; Φ = 0 and P = 0 at the surface) (read)
- `polytrope_k_coefficient_si`: [Liu1995] Eq. 3 (definition of A², A = ξ_1/R) (read)
- `polytrope_radius_m`: [Bludman2010] Table I, last column (M ~ R^{(n-3)/(n-1)}) (read); [Liu1995] Eq. 3 and Eq. 38 (read)
- `virial_mean_temperature_k`: [Liu1995] Eq. 44 and 45 (read); [McKee2007] TeX labels eq:iddot, eq:calt (virial theorem) (read)

### `kozmos.statistics.fitting`

- `chi_square`: [HoggBovyLang2010] TeX label eq:chisquared (read)
- `fit_line`: [HoggBovyLang2010] TeX label eq:fofx, eq:lsf (read)
- `linear_least_squares`: [HoggBovyLang2010] TeX label eq:lsf and the parameter covariance equation (read)

### `kozmos.statistics.mcmc`

- `autocorrelation_time`: [ForemanMackey2013] Section 3, 'Autocorrelation time': τ_f and C_f(T) equations (read)
- `ensemble_sample`: [ForemanMackey2013] Eq. proposal, acceptance; Algorithm 3 (parallel stretch move) (read)
- `rank_normalized_rhat`: [Vehtari2021] Section 4.1-4.2: rank normalization and folding equations (read)
- `sample_until_converged`: [Vehtari2021] Section 'Recommendations for practice': use the sample only if the rank-normalized, folded R-hat < 1.01 (read)
- `split_rhat`: [Vehtari2021] Section 3.1: B, W, var⁺ and R̂ equations (read)
- `stretch_factor`: [ForemanMackey2013] Eq. goodmanprop (g(z) ∝ 1/√z, z ∈ [1/a, a]) (read)

### `kozmos.statistics.robust`

- `biweight_location`: [Astropy2022] stats/biweight.py, biweight_location docstring (read)
- `biweight_scale`: [Astropy2022] stats/biweight.py, biweight_scale docstring (read)
- `mad_scale`: [Astropy2022] stats/funcs.py, median_absolute_deviation and mad_std (read)
- `sigma_clip`: [Astropy2022] stats/sigma_clipping.py, SigmaClip docstring (read)

### `kozmos.time.julian`

- `calendar_to_jd`: [ERFA2023] cal2jd.c; erfam.h ERFA_DJM0 (read)
- `calendar_to_mjd`: [ERFA2023] cal2jd.c (read)
- `earth_rotation_angle`: [ERFA2023] era00.c (read)
- `greenwich_mean_sidereal_time`: [ERFA2023] gmst06.c (read)
- `jd_to_calendar`: [ERFA2023] jd2cal.c (read)
- `local_mean_sidereal_time`: [Astropy2022] time/core.py, Time._sid_time_or_earth_rot_ang (read); [ERFA2023] gmst06.c (read)
- `mean_obliquity`: [ERFA2023] obl06.c (read)
- `tai_to_tt`: [ERFA2023] erfam.h ERFA_TTMTAI, ERFA_DAYSEC (read)
- `tt_to_tai`: [ERFA2023] erfam.h ERFA_TTMTAI, ERFA_DAYSEC (read)

### `kozmos.time.scales`

- `tai_minus_utc_s`: [ERFA2023] dat.c (read)
- `tt_minus_utc_s`: [ERFA2023] dat.c; erfam.h ERFA_TTMTAI (read)

### `kozmos.timeseries.periodogram`

- `false_alarm_probability`: [Astropy2022] timeseries/periodograms/lombscargle/_statistics.py, fap_single and fap_naive (read); [VanderPlas2018] TeX label eq:FAP-neff and the following paragraph (read)
- `frequency_grid`: [VanderPlas2018] Section 7.1, TeX label eq:n-eval (read)
- `lomb_scargle`: [VanderPlas2018] eq:floating-mean-model, eq:chi2-with-errors, eq:lomb-scargle-normalized (read)
- `lomb_scargle_classical`: [VanderPlas2018] eq:lomb-scargle-periodogram, eq:tau-def (read)

### `kozmos.visualization.stretch`

- `asinh_stretch`: [Lupton2004] Section 2, footnote: f(x) = asinh(α Q (x − m)) / Q (read)
- `lupton_rgb`: [Lupton2004] Section 2, TeX label EqIntensityStretch (read)
- `percentile_scale`: [Lupton2004] Section 2: linear mapping and the maximum value M (read)

## Constants

- `SPEED_OF_LIGHT` = 299792458 m s^-1 (exact) [CODATA2022] speed of light in vacuum (read)
- `PLANCK` = 6.62607015e-34 J Hz^-1 (exact) [CODATA2022] Planck constant (read)
- `BOLTZMANN` = 1.380649e-23 J K^-1 (exact) [CODATA2022] Boltzmann constant (read)
- `GRAVITATIONAL_CONSTANT` = 6.6743e-11 m^3 kg^-1 s^-2 (± 1.5e-15) [CODATA2022] Newtonian constant of gravitation (read)
- `ELEMENTARY_CHARGE` = 1.602176634e-19 C (exact) [CODATA2022] elementary charge (read)
- `ELECTRON_MASS` = 9.1093837139e-31 kg (± 2.8e-40) [CODATA2022] electron mass (read)
- `PROTON_MASS` = 1.67262192595e-27 kg (± 5.2e-37) [CODATA2022] proton mass (read)
- `ATOMIC_MASS_UNIT` = 1.66053906892e-27 kg (± 5.2e-37) [CODATA2022] atomic mass constant (read)
- `AVOGADRO` = 6.02214076e+23 mol^-1 (exact) [CODATA2022] Avogadro constant (read)
- `THOMSON_CROSS_SECTION` = 6.6524587051e-29 m^2 (± 6.2e-38) [CODATA2022] Thomson cross section (read)
- `STEFAN_BOLTZMANN` = 5.67037441918e-08 W m^-2 K^-4 (exact) [CODATA2022] Stefan-Boltzmann constant (5.670 374 419... e-8) (read; derivation: 2 pi^5 k^4 / (15 h^3 c^2))
- `WIEN_WAVELENGTH` = 0.002897771955 m K (exact) [CODATA2022] Wien wavelength displacement law constant (2.897 771 955... e-3) (read)
- `SECOND_RADIATION_CONSTANT` = 0.014387768775 m K (exact) [CODATA2022] second radiation constant (1.438 776 877... e-2) (read; derivation: h c / k)
- `SOLAR_RADIUS` = 695700000 m (exact) [Prsa2016] Table 1 (read)
- `SOLAR_LUMINOSITY` = 3.828e+26 W (exact) [Prsa2016] Table 1 (read)
- `SOLAR_EFFECTIVE_TEMPERATURE` = 5772 K (exact) [Prsa2016] Table 1 (read)
- `SOLAR_CONSTANT` = 1361 W m^-2 (exact) [Prsa2016] Table 1 (read)
- `GM_SUN` = 1.3271244e+20 m^3 s^-2 (exact) [Prsa2016] Table 1 (read)
- `GM_EARTH` = 3.986004e+14 m^3 s^-2 (exact) [Prsa2016] Table 1 (read)
- `GM_JUPITER` = 1.2668653e+17 m^3 s^-2 (exact) [Prsa2016] Table 1 (read)
- `EARTH_EQUATORIAL_RADIUS` = 6378100 m (exact) [Prsa2016] Table 1 (read)
- `EARTH_POLAR_RADIUS` = 6356800 m (exact) [Prsa2016] Table 1 (read)
- `JUPITER_EQUATORIAL_RADIUS` = 71492000 m (exact) [Prsa2016] Table 1 (read)
- `JUPITER_POLAR_RADIUS` = 66854000 m (exact) [Prsa2016] Table 1 (read)
- `BOLOMETRIC_ZERO_POINT_LUMINOSITY` = 3.0128e+28 W (exact) [Mamajek2015] L_0 definition (read)
- `ASTRONOMICAL_UNIT` = 149597870700 m (exact) [ERFA2023] erfam.h ERFA_DAU (read)
- `DAY_SECONDS` = 86400 s (exact) [ERFA2023] erfam.h ERFA_DAYSEC (read)
- `JULIAN_YEAR_DAYS` = 365.25 day (exact) [ERFA2023] erfam.h ERFA_DJY (read)
- `JULIAN_CENTURY_DAYS` = 36525 day (exact) [ERFA2023] erfam.h ERFA_DJC (read)
- `J2000_JD` = 2451545 day (exact) [ERFA2023] erfam.h ERFA_DJ00 (read)
- `MJD_ZERO_JD` = 2400000.5 day (exact) [ERFA2023] erfam.h ERFA_DJM0 (read)
- `TT_MINUS_TAI` = 32.184 s (exact) [ERFA2023] erfam.h ERFA_TTMTAI (read)
- `PARSEC` = 3.08567758149e+16 m (exact) [Mamajek2015] final note: the parsec is exactly (648000/π) au (read; derivation: au · 648000 / pi)
- `MEGAPARSEC` = 3.08567758149e+22 m (exact) [Mamajek2015] final note: the parsec is exactly (648000/π) au (read; derivation: 1e6 · parsec)
- `JULIAN_YEAR_SECONDS` = 31557600 s (exact) [ERFA2023] erfam.h ERFA_DJY · ERFA_DAYSEC (read; derivation: 365.25 · 86400)

## Number of functions per source

- [ADQL2023]: 2
- [Abramowicz2013]: 11
- [Armitage2011]: 3
- [Astropy2022]: 8
- [Babusiaux2018]: 2
- [Bardeen1972]: 1
- [Bludman2010]: 13
- [Brout2022]: 8
- [CODATA2022]: 3
- [CalabrettaGreisen2002]: 6
- [Carroll1997]: 1
- [Chabrier2003]: 8
- [Cohen2003]: 3
- [CraigChambers2026]: 3
- [EHT2019]: 4
- [ERFA2023]: 33
- [FITS2018]: 5
- [ForemanMackey2013]: 3
- [Groth1986]: 3
- [Hierarch2009]: 1
- [HoffleitWarren1991]: 1
- [Hogg1999]: 19
- [HoggBovyLang2010]: 3
- [Horne1986]: 3
- [Howell1989]: 1
- [IRAF2026]: 1
- [IRSA2026]: 3
- [Inayoshi2020]: 5
- [Kennicutt1998]: 2
- [Kennicutt2012]: 2
- [Komossa2015]: 2
- [Kroupa2001]: 8
- [Krumholz2015]: 17
- [Liu1995]: 11
- [Lupton2004]: 3
- [Mamajek2015]: 2
- [McKee2007]: 21
- [Moffat1969]: 1
- [Morton1991]: 3
- [MurrayCorreia2010]: 4
- [Page2005]: 7
- [Paxton2013]: 2
- [Photutils2026]: 3
- [PhotutilsPSF2026]: 2
- [Pols2011]: 1
- [Prsa2016]: 4
- [Reynolds2021]: 2
- [SExtractorDocs2026]: 2
- [SIBrochure9]: 2
- [Salpeter1955]: 4
- [Sathyaprakash2009]: 5
- [Schirmer2013]: 2
- [SciPyDocs2026]: 2
- [Scolnic2022]: 1
- [Shupe2005]: 4
- [SkyView2026]: 1
- [Stetson1987]: 1
- [TAP2019]: 3
- [TonryDavis1979]: 11
- [VanderPlas2018]: 4
- [Vehtari2021]: 3
- [VollmannEversberg2006]: 2
- [Wright2010]: 3
- [Zaghloul2011]: 1
- [Zinnecker2007]: 1
- [vanDokkum2001]: 1

## Functions awaiting source reading

