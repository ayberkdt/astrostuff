# Implementation status

This file is generated from the source registry by `python bibliography/registry_report.py`; do not edit it by hand.

- Public functions tied to a source: 414
- Functions whose records were all read from the source text: 414
- Source records: 476 (read: 476, pending: 0)
- Constants with a source: 35 (awaiting verification: 0)
- Test functions: 887 (parametrized and property tests count as one)
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

### `kozmos.black_holes.kerr`

- `frame_dragging_angular_velocity_rad_s`: [Abramowicz2013] 'Summary: characteristic radii and frequencies', omega (TeX label dragg-velocity-kerr) (read)
- `kerr_area_theorem_margin_m2`: [Page2005] Section 1, second law of black hole mechanics (Hawking's area theorem), no TeX label (read); [Sathyaprakash2009] Section 'Tests of general relativity' (no TeX label, article.tex line 5867): 'the final area must exceed the sum of the areas of the initial holes' (read)
- `kerr_bekenstein_hawking_entropy`: [Page2005] TeX label eq:2, S_bh = S_BH = A / 4 (Planck units), with A of TeX label eq:3 (read)
- `kerr_circular_orbit_angular_velocity_rad_s`: [Abramowicz2013] 'Summary: characteristic radii and frequencies', Omega_K^2 (TeX label keplerian-velocity-kerr, line 1095 of article.tex; the label sits on the following empty row, line 1096) (read)
- `kerr_circular_orbit_frequency_hz`: [Abramowicz2013] 'Summary: characteristic radii and frequencies', Omega_K^2 (TeX label keplerian-velocity-kerr) (read)
- `kerr_circular_orbit_specific_angular_momentum_rg`: [Pugliese2011] Eq. LPM (Kerr_A.tex lines 284-288) (read); [Abramowicz2013] 'Subsection: ISCO' (TeX labels circular-geodesic, redshift-factor, omega-covariant) and the Kerr metric (TeX label kerr-metric), with Omega_K (TeX label keplerian-velocity-kerr) (read)
- `kerr_circular_orbit_specific_energy`: [Pugliese2011] Eq. LPM (Kerr_A.tex lines 284-288) and the energies E(L) printed after it (lines 299-303, 312-316) (read); [Abramowicz2013] 'Subsection: ISCO' (TeX labels circular-geodesic, redshift-factor, omega-covariant) and the Kerr metric (TeX label kerr-metric), with Omega_K (TeX label keplerian-velocity-kerr) (read)
- `kerr_hawking_temperature_k`: [Page2005] Introduction (text before TeX label eq:2) and section 'Hawking Emission Formulae': T_bh = kappa / (2 pi), kappa of TeX label eq:3 (read)
- `kerr_horizon_area_m2`: [Visser2007] section 'Horizons', area of the horizons A_H = 4 pi (r_+^2 + a^2) = 8 pi (m^2 + sqrt(m^4 - m^2 a^2)) (equation without a label, Survey-grqc-3.tex line 1242) (read); [Page2005] TeX label eq:3, A = 4 pi (r_+^2 + a^2) = 4 pi M^2 [2 - Q_*^2 + 2 (1 - a_*^2 - Q_*^2)^{1/2}] (Q = 0) (read)
- `kerr_irreducible_mass_kg`: [Carroll1997] Section 7, Eq. 7.149 (TeX label 7.149): M_irr^2 = A / (16 pi G^2) (read)
- `kerr_marginally_bound_radius_rg`: [Abramowicz2013] 'Summary: characteristic radii and frequencies', r_mb (TeX label bound-radius-kerr) (read)
- `kerr_newman_horizon_radius_rg`: [Carroll1997] Section 7, Eq. 7.113 (TeX label 7.113): r_+- for Reissner-Nordstrom (read); [Page2005] TeX label eq:3, r_+ = M + (M^2 - a^2 - Q^2)^{1/2} = M [1 + (1 - a_*^2 - Q_*^2)^{1/2}] (read)
- `kerr_nodal_precession_frequency_hz`: [Reynolds2021] 'Spin from quasi-periodic oscillations' (TeX label qpos), relativistic precession model paragraph, Spin2020.tex line 465: test-particle Lense-Thirring precession frequency (the source spells it Lens-Thirring) (read); [Abramowicz2013] 'Summary: characteristic radii and frequencies', Omega_K and (omega_theta*)^2 (TeX labels keplerian-velocity-kerr, epicyclic-vertical) (read)
- `kerr_penrose_extractable_fraction`: [Carroll1997] Section 7, Eq. 7.152 (TeX label 7.152) and the sentence after it: about 29% of the energy of an extreme Kerr black hole (read)
- `kerr_periastron_precession_frequency_hz`: [Reynolds2021] 'Spin from quasi-periodic oscillations' (TeX label qpos), relativistic precession model paragraph, Spin2020.tex line 465: test-particle periastron precession frequency (read); [Abramowicz2013] 'Summary: characteristic radii and frequencies', Omega_K and (omega_r*)^2 (TeX labels keplerian-velocity-kerr, epicyclic-radial) (read)
- `kerr_radial_epicyclic_frequency_hz`: [Abramowicz2013] 'Summary: characteristic radii and frequencies', (omega_r*)^2 (TeX label epicyclic-radial) (read)
- `kerr_surface_gravity_m_s2`: [Carroll1997] Section 7, Eq. 7.155 (TeX label 7.155) (read); [Page2005] TeX label eq:3, kappa = 4 pi (r_+ - M) / A = (1/2) M^{-1} [1 + (1 - Q_*^2/2)(1 - a_*^2 - Q_*^2)^{-1/2}]^{-1} (Q = 0) (read)
- `kerr_vertical_epicyclic_frequency_hz`: [Abramowicz2013] 'Summary: characteristic radii and frequencies', (omega_theta*)^2 (TeX label epicyclic-vertical) (read)
- `merger_radiated_fraction_bound`: [Carroll1997] Section 7, Eq. 7.149 (M_irr) and Eq. 7.152 (M >= M_irr) (read); [Sathyaprakash2009] Section 'Tests of general relativity' (no TeX label, article.tex line 5867): the area theorem for a merger (read)

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

### `kozmos.cosmology.dark_energy`

- `acceleration_onset_redshift`: [Frieman2008] Eq. q (sec2.tex line 136-141): q = -a_ddot/(a H^2); acceleration for q < 0 (read)
- `age_gyr`: [ChevallierPolarski2001] age integral t(z) = ∫_0^{1/(1+z)} dx / (x H(x)) (age.tex line 343) (read); [Frieman2008] Eq. cosmic_time (sec2.tex lines 244-247): t(z) = ∫_z^∞ dz'/[(1+z') H(z')] (read)
- `big_rip_time_gyr`: [Frieman2008] sec9.tex lines 51-58: for w<-1, rho_DE ∝ a^beta, beta = -3(1+w); (t_inf - t_0) ≃ 2/(beta H0) (read)
- `cosmological_constant_lambda`: [Frieman2008] sec2.tex lines 59-62: Lambda = 8 pi G rho_VAC = -8 pi G p_VAC (read)
- `critical_density_kg_m3`: [Frieman2008] sec2.tex line 93: rho_crit = 3 H0^2 / 8 pi G (read); [Copeland2006] text after Eq. (conteq) and Eq. (acceleq): rho_c(t) = 3 H^2(t) / 8 pi G (hepthv3.tex line 632) (read)
- `de_eos`: [Linder2003] TeX label wa (expprl2.tex lines 147-150): w(a) = w0 + wa (1-a) = w0 + wa z/(1+z) (read)
- `de_mass_density_kg_m3`: [Frieman2008] Eq. q (sec2.tex line 140) with Eq. rho_scaling (sec2.tex lines 76-79) (read)
- `de_rho_ratio`: [Linder2003] TeX label wa and the sentence after it (expprl2.tex lines 147-152) (read); [Frieman2008] Eq. rho_scaling (sec2.tex lines 76-79) (read)
- `deceleration_parameter`: [Frieman2008] Eq. q (sec2.tex line 136-137): q = -a_ddot/(a H^2) = (1/2) sum_i Omega_i (1 + 3 w_i) (read)
- `hubble_parameter_km_s_mpc`: [Hogg1999] Eq. 14 (TeX label eq:ez, cosmography.tex lines 238-246): H(z) = H0 E(z) (read)
- `matter_de_equality_redshift`: [Frieman2008] Fig. growth caption (sec2.tex line 278): dark energy dominates when 1+z = (Omega_M/Omega_DE)^{1/(3w)} (read)
- `matter_radiation_equality_redshift`: [Frieman2008] sec2.tex lines 85-88: rho_M ∝ (1+z)^3, rho_R ∝ (1+z)^4 (read)
- `omega_curvature`: [Hogg1999] Eq. 14 (TeX label eq:ez, cosmography.tex lines 238-241) (read); [Frieman2008] Eq. feq (sec2.tex line 48): H^2 = 8 pi G rho / 3 - k/a^2 + Lambda/3 (read)
- `omega_de`: [Frieman2008] Eq. q (sec2.tex line 140) and Eq. rho_scaling with w = -1 (sec2.tex lines 76-79, 88-89) (read)
- `omega_matter`: [Frieman2008] Eq. q (sec2.tex line 140): Omega_i(z) = rho_i(z) / rho_crit(z); rho_M ∝ (1+z)^3 (sec2.tex line 86) (read)
- `omega_radiation`: [Frieman2008] Eq. q (sec2.tex line 140); rho_R ∝ (1+z)^4 (sec2.tex lines 87-88) (read)
- `total_eos`: [Frieman2008] Eq. q (sec2.tex line 136-137) and w_i = p_i/rho_i (sec2.tex line 68) (read)

### `kozmos.cosmology.distances`

- `angular_diameter_distance_mpc`: [Hogg1999] Eq. 18 (read)
- `angular_diameter_distance_z1_z2_mpc`: [Hogg1999] Eq. 19 (read)
- `comoving_distance_mpc`: [Hogg1999] Eq. 15 (read)
- `comoving_volume_element_mpc3_sr`: [Hogg1999] Eq. 28 (read)
- `comoving_volume_mpc3`: [Hogg1999] Eq. 29 (read)
- `distance_modulus`: [Hogg1999] Eq. 25 (read)
- `doppler_redshift`: [Hogg1999] Eq. 9 (read)
- `doppler_velocity_km_s`: [Hogg1999] Eq. 9 (read)
- `e_function`: [Frieman2008] Eq. rz (sec2.tex line 177) and Eq. rhow (sec2.tex lines 82-88) (read); [Linder2003] TeX label wa and the sentence after it (expprl2.tex lines 147-152) (read); [Brout2022] H(z) equation (TeX label eq:hz) (read); [Hogg1999] Eq. 14 (read)
- `hubble_distance_mpc`: [Hogg1999] Eq. 4 (read)
- `hubble_time_gyr`: [Hogg1999] Eq. 3 (read)
- `lookback_time_gyr`: [Hogg1999] Eq. 30 (read)
- `luminosity_distance_mpc`: [Hogg1999] Eq. 21 (read)
- `peculiar_velocity_km_s`: [Hogg1999] Eq. 10 (read)
- `transverse_comoving_distance_mpc`: [Hogg1999] Eq. 16 (read)

### `kozmos.cosmology.growth`

- `fsigma8`: [Linder2005] Eq. grot (groexpprdv2.tex line 192): the growth equation is linear and scale independent (read)
- `growth_factor`: [Linder2005] Eq. groq (groexpprdv2.tex lines 215-216) with g = delta/a (line 204) (read)
- `growth_index_gamma`: [Linder2005] Eq. gamma (groexpprdv2.tex line 366): gamma = 0.55 + 0.05 [1 + w(z=1)]; Eq. gammam1 (line 386): gamma = 0.55 + 0.02 [1 + w(z=1)] for w < -1 (read)
- `growth_log_slope`: [Linder2005] Eq. gapx (groexpprdv2.tex line 341): g(a) = exp ∫ dln a [Omega_m(a)^gamma - 1], g = delta/a (line 204) (read)
- `growth_log_slope_from_index`: [Linder2005] Eq. gapx (groexpprdv2.tex line 341) and f = 1 + dln g/dln a (read)
- `growth_suppression`: [Frieman2008] Eq. growth (sec2.tex lines 300-303), solution in matter domination delta ∝ a (read); [Linder2005] Eq. groq (groexpprdv2.tex lines 215-216) with g = delta/a (line 204), primes = d/dln a (line 210) (read)
- `growth_suppression_from_index`: [Linder2005] Eq. gapx (groexpprdv2.tex line 341): g(a) = exp ∫_0^a dln a [Omega_m(a)^gamma - 1] (read)

### `kozmos.cosmology.hubble_diagram`

- `best_absolute_magnitude`: [Brout2022] eq:likelihood (read)
- `log_likelihood`: [Brout2022] eq:likelihood, eq:sh0eslikelihood (read)
- `prepare_data`: [Brout2022] Section 4: z < ~0.01 is not used in the Hubble diagram; eq:dmuprime (read)
- `residual_vector`: [Brout2022] eq:dmu, eq:dmuprime, eq:dl (read)

### `kozmos.cosmology.scalar_field`

- `exponential_fixed_points`: [Copeland2006] TABLE I (TeX label crit, hepthv3.tex lines 3909-3946) and the eigenvalue list (lines 3949-3987, Eq. musca for point d) (read)
- `exponential_quintessence_rhs`: [Copeland2006] Eq. autoquin1, autoquin2 (hepthv3.tex lines 3831-3843) with epsilon = +1 and Gamma = 1 (line 3902) (read)
- `integrate_exponential_quintessence`: [Copeland2006] Eq. autoquin1, autoquin2 (hepthv3.tex lines 3831-3843); Eq. wphiquin, Omephiquin, weffquin (read)
- `integrate_quintessence`: [Copeland2006] Eq. phieqF (hepthv3.tex line 2586), Eq. H2 (line 2622), rho and p (lines 2614-2615), Eq. ddota (line 2625); Hdot = -(kappa^2/2)[phi_dot^2 + (1+w_m) rho_m] (line 3806) (read)
- `scalar_eos`: [Copeland2006] equation of state w_phi = (phi_dot^2 - 2V)/(phi_dot^2 + 2V) (hepthv3.tex lines 2662-2664) (read)
- `scalar_omega_phi`: [Copeland2006] Eq. Omephiquin (hepthv3.tex line 3871): Omega_phi = x^2 + y^2 (epsilon = +1) (read)
- `scalar_p`: [Copeland2006] text after Eq. (T_mu_nu): p = phi_dot^2/2 - V (hepthv3.tex line 2615) (read)
- `scalar_rho`: [Copeland2006] text after Eq. (T_mu_nu): rho = phi_dot^2/2 + V (hepthv3.tex line 2614) (read)
- `scalar_w_eff`: [Copeland2006] Eq. weffquin (hepthv3.tex line 3881): w_eff = w_m + (1-w_m) x^2 - (1+w_m) y^2 (read)
- `scalar_w_phi`: [Copeland2006] Eq. wphiquin (hepthv3.tex line 3865): w_phi = (x^2 - y^2)/(x^2 + y^2) (read)

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

### `kozmos.mechanics.energy`

- `accelerations_m_s2`: [DehnenRead2011] TeX label eqn:direct (line 255): a_i = -G sum_{j != i} m_j (x_i - x_j) / |x_i - x_j|^3; Plummer softening eq:plummer (line 1515) (read)
- `barycenter_m`: [MurrayCorreia2010] TeX label eq49 (position of the centre of mass of two bodies) (read)
- `barycenter_velocity_m_s`: [MurrayCorreia2010] TeX label eq50 and the sentence after it (dR/dt = V = constant) (read)
- `eccentricity_from_vis_viva_constant`: [MurrayCorreia2010] TeX label eq22 (h = sqrt(G(m1 + m2) a (1 - e^2))) and eq24, eq30 (read)
- `escape_speed_m_s`: [MurrayCorreia2010] TeX label eq24 (vis viva integral) (read)
- `kinetic_energy_j`: [DehnenRead2011] TeX line 1234 (virial theorem paragraph): T = sum_i (1/2) m_i |dx_i/dt|^2 (read)
- `potential_energy_j`: [DehnenRead2011] TeX label eqn:direct (line 255): H = sum p^2/2m - G sum_{j>i} m_i m_j / |x_i - x_j|; Plummer softening eq:plummer (line 1515) (read)
- `semi_major_axis_from_vis_viva_constant_m`: [MurrayCorreia2010] TeX label eq24 and eq30 (v^2 = mu (2/r - 1/a)) (read)
- `specific_angular_momentum_m2_s`: [MurrayCorreia2010] TeX label eq04 (r x dr/dt = h) (read)
- `total_angular_momentum`: [MurrayCorreia2010] TeX label eq04: r x dr/dt = h (angular momentum integral, per unit mass) (read)
- `total_energy_j`: [DehnenRead2011] TeX label eqn:direct (line 255), the Hamiltonian H = T + W (read)
- `total_momentum`: [DehnenRead2011] TeX text after eqn:direct (line 255): p_i = m_i dx_i/dt (read)
- `uniform_sphere_binding_energy_j`: [Chiang2020] TeX line 256 (after label eq:Ag): gravitational potential energy of a uniform density sphere -3 G M^2 / (5 R_p) (read)
- `virial_ratio`: [DehnenRead2011] TeX lines 1233-1236 (virial theorem): 2T + W = 0 (line 1233) with T = sum (1/2) m |dx/dt|^2 (line 1234) and W = sum m_i x_i . d2x_i/dt2 (line 1236) (read)
- `vis_viva_constant_m2_s2`: [MurrayCorreia2010] TeX label eq24 (vis viva integral): v^2/2 - G(m1 + m2)/r = C (read)

### `kozmos.mechanics.hamiltonian`

- `canonical_symmetry_variation`: [BanadosReyes2016] Eq. (invNoe) (line 421): delta_s q = eps dQ/dp, delta_s p = -eps dQ/dq (read)
- `conformal_particle_charges`: [BanadosReyes2016] conformal particle in Hamiltonian form, lines 473-475: H = p^2/2m + alpha/q^2, Q = -t H + p q / 2, K = t^2 H + 2 t Q - m q^2 / 2 (read)
- `hamilton_equations`: [BanadosReyes2016] Hamilton's equations after Eq. (Poissonpoint), lines 402-403: qdot = dH/dp, pdot = -dH/dq (read)
- `hamiltonian_from_lagrangian`: [BanadosReyes2016] Eq. (ha) (line 393) together with Eq. (action0) (line 97) (read)
- `integrate_hamiltonian`: [BanadosReyes2016] Hamilton's equations, lines 402-403 (after Eq. (Poissonpoint)) (read)
- `observable_derivative`: [BanadosReyes2016] Eq. (GH) (line 409): dG/dt = [G, H] + dG/dt (explicit) (read)
- `poisson_bracket`: [BanadosReyes2016] Eq. (Poissonpoint) (line 397): [F, G] = dF/dq dG/dp - dF/dp dG/dq (read)

### `kozmos.mechanics.integrators`

- `forest_ruth_step`: [Chin2003] TeX label fr (line 167): T4(eps) = T2(w1 eps) T2(w0 eps) T2(w1 eps), s = 2^(1/3), w1 = 1/(2 - s), w0 = -s/(2 - s); T2 of label second (line 160) is drift-kick-drift (read)
- `integrate`: [DehnenRead2011] TeX label eq:leapfrog (line 437), eqn:kdk:a (line 445) for scheme 'kdk' (read); [Chin2003] TeX label second (line 160) for scheme 'dkd' and label fr (line 167) for scheme 'forest_ruth' (read)
- `leapfrog_dkd_step`: [DehnenRead2011] TeX label eqn:opsplit (line 393), eqn:kick:drift (line 406); footnote after eqn:kdk:a (line 462) names the drift-kick-drift version (read)
- `leapfrog_kdk_step`: [DehnenRead2011] TeX label eq:leapfrog (line 437) and eqn:kdk:a (line 445): kick, drift, kick; eqn:kdk:b (line 471) (read)
- `relative_drift`: [DehnenRead2011] caption of figure fig:force_schematic (TeX lines 544-547): 'Fractional change in energy for the Kepler problem' (read)
- `symplectic_step_separable`: [Chin2003] TeX label second (line 160): e^(eps T/2) e^(eps V) e^(eps T/2) with T = p d/dq the drift and V = F d/dp the kick (lines 124-125), i.e. drift-kick-drift; order 4: label fr (line 167) (read); [DehnenRead2011] TeX label eqn:opsplit (line 393) and eqn:kick:drift (line 406): H = T(p) + V(x) and the two exact maps (drift, kick) (read)

### `kozmos.mechanics.lagrangian`

- `action`: [BanadosReyes2016] Eq. (action0) (line 97): I[q] = integral dt L(q, qdot, t) (read)
- `euler_lagrange_accelerations`: [BanadosReyes2016] Eq. (eom) (line 101) (read)
- `euler_lagrange_residual`: [BanadosReyes2016] Eq. (eom) (line 101) (read)
- `generalized_forces`: [BanadosReyes2016] Eq. (eom) (line 101): the term dL/dq^i (read)
- `generalized_momenta`: [BanadosReyes2016] Eq. (eom) (line 101) and Eq. (qn) (line 236): the combination dL/d(qdot^i) (read)
- `inertia_matrix`: [BanadosReyes2016] Eq. (eom) (line 101) (read)
- `integrate_lagrangian`: [BanadosReyes2016] Eq. (eom) (line 101) (read)
- `jacobi_function`: [BanadosReyes2016] Eq. (qn) (line 236) with Eq. (dr) (line 166) and the boundary term K = -eps L of Eq. (ttt) (line 173-175) (read)

### `kozmos.mechanics.noether`

- `boost_boundary_term`: [BanadosReyes2016] Eq. (sym3) (line 226); derivation in this project (see boost_generator) (read)
- `boost_generator`: [BanadosReyes2016] Eq. (sym), Eq. (sym3) (lines 139, 226) and Eq. (qn) (line 236) (read)
- `noether_charge`: [BanadosReyes2016] Eq. (qn) (line 236): Q = K - dL/d(qdot^i) delta_s q^i (read)
- `planar_rotation_generator`: [BanadosReyes2016] Eq. (rot) (line 159), specialised to a rotation about the third axis in the plane (read)
- `rotation_generator`: [BanadosReyes2016] Eq. (rot) (line 159): delta_s r = alpha x r; Q = -alpha . (m r x rdot) Eq. (angular) (line 247) (read)
- `symmetry_defect`: [BanadosReyes2016] Eq. (sym) (line 139) and Eq. (sym3) (line 226): a symmetry has delta_s L = dK/dt for any q(t) (read)
- `symmetry_variation`: [BanadosReyes2016] Eq. (sym) (line 139) and Eq. (sym3) (line 226): delta I = integral dK/dt; delta_s L = dL/dq.delta_s q + dL/dv.d(delta_s q)/dt from the variation of Eq. (action0) and line 193 (read)
- `t_translation_boundary_term`: [BanadosReyes2016] K = -eps L of Eq. (ttt) (line 173-175) and line 249 (K = -eps L) (read)
- `t_translation_generator`: [BanadosReyes2016] Eq. (dr) (line 166), Eq. (trans) (line 187), K = -eps L from Eq. (ttt) (line 173-175) (read)
- `translation_generator`: [BanadosReyes2016] Eq. (sss) (line 127): translations of x and y are a symmetry; Eq. (Ip2) (line 374): joint translation of two particles (read)
- `weyl_boundary_term`: [BanadosReyes2016] line 301: 'the boundary term is K = -tL' for the Weyl symmetry (read)
- `weyl_generator`: [BanadosReyes2016] line 292: delta_s x = -eps t xdot + (eps/2) x, for the action of Eq. (Iconformal) (line 264) (read)

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

### `kozmos.relativity.curvature`

- `einstein_tensor`: [Carroll1997] Section 3, Eq. 3.95 (TeX line 4591) (read)
- `kretschmann_scalar`: [Carroll1997] Section 7, Eq. 7.30 (TeX line 9287, left-hand side only) and Eq. 3.76 (TeX line 4406) (read)
- `ricci_scalar`: [Carroll1997] Section 3, Eq. 3.92 (TeX line 4572) (read)
- `ricci_tensor`: [Carroll1997] Section 3, Eq. 3.90 (TeX line 4556) (read)
- `riemann_tensor`: [Carroll1997] Section 3, Eq. 3.67 (TeX line 4242) (read)

### `kozmos.relativity.geodesics`

- `christoffel_symbols`: [Carroll1997] Section 3, Eq. 3.21 (TeX line 3411) (read)
- `geodesic_acceleration`: [Carroll1997] Section 3, Eq. 3.47 (TeX line 3853) (read)
- `integrate_geodesic`: [Carroll1997] Section 3, Eq. 3.47 (TeX line 3853) (read)
- `killing_conserved_quantity`: [Carroll1997] Section 5, Eq. 5.43 (TeX line 7582) and Section 7, Eq. 7.38 (TeX line 9369) (read)
- `killing_equation_defect`: [Carroll1997] Section 5, Eq. 5.41 and Eq. 5.31 (TeX lines 7564, 7417) (read)
- `normalize_tangent`: [Carroll1997] Section 7, Eq. 7.39 (TeX line 9376) and Section 1, Eq. 1.99 (TeX line 1676) (read)
- `stationary_axisymmetric_constants`: [Carroll1997] Section 7, Eq. 7.43 and Eq. 7.44 (TeX lines 9418, 9422) (read)
- `tangent_norm_squared`: [Carroll1997] Section 7, Eq. 7.39 (TeX line 9376) and Section 1, Eq. 1.99 (TeX line 1676) (read)

### `kozmos.relativity.metrics`

- `flrw_metric`: [Carroll1997] Section 8, Eq. 8.7 (TeX line 11409) (read)
- `kerr_eddington_finkelstein_metric`: [Visser2007] section 'Kerr's original coordinates', Eq. E:K1 (TeX lines 203-215) (read)
- `kerr_metric`: [Abramowicz2013] Kerr metric in Boyer-Lindquist coordinates, Eq. kerr-metric (TeX lines 543-580) (read)
- `metric_inverse`: [Carroll1997] Section 2, Eq. 2.28 (TeX line 2640) (read)
- `minkowski_metric`: [Carroll1997] Section 1, Eq. 1.8 (TeX line 447) (read)
- `reissner_nordstrom_metric`: [Carroll1997] Section 7, Eq. 7.110 and Eq. 7.111 (TeX lines 10602, 10607) (read)
- `schwarzschild_eddington_finkelstein_metric`: [Carroll1997] Section 7, Eq. 7.69 (TeX line 9861) (read)
- `schwarzschild_metric`: [Carroll1997] Section 7, Eq. 7.29 (TeX line 9217) (read)

### `kozmos.relativity.special`

- `aberrated_angle_rad`: [Carroll1997] Section 1, Eq. 1.14, 1.18 and 1.21 (TeX lines 489, 532, 566) (read)
- `add_velocities_collinear_m_s`: [Carroll1997] Section 1, Eq. 1.18 and Eq. 1.20 (TeX lines 532, 560) (read)
- `add_velocities_m_s`: [Carroll1997] Section 1, Eq. 1.98 (TeX line 1671), Eq. 1.11 (TeX line 471), Eq. 1.14 (TeX line 489) and Eq. 1.18 (TeX line 532) (read)
- `boost_four_vector`: [Carroll1997] Section 1, Eq. 1.11 (TeX line 471) and Eq. 1.18 (read)
- `boost_matrix`: [Carroll1997] Section 1, Eq. 1.18, 1.14 and 1.21 (TeX lines 532, 489, 566) (read)
- `contracted_length_m`: [Carroll1997] Section 1, Eq. 1.21 (TeX line 566) (read)
- `dilated_time_s`: [Carroll1997] Section 1, Eq. 1.97 (TeX line 1645) (read)
- `doppler_factor`: [Carroll1997] Section 1, Eq. 1.14, 1.18 and 1.21 (TeX lines 489, 532, 566) (read)
- `doppler_factor_source_frame`: [Carroll1997] Section 1, Eq. 1.14, 1.18 and 1.21 (TeX lines 489, 532, 566) (read)
- `energy_from_momentum_j`: [Carroll1997] Section 1, Eq. 1.99 and Eq. 1.100 (TeX lines 1676, 1686) (read)
- `four_momentum`: [Carroll1997] Section 1, Eq. 1.100 and Eq. 1.101 (TeX lines 1686, 1707) (read)
- `invariant_mass_kg`: [Carroll1997] Section 1, Eq. 1.99 and Eq. 1.100 (TeX lines 1676, 1686) (read)
- `kinetic_energy_j`: [Carroll1997] Section 1, Eq. 1.101 (TeX line 1707) (read)
- `lorentz_factor`: [Carroll1997] Section 1, Eq. 1.21 (TeX line 566) (read)
- `minkowski_inner_product`: [Carroll1997] Section 1, Eq. 1.8 and Eq. 1.9 (TeX lines 447, 452) (read)
- `proper_time_s`: [Carroll1997] Section 1, Eq. 1.97 (TeX line 1645) (read)
- `rapidity`: [Carroll1997] Section 1, Eq. 1.20 (TeX line 560) (read)
- `relativistic_momentum`: [Carroll1997] Section 1, Eq. 1.100 and Eq. 1.101 (TeX lines 1686, 1707) (read)
- `spacetime_interval_squared_m2`: [Carroll1997] Section 1, Eq. 1.9 (TeX line 452) (read)
- `total_energy_j`: [Carroll1997] Section 1, Eq. 1.100 and Eq. 1.101 (TeX lines 1686, 1707) (read)
- `velocity_from_rapidity_m_s`: [Carroll1997] Section 1, Eq. 1.20 (TeX line 560) (read)

### `kozmos.relativity.weak_field`

- `geodetic_precession_circular_rad_s`: [Will2014] subsubsection 'Geodetic precession', Eq. E41 (label at TeX line 3939) (read)
- `geodetic_precession_vector_rad_s`: [Will2014] subsubsection 'Geodetic precession', Eq. E41 (label at TeX line 3939) (read)
- `light_deflection_angle_rad`: [Will2014] subsubsection 'The deflection of light', Eq. E28 (label at TeX line 3014) (read)
- `perihelion_advance_per_orbit_rad`: [Will2014] subsection 'The perihelion shift of Mercury', Eq. E33 (label at TeX line 3430) (read)
- `perihelion_advance_rate_rad_s`: [Will2014] subsection 'The perihelion shift of Mercury', Eq. E33 and E34 (labels at TeX lines 3430, 3475) (read)
- `schwarzschild_clock_ratio`: [Carroll1997] Section 7, Eq. 7.58 (TeX line 9688) (read)
- `shapiro_round_trip_delay_s`: [Will2014] subsubsection 'The time delay of light', Eq. E31 (label at TeX line 3193) (read)
- `static_clock_ratio`: [Carroll1997] Section 7, Eq. 7.28 (TeX line 9208) and Eq. 7.58 (TeX line 9688) (read)

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
- [Abramowicz2013]: 22
- [Armitage2011]: 3
- [Astropy2022]: 8
- [Babusiaux2018]: 2
- [BanadosReyes2016]: 27
- [Bardeen1972]: 1
- [Bludman2010]: 13
- [Brout2022]: 8
- [CODATA2022]: 3
- [CalabrettaGreisen2002]: 6
- [Carroll1997]: 48
- [Chabrier2003]: 8
- [ChevallierPolarski2001]: 1
- [Chiang2020]: 1
- [Chin2003]: 3
- [Cohen2003]: 3
- [Copeland2006]: 11
- [CraigChambers2026]: 3
- [DehnenRead2011]: 11
- [EHT2019]: 4
- [ERFA2023]: 33
- [FITS2018]: 5
- [ForemanMackey2013]: 3
- [Frieman2008]: 17
- [Groth1986]: 3
- [Hierarch2009]: 1
- [HoffleitWarren1991]: 1
- [Hogg1999]: 21
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
- [Linder2003]: 3
- [Linder2005]: 7
- [Liu1995]: 11
- [Lupton2004]: 3
- [Mamajek2015]: 2
- [McKee2007]: 21
- [Moffat1969]: 1
- [Morton1991]: 3
- [MurrayCorreia2010]: 12
- [Page2005]: 13
- [Paxton2013]: 2
- [Photutils2026]: 3
- [PhotutilsPSF2026]: 2
- [Pols2011]: 1
- [Prsa2016]: 4
- [Pugliese2011]: 2
- [Reynolds2021]: 4
- [SExtractorDocs2026]: 2
- [SIBrochure9]: 2
- [Salpeter1955]: 4
- [Sathyaprakash2009]: 7
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
- [Visser2007]: 2
- [VollmannEversberg2006]: 2
- [Will2014]: 6
- [Wright2010]: 3
- [Zaghloul2011]: 1
- [Zinnecker2007]: 1
- [vanDokkum2001]: 1

## Functions awaiting source reading

