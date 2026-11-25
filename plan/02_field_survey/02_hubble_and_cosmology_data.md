# Hubble constant and cosmology: data and current status

Next to each piece of information in this document, it is stated how it was obtained: **read** (read directly from the source on 25 November 2026), **survey-page** (the web survey reported it by opening the page), **survey-summary** (the survey saw it only in a search result summary). The last two are not verified separately.

## 1. Pantheon+SH0ES data release

The project's primary dataset.

- Location: `https://github.com/PantheonPlusSH0ES/DataRelease`, `Pantheon+_Data/4_DISTANCES_AND_COVAR/` (read; GitHub API).
- Files (read):
  - `Pantheon+SH0ES.dat` (579 kB): 1701 rows, 47 columns.
  - `Pantheon+SH0ES_STAT+SYS.cov` (33 MB): the first line is the dimension (`1701`), followed by one matrix element per line.
  - `Pantheon+SH0ES_STATONLY.cov` (32 MB).
  - `README`.
- Contents of the table (read): 1701 light curve rows, 1543 unique supernova identifiers (`CID`); `zHD` range 0.00122-2.26137; 77 rows with `IS_CALIBRATOR = 1`.
- Columns to be used in the analysis: `zHD` (Hubble diagram redshift), `zHEL`, `m_b_corr` and `m_b_corr_err_DIAG` (standardized magnitude), `MU_SH0ES`, `CEPH_DIST` (Cepheid host distance modulus), `IS_CALIBRATOR`, `USED_IN_SH0ES_HF`, `x1`, `c`, `HOST_LOGMASS`, `VPEC`, `MWEBV`. The column definitions will be read from the `README` file in the repository and from the papers; the descriptions in this document were inferred from the column names.
- License (read): the repository has no license file defined (GitHub API `license: null`). The data are not redistributed; a script downloads them. Citation: [Scolnic2022; Brout2022] and, for the ladder, [Riess2022].
- The last push to the repository is dated 21 December 2022 (read); the data are static, but a digest of the downloaded files is still recorded.

## 2. Other supernova compilations

- Pantheon (2018): arXiv:1710.00845 (survey-page).
- Union3: arXiv:2311.12098; 2087 supernovae (survey-summary). Union3.1: arXiv:2601.19424 (survey-summary). Data location not verified.
- DES-SN5YR: arXiv:2401.02929 (survey-page); recalibration arXiv:2511.07517 (survey-summary). Data location not verified.
- Open Supernova Catalog: according to the survey, the project has ended and its data are archived (survey-summary).

Differences between compilations affect dark energy results; a cross-check with a second compilation is a later step in the roadmap.

## 3. Distance ladder data

- SH0ES Cepheid photometry: location of the tables not verified [Riess2022]. The `CEPH_DIST` column in the Pantheon+SH0ES table provides the host distances ready-made; reanalysis at the Cepheid level is a separate project.
- CCHP TRGB and JAGB distances: data location not verified [Freedman2019; Freedman2025].
- Cosmicflows-4: arXiv:2209.11238 (survey-page); data location not verified.
- NED-D: not verified.
- Hubble's 1929 table: the galaxy distance and velocity table in the original paper [Hubble1929]. The survey found no machine-readable copy; the table will be transcribed by hand from the paper and the transcription will be checked by a second reading.

## 4. BAO, CMB and independent measurements

- DESI DR1 BAO cosmology [DESI2025] (bibliographic record verified with Crossref). DESI DR2 BAO: arXiv:2503.14738 [DESI2025DR2] (bibliographic record verified with the arXiv API).
- DESI open data: according to the survey, DR1 was released on 19 March 2025, and access is by bulk file download (survey-page).
- Planck 2018 cosmological parameters [Planck2020]; the chains are in the Planck Legacy Archive, and the survey could not read the archive page.
- ACT DR6, SPT-3G, cosmic chronometer compilations, lens time delays: not surveyed.
- Standard siren: GW170817 [Abbott2017].

## 5. Current values of the Hubble constant

- SH0ES (Cepheid + SN Ia): `73.04 ± 1.04 km s⁻¹ Mpc⁻¹` [Riess2022] (survey-page: arXiv:2112.04510).
- Pantheon+ with SH0ES together: `73.5 ± 1.1` (`w0CDM`), `73.3 ± 1.1` (`w0waCDM`) [Brout2022] (survey-page).
- CCHP, TRGB only: `70.39 ± 1.22 ± 1.33 ± 0.70` [Freedman2025] (survey-page). With JWST data only, TRGB `68.81` and JAGB `67.80` (survey-summary). The values changed between versions of the paper.
- SH0ES HST+JWST combination `73.18 ± 0.88` (survey-summary; the primary paper was not found).
- The early-universe (Planck ΛCDM) value was not read from its source in this work.

The survey's conclusion: the tension is not resolved even in the newest publications; the difference between the Cepheid and TRGB ladders is a separate open debate; DESI DR2 reports that the BAO and CMB combination favors dynamical dark energy at the 3.1σ level (survey-summary).

Review sources: [DiValentino2021] (bibliographic record verified); the CosmoVerse report arXiv:2504.01669 (survey-summary).

## 6. Consequences for the project

1. Pantheon+SH0ES alone is sufficient for an end-to-end `H0` analysis: the Hubble flow supernovae, the Cepheid-anchored calibrators and the full covariance are in the same release.
2. The goal is to reproduce the published result. No other data are added until the reproduction holds.
3. Current values change quickly. No `H0` value is coded in the project as "current"; every value is cited with its bibliographic record and version.
4. Reading to be done: the full text of [Riess2022] and [Brout2022], the `README` file of the data repository, the 2025-2026 reviews.
