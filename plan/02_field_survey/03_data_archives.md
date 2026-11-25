# Data archives

The release status of the archives comes from a web survey and is dated 25 November 2026. **survey-page** is information the survey reported by opening the page; **not verified** is information the survey could not read or did not cover. None of the release information in this document is verified separately; when connecting to an archive for the first time, the release is read from the archive's own page.

## Astrometry and star catalogs

- **Gaia** (ESA). The current release is DR3 [Vallenari2023]. ESA's DR4 page, in its update of 14 September 2026, says in the future tense that the data "will be released" and gives no date (survey-page). Access: TAP/ADQL and DataLink, `gea.esac.esa.int/archive`. Use in the project: astrometric and photometric reference catalog, HR diagram, cluster membership, Cepheid parallaxes. When DR4 is out, the release name embedded in queries changes; tests are pinned to DR3.
- **2MASS, WISE, Pan-STARRS:** not verified [Skrutskie2006].

## Spectroscopic surveys

- **SDSS.** The survey saw the name "Data Release 20" on the page `sdss.org/dr20` (survey-page); the release date and access details were not read [York2000].
- **DESI.** DR1 19 March 2025, early data release 13 June 2023; access by bulk file download (survey-page).

## Imaging surveys and the time domain

- **Vera C. Rubin Observatory (LSST).** Data Preview 1: commissioning data, about 7 fields, observations from late 2024. "Early Data Preview 2": observations from April 2025 to January 2026 (survey-page). The status of DR1 is not verified.
- **DESI Legacy Imaging Surveys.** DR11 was seen in the navigation menu (survey-page); the release details were not read.
- **ZTF.** The open data releases page returned 404 in the survey; not verified [Bellm2019].
- **Euclid.** Not verified.

## Space telescopes (MAST)

- **TESS.** Access: `astroquery.mast`, bulk download, TESSCut, cloud access; the latest data release note on the MAST page is dated 12 November 2026 (survey-page) [Ricker2015].
- **Kepler, HST, JWST:** not verified.

## Exoplanets and small bodies

- **NASA Exoplanet Archive.** The recommended interface is TAP; most tables have been removed from the old `nstedAPI` interface (survey-page).
- **JPL Horizons and SBDB, Minor Planet Center:** not verified (the MPC page returned 403 in the survey).

## General services

- **SIMBAD, VizieR, CDS X-Match, NED, IRSA, HEASARC, ESO archive, AAVSO, SkyView:** not verified.
- **GWOSC** (gravitational wave open data): certificate error in the survey; the address `gwosc.org` will be tried.

## IVOA standards

The survey read the following recommendation versions from the IVOA document index (survey-page): TAP 1.1 (27 September 2019), ADQL 2.1 (15 December 2023), SIA 2.0 (23 December 2015), SSA 1.1 (10 February 2012), Simple Cone Search 1.03 (22 February 2008; 2.0 working draft), VOTable 1.5 (16 January 2025; 1.6 at the proposed recommendation stage), UCD1+ 1.3 (27 May 2018), HiPS 1.0 (19 May 2017), MOC 2.0 (27 July 2022), DataLink 1.1 (15 December 2023).

`archives/tap.py` and `io/votable.py` target TAP 1.1, ADQL 2.1 and VOTable 1.5.

## Endpoints tried on 25 November 2026

While the code was being written, the following services were queried directly and the results observed: IRSA TAP (`gaia_dr3_source` and 2MASS tables; works, CSV), NASA SkyView (DSS2, 2MASS, WISE cutouts; works, FITS), the Gaia AIP mirror (works, VOTable). The `tap-server/tap/sync` address of the ESA Gaia archive returned an HTML landing page in response to a query. The CDS TAPVizieR and hips2fits services could not be reached. Details: [../05_data_science/06_optical_and_infrared_observing.md](../05_data_science/06_optical_and_infrared_observing.md).

## Priority order

According to the roadmap, the archive modules are written in this order:

1. Pantheon+SH0ES (flat file download; TAP not needed).
2. Gaia TAP (astrometric reference and HR diagram).
3. MAST TESS light curves.
4. SDSS spectra.
5. JPL Horizons (ephemeris oracle).
6. NASA Exoplanet Archive, SIMBAD, VizieR.

Before first use, each archive's endpoint, data release and citation terms are read from its own documentation and written at the top of the module with the date.
