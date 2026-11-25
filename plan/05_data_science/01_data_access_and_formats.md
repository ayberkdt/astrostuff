# Data access and formats

Target modules: `io/`, `archives/`.

The current release status of the archives is in [../02_field_survey/03_data_archives.md](../02_field_survey/03_data_archives.md).

## 1. FITS

The common file format of astronomical data. The current standard version is 4.0; the definition published in a journal describes version 3.0 [Pence2010]. Version 4.0 has no journal reference; the document of the NASA FITS support office is cited [FITS2018].

- Structure: 2880-byte blocks; header (80-character cards: key, value, comment) and data unit; primary HDU and extensions (image, binary table, ASCII table).
- Mandatory keys: `SIMPLE`, `BITPIX`, `NAXIS`, `NAXISn`, `END`. Scaling: `BSCALE`, `BZERO` (the way an unsigned 16-bit integer is stored).
- Conventions: `CONTINUE` long strings, `HIERARCH`, multi-extension files, compressed images (tile compression), `CHECKSUM`/`DATASUM`.
- Time keys: `DATE-OBS`, `MJD-OBS`, `TIMESYS`, `EXPTIME`; the distinction between the start and the midpoint of the exposure [Rots2015].
- World coordinate system: [../03_mathematics/01_spherical_astronomy_and_transforms.md](../03_mathematics/01_spherical_astronomy_and_transforms.md), Section 8 [GreisenCalabretta2002; CalabrettaGreisen2002; Greisen2006].

**Implementation scope.** The reader and writer are written from scratch: header parsing, primary image, image extension, binary table. Compressed images and variable-length array columns are the second stage. Oracles: `astropy.io.fits` and `fitsio`. Test: the same file gives a byte-for-byte identical array with both readers; the written file passes the `astropy` verifier.

Real headers do not follow the standard: amateur software and old instruments write missing or faulty keys. The reader is tolerant, the writer is strict; every deviation is logged as a warning.

## 2. Table and catalog formats

- VOTable (XML based; the current recommendation is 1.5), CSV/ECSV (CSV that carries metadata), FITS binary table, Parquet (large catalogs), HDF5 (simulation output).
- Column metadata: unit, UCD (content descriptor), description; a column without a unit produces a warning when read.
- Large catalogs: HEALPix-partitioned Parquet (the HATS format) and lazy evaluation.

## 3. Virtual Observatory protocols

IVOA standards give common access to archives. Recommendation versions read from the IVOA document index during the survey: TAP 1.1 (2019), ADQL 2.1 (2023), SIA 2.0 (2015), SSA 1.1 (2012), Simple Cone Search 1.03 (2008), VOTable 1.5 (2025), HiPS 1.0 (2017), MOC 2.0 (2022), DataLink 1.1 (2023).

- **TAP:** catalog queries over HTTP; synchronous and asynchronous job model (create a job, poll the status, fetch the result).
- **ADQL:** the astronomy extension of SQL; geometric functions (`CONTAINS`, `POINT`, `CIRCLE`, `DISTANCE`), cross-matching.
- **Cone Search, SIA, SSA:** search for catalogs, images and spectra by position.
- **DataLink:** finding the data products attached to a record (Gaia spectra and light curves are obtained this way).
- **HiPS and MOC:** hierarchical image tiles and sky coverage maps.

**Implementation scope.** `archives/tap.py` writes the synchronous and asynchronous TAP client on top of the standard library; the VOTable parser is in `io/votable.py`. Oracles: `pyvo` and `astroquery` [Ginsburg2019].

## 4. Archive-specific access

A thin module for each archive: endpoint, table names, quality flags, recommended cuts and citation text.

- **Gaia:** TAP; `gaia_source` columns, `ruwe` and other quality measures, parallax zero point [Vallenari2023; Lindegren2021].
- **Pantheon+SH0ES:** the distance modulus table and covariance matrices from the GitHub data release [Scolnic2022; Brout2022].
- **MAST:** TESS and Kepler light curves, target pixel files, full-frame cutouts [Ricker2015].
- **SDSS, DESI:** spectra and redshifts [York2000].
- **ZTF:** light curves [Bellm2019].
- **NASA Exoplanet Archive:** planet parameters through TAP. According to the archive's documentation, most of the tables in the old `nstedAPI` interface have been removed; new code targets TAP.
- **JPL Horizons and SBDB:** ephemerides and minor body elements.
- **SIMBAD, VizieR, NED:** name resolution, published tables, extragalactic database.
- **AAVSO, Minor Planet Center:** variable star observations and astrometry submission.

## 5. Cache and source record

- Every download is written to a local cache; the key is a digest of the query text, the value is the metadata kept together with the data: URL, query, date, file digest (SHA-256), data release of the archive.
- The same query does not go to the network a second time; refreshing is an explicit argument.
- The data release of the archive (Gaia DR3, SDSS DR-n) is embedded in the query; the "latest" alias is not used, because the result changes over time.
- The network layer obeys rate limits, retries, and leaves no partial file on error.

## 6. Own observation data

Directory layout and file naming convention for an observing night; mapping of header keys to a common schema with an instrument definition file (gain, read noise, pixel scale, filter names); night log. The instrument definition is written a single time, and the reduction pipeline reads from it every night.
