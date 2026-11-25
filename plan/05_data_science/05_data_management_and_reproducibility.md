# Data management and reproducibility

The acceptance criterion of this project is that a result can be obtained again together with the data, code version and parameters that produced it. The principles follow the compilation of good enough practices in scientific computing and the FAIR principles [Wilson2017; Wilkinson2016].

## 1. Data layers

- `data/raw/`: files downloaded from an archive or coming from the telescope. Read-only; no code overwrites them. They are outside version control; the SHA-256 digest and the source of every file are kept in a record file.
- `data/interim/`: files that are reduced but not yet science products. They can be deleted and regenerated.
- `data/products/`: catalogs, light curves, spectra, fit results. Next to every product sits the record of the run that produced it.
- `data/reference/`: small files used by the tests; they are under version control.

## 2. Run record

Every analysis run produces a record: code version (git digest, a warning for a dirty working tree), digests of the input files, all parameters, random seed, dependency versions, start and end time, digests of the output files. The record is written next to the output as JSON.

Rule: the record is written first, the intermediate files are cleaned afterwards. Before a product is overwritten, the digest of its old version is entered in the record.

## 3. Configuration

- Parameters are in a configuration file (TOML), not in the code; the code documents the defaults, the file defines the run.
- Instrument definitions (gain, read noise, pixel scale, header key mapping) are separate files.
- There are no magic numbers: each threshold, window and tolerance value is named and its justification is written.

## 4. Workflow

- A long job is divided into steps; a step does not run again if the digest of its input has not changed.
- **Gate before expensive work:** before a run that takes hours (a large MCMC, batch reduction, a large download) starts, a small check script verifies from the data that the work is really needed: whether the output already exists, whether the input changed, whether a short trial run gives the expected result. The check produces output; if the exit code is not zero, the job does not start.
- Long runs save intermediate state and continue from where they stopped; chains are streamed to disk.
- First a small-scale trial (short chain, data subset), then the full run.

## 5. Environment

- Dependencies are defined in `pyproject.toml` with a lower bound, and in the lock file with the exact version.
- The version of the oracle libraries is written in the header of the reference output files; when an oracle is updated, a difference report is produced.
- Numerical results can depend on the operating system and the BLAS implementation in the last digits; tolerances are set accordingly.

## 6. Big data

- Out-of-core work: memory-mapped FITS, reading in chunks, Parquet and lazy queries.
- Sky partitioning: parallel processing and cross-matching with HEALPix cells.
- Computation on the archive side: whatever filtering is possible is done in the ADQL query, and only the needed rows and columns are downloaded.
- Order of speeding up: first correct and vectorized NumPy, measurement (profiling), then Numba or JAX if needed. No optimization is made without measurement.

## 7. Reporting

- Every result value is reported with its source: from which file and which run it was read.
- The uncertainty is always given with the value; statistical and systematic separately.
- Figures are produced by script; the data of the figure is saved separately. The production of figures and tables follows the figure standard of the project (journal profile, color-blind-safe palette, direct labeling).
- Negative results and failed attempts are also recorded.

## 8. Citation and license

- The citation requirement of every dataset used is kept with the data in the `archives` module and added automatically to the analysis report.
- `kozmos.bibliography_report()` lists the publications on which the functions called in an analysis rely.
- Software used as an oracle is also cited [Astropy2013; Astropy2018; Astropy2022; Harris2020; Virtanen2020].
