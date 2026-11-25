# Method and limits of the survey

How far any part of the plan can be trusted depends on how the information was obtained. There are four levels of confidence.

## 1. Read from the source and recorded

- **Bibliographic records.** Every entry in `bibliography/references.bib` was checked against an external record with `verify.py`: DOIs against Crossref, arXiv identifiers against the arXiv API, ISBNs against Open Library. The check compares title similarity and year; the result is in `verification_record.json`. This check shows that the record exists and that the title, year and journal are written correctly; it does not show that the content of the source is correctly reproduced in the plan.
- **Package versions.** The versions and upload dates of the 100 packages in `packages.txt` were read from the PyPI JSON API.
- **Pantheon+SH0ES data release.** The file list, row counts, column names, covariance dimension and license status were read from GitHub.
- **Package name.** It was queried that the name `kozmos` is free on PyPI.
- **Equations of the implemented functions.** The texts of the following sources were opened and the coded equations were taken from them: Hogg 1999, Brout et al. 2022, Hogg, Bovy and Lang 2010, Foreman-Mackey et al. 2013, VanderPlas 2018, Prša et al. 2016 and Mamajek et al. 2015 (arXiv TeX sources); CODATA 2022 (NIST `allascii.txt`); the ERFA source files (`cal2jd.c`, `jd2cal.c`, `era00.c`, `gmst06.c`, `obl06.c`, `sepp.c`, `s2c.c`, `c2s.c`, `anp.c`, `anpm.c`, `icrs2g.c`, `hd2ae.c`, `ae2hd.c`, `erfam.h`); the `README` file of the Pantheon+SH0ES data repository. The status per function is in [../STATUS.md](../STATUS.md).

## 2. Reported by the web survey by opening the page

A six-branch survey was carried out (core ecosystem, positional astronomy, data reduction, cosmology, statistics and archives, astrophysical domain codes). For each piece of information, the survey reported the address it read. Maintenance status, archive releases, IVOA standard versions and current `H0` values are at this level. This information has not been verified a second time; in this document it is marked "according to the survey" or "survey-page".

## 3. Seen by the survey only in a search summary

Union3, the DES-SN5YR recalibration, the CosmoVerse report, the combined SH0ES HST+JWST value and the JWST TRGB/JAGB values are at this level ("survey-summary"). They must be read from their source before use.

## 4. Recalled information

The equations, algorithm steps, numerical coefficients, equation and section numbers, typical magnitudes and the relative merits of methods in the mathematics, physics and data science sections come from the knowledge of the person who wrote the plan; apart from the sources listed above, the source texts were not opened and read. Citations show where to look for the equation; they do not show that the equation was transcribed from that source.

For this reason, before a function is written, the relevant section of the cited source is opened, the equation is taken from it, and the equation number is written into the `@source` record from that reading. If the equation in the plan differs from the one in the source, the source is correct and the plan is corrected.

## What the survey could not reach

- Some of the publisher pages (A&A, IERS, USNO, ESO, MPC, GWOSC, Euclid, ZTF, Planck Legacy Archive) refused automated access or returned empty.
- The PyPI HTML pages often failed to load; the versions were therefore read again from the JSON API.
- Most of the astrophysical domain codes branch could not be completed; the bibliographic records of the textbooks come not from the survey but from the Crossref and Open Library check.
- A few arXiv identifiers that the survey tried from memory turned out to be unrelated papers and were discarded. That identifiers and DOIs must not be written from memory, and that what is written must be checked, is shown by this.

## Sources with missing bibliographic records

Most of the topics whose records were missing in the first survey were later added to the bibliography and checked (the Hipparcos catalog, the SIP convention, HEALPix, the Cash statistic, NUTS, VSOP87, seeing, the lunar brightness model, the Moffat profile, the refractive index of air, the Landy-Szalay estimator, FITS 4.0). The record for the ELP lunar theory is still missing.

The records of sources that carry no DOI, arXiv identifier or ISBN will be verified by hand; the current list is in the "To be verified by hand" section of the file `bibliography/VERIFICATION_SUMMARY.md`. The ISBN of the [IERS2010] record could not be found in Open Library.

## Re-verification

```
python bibliography/verify.py
```

The script reruns the three checks and updates the record. Package versions and archive statuses change over time; the dated information in this document cannot be considered current until it is refreshed by a new run.
