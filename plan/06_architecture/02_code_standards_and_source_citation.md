# Code standards and source citation

Every public function is tied to the published source of the equation or algorithm it implements. The tie is made in three places, and all three are checked by machine: the docstring, the `@source` decorator and the test.

## The source citation system

### A single bibliography file

All sources are kept in BibTeX format in the file `bibliography/references.bib`. The key format is `FirstAuthorSurnameYear` (`Hogg1999`), with `Hogg1999a` for the same author and year; for institutional documents, a short name and the year (`IERS2010`, `SOFA2021`). Code, documents and tests use this key; no bibliographic entry is written anywhere else.

Every entry has a `doi`, or failing that an `eprint` (arXiv) or an `isbn`. `bibliography/verify.py` queries the DOIs at Crossref, compares the title and the year, and writes the result to the file `verification_record.json`. An entry that appears unverified in the record can be used in code, but its bibliographic details are not counted as verified.

### Docstring

The NumPy docstring format is used. The `References` section is mandatory and gives not only the key but also the location inside the source: equation number, section, page or table.

```python
@source("Hogg1999", where="Eq. 15", status="read")
def comoving_distance_mpc(z, cosmology):
    """Compute the line-of-sight comoving distance.

    D_C = D_H * ∫_0^z dz' / E(z'),  D_H = c / H0

    Parameters
    ----------
    z : float or ndarray
        Redshift, z >= 0.
    cosmology : Cosmology
        H0 [km s^-1 Mpc^-1] and the density parameters.

    Returns
    -------
    float or ndarray
        Comoving distance [Mpc].

    References
    ----------
    [Hogg1999] Eq. 15.

    Validity
    --------
    The radiation term is inside E(z); for z > 1e4 the resolution of the integral has not been tested.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py: relative difference < 1e-8 against astropy.cosmology.FlatLambdaCDM.
    """
```

The order of the sections is fixed: one-line summary, equation, `Parameters`, `Returns`, `References`, `Validity`, `Verification`. `Validity` writes the limit of the approximation (the truncated series term, the valid time range, the expected accuracy); if the limit is unknown, "not tested" is written.

### The `@source` decorator

The decorator in `kozmos/_sources.py` does not change the function; it only adds the record `(module, function, key, where, status, note)` to a registry. `where` is the position inside the source (equation number, section, table, file name or TeX label). `status` is the evidential status of the record: `read` says the equation was taken from the text of the source while the code was written, and `pending` says the source has not yet been opened. Three things are produced from the registry:

- `kozmos.sources_of(function)`: returns the bibliographic entries the function rests on.
- `kozmos.bibliography_report()`: gives the source list of the functions called in an analysis; the citation list comes out of here when writing a paper.
- The source → function reverse index: if an error is found in a source, the affected functions are listed.

### Machine check

`tests/test_source_rule.py` tests the following, and the test fails when any of them is broken:

1. Every function and class under `kozmos` whose name does not start with an underscore carries at least one `@source`.
2. Every key used exists in `references.bib`.
3. The `References` section in the docstring contains the same set as the keys in the decorator.
4. In every `@source` record `where` is filled and `status` is one of the two valid values.
5. The test file named in the `Verification` section of every public function exists.

Helpers that have no source (array formatting, argument checking) are named with an underscore and stay outside the rule. Functions that must be public but implement no equation are marked with `@infrastructure("reason")`; the test checks that these exemptions have a reason and are at most five. If a function rests not on a source but on a derivation in this project, the derivation is written under `plan/03_mathematics` or `docs/derivations` and cited with `@source("kozmos:derivation/<name>")`.

### Constants

Physical and astronomical constants are defined in `core/constants.py`, each with its value, unit, uncertainty and source. Physical constants are taken from the CODATA recommended values, the nominal solar and planetary values from the IAU 2015 B3 resolution [Prsa2016], and the astronomical unit from the IAU 2012 B2 resolution. No bare numeric constant is written in code; the `ruff` rules and review check this.

## Python standards

- Python 3.12 and later; type hints are mandatory, `mypy --strict` is clean.
- Formatting and checking are done with `ruff`; the configuration is in `pyproject.toml`.
- A module carries one topic; a module that exceeds 400 lines is reviewed for splitting.
- A function does one job. A function whose behavior changes with a flag argument is split in two.
- There are at most three positional arguments; more are given by keyword (the `*` separator).
- Errors are not passed over silently: invalid input raises `ValueError`, and the behavior (`nan` or error) for array elements outside the domain is written in the docstring.
- Mutable default arguments, global variables and side effects at import time are forbidden.
- Randomness is used by taking a `numpy.random.Generator` object as an argument; no global seed is used.
- Angles are radians inside. Degrees, hours and arcseconds appear only at the boundary, with the conversions in `core/angles.py`.
- Array functions take input with `numpy.asarray` and return a scalar for scalar input.

## Naming

- The function name says the quantity computed and its unit: `luminosity_distance_mpc`, `airmass`, `julian_date`.
- Transform functions have the form `source_to_target`: `equatorial_to_galactic`, `utc_to_tt`.
- Symbols follow the notation in the source and are explained in the docstring; if two sources use different symbols, the mapping is written in the `@source` note.

## Versions and changes

Every change that alters a numerical result (a new series term, a constant update, a different integration method) is written to the file `CHANGES.md`, with the affected function and the size of the change. If the tolerance of an oracle test is to be loosened, the rationale stays in the same record.
