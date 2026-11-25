"""Star samples from the Gaia DR3 catalog.

Queries go to the ``gaia_dr3_source`` table of the IRSA TAP service. The old
``tap-server`` address of the ESA Gaia archive is not used because on
25 November 2026 it returned a landing page instead of a query result; the
mirror at AIP (gaia.aip.de) works but returns VOTable.

Catalog: Gaia DR3 [Vallenari2023]; column definitions from the IRSA schema table [IRSA2026]. The absolute magnitude
definition and the quality-cut approach were read from the Gaia DR2 HR diagram paper [Babusiaux2018].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kozmos._sources import source
from kozmos.archives import tap

IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP/sync"
TABLE = "gaia_dr3_source"
ATTRIBUTION = "ESA Gaia DR3 (Gaia Collaboration, Vallenari et al. 2023); via the IRSA TAP service"
COLUMNS = (
    "source_id, ra, dec, parallax, parallax_over_error, pmra, pmdec, "
    "phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp, ruwe"
)


@source(
    "Babusiaux2018",
    where="Appendix B: archive query and filters; Section 2",
    status="read",
    note="the paper uses parallax_over_error > 10 and photometric filters; here a relative parallax error and "
    "a RUWE cut are applied (RUWE is not in the DR2 paper; it is a Gaia DR3 column)",
)
@source(
    "IRSA2026",
    where="TAP_SCHEMA.columns, table_name = 'gaia_dr3_source'",
    status="read",
    note="column definitions and units from the service schema table: parallax [mas], pmra/pmdec [mas/yr], "
    "ruwe 'Renormalised unit weight error', bp_rp 'BP - RP colour' [mag]",
)
def nearby_stars(
    cache_dir: Path,
    *,
    parallax_min_mas: float = 20.0,
    parallax_snr_min: float = 20.0,
    ruwe_max: float = 1.4,
    refresh: bool = False,
) -> dict[str, np.ndarray]:
    """Download stars around the Sun from Gaia DR3 with quality cuts.

    Default selection: parallax > 20 mas (within 50 pc), parallax S/N > 20,
    RUWE < 1.4, BP−RP color present. On 25 November 2026 this selection returned 31,298 stars.

    Parameters
    ----------
    cache_dir : Path
    parallax_min_mas : float
        Lower parallax limit [mas]; stars within 1000/parallax_min_mas parsec.
    parallax_snr_min : float
        Lower limit of ``parallax_over_error``.
    ruwe_max : float
        Upper limit of the renormalised unit weight error.
    refresh : bool
        Download again even if the cached file exists.

    Returns
    -------
    dict[str, ndarray]
        Column name → values; names are as in the Gaia archive.

    References
    ----------
    [Babusiaux2018] Appendix B; [IRSA2026] TAP_SCHEMA.columns.

    Validity
    --------
    Gaia saturates on very bright stars (roughly G < 3); the brightest stars
    are not in the sample. The RUWE cut removes some of the unresolved binaries.

    Verification
    ------------
    tests/unit/test_archives.py (needs the network, ``-m network``).
    """
    adql = (
        f"SELECT {COLUMNS} FROM {TABLE} WHERE parallax > {parallax_min_mas:g} "
        f"AND parallax_over_error > {parallax_snr_min:g} AND ruwe < {ruwe_max:g} AND bp_rp IS NOT NULL"
    )
    return tap.read_csv(tap.query(IRSA_TAP, adql, cache_dir, attribution=ATTRIBUTION, refresh=refresh))


@source(
    "Babusiaux2018",
    where="Section 2: M_G = G + 5 + 5 log10(ϖ/1000), ϖ in milliarcseconds",
    status="read",
    note="no extinction term is added",
)
def absolute_magnitude(apparent_magnitude: np.ndarray, parallax_mas: np.ndarray) -> np.ndarray:
    """Compute absolute magnitude from apparent magnitude and parallax: M = m + 5 + 5 log10(ϖ/1000).

    Parameters
    ----------
    apparent_magnitude : ndarray
    parallax_mas : ndarray
        Parallax [mas], positive.

    References
    ----------
    [Babusiaux2018] Section 2.

    Validity
    --------
    Takes the inverse of the parallax as the distance; usable when the relative
    parallax error is small (below about 10%). Interstellar extinction is not
    corrected; it is small within 50 pc (not tested).

    Verification
    ------------
    tests/unit/test_archives.py: M = m at 10 pc (ϖ = 100 mas); consistency with the distance modulus.
    """
    return (
        np.asarray(apparent_magnitude, dtype=float)
        + 5.0
        + 5.0 * np.log10(np.asarray(parallax_mas, dtype=float) / 1000.0)
    )


@source(
    "IRSA2026",
    where="TAP_SCHEMA.columns, table_name = 'gaia_dr3_source'",
    status="read",
    note="ra/dec [deg], ref_epoch 'Reference epoch' [yr], pmra 'Proper motion in right ascension direction' "
    "(μ_α cos δ) and pmdec [mas/yr], phot_g_mean_mag 'G-band mean magnitude' [mag]",
)
@source(
    "ADQL2023",
    where="Section 4.2.11 (CIRCLE), 4.2.12 (CONTAINS), 4.2.18 (POINT)",
    status="read",
    note="cone query; the coordinate-system string is a deprecated option in the document but works at IRSA",
)
def cone_search(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    cache_dir: Path,
    *,
    epoch_yr: float | None = None,
    refresh: bool = False,
) -> dict[str, np.ndarray]:
    """Download Gaia DR3 sources around a given direction; optionally move positions to an epoch.

    Parameters
    ----------
    ra_deg, dec_deg, radius_deg : float
        Center (ICRS) and radius of the cone [deg].
    cache_dir : Path
    epoch_yr : float, optional
        If given, ``ra`` and ``dec`` are moved to this Julian year with the proper motion; sources
        without a proper motion are left unchanged.
    refresh : bool
        Download again even if the cached file exists.

    Returns
    -------
    dict[str, ndarray]
        ``source_id``, ``ra``, ``dec`` [deg], ``ref_epoch`` [yr], ``pmra``, ``pmdec`` [mas/yr],
        ``phot_g_mean_mag``.

    References
    ----------
    [IRSA2026] TAP_SCHEMA.columns; [ADQL2023] Section 4.2.11, 4.2.12, 4.2.18; catalog: [Vallenari2023].

    Validity
    --------
    The epoch shift is linear: Δα = μ_α* Δt / cos δ, Δδ = μ_δ Δt; parallax and radial velocity
    are not applied. Very close to the poles the 1/cos δ factor is unreliable.

    Verification
    ------------
    tests/unit/test_archives.py (needs the network, ``-m network``): staying inside the cone and the size of the epoch shift.
    """
    adql = (
        f"SELECT source_id, ra, dec, ref_epoch, pmra, pmdec, phot_g_mean_mag FROM {TABLE} WHERE "
        f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra_deg:.6f}, {dec_deg:.6f}, "
        f"{radius_deg:.6f})) = 1"
    )
    s = tap.read_csv(tap.query(IRSA_TAP, adql, cache_dir, attribution=ATTRIBUTION, refresh=refresh))
    if epoch_yr is not None:
        dt = epoch_yr - s["ref_epoch"]
        mas = 1.0 / 3.6e6
        var = np.isfinite(s["pmra"]) & np.isfinite(s["pmdec"])
        s["ra"] = np.where(var, s["ra"] + s["pmra"] * dt * mas / np.cos(np.radians(s["dec"])), s["ra"])
        s["dec"] = np.where(var, s["dec"] + s["pmdec"] * dt * mas, s["dec"])
    return s
