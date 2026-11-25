"""Cone query of the 2MASS Point Source Catalog.

The query goes to the ``fp_psc`` table of the IRSA TAP service. Catalog:
[Skrutskie2006]; column definitions were read from the IRSA schema table [IRSA2026].
The catalog is an independent yardstick for testing the source detection and
astrometry code of this repository on a real image (``examples/real_image_validation.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kozmos._sources import source
from kozmos.archives import tap
from kozmos.archives.gaia import IRSA_TAP

TABLE = "fp_psc"
ATTRIBUTION = "2MASS Point Source Catalog (Skrutskie et al. 2006); via the IRSA TAP service"
COLUMNS = "ra, dec, err_maj, j_m, h_m, k_m, k_cmsig, k_snr, ph_qual"


@source(
    "IRSA2026",
    where="TAP_SCHEMA.columns, table_name = 'fp_psc'",
    status="read",
    note="ra/dec 'J2000 decimal deg'; err_maj semi-major axis of the position uncertainty ellipse [arcsec]; "
    "j_m/h_m/k_m 'selected default magnitude' [mag]; k_cmsig corrected K uncertainty; ph_qual photometric quality",
)
@source(
    "ADQL2023",
    where="Section 4.2.11 (CIRCLE), 4.2.12 (CONTAINS), 4.2.18 (POINT)",
    status="read",
    note="CONTAINS returns 1 if the first geometry is inside or on the boundary of the second; in WHERE it is "
    "compared with 0 or 1. CIRCLE takes the center coordinates and the radius in degrees. The coordinate-system "
    "string given as the first argument ('J2000') is a historical and deprecated option in the document; it was "
    "tested to work on the IRSA service",
)
def point_sources(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    cache_dir: Path,
    *,
    refresh: bool = False,
) -> dict[str, np.ndarray]:
    """Download the 2MASS point sources around a given direction.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Center of the cone (J2000) [deg].
    radius_deg : float
        Radius of the cone [deg].
    cache_dir : Path
    refresh : bool
        Download again even if the cached file exists.

    Returns
    -------
    dict[str, ndarray]
        ``ra``, ``dec`` [deg], ``err_maj`` [arcsec], ``j_m``, ``h_m``,
        ``k_m`` [Vega magnitude], ``k_cmsig`` [mag], ``k_snr``, ``ph_qual``.

    References
    ----------
    [IRSA2026] TAP_SCHEMA.columns; [ADQL2023] Section 4.2.11, 4.2.12, 4.2.18; catalog: [Skrutskie2006].

    Validity
    --------
    For sources with an empty ``k_cmsig`` the K magnitude is an upper limit (the
    source was not detected in that band); the third letter of the ``ph_qual``
    string is the quality of the K band. Inside nebulosity the catalog is also
    incomplete and may contain false sources.

    Verification
    ------------
    tests/unit/test_archives.py (needs the network, ``-m network``).
    """
    adql = (
        f"SELECT {COLUMNS} FROM {TABLE} WHERE CONTAINS(POINT('J2000', ra, dec), "
        f"CIRCLE('J2000', {ra_deg:.6f}, {dec_deg:.6f}, {radius_deg:.6f})) = 1"
    )
    return tap.read_csv(tap.query(IRSA_TAP, adql, cache_dir, attribution=ATTRIBUTION, refresh=refresh))
