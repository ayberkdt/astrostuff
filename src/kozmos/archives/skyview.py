"""Image cutouts from NASA SkyView.

SkyView resamples many surveys (DSS2, 2MASS, WISE ...) to the requested
position, field and pixel grid and returns them as FITS. The query parameters
(``Position``, ``Survey``, ``Pixels``, ``Size``, ``Return``) were read from the
examples in the SkyView batch-use documentation [SkyView2026].

Cutouts are resampled images: they do not replace the original archive data for
photometric calibration; they are for visual comparison and algorithm testing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kozmos._sources import source
from kozmos.archives import cache
from kozmos.io import fits

ENDPOINT = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"
ATTRIBUTION = "NASA SkyView (HEASARC); the surveys' own attributions are also required"

# Surveys used in the visible and infrared comparison, with effective wavelengths [µm].
# Wavelengths: [Cohen2003] Table 2 (λ_iso) for 2MASS, [Wright2010] band names for WISE; approximate for DSS2.
SURVEYS = {
    "DSS2 Blue": 0.44,
    "DSS2 Red": 0.64,
    "DSS2 IR": 0.85,
    "2MASS-J": 1.235,
    "2MASS-H": 1.662,
    "2MASS-K": 2.159,
    "WISE 3.4": 3.4,
    "WISE 4.6": 4.6,
    "WISE 12": 12.0,
    "WISE 22": 22.0,
}

ATTEMPTS = 3


@source(
    "SkyView2026",
    where="Batch Page: runquery.pl examples (Position, Survey, Pixels, Size, Return)",
    status="read",
    note="survey names ('DSS2 Red', '2MASS-K', 'WISE 12') were verified from working queries, not from the document",
)
def download_cutout(
    ra_deg: float,
    dec_deg: float,
    survey: str,
    cache_dir: Path,
    *,
    size_deg: float = 0.3,
    pixels: int = 400,
    refresh: bool = False,
) -> tuple[np.ndarray, dict]:
    """Download a square image cutout from a survey around a given position.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Center position (ICRS) [deg].
    survey : str
        SkyView survey name; known names are in the ``SURVEYS`` dictionary.
    cache_dir : Path
    size_deg : float
        Side length of the image [deg].
    pixels : int
        Number of pixels per side.
    refresh : bool
        Download again even if the cached file exists.

    Returns
    -------
    (image, header) : ``image[y, x]`` and the FITS header.

    References
    ----------
    [SkyView2026] Batch Page.

    Validity
    --------
    The image is resampled; pixel values are in the original unit of the survey
    and change from survey to survey. North is up and east is left (SkyView
    default; checked from the sign of CDELT1 in the header).

    If the service does not return valid FITS the request is repeated three times; invalid responses are not cached.

    Verification
    ------------
    tests/unit/test_archives.py: retry on an invalid response (no network) and a real query (``-m network``).
    """
    name = f"skyview_{cache.key(survey, f'{ra_deg:.6f}', f'{dec_deg:.6f}', str(size_deg), str(pixels))}.fits"
    target = Path(cache_dir) / name
    for attempt in range(1, ATTEMPTS + 1):
        path = cache.download(
            ENDPOINT,
            target,
            parameters={
                "Position": f"{ra_deg:.6f},{dec_deg:.6f}",
                "Survey": survey,
                "Pixels": str(pixels),
                "Size": f"{size_deg:g}",
                "Return": "FITS",
            },
            attribution=f"{ATTRIBUTION}; survey: {survey}",
            refresh=refresh or attempt > 1,
        )
        try:
            return fits.read_fits(path)
        except ValueError as error:
            # The service sometimes returns a non-FITS or truncated response with status 200 (seen in a test on
            # 25 November 2026). The invalid response is not left in the cache; it is kept under the name
            # ``.invalid`` so it can be inspected.
            invalid = target.with_name(f"{target.name}.invalid{attempt}")
            path.replace(invalid)
            if attempt == ATTEMPTS:
                head = invalid.read_bytes()[:80]
                raise ValueError(
                    f"SkyView did not return valid FITS in {ATTEMPTS} attempts ({error}); start of the last response: "
                    f"{head!r}; responses were kept under {invalid.parent} with the .invalid name"
                ) from error
    raise AssertionError("unreachable")
