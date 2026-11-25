"""Infrared photometry: 2MASS and WISE Vega zero points, magnitude-flux conversions.

The zero-magnitude flux densities were read from the arXiv TeX files of the
source papers: [Cohen2003] Table 2 (F_ν(iso) and λ(iso)) for 2MASS and
[Wright2010] Section 4.4 (F_ν° and F_λ°) for WISE. The Wien displacement
constant is the CODATA 2022 value [CODATA2022].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import infrastructure, source
from kozmos.core.constants import WIEN_WAVELENGTH

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Band:
    """Zero point of a photometric band in the Vega system."""

    name: str
    wavelength_um: float
    zero_flux_jy: float
    zero_flux_uncertainty_jy: float
    source: str


# [Cohen2003] Table 2: λ(iso) [µm], F_ν(iso) [Jy] and its uncertainty.
# [Wright2010] Section 4.4: F_ν° = 306.681, 170.663, 29.0448, 8.2839 Jy (W1..W4); overall systematic uncertainty ±1.5%.
BANDS: dict[str, Band] = {
    "J": Band("2MASS J", 1.235, 1594.0, 27.80, "Cohen2003"),
    "H": Band("2MASS H", 1.662, 1024.0, 19.95, "Cohen2003"),
    "Ks": Band("2MASS Ks", 2.159, 666.7, 12.55, "Cohen2003"),
    "W1": Band("WISE W1", 3.4, 306.681, 0.015 * 306.681, "Wright2010"),
    "W2": Band("WISE W2", 4.6, 170.663, 0.015 * 170.663, "Wright2010"),
    "W3": Band("WISE W3", 12.0, 29.0448, 0.015 * 29.0448, "Wright2010"),
    "W4": Band("WISE W4", 22.0, 8.2839, 0.015 * 8.2839, "Wright2010"),
}


@infrastructure("looks up the band dictionary; applies no equation")
def band(name: str) -> Band:
    """Return the zero-point record for a band name (``"J"``, ``"H"``, ``"Ks"``, ``"W1"`` ... ``"W4"``)."""
    try:
        return BANDS[name]
    except KeyError:
        raise ValueError(f"unknown band {name!r}; known bands: {', '.join(BANDS)}") from None


@source("Cohen2003", where="Table 2: F_ν(iso)", status="read", note="2MASS J, H, Ks: 1594, 1024, 666.7 Jy")
@source(
    "Wright2010", where="Section 4.4: F_ν°", status="read", note="WISE W1..W4: 306.681, 170.663, 29.0448, 8.2839 Jy"
)
def flux_jy_from_vega_magnitude(magnitude: ArrayLike, band_name: str) -> Array:
    """Compute the flux density from a Vega-system magnitude: F_ν = F_ν° · 10^(−0.4 m).

    Parameters
    ----------
    magnitude : float or ndarray
        Vega magnitude.
    band_name : str
        ``"J"``, ``"H"``, ``"Ks"``, ``"W1"``, ``"W2"``, ``"W3"``, ``"W4"``.

    Returns
    -------
    float or ndarray
        Flux density [Jy].

    References
    ----------
    [Cohen2003] Table 2; [Wright2010] Section 4.4.

    Validity
    --------
    The zero points are defined for a particular spectral shape; very red or
    very blue sources need a color correction. [Wright2010] mentions
    corrections of 8% and 4% for red sources in W3 and W4; they are not
    applied here.

    Verification
    ------------
    tests/unit/test_infrared.py: table value at zero magnitude; one tenth at magnitude 2.5.
    """
    return band(band_name).zero_flux_jy * 10.0 ** (-0.4 * np.asarray(magnitude, dtype=float))


@source("Cohen2003", where="Table 2: F_ν(iso)", status="read")
@source("Wright2010", where="Section 4.4: F_ν°", status="read")
def vega_magnitude_from_flux(flux_jy: ArrayLike, band_name: str) -> Array:
    """Compute the Vega magnitude from a flux density: m = −2.5 log10(F_ν / F_ν°).

    References
    ----------
    [Cohen2003] Table 2; [Wright2010] Section 4.4.

    Verification
    ------------
    tests/unit/test_infrared.py: round trip with ``flux_jy_from_vega_magnitude``.
    """
    return -2.5 * np.log10(np.asarray(flux_jy, dtype=float) / band(band_name).zero_flux_jy)


@source(
    "Cohen2003",
    where="Table 2: F_ν(iso)",
    status="read",
    note="difference using the definition m_AB = −2.5 log10(F_ν / 3631 Jy): m_AB − m_Vega = 2.5 log10(3631 / F_ν°). "
    "The citation for the 3631 Jy constant, [OkeGunn1983], was not read",
)
@source("Wright2010", where="Section 4.4: F_ν°", status="read")
def vega_ab_offset(band_name: str) -> float:
    """Compute the difference between AB and Vega magnitudes in a band: m_AB − m_Vega.

    References
    ----------
    [Cohen2003] Table 2; [Wright2010] Section 4.4.

    Validity
    --------
    The AB zero point is taken as 3631 Jy; the source of this value was not read in this work.

    Verification
    ------------
    tests/unit/test_infrared.py: about 1.84 mag in the Ks band (computed from the table value).
    """
    return float(2.5 * np.log10(3631.0 / band(band_name).zero_flux_jy))


@source("CODATA2022", where="Wien wavelength displacement law constant", status="read", note="λ_peak · T = b")
def peak_wavelength_um(temperature_k: ArrayLike) -> Array:
    """Compute the peak wavelength of blackbody radiation (B_λ): λ_peak = b / T.

    Returns
    -------
    float or ndarray
        Peak wavelength [µm].

    References
    ----------
    [CODATA2022] Wien displacement constant.

    Validity
    --------
    The peak is defined for B_λ; the peak of B_ν is at a different wavelength.

    Verification
    ------------
    tests/unit/test_infrared.py: 0.502 µm for 5772 K; about 97 µm for 30 K dust.
    """
    return WIEN_WAVELENGTH.value / np.asarray(temperature_k, dtype=float) * 1e6


@source("CODATA2022", where="Wien wavelength displacement law constant", status="read", note="T = b / λ_peak")
def peak_temperature_k(wavelength_um: ArrayLike) -> Array:
    """Compute the blackbody temperature from a peak wavelength: T = b / λ_peak.

    Gives the order of magnitude of the thermal source temperature to which a
    band is most sensitive: about 1340 K for 2MASS Ks, about 130 K for WISE W4.

    References
    ----------
    [CODATA2022] Wien displacement constant.

    Verification
    ------------
    tests/unit/test_infrared.py: round trip with ``peak_wavelength_um``.
    """
    return WIEN_WAVELENGTH.value / (np.asarray(wavelength_um, dtype=float) * 1e-6)
