"""Magnitudes, distance modulus and blackbody radiation.

The bolometric magnitude zero point was read from IAU 2015 Resolution B2
[Mamajek2015] and the distance modulus definition from [Hogg1999] Eq. 25. The
Planck function was read from the source code of the astropy ``BlackBody``
model [Astropy2022]. Constants are CODATA 2022 values [CODATA2022].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.constants import BOLOMETRIC_ZERO_POINT_LUMINOSITY, BOLTZMANN, PLANCK, SPEED_OF_LIGHT, STEFAN_BOLTZMANN

Array = NDArray[np.float64]


@source("Mamajek2015", where="M_Bol definition; L_0 = 3.0128e28 W", status="read", note="M_Bol = −2,5 log10(L / L_0)")
def absolute_bolometric_magnitude(luminosity_w: ArrayLike) -> Array:
    """Compute the absolute bolometric magnitude from the luminosity.

    M_Bol = −2.5 log10(L / L_0), L_0 = 3.0128 × 10²⁸ W.

    Parameters
    ----------
    luminosity_w : float or ndarray
        Luminosity [W], positive.

    References
    ----------
    [Mamajek2015] L_0 definition.

    Verification
    ------------
    tests/unit/test_magnitudes.py: 4.74 mag for the nominal solar luminosity (the value given in the resolution).
    """
    return -2.5 * np.log10(np.asarray(luminosity_w, dtype=float) / BOLOMETRIC_ZERO_POINT_LUMINOSITY.value)


@source("Mamajek2015", where="M_Bol definition; L_0 = 3.0128e28 W", status="read")
def luminosity_from_bolometric_magnitude_w(absolute_bolometric_magnitude_mag: ArrayLike) -> Array:
    """Compute the luminosity from the absolute bolometric magnitude: L = L_0 · 10^(−0.4 M_Bol).

    References
    ----------
    [Mamajek2015] L_0 definition.

    Verification
    ------------
    tests/property/test_magnitudes_property.py: round trip with ``absolute_bolometric_magnitude``.
    """
    m = np.asarray(absolute_bolometric_magnitude_mag, dtype=float)
    return BOLOMETRIC_ZERO_POINT_LUMINOSITY.value * 10.0 ** (-0.4 * m)


@source("Hogg1999", where="Eq. 25", status="read", note="DM = 5 log10(D_L / 10 pc)")
def distance_modulus_from_pc(distance_pc: ArrayLike) -> Array:
    """Compute the distance modulus from a distance in parsecs.

    References
    ----------
    [Hogg1999] Eq. 25.

    Validity
    --------
    Does not include extinction or the K correction.

    Verification
    ------------
    tests/unit/test_magnitudes.py: 0 mag for 10 pc.
    """
    return 5.0 * np.log10(np.asarray(distance_pc, dtype=float) / 10.0)


@source("Hogg1999", where="Eq. 25", status="read")
def pc_from_distance_modulus(distance_modulus_mag: ArrayLike) -> Array:
    """Compute the distance in parsecs from the distance modulus: d = 10^(μ/5 + 1).

    References
    ----------
    [Hogg1999] Eq. 25.

    Verification
    ------------
    tests/property/test_magnitudes_property.py: round trip with ``distance_modulus_from_pc``.
    """
    return 10.0 ** (np.asarray(distance_modulus_mag, dtype=float) / 5.0 + 1.0)


@source(
    "Astropy2022",
    where="modeling/physical_models.py, BlackBody.evaluate",
    status="read",
    note="bb_nu = 2 h ν³ / (c² expm1(hν/kT)); astropy 8.0.0 source code. The textbook citation [RybickiLightman1979] was not read",
)
def planck_frequency(frequency_hz: ArrayLike, temperature_k: ArrayLike) -> Array:
    """Compute the specific intensity of a blackbody per unit frequency.

    B_ν(T) = (2 h ν³ / c²) / (exp(hν / kT) − 1)

    Returns
    -------
    float or ndarray
        B_ν [W m⁻² Hz⁻¹ sr⁻¹].

    References
    ----------
    [Astropy2022] BlackBody.evaluate.

    Validity
    --------
    ν > 0, T > 0. ``expm1`` is used for small hν/kT; for large values the
    exponential overflow resolves to zero.

    Verification
    ------------
    tests/unit/test_magnitudes.py: frequency integral equals σT⁴/π; tests/oracle/test_radiation_oracle.py: astropy BlackBody.
    """
    nu = np.asarray(frequency_hz, dtype=float)
    t = np.asarray(temperature_k, dtype=float)
    x = PLANCK.value * nu / (BOLTZMANN.value * t)
    with np.errstate(over="ignore"):
        return 2.0 * PLANCK.value * nu**3 / SPEED_OF_LIGHT.value**2 / np.expm1(x)


@source(
    "Astropy2022",
    where="modeling/physical_models.py, BlackBody.evaluate",
    status="read",
    note="from the transformation B_λ = B_ν c/λ² (derivation in this project): B_λ = (2hc²/λ⁵) / (exp(hc/λkT) − 1)",
)
def planck_wavelength(wavelength_m: ArrayLike, temperature_k: ArrayLike) -> Array:
    """Compute the specific intensity of a blackbody per unit wavelength.

    Returns
    -------
    float or ndarray
        B_λ [W m⁻³ sr⁻¹].

    References
    ----------
    [Astropy2022] BlackBody.evaluate.

    Verification
    ------------
    tests/unit/test_magnitudes.py: the peak wavelength agrees with the Wien constant [CODATA2022].
    """
    lam = np.asarray(wavelength_m, dtype=float)
    t = np.asarray(temperature_k, dtype=float)
    x = PLANCK.value * SPEED_OF_LIGHT.value / (lam * BOLTZMANN.value * t)
    with np.errstate(over="ignore"):
        return 2.0 * PLANCK.value * SPEED_OF_LIGHT.value**2 / lam**5 / np.expm1(x)


@source(
    "CODATA2022",
    where="Stefan-Boltzmann constant",
    status="read",
    note="F = σ T⁴; the textbook citation for the law itself is still pending",
)
def blackbody_flux(temperature_k: ArrayLike) -> Array:
    """Compute the total flux leaving a blackbody surface: F = σ T⁴ [W m⁻²].

    References
    ----------
    [CODATA2022] Stefan-Boltzmann constant.

    Verification
    ------------
    tests/unit/test_magnitudes.py: nominal luminosity from the nominal solar radius and effective temperature [Prsa2016].
    """
    return STEFAN_BOLTZMANN.value * np.asarray(temperature_k, dtype=float) ** 4
