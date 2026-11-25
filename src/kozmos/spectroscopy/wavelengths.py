"""Wavelength conversions: air and vacuum, Doppler shifts, resolving power.

The air-vacuum relation is the IAU standard as printed in Morton's atomic-data
compilation [Morton1991] (Eq. 3, read from the ADS scan on 25 November 2026).
The Doppler relations build on ``kozmos.cosmology.distances`` ([Hogg1999] Eq. 9).
The resolving-power relations rest on the one worked example printed in
[TonryDavis1979] Sect. IV.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.constants import SPEED_OF_LIGHT_KM_S
from kozmos.cosmology.distances import doppler_redshift, doppler_velocity_km_s

Array = NDArray[np.float64]

# Shortest wavelength (air or vacuum) accepted by the air-vacuum functions [angstrom]. [Morton1991] lists air
# wavelengths only longward of 2000 angstrom (air), 2000.648 angstrom (vacuum). This project's choice.
_MIN_WAVELENGTH_ANGSTROM = 2000.0


def _refractivity(wavelength_vacuum_angstrom: Array) -> Array:
    """n - 1 of [Morton1991] Eq. 3 with sigma = 1e4 / lambda_vac (lambda in angstrom, sigma in micron^-1)."""
    sigma2 = (1.0e4 / wavelength_vacuum_angstrom) ** 2
    return 6.4328e-5 + 2.94981e-2 / (146.0 - sigma2) + 2.5540e-4 / (41.0 - sigma2)


def _check_range(wavelength_angstrom: Array) -> None:
    if np.any(~np.isfinite(wavelength_angstrom)) or np.any(wavelength_angstrom < _MIN_WAVELENGTH_ANGSTROM):
        raise ValueError(
            f"wavelengths must be finite and >= {_MIN_WAVELENGTH_ANGSTROM} angstrom (air wavelengths are not defined "
            "shortward of 2000 angstrom in the source)"
        )


@source(
    "Morton1991",
    where="Eq. 3 (Sect. 2, IAU standard after Oosterhoff 1957 and Edlen 1953)",
    status="read",
    note=(
        "n - 1 = 6.4328e-5 + 2.94981e-2/(146 - sigma^2) + 2.5540e-4/(41 - sigma^2), sigma = 1e4/lambda with lambda in "
        "angstrom. The source does not say whether lambda in sigma is the air or the vacuum wavelength; this "
        "project's choice: the vacuum wavelength (an Edlen-type dispersion formula is a function of the vacuum "
        "wavenumber), which makes the pair of conversions exact inverses."
    ),
)
def air_refractive_index(wavelength_vacuum_angstrom: ArrayLike) -> Array:
    """Compute the refractive index of standard air from the IAU formula.

    n = lambda_vac / lambda_air = 1 + 6.4328e-5 + 2.94981e-2 / (146 - sigma^2) + 2.5540e-4 / (41 - sigma^2),
    sigma = 1e4 / lambda_vac

    Parameters
    ----------
    wavelength_vacuum_angstrom : float or ndarray
        Vacuum wavelength [angstrom], at least 2000.

    Returns
    -------
    ndarray
        Refractive index n (dimensionless), about 1.00028 at 5000 angstrom.

    References
    ----------
    [Morton1991] Eq. 3.

    Validity
    --------
    The formula has a pole at sigma^2 = 41 (1562 angstrom); this function raises ValueError below 2000 angstrom,
    where the source lists no air wavelengths. The source states that Edlen (1966) and Peck & Reeder (1972) proposed
    improvements that mainly affect infrared wavelengths and that it used this formula anyway; the size of that
    difference was not measured here (no independent implementation is installed). The conditions of "standard
    air" (temperature, pressure, humidity, CO2) are not given in the text read.

    Verification
    ------------
    tests/unit/test_spectroscopy.py: values of the Morton table (Na I multiplets, 2000 angstrom pair), range guard.
    """
    wavelength = np.asarray(wavelength_vacuum_angstrom, dtype=float)
    _check_range(wavelength)
    return 1.0 + _refractivity(wavelength)


@source(
    "Morton1991",
    where="Eq. 3 (Sect. 2)",
    status="read",
    note="lambda_air = lambda_vac / n with n from Eq. 3 evaluated at sigma = 1e4/lambda_vac (this project's choice).",
)
def vacuum_to_air(wavelength_vacuum_angstrom: ArrayLike) -> Array:
    """Convert vacuum wavelengths to air wavelengths with the IAU standard relation.

    lambda_air = lambda_vac / n(lambda_vac)

    Parameters
    ----------
    wavelength_vacuum_angstrom : float or ndarray
        Vacuum wavelength [angstrom], at least 2000.

    Returns
    -------
    ndarray
        Air wavelength [angstrom].

    References
    ----------
    [Morton1991] Eq. 3.

    Validity
    --------
    Same as ``air_refractive_index``. Measured against the 12 air/vacuum pairs printed in the source table
    (Na I multiplets, 2680-5898 angstrom): largest difference 0.9 milli-angstrom, the size of the rounding of the
    printed 3-decimal wavelengths. The alternative convention (sigma from the air wavelength) differs from this one
    by 0.098 milli-angstrom at 2000 angstrom, 0.018 at 5000 and 0.004 at 20000 (measured, see tests).

    Verification
    ------------
    tests/unit/test_spectroscopy.py: table pairs, the 2000 angstrom pair, exact round trip.
    """
    wavelength = np.asarray(wavelength_vacuum_angstrom, dtype=float)
    _check_range(wavelength)
    return wavelength / (1.0 + _refractivity(wavelength))


@source(
    "Morton1991",
    where="Eq. 3 (Sect. 2)",
    status="read",
    note=(
        "inverse of the vacuum-to-air relation by fixed-point iteration lambda_vac = lambda_air * n(lambda_vac), "
        "start lambda_vac = lambda_air (iteration scheme: this project's choice)."
    ),
)
def air_to_vacuum(wavelength_air_angstrom: ArrayLike) -> Array:
    """Convert air wavelengths to vacuum wavelengths with the IAU standard relation.

    lambda_vac = lambda_air * n(lambda_vac), solved by fixed-point iteration (the map has contraction factor about
    3e-4, so five iterations reach machine precision)

    Parameters
    ----------
    wavelength_air_angstrom : float or ndarray
        Air wavelength [angstrom], at least 2000.

    Returns
    -------
    ndarray
        Vacuum wavelength [angstrom].

    References
    ----------
    [Morton1991] Eq. 3.

    Validity
    --------
    Same as ``vacuum_to_air``. Round trip with ``vacuum_to_air``: largest error 1.8e-12 angstrom between 2001
    and 30000 angstrom (measured, see tests).

    Verification
    ------------
    tests/unit/test_spectroscopy.py: table pairs, the 2000 angstrom pair, exact round trip.
    """
    air = np.asarray(wavelength_air_angstrom, dtype=float)
    _check_range(air)
    vacuum = air.copy()
    for _ in range(20):
        updated = air * (1.0 + _refractivity(vacuum))
        done = np.max(np.abs(updated - vacuum)) <= 1e-13 * np.max(updated)
        vacuum = updated
        if done:
            break
    return vacuum


@source(
    "Hogg1999",
    where="Eq. 9 and the redshift definition z = lambda_o/lambda_e - 1 printed just before it",
    status="read",
    note=(
        "relativistic: lambda_o = lambda_e (1 + z) with z from kozmos.cosmology.distances.doppler_redshift (Eq. 9). "
        "Non-relativistic: lambda_o = lambda_e (1 + v/c), the first-order expansion of Eq. 9 (this project's "
        "derivation, not a printed equation)."
    ),
)
def doppler_shifted_wavelength(
    wavelength_rest: ArrayLike, velocity_km_s: ArrayLike, relativistic: bool = True
) -> Array:
    """Compute the observed wavelength of a source with a given radial velocity.

    relativistic:      lambda_o = lambda_e sqrt((1 + v/c) / (1 - v/c))
    non-relativistic:  lambda_o = lambda_e (1 + v/c)

    Parameters
    ----------
    wavelength_rest : float or ndarray
        Rest (emitted) wavelength, any length unit; the result has the same unit.
    velocity_km_s : float or ndarray
        Radial velocity, positive for recession [km/s].
    relativistic : bool
        Use the special-relativistic relation (default) or its first-order form.

    Returns
    -------
    ndarray
        Observed wavelength, same unit as ``wavelength_rest``.

    References
    ----------
    [Hogg1999] Eq. 9.

    Validity
    --------
    Peculiar (not cosmological) velocities, |v| < c. The two forms differ by (v/c)^2 / 2 in relative wavelength
    (5e-9 at 30 km/s, 5e-7 at 300 km/s); the tests check this expansion.

    Verification
    ------------
    tests/unit/test_spectroscopy.py: small-velocity limit, difference of the two forms, round trip.
    tests/oracle/test_spectroscopy_oracle.py: astropy doppler_relativistic equivalency.
    """
    rest = np.asarray(wavelength_rest, dtype=float)
    if relativistic:
        return rest * (1.0 + doppler_redshift(velocity_km_s))
    return rest * (1.0 + np.asarray(velocity_km_s, dtype=float) / SPEED_OF_LIGHT_KM_S)


@source(
    "Hogg1999",
    where="Eq. 9 and the redshift definition z = lambda_o/lambda_e - 1 printed just before it",
    status="read",
    note=(
        "relativistic: z = lambda_o/lambda_e - 1, then kozmos.cosmology.distances.doppler_velocity_km_s (Eq. 9 "
        "solved for v). Non-relativistic: v = c z (this project's first-order form)."
    ),
)
def radial_velocity_from_wavelengths(
    wavelength_observed: ArrayLike, wavelength_rest: ArrayLike, relativistic: bool = True
) -> Array:
    """Compute the radial velocity from an observed and a rest wavelength.

    z = lambda_o / lambda_e - 1; relativistic v = c ((1+z)^2 - 1) / ((1+z)^2 + 1); non-relativistic v = c z

    Parameters
    ----------
    wavelength_observed : float or ndarray
        Observed wavelength, any length unit.
    wavelength_rest : float or ndarray
        Rest wavelength, same unit.
    relativistic : bool
        Use the special-relativistic relation (default) or v = c z.

    Returns
    -------
    ndarray
        Radial velocity, positive for recession [km/s].

    References
    ----------
    [Hogg1999] Eq. 9.

    Validity
    --------
    Same as ``doppler_shifted_wavelength``; the observed and rest wavelengths must both be in air or both in vacuum.

    Verification
    ------------
    tests/unit/test_spectroscopy.py: round trip with ``doppler_shifted_wavelength``.
    """
    z = np.asarray(wavelength_observed, dtype=float) / np.asarray(wavelength_rest, dtype=float) - 1.0
    if relativistic:
        return doppler_velocity_km_s(z)
    return SPEED_OF_LIGHT_KM_S * z


@source(
    "TonryDavis1979",
    where="Sect. IV: rms resolution about 2 A, given as 120 km per second at 5000 A",
    status="read",
    note=(
        "R = lambda / delta_lambda is the definition of resolving power; the paper prints only the numerical example "
        "(2 A at 5000 A is 120 km/s), which the tests reproduce. The relation itself is this project's statement."
    ),
)
def resolving_power(wavelength: ArrayLike, delta_wavelength: ArrayLike) -> Array:
    """Compute the resolving power from a wavelength and its resolution element.

    R = lambda / delta_lambda

    Parameters
    ----------
    wavelength : float or ndarray
        Wavelength, any length unit.
    delta_wavelength : float or ndarray
        Resolution element (FWHM or rms, as the caller defines it), same unit.

    Returns
    -------
    ndarray
        Resolving power R (dimensionless).

    References
    ----------
    [TonryDavis1979] Sect. IV (worked example).

    Validity
    --------
    Pure ratio; the meaning of ``delta_wavelength`` (FWHM versus rms) is the caller's. The source's example uses
    an rms resolution.

    Verification
    ------------
    tests/unit/test_spectroscopy.py: the 2 A at 5000 A example (R = 2500, 119.9 km/s).
    """
    return np.asarray(wavelength, dtype=float) / np.asarray(delta_wavelength, dtype=float)


@source(
    "TonryDavis1979",
    where="Sect. IV: rms resolution about 2 A, given as 120 km per second at 5000 A",
    status="read",
    note="delta_v = c delta_lambda / lambda = c / R, the first-order Doppler relation (this project's statement).",
)
def velocity_resolution_km_s(resolving_power_value: ArrayLike) -> Array:
    """Convert a resolving power to a velocity resolution.

    delta_v = c / R

    Parameters
    ----------
    resolving_power_value : float or ndarray
        Resolving power R (dimensionless).

    Returns
    -------
    ndarray
        Velocity resolution [km/s].

    References
    ----------
    [TonryDavis1979] Sect. IV (worked example).

    Validity
    --------
    First order in delta_lambda / lambda: the relative difference from the relativistic Doppler relation is of
    order 1/R (0.05 percent at R = 1000, measured in the tests).

    Verification
    ------------
    tests/unit/test_spectroscopy.py: R = 2500 gives 119.9 km/s; round trip with ``resolving_power_from_velocity``.
    """
    return SPEED_OF_LIGHT_KM_S / np.asarray(resolving_power_value, dtype=float)


@source(
    "TonryDavis1979",
    where="Sect. IV: rms resolution about 2 A, given as 120 km per second at 5000 A",
    status="read",
    note="R = c / delta_v, inverse of velocity_resolution_km_s (this project's statement).",
)
def resolving_power_from_velocity(velocity_resolution_km_s_value: ArrayLike) -> Array:
    """Convert a velocity resolution to a resolving power.

    R = c / delta_v

    Parameters
    ----------
    velocity_resolution_km_s_value : float or ndarray
        Velocity resolution [km/s].

    Returns
    -------
    ndarray
        Resolving power R (dimensionless).

    References
    ----------
    [TonryDavis1979] Sect. IV (worked example).

    Validity
    --------
    Same as ``velocity_resolution_km_s``.

    Verification
    ------------
    tests/unit/test_spectroscopy.py: round trip with ``velocity_resolution_km_s``.
    """
    return SPEED_OF_LIGHT_KM_S / np.asarray(velocity_resolution_km_s_value, dtype=float)
