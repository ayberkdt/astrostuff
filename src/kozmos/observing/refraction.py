"""Atmospheric refraction: the A tan Z + B tan³ Z model.

The model and the computation of its constants were read from the ERFA source
code (``refco.c``) [ERFA2023]. Z is the refraction-affected ("observed") zenith
distance; dZ = A tan Z + B tan³ Z is added to Z to obtain the zenith distance
in vacuum ("topocentric"). Refraction raises an object above the horizon.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source

Array = NDArray[np.float64]


@source(
    "ERFA2023",
    where="refco.c",
    status="read",
    note="water vapor pressure, refractive index (γ = n − 1) and β (scale height / geocentric distance) "
    "with the expressions in refco.c; A = γ(1 − β), B = −γ(β − γ/2). ERFA bases these expressions on Gill (1982), "
    "Crane (1976), Hohenkerk and Sinclair (1985), Rueger (2002), Stone (1996) and Green (1987); "
    "those sources were not read. Only the optical/infrared branch (λ <= 100 µm) was implemented",
)
def refraction_constants(
    pressure_hpa: float, temperature_c: float, relative_humidity: float, wavelength_um: float = 0.55
) -> tuple[float, float]:
    """Compute the A and B constants of the refraction model.

    Parameters
    ----------
    pressure_hpa : float
        Pressure at the observer [hPa].
    temperature_c : float
        Ambient temperature [°C].
    relative_humidity : float
        Relative humidity, 0-1.
    wavelength_um : float
        Wavelength [µm]; range 0.1-100.

    Returns
    -------
    (A, B) [rad]

    References
    ----------
    [ERFA2023] refco.c.

    Validity
    --------
    Compared with ray tracing, the source gives a worst-case error of 62 and a root-mean-square
    error of 8 milliarcseconds in the optical/infrared (zenith distance 15°-75°). Height above sea
    level enters only through the pressure. The radio branch was not implemented; λ > 100 µm raises
    ``ValueError``. Inputs are clipped to the safe ranges in the source.

    Verification
    ------------
    tests/oracle/test_precession_oracle.py: equality with erfa.refco; agreement within 0.02 arcseconds with the table in refco.c.
    """
    if wavelength_um > 100.0:
        raise ValueError("only the optical/infrared branch is implemented (wavelength <= 100 µm)")
    t = min(max(temperature_c, -150.0), 200.0)
    p = min(max(pressure_hpa, 0.0), 10000.0)
    r = min(max(relative_humidity, 0.0), 1.0)
    w = max(wavelength_um, 0.1)
    if p > 0.0:
        ps = 10.0 ** ((0.7859 + 0.03477 * t) / (1.0 + 0.00412 * t)) * (1.0 + p * (4.5e-6 + 6e-10 * t * t))
        pw = r * ps / (1.0 - (1.0 - r) * ps / p)
    else:
        pw = 0.0
    tk = t + 273.15
    wlsq = w * w
    gamma = ((77.53484e-6 + (4.39108e-7 + 3.666e-9 / wlsq) / wlsq) * p - 11.2684e-6 * pw) / tk
    beta = 4.4474e-6 * tk
    return gamma * (1.0 - beta), -gamma * (beta - gamma / 2.0)


@source(
    "ERFA2023",
    where="refco.c: 'dZ = A tan Z + B tan^3 Z', Z is the observed zenith distance",
    status="read",
    note="for topocentric → observed direction, the equation Z_observed = Z_topo − dZ(Z_observed) is solved by "
    "fixed-point iteration (the iteration is this project's choice)",
)
def observed_altitude(altitude_rad: ArrayLike, a_rad: float, b_rad: float, iterations: int = 4) -> Array:
    """Compute the refracted (observed) altitude from the altitude in vacuum (topocentric).

    Parameters
    ----------
    altitude_rad : float or ndarray
        Altitude without refraction [rad].
    a_rad, b_rad : float
        Outputs of ``refraction_constants``.

    Returns
    -------
    ndarray
        Observed altitude [rad]; always greater than or equal to the input.

    References
    ----------
    [ERFA2023] refco.c.

    Validity
    --------
    The tan Z model is invalid near the horizon: the source table deviates from ray tracing
    by 0.6 arcseconds at a zenith distance of 80° and the difference grows quickly toward the
    horizon. For directions with an observed zenith distance above 85° (altitude below about 5°)
    the result is ``nan``.

    Verification
    ------------
    tests/oracle/test_precession_oracle.py: round trip with the forward model; refco.c table.
    """
    z_topo = np.pi / 2.0 - np.asarray(altitude_rad, dtype=float)
    z = z_topo.copy()
    for _ in range(iterations):
        tan_z = np.tan(np.clip(z, 0.0, np.radians(85.0)))
        z = z_topo - (a_rad * tan_z + b_rad * tan_z**3)
    return np.where(z <= np.radians(85.0), np.pi / 2.0 - z, np.nan)
