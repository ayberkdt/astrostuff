"""Cosmological distance measures.

Definitions follow Hogg's review [Hogg1999]; the equations were read from the
paper's arXiv TeX source (astro-ph/9905116) on 25 November 2026 and are
numbered as in that source. The generalization to a dark energy equation of
state parameter ``w`` is taken from the H(z) expression of the Pantheon+
analysis [Brout2022].

There is no radiation term: [Hogg1999] Eq. 14 contains only matter, curvature
and the cosmological constant. Results therefore deviate from the physical
universe when ``z`` exceeds a few hundred.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.constants import JULIAN_YEAR_SECONDS, MEGAPARSEC, SPEED_OF_LIGHT_KM_S

Array = NDArray[np.float64]

# Gauss-Legendre nodes: for the integral of 1/E(z') over [0, z].
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(64)


@dataclass(frozen=True, slots=True)
class Cosmology:
    """Parameters of a Friedmann-Lemaître-Robertson-Walker universe.

    Attributes
    ----------
    h0 : float
        Hubble constant [km s^-1 Mpc^-1].
    omega_m : float
        Matter density parameter.
    omega_lambda : float
        Dark energy density parameter.
    w : float
        Dark energy equation of state parameter; −1 is the cosmological constant.
    """

    h0: float
    omega_m: float
    omega_lambda: float
    w: float = -1.0

    def __post_init__(self) -> None:
        if self.h0 <= 0:
            raise ValueError("H0 must be positive")
        if self.omega_m < 0:
            raise ValueError("Ωm must not be negative")

    @property
    def omega_k(self) -> float:
        """Curvature density parameter: Ωk = 1 − Ωm − ΩΛ ([Hogg1999] Eq. 7)."""
        return 1.0 - self.omega_m - self.omega_lambda

    @classmethod
    def flat(cls, h0: float, omega_m: float, w: float = -1.0) -> Cosmology:
        """Build a flat universe: ΩΛ = 1 − Ωm."""
        return cls(h0=h0, omega_m=omega_m, omega_lambda=1.0 - omega_m, w=w)


@source("Hogg1999", where="Eq. 4", status="read")
def hubble_distance_mpc(cosmology: Cosmology) -> float:
    """Compute the Hubble distance: D_H = c / H0.

    Returns
    -------
    float
        Hubble distance [Mpc].

    References
    ----------
    [Hogg1999] Eq. 4.

    Verification
    ------------
    tests/unit/test_cosmology.py: about 3000 Mpc for h = 1 (the value in the paper).
    """
    return SPEED_OF_LIGHT_KM_S / cosmology.h0


@source("Hogg1999", where="Eq. 3", status="read")
def hubble_time_gyr(cosmology: Cosmology) -> float:
    """Compute the Hubble time: t_H = 1 / H0.

    Returns
    -------
    float
        Hubble time [Gyr, Julian year].

    References
    ----------
    [Hogg1999] Eq. 3.

    Verification
    ------------
    tests/unit/test_cosmology.py: 9.78 Gyr for h = 1 (the value in the paper).
    """
    h0_si = cosmology.h0 * 1e3 / MEGAPARSEC.value  # s^-1
    return 1.0 / h0_si / JULIAN_YEAR_SECONDS.value / 1e9


@source("Hogg1999", where="Eq. 14", status="read")
@source(
    "Brout2022",
    where="H(z) equation (TeX label eq:hz)",
    status="read",
    note="the ΩΛ term is multiplied by (1+z)^{3(1+w)}; Brout writes this for a flat universe, the curvature term is from Hogg",
)
def e_function(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the dimensionless Hubble parameter: E(z) = H(z)/H0.

    E(z) = sqrt( Ωm (1+z)³ + Ωk (1+z)² + ΩΛ (1+z)^{3(1+w)} )

    Parameters
    ----------
    z : float or ndarray
        Redshift, z > −1.
    cosmology : Cosmology

    References
    ----------
    [Hogg1999] Eq. 14; [Brout2022] H(z) equation.

    Validity
    --------
    There is no radiation term.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py: relative difference from astropy.cosmology efunc < 1e-12.
    """
    zp1 = 1.0 + np.asarray(z, dtype=float)
    k = cosmology
    return np.sqrt(k.omega_m * zp1**3 + k.omega_k * zp1**2 + k.omega_lambda * zp1 ** (3.0 * (1.0 + k.w)))


def _integral_from_zero(z: Array, integrand) -> Array:  # type: ignore[no-untyped-def]
    """64-point Gauss-Legendre integral over [0, z]; one pass for an array of z."""
    half = 0.5 * z[..., None]
    nodes = half * (_NODES + 1.0)
    return np.sum(_WEIGHTS * integrand(nodes), axis=-1) * half[..., 0]


@source("Hogg1999", where="Eq. 15", status="read")
def comoving_distance_mpc(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the line-of-sight comoving distance.

    D_C = D_H ∫₀ᶻ dz′ / E(z′)

    Returns
    -------
    float or ndarray
        Comoving distance [Mpc].

    References
    ----------
    [Hogg1999] Eq. 15.

    Validity
    --------
    The 64-point Gauss-Legendre integral agrees with astropy to a relative
    difference below 1e-9 over 0 <= z <= 10; larger z has not been tested.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py; tests/unit/test_cosmology.py (Einstein-de Sitter closed-form solution).
    """
    zz = np.asarray(z, dtype=float)
    if np.any(zz < 0):
        raise ValueError("z must not be negative")
    return hubble_distance_mpc(cosmology) * _integral_from_zero(zz, lambda x: 1.0 / e_function(x, cosmology))


@source("Hogg1999", where="Eq. 16", status="read")
def transverse_comoving_distance_mpc(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the transverse comoving distance.

    Ωk > 0: D_M = D_H/√Ωk · sinh(√Ωk D_C/D_H); Ωk = 0: D_M = D_C;
    Ωk < 0: D_M = D_H/√|Ωk| · sin(√|Ωk| D_C/D_H).

    References
    ----------
    [Hogg1999] Eq. 16.

    Validity
    --------
    |Ωk| < 1e-12 is treated as a flat universe; the branches are continuous at this limit.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py; tests/property/test_cosmology_property.py (continuity as Ωk → 0).
    """
    dc = comoving_distance_mpc(z, cosmology)
    ok = cosmology.omega_k
    if abs(ok) < 1e-12:
        return dc
    dh = hubble_distance_mpc(cosmology)
    sqrt_ok = np.sqrt(abs(ok))
    if ok > 0:
        return dh / sqrt_ok * np.sinh(sqrt_ok * dc / dh)
    return dh / sqrt_ok * np.sin(sqrt_ok * dc / dh)


@source("Hogg1999", where="Eq. 18", status="read")
def angular_diameter_distance_mpc(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the angular diameter distance: D_A = D_M / (1 + z).

    References
    ----------
    [Hogg1999] Eq. 18.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py
    """
    return transverse_comoving_distance_mpc(z, cosmology) / (1.0 + np.asarray(z, dtype=float))


@source("Hogg1999", where="Eq. 21", status="read")
def luminosity_distance_mpc(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the luminosity distance: D_L = (1 + z) D_M.

    References
    ----------
    [Hogg1999] Eq. 21.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py; tests/property/test_cosmology_property.py (cz/H0 in the z → 0 limit).
    """
    return (1.0 + np.asarray(z, dtype=float)) * transverse_comoving_distance_mpc(z, cosmology)


@source("Hogg1999", where="Eq. 25", status="read", note="if D_L is in Mpc, μ = 5 log10(D_L) + 25")
def distance_modulus(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the distance modulus: μ = 5 log10(D_L / 10 pc).

    Returns
    -------
    float or ndarray
        Distance modulus [mag].

    References
    ----------
    [Hogg1999] Eq. 25.

    Validity
    --------
    z > 0; returns −∞ for z = 0.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py: difference from astropy distmod < 1e-9 mag.
    """
    with np.errstate(divide="ignore"):
        return 5.0 * np.log10(luminosity_distance_mpc(z, cosmology)) + 25.0


@source("Hogg1999", where="Eq. 30", status="read")
def lookback_time_gyr(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the lookback time: t_L = t_H ∫₀ᶻ dz′ / [(1+z′) E(z′)].

    Returns
    -------
    float or ndarray
        Lookback time [Gyr].

    References
    ----------
    [Hogg1999] Eq. 30.

    Validity
    --------
    Tested over 0 <= z <= 10.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py: relative difference from astropy lookback_time < 1e-8.
    """
    zz = np.asarray(z, dtype=float)
    if np.any(zz < 0):
        raise ValueError("z must not be negative")
    return hubble_time_gyr(cosmology) * _integral_from_zero(zz, lambda x: 1.0 / ((1.0 + x) * e_function(x, cosmology)))


@source("Hogg1999", where="Eq. 28", status="read")
def comoving_volume_element_mpc3(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the comoving volume per unit solid angle and per unit z.

    dV_C / (dΩ dz) = D_H (1+z)² D_A² / E(z)

    Returns
    -------
    float or ndarray
        Volume element [Mpc³ sr⁻¹].

    References
    ----------
    [Hogg1999] Eq. 28.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py: astropy differential_comoving_volume.
    """
    zz = np.asarray(z, dtype=float)
    da = angular_diameter_distance_mpc(zz, cosmology)
    return hubble_distance_mpc(cosmology) * (1.0 + zz) ** 2 * da**2 / e_function(zz, cosmology)


@source("Hogg1999", where="Eq. 19", status="read", note="the source states that the formula is not valid for Ωk < 0")
def angular_diameter_distance_z1_z2_mpc(z1: ArrayLike, z2: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the angular diameter distance of an object at z2 as measured by an observer at z1.

    D_A12 = [D_M2 sqrt(1 + Ωk D_M1²/D_H²) − D_M1 sqrt(1 + Ωk D_M2²/D_H²)] / (1 + z2)

    References
    ----------
    [Hogg1999] Eq. 19.

    Validity
    --------
    Ωk >= 0 and z1 <= z2. Used for the lens-source distance in gravitational lensing.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py: astropy angular_diameter_distance_z1z2.
    """
    if cosmology.omega_k < -1e-12:
        raise ValueError("Eq. 19 is not valid in a closed universe (Ωk < 0)")
    ok = max(cosmology.omega_k, 0.0)
    dh = hubble_distance_mpc(cosmology)
    dm1 = transverse_comoving_distance_mpc(z1, cosmology)
    dm2 = transverse_comoving_distance_mpc(z2, cosmology)
    ic = dm2 * np.sqrt(1.0 + ok * dm1**2 / dh**2) - dm1 * np.sqrt(1.0 + ok * dm2**2 / dh**2)
    return ic / (1.0 + np.asarray(z2, dtype=float))


@source("Hogg1999", where="Eq. 29", status="read", note="all-sky volume; three curvature branches")
def comoving_volume_mpc3(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the all-sky comoving volume out to z.

    In a flat universe V_C = (4π/3) D_M³; in a curved universe the closed form given in the source.

    Returns
    -------
    float or ndarray
        Comoving volume [Mpc³].

    References
    ----------
    [Hogg1999] Eq. 29.

    Validity
    --------
    |Ωk| < 1e-8 is treated as a flat universe: in the curved branch the difference
    of two large terms loses precision for small |Ωk|.

    Verification
    ------------
    tests/oracle/test_cosmology_oracle.py: astropy comoving_volume.
    """
    dm = transverse_comoving_distance_mpc(z, cosmology)
    ok = cosmology.omega_k
    if abs(ok) < 1e-8:
        return 4.0 * np.pi / 3.0 * dm**3
    dh = hubble_distance_mpc(cosmology)
    x = dm / dh
    sqrt_ok = np.sqrt(abs(ok))
    inverse = np.arcsinh(sqrt_ok * x) if ok > 0 else np.arcsin(sqrt_ok * x)
    return 4.0 * np.pi * dh**3 / (2.0 * ok) * (x * np.sqrt(1.0 + ok * x**2) - inverse / sqrt_ok)


@source("Hogg1999", where="Eq. 9", status="read")
def doppler_redshift(velocity_km_s: ArrayLike) -> Array:
    """Compute the relativistic Doppler redshift from a line-of-sight velocity.

    1 + z = sqrt((1 + v/c) / (1 − v/c))

    References
    ----------
    [Hogg1999] Eq. 9.

    Validity
    --------
    |v| < c. A cosmological redshift is not converted to a velocity with this
    formula; the source stresses that it is meaningful only for peculiar
    velocities.

    Verification
    ------------
    tests/unit/test_cosmology.py: small-velocity limit z ≈ v/c and round trip with the inverse function.
    """
    beta = np.asarray(velocity_km_s, dtype=float) / SPEED_OF_LIGHT_KM_S
    return np.sqrt((1.0 + beta) / (1.0 - beta)) - 1.0


@source("Hogg1999", where="Eq. 9", status="read", note="solution of Eq. 9 for v: β = ((1+z)² − 1)/((1+z)² + 1)")
def doppler_velocity_km_s(z: ArrayLike) -> Array:
    """Compute the line-of-sight velocity from a Doppler redshift.

    References
    ----------
    [Hogg1999] Eq. 9.

    Verification
    ------------
    tests/unit/test_cosmology.py: round trip with ``doppler_redshift``.
    """
    square = (1.0 + np.asarray(z, dtype=float)) ** 2
    return SPEED_OF_LIGHT_KM_S * (square - 1.0) / (square + 1.0)


@source(
    "Hogg1999", where="Eq. 10", status="read", note="v_pec = c (z_obs − z_cos)/(1 + z); small-velocity approximation"
)
def peculiar_velocity_km_s(z_observed: ArrayLike, z_cosmological: ArrayLike) -> Array:
    """Compute the peculiar velocity from the observed and cosmological redshift.

    v_pec = c (z_obs − z_cos) / (1 + z)

    References
    ----------
    [Hogg1999] Eq. 10.

    Validity
    --------
    v_pec << c. The cosmological redshift is used for z in the denominator;
    the source does not say which one to use, and the difference is second order.

    Verification
    ------------
    tests/unit/test_cosmology.py: consistency with the rule (1 + z_obs) = (1 + z_cos)(1 + v/c).
    """
    z_obs = np.asarray(z_observed, dtype=float)
    z_cos = np.asarray(z_cosmological, dtype=float)
    return SPEED_OF_LIGHT_KM_S * (z_obs - z_cos) / (1.0 + z_cos)
