"""Dark energy background quantities for the w0-wa (CPL) model.

The background model is the one of ``kozmos.cosmology.distances``: matter, radiation, curvature and a
dark energy component with the equation of state w(a) = w0 + wa (1 - a) of [Linder2003] (``Cosmology.w``
is w0, ``Cosmology.wa`` is wa). The Friedmann equations, the density scaling of each component, the
critical density and the deceleration parameter are read from [Frieman2008] (arXiv 0803.0982, TeX files
``INCLUDE/sec2.tex`` and ``INCLUDE/sec9.tex``) and [Copeland2006] (arXiv hep-th/0603057).

Conventions: z is the redshift (z > -1), ``Omega_i(z) = rho_i(z) / rho_crit(z)`` with the critical density
at the same redshift ([Frieman2008] Eq. q), so the density parameters of matter, radiation, dark energy
and curvature add to one at every redshift. The dark energy density parameter today is
``Cosmology.omega_lambda`` for every w and wa.

Bouncing models: when E(z)^2 = H^2/H0^2 is zero or negative at some redshift (for example a closed model with a
large dark energy density such as ``Cosmology(70, 0.1, 1.5)``) the model has no real expansion rate there. The
density parameters, q and w_tot are then NaN at those redshifts, never a finite value computed from a negative
denominator.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from kozmos._sources import source
from kozmos.core.constants import GRAVITATIONAL_CONSTANT, MEGAPARSEC, SPEED_OF_LIGHT
from kozmos.cosmology.distances import (
    _NODES,
    _WEIGHTS,
    Cosmology,
    e_function,
    hubble_time_gyr,
)

Array = NDArray[np.float64]

_SCAN_POINTS = 4001  # grid points of the sign-change scan in ln(1+z): step 9.8e-4 for z_max = 50
_POLE_RTOL = 1e-3  # a root must satisfy |f(root)| <= _POLE_RTOL max(|f| at the bracket ends), else it is a pole


def _log_de_ratio(zp1: Array, k: Cosmology) -> Array:
    """ln of rho_DE(z)/rho_DE(0) for w(a) = w + wa (1 - a); zp1 = 1 + z."""
    return 3.0 * (1.0 + k.w + k.wa) * np.log(zp1) - 3.0 * k.wa * (zp1 - 1.0) / zp1


def _densities(z: ArrayLike, k: Cosmology) -> tuple[Array, Array, Array, Array]:
    """Matter, radiation, curvature and dark energy terms of E(z)^2 (density relative to critical today)."""
    zp1 = 1.0 + np.asarray(z, dtype=float)
    return (
        k.omega_m * zp1**3,
        k.omega_r * zp1**4,
        k.omega_k * zp1**2,
        k.omega_lambda * np.exp(_log_de_ratio(zp1, k)),
    )


def _e_squared(z: ArrayLike, k: Cosmology) -> Array:
    """E(z)^2 = H(z)^2/H0^2 as the sum of the four terms of ``_densities``."""
    m = _densities(z, k)
    return m[0] + m[1] + m[2] + m[3]


def _fraction(term: Array, total: Array) -> Array:
    """``term / total`` where ``total`` = E(z)^2 is positive; NaN where E^2 <= 0 or not finite (no real H)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        value = term / total
    return np.where(total > 0.0, value, np.nan)[()]


@source(
    "Linder2003",
    where="TeX label wa (expprl2.tex lines 147-150): w(a) = w0 + wa (1-a) = w0 + wa z/(1+z)",
    status="read",
)
def de_eos(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the dark energy equation of state: w(z) = w0 + wa z/(1+z).

    Parameters
    ----------
    z : float or ndarray
        Redshift, z > -1.
    cosmology : Cosmology
        ``cosmology.w`` is w0 and ``cosmology.wa`` is wa.

    Returns
    -------
    float or ndarray
        w(z) = p/rho of the dark energy, dimensionless.

    References
    ----------
    [Linder2003] equation after the sentence "I suggest a new model" (TeX label wa).

    Validity
    --------
    The CPL form is a parametrization, bounded for z -> infinity (w -> w0 + wa); it is not valid for
    z -> -1 (the far future), where w diverges for wa != 0.

    Verification
    ------------
    tests/oracle/test_dark_energy_oracle.py: astropy Flatw0waCDM.w, exact to rounding;
    tests/unit/test_dark_energy.py: w(0) = w0 and the closed form of the density factor.
    """
    zz = np.asarray(z, dtype=float)
    return cosmology.w + cosmology.wa * zz / (1.0 + zz)


@source(
    "Frieman2008",
    where="Eq. rho_scaling (sec2.tex lines 76-79)",
    status="read",
    note="rho_i ∝ exp[3 ∫_0^z (1+w_i) dln(1+z')]; the integral is done in closed form for the CPL w(z)",
)
@source(
    "Linder2003",
    where="TeX label wa and the sentence after it (expprl2.tex lines 147-152)",
    status="read",
    note="the exponential factor resolves to a^{-3(1+w0+wa)} exp(-3 wa (1-a)) for w(a) = w0 + wa (1-a)",
)
def de_rho_ratio(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the dark energy density relative to today: rho_DE(z) / rho_DE(0).

    rho_DE(z)/rho_DE(0) = (1+z)^{3(1+w0+wa)} exp(-3 wa z/(1+z))

    Parameters
    ----------
    z : float or ndarray
        Redshift, z > -1.
    cosmology : Cosmology

    Returns
    -------
    float or ndarray
        Density ratio, dimensionless; 1 at z = 0 and for w = -1, wa = 0 at every z.

    References
    ----------
    [Frieman2008] Eq. rho_scaling; [Linder2003] resolved form after Eq. wa.

    Validity
    --------
    For the CPL equation of state only. The closed form is checked against the integral of Eq.
    rho_scaling in the tests.

    Verification
    ------------
    tests/unit/test_dark_energy.py: equality with the numerical integral of 3 (1+w) dln(1+z);
    tests/oracle/test_dark_energy_oracle.py: astropy de_density_scale, relative difference < 1e-14.
    """
    return np.exp(_log_de_ratio(1.0 + np.asarray(z, dtype=float), cosmology))


@source(
    "Frieman2008",
    where="Eq. q (sec2.tex line 140): Omega_i(z) = rho_i(z) / rho_crit(z); rho_M ∝ (1+z)^3 (sec2.tex line 86)",
    status="read",
    note="Omega_m(z) = Omega_m (1+z)^3 / E(z)^2, with E(z)^2 = H(z)^2/H0^2 from Eq. feq",
)
def omega_matter(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the matter density parameter at redshift z: Omega_m(z) = Omega_m (1+z)^3 / E(z)^2.

    Parameters
    ----------
    z : float or ndarray
        Redshift, z > -1.
    cosmology : Cosmology

    Returns
    -------
    float or ndarray
        Omega_m(z), dimensionless.

    References
    ----------
    [Frieman2008] Eq. q (definition of Omega_i(z)) and the scaling of rho_M below Eq. rhow.

    Validity
    --------
    Matter is pressureless; baryons and dark matter are not separated. NaN where E(z)^2 <= 0 (a bouncing
    model has no real expansion rate there).

    Verification
    ------------
    tests/oracle/test_dark_energy_oracle.py: astropy Om(z), relative difference < 1e-14;
    tests/unit/test_dark_energy.py: the four density parameters add to one; NaN where E^2 <= 0.
    """
    m = _densities(z, cosmology)
    return _fraction(m[0], m[0] + m[1] + m[2] + m[3])


@source(
    "Frieman2008",
    where="Eq. q (sec2.tex line 140); rho_R ∝ (1+z)^4 (sec2.tex lines 87-88)",
    status="read",
)
def omega_radiation(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the radiation density parameter at redshift z: Omega_r(z) = Omega_r (1+z)^4 / E(z)^2.

    Parameters
    ----------
    z : float or ndarray
        Redshift, z > -1.
    cosmology : Cosmology

    Returns
    -------
    float or ndarray
        Omega_r(z), dimensionless.

    References
    ----------
    [Frieman2008] Eq. q and the scaling of rho_R after Eq. rhow.

    Validity
    --------
    Photons plus massless neutrinos (a single ``omega_r``). NaN where E(z)^2 <= 0.

    Verification
    ------------
    tests/oracle/test_dark_energy_oracle.py: astropy (Ogamma(z) + Onu(z)), relative difference < 1e-14;
    tests/unit/test_dark_energy.py: NaN where E^2 <= 0.
    """
    m = _densities(z, cosmology)
    return _fraction(m[1], m[0] + m[1] + m[2] + m[3])


@source(
    "Frieman2008",
    where="Eq. q (sec2.tex line 140) and Eq. rho_scaling with w = -1 (sec2.tex lines 76-79, 88-89)",
    status="read",
    note="Omega_DE(z) = Omega_DE rho_DE(z)/rho_DE(0) / E(z)^2",
)
def omega_de(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the dark energy density parameter at redshift z.

    Omega_DE(z) = Omega_DE,0 [rho_DE(z)/rho_DE(0)] / E(z)^2

    Parameters
    ----------
    z : float or ndarray
        Redshift, z > -1.
    cosmology : Cosmology
        ``cosmology.omega_lambda`` is the dark energy density parameter today.

    Returns
    -------
    float or ndarray
        Omega_DE(z), dimensionless.

    References
    ----------
    [Frieman2008] Eq. q and Eq. rho_scaling.

    Validity
    --------
    CPL equation of state; the dark energy is assumed smooth. NaN where E(z)^2 <= 0.

    Verification
    ------------
    tests/oracle/test_dark_energy_oracle.py: astropy Ode(z), relative difference < 1e-14;
    tests/unit/test_dark_energy.py: NaN where E^2 <= 0.
    """
    m = _densities(z, cosmology)
    return _fraction(m[3], m[0] + m[1] + m[2] + m[3])


@source(
    "Frieman2008",
    where="Eq. feq (sec2.tex line 48): H^2 = 8 pi G rho / 3 - k/a^2 + Lambda/3",
    status="read",
    note="derivation in this project from Eq. feq: Omega_k(z) = -k/(a^2 H^2) = Omega_k (1+z)^2 / E(z)^2",
)
@source("Hogg1999", where="Eq. 14 (TeX label eq:ez, cosmography.tex lines 238-241)", status="read")
def omega_curvature(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the curvature density parameter at redshift z: Omega_k(z) = Omega_k (1+z)^2 / E(z)^2.

    Parameters
    ----------
    z : float or ndarray
        Redshift, z > -1.
    cosmology : Cosmology

    Returns
    -------
    float or ndarray
        Omega_k(z), dimensionless; zero for a flat model.

    References
    ----------
    [Frieman2008] Eq. feq (the -k/a^2 term); [Hogg1999] Eq. 14 (the Omega_k (1+z)^2 term of E(z)).

    Validity
    --------
    Derivation in this project: dividing Eq. feq by H^2 and writing the k term as Omega_k (1+z)^2 H0^2.
    NaN where E(z)^2 <= 0.

    Verification
    ------------
    tests/oracle/test_dark_energy_oracle.py: astropy Ok(z), relative difference < 1e-14;
    tests/unit/test_dark_energy.py: NaN where E^2 <= 0.
    """
    m = _densities(z, cosmology)
    return _fraction(m[2], m[0] + m[1] + m[2] + m[3])


@source("Hogg1999", where="Eq. 14 (TeX label eq:ez, cosmography.tex lines 238-246): H(z) = H0 E(z)", status="read")
def hubble_parameter_km_s_mpc(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the Hubble parameter at redshift z: H(z) = H0 E(z).

    Parameters
    ----------
    z : float or ndarray
        Redshift, z > -1.
    cosmology : Cosmology

    Returns
    -------
    float or ndarray
        H(z) [km s^-1 Mpc^-1].

    References
    ----------
    [Hogg1999] Eq. 14 and the sentence after it ("H(z) = H0 E(z)").

    Validity
    --------
    E(z) of ``kozmos.cosmology.distances.e_function`` (matter, radiation, curvature, CPL dark energy).

    Verification
    ------------
    tests/oracle/test_dark_energy_oracle.py: astropy H(z), relative difference < 1e-14.
    """
    return cosmology.h0 * e_function(z, cosmology)


def _h_si(z: ArrayLike, k: Cosmology) -> Array:
    """Hubble parameter in s^-1."""
    return k.h0 * 1e3 / MEGAPARSEC.value * e_function(z, k)


@source(
    "Copeland2006",
    where="text after Eq. (conteq) and Eq. (acceleq): rho_c(t) = 3 H^2(t) / 8 pi G (hepthv3.tex line 632)",
    status="read",
)
@source("Frieman2008", where="sec2.tex line 93: rho_crit = 3 H0^2 / 8 pi G", status="read")
def critical_density_kg_m3(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the critical density at redshift z: rho_c(z) = 3 H(z)^2 / (8 pi G).

    Parameters
    ----------
    z : float or ndarray
        Redshift, z > -1.
    cosmology : Cosmology

    Returns
    -------
    float or ndarray
        Critical (mass) density [kg m^-3]; the energy density divided by c^2.

    References
    ----------
    [Copeland2006] rho_c(t) = 3 H^2(t)/(8 pi G); [Frieman2008] rho_crit = 3 H0^2/(8 pi G).

    Validity
    --------
    G is the CODATA 2022 value of ``kozmos.core.constants`` (relative uncertainty 2.2e-5), which
    limits the absolute accuracy of the density; the ratio to another density computed with the same G
    is not affected. Both sources use c = 1; the SI value is the mass density.

    Verification
    ------------
    tests/oracle/test_dark_energy_oracle.py: astropy critical_density, relative difference < 1e-14.
    """
    return 3.0 * _h_si(z, cosmology) ** 2 / (8.0 * np.pi * GRAVITATIONAL_CONSTANT.value)


@source(
    "Frieman2008",
    where="Eq. q (sec2.tex line 140) with Eq. rho_scaling (sec2.tex lines 76-79)",
    status="read",
    note="derivation in this project: rho_DE(z) = Omega_DE,0 rho_crit(0) rho_DE(z)/rho_DE(0)",
)
def de_mass_density_kg_m3(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the dark energy (mass) density at redshift z.

    rho_DE(z) = Omega_DE,0 rho_crit(0) (1+z)^{3(1+w0+wa)} exp(-3 wa z/(1+z))

    Parameters
    ----------
    z : float or ndarray
        Redshift, z > -1.
    cosmology : Cosmology

    Returns
    -------
    float or ndarray
        Dark energy mass density [kg m^-3].

    References
    ----------
    [Frieman2008] Eq. q and Eq. rho_scaling.

    Validity
    --------
    Same as ``critical_density_kg_m3`` (CODATA 2022 G).

    Verification
    ------------
    tests/oracle/test_dark_energy_oracle.py: astropy Ode0 * critical_density0 * de_density_scale(z).
    """
    rho_crit0 = critical_density_kg_m3(0.0, cosmology)
    return cosmology.omega_lambda * rho_crit0 * de_rho_ratio(z, cosmology)


@source(
    "Frieman2008",
    where="sec2.tex lines 59-62: Lambda = 8 pi G rho_VAC = -8 pi G p_VAC",
    status="read",
    note="derivation in this project: Lambda = 8 pi G Omega_Lambda rho_crit(0) / c^2 = 3 H0^2 Omega_Lambda / c^2 (the "
    "source uses c = 1, so Lambda has the dimension of H^2; dividing by c^2 gives m^-2)",
)
def cosmological_constant_lambda(cosmology: Cosmology) -> float:
    """Compute the cosmological constant for w = -1: Lambda = 3 H0^2 Omega_Lambda / c^2.

    Parameters
    ----------
    cosmology : Cosmology
        Must have ``w == -1`` and ``wa == 0``.

    Returns
    -------
    float
        Lambda [m^-2]. The name carries no unit suffix because the suffix grammar of
        ``kozmos.core.units`` reads ``_m2`` as m^2 and cannot express m^-2.

    References
    ----------
    [Frieman2008] Lambda = 8 pi G rho_VAC with p_VAC = -rho_VAC (sec2.tex lines 59-62).

    Validity
    --------
    Raises ``ValueError`` for w != -1 or wa != 0, where the dark energy density is not constant and no
    single Lambda exists. The value 3 H0^2 Omega_Lambda / c^2 does not depend on G.

    Verification
    ------------
    tests/unit/test_dark_energy.py: closed form and ratio to 8 pi G rho_DE / c^2;
    tests/oracle/test_dark_energy_oracle.py: astropy Ode0 * critical_density0 * 8 pi G / c^2.
    """
    if cosmology.w != -1.0 or cosmology.wa != 0.0:
        raise ValueError("a cosmological constant requires w = -1 and wa = 0")
    h0_si = cosmology.h0 * 1e3 / MEGAPARSEC.value
    return 3.0 * h0_si**2 * cosmology.omega_lambda / SPEED_OF_LIGHT.value**2


@source(
    "Frieman2008",
    where="Eq. q (sec2.tex line 136-137): q = -a_ddot/(a H^2) = (1/2) sum_i Omega_i (1 + 3 w_i)",
    status="read",
    note="components: matter w=0, radiation w=1/3 (sec2.tex lines 85-88), dark energy w(z) of the CPL model; the "
    "curvature term has w=-1/3 and drops out",
)
def deceleration_parameter(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the deceleration parameter: q(z) = (1/2) sum_i Omega_i(z) [1 + 3 w_i(z)].

    q(z) = (1/2) Omega_m(z) + Omega_r(z) + (1/2) Omega_DE(z) [1 + 3 w(z)]

    Parameters
    ----------
    z : float or ndarray
        Redshift, z > -1.
    cosmology : Cosmology

    Returns
    -------
    float or ndarray
        q(z), dimensionless; q < 0 means accelerating expansion.

    References
    ----------
    [Frieman2008] Eq. q.

    Validity
    --------
    Curvature enters only through the density parameters (an Omega_k term has 1 + 3 w = 0). NaN where
    E(z)^2 <= 0: q diverges as E^2 -> 0 and has no meaning where H^2 is negative.

    Verification
    ------------
    tests/unit/test_dark_energy.py: q = -1 - (1/2) dln(H^2)/dln(a) by finite differences of
    e_function, and the lambda-CDM closed form q = (1/2) Omega_m(z) - Omega_Lambda(z); NaN where E^2 <= 0
    (before the fix q(z = 1) = 11.0 for Cosmology(70, 0.1, 1.5), where E^2 < 0);
    tests/oracle/test_dark_energy_oracle.py: q from astropy density parameters.
    """
    om = omega_matter(z, cosmology)
    orad = omega_radiation(z, cosmology)
    ode = omega_de(z, cosmology)
    return 0.5 * om + orad + 0.5 * ode * (1.0 + 3.0 * de_eos(z, cosmology))


@source(
    "Frieman2008",
    where="Eq. q (sec2.tex line 136-137) and w_i = p_i/rho_i (sec2.tex line 68)",
    status="read",
    note="derivation in this project: w_tot = sum_i rho_i w_i / sum_i rho_i = [Omega_r/3 + Omega_DE w(z)] / "
    "[Omega_m + Omega_r + Omega_DE]; for a flat model q = (1 + 3 w_tot)/2",
)
def total_eos(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the total effective equation of state of the fluids: w_tot = p_tot / rho_tot.

    Curvature is not a fluid and is excluded from both sums.

    Parameters
    ----------
    z : float or ndarray
        Redshift, z > -1.
    cosmology : Cosmology

    Returns
    -------
    float or ndarray
        w_tot(z), dimensionless.

    References
    ----------
    [Frieman2008] Eq. q and the definition w_i = p_i / rho_i.

    Validity
    --------
    Acceleration of a flat model is equivalent to w_tot < -1/3; with curvature use q(z) instead. NaN where
    E(z)^2 <= 0.

    Verification
    ------------
    tests/unit/test_dark_energy.py: q = (1 + 3 w_tot)/2 for flat models, w_tot = 0 for Einstein-de Sitter and
    1/3 for pure radiation; NaN where E^2 <= 0.
    """
    om = omega_matter(z, cosmology)
    orad = omega_radiation(z, cosmology)
    ode = omega_de(z, cosmology)
    return (orad / 3.0 + ode * de_eos(z, cosmology)) / (om + orad + ode)


def _first_sign_change(f, z_max: float, what: str = "f") -> float | None:  # type: ignore[no-untyped-def]
    """First root of ``f`` on [0, z_max] found by a scan in ln(1+z) followed by ``brentq``; None if there is none.

    A root is a grid node where ``f`` is exactly zero or a bracket with a sign change. Raises ``ValueError`` if
    ``f`` is not finite at a grid node before the first root (a sign change across such a point is not a root),
    or if the refined root is a pole (|f(root)| not small against ``f`` at the bracket ends).
    """
    if not z_max > 0.0:
        raise ValueError("z_max must be positive")
    grid = np.expm1(np.linspace(0.0, np.log1p(z_max), _SCAN_POINTS))
    values = np.asarray(f(grid), dtype=float)
    bad = np.nonzero(~np.isfinite(values))[0]
    finite = int(bad[0]) if bad.size else values.size  # values[:finite] is the finite prefix
    signs = np.sign(values[:finite])
    zeros = np.nonzero(signs == 0.0)[0]
    changes = np.nonzero(signs[:-1] * signs[1:] < 0)[0]
    if zeros.size and (changes.size == 0 or zeros[0] < changes[0]):
        return float(grid[zeros[0]])  # the root is exactly on a grid node
    if changes.size:
        i = int(changes[0])

        def scalar(x: float) -> float:
            return float(f(x))

        root = float(brentq(scalar, grid[i], grid[i + 1], xtol=1e-14, rtol=4 * np.finfo(float).eps))
        residual = scalar(root)
        ends = max(abs(values[i]), abs(values[i + 1]))
        if not np.isfinite(residual) or abs(residual) > _POLE_RTOL * ends:
            raise ValueError(
                f"the sign change of {what} near z = {root:.6g} is a pole (E^2 <= 0 or divergence), not a root"
            )
        return root
    if finite < values.size:
        raise ValueError(
            f"{what} is not finite at z = {grid[finite]:.6g} (E^2 <= 0: the model bounces) before any root was found"
        )
    return None


@source(
    "Frieman2008",
    where="Eq. q (sec2.tex line 136-141): q = -a_ddot/(a H^2); acceleration for q < 0",
    status="read",
    note="this project's definition: the expansion must accelerate today (q(0) < 0); the onset is the first zero of "
    "q(z) found going back from z = 0, i.e. where the accelerating phase that includes today began",
)
def acceleration_onset_redshift(cosmology: Cosmology, z_max: float = 50.0) -> float:
    """Find the redshift at which the present accelerating phase began: q(z_acc) = 0 with q < 0 on [0, z_acc).

    The expansion must accelerate today (q(0) < 0). Going back in time from z = 0, q stays negative until the
    first zero, which is the returned redshift; a model that does not accelerate today has no such onset and
    raises ``ValueError`` (the first zero of q for such a model is where an earlier accelerating phase ENDS).

    Parameters
    ----------
    cosmology : Cosmology
    z_max : float
        Upper limit of the search, default 50.

    Returns
    -------
    float
        Redshift z_acc with q(z_acc) = 0, the first sign change of q(z) from z = 0 outwards.

    Raises
    ------
    ValueError
        If q(0) >= 0 (no acceleration today); if E(z)^2 <= 0 or q is not finite on the scan before the first zero
        (a sign change across a pole of q is not a root); or if q stays negative on [0, z_max].

    References
    ----------
    [Frieman2008] Eq. q and the statement that acceleration needs q < 0 (a_ddot > 0).

    Validity
    --------
    The scan uses 4001 points equally spaced in ln(1+z) (step 9.8e-4 for z_max = 50); two roots closer than
    one grid step would be missed (tests/unit/test_dark_energy.py shows a pair 2.5e-3 apart is found, and that
    a 50-point grid misses it). The root is refined with a bracketing method to the double-precision limit,
    and rejected if q at the root is not small against q at the bracket ends (a pole).

    Verification
    ------------
    tests/unit/test_dark_energy.py: lambda-CDM closed form (1+z)^3 = 2 Omega_Lambda / Omega_m for flat models without
    radiation; the w0 = -0.3, wa = -0.9 model (q(0) = +0.185, formerly returned 0.390 where acceleration ends)
    and Cosmology(70, 0.1, 1.5) (formerly returned the pole z = 0.9166 with q = 1.7e14) raise;
    tests/oracle/test_dark_energy_oracle.py: sign change of astropy-derived q.
    """
    q0 = float(deceleration_parameter(0.0, cosmology))
    if not np.isfinite(q0) or q0 >= 0.0:
        raise ValueError(
            f"the expansion does not accelerate today (q(0) = {q0:.6g}); there is no onset of a present accelerating phase"
        )
    root = _first_sign_change(lambda zz: deceleration_parameter(zz, cosmology), z_max, "q(z)")
    if root is None:
        raise ValueError(f"q(z) stays negative on [0, {z_max}]: no onset of acceleration in the search range")
    return root


@source(
    "Frieman2008",
    where="Fig. growth caption (sec2.tex line 278): dark energy dominates when 1+z = (Omega_M/Omega_DE)^{1/(3w)}",
    status="read",
    note="the caption is the equality rho_M = rho_DE for constant w; here generalized by solving "
    "Omega_m (1+z)^3 = Omega_DE rho_DE(z)/rho_DE(0) with the CPL density factor, for the equality that "
    "begins the dark-energy-dominated era of today (this project's definition)",
)
def matter_de_equality_redshift(cosmology: Cosmology, z_max: float = 50.0) -> float:
    """Find the redshift at which the present dark-energy-dominated era began: rho_m(z) = rho_DE(z).

    Dark energy must dominate today (rho_DE(0) >= rho_m(0), i.e. Omega_DE >= Omega_m); going back in time the
    first equality is the returned redshift. If matter dominates today the first equality would instead mark
    the END of an earlier dark-energy-dominated era, so ``ValueError`` is raised.

    Parameters
    ----------
    cosmology : Cosmology
    z_max : float
        Upper limit of the search, default 50.

    Returns
    -------
    float
        z_eq of the first sign change of ln[rho_m(z)/rho_DE(z)] from z = 0 outwards (0 if the densities are
        equal today).

    Raises
    ------
    ValueError
        If Omega_m or Omega_DE is not positive, if matter dominates today (Omega_m > Omega_DE), or if there is
        no equality on [0, z_max].

    References
    ----------
    [Frieman2008] Fig. growth caption: 1 + z = (Omega_M/Omega_DE)^{1/(3w)} for constant w.

    Validity
    --------
    Radiation and curvature are ignored in the condition. For constant w and a flat model the root is
    the closed form of the caption, which the tests use. The scan resolution is that of
    ``acceleration_onset_redshift``.

    Verification
    ------------
    tests/unit/test_dark_energy.py: closed form for constant w, both signs of 1 + w; a model with matter
    dominating today but dark energy dominating earlier raises;
    tests/oracle/test_dark_energy_oracle.py: equality of astropy Om(z) and Ode(z) at the returned redshift
    for a flat model without radiation.
    """
    if cosmology.omega_m <= 0.0 or cosmology.omega_lambda <= 0.0:
        raise ValueError("matter-dark energy equality needs positive Omega_m and Omega_DE")
    if cosmology.omega_m > cosmology.omega_lambda:
        raise ValueError(
            f"matter dominates dark energy today (Omega_m = {cosmology.omega_m:.6g} > Omega_DE = "
            f"{cosmology.omega_lambda:.6g}); there is no onset of a present dark-energy-dominated era"
        )

    def log_ratio(zz: Array) -> Array:
        zp1 = 1.0 + np.asarray(zz, dtype=float)
        return np.log(cosmology.omega_m / cosmology.omega_lambda) + 3.0 * np.log(zp1) - _log_de_ratio(zp1, cosmology)

    root = _first_sign_change(log_ratio, z_max, "ln(rho_m/rho_DE)")
    if root is None:
        raise ValueError(f"rho_m and rho_DE are not equal on [0, {z_max}]")
    return root


@source(
    "Frieman2008",
    where="sec2.tex lines 85-88: rho_M ∝ (1+z)^3, rho_R ∝ (1+z)^4",
    status="read",
    note="derivation in this project: Omega_m (1+z)^3 = Omega_r (1+z)^4 gives 1+z = Omega_m / Omega_r",
)
def matter_radiation_equality_redshift(cosmology: Cosmology) -> float:
    """Compute the redshift of matter-radiation equality: 1 + z_eq = Omega_m / Omega_r.

    Parameters
    ----------
    cosmology : Cosmology
        ``omega_r`` must be positive.

    Returns
    -------
    float
        z_eq, dimensionless.

    References
    ----------
    [Frieman2008] Scalings rho_M ∝ (1+z)^3 and rho_R ∝ (1+z)^4.

    Validity
    --------
    Massless neutrinos only (all of ``omega_r`` scales as (1+z)^4); the value with massive neutrinos
    differs. The dark energy density is not included in the condition.

    Verification
    ------------
    tests/unit/test_dark_energy.py: equal densities at the returned redshift.
    """
    if cosmology.omega_r <= 0.0:
        raise ValueError("matter-radiation equality needs omega_r > 0")
    return cosmology.omega_m / cosmology.omega_r - 1.0


@source(
    "Frieman2008",
    where="Eq. cosmic_time (sec2.tex lines 244-247): t(z) = ∫_z^∞ dz'/[(1+z') H(z')]",
    status="read",
    note="the substitution a = u^2 and the factor u^3 are a derivation in this project (below)",
)
@source(
    "ChevallierPolarski2001",
    where="age integral t(z) = ∫_0^{1/(1+z)} dx / (x H(x)) (age.tex line 343)",
    status="read",
)
def age_gyr(z: ArrayLike, cosmology: Cosmology) -> Array:
    """Compute the age of the universe at redshift z: t(z) = ∫_0^{a} da' / (a' H(a')), a = 1/(1+z).

    With a = u^2 the integral is regular at the lower limit:
    t(z) = t_H ∫_0^{sqrt(a)} 2 u^3 du / sqrt(Omega_r + Omega_m u^2 + Omega_k u^4 + Omega_DE u^8 f(u^2)),
    f(a) = a^{-3(1+w0+wa)} exp(-3 wa (1-a)). The integral is evaluated with 64-point Gauss-Legendre.

    Parameters
    ----------
    z : float or ndarray
        Redshift, z > -1.
    cosmology : Cosmology

    Returns
    -------
    float or ndarray
        Age [Gyr, Julian year]; ``age_gyr(0, c)`` is the present age.

    References
    ----------
    [Frieman2008] Eq. cosmic_time; [ChevallierPolarski2001] the integral over the scale factor.

    Validity
    --------
    Requires Omega_m > 0 or Omega_r > 0 (otherwise the age diverges). The integrand has no singularity but
    the 64-point rule is not adaptive; the measured difference from astropy is in the tests. A model
    whose dark energy dominates at early times (w0 + wa > -1/3 with a positive dark energy density) has a
    different early expansion and the quadrature is not tested there. Closed (Omega_k < 0) universes that
    recollapse have no age at some z; these are not detected.

    Verification
    ------------
    tests/oracle/test_dark_energy_oracle.py: astropy age, relative difference measured in the test;
    tests/unit/test_dark_energy.py: Einstein-de Sitter closed form t = (2/3) t_H (1+z)^{-3/2}.
    """
    k = cosmology
    if k.omega_m <= 0.0 and k.omega_r <= 0.0:
        raise ValueError("the age is infinite without matter or radiation")
    zz = np.asarray(z, dtype=float)
    top = (1.0 + zz) ** -0.5  # sqrt(a)
    u = 0.5 * top[..., None] * (_NODES + 1.0)
    a = u * u
    de = k.omega_lambda * u**8 * np.exp(-3.0 * (1.0 + k.w + k.wa) * np.log(a) - 3.0 * k.wa * (1.0 - a))
    denominator = np.sqrt(k.omega_r + k.omega_m * u**2 + k.omega_k * u**4 + de)
    integrand = 2.0 * u**3 / denominator
    return hubble_time_gyr(k) * 0.5 * top * np.sum(_WEIGHTS * integrand, axis=-1)


@source(
    "Frieman2008",
    where="sec9.tex lines 51-58: for w<-1, rho_DE ∝ a^beta, beta = -3(1+w); (t_inf - t_0) ≃ 2/(beta H0)",
    status="read",
    note="the source gives this as an approximation (≃) for a dark-energy-dominated universe",
)
def big_rip_time_gyr(cosmology: Cosmology) -> float:
    """Compute the time from today until the big rip: t_inf - t_0 = 2 / (beta H0), beta = -3 (1 + w).

    Parameters
    ----------
    cosmology : Cosmology
        Must have ``w < -1`` and ``wa == 0``.

    Returns
    -------
    float
        Time until the scale factor diverges [Gyr, Julian year].

    Raises
    ------
    ValueError
        If w >= -1 or wa != 0.

    References
    ----------
    [Frieman2008] sec9.tex lines 51-58 (the "big rip" scenario).

    Validity
    --------
    The source writes the result with an approximation sign: matter and curvature are neglected, i.e.
    the formula is the exact time for a universe filled only with the phantom component and is an
    estimate otherwise. The size of the error for Omega_m = 0.3 is measured in the tests against the
    integral of da / (a H) from a = 1 to infinity.

    Verification
    ------------
    tests/unit/test_dark_energy.py: exact for Omega_DE = 1 and compared with the numerical integral for
    Omega_m = 0.3.
    """
    if cosmology.w >= -1.0 or cosmology.wa != 0.0:
        raise ValueError("a big rip needs a constant w < -1")
    beta = -3.0 * (1.0 + cosmology.w)
    return 2.0 * hubble_time_gyr(cosmology) / beta
