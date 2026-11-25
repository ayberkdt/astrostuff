"""Polytropes: the Lane-Emden equation and structure quantities of polytropic spheres.

The polytropic equation of state P = K ρ^{1+1/n} and the Lane-Emden equation were read from the
text of [Liu1995] (arXiv:astro-ph/9512061), and the ξ_1 and ρ_c/ρ̄ values and the closed-form
solutions for n = 0, 1, 5 from the text of [Bludman2010] (arXiv:1008.4281), on 25 November 2026.
Numbered equations of the sources are cited with ``Eq.``; unnumbered ones are cited by
section or table name.

The dimensionless variables are the same as in [Liu1995] Eq. 3: ξ = A r,
A² = 4πG ρ_c^{(n-1)/n} / ((n+1) K) and θ = (ρ/ρ_c)^{1/n}. The functions in this module
use SI units; n is a single number, not an array, because a separate differential equation
is solved for each n (the result is cached).

In this file the mean molecular weight μ is multiplied by the proton mass; [Liu1995] Eq. 42 uses the mass of
atomic hydrogen (difference 0.05%). The Eddington functions in the ``scaling`` module instead use the atomic
mass unit, like their source [Bludman2010]; it differs from the proton mass by 0.73%. With the same μ the results
of the two modules diverge by this much (Sun, n = 3, μ = 0.62: central temperature 1.2241e7 and 1.2151e7 K).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from kozmos._sources import source
from kozmos.core.constants import BOLTZMANN, GRAVITATIONAL_CONSTANT, PROTON_MASS

Array = NDArray[np.float64]

# First step from the center: Taylor series [Liu1995] Eq. 5, [Bludman2010] Table II.
_XI_START = 1e-5
_XI_END = 1e5


def _valid_n(n: float) -> float:
    n = float(n)
    if not 0.0 <= n < 5.0:
        raise ValueError("the polytropic index must satisfy 0 <= n < 5 (the radius is infinite for n >= 5)")
    return n


@lru_cache(maxsize=64)
def _solve(n: float):  # type: ignore[no-untyped-def]
    """Solve the Lane-Emden equation to the first zero with high accuracy; (solution, ξ_1, θ'(ξ_1))."""

    def rhs(x, y):  # type: ignore[no-untyped-def]
        return [y[1], -(max(y[0], 0.0) ** n) - 2.0 * y[1] / x]

    def zero_crossing(x, y):  # type: ignore[no-untyped-def]
        return y[0]

    zero_crossing.terminal = True  # type: ignore[attr-defined]
    zero_crossing.direction = -1  # type: ignore[attr-defined]
    x0 = _XI_START
    y0 = [1.0 - x0**2 / 6.0 + n * x0**4 / 120.0, -x0 / 3.0 + n * x0**3 / 30.0]
    c = solve_ivp(
        rhs, (x0, _XI_END), y0, method="DOP853", rtol=1e-12, atol=1e-14, events=zero_crossing, dense_output=True
    )
    if c.t_events[0].size == 0:
        raise ValueError(f"θ did not reach its first zero for n = {n}")
    return c.sol, float(c.t_events[0][0]), float(c.y_events[0][0][1])


@source("Liu1995", where="Eq. 2, 4, 5 (Lane-Emden equation, central conditions, power series)", status="read")
@source("Bludman2010", where="Table II (Taylor series 1 - ξ²/6 + n ξ⁴/120 ...)", status="read")
def lane_emden_solve(n: float, n_points: int = 400) -> tuple[Array, Array, Array]:
    """Solve the Lane-Emden equation from the center to the first zero.

    θ'' + (2/ξ) θ' = -θ^n,   θ(0) = 1,  θ'(0) = 0

    Parameters
    ----------
    n : float
        Polytropic index, 0 <= n < 5.
    n_points : int
        Number of equally spaced output points on [0, ξ_1] (at least 2).

    Returns
    -------
    xi, theta, dtheta_dxi : ndarray
        Dimensionless radius, θ and its derivative. The last point is the surface, θ = 0.

    References
    ----------
    [Liu1995] Eq. 2, 4, 5; [Bludman2010] Table II.

    Validity
    --------
    For n >= 5, θ does not reach zero (infinite radius at n = 5). For 0 < n < 1 the derivative of θ^n
    is infinite at the surface; ξ_1 is still found (for n = 0.5 it agrees with the value of an independent solver within 1e-8).

    Verification
    ------------
    tests/unit/test_stars.py: closed-form solutions for n = 0, 1, 5 and the ξ_1 values of [Bludman2010] Table I.
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    n = _valid_n(n)
    solution, xi1, _ = _solve(n)
    xi = np.linspace(0.0, xi1, int(n_points))
    theta = np.ones_like(xi)
    dtheta = np.zeros_like(xi)
    inside = xi >= _XI_START
    y = solution(xi[inside])
    theta[inside], dtheta[inside] = y[0], y[1]
    small = ~inside
    theta[small] = 1.0 - xi[small] ** 2 / 6.0 + n * xi[small] ** 4 / 120.0
    dtheta[small] = -xi[small] / 3.0 + n * xi[small] ** 3 / 30.0
    return xi, theta, dtheta


@source("Bludman2010", where="Table I, ξ_1n column", status="read")
@source("Liu1995", where="Eq. 2 (the first zero of θ is the surface of the star)", status="read")
def lane_emden_first_zero(n: float) -> float:
    """Find the first zero of the Lane-Emden function: θ(ξ_1) = 0.

    Parameters
    ----------
    n : float
        Polytropic index, 0 <= n < 5.

    Returns
    -------
    float
        ξ_1, the dimensionless radius of the star (R = ξ_1 / A).

    References
    ----------
    [Bludman2010] Table I; [Liu1995] Eq. 2.

    Validity
    --------
    ξ_1 is infinite for n >= 5; ``ValueError`` is raised.

    Verification
    ------------
    tests/unit/test_stars.py: Table I values for n = 0, 1, 1.5, 2, 3, 4, 4.5 (four significant digits).
    """
    return _solve(_valid_n(n))[1]


@source("Liu1995", where="Eq. 38 (m = 4π ρ_c r³ (-θ'/ξ))", status="read")
@source(
    "Bludman2010",
    where="Table I (0ω_3 = 2.018) and Section II, M = 4π 0ω_3 (K/πG)^{3/2}",
    status="read",
    note="the 0ω_n column in the table equals -ξ_1² θ'(ξ_1) only for n = 3; for other n it uses a different normalization",
)
def lane_emden_mass_coefficient(n: float) -> float:
    """Compute the dimensionless total mass: ω_n = -ξ_1² θ'(ξ_1).

    M = 4π ρ_c (R/ξ_1)³ ω_n   ([Liu1995] Eq. 38, at r = R)

    Parameters
    ----------
    n : float
        Polytropic index, 0 <= n < 5.

    Returns
    -------
    float
        ω_n; 2.018 for n = 3.

    References
    ----------
    [Liu1995] Eq. 38; [Bludman2010] Table I and Section II.

    Verification
    ------------
    tests/unit/test_stars.py: 2.018 for n = 3 and the identity ξ_1³ / (3 ρ_c/ρ̄).
    """
    _, xi1, dtheta1 = _solve(_valid_n(n))
    return -(xi1**2) * dtheta1


@source("Bludman2010", where="Table I, ρ_cn/ρ̄_n column; definition ρ̄_n = 3M/4πR³", status="read")
@source("Liu1995", where="Eq. 38 (m = M at r = R)", status="read")
def lane_emden_central_to_mean_density(n: float) -> float:
    """Compute the ratio of central density to mean density.

    ρ_c / ρ̄ = ξ_1 / (-3 θ'(ξ_1))

    Parameters
    ----------
    n : float
        Polytropic index, 0 <= n < 5.

    Returns
    -------
    float
        ρ_c/ρ̄ >= 1; 1 at n = 0 and 54.18 at n = 3.

    References
    ----------
    [Bludman2010] Table I; [Liu1995] Eq. 38.

    Verification
    ------------
    tests/unit/test_stars.py: n = 0 (1), n = 1 (π²/3) and the Table I values.
    """
    _, xi1, dtheta1 = _solve(_valid_n(n))
    return xi1 / (-3.0 * dtheta1)


@source("Bludman2010", where="Table II (rows n = 0, 1, 5)", status="read")
def lane_emden_analytic(xi: ArrayLike, n: int) -> Array:
    """Compute the closed-form solutions of the Lane-Emden equation for n = 0, 1, 5.

    n = 0:  θ = 1 - ξ²/6
    n = 1:  θ = sin ξ / ξ
    n = 5:  θ = (1 + ξ²/3)^{-1/2}

    Parameters
    ----------
    xi : float or ndarray
        Dimensionless radius, ξ >= 0.
    n : {0, 1, 5}
        Polytropic index.

    Returns
    -------
    float or ndarray
        θ(ξ). For n = 0 and 1, the region beyond ξ_1 is unphysical (θ < 0).

    References
    ----------
    [Bludman2010] Table II.

    Verification
    ------------
    tests/unit/test_stars.py: agreement with the numerical solution within 1e-9 for n = 0 and n = 1; residual of the equation for n = 5
    (n = 5 is not solved numerically: the radius is infinite).
    """
    x = np.asarray(xi, dtype=float)
    if np.any(x < 0):
        raise ValueError("ξ cannot be negative")
    if n == 0:
        return 1.0 - x**2 / 6.0
    if n == 1:
        return np.sinc(x / np.pi)
    if n == 5:
        return (1.0 + x**2 / 3.0) ** -0.5
    raise ValueError("a closed-form solution exists only for n = 0, 1, 5")


@source("Liu1995", where="Eq. 38 (at r = R), [Bludman2010] Table I for the definition of ρ̄", status="read")
def polytrope_central_density_kg_m3(mass_kg: ArrayLike, radius_m: ArrayLike, n: float) -> Array:
    """Compute the central density of a polytrope with the given mass and radius.

    ρ_c = (ρ_c/ρ̄) · 3M / (4π R³)

    Parameters
    ----------
    mass_kg : float or ndarray
        Total mass [kg].
    radius_m : float or ndarray
        Radius [m].
    n : float
        Polytropic index, 0 <= n < 5.

    Returns
    -------
    float or ndarray
        Central density [kg m^-3].

    References
    ----------
    [Liu1995] Eq. 38.

    Verification
    ------------
    tests/unit/test_stars.py: 76.39 g cm^-3 for the n = 3 solar model (numerical value of [Liu1995]).
    """
    m = np.asarray(mass_kg, dtype=float)
    r = np.asarray(radius_m, dtype=float)
    return lane_emden_central_to_mean_density(n) * 3.0 * m / (4.0 * np.pi * r**3)


@source("Liu1995", where="Eq. 3 (definition of A², A = ξ_1/R)", status="read")
def polytrope_k_coefficient_si(mass_kg: ArrayLike, radius_m: ArrayLike, n: float) -> Array:
    """Compute the polytropic constant from the given mass and radius: P = K ρ^{1+1/n}.

    K = 4π G ρ_c^{(n-1)/n} R² / ((n+1) ξ_1²)

    Parameters
    ----------
    mass_kg, radius_m : float or ndarray
        Mass [kg] and radius [m].
    n : float
        Polytropic index, 0 < n < 5 (K is undefined at n = 0).

    Returns
    -------
    float or ndarray
        K [Pa m^{3(1+1/n)} kg^{-(1+1/n)}], that is, it gives P in Pa for ρ in kg m^-3.

    References
    ----------
    [Liu1995] Eq. 3.

    Validity
    --------
    The constant K is determined by mass and radius together; for n = 3, K depends only on the mass.

    Verification
    ------------
    tests/unit/test_stars.py: 3.84e14 dyn cm^-2 g^{-4/3} for the n = 3 solar model ([Araujo2011]).
    """
    n = _valid_n(n)
    if n == 0.0:
        raise ValueError("K is undefined for n = 0 (equation of state P = K ρ^∞)")
    xi1 = lane_emden_first_zero(n)
    r = np.asarray(radius_m, dtype=float)
    rho_c = polytrope_central_density_kg_m3(mass_kg, radius_m, n)
    return 4.0 * np.pi * GRAVITATIONAL_CONSTANT.value * rho_c ** ((n - 1.0) / n) * r**2 / ((n + 1.0) * xi1**2)


@source(
    "Liu1995",
    where="Eq. 41 (P = K ρ_c^{(n+1)/n} ω^{n+1}, ω = 1 at the center) and Eq. 3",
    status="read",
    note="combining Eq. 41 with Eq. 3 gives P_c = 4π G R² ρ_c² / ((n+1) ξ_1²); this includes n = 0",
)
def polytrope_central_pressure_pa(mass_kg: ArrayLike, radius_m: ArrayLike, n: float) -> Array:
    """Compute the central pressure of a polytrope with the given mass and radius.

    P_c = K ρ_c^{(n+1)/n} = 4π G R² ρ_c² / ((n+1) ξ_1²)

    Parameters
    ----------
    mass_kg, radius_m : float or ndarray
        Mass [kg] and radius [m].
    n : float
        Polytropic index, 0 <= n < 5.

    Returns
    -------
    float or ndarray
        Central pressure [Pa].

    References
    ----------
    [Liu1995] Eq. 41 and Eq. 3.

    Verification
    ------------
    tests/unit/test_stars.py: 1.24e17 dyn cm^-2 for the n = 3 Sun (numerical value quoted by [Liu1995]);
    closed-form solution for a uniform sphere for n = 0, (2π/3) G ρ² R².
    """
    n = _valid_n(n)
    xi1 = lane_emden_first_zero(n)
    r = np.asarray(radius_m, dtype=float)
    rho_c = polytrope_central_density_kg_m3(mass_kg, radius_m, n)
    return 4.0 * np.pi * GRAVITATIONAL_CONSTANT.value * r**2 * rho_c**2 / ((n + 1.0) * xi1**2)


@source("Liu1995", where="Eq. 3 and Eq. 38", status="read")
@source(
    "Bludman2010",
    where="Table I, last column (M ~ R^{(n-3)/(n-1)})",
    status="read",
    note="the exponent was checked one by one against 1 - ω̃_n in the table for n = 0, 1.5, 2, 4",
)
def polytrope_radius_m(mass_kg: ArrayLike, k_si: float, n: float) -> Array:
    """Solve for the radius of the star for the given mass and polytropic constant.

    M = 4π (ω_n/ξ_1³) B^{n/(n-1)} R^{(n-3)/(n-1)},   B = (n+1) K ξ_1² / (4π G)

    Parameters
    ----------
    mass_kg : float or ndarray
        Total mass [kg].
    k_si : float
        Polytropic constant K (same unit as ``polytrope_k_coefficient_si``).
    n : float
        Polytropic index, 0 < n < 5 and n ∉ {1, 3} (for n = 1 and 3 the radius is independent of the mass).

    Returns
    -------
    float or ndarray
        Radius [m]. R ∝ M^{-1/3} for n = 3/2 (non-relativistic degenerate gas).

    References
    ----------
    [Liu1995] Eq. 3 and 38; [Bludman2010] Table I.

    Verification
    ------------
    tests/unit/test_stars.py: round trip from K to radius; for n = 3/2, when the mass increases eightfold the radius halves.
    """
    n = _valid_n(n)
    if n <= 0.0 or n in (1.0, 3.0):
        raise ValueError("the radius can be found this way only for 0 < n < 5, n ∉ {1, 3}")
    if k_si <= 0:
        raise ValueError("K must be positive")
    xi1 = lane_emden_first_zero(n)
    omega = lane_emden_mass_coefficient(n)
    b = (n + 1.0) * k_si * xi1**2 / (4.0 * np.pi * GRAVITATIONAL_CONSTANT.value)
    m = np.asarray(mass_kg, dtype=float)
    base = m * xi1**3 / (4.0 * np.pi * omega * b ** (n / (n - 1.0)))
    return base ** ((n - 1.0) / (n - 3.0))


@source("Liu1995", where="Eq. 42 (T = μ m_p K ρ_c^{1/n} ω / k, ω = 1 at the center)", status="read")
def polytrope_central_temperature_k(
    mass_kg: ArrayLike, radius_m: ArrayLike, n: float, mean_molecular_weight: float
) -> Array:
    """Compute the central temperature of an ideal-gas polytrope.

    T_c = μ m_p P_c / (k ρ_c)

    Parameters
    ----------
    mass_kg, radius_m : float or ndarray
        Mass [kg] and radius [m].
    n : float
        Polytropic index, 0 <= n < 5.
    mean_molecular_weight : float
        μ, mean mass per particle (in units of m_p).

    Returns
    -------
    float or ndarray
        Central temperature [K].

    References
    ----------
    [Liu1995] Eq. 42.

    Validity
    --------
    The ideal-gas equation of state is valid only if radiation pressure and degeneracy can be ignored.

    Verification
    ------------
    tests/unit/test_stars.py: 1.2e7 K for the n = 3, μ = 0.62 Sun ([Liu1995] text).
    """
    pc = polytrope_central_pressure_pa(mass_kg, radius_m, n)
    rho_c = polytrope_central_density_kg_m3(mass_kg, radius_m, n)
    return mean_molecular_weight * PROTON_MASS.value * pc / (BOLTZMANN.value * rho_c)


@source(
    "Liu1995",
    where="Eq. 44 and 45 (E_g; Φ = 0 and P = 0 at the surface)",
    status="read",
    note="in Eq. 44 take r = R, m = M; the first term remains: -3/(5-n) G M²/R",
)
def polytrope_gravitational_energy_j(mass_kg: ArrayLike, radius_m: ArrayLike, n: float) -> Array:
    """Compute the gravitational self-binding energy of a polytrope.

    E_g = -3/(5-n) · G M² / R

    Parameters
    ----------
    mass_kg, radius_m : float or ndarray
        Mass [kg] and radius [m].
    n : float
        Polytropic index, 0 <= n < 5.

    Returns
    -------
    float or ndarray
        E_g [J], negative.

    References
    ----------
    [Liu1995] Eq. 44, 45.

    Verification
    ------------
    tests/unit/test_stars.py: -3/5 for n = 0 and -3/4 GM²/R for n = 1 (closed-form solutions).
    """
    n = _valid_n(n)
    m = np.asarray(mass_kg, dtype=float)
    r = np.asarray(radius_m, dtype=float)
    return -3.0 / (5.0 - n) * GRAVITATIONAL_CONSTANT.value * m**2 / r


@source("McKee2007", where="TeX labels eq:iddot, eq:calt (virial theorem)", status="read")
@source(
    "Liu1995",
    where="Eq. 44 and 45",
    status="read",
    note="Eq. 44 has three terms; with P = 0 at the surface and Eq. 45, E_g = -3/(5-n) G M²/R. For a stationary sphere with no surface term and no magnetic field, 2T + W = 0; T = (3/2)(k/μ m_p) M T̄ with T̄ the mass-weighted mean",
)
def virial_mean_temperature_k(
    mass_kg: ArrayLike, radius_m: ArrayLike, mean_molecular_weight: float, n: float = 3.0
) -> Array:
    """Compute the mass-weighted mean temperature from the virial theorem.

    2T + W = 0,  T = (3/2) (k / μ m_p) M T̄,  W = -3/(5-n) G M²/R
    ⇒  T̄ = G M μ m_p / ((5-n) k R)

    Parameters
    ----------
    mass_kg, radius_m : float or ndarray
        Mass [kg] and radius [m].
    mean_molecular_weight : float
        μ (in units of m_p).
    n : float
        Polytropic index used to compute W; the default is 3 (Eddington standard model).

    Returns
    -------
    float or ndarray
        Mass-weighted mean temperature [K].

    References
    ----------
    [McKee2007] virial theorem (eq:iddot, eq:calt); [Liu1995] Eq. 44.

    Validity
    --------
    Radiation pressure is ignored; an ideal gas and stationary equilibrium are assumed.

    Verification
    ------------
    tests/unit/test_stars.py: identity 2T + W = 0; for the n = 3 Sun it stays below the central temperature.
    """
    n = _valid_n(n)
    m = np.asarray(mass_kg, dtype=float)
    r = np.asarray(radius_m, dtype=float)
    return (
        GRAVITATIONAL_CONSTANT.value * m * mean_molecular_weight * PROTON_MASS.value / ((5.0 - n) * BOLTZMANN.value * r)
    )
