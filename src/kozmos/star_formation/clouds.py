"""Stability, timescales, and star formation rate of molecular clouds.

The equations were taken from the arXiv TeX files of two open sources
(25 November 2026): the ARA&A review by McKee and Ostriker (arXiv:0707.3514,
[McKee2007]) and the open lecture notes by Krumholz (arXiv:1511.03457,
[Krumholz2015]). The TeX source of the review has no equation numbers; the
``where`` field gives the TeX label (``\\label``) instead.

The authors use different numerical factors for the Jeans quantities. In this
module both definitions can be selected with the ``definition`` argument, and
the difference is stated explicitly in the docstring of the relevant function.

Units are SI (kg, m, s, K, Pa). For magnetic quantities the conversion from
Gaussian units to SI uses exact factors (1 T = 10⁴ G, 1 Wb = 10⁸ Mx).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from kozmos._sources import source
from kozmos.core.constants import BOLTZMANN, GRAVITATIONAL_CONSTANT, PROTON_MASS

Array = NDArray[np.float64]

# The proton mass is used in place of the hydrogen atom mass m_H. The difference
# is about 0.15 % (electron mass minus binding energy), far below the uncertainty
# of the μ values quoted by the sources.
_M_H = PROTON_MASS.value

# Mean molecular weight: mixture of H2 and He (1 He per 10 H), per PARTICLE [Krumholz2015].
# This is the default of the density functions: n is the number density of all particles (H2 + He).
MEAN_MOLECULAR_WEIGHT = 2.33
# If n is only the number density of H2 molecules (n_H2, common in observations), the mass per molecule is 2.8 m_H;
# if n is the density of hydrogen nuclei (n_H), the mass per nucleus is 1.4 m_H. Both follow from the same helium
# abundance (2.33 · 1.2 = 2.8; 2.8 / 2 = 1.4); the value 1.4 for n_H appears in the numerical example of
# [McKee2007] eq:tff.
MOLECULAR_WEIGHT_PER_H2 = 2.8
MOLECULAR_WEIGHT_PER_H = 1.4


def _positive(**values: ArrayLike) -> list[Array]:
    """Check that the arguments are strictly positive; convert them to arrays and return them."""
    out = []
    for name, d in values.items():
        a = np.asarray(d, dtype=float)
        if np.any(~(a > 0.0)):
            raise ValueError(f"{name} must be strictly positive")
        out.append(a)
    return out


@source(
    "Krumholz2015",
    where="Section 4 (Gas Flows and Turbulence), the c_s expression in the Reynolds and Mach numbers subsection",
    status="read",
    note="The source takes μ = 2.33 and gives 0.18 km/s for 10 K; μ is an independent variable here",
)
def sound_speed_m_s(temperature_k: ArrayLike, mean_molecular_weight: float = MEAN_MOLECULAR_WEIGHT) -> Array:
    """Compute the isothermal sound speed: c_s = sqrt(k_B T / (μ m_H)).

    Parameters
    ----------
    temperature_k : float or ndarray
        Gas temperature [K], positive.
    mean_molecular_weight : float
        Mean mass per particle in units of m_H; 2.33 for H2 and He.

    Returns
    -------
    float or ndarray
        Sound speed [m/s].

    References
    ----------
    [Krumholz2015] Section 4.

    Validity
    --------
    Isothermal (γ = 1) gas; as the density rises the gas becomes adiabatic and the result no longer applies.

    Verification
    ------------
    tests/unit/test_star_formation.py: 0.18 km/s for 10 K (the value in the source).
    """
    (t,) = _positive(temperature_k=temperature_k)
    (mu,) = _positive(mean_molecular_weight=mean_molecular_weight)
    return np.sqrt(BOLTZMANN.value * t / (mu * _M_H))


@source(
    "Krumholz2015",
    where="Section 4 (Gas Flows and Turbulence), Reynolds and Mach numbers subsection: M ~ V/c_s",
    status="read",
)
def mach_number(velocity_m_s: ArrayLike, sound_speed_m_s: ArrayLike) -> Array:
    """Compute the Mach number: M = V / c_s.

    For M ≫ 1 the pressure term is unimportant in the behavior of the flow; the
    source finds M ~ 20 for a molecular cloud. It gives the relative weight of
    thermal and turbulent support.

    Parameters
    ----------
    velocity_m_s : float or ndarray
        Typical flow speed or turbulent velocity dispersion [m/s], non-negative.
    sound_speed_m_s : float or ndarray
        Sound speed [m/s], positive.

    Returns
    -------
    float or ndarray
        Dimensionless Mach number.

    References
    ----------
    [Krumholz2015] Section 4.

    Verification
    ------------
    tests/unit/test_star_formation.py: M = 1 for a speed equal to the sound speed.
    """
    v = np.asarray(velocity_m_s, dtype=float)
    if np.any(v < 0):
        raise ValueError("velocity cannot be negative")
    (cs,) = _positive(sound_speed=sound_speed_m_s)
    return v / cs


@source(
    "Krumholz2015",
    where="Section 8 (Giant Molecular Clouds), Scaling Relations: n = 3M / (4π R³ μ m_H)",
    status="read",
    note="The source uses μ = 2.3 (per H2 molecule); here μ is an independent variable",
)
def mass_density_kg_m3(number_density_m3: ArrayLike, mean_molecular_weight: float = MEAN_MOLECULAR_WEIGHT) -> Array:
    """Compute the mass density from the number density: ρ = μ m_H n.

    Parameters
    ----------
    number_density_m3 : float or ndarray
        Particle number density [m⁻³], non-negative.
    mean_molecular_weight : float
        Mean mass per particle in units of m_H.

    Returns
    -------
    float or ndarray
        Mass density [kg/m³].

    References
    ----------
    [Krumholz2015] Section 8.

    Verification
    ------------
    tests/unit/test_star_formation.py: round trip with the number density.
    """
    n = np.asarray(number_density_m3, dtype=float)
    if np.any(n < 0):
        raise ValueError("number density cannot be negative")
    (mu,) = _positive(mean_molecular_weight=mean_molecular_weight)
    return n * mu * _M_H


@source(
    "Krumholz2015",
    where="Section 8 (Giant Molecular Clouds), Scaling Relations: n = 3M / (4π R³ μ m_H)",
    status="read",
)
def number_density_m3(mass_density_kg_m3: ArrayLike, mean_molecular_weight: float = MEAN_MOLECULAR_WEIGHT) -> Array:
    """Compute the number density from the mass density: n = ρ / (μ m_H).

    Parameters
    ----------
    mass_density_kg_m3 : float or ndarray
        Mass density [kg/m³], non-negative.
    mean_molecular_weight : float
        Mean mass per particle in units of m_H.

    Returns
    -------
    float or ndarray
        Number density [m⁻³].

    References
    ----------
    [Krumholz2015] Section 8.

    Verification
    ------------
    tests/unit/test_star_formation.py: round trip with ``mass_density_kg_m3``.
    """
    rho = np.asarray(mass_density_kg_m3, dtype=float)
    if np.any(rho < 0):
        raise ValueError("density cannot be negative")
    (mu,) = _positive(mean_molecular_weight=mean_molecular_weight)
    return rho / (mu * _M_H)


@source(
    "Krumholz2015",
    where="Section 8 (Giant Molecular Clouds), Mass Measurement, Optically Thick Lines: N = M / (μ m_H π R²)",
    status="read",
    note="The source writes the relation between N and Σ for the mean column density; Σ = μ m_H N is a rearrangement",
)
def surface_density_kg_m2(column_density_m2: ArrayLike, mean_molecular_weight: float = MEAN_MOLECULAR_WEIGHT) -> Array:
    """Compute the surface density from the column density: Σ = μ m_H N.

    Parameters
    ----------
    column_density_m2 : float or ndarray
        Particle column density [m⁻²], non-negative.
    mean_molecular_weight : float
        Mean mass per particle in units of m_H.

    Returns
    -------
    float or ndarray
        Surface density [kg/m²].

    References
    ----------
    [Krumholz2015] Section 8.

    Verification
    ------------
    tests/unit/test_star_formation.py: round trip with ``column_density_m2``.
    """
    n = np.asarray(column_density_m2, dtype=float)
    if np.any(n < 0):
        raise ValueError("column density cannot be negative")
    (mu,) = _positive(mean_molecular_weight=mean_molecular_weight)
    return n * mu * _M_H


@source(
    "Krumholz2015",
    where="Section 8 (Giant Molecular Clouds), Mass Measurement, Optically Thick Lines: N = M / (μ m_H π R²)",
    status="read",
)
def column_density_m2(surface_density_kg_m2: ArrayLike, mean_molecular_weight: float = MEAN_MOLECULAR_WEIGHT) -> Array:
    """Compute the column density from the surface density: N = Σ / (μ m_H).

    Parameters
    ----------
    surface_density_kg_m2 : float or ndarray
        Surface density [kg/m²], non-negative.
    mean_molecular_weight : float
        Mean mass per particle in units of m_H.

    Returns
    -------
    float or ndarray
        Particle column density [m⁻²].

    References
    ----------
    [Krumholz2015] Section 8.

    Verification
    ------------
    tests/unit/test_star_formation.py: round trip with ``surface_density_kg_m2``.
    """
    s = np.asarray(surface_density_kg_m2, dtype=float)
    if np.any(s < 0):
        raise ValueError("surface density cannot be negative")
    (mu,) = _positive(mean_molecular_weight=mean_molecular_weight)
    return s / (mu * _M_H)


@source("McKee2007", where="TeX label LJeans", status="read")
@source("Krumholz2015", where="Section 6 (Gravitational Instability and Collapse), λ_J = 2π/k_J", status="read")
def jeans_length_m(sound_speed_m_s: ArrayLike, density_kg_m3: ArrayLike) -> Array:
    """Compute the Jeans length: λ_J = sqrt(π c_s² / (G ρ)).

    Parameters
    ----------
    sound_speed_m_s : float or ndarray
        Isothermal sound speed [m/s], positive.
    density_kg_m3 : float or ndarray
        Density of the uniform medium [kg/m³], positive.

    Returns
    -------
    float or ndarray
        Jeans length [m]: the smallest wavelength of gravitational fragmentation.

    References
    ----------
    [McKee2007] LJeans; [Krumholz2015] Section 6 (derived from linear analysis).
    The two sources give the same expression.

    Validity
    --------
    The default μ = 2.33 is per particle: the number density n must be the density of all particles (H2 + He).
    If only the H2 density (n_H2) is given, use ``MOLECULAR_WEIGHT_PER_H2`` (2.8); if the density of
    hydrogen nuclei (n_H) is given, use ``MOLECULAR_WEIGHT_PER_H`` (1.4). If n_H2 is passed with the default,
    the density comes out about 17 % too low and the free-fall time 9 % too long.
    Uniform, infinite, isothermal medium (the "Jeans swindle"); pressure support only.

    Verification
    ------------
    tests/unit/test_star_formation.py: 3.4 pc for c_s = 0.2 km/s, ρ = 100 m_p.
    """
    cs, rho = _positive(sound_speed=sound_speed_m_s, density=density_kg_m3)
    return np.sqrt(np.pi * cs**2 / (GRAVITATIONAL_CONSTANT.value * rho))


@source(
    "McKee2007",
    where="TeX label LJeans, the paragraph that follows (M_J = 2.47 M_BE)",
    status="read",
    note="definition='sphere': M_J = (4π/3)(λ_J/2)³ ρ (McKee-Ostriker, definition of Binney-Tremaine 1987)",
)
@source(
    "Krumholz2015",
    where="Section 6, definition of the Jeans mass and its footnote",
    status="read",
    note="definition='cube': M_J = ρ λ_J³ / 8 (cube of side λ_J/2); the ratio between the two is 4π/3",
)
def jeans_mass_kg(sound_speed_m_s: ArrayLike, density_kg_m3: ArrayLike, definition: str = "sphere") -> Array:
    """Compute the Jeans mass; the numerical factor depends on the author.

    ``definition='sphere'``: M_J = (4π/3)(λ_J/2)³ ρ = (π^{5/2}/6) c_s³ / sqrt(G³ ρ)
    ([McKee2007]; 2.47 times the Bonnor-Ebert mass).
    ``definition='cube'``: M_J = ρ λ_J³ / 8 = (π^{3/2}/8) c_s³ / sqrt(G³ ρ) ([Krumholz2015]).
    The sources state explicitly that this factor is uncertain; the scaling relation
    c_s³ / sqrt(G³ ρ) is the same in both definitions.

    Parameters
    ----------
    sound_speed_m_s : float or ndarray
        Isothermal sound speed [m/s], positive.
    density_kg_m3 : float or ndarray
        Density [kg/m³], positive.
    definition : {"sphere", "cube"}
        Definition of the factor.

    Returns
    -------
    float or ndarray
        Jeans mass [kg].

    References
    ----------
    [McKee2007] paragraph after LJeans; [Krumholz2015] Section 6.

    Validity
    --------
    Same as ``jeans_length_m``.

    Verification
    ------------
    tests/unit/test_star_formation.py: ratio of the two definitions is 4π/3; M_J = 2.47 M_BE.
    """
    cs, rho = _positive(sound_speed=sound_speed_m_s, density=density_kg_m3)
    lam = np.sqrt(np.pi * cs**2 / (GRAVITATIONAL_CONSTANT.value * rho))
    if definition == "sphere":
        return (4.0 * np.pi / 3.0) * (lam / 2.0) ** 3 * rho
    if definition == "cube":
        return rho * lam**3 / 8.0
    raise ValueError("definition must be 'sphere' or 'cube'")


@source(
    "McKee2007",
    where="TeX label MBE_def",
    status="read",
    note="Coefficient 1.182 (the exercise solution in Krumholz2015 gives 1.18)",
)
@source("Krumholz2015", where="problems/probsol2.tex, part (h) of the Bonnor-Ebert sphere exercise", status="read")
def bonnor_ebert_critical_mass_kg(sound_speed_m_s: ArrayLike, surface_pressure_pa: ArrayLike) -> Array:
    """Compute the Bonnor-Ebert mass: M_BE = 1.182 c_s⁴ / sqrt(G³ P_0).

    This is the largest mass of an isothermal sphere without a magnetic field
    that can be in hydrostatic equilibrium at the given surface pressure
    (stable or unstable).

    Parameters
    ----------
    sound_speed_m_s : float or ndarray
        Isothermal sound speed [m/s], positive.
    surface_pressure_pa : float or ndarray
        External pressure at the surface P_0 = ρ_0 c_s² [Pa], positive.

    Returns
    -------
    float or ndarray
        Critical mass [kg].

    References
    ----------
    [McKee2007] MBE_def; [Krumholz2015] probsol2 (field: m_max = 1.18).

    Validity
    --------
    Isothermal, non-rotating sphere without a magnetic field and without turbulence.

    Verification
    ------------
    tests/unit/test_star_formation.py: 0.66 and 0.67 solar masses from the sources
    for T = 10 K, P/k_B = 3e5 cm⁻³ K; agreement with the coefficient derived from the ODE.
    """
    cs, p = _positive(sound_speed=sound_speed_m_s, surface_pressure=surface_pressure_pa)
    return 1.182 * cs**4 / np.sqrt(GRAVITATIONAL_CONSTANT.value**3 * p)


@source(
    "McKee2007",
    where="TeX label MBE_def, the next paragraph: R_BE = 0.486 c_s / sqrt(G ρ_0)",
    status="read",
)
def bonnor_ebert_radius_m(sound_speed_m_s: ArrayLike, surface_value_density_kg_m3: ArrayLike) -> Array:
    """Compute the radius of the critical Bonnor-Ebert sphere: R_BE = 0.486 c_s / sqrt(G ρ_0).

    Parameters
    ----------
    sound_speed_m_s : float or ndarray
        Isothermal sound speed [m/s], positive.
    surface_value_density_kg_m3 : float or ndarray
        Density ρ_0 at the surface of the sphere [kg/m³], positive.

    Returns
    -------
    float or ndarray
        Radius [m].

    References
    ----------
    [McKee2007] paragraph after MBE_def.

    Validity
    --------
    Same as ``bonnor_ebert_critical_mass_kg``.

    Verification
    ------------
    tests/unit/test_star_formation.py: agreement with ξ and the density contrast
    derived from the ODE; R_BE ≈ 0.486/√π times the Jeans length.
    """
    cs, rho = _positive(sound_speed=sound_speed_m_s, surface_density=surface_value_density_kg_m3)
    return 0.486 * cs / np.sqrt(GRAVITATIONAL_CONSTANT.value * rho)


def _be_ode(xi: float, y: list[float]) -> list[float]:
    """Dimensionless isothermal sphere equation: ψ'' + 2ψ'/ξ = exp(−ψ)."""
    psi, dpsi = y
    return [dpsi, -2.0 * dpsi / xi + np.exp(-psi)]


@lru_cache(maxsize=1)
def _be_coefficients() -> tuple[float, float, float]:
    """Solve the ODE and compute the triple (m_max, ρ_c/ρ_s, ξ_s).

    m(ξ) = (4π)^{-1/2} ξ² ψ' exp(−ψ/2); the mass integral in the source
    ∫ ξ² e^{−ψ} dξ = ξ² ψ' is obtained from the identity and P_s = ρ_s c_s².
    """
    xi0 = 1e-4
    solution = solve_ivp(_be_ode, (xi0, 30.0), [xi0**2 / 6.0, xi0 / 3.0], rtol=1e-11, atol=1e-13, dense_output=True)

    def neg_mass(xi: float) -> float:
        psi, dpsi = solution.sol(xi)
        return float(-(xi**2) * dpsi * np.exp(-psi / 2.0) / np.sqrt(4.0 * np.pi))

    grid = np.linspace(2.0, 15.0, 2000)
    best = grid[int(np.argmin([neg_mass(x) for x in grid]))]
    r = minimize_scalar(neg_mass, bounds=(best - 0.05, best + 0.05), method="bounded", options={"xatol": 1e-10})
    xi_s = float(r.x)
    psi_s = float(solution.sol(xi_s)[0])
    return float(-r.fun), float(np.exp(psi_s)), xi_s


@source(
    "Krumholz2015",
    where="problems/probset2.tex part (e) (mass expression); probsol2.tex parts (c)-(d) (ODE), part (f) (m_max = 1.18, ρ_c/ρ_s = 14.0)",
    status="read",
    note="The ODE and the expression m(ξ) = (4π)^{-1/2} ξ² ψ' e^{-ψ/2} are from the source; only the value ξ_s ≈ 6.45 is not in the source and comes from the numerical solution",
)
def bonnor_ebert_critical_coefficients() -> tuple[float, float, float]:
    """Compute the critical Bonnor-Ebert coefficients from the isothermal sphere equation.

    The equation ψ'' + 2ψ'/ξ = e^{−ψ} (ψ(0) = ψ'(0) = 0) is solved numerically;
    the sphere is critical when the dimensionless mass m = M sqrt(P_s) G^{3/2} / c_s⁴
    reaches its maximum.

    Returns
    -------
    tuple of float
        (m_max, ρ_c/ρ_s, ξ_s): the largest dimensionless mass (≈ 1.18), the critical
        central-to-surface density contrast (≈ 14.0), and the dimensionless surface
        radius ξ_s = R/r_0 (≈ 6.45).

    References
    ----------
    [Krumholz2015] probsol2.

    Validity
    --------
    Spheres with ξ_s > ξ_critical are unstable; ρ_c/ρ_s > 14 means unstable.

    Verification
    ------------
    tests/unit/test_star_formation.py: with 1.18 and 14.0 in the source; consistency
    with 1.182 and R_BE = 0.486 c_s/sqrt(G ρ_0) in McKee2007.
    """
    return _be_coefficients()


@source("McKee2007", where="TeX label alpha_vir_def", status="read")
@source(
    "Krumholz2015",
    where="Section 8 (Giant Molecular Clouds), Larson's third law: α_vir = 5σ²R/(GM)",
    status="read",
)
def virial_parameter(velocity_dispersion_m_s: ArrayLike, radius_m: ArrayLike, mass_kg: ArrayLike) -> Array:
    """Compute the virial parameter: α_vir = 5 σ² R / (G M).

    The coefficient is chosen so that α_vir = 1 for a uniform sphere in virial
    equilibrium without a magnetic field (this sphere is not in hydrostatic
    equilibrium). α_vir ≲ 1 corresponds to bound clouds and α_vir ≫ 1 to clouds
    held by external pressure.

    Parameters
    ----------
    velocity_dispersion_m_s : float or ndarray
        One-dimensional total (thermal + turbulent) velocity dispersion σ [m/s], positive.
    radius_m : float or ndarray
        Cloud radius [m], positive.
    mass_kg : float or ndarray
        Cloud mass [kg], positive.

    Returns
    -------
    float or ndarray
        Dimensionless virial parameter.

    References
    ----------
    [McKee2007] alpha_vir_def; [Krumholz2015] Section 8.

    Validity
    --------
    The magnetic term and the surface pressure are not taken into account.

    Verification
    ------------
    tests/unit/test_star_formation.py: α_vir = 1 for σ² = GM/(5R); consistency
    with the surface density form.
    """
    sigma, r, m = _positive(velocity_dispersion=velocity_dispersion_m_s, radius=radius_m, mass=mass_kg)
    return 5.0 * sigma**2 * r / (GRAVITATIONAL_CONSTANT.value * m)


@source(
    "McKee2007",
    where="TeX label alpha_vir_def",
    status="read",
    note="Written by substituting M = π R² Σ; identical to the (5/π pc) σ_pc²/(GΣ) form in Section 8 of Krumholz2015",
)
def virial_parameter_from_surface_density(
    velocity_dispersion_m_s: ArrayLike, radius_m: ArrayLike, surface_density_kg_m2: ArrayLike
) -> Array:
    """Compute the virial parameter from the surface density: α_vir = 5 σ² / (π G Σ R).

    Parameters
    ----------
    velocity_dispersion_m_s : float or ndarray
        One-dimensional velocity dispersion [m/s], positive.
    radius_m : float or ndarray
        Cloud radius [m], positive.
    surface_density_kg_m2 : float or ndarray
        Mean surface density Σ = M/(π R²) [kg/m²], positive.

    Returns
    -------
    float or ndarray
        Dimensionless virial parameter.

    References
    ----------
    [McKee2007] alpha_vir_def.

    Validity
    --------
    Same as ``virial_parameter``; the surface density is taken over the projected disk.

    Verification
    ------------
    tests/unit/test_star_formation.py: equality with ``virial_parameter`` for M = π R² Σ.
    """
    sigma, r, s = _positive(
        velocity_dispersion=velocity_dispersion_m_s, radius=radius_m, surface_density=surface_density_kg_m2
    )
    return 5.0 * sigma**2 / (np.pi * GRAVITATIONAL_CONSTANT.value * s * r)


@source(
    "McKee2007",
    where="TeX label eq:tff",
    status="read",
    note="The numerical value is given by taking the He fraction as 10 % for n_H",
)
@source(
    "Krumholz2015",
    where="Section 6, definition of t_ff after the Jeans analysis; Section 8 timescales",
    status="read",
)
def free_fall_time_s(density_kg_m3: ArrayLike) -> Array:
    """Compute the free-fall time: t_ff = sqrt(3π / (32 G ρ)).

    This is the time for a pressureless, uniform sphere to collapse to a point
    under its own gravity.

    Parameters
    ----------
    density_kg_m3 : float or ndarray
        Mean mass density [kg/m³], positive.

    Returns
    -------
    float or ndarray
        Free-fall time [s].

    References
    ----------
    [McKee2007] eq:tff; [Krumholz2015] Sections 6 and 8.

    Validity
    --------
    Pressure, turbulence, and magnetic support are ignored; real clouds collapse more slowly.

    Verification
    ------------
    tests/unit/test_star_formation.py: 1.37 million years for n_H = 10³ cm⁻³.
    """
    (rho,) = _positive(density=density_kg_m3)
    return np.sqrt(3.0 * np.pi / (32.0 * GRAVITATIONAL_CONSTANT.value * rho))


@source(
    "Krumholz2015",
    where="Section 8 (Giant Molecular Clouds), Molecular Cloud Timescales: t_cr = R/σ",
    status="read",
)
def crossing_time_s(radius_m: ArrayLike, velocity_dispersion_m_s: ArrayLike) -> Array:
    """Compute the crossing time: t_cr = R / σ.

    If the sound speed is given as σ, the result is the sound crossing time; if
    the turbulent velocity dispersion is given, it is the turbulent crossing time.

    Parameters
    ----------
    radius_m : float or ndarray
        Cloud radius [m], positive.
    velocity_dispersion_m_s : float or ndarray
        Velocity dispersion or sound speed [m/s], positive.

    Returns
    -------
    float or ndarray
        Crossing time [s].

    References
    ----------
    [Krumholz2015] Section 8.

    Verification
    ------------
    tests/unit/test_star_formation.py: t_cr ≈ 2 t_ff in a virial cloud (source: for α_vir = 1
    the free-fall time is half the crossing time).
    """
    r, sigma = _positive(radius=radius_m, velocity_dispersion=velocity_dispersion_m_s)
    return r / sigma


@source(
    "McKee2007",
    where="TeX label eq:mphi",
    status="read",
    note="c_Φ = 0.17 is the value of Tomisaka (1988) (quoted in the source; the original was not read)",
)
@source(
    "Krumholz2015",
    where="Section 6, Magnetic Support and the Magnetic Critical Mass",
    status="read",
    note="The source derives sqrt(5/2)/(3π) = 0.17 for a uniform sphere and gives 0.12 for Tomisaka (1998); McKee2007 cites a different paper for 0.17 (Tomisaka, Ikeuchi, and Nakamura 1988); neither original paper was read",
)
def magnetic_critical_mass_kg(magnetic_flux_wb: ArrayLike, c_phi: float = 0.17) -> Array:
    """Compute the magnetic critical mass: M_Φ = c_Φ Φ / G^{1/2} (in Gaussian units).

    If M > M_Φ the cloud is magnetically supercritical and the field cannot
    halt the collapse; if M < M_Φ it cannot collapse. Φ = π R² B is the flux
    for a uniform field. Conversion to SI: Φ[Mx] = 10⁸ Φ[Wb], G_cgs = 10³ G_SI, kg = 10⁻³ g.

    Parameters
    ----------
    magnetic_flux_wb : float or ndarray
        Magnetic flux through the cloud [Wb], positive.
    c_phi : float
        Coefficient that depends on the internal density and field distribution; 0.17 (polar field,
        constant mass-to-flux ratio), sqrt(5/2)/(3π) ≈ 0.168 for a uniform sphere.

    Returns
    -------
    float or ndarray
        Critical mass [kg].

    References
    ----------
    [McKee2007] eq:mphi; [Krumholz2015] Section 6.

    Validity
    --------
    Cold cloud, ideal MHD (flux frozen in); for a dipole or random field the
    total flux does not determine the mass.

    Verification
    ------------
    tests/unit/test_star_formation.py: consistency with the sheet equivalent (Σ/B)_cr = 1/(2π√G);
    linear scaling with Φ.
    """
    (phi,) = _positive(magnetic_flux=magnetic_flux_wb)
    if not c_phi > 0:
        raise ValueError("c_phi must be positive")
    g_cgs = GRAVITATIONAL_CONSTANT.value * 1e3
    return c_phi * (phi * 1e8) / np.sqrt(g_cgs) * 1e-3


@source(
    "McKee2007",
    where="After TeX label eq:mphi: G^{1/2}(Σ/B)_cr = 1/(2π) (infinite cold sheet)",
    status="read",
    note="The source quotes the value for Nakano-Nakamura (1978); the original was not read. The normalized ratio was written as 2π√G Σ/B",
)
def normalized_mass_to_flux_ratio(surface_density_kg_m2: ArrayLike, magnetic_field_t: ArrayLike) -> Array:
    """Compute the mass-to-flux ratio relative to its critical value: μ/μ_cr = 2π √G Σ / B.

    The critical value is G^{1/2}(Σ/B)_cr = 1/(2π) (in Gaussian units). If the
    ratio is greater than 1, the cloud is magnetically supercritical.

    Parameters
    ----------
    surface_density_kg_m2 : float or ndarray
        Surface density [kg/m²], positive.
    magnetic_field_t : float or ndarray
        Field component perpendicular to the surface [T], positive. (1 T = 10⁴ G; Σ[g/cm²] = 0.1 Σ[kg/m²].)

    Returns
    -------
    float or ndarray
        Dimensionless normalized mass-to-flux ratio.

    References
    ----------
    [McKee2007] paragraph after eq:mphi.

    Validity
    --------
    Infinite cold sheet; for spherical clouds it is approximately equivalent, since c_Φ ≈ 0.17 ≈ 1/(2π).

    Verification
    ------------
    tests/unit/test_star_formation.py: ratio 1 for Σ = B/(2π√G); consistency with the spherical M_Φ.
    """
    s, b = _positive(surface_density=surface_density_kg_m2, magnetic_field=magnetic_field_t)
    g_cgs = GRAVITATIONAL_CONSTANT.value * 1e3
    return 2.0 * np.pi * np.sqrt(g_cgs) * (s * 0.1) / (b * 1e4)


@source(
    "McKee2007",
    where="TeX label GMC_lws (PCA fit of Heyer and Brunt 2004)",
    status="read",
    note="The defaults 0.9 km/s and 0.56 are the PCA fit quoted in McKee2007; L_pca is about a quarter of the cloud diameter",
)
@source(
    "Krumholz2015",
    where="Section 8 (Giant Molecular Clouds), Scaling Relations: Larson's second law",
    status="read",
    note="Alternatives quoted in the source: 0.72 and 0.5 (Solomon 1987, Milky Way), 0.44 and 0.60 (Bolatto 2008); the originals were not read",
)
def linewidth_size_km_s(size_pc: ArrayLike, coefficient_km_s: float = 0.9, exponent: float = 0.56) -> Array:
    """Compute the Larson linewidth-size relation: δv = a (L / 1 pc)^q.

    Parameters
    ----------
    size_pc : float or ndarray
        Scale or cloud radius [pc], positive.
    coefficient_km_s : float
        Velocity dispersion a at 1 pc [km/s].
    exponent : float
        Power-law exponent q (≈ 0.5 for turbulence; Kolmogorov 1/3).

    Returns
    -------
    float or ndarray
        Velocity dispersion [km/s].

    References
    ----------
    [McKee2007] GMC_lws; [Krumholz2015] Section 8.

    Validity
    --------
    Molecular clouds in the Milky Way and the scales of 0.03-30 pc within them; the
    definition of size (PCA scale or cloud radius) changes the normalization.

    Verification
    ------------
    tests/unit/test_star_formation.py: a at 1 pc; scaling with the exponent; the alternatives in the source.
    """
    (size,) = _positive(size=size_pc)
    return coefficient_km_s * size**exponent


@source(
    "McKee2007",
    where="TeX label eq:mdssis (Shu expansion wave solution)",
    status="read",
    note="The source writes the general form as Φ c_s³/G (TeX eq:mds); Φ = 0.975 for the Shu solution",
)
def isothermal_collapse_accretion_rate_kg_s(sound_speed_m_s: ArrayLike, coefficient: float = 0.975) -> Array:
    """Compute the mass accretion rate onto the protostar in isothermal core collapse: Ṁ = 0.975 c_s³ / G.

    In the Shu expansion wave solution the rate is constant in time; the mass
    inside the wave is 2Ṁt and about half of it is in the protostar.

    Parameters
    ----------
    sound_speed_m_s : float or ndarray
        Isothermal sound speed [m/s], positive.
    coefficient : float
        Dimensionless factor Φ; 0.975 for Shu, the Larson-Penston-Hunter solution is larger.

    Returns
    -------
    float or ndarray
        Mass accretion rate [kg/s].

    References
    ----------
    [McKee2007] eq:mdssis.

    Validity
    --------
    Non-rotating, isothermal core without a magnetic field; protostellar outflows reduce
    the rate by a factor ε_core < 1.

    Verification
    ------------
    tests/unit/test_star_formation.py: 1.54e-6 solar masses per year for 10 K.
    """
    (cs,) = _positive(sound_speed=sound_speed_m_s)
    if not coefficient > 0:
        raise ValueError("coefficient must be positive")
    return coefficient * cs**3 / GRAVITATIONAL_CONSTANT.value


@source("McKee2007", where="TeX label eq:epsff", status="read")
@source(
    "Krumholz2015",
    where="Section 8 (Giant Molecular Clouds), Depletion Time: ε_ff = t_ff / t_dep",
    status="read",
)
def free_fall_efficiency(free_fall_time_s: ArrayLike, depletion_time_s: ArrayLike) -> Array:
    """Compute the star formation efficiency per free-fall time: ε_ff = t_ff / t_dep.

    This is the fraction of the mass of a cloud that is converted into stars in one free-fall time.

    Parameters
    ----------
    free_fall_time_s : float or ndarray
        Free-fall time [s], positive.
    depletion_time_s : float or ndarray
        Gas depletion time M/Ṁ_* [s], positive.

    Returns
    -------
    float or ndarray
        Dimensionless efficiency.

    References
    ----------
    [McKee2007] eq:epsff; [Krumholz2015] Section 8. The sources state that the
    typical observed value is ε_ff ~ 0.01.

    Verification
    ------------
    tests/unit/test_star_formation.py: round trip with ``cloud_star_formation_rate_kg_s``.
    """
    tff, tdep = _positive(free_fall_time=free_fall_time_s, depletion_time=depletion_time_s)
    return tff / tdep


@source(
    "McKee2007",
    where="TeX label eq:epsff",
    status="read",
    note="Ṁ_* = ε_ff M / t_ff; typical value for ε_ff in the source is ~0.01 (Krumholz2015 Sections 8-10)",
)
def cloud_star_formation_rate_kg_s(
    cloud_mass_kg: ArrayLike, mean_density_kg_m3: ArrayLike, ff_efficiency: float = 0.01
) -> Array:
    """Compute the star formation rate of a cloud: Ṁ_* = ε_ff M / t_ff.

    Parameters
    ----------
    cloud_mass_kg : float or ndarray
        Gas mass [kg], positive.
    mean_density_kg_m3 : float or ndarray
        Mean density of the cloud [kg/m³]; t_ff is computed with this density.
    ff_efficiency : float
        Efficiency per free-fall time ε_ff, in the range [0, 1];
        typical observed value ≈ 0.01.

    Returns
    -------
    float or ndarray
        Star formation rate [kg/s].

    References
    ----------
    [McKee2007] eq:epsff.

    Validity
    --------
    ε_ff is a measured or assumed constant; this function does not predict it.

    Verification
    ------------
    tests/unit/test_star_formation.py: M/Ṁ_* = t_ff/ε_ff; linear scaling with ε_ff.
    """
    m, rho = _positive(mass=cloud_mass_kg, density=mean_density_kg_m3)
    if not 0.0 <= ff_efficiency <= 1.0:
        raise ValueError("ε_ff must be in the range [0, 1]")
    return ff_efficiency * m / free_fall_time_s(rho)
