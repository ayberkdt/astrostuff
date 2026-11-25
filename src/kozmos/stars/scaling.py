"""Stellar scales: radiation, gravity, timescales, the Eddington limit and the Chandrasekhar mass.

The texts of the sources were read from the arXiv TeX files on 25 November 2026:

- Stefan-Boltzmann and surface gravity forms: IAU 2015 B3 paper [Prsa2016].
- Free-fall time: [McKee2007]; Kelvin-Helmholtz time: [Zinnecker2007].
- Eddington luminosity and critical angular velocity: [Paxton2013].
- Eddington standard model, M_*, and the white dwarf K constant with the Chandrasekhar mass: [Bludman2010].

The nuclear timescale follows the Pols lecture notes [Pols2011, Eq. 2.37].
All functions use SI units; times given in years are Julian years.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from kozmos._sources import source
from kozmos.core.constants import (
    ATOMIC_MASS_UNIT,
    GRAVITATIONAL_CONSTANT,
    JULIAN_YEAR_SECONDS,
    PLANCK,
    SPEED_OF_LIGHT,
    STEFAN_BOLTZMANN,
)
from kozmos.stars.structure import lane_emden_mass_coefficient

Array = NDArray[np.float64]


@source(
    "Prsa2016",
    where="Table 'formu', M_Bol row (4πσR² term); text: T_eff from the Stefan-Boltzmann law",
    status="read",
)
def luminosity_w(radius_m: ArrayLike, effective_temperature_k: ArrayLike) -> Array:
    """Compute the luminosity of a spherical blackbody star: L = 4π R² σ T_eff⁴.

    Parameters
    ----------
    radius_m : float or ndarray
        Radius [m].
    effective_temperature_k : float or ndarray
        Effective temperature [K].

    Returns
    -------
    float or ndarray
        Luminosity [W].

    References
    ----------
    [Prsa2016] Table 'formu' (4πσR² in the M_Bol row) and the derivation of T_eff from the Stefan-Boltzmann law.

    Verification
    ------------
    tests/unit/test_stars.py: nominal solar values agree with 3.828e26 W within 0.1 percent
    ([Prsa2016]: the nominal values are mutually consistent with the current σ).
    """
    r = np.asarray(radius_m, dtype=float)
    t = np.asarray(effective_temperature_k, dtype=float)
    return 4.0 * np.pi * r**2 * STEFAN_BOLTZMANN.value * t**4


@source("Prsa2016", where="text: T_eff from the Stefan-Boltzmann law; Table 'formu', M_Bol row", status="read")
def effective_temperature_k(luminosity_w: ArrayLike, radius_m: ArrayLike) -> Array:
    """Compute the effective temperature from luminosity and radius: T_eff = (L / 4π R² σ)^{1/4}.

    Parameters
    ----------
    luminosity_w : float or ndarray
        Luminosity [W].
    radius_m : float or ndarray
        Radius [m].

    Returns
    -------
    float or ndarray
        Effective temperature [K].

    References
    ----------
    [Prsa2016] T_eff from the Stefan-Boltzmann law (text) and Table 'formu'.

    Verification
    ------------
    tests/unit/test_stars.py: nominal Sun gives 5772 K within 0.1 percent; round trip with ``luminosity_w``.
    """
    lum = np.asarray(luminosity_w, dtype=float)
    r = np.asarray(radius_m, dtype=float)
    return (lum / (4.0 * np.pi * r**2 * STEFAN_BOLTZMANN.value)) ** 0.25


@source(
    "Prsa2016",
    where="Table 'formu', 'log of surface gravity' row: log g = log(10^6 GM) - 2 log(100 R) [cgs]",
    status="read",
)
def surface_gravity_m_s2(mass_kg: ArrayLike, radius_m: ArrayLike) -> Array:
    """Compute the surface gravitational acceleration: g = G M / R².

    Parameters
    ----------
    mass_kg, radius_m : float or ndarray
        Mass [kg] and radius [m].

    Returns
    -------
    float or ndarray
        Surface gravity [m s^-2].

    References
    ----------
    [Prsa2016] Table 'formu' (log g row, the cgs logarithm of g = GM/R²).

    Verification
    ------------
    tests/unit/test_stars.py: nominal Sun gives log g = 4.438068 ([Prsa2016]).
    """
    m = np.asarray(mass_kg, dtype=float)
    r = np.asarray(radius_m, dtype=float)
    return GRAVITATIONAL_CONSTANT.value * m / r**2


@source(
    "Prsa2016",
    where="Table 'formu', 'log of surface gravity' row: 4.438068... + log M - 2 log R",
    status="read",
)
def log_surface_gravity_cgs(mass_kg: ArrayLike, radius_m: ArrayLike) -> Array:
    """Compute the cgs logarithm of the surface gravity: log₁₀ g [cm s^-2].

    Parameters
    ----------
    mass_kg, radius_m : float or ndarray
        Mass [kg] and radius [m].

    Returns
    -------
    float or ndarray
        log g (cgs); 4.438 for the Sun.

    References
    ----------
    [Prsa2016] Table 'formu', log g row.

    Verification
    ------------
    tests/unit/test_stars.py: nominal Sun gives 4.438068 (six decimals, [Prsa2016]).
    """
    return np.log10(100.0 * surface_gravity_m_s2(mass_kg, radius_m))


@source("Bludman2010", where="Section I, text before Table I: ρ̄_n = 3M/4πR³", status="read")
def mean_density_kg_m3(mass_kg: ArrayLike, radius_m: ArrayLike) -> Array:
    """Compute the mean density of a spherical body: ρ̄ = 3M / 4π R³.

    Parameters
    ----------
    mass_kg, radius_m : float or ndarray
        Mass [kg] and radius [m].

    Returns
    -------
    float or ndarray
        Mean density [kg m^-3].

    References
    ----------
    [Bludman2010] definition of ρ̄_n.

    Verification
    ------------
    tests/unit/test_stars.py: nominal Sun gives about 1.41 g cm^-3.
    """
    m = np.asarray(mass_kg, dtype=float)
    r = np.asarray(radius_m, dtype=float)
    return 3.0 * m / (4.0 * np.pi * r**3)


@source("McKee2007", where="TeX etiketi eq:tff", status="read")
def free_fall_time_s(density_kg_m3: ArrayLike) -> Array:
    """Compute the free-fall (dynamical) time of a pressureless homogeneous sphere.

    t_ff = ( 3π / (32 G ρ̄) )^{1/2}

    Parameters
    ----------
    density_kg_m3 : float or ndarray
        Mean density [kg m^-3], positive.

    Returns
    -------
    float or ndarray
        Free-fall time [s].

    References
    ----------
    [McKee2007] eq:tff.

    Validity
    --------
    Used for stars as the dynamical timescale with ρ̄; the pressure and density
    distribution are ignored.

    Verification
    ------------
    tests/unit/test_stars.py: 1.37e6 yr for n_H = 10³ cm^-3 (the number in the [McKee2007] text).
    """
    rho = np.asarray(density_kg_m3, dtype=float)
    if np.any(rho <= 0):
        raise ValueError("density must be positive")
    return np.sqrt(3.0 * np.pi / (32.0 * GRAVITATIONAL_CONSTANT.value * rho))


@source(
    "Zinnecker2007",
    where="Section 'The Accretion Phase / Formation of the Hydrostatic Core' text and GLOSSARY: t_KH = G M²/(R L)",
    status="read",
    note="the paper writes τ_KH ~ G M²/(R L) as an order of magnitude; the prefactor is taken as 1",
)
def kelvin_helmholtz_timescale_s(mass_kg: ArrayLike, radius_m: ArrayLike, luminosity_w: ArrayLike) -> Array:
    """Compute the Kelvin-Helmholtz (thermal) timescale: t_KH = G M² / (R L).

    Parameters
    ----------
    mass_kg, radius_m : float or ndarray
        Mass [kg] and radius [m].
    luminosity_w : float or ndarray
        Luminosity [W], positive.

    Returns
    -------
    float or ndarray
        Timescale [s].

    References
    ----------
    [Zinnecker2007] definition of t_KH.

    Validity
    --------
    This is an order-of-magnitude estimate; the prefactor (G M²/R) is only the scale of the gravitational energy.

    Verification
    ------------
    tests/unit/test_stars.py: between 10⁷ and 10⁸ yr for the Sun ([Zinnecker2007]: ~10⁷ yr for solar-type stars).
    """
    m = np.asarray(mass_kg, dtype=float)
    r = np.asarray(radius_m, dtype=float)
    lum = np.asarray(luminosity_w, dtype=float)
    if np.any(lum <= 0):
        raise ValueError("luminosity must be positive")
    return GRAVITATIONAL_CONSTANT.value * m**2 / (r * lum)


@source(
    "Pols2011",
    where="Section 2.4.2, Eq. 2.37",
    status="read",
    note="τ_nuc = φ f_nuc M c² / L ≈ 10^10 (M/M_⊙)(L_⊙/L) yr. For hydrogen burning the source gives φ ≈ 0.007 and, in a "
    "Sun-like star, f_nuc ≈ 0.1 (70% of the mass is hydrogen, burning in the inner ~10%)",
)
def nuclear_timescale_yr(
    mass_kg: ArrayLike, luminosity_w: ArrayLike, yield_fraction: float, fuel_fraction: float
) -> Array:
    """Compute the nuclear timescale and the main-sequence lifetime estimate.

    t_nuc = ε f M c² / L

    Parameters
    ----------
    mass_kg : float or ndarray
        Stellar mass [kg].
    luminosity_w : float or ndarray
        Mean luminosity [W], positive.
    yield_fraction : float
        ε, the fraction of the fuel's rest mass converted to radiation (between 0 and 1); the value is supplied by the user.
    fuel_fraction : float
        f, the ratio of the burning core to the total mass (between 0 and 1).

    Returns
    -------
    float or ndarray
        Timescale [yr].

    References
    ----------
    [Pols2011] Section 2.4.2, Eq. 2.37.

    Validity
    --------
    ε and f are supplied by the user. The values in the source: ε ≈ 0.007 for hydrogen burning, f ≈ 0.1 in a
    Sun-like star; these give about 10^10 yr for the Sun. For helium and heavier
    fuels ε is smaller by at least a factor of ten.

    Verification
    ------------
    tests/unit/test_stars.py: linear in mass, inversely proportional to luminosity; closed value by definition;
    the solar value in the source (about 10^10 yr).
    """
    if not (0.0 < yield_fraction < 1.0 and 0.0 < fuel_fraction <= 1.0):
        raise ValueError("yield_fraction must be in (0, 1) and fuel_fraction in (0, 1]")
    m = np.asarray(mass_kg, dtype=float)
    lum = np.asarray(luminosity_w, dtype=float)
    if np.any(lum <= 0):
        raise ValueError("luminosity must be positive")
    return yield_fraction * fuel_fraction * m * SPEED_OF_LIGHT.value**2 / lum / JULIAN_YEAR_SECONDS.value


@source(
    "Paxton2013",
    where="Section 'Rotationally-Enhanced Mass Loss' (TeX label s.rotation-mass-loss): L_Edd = 4π c G M/κ",
    status="read",
)
@source("Bludman2010", where="Section II.B, Eddington standard model: L_Edd := 4π c G M/κ_p", status="read")
def eddington_luminosity_w(mass_kg: ArrayLike, opacity_m2_kg: ArrayLike) -> Array:
    """Compute the Eddington luminosity limit: L_Edd = 4π c G M / κ.

    Parameters
    ----------
    mass_kg : float or ndarray
        Mass [kg].
    opacity_m2_kg : float or ndarray
        κ, opacity per unit mass [m² kg^-1], positive.

    Returns
    -------
    float or ndarray
        L_Edd [W].

    References
    ----------
    [Paxton2013] s.rotation-mass-loss; [Bludman2010] Eddington standard model.

    Verification
    ------------
    tests/unit/test_stars.py: linear in mass, inversely proportional to κ; closed value.
    """
    m = np.asarray(mass_kg, dtype=float)
    kappa = np.asarray(opacity_m2_kg, dtype=float)
    if np.any(kappa <= 0):
        raise ValueError("opacity must be positive")
    return 4.0 * np.pi * SPEED_OF_LIGHT.value * GRAVITATIONAL_CONSTANT.value * m / kappa


@source(
    "Paxton2013", where="sentence after the equation labeled mdotomega: Ω_crit² = (1 - L/L_Edd) G M / R³", status="read"
)
def critical_angular_velocity_rad_s(
    mass_kg: ArrayLike, radius_m: ArrayLike, luminosity_w: ArrayLike, opacity_m2_kg: ArrayLike
) -> Array:
    """Compute the critical (break-up) angular velocity reduced by radiation pressure.

    Ω_crit² = (1 - L / L_Edd) G M / R³

    Parameters
    ----------
    mass_kg, radius_m : float or ndarray
        Mass [kg] and radius [m].
    luminosity_w : float or ndarray
        Luminosity [W]; cannot exceed L_Edd.
    opacity_m2_kg : float or ndarray
        Surface opacity κ [m² kg^-1].

    Returns
    -------
    float or ndarray
        Ω_crit [rad s^-1]. The Kepler angular velocity for L = 0, zero for L → L_Edd.

    References
    ----------
    [Paxton2013] definition of Ω_crit.

    Validity
    --------
    It is not defined for L > L_Edd; ``ValueError`` is raised.

    Verification
    ------------
    tests/unit/test_stars.py: boundary values L = 0 and L = L_Edd.
    """
    m = np.asarray(mass_kg, dtype=float)
    r = np.asarray(radius_m, dtype=float)
    ratio = np.asarray(luminosity_w, dtype=float) / eddington_luminosity_w(m, opacity_m2_kg)
    if np.any(ratio > 1.0):
        raise ValueError("L exceeds the Eddington limit; the critical angular velocity is undefined")
    return np.sqrt((1.0 - ratio) * GRAVITATIONAL_CONSTANT.value * m / r**3)


@source(
    "Bludman2010",
    where="Section II.B: M_* := 3√10 0ω_3/π³ (hc/(G m_H^{4/3}))^{3/2} = 18.3 M_⊙",
    status="read",
    note="m_H is defined in the text as the atomic mass unit; 0ω_3 comes from the numerical solution",
)
def eddington_scale_mass_kg() -> float:
    """Compute the scale mass M_* of the Eddington quartic equation.

    M_* = 3√10 ω_3 / π³ · ( h c / (G m_H^{4/3}) )^{3/2}

    Returns
    -------
    float
        M_* [kg]; about 18.3 solar masses.

    References
    ----------
    [Bludman2010] Section II.B, definition of M_*.

    Verification
    ------------
    tests/unit/test_stars.py: 18.3 M_⊙ ([Bludman2010]) within 0.5 percent.
    """
    omega3 = lane_emden_mass_coefficient(3.0)
    mh = ATOMIC_MASS_UNIT.value
    return float(
        3.0
        * np.sqrt(10.0)
        * omega3
        / np.pi**3
        * (PLANCK.value * SPEED_OF_LIGHT.value / (GRAVITATIONAL_CONSTANT.value * mh ** (4.0 / 3.0))) ** 1.5
    )


@source("Bludman2010", where="Section II.B, Eddington quartic equation: (1-β)/β⁴ = (M μ²/M_*)²", status="read")
def eddington_beta(mass_kg: ArrayLike, mean_molecular_weight: float) -> Array:
    """Solve for β, the ratio of gas pressure to total pressure in the Eddington standard model.

    (1 - β) / β⁴ = ( M μ² / M_* )²

    Parameters
    ----------
    mass_kg : float or ndarray
        Stellar mass [kg], positive.
    mean_molecular_weight : float
        μ (in units of m_H), positive.

    Returns
    -------
    float or ndarray
        β ∈ (0, 1]; tends to 1 at small mass and to 0 at large mass.

    References
    ----------
    [Bludman2010] Eddington quartic equation.

    Validity
    --------
    A star with constant radiation entropy (Eddington standard model, n = 3) is assumed.

    Verification
    ------------
    tests/unit/test_stars.py: residual of the equation; β decreases monotonically with mass.
    """
    if mean_molecular_weight <= 0:
        raise ValueError("μ must be positive")
    y = np.asarray(mass_kg, dtype=float) * mean_molecular_weight**2 / eddington_scale_mass_kg()
    if np.any(y < 0):
        raise ValueError("the mass cannot be negative")

    def root(y_single: float) -> float:
        return float(brentq(lambda b: 1.0 - b - y_single**2 * b**4, 0.0, 1.0, xtol=1e-15, rtol=1e-15))

    return np.asarray(np.vectorize(root, otypes=[float])(y))[()]


@source("Bludman2010", where="Section II.B: L = L_Edd [1 - β(M)]", status="read")
def standard_model_luminosity_w(mass_kg: ArrayLike, mean_molecular_weight: float, opacity_m2_kg: float) -> Array:
    """Compute the luminosity of the Eddington standard model: L = L_Edd (1 - β).

    Parameters
    ----------
    mass_kg : float or ndarray
        Stellar mass [kg], positive.
    mean_molecular_weight : float
        μ (in units of m_H).
    opacity_m2_kg : float
        Photospheric opacity κ [m² kg^-1].

    Returns
    -------
    float or ndarray
        Luminosity [W]; grows as M³ at small mass and approaches L_Edd at large mass.

    References
    ----------
    [Bludman2010] L = L_Edd [1 - β(M)].

    Verification
    ------------
    tests/unit/test_stars.py: L/L_Edd = 1 - β; M³ law at small mass and the coefficient 0.003 ≈ (1/18.3)².
    """
    beta = eddington_beta(mass_kg, mean_molecular_weight)
    return eddington_luminosity_w(mass_kg, opacity_m2_kg) * (1.0 - beta)


@source(
    "Bludman2010",
    where="Section II.A (relativistic degenerate stars): K_WD = (hc/8)(3/π)^{1/3}(m_H μ_e)^{-4/3}",
    status="read",
    note="m_H is defined in the text as the atomic mass unit",
)
def white_dwarf_k_coefficient_si(molecular_weight_per_electron: float) -> float:
    """Compute the polytropic constant of a relativistic degenerate electron gas: P = K ρ^{4/3}.

    K_WD = (h c / 8) (3/π)^{1/3} (m_H μ_e)^{-4/3}

    Parameters
    ----------
    molecular_weight_per_electron : float
        μ_e, mean mass per electron (in units of m_H); 2 for He and C/O.

    Returns
    -------
    float
        K [Pa m⁴ kg^{-4/3}].

    References
    ----------
    [Bludman2010] expression for K_WD.

    Verification
    ------------
    tests/unit/test_stars.py: μ_e^{-4/3} scaling; together with the Chandrasekhar mass, 5.83/μ_e² M_⊙.
    """
    if molecular_weight_per_electron <= 0:
        raise ValueError("μ_e must be positive")
    return float(
        PLANCK.value
        * SPEED_OF_LIGHT.value
        / 8.0
        * (3.0 / np.pi) ** (1.0 / 3.0)
        * (ATOMIC_MASS_UNIT.value * molecular_weight_per_electron) ** (-4.0 / 3.0)
    )


@source(
    "Bludman2010",
    where="Section II (introduction) M = 4π 0ω_3 (K/πG)^{3/2}; Section II.A M_Ch = 5.83 M_⊙/μ_e² = 1.46 M_⊙",
    status="read",
    note="0ω_3 = -ξ_1² θ'(ξ_1) (n = 3) comes from the numerical solution; compared with the 5.83 in the text",
)
def chandrasekhar_mass_kg(molecular_weight_per_electron: float) -> float:
    """Compute the Chandrasekhar mass: the mass of an n = 3 polytrope depends only on K.

    M_Ch = 4π ω_3 ( K_WD / (π G) )^{3/2}

    Parameters
    ----------
    molecular_weight_per_electron : float
        μ_e (in units of m_H); 2 for He and C/O.

    Returns
    -------
    float
        M_Ch [kg]; about 1.46 solar masses for μ_e = 2.

    References
    ----------
    [Bludman2010] Section II and II.A.

    Validity
    --------
    Non-relativistic corrections, Coulomb effects and general relativity are ignored.

    Verification
    ------------
    tests/unit/test_stars.py: 5.83/μ_e² M_⊙, and 1.46 M_⊙ for μ_e = 2 ([Bludman2010]).
    """
    k = white_dwarf_k_coefficient_si(molecular_weight_per_electron)
    omega3 = lane_emden_mass_coefficient(3.0)
    return float(4.0 * np.pi * omega3 * (k / (np.pi * GRAVITATIONAL_CONSTANT.value)) ** 1.5)
