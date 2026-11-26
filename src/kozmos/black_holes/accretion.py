"""Black hole accretion: Eddington limit, Bondi flow, thin-disk temperature and tidal disruption.

The equations were read from the arXiv TeX files of the following sources on 25 November 2026:
Inayoshi, Visbal and Haiman [Inayoshi2020] (arXiv:1911.05791), Armitage [Armitage2011]
(arXiv:1011.1496), Komossa [Komossa2015] (arXiv:1505.01093) and Abramowicz and Fragile
[Abramowicz2013] (arXiv:1104.5499).

[Inayoshi2020] does not give the value of the electron scattering opacity κ_es. Here
κ_es = σ_T / m_p is used for fully ionized pure hydrogen (0.40 cm² g⁻¹); this choice belongs
to this module, not to the source, and it was compared with the numerical values of the source:
0.4505 Gyr (source 0.45) and 2.22e-5 M☉/yr (source 2.3e-5; 3.5% lower).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.constants import (
    GRAVITATIONAL_CONSTANT,
    PROTON_MASS,
    SPEED_OF_LIGHT,
    STEFAN_BOLTZMANN,
    THOMSON_CROSS_SECTION,
)

Array = NDArray[np.float64]

_G = GRAVITATIONAL_CONSTANT.value
_C = SPEED_OF_LIGHT.value
_M_P = PROTON_MASS.value
_SIGMA_T = THOMSON_CROSS_SECTION.value
_SIGMA_SB = STEFAN_BOLTZMANN.value
# Solar mass [kg]: nominal GM_sun / G (from the IAU 2015 B3 and CODATA 2022 constants).
_KAPPA_ES = _SIGMA_T / _M_P  # pure hydrogen, fully ionized [m² kg⁻¹]


def _positive(value: ArrayLike, name: str) -> Array:
    """Convert the value to an array and check that it is positive."""
    x = np.asarray(value, dtype=float)
    if np.any(~(x > 0.0)):
        raise ValueError(f"{name} must be positive")
    return x


def _efficiency(efficiency: ArrayLike) -> Array:
    """Convert the radiative efficiency to an array and check that 0 < η < 1."""
    e = np.asarray(efficiency, dtype=float)
    if np.any(~((e > 0.0) & (e < 1.0))):
        raise ValueError("radiative efficiency must be between 0 and 1")
    return e


@source(
    "Inayoshi2020",
    where="§'Growing BHs by Accretion: Is There an Eddington Limit?', L_Edd = 4π c G M / κ_es",
    status="read",
    note="κ_es is not defined in the source; σ_T/m_p was used (see the module docstring). The Abramowicz2013 §1 footnote "
    "gives L_Edd ≡ 1.2e38 (M/M☉) erg/s; this is about 5% smaller than the output here (1.26e38).",
)
def eddington_luminosity_w(mass_kg: ArrayLike) -> Array:
    """Compute the Eddington luminosity: L_Edd = 4π G M m_p c / σ_T.

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].

    Returns
    -------
    float or ndarray
        L_Edd [W]; ≈ 1.26e31 W (1.26e38 erg/s) for 1 M☉.

    References
    ----------
    [Inayoshi2020] §'Growing BHs by Accretion' (L_Edd = 4π c G M/κ_es).

    Validity
    --------
    Spherical geometry, fully ionized hydrogen, Thomson scattering only. In a disk geometry
    the radiation can escape vertically, so the limit can be exceeded (Abramowicz2013, thin disk section).

    Verification
    ------------
    tests/unit/test_black_holes.py: agrees with 1.2e38 erg/s (M/M☉) [Abramowicz2013 §1] within 6% (the source rounds to two digits).
    """
    return 4.0 * np.pi * _C * _G * _positive(mass_kg, "mass") / _KAPPA_ES


@source(
    "Abramowicz2013",
    where="§1 (Introduction), efficiency definition η ≡ L/(Ṁ c²)",
    status="read",
    note="L = η Ṁ c² is the definition itself.",
)
def accretion_luminosity_w(accretion_rate_kg_s: ArrayLike, efficiency: ArrayLike) -> Array:
    """Compute the accretion luminosity: L = η Ṁ c².

    Parameters
    ----------
    accretion_rate_kg_s : float or ndarray
        Accretion rate Ṁ [kg s^-1], positive.
    efficiency : float or ndarray
        Radiative efficiency η, 0 < η < 1.

    Returns
    -------
    float or ndarray
        L [W].

    References
    ----------
    [Abramowicz2013] §1, η ≡ L/(Ṁ c²).

    Verification
    ------------
    tests/unit/test_black_holes.py: round trip through L_Edd with the Eddington accretion rate.
    """
    return _efficiency(efficiency) * _positive(accretion_rate_kg_s, "accretion rate") * _C**2


@source(
    "Inayoshi2020",
    where="§'Growing BHs by Accretion' and Table 3, Ṁ_Edd ≡ L_Edd / (0.1 c²) (a footnote also mentions the L_Edd/c² definition)",
    status="read",
    note="The source fixes η = 0.1; here η is a parameter and η = 0.1 is the source's definition.",
)
def eddington_accretion_rate_kg_s(mass_kg: ArrayLike, efficiency: ArrayLike = 0.1) -> Array:
    """Compute the Eddington accretion rate: Ṁ_Edd = L_Edd / (η c²).

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].
    efficiency : float or ndarray, optional
        Radiative efficiency η, 0 < η < 1; default 0.1 (the source's definition).

    Returns
    -------
    float or ndarray
        Ṁ_Edd [kg s^-1]. 2.22e-5 M☉/yr for 10³ M☉ (η = 0.1); the [Inayoshi2020] Table 3 value is 2.3e-5.

    References
    ----------
    [Inayoshi2020] Table 3 and §'Growing BHs by Accretion'.

    Verification
    ------------
    tests/unit/test_black_holes.py: agrees with the Table 3 value 2.3e-5 M☉/yr within 5%; 1/η scaling.
    """
    return eddington_luminosity_w(mass_kg) / (_efficiency(efficiency) * _C**2)


@source(
    "Inayoshi2020",
    where="§'Growing BHs by Accretion', t_grow ≈ 0.45 ε/(1−ε) f_duty^{-1} ln(M/M_seed) Gyr; §1 t_Edd ≈ 50 Myr (ε = 0.1)",
    status="read",
    note="The source gives only the 0.45 Gyr coefficient; here the coefficient is computed as σ_T c/(4π G m_p) and "
    "compared with 0.45 Gyr in the test. The e-folding time is the part of the t_grow expression before the ln factor.",
)
def salpeter_time_s(efficiency: ArrayLike = 0.1) -> Array:
    """Compute the e-folding (Salpeter) time of Eddington-limited growth.

    t_Sal = [η / (1 − η)] σ_T c / (4π G m_p)

    Mass M(t) = M0 exp(t / t_Sal). ≈ 50 Myr for η = 0.1.

    Parameters
    ----------
    efficiency : float or ndarray, optional
        Radiative efficiency η, 0 < η < 1; default 0.1.

    Returns
    -------
    float or ndarray
        t_Sal [s].

    References
    ----------
    [Inayoshi2020] the t_grow expression and t_Edd ≈ 50 Myr in §1.

    Validity
    --------
    Continuous Eddington-limited accretion, duty cycle 1; the fraction (1 − η) of the
    energy falling onto the mass is added to the black hole.

    Verification
    ------------
    tests/unit/test_black_holes.py: 0.45 Gyr coefficient and 50 Myr; exponential growth of M(t).
    """
    e = _efficiency(efficiency)
    return e / (1.0 - e) * _SIGMA_T * _C / (4.0 * np.pi * _G * _M_P)


@source(
    "Inayoshi2020",
    where="§'Inflow from large scales' (R_B ≡ G M / c_s²) and Table 3 (TeX label tab:phys)",
    status="read",
)
def bondi_radius_m(mass_kg: ArrayLike, sound_speed_m_s: ArrayLike) -> Array:
    """Compute the Bondi radius: R_B = G M / c_s².

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].
    sound_speed_m_s : float or ndarray
        Sound speed of the gas [m s^-1], positive.

    Returns
    -------
    float or ndarray
        R_B [m].

    References
    ----------
    [Inayoshi2020] R_B definition and Table 3.

    Verification
    ------------
    tests/unit/test_black_holes.py: Table 3 value 0.6 pc (10³ M☉, T = 10³ K); M and c_s scaling.
    """
    return _G * _positive(mass_kg, "mass") / _positive(sound_speed_m_s, "sound speed") ** 2


@source("Inayoshi2020", where="Table 3, Ṁ_B ≡ π e^{3/2} ρ G² M² / c_s³", status="read")
def bondi_accretion_rate_kg_s(mass_kg: ArrayLike, density_kg_m3: ArrayLike, sound_speed_m_s: ArrayLike) -> Array:
    """Compute the Bondi accretion rate: Ṁ_B = π e^{3/2} ρ G² M² / c_s³.

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].
    density_kg_m3 : float or ndarray
        Mass density ρ of the gas far from the black hole [kg m^-3].
    sound_speed_m_s : float or ndarray
        Sound speed of the gas [m s^-1].

    Returns
    -------
    float or ndarray
        Ṁ_B [kg s^-1].

    References
    ----------
    [Inayoshi2020] Table 3.

    Validity
    --------
    The definition in the source is for a spherical isothermal flow; angular momentum,
    radiation feedback and relativistic effects are neglected.

    Verification
    ------------
    tests/unit/test_black_holes.py: Table 3 value 4.5e-3 M☉/yr (n_H = 1e4 cm^-3, T = 10³ K, 10³ M☉);
    identity Ṁ_B = π e^{3/2} ρ R_B² c_s.
    """
    m = _positive(mass_kg, "mass")
    rho = _positive(density_kg_m3, "density")
    cs = _positive(sound_speed_m_s, "sound speed")
    return np.pi * np.exp(1.5) * rho * (_G * m) ** 2 / cs**3


@source(
    "Armitage2011",
    where="§2 'Accretion Disk Evolution', T_eff⁴ = 3 G M Ṁ / (8π σ r³) [1 − √(r_in/r)] (same eqnarray as TeX label eq:2_nusigma, second line)",
    status="read",
)
def thin_disk_effective_temperature_k(
    radius_m: ArrayLike, mass_kg: ArrayLike, accretion_rate_kg_s: ArrayLike, inner_radius_m: ArrayLike
) -> Array:
    """Compute the effective temperature profile of a steady, thin, Newtonian disk.

    T_eff(r) = { 3 G M Ṁ / (8π σ_SB r³) · [1 − √(r_in / r)] }^{1/4}

    Parameters
    ----------
    radius_m : float or ndarray
        Disk radius r [m]; must satisfy r ≥ r_in.
    mass_kg : float or ndarray
        Central mass [kg].
    accretion_rate_kg_s : float or ndarray
        Accretion rate Ṁ [kg s^-1].
    inner_radius_m : float or ndarray
        Inner radius r_in [m] (the torque is zero here); typically the ISCO for a black hole.

    Returns
    -------
    float or ndarray
        T_eff [K]; 0 at r = r_in.

    References
    ----------
    [Armitage2011] §2 'Accretion Disk Evolution', steady viscous disk equations.

    Validity
    --------
    Newtonian gravity (no relativistic correction factors), zero torque at r_in, local
    radiative equilibrium in each annulus, thin and optically thick disk. For the
    Novikov-Thorne relativistic corrections see Abramowicz2013; they are not included here.
    The source was written for protoplanetary disks; the profile is independent of the
    viscosity law (α) and relies only on conservation of energy and angular momentum;
    around a black hole it is usable only in the Newtonian limit (r ≫ r_G).

    Verification
    ------------
    tests/unit/test_black_holes.py: r^{-3/4} for r ≫ r_in; Ṁ^{1/4} scaling; maximum value.
    """
    r = _positive(radius_m, "radius")
    r_ic = _positive(inner_radius_m, "inner radius")
    if np.any(r < r_ic):
        raise ValueError("radius cannot be smaller than the inner radius")
    m = _positive(mass_kg, "mass")
    mdot = _positive(accretion_rate_kg_s, "accretion rate")
    t4 = 3.0 * _G * m * mdot / (8.0 * np.pi * _SIGMA_SB * r**3) * (1.0 - np.sqrt(r_ic / r))
    return t4**0.25


@source(
    "Armitage2011",
    where="§2 'Accretion Disk Evolution', T_eff profile; the location of the maximum was derived by this project",
    status="read",
    note="The profile equation was read; the solution of dT/dr = 0, u = √(r_in/r) = 6/7, is this project's own derivation "
    "and is verified in the test by numerical maximization.",
)
def thin_disk_peak_temperature_radius_m(inner_radius_m: ArrayLike) -> Array:
    """Compute the radius at which the effective temperature of a thin disk is highest: r = (49/36) r_in.

    Parameters
    ----------
    inner_radius_m : float or ndarray
        Inner radius r_in [m].

    Returns
    -------
    float or ndarray
        Radius r [m] at which T_eff peaks.

    References
    ----------
    [Armitage2011] §2, T_eff profile (the maximization is this project's derivation).

    Validity
    --------
    Newtonian disk, zero torque at r_in; the validity conditions of the [Armitage2011] profile.
    Another review quotes the maximum at 25/9 r_in for the same profile; maximising the profile gives 49/36 r_in
    (checked numerically in the test), which is what this function returns.

    Verification
    ------------
    tests/unit/test_black_holes.py: numerical maximization with scipy.
    """
    return 49.0 / 36.0 * _positive(inner_radius_m, "inner radius")


@source(
    "Armitage2011",
    where="§2 'Accretion Disk Evolution', T_eff profile; the maximum value was derived by this project",
    status="read",
    note="T_max⁴ = [3 G M Ṁ / (8π σ r_in³)] (6/7)⁶ (1/7): for u = √(r_in/r), u⁶(1−u) is largest at u = 6/7; "
    "this is the project's own derivation, verified in the test against the numerical maximum of the profile function.",
)
def thin_disk_peak_temperature_k(
    mass_kg: ArrayLike, accretion_rate_kg_s: ArrayLike, inner_radius_m: ArrayLike
) -> Array:
    """Compute the maximum effective temperature of a thin disk.

    T_max = { 3 G M Ṁ / (8π σ_SB r_in³) · (6/7)⁶ / 7 }^{1/4} ≈ 0.488 (3 G M Ṁ / 8π σ r_in³)^{1/4}

    Parameters
    ----------
    mass_kg : float or ndarray
        Central mass [kg].
    accretion_rate_kg_s : float or ndarray
        Accretion rate Ṁ [kg s^-1].
    inner_radius_m : float or ndarray
        Inner radius r_in [m].

    Returns
    -------
    float or ndarray
        T_max [K].

    References
    ----------
    [Armitage2011] §2, T_eff profile (the maximization is this project's derivation).

    Validity
    --------
    Conditions of the [Armitage2011] profile: Newtonian, zero torque at r_in.

    Verification
    ------------
    tests/unit/test_black_holes.py: value of the profile function at r = 49/36 r_in and numerical maximization;
    tests/property/test_black_holes_property.py: M^{1/4} Ṁ^{1/4} r_in^{-3/4} scaling.
    """
    m = _positive(mass_kg, "mass")
    mdot = _positive(accretion_rate_kg_s, "accretion rate")
    r_ic = _positive(inner_radius_m, "inner radius")
    t4 = 3.0 * _G * m * mdot / (8.0 * np.pi * _SIGMA_SB * r_ic**3) * (6.0 / 7.0) ** 6 / 7.0
    return t4**0.25


@source(
    "Komossa2015",
    where="§1 (Introduction), tidal radius equation for r_t (no label): r_t ≃ 7e12 (M/10⁶ M☉)^{1/3} (m*/M☉)^{-1/3} (r*/R☉) cm",
    status="read",
    note="The source gives only the numerical scaling. Here r_t = r* (M/m*)^{1/3} is written; the coefficient "
    "R☉ (10⁶)^{1/3} = 6.96e12 cm agrees with the 7e12 cm in the source within 1% and is checked in the test.",
)
def tidal_disruption_radius_m(
    black_hole_mass_kg: ArrayLike, stellar_mass_kg: ArrayLike, stellar_radius_m: ArrayLike
) -> Array:
    """Compute the radius at which a star is disrupted by tidal forces: r_t = r* (M / m*)^{1/3}.

    Parameters
    ----------
    black_hole_mass_kg : float or ndarray
        Black hole mass M [kg].
    stellar_mass_kg : float or ndarray
        Stellar mass m* [kg].
    stellar_radius_m : float or ndarray
        Stellar radius r* [m].

    Returns
    -------
    float or ndarray
        r_t [m].

    References
    ----------
    [Komossa2015] §1, tidal radius equation.

    Validity
    --------
    Order-of-magnitude estimate (the O(1) factor depends on the structure of the star);
    Newtonian tidal force, non-rotating black hole. Rotation can change the disruption
    threshold (source §1).

    Verification
    ------------
    tests/unit/test_black_holes.py: 7e12 cm coefficient (source); M^{1/3} scaling.
    """
    m_kd = _positive(black_hole_mass_kg, "black hole mass")
    m_y = _positive(stellar_mass_kg, "stellar mass")
    r_y = _positive(stellar_radius_m, "stellar radius")
    return r_y * (m_kd / m_y) ** (1.0 / 3.0)


@source(
    "Komossa2015",
    where="§1, the r_t equation and the sentence after it: Sun-like stars are swallowed whole above ≳ 10⁸ M☉, white dwarfs are disrupted below ≲ 10⁵ M☉",
    status="read",
    note="The solution for M from the condition r_t = R_S was derived by this project: M = (r* c² / 2G)^{3/2} / √m*. The values "
    "10⁸ and 10⁵ M☉, which the source gives only as orders of magnitude, are compared in the test.",
)
def hills_mass_kg(stellar_mass_kg: ArrayLike, stellar_radius_m: ArrayLike) -> Array:
    """Compute the black hole mass at which the tidal radius equals the Schwarzschild radius (Hills mass).

    r* (M / m*)^{1/3} = 2 G M / c²  ⇒  M_H = (r* c² / 2G)^{3/2} / √m*

    Above this mass the star is swallowed whole without a flare.

    Parameters
    ----------
    stellar_mass_kg : float or ndarray
        Stellar mass m* [kg].
    stellar_radius_m : float or ndarray
        Stellar radius r* [m].

    Returns
    -------
    float or ndarray
        M_H [kg]. 1.14e8 M☉ for the Sun.

    References
    ----------
    [Komossa2015] §1 (r_t equation and swallowing threshold).

    Validity
    --------
    Non-rotating black hole, Newtonian tidal radius and the Schwarzschild radius for the
    horizon; Kerr and relativistic tidal corrections change this threshold.

    Verification
    ------------
    tests/unit/test_black_holes.py: Sun ≈ 1e8 M☉, white dwarf ~1e5 M☉; identity r_t(M_H) = R_S.
    """
    m_y = _positive(stellar_mass_kg, "stellar mass")
    r_y = _positive(stellar_radius_m, "stellar radius")
    return (r_y * _C**2 / (2.0 * _G)) ** 1.5 / np.sqrt(m_y)
