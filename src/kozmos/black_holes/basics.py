"""Black hole geometry, Kerr orbits, Hawking thermodynamics and binary black holes.

The equations were read from the following sources on 25 November 2026: Abramowicz and Fragile
[Abramowicz2013] (arXiv:1104.5499, TeX), Page [Page2005] (arXiv:hep-th/0409024, TeX),
Sathyaprakash and Schutz [Sathyaprakash2009] (arXiv:0903.0338, TeX) and Event Horizon
Telescope Paper I [EHT2019] (arXiv:1906.11238, PDF only; there is no TeX source, so the
section and page are given instead of an equation number).

Anything that was not read carries ``status="pending"``, and the ``note`` field of the
function states what was read and what was not.

For a Kerr black hole, ``spin`` is the dimensionless spin a_* = a/M = cJ/(G M²);
a positive value means that the orbit runs in the same direction as the rotation of the
black hole (prograde), a negative value means the opposite direction (retrograde). Radii
with the suffix ``_rg`` are in units of r_G = G M / c².
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.constants import BOLTZMANN, GRAVITATIONAL_CONSTANT, PLANCK, SPEED_OF_LIGHT

Array = NDArray[np.float64]

_G = GRAVITATIONAL_CONSTANT.value
_C = SPEED_OF_LIGHT.value
_HBAR = PLANCK.value / (2.0 * np.pi)  # reduced Planck constant, ħ = h / 2π
_RAD_UAS = 180.0 / np.pi * 3600.0 * 1e6  # radian → microarcsecond


def _positive(value: ArrayLike, name: str) -> Array:
    """Convert the value to an array and check that it is positive."""
    x = np.asarray(value, dtype=float)
    if np.any(~(x > 0.0)):
        raise ValueError(f"{name} must be positive")
    return x


def _spin(spin: ArrayLike) -> Array:
    """Convert the dimensionless spin to an array and check that |a_*| ≤ 1."""
    a = np.asarray(spin, dtype=float)
    if np.any(~(np.abs(a) <= 1.0)):
        raise ValueError("dimensionless spin must satisfy |a*| ≤ 1")
    return a


# --------------------------------------------------------------------------------------
# Schwarzschild geometry
# --------------------------------------------------------------------------------------


@source("EHT2019", where="§1 (Introduction), definition r_g ≡ GM/c², page L1", status="read")
def gravitational_radius_m(mass_kg: ArrayLike) -> Array:
    """Compute the gravitational radius: r_G = G M / c².

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg], positive.

    Returns
    -------
    float or ndarray
        r_G [m].

    References
    ----------
    [EHT2019] §1, "r_g ≡ GM/c² ... characteristic lengthscale".

    Verification
    ------------
    tests/unit/test_black_holes.py: 1.477 km for one solar mass; linear in mass.
    """
    return _G * _positive(mass_kg, "mass") / _C**2


@source("EHT2019", where="§1 (Introduction), R_S = 2 r_g, page L1", status="read")
def schwarzschild_radius_m(mass_kg: ArrayLike) -> Array:
    """Compute the Schwarzschild radius (horizon of a non-rotating black hole): R_S = 2 G M / c².

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg], positive.

    Returns
    -------
    float or ndarray
        R_S [m].

    References
    ----------
    [EHT2019] §1, "R_S = 2 r_g".

    Verification
    ------------
    tests/unit/test_black_holes.py: 2.95 km for one solar mass; same as the a* = 0 limit of the Kerr horizon.
    """
    return 2.0 * gravitational_radius_m(mass_kg)


@source(
    "Carroll1997",
    where="Section 7, Eq. 7.58-7.61",
    status="read",
    note="for two observers at fixed positions, ω2/ω1 = [(1 − 2GM/r1)/(1 − 2GM/r2)]^{1/2} (G = c = 1); "
    "here the receiver is at infinity (r2 → ∞): 1 + z = (1 − R_S/r)^{−1/2}",
)
def gravitational_redshift(radius_m: ArrayLike, mass_kg: ArrayLike) -> Array:
    """Compute the gravitational redshift of a static emitter as seen by an observer at infinity.

    1 + z = (1 − R_S / r)^(−1/2)

    Parameters
    ----------
    radius_m : float or ndarray
        Schwarzschild radial coordinate of the emitter [m]; must be larger than R_S.
    mass_kg : float or ndarray
        Black hole mass [kg].

    Returns
    -------
    float or ndarray
        Redshift z (dimensionless).

    References
    ----------
    [Carroll1997] Eq. 7.58-7.61.

    Validity
    --------
    Static (non-orbiting) emitter only; r > R_S. z → ∞ as the horizon is approached.

    Verification
    ------------
    tests/unit/test_black_holes.py: z = √3 − 1 for r = 3 R_S/2; z ≈ R_S/(2r) for r → ∞.
    """
    r = _positive(radius_m, "radius")
    rs = schwarzschild_radius_m(mass_kg)
    if np.any(r <= rs):
        raise ValueError("radius must be larger than the Schwarzschild radius")
    return (1.0 - rs / r) ** -0.5 - 1.0


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies' (TeX label schwarzschild-frequencies), x_ph = 3",
    status="read",
    note="The a* = 0 value of the Kerr formula (TeX label photon-radius-kerr) also gives 3 r_G.",
)
def photon_sphere_radius_m(mass_kg: ArrayLike) -> Array:
    """Compute the photon sphere radius of a non-rotating black hole: r_ph = 3 G M / c².

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].

    Returns
    -------
    float or ndarray
        r_ph [m].

    References
    ----------
    [Abramowicz2013] x_ph = r_ph/r_G = 3 for Schwarzschild.

    Verification
    ------------
    tests/unit/test_black_holes.py: the a* = 0 value of the Kerr formula and the ratio to R_S (3/2).
    """
    return 3.0 * gravitational_radius_m(mass_kg)


@source(
    "EHT2019",
    where="§1 (Introduction), 'photon capture radius' R_c = √27 r_g, page L1",
    status="read",
    note="Read from the PDF; text extraction drops the √ symbol, and R_c = √27 r_g was verified on the page image.",
)
def critical_impact_parameter_m(mass_kg: ArrayLike) -> Array:
    """Compute the critical impact parameter of a non-rotating black hole: b_c = √27 G M / c².

    Photons arriving with b < b_c are captured, those with b > b_c escape; b = b_c is the
    unstable circular photon orbit (photon ring).

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].

    Returns
    -------
    float or ndarray
        b_c [m].

    References
    ----------
    [EHT2019] §1, "R_c = √27 r_g".

    Verification
    ------------
    tests/unit/test_black_holes.py: independent computation b_c = r_ph / √(1 − R_S/r_ph)
    from the photon sphere radius and the Schwarzschild metric.
    """
    return np.sqrt(27.0) * gravitational_radius_m(mass_kg)


@source(
    "EHT2019",
    where="§1 R_c = √27 r_g; §7 'Model Comparison and Parameter Estimation' θ_g ≡ GM/(c² D)",
    status="read",
    note="Diameter = 2 R_c / D is the small-angle approximation (geometry, not written in the source). The 37.6 and 21.3 µas "
    "values that the EHT gives for Schwarzschild are reproduced in the test.",
)
def shadow_angular_diameter_uas(mass_kg: ArrayLike, distance_m: ArrayLike) -> Array:
    """Compute the angular diameter of the shadow of a non-rotating black hole on the sky.

    θ_shadow = 2 √27 θ_g,  θ_g = G M / (c² D)

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].
    distance_m : float or ndarray
        Distance to the observer [m]; D ≫ r_G is assumed.

    Returns
    -------
    float or ndarray
        Angular diameter [µas, microarcseconds].

    References
    ----------
    [EHT2019] §1 (R_c) and §7 (definition of θ_g; the d = α θ_g given there is the diameter of the emission ring, not of the shadow).

    Validity
    --------
    Non-rotating black hole, small-angle approximation. Rotation changes the shadow size by
    about 4% (source, §1); for Kerr, α depends on spin and inclination and is not computed here.

    Verification
    ------------
    tests/unit/test_black_holes.py: 37.6 µas (stellar mass) and 21.3 µas (gas mass) for M87*.
    """
    d = _positive(distance_m, "distance")
    return 2.0 * critical_impact_parameter_m(mass_kg) / d * _RAD_UAS


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies' (TeX label schwarzschild-frequencies), x_ms = 6",
    status="read",
)
def schwarzschild_isco_radius_m(mass_kg: ArrayLike) -> Array:
    """Compute the radius of the innermost stable circular orbit (ISCO) of a non-rotating black hole: r_ISCO = 6 G M / c².

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].

    Returns
    -------
    float or ndarray
        r_ISCO [m].

    References
    ----------
    [Abramowicz2013] x_ms = r_ms/r_G = 6 for Schwarzschild.

    Verification
    ------------
    tests/unit/test_black_holes.py: the a* = 0 value of the Kerr ISCO formula; 3 R_S.
    """
    return 6.0 * gravitational_radius_m(mass_kg)


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies', Ω_K² = GM/r³ (TeX label schwarzschild-frequencies), x_ms = 6",
    status="read",
    note="The conversion f = Ω_K / (2π) is a definition.",
)
def isco_orbital_frequency_hz(mass_kg: ArrayLike) -> Array:
    """Compute the Kepler orbital frequency at the ISCO (non-rotating black hole).

    Ω_K² = G M / r³,  r = 6 G M / c²  ⇒  f_orb = c³ / (2π 6^{3/2} G M)

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].

    Returns
    -------
    float or ndarray
        Orbital frequency [Hz], as measured by an observer at infinity.

    References
    ----------
    [Abramowicz2013] Ω_K² = GM/r³ and x_ms = 6 for Schwarzschild.

    Validity
    --------
    r is the Schwarzschild radial coordinate; Ω_K is measured with respect to the time of an observer at infinity.

    Verification
    ------------
    tests/unit/test_black_holes.py: independent computation with Ω_K² = GM/r³; 1/M scaling.
    """
    m = _positive(mass_kg, "mass")
    r = schwarzschild_isco_radius_m(m)
    return np.sqrt(_G * m / r**3) / (2.0 * np.pi)


# --------------------------------------------------------------------------------------
# Kerr geometry
# --------------------------------------------------------------------------------------


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies', TeX label stable-radius-kerr (formula of Bardeen et al.)",
    status="read",
    note="The formula of the source is for the prograde orbit; Z1 and Z2 are even in a*. Both signs are "
    "verified independently in the test with the condition (ω_r*)² = 0 of the epicyclic-radial equation of Abramowicz2013.",
)
@source(
    "Reynolds2021",
    where="Section 2: r_isco equation and the definitions of Z1, Z2",
    status="read",
    note="r_isco = (3 + Z2 ∓ [(3 − Z1)(3 + Z1 + 2Z2)]^{1/2}) r_g; the ∓ sign is for prograde and retrograde "
    "orbits, respectively. The same expression is Eq. 2.21 in the original source [Bardeen1972].",
)
def kerr_isco_radius_rg(spin: ArrayLike) -> Array:
    """Compute the ISCO radius of a Kerr black hole (Bardeen, Press, Teukolsky formula).

    r_ISCO / r_G = 3 + Z2 ∓ √[(3 − Z1)(3 + Z1 + 2 Z2)]

    Z1 = 1 + (1 − a*²)^{1/3} [(1 + a*)^{1/3} + (1 − a*)^{1/3}],  Z2 = √(3 a*² + Z1²)

    The minus sign is for prograde orbits (a* ≥ 0), the plus sign for retrograde orbits (a* < 0).

    Parameters
    ----------
    spin : float or ndarray
        Dimensionless spin a_* ∈ [−1, 1]; a negative value means a retrograde orbit.

    Returns
    -------
    float or ndarray
        r_ISCO / r_G (dimensionless): 6 for a* = 0, 1 for a* = 1, 9 for a* = −1.

    References
    ----------
    [Abramowicz2013] stable-radius-kerr; [Reynolds2021] Section 2.

    Validity
    --------
    Kerr metric, equatorial plane, test particle. Exact to rounding at |a*| = 1 and for moderate spins; just below
    |a*| = 1 the terms 1 - a*^2 and 3 - Z1 cancel and the relative error grows to 5e-12 at 1 - |a*| = 1e-8 and
    4e-9 at 1e-15 (measured against a 60-digit calculation).

    Verification
    ------------
    tests/unit/test_black_holes.py: limiting values (1, 6, 9) and the condition (ω_r*)² = 0;
    tests/property/test_black_holes_property.py: monotonic in spin.
    """
    a_signed = _spin(spin)
    a = np.abs(a_signed)
    z1 = 1.0 + np.cbrt(1.0 - a**2) * (np.cbrt(1.0 + a) + np.cbrt(1.0 - a))
    z2 = np.sqrt(3.0 * a**2 + z1**2)
    root = np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    sign = np.where(a_signed >= 0.0, 1.0, -1.0)
    return 3.0 + z2 - sign * root


def _isco_energy(spin: ArrayLike) -> Array:
    """Return the energy per unit mass at the ISCO, E = √(1 − 2/(3 x_ISCO)); helper."""
    x = kerr_isco_radius_rg(spin)
    return np.sqrt(1.0 - 2.0 / (3.0 * x))


@source(
    "Bardeen1972",
    where="Eq. 2.20: stability condition 1 − (E/μ)² ≥ (2/3)(M/r), equality at r = r_ms",
    status="read",
    note="Read from the ADS scan. Equality of the condition at r_ms gives Ẽ_ms = (1 − 2/(3 r̃_ms))^{1/2}. The test "
    "checks that the circular-orbit energy Ẽ(r) has its minimum at the ISCO (dẼ/dr = 0) and the Schwarzschild value √(8/9).",
)
def kerr_isco_specific_energy(spin: ArrayLike) -> Array:
    """Compute the energy per unit rest mass of the circular orbit at the ISCO.

    E_ISCO = √(1 − 2 / (3 x_ISCO)),  x_ISCO = r_ISCO / r_G

    Parameters
    ----------
    spin : float or ndarray
        Dimensionless spin a_* ∈ [−1, 1]; a negative value means a retrograde orbit.

    Returns
    -------
    float or ndarray
        E_ISCO / (m c²): √(8/9) ≈ 0.9428 for a* = 0.

    References
    ----------
    [Bardeen1972] Eq. 2.20.

    Validity
    --------
    Kerr metric, equatorial plane, test particle.

    Verification
    ------------
    tests/unit/test_black_holes.py: √(8/9) and the minimum of the independent circular-orbit energy.
    """
    return _isco_energy(spin)


@source(
    "Abramowicz2013",
    where="§1 (Introduction), efficiency definition η ≡ L/(Ṁ c²)",
    status="read",
    note="Inayoshi2020 gives '~0.06' for a non-rotating black hole (agrees with 0.057 in the test).",
)
@source(
    "Reynolds2021",
    where="subsection 'Radiative efficiency' (TeX labels efficiency and eqn:eta, Spin2020.tex lines 470-476): "
    "η = 1 − e_isco/c² for a Novikov-Thorne disk",
    status="read",
    note="e_isco: total energy of a unit mass on the circular orbit at the ISCO; the closed form of e_isco "
    "is in kerr_isco_specific_energy [Bardeen1972, Eq. 2.20].",
)
def isco_radiative_efficiency(spin: ArrayLike) -> Array:
    """Compute the radiative efficiency for a thin disk: η = 1 − E_ISCO.

    η ≡ L / (Ṁ c²). For a non-rotating black hole 1 − √(8/9) ≈ 0.0572; ≈ 0.42 for a* = 1.

    Parameters
    ----------
    spin : float or ndarray
        Dimensionless spin a_* ∈ [−1, 1]; a negative value means a retrograde disk.

    Returns
    -------
    float or ndarray
        Efficiency η (dimensionless).

    References
    ----------
    [Abramowicz2013] §1; [Reynolds2021] subsection 'Radiative efficiency', Eq. eta.

    Validity
    --------
    Geometrically thin, radiatively efficient disk; radiation coming from inside the ISCO and
    the recapture of light that falls into the horizon are neglected.

    Verification
    ------------
    tests/unit/test_black_holes.py: 0.057 (Inayoshi2020 '~0.06'); tests/property/test_black_holes_property.py:
    increases with spin.
    """
    return 1.0 - _isco_energy(spin)


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies', r_H (same eqnarray, line without a label)",
    status="read",
)
@source("Page2005", where="TeX label eq:3, r_+ (Q = 0)", status="read")
def kerr_horizon_radius_rg(spin: ArrayLike) -> Array:
    """Compute the event horizon radius of a Kerr black hole: r_+ = r_G (1 + √(1 − a*²)).

    Parameters
    ----------
    spin : float or ndarray
        Dimensionless spin a_* ∈ [−1, 1].

    Returns
    -------
    float or ndarray
        r_+ / r_G: 2 for a* = 0, 1 for a* = ±1.

    References
    ----------
    [Abramowicz2013] r_H; [Page2005] TeX label eq:3 (with the charge set to zero).

    Verification
    ------------
    tests/unit/test_black_holes.py: values for a* = 0 and 1; agreement with R_S/2r_G.
    """
    a = _spin(spin)
    return 1.0 + np.sqrt(1.0 - a**2)


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies', r_0 (ergosphere; same eqnarray, line without a label)",
    status="read",
)
def kerr_ergosphere_radius_rg(spin: ArrayLike, polar_angle_rad: ArrayLike = np.pi / 2) -> Array:
    """Compute the ergosphere radius of a Kerr black hole: r_0 = r_G (1 + √(1 − a*² cos²θ)).

    Parameters
    ----------
    spin : float or ndarray
        Dimensionless spin a_* ∈ [−1, 1].
    polar_angle_rad : float or ndarray, optional
        Polar angle θ [rad]; the default is the equator (π/2), where the ergosphere is widest.

    Returns
    -------
    float or ndarray
        r_0 / r_G: 2 at the equator, tangent to the horizon (r_+) at the poles.

    References
    ----------
    [Abramowicz2013] r_0.

    Verification
    ------------
    tests/unit/test_black_holes.py: 2 r_G at the equator; horizon radius at the pole;
    tests/property/test_black_holes_property.py: horizon ≤ ergosphere.
    """
    a = _spin(spin)
    theta = np.asarray(polar_angle_rad, dtype=float)
    return 1.0 + np.sqrt(1.0 - (a * np.cos(theta)) ** 2)


@source(
    "Page2005",
    where="TeX label eq:3, Ω = 4π a / A (Q = 0)",
    status="read",
    note="The source uses units G = c = 1; converted to SI with Ω_H = c³ Ω̃ / (G M).",
)
def horizon_angular_velocity_rad_s(mass_kg: ArrayLike, spin: ArrayLike) -> Array:
    """Compute the angular velocity of the Kerr horizon.

    Ω_H = (c³ / G M) · a* / [2 (1 + √(1 − a*²))]

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].
    spin : float or ndarray
        Dimensionless spin a_* ∈ [−1, 1].

    Returns
    -------
    float or ndarray
        Ω_H [rad s^-1]; the sign is the same as the sign of the spin.

    References
    ----------
    [Page2005] TeX label eq:3 (Ω, with the charge set to zero).

    Verification
    ------------
    tests/unit/test_black_holes.py: Ω_H = c³/(2 G M) for a* = 1; a* c³/(4 G M) for small a*.
    """
    m = _positive(mass_kg, "mass")
    a = _spin(spin)
    return _C**3 / (_G * m) * a / (2.0 * (1.0 + np.sqrt(1.0 - a**2)))


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies', TeX label photon-radius-kerr",
    status="read",
    note="As printed in the source, the formula 2 r_G{1 + cos[(2/3) arccos(a*)]} gives 4 r_G for a* = 1, which is the retrograde "
    "orbit; the prograde orbit should be r_G at a* = 1. Therefore arccos(−a*) is used here for the prograde orbit: "
    "the sign of a* in the source appears to be counted as positive for retrograde. The result is verified in the test "
    "against the roots of the cubic y³ − 3y + 2a* = 0 (y = √(r/r_G)) in the same source.",
)
def kerr_photon_orbit_radius_rg(spin: ArrayLike) -> Array:
    """Compute the radius of the equatorial circular photon orbit of a Kerr black hole.

    r_ph / r_G = 2 {1 + cos[(2/3) arccos(−a*)]}

    Parameters
    ----------
    spin : float or ndarray
        Dimensionless spin a_* ∈ [−1, 1]; for a positive value the photon orbits in the prograde direction, for a negative value in the retrograde direction.

    Returns
    -------
    float or ndarray
        r_ph / r_G: 3 for a* = 0, 1 for a* = 1, 4 for a* = −1.

    References
    ----------
    [Abramowicz2013] TeX label photon-radius-kerr (the sign correction is in the note field).

    Validity
    --------
    Kerr metric, equatorial plane. This is the radius of the orbit, not the edge of the shadow.

    Verification
    ------------
    tests/unit/test_black_holes.py: limiting values and the cubic y³ − 3y + 2a* = 0.
    """
    a = _spin(spin)
    return 2.0 * (1.0 + np.cos((2.0 / 3.0) * np.arccos(-a)))


# --------------------------------------------------------------------------------------
# Hawking thermodynamics
# --------------------------------------------------------------------------------------


@source(
    "Abramowicz2013",
    where="§1 (Introduction), Hawking temperature footnote: T_H = ħc³/(8π G M k_B)",
    status="read",
)
@source(
    "Page2005",
    where="§1; T = κ/(2π) and TeX label eq:3 (Schwarzschild: κ = 1/(4M), in Planck units)",
    status="read",
)
def hawking_temperature_k(mass_kg: ArrayLike) -> Array:
    """Compute the Hawking temperature of a non-rotating black hole: T_H = ħ c³ / (8π G M k_B).

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].

    Returns
    -------
    float or ndarray
        T_H [K].

    References
    ----------
    [Abramowicz2013] §1 footnote (explicit SI form); [Page2005] T = κ/(2π), κ = 1/(4M).

    Validity
    --------
    Semiclassical limit: the mass is much larger than the Planck mass. Absorption of
    background radiation and the back-reaction of the mass loss are neglected.

    Verification
    ------------
    tests/unit/test_black_holes.py: ~1e-8 K for a stellar-mass black hole (source footnote);
    independent computation with κ/(2π); T·M is constant.
    """
    m = _positive(mass_kg, "mass")
    return _HBAR * _C**3 / (8.0 * np.pi * _G * m * BOLTZMANN.value)


@source(
    "Page2005",
    where="TeX label eq:2, S_bh = S_BH ≡ A/4 (Planck units); A = 16π M² (Schwarzschild, paragraph after eq:3)",
    status="read",
    note="The conversion from Planck units to SI (S/k_B = A c³ / (4 G ħ)) was done by this project; it is not written in the source.",
)
def bekenstein_hawking_entropy(mass_kg: ArrayLike) -> Array:
    """Compute the Bekenstein-Hawking entropy of a non-rotating black hole in units of the Boltzmann constant.

    S / k_B = A c³ / (4 G ħ) = 4π G M² / (ħ c),  A = 16π G² M² / c⁴

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].

    Returns
    -------
    float or ndarray
        S / k_B (dimensionless). ~1e77 for one solar mass.

    References
    ----------
    [Page2005] TeX label eq:2 (S = A/4) and the Schwarzschild area A = 16π M².

    Verification
    ------------
    tests/unit/test_black_holes.py: independent computation with A/4 (from the horizon area); M² scaling.
    """
    m = _positive(mass_kg, "mass")
    area = 16.0 * np.pi * (_G * m / _C**2) ** 2
    return area * _C**3 / (4.0 * _G * _HBAR)


@source(
    "Page2005",
    where="TeX label eq:15",
    status="read",
    note="−dM/dt = 1e-5 M⁻² (8.1830 n_1/2 + 3.3638 n_1 + 0.3836 n_2), in Planck units; n_s is the number of massless "
    "spin-s species with two polarizations. The conversion to SI (factor ħ c⁴/G²) is done in this project",
)
def hawking_mass_loss_rate_kg_s(
    mass_kg: ArrayLike, *, spin_half: float = 0.0, spin_one: float = 1.0, spin_two: float = 1.0
) -> Array:
    """Compute the mass loss rate of a non-rotating black hole through Hawking radiation.

    −dM/dt = α ħ c⁴ / (G² M²),  α = 1e-5 (8.1830 n_½ + 3.3638 n_1 + 0.3836 n_2)

    The coefficients come from numerical greybody calculations. The default species are the
    photon (n_1 = 1) and the graviton (n_2 = 1).

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].
    spin_half, spin_one, spin_two : float
        Numbers of massless spin-½, spin-1 and spin-2 species with two polarizations.

    Returns
    -------
    float or ndarray
        Mass loss rate −dM/dt [kg s⁻¹], positive.

    References
    ----------
    [Page2005] eq:15.

    Validity
    --------
    Massless species only, and a non-rotating, uncharged black hole. A massive species
    becomes active when the Hawking temperature approaches the rest-mass energy of that
    species; for electrons this is below about 1e13-1e14 kg (order-of-magnitude
    estimate, k T_H ~ m_e c²). The mass below which neutrinos count as "massless"
    depends on their masses; they are not counted by default.

    Verification
    ------------
    tests/unit/test_black_holes.py: 1.62 times the blackbody estimate for photons only; M⁻² scaling.
    """
    m = _positive(mass_kg, "mass")
    counts = np.array([spin_half, spin_one, spin_two], dtype=float)
    if np.any(~(counts >= 0.0)) or not np.any(counts > 0.0):
        raise ValueError("the numbers of emitted species must not be negative, and at least one must be positive")
    alpha = 1e-5 * (8.1830 * spin_half + 3.3638 * spin_one + 0.3836 * spin_two)
    return alpha * _HBAR * _C**4 / (_G**2 * m**2)


@source(
    "Page2005",
    where="TeX label eq:15",
    status="read",
    note="integral of eq:15 over M (derivation in this project): t = G² M³ / (3 α ħ c⁴); for α with fixed species numbers",
)
def hawking_evaporation_time_s(
    mass_kg: ArrayLike, *, spin_half: float = 0.0, spin_one: float = 1.0, spin_two: float = 1.0
) -> Array:
    """Compute the complete evaporation time through Hawking radiation with greybody coefficients.

    t = G² M³ / (3 α ħ c⁴),  α = 1e-5 (8.1830 n_½ + 3.3638 n_1 + 0.3836 n_2)

    Parameters
    ----------
    mass_kg : float or ndarray
        Initial mass [kg].
    spin_half, spin_one, spin_two : float
        Numbers of massless species; the default is photon and graviton.

    Returns
    -------
    float or ndarray
        Evaporation time [s]. About 1.2e67 yr for one solar mass with the default species.

    References
    ----------
    [Page2005] eq:15.

    Validity
    --------
    The species numbers are held fixed during the evaporation. In reality the
    temperature rises as the mass decreases and massive species become active; for
    small initial masses (below about 1e14 kg) the true time is therefore shorter
    than the value here, and the result is only an upper limit.

    Verification
    ------------
    tests/unit/test_black_holes.py: numerical integral of ``hawking_mass_loss_rate_kg_s``; M³ scaling.
    """
    m = _positive(mass_kg, "mass")
    counts = np.array([spin_half, spin_one, spin_two], dtype=float)
    if np.any(~(counts >= 0.0)) or not np.any(counts > 0.0):
        raise ValueError("the numbers of emitted species must not be negative, and at least one must be positive")
    alpha = 1e-5 * (8.1830 * spin_half + 3.3638 * spin_one + 0.3836 * spin_two)
    return _G**2 * m**3 / (3.0 * alpha * _HBAR * _C**4)


@source(
    "Page2005",
    where="TeX label eq:3 (A, κ) and §1 (T = κ/2π)",
    status="read",
    note="The blackbody assumption L = σ T_H⁴ A and the integral t = 5120π G² M³ / (ħ c⁴) are this project's derivation; "
    "the source gives only T_H and A. The eq:15 greybody coefficient of the source is 1.62 times this estimate for photons only",
)
def hawking_evaporation_time_blackbody_s(mass_kg: ArrayLike) -> Array:
    """Compute the textbook blackbody estimate of the evaporation time.

    L = σ_SB T_H⁴ A = ħ c⁶ / (15360π G² M²),  t = 5120π G² M³ / (ħ c⁴)

    Returns
    -------
    float or ndarray
        Time [s]. About 2.1e67 yr for one solar mass.

    References
    ----------
    [Page2005] eq:3, §1.

    Validity
    --------
    Photon only, greybody factor 1. This assumption underestimates the emission rate by a
    factor of 1.62 even for photons only and so makes the time longer than it is;
    it is meant for comparison and teaching. Use ``hawking_evaporation_time_s`` for calculations.

    Verification
    ------------
    tests/unit/test_black_holes.py: independent integral with L = σ T⁴ A and M³ scaling.
    """
    m = _positive(mass_kg, "mass")
    return 5120.0 * np.pi * _G**2 * m**3 / (_HBAR * _C**4)


# --------------------------------------------------------------------------------------
# Binary black holes
# --------------------------------------------------------------------------------------


@source("Sathyaprakash2009", where="TeX label eqn:chirpmassdef, 𝓜 = μ^{3/5} M^{2/5} = ν^{3/5} M", status="read")
def chirp_mass_kg(mass1_kg: ArrayLike, mass2_kg: ArrayLike) -> Array:
    """Compute the chirp mass of a binary system.

    𝓜 = (m1 m2)^{3/5} / (m1 + m2)^{1/5}

    Parameters
    ----------
    mass1_kg, mass2_kg : float or ndarray
        Component masses [kg], positive.

    Returns
    -------
    float or ndarray
        Chirp mass [kg]; m·2^{-1/5} for equal masses m.

    References
    ----------
    [Sathyaprakash2009] TeX label eqn:chirpmassdef.

    Verification
    ------------
    tests/unit/test_black_holes.py: agreement with the μ^{3/5}M^{2/5} and ν^{3/5}M forms in the source;
    tests/property/test_black_holes_property.py: symmetry and upper bound.
    """
    m1 = _positive(mass1_kg, "mass 1")
    m2 = _positive(mass2_kg, "mass 2")
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


@source(
    "Sathyaprakash2009",
    where="TeX label eqn:circularPM (Ṗ_b, attributed to Peters-Mathews 1963), f_gw = 2/P_b",
    status="read",
    note="The source uses units G = c = 1. f_gw = 2 f_orb (text after the paragraph 'Waveform polarizations.': the radiation is emitted at twice the orbital "
    "frequency); the conversion ḟ = −2 Ṗ_b / P_b² was done by this project and checked in the test against the Ṗ_b equation.",
)
def gw_frequency_derivative_hz_s(chirp_mass_kg: ArrayLike, frequency_hz: ArrayLike) -> Array:
    """Compute the rate at which the gravitational wave frequency of a circular binary increases with time.

    ḟ = (96/5) π^{8/3} (G 𝓜 / c³)^{5/3} f^{11/3}

    Parameters
    ----------
    chirp_mass_kg : float or ndarray
        Chirp mass 𝓜 [kg].
    frequency_hz : float or ndarray
        Gravitational wave frequency f [Hz] (twice the orbital frequency).

    Returns
    -------
    float or ndarray
        df/dt [Hz s^-1].

    References
    ----------
    [Sathyaprakash2009] TeX label eqn:circularPM.

    Validity
    --------
    Circular orbit, quadrupole radiation at Newtonian order; higher-order PN corrections
    matter when approaching the ISCO.

    Verification
    ------------
    tests/unit/test_black_holes.py: independent derivation from the Ṗ_b equation; 𝓜 and f scaling.
    """
    mc = _positive(chirp_mass_kg, "chirp mass")
    f = _positive(frequency_hz, "frequency")
    return 96.0 / 5.0 * np.pi ** (8.0 / 3.0) * (_G * mc / _C**3) ** (5.0 / 3.0) * f ** (11.0 / 3.0)


@source(
    "Sathyaprakash2009",
    where="§'Coalescing binaries' paragraph, Newtonian-order chirp time τ0 (equation without a label)",
    status="read",
    note="τ0 = 5/(256 π ν f_L) (π M f_L)^{-5/3} (G = c = 1) in the source was converted to this form with 𝓜^{5/3} = ν M^{5/3}. "
    "The text of Peters1964 itself was not read.",
)
def time_to_merger_s(chirp_mass_kg: ArrayLike, frequency_hz: ArrayLike) -> Array:
    """Compute the time remaining until merger of a circular binary at wave frequency f.

    τ = (5/256) (π f)^{−8/3} (G 𝓜 / c³)^{−5/3}

    Parameters
    ----------
    chirp_mass_kg : float or ndarray
        Chirp mass 𝓜 [kg].
    frequency_hz : float or ndarray
        Gravitational wave frequency f [Hz].

    Returns
    -------
    float or ndarray
        Remaining time [s]: until the moment at which the frequency formally diverges.

    References
    ----------
    [Sathyaprakash2009] Newtonian-order τ0 (equation without a label, 'Coalescing binaries').

    Validity
    --------
    Newtonian order, circular orbit; for systems of 10 M☉ and above, PN corrections change
    the time at frequencies close to the ISCO.

    Verification
    ------------
    tests/unit/test_black_holes.py: agreement with the inverse integral of ḟ and with the ν form of the source.
    """
    mc = _positive(chirp_mass_kg, "chirp mass")
    f = _positive(frequency_hz, "frequency")
    return 5.0 / 256.0 * (np.pi * f) ** (-8.0 / 3.0) * (_G * mc / _C**3) ** (-5.0 / 3.0)


@source(
    "Sathyaprakash2009",
    where="§'Coalescing binaries' paragraph, τ0 (equation without a label); its inverse was taken by this project",
    status="read",
    note="The τ0 equation was read; f(τ) is its algebraic inverse and is checked in the test by a round trip.",
)
def gw_frequency_from_time_to_merger_hz(chirp_mass_kg: ArrayLike, remaining_time_s: ArrayLike) -> Array:
    """Compute the wave frequency when the time remaining until merger is τ.

    f = (1/π) (5 / 256 τ)^{3/8} (G 𝓜 / c³)^{−5/8}

    Parameters
    ----------
    chirp_mass_kg : float or ndarray
        Chirp mass 𝓜 [kg].
    remaining_time_s : float or ndarray
        Time remaining until merger [s], positive.

    Returns
    -------
    float or ndarray
        Gravitational wave frequency [Hz].

    References
    ----------
    [Sathyaprakash2009] inverse of the Newtonian-order τ0.

    Validity
    --------
    Newtonian order, circular orbit; values above the ISCO frequency are not physical.

    Verification
    ------------
    tests/unit/test_black_holes.py: round trip with time_to_merger_s;
    tests/property/test_black_holes_property.py: f increases as τ decreases.
    """
    mc = _positive(chirp_mass_kg, "chirp mass")
    tau = _positive(remaining_time_s, "remaining time")
    return (5.0 / (256.0 * tau)) ** (3.0 / 8.0) * (_G * mc / _C**3) ** (-5.0 / 8.0) / np.pi


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies' (TeX label schwarzschild-frequencies), Ω_K² = GM/r³, x_ms = 6",
    status="read",
)
@source(
    "Sathyaprakash2009",
    where="text after the paragraph 'Waveform polarizations.': at lowest order the radiation is emitted at twice the orbital frequency",
    status="read",
    note="For the orbital frequency, the test-particle ISCO (6 G M/c²) is applied to the total mass; for an equal-mass binary "
    "this is a heuristic limit, and the source does not give a binary ISCO.",
)
def gw_frequency_at_isco_hz(total_mass_kg: ArrayLike) -> Array:
    """Compute the gravitational wave frequency at the ISCO of a binary (test-particle limit).

    f_gw = 2 f_orb = c³ / (6^{3/2} π G M)

    Parameters
    ----------
    total_mass_kg : float or ndarray
        Total mass M of the binary [kg].

    Returns
    -------
    float or ndarray
        f_gw [Hz]; about 4.4 kHz for 1 M☉, inversely proportional to M.

    References
    ----------
    [Abramowicz2013] Ω_K² = GM/r³ and x_ms = 6; [Sathyaprakash2009] f_gw = 2 f_orb.

    Validity
    --------
    Test-particle (ν → 0) limit; for an equal-mass binary the true merger frequency differs from this.

    Verification
    ------------
    tests/unit/test_black_holes.py: twice isco_orbital_frequency_hz; 4.4 kHz (M☉/M).
    """
    return 2.0 * isco_orbital_frequency_hz(total_mass_kg)
