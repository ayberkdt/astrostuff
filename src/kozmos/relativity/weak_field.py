"""Weak-field and post-Newtonian effects of general relativity in SI units.

Will [Will2014] writes the post-Newtonian results with G = c = 1 and the PPN parameters gamma (space curvature
per unit mass) and beta (nonlinearity of the superposition law); both are 1 in general relativity. Here the
mass M enters through G M / c^2 (a length) or G M / c^3 (a time), restored by dimensional analysis; the
``note`` of each record says so.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.constants import GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT

Array = NDArray[np.float64]

_G = GRAVITATIONAL_CONSTANT.value
_C = SPEED_OF_LIGHT.value


def _positive(value: ArrayLike, name: str) -> Array:
    x = np.asarray(value, dtype=float)
    if np.any(~(x > 0.0)):
        raise ValueError(f"{name} must be positive")
    return x


@source(
    "Will2014",
    where="subsection 'The perihelion shift of Mercury', Eq. E33 (label at TeX line 3430)",
    status="read",
    note="Eq. E33 with the mass-ratio term (alpha_1, alpha_2, alpha_3, zeta_2 times eta) dropped, as the text does "
    "for Mercury; m = G M / c^2; the J_2 term (J_2 R^2 / (2 m p)) is kept and multiplied out: 3 pi J_2 R^2 / p^2",
)
def perihelion_advance_per_orbit_rad(
    total_mass_kg: ArrayLike,
    semi_major_axis_m: ArrayLike,
    eccentricity: ArrayLike,
    gamma_ppn: float = 1.0,
    beta_ppn: float = 1.0,
    quadrupole_j2: float = 0.0,
    body_radius_m: float = 0.0,
) -> Array:
    """Compute the advance of the perihelion per orbit.

    Delta_omega = (6 pi G M / (c^2 p)) (2 + 2 gamma - beta) / 3 + 3 pi J_2 R^2 / p^2, with the semi-latus
    rectum p = a (1 - e^2).

    Parameters
    ----------
    total_mass_kg : float or ndarray
        Total mass m_1 + m_2 of the two bodies [kg].
    semi_major_axis_m : float or ndarray
        Semi-major axis a [m].
    eccentricity : float or ndarray
        Eccentricity e, 0 <= e < 1.
    gamma_ppn, beta_ppn : float, optional
        PPN parameters gamma and beta (1 in general relativity).
    quadrupole_j2 : float, optional
        Dimensionless quadrupole moment J_2 of the central body (0 by default).
    body_radius_m : float, optional
        Mean radius R of the central body for the J_2 term [m].

    Returns
    -------
    float or ndarray
        Advance per orbit [rad], positive in the direction of the orbital motion.

    References
    ----------
    [Will2014] Eq. E33 (with the numerical value for Mercury in Eq. E34).

    Validity
    --------
    Post-Newtonian order: relative corrections of order G M / (c^2 p). The mass-ratio term (zero in any fully
    conservative theory of gravity, and negligible for Mercury where eta is about 2e-7, as the source says)
    is dropped, so for comparable masses the result needs that term. Only the conservative lowest order is
    included; the Lense-Thirring contribution of a rotating central body is not.

    Verification
    ------------
    tests/unit/test_weak_field.py: Mercury gives 42.98 arcsec per century (Eq. E34) with orbital elements of
    the JPL table; agrees with 6 pi G M / (c^2 p); gamma = beta = 1 gives the general-relativity factor 1;
    the J_2 term of Eq. E34 is reproduced (coefficient 2.955e-4, printed as 3e-4) and is linear in J_2.
    """
    mass = _positive(total_mass_kg, "total_mass_kg")
    a = _positive(semi_major_axis_m, "semi_major_axis_m")
    e = np.asarray(eccentricity, dtype=float)
    if np.any(~((e >= 0.0) & (e < 1.0))):
        raise ValueError("eccentricity must satisfy 0 <= e < 1")
    p = a * (1.0 - e**2)
    m_length = _G * mass / _C**2
    post_newtonian = 6.0 * np.pi * m_length / p * (2.0 + 2.0 * gamma_ppn - beta_ppn) / 3.0
    quadrupole = 3.0 * np.pi * quadrupole_j2 * body_radius_m**2 / p**2
    return post_newtonian + quadrupole


@source(
    "Will2014",
    where="subsection 'The perihelion shift of Mercury', Eq. E33 and E34 (labels at TeX lines 3430, 3475)",
    status="read",
    note="derivation in this project: Eq. E33 per orbit divided by the Keplerian period P = 2 pi sqrt(a^3 / (G M)); "
    "Eq. E34 is the same rate for Mercury in arcsec per century",
)
def perihelion_advance_rate_rad_s(
    total_mass_kg: ArrayLike,
    semi_major_axis_m: ArrayLike,
    eccentricity: ArrayLike,
    gamma_ppn: float = 1.0,
    beta_ppn: float = 1.0,
    quadrupole_j2: float = 0.0,
    body_radius_m: float = 0.0,
) -> Array:
    """Compute the mean rate of the perihelion advance: the advance per orbit divided by the Keplerian period.

    Parameters
    ----------
    total_mass_kg : float or ndarray
        Total mass [kg].
    semi_major_axis_m : float or ndarray
        Semi-major axis [m].
    eccentricity : float or ndarray
        Eccentricity, 0 <= e < 1.
    gamma_ppn, beta_ppn : float, optional
        PPN parameters (1 in general relativity).
    quadrupole_j2 : float, optional
        Dimensionless quadrupole moment J_2 of the central body.
    body_radius_m : float, optional
        Radius of the central body [m].

    Returns
    -------
    float or ndarray
        Rate of advance [rad s^-1].

    References
    ----------
    [Will2014] Eq. E33, Eq. E34.

    Validity
    --------
    The Newtonian period 2 pi sqrt(a^3 / (G M)) is used; its relative error against the true period is of the
    post-Newtonian order, far below the other approximations of the formula.

    Verification
    ------------
    tests/unit/test_weak_field.py: 42.98 arcsec per century for Mercury (Eq. E34).
    """
    mass = _positive(total_mass_kg, "total_mass_kg")
    a = _positive(semi_major_axis_m, "semi_major_axis_m")
    period = 2.0 * np.pi * np.sqrt(a**3 / (_G * mass))
    advance = perihelion_advance_per_orbit_rad(mass, a, eccentricity, gamma_ppn, beta_ppn, quadrupole_j2, body_radius_m)
    return advance / period


@source(
    "Will2014",
    where="subsubsection 'The deflection of light', Eq. E28 (label at TeX line 3014)",
    status="read",
    note="Eq. E28: delta theta = (1/2)(1 + gamma)(4 M / d)(1 + cos Phi)/2 with M = G M_kg / c^2 and Phi the "
    "angle between the Earth-Sun line and the incoming direction of the photon",
)
def light_deflection_angle_rad(
    mass_kg: ArrayLike,
    closest_approach_m: ArrayLike,
    gamma_ppn: float = 1.0,
    incoming_angle_rad: ArrayLike = 0.0,
) -> Array:
    """Compute the deflection of a light ray passing a mass: (1 + gamma)/2 * 4 G M / (c^2 d) * (1 + cos Phi)/2.

    Parameters
    ----------
    mass_kg : float or ndarray
        Mass of the deflecting body [kg].
    closest_approach_m : float or ndarray
        Distance d of closest approach of the ray to the body [m].
    gamma_ppn : float, optional
        PPN parameter gamma (1 in general relativity; the light deflection coefficient is (1 + gamma)/2).
    incoming_angle_rad : float or ndarray, optional
        Angle Phi between the line from the observer to the body and the incoming direction of the photon [rad];
        Phi = 0 gives the full deflection of a ray from infinity to infinity, the value (1 + cos Phi)/2 accounts
        for an observer at a finite distance.

    Returns
    -------
    float or ndarray
        Deflection angle [rad].

    References
    ----------
    [Will2014] Eq. E28, Eq. E29 (grazing ray at the Sun, 1.7505 arcsec for gamma = 1).

    Validity
    --------
    Weak field, d >> G M / c^2, first order in G M / (c^2 d). The next-order term is not in the source read
    for this function and is not included; tests/unit/test_geodesics.py measures the size of the departure by
    integration of null geodesics.

    Verification
    ------------
    tests/unit/test_weak_field.py: solar limb gives 1.7505 arcsec of Eq. E29 for the nominal solar radius;
    tests/unit/test_geodesics.py: null geodesic integration agrees with 4 G M / (c^2 b) to next order.
    """
    mass = _positive(mass_kg, "mass_kg")
    d = _positive(closest_approach_m, "closest_approach_m")
    factor = 0.5 * (1.0 + gamma_ppn) * 0.5 * (1.0 + np.cos(np.asarray(incoming_angle_rad, dtype=float)))
    return factor * 4.0 * _G * mass / (_C**2 * d)


@source(
    "Will2014",
    where="subsubsection 'The time delay of light', Eq. E31 (label at TeX line 3193)",
    status="read",
    note="Eq. E31: delta t = 2(1 + gamma) M ln[(r_E + x_E.n)(r_e - x_e.n)/d^2] with M = G M_kg / c^3 (a time); the "
    "text does not define n; here n is the unit vector from the source to the Earth, the choice that gives "
    "ln(4 r_E r_e / d^2) at superior conjunction (derivation in this project). Identity, derived here from "
    "Eq. E31: d^2 = (r_E - x_E.n)(r_E + x_E.n), so the argument is (r_e - x_e.n)/(r_E - x_E.n); with R = "
    "|x_E - x_e|, n = (x_E - x_e)/R and 2 x_E.x_e = r_E^2 + r_e^2 - R^2 one finds r_e - x_e.n = "
    "(R + r_e - r_E)(R + r_e + r_E)/(2R) and r_E - x_E.n = (r_E + r_e - R)(R + r_e - r_E)/(2R); the common "
    "factor cancels and delta t = 2(1 + gamma) M ln[(r_E + r_e + R)/(r_E + r_e - R)], the form evaluated here",
)
def shapiro_round_trip_delay_s(
    mass_kg: ArrayLike,
    earth_position_m: ArrayLike,
    source_position_m: ArrayLike,
    gamma_ppn: float = 1.0,
) -> Array:
    """Compute the extra round-trip travel time of a radar signal passing a mass (Shapiro delay).

    delta t = 2 (1 + gamma) G M / c^3 ln[(r_E + x_E . n)(r_e - x_e . n) / d^2], where n is the unit vector from
    the reflector (source) to the Earth, d the distance of the straight line between them from the mass. The code
    evaluates the algebraically identical form ln[(r_E + r_e + R) / (r_E + r_e - R)] with R = |x_E - x_e| (see
    the note of the source record for the two-line derivation), which has no 0/0 for collinear same-side
    geometries. The difference r_E + r_e - R is not formed by subtraction (behind the mass it would lose
    1e-16 (r_E + r_e) / (r_E + r_e - R), 1.4e-4 of the delay for a ray 3e4 m from the centre of the Sun): it is
    2 (r_E r_e + x_E . x_e) / (r_E + r_e + R), with the bracket taken as |x_E x x_e|^2 / (r_E r_e - x_E . x_e)
    when the two points lie on opposite sides. Measured against 50-digit arithmetic: 2e-16 for impact
    parameters from 7e8 m down to 3e4 m.

    Parameters
    ----------
    mass_kg : float or ndarray
        Mass of the central body [kg].
    earth_position_m : ndarray, shape (..., 3)
        Position of the Earth (the transmitter) relative to the mass [m].
    source_position_m : ndarray, shape (..., 3)
        Position of the reflector (planet or spacecraft) relative to the mass [m].
    gamma_ppn : float, optional
        PPN parameter gamma (1 in general relativity).

    Returns
    -------
    float or ndarray
        Additional round-trip delay [s] relative to the Newtonian (flat-space) light travel time.

    References
    ----------
    [Will2014] Eq. E31 (rounded numerical form: Eq. E32).

    Validity
    --------
    Weak field; both end points outside the body and the straight segment not passing through it. ValueError is
    raised when r_E + r_e = R (the segment contains the mass) or when the two positions coincide (R = 0). The
    segment may be collinear with the mass on the same side (d = 0 there is harmless; the delay is finite).
    The formula is first order in G M / c^2 and is the coordinate-time delay; no clock-rate correction is
    applied to turn it into a delay measured on Earth. The definition of n is derived in this project because
    the text does not define it.

    Verification
    ------------
    tests/unit/test_weak_field.py: superior conjunction reproduces ln(4 r_E r_e / d^2); Eq. E32 coefficients
    (240 microseconds and 20 microseconds per e-fold, both rounded in the source) are matched within the
    rounding of the printed numbers.
    """
    mass = _positive(mass_kg, "mass_kg")
    x_earth = np.asarray(earth_position_m, dtype=float)
    x_source = np.asarray(source_position_m, dtype=float)
    if x_earth.shape[-1:] != (3,) or x_source.shape[-1:] != (3,):
        raise ValueError("positions must have a last axis of length 3")
    chord = np.linalg.norm(x_earth - x_source, axis=-1)
    path_sum = np.linalg.norm(x_earth, axis=-1) + np.linalg.norm(x_source, axis=-1)
    if np.any(~(chord > 0.0)):
        raise ValueError("the two positions coincide")
    # r_E + r_e - R is b^2 R / (2 r_E r_e) for a small impact parameter b; on a line through the mass it is a rounding
    # residue of three norms, zero on one platform and one unit in the last place on another. Anything within eight
    # rounding errors of r_E + r_e counts as through the mass (b below about 4e-8 of the distances).
    if np.any(~(path_sum - chord > 8.0 * np.finfo(float).eps * path_sum)):
        raise ValueError("the straight line between the end points passes through the mass")
    # s - R = (s^2 - R^2) / (s + R) = 2 (r_E r_e + x_E . x_e) / (s + R). Behind the mass the bracket cancels in
    # its turn; there r_E r_e + x_E . x_e = |x_E x x_e|^2 / (r_E r_e - x_E . x_e), whose denominator does not.
    # Formed as s - R the difference lost 1e-16 s / (s - R): 1.4e-4 of the delay for a ray 3e4 m from the centre.
    r_earth = np.linalg.norm(x_earth, axis=-1)
    r_source = np.linalg.norm(x_source, axis=-1)
    dot = np.sum(x_earth * x_source, axis=-1)
    cross = np.cross(x_earth, x_source)
    with np.errstate(divide="ignore", invalid="ignore"):
        behind = np.sum(cross * cross, axis=-1) / (r_earth * r_source - dot)
    bracket = np.where(dot < 0.0, behind, r_earth * r_source + dot)
    difference = 2.0 * bracket / (path_sum + chord)
    argument = 1.0 + 2.0 * chord / difference  # (s + R) / (s - R)
    return 2.0 * (1.0 + gamma_ppn) * _G * mass / _C**3 * np.log(argument)


@source(
    "Carroll1997",
    where="Section 7, Eq. 7.28 (TeX line 9208) and Eq. 7.58 (TeX line 9688)",
    status="read",
    note="derivation in this project: Eq. 7.28 gives g_00 = -(1 + 2 Phi) in the weak-field limit and, for a "
    "static clock, d tau / d t = sqrt(-g_00) (the same step as Eq. 7.58 from Eq. 7.45), so "
    "d tau / d t = sqrt(1 + 2 Phi / c^2)",
)
def static_clock_ratio(potential_m2_s2: ArrayLike) -> Array:
    """Compute the rate of a static clock in a weak static gravitational field: d tau / d t = sqrt(1 + 2 Phi / c^2).

    Parameters
    ----------
    potential_m2_s2 : float or ndarray
        Newtonian potential Phi at the clock, with Phi -> 0 at infinity (negative near a mass) [m^2 s^-2].

    Returns
    -------
    float or ndarray
        Ratio of proper time of the clock to the coordinate time (the time of a clock at infinity), smaller than
        1 deep in a potential well.

    References
    ----------
    [Carroll1997] Eq. 7.28, Eq. 7.58.

    Validity
    --------
    |Phi| << c^2 and a static clock. The result is real only for Phi > -c^2 / 2. In the weak field the metric
    function g_00 = -(1 + 2 Phi / c^2) is accurate to first order in Phi / c^2 only, so the second-order term
    of the ratio is not meaningful; compare schwarzschild_clock_ratio for the exact vacuum value.

    Verification
    ------------
    tests/unit/test_weak_field.py: identical to schwarzschild_clock_ratio for Phi = -G M / r (the same
    function of the potential); the ratio of two clocks reproduces the frequency shift 1 + Phi_1 - Phi_2 of
    Eq. 7.62 to the neglected second order (measured 2.3e-12 at the solar surface).
    """
    arg = 1.0 + 2.0 * np.asarray(potential_m2_s2, dtype=float) / _C**2
    if np.any(~(arg > 0.0)):
        raise ValueError("the potential must satisfy Phi > -c^2 / 2")
    return np.sqrt(arg)


@source(
    "Carroll1997",
    where="Section 7, Eq. 7.58 (TeX line 9688)",
    status="read",
    note="d tau_i / d t = (1 - 2GM/r_i)^{1/2}, G = c = 1 restored: 1 - 2 G M / (r c^2)",
)
def schwarzschild_clock_ratio(radius_m: ArrayLike, mass_kg: ArrayLike) -> Array:
    """Compute the rate of a static clock in the Schwarzschild geometry: d tau / d t = sqrt(1 - 2 G M / (r c^2)).

    Parameters
    ----------
    radius_m : float or ndarray
        Schwarzschild radial coordinate of the clock [m]; larger than 2 G M / c^2.
    mass_kg : float or ndarray
        Mass of the central body [kg].

    Returns
    -------
    float or ndarray
        Ratio of the proper time of the static clock to the time of a clock at infinity.

    References
    ----------
    [Carroll1997] Eq. 7.58.

    Validity
    --------
    Exact for the vacuum exterior of a spherical mass, static clock (not orbiting). r > 2 G M / c^2 is
    enforced.

    Verification
    ------------
    tests/unit/test_weak_field.py: closed-form series at the solar surface; ratio of two radii equals the exact
    frequency shift of Eq. 7.61; equals static_clock_ratio for Phi = -G M / r; the horizon is rejected.
    """
    r = _positive(radius_m, "radius_m")
    mass = _positive(mass_kg, "mass_kg")
    arg = 1.0 - 2.0 * _G * mass / (r * _C**2)
    if np.any(~(arg > 0.0)):
        raise ValueError("radius_m must be larger than the Schwarzschild radius")
    return np.sqrt(arg)


@source(
    "Will2014",
    where="subsubsection 'Geodetic precession', Eq. E41 (label at TeX line 3939)",
    status="read",
    note="Eq. E41: Omega_G = (gamma + 1/2) v x grad U with G = c = 1; SI: divide by c^2 with v and grad U in "
    "m s^-1 and m s^-2 (U = G M / r > 0 is the Newtonian potential in the sign convention of the text)",
)
def geodetic_precession_vector_rad_s(
    velocity_m_s: ArrayLike, potential_gradient_m_s2: ArrayLike, gamma_ppn: float = 1.0
) -> Array:
    """Compute the geodetic precession vector of a gyroscope: Omega_G = (gamma + 1/2) v x grad(U) / c^2.

    Parameters
    ----------
    velocity_m_s : ndarray, shape (..., 3)
        Velocity of the gyroscope [m s^-1].
    potential_gradient_m_s2 : ndarray, shape (..., 3)
        Gradient of the Newtonian potential U = G M / r (positive, pointing towards the mass; the sign convention
        of the source) at the gyroscope [m s^-2].
    gamma_ppn : float, optional
        PPN parameter gamma (1 in general relativity).

    Returns
    -------
    ndarray, shape (..., 3)
        Precession vector of the spin, dS/dtau = Omega_G x S [rad s^-1].

    References
    ----------
    [Will2014] Eq. E41.

    Validity
    --------
    Weak field and slow motion (v << c); only the mass monopole of the source; the frame-dragging
    (Lense-Thirring) precession, caused by the rotation of the source, is not included and its formula was not
    printed in the source (only its size for Gravity Probe B, 38 milliarcsec per year, was).

    Verification
    ------------
    tests/unit/test_weak_field.py: Gravity Probe B (orbit altitude 642 km) gives about 6.63 arcsec per year
    (the source quotes 6630 milliarcsec per year); the Earth-Moon system about 2 arcsec per century (the text:
    'about 2'); direction perpendicular to the orbital plane.
    """
    v = np.asarray(velocity_m_s, dtype=float)
    grad = np.asarray(potential_gradient_m_s2, dtype=float)
    if v.shape[-1:] != (3,) or grad.shape[-1:] != (3,):
        raise ValueError("velocity and potential gradient must have a last axis of length 3")
    return (gamma_ppn + 0.5) * np.cross(v, grad) / _C**2


@source(
    "Will2014",
    where="subsubsection 'Geodetic precession', Eq. E41 (label at TeX line 3939)",
    status="read",
    note="derivation in this project from Eq. E41 for a circular orbit: v = sqrt(G M / r), |grad U| = G M / r^2, "
    "v perpendicular to grad U, so |Omega_G| = (gamma + 1/2)(G M)^{3/2} / (c^2 r^{5/2})",
)
def geodetic_precession_circular_rad_s(mass_kg: ArrayLike, radius_m: ArrayLike, gamma_ppn: float = 1.0) -> Array:
    """Compute the geodetic precession rate of a gyroscope on a circular orbit: (gamma + 1/2)(G M)^(3/2) / (c^2 r^(5/2)).

    Parameters
    ----------
    mass_kg : float or ndarray
        Mass of the central body [kg].
    radius_m : float or ndarray
        Orbital radius [m].
    gamma_ppn : float, optional
        PPN parameter gamma (1 in general relativity).

    Returns
    -------
    float or ndarray
        Precession rate [rad s^-1].

    References
    ----------
    [Will2014] Eq. E41.

    Validity
    --------
    Circular Keplerian orbit, weak field, point-mass potential (no oblateness, no frame dragging).

    Verification
    ------------
    tests/unit/test_weak_field.py: equals the magnitude from geodetic_precession_vector_rad_s for a circular
    orbit; Gravity Probe B about 6.63 arcsec per year; Earth-Moon system about 2 arcsec per century.
    """
    mass = _positive(mass_kg, "mass_kg")
    r = _positive(radius_m, "radius_m")
    return (gamma_ppn + 0.5) * (_G * mass) ** 1.5 / (_C**2 * r**2.5)
