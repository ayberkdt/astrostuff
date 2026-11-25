"""Kerr geometry: circular geodesics, epicyclic frequencies, frame dragging and horizon thermodynamics.

Companion of ``kozmos.black_holes.basics``, which holds the ISCO, the horizon, the ergosphere, the photon
orbit and the Schwarzschild thermodynamics. Nothing from there is repeated here.

The equations were read on 26 November 2026 from the TeX sources of Abramowicz and Fragile [Abramowicz2013]
(arXiv:1104.5499), Page [Page2005] (arXiv:hep-th/0409024), Carroll [Carroll1997] (arXiv:gr-qc/9712019),
Visser [Visser2007] (arXiv:0706.0622), Sathyaprakash and Schutz [Sathyaprakash2009] (arXiv:0903.0338) and
Reynolds [Reynolds2021] (arXiv:2011.08948). The circular-orbit formulas of Bardeen, Press and Teukolsky
(specific energy and angular momentum at arbitrary radius) were not available typeset; the specific energy
and angular momentum here are computed from the metric components and the angular velocity that were read
(see ``kerr_circular_orbit_specific_energy``).

Conventions are those of ``basics``: ``spin`` is the dimensionless a_* = a/M = cJ/(G M^2) in [-1, 1]; a
negative value means that the orbit runs against the rotation of the black hole (retrograde), i.e. the
orbit of angular velocity Omega > 0 around a hole of spin -|a_*|. Radii with the suffix ``_rg`` are in units
of r_G = G M / c^2. Angular momenta, frequencies and precession frequencies of an orbit are counted along the
direction of the orbit, so a retrograde nodal precession comes out negative.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.black_holes.basics import _positive, _spin, kerr_photon_orbit_radius_rg
from kozmos.core.constants import BOLTZMANN, GRAVITATIONAL_CONSTANT, PLANCK, SPEED_OF_LIGHT

Array = NDArray[np.float64]

_G = GRAVITATIONAL_CONSTANT.value
_C = SPEED_OF_LIGHT.value
_HBAR = PLANCK.value / (2.0 * np.pi)  # reduced Planck constant, hbar = h / 2 pi

# A radial epicyclic frequency squared that is negative by no more than this fraction of the orbital frequency
# squared is rounding noise at the ISCO and is set to zero (the roundoff measured at the basics ISCO is ~1e-16).
_EPICYCLIC_ROUNDOFF = 1e-12


def _orbit_radius(radius_rg: ArrayLike, spin: Array) -> Array:
    """Check that a circular orbit exists at the radius: r must exceed the equatorial photon orbit."""
    r = _positive(radius_rg, "radius")
    if np.any(r <= kerr_photon_orbit_radius_rg(spin)):
        raise ValueError("radius must be larger than the circular photon orbit: no circular geodesic exists inside it")
    return r


def _circular_angular_momentum(r: Array, a: Array) -> Array:
    """L of a circular orbit, [Pugliese2011] Eq. LPM with the signed spin: (x^2 - 2 a sqrt(x) + a^2) / sqrt(x^2 (x - 3) + 2 a x^{3/2})."""
    radicand = r**2 * (r - 3.0) + 2.0 * a * r**1.5
    if np.any(~(radicand > 0.0)):
        raise ValueError(
            "the radius is too close to the photon orbit: the orbit cannot be resolved in double precision"
        )
    return (r**2 - 2.0 * a * np.sqrt(r) + a**2) / np.sqrt(radicand)


def _omega_rg(r: Array, a: Array) -> Array:
    """Orbital angular velocity in units of c^3 / (G M): 1 / (r^{3/2} + a)."""
    return 1.0 / (r**1.5 + a)


def _time_unit_rate(mass_kg: ArrayLike) -> Array:
    """The rate c^3 / (G M) [s^-1] that converts an angular frequency in geometrized units to SI."""
    return _C**3 / (_G * _positive(mass_kg, "mass"))


def _epicyclic_squared(r: Array, a: Array, vertical: bool) -> Array:
    """Epicyclic frequency squared divided by the orbital frequency squared (dimensionless)."""
    if vertical:
        return 1.0 - 4.0 * a * r**-1.5 + 3.0 * a**2 / r**2
    return 1.0 - 6.0 / r + 8.0 * a * r**-1.5 - 3.0 * a**2 / r**2


def _radial_ratio(r: Array, a: Array) -> Array:
    """Radial epicyclic frequency over orbital frequency; NaN where the orbit is radially unstable."""
    q = _epicyclic_squared(r, a, vertical=False)
    q = np.where((q < 0.0) & (q >= -_EPICYCLIC_ROUNDOFF), 0.0, q)
    return np.sqrt(np.where(q >= 0.0, q, np.nan))


# --------------------------------------------------------------------------------------
# Circular equatorial geodesics
# --------------------------------------------------------------------------------------


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies', Omega_K^2 (TeX label keplerian-velocity-kerr, "
    "line 1095 of article.tex; the label sits on the following empty row, line 1096)",
    status="read",
    note="The source gives Omega_K^2 = GM / (r^{3/2} + a M^{1/2})^2 for the prograde orbit (G = c = 1 in the "
    "rescaling of TeX label rescaling); the retrograde orbit is the same expression with the sign of a reversed, "
    "which is the spin convention of this module. Conversion to SI: Omega = c^3 / (G M) * 1 / (x^{3/2} + a_*).",
)
def kerr_circular_orbit_angular_velocity_rad_s(mass_kg: ArrayLike, radius_rg: ArrayLike, spin: ArrayLike) -> Array:
    """Compute the angular velocity of a circular equatorial geodesic of a Kerr black hole.

    Omega = c^3 / (G M) * 1 / (x^{3/2} + a_*),  x = r / r_G

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].
    radius_rg : float or ndarray
        Boyer-Lindquist radius of the orbit in units of r_G; must exceed the circular photon orbit of the same
        spin.
    spin : float or ndarray
        Dimensionless spin a_* in [-1, 1]; a negative value means a retrograde orbit.

    Returns
    -------
    float or ndarray
        Omega = d(phi)/dt [rad s^-1], measured with respect to the time of an observer at infinity, positive in
        the direction of the orbit.

    References
    ----------
    [Abramowicz2013] TeX label keplerian-velocity-kerr.

    Validity
    --------
    Kerr metric, equatorial plane, test particle on a geodesic. The orbit exists for r larger than the photon
    orbit; it is bound for r above the marginally bound orbit and stable for r above the ISCO. A ValueError is
    raised for r at or below the photon orbit.

    Verification
    ------------
    tests/unit/test_kerr.py: Kepler's law for a_* = 0; the closed form at a_* = 1, r = 1 ... 9; the orbit
    through the ISCO of basics.
    """
    a = _spin(spin)
    r = _orbit_radius(radius_rg, a)
    return _time_unit_rate(mass_kg) * _omega_rg(r, a)


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies', Omega_K^2 (TeX label keplerian-velocity-kerr)",
    status="read",
    note="The conversion f = Omega / (2 pi) is a definition.",
)
def kerr_circular_orbit_frequency_hz(mass_kg: ArrayLike, radius_rg: ArrayLike, spin: ArrayLike) -> Array:
    """Compute the orbital frequency of a circular equatorial geodesic of a Kerr black hole.

    f = Omega / (2 pi) = c^3 / (2 pi G M) * 1 / (x^{3/2} + a_*)

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].
    radius_rg : float or ndarray
        Boyer-Lindquist radius in units of r_G, larger than the circular photon orbit.
    spin : float or ndarray
        Dimensionless spin a_* in [-1, 1]; a negative value means a retrograde orbit.

    Returns
    -------
    float or ndarray
        Orbital frequency [Hz] as measured at infinity.

    References
    ----------
    [Abramowicz2013] TeX label keplerian-velocity-kerr.

    Validity
    --------
    See ``kerr_circular_orbit_angular_velocity_rad_s``.

    Verification
    ------------
    tests/unit/test_kerr.py: equals basics.isco_orbital_frequency_hz at a_* = 0, r = 6; 1/M scaling.
    """
    return kerr_circular_orbit_angular_velocity_rad_s(mass_kg, radius_rg, spin) / (2.0 * np.pi)


@source(
    "Abramowicz2013",
    where="'Subsection: ISCO' (TeX labels circular-geodesic, redshift-factor, omega-covariant) and the Kerr metric "
    "(TeX label kerr-metric), with Omega_K (TeX label keplerian-velocity-kerr)",
    status="read",
    note="Derivation in this project from the read equations, not a transcription: u^mu = A (eta^mu + Omega xi^mu), "
    "E = -p_t / m = -A (g_tt + Omega g_tphi), with A = u^t = L / (g_tphi + Omega g_phiphi) and L from [Pugliese2011] (from "
    "TeX labels constant-motion-energy and omega-covariant, u_t = A (g_tt + Omega g_tphi)), evaluated at theta = pi/2 "
    "with the metric components of kerr-metric and the Omega_K of keplerian-velocity-kerr. The test compares the "
    "result with the energy printed in [Pugliese2011] (the two unlabelled equations after Eq. LPM), with "
    "L/E = l_K of TeX label keplerian-momentum-kerr, with the ISCO energy of basics and with dE = Omega dL.",
)
@source(
    "Pugliese2011",
    where="Eq. LPM (Kerr_A.tex lines 284-288) and the energies E(L) printed after it (lines 299-303, 312-316)",
    status="read",
    note="L enters through the normalisation u^t = L / (g_tphi + Omega g_phiphi); the printed energy is used in "
    "the test only. In the source the prograde orbit is the lower sign (L_-) with a > 0",
)
def kerr_circular_orbit_specific_energy(radius_rg: ArrayLike, spin: ArrayLike) -> Array:
    """Compute the energy per unit rest mass of a circular equatorial geodesic of a Kerr black hole.

    E = -A (g_tt + Omega g_tphi),  A = u^t = L / (g_tphi + Omega g_phiphi)

    with the equatorial Kerr metric in units G = c = M = 1: g_tt = -(1 - 2/x), g_tphi = -2 a_*/x,
    g_phiphi = x^2 + a_*^2 + 2 a_*^2/x, and Omega = 1/(x^{3/2} + a_*). It reduces to (1 - 2/x)/sqrt(1 - 3/x)
    for a_* = 0.

    Parameters
    ----------
    radius_rg : float or ndarray
        Boyer-Lindquist radius x = r / r_G, larger than the circular photon orbit.
    spin : float or ndarray
        Dimensionless spin a_* in [-1, 1]; a negative value means a retrograde orbit.

    Returns
    -------
    float or ndarray
        E / (m c^2), dimensionless. It equals 1 at the marginally bound orbit and diverges at the photon orbit.

    References
    ----------
    [Abramowicz2013] TeX labels circular-geodesic, redshift-factor, omega-covariant, kerr-metric;
    [Pugliese2011] Eq. LPM.

    Validity
    --------
    Kerr metric, equatorial plane, test particle; stable only for r above the ISCO, bound (E < 1) only for r
    above the marginally bound orbit. A ValueError is raised for r at or below the photon orbit. Within a
    relative distance eps of the photon orbit the rounding error is about 1e-16 / eps.

    Verification
    ------------
    tests/unit/test_kerr.py: the energy printed in the second source on a grid of radii and spins, both orbit
    senses; Schwarzschild closed form; equals basics.kerr_isco_specific_energy at the ISCO;
    dE/dr = 0 at the ISCO; E = 1 at the marginally bound orbit; divergence at the photon orbit.
    """
    a = _spin(spin)
    r = _orbit_radius(radius_rg, a)
    omega = _omega_rg(r, a)
    g_tt = -(1.0 - 2.0 / r)
    g_tp = -2.0 * a / r
    g_pp = r**2 + a**2 + 2.0 * a**2 / r
    # u^t = L / (g_tphi + Omega g_phiphi): the normalisation is taken from the closed form of L, which has one
    # cancelling difference near the photon orbit where the direct normalisation has it squared
    return -(g_tt + omega * g_tp) * _circular_angular_momentum(r, a) / (g_tp + omega * g_pp)


@source(
    "Abramowicz2013",
    where="'Subsection: ISCO' (TeX labels circular-geodesic, redshift-factor, omega-covariant) and the Kerr metric "
    "(TeX label kerr-metric), with Omega_K (TeX label keplerian-velocity-kerr)",
    status="read",
    note="Not used for the value any more; kept as the independent check: L / (m c) = u_phi = A (g_tphi + Omega g_phiphi) "
    "(TeX label omega-covariant gives u_t; u_phi follows from u_mu = g_mu_nu u^nu with u^mu = A (eta^mu + Omega "
    "xi^mu) of circular-geodesic). The test evaluates this expression and compares it with the closed form; "
    "the ratio L/E is checked against l_K of TeX label keplerian-momentum-kerr.",
)
@source(
    "Pugliese2011",
    where="Eq. LPM (Kerr_A.tex lines 284-288)",
    status="read",
    note="L_pm / (mu M) = |a^2 pm 2 a sqrt(r) + r^2| / sqrt(r^2 (r - 3) mp 2 a r^{3/2}) with M = 1; the prograde orbit "
    "is the lower sign with a > 0, the retrograde orbit the upper sign, so both are the lower-sign expression "
    "with the signed spin of this module. The numerator is a perfect square plus a positive term for |a| <= 1 "
    "and r above the photon orbit, so the absolute value is dropped",
)
def kerr_circular_orbit_specific_angular_momentum_rg(radius_rg: ArrayLike, spin: ArrayLike) -> Array:
    """Compute the angular momentum per unit rest mass of a circular equatorial geodesic, in units of r_G.

    L / (m c) / r_G = (x^2 - 2 a_* sqrt(x) + a_*^2) / sqrt(x^2 (x - 3) + 2 a_* x^{3/2})

    with the same metric components, Omega and redshift factor A as in ``kerr_circular_orbit_specific_energy``.
    It reduces to sqrt(x) / sqrt(1 - 3/x) for a_* = 0.

    Parameters
    ----------
    radius_rg : float or ndarray
        Boyer-Lindquist radius x = r / r_G, larger than the circular photon orbit.
    spin : float or ndarray
        Dimensionless spin a_* in [-1, 1]; a negative value means a retrograde orbit.

    Returns
    -------
    float or ndarray
        L / (m c r_G) = L / (G m M / c), dimensionless and positive, counted along the direction of the orbit
        (for a retrograde orbit it is the magnitude).

    References
    ----------
    [Abramowicz2013] TeX labels circular-geodesic, omega-covariant, kerr-metric.
    [Pugliese2011] Eq. LPM.

    Validity
    --------
    Kerr metric, equatorial plane, test particle; stable only for r above the ISCO. A ValueError is raised for r
    at or below the photon orbit.

    Verification
    ------------
    tests/unit/test_kerr.py: Schwarzschild closed form; L/E equals the Keplerian l_K of the source; dL/dr = 0
    at the ISCO; dE = Omega dL.
    """
    a = _spin(spin)
    r = _orbit_radius(radius_rg, a)
    return _circular_angular_momentum(r, a)


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies', r_mb (TeX label bound-radius-kerr)",
    status="read",
    note="The formula is printed for a_* > 0 (prograde: 1 for a_* = 1, 4 for a_* = 0). For a_* < 0 the same "
    "expression with the signed spin gives the retrograde orbit (5.83 for a_* = -1); the test checks both signs "
    "against the condition E = 1 of kerr_circular_orbit_specific_energy, which is independent of this formula.",
)
def kerr_marginally_bound_radius_rg(spin: ArrayLike) -> Array:
    """Compute the radius of the marginally bound circular orbit of a Kerr black hole.

    r_mb / r_G = 2 (1 - a_*/2 + sqrt(1 - a_*))

    Parameters
    ----------
    spin : float or ndarray
        Dimensionless spin a_* in [-1, 1]; a negative value means a retrograde orbit.

    Returns
    -------
    float or ndarray
        r_mb / r_G: 4 for a_* = 0, 1 for a_* = 1, 3 + 2 sqrt(2) for a_* = -1.

    References
    ----------
    [Abramowicz2013] TeX label bound-radius-kerr.

    Validity
    --------
    Kerr metric, equatorial plane, test particle. Circular orbits with r above this radius are bound
    (E < 1); the orbits between r_mb and the ISCO are bound but unstable.

    Verification
    ------------
    tests/unit/test_kerr.py: 4 and 1 at a_* = 0 and 1; E = 1 there for both signs of the spin; ordering
    r_ph < r_mb < r_isco.
    """
    a = _spin(spin)
    return 2.0 * (1.0 - a / 2.0 + np.sqrt(1.0 - a))


# --------------------------------------------------------------------------------------
# Epicyclic and precession frequencies
# --------------------------------------------------------------------------------------


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies', (omega_r*)^2 (TeX label epicyclic-radial)",
    status="read",
    note="The source writes x = r/M and a_* for a prograde orbit; the signed spin gives the retrograde orbit. The "
    "orbit is radially unstable where the bracket is negative (source, text after epicyclic-frequencies-"
    "relativistic: omega_r^2 < 0 inside the ISCO); this function then returns NaN, because there is no real "
    "frequency. Values of the bracket between -1e-12 and 0 are rounding noise at the ISCO and are set to zero.",
)
def kerr_radial_epicyclic_frequency_hz(mass_kg: ArrayLike, radius_rg: ArrayLike, spin: ArrayLike) -> Array:
    """Compute the radial epicyclic frequency of a circular equatorial geodesic, as measured at infinity.

    (omega_r*)^2 = Omega^2 (1 - 6/x + 8 a_* x^{-3/2} - 3 a_*^2 x^{-2});  f_r = omega_r* / (2 pi)

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].
    radius_rg : float or ndarray
        Boyer-Lindquist radius in units of r_G, larger than the circular photon orbit.
    spin : float or ndarray
        Dimensionless spin a_* in [-1, 1]; a negative value means a retrograde orbit.

    Returns
    -------
    float or ndarray
        f_r [Hz]; it vanishes at the ISCO and is NaN (no real frequency, radially unstable orbit) inside it.
        Because f_r is the square root of a difference that vanishes there, a radius within rounding of the
        ISCO gives f_r / f_orbit of order 1e-8, not exactly zero, and a radius inside the ISCO by less than
        1e-12 in that difference gives 0 instead of NaN.

    References
    ----------
    [Abramowicz2013] TeX label epicyclic-radial.

    Validity
    --------
    Kerr metric, equatorial plane, test particle, linear perturbation. Inside the ISCO the squared frequency is
    negative: the perturbation grows exponentially and NaN is returned.

    Verification
    ------------
    tests/unit/test_kerr.py: Schwarzschild limit Omega sqrt(1 - 6/x); zero at the ISCO of basics for both signs;
    NaN inside; maximum of f_r outside the ISCO.
    """
    a = _spin(spin)
    r = _orbit_radius(radius_rg, a)
    return kerr_circular_orbit_frequency_hz(mass_kg, r, a) * _radial_ratio(r, a)


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies', (omega_theta*)^2 (TeX label epicyclic-vertical)",
    status="read",
    note="The source writes x = r/M and a_* for a prograde orbit; the signed spin gives the retrograde orbit.",
)
def kerr_vertical_epicyclic_frequency_hz(mass_kg: ArrayLike, radius_rg: ArrayLike, spin: ArrayLike) -> Array:
    """Compute the vertical epicyclic frequency of a circular equatorial geodesic, as measured at infinity.

    (omega_theta*)^2 = Omega^2 (1 - 4 a_* x^{-3/2} + 3 a_*^2 x^{-2});  f_theta = omega_theta* / (2 pi)

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].
    radius_rg : float or ndarray
        Boyer-Lindquist radius in units of r_G, larger than the circular photon orbit.
    spin : float or ndarray
        Dimensionless spin a_* in [-1, 1]; a negative value means a retrograde orbit.

    Returns
    -------
    float or ndarray
        f_theta [Hz]; equal to the orbital frequency for a_* = 0.

    References
    ----------
    [Abramowicz2013] TeX label epicyclic-vertical.

    Validity
    --------
    Kerr metric, equatorial plane, test particle, linear perturbation. The squared bracket is positive for
    every r above the photon orbit that was tested (see the test); a negative value would give NaN.

    Verification
    ------------
    tests/unit/test_kerr.py: equals the orbital frequency for a_* = 0; weak-field expansion Omega (1 - 2 a_*
    x^{-3/2}).
    """
    a = _spin(spin)
    r = _orbit_radius(radius_rg, a)
    q = _epicyclic_squared(r, a, vertical=True)
    return kerr_circular_orbit_frequency_hz(mass_kg, r, a) * np.sqrt(np.where(q >= 0.0, q, np.nan))


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies', Omega_K and (omega_r*)^2 (TeX labels "
    "keplerian-velocity-kerr, epicyclic-radial)",
    status="read",
)
@source(
    "Reynolds2021",
    where="'Spin from quasi-periodic oscillations' (TeX label qpos), relativistic precession model paragraph, "
    "Spin2020.tex line 465: test-particle periastron precession frequency",
    status="read",
    note="Reynolds names the test-particle periastron precession and orbital frequencies of the RPM but prints no "
    "formula. The expression nu_per = nu_phi - nu_r is this project's derivation: one radial period lasts "
    "2 pi / omega_r*, during which the azimuth advances by 2 pi Omega / omega_r*; the pericentre advances by the "
    "excess over 2 pi per radial period, i.e. at the rate Omega - omega_r*.",
)
def kerr_periastron_precession_frequency_hz(mass_kg: ArrayLike, radius_rg: ArrayLike, spin: ArrayLike) -> Array:
    """Compute the periastron precession frequency of a slightly eccentric orbit in the Kerr geometry.

    f_per = f_phi - f_r = (Omega - omega_r*) / (2 pi)

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].
    radius_rg : float or ndarray
        Boyer-Lindquist radius of the guiding circular orbit in units of r_G.
    spin : float or ndarray
        Dimensionless spin a_* in [-1, 1]; a negative value means a retrograde orbit.

    Returns
    -------
    float or ndarray
        f_per [Hz]; NaN inside the ISCO (no real radial epicyclic frequency). It tends to
        3 Omega / x for large x and a_* = 0.

    References
    ----------
    [Abramowicz2013] TeX labels keplerian-velocity-kerr, epicyclic-radial; [Reynolds2021] relativistic
    precession model.

    Validity
    --------
    Test particle, small eccentricity (linear epicyclic theory). The periastron frequency is relative to the
    Boyer-Lindquist azimuth; the result is defined for a stable circular orbit only.

    Verification
    ------------
    tests/unit/test_kerr.py: the weak-field Schwarzschild value 3 x^{-5/2} (expansion of the source formula);
    equals f_phi - f_r by construction of the test inputs; NaN inside the ISCO.
    """
    a = _spin(spin)
    r = _orbit_radius(radius_rg, a)
    return kerr_circular_orbit_frequency_hz(mass_kg, r, a) * (1.0 - _radial_ratio(r, a))


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies', Omega_K and (omega_theta*)^2 (TeX labels "
    "keplerian-velocity-kerr, epicyclic-vertical)",
    status="read",
)
@source(
    "Reynolds2021",
    where="'Spin from quasi-periodic oscillations' (TeX label qpos), relativistic precession model paragraph, "
    "Spin2020.tex line 465: test-particle Lense-Thirring precession frequency (the source spells it Lens-Thirring)",
    status="read",
    note="At line 465 Reynolds names the test-particle Lense-Thirring precession frequency without a formula; line 236 "
    "of the same file prints the weak-field rate Omega_LT = 2a(GM)^2/c^5r^3, which is not used here (as printed it "
    "does not have the dimension of a frequency for a dimensionless a); the far-field limit 2 a_* x^-3 of this "
    "function, in units of c^3/(GM), is checked in the test. The "
    "expression nu_nod = nu_phi - nu_theta is this project's derivation, the same argument as for the periastron: "
    "the orbital plane (the node) advances at the rate Omega - omega_theta*.",
)
def kerr_nodal_precession_frequency_hz(mass_kg: ArrayLike, radius_rg: ArrayLike, spin: ArrayLike) -> Array:
    """Compute the nodal (Lense-Thirring) precession frequency of a slightly inclined orbit in the Kerr geometry.

    f_nod = f_phi - f_theta = (Omega - omega_theta*) / (2 pi)

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].
    radius_rg : float or ndarray
        Boyer-Lindquist radius of the guiding circular orbit in units of r_G.
    spin : float or ndarray
        Dimensionless spin a_* in [-1, 1]; a negative value means a retrograde orbit.

    Returns
    -------
    float or ndarray
        f_nod [Hz], counted along the direction of the orbit: positive for a prograde orbit (the orbital plane
        is dragged along the rotation of the hole), negative for a retrograde orbit, zero for a_* = 0. It tends
        to 2 a_* Omega x^{-3/2} for large x.

    References
    ----------
    [Abramowicz2013] TeX labels keplerian-velocity-kerr, epicyclic-vertical; [Reynolds2021] relativistic
    precession model.

    Validity
    --------
    Test particle, small inclination (linear epicyclic theory); a stable or an unstable circular orbit.

    Verification
    ------------
    tests/unit/test_kerr.py: zero for a_* = 0; weak-field value 2 a_* x^{-3} (expansion of the source formula);
    sign reverses with the sign of the spin.
    """
    a = _spin(spin)
    r = _orbit_radius(radius_rg, a)
    q = _epicyclic_squared(r, a, vertical=True)
    return kerr_circular_orbit_frequency_hz(mass_kg, r, a) * (1.0 - np.sqrt(np.where(q >= 0.0, q, np.nan)))


# --------------------------------------------------------------------------------------
# Frame dragging
# --------------------------------------------------------------------------------------


@source(
    "Abramowicz2013",
    where="'Summary: characteristic radii and frequencies', omega (TeX label dragg-velocity-kerr)",
    status="read",
    note="G = c = 1 in the source; converted to SI with omega = c^3 / (G M) * omega~. The text before TeX label "
    "omega-covariant gives the definition omega = -g_tphi / g_phiphi, which this expression equals "
    "(checked in the test from the metric components of kerr-metric).",
)
def frame_dragging_angular_velocity_rad_s(
    mass_kg: ArrayLike, radius_rg: ArrayLike, spin: ArrayLike, polar_angle_rad: ArrayLike = np.pi / 2
) -> Array:
    """Compute the angular velocity of zero-angular-momentum observers (frame dragging) in the Kerr geometry.

    omega = 2 a_* x / [(a_*^2 + x^2) rho^2 + 2 a_*^2 x sin^2(theta)] * c^3 / (G M),  rho^2 = x^2 + a_*^2 cos^2(theta)

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].
    radius_rg : float or ndarray
        Boyer-Lindquist radius x = r / r_G; must be positive. Values inside the horizon are not excluded
        (the expression is smooth there) but have no meaning for an observer at infinity.
    spin : float or ndarray
        Dimensionless spin a_* in [-1, 1]; the sign of omega is the sign of the spin.
    polar_angle_rad : float or ndarray, optional
        Polar angle theta [rad]; the default is the equator.

    Returns
    -------
    float or ndarray
        omega = -g_tphi / g_phiphi [rad s^-1]; it equals the horizon angular velocity at the horizon and falls
        as 2 a_* / x^3 far away.

    References
    ----------
    [Abramowicz2013] TeX label dragg-velocity-kerr.

    Validity
    --------
    Kerr metric, Boyer-Lindquist coordinates, observers outside the horizon.

    Verification
    ------------
    tests/unit/test_kerr.py: equals basics.horizon_angular_velocity_rad_s at r_+; equals -g_tphi/g_phiphi from the
    metric of the source at several angles; 2 a_* / x^3 far away.
    """
    m = _positive(mass_kg, "mass")
    a = _spin(spin)
    x = _positive(radius_rg, "radius")
    theta = np.asarray(polar_angle_rad, dtype=float)
    rho2 = x**2 + (a * np.cos(theta)) ** 2
    omega = 2.0 * a * x / ((a**2 + x**2) * rho2 + 2.0 * a**2 * x * np.sin(theta) ** 2)
    return _C**3 / (_G * m) * omega


# --------------------------------------------------------------------------------------
# Horizon thermodynamics
# --------------------------------------------------------------------------------------


@source(
    "Page2005",
    where="TeX label eq:3, A = 4 pi (r_+^2 + a^2) = 4 pi M^2 [2 - Q_*^2 + 2 (1 - a_*^2 - Q_*^2)^{1/2}] (Q = 0)",
    status="read",
    note="G = c = 1 in the source; restored to SI with M -> G M / c^2.",
)
@source(
    "Visser2007",
    where="section 'Horizons', area of the horizons A_H = 4 pi (r_+^2 + a^2) = 8 pi (m^2 + sqrt(m^4 - m^2 a^2)) "
    "(equation without a label, Survey-grqc-3.tex line 1242)",
    status="read",
)
def kerr_horizon_area_m2(mass_kg: ArrayLike, spin: ArrayLike) -> Array:
    """Compute the area of the event horizon of a Kerr black hole.

    A = 8 pi r_G^2 (1 + sqrt(1 - a_*^2)) = 8 pi (G M / c^2)^2 (1 + sqrt(1 - a_*^2))

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].
    spin : float or ndarray
        Dimensionless spin a_* in [-1, 1]; the area depends on |a_*| only.

    Returns
    -------
    float or ndarray
        A [m^2]: 16 pi r_G^2 for a_* = 0, 8 pi r_G^2 for |a_*| = 1.

    References
    ----------
    [Page2005] TeX label eq:3; [Visser2007] horizon area.

    Validity
    --------
    Stationary, uncharged Kerr black hole; the area of the event horizon (outer root of Delta = 0).

    Verification
    ------------
    tests/unit/test_kerr.py: 4 pi (r_+^2 + a^2) with the horizon radius of basics; ratio 1/2 between |a_*| = 1
    and a_* = 0; the a_* = 0 value 16 pi r_G^2 (Page, text after TeX label eq:3).
    """
    m = _positive(mass_kg, "mass")
    a = _spin(spin)
    return 8.0 * np.pi * (_G * m / _C**2) ** 2 * (1.0 + np.sqrt(1.0 - a**2))


@source(
    "Carroll1997",
    where="Section 7, Eq. 7.149 (TeX label 7.149): M_irr^2 = A / (16 pi G^2)",
    status="read",
    note="With A from Page2005 TeX label eq:3 the definition gives M_irr = M sqrt[(1 + sqrt(1 - a_*^2)) / 2] "
    "(derivation in this project; it is also the third and fourth line of Eq. 7.149 with J = G M^2 a_* / c, "
    "up to the G factors that the source marks as unchecked).",
)
def kerr_irreducible_mass_kg(mass_kg: ArrayLike, spin: ArrayLike) -> Array:
    """Compute the irreducible mass of a Kerr black hole.

    M_irr = sqrt(A c^4 / (16 pi G^2)) = M sqrt[(1 + sqrt(1 - a_*^2)) / 2]

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].
    spin : float or ndarray
        Dimensionless spin a_* in [-1, 1]; M_irr depends on |a_*| only.

    Returns
    -------
    float or ndarray
        M_irr [kg]: M for a_* = 0, M / sqrt(2) for |a_*| = 1. It cannot be decreased by any classical process.

    References
    ----------
    [Carroll1997] Eq. 7.149.

    Validity
    --------
    Stationary Kerr black hole. M_irr is the mass that remains when all rotational energy has been extracted
    by reversible processes (source, Eq. 7.152).

    Verification
    ------------
    tests/unit/test_kerr.py: M at a_* = 0 and M / sqrt(2) at |a_*| = 1; equals sqrt(A c^4 / (16 pi G^2)) with the
    area function; the Eq. 7.149 form sqrt{[M^2 + sqrt(M^4 - (J/G)^2)] / 2}.
    """
    m = _positive(mass_kg, "mass")
    a = _spin(spin)
    return m * np.sqrt((1.0 + np.sqrt(1.0 - a**2)) / 2.0)


@source(
    "Carroll1997",
    where="Section 7, Eq. 7.152 (TeX label 7.152) and the sentence after it: about 29% of the energy of an "
    "extreme Kerr black hole",
    status="read",
    note="1 - M_irr / M with M_irr of Eq. 7.149 (derivation in this project: M_irr / M = sqrt[(1 + sqrt(1 - a_*^2))"
    " / 2]).",
)
def kerr_penrose_extractable_fraction(spin: ArrayLike) -> Array:
    """Compute the maximum fraction of the mass-energy of a Kerr black hole that can be extracted.

    (M - M_irr) / M = 1 - sqrt[(1 + sqrt(1 - a_*^2)) / 2]

    Parameters
    ----------
    spin : float or ndarray
        Dimensionless spin a_* in [-1, 1].

    Returns
    -------
    float or ndarray
        Fraction of M c^2 that is rotational energy: 0 for a_* = 0 and 1 - 1/sqrt(2) = 0.2929 for |a_*| = 1.

    References
    ----------
    [Carroll1997] Eq. 7.152.

    Validity
    --------
    Upper limit for reversible extraction (Penrose process, Blandford-Znajek and other processes that keep the
    area constant); the source says "approximately 29%" for the extreme black hole.

    Verification
    ------------
    tests/unit/test_kerr.py: 0 and 0.2929 at a_* = 0 and 1; 1 - M_irr / M with the irreducible-mass function;
    monotonic in |a_*|.
    """
    a = _spin(spin)
    return 1.0 - np.sqrt((1.0 + np.sqrt(1.0 - a**2)) / 2.0)


@source(
    "Page2005",
    where="TeX label eq:3, kappa = 4 pi (r_+ - M) / A = (1/2) M^{-1} [1 + (1 - Q_*^2/2)(1 - a_*^2 - Q_*^2)^{-1/2}]^{-1} (Q = 0)",
    status="read",
    note="G = c = 1 in the source, kappa in 1/M. Here kappa is the acceleration of the Killing horizon, "
    "kappa_SI = c^4 kappa~ / (G) (conversion in this project), so that T = hbar kappa_SI / (2 pi c k_B) "
    "reproduces T = hbar c^3 / (8 pi G M k_B) of basics for a_* = 0. Carroll1997 Eq. 7.155 gives the same "
    "expression in the form sqrt(G^2 M^2 - a^2) / [2 G M (G M + sqrt(G^2 M^2 - a^2))].",
)
@source(
    "Carroll1997",
    where="Section 7, Eq. 7.155 (TeX label 7.155)",
    status="read",
)
def kerr_surface_gravity_m_s2(mass_kg: ArrayLike, spin: ArrayLike) -> Array:
    """Compute the surface gravity of the horizon of a Kerr black hole.

    kappa = c^4 / (2 G M) * sqrt(1 - a_*^2) / (1 + sqrt(1 - a_*^2))

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].
    spin : float or ndarray
        Dimensionless spin a_* in [-1, 1].

    Returns
    -------
    float or ndarray
        kappa [m s^-2] (the acceleration scale of the Killing horizon, with respect to the time at infinity):
        c^4 / (4 G M) for a_* = 0, zero for |a_*| = 1 (extremal black hole).

    References
    ----------
    [Page2005] TeX label eq:3; [Carroll1997] Eq. 7.155.

    Validity
    --------
    Stationary, uncharged Kerr black hole. The unit in which a source states kappa is geometrized (1/length);
    the conversion to an acceleration is made here.

    Verification
    ------------
    tests/unit/test_kerr.py: 4 pi (r_+ - M) / A from the horizon radius and area; Eq. 7.155 form; zero at
    |a_*| = 1; equals c^4/(4 G M) at a_* = 0.
    """
    m = _positive(mass_kg, "mass")
    a = _spin(spin)
    root = np.sqrt(1.0 - a**2)
    return _C**4 / (2.0 * _G * m) * root / (1.0 + root)


@source(
    "Page2005",
    where="Introduction (text before TeX label eq:2) and section 'Hawking Emission Formulae': T_bh = kappa / (2 pi), kappa of TeX label eq:3",
    status="read",
    note="T = kappa / (2 pi) in geometrized units with hbar kept; in SI T = hbar kappa_SI / (2 pi c k_B) "
    "(conversion in this project). For Kerr emission the Planck factor of the source is "
    "1 / (exp[2 pi (omega - m Omega) / kappa] -/+ 1); T_bh is the horizon temperature that stands in "
    "kappa / (2 pi) of that factor.",
)
def kerr_hawking_temperature_k(mass_kg: ArrayLike, spin: ArrayLike) -> Array:
    """Compute the Hawking temperature of a Kerr black hole: T = hbar kappa / (2 pi c k_B).

    T = hbar c^3 / (4 pi G M k_B) * sqrt(1 - a_*^2) / (1 + sqrt(1 - a_*^2))

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].
    spin : float or ndarray
        Dimensionless spin a_* in [-1, 1].

    Returns
    -------
    float or ndarray
        T [K]; equals ``basics.hawking_temperature_k`` for a_* = 0 and tends to zero as |a_*| -> 1.

    References
    ----------
    [Page2005] T = kappa / (2 pi) and TeX label eq:3.

    Validity
    --------
    Semiclassical limit; T = kappa / (2 pi) is the horizon temperature of the first law. The emission
    spectrum of a rotating hole is not a blackbody of this temperature: the Planck factor of the source
    contains omega - m Omega (mode number m, horizon angular velocity Omega) and greybody factors.

    Verification
    ------------
    tests/unit/test_kerr.py: equals basics.hawking_temperature_k at a_* = 0; falls monotonically to zero; first
    law dM c^2 = T dS + Omega_H dJ by finite differences.
    """
    return _HBAR * kerr_surface_gravity_m_s2(mass_kg, spin) / (2.0 * np.pi * _C * BOLTZMANN.value)


@source(
    "Page2005",
    where="TeX label eq:2, S_bh = S_BH = A / 4 (Planck units), with A of TeX label eq:3",
    status="read",
    note="The conversion from Planck units to SI (S / k_B = A c^3 / (4 G hbar)) is the one of basics."
    "bekenstein_hawking_entropy; it is this project's, not written in the source.",
)
def kerr_bekenstein_hawking_entropy(mass_kg: ArrayLike, spin: ArrayLike) -> Array:
    """Compute the Bekenstein-Hawking entropy of a Kerr black hole in units of the Boltzmann constant.

    S / k_B = A c^3 / (4 G hbar)

    Parameters
    ----------
    mass_kg : float or ndarray
        Black hole mass [kg].
    spin : float or ndarray
        Dimensionless spin a_* in [-1, 1].

    Returns
    -------
    float or ndarray
        S / k_B (dimensionless); half the Schwarzschild value for |a_*| = 1.

    References
    ----------
    [Page2005] TeX label eq:2 (S = A/4) and TeX label eq:3 (A).

    Validity
    --------
    Semiclassical black hole thermodynamics; the entropy is a function of the horizon area only.

    Verification
    ------------
    tests/unit/test_kerr.py: equals basics.bekenstein_hawking_entropy at a_* = 0; A/4 with the area function;
    first law by finite differences.
    """
    return kerr_horizon_area_m2(mass_kg, spin) * _C**3 / (4.0 * _G * _HBAR)


# --------------------------------------------------------------------------------------
# Area theorem
# --------------------------------------------------------------------------------------


@source(
    "Sathyaprakash2009",
    where="Section 'Tests of general relativity' (no TeX label, article.tex line 5867): 'the final area must exceed "
    "the sum of the areas of the initial holes'",
    status="read",
    note="The area of each hole is that of kerr_horizon_area_m2 (Page2005 TeX label eq:3). The margin A_f - A_1 - A_2 "
    "is the quantity that the sentence says must be positive; the function itself is this project's.",
)
@source(
    "Page2005",
    where="Section 1, second law of black hole mechanics (Hawking's area theorem), no TeX label",
    status="read",
)
def kerr_area_theorem_margin_m2(
    mass1_kg: ArrayLike,
    spin1: ArrayLike,
    mass2_kg: ArrayLike,
    spin2: ArrayLike,
    final_mass_kg: ArrayLike,
    final_spin: ArrayLike,
) -> Array:
    """Compute the area increase in a black hole merger: A_final - A_1 - A_2.

    Parameters
    ----------
    mass1_kg, mass2_kg : float or ndarray
        Masses of the two initial black holes [kg].
    spin1, spin2 : float or ndarray
        Dimensionless spins a_* of the two initial black holes, in [-1, 1].
    final_mass_kg : float or ndarray
        Mass of the final black hole [kg].
    final_spin : float or ndarray
        Dimensionless spin of the final black hole, in [-1, 1].

    Returns
    -------
    float or ndarray
        A_final - A_1 - A_2 [m^2]; the area theorem requires it to be non-negative. A negative value means that
        the numbers cannot describe a merger in classical general relativity.

    References
    ----------
    [Sathyaprakash2009] 'Tests of general relativity'; [Page2005] second law of black hole mechanics.

    Validity
    --------
    Each hole is a stationary Kerr black hole before and after (the area is defined on the event horizon).
    Masses are the total masses at infinite separation and after ringdown; the radiated energy is the
    difference between the sum of the initial masses and the final mass.

    Verification
    ------------
    tests/unit/test_kerr.py: zero at the area-theorem limit M_f = sqrt(2) M for two equal non-spinning holes;
    positive for generic merger numbers; sign changes with the final mass.
    """
    return (
        kerr_horizon_area_m2(final_mass_kg, final_spin)
        - kerr_horizon_area_m2(mass1_kg, spin1)
        - kerr_horizon_area_m2(mass2_kg, spin2)
    )


@source(
    "Sathyaprakash2009",
    where="Section 'Tests of general relativity' (no TeX label, article.tex line 5867): the area theorem for a merger",
    status="read",
)
@source(
    "Carroll1997",
    where="Section 7, Eq. 7.149 (M_irr) and Eq. 7.152 (M >= M_irr)",
    status="read",
    note="Derivation in this project: the area theorem gives M_irr,f^2 >= M_irr,1^2 + M_irr,2^2 (Eq. 7.149: "
    "M_irr^2 = A / (16 pi G^2)); M_f >= M_irr,f because M_irr / M <= 1; so the radiated energy "
    "M_1 + M_2 - M_f is at most M_1 + M_2 - sqrt(M_irr,1^2 + M_irr,2^2). For two equal non-spinning holes this is "
    "1 - 1/sqrt(2) of the total. No source read here states the number 1 - 1/sqrt(2).",
)
def merger_radiated_fraction_bound(
    mass1_kg: ArrayLike, spin1: ArrayLike, mass2_kg: ArrayLike, spin2: ArrayLike
) -> Array:
    """Compute the upper bound that the area theorem puts on the fraction of the mass radiated in a merger.

    (E_rad / (M_1 + M_2) c^2) <= 1 - sqrt(M_irr,1^2 + M_irr,2^2) / (M_1 + M_2)

    Parameters
    ----------
    mass1_kg, mass2_kg : float or ndarray
        Masses of the two initial black holes [kg].
    spin1, spin2 : float or ndarray
        Dimensionless spins a_* of the two initial black holes, in [-1, 1].

    Returns
    -------
    float or ndarray
        Upper bound of the radiated fraction of the total initial mass (dimensionless): 1 - 1/sqrt(2) = 0.2929
        for two equal non-spinning holes.

    References
    ----------
    [Sathyaprakash2009] 'Tests of general relativity'; [Carroll1997] Eq. 7.149, 7.152.

    Validity
    --------
    Bound only: it assumes that the final hole is Schwarzschild with the combined irreducible mass, which
    angular momentum conservation does not allow for a generic merger. Real mergers radiate much less (a few
    per cent of the total mass for comparable-mass black holes).

    Verification
    ------------
    tests/unit/test_kerr.py: 1 - 1/sqrt(2) for equal non-spinning holes; bounded by 1; consistent with the area
    margin function (a final mass at the bound gives zero margin for a_final = 0).
    """
    m1 = _positive(mass1_kg, "mass 1")
    m2 = _positive(mass2_kg, "mass 2")
    irr1 = kerr_irreducible_mass_kg(m1, spin1)
    irr2 = kerr_irreducible_mass_kg(m2, spin2)
    return 1.0 - np.sqrt(irr1**2 + irr2**2) / (m1 + m2)


# --------------------------------------------------------------------------------------
# Charged black hole
# --------------------------------------------------------------------------------------


@source(
    "Page2005",
    where="TeX label eq:3, r_+ = M + (M^2 - a^2 - Q^2)^{1/2} = M [1 + (1 - a_*^2 - Q_*^2)^{1/2}]",
    status="read",
    note="Kerr-Newman in geometrized units G = c = k = 4 pi epsilon_0 = 1; Q_* = Q / M. Carroll1997 Eq. 7.113 "
    "gives the same root for the Reissner-Nordstrom black hole (a_* = 0, G (p^2 + q^2) = Q^2).",
)
@source("Carroll1997", where="Section 7, Eq. 7.113 (TeX label 7.113): r_+- for Reissner-Nordstrom", status="read")
def kerr_newman_horizon_radius_rg(spin: ArrayLike, charge_parameter: ArrayLike) -> Array:
    """Compute the outer horizon radius of a Kerr-Newman black hole: r_+ = r_G [1 + sqrt(1 - a_*^2 - Q_*^2)].

    Parameters
    ----------
    spin : float or ndarray
        Dimensionless spin a_* in [-1, 1].
    charge_parameter : float or ndarray
        Dimensionless charge Q_* = Q / M in geometrized units (the charge in the units in which G, c and
        4 pi epsilon_0 are 1, divided by the mass in the same units).

    Returns
    -------
    float or ndarray
        r_+ / r_G: 2 for a_* = Q_* = 0 (Schwarzschild), 1 for a_*^2 + Q_*^2 = 1 (extremal).

    References
    ----------
    [Page2005] TeX label eq:3; [Carroll1997] Eq. 7.113.

    Validity
    --------
    Requires a_*^2 + Q_*^2 <= 1 (otherwise there is no horizon and a ValueError is raised). Astrophysical black
    holes are expected to be almost neutral.

    Verification
    ------------
    tests/unit/test_kerr.py: equals basics.kerr_horizon_radius_rg at Q_* = 0; Reissner-Nordstrom 1 + sqrt(1 - Q_*^2);
    root of Delta = r^2 - 2 r + a_*^2 + Q_*^2.
    """
    a = _spin(spin)
    q = np.asarray(charge_parameter, dtype=float)
    disc = 1.0 - a**2 - q**2
    if np.any(~(disc >= 0.0)):
        raise ValueError("a_*^2 + Q_*^2 must not exceed 1: there is no horizon (naked singularity)")
    return 1.0 + np.sqrt(disc)
