"""Energy, momentum and virial bookkeeping of gravitating particle systems, and two-body orbit relations (SI).

The equations were read from the TeX sources of Dehnen and Read [DehnenRead2011] (arXiv:1105.1082, ``nbody.tex``),
of Murray and Correia [MurrayCorreia2010] (arXiv:1009.1738, ``Chapter2.tex``) and of Chiang et al. [Chiang2020]
(arXiv:2007.01679, ``ms_arxiv.tex``); equations are cited by the TeX labels with the TeX line number.

Arrays of N particles: ``positions_m`` and ``velocities_m_s`` have shape (N, 3), ``masses_kg`` has shape (N,).
The grammar of unit suffixes of kozmos cannot express kg m s^-1 or kg m^2 s^-1, so the momenta carry no suffix; their
SI units are stated in the docstrings.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.constants import GRAVITATIONAL_CONSTANT

Array = NDArray[np.float64]

_G = GRAVITATIONAL_CONSTANT.value


def _particles(masses_kg: ArrayLike, positions_m: ArrayLike, velocities_m_s: ArrayLike | None = None):
    m = np.asarray(masses_kg, dtype=float)
    x = np.asarray(positions_m, dtype=float)
    if m.ndim != 1 or x.ndim != 2 or x.shape[1] != 3 or x.shape[0] != m.shape[0]:
        raise ValueError("masses_kg must have shape (N,) and positions_m shape (N, 3)")
    if np.any(~(m > 0.0)):
        raise ValueError("masses must be positive")
    if velocities_m_s is None:
        return m, x
    v = np.asarray(velocities_m_s, dtype=float)
    if v.shape != x.shape:
        raise ValueError("velocities_m_s must have the same shape as positions_m")
    return m, x, v


def _softening(softening_m: float) -> float:
    eps = float(softening_m)
    if not (eps >= 0.0) or not np.isfinite(eps):
        raise ValueError("softening_m must be finite and >= 0")
    return eps


@source(
    "DehnenRead2011",
    where="TeX line 1234 (virial theorem paragraph): T = sum_i (1/2) m_i |dx_i/dt|^2",
    status="read",
)
def kinetic_energy_j(masses_kg: ArrayLike, velocities_m_s: ArrayLike) -> float:
    """Compute the total kinetic energy: T = sum_i (1/2) m_i |v_i|^2.

    Parameters
    ----------
    masses_kg : array_like
        Masses [kg], shape (N,).
    velocities_m_s : array_like
        Velocities [m s^-1], shape (N, 3).

    Returns
    -------
    float
        T [J].

    References
    ----------
    [DehnenRead2011] TeX line 1234 (T in the virial theorem).

    Validity
    --------
    Non-relativistic.

    Verification
    ------------
    tests/unit/test_energy.py: closed form for a circular two-body orbit; equal to the sum of the two (1/2) m v^2.
    """
    m = np.asarray(masses_kg, dtype=float)
    v = np.asarray(velocities_m_s, dtype=float)
    if m.ndim != 1 or v.ndim != 2 or v.shape[0] != m.shape[0] or v.shape[1] != 3:
        raise ValueError("masses_kg must have shape (N,) and velocities_m_s shape (N, 3)")
    return float(0.5 * np.sum(m * np.einsum("ij,ij->i", v, v)))


def _pair_geometry(x: Array, eps: float):
    """Return dx[i, j] = x_j - x_i and r2[i, j] + eps^2 with the diagonal set to 1 (it is never used)."""
    dx = x[None, :, :] - x[:, None, :]
    r2 = np.einsum("ijk,ijk->ij", dx, dx) + eps * eps
    np.fill_diagonal(r2, 1.0)
    if np.any(r2 <= 0.0):
        raise ValueError("two particles coincide and softening_m is zero")
    return dx, r2


@source(
    "DehnenRead2011",
    where="TeX label eqn:direct (line 255): H = sum p^2/2m - G sum_{j>i} m_i m_j / |x_i - x_j|; Plummer softening "
    "eq:plummer (line 1515)",
    status="read",
    note="derivation in this project from eq:plummer: the potential of particle j at x is -G m_j / sqrt(|x - x_j|^2 + "
    "eps^2), so the pair energy is -G m_i m_j / sqrt(r_ij^2 + eps^2) and the system energy is the sum over pairs "
    "(each pair once); with eps = 0 this is the potential term of eqn:direct",
)
def potential_energy_j(masses_kg: ArrayLike, positions_m: ArrayLike, softening_m: float = 0.0) -> float:
    """Compute the pairwise gravitational potential energy: W = -G sum_{i<j} m_i m_j / sqrt(r_ij^2 + eps^2).

    Parameters
    ----------
    masses_kg : array_like
        Masses [kg], shape (N,).
    positions_m : array_like
        Positions [m], shape (N, 3).
    softening_m : float
        Plummer softening length eps [m], >= 0 (0 gives the point-mass potential).

    Returns
    -------
    float
        W [J], negative.

    References
    ----------
    [DehnenRead2011] eqn:direct, eq:plummer.

    Validity
    --------
    O(N^2) time and 56 N^2 bytes of peak memory (measured with ``tracemalloc`` on Windows, eps > 0; 48.0 N^2 at
    N = 500 on Linux, where the test suite runs as well: 56.0005 N^2 at N = 1000
    and 56.0001 N^2 at N = 2000, i.e. 56 MB and 224 MB; the N x N x 3 displacement array and its relatives are
    the bulk); for N = 10^4 this is about 5.6 GB. With a nonzero eps the energy is the exact conserved
    energy of the softened equations of motion of ``accelerations_m_s2`` (same eps). Two coincident particles with
    eps = 0 raise ValueError.

    Verification
    ------------
    tests/unit/test_energy.py: two-body closed form -G m1 m2 / r; the force of accelerations_m_s2 is minus the
    numerical gradient of this energy (central differences), for eps = 0 and eps > 0.
    """
    m, x = _particles(masses_kg, positions_m)
    eps = _softening(softening_m)
    _, r2 = _pair_geometry(x, eps)
    pair = m[:, None] * m[None, :] / np.sqrt(r2)
    np.fill_diagonal(pair, 0.0)
    return float(-0.5 * _G * np.sum(pair))


@source(
    "DehnenRead2011",
    where="TeX label eqn:direct (line 255): a_i = -G sum_{j != i} m_j (x_i - x_j) / |x_i - x_j|^3; Plummer "
    "softening eq:plummer (line 1515)",
    status="read",
    note="the softened acceleration is derivation in this project from eq:plummer: a_i = -grad_i of the softened "
    "pair energy / m_i = G sum_{j != i} m_j (x_j - x_i) / (r_ij^2 + eps^2)^{3/2}",
)
def accelerations_m_s2(masses_kg: ArrayLike, positions_m: ArrayLike, softening_m: float = 0.0) -> Array:
    """Compute the gravitational accelerations by direct summation (Plummer-softened).

    a_i = G sum_{j != i} m_j (x_j - x_i) / (r_ij^2 + eps^2)^(3/2)

    Parameters
    ----------
    masses_kg : array_like
        Masses [kg], shape (N,).
    positions_m : array_like
        Positions [m], shape (N, 3).
    softening_m : float
        Plummer softening length eps [m], >= 0.

    Returns
    -------
    ndarray
        Accelerations [m s^-2], shape (N, 3).

    References
    ----------
    [DehnenRead2011] eqn:direct, eq:plummer.

    Validity
    --------
    O(N^2) time; peak memory 48 N^2 bytes (measured with ``tracemalloc``, eps > 0: 48.07 N^2 at N = 1000 and
    48.02 N^2 at N = 2000); N = 10^4 needs about 4.8 GB. For eps = 0 two coincident particles raise ValueError.
    The sum of m_i a_i is zero to rounding for any eps (equal and opposite pair forces).

    Verification
    ------------
    tests/unit/test_energy.py: two-body closed form G m / r^2; sum of m a = 0; equals minus the gradient of
    potential_energy_j; a leapfrog run conserves the total energy.
    """
    m, x = _particles(masses_kg, positions_m)
    eps = _softening(softening_m)
    dx, r2 = _pair_geometry(x, eps)
    w = m[None, :] * r2**-1.5
    np.fill_diagonal(w, 0.0)
    return _G * np.einsum("ij,ijk->ik", w, dx)


@source(
    "DehnenRead2011",
    where="TeX label eqn:direct (line 255), the Hamiltonian H = T + W",
    status="read",
)
def total_energy_j(
    masses_kg: ArrayLike, positions_m: ArrayLike, velocities_m_s: ArrayLike, softening_m: float = 0.0
) -> float:
    """Compute the total energy of the particle system: E = T + W.

    Parameters
    ----------
    masses_kg : array_like
        Masses [kg], shape (N,).
    positions_m : array_like
        Positions [m], shape (N, 3).
    velocities_m_s : array_like
        Velocities [m s^-1], shape (N, 3).
    softening_m : float
        Plummer softening length [m], >= 0; must be the same as in the acceleration used for the evolution.

    Returns
    -------
    float
        E [J].

    References
    ----------
    [DehnenRead2011] eqn:direct (the Hamiltonian).

    Validity
    --------
    Conserved by the equations of motion of ``accelerations_m_s2`` with the same softening. Memory and time as
    ``potential_energy_j``.

    Verification
    ------------
    tests/unit/test_energy.py: -G m1 m2 / (2 a) for a two-body orbit; conservation along a leapfrog run.
    """
    m, x, v = _particles(masses_kg, positions_m, velocities_m_s)
    return kinetic_energy_j(m, v) + potential_energy_j(m, x, softening_m)


@source(
    "MurrayCorreia2010",
    where="TeX label eq49 (position of the centre of mass of two bodies)",
    status="read",
    note="the source prints R = (m1 r1 + m2 r2)/(m1 + m2); the sum over N particles is the obvious extension",
)
def barycenter_m(masses_kg: ArrayLike, positions_m: ArrayLike) -> Array:
    """Compute the centre-of-mass position: R = sum_i m_i x_i / sum_i m_i.

    Parameters
    ----------
    masses_kg : array_like
        Masses [kg], shape (N,).
    positions_m : array_like
        Positions [m], shape (N, 3).

    Returns
    -------
    ndarray
        R [m], shape (3,).

    References
    ----------
    [MurrayCorreia2010] eq49 (two bodies); the N-body sum is an extension.

    Validity
    --------
    Newtonian, positions in one inertial frame.

    Verification
    ------------
    tests/unit/test_energy.py: two-body closed form; invariance of the recentred system.
    """
    m, x = _particles(masses_kg, positions_m)
    return np.einsum("i,ij->j", m, x) / np.sum(m)


@source(
    "MurrayCorreia2010",
    where="TeX label eq50 and the sentence after it (dR/dt = V = constant)",
    status="read",
    note="derivation in this project from eq49: V = dR/dt = sum m_i v_i / sum m_i",
)
def barycenter_velocity_m_s(masses_kg: ArrayLike, velocities_m_s: ArrayLike) -> Array:
    """Compute the velocity of the centre of mass: V = sum_i m_i v_i / sum_i m_i.

    Parameters
    ----------
    masses_kg : array_like
        Masses [kg], shape (N,).
    velocities_m_s : array_like
        Velocities [m s^-1], shape (N, 3).

    Returns
    -------
    ndarray
        V [m s^-1], shape (3,).

    References
    ----------
    [MurrayCorreia2010] eq49, eq50.

    Validity
    --------
    Newtonian, one inertial frame. Constant in time for an isolated system.

    Verification
    ------------
    tests/unit/test_energy.py: two-body closed form; constant along a three-body leapfrog run.
    """
    m = np.asarray(masses_kg, dtype=float)
    v = np.asarray(velocities_m_s, dtype=float)
    if m.ndim != 1 or v.ndim != 2 or v.shape[0] != m.shape[0] or v.shape[1] != 3:
        raise ValueError("masses_kg must have shape (N,) and velocities_m_s shape (N, 3)")
    return np.einsum("i,ij->j", m, v) / np.sum(m)


@source(
    "DehnenRead2011",
    where="TeX text after eqn:direct (line 255): p_i = m_i dx_i/dt",
    status="read",
    note="derivation in this project: the total momentum is the sum of the p_i of the source",
)
def total_momentum(masses_kg: ArrayLike, velocities_m_s: ArrayLike) -> Array:
    """Compute the total linear momentum: P = sum_i m_i v_i, in kg m/s.

    Parameters
    ----------
    masses_kg : array_like
        Masses [kg], shape (N,).
    velocities_m_s : array_like
        Velocities [m s^-1], shape (N, 3).

    Returns
    -------
    ndarray
        P, shape (3,), in kg m/s.

    References
    ----------
    [DehnenRead2011] p_i = m_i dx_i/dt after eqn:direct.

    Validity
    --------
    Newtonian. Conserved for pairwise forces (also softened ones).

    Verification
    ------------
    tests/unit/test_energy.py: conserved to rounding along a three-body leapfrog run.
    """
    m = np.asarray(masses_kg, dtype=float)
    v = np.asarray(velocities_m_s, dtype=float)
    if m.ndim != 1 or v.ndim != 2 or v.shape[0] != m.shape[0] or v.shape[1] != 3:
        raise ValueError("masses_kg must have shape (N,) and velocities_m_s shape (N, 3)")
    return np.einsum("i,ij->j", m, v)


@source(
    "MurrayCorreia2010",
    where="TeX label eq04: r x dr/dt = h (angular momentum integral, per unit mass)",
    status="read",
    note="derivation in this project: the angular momentum of N particles about the origin is "
    "L = sum_i m_i x_i x v_i (m_i times eq04 for each particle, summed)",
)
def total_angular_momentum(masses_kg: ArrayLike, positions_m: ArrayLike, velocities_m_s: ArrayLike) -> Array:
    """Compute the total angular momentum about the origin: L = sum_i m_i x_i x v_i, in kg m^2/s.

    Parameters
    ----------
    masses_kg : array_like
        Masses [kg], shape (N,).
    positions_m : array_like
        Positions [m], shape (N, 3).
    velocities_m_s : array_like
        Velocities [m s^-1], shape (N, 3).

    Returns
    -------
    ndarray
        L, shape (3,), in kg m^2/s.

    References
    ----------
    [MurrayCorreia2010] eq04 (per unit mass, two-body relative motion).

    Validity
    --------
    About the origin of the coordinates; conserved for central pairwise forces, also with softening (the force
    on each particle of a pair is along the line joining them).

    Verification
    ------------
    tests/unit/test_energy.py: two-body closed form m_red sqrt(mu a (1 - e^2)); conserved to rounding along a
    three-body leapfrog run.
    """
    m, x, v = _particles(masses_kg, positions_m, velocities_m_s)
    return np.einsum("i,ij->j", m, np.cross(x, v))


@source(
    "DehnenRead2011",
    where="TeX lines 1233-1236 (virial theorem): 2T + W = 0 (line 1233) with T = sum (1/2) m |dx/dt|^2 (line 1234) and "
    "W = sum m_i x_i . d2x_i/dt2 (line 1236)",
    status="read",
)
def virial_ratio(
    masses_kg: ArrayLike, positions_m: ArrayLike, velocities_m_s: ArrayLike, softening_m: float = 0.0
) -> float:
    """Compute the virial ratio 2T / |W| with W = sum_i m_i x_i . a_i (equals 1 in virial equilibrium).

    Parameters
    ----------
    masses_kg : array_like
        Masses [kg], shape (N,).
    positions_m : array_like
        Positions [m], shape (N, 3).
    velocities_m_s : array_like
        Velocities [m s^-1], shape (N, 3).
    softening_m : float
        Plummer softening length [m], >= 0 (the accelerations used in W are the softened ones).

    Returns
    -------
    float
        2T / |W|, dimensionless.

    References
    ----------
    [DehnenRead2011] virial theorem 2T + W = 0 (TeX line 1233).

    Validity
    --------
    The virial theorem holds for the time average of an equilibrium system; the instantaneous ratio of a
    snapshot is 1 only on a circular two-body orbit and in a few other special cases. W is the source's
    definition (virial of the accelerations), computed with ``accelerations_m_s2``. One particle has no potential
    energy (W = 0) and the ratio is undefined: ValueError for N < 2, and also when W = 0 for N >= 2 (all
    particles at one point, possible only with a nonzero softening).

    Verification
    ------------
    tests/unit/test_energy.py: exactly 1 for a circular two-body orbit; time average over an eccentric orbit
    close to 1; W equals the potential energy for eps = 0.
    """
    m, x, v = _particles(masses_kg, positions_m, velocities_m_s)
    if m.size < 2:
        raise ValueError("the virial ratio needs at least two particles (one particle has no potential energy)")
    a = accelerations_m_s2(m, x, softening_m)
    w = np.sum(m * np.einsum("ij,ij->i", x, a))
    if w == 0.0:
        raise ValueError("W = 0 (all particles at one point): the virial ratio 2T / |W| is undefined")
    return float(2.0 * kinetic_energy_j(m, v) / abs(w))


@source(
    "MurrayCorreia2010",
    where="TeX label eq24 (vis viva integral): v^2/2 - G(m1 + m2)/r = C",
    status="read",
)
def vis_viva_constant_m2_s2(relative_position_m: ArrayLike, relative_velocity_m_s: ArrayLike, gm_m3_s2: float) -> Array:
    """Compute the vis viva constant (specific orbital energy): C = v^2/2 - mu/r.

    Parameters
    ----------
    relative_position_m : array_like
        Position of body 2 relative to body 1 [m], shape (..., 3).
    relative_velocity_m_s : array_like
        Velocity of body 2 relative to body 1 [m s^-1], shape (..., 3).
    gm_m3_s2 : float
        mu = G (m1 + m2) [m^3 s^-2].

    Returns
    -------
    ndarray
        C [m^2 s^-2] (per unit reduced mass; multiply by the reduced mass for the orbital energy in J).

    References
    ----------
    [MurrayCorreia2010] eq24.

    Validity
    --------
    Two bodies, Newtonian. C < 0 elliptic, C = 0 parabolic, C > 0 hyperbolic.

    Verification
    ------------
    tests/unit/test_energy.py: -mu/(2 a) on a circular orbit and at periapsis of an eccentric orbit; equals
    the energy of total_energy_j divided by the reduced mass.
    """
    r = np.asarray(relative_position_m, dtype=float)
    v = np.asarray(relative_velocity_m_s, dtype=float)
    return 0.5 * np.einsum("...i,...i->...", v, v) - gm_m3_s2 / np.linalg.norm(r, axis=-1)


@source(
    "MurrayCorreia2010",
    where="TeX label eq04 (r x dr/dt = h)",
    status="read",
)
def specific_angular_momentum_m2_s(relative_position_m: ArrayLike, relative_velocity_m_s: ArrayLike) -> Array:
    """Compute the specific angular momentum vector of the relative orbit: h = r x v.

    Parameters
    ----------
    relative_position_m : array_like
        Position of body 2 relative to body 1 [m], shape (..., 3).
    relative_velocity_m_s : array_like
        Velocity of body 2 relative to body 1 [m s^-1], shape (..., 3).

    Returns
    -------
    ndarray
        h [m^2 s^-1], shape (..., 3).

    References
    ----------
    [MurrayCorreia2010] eq04.

    Validity
    --------
    Two bodies, central force.

    Verification
    ------------
    tests/unit/test_energy.py: |h| = sqrt(mu a (1 - e^2)) (eq22 of the source) for a Kepler orbit.
    """
    return np.cross(np.asarray(relative_position_m, dtype=float), np.asarray(relative_velocity_m_s, dtype=float))


@source(
    "MurrayCorreia2010",
    where="TeX label eq24 and eq30 (v^2 = mu (2/r - 1/a))",
    status="read",
    note="derivation in this project: eliminating v^2 between eq24 (v^2/2 - mu/r = C) and eq30 gives "
    "C = -mu/(2a), so a = -mu/(2 C)",
)
def semi_major_axis_from_vis_viva_constant_m(vis_viva_constant_m2_s2: ArrayLike, gm_m3_s2: float) -> Array:
    """Convert the vis viva constant to the semi-major axis: a = -mu / (2 C).

    Parameters
    ----------
    vis_viva_constant_m2_s2 : array_like
        C [m^2 s^-2], negative for an ellipse.
    gm_m3_s2 : float
        mu = G (m1 + m2) [m^3 s^-2].

    Returns
    -------
    ndarray
        a [m]; negative for C > 0 (hyperbola, by the same formula) and +inf for C = 0 (see Validity).

    References
    ----------
    [MurrayCorreia2010] eq24, eq30.

    Validity
    --------
    Two-body Keplerian motion. The parabolic case C = 0 (either +0.0 or -0.0) returns +inf by convention: the
    formula -mu / (2 C) would give -inf for +0.0 and +inf for -0.0, a distinction without physical content (a
    parabola has an infinite semi-major axis; the sign is that of the limit from the ellipse, C -> 0 from below).

    Verification
    ------------
    tests/unit/test_energy.py: round trip with vis_viva_constant_m2_s2 on a circular orbit, C = +0.0 and -0.0 give
    +inf, and consistency with
    kozmos.ephemeris.kepler.orbital_speed_m_s (vis viva).
    """
    c = np.asarray(vis_viva_constant_m2_s2, dtype=float)
    with np.errstate(divide="ignore"):
        a = -gm_m3_s2 / (2.0 * c)
    return np.where(c == 0.0, np.inf, a)[()]


@source(
    "MurrayCorreia2010",
    where="TeX label eq22 (h = sqrt(G(m1 + m2) a (1 - e^2))) and eq24, eq30",
    status="read",
    note="derivation in this project: h^2 = mu a (1 - e^2) (eq22) with a = -mu/(2C) (eq24 with eq30) gives "
    "e^2 = 1 + 2 C h^2 / mu^2",
)
def eccentricity_from_vis_viva_constant(
    vis_viva_constant_m2_s2: ArrayLike, specific_angular_momentum_m2_s: ArrayLike, gm_m3_s2: float
) -> Array:
    """Compute the eccentricity from the vis viva constant and the angular momentum.

    e = sqrt(1 + 2 C h^2 / mu^2)

    Parameters
    ----------
    vis_viva_constant_m2_s2 : array_like
        C [m^2 s^-2].
    specific_angular_momentum_m2_s : array_like
        Magnitude of h [m^2 s^-1], >= 0.
    gm_m3_s2 : float
        mu = G (m1 + m2) [m^3 s^-2].

    Returns
    -------
    ndarray
        e, dimensionless (0 <= e < 1 elliptic).

    References
    ----------
    [MurrayCorreia2010] eq22, eq24, eq30.

    Validity
    --------
    Two-body Keplerian motion. The radicand is clipped at 0 when rounding makes it slightly negative (circular
    orbit); a value below -1e-9 raises ValueError. Accuracy limit: e^2 = 1 + 2 C h^2 / mu^2 is a difference of two
    terms that cancel for e -> 0 (2 C h^2 / mu^2 = -(1 - e^2)), and C itself carries a rounding error of order
    eps |C| (eps = 2.2e-16) from v^2 / 2 - mu / r, so e^2 has an absolute error of a few eps and the eccentricity
    cannot be resolved below about sqrt(eps) = 1.5e-8: a circular orbit comes out as e up to about 2 sqrt(eps)
    = 3e-8 instead of 0 (measured maximum 3.2e-8 over 60000 random circular orbits, see the test), and for larger e
    the absolute error falls as a few eps / (2 e). No cancellation-free formula exists from the inputs
    (C, h, mu) of this function. (From a position and velocity, the eccentricity vector v x h / mu - r / |r| has
    no such limit: measured 1.5e-15 on the same orbits; it is not implemented here.)

    Verification
    ------------
    tests/unit/test_energy.py: e recovered for e = 0.1 ... 0.95 from the position and velocity at periapsis (random
    orbits), and the sqrt(eps) accuracy limit measured on random circular orbits.
    """
    c = np.asarray(vis_viva_constant_m2_s2, dtype=float)
    h = np.asarray(specific_angular_momentum_m2_s, dtype=float)
    e2 = 1.0 + 2.0 * c * h**2 / gm_m3_s2**2
    if np.any(e2 < -1e-9):
        raise ValueError("inconsistent C and h: e^2 < 0")
    return np.sqrt(np.maximum(e2, 0.0))


@source(
    "MurrayCorreia2010",
    where="TeX label eq24 (vis viva integral)",
    status="read",
    note="derivation in this project: a body at rest at infinity has C = 0 in eq24, so |v| = sqrt(2 mu / r) at "
    "distance r",
)
def escape_speed_m_s(gm_m3_s2: ArrayLike, distance_m: ArrayLike) -> Array:
    """Compute the escape speed from distance r of a mass M: v_esc = sqrt(2 G M / r).

    Parameters
    ----------
    gm_m3_s2 : float or ndarray
        G M [m^3 s^-2] (for two bodies G (m1 + m2) with the relative speed).
    distance_m : float or ndarray
        Distance [m], > 0.

    Returns
    -------
    float or ndarray
        v_esc [m s^-1].

    References
    ----------
    [MurrayCorreia2010] eq24.

    Validity
    --------
    Newtonian point mass or spherically symmetric body, outside the surface. It is sqrt(2) times the circular
    speed at the same distance.

    Verification
    ------------
    tests/unit/test_energy.py: sqrt(2) times the circular speed from kozmos.ephemeris.kepler.orbital_speed_m_s;
    zero total specific energy at v_esc.
    """
    r = np.asarray(distance_m, dtype=float)
    gm = np.asarray(gm_m3_s2, dtype=float)
    if np.any(~(r > 0.0)):
        raise ValueError("distance_m must be positive")
    if np.any(gm < 0.0):
        raise ValueError("gm_m3_s2 must be >= 0 (a negative G M has no escape speed)")
    return np.sqrt(2.0 * gm / r)


@source(
    "Chiang2020",
    where="TeX line 256 (after label eq:Ag): gravitational potential energy of a uniform density sphere "
    "-3 G M^2 / (5 R_p)",
    status="read",
)
def uniform_sphere_binding_energy_j(mass_kg: ArrayLike, radius_m: ArrayLike) -> Array:
    """Compute the gravitational binding energy of a uniform sphere: E_b = 3 G M^2 / (5 R) (positive; W = -E_b).

    Parameters
    ----------
    mass_kg : float or ndarray
        Mass [kg], > 0.
    radius_m : float or ndarray
        Radius [m], > 0.

    Returns
    -------
    float or ndarray
        E_b [J]; the gravitational potential energy of the sphere is -E_b.

    References
    ----------
    [Chiang2020] TeX line 256.

    Validity
    --------
    Newtonian, constant density, isolated sphere. The source prints the potential energy -3GM^2/(5R_p) as the
    value the halo expression reduces to for a uniform sphere; the sign convention here is E_b = -W.

    Verification
    ------------
    tests/unit/test_energy.py: independent oracle by direct numerical integration of the shell energies
    -G M(r) dm / r over thin shells.
    """
    m = np.asarray(mass_kg, dtype=float)
    r = np.asarray(radius_m, dtype=float)
    if np.any(~(m > 0.0)) or np.any(~(r > 0.0)):
        raise ValueError("mass_kg and radius_m must be positive")
    return 0.6 * _G * m**2 / r
