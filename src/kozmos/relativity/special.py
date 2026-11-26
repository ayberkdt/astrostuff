"""Special relativity in SI units.

Conventions: the metric signature is (-, +, +, +) as in Carroll [Carroll1997] Eq. 1.8; a four-vector
carries the time component first and multiplied by c, (c t, x, y, z) for events and (E/c, p_x, p_y, p_z)
for the four-momentum. Carroll sets c = 1; c is restored here by dimensional analysis, which is stated in the
``note`` of every record where it matters.

Equations that appear typeset in the source are cited with their number. Results that the source does not
print (the general boost matrix for an arbitrary direction, the Doppler factor and the aberration formula,
velocity addition) are derived in this project from printed equations, and the ``note`` field says so; each
is verified against an independent computation in tests/unit/test_special_relativity.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.constants import SPEED_OF_LIGHT

Array = NDArray[np.float64]

_C = SPEED_OF_LIGHT.value


def _beta(velocity_m_s: ArrayLike) -> Array:
    """Return v / c as an array and check |v| < c (signed velocities are allowed)."""
    beta = np.asarray(velocity_m_s, dtype=float) / _C
    if np.any(~(np.abs(beta) < 1.0)):
        raise ValueError("speed must be smaller than the speed of light")
    return beta


def _speed_beta(speed_m_s: ArrayLike) -> Array:
    """Return v / c for a speed and check 0 <= v < c (a negative speed raises ValueError)."""
    if np.any(np.asarray(speed_m_s, dtype=float) < 0.0):
        raise ValueError("speed must be non-negative; the direction is set by the angle")
    return _beta(speed_m_s)


def _gamma(velocity_m_s: ArrayLike) -> Array:
    """Lorentz factor c / sqrt((c - v)(c + v)) from the velocity itself.

    c - v is exact in floating point for c/2 <= v <= 2c, so the result is good to rounding however close v is
    to c. Formed from beta = v / c it is not: the rounding of beta is 1e-16, and 1 - beta inherits it as a
    relative error of 1e-16 / (1 - beta) (1.3e-5 at 1 - beta = 1e-12).
    """
    v = np.asarray(velocity_m_s, dtype=float)
    return _C / np.sqrt((_C - v) * (_C + v))


def _vector3(value: ArrayLike, name: str) -> Array:
    v = np.asarray(value, dtype=float)
    if v.shape[-1:] != (3,):
        raise ValueError(f"{name} must have a last axis of length 3")
    return v


def _vector4(value: ArrayLike, name: str) -> Array:
    v = np.asarray(value, dtype=float)
    if v.shape[-1:] != (4,):
        raise ValueError(f"{name} must have a last axis of length 4")
    return v


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.21 (TeX line 566)",
    status="read",
    note="Eq. 1.21 prints gamma = 1/sqrt(1 - v^2) with c = 1; here gamma = 1/sqrt(1 - (v/c)^2)",
)
def lorentz_factor(velocity_m_s: ArrayLike) -> Array:
    """Compute the Lorentz factor: gamma = 1 / sqrt(1 - v^2 / c^2).

    Parameters
    ----------
    velocity_m_s : float or ndarray
        Speed [m s^-1] (the sign is irrelevant), with |v| < c.

    Returns
    -------
    float or ndarray
        gamma (dimensionless), at least 1.

    References
    ----------
    [Carroll1997] Eq. 1.21.

    Validity
    --------
    |v| < c is enforced (ValueError otherwise). gamma is formed as c / sqrt((c - v)(c + v)) from the velocity
    itself, c - v being exact in floating point near c: measured against 60-digit arithmetic, 2e-16 from
    v = 1e-12 m/s to c (1 - 1e-15). (From beta = v / c it lost 1e-16 / (1 - beta): 1.3e-5 at
    1 - beta = 1e-12.) The functions that take a velocity vector (boost_matrix, four_momentum,
    add_velocities_m_s) form 1 - |v|^2 / c^2 from the components and keep that loss: 4e-9 at
    1 - beta = 1e-9.

    Verification
    ------------
    tests/unit/test_special_relativity.py: gamma = 1 + beta^2 / 2 + 3 beta^4 / 8 at low velocity; gamma(0.6 c) =
    1.25 and gamma(0.8 c) = 5/3; gamma = cosh(rapidity); speeds at or above c are rejected.
    """
    _beta(velocity_m_s)  # validation
    return _gamma(velocity_m_s)


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.20 (TeX line 560)",
    status="read",
    note="Eq. 1.20 prints v = tanh(phi) for the boost parameter phi of Eq. 1.18; here phi = artanh(v/c)",
)
def rapidity(velocity_m_s: ArrayLike) -> Array:
    """Compute the rapidity (boost parameter): phi = artanh(v / c).

    Parameters
    ----------
    velocity_m_s : float or ndarray
        Velocity along the boost axis [m s^-1], |v| < c; the sign gives the direction.

    Returns
    -------
    float or ndarray
        phi (dimensionless).

    References
    ----------
    [Carroll1997] Eq. 1.18 (boost matrix with parameter phi) and Eq. 1.20 (v = tanh phi).

    Validity
    --------
    |v| < c is enforced.

    Verification
    ------------
    tests/unit/test_special_relativity.py: rapidities add for collinear boosts; tanh(rapidity) returns v / c;
    boost matrix of Eq. 1.18 reproduced from the rapidity.
    """
    _beta(velocity_m_s)  # validation
    v = np.asarray(velocity_m_s, dtype=float)
    # artanh(v / c) = log1p(2 v / (c - v)) / 2: c - v is exact near c (from beta the result loses 1e-16 / (1 - beta)),
    # and log1p keeps the small velocities (log of the ratio (c + v) / (c - v) loses 1e-16 c / v there)
    # (an odd function, evaluated for |v|: for v near -c the argument of log1p would be near -1)
    speed = np.abs(v)
    return np.sign(v) * 0.5 * np.log1p(2.0 * speed / (_C - speed))


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.20 (TeX line 560)",
    status="read",
    note="inverse of Eq. 1.20: v = c tanh(phi)",
)
def velocity_from_rapidity_m_s(rapidity: ArrayLike) -> Array:
    """Compute the velocity belonging to a rapidity: v = c tanh(phi).

    Parameters
    ----------
    rapidity : float or ndarray
        Rapidity phi (dimensionless).

    Returns
    -------
    float or ndarray
        Velocity [m s^-1].

    References
    ----------
    [Carroll1997] Eq. 1.20.

    Validity
    --------
    Any real rapidity; the result is always smaller than c in magnitude (it can round to exactly c for
    |phi| above about 19).

    Verification
    ------------
    tests/unit/test_special_relativity.py: round trip with rapidity(); never exceeds c.
    """
    return _C * np.tanh(np.asarray(rapidity, dtype=float))


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.18 and Eq. 1.20 (TeX lines 532, 560)",
    status="read",
    note="derivation in this project from Eq. 1.18 and Eq. 1.20: two boosts along one axis multiply as "
    "hyperbolic rotations, phi = phi_1 + phi_2, hence tanh(phi_1 + phi_2) = (t_1 + t_2) / (1 + t_1 t_2); "
    "the formula is not typeset in the source",
)
def add_velocities_collinear_m_s(velocity_m_s: ArrayLike, frame_velocity_m_s: ArrayLike) -> Array:
    """Compose two collinear velocities: w = (v + u) / (1 + v u / c^2).

    An object moves with velocity v in a frame S', and S' moves with velocity u relative to S, both along the
    same axis; w is the velocity of the object in S.

    Parameters
    ----------
    velocity_m_s : float or ndarray
        Velocity v of the object in S' [m s^-1], |v| < c.
    frame_velocity_m_s : float or ndarray
        Velocity u of S' relative to S [m s^-1], |u| < c.

    Returns
    -------
    float or ndarray
        Velocity w of the object in S [m s^-1].

    References
    ----------
    [Carroll1997] Eq. 1.18, Eq. 1.20 (rapidity addition; the composition formula is derived here).

    Validity
    --------
    |v| < c and |u| < c are enforced; then |w| < c mathematically. In floating point |w| can equal c when
    both inputs lie within a few units of 1e-16 of c.

    Verification
    ------------
    tests/unit/test_special_relativity.py: agrees with tanh(rapidity_1 + rapidity_2); |w| <= c for random
    inputs; w = v + u at low velocity; inputs at c are rejected.
    """
    b1 = _beta(velocity_m_s)
    b2 = _beta(frame_velocity_m_s)
    return _C * (b1 + b2) / (1.0 + b1 * b2)


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.98 (TeX line 1671), Eq. 1.11 (TeX line 471), Eq. 1.14 (TeX line 489) and Eq. 1.18 "
    "(TeX line 532)",
    status="read",
    note="derivation in this project: the four-velocity of the object in S' is gamma_v (c, v) (Eq. 1.98 with c "
    "restored); S coordinates are obtained from S' coordinates by the boost of -u; Eq. 1.14 prints only the condition "
    "Lambda^T eta Lambda = eta, from which the inverse Lambda^-1 = eta Lambda^T eta is derived here, and the "
    "boost of -u equals that inverse (Eq. 1.18 with the parameter phi -> -phi); "
    "w = c (spatial part) / (time part)",
)
def add_velocities_m_s(velocity_m_s: ArrayLike, frame_velocity_m_s: ArrayLike) -> Array:
    """Compose two velocities of arbitrary directions by transforming the four-velocity.

    An object moves with velocity v in a frame S', and S' moves with velocity u relative to S; w is the velocity
    of the object in S.

    Parameters
    ----------
    velocity_m_s : ndarray, shape (3,)
        Velocity v of the object in S' [m s^-1], |v| < c.
    frame_velocity_m_s : ndarray, shape (3,)
        Velocity u of S' relative to S [m s^-1], |u| < c.

    Returns
    -------
    ndarray, shape (3,)
        Velocity w of the object in S [m s^-1], |w| < c.

    References
    ----------
    [Carroll1997] Eq. 1.98 (four-velocity), Eq. 1.11 and 1.18 (transformation of a four-vector).

    Validity
    --------
    The operation is not commutative in general for non-collinear velocities (the two orders differ by a Wigner
    rotation of the frames). Accuracy is rounding level (the test file measures it).

    Verification
    ------------
    tests/unit/test_special_relativity.py: agrees with the collinear formula for parallel vectors; satisfies the
    invariant identity gamma_v = gamma_u gamma_w (1 - u.w / c^2); |w| < c for random inputs; adding -u returns v.
    """
    v = _vector3(velocity_m_s, "velocity_m_s")
    u = _vector3(frame_velocity_m_s, "frame_velocity_m_s")
    if v.ndim != 1 or u.ndim != 1:
        raise ValueError("add_velocities_m_s takes single velocity vectors of shape (3,)")
    four_velocity = four_momentum(1.0, v) / _C  # gamma_v (c, v) / c, the scale drops out of the ratio
    in_s = boost_matrix(-u) @ four_velocity
    return _C * in_s[1:] / in_s[0]


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.18, 1.14 and 1.21 (TeX lines 532, 489, 566)",
    status="read",
    note="derivation in this project: Eq. 1.18 is the boost along x; rotating the axis onto the direction of v "
    "gives Lambda^0_0 = gamma, Lambda^0_i = Lambda^i_0 = -gamma beta_i, "
    "Lambda^i_j = delta_ij + gamma^2/(gamma + 1) beta_i beta_j (the source declines to print the general "
    "six-parameter Lorentz matrix and prints only the boost along x); the sign of Eq. 1.21 "
    "(x' = gamma (x - v t)) is kept",
)
def boost_matrix(velocity_m_s: ArrayLike) -> Array:
    """Build the Lorentz boost matrix Lambda^mu'_nu for a frame moving with velocity v.

    The new frame S' moves with velocity v relative to S; coordinates are (c t, x, y, z) and
    x'^mu = Lambda^mu'_nu x^nu. For v along x the matrix is the one of Eq. 1.18 with tanh(phi) = v / c.

    Parameters
    ----------
    velocity_m_s : ndarray, shape (3,)
        Velocity of S' relative to S [m s^-1], |v| < c.

    Returns
    -------
    ndarray, shape (4, 4)
        Lambda^mu'_nu.

    References
    ----------
    [Carroll1997] Eq. 1.14 (defining condition Lambda^T eta Lambda = eta), Eq. 1.18 (boost), Eq. 1.21.

    Validity
    --------
    Pure boost only (no rotation); the composition of two boosts in different directions is in general a boost
    times a rotation (Wigner rotation), so the matrices do not commute. The factor gamma^2 / (gamma + 1) is
    used instead of (gamma - 1) / beta^2 so that v = 0 and small v are exact.

    Verification
    ------------
    tests/unit/test_special_relativity.py: Lambda^T eta Lambda = eta to rounding for random v; equals the
    matrix of Eq. 1.18 along x; inverse is the boost of -v; collinear composition adds rapidities.
    """
    beta = _vector3(velocity_m_s, "velocity_m_s") / _C
    if beta.ndim != 1:
        raise ValueError("boost_matrix takes a single velocity vector of shape (3,)")
    b2 = float(beta @ beta)
    if not b2 < 1.0:
        raise ValueError("speed must be smaller than the speed of light")
    gamma = 1.0 / np.sqrt(1.0 - b2)
    matrix = np.empty((4, 4))
    matrix[0, 0] = gamma
    matrix[0, 1:] = -gamma * beta
    matrix[1:, 0] = -gamma * beta
    matrix[1:, 1:] = np.eye(3) + gamma * gamma / (gamma + 1.0) * np.outer(beta, beta)
    return matrix


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.11 (TeX line 471) and Eq. 1.18",
    status="read",
    note="x'^mu = Lambda^mu'_nu x^nu with Lambda from boost_matrix",
)
def boost_four_vector(four_vector: ArrayLike, velocity_m_s: ArrayLike) -> Array:
    """Transform a four-vector to a frame moving with velocity v: A'^mu = Lambda^mu'_nu A^nu.

    Parameters
    ----------
    four_vector : ndarray, shape (..., 4)
        Components (A^0, A^1, A^2, A^3) with the time component in the same unit as the space components
        (c t for an event [m], E / c for a four-momentum).
    velocity_m_s : ndarray, shape (3,)
        Velocity of the new frame relative to the old one [m s^-1].

    Returns
    -------
    ndarray, shape (..., 4)
        Components in the new frame.

    References
    ----------
    [Carroll1997] Eq. 1.11, Eq. 1.18.

    Validity
    --------
    Pure boost; see boost_matrix.

    Verification
    ------------
    tests/unit/test_special_relativity.py: the interval and the four-momentum norm are invariant for random
    boosts; an event at x' = 0 moves with velocity v; boost followed by the opposite boost returns the input.
    """
    matrix = boost_matrix(velocity_m_s)
    return np.einsum("ij,...j->...i", matrix, _vector4(four_vector, "four_vector"))


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.8 and Eq. 1.9 (TeX lines 447, 452)",
    status="read",
    note="eta = diag(-1, 1, 1, 1); the inner product of two different vectors is the polarisation of Eq. 1.9",
)
def minkowski_inner_product(four_vector_a: ArrayLike, four_vector_b: ArrayLike) -> Array:
    """Compute the Minkowski inner product eta_mu_nu A^mu B^nu = -A^0 B^0 + A . B (signature -+++).

    Parameters
    ----------
    four_vector_a, four_vector_b : ndarray, shape (..., 4)
        Four-vectors with the time component first, in the same unit for all four components.

    Returns
    -------
    float or ndarray
        The inner product, in the square of the unit of the components. Negative for timelike vectors,
        positive for spacelike ones, zero for null vectors.

    References
    ----------
    [Carroll1997] Eq. 1.8, Eq. 1.9.

    Validity
    --------
    Signature (-, +, +, +); many field-theory texts use the opposite sign (Carroll says so after Eq. 1.8).

    Verification
    ------------
    tests/unit/test_special_relativity.py: sign for timelike, spacelike and null vectors; invariance under
    random boosts to rounding; symmetric in its two arguments.
    """
    a = _vector4(four_vector_a, "four_vector_a")
    b = _vector4(four_vector_b, "four_vector_b")
    return -a[..., 0] * b[..., 0] + np.sum(a[..., 1:] * b[..., 1:], axis=-1)


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.9 (TeX line 452)",
    status="read",
    note="s^2 = eta_mu_nu Delta x^mu Delta x^nu with the coordinates (c Delta t, Delta x, Delta y, Delta z)",
)
def spacetime_interval_squared_m2(event_separation_m: ArrayLike) -> Array:
    """Compute the squared spacetime interval: s^2 = -(c dt)^2 + dx^2 + dy^2 + dz^2.

    Parameters
    ----------
    event_separation_m : ndarray, shape (..., 4)
        Separation of two events, (c dt, dx, dy, dz) [m].

    Returns
    -------
    float or ndarray
        s^2 [m^2]; negative for timelike, positive for spacelike separation.

    References
    ----------
    [Carroll1997] Eq. 1.9.

    Validity
    --------
    Flat spacetime, Cartesian inertial coordinates.

    Verification
    ------------
    tests/unit/test_special_relativity.py: invariance under random boosts to rounding; light-like separation
    gives zero.
    """
    d = _vector4(event_separation_m, "event_separation_m")
    return -(d[..., 0] ** 2) + np.sum(d[..., 1:] ** 2, axis=-1)


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.97 (TeX line 1645)",
    status="read",
    note="derivation in this project: for a clock moving with constant speed v, dtau = sqrt(1 - v^2/c^2) dt "
    "from Eq. 1.97, hence Delta t = gamma Delta tau",
)
def dilated_time_s(proper_time_s: ArrayLike, velocity_m_s: ArrayLike) -> Array:
    """Compute the coordinate time that elapses while a moving clock shows a given proper time: gamma tau.

    Parameters
    ----------
    proper_time_s : float or ndarray
        Proper time shown by the moving clock [s].
    velocity_m_s : float or ndarray
        Speed of the clock in the frame of the observer [m s^-1], |v| < c.

    Returns
    -------
    float or ndarray
        Coordinate time [s].

    References
    ----------
    [Carroll1997] Eq. 1.97 (proper time along a worldline).

    Validity
    --------
    Constant speed (inertial clock) in flat spacetime.

    Verification
    ------------
    tests/unit/test_special_relativity.py: equals the time component of the boosted four-vector of the
    event pair (c tau, 0, 0, 0); low-velocity limit tau (1 + beta^2 / 2).
    """
    return np.asarray(proper_time_s, dtype=float) * lorentz_factor(velocity_m_s)


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.97 (TeX line 1645)",
    status="read",
    note="derivation in this project: Delta tau = sqrt(1 - v^2/c^2) Delta t for constant speed",
)
def proper_time_s(coordinate_time_s: ArrayLike, velocity_m_s: ArrayLike) -> Array:
    """Compute the proper time of a moving clock for a given coordinate time: tau = t / gamma.

    Parameters
    ----------
    coordinate_time_s : float or ndarray
        Time elapsed in the frame of the observer [s].
    velocity_m_s : float or ndarray
        Speed of the clock [m s^-1], |v| < c.

    Returns
    -------
    float or ndarray
        Proper time [s].

    References
    ----------
    [Carroll1997] Eq. 1.97.

    Validity
    --------
    Constant speed in flat spacetime; for variable speed integrate Eq. 1.97.

    Verification
    ------------
    tests/unit/test_special_relativity.py: inverse of dilated_time_s; equals the square root of minus the
    squared interval divided by c.
    """
    return np.asarray(coordinate_time_s, dtype=float) / lorentz_factor(velocity_m_s)


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.21 (TeX line 566)",
    status="read",
    note="derivation in this project: two events simultaneous in S at separation L give a separation "
    "L0 = gamma L in S' (x'_1 - x'_2 = gamma (x_1 - x_2) by Eq. 1.21), hence L = L0 / gamma",
)
def contracted_length_m(proper_length_m: ArrayLike, velocity_m_s: ArrayLike) -> Array:
    """Compute the length of a moving rod along its direction of motion: L = L0 / gamma.

    Parameters
    ----------
    proper_length_m : float or ndarray
        Length in the rest frame of the rod [m].
    velocity_m_s : float or ndarray
        Speed of the rod along its length in the frame of the observer [m s^-1], |v| < c.

    Returns
    -------
    float or ndarray
        Length measured by the observer [m].

    References
    ----------
    [Carroll1997] Eq. 1.21.

    Validity
    --------
    Rod moving along its own axis; lengths transverse to the motion are unchanged.

    Verification
    ------------
    tests/unit/test_special_relativity.py: agrees with the end points of a rod transformed by the boost matrix
    and read at equal time in the observer frame; low-velocity limit L0 (1 - beta^2 / 2).
    """
    return np.asarray(proper_length_m, dtype=float) / lorentz_factor(velocity_m_s)


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.14, 1.18 and 1.21 (TeX lines 489, 532, 566)",
    status="read",
    note="derivation in this project from the boost of a null four-vector k = (omega/c)(1, n): with the boost "
    "of boost_matrix, omega_obs = omega_src * gamma (1 + beta cos(theta')) and equivalently "
    "omega_obs = omega_src / (gamma (1 - beta cos(theta))); the formula is not typeset in the source",
)
def doppler_factor(velocity_m_s: ArrayLike, angle_rad: ArrayLike) -> Array:
    """Compute the relativistic Doppler factor nu_obs / nu_src = 1 / (gamma (1 - beta cos(theta))).

    theta is the angle, in the frame of the observer, between the velocity of the source and the direction in
    which the photon travels from the source to the observer. theta = 0: the source approaches along the line
    of sight; theta = pi: it recedes.

    Parameters
    ----------
    velocity_m_s : float or ndarray
        Speed of the source in the observer frame [m s^-1], 0 <= v < c; a negative value raises ValueError.
    angle_rad : float or ndarray
        Angle theta between the source velocity and the photon direction, observer frame [rad].

    Returns
    -------
    float or ndarray
        Ratio of received to emitted frequency (dimensionless).

    References
    ----------
    [Carroll1997] Eq. 1.14, 1.18 (Lorentz transformation of a four-vector); the formula is derived here.

    Validity
    --------
    Flat spacetime, free source (constant velocity), photon propagating in vacuum. For theta = pi/2 in the
    observer frame the factor is 1 / gamma (transverse redshift); for theta' = pi/2 in the source frame it is
    gamma, see doppler_factor_source_frame.

    Verification
    ------------
    tests/unit/test_special_relativity.py: compared with the frequency (time component) of a null four-vector
    boosted with boost_matrix for random 3D velocities; longitudinal limits sqrt((1 + beta)/(1 - beta)) and
    its inverse; transverse limit 1 / gamma; beta -> 0 limit 1 + beta cos(theta).
    """
    _speed_beta(velocity_m_s)  # validation
    # 1 - beta cos(theta) = [(c - v) + 2 v sin^2(theta / 2)] / c, without the rounding of beta
    v = np.asarray(velocity_m_s, dtype=float)
    half = 0.5 * np.asarray(angle_rad, dtype=float)
    return _C / (_gamma(v) * ((_C - v) + 2.0 * v * np.sin(half) ** 2))


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.14, 1.18 and 1.21 (TeX lines 489, 532, 566)",
    status="read",
    note="derivation in this project from the boost of a null four-vector, see doppler_factor: "
    "omega_obs = omega_src gamma (1 + beta cos(theta')) with theta' measured in the source frame",
)
def doppler_factor_source_frame(velocity_m_s: ArrayLike, angle_source_rad: ArrayLike) -> Array:
    """Compute nu_obs / nu_src = gamma (1 + beta cos(theta')) with the emission angle in the source frame.

    theta' is the angle, in the rest frame of the source, between the direction of motion of the source (as
    seen by the observer) and the direction in which the photon is emitted.

    Parameters
    ----------
    velocity_m_s : float or ndarray
        Speed of the source in the observer frame [m s^-1], 0 <= v < c; a negative value raises ValueError.
    angle_source_rad : float or ndarray
        Emission angle theta' in the source frame [rad].

    Returns
    -------
    float or ndarray
        Ratio of received to emitted frequency (dimensionless).

    References
    ----------
    [Carroll1997] Eq. 1.14, 1.18.

    Validity
    --------
    Same as doppler_factor. Emission at theta' = pi/2 in the source frame gives gamma (blueshift).

    Verification
    ------------
    tests/unit/test_special_relativity.py: compared with the frequency (time component) of a null four-vector
    boosted with boost_matrix for random 3D velocities; transverse limit gamma for theta' = pi/2.
    """
    _speed_beta(velocity_m_s)  # validation
    # 1 + beta cos(theta') = [(c - v) + 2 v cos^2(theta' / 2)] / c
    v = np.asarray(velocity_m_s, dtype=float)
    half = 0.5 * np.asarray(angle_source_rad, dtype=float)
    return _gamma(v) * ((_C - v) + 2.0 * v * np.cos(half) ** 2) / _C


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.14, 1.18 and 1.21 (TeX lines 489, 532, 566)",
    status="read",
    note="derivation in this project from the boost of a null four-vector: the spatial components give "
    "omega_obs cos(theta) = gamma omega_src (cos(theta') + beta) and omega_obs sin(theta) = omega_src "
    "sin(theta'), hence tan(theta) = sin(theta') / (gamma (cos(theta') + beta)); not typeset in the source",
)
def aberrated_angle_rad(velocity_m_s: ArrayLike, angle_source_rad: ArrayLike) -> Array:
    """Compute the direction of a photon in the observer frame from its direction in the source frame.

    theta = atan2(sin(theta'), gamma (cos(theta') + beta)), both angles measured from the direction of motion
    of the source (as seen by the observer) to the direction of the photon momentum.

    Parameters
    ----------
    velocity_m_s : float or ndarray
        Speed of the source in the observer frame [m s^-1], 0 <= v < c; a negative value raises ValueError.
    angle_source_rad : float or ndarray
        Angle theta' in the source frame [rad].

    Returns
    -------
    float or ndarray
        Angle theta in the observer frame [rad], in [0, pi] for theta' in [0, pi].

    References
    ----------
    [Carroll1997] Eq. 1.14, 1.18.

    Validity
    --------
    Same as doppler_factor. The photon is beamed forward: theta < theta' for 0 < theta' < pi and beta > 0.

    Verification
    ------------
    tests/unit/test_special_relativity.py: compared with the direction of a null four-vector boosted with
    boost_matrix for random 3D velocities; theta = theta' for beta = 0; theta' = pi/2 gives
    cos(theta) = beta.
    """
    beta = np.asarray(_speed_beta(velocity_m_s))
    gamma = _gamma(velocity_m_s)
    theta_source = np.asarray(angle_source_rad, dtype=float)
    return np.arctan2(np.sin(theta_source), gamma * (np.cos(theta_source) + beta))


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.100 and Eq. 1.101 (TeX lines 1686, 1707)",
    status="read",
    note="p^mu = m U^mu with U^mu = gamma (c, v) (Eq. 1.98 with c restored); the time component is E / c, "
    "so p^mu = gamma m (c, v)",
)
def four_momentum(mass_kg: ArrayLike, velocity_m_s: ArrayLike) -> Array:
    """Compute the energy-momentum four-vector p^mu = gamma m (c, v), components (E / c, p_x, p_y, p_z).

    Parameters
    ----------
    mass_kg : float or ndarray
        Rest mass [kg].
    velocity_m_s : ndarray, shape (..., 3)
        Velocity vector [m s^-1], |v| < c.

    Returns
    -------
    ndarray, shape (..., 4)
        (E / c, p_x, p_y, p_z) [kg m s^-1].

    References
    ----------
    [Carroll1997] Eq. 1.100, Eq. 1.101.

    Validity
    --------
    Massive particle, |v| < c. A photon has no rest-frame velocity; build its four-momentum as
    (E / c)(1, n) directly.

    Verification
    ------------
    tests/unit/test_special_relativity.py: p . p = -(m c)^2 to rounding; the boosted four-momentum of a
    particle at rest equals four_momentum of the boost velocity; E = m c^2 at rest.
    """
    v = _vector3(velocity_m_s, "velocity_m_s")
    beta = v / _C
    b2 = np.sum(beta * beta, axis=-1)
    if np.any(~(b2 < 1.0)):
        raise ValueError("speed must be smaller than the speed of light")
    gamma = 1.0 / np.sqrt(1.0 - b2)
    m = np.asarray(mass_kg, dtype=float)
    time_part = (gamma * m * _C)[..., None]
    space_part = (gamma * m)[..., None] * v
    return np.concatenate([time_part, space_part], axis=-1)


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.100 and Eq. 1.101 (TeX lines 1686, 1707)",
    status="read",
    note="E = p^0 c = gamma m c^2 (the text says the energy is p^0 and that p^0 = m in the rest frame, E = m c^2)",
)
def total_energy_j(mass_kg: ArrayLike, velocity_m_s: ArrayLike) -> Array:
    """Compute the total energy: E = gamma m c^2.

    Parameters
    ----------
    mass_kg : float or ndarray
        Rest mass [kg].
    velocity_m_s : float or ndarray
        Speed [m s^-1], |v| < c.

    Returns
    -------
    float or ndarray
        Total energy including rest energy [J].

    References
    ----------
    [Carroll1997] Eq. 1.100, 1.101.

    Validity
    --------
    Massive particle in flat spacetime.

    Verification
    ------------
    tests/unit/test_special_relativity.py: E = m c^2 at rest; equals c times the time component of
    four_momentum; E^2 = (p c)^2 + (m c^2)^2.
    """
    return np.asarray(mass_kg, dtype=float) * _C**2 * lorentz_factor(velocity_m_s)


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.101 (TeX line 1707)",
    status="read",
    note="the text expands p^0 = m + m v^2 / 2 for small v and calls the second term the kinetic energy; "
    "the exact kinetic energy (gamma - 1) m c^2 is the difference of Eq. 1.101's p^0 and the rest energy",
)
def kinetic_energy_j(mass_kg: ArrayLike, velocity_m_s: ArrayLike) -> Array:
    """Compute the kinetic energy: T = (gamma - 1) m c^2.

    Parameters
    ----------
    mass_kg : float or ndarray
        Rest mass [kg].
    velocity_m_s : float or ndarray
        Speed [m s^-1], |v| < c.

    Returns
    -------
    float or ndarray
        Kinetic energy [J].

    References
    ----------
    [Carroll1997] Eq. 1.101 and the text after it (small-v expansion p^0 = m + m v^2 / 2).

    Validity
    --------
    Evaluated as m c^2 gamma^2 beta^2 / (gamma + 1), which has no cancellation at low velocity.

    Verification
    ------------
    tests/unit/test_special_relativity.py: tends to m v^2 / 2 for beta -> 0 (relative difference 3 beta^2 / 4);
    equals total_energy_j minus m c^2 at high velocity.
    """
    beta = np.asarray(_beta(velocity_m_s))
    gamma = _gamma(velocity_m_s)
    return np.asarray(mass_kg, dtype=float) * _C**2 * gamma**2 * beta**2 / (gamma + 1.0)


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.100 and Eq. 1.101 (TeX lines 1686, 1707)",
    status="read",
    note="the text identifies the space components of p^mu with the momentum, p^1 = gamma m v; "
    "here as a vector, p = gamma m v",
)
def relativistic_momentum(mass_kg: ArrayLike, velocity_m_s: ArrayLike) -> Array:
    """Compute the momentum vector: p = gamma m v.

    Parameters
    ----------
    mass_kg : float or ndarray
        Rest mass [kg].
    velocity_m_s : ndarray, shape (..., 3)
        Velocity vector [m s^-1], |v| < c.

    Returns
    -------
    ndarray, shape (..., 3)
        Momentum [kg m s^-1].

    References
    ----------
    [Carroll1997] Eq. 1.101.

    Validity
    --------
    Massive particle in flat spacetime.

    Verification
    ------------
    tests/unit/test_special_relativity.py: equals the space part of four_momentum; low-velocity limit m v.
    """
    return four_momentum(mass_kg, velocity_m_s)[..., 1:]


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.99 and Eq. 1.100 (TeX lines 1676, 1686)",
    status="read",
    note="derivation in this project: Eq. 1.99 gives eta U U = -1, so with p = m U (Eq. 1.100) "
    "-(p^0)^2 + |p|^2 = -m^2 in c = 1 units, i.e. E^2 = (p c)^2 + (m c^2)^2",
)
def energy_from_momentum_j(mass_kg: ArrayLike, momentum: ArrayLike) -> Array:
    """Compute the total energy from the momentum magnitude: E = sqrt((p c)^2 + (m c^2)^2).

    Parameters
    ----------
    mass_kg : float or ndarray
        Rest mass [kg]; zero for a photon.
    momentum : float or ndarray
        Magnitude of the momentum [kg m s^-1].

    Returns
    -------
    float or ndarray
        Total energy [J].

    References
    ----------
    [Carroll1997] Eq. 1.99, Eq. 1.100.

    Validity
    --------
    Flat spacetime. np.hypot of the two terms is used, so there is no overflow for extreme values.

    Verification
    ------------
    tests/unit/test_special_relativity.py: agrees with gamma m c^2 for random velocities; E = p c for
    m = 0; E = m c^2 at p = 0.
    """
    p = np.abs(np.asarray(momentum, dtype=float))
    return np.hypot(p * _C, np.asarray(mass_kg, dtype=float) * _C**2)


@source(
    "Carroll1997",
    where="Section 1, Eq. 1.99 and Eq. 1.100 (TeX lines 1676, 1686)",
    status="read",
    note="derivation in this project: the four-momenta add, and the norm of the sum, "
    "eta P P = -(M c)^2, defines the invariant mass M (Eq. 1.99 and 1.100 for a single particle)",
)
def invariant_mass_kg(four_momenta: ArrayLike) -> Array:
    """Compute the invariant mass of a system of particles: M c = sqrt(-eta_mu_nu P^mu P^nu), P = sum p_i.

    Parameters
    ----------
    four_momenta : ndarray, shape (..., N, 4)
        Four-momenta (E / c, p_x, p_y, p_z) of the N particles in one frame [kg m s^-1].

    Returns
    -------
    float or ndarray
        Invariant mass of the system [kg].

    References
    ----------
    [Carroll1997] Eq. 1.99, Eq. 1.100.

    Validity
    --------
    The sum must be timelike or null. A sum that is spacelike beyond rounding raises ValueError; a slightly
    negative value of -P.P, smaller in magnitude than 1e-12 of (P^0)^2, is set to zero (massless system,
    for example two collinear photons).

    Verification
    ------------
    tests/unit/test_special_relativity.py: two photons give M c^2 = sqrt(2 E1 E2 (1 - cos(angle))); invariant
    under a random boost of all particles; a single particle returns its rest mass.
    """
    p = _vector4(four_momenta, "four_momenta")
    if p.ndim < 2:
        raise ValueError("four_momenta must have shape (..., N, 4)")
    total = np.sum(p, axis=-2)
    s = -minkowski_inner_product(total, total)
    scale = total[..., 0] ** 2
    if np.any(s < -1e-12 * scale):
        raise ValueError("the total four-momentum is spacelike")
    return np.sqrt(np.maximum(s, 0.0)) / _C
