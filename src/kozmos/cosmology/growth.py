"""Linear growth of matter perturbations on the w0-wa background.

The growth equation is the one of [Linder2005] (arXiv astro-ph/0507263, TeX label ``eq.groq``) written for
g = delta/a with derivatives with respect to ln a. The background (deceleration parameter q and Omega_m(a))
comes from ``kozmos.cosmology.dark_energy``. General relativity, pressureless matter, all other
components smooth, scales well inside the Hubble radius ([Linder2005] Eq. grot; [Frieman2008] Eq. growth).

Normalizations used below:

* ``growth_suppression`` returns g(z) with g -> 1 deep in matter domination (the normalization of
  [Linder2005] Eq. gapx), so g = 1 for an Einstein-de Sitter universe at all times;
* ``growth_factor`` returns D(z) = a g(a) / g(1), the growing mode normalized to 1 today.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from kozmos._sources import source
from kozmos.cosmology.dark_energy import _e_squared, de_eos, deceleration_parameter, omega_matter
from kozmos.cosmology.distances import _NODES, _WEIGHTS, Cosmology

Array = NDArray[np.float64]

_Z_START = 1000.0
_RTOL = 1e-10
_A_FIT_START = 1e-4  # lower limit of the Eq. gapx integral; the integrand is O(a^3) there
_BOUNCE_SCAN_POINTS = 4001  # grid points in ln(1+z) of the E^2 > 0 check before the integration


def _require_positive_e_squared(z: ArrayLike, cosmology: Cosmology) -> None:
    """Raise ``ValueError`` if E(z)^2 is not positive (or not finite) at any of the redshifts z."""
    e2 = _e_squared(z, cosmology)
    if not np.all(e2 > 0.0):
        bad = np.atleast_1d(np.asarray(z, dtype=float))[np.atleast_1d(~(e2 > 0.0))]
        raise ValueError(
            f"E(z)^2 <= 0 at z = {float(bad[0]):.6g}: the model bounces (no real expansion rate) inside the "
            "integration range, so the growth equation has no solution there"
        )


@lru_cache(maxsize=64)
def _solve(cosmology: Cosmology, z_start: float, rtol: float):  # type: ignore[no-untyped-def]
    """Dense solution (g, dg/dln a) of Linder's Eq. groq from z_start to today.

    Raises ``ValueError`` if E^2 <= 0 anywhere on [0, z_start] (checked on a grid of 4001 points in ln(1+z) and
    at every right-hand-side evaluation of the solver).
    """
    _require_positive_e_squared(np.expm1(np.linspace(0.0, np.log1p(z_start), _BOUNCE_SCAN_POINTS)), cosmology)

    def rhs(n: float, y: Array) -> list[float]:
        z = np.expm1(-n)
        q = float(deceleration_parameter(z, cosmology))
        if not np.isfinite(q):  # NaN exactly where E^2 <= 0
            _require_positive_e_squared(z, cosmology)
            raise ValueError(f"the deceleration parameter is not finite at z = {float(z):.6g}")
        om = float(omega_matter(z, cosmology))
        return [y[1], -(3.0 - q) * y[1] - (2.0 - q - 1.5 * om) * y[0]]

    n_start = -np.log1p(z_start)
    return solve_ivp(rhs, (n_start, 0.0), [1.0, 0.0], method="DOP853", rtol=rtol, atol=1e-12, dense_output=True)


def _evaluate(z: ArrayLike, cosmology: Cosmology, z_start: float, rtol: float) -> tuple[Array, Array, Array]:
    """Return a, g, dg/dln a at the redshifts z (0 <= z <= z_start); z may have any shape."""
    zz = np.asarray(z, dtype=float)
    if np.any(zz < 0) or np.any(zz > z_start):
        raise ValueError(f"z must lie in [0, z_start] = [0, {z_start}]")
    solution = _solve(cosmology, float(z_start), float(rtol))
    n = -np.log1p(zz)
    y = solution.sol(n.reshape(-1))  # the dense output accepts only scalars or 1-D arrays
    return np.exp(n), y[0].reshape(zz.shape)[()], y[1].reshape(zz.shape)[()]


@source(
    "Linder2005",
    where="Eq. groq (groexpprdv2.tex lines 215-216) with g = delta/a (line 204), primes = d/dln a (line 210)",
    status="read",
    note="G/G_Newton = 1; the equations groh, groq, grow, groom are stated to be equivalent (line 224)",
)
@source(
    "Frieman2008", where="Eq. growth (sec2.tex lines 300-303), solution in matter domination delta ∝ a", status="read"
)
def growth_suppression(z: ArrayLike, cosmology: Cosmology, z_start: float = _Z_START, rtol: float = _RTOL) -> Array:
    """Compute g(z) = delta/a for the growing mode, normalized to g = 1 at the start of the integration.

    The ODE g'' + (3 - q) g' + [2 - q - (3/2) Omega_m(a)] g = 0 (primes: d/dln a) is integrated with
    initial conditions g = 1, g' = 0 at z_start, where the growing mode of a matter-dominated universe is
    delta ∝ a.

    Parameters
    ----------
    z : float or ndarray
        Redshift, 0 <= z <= z_start.
    cosmology : Cosmology
    z_start : float
        Redshift of the initial conditions, default 1000.
    rtol : float
        Relative tolerance of the ODE solver, default 1e-10.

    Returns
    -------
    float or ndarray
        g(z), dimensionless; 1 for Einstein-de Sitter at all z.

    References
    ----------
    [Linder2005] Eq. groq; [Frieman2008] Eq. growth.

    Validity
    --------
    Raises ``ValueError`` if E(z)^2 <= 0 anywhere on [0, z_start] (a bouncing model such as
    ``Cosmology(70, 0.1, 1.5)``, for which the solver formerly returned D = [1, 0.013, -9e-89] at z = [0, 1, 5]);
    ``z`` may have any shape. Initial conditions assume matter domination at z_start. Measured change of D(z) for 0 <= z <= 10 when
    z_start moves from 1000 to the reference 5000 (tests/unit/test_growth.py): 4.5e-11 for flat lambda-CDM,
    3.2e-9 for w0 = -0.8, wa = 0.5 (the dark energy is not negligible early), 1.0e-6 with omega_r = 8.5e-5
    (Omega_r(z)/Omega_m(z) = 0.28 at z = 1000, so the conditions are only approximately matter dominated and
    the error does not vanish as z_start grows until radiation is treated in the initial conditions).
    Changing rtol from 1e-10 to 1e-13 changes D by 6e-12 (lambda-CDM). Radiation is part of the background (it
    enters q and Omega_m) but the Meszaros effect before z_start is not integrated. No dark energy
    perturbations (smooth dark energy), no massive neutrinos, sub-horizon scales only.

    Verification
    ------------
    tests/unit/test_growth.py: Einstein-de Sitter g = 1, the lambda-CDM integral solution (D differs by 6.1e-12),
    an independent integration of Linder's Eq. grow for a w0-wa model (1.4e-12), independent integrations of the
    equation in a for curved, phantom and radiation models, insensitivity to z_start and rtol, ValueError for a
    bouncing model, 2-D redshift arrays.
    """
    _, g, _ = _evaluate(z, cosmology, z_start, rtol)
    return g


@source(
    "Linder2005",
    where="Eq. groq (groexpprdv2.tex lines 215-216) with g = delta/a (line 204)",
    status="read",
    note="derivation in this project: D = delta/delta(today) = a g(a)/g(1)",
)
def growth_factor(z: ArrayLike, cosmology: Cosmology, z_start: float = _Z_START, rtol: float = _RTOL) -> Array:
    """Compute the linear growth factor D(z) = delta(z)/delta(0) = a g(a) / g(1) (D = 1 today).

    Parameters
    ----------
    z : float or ndarray
        Redshift, 0 <= z <= z_start.
    cosmology : Cosmology
    z_start : float
        Redshift of the initial conditions, default 1000.
    rtol : float
        Relative tolerance of the ODE solver, default 1e-10.

    Returns
    -------
    float or ndarray
        D(z), dimensionless; 1 at z = 0, and (1+z)^-1 for Einstein-de Sitter.

    References
    ----------
    [Linder2005] Eq. groq and the definition g = delta/a.

    Validity
    --------
    See ``growth_suppression``.

    Verification
    ------------
    tests/unit/test_growth.py: Einstein-de Sitter D = a, lambda-CDM integral closed form.
    """
    a, g, _ = _evaluate(z, cosmology, z_start, rtol)
    g0 = _evaluate(0.0, cosmology, z_start, rtol)[1]
    return a * g / g0


@source(
    "Linder2005",
    where="Eq. gapx (groexpprdv2.tex line 341): g(a) = exp ∫ dln a [Omega_m(a)^gamma - 1], g = delta/a (line 204)",
    status="read",
    note="derivation in this project: f = dln(delta)/dln(a) = 1 + dln(g)/dln(a) = 1 + g'/g, "
    "evaluated from the ODE solution",
)
def growth_log_slope(z: ArrayLike, cosmology: Cosmology, z_start: float = _Z_START, rtol: float = _RTOL) -> Array:
    """Compute the logarithmic growth slope f = dln(delta)/dln(a) = 1 + g'/g from the ODE solution.

    Parameters
    ----------
    z : float or ndarray
        Redshift, 0 <= z <= z_start.
    cosmology : Cosmology
    z_start : float
        Redshift of the initial conditions, default 1000.
    rtol : float
        Relative tolerance of the ODE solver, default 1e-10.

    Returns
    -------
    float or ndarray
        f(z), dimensionless; 1 for Einstein-de Sitter.

    References
    ----------
    [Linder2005] Eq. gapx and g = delta/a: f = Omega_m^gamma is equivalent to the integrand of Eq. gapx.

    Validity
    --------
    See ``growth_suppression``. f is the derivative of the dense output of the solver, which has the
    same relative accuracy as g' itself (the derivative is a component of the ODE state).

    Verification
    ------------
    tests/unit/test_growth.py: Einstein-de Sitter f = 1, the lambda-CDM integral solution (difference 3.6e-11) and
    the finite difference of ln D (difference 1.7e-9).
    """
    _, g, dg = _evaluate(z, cosmology, z_start, rtol)
    return 1.0 + dg / g


@source(
    "Linder2005",
    where="Eq. gamma (groexpprdv2.tex line 366): gamma = 0.55 + 0.05 [1 + w(z=1)]; Eq. gammam1 (line 386): "
    "gamma = 0.55 + 0.02 [1 + w(z=1)] for w < -1",
    status="read",
    note="w(z=1) = w0 + wa/2 from w(a) = w0 + wa (1-a) ([Linder2003] Eq. wa); the paper states the fit for "
    "constant w and for the SUGRA model (w0=-0.82, wa=0.58, 0.25 percent accuracy)",
)
def growth_index_gamma(cosmology: Cosmology, phantom_variant: bool = False) -> float:
    """Compute the growth index of Linder's fit: gamma = 0.55 + 0.05 [1 + w(z=1)].

    Parameters
    ----------
    cosmology : Cosmology
        w(z=1) = w + wa/2 is evaluated with the CPL form.
    phantom_variant : bool
        If True and 1 + w(z=1) < 0, use the coefficient 0.02 instead of 0.05 (Eq. gammam1).

    Returns
    -------
    float
        gamma, dimensionless (0.55 for a cosmological constant).

    References
    ----------
    [Linder2005] Eq. gamma and Eq. gammam1; [Linder2003] w(a) (for w(z=1) of the CPL model).

    Validity
    --------
    Stated range of the source: Omega_m in [0.2, 1], flat universe, general relativity, accuracy 0.05 percent
    for a cosmological constant, 0.2 percent for w = -0.8, 0.4 percent for w = -0.5, 0.3 percent (0.5 percent)
    for w = -1.2 (-1.5), 0.25 percent for w0 = -0.82, wa = 0.58 (the source's SUGRA background). Measured here,
    for g over 0 <= z <= 100 against the ODE solution: 0.050 percent (lambda-CDM, Omega_m = 0.22), 0.19 percent
    (w = -0.8), 0.43 percent (w = -0.5, above the stated 0.4), 0.26 percent (w = -1.2), 0.51 percent
    (w = -1.5, above the stated 0.5), 0.052 and 0.036 percent with Eq. gammam1, 0.41 percent for the CPL model
    with the SUGRA parameters (a different background from the source's SUGRA solution, so not a test of the
    0.25 percent claim).
    The fit of f itself is less accurate (0.57 percent for lambda-CDM, Omega_m = 0.3). Radiation is not part of the
    source's background.

    Verification
    ------------
    tests/unit/test_growth.py: gamma values and the accuracy of the fit against the ODE solution.
    """
    w1 = float(de_eos(1.0, cosmology))
    coefficient = 0.02 if (phantom_variant and 1.0 + w1 < 0.0) else 0.05
    return 0.55 + coefficient * (1.0 + w1)


@source(
    "Linder2005",
    where="Eq. gapx (groexpprdv2.tex line 341) and f = 1 + dln g/dln a",
    status="read",
    note="derivation in this project: the integrand of Eq. gapx is Omega_m(a)^gamma - 1 = dln g/dln a, so "
    "f = Omega_m(a)^gamma",
)
def growth_log_slope_from_index(z: ArrayLike, cosmology: Cosmology, gamma: float | None = None) -> Array:
    """Compute the growth-index approximation of the logarithmic growth slope: f = Omega_m(z)^gamma.

    Parameters
    ----------
    z : float or ndarray
        Redshift, z >= 0.
    cosmology : Cosmology
    gamma : float, optional
        Growth index; default ``growth_index_gamma(cosmology)``.

    Returns
    -------
    float or ndarray
        f(z) approximation, dimensionless.

    References
    ----------
    [Linder2005] Eq. gapx.

    Validity
    --------
    See ``growth_index_gamma``.

    Verification
    ------------
    tests/unit/test_growth.py: difference from the ODE solution over Omega_m in [0.22, 1].
    """
    if gamma is None:
        gamma = growth_index_gamma(cosmology)
    return omega_matter(z, cosmology) ** gamma


@source(
    "Linder2005",
    where="Eq. gapx (groexpprdv2.tex line 341): g(a) = exp ∫_0^a dln a [Omega_m(a)^gamma - 1]",
    status="read",
)
def growth_suppression_from_index(z: ArrayLike, cosmology: Cosmology, gamma: float | None = None) -> Array:
    """Compute g(z) = delta/a of the growth-index fit: g(a) = exp ∫_0^a dln a' [Omega_m(a')^gamma - 1].

    The lower limit is a = 1e-4 (the integrand is O(a^3) there, so the neglected part is below 1e-11); the integral
    is done with one 64-point Gauss-Legendre rule in ln a (at most 9.2 e-folds wide).

    Parameters
    ----------
    z : float or ndarray
        Redshift, 0 <= z <= 1e4 - 1; any shape.
    cosmology : Cosmology
    gamma : float, optional
        Growth index; default ``growth_index_gamma(cosmology)``.

    Returns
    -------
    float or ndarray
        g(z), dimensionless, normalized to 1 at a -> 0 like ``growth_suppression``.

    References
    ----------
    [Linder2005] Eq. gapx.

    Validity
    --------
    The source gives Eq. gapx without radiation; with ``omega_r > 0`` Omega_m(a) does not tend to 1 at small a
    and the fit loses its meaning. See ``growth_index_gamma`` for the stated accuracy. NaN when E(a)^2 <= 0 anywhere between a = 1e-4 and
    the requested redshift (the path of the integral crosses a bounce).
    One 64-point rule over at most 9.2 e-folds reproduces a tight scipy quadrature to 4.9e-15 (largest of nine
    flat, curved, phantom, w0-wa and radiation models, z up to 9999; the former 10 panels gave 3.3e-16), so
    panels are not needed; the lower limit a = 1e-4, not the quadrature, limits the accuracy.

    Verification
    ------------
    tests/unit/test_growth.py: difference from the ODE solution over the range stated by the source; equality with
    a scipy quadrature of the integral for several models; 2-D redshift arrays.
    """
    if gamma is None:
        gamma = growth_index_gamma(cosmology)
    zz = np.asarray(z, dtype=float)
    if np.any(zz < 0) or np.any(zz > 1.0 / _A_FIT_START - 1.0):
        raise ValueError("z must lie in [0, 1/1e-4 - 1]")

    n_low = np.log(_A_FIT_START)
    half = 0.5 * (-np.log1p(zz.reshape(-1)) - n_low)  # half-width of the interval in N = ln a, shape (m,)
    nodes = n_low + half[:, None] * (_NODES + 1.0)  # shape (m, 64)
    integral = half * np.sum(_WEIGHTS * (omega_matter(np.expm1(-nodes), cosmology) ** gamma - 1.0), axis=-1)
    return np.exp(integral).reshape(zz.shape)[()]


@source(
    "Linder2005",
    where="Eq. grot (groexpprdv2.tex line 192): the growth equation is linear and scale independent",
    status="read",
    note="derivation in this project: every mode obeys the same linear equation, so sigma_8(z) = sigma_8(0) D(z) and "
    "f sigma_8(z) = f(z) sigma_8(0) D(z)",
)
def fsigma8(
    z: ArrayLike, cosmology: Cosmology, sigma8_today: float, z_start: float = _Z_START, rtol: float = _RTOL
) -> Array:
    """Compute f sigma_8(z) = f(z) sigma_8(0) D(z) for a linear matter power spectrum.

    Parameters
    ----------
    z : float or ndarray
        Redshift, 0 <= z <= z_start.
    cosmology : Cosmology
    sigma8_today : float
        rms linear matter fluctuation in 8 Mpc/h spheres today.
    z_start : float
        Redshift of the initial conditions, default 1000.
    rtol : float
        Relative tolerance of the ODE solver, default 1e-10.

    Returns
    -------
    float or ndarray
        f sigma_8(z), dimensionless.

    References
    ----------
    [Linder2005] Eq. grot (linear, scale-independent growth equation).

    Validity
    --------
    Scale-independent growth only: general relativity with smooth dark energy and no massive neutrinos.
    The product f sigma_8 is a definition; the sources read in this project do not print it.

    Verification
    ------------
    tests/unit/test_growth.py: Einstein-de Sitter f sigma_8 = sigma_8(0)/(1+z) and equality with the product of
    the other functions of this module.
    """
    return growth_log_slope(z, cosmology, z_start, rtol) * sigma8_today * growth_factor(z, cosmology, z_start, rtol)
