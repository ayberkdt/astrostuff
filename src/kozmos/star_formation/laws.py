"""Star formation laws on galactic scales and rate calibrations from luminosity.

The equations were read from the arXiv TeX sources (25 November 2026): Kennicutt
(1998, astro-ph/9712213; equation numbers are the order in the text of the paper),
Kennicutt and Evans (2012, arXiv:1204.3552, Table 1), and McKee and Ostriker
(2007, arXiv:0707.3514; TeX labels).

Units are the real units used by astronomers and are written in the argument
names: surface density in M☉ pc⁻², star formation surface density in
M☉ yr⁻¹ kpc⁻², angular velocity in km s⁻¹ kpc⁻¹. Luminosity is taken in W (1 erg/s = 10⁻⁷ W
is an exact factor).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source
from kozmos.core.constants import GM_SUN, GRAVITATIONAL_CONSTANT, JULIAN_YEAR_SECONDS, PARSEC

Array = NDArray[np.float64]

# Solar mass: GM_☉ / G; GM_☉ is the nominal IAU value, G is the CODATA value.
_SOLAR_MASS_KG = GM_SUN.value / GRAVITATIONAL_CONSTANT.value
_PC_M = PARSEC.value
_YEAR_S = JULIAN_YEAR_SECONDS.value

# Kennicutt and Evans (2012) Table 1: log C_x (Kroupa IMF, Starburst99, solar metallicity).
# L_x units: erg/s for FUV, NUV, TIR, 24 µm, 70 µm, 2-10 keV; erg/s for Hα;
# erg/s/Hz for 1.4 GHz.
_LOG_C = {
    "FUV": 43.35,
    "NUV": 43.17,
    "Halpha": 41.27,
    "TIR": 43.41,
    "24um": 42.69,
    "70um": 43.23,
    "1.4GHz": 28.20,
    "2-10keV": 39.77,
}


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
    "Kennicutt1998",
    where="Eq. 4 (order in the text; referred to as 'equation (4)')",
    status="read",
    note="Source: Σ_SFR = (2.5 ± 0.7)e-4 (Σ_gas / 1 M☉ pc⁻²)^{1.4 ± 0.15} M☉ yr⁻¹ kpc⁻²; "
    "McKee2007 TeX label eq:ks quotes the same value. According to Kennicutt2012, this paper assumes a Salpeter IMF (0.1-100 M☉)",
)
@source("McKee2007", where="TeX label eq:ks", status="read")
def kennicutt_schmidt_sfr_surface_density(
    gas_surface_density_msun_pc2: ArrayLike, index: float = 1.4, amplitude: float = 2.5e-4
) -> Array:
    """Compute the Kennicutt-Schmidt law: Σ_SFR = A (Σ_gas / 1 M☉ pc⁻²)^N.

    Parameters
    ----------
    gas_surface_density_msun_pc2 : float or ndarray
        Total (HI + H2) gas surface density [M☉ pc⁻²], positive.
    index : float
        Power-law index N; 1.4 ± 0.15 in the source.
    amplitude : float
        Σ_SFR for Σ_gas = 1 M☉ pc⁻² [M☉ yr⁻¹ kpc⁻²]; (2.5 ± 0.7)e-4 in the source.

    Returns
    -------
    float or ndarray
        Star formation rate surface density [M☉ yr⁻¹ kpc⁻²].

    References
    ----------
    [Kennicutt1998] Eq. 4; [McKee2007] eq:ks.

    Validity
    --------
    The source states that a single power law represents the global rate well, from
    gas-poor spiral disks to the brightest starburst nuclei.
    It applies to galaxy-scale averages, not to a single star-forming region.

    Verification
    ------------
    tests/unit/test_star_formation.py: recovery of A and N; scaling check.
    """
    (s,) = _positive(gas_surface_density=gas_surface_density_msun_pc2)
    if not (amplitude > 0 and index > 0):
        raise ValueError("amplitude and index must be positive")
    return amplitude * s**index


@source(
    "Kennicutt1998",
    where="Eq. 7 (order in the text; referred to as 'equation (7)')",
    status="read",
    note="The source writes Σ_SFR ≃ 0.017 Σ_gas Ω_gas; Ω_gas is the gas rotation rate of the disk. "
    "McKee2007 TeX label eq:ks quotes the same coefficient",
)
@source("McKee2007", where="TeX label eq:ks", status="read")
def kennicutt_orbital_sfr_surface_density(
    gas_surface_density_msun_pc2: ArrayLike, omega_km_s_kpc: ArrayLike, coefficient: float = 0.017
) -> Array:
    """Compute the star formation law with the orbital time: Σ_SFR = 0.017 Σ_gas Ω.

    About 10 % of the gas mass per orbit (0.017·2π ≈ 0.107) turns into stars.

    Parameters
    ----------
    gas_surface_density_msun_pc2 : float or ndarray
        Gas surface density [M☉ pc⁻²], positive.
    omega_km_s_kpc : float or ndarray
        Angular rotation rate of the gas Ω [km s⁻¹ kpc⁻¹], positive.
    coefficient : float
        Dimensionless efficiency coefficient; 0.017 in the source.

    Returns
    -------
    float or ndarray
        Star formation rate surface density [M☉ yr⁻¹ kpc⁻²].

    References
    ----------
    [Kennicutt1998] Eq. 7; [McKee2007] eq:ks.

    Validity
    --------
    The source lowers the efficiency of 21 % per orbit at the outer edge to ≈ 10 %, because the
    mean orbital period inside the disk is about half that at the edge; 0.017 is the
    coefficient of this local form.

    Verification
    ------------
    tests/unit/test_star_formation.py: ~10 % efficiency per orbit.
    """
    s, om = _positive(gas_surface_density=gas_surface_density_msun_pc2, omega=omega_km_s_kpc)
    if not coefficient > 0:
        raise ValueError("coefficient must be positive")
    om_yr = om * 1e3 / (1e3 * _PC_M) * _YEAR_S  # km s^-1 kpc^-1 -> yr^-1
    return coefficient * (s * 1e6) * om_yr


@source(
    "Krumholz2015",
    where="Section 8 (Giant Molecular Clouds), Depletion Time: t_dep = Σ_gas / Σ̇_*",
    status="read",
    note="McKee2007 defines the same quantity as t_{S*} = M_S / Ṁ_* (before TeX label eq:epsff)",
)
def depletion_time_yr(gas_surface_density_msun_pc2: ArrayLike, sfr_surface_density_msun_yr_kpc2: ArrayLike) -> Array:
    """Compute the gas depletion time: t_dep = Σ_gas / Σ_SFR.

    Parameters
    ----------
    gas_surface_density_msun_pc2 : float or ndarray
        Gas surface density [M☉ pc⁻²], positive.
    sfr_surface_density_msun_yr_kpc2 : float or ndarray
        Star formation surface density [M☉ yr⁻¹ kpc⁻²], positive.

    Returns
    -------
    float or ndarray
        Depletion time [yr].

    References
    ----------
    [Krumholz2015] Section 8. The source gives typical values of ≈ 1 Gyr for the
    Milky Way and ≈ 2 Gyr for the molecular gas of nearby galaxies.

    Verification
    ------------
    tests/unit/test_star_formation.py: Σ^{-0.4} scaling with the Kennicutt-Schmidt law.
    """
    s, sfr = _positive(
        gas_surface_density=gas_surface_density_msun_pc2,
        sfr_surface_density=sfr_surface_density_msun_yr_kpc2,
    )
    return s * 1e6 / sfr


@source(
    "McKee2007",
    where="TeX label Qdef",
    status="read",
    note="The source uses the form with π for a fluid (gas) disk",
)
def toomre_q_gas(kappa_km_s_kpc: ArrayLike, sound_speed_km_s: ArrayLike, surface_density_msun_pc2: ArrayLike) -> Array:
    """Compute the Toomre parameter for a gas disk: Q = κ c_s / (π G Σ).

    Parameters
    ----------
    kappa_km_s_kpc : float or ndarray
        Epicyclic frequency κ [km s⁻¹ kpc⁻¹], positive.
    sound_speed_km_s : float or ndarray
        Mean sound speed (or velocity dispersion) of the gas [km s⁻¹], positive.
    surface_density_msun_pc2 : float or ndarray
        Gas surface density [M☉ pc⁻²], positive.

    Returns
    -------
    float or ndarray
        Dimensionless Q. The source reports that in models with realistic thickness, bound
        clouds form when Q < Q_cr ≈ 1.5.

    References
    ----------
    [McKee2007] Qdef.

    Validity
    --------
    One-component, fluid, thin disk; thickness, magnetic field, and the stellar component are neglected.

    Verification
    ------------
    tests/unit/test_star_formation.py: Q = 1.4 for c_s = 7 km/s, κ = 36 km/s/kpc, Σ = 13 M☉/pc².
    """
    kappa, cs, s = _positive(
        kappa=kappa_km_s_kpc, sound_speed=sound_speed_km_s, surface_density=surface_density_msun_pc2
    )
    kappa_si = kappa * 1e3 / (1e3 * _PC_M)  # s^-1
    cs_si = cs * 1e3
    s_si = s * _SOLAR_MASS_KG / _PC_M**2
    return kappa_si * cs_si / (np.pi * GRAVITATIONAL_CONSTANT.value * s_si)


@source("McKee2007", where="TeX label MJ2D (MJ2Ddef)", status="read")
def two_dimensional_jeans_mass_msun(sound_speed_km_s: ArrayLike, surface_density_msun_pc2: ArrayLike) -> Array:
    """Compute the two-dimensional Jeans mass: M_J,2D = c_s⁴ / (G² Σ).

    According to the source, the mass of clouds formed by gravity in gas disks
    of uniform surface density is a few to ten times this value.

    Parameters
    ----------
    sound_speed_km_s : float or ndarray
        Mean sound speed of the gas [km s⁻¹], positive.
    surface_density_msun_pc2 : float or ndarray
        Gas surface density [M☉ pc⁻²], positive.

    Returns
    -------
    float or ndarray
        Mass [M☉].

    References
    ----------
    [McKee2007] MJ2D.

    Validity
    --------
    Definition in the source; disk thickness and field effects change the coefficient.

    Verification
    ------------
    tests/unit/test_star_formation.py: 1e7 M☉ for c_s = 7 km/s, Σ = 13 M☉/pc².
    """
    cs, s = _positive(sound_speed=sound_speed_km_s, surface_density=surface_density_msun_pc2)
    s_si = s * _SOLAR_MASS_KG / _PC_M**2
    return (cs * 1e3) ** 4 / (GRAVITATIONAL_CONSTANT.value**2 * s_si) / _SOLAR_MASS_KG


@source("McKee2007", where="TeX label SFR-ff", status="read")
def free_fall_sfr_surface_density(
    gas_surface_density_msun_pc2: ArrayLike,
    free_fall_time_yr: ArrayLike,
    ff_efficiency: float = 0.01,
    cloud_gas_fraction: float = 1.0,
) -> Array:
    """Compute the star formation rate surface density from the cloud free-fall time.

    Σ_SFR = ε_ff,GMC Σ_gas f_GMC / t_ff,GMC.

    Parameters
    ----------
    gas_surface_density_msun_pc2 : float or ndarray
        Gas surface density averaged over a large population of clouds [M☉ pc⁻²], positive.
    free_fall_time_yr : float or ndarray
        t_ff from the mean density of giant molecular clouds [yr], positive.
    ff_efficiency : float
        ε_ff,GMC, [0, 1]; typical observed value ≈ 0.01.
    cloud_gas_fraction : float
        Fraction f_GMC of the gas that is in clouds, (0, 1].

    Returns
    -------
    float or ndarray
        Star formation rate surface density [M☉ yr⁻¹ kpc⁻²].

    References
    ----------
    [McKee2007] SFR-ff.

    Validity
    --------
    The surface density is an average over a population of clouds; the rate of a single cloud varies widely.

    Verification
    ------------
    tests/unit/test_star_formation.py: linear scaling with ε_ff and f_GMC; inverse proportion to t_ff.
    """
    s, tff = _positive(gas_surface_density=gas_surface_density_msun_pc2, free_fall_time=free_fall_time_yr)
    if not 0.0 <= ff_efficiency <= 1.0:
        raise ValueError("ε_ff must be in the range [0, 1]")
    if not 0.0 < cloud_gas_fraction <= 1.0:
        raise ValueError("f_GMC must be in the range (0, 1]")
    return ff_efficiency * (s * 1e6) * cloud_gas_fraction / tff


@source(
    "McKee2007",
    where="TeX label SFR-lc",
    status="read",
    note="The source gives a mean observed ε_GMC ≈ 0.05 for the Milky Way (t_GMC/t_ff ≈ 5)",
)
def life_cycle_sfr_surface_density(
    gas_surface_density_msun_pc2: ArrayLike, cloud_lifetime_efficiency: float, life_cycle_time_yr: ArrayLike
) -> Array:
    """Compute the star formation rate surface density from the life cycle time of the gas.

    Σ_SFR = ε_GMC Σ_gas / t_lc; here t_lc = t_diffuse + t_GMC is the time from the
    gas turning into clouds until it is dispersed after forming stars.

    Parameters
    ----------
    gas_surface_density_msun_pc2 : float or ndarray
        Gas surface density [M☉ pc⁻²], positive.
    cloud_lifetime_efficiency : float
        Fraction ε_GMC of the cloud mass that is converted into stars over the lifetime of the cloud, [0, 1].
    life_cycle_time_yr : float or ndarray
        Life cycle time t_lc [yr], positive.

    Returns
    -------
    float or ndarray
        Star formation rate surface density [M☉ yr⁻¹ kpc⁻²].

    References
    ----------
    [McKee2007] SFR-lc.

    Validity
    --------
    Assumes a steady state (formation and destruction rates are equal).

    Verification
    ------------
    tests/unit/test_star_formation.py: linear in ε_GMC and Σ_gas, inversely proportional to t_lc.
    """
    s, tlc = _positive(gas_surface_density=gas_surface_density_msun_pc2, life_cycle=life_cycle_time_yr)
    if not 0.0 <= cloud_lifetime_efficiency <= 1.0:
        raise ValueError("ε_GMC must be in the range [0, 1]")
    return cloud_lifetime_efficiency * (s * 1e6) / tlc


@source(
    "Kennicutt2012",
    where="Table 1 (TeX label sfrequations) and the equation log SFR = log L_x − log C_x in the subsection 'An Updated Compendium of Integrated SFR Calibrations'",
    status="read",
    note="Kroupa IMF (Kroupa and Weidner 2003), Starburst99, solar metallicity. IR bands are for continuous star formation "
    "(0-100 Myr) and full attenuation; factor of 0.68 for Hα and 0.86 for TIR relative to K98. "
    "Units: erg/s for FUV, NUV, Hα, TIR, 24 µm, 70 µm, 2-10 keV; erg/s/Hz for 1.4 GHz",
)
def sfr_calibration_msun_yr(luminosity_w: ArrayLike, band: str) -> Array:
    """Compute the star formation rate from luminosity: log SFR = log L_x − log C_x.

    Parameters
    ----------
    luminosity_w : float or ndarray
        Luminosity [W] (W/Hz for the 1.4 GHz band), positive. Converted to the cgs
        units of the source with 1 erg/s = 10⁻⁷ W.
    band : str
        "FUV", "NUV", "Halpha", "TIR", "24um", "70um", "1.4GHz", or "2-10keV".
        L_x = ν L_ν for FUV and NUV; the 3-1100 µm integral is used for TIR.

    Returns
    -------
    float or ndarray
        Star formation rate [M☉ yr⁻¹].

    References
    ----------
    [Kennicutt2012] Table 1.

    Validity
    --------
    All coefficients are for solar metallicity and are sensitive to metallicity;
    the IR bands are at the starburst limit where attenuation is complete and
    they break down for faint sources. The age range differs for each band (Table 1, second column).

    Verification
    ------------
    tests/unit/test_star_formation.py: with the Table 1 values; SFR = L / 10^41.27 for Hα.
    """
    if band not in _LOG_C:
        raise ValueError(f"unknown band {band!r}; valid: {sorted(_LOG_C)}")
    (l_w,) = _positive(luminosity=luminosity_w)
    l_erg = l_w * 1e7
    return l_erg / 10.0 ** _LOG_C[band]
