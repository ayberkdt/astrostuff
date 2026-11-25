"""Four figures from the physics modules: black hole, polytropes, mass functions, Jeans mass.

Each figure is the direct output of ``kozmos`` functions; no extra data is used.

Usage:
    python examples/physics_gallery.py
"""

from __future__ import annotations

from pathlib import Path

import _plotting
import numpy as np

from kozmos.black_holes import basics
from kozmos.core.constants import GM_SUN, GRAVITATIONAL_CONSTANT, PROTON_MASS
from kozmos.star_formation import clouds
from kozmos.stars import imf, structure

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"
SOLAR_MASS_KG = GM_SUN.value / GRAVITATIONAL_CONSTANT.value  # nominal GM / G
ACCENT, GRAY = "#c2410c", "#6b7280"


def black_hole() -> None:
    spin = np.linspace(0.0, 0.998, 400)
    isco = basics.kerr_isco_radius_rg(spin)
    horizon = basics.kerr_horizon_radius_rg(spin)
    efficiency_percent = 100.0 * basics.isco_radiative_efficiency(spin)
    fig, axes = _plotting.figure("web", target="web-640", ratio=0.48, mosaic=[["radius", "efficiency"]])
    ax = axes["radius"]
    ax.plot(spin, isco, color=ACCENT, lw=2.0, label="Innermost stable orbit")
    ax.plot(spin, horizon, color=GRAY, lw=1.6, ls="--", label="Event horizon")
    ax.set_xlabel("Spin parameter a*")
    ax.set_ylabel("Radius (GM/c²)")
    ax.set_ylim(0, 6.6)
    ax.text(0.42, 5.6, "Innermost stable orbit", color=ACCENT)
    ax.text(0.04, 1.05, "Event horizon", color="#4b5563")
    ax = axes["efficiency"]
    ax.plot(spin, efficiency_percent, color="#111827", lw=2.0)
    ax.set_xlabel("Spin parameter a*")
    ax.set_ylabel("Accretion efficiency (%)")
    ax.set_ylim(0, 35)
    ax.annotate(
        f"a* = 0: {efficiency_percent[0]:.1f}%",
        (0.0, efficiency_percent[0]),
        xytext=(0.06, 11.5),
        color="#4b5563",
        arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=0.8),
    )
    ax.annotate(
        f"a* = 0.998: {efficiency_percent[-1]:.0f}%",
        (0.998, efficiency_percent[-1]),
        xytext=(0.36, 29.0),
        color="#4b5563",
        arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=0.8),
    )
    _plotting.save(fig, str(OUT_DIR / "black_hole_spin"), target="web-640", formats=("png",))
    print(
        f"black hole: ISCO {isco[0]:.2f} → {isco[-1]:.2f} r_g; efficiency {efficiency_percent[0]:.2f}% → {efficiency_percent[-1]:.1f}%"
    )


def lane_emden() -> None:
    fig, ax = _plotting.figure("web", target="web-640", ratio=0.62)
    for n in (0.0, 1.0, 1.5, 3.0):
        xi, theta, _ = structure.lane_emden_solve(n)
        (c,) = ax.plot(xi, theta, lw=1.9, label=f"n = {n:g}")
        ax.plot(xi[-1], 0.0, "o", ms=3.5, color=c.get_color())
    ax.set_xlabel("Dimensionless radius ξ")
    ax.set_ylabel("Dimensionless temperature θ (density = central density × θⁿ)")
    ax.set_xlim(0, 7.6)
    ax.set_ylim(0, 1.03)
    _plotting.legend.place(ax)
    _plotting.save(fig, str(OUT_DIR / "lane_emden"), target="web-640", formats=("png",))
    print(
        "Lane-Emden first zeros:",
        {n: round(structure.lane_emden_first_zero(n), 4) for n in (0, 1, 1.5, 3)},
        "central/mean density for n=3:",
        round(structure.lane_emden_central_to_mean_density(3.0), 2),
    )


def mass_functions() -> None:
    m = np.logspace(-1, np.log10(50.0), 300)
    fig, ax = _plotting.figure("web", target="web-640", ratio=0.62)
    series = [
        ("Salpeter (1955)", imf.salpeter_dn_dm(m, 0.1, 50.0)),
        ("Kroupa (2001)", imf.kroupa_dn_dm(m, 0.1, 50.0)),
        ("Chabrier (2003)", imf.chabrier_dn_dm(m, 0.1, 50.0)),
    ]
    for name, dn in series:
        ax.plot(m, m * dn, lw=1.9, label=name)  # m dN/dm: number per logarithmic mass interval
    ax.set_xscale("log")  # datawiz: allowed as the usual scale of a mass function; ticks are plain numbers
    ax.set_yscale("log")
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    ax.minorticks_off()
    ax.set_xlabel("Stellar mass (solar masses)")
    ax.set_ylabel("Number of stars per logarithmic mass interval, m dN/dm")
    _plotting.legend.place(ax)
    _plotting.save(fig, str(OUT_DIR / "mass_functions"), target="web-640", formats=("png",))
    print(
        "mean masses (0.1-50 M☉):",
        {name: round(float(np.trapezoid(m * dn, m) / np.trapezoid(dn, m)), 3) for name, dn in series},
    )


def jeans() -> None:
    n_cm3 = np.logspace(1, 7, 200)  # hydrogen molecule number density [cm^-3]
    rho = clouds.mass_density_kg_m3(n_cm3 * 1e6)
    fig, ax = _plotting.figure("web", target="web-640", ratio=0.62)
    for temperature in (10.0, 20.0, 50.0):
        mass = clouds.jeans_mass_kg(clouds.sound_speed_m_s(temperature), rho) / SOLAR_MASS_KG
        ax.plot(n_cm3, mass, lw=1.9, label=f"{temperature:g} K")
    ax.set_xscale("log")  # datawiz: allowed because the density spans six orders of magnitude
    ax.set_yscale("log")
    ax.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ax.set_xticklabels(["10", "10²", "10³", "10⁴", "10⁵", "10⁶", "10⁷"])
    ax.set_yticks([0.1, 1, 10, 100, 1000])
    ax.set_yticklabels(["0.1", "1", "10", "100", "1000"])
    ax.minorticks_off()
    ax.set_xlabel("Gas particle density (cm⁻³)")
    ax.set_ylabel("Jeans mass (solar masses)")
    _plotting.labels.line_end(ax)
    _plotting.save(fig, str(OUT_DIR / "jeans_mass"), target="web-640", formats=("png",))
    example = clouds.jeans_mass_kg(clouds.sound_speed_m_s(10.0), clouds.mass_density_kg_m3(1e4 * 1e6)) / SOLAR_MASS_KG
    print(f"Jeans mass, 10 K and 10^4 cm^-3: {float(example):.2f} M☉; proton mass {PROTON_MASS.value:.3e} kg")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    black_hole()
    lane_emden()
    mass_functions()
    jeans()
