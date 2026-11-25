"""Figures from the mechanics, relativity and dark energy modules, and the banner of the README.

Every curve is the output of ``kozmos`` functions:

- ``banner``: light rays around a non-rotating black hole, integrated as null geodesics of the
  Schwarzschild metric;
- ``relativity_orbits``: a bound orbit whose periapsis advances, and the deflection of light
  against the impact parameter;
- ``energy_conservation``: energy error of the symplectic integrators and of a general-purpose
  Runge-Kutta solver on a Kepler orbit;
- ``quintessence``: phase plane of a scalar field with an exponential potential;
- ``dark_energy_w``: constraint on (Ωm, w) from ``projects/dark_energy/wcdm_analysis.py``.

Usage:
    python examples/relativity_gallery.py [name ...]      # no name: all figures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import _plotting
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from scipy.integrate import solve_ivp
from scipy.stats import chi2 as chi2_distribution

from kozmos.cosmology import scalar_field
from kozmos.mechanics import integrators
from kozmos.relativity import geodesics, metrics

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"
ACCENT, GRAY, INK = "#c2410c", "#6b7280", "#111827"
RAY = "#f5c98b"  # light rays on the dark banner
LIFT = 30.0
MASS = 1.0  # geometric units: lengths in G M / c^2


def _schwarzschild(x):
    return metrics.schwarzschild_metric(x, MASS)


def _light_ray(impact: float, start_x: float, stop_radius: float) -> tuple[np.ndarray, np.ndarray, bool]:
    """Null geodesic launched along +x from (start_x, impact); returns r, phi and whether it was captured."""
    r0 = np.hypot(start_x, impact)
    phi0 = np.arctan2(impact, start_x)
    x = np.array([0.0, r0, np.pi / 2, phi0])
    u = geodesics.normalize_tangent(_schwarzschild, x, np.array([np.cos(phi0), 0.0, -np.sin(phi0) / r0]), kind="null")
    radii, angles = [r0], [phi0]
    while True:
        r = x[1]
        if r < 2.08 * MASS:
            return np.array(radii), np.array(angles), True
        if r > stop_radius and u[1] > 0:
            return np.array(radii), np.array(angles), False
        # short steps near the hole, where the coordinates are singular at r = 2M; long steps far away
        span = max(0.02, 0.25 * (r - 2.0 * MASS)) if r < 12 else 0.5 * r
        samples = np.linspace(0.0, span, 12 if r < 40 else 3)
        xs, us = geodesics.integrate_geodesic(_schwarzschild, x, u, samples, rtol=1e-9, atol=1e-11)
        radii += list(xs[1:, 1])
        angles += list(xs[1:, 3])
        x, u = xs[-1], us[-1]


def banner() -> None:
    impacts = np.concatenate([np.linspace(-11.0, -5.6, 10), np.linspace(-5.0, 5.0, 11), np.linspace(5.6, 11.0, 10)])
    impacts = np.sort(np.concatenate([impacts, [-5.25, 5.25]]))
    fig, ax = _plotting.figure("web", target="web-640", ratio=0.36, koyu=True)
    background = fig.get_facecolor()
    captured_count = 0
    for b in impacts:
        r, phi, captured = _light_ray(float(b), -200.0, 60.0)
        captured_count += captured
        # the plotting tool draws a reference line at y = 0 when the axis spans it: the picture is shifted upward
        x, y = r * np.cos(phi), r * np.sin(phi) + LIFT
        keep = x > -12.0
        closeness = np.exp(-(((abs(b) - np.sqrt(27.0)) / 2.2) ** 2))  # brightest near the critical impact parameter
        # one hue for every ray: brightness, not color, carries the distance from the critical impact parameter
        ax.plot(x[keep], y[keep], color=RAY, lw=0.9, alpha=0.3 + 0.65 * closeness, solid_capstyle="round")
    ax.add_patch(Circle((0, LIFT), 2.0 * MASS, facecolor="black", edgecolor="#64748b", lw=0.6, zorder=5))
    ax.add_patch(Circle((0, LIFT), 3.0 * MASS, fill=False, ls=(0, (2, 3)), lw=0.8, color="#cbd5e1", zorder=6))
    ax.set_xlim(-40, 15)
    ax.set_ylim(LIFT - 9.9, LIFT + 9.9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.axis("off")
    ax.text(-39, LIFT + 2.6, "kozmos", fontsize=30, fontweight="bold", color="#f8fafc", va="center")
    ax.text(
        -39,
        LIFT - 1.6,
        "astronomy and astrophysics,\nevery equation traced to its source",
        color="#cbd5e1",
        va="center",
    )
    ax.text(-39, LIFT - 6.8, "light bent by a black hole,\ncomputed with this library", color="#94a3b8", va="center")
    fig.patch.set_facecolor(background)
    _plotting.save(fig, str(OUT_DIR / "banner"), target="web-640", formats=("png",))
    print(f"banner: {len(impacts)} rays, {captured_count} captured")


def _deflection(impact: float) -> float:
    start = -4000.0
    r, phi, captured = _light_ray(impact, start, 4000.0)
    if captured:
        return float("nan")
    swept = abs(np.unwrap(phi)[-1] - phi[0])
    # the asymptotes are reached only at infinity: add the flat-space angle still to go at both ends
    return float(swept - np.pi + np.arcsin(impact / r[0]) + np.arcsin(impact / r[-1]))


def relativity_orbits() -> None:
    # bound orbit between r = 7 and r = 30: the angular momentum that makes both radii turning points
    periapsis, apoapsis = 7.0, 30.0
    f = lambda r: 1.0 - 2.0 * MASS / r  # noqa: E731
    l2 = (f(apoapsis) - f(periapsis)) / (f(periapsis) / periapsis**2 - f(apoapsis) / apoapsis**2)
    x0 = np.array([0.0, apoapsis, np.pi / 2, 0.0])
    u0 = geodesics.normalize_tangent(_schwarzschild, x0, np.array([0.0, 0.0, np.sqrt(l2) / apoapsis**2]))
    proper_time = np.linspace(0.0, 2500.0, 5000)
    xs, us = geodesics.integrate_geodesic(_schwarzschild, x0, u0, proper_time, rtol=1e-10, atol=1e-12)
    r, phi = xs[:, 1], xs[:, 3]
    minima = np.where((r[1:-1] < r[:-2]) & (r[1:-1] < r[2:]))[0] + 1
    advance_deg = float(np.degrees(np.mean(np.diff(phi[minima])) - 2.0 * np.pi))
    energy = [
        geodesics.stationary_axisymmetric_constants(_schwarzschild, a, b)[0]
        for a, b in zip(xs[::500], us[::500], strict=True)
    ]

    impacts = np.geomspace(5.4, 400.0, 16)
    angles = np.array([_deflection(float(b)) for b in impacts])

    fig, axes = _plotting.figure("web", target="web-640", ratio=0.5, mosaic=[["orbit", "light"]])
    ax = axes["orbit"]
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi)])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    shade = np.linspace(0.25, 1.0, len(segments))
    colors = np.column_stack([np.full_like(shade, 0.76), np.full_like(shade, 0.25), np.full_like(shade, 0.05), shade])
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=1.1))
    ax.add_patch(Circle((0, 0), 2.0, color=INK))
    ax.set_xlim(-36, 36)
    ax.set_ylim(-36, 48)
    ax.set_aspect("equal")
    ax.set_xlabel("x (GM/c²)")
    ax.set_ylabel("y (GM/c²)")
    ax.text(-34, 46, f"periapsis advances\n{advance_deg:.0f}° per orbit", color=ACCENT, va="top")

    ax = axes["light"]
    weak = 4.0 * MASS / impacts
    ax.plot(impacts, np.degrees(weak), color=GRAY, lw=1.4, ls="--")
    ax.plot(impacts, np.degrees(angles), color=ACCENT, lw=0, marker="o", ms=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Impact parameter (GM/c²)")
    ax.set_ylabel("Deflection of light (degrees)")
    ax.text(30, 22, "integrated null geodesics", color=ACCENT)
    ax.text(7, 1.1, "weak-field formula\n4GM/(c²b)", color="#4b5563", va="top")
    _plotting.save(fig, str(OUT_DIR / "relativity_orbits"), target="web-640", formats=("png",))
    record = {
        "orbit": {
            "periapsis_rg": float(r[minima].mean()),
            "apoapsis_rg": apoapsis,
            "periapsis_advance_deg_per_orbit": advance_deg,
            "periapses_measured": int(len(minima)),
            "energy_relative_drift": float(np.max(np.abs(np.array(energy) / energy[0] - 1.0))),
        },
        "deflection": [
            {"impact_rg": float(b), "integrated_deg": float(np.degrees(a)), "weak_field_deg": float(np.degrees(w))}
            for b, a, w in zip(impacts, angles, weak, strict=True)
        ],
    }
    (OUT_DIR / "relativity_orbits.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"relativity_orbits: advance {advance_deg:.2f} deg/orbit")


def energy_conservation() -> None:
    eccentricity, periods, steps_per_period = 0.5, 1000, 200
    q0 = np.array([1.0 - eccentricity, 0.0])  # GM = 1, a = 1: period 2 pi
    v0 = np.array([0.0, np.sqrt((1.0 + eccentricity) / (1.0 - eccentricity))])

    def acceleration(q, t):
        return -q / np.sum(q**2) ** 1.5

    def energy(q, v):
        return 0.5 * np.sum(v**2, axis=-1) - 1.0 / np.sqrt(np.sum(q**2, axis=-1))

    dt = 2.0 * np.pi / steps_per_period
    orbit = np.arange(1, periods + 1)
    curves = {}
    for scheme in ("kdk", "forest_ruth"):
        _, q, v = integrators.integrate(acceleration, q0, v0, dt, periods * steps_per_period, scheme=scheme)
        error = np.abs(integrators.relative_drift(energy(q, v)))[1:]
        curves[scheme] = error.reshape(periods, steps_per_period).max(axis=1)
    times = np.linspace(0.0, 2.0 * np.pi * periods, periods * 20 + 1)
    solution = solve_ivp(
        lambda t, s: np.concatenate([s[2:], acceleration(s[:2], t)]), (0.0, times[-1]), np.concatenate([q0, v0]),
        method="RK45", t_eval=times, rtol=1e-6, atol=1e-9,
    )  # fmt: skip
    error = np.abs(integrators.relative_drift(energy(solution.y[:2].T, solution.y[2:].T)))[1:]
    curves["rk45"] = error.reshape(periods, 20).max(axis=1)

    fig, ax = _plotting.figure("web", target="web-640", ratio=0.6)
    ax.plot(orbit, curves["rk45"], color=GRAY, lw=1.6)
    ax.plot(orbit, curves["kdk"], color=INK, lw=1.6)
    ax.plot(orbit, curves["forest_ruth"], color=ACCENT, lw=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Orbit number")
    ax.set_ylabel("Largest energy error in the orbit (relative)")
    ax.set_ylim(1.5e-6, 8e-3)
    ax.text(25, 4.5e-5, "Runge-Kutta (SciPy RK45, rtol 1e-6):\nthe error accumulates", color="#4b5563", va="top")
    ax.text(1.3, curves["kdk"][0] * 0.7, "leapfrog: bounded", color=INK, va="top")
    ax.text(25, curves["forest_ruth"][0] * 0.7, "fourth-order symplectic: bounded", color=ACCENT, va="top")
    _plotting.save(fig, str(OUT_DIR / "energy_conservation"), target="web-640", formats=("png",))
    record = {
        "eccentricity": eccentricity,
        "periods": periods,
        "steps_per_period": steps_per_period,
        "rk45_function_evaluations_per_period": solution.nfev / periods,
        "energy_error": {
            k: {"first_orbit": float(v[0]), "last_orbit": float(v[-1]), "max": float(v.max())}
            for k, v in curves.items()
        },
    }
    (OUT_DIR / "energy_conservation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"energy_conservation: {record['energy_error']}")


def quintessence() -> None:
    lam, w_background = 2.0, 0.0
    fig, ax = _plotting.figure("web", target="web-640", ratio=0.56)
    theta = np.linspace(0.0, np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color=GRAY, lw=1.0)
    starts = [(x, y) for x in (-0.95, -0.6, -0.2, 0.2, 0.6, 0.9) for y in (0.02, 0.3)]
    ends = []
    for x0, y0 in starts:
        if x0**2 + y0**2 >= 1.0:
            continue
        solution = scalar_field.integrate_exponential_quintessence(
            x0, y0, lam, w_background, (0.0, 40.0), n_eval=np.linspace(0.0, 40.0, 4000)
        )
        ax.plot(solution.x, solution.y, color=GRAY, lw=0.9, alpha=0.8)
        ends.append((float(solution.x[-1]), float(solution.y[-1])))
    points = scalar_field.exponential_fixed_points(lam, w_background)
    attractor = next(p for p in points if "stable" in p.stability and "unstable" not in p.stability)
    for p in points:
        if p.y < 0:
            continue
        ax.plot(
            p.x, p.y, marker="o", ms=7 if p is attractor else 5, color=ACCENT if p is attractor else INK, lw=0, zorder=5
        )
    ax.set_xlim(-1.08, 1.55)
    ax.set_ylim(-0.03, 1.08)
    ax.set_aspect("equal")
    ax.set_xlabel("x  (x² = kinetic share of the density)")
    ax.set_ylabel("y  (y² = potential share)")
    omega = float(scalar_field.scalar_omega_phi(attractor.x, attractor.y))
    ax.text(
        0.80,
        0.84,
        f"attractor: the field keeps\n{100 * omega:.0f}% of the density",
        color=ACCENT,
        va="bottom",
    )
    ax.text(-0.69, 0.78, "field only\n(Ω = 1)", color="#4b5563", ha="right", va="bottom")
    _plotting.save(fig, str(OUT_DIR / "quintessence"), target="web-640", formats=("png",))
    spread = float(np.max(np.hypot(*(np.array(ends) - [attractor.x, attractor.y]).T)))
    record = {
        "lam": lam,
        "w_background": w_background,
        "fixed_points": [{"name": p.name, "x": p.x, "y": p.y, "stability": p.stability} for p in points],
        "attractor_omega_phi": omega,
        "largest_distance_of_a_trajectory_end_from_the_attractor": spread,
        "trajectories": len(ends),
    }
    (OUT_DIR / "quintessence.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"quintessence: attractor ({attractor.x:.4f}, {attractor.y:.4f}), spread {spread:.1e}")


def dark_energy_w() -> None:
    folder = ROOT / "projects" / "dark_energy"
    grid = np.load(folder / "wcdm_posterior.npz")
    result = json.loads((folder / "wcdm_result.json").read_text(encoding="utf-8"))
    omega_m, w, chi2 = grid["omega_m"], grid["w"], grid["chi2"]
    delta = (chi2 - chi2.min()).T
    levels = [0.0, float(chi2_distribution.ppf(0.6827, 2)), float(chi2_distribution.ppf(0.9545, 2))]
    fig, ax = _plotting.figure("web", target="web-640", ratio=0.66)
    ax.contourf(omega_m, w, delta, levels=levels, colors=["#f4b79a", "#fbe3d6"])
    ax.contour(omega_m, w, delta, levels=levels[1:], colors=[ACCENT], linewidths=[1.4, 0.9])
    ax.axhline(-1.0, color=INK, lw=1.2, ls="--")
    best = result["best_fit"]
    ax.plot(best["omega_m"], best["w"], marker="o", ms=6, color=ACCENT, lw=0)
    p = result["published"]
    ax.errorbar(
        p["omega_m"]["value"], p["w"]["value"],
        xerr=[[p["omega_m"]["minus"]], [p["omega_m"]["plus"]]], yerr=[[p["w"]["minus"]], [p["w"]["plus"]]],
        color=INK, marker="s", ms=4, lw=1.0, capsize=2,
    )  # fmt: skip
    ax.set_xlim(0.05, 0.5)
    ax.set_ylim(-1.5, -0.5)
    ax.set_xlabel("Matter density Ωm")
    ax.set_ylabel("Dark energy equation of state w")
    ax.text(0.06, -0.97, "cosmological constant (w = −1)", color=INK, va="bottom")
    ax.text(0.33, -0.66, "published value\n(Brout et al. 2022)", color=INK, va="bottom")
    ax.text(0.07, -1.42, "this library:\n68% and 95% regions", color=ACCENT, va="bottom")
    _plotting.save(fig, str(OUT_DIR / "dark_energy_w"), target="web-640", formats=("png",))
    print(f"dark_energy_w: levels {levels[1:]}")


FIGURES = {
    "banner": banner,
    "relativity_orbits": relativity_orbits,
    "energy_conservation": energy_conservation,
    "quintessence": quintessence,
    "dark_energy_w": dark_energy_w,
}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
