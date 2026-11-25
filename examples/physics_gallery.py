"""Fizik modüllerinden dört şekil: kara delik, politroplar, kütle fonksiyonları, Jeans kütlesi.

Her şekil ``kozmos`` fonksiyonlarının doğrudan çıktısıdır; ek veri kullanılmaz.

Kullanım:
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
CIKTI = ROOT / "figures"
GUNES_KUTLESI_KG = GM_SUN.value / GRAVITATIONAL_CONSTANT.value  # nominal GM / G
VURGU, GRI = "#c2410c", "#6b7280"


def kara_delik() -> None:
    spin = np.linspace(0.0, 0.998, 400)
    isco = basics.kerr_isco_radius_rg(spin)
    ufuk = basics.kerr_horizon_radius_rg(spin)
    yield_fraction = 100.0 * basics.isco_radiative_efficiency(spin)
    fig, eksenler = _plotting.figure("web", target="web-640", ratio=0.48, decimal=",", mosaic=[["yaricap", "verim"]])
    ax = eksenler["yaricap"]
    ax.plot(spin, isco, color=VURGU, lw=2.0, label="En içteki kararlı yörünge")
    ax.plot(spin, ufuk, color=GRI, lw=1.6, ls="--", label="Olay ufku")
    ax.set_xlabel("Dönme parametresi a*")
    ax.set_ylabel("Yarıçap (GM/c²)")
    ax.set_ylim(0, 6.6)
    ax.text(0.42, 5.6, "En içteki kararlı yörünge", color=VURGU)
    ax.text(0.04, 1.05, "Olay ufku", color="#4b5563")
    ax = eksenler["verim"]
    ax.plot(spin, yield_fraction, color="#111827", lw=2.0)
    ax.set_xlabel("Dönme parametresi a*")
    ax.set_ylabel("Yığılma verimi (%)")
    ax.set_ylim(0, 35)
    ax.annotate(
        f"a* = 0: %{yield_fraction[0]:.1f}".replace(".", ","),
        (0.0, yield_fraction[0]),
        xytext=(0.06, 11.5),
        color="#4b5563",
        arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=0.8),
    )
    ax.annotate(
        f"a* = 0,998: %{yield_fraction[-1]:.0f}",
        (0.998, yield_fraction[-1]),
        xytext=(0.36, 29.0),
        color="#4b5563",
        arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=0.8),
    )
    _plotting.save(fig, str(CIKTI / "kara_delik_donme"), target="web-640", formats=("png",))
    print(
        f"kara delik: ISCO {isco[0]:.2f} → {isco[-1]:.2f} r_g; verim %{yield_fraction[0]:.2f} → %{yield_fraction[-1]:.1f}"
    )


def lane_emden() -> None:
    fig, ax = _plotting.figure("web", target="web-640", ratio=0.62, decimal=",")
    for n in (0.0, 1.0, 1.5, 3.0):
        xi, teta, _ = structure.lane_emden_solve(n)
        (c,) = ax.plot(xi, teta, lw=1.9, label=f"n = {n:g}".replace(".", ","))
        ax.plot(xi[-1], 0.0, "o", ms=3.5, color=c.get_color())
    ax.set_xlabel("Boyutsuz yarıçap ξ")
    ax.set_ylabel("Boyutsuz sıcaklık θ (yoğunluk = merkez yoğunluğu × θⁿ)")
    ax.set_xlim(0, 7.6)
    ax.set_ylim(0, 1.03)
    _plotting.legend.place(ax)
    _plotting.save(fig, str(CIKTI / "lane_emden"), target="web-640", formats=("png",))
    print(
        "Lane-Emden ilk sıfırlar:",
        {n: round(structure.lane_emden_first_zero(n), 4) for n in (0, 1, 1.5, 3)},
        "merkez/ortalama yoğunluk n=3:",
        round(structure.lane_emden_central_to_mean_density(3.0), 2),
    )


def kutle_fonksiyonlari() -> None:
    m = np.logspace(-1, np.log10(50.0), 300)
    fig, ax = _plotting.figure("web", target="web-640", ratio=0.62, decimal=",")
    seriler = [
        ("Salpeter (1955)", imf.salpeter_dn_dm(m, 0.1, 50.0)),
        ("Kroupa (2001)", imf.kroupa_dn_dm(m, 0.1, 50.0)),
        ("Chabrier (2003)", imf.chabrier_dn_dm(m, 0.1, 50.0)),
    ]
    for name, dn in seriler:
        ax.plot(m, m * dn, lw=1.9, label=name)  # m dN/dm: logaritmik kütle aralığı başına sayı
    ax.set_xscale("log")  # datawiz: izin kütle fonksiyonunun alışılmış ölçeği; tikler düz sayı
    ax.set_yscale("log")
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_xticklabels(["0,1", "0,3", "1", "3", "10", "30"])
    ax.minorticks_off()
    ax.set_xlabel("Yıldız kütlesi (Güneş kütlesi)")
    ax.set_ylabel("Logaritmik kütle aralığı başına yıldız sayısı, m dN/dm")
    _plotting.legend.place(ax)
    _plotting.save(fig, str(CIKTI / "kutle_fonksiyonlari"), target="web-640", formats=("png",))
    print(
        "ortalama kütleler (0,1-50 M☉):",
        {name: round(float(np.trapezoid(m * dn, m) / np.trapezoid(dn, m)), 3) for name, dn in seriler},
    )


def jeans() -> None:
    n_cm3 = np.logspace(1, 7, 200)  # hidrojen molekülü sayı yoğunluğu [cm^-3]
    rho = clouds.mass_density_kg_m3(n_cm3 * 1e6)
    fig, ax = _plotting.figure("web", target="web-640", ratio=0.62, decimal=",")
    for sicaklik in (10.0, 20.0, 50.0):
        mass = clouds.jeans_mass_kg(clouds.sound_speed_m_s(sicaklik), rho) / GUNES_KUTLESI_KG
        ax.plot(n_cm3, mass, lw=1.9, label=f"{sicaklik:g} K")
    ax.set_xscale("log")  # datawiz: izin yoğunluk altı mertebe tarıyor
    ax.set_yscale("log")
    ax.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ax.set_xticklabels(["10", "10²", "10³", "10⁴", "10⁵", "10⁶", "10⁷"])
    ax.set_yticks([0.1, 1, 10, 100, 1000])
    ax.set_yticklabels(["0,1", "1", "10", "100", "1000"])
    ax.minorticks_off()
    ax.set_xlabel("Gaz parçacık yoğunluğu (cm⁻³)")
    ax.set_ylabel("Jeans kütlesi (Güneş kütlesi)")
    _plotting.labels.line_end(ax)
    _plotting.save(fig, str(CIKTI / "jeans_kutlesi"), target="web-640", formats=("png",))
    ornek = clouds.jeans_mass_kg(clouds.sound_speed_m_s(10.0), clouds.mass_density_kg_m3(1e4 * 1e6)) / GUNES_KUTLESI_KG
    print(f"Jeans kütlesi, 10 K ve 10^4 cm^-3: {float(ornek):.2f} M☉; proton kütlesi {PROTON_MASS.value:.3e} kg")


if __name__ == "__main__":
    CIKTI.mkdir(exist_ok=True)
    kara_delik()
    lane_emden()
    kutle_fonksiyonlari()
    jeans()
