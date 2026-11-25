"""Çıplak gözle görülen yıldızların galaktik koordinatlarda tüm gökyüzü haritası.

Veri: Parlak Yıldız Kataloğu (``kozmos.archives.heasarc.parlak_yildizlar``).
Koordinat dönüşümü ``kozmos.coordinates.transforms.icrs_den_galaktik`` ile yapılır;
izdüşüm matplotlib'in Hammer izdüşümüdür.

Kullanım:
    python examples/sky_map.py
"""

from __future__ import annotations

from pathlib import Path

import _plotting
import numpy as np

from kozmos.archives import heasarc
from kozmos.coordinates.transforms import icrs_to_galactic
from kozmos.core.angles import wrap_pm_pi

ROOT = Path(__file__).resolve().parents[1]
ADLAR = {"HR 2491": "Sirius", "HR 2326": "Canopus", "HR 5340": "Arcturus", "HR 7001": "Vega", "HR 1708": "Capella"}


def main() -> None:
    s = heasarc.bright_stars(ROOT / "data" / "raw" / "heasarc")
    gecerli = np.isfinite(s["vmag"]) & np.isfinite(s["ra"]) & np.isfinite(s["dec"])
    boylam, enlem = icrs_to_galactic(np.radians(s["ra"][gecerli]), np.radians(s["dec"][gecerli]))
    magnitude = s["vmag"][gecerli]
    x = -wrap_pm_pi(boylam)  # gökyüzü haritası kuralı: boylam sola doğru artar

    fig, ax = _plotting.figure("web", target="web-640", ratio=0.56, subplot_kw={"projection": "hammer"})
    size = (
        0.25 + 14.0 * (10 ** (-0.4 * (magnitude - magnitude.min()))) ** 0.55
    )  # akının 0,55. kuvveti: soluklar da görünsün
    ax.scatter(x, enlem, s=size, color="#374151", linewidths=0, alpha=0.75, rasterized=True)
    ax.set_xticklabels([])
    ax.set_yticks(np.radians([-60, -30, 0, 30, 60]))
    ax.set_yticklabels(["−60°", "−30°", "0°", "30°", "60°"])
    ax.grid(True, color="#d1d5db", lw=0.5)
    for hr, name in ADLAR.items():
        i = int(np.flatnonzero(s["name"][gecerli] == hr)[0])
        ax.annotate(
            name, (x[i], enlem[i]), xytext=(6, 6), textcoords="offset points", color="#c2410c", fontweight="bold"
        )
    ax.set_xlabel("Galaktik boylam (merkez ortada, sola doğru artar)")
    ax.set_ylabel("Galaktik enlem")

    cikti = ROOT / "figures"
    cikti.mkdir(exist_ok=True)
    _plotting.save(fig, str(cikti / "gokyuzu_haritasi"), target="web-640", formats=("png",))

    # |b| < 20° kuşağı gökyüzünün sin(20°) = %34,2'sini kaplar; düzgün dağılımda yıldızların bu kadarı orada olurdu.
    kusak = np.abs(enlem) < np.radians(20.0)
    print(
        f"yıldız sayısı {magnitude.size}; |b| < 20° içinde %{100 * kusak.mean():.1f} (düzgün dağılım beklentisi %{100 * np.sin(np.radians(20)):.1f})"
    )
    parlak = magnitude < 3.0
    print(f"V < 3 olan {parlak.sum()} yıldızın %{100 * kusak[parlak].mean():.1f}'i aynı kuşakta")


if __name__ == "__main__":
    main()
