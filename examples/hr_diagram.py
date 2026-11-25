"""Gaia DR3 ile Güneş çevresinin Hertzsprung-Russell diyagramı.

Veri: 50 pc içindeki, paralaks S/N > 20 ve RUWE < 1,4 olan yıldızlar
(``kozmos.archives.gaia.yakin_yildizlar``). Mutlak kadir
M_G = G + 5 + 5 log10(ϖ/1000) [Babusiaux2018]; sönümleme düzeltilmez.

Kullanım:
    python examples/hr_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import _plotting
import numpy as np

from kozmos.archives import gaia

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    s = gaia.nearby_stars(ROOT / "data" / "raw" / "gaia")
    renk = s["bp_rp"]
    magnitude = gaia.absolute_magnitude(s["phot_g_mean_mag"], s["parallax"])

    fig, ax = _plotting.figure("web", target="web-640", ratio=1.05, decimal=",")
    ax.scatter(renk, magnitude, s=1.6, alpha=0.35, linewidths=0, color="#6b7280", rasterized=True)
    ax.set_xlim(-0.7, 5.2)
    ax.set_ylim(17.5, -0.5)  # parlak yukarıda
    ax.set_xlabel("Renk, BP − RP (mag)")
    ax.set_ylabel("Mutlak kadir, G bandı (mag)")
    labels = dict(color="#c2410c", fontweight="bold")
    ax.text(2.55, 6.3, "Anakol", **labels)
    ax.text(0.95, 13.6, "Beyaz cüceler", **labels)
    ax.text(1.42, 2.1, "Devler", **labels)
    number = f"{renk.size:,}".replace(",", " ")
    ax.text(3.05, 1.2, number + " yıldız\nGüneş'e 50 pc'den yakın", color="#4b5563", va="top")

    cikti = ROOT / "figures"
    cikti.mkdir(exist_ok=True)
    _plotting.save(fig, str(cikti / "hr_diyagrami"), target="web-640", formats=("png",))

    # Bölge sayıları: kaba kutularla, yalnızca altyazı için.
    beyaz_cuce = (magnitude > 10.0 + 2.6 * renk) & (renk < 1.9)
    dev = (magnitude < 3.0) & (renk > 0.85)
    print(f"toplam {renk.size}, beyaz cüce bölgesi {beyaz_cuce.sum()}, dev bölgesi {dev.sum()}")
    print(
        f"renk aralığı {np.nanmin(renk):.2f}..{np.nanmax(renk):.2f}, kadir aralığı {magnitude.min():.1f}..{magnitude.max():.1f}"
    )


if __name__ == "__main__":
    main()
