"""Çıplak gözle görülen yıldızların galaktik koordinatlarda tüm gökyüzü haritası.

Veri: Parlak Yıldız Kataloğu (``kozmos.arsiv.heasarc.parlak_yildizlar``).
Koordinat dönüşümü ``kozmos.koordinat.donusum.icrs_den_galaktik`` ile yapılır;
izdüşüm matplotlib'in Hammer izdüşümüdür.

Kullanım:
    python ornekler/gokyuzu_haritasi.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kozmos.arsiv import heasarc
from kozmos.koordinat.donusum import icrs_den_galaktik
from kozmos.temel.aci import eksi_pi_arti_pi

try:  # datawiz yerel bir şekil kitaplığıdır; yoksa düz matplotlib ile çizilir ve denetim atlanır.
    import datawiz as dw
except ImportError:  # pragma: no cover
    dw = None

KOK = Path(__file__).resolve().parents[1]
ADLAR = {"HR 2491": "Sirius", "HR 2326": "Canopus", "HR 5340": "Arcturus", "HR 7001": "Vega", "HR 1708": "Capella"}


def main() -> None:
    s = heasarc.parlak_yildizlar(KOK / "veri" / "ham" / "heasarc")
    gecerli = np.isfinite(s["vmag"]) & np.isfinite(s["ra"]) & np.isfinite(s["dec"])
    boylam, enlem = icrs_den_galaktik(np.radians(s["ra"][gecerli]), np.radians(s["dec"][gecerli]))
    kadir = s["vmag"][gecerli]
    x = -eksi_pi_arti_pi(boylam)  # gökyüzü haritası kuralı: boylam sola doğru artar

    if dw is not None:
        fig, ax = dw.figur("web", hedef="web-640", oran=0.56, subplot_kw={"projection": "hammer"})
    else:
        fig, ax = plt.subplots(figsize=(6.7, 3.8), subplot_kw={"projection": "hammer"})
    boyut = 0.25 + 14.0 * (10 ** (-0.4 * (kadir - kadir.min()))) ** 0.55  # akının 0,55. kuvveti: soluklar da görünsün
    ax.scatter(x, enlem, s=boyut, color="#374151", linewidths=0, alpha=0.75, rasterized=True)
    ax.set_xticklabels([])
    ax.set_yticks(np.radians([-60, -30, 0, 30, 60]))
    ax.set_yticklabels(["−60°", "−30°", "0°", "30°", "60°"])
    ax.grid(True, color="#d1d5db", lw=0.5)
    for hr, ad in ADLAR.items():
        i = int(np.flatnonzero(s["name"][gecerli] == hr)[0])
        ax.annotate(ad, (x[i], enlem[i]), xytext=(6, 6), textcoords="offset points", color="#c2410c", fontweight="bold")
    ax.set_xlabel("Galaktik boylam (merkez ortada, sola doğru artar)")
    ax.set_ylabel("Galaktik enlem")

    cikti = KOK / "belgeler"
    cikti.mkdir(exist_ok=True)
    if dw is not None:
        dw.kaydet(fig, str(cikti / "gokyuzu_haritasi"), hedef="web-640", bicimler=("png",))
    else:
        fig.savefig(cikti / "gokyuzu_haritasi.png", dpi=250, bbox_inches="tight")

    # |b| < 20° kuşağı gökyüzünün sin(20°) = %34,2'sini kaplar; düzgün dağılımda yıldızların bu kadarı orada olurdu.
    kusak = np.abs(enlem) < np.radians(20.0)
    print(
        f"yıldız sayısı {kadir.size}; |b| < 20° içinde %{100 * kusak.mean():.1f} (düzgün dağılım beklentisi %{100 * np.sin(np.radians(20)):.1f})"
    )
    parlak = kadir < 3.0
    print(f"V < 3 olan {parlak.sum()} yıldızın %{100 * kusak[parlak].mean():.1f}'i aynı kuşakta")


if __name__ == "__main__":
    main()
