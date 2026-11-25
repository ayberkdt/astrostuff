"""Aynı alanın optik ve kızılötesi görüntüleri: Alev Bulutsusu (NGC 2024).

Yıldız oluşum bölgelerinde toz, görünür ışığı soğurur; yakın kızılötesinde
sönümleme çok daha küçüktür ve tozun içindeki genç yıldızlar görünür; orta
kızılötesinde ısınmış tozun kendi ışıması baskındır.

Görüntüler NASA SkyView'dan indirilir (DSS2 Red, 2MASS J/H/K, WISE 12 µm),
arka plan ve kaynak saptama ``kozmos.photometry`` ile yapılır, germe
[Lupton2004] asinh biçimidir.

Kullanım:
    python examples/optical_infrared.py
"""

from __future__ import annotations

from pathlib import Path

import _plotting
import matplotlib.pyplot as plt
import numpy as np

from kozmos.archives import skyview
from kozmos.photometry import sources
from kozmos.visualization import stretch

ROOT = Path(__file__).resolve().parents[1]
ONBELLEK = ROOT / "data" / "raw" / "skyview"
RA, DEC, ALAN, PIKSEL = 85.42, -1.90, 0.3, 450
PALET = plt.get_cmap("gray").with_extremes(bad="#7c2d12")  # NaN pikseller kiremit rengi
PANELLER = [
    ("DSS2 Red", "Optik, 0,64 µm"),
    ("2MASS-K", "Yakın kızılötesi, 2,16 µm"),
    ("WISE 12", "Orta kızılötesi, 12 µm"),
]


def hazirla(survey: str) -> tuple[np.ndarray, int]:
    """Kesiti indirir, arka planı çıkarır, kaynakları sayar ve [0, 1] aralığına gerer."""
    image, _ = skyview.download_cutout(RA, DEC, survey, ONBELLEK, size_deg=ALAN, pixels=PIKSEL)
    harita, rms = sources.background_map(image, cell=75)
    sources_of = sources.detect_sources(image, harita, rms, threshold_sigma=4.0, min_pixels=6)
    duzey, sacilim = sources.estimate_background(image)
    gerilmis = stretch.asinh_stretch(image, black=duzey - sacilim, alpha=1.0 / (8.0 * sacilim), q=6.0)
    olcekli = stretch.percentile_scale(gerilmis, 0.5, 99.9)
    olcekli[~np.isfinite(image)] = np.nan  # veri taşımayan pikseller siyaha boyanmaz, ayrı gösterilir
    return olcekli, len(sources_of)


def main() -> None:
    cikti = ROOT / "figures"
    cikti.mkdir(exist_ok=True)

    fig, eksenler = _plotting.figure(
        "web", target="web-640", ratio=0.40, decimal=",", mosaic=[["a", "b", "c"]], gridspec_kw={"wspace": 0.04}
    )
    sayilar = {}
    for ax, (survey, header) in zip(eksenler.values(), PANELLER, strict=True):
        image, n = hazirla(survey)
        sayilar[survey] = n
        # datawiz: izin göreli parlaklık görüntüsü; asinh gerilmiş değerlerin renk çubuğu okura nicelik vermez
        ax.imshow(image, origin="lower", cmap=PALET, vmin=0, vmax=1, interpolation="nearest")
        bos = ~np.isfinite(image)
        if bos.any():
            yy, xx = np.nonzero(bos)
            ax.text(xx.mean(), yy.mean(), "veri yok", color="white", ha="center", va="center")
            print(f"{survey}: {int(bos.sum())} piksel veri taşımıyor (NaN)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for kenar in ax.spines.values():
            kenar.set_visible(False)
        ax.set_title(f"{header}\n{n} kaynak saptandı", loc="left")
    # Ölçek çubuğu: 5 yay dakikası.
    ax = eksenler["a"]
    uzunluk = 5.0 / 60.0 / ALAN * PIKSEL
    ax.plot([22, 22 + uzunluk], [24, 24], color="white", lw=2, solid_capstyle="butt")
    ax.text(22, 36, "5′", color="white", va="bottom")
    _plotting.save(fig, str(cikti / "optik_kizilotesi"), target="web-640", formats=("png",))
    print("kaynak sayıları:", sayilar)

    # Yakın kızılötesi renkli bileşim: K → kırmızı, H → yeşil, J → mavi.
    bantlar = []
    for survey in ("2MASS-K", "2MASS-H", "2MASS-J"):
        image, _ = skyview.download_cutout(RA, DEC, survey, ONBELLEK, size_deg=ALAN, pixels=PIKSEL)
        harita, rms = sources.background_map(image, cell=75)
        bantlar.append((image - harita) / np.median(rms))  # gürültü birimine ölçekle: bantlar karşılaştırılabilir olur
    rgb = stretch.lupton_rgb(*bantlar, black=-1.0, alpha=0.06, q=8.0)
    plt.imsave(cikti / "alev_bulutsusu_jhk.png", np.clip(rgb, 0, 1), origin="lower")
    print("renkli bileşim:", cikti / "alev_bulutsusu_jhk.png", rgb.shape)


if __name__ == "__main__":
    main()
