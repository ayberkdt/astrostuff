"""Aynı alanın optik ve kızılötesi görüntüleri: Alev Bulutsusu (NGC 2024).

Yıldız oluşum bölgelerinde toz, görünür ışığı soğurur; yakın kızılötesinde
sönümleme çok daha küçüktür ve tozun içindeki genç yıldızlar görünür; orta
kızılötesinde ısınmış tozun kendi ışıması baskındır.

Görüntüler NASA SkyView'dan indirilir (DSS2 Red, 2MASS J/H/K, WISE 12 µm),
arka plan ve kaynak saptama ``kozmos.fotometri`` ile yapılır, germe
[Lupton2004] asinh biçimidir.

Kullanım:
    python ornekler/optik_kizilotesi.py
"""

from __future__ import annotations

from pathlib import Path

import datawiz as dw
import matplotlib.pyplot as plt
import numpy as np

from kozmos.arsiv import skyview
from kozmos.fotometri import kaynak as fk
from kozmos.gorsel import germe

KOK = Path(__file__).resolve().parents[1]
ONBELLEK = KOK / "veri" / "ham" / "skyview"
RA, DEC, ALAN, PIKSEL = 85.42, -1.90, 0.3, 450
PANELLER = [
    ("DSS2 Red", "Optik, 0,64 µm"),
    ("2MASS-K", "Yakın kızılötesi, 2,16 µm"),
    ("WISE 12", "Orta kızılötesi, 12 µm"),
]


def hazirla(tarama: str) -> tuple[np.ndarray, int]:
    """Kesiti indirir, arka planı çıkarır, kaynakları sayar ve [0, 1] aralığına gerer."""
    goruntu, _ = skyview.kesit_indir(RA, DEC, tarama, ONBELLEK, alan_derece=ALAN, piksel=PIKSEL)
    harita, rms = fk.arka_plan_haritasi(goruntu, hucre=75)
    kaynaklar = fk.kaynak_bul(goruntu, harita, rms, esik_sigma=4.0, en_az_piksel=6)
    duzey, sacilim = fk.arka_plan_kestir(goruntu)
    gerilmis = germe.asinh_ger(goruntu, siyah=duzey - sacilim, alfa=1.0 / (8.0 * sacilim), q=6.0)
    return germe.yuzdelik_olcekle(gerilmis, 0.5, 99.9), len(kaynaklar)


def main() -> None:
    cikti = KOK / "belgeler"
    cikti.mkdir(exist_ok=True)

    fig, eksenler = dw.figur(
        "web", hedef="web-640", oran=0.40, ondalik=",", mozaik=[["a", "b", "c"]], gridspec_kw={"wspace": 0.04}
    )
    sayilar = {}
    for ax, (tarama, baslik) in zip(eksenler.values(), PANELLER, strict=True):
        goruntu, n = hazirla(tarama)
        sayilar[tarama] = n
        # datawiz: izin göreli parlaklık görüntüsü; asinh gerilmiş değerlerin renk çubuğu okura nicelik vermez
        ax.imshow(goruntu, origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for kenar in ax.spines.values():
            kenar.set_visible(False)
        ax.set_title(f"{baslik}\n{n} kaynak saptandı", loc="left")
    # Ölçek çubuğu: 5 yay dakikası.
    ax = eksenler["a"]
    uzunluk = 5.0 / 60.0 / ALAN * PIKSEL
    ax.plot([22, 22 + uzunluk], [24, 24], color="white", lw=2, solid_capstyle="butt")
    ax.text(22, 36, "5′", color="white", va="bottom")
    dw.kaydet(fig, str(cikti / "optik_kizilotesi"), hedef="web-640", bicimler=("png",))
    print("kaynak sayıları:", sayilar)

    # Yakın kızılötesi renkli bileşim: K → kırmızı, H → yeşil, J → mavi.
    bantlar = []
    for tarama in ("2MASS-K", "2MASS-H", "2MASS-J"):
        goruntu, _ = skyview.kesit_indir(RA, DEC, tarama, ONBELLEK, alan_derece=ALAN, piksel=PIKSEL)
        harita, rms = fk.arka_plan_haritasi(goruntu, hucre=75)
        bantlar.append(
            (goruntu - harita) / np.median(rms)
        )  # gürültü birimine ölçekle: bantlar karşılaştırılabilir olur
    rgb = germe.lupton_rgb(*bantlar, siyah=-1.0, alfa=0.06, q=8.0)
    plt.imsave(cikti / "alev_bulutsusu_jhk.png", np.clip(rgb, 0, 1), origin="lower")
    print("renkli bileşim:", cikti / "alev_bulutsusu_jhk.png", rgb.shape)


if __name__ == "__main__":
    main()
