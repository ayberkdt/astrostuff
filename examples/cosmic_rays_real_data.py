"""Kozmik ışın temizliğinin gerçek HST karelerinde sınanması.

Veri: Hubble STIS CCD görüntüleri (MAST). Her dosya aynı yönelimde art arda
alınmış iki pozu taşır. Kozmik ışınlar iki pozda farklı piksellere düşer; bu
yüzden iki pozun farkı, tek kareye bakan algoritma için bağımsız bir "gerçek"
verir. İki ayrı alan türü sınanır:

- o5d903060: M5 (NGC 5904) küresel kümesi, yıldızca kalabalık alan.
- o4gb010h0: Abell 1795 gökada kümesi, yaygın kaynaklı seyrek alan.

Ölçülenler: gerçek kozmik ışın piksellerinin tek kareden bulunan kesri
(tamlık), işaretlenen piksellerin gerçek kozmik ışına bitişik olan kesri
(saflık) ve temizlenen karenin ikinci poza yakınlığı. STIS CCD'sinde PSF eksik
örneklenir; kaynak makale bu durumda f_lim'in büyütülmesini söyler
[vanDokkum2001], bu yüzden iki değer karşılaştırılır.

Kullanım:
    python examples/cosmic_rays_real_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import _plotting
import numpy as np
from scipy import ndimage

from kozmos.archives import cache
from kozmos.io import fits
from kozmos.reduction import cosmic_rays
from kozmos.visualization import stretch

ROOT = Path(__file__).resolve().parents[1]
GOZLEMLER = ("o5d903060", "o4gb010h0")
ADRES = "https://mast.stsci.edu/api/v0.1/Download/file"
GERCEK_ESIGI = 6.0  # iki pozun farkı, fark gürültüsünün bu katını aşarsa kozmik ışın sayılır
KESIT = (slice(430, 590), slice(430, 590))
AYARLAR = {
    "f_lim=2, komşu eşiği yok": {"f_lim": 2.0, "komsu_esik_orani": 1.0},
    "f_lim=2": {"f_lim": 2.0},
    "f_lim=5": {"f_lim": 5.0},
}


def olc(gozlem: str) -> tuple[dict, dict]:
    target = ROOT / "data" / "raw" / "hst" / f"{gozlem}_flt.fits"
    cache.download(
        ADRES, target, parameters={"uri": f"mast:HST/product/{gozlem}_flt.fits"}, attribution="HST STIS, MAST üzerinden"
    )
    birimler = fits.list_hdus(target)
    ana = birimler[0][1]
    bilim = [i for i, (_, b, _) in enumerate(birimler) if b.get("EXTNAME") == "SCI"]
    if len(bilim) < 2:
        raise SystemExit(f"{gozlem}: iki poz bekleniyordu, {len(bilim)} bulundu")
    kazanc, okuma = float(ana["ATODGAIN"]), float(ana["READNSE"])
    poz1, baslik1 = fits.read_fits(target, hdu=bilim[0])
    poz2, baslik2 = fits.read_fits(target, hdu=bilim[1])
    ratio = float(baslik1["EXPTIME"]) / float(baslik2["EXPTIME"])
    poz2 = poz2 * ratio  # poz süreleri farklıysa ikinci poz birincinin ölçeğine getirilir

    # Gerçek: iki pozun farkı. Fark gürültüsü iki pozun küçüğünden (kozmik ışınsız değer) kurulur.
    taban = np.minimum(poz1, poz2)
    sigma_fark = np.sqrt((1.0 + ratio) * (np.clip(kazanc * taban, 0.0, None) + okuma**2)) / kazanc
    gercek1 = (poz1 - poz2) > GERCEK_ESIGI * sigma_fark
    gercek2 = (poz2 - poz1) > GERCEK_ESIGI * sigma_fark
    cevre = ndimage.binary_dilation(gercek1, structure=np.ones((3, 3)))  # gerçek piksellere bitişik komşular
    parlak = gercek1 & ((poz1 - poz2) > 20.0 * sigma_fark)

    sonuc = {
        "data": {
            "hedef": str(ana["TARGNAME"]).strip(),
            "program": int(ana["PROPOSID"]),
            "poz_suresi_s": [float(baslik1["EXPTIME"]), float(baslik2["EXPTIME"])],
            "kazanc_e_adu": kazanc,
            "okuma_gurultusu_e": okuma,
            "gokyuzu_duzeyi_adu": float(np.median(poz1)),
        },
        "gercek": {
            "esik_sigma": GERCEK_ESIGI,
            "poz1_kozmik_isin_pikseli": int(gercek1.sum()),
            "poz2_kozmik_isin_pikseli": int(gercek2.sum()),
            "poz1_kesri": float(gercek1.mean()),
        },
        "ayarlar": {},
    }
    temizler = {}
    for name, secenek in AYARLAR.items():
        temiz, mask = cosmic_rays.detect_cosmic_rays(poz1, gain_e_adu=kazanc, read_noise_e=okuma, **secenek)
        temizler[name] = temiz
        sonuc["ayarlar"][name] = {
            "isaretlenen_piksel": int(mask.sum()),
            "tamlik": float((mask & gercek1).sum() / gercek1.sum()),
            "tamlik_20_sigma_ustu": float((mask & parlak).sum() / parlak.sum()),
            "saflik_bitisik_dahil": float((mask & cevre).sum() / mask.sum()),
            "gercek_disi_isaretlenen_kesir": float((mask & ~cevre).sum() / mask.size),
            # Temizlik niteliği: gerçek kozmik ışın piksellerinde ikinci pozdan ortanca mutlak sapma [fark gürültüsü birimi].
            "temizlik_oncesi_sapma_sigma": float(np.median(np.abs(poz1 - poz2)[gercek1] / sigma_fark[gercek1])),
            "temizlik_sonrasi_sapma_sigma": float(np.median(np.abs(temiz - poz2)[gercek1] / sigma_fark[gercek1])),
        }
    duzey = float(np.median(poz1))
    return sonuc, {
        "poz1": poz1,
        "temiz": temizler["f_lim=2"],
        "poz2": poz2,
        "duzey": duzey,
        "sacilim": float(np.sqrt(kazanc * max(duzey, 0.0) + okuma**2) / kazanc),
    }


def main() -> None:
    sonuc, goruntuler = {}, {}
    for gozlem in GOZLEMLER:
        sonuc[gozlem], goruntuler[gozlem] = olc(gozlem)
    cikti = ROOT / "figures"
    (cikti / "kozmik_isin_gercek_veri.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(sonuc, ensure_ascii=False, indent=1))

    # Şekil (M5 alanı): ham poz, temizlenmiş poz (f_lim = 2) ve ikinci poz; ortak germe.
    g = goruntuler[GOZLEMLER[0]]
    sure = sonuc[GOZLEMLER[0]]["data"]["poz_suresi_s"][0]

    def ger(image: np.ndarray) -> np.ndarray:
        return stretch.asinh_stretch(
            image[KESIT], black=g["duzey"] - g["sacilim"], alpha=1.0 / (10.0 * g["sacilim"]), q=6.0
        )

    high = float(np.percentile(ger(g["poz2"]), 99.8))
    paneller = [
        (f"Birinci poz ({sure:.0f} s)", g["poz1"]),
        ("Temizlenmiş birinci poz", g["temiz"]),
        (f"İkinci poz ({sure:.0f} s)", g["poz2"]),
    ]
    fig, eksenler = _plotting.figure(
        "web", target="web-640", ratio=0.40, decimal=",", mosaic=[["a", "b", "c"]], gridspec_kw={"wspace": 0.04}
    )
    for ax, (header, image) in zip(eksenler.values(), paneller, strict=True):
        # datawiz: izin göreli parlaklık görüntüsü; asinh gerilmiş değerlerin renk çubuğu okura nicelik vermez
        ax.imshow(
            np.clip(ger(image) / high, 0, 1), origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for kenar in ax.spines.values():
            kenar.set_visible(False)
        ax.set_title(header, loc="left")
    _plotting.save(fig, str(cikti / "kozmik_isin_gercek_veri"), target="web-640", formats=("png",))


if __name__ == "__main__":
    main()
