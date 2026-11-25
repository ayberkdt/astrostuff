"""Kozmik ışın temizliğinin gerçek HST karesinde sınanması.

Veri: Hubble STIS CCD, M5 (NGC 5904) küresel kümesi, program 8310, gözlem
o5d903060 (MAST). Dosya aynı yönelimde art arda alınmış iki 400 saniyelik pozu
taşır. Kozmik ışınlar iki pozda farklı piksellere düşer; bu yüzden iki pozun
farkı, tek kareye bakan algoritma için bağımsız bir "gerçek" verir.

Ölçülenler: gerçek kozmik ışın piksellerinin tek kareden bulunan kesri
(tamlık), işaretlenen piksellerin gerçek kozmik ışına bitişik olan kesri
(saflık) ve temizlenen karenin ikinci poza yakınlığı. STIS CCD'sinde PSF eksik
örneklenir; kaynak makale bu durumda f_lim'in büyütülmesini söyler
[vanDokkum2001], bu yüzden iki değer karşılaştırılır.

Kullanım:
    python ornekler/kozmik_isin_gercek_veri.py
"""

from __future__ import annotations

import json
from pathlib import Path

import _cizim as dw
import numpy as np
from scipy import ndimage

from kozmos.arsiv import onbellek
from kozmos.gorsel import germe
from kozmos.indirgeme import kozmik_isin
from kozmos.io import fits

KOK = Path(__file__).resolve().parents[1]
GOZLEM = "o5d903060"
ADRES = "https://mast.stsci.edu/api/v0.1/Download/file"
GERCEK_ESIGI = 6.0  # iki pozun farkı, fark gürültüsünün bu katını aşarsa kozmik ışın sayılır
KESIT = (slice(430, 590), slice(430, 590))


def main() -> None:
    hedef = KOK / "veri" / "ham" / "hst" / f"{GOZLEM}_flt.fits"
    onbellek.indir(
        ADRES,
        hedef,
        parametreler={"uri": f"mast:HST/product/{GOZLEM}_flt.fits"},
        atif="HST STIS, program 8310, MAST üzerinden",
    )
    birimler = fits.hdu_listesi(hedef)
    ana = birimler[0][1]
    kazanc, okuma = float(ana["ATODGAIN"]), float(ana["READNSE"])
    poz1, baslik1 = fits.fits_oku(hedef, hdu=1)
    poz2, baslik2 = fits.fits_oku(hedef, hdu=4)
    assert baslik1["EXTNAME"] == baslik2["EXTNAME"] == "SCI" and baslik1["EXPTIME"] == baslik2["EXPTIME"]

    # Gerçek: iki pozun farkı. Fark gürültüsü iki pozun küçüğünden (kozmik ışınsız değer) kurulur.
    taban = np.minimum(poz1, poz2)
    sigma_fark = np.sqrt(2.0 * (np.clip(kazanc * taban, 0.0, None) + okuma**2)) / kazanc
    gercek1 = (poz1 - poz2) > GERCEK_ESIGI * sigma_fark
    gercek2 = (poz2 - poz1) > GERCEK_ESIGI * sigma_fark
    cevre = ndimage.binary_dilation(gercek1, structure=np.ones((3, 3)))  # gerçek piksellere bitişik komşular

    sonuc = {
        "veri": {
            "gozlem": GOZLEM,
            "hedef": ana["TARGNAME"],
            "program": int(ana["PROPOSID"]),
            "poz_suresi_s": float(baslik1["EXPTIME"]),
            "kazanc_e_adu": kazanc,
            "okuma_gurultusu_e": okuma,
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
    ayarlar = {
        "f_lim=2, komşu eşiği yok": {"f_lim": 2.0, "komsu_esik_orani": 1.0},
        "f_lim=2": {"f_lim": 2.0},
        "f_lim=5": {"f_lim": 5.0},
    }
    for ad, secenek in ayarlar.items():
        temiz, maske = kozmik_isin.kozmik_isin_bul(poz1, kazanc_e_adu=kazanc, okuma_gurultusu_e=okuma, **secenek)
        temizler[ad] = (temiz, maske)
        parlak = gercek1 & ((poz1 - poz2) > 20.0 * sigma_fark)
        sonuc["ayarlar"][ad] = {
            "isaretlenen_piksel": int(maske.sum()),
            "tamlik": float((maske & gercek1).sum() / gercek1.sum()),
            "tamlik_20_sigma_ustu": float((maske & parlak).sum() / parlak.sum()),
            "saflik_bitisik_dahil": float((maske & cevre).sum() / maske.sum()),
            "gercek_disi_isaretlenen_kesir": float((maske & ~cevre).sum() / maske.size),
            # Temizlik niteliği: gerçek kozmik ışın piksellerinde ikinci pozdan ortanca mutlak sapma [fark gürültüsü birimi].
            "temizlik_oncesi_sapma_sigma": float(np.median(np.abs(poz1 - poz2)[gercek1] / sigma_fark[gercek1])),
            "temizlik_sonrasi_sapma_sigma": float(np.median(np.abs(temiz - poz2)[gercek1] / sigma_fark[gercek1])),
        }
    cikti = KOK / "belgeler"
    (cikti / "kozmik_isin_gercek_veri.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(sonuc, ensure_ascii=False, indent=1))

    # Şekil: ham poz, temizlenmiş poz (f_lim = 2) ve ikinci poz; ortak germe.
    duzey = float(np.median(poz1))
    sacilim = float(np.sqrt(kazanc * duzey + okuma**2) / kazanc)
    paneller = [
        ("Birinci poz (400 s)", poz1),
        ("Temizlenmiş birinci poz", temizler["f_lim=2"][0]),
        ("İkinci poz (400 s)", poz2),
    ]
    fig, eksenler = dw.figur(
        "web", hedef="web-640", oran=0.40, ondalik=",", mozaik=[["a", "b", "c"]], gridspec_kw={"wspace": 0.04}
    )
    for ax, (baslik, goruntu) in zip(eksenler.values(), paneller, strict=True):
        gerilmis = germe.asinh_ger(goruntu[KESIT], siyah=duzey - sacilim, alfa=1.0 / (10.0 * sacilim), q=6.0)
        ust = float(
            np.percentile(germe.asinh_ger(poz2[KESIT], siyah=duzey - sacilim, alfa=1.0 / (10.0 * sacilim), q=6.0), 99.8)
        )
        # datawiz: izin göreli parlaklık görüntüsü; asinh gerilmiş değerlerin renk çubuğu okura nicelik vermez
        ax.imshow(np.clip(gerilmis / ust, 0, 1), origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for kenar in ax.spines.values():
            kenar.set_visible(False)
        ax.set_title(baslik, loc="left")
    dw.kaydet(fig, str(cikti / "kozmik_isin_gercek_veri"), hedef="web-640", bicimler=("png",))


if __name__ == "__main__":
    main()
