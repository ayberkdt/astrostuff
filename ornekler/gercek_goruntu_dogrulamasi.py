"""Kaynak saptama ve astrometrinin gerçek görüntüde bağımsız katalogla sınanması.

Görüntü: Alev Bulutsusu'nun (NGC 2024) 2MASS K bandı kesiti, 1 yay saniyesi/piksel
(NASA SkyView). Ölçüt: aynı alanın 2MASS Nokta Kaynak Kataloğu (IRSA).

Ölçülenler:

1. Saflık: saptanan kaynakların katalogda karşılığı olan kesri.
2. Katalog tamlığı: katalog yıldızlarının K kadirine göre saptanan kesri.
3. Astrometri: WCS ile hesaplanan konumların katalog konumlarından farkı.
4. Fotometrik sıfır noktası: açıklık kadiri ile katalog K kadirinin farkı.
5. Yapay yıldız tamlığı: görüntüye eklenen yıldızların geri bulunma kesri.

Katalog aynı taramanın verisinden üretilmiştir; bu sınama kodun doğruluğunu
ölçer, 2MASS'in kendisini değil. SkyView görüntüyü istenen ızgaraya yeniden
örnekler; konum artıkları bu yeniden örneklemeyi de içerir.

Kullanım:
    python ornekler/gercek_goruntu_dogrulamasi.py
"""

from __future__ import annotations

import json
from pathlib import Path

import _cizim as dw
import numpy as np

from kozmos.arsiv import ikimass, skyview
from kozmos.astrometri import cozum, eslestirme, ucgen, wcs
from kozmos.fotometri import kaynak as fk
from kozmos.gozlem import dedektor as dd
from kozmos.istatistik import saglam

KOK = Path(__file__).resolve().parents[1]
RA, DEC, ALAN, PIKSEL = 85.42, -1.90, 0.15, 540
ESLEME_YARICAPI = 2.0  # yay saniyesi
KENAR = 12  # piksel; kenara bu kadar yakın katalog yıldızları tamlık hesabına girmez
ACIKLIK, HALKA = 4.0, (8.0, 12.0)
TOHUM = 20261125


def sapta(goruntu: np.ndarray, ayristir: bool = True) -> fk.KaynakListesi:
    harita, rms = fk.arka_plan_haritasi(goruntu, hucre=60)
    return fk.kaynak_bul(goruntu, harita, rms, esik_sigma=4.0, en_az_piksel=5, ayristirma_duzeyi=32 if ayristir else 0)


def main() -> None:
    goruntu, baslik = skyview.kesit_indir(
        RA, DEC, "2MASS-K", KOK / "veri" / "ham" / "skyview", alan_derece=ALAN, piksel=PIKSEL
    )
    katalog = ikimass.nokta_kaynaklari(RA, DEC, ALAN * 0.75, KOK / "veri" / "ham" / "2mass")
    w = wcs.basliktan_wcs(baslik)
    olcek = abs(float(baslik["CDELT1"])) * 3600.0  # yay saniyesi / piksel

    # Katalog: K bandında gerçekten saptanmış (k_cmsig dolu) ve görüntünün içine düşen yıldızlar.
    kx, ky = wcs.gok_den_piksel(w, katalog["ra"], katalog["dec"])
    icinde = (kx > KENAR) & (kx < PIKSEL - 1 - KENAR) & (ky > KENAR) & (ky < PIKSEL - 1 - KENAR)
    gecerli = icinde & np.isfinite(katalog["k_cmsig"])
    k_ra, k_dec, k_m, kx, ky = (d[gecerli] for d in (katalog["ra"], katalog["dec"], katalog["k_m"], kx, ky))

    kaynaklar = sapta(goruntu)
    s_ra, s_dec = wcs.piksel_den_gok(w, kaynaklar.x, kaynaklar.y)
    ic_kaynak = (
        (kaynaklar.x > KENAR)
        & (kaynaklar.x < PIKSEL - 1 - KENAR)
        & (kaynaklar.y > KENAR)
        & (kaynaklar.y < PIKSEL - 1 - KENAR)
    )

    # 1. Saflık: saptamadan kataloğun tamamına (K üst sınırlı kaynaklar dahil).
    tum = eslestirme.capraz_esle(s_ra, s_dec, katalog["ra"], katalog["dec"], ESLEME_YARICAPI)
    saflik = float(tum.eslesti[ic_kaynak].mean())

    # 2. Katalog tamlığı: katalogdan saptamaya.
    ters = eslestirme.capraz_esle(k_ra, k_dec, s_ra, s_dec, ESLEME_YARICAPI)
    kenarlar = np.arange(6.0, 16.01, 1.0)
    orta = 0.5 * (kenarlar[:-1] + kenarlar[1:])
    sayi = np.array([np.sum((k_m >= a) & (k_m < b)) for a, b in zip(kenarlar[:-1], kenarlar[1:], strict=True)])
    bulunan = np.array(
        [np.sum(ters.eslesti[(k_m >= a) & (k_m < b)]) for a, b in zip(kenarlar[:-1], kenarlar[1:], strict=True)]
    )

    # 3. Astrometri: eşleşen çiftlerde konum farkı [yay saniyesi].
    e = ters.eslesti
    d_ra = (s_ra[ters.indis[e]] - k_ra[e]) * np.cos(np.radians(k_dec[e])) * 3600.0
    d_dec = (s_dec[ters.indis[e]] - k_dec[e]) * 3600.0

    # 4. Sıfır noktası ve PSF genişliği: parlak, yalıtılmış, doymamış yıldızlardan.
    yalitik = np.array([np.sum(np.hypot(kx - x, ky - y) < 15.0) == 1 for x, y in zip(kx, ky, strict=True)])
    secim = e & yalitik & (k_m > 9.0) & (k_m < 12.5)
    aki, _ = fk.aciklik_fotometrisi(goruntu, kx[secim], ky[secim], ACIKLIK, halka=HALKA)
    dar, _ = fk.aciklik_fotometrisi(goruntu, kx[secim], ky[secim], 1.5, halka=HALKA)
    genis, _ = fk.aciklik_fotometrisi(goruntu, kx[secim], ky[secim], 6.0, halka=HALKA)
    iyi = (aki > 0) & (dar > 0) & (genis > dar)
    sifir_noktalari = k_m[secim][iyi] + 2.5 * np.log10(aki[iyi])
    sifir = float(np.median(sifir_noktalari))
    sifir_sacilim = float(saglam.mad_olcek(sifir_noktalari))
    # Gauss PSF'te r yarıçapında çevrelenen akı kesri 1 − exp(−r²/2σ²): iki açıklığın oranından σ.
    oran = np.median(dar[iyi] / genis[iyi])
    sigmalar = np.linspace(0.5, 4.0, 3501)
    model = (1 - np.exp(-(1.5**2) / (2 * sigmalar**2))) / (1 - np.exp(-(6.0**2) / (2 * sigmalar**2)))
    sigma_psf = float(sigmalar[np.argmin(np.abs(model - oran))])
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_psf

    # 5. Yapay yıldız deneyi: aynı görüntüye bilinen kadirde Gauss yıldızlar eklenir ve yeniden saptanır.
    uretec = np.random.default_rng(TOHUM)
    aciklik_kesri = 1.0 - np.exp(-(ACIKLIK**2) / (2 * sigma_psf**2))
    yapay_kadir = np.arange(10.5, 16.01, 0.5)
    yapay_kesir = []
    for m in yapay_kadir:
        bulundu = toplam = 0
        for _ in range(4):
            x = uretec.uniform(KENAR, PIKSEL - 1 - KENAR, 40)
            y = uretec.uniform(KENAR, PIKSEL - 1 - KENAR, 40)
            toplam_aki = 10 ** (-0.4 * (m - sifir)) / aciklik_kesri
            yeni = sapta(goruntu + dd.yildiz_alani(goruntu.shape, x, y, np.full(40, toplam_aki), fwhm))
            uzaklik = np.hypot(yeni.x[None, :] - x[:, None], yeni.y[None, :] - y[:, None]).min(axis=1)
            bulundu += int(np.sum(uzaklik < 1.5))
            toplam += 40
        yapay_kesir.append(bulundu / toplam)
    yapay_kesir = np.array(yapay_kesir)

    # 6. Görüntüden astrometrik çözüm: başlıktaki WCS bilerek bozulur (0,4° dönme, %0,4 ölçek, 18 ve −14 piksel
    # kaydırma) ve yalnızca saptanan kaynaklarla katalogdan yeniden çözülür; sonuç başlıktaki WCS ile karşılaştırılır.
    aci = np.radians(0.4)
    don = np.array([[np.cos(aci), -np.sin(aci)], [np.sin(aci), np.cos(aci)]])
    bozuk = wcs.Wcs(crpix=w.crpix + np.array([18.0, -14.0]), crval=w.crval, cd=1.004 * don @ w.cd)
    izgara_x, izgara_y = np.meshgrid(np.linspace(0, PIKSEL - 1, 5), np.linspace(0, PIKSEL - 1, 5))

    def basliktan_fark(aday: wcs.Wcs) -> float:
        r1, d1 = wcs.piksel_den_gok(aday, izgara_x, izgara_y)
        r2, d2 = wcs.piksel_den_gok(w, izgara_x, izgara_y)
        return float(np.max(np.hypot((r1 - r2) * np.cos(np.radians(d2)), d1 - d2)) * 3600.0)

    kaymasiz, oy = cozum.kaydirma_bul(
        bozuk, kaynaklar.x, kaynaklar.y, katalog["ra"], katalog["dec"], en_cok_piksel=40.0
    )
    cozulen = cozum.wcs_iyilestir(
        kaymasiz, kaynaklar.x, kaynaklar.y, katalog["ra"], katalog["dec"], yaricap_yaysaniye=6.0
    )
    astrometrik_cozum = {
        "bozulma": {"donme_derece": 0.4, "olcek_carpani": 1.004, "kaydirma_piksel": [18.0, -14.0]},
        "bozuk_wcs_en_buyuk_hata_yaysaniye": basliktan_fark(bozuk),
        "kaydirma_oyu": oy,
        "kaydirma_sonrasi_en_buyuk_hata_yaysaniye": basliktan_fark(kaymasiz),
        "cozum_en_buyuk_hata_yaysaniye": basliktan_fark(cozulen.wcs),
        "cozume_giren_yildiz": cozulen.yildiz_sayisi,
        "katalog_artik_rms_yaysaniye": cozulen.artik_rms_yaysaniye,
    }

    # 7. Kör çözüm: başlıktaki WCS hiç kullanılmaz; yalnızca saptanan kaynakların piksel konumları ve akıları ile
    # aynı bölgenin katalogu verilir (ölçek, yönelim, ayna ve teğet noktası bilinmez).
    kor, cift = ucgen.kor_cozum(
        kaynaklar.x, kaynaklar.y, kaynaklar.aki, katalog["ra"], katalog["dec"], katalog["k_m"], en_parlak=30
    )
    astrometrik_cozum["kor_cozum"] = {
        "ucgen_eslemesinde_atanan_cift": cift,
        "cozume_giren_yildiz": kor.yildiz_sayisi,
        "basliktan_en_buyuk_fark_yaysaniye": basliktan_fark(kor.wcs),
        "katalog_artik_rms_yaysaniye": kor.artik_rms_yaysaniye,
        "olcek_yaysaniye_piksel": float(np.sqrt(abs(np.linalg.det(kor.wcs.cd))) * 3600.0),
    }

    sonuc = {
        "goruntu": {
            "tarama": "2MASS-K (SkyView)",
            "merkez_derece": [RA, DEC],
            "olcek_yaysaniye_piksel": olcek,
            "piksel": PIKSEL,
        },
        "katalog": {"toplam": int(len(katalog["ra"])), "goruntude_k_saptanmis": int(gecerli.sum())},
        "saptanan_kaynak": int(ic_kaynak.sum()),
        "saptanan_kaynak_ayristirmasiz": len(sapta(goruntu, ayristir=False)),
        "esleme_yaricapi_yaysaniye": ESLEME_YARICAPI,
        "saflik": saflik,
        "katalog_tamligi": {
            "k_orta": orta.tolist(),
            "katalog_sayisi": sayi.tolist(),
            "bulunan": bulunan.tolist(),
        },
        "astrometri_yaysaniye": {
            "cift_sayisi": int(e.sum()),
            "ortanca_d_ra_cos_dec": float(np.median(d_ra)),
            "ortanca_d_dec": float(np.median(d_dec)),
            "sacilim_d_ra": float(saglam.mad_olcek(d_ra)),
            "sacilim_d_dec": float(saglam.mad_olcek(d_dec)),
            "ortanca_ayriklik": float(np.median(ters.ayriklik_yaysaniye[e])),
        },
        "fotometri": {
            "yildiz_sayisi": int(iyi.sum()),
            "sifir_noktasi_kadir": sifir,
            "sacilim_kadir": sifir_sacilim,
            "psf_fwhm_piksel": fwhm,
        },
        "astrometrik_cozum": astrometrik_cozum,
        "yapay_yildiz": {"k": yapay_kadir.tolist(), "bulunma_kesri": yapay_kesir.tolist(), "tohum": TOHUM},
    }
    cikti = KOK / "belgeler"
    (cikti / "gercek_goruntu_dogrulamasi.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(sonuc, ensure_ascii=False, indent=2))

    # Şekil: (a) tamlık eğrileri, (b) konum artıkları.
    fig, eksenler = dw.figur("web", hedef="web-640", oran=0.48, ondalik=",", mozaik=[["a", "b"]])
    a, b = eksenler["a"], eksenler["b"]
    dolu = sayi >= 5
    a.plot(orta[dolu], 100.0 * bulunan[dolu] / sayi[dolu], "o-", label="Katalog yıldızları")
    a.plot(yapay_kadir, 100.0 * yapay_kesir, "s--", label="Yapay yıldızlar")
    a.set_xlabel("K kadiri (Vega)")
    a.set_ylabel("Saptanan kesir (%)")
    a.set_ylim(0, 105)
    b.scatter(d_ra, d_dec, s=8, alpha=0.6, linewidths=0)
    b.axhline(0.0, color="0.6", lw=0.6, zorder=0)
    b.axvline(0.0, color="0.6", lw=0.6, zorder=0)
    b.set_xlim(-2.1, 2.1)
    b.set_ylim(-2.1, 2.1)
    b.set_aspect("equal")
    b.set_xlabel("Δα cos δ (yay saniyesi)")
    b.set_ylabel("Δδ (yay saniyesi)")
    dw.lejant.yerlestir(a)
    dw.kaydet(fig, str(cikti / "gercek_goruntu_dogrulamasi"), hedef="web-640", bicimler=("png",))


if __name__ == "__main__":
    main()
