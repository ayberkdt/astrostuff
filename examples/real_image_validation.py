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
    python examples/real_image_validation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import _plotting
import numpy as np

from kozmos.archives import skyview, twomass
from kozmos.astrometry import matching, solve, triangles, wcs
from kozmos.observing import detector
from kozmos.photometry import sources
from kozmos.statistics import robust

ROOT = Path(__file__).resolve().parents[1]
RA, DEC, ALAN, PIKSEL = 85.42, -1.90, 0.15, 540
ESLEME_YARICAPI = 2.0  # yay saniyesi
KENAR = 12  # piksel; kenara bu kadar yakın katalog yıldızları tamlık hesabına girmez
ACIKLIK, HALKA = 4.0, (8.0, 12.0)
TOHUM = 20261125


def sapta(image: np.ndarray, ayristir: bool = True) -> sources.SourceList:
    harita, rms = sources.background_map(image, cell=60)
    return sources.detect_sources(
        image, harita, rms, threshold_sigma=4.0, min_pixels=5, deblend_levels=32 if ayristir else 0
    )


def main() -> None:
    image, header = skyview.download_cutout(
        RA, DEC, "2MASS-K", ROOT / "data" / "raw" / "skyview", size_deg=ALAN, pixels=PIKSEL
    )
    katalog = twomass.point_sources(RA, DEC, ALAN * 0.75, ROOT / "data" / "raw" / "2mass")
    w = wcs.wcs_from_header(header)
    olcek = abs(float(header["CDELT1"])) * 3600.0  # yay saniyesi / piksel

    # Katalog: K bandında gerçekten saptanmış (k_cmsig dolu) ve görüntünün içine düşen yıldızlar.
    kx, ky = wcs.sky_to_pixel(w, katalog["ra"], katalog["dec"])
    icinde = (kx > KENAR) & (kx < PIKSEL - 1 - KENAR) & (ky > KENAR) & (ky < PIKSEL - 1 - KENAR)
    gecerli = icinde & np.isfinite(katalog["k_cmsig"])
    k_ra, k_dec, k_m, kx, ky = (d[gecerli] for d in (katalog["ra"], katalog["dec"], katalog["k_m"], kx, ky))

    sources_of = sapta(image)
    s_ra, s_dec = wcs.pixel_to_sky(w, sources_of.x, sources_of.y)
    ic_kaynak = (
        (sources_of.x > KENAR)
        & (sources_of.x < PIKSEL - 1 - KENAR)
        & (sources_of.y > KENAR)
        & (sources_of.y < PIKSEL - 1 - KENAR)
    )

    # 1. Saflık: saptamadan kataloğun tamamına (K üst sınırlı kaynaklar dahil).
    tum = matching.cross_match(s_ra, s_dec, katalog["ra"], katalog["dec"], ESLEME_YARICAPI)
    saflik = float(tum.matched[ic_kaynak].mean())

    # 2. Katalog tamlığı: katalogdan saptamaya.
    inverse = matching.cross_match(k_ra, k_dec, s_ra, s_dec, ESLEME_YARICAPI)
    kenarlar = np.arange(6.0, 16.01, 1.0)
    orta = 0.5 * (kenarlar[:-1] + kenarlar[1:])
    number = np.array([np.sum((k_m >= a) & (k_m < b)) for a, b in zip(kenarlar[:-1], kenarlar[1:], strict=True)])
    bulunan = np.array(
        [np.sum(inverse.matched[(k_m >= a) & (k_m < b)]) for a, b in zip(kenarlar[:-1], kenarlar[1:], strict=True)]
    )

    # 3. Astrometri: eşleşen çiftlerde konum farkı [yay saniyesi].
    e = inverse.matched
    d_ra = (s_ra[inverse.index[e]] - k_ra[e]) * np.cos(np.radians(k_dec[e])) * 3600.0
    d_dec = (s_dec[inverse.index[e]] - k_dec[e]) * 3600.0

    # 4. Sıfır noktası ve PSF genişliği: parlak, yalıtılmış, doymamış yıldızlardan.
    yalitik = np.array([np.sum(np.hypot(kx - x, ky - y) < 15.0) == 1 for x, y in zip(kx, ky, strict=True)])
    secim = e & yalitik & (k_m > 9.0) & (k_m < 12.5)
    flux, _ = sources.aperture_photometry(image, kx[secim], ky[secim], ACIKLIK, annulus=HALKA)
    dar, _ = sources.aperture_photometry(image, kx[secim], ky[secim], 1.5, annulus=HALKA)
    genis, _ = sources.aperture_photometry(image, kx[secim], ky[secim], 6.0, annulus=HALKA)
    iyi = (flux > 0) & (dar > 0) & (genis > dar)
    sifir_noktalari = k_m[secim][iyi] + 2.5 * np.log10(flux[iyi])
    sifir = float(np.median(sifir_noktalari))
    sifir_sacilim = float(robust.mad_scale(sifir_noktalari))
    # Gauss PSF'te r yarıçapında çevrelenen akı kesri 1 − exp(−r²/2σ²): iki açıklığın oranından σ.
    ratio = np.median(dar[iyi] / genis[iyi])
    sigmalar = np.linspace(0.5, 4.0, 3501)
    model = (1 - np.exp(-(1.5**2) / (2 * sigmalar**2))) / (1 - np.exp(-(6.0**2) / (2 * sigmalar**2)))
    sigma_psf = float(sigmalar[np.argmin(np.abs(model - ratio))])
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_psf

    # 5. Yapay yıldız deneyi: aynı görüntüye bilinen kadirde Gauss yıldızlar eklenir ve yeniden saptanır.
    rng = np.random.default_rng(TOHUM)
    aciklik_kesri = 1.0 - np.exp(-(ACIKLIK**2) / (2 * sigma_psf**2))
    yapay_kadir = np.arange(10.5, 16.01, 0.5)
    yapay_kesir = []
    for m in yapay_kadir:
        bulundu = toplam = 0
        for _ in range(4):
            x = rng.uniform(KENAR, PIKSEL - 1 - KENAR, 40)
            y = rng.uniform(KENAR, PIKSEL - 1 - KENAR, 40)
            toplam_aki = 10 ** (-0.4 * (m - sifir)) / aciklik_kesri
            yeni = sapta(image + detector.star_field(image.shape, x, y, np.full(40, toplam_aki), fwhm))
            uzaklik = np.hypot(yeni.x[None, :] - x[:, None], yeni.y[None, :] - y[:, None]).min(axis=1)
            bulundu += int(np.sum(uzaklik < 1.5))
            toplam += 40
        yapay_kesir.append(bulundu / toplam)
    yapay_kesir = np.array(yapay_kesir)

    # 6. Görüntüden astrometrik çözüm: başlıktaki WCS bilerek bozulur (0,4° dönme, %0,4 ölçek, 18 ve −14 piksel
    # kaydırma) ve yalnızca saptanan kaynaklarla katalogdan yeniden çözülür; sonuç başlıktaki WCS ile karşılaştırılır.
    angle = np.radians(0.4)
    don = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    bozuk = wcs.Wcs(crpix=w.crpix + np.array([18.0, -14.0]), crval=w.crval, cd=1.004 * don @ w.cd)
    izgara_x, izgara_y = np.meshgrid(np.linspace(0, PIKSEL - 1, 5), np.linspace(0, PIKSEL - 1, 5))

    def basliktan_fark(aday: wcs.Wcs) -> float:
        r1, d1 = wcs.pixel_to_sky(aday, izgara_x, izgara_y)
        r2, d2 = wcs.pixel_to_sky(w, izgara_x, izgara_y)
        return float(np.max(np.hypot((r1 - r2) * np.cos(np.radians(d2)), d1 - d2)) * 3600.0)

    kaymasiz, oy = solve.find_shift(
        bozuk, sources_of.x, sources_of.y, katalog["ra"], katalog["dec"], max_shift_pix=40.0
    )
    cozulen = solve.refine_wcs(kaymasiz, sources_of.x, sources_of.y, katalog["ra"], katalog["dec"], radius_arcsec=6.0)
    astrometrik_cozum = {
        "bozulma": {"donme_derece": 0.4, "olcek_carpani": 1.004, "kaydirma_piksel": [18.0, -14.0]},
        "bozuk_wcs_en_buyuk_hata_yaysaniye": basliktan_fark(bozuk),
        "kaydirma_oyu": oy,
        "kaydirma_sonrasi_en_buyuk_hata_yaysaniye": basliktan_fark(kaymasiz),
        "cozum_en_buyuk_hata_yaysaniye": basliktan_fark(cozulen.wcs),
        "cozume_giren_yildiz": cozulen.n_stars,
        "katalog_artik_rms_yaysaniye": cozulen.residual_rms_arcsec,
    }

    # 7. Kör çözüm: başlıktaki WCS hiç kullanılmaz; yalnızca saptanan kaynakların piksel konumları ve akıları ile
    # aynı bölgenin katalogu verilir (ölçek, yönelim, ayna ve teğet noktası bilinmez).
    kor, cift = triangles.blind_solve(
        sources_of.x, sources_of.y, sources_of.flux, katalog["ra"], katalog["dec"], katalog["k_m"], n_brightest=30
    )
    astrometrik_cozum["kor_cozum"] = {
        "ucgen_eslemesinde_atanan_cift": cift,
        "cozume_giren_yildiz": kor.n_stars,
        "basliktan_en_buyuk_fark_yaysaniye": basliktan_fark(kor.wcs),
        "katalog_artik_rms_yaysaniye": kor.residual_rms_arcsec,
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
        "saptanan_kaynak_ayristirmasiz": len(sapta(image, ayristir=False)),
        "esleme_yaricapi_yaysaniye": ESLEME_YARICAPI,
        "saflik": saflik,
        "katalog_tamligi": {
            "k_orta": orta.tolist(),
            "katalog_sayisi": number.tolist(),
            "bulunan": bulunan.tolist(),
        },
        "astrometri_yaysaniye": {
            "cift_sayisi": int(e.sum()),
            "ortanca_d_ra_cos_dec": float(np.median(d_ra)),
            "ortanca_d_dec": float(np.median(d_dec)),
            "sacilim_d_ra": float(robust.mad_scale(d_ra)),
            "sacilim_d_dec": float(robust.mad_scale(d_dec)),
            "ortanca_ayriklik": float(np.median(inverse.separation_arcsec[e])),
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
    cikti = ROOT / "figures"
    (cikti / "gercek_goruntu_dogrulamasi.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(sonuc, ensure_ascii=False, indent=2))

    # Şekil: (a) tamlık eğrileri, (b) konum artıkları.
    fig, eksenler = _plotting.figure("web", target="web-640", ratio=0.48, decimal=",", mosaic=[["a", "b"]])
    a, b = eksenler["a"], eksenler["b"]
    dolu = number >= 5
    a.plot(orta[dolu], 100.0 * bulunan[dolu] / number[dolu], "o-", label="Katalog yıldızları")
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
    _plotting.legend.place(a)
    _plotting.save(fig, str(cikti / "gercek_goruntu_dogrulamasi"), target="web-640", formats=("png",))


if __name__ == "__main__":
    main()
