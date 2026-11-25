"""Gerçek ham yakın kızılötesi karelerin uçtan uca indirgenmesi.

Veri: ESO NTT/SOFI, Ks bandı, SN 2011bm alanı, program 184.D-1152(F), 15 Mart
2012; dört köşeli kaydırma düzeninde 16 ham kare (her biri 15 × 6 s). Kareler
ESO arşivinden ``.fits.Z`` olarak iner; açılmış ``.fits`` dosyaları beklenir
(``gzip -d`` ``.Z`` dosyalarını açar). Dosya kimlikleri aşağıdaki ``KARE_KIMLIKLERI``
listesindedir; her biri ``https://dataportal.eso.org/dataportal_new/file/<kimlik>``
adresinden iner ve ``veri/ham/eso/`` altına, adındaki ``:`` yerine ``-`` konarak kaydedilir.

Adımlar:

1. Maskesiz durağan gökyüzü modeli, bozuk piksel haritası, kare başına saptama.
2. Kareler arası öteleme: başlıktaki kaydırma komutlarından başlar, yıldızlarla inceltilir.
3. İlk yığından nesne maskesi; maskeli durağan ve dinamik gökyüzü modelleri.
4. Yığma, kaynak saptama, 2MASS kataloğuyla astrometrik çözüm ve sıfır noktası.

Bu sınamada "gerçek" bilinmez; ölçütler iç tutarlılıktır (artık arka planın
düzlüğü, aynı yıldızın 16 karedeki akısının yinelenebilirliği) ve bağımsız
katalogdur (2MASS konumları ve K kadirleri).

Kullanım:
    python ornekler/kizilotesi_indirgeme.py
"""

from __future__ import annotations

import json
from pathlib import Path

import _cizim as dw
import numpy as np
from scipy import ndimage

from kozmos.arsiv import ikimass
from kozmos.astrometri import cozum, eslestirme, wcs
from kozmos.fotometri import kaynak as fk
from kozmos.gorsel import germe
from kozmos.indirgeme import gokyuzu
from kozmos.io import fits
from kozmos.istatistik import saglam

KOK = Path(__file__).resolve().parents[1]
DIZIN = KOK / "veri" / "ham" / "eso"
ACIKLIK, HALKA = 5.0, (10.0, 15.0)
KARE_KIMLIKLERI = (
    "SOFI.2012-03-15T07:15:00.783",
    "SOFI.2012-03-15T07:16:53.872",
    "SOFI.2012-03-15T07:18:47.284",
    "SOFI.2012-03-15T07:20:40.514",
    "SOFI.2012-03-15T07:22:37.692",
    "SOFI.2012-03-15T07:24:31.062",
    "SOFI.2012-03-15T07:26:24.182",
    "SOFI.2012-03-15T07:28:17.314",
    "SOFI.2012-03-15T07:30:12.942",
    "SOFI.2012-03-15T07:32:06.113",
    "SOFI.2012-03-15T07:33:59.183",
    "SOFI.2012-03-15T07:35:52.612",
    "SOFI.2012-03-15T07:37:48.142",
    "SOFI.2012-03-15T07:39:41.265",
    "SOFI.2012-03-15T07:41:34.245",
    "SOFI.2012-03-15T07:43:27.413",
)


def sapta(goruntu: np.ndarray, esik: float) -> fk.KaynakListesi:
    harita, rms = fk.arka_plan_haritasi(goruntu, hucre=64)
    return fk.kaynak_bul(goruntu, harita, rms, esik_sigma=esik, en_az_piksel=8, yumusatma_fwhm=3.0)


def eslesen_fark(liste0, liste1, tahmin, yaricap: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """liste1 ≈ liste0 + tahmin varsayımıyla ``yaricap`` içinde eşleşen çiftlerin (dx, dy) farkları."""
    dx = liste1.x[:, None] - liste0.x[None, :]
    dy = liste1.y[:, None] - liste0.y[None, :]
    yakin = np.hypot(dx - tahmin[0], dy - tahmin[1]) < yaricap
    return dx[yakin], dy[yakin]


def yigin(kareler: np.ndarray, oteleme: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kareleri tam piksel ötelemeyle ortak tuvale yerleştirip ortalar; (yığın, kapsama, tuval kökeni)."""
    tam = np.round(oteleme).astype(int)
    koken = tam.max(axis=0)  # tuval pikseli = kare pikseli − öteleme + köken
    ny, nx = kareler.shape[1:]
    toplam = np.zeros((ny + int(koken[1] - tam[:, 1].min()), nx + int(koken[0] - tam[:, 0].min())))
    sayi = np.zeros_like(toplam)
    for kare, (dx, dy) in zip(kareler, tam, strict=True):
        x0, y0 = int(koken[0] - dx), int(koken[1] - dy)
        iyi = np.isfinite(kare)
        toplam[y0 : y0 + ny, x0 : x0 + nx] += np.where(iyi, kare, 0.0)
        sayi[y0 : y0 + ny, x0 : x0 + nx] += iyi
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sayi > 0, toplam / sayi, np.nan), sayi, koken


def buyuk_olcekli_yapi(kareler: np.ndarray, maskeler: np.ndarray, kutu: int = 64) -> float:
    """Maskesiz piksellerin 64 piksellik kutu ortancalarının kare içi saçılımı; kareler üzerinden ortanca [ADU]."""
    sonuc = []
    for kare, maske in zip(kareler, maskeler, strict=True):
        d = np.where(maske, np.nan, kare)
        n = d.shape[0] // kutu
        bloklar = d[: n * kutu, : n * kutu].reshape(n, kutu, n, kutu).swapaxes(1, 2).reshape(n, n, -1)
        sonuc.append(np.nanstd(np.nanmedian(bloklar, axis=2)))
    return float(np.median(sonuc))


def main() -> None:
    yollar = sorted(DIZIN.glob("SOFI*.fits"))
    if len(yollar) < 8:
        raise SystemExit(f"{DIZIN} altında açılmış SOFI kareleri yok; kimlikler KARE_KIMLIKLERI listesinde")
    kareler, zaman, basliklar = [], [], []
    for yol in yollar:
        veri, baslik = fits.fits_oku(yol)
        kareler.append(veri)
        zaman.append(float(baslik["MJD-OBS"]) * 1440.0)  # dakika
        basliklar.append(baslik)
    kareler, zaman = np.array(kareler), np.array(zaman)
    zaman -= zaman[0]
    duzey = np.median(kareler, axis=(1, 2))
    k_sayi, (ny, nx) = len(kareler), kareler.shape[1:]
    b0 = basliklar[0]

    # 1. Maskesiz durağan model. Bozuk pikseller (sıcak, ölü): gökyüzü modelinde yerel ortancadan sapanlar.
    # Ham karede karanlık ve düz alan düzeltmesi yoktur; bu pikseller her karede aynı yerde durur.
    ilk, ilk_model = gokyuzu.duragan_gokyuzu_cikar(kareler)
    yerel = ilk_model - ndimage.median_filter(ilk_model, size=5, mode="nearest")
    bozuk = ndimage.binary_dilation(np.abs(yerel) > 6.0 * saglam.mad_olcek(yerel), structure=np.ones((3, 3)))
    ilk_temiz = ilk.copy()
    ilk_temiz[:, bozuk] = 0.0
    listeler = [sapta(kare, esik=6.0) for kare in ilk_temiz]

    # 2. Öteleme. Alan seyrektir (karede bir avuç yıldız); kör oylama güvenilmez. Başlıktaki kaydırma komutları
    # başlangıç tahminidir; komutun dedektör eksenleriyle eşleşmesi ve işareti (sekiz olasılık) en çok yıldız
    # eşleştiren seçenek olarak veriden bulunur, sonra öteleme eşleşen yıldızlarla inceltilir.
    komut = np.array([[b["ESO SEQ CUMOFFSETX"], b["ESO SEQ CUMOFFSETY"]] for b in basliklar])
    komut -= komut[0]
    adaylar = []
    for sira in ((0, 1), (1, 0)):
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                tahmin = np.column_stack([sx * komut[:, sira[0]], sy * komut[:, sira[1]]])
                puan = sum(eslesen_fark(listeler[0], listeler[i], tahmin[i])[0].size for i in range(1, k_sayi))
                adaylar.append((puan, sira, (sx, sy), tahmin))
    adaylar.sort(key=lambda aday: -aday[0])
    _, komut_sirasi, komut_isareti, tahmin = adaylar[0]
    oteleme, eslesen = [(0.0, 0.0)], []
    for i in range(1, k_sayi):
        dx, dy = eslesen_fark(listeler[0], listeler[i], tahmin[i])
        oteleme.append((float(np.median(dx)), float(np.median(dy))) if dx.size else tuple(tahmin[i]))
        eslesen.append(int(dx.size))
    oteleme = np.array(oteleme)
    tam = np.round(oteleme).astype(int)

    # 3. İlk yığından nesne maskesi (yığın koordinatında), her karenin koordinatına geri taşınır.
    ilk_yigin, kapsama, koken = yigin(np.where(bozuk[None], np.nan, ilk), oteleme)
    tam_kapsama = kapsama >= 0.5 * k_sayi
    dolgu = np.where(tam_kapsama, ilk_yigin, np.nanmedian(ilk_yigin[tam_kapsama]))
    y_harita, _ = fk.arka_plan_haritasi(dolgu, hucre=64)
    puruzsuz = ndimage.gaussian_filter(dolgu - y_harita, 1.5)
    nesne = (puruzsuz > 3.0 * saglam.mad_olcek(puruzsuz[tam_kapsama])) & tam_kapsama
    nesne = ndimage.binary_dilation(nesne, structure=np.ones((3, 3)), iterations=3)
    maskeler = np.array(
        [
            nesne[int(koken[1] - dy) : int(koken[1] - dy) + ny, int(koken[0] - dx) : int(koken[0] - dx) + nx]
            for dx, dy in tam
        ]
    )

    duragan, model = gokyuzu.duragan_gokyuzu_cikar(kareler, maskeler=maskeler)
    dinamik = gokyuzu.dinamik_gokyuzu_cikar(kareler, zaman, komsu_sayisi=8, maskeler=maskeler)
    sabit = kareler - duzey[:, None, None]  # yalnızca düzey çıkarılmış: karşılaştırma tabanı
    disarida = maskeler | bozuk[None]  # ölçütlerde nesneler ve bozuk pikseller dışarıda tutulur

    def gurultu(dizi: np.ndarray) -> float:
        return float(np.median([saglam.mad_olcek(k[~m]) for k, m in zip(dizi, disarida, strict=True)]))

    # 4. Yığın (bozuk pikseller girmez), saptama, astrometri, sıfır noktası.
    son_yigin, kapsama, koken = yigin(np.where(bozuk[None], np.nan, dinamik), oteleme)
    tam_kapsama = kapsama >= 0.5 * k_sayi
    yigin_g = np.where(tam_kapsama, son_yigin, np.nanmedian(son_yigin[tam_kapsama]))
    kaynaklar = sapta(yigin_g, esik=5.0)
    yi, xi = np.round(kaynaklar.y).astype(int), np.round(kaynaklar.x).astype(int)
    icte = ndimage.binary_erosion(tam_kapsama, iterations=15)[yi, xi]
    kx, ky = kaynaklar.x[icte], kaynaklar.y[icte]

    # Yığın pikseli = ilk karenin pikseli + köken (ilk karenin ötelemesi sıfır).
    baslik_wcs = wcs.basliktan_wcs(b0)
    baslik_wcs = wcs.Wcs(crpix=baslik_wcs.crpix + koken, crval=baslik_wcs.crval, cd=baslik_wcs.cd)
    katalog = ikimass.nokta_kaynaklari(float(b0["CRVAL1"]), float(b0["CRVAL2"]), 0.09, KOK / "veri" / "ham" / "2mass")
    kaymasiz, oy = cozum.kaydirma_bul(
        baslik_wcs, kx, ky, katalog["ra"], katalog["dec"], en_cok_piksel=320.0, kutu_piksel=4.0
    )
    cozulen = cozum.wcs_iyilestir(
        kaymasiz, kx, ky, katalog["ra"], katalog["dec"], yaricap_yaysaniye=4.0, son_yaricap_yaysaniye=1.5
    )
    ra, dec = wcs.piksel_den_gok(cozulen.wcs, kx, ky)
    e = eslestirme.capraz_esle(ra, dec, katalog["ra"], katalog["dec"], 1.5)
    aki, _ = fk.aciklik_fotometrisi(yigin_g, kx, ky, ACIKLIK, halka=HALKA)
    k_iyi = e.eslesti & (katalog["k_cmsig"][e.indis] < 0.2) & (aki > 0)  # nan < 0.2 yanlıştır: üst sınırlar düşer
    # Çözülen WCS ile yığının tam kapsanan bölgesine düşen katalog yıldızları ve bunların saptanan kesri.
    px, py = wcs.gok_den_piksel(cozulen.wcs, katalog["ra"], katalog["dec"])
    ic_bolge = ndimage.binary_erosion(tam_kapsama, iterations=15)
    alanda = np.isfinite(px) & (px > 0) & (px < yigin_g.shape[1] - 1) & (py > 0) & (py < yigin_g.shape[0] - 1)
    alanda[alanda] = ic_bolge[np.round(py[alanda]).astype(int), np.round(px[alanda]).astype(int)]
    ters = eslestirme.capraz_esle(katalog["ra"][alanda], katalog["dec"][alanda], ra, dec, 1.5)
    sifir = katalog["k_m"][e.indis][k_iyi] + 2.5 * np.log10(aki[k_iyi])

    def yinelenebilirlik(dizi: np.ndarray, duz_alan: np.ndarray | None = None) -> tuple[float, float, int]:
        """Parlak yıldızların 16 karedeki açıklık akısının bağıl sağlam saçılımı (yıldızlar üzerinden ortanca)."""
        olcumler, beklenen = [], []
        piksel_sigma = gurultu(dizi)
        for i in np.argsort(-aki):
            x_k = kx[i] - koken[0] + oteleme[:, 0]
            y_k = ky[i] - koken[1] + oteleme[:, 1]
            if np.any((x_k < 20) | (x_k > nx - 21) | (y_k < 20) | (y_k > ny - 21)):
                continue
            if max(kareler[k][int(round(y_k[k])), int(round(x_k[k]))] for k in range(k_sayi)) > 20000.0:
                continue  # doyma bölgesinden uzak dur
            f = np.array(
                [
                    float(
                        fk.aciklik_fotometrisi(
                            np.nan_to_num(dizi[k] if duz_alan is None else dizi[k] / duz_alan),
                            x_k[k],
                            y_k[k],
                            ACIKLIK,
                            halka=HALKA,
                        )[0]
                    )
                    for k in range(k_sayi)
                ]
            )
            if np.median(f) > 0:
                olcumler.append(saglam.mad_olcek(f) / np.median(f))
                # Beklenen: açıklıktaki n_pix pikselin ve halkadan kestirilen arka planın gürültüsü (kaynak fotonu hariç).
                n_pix, n_b = np.pi * ACIKLIK**2, np.pi * (HALKA[1] ** 2 - HALKA[0] ** 2)
                beklenen.append(piksel_sigma * np.sqrt(n_pix * (1.0 + n_pix / n_b)) / np.median(f))
            if len(olcumler) >= 10:
                break
        return float(np.median(olcumler)), float(np.median(np.array(olcumler) / np.array(beklenen))), len(olcumler)

    yinele = {}
    for ad, dizi, duz in (
        ("duragan_maskesiz", ilk, None),
        ("duragan_maskeli", duragan, None),
        ("dinamik_maskeli", dinamik, None),
        ("dinamik_maskeli_duz_alanli", dinamik, model),
    ):
        sacilim, oran, n_yildiz = yinelenebilirlik(dizi, duz)
        yinele[ad] = {"bagil_sacilim": sacilim, "beklenen_gurultuye_orani": oran}
        yinele["yildiz_sayisi"] = n_yildiz

    sonuc = {
        "veri": {
            "alet": "ESO NTT / SOFI",
            "program": "184.D-1152(F)",
            "hedef": str(b0["OBJECT"]).strip(),
            "suzgec": str(b0["ESO INS FILT1 NAME"]).strip(),
            "kare_sayisi": k_sayi,
            "dit_s": float(b0["ESO DET DIT"]),
            "ndit": int(b0["ESO DET NDIT"]),
            "dizi_suresi_dakika": float(zaman[-1]),
        },
        "gokyuzu": {
            "duzey_adu_en_az": float(duzey.min()),
            "duzey_adu_en_cok": float(duzey.max()),
            "model_sacilimi": float(np.std(model)),
        },
        "bozuk_piksel_kesri": float(bozuk.mean()),
        "nesne_maskesi_kesri": float(maskeler.mean()),
        "oteleme": {
            "kare_basina_saptanan_ortanca": float(np.median([len(liste) for liste in listeler])),
            "en_buyuk_piksel": float(np.abs(oteleme).max()),
            "kare_basina_eslesen_en_az": int(min(eslesen)),
            "en_iyi_ve_ikinci_secenegin_puani": [int(adaylar[0][0]), int(adaylar[1][0])],
            "baslik_komutu_eksen_sirasi": list(komut_sirasi),
            "baslik_komutu_isareti": list(komut_isareti),
            "baslik_komutundan_fark_rms_piksel": float(np.sqrt(np.mean((oteleme - tahmin) ** 2))),
        },
        "buyuk_olcekli_artik_adu": {
            "yalniz_duzey_cikarilmis": buyuk_olcekli_yapi(sabit, disarida),
            "duragan_maskesiz": buyuk_olcekli_yapi(ilk, disarida),
            "duragan_maskeli": buyuk_olcekli_yapi(duragan, disarida),
            "dinamik_maskeli": buyuk_olcekli_yapi(dinamik, disarida),
        },
        "piksel_gurultusu_adu": {
            "yalniz_duzey_cikarilmis": gurultu(sabit),
            "duragan_maskeli": gurultu(duragan),
            "dinamik_maskeli": gurultu(dinamik),
        },
        "yinelenebilirlik_bagil_sacilim": yinele,
        "yigin": {"boyut": list(yigin_g.shape), "saptanan_kaynak": int(icte.sum())},
        "astrometri": {
            "katalog_yildizi": int(len(katalog["ra"])),
            "kaydirma_oyu": oy,
            "baslik_wcs_kaymasi_piksel": (kaymasiz.crpix - baslik_wcs.crpix).tolist(),
            "cozume_giren_yildiz": cozulen.yildiz_sayisi,
            "alandaki_katalog_yildizi": int(alanda.sum()),
            "alandaki_katalog_yildizindan_saptanan": int(ters.eslesti.sum()),
            "saptanmayanlarin_k_kadiri": np.sort(katalog["k_m"][alanda][~ters.eslesti]).round(2).tolist(),
            "saptananlarin_k_kadiri": np.sort(katalog["k_m"][alanda][ters.eslesti]).round(2).tolist(),
            "artik_rms_yaysaniye": cozulen.artik_rms_yaysaniye,
            "olcek_yaysaniye_piksel": float(np.sqrt(abs(np.linalg.det(cozulen.wcs.cd))) * 3600.0),
            "baslik_olcegi_yaysaniye_piksel": float(b0["ESO INS PIXSCALE"]),
        },
        "fotometri": {
            "yildiz_sayisi": int(k_iyi.sum()),
            "sifir_noktasi_kadir": float(np.median(sifir)) if sifir.size else None,
            "sacilim_kadir": float(saglam.mad_olcek(sifir)) if sifir.size > 2 else None,
        },
    }
    cikti = KOK / "belgeler"
    (cikti / "kizilotesi_indirgeme.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(sonuc, ensure_ascii=False, indent=1))

    # Şekil: ham kare, gökyüzü çıkarılmış kare, yığının aynı bölgesi.
    def ger(goruntu: np.ndarray, sacilim: float, taban: float) -> np.ndarray:
        g = germe.asinh_ger(goruntu, siyah=taban - 2.0 * sacilim, alfa=1.0 / (10.0 * sacilim), q=6.0)
        return germe.yuzdelik_olcekle(g, 0.5, 99.8)

    # Yığının bütün karelerin örtüştüğü orta bölgesi (dört köşeli kaydırmada kenarlar yalnızca bazı karelerde vardır).
    ys, xs = np.nonzero(kapsama >= k_sayi - 1)
    orta = yigin_g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    paneller = [
        ("Ham kare (15 × 6 s)", ger(kareler[0], gurultu(sabit), float(duzey[0]))),
        ("Gökyüzü çıkarılmış kare", ger(np.where(bozuk, 0.0, dinamik[0]), gurultu(dinamik), 0.0)),
        (f"{k_sayi} karenin yığını (orta bölge)", ger(orta, saglam.mad_olcek(orta), 0.0)),
    ]
    fig, eksenler = dw.figur(
        "web", hedef="web-640", oran=0.40, ondalik=",", mozaik=[["a", "b", "c"]], gridspec_kw={"wspace": 0.04}
    )
    for ax, (baslik, goruntu) in zip(eksenler.values(), paneller, strict=True):
        # datawiz: izin göreli parlaklık görüntüsü; asinh gerilmiş değerlerin renk çubuğu okura nicelik vermez
        ax.imshow(goruntu, origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for kenar in ax.spines.values():
            kenar.set_visible(False)
        ax.set_title(baslik, loc="left")
    dw.kaydet(fig, str(cikti / "kizilotesi_indirgeme"), hedef="web-640", bicimler=("png",))


if __name__ == "__main__":
    main()
