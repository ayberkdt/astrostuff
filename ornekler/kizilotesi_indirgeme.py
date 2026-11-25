"""Gerçek ham yakın kızılötesi karelerin uçtan uca indirgenmesi.

Veri: ESO NTT/SOFI, Ks bandı, program 184.D-1152(F), Mart 2012. İki bilim
dizisi ve aynı gecenin kubbe düz alanları:

- SN 2011bm alanı: 16 kare (15 × 6 s), yüksek galaktik enlem, seyrek alan.
- SN 2005at alanı (NGC 6744): 32 kare (10 × 7 s), yıldızca zengin; gökada alanı doldurur.
- Ks kubbe düz alanı: 8 kare (lamba açık/kapalı, maskeli/maskesiz).

Kareler ESO arşivinden ``.fits.Z`` olarak iner; açılmış ``.fits`` dosyaları
``veri/ham/eso/`` altında beklenir (``gzip -d`` ``.Z`` dosyalarını açar; düz
alan dosyaları ``duzalan_`` önekiyle). Dosya kimlikleri
``ornekler/eso_sofi_kareleri.txt`` içindedir; her biri
``https://dataportal.eso.org/dataportal_new/file/<kimlik>`` adresinden iner.

Adımlar:

1. Düz alan: maskesiz lamba açık karelerin ortalaması eksi maskesiz lamba kapalı karelerin ortalaması.
2. Maskesiz durağan gökyüzü modeli, bozuk piksel haritası, kare başına saptama.
3. Kareler arası öteleme: başlıktaki kaydırma komutlarından başlar, yıldızlarla inceltilir.
4. İlk yığından nesne maskesi; maskeli durağan ve dinamik gökyüzü modelleri; düz alana bölme.
5. Yığma, kaynak saptama, 2MASS kataloğuyla astrometrik çözüm (başlıktan ve başlıksız) ve sıfır noktası.

Bu sınamada "gerçek" bilinmez; ölçütler iç tutarlılıktır (artık arka planın
düzlüğü, aynı yıldızın karelerdeki akısının yinelenebilirliği) ve bağımsız
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

from kozmos.arsiv import gaia, ikimass
from kozmos.astrometri import cozum, eslestirme, ucgen, wcs
from kozmos.fotometri import kaynak as fk
from kozmos.gorsel import germe
from kozmos.indirgeme import gokyuzu
from kozmos.io import fits
from kozmos.istatistik import saglam

KOK = Path(__file__).resolve().parents[1]
DIZIN = KOK / "veri" / "ham" / "eso"
ACIKLIK, HALKA = 5.0, (10.0, 15.0)
DIZILER = {"SN2011bm": "SOFI.2012-03-15T07*.fits", "SN2005at": "SOFI.2012-03-14T09*.fits"}


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
        with np.errstate(all="ignore"):
            sonuc.append(np.nanstd(np.nanmedian(bloklar, axis=2)))
    return float(np.median(sonuc))


def duz_alan_kur() -> tuple[np.ndarray, dict]:
    """Kubbe düz alanı: (lamba açık − lamba kapalı), ortancaya bölünmüş.

    Dizi lamba kapalı, kapalı+maske, açık+maske, açık, açık, açık+maske, kapalı+maske, kapalı sırasındadır;
    bu sıra başlıkta yazmaz, karelerin düzeylerinden okunmuştur. Maske ışığın bir bölümünü kestiği için her
    lamba durumunda en yüksek düzeyli iki kare maskesizdir. Maskeli kareler (aydınlatmaya bağlı sıfır düzeyi
    düzeltmesi içindir) burada kullanılmaz.
    """
    yollar = sorted(DIZIN.glob("duzalan_SOFI*.fits"))
    if len(yollar) != 8:
        raise SystemExit(f"{DIZIN} altında 8 düz alan karesi bekleniyor, {len(yollar)} bulundu")
    kareler = np.array([fits.fits_oku(yol)[0] for yol in yollar])
    duzey = np.median(kareler, axis=(1, 2))
    sira = np.argsort(duzey)
    kapali, acik = sira[2:4], sira[6:8]  # her dörtlünün en yüksek ikisi
    fark = kareler[acik].mean(axis=0) - kareler[kapali].mean(axis=0)
    duz = fark / np.median(fark)
    return duz, {
        "kare_duzeyleri_adu": [round(float(d), 1) for d in duzey],
        "lamba_acik_eksi_kapali_adu": float(np.median(fark)),
        "duz_alan_sacilimi": float(saglam.mad_olcek(duz)),
    }


def indirge(ad: str, desen: str, duz: np.ndarray) -> tuple[dict, dict]:
    yollar = sorted(DIZIN.glob(desen))
    if len(yollar) < 8:
        raise SystemExit(f"{DIZIN} altında {ad} kareleri yok; kimlikler ornekler/eso_sofi_kareleri.txt içinde")
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

    # Maskesiz durağan model. Bozuk pikseller: gökyüzü modelinde yerel ortancadan sapanlar ve düz alanı aykırı olanlar.
    ilk, ilk_model = gokyuzu.duragan_gokyuzu_cikar(kareler)
    yerel = ilk_model - ndimage.median_filter(ilk_model, size=5, mode="nearest")
    bozuk = np.abs(yerel) > 6.0 * saglam.mad_olcek(yerel)
    bozuk |= (duz < 0.5) | (duz > 1.5)
    bozuk = ndimage.binary_dilation(bozuk, structure=np.ones((3, 3)))
    ilk_temiz = ilk.copy()
    ilk_temiz[:, bozuk] = 0.0
    listeler = [sapta(kare, esik=6.0) for kare in ilk_temiz]

    # Öteleme: başlıktaki kaydırma komutları başlangıç tahminidir; komutun dedektör eksenleriyle eşleşmesi ve
    # işareti (sekiz olasılık) en çok yıldız eşleştiren seçenek olarak veriden bulunur, sonra yıldızlarla inceltilir.
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

    # İlk yığından nesne maskesi (yığın koordinatında), her karenin koordinatına geri taşınır.
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

    duragan, _ = gokyuzu.duragan_gokyuzu_cikar(kareler, maskeler=maskeler)
    dinamik = gokyuzu.dinamik_gokyuzu_cikar(kareler, zaman, komsu_sayisi=8, maskeler=maskeler)
    sabit = kareler - duzey[:, None, None]  # yalnızca düzey çıkarılmış: karşılaştırma tabanı
    disarida = maskeler | bozuk[None]
    guvenli_duz = np.where(bozuk, 1.0, duz)
    duragan_duz = duragan / guvenli_duz  # gökyüzü çıkarılmış kare düz alana bölünür

    def gurultu(dizi: np.ndarray) -> float:
        return float(np.median([saglam.mad_olcek(k[~m]) for k, m in zip(dizi, disarida, strict=True)]))

    # Yığın (düz alanlı durağan model; bozuk pikseller girmez), saptama.
    son_yigin, kapsama, koken = yigin(np.where(bozuk[None], np.nan, duragan_duz), oteleme)
    tam_kapsama = kapsama >= 0.5 * k_sayi
    yigin_g = np.where(tam_kapsama, son_yigin, np.nanmedian(son_yigin[tam_kapsama]))
    kaynaklar = sapta(yigin_g, esik=5.0)
    ic_bolge = ndimage.binary_erosion(tam_kapsama, iterations=15)
    icte = ic_bolge[np.round(kaynaklar.y).astype(int), np.round(kaynaklar.x).astype(int)]
    kx, ky = kaynaklar.x[icte], kaynaklar.y[icte]
    # Kör çözümde parlaklık ölçütü tepe değeridir: bölüt akısı yaygın gökada düğümlerini yıldızların önüne geçirir.
    k_aki = kaynaklar.tepe[icte]

    # Astrometri: (a) başlıktaki WCS'ten başlayarak, (b) başlık kullanılmadan üçgen eşlemeyle.
    baslik_wcs = wcs.basliktan_wcs(b0)
    baslik_wcs = wcs.Wcs(crpix=baslik_wcs.crpix + koken, crval=baslik_wcs.crval, cd=baslik_wcs.cd)
    katalog = ikimass.nokta_kaynaklari(float(b0["CRVAL1"]), float(b0["CRVAL2"]), 0.09, KOK / "veri" / "ham" / "2mass")
    kaymasiz, oy = cozum.kaydirma_bul(
        baslik_wcs, kx, ky, katalog["ra"], katalog["dec"], en_cok_piksel=320.0, kutu_piksel=4.0
    )
    cozulen = cozum.wcs_iyilestir(
        kaymasiz, kx, ky, katalog["ra"], katalog["dec"], yaricap_yaysaniye=4.0, son_yaricap_yaysaniye=1.5
    )
    px, py = wcs.gok_den_piksel(cozulen.wcs, katalog["ra"], katalog["dec"])
    alanda = np.isfinite(px) & (px > 0) & (px < yigin_g.shape[1] - 1) & (py > 0) & (py < yigin_g.shape[0] - 1)
    alanda[alanda] = ic_bolge[np.round(py[alanda]).astype(int), np.round(px[alanda]).astype(int)]
    ra, dec = wcs.piksel_den_gok(cozulen.wcs, kx, ky)
    ters = eslestirme.capraz_esle(katalog["ra"][alanda], katalog["dec"][alanda], ra, dec, 1.5)
    try:
        # Kör çözüm: başlıktaki WCS kullanılmaz. Yalnızca yaklaşık gök konumu ve alan genişliği bilgisi kullanılır:
        # katalog, yığının kısa kenarının yarısı kadar yarıçaplı bir koniyle sınırlanır.
        yaricap = 0.5 * min(yigin_g.shape) * float(b0["ESO INS PIXSCALE"]) / 3600.0
        uzaklik = np.hypot(
            (katalog["ra"] - float(b0["CRVAL1"])) * np.cos(np.radians(float(b0["CRVAL2"]))),
            katalog["dec"] - float(b0["CRVAL2"]),
        )
        yakin = uzaklik < yaricap
        kor, cift = ucgen.kor_cozum(
            kx,
            ky,
            k_aki,
            katalog["ra"][yakin],
            katalog["dec"][yakin],
            katalog["k_m"][yakin],
            en_parlak=40,
            tolerans=0.015,
        )
        izgara_x, izgara_y = np.meshgrid(
            np.linspace(0, yigin_g.shape[1] - 1, 4), np.linspace(0, yigin_g.shape[0] - 1, 4)
        )
        r1, d1 = wcs.piksel_den_gok(kor.wcs, izgara_x, izgara_y)
        r2, d2 = wcs.piksel_den_gok(cozulen.wcs, izgara_x, izgara_y)
        kor_sonuc = {
            "ucgen_eslemesinde_atanan_cift": cift,
            "cozume_giren_yildiz": kor.yildiz_sayisi,
            "basliktan_baslayan_cozumden_en_buyuk_fark_yaysaniye": float(
                np.max(np.hypot((r1 - r2) * np.cos(np.radians(d2)), d1 - d2)) * 3600.0
            ),
        }
    except ValueError as hata:
        # Tanı: başlıktan başlayan çözümle, üçgen eşlemeye giren iki parlak listenin kaç ortak yıldızı vardı?
        g = np.argsort(-k_aki)[:40]
        kk = np.argsort(katalog["k_m"][yakin])[:40]
        ortak = eslestirme.capraz_esle(ra[g], dec[g], katalog["ra"][yakin][kk], katalog["dec"][yakin][kk], 1.5)
        kor_sonuc = {
            "basarisiz": str(hata),
            "konideki_katalog_yildizi": int(yakin.sum()),
            "parlak_listelerin_ortak_yildizi": int(ortak.eslesti.sum()),
        }

    # Gaia DR3 ile astrometri: 2MASS'ten çok daha derin olduğu için seyrek alanda da yeterli yıldız verir.
    # Konumlar öz hareketle gözlem dönemine taşınır. Kör çözümde parlaklık sırası G kadirinden alınır.
    donem = 2000.0 + (float(b0["MJD-OBS"]) - 51544.5) / 365.25
    g_kat = gaia.koni_sorgusu(
        float(b0["CRVAL1"]), float(b0["CRVAL2"]), 0.09, KOK / "veri" / "ham" / "gaia", donem_yil=donem
    )
    g_kaymasiz, g_oy = cozum.kaydirma_bul(
        baslik_wcs, kx, ky, g_kat["ra"], g_kat["dec"], en_cok_piksel=320.0, kutu_piksel=4.0
    )
    g_cozum = cozum.wcs_iyilestir(
        g_kaymasiz, kx, ky, g_kat["ra"], g_kat["dec"], yaricap_yaysaniye=4.0, son_yaricap_yaysaniye=1.0
    )
    kose_x, kose_y = np.meshgrid(np.linspace(0, yigin_g.shape[1] - 1, 4), np.linspace(0, yigin_g.shape[0] - 1, 4))

    def cozum_farki(w1: wcs.Wcs, w2: wcs.Wcs) -> float:
        a1, d1 = wcs.piksel_den_gok(w1, kose_x, kose_y)
        a2, d2 = wcs.piksel_den_gok(w2, kose_x, kose_y)
        return float(np.max(np.hypot((a1 - a2) * np.cos(np.radians(d2)), d1 - d2)) * 3600.0)

    gaia_sonuc = {
        "donem_yil": donem,
        "konideki_gaia_kaynagi": int(len(g_kat["ra"])),
        "kaydirma_oyu": g_oy,
        "cozume_giren_yildiz": g_cozum.yildiz_sayisi,
        "artik_rms_yaysaniye": g_cozum.artik_rms_yaysaniye,
        "2mass_cozumunden_en_buyuk_fark_yaysaniye": cozum_farki(g_cozum.wcs, cozulen.wcs),
    }
    try:
        g_yakin = (
            np.hypot(
                (g_kat["ra"] - float(b0["CRVAL1"])) * np.cos(np.radians(float(b0["CRVAL2"]))),
                g_kat["dec"] - float(b0["CRVAL2"]),
            )
            < 0.5 * min(yigin_g.shape) * float(b0["ESO INS PIXSCALE"]) / 3600.0
        )
        g_kor, g_cift = ucgen.kor_cozum(
            kx,
            ky,
            k_aki,
            g_kat["ra"][g_yakin],
            g_kat["dec"][g_yakin],
            g_kat["phot_g_mean_mag"][g_yakin],
            en_parlak=40,
            tolerans=0.015,
            son_yaricap_yaysaniye=1.0,
            en_az_dogrulama_kesri=0.2,  # seyrek alanda parlak listelerin ortak yıldızı azdır; mutlak alt sınır (8) korunur
        )
        gaia_sonuc["kor_cozum"] = {
            "ucgen_eslemesinde_atanan_cift": g_cift,
            "cozume_giren_yildiz": g_kor.yildiz_sayisi,
            "artik_rms_yaysaniye": g_kor.artik_rms_yaysaniye,
            "basliktan_baslayan_gaia_cozumunden_en_buyuk_fark_yaysaniye": cozum_farki(g_kor.wcs, g_cozum.wcs),
        }
    except ValueError as hata:
        gaia_sonuc["kor_cozum"] = {"basarisiz": str(hata)}

    # Sıfır noktası: 2MASS K belirsizliği 0,15 kadirden küçük, doymamış, eşleşen yıldızlar.
    e = eslestirme.capraz_esle(ra, dec, katalog["ra"], katalog["dec"], 1.5)
    aki, _ = fk.aciklik_fotometrisi(yigin_g, kx, ky, ACIKLIK, halka=HALKA)
    tepe = kaynaklar.tepe[icte]
    k_iyi = e.eslesti & (katalog["k_cmsig"][e.indis] < 0.15) & (aki > 0) & (tepe < 15000.0)
    sifir = katalog["k_m"][e.indis][k_iyi] + 2.5 * np.log10(aki[k_iyi])

    def yinelenebilirlik(dizi: np.ndarray) -> dict:
        """Parlak yıldızların karelerdeki açıklık akısının bağıl sağlam saçılımı ve beklenen gürültüye oranı."""
        olcumler, beklenen = [], []
        piksel_sigma = gurultu(dizi)
        n_pix, n_b = np.pi * ACIKLIK**2, np.pi * (HALKA[1] ** 2 - HALKA[0] ** 2)
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
                        fk.aciklik_fotometrisi(np.where(bozuk, 0.0, dizi[k]), x_k[k], y_k[k], ACIKLIK, halka=HALKA)[0]
                    )
                    for k in range(k_sayi)
                ]
            )
            if np.median(f) > 30.0 * piksel_sigma * np.sqrt(n_pix):  # S/G > 30: gürültü değil düzeltme hatası ölçülsün
                olcumler.append(saglam.mad_olcek(f) / np.median(f))
                beklenen.append(piksel_sigma * np.sqrt(n_pix * (1.0 + n_pix / n_b)) / np.median(f))
            if len(olcumler) >= 15:
                break
        if not olcumler:
            return {"yildiz_sayisi": 0}
        return {
            "yildiz_sayisi": len(olcumler),
            "bagil_sacilim": float(np.median(olcumler)),
            "beklenen_arka_plan_gurultusu": float(np.median(beklenen)),
        }

    sonuc = {
        "veri": {
            "hedef": str(b0["OBJECT"]).strip(),
            "suzgec": str(b0["ESO INS FILT1 NAME"]).strip(),
            "kare_sayisi": k_sayi,
            "dit_s": float(b0["ESO DET DIT"]),
            "ndit": int(b0["ESO DET NDIT"]),
            "dizi_suresi_dakika": float(zaman[-1]),
            "hava_kutlesi": float(b0["ESO TEL AIRM START"]),
        },
        "gokyuzu_duzeyi_adu": [float(duzey.min()), float(duzey.max())],
        "bozuk_piksel_kesri": float(bozuk.mean()),
        "nesne_maskesi_kesri": float(maskeler.mean()),
        "oteleme": {
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
        "yinelenebilirlik": {
            "duz_alansiz": yinelenebilirlik(duragan),
            "duz_alanli": yinelenebilirlik(duragan_duz),
        },
        "yigin": {"boyut": list(yigin_g.shape), "saptanan_kaynak": int(icte.sum())},
        "astrometri": {
            "katalog_yildizi": int(len(katalog["ra"])),
            "kaydirma_oyu": oy,
            "baslik_wcs_kaymasi_piksel": (kaymasiz.crpix - baslik_wcs.crpix).tolist(),
            "cozume_giren_yildiz": cozulen.yildiz_sayisi,
            "alandaki_katalog_yildizi": int(alanda.sum()),
            "alandaki_katalog_yildizindan_saptanan": int(ters.eslesti.sum()),
            "artik_rms_yaysaniye": cozulen.artik_rms_yaysaniye,
            "olcek_yaysaniye_piksel": float(np.sqrt(abs(np.linalg.det(cozulen.wcs.cd))) * 3600.0),
            "baslik_olcegi_yaysaniye_piksel": float(b0["ESO INS PIXSCALE"]),
            "kor_cozum": kor_sonuc,
            "gaia": gaia_sonuc,
        },
        "fotometri": {
            "yildiz_sayisi": int(k_iyi.sum()),
            "sifir_noktasi_kadir": float(np.median(sifir)) if sifir.size else None,
            "sacilim_kadir": float(saglam.mad_olcek(sifir)) if sifir.size > 2 else None,
            "tek_tek_sifir_noktalari": np.sort(sifir).round(3).tolist() if sifir.size <= 6 else None,
            "ortanca_katalog_belirsizligi_kadir": float(np.median(katalog["k_cmsig"][e.indis][k_iyi]))
            if sifir.size
            else None,
        },
    }
    ys, xs = np.nonzero(kapsama >= k_sayi - 1)
    goruntuler = {
        "ham": (kareler[0], gurultu(sabit), float(duzey[0])),
        "cikarilmis": (np.where(bozuk, 0.0, duragan_duz[0]), gurultu(duragan_duz), 0.0),
        "yigin": (yigin_g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], None, 0.0),
    }
    return sonuc, goruntuler


def main() -> None:
    duz, duz_bilgi = duz_alan_kur()
    sonuc = {
        "veri": {"alet": "ESO NTT / SOFI", "program": "184.D-1152(F)", "kaynak": "ESO Bilim Arşivi"},
        "duz_alan": duz_bilgi,
        "diziler": {},
    }
    goruntuler = {}
    for ad, desen in DIZILER.items():
        sonuc["diziler"][ad], goruntuler[ad] = indirge(ad, desen, duz)
    # Sıfır noktası aktarımı: seyrek alanda yeterli 2MASS yıldızı yoktur. Zengin alanın sıfır noktası, poz süresi
    # farkı için düzeltilip seyrek alana taşınır ve oradaki az sayıdaki yıldızla karşılaştırılır. Varsayım: piksel
    # değerleri tek DIT başınadır (NDIT ortalaması); SOFI belgelerinden doğrulanmadı. Hava kütlesi düzeltmesi yoktur.
    zengin, seyrek = sonuc["diziler"]["SN2005at"], sonuc["diziler"]["SN2011bm"]
    aktarilan = zengin["fotometri"]["sifir_noktasi_kadir"] - 2.5 * np.log10(
        zengin["veri"]["dit_s"] / seyrek["veri"]["dit_s"]
    )
    sonuc["sifir_noktasi_aktarimi"] = {
        "zengin_alan_sifir_noktasi": zengin["fotometri"]["sifir_noktasi_kadir"],
        "dit_duzeltmesi_kadir": float(-2.5 * np.log10(zengin["veri"]["dit_s"] / seyrek["veri"]["dit_s"])),
        "seyrek_alana_aktarilan": float(aktarilan),
        "seyrek_alandaki_yildizlarin_sifir_noktalari": seyrek["fotometri"]["tek_tek_sifir_noktalari"],
        "hava_kutleleri": [zengin["veri"]["hava_kutlesi"], seyrek["veri"]["hava_kutlesi"]],
    }
    cikti = KOK / "belgeler"
    (cikti / "kizilotesi_indirgeme.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(sonuc, ensure_ascii=False, indent=1))

    # Şekil: ham kare, gökyüzü çıkarılmış kare, yığının bütün karelerin örtüştüğü orta bölgesi (SN 2011bm dizisi).
    def ger(goruntu: np.ndarray, sacilim: float | None, taban: float) -> np.ndarray:
        sacilim = saglam.mad_olcek(goruntu) if sacilim is None else sacilim
        g = germe.asinh_ger(goruntu, siyah=taban - 2.0 * sacilim, alfa=1.0 / (10.0 * sacilim), q=6.0)
        return germe.yuzdelik_olcekle(g, 0.5, 99.8)

    g = goruntuler["SN2011bm"]
    paneller = [
        ("Ham kare (15 × 6 s)", ger(*g["ham"])),
        ("Gökyüzü çıkarılmış kare", ger(*g["cikarilmis"])),
        ("16 karenin yığını (orta bölge)", ger(*g["yigin"])),
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
