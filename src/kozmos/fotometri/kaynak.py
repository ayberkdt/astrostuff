"""Arka plan kestirimi, kaynak saptama ve açıklık fotometrisi.

Arka plan kestiricisi SExtractor belgelerinin "Modeling the background"
bölümünden okunmuştur [SExtractorBelge2026]: ortanca çevresinde ±3σ yinelemeli
kırpma, kip = 2,5 × ortanca − 1,5 × ortalama, kip ile ortanca %30'dan fazla
ayrışırsa ortanca; harita için hücre ızgarası, ortanca süzgeci ve bikübik
spline. Yöntemin makalesi [BertinArnouts1996] bu çalışmada okunmamıştır.

Kaynak saptama (eşik üstü bağlı piksel grupları) ve açıklık fotometrisi
(alt piksel örneklemeli daire) photutils belgelerindeki tanımları izler
[Photutils2026]. İkisi de gerçeği bilinen sentetik görüntülerle sınanır.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from kozmos._kaynak import kaynak

Dizi = NDArray[np.float64]


@kaynak(
    "SExtractorBelge2026",
    yer="Modeling the background: Background estimation, Denk. 4",
    teyit="okundu",
    not_="ortanca çevresinde ±3σ kırpma yakınsayana kadar; Mode = 2.5 × Median − 1.5 × Mean; "
    "kip ile ortanca %30'dan fazla ayrışırsa ortanca",
)
def arka_plan_kestir(goruntu: ArrayLike, *, sigma: float = 3.0, en_cok_yineleme: int = 50) -> tuple[float, float]:
    """Bir görüntü parçasının arka plan düzeyini ve gürültüsünü kestirir.

    Döndürür
    --------
    (düzey, rms) : arka plan düzeyi ve kırpılmış dağılımın standart sapması.

    Kaynaklar
    ---------
    [SExtractorBelge2026] Background estimation.

    Geçerlilik
    ----------
    Kaynakların parçanın küçük bir bölümünü kapladığını varsayar. "%30 ayrışma"
    ölçütü burada |kip − ortanca| > 0,3 σ olarak uygulanmıştır; belgedeki
    ifadenin neye göre %30 olduğu açık değildir.

    Doğrulama
    ---------
    tests/birim/test_fotometri.py: yıldız eklenmiş sentetik alanda düzeyin yansız geri kazanımı.
    """
    x = np.asarray(goruntu, dtype=float).ravel()
    x = x[np.isfinite(x)]
    for _ in range(en_cok_yineleme):
        ortanca, olcek = np.median(x), x.std()
        kalan = x[np.abs(x - ortanca) <= sigma * olcek]
        if kalan.size == x.size:
            break
        x = kalan
    ortanca, ortalama, olcek = float(np.median(x)), float(x.mean()), float(x.std())
    kip = 2.5 * ortanca - 1.5 * ortalama
    return (ortanca if abs(kip - ortanca) > 0.3 * olcek else kip), olcek


@kaynak(
    "SExtractorBelge2026",
    yer="Modeling the background: Background map",
    teyit="okundu",
    not_="hücre ızgarasında yerel kestirim, ortanca süzgeci, hücreler arası bikübik spline",
)
def arka_plan_haritasi(goruntu: ArrayLike, hucre: int = 64, suzgec: int = 3) -> tuple[Dizi, Dizi]:
    """Yavaş değişen arka planın ve gürültüsünün haritasını çıkarır.

    Parametreler
    ------------
    goruntu : ndarray (Y, X)
    hucre : int
        Hücre kenarı [piksel]; kaynaklardan büyük, arka plan değişim ölçeğinden küçük seçilir.
    suzgec : int
        Hücre haritasına uygulanan ortanca süzgecinin genişliği (hücre cinsinden); 1 süzgeçsizdir.

    Döndürür
    --------
    (arka plan, rms) : görüntüyle aynı biçimde iki harita.

    Kaynaklar
    ---------
    [SExtractorBelge2026] Background map.

    Geçerlilik
    ----------
    Hücreden büyük yaygın kaynaklar (bulutsu, gökada) arka plan sayılır ve
    çıkarılır. Enterpolasyon ``scipy.ndimage.zoom`` ile kübik spline'dır; belgedeki
    "doğal bikübik spline" ile kenar davranışı farklı olabilir.

    Doğrulama
    ---------
    tests/birim/test_fotometri.py: eğimli arka planın %1 içinde geri kazanımı.
    """
    g = np.asarray(goruntu, dtype=float)
    ny, nx = g.shape
    hy, hx = max(1, ny // hucre), max(1, nx // hucre)
    duzey, rms = np.empty((hy, hx)), np.empty((hy, hx))
    for i in range(hy):
        for j in range(hx):
            parca = g[i * ny // hy : (i + 1) * ny // hy, j * nx // hx : (j + 1) * nx // hx]
            duzey[i, j], rms[i, j] = arka_plan_kestir(parca)
    if suzgec > 1:
        duzey = ndimage.median_filter(duzey, size=suzgec, mode="nearest")
        rms = ndimage.median_filter(rms, size=suzgec, mode="nearest")
    derece = 3 if min(hy, hx) >= 4 else 1
    buyut = lambda h: ndimage.zoom(h, (ny / hy, nx / hx), order=derece, mode="nearest", grid_mode=True)  # noqa: E731
    return buyut(duzey)[:ny, :nx], np.clip(buyut(rms)[:ny, :nx], 1e-300, None)


@dataclass(frozen=True, slots=True)
class KaynakListesi:
    """Saptanan kaynakların ölçümleri.

    Öznitelikler
    ------------
    x, y : ndarray
        Akı ağırlıklı merkezler [piksel]; ``goruntu[y, x]``.
    aki : ndarray
        Bölütteki arka plan çıkarılmış toplam akı.
    tepe : ndarray
        En parlak pikselin değeri.
    piksel_sayisi : ndarray
    """

    x: Dizi
    y: Dizi
    aki: Dizi
    tepe: Dizi
    piksel_sayisi: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.x.size)


def _isaretciler(alt: Dizi, maske: NDArray[np.bool_], duzeyler: Dizi, esik_aki: float) -> list:
    """Bir bölütün ayrı kaynak sayılacak tepelerinin (y, x) konumları (çok eşikli ağaç, eyer ölçütü)."""
    bekleyen = [(maske, 0)]
    tepeler = []
    while bekleyen:
        m, i0 = bekleyen.pop()
        bolundu = False
        for i in range(i0, duzeyler.size):
            parcalar, n = ndimage.label(m & (alt > duzeyler[i]), structure=np.ones((3, 3)))
            if n == 0:
                break
            if n == 1:
                continue
            kimlik = np.arange(1, n + 1)
            # Eyer düzeyinin üstünde kalan akı: dalın kendi tepesinin anlamlılığı.
            ust_aki = ndimage.sum_labels(alt - duzeyler[i], parcalar, kimlik)
            anlamli = kimlik[ust_aki >= esik_aki]
            if anlamli.size >= 2:
                bekleyen.extend((parcalar == k, i + 1) for k in anlamli)
                bolundu = True
                break
            if anlamli.size == 1:
                m = parcalar == anlamli[0]
            else:
                break
        if not bolundu:
            tepeler.append(np.unravel_index(np.argmax(np.where(m, alt, -np.inf)), alt.shape))
    return tepeler


def _taskin(alt: Dizi, maske: NDArray[np.bool_], tepeler: list) -> NDArray[np.int32]:
    """İşaretçilerden başlayan öncelik kuyruklu taşkın: her piksel, ona en yüksek düzeyden ulaşan havzaya girer.

    ``scipy.ndimage.watershed_ift`` komşu piksel farkını maliyet sayar (eğim görüntüsü bekler) ve parlaklık
    görüntüsünde eyeri aşar; bu yüzden kullanılmamıştır.
    """
    ny, nx = alt.shape
    havza = np.zeros(alt.shape, dtype=np.int32)
    kuyruk = []
    for j, (y, x) in enumerate(tepeler, start=1):
        havza[y, x] = j
        heapq.heappush(kuyruk, (-float(alt[y, x]), j, int(y), int(x)))
    while kuyruk:
        duzey, j, y, x = heapq.heappop(kuyruk)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < ny and 0 <= xx < nx and maske[yy, xx] and havza[yy, xx] == 0:
                    havza[yy, xx] = j
                    heapq.heappush(kuyruk, (max(duzey, -float(alt[yy, xx])), j, yy, xx))
    return havza


def _ayristir(saptama: Dizi, etiket: NDArray[np.int32], duzey_sayisi: int, karsitlik: float, en_az_piksel: int):
    """Birden çok anlamlı tepe taşıyan bölütleri su havzalarına böler; yeni etiket haritası döner."""
    yeni = etiket.copy()
    sonraki = int(etiket.max()) + 1
    for kimlik, dilim in enumerate(ndimage.find_objects(etiket), start=1):
        if dilim is None:
            continue
        maske = etiket[dilim] == kimlik
        alt = np.where(maske, saptama[dilim], 0.0)
        en_az, en_cok = float(alt[maske].min()), float(alt[maske].max())
        if maske.sum() < 2 * en_az_piksel or en_az <= 0 or en_cok <= en_az:
            continue
        duzeyler = np.geomspace(en_az, en_cok, duzey_sayisi + 2)[1:-1]
        tepeler = _isaretciler(alt, maske, duzeyler, karsitlik * float(alt[maske].sum()))
        tepeler.sort(key=lambda yx: -alt[yx])  # en parlak tepe eski kimliği korur
        while len(tepeler) >= 2:
            havza = _taskin(alt, maske, tepeler)
            boyut = np.bincount(havza[maske], minlength=len(tepeler) + 1)[1:]
            if boyut.min() >= en_az_piksel:
                break
            del tepeler[int(np.argmin(boyut))]  # en az piksel koşulunu sağlamayan havza komşusuna katılır
        if len(tepeler) < 2:
            continue
        for j in range(2, len(tepeler) + 1):
            yeni[dilim][maske & (havza == j)] = sonraki
            sonraki += 1
    return yeni


@kaynak(
    "Photutils2026",
    yer="Kullanım kılavuzu: Image Segmentation ve Source Deblending",
    teyit="okundu",
    not_="kaynak: her biri eşik üstünde olan en az belirli sayıda bağlı piksel; eşik arka plan gürültüsünün katı; "
    "eşiklemeden önce süzgeçleme; varsayılan bağlantı 8 komşuluk. Ayrıştırma: çok eşikleme + su havzası; "
    "'saddle' ölçütü, tepenin komşularından ayrıldığı eyer düzeyinin üstünde tuttuğu akının toplam akıya oranıdır; "
    "düzeyler kaynağın en küçük ve en büyük değeri arasındadır; ayrışmak için kaynaklar arasında eyer noktası gerekir. "
    "Düzeylerin geometrik aralıklanması ve havzaların öncelik kuyruklu taşkınla kurulması bu projenin seçimidir. "
    "Yöntemin makalesi [BertinArnouts1996] okunmadı",
)
def kaynak_bul(
    goruntu: ArrayLike,
    arka_plan: ArrayLike | float,
    rms: ArrayLike | float,
    *,
    esik_sigma: float = 3.0,
    en_az_piksel: int = 5,
    yumusatma_fwhm: float | None = None,
    ayristirma_duzeyi: int = 0,
    karsitlik: float = 0.005,
) -> KaynakListesi:
    """Arka plan üstündeki bağlı piksel gruplarını kaynak olarak saptar.

    Parametreler
    ------------
    goruntu : ndarray (Y, X)
    arka_plan, rms : ndarray ya da float
        ``arka_plan_haritasi`` çıktıları.
    esik_sigma : float
        Saptama eşiği (rms katı).
    en_az_piksel : int
        Bir kaynağın en az bağlı piksel sayısı; tek piksellik gürültü tepelerini ve kozmik ışınları eler.
    yumusatma_fwhm : float, isteğe bağlı
        Verilirse saptama görüntüsü bu genişlikte Gauss ile yumuşatılır (eşleşmiş süzgeç yaklaşımı);
        eşik yumuşatılmış gürültüye göre ölçeklenir.
    ayristirma_duzeyi : int
        Sıfırdan büyükse örtüşen kaynaklar bu sayıda eşik düzeyiyle ayrıştırılır (0: ayrıştırma yok).
    karsitlik : float
        Bir tepenin ayrı kaynak sayılması için eyer düzeyinin üstündeki akısının bölütün toplam akısına en küçük oranı.

    Döndürür
    --------
    KaynakListesi
        Akıya göre azalan sırada.

    Kaynaklar
    ---------
    [Photutils2026] Image Segmentation, Source Deblending.

    Geçerlilik
    ----------
    Ayrıştırma kapalıyken örtüşen kaynaklar tek kaynak sayılır. Açıkken aralarında
    eyer noktası olmayan (tek tepeli) çiftler yine ayrılmaz; eyer sığsa da ayrılmaz:
    2 FWHM aralıklı sentetik çiftlerin 32 düzey ve 0,005 karşıtlıkla akı oranı 2'ye
    kadar %92-98'i, oran 3'te %14'ü, oran 4'te %1'i ayrışır; 128 düzey ve 0,001
    karşıtlıkla oran 3'te %99'u, oran 4'te %59'u ayrışır (200 gerçekleme, rastgele yön); gürültülü saptama
    görüntüsünde küçük ``karsitlik`` gürültü tepelerini kaynak diye böler,
    yumuşatma bunu azaltır. Yaygın yapının (bulutsu) parlak bölgeleri de "kaynak" olarak çıkar.

    Doğrulama
    ---------
    tests/birim/test_fotometri.py: sentetik yıldız alanında tamlık, konum ve yanlış saptama sayısı; yakın çiftlerin ayrışması.
    """
    g = np.asarray(goruntu, dtype=float) - arka_plan
    saptama, esik = g, esik_sigma * np.asarray(rms, dtype=float)
    if yumusatma_fwhm:
        s = yumusatma_fwhm / 2.3548200450309493
        saptama = ndimage.gaussian_filter(g, s, mode="nearest")
        esik = esik / (2.0 * np.sqrt(np.pi) * s)  # beyaz gürültünün Gauss yumuşatmasından sonraki rms'i
    etiket, n = ndimage.label(saptama > esik, structure=np.ones((3, 3)))
    if n:
        sayi = np.bincount(etiket.ravel(), minlength=n + 1)
        etiket[sayi[etiket] < en_az_piksel] = 0
        if ayristirma_duzeyi > 0:
            etiket = _ayristir(saptama, etiket, ayristirma_duzeyi, karsitlik, en_az_piksel)
    gecerli = np.unique(etiket[etiket > 0])
    if gecerli.size == 0:
        bos = np.empty(0)
        return KaynakListesi(bos, bos, bos, bos, np.empty(0, dtype=np.int64))
    sayi = np.asarray(ndimage.sum_labels(np.ones_like(g), etiket, gecerli)).astype(np.int64)
    pozitif = np.clip(g, 0.0, None)
    merkez = np.array(ndimage.center_of_mass(pozitif, etiket, gecerli)).reshape(-1, 2)
    aki = np.asarray(ndimage.sum_labels(g, etiket, gecerli))
    tepe = np.asarray(ndimage.maximum(g, etiket, gecerli))
    sira = np.argsort(-aki)
    return KaynakListesi(merkez[sira, 1], merkez[sira, 0], aki[sira], tepe[sira], sayi[sira])


@kaynak(
    "Photutils2026",
    yer="Kullanım kılavuzu: Aperture Photometry, 'subpixel' yöntemi",
    teyit="okundu",
    not_="piksel N×N alt piksele bölünür; ağırlık, merkezi açıklığın içinde kalan alt piksellerin kesridir. "
    "Halkadan ortanca ile yerel gökyüzü çıkarma bu projenin seçimidir",
)
def aciklik_fotometrisi(
    goruntu: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    yaricap: float,
    *,
    halka: tuple[float, float] | None = None,
    alt_piksel: int = 5,
) -> tuple[Dizi, Dizi]:
    """Dairesel açıklıkta akı ölçer; isteğe bağlı halkadan yerel gökyüzünü çıkarır.

    Parametreler
    ------------
    goruntu : ndarray (Y, X)
    x, y : ndarray (N,)
        Kaynak merkezleri [piksel].
    yaricap : float
        Açıklık yarıçapı [piksel].
    halka : (iç, dış), isteğe bağlı
        Gökyüzü halkasının yarıçapları; verilirse halkadaki ortanca piksel başına gökyüzü sayılır.
    alt_piksel : int
        Kenar piksellerinin kapsanan kesri için piksel başına örnekleme sayısı (her eksende).

    Döndürür
    --------
    (akı, alan) : gökyüzü çıkarılmış akı ve açıklığın etkin alanı [piksel²].

    Kaynaklar
    ---------
    [Photutils2026] Aperture Photometry.

    Geçerlilik
    ----------
    Alt piksel örneklemesi daire alanını yaklaşık 1/alt_piksel² bağıl hatayla verir.
    Kenar pikselinin kapsanan kesri alanla ölçülür ve piksel içinde akı düzgün sayılır; açıklık
    PSF'in dik olduğu bölgeyi kesiyorsa (yarıçap FWHM'den küçükse) akı yüzde birkaç eksik ölçülür.
    Görüntü kenarına taşan açıklıklarda yalnızca görüntü içindeki pikseller toplanır.

    Doğrulama
    ---------
    tests/birim/test_fotometri.py: düz görüntüde alan πr²; Gauss yıldızda çevrelenen akı kesri 1 − exp(−r²/2σ²).
    """
    g = np.asarray(goruntu, dtype=float)
    ny, nx = g.shape
    kaydirma = (np.arange(alt_piksel) + 0.5) / alt_piksel - 0.5
    akilar, alanlar = [], []
    for x0, y0 in zip(
        np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float)), strict=True
    ):
        dis = halka[1] if halka else yaricap
        x1, x2 = max(0, int(np.floor(x0 - dis - 1))), min(nx, int(np.ceil(x0 + dis + 2)))
        y1, y2 = max(0, int(np.floor(y0 - dis - 1))), min(ny, int(np.ceil(y0 + dis + 2)))
        parca = g[y1:y2, x1:x2]
        yy, xx = np.mgrid[y1:y2, x1:x2]
        ax = xx[..., None, None] + kaydirma[None, None, None, :] - x0
        ay = yy[..., None, None] + kaydirma[None, None, :, None] - y0
        kesir = (ax**2 + ay**2 <= yaricap**2).mean(axis=(-1, -2))
        gokyuzu = 0.0
        if halka:
            r2 = (xx - x0) ** 2 + (yy - y0) ** 2
            secim = (r2 >= halka[0] ** 2) & (r2 <= halka[1] ** 2)
            gokyuzu = float(np.median(parca[secim])) if secim.any() else 0.0
        alan = float(kesir.sum())
        akilar.append(float((parca * kesir).sum()) - gokyuzu * alan)
        alanlar.append(alan)
    return np.array(akilar), np.array(alanlar)
