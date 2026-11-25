"""Arka plan kestirimi, kaynak saptama ve açıklık fotometrisi.

Arka plan kestiricisi SExtractor figuresinin "Modeling the background"
bölümünden okunmuştur [SExtractorBelge2026]: ortanca çevresinde ±3σ yinelemeli
kırpma, kip = 2,5 × ortanca − 1,5 × ortalama, kip ile ortanca %30'dan fazla
ayrışırsa ortanca; harita için hücre ızgarası, ortanca süzgeci ve bikübik
spline. Yöntemin makalesi [BertinArnouts1996] bu çalışmada okunmamıştır.

Kaynak saptama (eşik üstü bağlı piksel grupları) ve açıklık fotometrisi
(alt piksel örneklemeli daire) photutils figuresindeki tanımları izler
[Photutils2026]. İkisi de gerçeği bilinen sentetik görüntülerle sınanır.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from kozmos._sources import source

Array = NDArray[np.float64]


@source(
    "SExtractorBelge2026",
    where="Modeling the background: Background estimation, Denk. 4",
    status="read",
    note="ortanca çevresinde ±3σ kırpma yakınsayana kadar; Mode = 2.5 × Median − 1.5 × Mean; "
    "kip ile ortanca %30'dan fazla ayrışırsa ortanca",
)
def estimate_background(image: ArrayLike, *, sigma: float = 3.0, max_iterations: int = 50) -> tuple[float, float]:
    """Bir görüntü parçasının arka plan düzeyini ve gürültüsünü kestirir.

    Returns
    -------
    (düzey, rms) : arka plan düzeyi ve kırpılmış dağılımın standart sapması.

    References
    ----------
    [SExtractorBelge2026] Background estimation.

    Validity
    --------
    Kaynakların parçanın küçük bir bölümünü kapladığını varsayar. "%30 ayrışma"
    ölçütü burada |kip − ortanca| > 0,3 σ olarak uygulanmıştır; belgedeki
    ifadenin neye göre %30 olduğu açık değildir.

    Verification
    ------------
    tests/unit/test_photometry.py: yıldız eklenmiş sentetik alanda düzeyin yansız geri kazanımı.
    """
    x = np.asarray(image, dtype=float).ravel()
    x = x[np.isfinite(x)]
    for _ in range(max_iterations):
        ortanca, olcek = np.median(x), x.std()
        kalan = x[np.abs(x - ortanca) <= sigma * olcek]
        if kalan.size == x.size:
            break
        x = kalan
    ortanca, ortalama, olcek = float(np.median(x)), float(x.mean()), float(x.std())
    kip = 2.5 * ortanca - 1.5 * ortalama
    return (ortanca if abs(kip - ortanca) > 0.3 * olcek else kip), olcek


@source(
    "SExtractorBelge2026",
    where="Modeling the background: Background map",
    status="read",
    note="hücre ızgarasında yerel kestirim, ortanca süzgeci, hücreler arası bikübik spline",
)
def background_map(image: ArrayLike, cell: int = 64, filter_size: int = 3) -> tuple[Array, Array]:
    """Yavaş değişen arka planın ve gürültüsünün haritasını çıkarır.

    Parameters
    ----------
    goruntu : ndarray (Y, X)
    hucre : int
        Hücre kenarı [piksel]; kaynaklardan büyük, arka plan değişim ölçeğinden küçük seçilir.
    suzgec : int
        Hücre haritasına uygulanan ortanca süzgecinin genişliği (hücre cinsinden); 1 süzgeçsizdir.

    Returns
    -------
    (arka plan, rms) : görüntüyle aynı biçimde iki harita.

    References
    ----------
    [SExtractorBelge2026] Background map.

    Validity
    --------
    Hücreden büyük yaygın kaynaklar (bulutsu, gökada) arka plan sayılır ve
    çıkarılır. Enterpolasyon ``scipy.ndimage.zoom`` ile kübik spline'dır; belgedeki
    "doğal bikübik spline" ile kenar davranışı farklı olabilir.

    Verification
    ------------
    tests/unit/test_photometry.py: eğimli arka planın %1 içinde geri kazanımı.
    """
    g = np.asarray(image, dtype=float)
    ny, nx = g.shape
    hy, hx = max(1, ny // cell), max(1, nx // cell)
    duzey, rms = np.empty((hy, hx)), np.empty((hy, hx))
    for i in range(hy):
        for j in range(hx):
            parca = g[i * ny // hy : (i + 1) * ny // hy, j * nx // hx : (j + 1) * nx // hx]
            duzey[i, j], rms[i, j] = estimate_background(parca)
    if filter_size > 1:
        duzey = ndimage.median_filter(duzey, size=filter_size, mode="nearest")
        rms = ndimage.median_filter(rms, size=filter_size, mode="nearest")
    derece = 3 if min(hy, hx) >= 4 else 1
    buyut = lambda h: ndimage.zoom(h, (ny / hy, nx / hx), order=derece, mode="nearest", grid_mode=True)  # noqa: E731
    return buyut(duzey)[:ny, :nx], np.clip(buyut(rms)[:ny, :nx], 1e-300, None)


@dataclass(frozen=True, slots=True)
class SourceList:
    """Saptanan kaynakların ölçümleri.

    Attributes
    ----------
    x, y : ndarray
        Akı ağırlıklı merkezler [piksel]; ``goruntu[y, x]``.
    aki : ndarray
        Bölütteki arka plan çıkarılmış toplam akı.
    tepe : ndarray
        En parlak pikselin değeri.
    piksel_sayisi : ndarray
    """

    x: Array
    y: Array
    flux: Array
    peak: Array
    n_pixels: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.x.size)


def _markers(low: Array, mask: NDArray[np.bool_], levels: Array, flux_threshold: float) -> list:
    """Bir bölütün ayrı kaynak sayılacak tepelerinin (y, x) konumları (çok eşikli ağaç, eyer ölçütü)."""
    bekleyen = [(mask, 0)]
    peaks = []
    while bekleyen:
        m, i0 = bekleyen.pop()
        bolundu = False
        for i in range(i0, levels.size):
            parts, n = ndimage.label(m & (low > levels[i]), structure=np.ones((3, 3)))
            if n == 0:
                break
            if n == 1:
                continue
            kimlik = np.arange(1, n + 1)
            # Eyer düzeyinin üstünde kalan akı: dalın kendi tepesinin anlamlılığı.
            ust_aki = ndimage.sum_labels(low - levels[i], parts, kimlik)
            anlamli = kimlik[ust_aki >= flux_threshold]
            if anlamli.size >= 2:
                bekleyen.extend((parts == k, i + 1) for k in anlamli)
                bolundu = True
                break
            if anlamli.size == 1:
                m = parts == anlamli[0]
            else:
                break
        if not bolundu:
            peaks.append(np.unravel_index(np.argmax(np.where(m, low, -np.inf)), low.shape))
    return peaks


def _flood(low: Array, mask: NDArray[np.bool_], peaks: list) -> NDArray[np.int32]:
    """İşaretçilerden başlayan öncelik kuyruklu taşkın: her piksel, ona en yüksek düzeyden ulaşan havzaya girer.

    ``scipy.ndimage.watershed_ift`` komşu piksel farkını maliyet sayar (eğim görüntüsü bekler) ve parlaklık
    görüntüsünde eyeri aşar; bu yüzden kullanılmamıştır.
    """
    ny, nx = low.shape
    havza = np.zeros(low.shape, dtype=np.int32)
    kuyruk = []
    for j, (y, x) in enumerate(peaks, start=1):
        havza[y, x] = j
        heapq.heappush(kuyruk, (-float(low[y, x]), j, int(y), int(x)))
    while kuyruk:
        duzey, j, y, x = heapq.heappop(kuyruk)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < ny and 0 <= xx < nx and mask[yy, xx] and havza[yy, xx] == 0:
                    havza[yy, xx] = j
                    heapq.heappush(kuyruk, (max(duzey, -float(low[yy, xx])), j, yy, xx))
    return havza


def _deblend(detection_image: Array, labels: NDArray[np.int32], n_levels: int, contrast: float, min_pixels: int):
    """Birden çok anlamlı tepe taşıyan bölütleri su havzalarına böler; yeni etiket haritası döner."""
    yeni = labels.copy()
    sonraki = int(labels.max()) + 1
    for kimlik, dilim in enumerate(ndimage.find_objects(labels), start=1):
        if dilim is None:
            continue
        mask = labels[dilim] == kimlik
        low = np.where(mask, detection_image[dilim], 0.0)
        en_az, en_cok = float(low[mask].min()), float(low[mask].max())
        if mask.sum() < 2 * min_pixels or en_az <= 0 or en_cok <= en_az:
            continue
        levels = np.geomspace(en_az, en_cok, n_levels + 2)[1:-1]
        peaks = _markers(low, mask, levels, contrast * float(low[mask].sum()))
        peaks.sort(key=lambda yx: -low[yx])  # en parlak tepe eski kimliği korur
        while len(peaks) >= 2:
            havza = _flood(low, mask, peaks)
            size = np.bincount(havza[mask], minlength=len(peaks) + 1)[1:]
            if size.min() >= min_pixels:
                break
            del peaks[int(np.argmin(size))]  # en az piksel koşulunu sağlamayan havza komşusuna katılır
        if len(peaks) < 2:
            continue
        for j in range(2, len(peaks) + 1):
            yeni[dilim][mask & (havza == j)] = sonraki
            sonraki += 1
    return yeni


@source(
    "Photutils2026",
    where="Kullanım kılavuzu: Image Segmentation ve Source Deblending",
    status="read",
    note="kaynak: her biri eşik üstünde olan en az belirli sayıda bağlı piksel; eşik arka plan gürültüsünün katı; "
    "eşiklemeden önce süzgeçleme; varsayılan bağlantı 8 komşuluk. Ayrıştırma: çok eşikleme + su havzası; "
    "'saddle' ölçütü, tepenin komşularından ayrıldığı eyer düzeyinin üstünde tuttuğu akının toplam akıya oranıdır; "
    "düzeyler kaynağın en küçük ve en büyük değeri arasındadır; ayrışmak için kaynaklar arasında eyer noktası gerekir. "
    "Düzeylerin geometrik aralıklanması ve havzaların öncelik kuyruklu taşkınla kurulması bu projenin seçimidir. "
    "Yöntemin makalesi [BertinArnouts1996] okunmadı",
)
def detect_sources(
    image: ArrayLike,
    background: ArrayLike | float,
    rms: ArrayLike | float,
    *,
    threshold_sigma: float = 3.0,
    min_pixels: int = 5,
    smoothing_fwhm: float | None = None,
    deblend_levels: int = 0,
    contrast: float = 0.005,
) -> SourceList:
    """Arka plan üstündeki bağlı piksel gruplarını kaynak olarak saptar.

    Parameters
    ----------
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

    Returns
    -------
    KaynakListesi
        Akıya göre azalan sırada.

    References
    ----------
    [Photutils2026] Image Segmentation, Source Deblending.

    Validity
    --------
    Ayrıştırma kapalıyken örtüşen kaynaklar tek kaynak sayılır. Açıkken aralarında
    eyer noktası olmayan (tek tepeli) çiftler yine ayrılmaz; eyer sığsa da ayrılmaz:
    2 FWHM aralıklı sentetik çiftlerin 32 düzey ve 0,005 karşıtlıkla akı oranı 2'ye
    kadar %92-98'i, oran 3'te %14'ü, oran 4'te %1'i ayrışır; 128 düzey ve 0,001
    karşıtlıkla oran 3'te %99'u, oran 4'te %59'u ayrışır (200 gerçekleme, rastgele yön); gürültülü saptama
    görüntüsünde küçük ``karsitlik`` gürültü tepelerini kaynak diye böler,
    yumuşatma bunu azaltır. Yaygın yapının (bulutsu) parlak bölgeleri de "kaynak" olarak çıkar.

    Verification
    ------------
    tests/unit/test_photometry.py: sentetik yıldız alanında tamlık, konum ve yanlış saptama sayısı; yakın çiftlerin ayrışması.
    """
    g = np.asarray(image, dtype=float) - background
    detection_image, threshold = g, threshold_sigma * np.asarray(rms, dtype=float)
    if smoothing_fwhm:
        s = smoothing_fwhm / 2.3548200450309493
        detection_image = ndimage.gaussian_filter(g, s, mode="nearest")
        threshold = threshold / (2.0 * np.sqrt(np.pi) * s)  # beyaz gürültünün Gauss yumuşatmasından sonraki rms'i
    labels, n = ndimage.label(detection_image > threshold, structure=np.ones((3, 3)))
    if n:
        number = np.bincount(labels.ravel(), minlength=n + 1)
        labels[number[labels] < min_pixels] = 0
        if deblend_levels > 0:
            labels = _deblend(detection_image, labels, deblend_levels, contrast, min_pixels)
    gecerli = np.unique(labels[labels > 0])
    if gecerli.size == 0:
        bos = np.empty(0)
        return SourceList(bos, bos, bos, bos, np.empty(0, dtype=np.int64))
    number = np.asarray(ndimage.sum_labels(np.ones_like(g), labels, gecerli)).astype(np.int64)
    pozitif = np.clip(g, 0.0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        merkez = np.array(ndimage.center_of_mass(pozitif, labels, gecerli)).reshape(-1, 2)
    # Yumuşatılmış görüntüde saptanan ama özgün görüntüde artı pikseli olmayan bölüt: ağırlık toplamı sıfırdır.
    # Gerçek veride (SOFI yığını) görüldü; bu bölütlerde merkez, bölütün geometrik ortasıdır.
    eksik = ~np.isfinite(merkez).all(axis=1)
    if eksik.any():
        merkez[eksik] = np.array(ndimage.center_of_mass(np.ones_like(g), labels, gecerli[eksik])).reshape(-1, 2)
    flux = np.asarray(ndimage.sum_labels(g, labels, gecerli))
    peak = np.asarray(ndimage.maximum(g, labels, gecerli))
    sira = np.argsort(-flux)
    return SourceList(merkez[sira, 1], merkez[sira, 0], flux[sira], peak[sira], number[sira])


@source(
    "Photutils2026",
    where="Kullanım kılavuzu: Aperture Photometry, 'subpixel' yöntemi",
    status="read",
    note="piksel N×N alt piksele bölünür; ağırlık, merkezi açıklığın içinde kalan alt piksellerin kesridir. "
    "Halkadan ortanca ile yerel gökyüzü çıkarma bu projenin seçimidir",
)
def aperture_photometry(
    image: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    radius: float,
    *,
    annulus: tuple[float, float] | None = None,
    subpixels: int = 5,
) -> tuple[Array, Array]:
    """Dairesel açıklıkta akı ölçer; isteğe bağlı halkadan yerel gökyüzünü çıkarır.

    Parameters
    ----------
    goruntu : ndarray (Y, X)
    x, y : ndarray (N,)
        Kaynak merkezleri [piksel].
    yaricap : float
        Açıklık yarıçapı [piksel].
    halka : (iç, dış), isteğe bağlı
        Gökyüzü halkasının yarıçapları; verilirse halkadaki ortanca piksel başına gökyüzü sayılır.
    alt_piksel : int
        Kenar piksellerinin kapsanan kesri için piksel başına örnekleme sayısı (her eksende).

    Returns
    -------
    (akı, alan) : gökyüzü çıkarılmış akı ve açıklığın etkin alanı [piksel²].

    References
    ----------
    [Photutils2026] Aperture Photometry.

    Validity
    --------
    Alt piksel örneklemesi daire alanını yaklaşık 1/alt_piksel² bağıl hatayla verir.
    Kenar pikselinin kapsanan kesri alanla ölçülür ve piksel içinde akı düzgün sayılır; açıklık
    PSF'in dik olduğu bölgeyi kesiyorsa (yarıçap FWHM'den küçükse) akı yüzde birkaç eksik ölçülür.
    Görüntü kenarına taşan açıklıklarda yalnızca görüntü içindeki pikseller toplanır.

    Verification
    ------------
    tests/unit/test_photometry.py: düz görüntüde alan πr²; Gauss yıldızda çevrelenen akı kesri 1 − exp(−r²/2σ²).
    """
    g = np.asarray(image, dtype=float)
    ny, nx = g.shape
    kaydirma = (np.arange(subpixels) + 0.5) / subpixels - 0.5
    akilar, alanlar = [], []
    for x0, y0 in zip(
        np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float)), strict=True
    ):
        dis = annulus[1] if annulus else radius
        x1, x2 = max(0, int(np.floor(x0 - dis - 1))), min(nx, int(np.ceil(x0 + dis + 2)))
        y1, y2 = max(0, int(np.floor(y0 - dis - 1))), min(ny, int(np.ceil(y0 + dis + 2)))
        parca = g[y1:y2, x1:x2]
        yy, xx = np.mgrid[y1:y2, x1:x2]
        ax = xx[..., None, None] + kaydirma[None, None, None, :] - x0
        month = yy[..., None, None] + kaydirma[None, None, :, None] - y0
        kesir = (ax**2 + month**2 <= radius**2).mean(axis=(-1, -2))
        gokyuzu = 0.0
        if annulus:
            r2 = (xx - x0) ** 2 + (yy - y0) ** 2
            secim = (r2 >= annulus[0] ** 2) & (r2 <= annulus[1] ** 2)
            gokyuzu = float(np.median(parca[secim])) if secim.any() else 0.0
        alan = float(kesir.sum())
        akilar.append(float((parca * kesir).sum()) - gokyuzu * alan)
        alanlar.append(alan)
    return np.array(akilar), np.array(alanlar)
