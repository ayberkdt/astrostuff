"""Arka plan kestirimi, kaynak saptama ve açıklık fotometrisi.

Arka plan kestiricisi SExtractor belgelerinin "Modeling the background"
bölümünden okunmuştur [SExtractorBelge2026]: ortanca çevresinde ±3σ yinelemeli
kırpma, kip = 2,5 × ortanca − 1,5 × ortalama, kip ile ortanca %30'dan fazla
ayrışırsa ortanca; harita için hücre ızgarası, ortanca süzgeci ve bikübik
spline. Yöntemin makalesi [BertinArnouts1996] bu çalışmada okunmamıştır.

Kaynak saptama (eşik üstü bağlı piksel grupları) ve açıklık fotometrisi
(alt piksel örneklemeli daire) yaygın uygulamayı izler; bu iki fonksiyonun
kaynak metni açılmamıştır ve kayıtları "bekliyor" durumundadır. İkisi de
gerçeği bilinen sentetik görüntülerle sınanır.
"""

from __future__ import annotations

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


@kaynak(
    "BertinArnouts1996",
    yer="saptama: eşik üstü bağlı pikseller",
    teyit="bekliyor",
    not_="arka plan çıkarılmış görüntüde k·rms üstündeki 8-komşulukla bağlı piksel grupları; en az piksel "
    "koşulu. Makale ve belgenin saptama bölümü okunmadı; ayrıştırma (deblending) ve süzgeçleme yok",
)
def kaynak_bul(
    goruntu: ArrayLike,
    arka_plan: ArrayLike | float,
    rms: ArrayLike | float,
    *,
    esik_sigma: float = 3.0,
    en_az_piksel: int = 5,
    yumusatma_fwhm: float | None = None,
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

    Döndürür
    --------
    KaynakListesi
        Akıya göre azalan sırada.

    Kaynaklar
    ---------
    [BertinArnouts1996] saptama.

    Geçerlilik
    ----------
    Örtüşen kaynaklar tek kaynak sayılır. Yaygın yapının (bulutsu) parlak
    bölgeleri de "kaynak" olarak çıkar.

    Doğrulama
    ---------
    tests/birim/test_fotometri.py: sentetik yıldız alanında tamlık, konum ve yanlış saptama sayısı.
    """
    g = np.asarray(goruntu, dtype=float) - arka_plan
    saptama, esik = g, esik_sigma * np.asarray(rms, dtype=float)
    if yumusatma_fwhm:
        s = yumusatma_fwhm / 2.3548200450309493
        saptama = ndimage.gaussian_filter(g, s, mode="nearest")
        esik = esik / (2.0 * np.sqrt(np.pi) * s)  # beyaz gürültünün Gauss yumuşatmasından sonraki rms'i
    etiket, n = ndimage.label(saptama > esik, structure=np.ones((3, 3)))
    if n == 0:
        bos = np.empty(0)
        return KaynakListesi(bos, bos, bos, bos, np.empty(0, dtype=np.int64))
    kimlik = np.arange(1, n + 1)
    sayi = ndimage.sum_labels(np.ones_like(g), etiket, kimlik).astype(np.int64)
    gecerli = kimlik[sayi >= en_az_piksel]
    pozitif = np.clip(g, 0.0, None)
    merkez = np.array(ndimage.center_of_mass(pozitif, etiket, gecerli)).reshape(-1, 2)
    aki = np.asarray(ndimage.sum_labels(g, etiket, gecerli))
    tepe = np.asarray(ndimage.maximum(g, etiket, gecerli))
    sira = np.argsort(-aki)
    return KaynakListesi(merkez[sira, 1], merkez[sira, 0], aki[sira], tepe[sira], sayi[sayi >= en_az_piksel][sira])


@kaynak(
    "Howell2006",
    yer="Bölüm 5: açıklık fotometrisi",
    teyit="bekliyor",
    not_="dairesel açıklık toplamı eksi halkadan kestirilen yerel gökyüzü; kenar pikselleri alt piksel "
    "örneklemesiyle ağırlıklandırılır. Kaynak metni açılmadı",
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
    [Howell2006] Bölüm 5.

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
