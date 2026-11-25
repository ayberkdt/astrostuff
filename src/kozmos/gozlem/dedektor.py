"""CCD/CMOS dedektörün sinyal modeli, sentetik kare üreteci ve sinyal-gürültü oranı.

Model ve CCD denklemi [Howell2006] kitabının 3. ve 4. bölümlerine dayanır;
kitabın metni bu çalışmada açılmamıştır ve kayıtlar "bekliyor" durumundadır.
Üretecin amacı indirgeme algoritmalarını gerçeğin bilindiği veride
sınamaktır: üretilen karenin altındaki gerçek sahne de döndürülür.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._kaynak import kaynak

Dizi = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Dedektor:
    """Dedektörün sabit özellikleri.

    Öznitelikler
    ------------
    kazanc_e_adu : float
        Kazanç [e⁻/ADU].
    okuma_gurultusu_e : float
        Okuma gürültüsü [e⁻, RMS].
    karanlik_akim_e_s : float
        Karanlık akım [e⁻/s/piksel].
    bias_adu : float
        Bias düzeyi [ADU].
    """

    kazanc_e_adu: float = 1.5
    okuma_gurultusu_e: float = 5.0
    karanlik_akim_e_s: float = 0.05
    bias_adu: float = 1000.0


@kaynak(
    "Howell2006",
    yer="Bölüm 3-4: CCD sinyal modeli",
    teyit="bekliyor",
    not_="ham[ADU] = Poisson((sahne·düz + karanlık)·t)/kazanç + bias + N(0, okuma/kazanç)",
)
def ham_kare_uret(
    sahne_e_s: ArrayLike,
    poz_s: float,
    dedektor: Dedektor,
    uretec: np.random.Generator,
    *,
    duz_alan: ArrayLike | float = 1.0,
) -> Dizi:
    """Verilen sahneden gürültülü ham dedektör karesi üretir.

    ham = Poisson[(sahne · düz_alan + karanlık_akım) · t] / kazanç + bias + okuma gürültüsü

    Parametreler
    ------------
    sahne_e_s : ndarray (Y, X)
        Dedektöre düşen akı [e⁻/s/piksel]: gökyüzü ve kaynakların toplamı. Bias ve
        karanlık kareleri için sıfır verilir.
    poz_s : float
        Poz süresi [s]; bias karesi için 0.
    dedektor : Dedektor
    uretec : numpy.random.Generator
    duz_alan : ndarray (Y, X) ya da float
        Pikselden piksele göreli duyarlılık (ortalaması 1 dolayında).

    Döndürür
    --------
    ndarray (Y, X)
        Ham kare [ADU]. Sayısallaştırma (tamsayıya yuvarlama) uygulanmaz.

    Kaynaklar
    ---------
    [Howell2006] Bölüm 3-4.

    Geçerlilik
    ----------
    Doğrusal dedektör: doyma, doğrusallıktan sapma, yük aktarım verimsizliği,
    sıcak piksel ve kozmik ışın içermez. Karanlık akım düz alandan etkilenmez.

    Doğrulama
    ---------
    tests/birim/test_indirgeme.py: üretilen karelerden kazanç ve okuma gürültüsünün geri kazanımı.
    """
    sahne = np.asarray(sahne_e_s, dtype=float)
    beklenen = (sahne * duz_alan + dedektor.karanlik_akim_e_s) * poz_s
    elektron = uretec.poisson(np.maximum(beklenen, 0.0)).astype(float)
    okuma = uretec.normal(0.0, dedektor.okuma_gurultusu_e, sahne.shape)
    return (elektron + okuma) / dedektor.kazanc_e_adu + dedektor.bias_adu


@kaynak(
    "Howell2006",
    yer="Bölüm 5: nokta dağılım fonksiyonu (dairesel Gauss yaklaşımı)",
    teyit="bekliyor",
    not_="piksel alanı üzerinde tam integral: hata fonksiyonu farkları; FWHM = 2 sqrt(2 ln 2) σ",
)
def yildiz_alani(bicim: tuple[int, int], x: ArrayLike, y: ArrayLike, aki_e_s: ArrayLike, fwhm_piksel: float) -> Dizi:
    """Dairesel Gauss PSF'li nokta kaynaklardan oluşan sahne üretir.

    Her piksel, profilin piksel merkezindeki değeri değil piksel alanı
    üzerindeki integralidir; bu yüzden dar PSF'te de toplam akı korunur.

    Parametreler
    ------------
    bicim : (Y, X)
    x, y : ndarray (N,)
        Kaynak merkezleri [piksel]; piksel merkezi tamsayı koordinattadır.
    aki_e_s : ndarray (N,)
        Kaynakların toplam akısı [e⁻/s].
    fwhm_piksel : float

    Döndürür
    --------
    ndarray (Y, X) [e⁻/s/piksel]

    Kaynaklar
    ---------
    [Howell2006] Bölüm 5.

    Geçerlilik
    ----------
    Gerçek PSF'in kanatları Gauss'tan geniştir (Moffat); bu model yalnızca
    algoritma sınaması içindir.

    Doğrulama
    ---------
    tests/birim/test_indirgeme.py: kenardan uzak kaynakta toplam akının korunumu.
    """
    from scipy.special import erf

    ny, nx = bicim
    sigma = fwhm_piksel / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    kenar_x = np.arange(nx + 1) - 0.5
    kenar_y = np.arange(ny + 1) - 0.5
    sahne = np.zeros(bicim)
    for x0, y0, f in zip(np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(aki_e_s), strict=True):
        px = 0.5 * np.diff(erf((kenar_x - x0) / (np.sqrt(2.0) * sigma)))
        py = 0.5 * np.diff(erf((kenar_y - y0) / (np.sqrt(2.0) * sigma)))
        sahne += f * np.outer(py, px)
    return sahne


@kaynak(
    "Howell2006",
    yer="Bölüm 4.4: CCD denklemi",
    teyit="bekliyor",
    not_="S/N = N* / sqrt(N* + n_pix (1 + n_pix/n_B)(N_S + N_D + N_R²)); sayısallaştırma terimi yok",
)
@kaynak("MerlineHowell1995", yer="arka plan kestirimi terimi (1 + n_pix/n_B)", teyit="bekliyor")
def sinyal_gurultu_orani(
    kaynak_e: ArrayLike,
    gokyuzu_e_piksel: ArrayLike,
    karanlik_e_piksel: ArrayLike,
    okuma_gurultusu_e: float,
    piksel_sayisi: float,
    arka_plan_piksel_sayisi: float = np.inf,
) -> Dizi:
    """Açıklık fotometrisi için sinyal-gürültü oranını CCD denklemiyle hesaplar.

    S/N = N* / sqrt( N* + n_pix (1 + n_pix/n_B) (N_S + N_D + N_R²) )

    Parametreler
    ------------
    kaynak_e : float ya da ndarray
        Açıklıktaki kaynak elektronları N*.
    gokyuzu_e_piksel, karanlik_e_piksel : float ya da ndarray
        Piksel başına gökyüzü ve karanlık akım elektronları.
    okuma_gurultusu_e : float
        Okuma gürültüsü [e⁻ RMS].
    piksel_sayisi : float
        Açıklıktaki piksel sayısı n_pix.
    arka_plan_piksel_sayisi : float
        Arka plan kestiriminde kullanılan piksel sayısı n_B; sonsuz ise terim 1 olur.

    Kaynaklar
    ---------
    [Howell2006] Bölüm 4.4; [MerlineHowell1995].

    Geçerlilik
    ----------
    Poisson ve Gauss gürültülerinin bağımsız toplamı; ilintili gürültü
    (yeniden örnekleme, düz alan hatası) içermez.

    Doğrulama
    ---------
    tests/birim/test_indirgeme.py: kaynak baskın limitte sqrt(N*); sentetik karelerde ölçülen saçılım.
    """
    n = np.asarray(kaynak_e, dtype=float)
    arka = piksel_sayisi * (1.0 + piksel_sayisi / arka_plan_piksel_sayisi)
    gurultu2 = n + arka * (
        np.asarray(gokyuzu_e_piksel, dtype=float) + np.asarray(karanlik_e_piksel, dtype=float) + okuma_gurultusu_e**2
    )
    return n / np.sqrt(gurultu2)
