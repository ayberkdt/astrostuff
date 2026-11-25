"""NASA SkyView'dan görüntü kesitleri.

SkyView, çok sayıda taramayı (DSS2, 2MASS, WISE ...) istenen konum, alan ve
piksel ızgarasına yeniden örnekleyip FITS olarak verir. Sorgu parametreleri
(``Position``, ``Survey``, ``Pixels``, ``Size``, ``Return``) SkyView'ın toplu
kullanım belgesindeki örneklerden okunmuştur [SkyView2026].

Kesitler yeniden örneklenmiş görüntülerdir: fotometrik kalibrasyon için özgün
arşiv verisinin yerini tutmaz, görsel karşılaştırma ve algoritma sınaması içindir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kozmos._kaynak import kaynak
from kozmos.arsiv import onbellek
from kozmos.io import fits

UC_NOKTA = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"
ATIF = "NASA SkyView (HEASARC); taramaların kendi atıfları ayrıca gerekir"

# Görünür ve kızılötesi karşılaştırmasında kullanılan taramalar ve etkin dalga boyları [µm].
# Dalga boyları 2MASS için [Cohen2003] Tablo 2 (λ_iso), WISE için [Wright2010] bant adları; DSS2 için yaklaşık.
TARAMALAR = {
    "DSS2 Blue": 0.44,
    "DSS2 Red": 0.64,
    "DSS2 IR": 0.85,
    "2MASS-J": 1.235,
    "2MASS-H": 1.662,
    "2MASS-K": 2.159,
    "WISE 3.4": 3.4,
    "WISE 4.6": 4.6,
    "WISE 12": 12.0,
    "WISE 22": 22.0,
}

DENEME_SAYISI = 3


@kaynak(
    "SkyView2026",
    yer="Batch Page: runquery.pl örnekleri (Position, Survey, Pixels, Size, Return)",
    teyit="okundu",
    not_="tarama adları ('DSS2 Red', '2MASS-K', 'WISE 12') belgeden değil, çalışan sorgulardan doğrulandı",
)
def kesit_indir(
    sag_acikllik_derece: float,
    dik_acikllik_derece: float,
    tarama: str,
    onbellek_dizini: Path,
    *,
    alan_derece: float = 0.3,
    piksel: int = 400,
    yenile: bool = False,
) -> tuple[np.ndarray, dict]:
    """Verilen konum çevresinde bir taramadan kare görüntü kesiti indirir.

    Parametreler
    ------------
    sag_acikllik_derece, dik_acikllik_derece : float
        Merkez konumu (ICRS) [derece].
    tarama : str
        SkyView tarama adı; bilinen adlar ``TARAMALAR`` sözlüğündedir.
    onbellek_dizini : Path
    alan_derece : float
        Görüntünün kenar uzunluğu [derece].
    piksel : int
        Kenar başına piksel sayısı.

    Döndürür
    --------
    (görüntü, başlık) : ``görüntü[y, x]`` ve FITS başlığı.

    Kaynaklar
    ---------
    [SkyView2026] Batch Page.

    Geçerlilik
    ----------
    Görüntü yeniden örneklenmiştir; piksel değerleri taramanın özgün birimindedir
    ve taramadan taramaya değişir. Kuzey yukarı, doğu soldadır (SkyView varsayılanı;
    başlıktaki CDELT1 işaretinden denetlenir).

    Servis geçerli FITS döndürmezse istek üç kez yinelenir; bozuk yanıtlar önbelleğe alınmaz.

    Doğrulama
    ---------
    tests/birim/test_arsiv.py: bozuk yanıtın yeniden denenmesi (ağsız) ve gerçek sorgu (``-m ag``).
    """
    ad = f"skyview_{onbellek.anahtar(tarama, f'{sag_acikllik_derece:.6f}', f'{dik_acikllik_derece:.6f}', str(alan_derece), str(piksel))}.fits"
    hedef = Path(onbellek_dizini) / ad
    for deneme in range(1, DENEME_SAYISI + 1):
        yol = onbellek.indir(
            UC_NOKTA,
            hedef,
            parametreler={
                "Position": f"{sag_acikllik_derece:.6f},{dik_acikllik_derece:.6f}",
                "Survey": tarama,
                "Pixels": str(piksel),
                "Size": f"{alan_derece:g}",
                "Return": "FITS",
            },
            atif=f"{ATIF}; tarama: {tarama}",
            yenile=yenile or deneme > 1,
        )
        try:
            return fits.fits_oku(yol)
        except ValueError as hata:
            # Servis ara sıra 200 koduyla FITS olmayan ya da yarım bir yanıt döndürür (25 Kasım 2026'da testte
            # görüldü). Bozuk yanıt önbellekte bırakılmaz; incelenebilsin diye ``.hatali`` adıyla saklanır.
            hatali = hedef.with_name(f"{hedef.name}.hatali{deneme}")
            yol.replace(hatali)
            if deneme == DENEME_SAYISI:
                bas = hatali.read_bytes()[:80]
                raise ValueError(
                    f"SkyView {DENEME_SAYISI} denemede geçerli FITS döndürmedi ({hata}); son yanıtın başı: {bas!r}; "
                    f"yanıtlar {hatali.parent} altında .hatali adıyla saklandı"
                ) from hata
    raise AssertionError("ulaşılamaz")
