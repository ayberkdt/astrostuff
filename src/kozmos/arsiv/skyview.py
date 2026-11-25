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

    Doğrulama
    ---------
    tests/birim/test_arsiv.py (ağ gerektirir, ``-m ag``).
    """
    ad = f"skyview_{onbellek.anahtar(tarama, f'{sag_acikllik_derece:.6f}', f'{dik_acikllik_derece:.6f}', str(alan_derece), str(piksel))}.fits"
    yol = onbellek.indir(
        UC_NOKTA,
        Path(onbellek_dizini) / ad,
        parametreler={
            "Position": f"{sag_acikllik_derece:.6f},{dik_acikllik_derece:.6f}",
            "Survey": tarama,
            "Pixels": str(piksel),
            "Size": f"{alan_derece:g}",
            "Return": "FITS",
        },
        atif=f"{ATIF}; tarama: {tarama}",
        yenile=yenile,
    )
    return fits.fits_oku(yol)
