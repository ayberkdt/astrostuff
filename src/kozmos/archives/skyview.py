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

from kozmos._sources import source
from kozmos.archives import cache
from kozmos.io import fits

ENDPOINT = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"
ATTRIBUTION = "NASA SkyView (HEASARC); taramaların kendi atıfları ayrıca gerekir"

# Görünür ve kızılötesi karşılaştırmasında kullanılan taramalar ve etkin dalga boyları [µm].
# Dalga boyları 2MASS için [Cohen2003] Tablo 2 (λ_iso), WISE için [Wright2010] bant adları; DSS2 için yaklaşık.
SURVEYS = {
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

ATTEMPTS = 3


@source(
    "SkyView2026",
    where="Batch Page: runquery.pl örnekleri (Position, Survey, Pixels, Size, Return)",
    status="read",
    note="tarama adları ('DSS2 Red', '2MASS-K', 'WISE 12') belgeden değil, çalışan sorgulardan doğrulandı",
)
def download_cutout(
    ra_deg: float,
    dec_deg: float,
    survey: str,
    cache_dir: Path,
    *,
    size_deg: float = 0.3,
    pixels: int = 400,
    refresh: bool = False,
) -> tuple[np.ndarray, dict]:
    """Verilen konum çevresinde bir taramadan kare görüntü kesiti indirir.

    Parameters
    ----------
    sag_acikllik_derece, dik_acikllik_derece : float
        Merkez konumu (ICRS) [derece].
    tarama : str
        SkyView tarama adı; bilinen adlar ``TARAMALAR`` sözlüğündedir.
    onbellek_dizini : Path
    alan_derece : float
        Görüntünün kenar uzunluğu [derece].
    piksel : int
        Kenar başına piksel sayısı.

    Returns
    -------
    (görüntü, başlık) : ``görüntü[y, x]`` ve FITS başlığı.

    References
    ----------
    [SkyView2026] Batch Page.

    Validity
    --------
    Görüntü yeniden örneklenmiştir; piksel değerleri taramanın özgün birimindedir
    ve taramadan taramaya değişir. Kuzey yukarı, doğu soldadır (SkyView varsayılanı;
    başlıktaki CDELT1 işaretinden denetlenir).

    Servis geçerli FITS döndürmezse istek üç kez yinelenir; bozuk yanıtlar önbelleğe alınmaz.

    Verification
    ------------
    tests/unit/test_archives.py: bozuk yanıtın yeniden denenmesi (ağsız) ve gerçek sorgu (``-m ag``).
    """
    name = f"skyview_{cache.key(survey, f'{ra_deg:.6f}', f'{dec_deg:.6f}', str(size_deg), str(pixels))}.fits"
    target = Path(cache_dir) / name
    for deneme in range(1, ATTEMPTS + 1):
        path = cache.download(
            ENDPOINT,
            target,
            parameters={
                "Position": f"{ra_deg:.6f},{dec_deg:.6f}",
                "Survey": survey,
                "Pixels": str(pixels),
                "Size": f"{size_deg:g}",
                "Return": "FITS",
            },
            attribution=f"{ATTRIBUTION}; tarama: {survey}",
            refresh=refresh or deneme > 1,
        )
        try:
            return fits.read_fits(path)
        except ValueError as hata:
            # Servis ara sıra 200 koduyla FITS olmayan ya da yarım bir yanıt döndürür (25 Kasım 2026'da testte
            # görüldü). Bozuk yanıt önbellekte bırakılmaz; incelenebilsin diye ``.hatali`` adıyla saklanır.
            hatali = target.with_name(f"{target.name}.hatali{deneme}")
            path.replace(hatali)
            if deneme == ATTEMPTS:
                bas = hatali.read_bytes()[:80]
                raise ValueError(
                    f"SkyView {ATTEMPTS} denemede geçerli FITS döndürmedi ({hata}); son yanıtın başı: {bas!r}; "
                    f"yanıtlar {hatali.parent} altında .hatali adıyla saklandı"
                ) from hata
    raise AssertionError("ulaşılamaz")
