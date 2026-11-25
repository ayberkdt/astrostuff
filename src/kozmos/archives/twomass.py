"""2MASS Nokta Kaynak Kataloğu'ndan koni sorgusu.

Sorgu IRSA'nın TAP servisindeki ``fp_psc`` tablosuna gider. Katalog:
[Skrutskie2006]; sütun tanımları IRSA şema tablosundan okunmuştur [IRSA2026].
Katalog, bu depodaki kaynak saptama ve astrometri kodunun gerçek görüntüde
sınanması için bağımsız ölçüttür (``examples/real_image_validation.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kozmos._sources import source
from kozmos.archives import tap
from kozmos.archives.gaia import IRSA_TAP

TABLE = "fp_psc"
ATTRIBUTION = "2MASS Nokta Kaynak Kataloğu (Skrutskie ve ark. 2006); IRSA TAP servisi üzerinden"
COLUMNS = "ra, dec, err_maj, j_m, h_m, k_m, k_cmsig, k_snr, ph_qual"


@source(
    "IRSA2026",
    where="TAP_SCHEMA.columns, table_name = 'fp_psc'",
    status="read",
    note="ra/dec 'J2000 decimal deg'; err_maj konum belirsizliği elipsinin yarı büyük ekseni [yay saniyesi]; "
    "j_m/h_m/k_m 'selected default magnitude' [mag]; k_cmsig düzeltilmiş K belirsizliği; ph_qual fotometrik nitelik",
)
@source(
    "ADQL2023",
    where="Bölüm 4.2.11 (CIRCLE), 4.2.12 (CONTAINS), 4.2.18 (POINT)",
    status="read",
    note="CONTAINS, ilk geometri ikincinin içinde ya da sınırındaysa 1 döndürür; WHERE içinde 0 ya da 1 ile "
    "karşılaştırılır. CIRCLE merkez koordinatlarını ve yarıçapı derece alır. İlk argüman olarak verilen koordinat "
    "sistemi dizgisi ('J2000') belgede tarihsel ve kullanımdan kalkmakta olan seçenektir; IRSA servisinde çalıştığı sınandı",
)
def point_sources(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    cache_dir: Path,
    *,
    refresh: bool = False,
) -> dict[str, np.ndarray]:
    """Verilen yönün çevresindeki 2MASS nokta kaynaklarını indirir.

    Parameters
    ----------
    sag_acikllik_derece, dik_acikllik_derece : float
        Koninin merkezi (J2000) [derece].
    yaricap_derece : float
        Koninin yarıçapı [derece].
    onbellek_dizini : Path

    Returns
    -------
    dict[str, ndarray]
        ``ra``, ``dec`` [derece], ``err_maj`` [yay saniyesi], ``j_m``, ``h_m``,
        ``k_m`` [Vega kadiri], ``k_cmsig`` [kadir], ``k_snr``, ``ph_qual``.

    References
    ----------
    [IRSA2026] TAP_SCHEMA.columns; [ADQL2023] Bölüm 4.2.11, 4.2.12, 4.2.18; katalog: [Skrutskie2006].

    Validity
    --------
    ``k_cmsig`` boş olan kaynaklarda K kadiri üst sınırdır (kaynak o bantta
    saptanmamıştır); ``ph_qual`` dizgisinin üçüncü harfi K bandının niteliğidir.
    Bulutsuluk içinde katalog da eksiktir ve yanlış kaynak içerebilir.

    Verification
    ------------
    tests/unit/test_archives.py (ağ gerektirir, ``-m ag``).
    """
    adql = (
        f"SELECT {COLUMNS} FROM {TABLE} WHERE CONTAINS(POINT('J2000', ra, dec), "
        f"CIRCLE('J2000', {ra_deg:.6f}, {dec_deg:.6f}, {radius_deg:.6f})) = 1"
    )
    return tap.read_csv(tap.query(IRSA_TAP, adql, cache_dir, attribution=ATTRIBUTION, refresh=refresh))
