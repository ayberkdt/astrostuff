"""2MASS Nokta Kaynak Kataloğu'ndan koni sorgusu.

Sorgu IRSA'nın TAP servisindeki ``fp_psc`` tablosuna gider. Katalog:
[Skrutskie2006]; sütun tanımları IRSA şema tablosundan okunmuştur [IRSA2026].
Katalog, bu depodaki kaynak saptama ve astrometri kodunun gerçek görüntüde
sınanması için bağımsız ölçüttür (``ornekler/gercek_goruntu_dogrulamasi.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kozmos._kaynak import kaynak
from kozmos.arsiv import tap
from kozmos.arsiv.gaia import IRSA_TAP

TABLO = "fp_psc"
ATIF = "2MASS Nokta Kaynak Kataloğu (Skrutskie ve ark. 2006); IRSA TAP servisi üzerinden"
SUTUNLAR = "ra, dec, err_maj, j_m, h_m, k_m, k_cmsig, k_snr, ph_qual"


@kaynak(
    "IRSA2026",
    yer="TAP_SCHEMA.columns, table_name = 'fp_psc'",
    teyit="okundu",
    not_="ra/dec 'J2000 decimal deg'; err_maj konum belirsizliği elipsinin yarı büyük ekseni [yay saniyesi]; "
    "j_m/h_m/k_m 'selected default magnitude' [mag]; k_cmsig düzeltilmiş K belirsizliği; ph_qual fotometrik nitelik",
)
@kaynak(
    "ADQL2023",
    yer="Bölüm 4.2.11 (CIRCLE), 4.2.12 (CONTAINS), 4.2.18 (POINT)",
    teyit="okundu",
    not_="CONTAINS, ilk geometri ikincinin içinde ya da sınırındaysa 1 döndürür; WHERE içinde 0 ya da 1 ile "
    "karşılaştırılır. CIRCLE merkez koordinatlarını ve yarıçapı derece alır. İlk argüman olarak verilen koordinat "
    "sistemi dizgisi ('J2000') belgede tarihsel ve kullanımdan kalkmakta olan seçenektir; IRSA servisinde çalıştığı sınandı",
)
def nokta_kaynaklari(
    sag_acikllik_derece: float,
    dik_acikllik_derece: float,
    yaricap_derece: float,
    onbellek_dizini: Path,
    *,
    yenile: bool = False,
) -> dict[str, np.ndarray]:
    """Verilen yönün çevresindeki 2MASS nokta kaynaklarını indirir.

    Parametreler
    ------------
    sag_acikllik_derece, dik_acikllik_derece : float
        Koninin merkezi (J2000) [derece].
    yaricap_derece : float
        Koninin yarıçapı [derece].
    onbellek_dizini : Path

    Döndürür
    --------
    dict[str, ndarray]
        ``ra``, ``dec`` [derece], ``err_maj`` [yay saniyesi], ``j_m``, ``h_m``,
        ``k_m`` [Vega kadiri], ``k_cmsig`` [kadir], ``k_snr``, ``ph_qual``.

    Kaynaklar
    ---------
    [IRSA2026] TAP_SCHEMA.columns; [ADQL2023] Bölüm 4.2.11, 4.2.12, 4.2.18; katalog: [Skrutskie2006].

    Geçerlilik
    ----------
    ``k_cmsig`` boş olan kaynaklarda K kadiri üst sınırdır (kaynak o bantta
    saptanmamıştır); ``ph_qual`` dizgisinin üçüncü harfi K bandının niteliğidir.
    Bulutsuluk içinde katalog da eksiktir ve yanlış kaynak içerebilir.

    Doğrulama
    ---------
    tests/birim/test_arsiv.py (ağ gerektirir, ``-m ag``).
    """
    adql = (
        f"SELECT {SUTUNLAR} FROM {TABLO} WHERE CONTAINS(POINT('J2000', ra, dec), "
        f"CIRCLE('J2000', {sag_acikllik_derece:.6f}, {dik_acikllik_derece:.6f}, {yaricap_derece:.6f})) = 1"
    )
    return tap.csv_oku(tap.sorgula(IRSA_TAP, adql, onbellek_dizini, atif=ATIF, yenile=yenile))
