"""Gaia DR3 kataloğundan yıldız örneklemleri.

Sorgular IRSA'nın TAP servisindeki ``gaia_dr3_source`` tablosuna gider. ESA
Gaia arşivinin eski ``tap-server`` adresi 25 Kasım 2026'da sorguya açılış
sayfası döndürdüğü için kullanılmamıştır; AIP'deki ayna (gaia.aip.de)
çalışmakta ama VOTable döndürmektedir.

Katalog: Gaia DR3 [Vallenari2023]; sütun tanımları IRSA şema tablosundan [IRSA2026]. Mutlak kadir tanımı ve kalite kesimi
yaklaşımı Gaia DR2 HR diyagramı makalesinden okunmuştur [Babusiaux2018].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kozmos._sources import source
from kozmos.archives import tap

IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP/sync"
TABLE = "gaia_dr3_source"
ATTRIBUTION = "ESA Gaia DR3 (Gaia Collaboration, Vallenari ve ark. 2023); IRSA TAP servisi üzerinden"
COLUMNS = (
    "source_id, ra, dec, parallax, parallax_over_error, pmra, pmdec, "
    "phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp, ruwe"
)


@source(
    "Babusiaux2018",
    where="Ek B: arşiv sorgusu ve süzgeçler; Bölüm 2",
    status="read",
    note="makale parallax_over_error > 10 ve fotometrik süzgeçler kullanır; burada göreli paralaks hatası ve "
    "RUWE kesimi uygulanır (RUWE DR2 makalesinde yoktur; Gaia DR3 sütunudur)",
)
@source(
    "IRSA2026",
    where="TAP_SCHEMA.columns, table_name = 'gaia_dr3_source'",
    status="read",
    note="sütun tanımları ve birimleri servisin şema tablosundan: parallax [mas], pmra/pmdec [mas/yr], "
    "ruwe 'Renormalised unit weight error', bp_rp 'BP - RP colour' [mag]",
)
def nearby_stars(
    cache_dir: Path,
    *,
    parallax_min_mas: float = 20.0,
    parallax_snr_min: float = 20.0,
    ruwe_max: float = 1.4,
    refresh: bool = False,
) -> dict[str, np.ndarray]:
    """Güneş çevresindeki yıldızları Gaia DR3'ten kalite kesimleriyle indirir.

    Varsayılan seçim: paralaks > 20 mas (50 pc içi), paralaks S/N > 20,
    RUWE < 1,4, BP−RP rengi var. 25 Kasım 2026'da bu seçim 31 298 yıldız vermiştir.

    Parameters
    ----------
    onbellek_dizini : Path
    paralaks_alt_mas : float
        Alt paralaks sınırı [mas]; 1000/paralaks_alt parsek içindeki yıldızlar.
    paralaks_sn_alt : float
        ``parallax_over_error`` alt sınırı.
    ruwe_ust : float
        Yeniden normalleştirilmiş birim ağırlık hatasının üst sınırı.

    Returns
    -------
    dict[str, ndarray]
        Sütun adı → değerler; adlar Gaia arşivindeki gibidir.

    References
    ----------
    [Babusiaux2018] Ek B; [IRSA2026] TAP_SCHEMA.columns.

    Validity
    --------
    Gaia çok parlak yıldızlarda (yaklaşık G < 3) doyar; en parlak yıldızlar
    örneklemde yoktur. RUWE kesimi çözülmemiş çiftlerin bir kısmını ayıklar.

    Verification
    ------------
    tests/unit/test_archives.py (ağ gerektirir, ``-m ag``).
    """
    adql = (
        f"SELECT {COLUMNS} FROM {TABLE} WHERE parallax > {parallax_min_mas:g} "
        f"AND parallax_over_error > {parallax_snr_min:g} AND ruwe < {ruwe_max:g} AND bp_rp IS NOT NULL"
    )
    return tap.read_csv(tap.query(IRSA_TAP, adql, cache_dir, attribution=ATTRIBUTION, refresh=refresh))


@source(
    "Babusiaux2018",
    where="Bölüm 2: M_G = G + 5 + 5 log10(ϖ/1000), ϖ miliyay saniyesi",
    status="read",
    note="sönümleme terimi eklenmez",
)
def absolute_magnitude(apparent_magnitude: np.ndarray, parallax_mas: np.ndarray) -> np.ndarray:
    """Görünen kadir ve paralakstan mutlak kadiri hesaplar: M = m + 5 + 5 log10(ϖ/1000).

    Parameters
    ----------
    gorunen_kadir : ndarray
    paralaks_mas : ndarray
        Paralaks [mas], pozitif.

    References
    ----------
    [Babusiaux2018] Bölüm 2.

    Validity
    --------
    Paralaksın tersini uzaklık sayar; göreli paralaks hatası küçükken
    (yaklaşık %10 altı) kullanılabilir. Yıldızlararası sönümleme düzeltilmez;
    50 pc içinde küçüktür (sınanmadı).

    Verification
    ------------
    tests/unit/test_archives.py: 10 pc (ϖ = 100 mas) için M = m; uzaklık modülüyle tutarlılık.
    """
    return (
        np.asarray(apparent_magnitude, dtype=float)
        + 5.0
        + 5.0 * np.log10(np.asarray(parallax_mas, dtype=float) / 1000.0)
    )


@source(
    "IRSA2026",
    where="TAP_SCHEMA.columns, table_name = 'gaia_dr3_source'",
    status="read",
    note="ra/dec [deg], ref_epoch 'Reference epoch' [yr], pmra 'Proper motion in right ascension direction' "
    "(μ_α cos δ) ve pmdec [mas/yr], phot_g_mean_mag 'G-band mean magnitude' [mag]",
)
@source(
    "ADQL2023",
    where="Bölüm 4.2.11 (CIRCLE), 4.2.12 (CONTAINS), 4.2.18 (POINT)",
    status="read",
    note="koni sorgusu; koordinat sistemi dizgisi belgede kullanımdan kalkmakta olan seçenektir, IRSA'da çalışır",
)
def cone_search(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    cache_dir: Path,
    *,
    epoch_yr: float | None = None,
    refresh: bool = False,
) -> dict[str, np.ndarray]:
    """Verilen yönün çevresindeki Gaia DR3 kaynaklarını indirir; istenirse konumları bir döneme taşır.

    Parameters
    ----------
    sag_acikllik_derece, dik_acikllik_derece, yaricap_derece : float
        Koninin merkezi (ICRS) ve yarıçapı [derece].
    onbellek_dizini : Path
    donem_yil : float, isteğe bağlı
        Verilirse ``ra`` ve ``dec`` öz hareketle bu Jülyen yılına taşınır; öz hareketi olmayan
        kaynaklar olduğu gibi kalır.

    Returns
    -------
    dict[str, ndarray]
        ``source_id``, ``ra``, ``dec`` [derece], ``ref_epoch`` [yıl], ``pmra``, ``pmdec`` [mas/yıl],
        ``phot_g_mean_mag``.

    References
    ----------
    [IRSA2026] TAP_SCHEMA.columns; [ADQL2023] Bölüm 4.2.11, 4.2.12, 4.2.18; katalog: [Vallenari2023].

    Validity
    --------
    Dönem taşıması doğrusaldır: Δα = μ_α* Δt / cos δ, Δδ = μ_δ Δt; paralaks ve dikine hız
    uygulanmaz. Kutuplara çok yakın yönlerde 1/cos δ çarpanı güvenilmezdir.

    Verification
    ------------
    tests/unit/test_archives.py (ağ gerektirir, ``-m ag``): koni içinde kalma ve dönem taşımasının büyüklüğü.
    """
    adql = (
        f"SELECT source_id, ra, dec, ref_epoch, pmra, pmdec, phot_g_mean_mag FROM {TABLE} WHERE "
        f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra_deg:.6f}, {dec_deg:.6f}, "
        f"{radius_deg:.6f})) = 1"
    )
    s = tap.read_csv(tap.query(IRSA_TAP, adql, cache_dir, attribution=ATTRIBUTION, refresh=refresh))
    if epoch_yr is not None:
        dt = epoch_yr - s["ref_epoch"]
        mas = 1.0 / 3.6e6
        var = np.isfinite(s["pmra"]) & np.isfinite(s["pmdec"])
        s["ra"] = np.where(var, s["ra"] + s["pmra"] * dt * mas / np.cos(np.radians(s["dec"])), s["ra"])
        s["dec"] = np.where(var, s["dec"] + s["pmdec"] * dt * mas, s["dec"])
    return s
