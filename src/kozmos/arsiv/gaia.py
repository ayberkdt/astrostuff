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

from kozmos._kaynak import kaynak
from kozmos.arsiv import tap

IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP/sync"
TABLO = "gaia_dr3_source"
ATIF = "ESA Gaia DR3 (Gaia Collaboration, Vallenari ve ark. 2023); IRSA TAP servisi üzerinden"
SUTUNLAR = (
    "source_id, ra, dec, parallax, parallax_over_error, pmra, pmdec, "
    "phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp, ruwe"
)


@kaynak(
    "Babusiaux2018",
    yer="Ek B: arşiv sorgusu ve süzgeçler; Bölüm 2",
    teyit="okundu",
    not_="makale parallax_over_error > 10 ve fotometrik süzgeçler kullanır; burada göreli paralaks hatası ve "
    "RUWE kesimi uygulanır (RUWE DR2 makalesinde yoktur; Gaia DR3 sütunudur)",
)
@kaynak(
    "IRSA2026",
    yer="TAP_SCHEMA.columns, table_name = 'gaia_dr3_source'",
    teyit="okundu",
    not_="sütun tanımları ve birimleri servisin şema tablosundan: parallax [mas], pmra/pmdec [mas/yr], "
    "ruwe 'Renormalised unit weight error', bp_rp 'BP - RP colour' [mag]",
)
def yakin_yildizlar(
    onbellek_dizini: Path,
    *,
    paralaks_alt_mas: float = 20.0,
    paralaks_sn_alt: float = 20.0,
    ruwe_ust: float = 1.4,
    yenile: bool = False,
) -> dict[str, np.ndarray]:
    """Güneş çevresindeki yıldızları Gaia DR3'ten kalite kesimleriyle indirir.

    Varsayılan seçim: paralaks > 20 mas (50 pc içi), paralaks S/N > 20,
    RUWE < 1,4, BP−RP rengi var. 25 Kasım 2026'da bu seçim 31 298 yıldız vermiştir.

    Parametreler
    ------------
    onbellek_dizini : Path
    paralaks_alt_mas : float
        Alt paralaks sınırı [mas]; 1000/paralaks_alt parsek içindeki yıldızlar.
    paralaks_sn_alt : float
        ``parallax_over_error`` alt sınırı.
    ruwe_ust : float
        Yeniden normalleştirilmiş birim ağırlık hatasının üst sınırı.

    Döndürür
    --------
    dict[str, ndarray]
        Sütun adı → değerler; adlar Gaia arşivindeki gibidir.

    Kaynaklar
    ---------
    [Babusiaux2018] Ek B; [IRSA2026] TAP_SCHEMA.columns.

    Geçerlilik
    ----------
    Gaia çok parlak yıldızlarda (yaklaşık G < 3) doyar; en parlak yıldızlar
    örneklemde yoktur. RUWE kesimi çözülmemiş çiftlerin bir kısmını ayıklar.

    Doğrulama
    ---------
    tests/birim/test_arsiv.py (ağ gerektirir, ``-m ag``).
    """
    adql = (
        f"SELECT {SUTUNLAR} FROM {TABLO} WHERE parallax > {paralaks_alt_mas:g} "
        f"AND parallax_over_error > {paralaks_sn_alt:g} AND ruwe < {ruwe_ust:g} AND bp_rp IS NOT NULL"
    )
    return tap.csv_oku(tap.sorgula(IRSA_TAP, adql, onbellek_dizini, atif=ATIF, yenile=yenile))


@kaynak(
    "Babusiaux2018",
    yer="Bölüm 2: M_G = G + 5 + 5 log10(ϖ/1000), ϖ miliyay saniyesi",
    teyit="okundu",
    not_="sönümleme terimi eklenmez",
)
def mutlak_kadir(gorunen_kadir: np.ndarray, paralaks_mas: np.ndarray) -> np.ndarray:
    """Görünen kadir ve paralakstan mutlak kadiri hesaplar: M = m + 5 + 5 log10(ϖ/1000).

    Parametreler
    ------------
    gorunen_kadir : ndarray
    paralaks_mas : ndarray
        Paralaks [mas], pozitif.

    Kaynaklar
    ---------
    [Babusiaux2018] Bölüm 2.

    Geçerlilik
    ----------
    Paralaksın tersini uzaklık sayar; göreli paralaks hatası küçükken
    (yaklaşık %10 altı) kullanılabilir. Yıldızlararası sönümleme düzeltilmez;
    50 pc içinde küçüktür (sınanmadı).

    Doğrulama
    ---------
    tests/birim/test_arsiv.py: 10 pc (ϖ = 100 mas) için M = m; uzaklık modülüyle tutarlılık.
    """
    return np.asarray(gorunen_kadir, dtype=float) + 5.0 + 5.0 * np.log10(np.asarray(paralaks_mas, dtype=float) / 1000.0)


@kaynak(
    "IRSA2026",
    yer="TAP_SCHEMA.columns, table_name = 'gaia_dr3_source'",
    teyit="okundu",
    not_="ra/dec [deg], ref_epoch 'Reference epoch' [yr], pmra 'Proper motion in right ascension direction' "
    "(μ_α cos δ) ve pmdec [mas/yr], phot_g_mean_mag 'G-band mean magnitude' [mag]",
)
@kaynak(
    "ADQL2023",
    yer="Bölüm 4.2.11 (CIRCLE), 4.2.12 (CONTAINS), 4.2.18 (POINT)",
    teyit="okundu",
    not_="koni sorgusu; koordinat sistemi dizgisi belgede kullanımdan kalkmakta olan seçenektir, IRSA'da çalışır",
)
def koni_sorgusu(
    sag_acikllik_derece: float,
    dik_acikllik_derece: float,
    yaricap_derece: float,
    onbellek_dizini: Path,
    *,
    donem_yil: float | None = None,
    yenile: bool = False,
) -> dict[str, np.ndarray]:
    """Verilen yönün çevresindeki Gaia DR3 kaynaklarını indirir; istenirse konumları bir döneme taşır.

    Parametreler
    ------------
    sag_acikllik_derece, dik_acikllik_derece, yaricap_derece : float
        Koninin merkezi (ICRS) ve yarıçapı [derece].
    onbellek_dizini : Path
    donem_yil : float, isteğe bağlı
        Verilirse ``ra`` ve ``dec`` öz hareketle bu Jülyen yılına taşınır; öz hareketi olmayan
        kaynaklar olduğu gibi kalır.

    Döndürür
    --------
    dict[str, ndarray]
        ``source_id``, ``ra``, ``dec`` [derece], ``ref_epoch`` [yıl], ``pmra``, ``pmdec`` [mas/yıl],
        ``phot_g_mean_mag``.

    Kaynaklar
    ---------
    [IRSA2026] TAP_SCHEMA.columns; [ADQL2023] Bölüm 4.2.11, 4.2.12, 4.2.18; katalog: [Vallenari2023].

    Geçerlilik
    ----------
    Dönem taşıması doğrusaldır: Δα = μ_α* Δt / cos δ, Δδ = μ_δ Δt; paralaks ve dikine hız
    uygulanmaz. Kutuplara çok yakın yönlerde 1/cos δ çarpanı güvenilmezdir.

    Doğrulama
    ---------
    tests/birim/test_arsiv.py (ağ gerektirir, ``-m ag``): koni içinde kalma ve dönem taşımasının büyüklüğü.
    """
    adql = (
        f"SELECT source_id, ra, dec, ref_epoch, pmra, pmdec, phot_g_mean_mag FROM {TABLO} WHERE "
        f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {sag_acikllik_derece:.6f}, {dik_acikllik_derece:.6f}, "
        f"{yaricap_derece:.6f})) = 1"
    )
    s = tap.csv_oku(tap.sorgula(IRSA_TAP, adql, onbellek_dizini, atif=ATIF, yenile=yenile))
    if donem_yil is not None:
        dt = donem_yil - s["ref_epoch"]
        mas = 1.0 / 3.6e6
        var = np.isfinite(s["pmra"]) & np.isfinite(s["pmdec"])
        s["ra"] = np.where(var, s["ra"] + s["pmra"] * dt * mas / np.cos(np.radians(s["dec"])), s["ra"])
        s["dec"] = np.where(var, s["dec"] + s["pmdec"] * dt * mas, s["dec"])
    return s
