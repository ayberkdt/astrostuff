"""Pantheon+SH0ES verisinden Ωm ve H0 kestirimi.

İki analiz yapar ve sonucu yayımlanan değerlerle karşılaştırır [Brout2022]:

1. Yalnızca süpernovalar, düz ΛCDM: Ωm. Mutlak kadir M (H0 ile dejenere)
   düz önselle analitik olarak marjinalleştirilir; Ωm sonsalı ızgarada
   hesaplanır.
2. Sefeid kalibratörleri eklenir: (H0, Ωm, M) için kendi topluluk
   örnekleyicimizle MCMC.

Kullanım:
    python projeler/hubble/hubble_analizi.py [--adim 1500] [--tohum 20261125]

Çıktı: ``projeler/hubble/sonuc.json`` (çalıştırma kaydı ve sonuçlar) ve
``projeler/hubble/zincir.npy``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy

import kozmos
from kozmos.arsiv import pantheon
from kozmos.istatistik.mcmc import (
    bolunmus_rhat,
    oz_ilinti_suresi,
    rhat_sira_normallestirilmis,
    topluluk_ornekle,
)
from kozmos.kozmoloji.hubble_diyagrami import (
    HubbleVerisi,
    en_iyi_mutlak_kadir,
    log_olabilirlik,
    veri_hazirla,
)
from kozmos.kozmoloji.uzaklik import Kozmoloji

KOK = Path(__file__).resolve().parents[2]
ONBELLEK = KOK / "veri" / "ham" / "pantheon"
CIKTI = Path(__file__).resolve().parent

# Yayımlanan değerler: Brout ve ark. 2022, arXiv:2202.04077 TeX kaynağı (main.tex,
# \fLCDMOMPan ve \fLCDMHPanSH makroları), 25 Kasım 2026'da okundu.
YAYIMLANAN = {
    "omega_m_duz_lcdm": {"deger": 0.334, "belirsizlik": 0.018, "kaynak": "Brout2022, düz ΛCDM, yalnızca SN"},
    "h0_duz_lcdm": {"deger": 73.6, "belirsizlik": 1.1, "kaynak": "Brout2022, düz ΛCDM, Pantheon+ ve SH0ES"},
}
H0_NOMINAL = 70.0  # Analiz 1'de M ile dejenere; sonucu etkilemez.


def omega_m_izgara(veri: HubbleVerisi, izgara: np.ndarray) -> dict[str, float | list[float]]:
    """Analiz 1: M marjinalleştirilmiş Ωm sonsalı (düz önsel, 0 < Ωm < 1)."""
    ki2 = np.array([en_iyi_mutlak_kadir(veri, Kozmoloji.duz(H0_NOMINAL, om))[2] for om in izgara])
    yogunluk = np.exp(-0.5 * (ki2 - ki2.min()))
    yogunluk /= np.trapezoid(yogunluk, izgara)
    ortalama = float(np.trapezoid(izgara * yogunluk, izgara))
    sapma = float(np.sqrt(np.trapezoid((izgara - ortalama) ** 2 * yogunluk, izgara)))
    return {
        "ortalama": ortalama,
        "standart_sapma": sapma,
        "en_olasi": float(izgara[np.argmin(ki2)]),
        "ki_kare_min": float(ki2.min()),
        "serbestlik_derecesi": len(veri) - 2,
        "izgara": izgara.tolist(),
        "yogunluk": yogunluk.tolist(),
    }


def h0_mcmc(veri: HubbleVerisi, adim: int, uretec: np.random.Generator) -> tuple[dict[str, object], np.ndarray]:
    """Analiz 2: (H0, Ωm, M) için topluluk örnekleyici."""

    def log_sonsal(p: np.ndarray) -> float:
        h0, om, m = p
        if not (50.0 < h0 < 100.0 and 0.05 < om < 0.95 and -21.0 < m < -18.0):
            return -np.inf
        return log_olabilirlik(veri, Kozmoloji.duz(h0, om), m)

    yuruyucu = 32
    baslangic = np.array([73.0, 0.33, -19.25]) + uretec.normal(0, [0.5, 0.01, 0.01], (yuruyucu, 3))
    zincir = topluluk_ornekle(log_sonsal, baslangic, adim, uretec)
    tau = oz_ilinti_suresi(zincir.ornekler)
    atilacak = int(min(adim // 2, max(100, 5 * tau.max())))
    kalan = zincir.ornekler[atilacak:]
    duz = kalan.reshape(-1, 3)
    adlar = ["h0", "omega_m", "mutlak_kadir"]
    sonuc: dict[str, object] = {
        ad: {
            "ortalama": float(duz[:, i].mean()),
            "standart_sapma": float(duz[:, i].std(ddof=1)),
            "yuzdelikler_16_50_84": [float(q) for q in np.percentile(duz[:, i], [16, 50, 84])],
        }
        for i, ad in enumerate(adlar)
    }
    sonuc["tanilar"] = {
        "adim": adim,
        "yuruyucu": yuruyucu,
        "atilan_isinma": atilacak,
        "oz_ilinti_suresi": tau.tolist(),
        "etkin_ornek_sayisi": (kalan.shape[0] * yuruyucu / tau).tolist(),
        "bolunmus_rhat": bolunmus_rhat(kalan).tolist(),
        "rhat_sira_normallestirilmis": rhat_sira_normallestirilmis(kalan).tolist(),
        "ortalama_kabul_orani": float(zincir.kabul_orani.mean()),
        "zincir_yeterince_uzun": bool(adim > 50 * tau.max()),
    }
    en_iyi = duz[np.argmax(zincir.log_olasilik[atilacak:].reshape(-1))]
    sonuc["ki_kare_min"] = float(-2 * zincir.log_olasilik[atilacak:].max())
    sonuc["serbestlik_derecesi"] = len(veri) - 3
    sonuc["en_olasi"] = dict(zip(adlar, map(float, en_iyi), strict=True))
    return sonuc, zincir.ornekler


def git_ozeti() -> str:
    try:
        ozet = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=KOK, capture_output=True, text=True, check=True
        ).stdout.strip()
        kirli = subprocess.run(
            ["git", "status", "--porcelain"], cwd=KOK, capture_output=True, text=True, check=True
        ).stdout.strip()
        return ozet + (" (çalışma ağacı kirli)" if kirli else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "git deposu yok"


def sigma_farki(deger: float, sapma: float, yayim: dict[str, object]) -> float:
    return (deger - float(yayim["deger"])) / float(np.hypot(sapma, float(yayim["belirsizlik"])))  # type: ignore[arg-type]


def main() -> None:
    ayristirici = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ayristirici.add_argument("--adim", type=int, default=1500)
    ayristirici.add_argument("--tohum", type=int, default=20261125)
    secenek = ayristirici.parse_args()
    basla = time.time()
    uretec = np.random.default_rng(secenek.tohum)

    ham = pantheon.yukle(ONBELLEK)
    asimetri = float(np.abs(ham.kovaryans - ham.kovaryans.T).max())
    kov = 0.5 * (ham.kovaryans + ham.kovaryans.T)

    yalniz_sn = veri_hazirla(ham.sutunlar, kov, kalibratorler=False)
    a1 = omega_m_izgara(yalniz_sn, np.linspace(0.15, 0.60, 181))
    print(
        f"Analiz 1 (yalnızca SN, N={len(yalniz_sn)}): Ωm = {a1['ortalama']:.3f} ± {a1['standart_sapma']:.3f}"
        f"   χ²/sd = {a1['ki_kare_min']:.1f}/{a1['serbestlik_derecesi']}"
    )

    sefeidli = veri_hazirla(ham.sutunlar, kov, kalibratorler=True)
    a2, ornekler = h0_mcmc(sefeidli, secenek.adim, uretec)
    h0 = a2["h0"]
    print(
        f"Analiz 2 (SN + Sefeid, N={len(sefeidli)}, kalibratör={int(sefeidli.kalibrator.sum())}): "
        f"H0 = {h0['ortalama']:.2f} ± {h0['standart_sapma']:.2f}   "  # type: ignore[index]
        f"Ωm = {a2['omega_m']['ortalama']:.3f} ± {a2['omega_m']['standart_sapma']:.3f}"
    )  # type: ignore[index]
    print(f"  tanılar: {a2['tanilar']}")

    kaynak_kaydi = json.loads((ONBELLEK / "KAYNAK_KAYDI.json").read_text(encoding="utf-8"))
    kayit = {
        "zaman": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "sure_s": round(time.time() - basla, 1),
        "kod": {"kozmos": kozmos.__version__, "git": git_ozeti()},
        "ortam": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "parametreler": {"adim": secenek.adim, "tohum": secenek.tohum, "z_alt": 0.01, "h0_nominal": H0_NOMINAL},
        "veri": {dosya: {"sha256": k["sha256"], "bayt": k["bayt"]} for dosya, k in kaynak_kaydi.items()},
        "kovaryans_en_buyuk_asimetri": asimetri,
        "analiz_1_yalniz_sn": {k: v for k, v in a1.items() if k not in ("izgara", "yogunluk")},
        "analiz_2_sn_ve_sefeid": a2,
        "yayimlanan": YAYIMLANAN,
        "karsilastirma_sigma": {
            "omega_m": sigma_farki(a1["ortalama"], a1["standart_sapma"], YAYIMLANAN["omega_m_duz_lcdm"]),  # type: ignore[arg-type]
            "h0": sigma_farki(h0["ortalama"], h0["standart_sapma"], YAYIMLANAN["h0_duz_lcdm"]),  # type: ignore[index]
        },
    }
    (CIKTI / "sonuc.json").write_text(json.dumps(kayit, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(CIKTI / "zincir.npy", ornekler.astype(np.float32))
    np.savez(CIKTI / "omega_m_sonsal.npz", izgara=np.array(a1["izgara"]), yogunluk=np.array(a1["yogunluk"]))
    print(f"kayıt: {CIKTI / 'sonuc.json'}  ({kayit['sure_s']} s)")
    print(
        f"yayımlananla fark: Ωm {kayit['karsilastirma_sigma']['omega_m']:+.2f}σ, H0 {kayit['karsilastirma_sigma']['h0']:+.2f}σ"
    )


if __name__ == "__main__":
    main()
