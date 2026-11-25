"""Pantheon+SH0ES verisinden Ωm ve H0 kestirimi.

İki analiz yapar ve sonucu yayımlanan değerlerle karşılaştırır [Brout2022]:

1. Yalnızca süpernovalar, düz ΛCDM: Ωm. Mutlak kadir M (H0 ile dejenere)
   düz önselle analitik olarak marjinalleştirilir; Ωm sonsalı ızgarada
   hesaplanır.
2. Sefeid kalibratörleri eklenir: (H0, Ωm, M) için kendi topluluk
   örnekleyicimizle MCMC.

Kullanım:
    python projects/hubble/hubble_analysis.py [--adim 1500] [--tohum 20261125]

Çıktı: ``projects/hubble/result.json`` (çalıştırma kaydı ve sonuçlar) ve
``projects/hubble/zincir.npy``.
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
from kozmos.archives import pantheon
from kozmos.cosmology.distances import Cosmology
from kozmos.cosmology.hubble_diagram import HubbleData, best_absolute_magnitude, log_likelihood, prepare_data
from kozmos.statistics.mcmc import autocorrelation_time, ensemble_sample, rank_normalized_rhat, split_rhat

ROOT = Path(__file__).resolve().parents[2]
ONBELLEK = ROOT / "data" / "raw" / "pantheon"
CIKTI = Path(__file__).resolve().parent

# Yayımlanan değerler: Brout ve ark. 2022, arXiv:2202.04077 TeX kaynağı (main.tex,
# \fLCDMOMPan ve \fLCDMHPanSH makroları), 25 Kasım 2026'da okundu.
YAYIMLANAN = {
    "omega_m_duz_lcdm": {"deger": 0.334, "belirsizlik": 0.018, "kaynak": "Brout2022, düz ΛCDM, yalnızca SN"},
    "h0_duz_lcdm": {"deger": 73.6, "belirsizlik": 1.1, "kaynak": "Brout2022, düz ΛCDM, Pantheon+ ve SH0ES"},
}
H0_NOMINAL = 70.0  # Analiz 1'de M ile dejenere; sonucu etkilemez.


def omega_m_izgara(data: HubbleData, izgara: np.ndarray) -> dict[str, float | list[float]]:
    """Analiz 1: M marjinalleştirilmiş Ωm sonsalı (düz önsel, 0 < Ωm < 1)."""
    ki2 = np.array([best_absolute_magnitude(data, Cosmology.flat(H0_NOMINAL, om))[2] for om in izgara])
    density = np.exp(-0.5 * (ki2 - ki2.min()))
    density /= np.trapezoid(density, izgara)
    ortalama = float(np.trapezoid(izgara * density, izgara))
    sapma = float(np.sqrt(np.trapezoid((izgara - ortalama) ** 2 * density, izgara)))
    return {
        "ortalama": ortalama,
        "standart_sapma": sapma,
        "en_olasi": float(izgara[np.argmin(ki2)]),
        "ki_kare_min": float(ki2.min()),
        "serbestlik_derecesi": len(data) - 2,
        "izgara": izgara.tolist(),
        "yogunluk": density.tolist(),
    }


def h0_mcmc(data: HubbleData, adim: int, rng: np.random.Generator) -> tuple[dict[str, object], np.ndarray]:
    """Analiz 2: (H0, Ωm, M) için topluluk örnekleyici."""

    def log_sonsal(p: np.ndarray) -> float:
        h0, om, m = p
        if not (50.0 < h0 < 100.0 and 0.05 < om < 0.95 and -21.0 < m < -18.0):
            return -np.inf
        return log_likelihood(data, Cosmology.flat(h0, om), m)

    yuruyucu = 32
    initial = np.array([73.0, 0.33, -19.25]) + rng.normal(0, [0.5, 0.01, 0.01], (yuruyucu, 3))
    chain = ensemble_sample(log_sonsal, initial, adim, rng)
    tau = autocorrelation_time(chain.samples)
    discard = int(min(adim // 2, max(100, 5 * tau.max())))
    kalan = chain.samples[discard:]
    flat = kalan.reshape(-1, 3)
    adlar = ["h0", "omega_m", "mutlak_kadir"]
    sonuc: dict[str, object] = {
        name: {
            "ortalama": float(flat[:, i].mean()),
            "standart_sapma": float(flat[:, i].std(ddof=1)),
            "yuzdelikler_16_50_84": [float(q) for q in np.percentile(flat[:, i], [16, 50, 84])],
        }
        for i, name in enumerate(adlar)
    }
    sonuc["tanilar"] = {
        "adim": adim,
        "yuruyucu": yuruyucu,
        "atilan_isinma": discard,
        "oz_ilinti_suresi": tau.tolist(),
        "etkin_ornek_sayisi": (kalan.shape[0] * yuruyucu / tau).tolist(),
        "bolunmus_rhat": split_rhat(kalan).tolist(),
        "rhat_sira_normallestirilmis": rank_normalized_rhat(kalan).tolist(),
        "ortalama_kabul_orani": float(chain.acceptance_fraction.mean()),
        "zincir_yeterince_uzun": bool(adim > 50 * tau.max()),
    }
    en_iyi = flat[np.argmax(chain.log_probability[discard:].reshape(-1))]
    sonuc["ki_kare_min"] = float(-2 * chain.log_probability[discard:].max())
    sonuc["serbestlik_derecesi"] = len(data) - 3
    sonuc["en_olasi"] = dict(zip(adlar, map(float, en_iyi), strict=True))
    return sonuc, chain.samples


def git_ozeti() -> str:
    try:
        ozet = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        kirli = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        return ozet + (" (çalışma ağacı kirli)" if kirli else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "git deposu yok"


def sigma_farki(value: float, sapma: float, yayim: dict[str, object]) -> float:
    return (value - float(yayim["deger"])) / float(np.hypot(sapma, float(yayim["belirsizlik"])))  # type: ignore[arg-type]


def main() -> None:
    ayristirici = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ayristirici.add_argument("--adim", type=int, default=1500)
    ayristirici.add_argument("--tohum", type=int, default=20261125)
    secenek = ayristirici.parse_args()
    basla = time.time()
    rng = np.random.default_rng(secenek.tohum)

    raw = pantheon.load(ONBELLEK)
    asimetri = float(np.abs(raw.covariance - raw.covariance.T).max())
    kov = 0.5 * (raw.covariance + raw.covariance.T)

    yalniz_sn = prepare_data(raw.columns, kov, calibrators=False)
    a1 = omega_m_izgara(yalniz_sn, np.linspace(0.15, 0.60, 181))
    print(
        f"Analiz 1 (yalnızca SN, N={len(yalniz_sn)}): Ωm = {a1['ortalama']:.3f} ± {a1['standart_sapma']:.3f}"
        f"   χ²/sd = {a1['ki_kare_min']:.1f}/{a1['serbestlik_derecesi']}"
    )

    sefeidli = prepare_data(raw.columns, kov, calibrators=True)
    a2, samples = h0_mcmc(sefeidli, secenek.adim, rng)
    h0 = a2["h0"]
    print(
        f"Analiz 2 (SN + Sefeid, N={len(sefeidli)}, kalibratör={int(sefeidli.calibrator.sum())}): "
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
        "data": {dosya: {"sha256": k["sha256"], "bayt": k["bayt"]} for dosya, k in kaynak_kaydi.items()},
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
    np.save(CIKTI / "zincir.npy", samples.astype(np.float32))
    np.savez(CIKTI / "omega_m_sonsal.npz", izgara=np.array(a1["izgara"]), density=np.array(a1["yogunluk"]))
    print(f"kayıt: {CIKTI / 'sonuc.json'}  ({kayit['sure_s']} s)")
    print(
        f"yayımlananla fark: Ωm {kayit['karsilastirma_sigma']['omega_m']:+.2f}σ, H0 {kayit['karsilastirma_sigma']['h0']:+.2f}σ"
    )


if __name__ == "__main__":
    main()
