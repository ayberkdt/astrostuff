"""H0 belirsizliğinin zincirden bağımsız, kapalı biçimli denetimi.

Soru: bu depodaki zincir H0 = 73,53 ± 1,00 verirken yayımlanan değer neden
73,6 ± 1,1 [Brout2022]?

Yöntem: Ωm sabitken model, M ve x = 5 log10(H0 / 70) parametrelerinde
doğrusaldır. Hubble akışındaki süpernovalarda artık m − M − μ(z; 70, Ωm) + x,
kalibratörlerde m − M − μ_Sefeid olur. Doğrusal Gauss modelinde parametre
kovaryansı (Aᵀ C⁻¹ A)⁻¹'dir; örnekleyiciye, zincir uzunluğuna ve yakınsamaya
bağlı değildir. Ωm'in katkısı, Ωm ızgarasındaki en iyi x'in saçılımından
toplam varyans yasasıyla eklenir.

Kullanım:
    python projeler/hubble/h0_belirsizlik_denetimi.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import linalg

from kozmos.arsiv import pantheon
from kozmos.kozmoloji.uzaklik import Kozmoloji, uzaklik_modulu

KOK = Path(__file__).resolve().parents[2]
BURASI = Path(__file__).resolve().parent
H0_NOMINAL = 70.0


def cozum(s: dict, kovaryans: np.ndarray, z_alt: float, z_ust: float = np.inf) -> dict[str, float]:
    z = s["zHD"]
    kal = s["IS_CALIBRATOR"] == 1
    secim = kal | ((z > z_alt) & (z < z_ust))
    z, kal = z[secim], kal[secim]
    m = s["m_b_corr"][secim]
    carpan = linalg.cho_factor(kovaryans[np.ix_(secim, secim)], lower=True)
    tasarim = np.column_stack([np.ones(z.size), np.where(kal, 0.0, -1.0)])  # sütunlar: M, x
    ters_kov = np.linalg.inv(tasarim.T @ linalg.cho_solve(carpan, tasarim))

    def uyum(omega_m: float) -> tuple[float, float]:
        mu = np.where(kal, s["CEPH_DIST"][secim], uzaklik_modulu(z, Kozmoloji.duz(H0_NOMINAL, omega_m)))
        p = ters_kov @ (tasarim.T @ linalg.cho_solve(carpan, m - mu))
        artik = m - mu - tasarim @ p
        return float(p[1]), float(artik @ linalg.cho_solve(carpan, artik))

    izgara = np.linspace(0.20, 0.50, 121)
    x, ki = np.array([uyum(o) for o in izgara]).T
    agirlik = np.exp(-0.5 * (ki - ki.min()))
    agirlik /= agirlik.sum()
    x_ort = float(agirlik @ x)
    # Toplam varyans: Ωm sabitkenki varyans + Ωm sonsalı boyunca en iyi x'in varyansı.
    var_x = float(ters_kov[1, 1] + agirlik @ (x - x_ort) ** 2)
    h0 = H0_NOMINAL * 10 ** (x_ort / 5.0)
    donusum = h0 * np.log(10.0) / 5.0
    return {
        "sn_sayisi": int(secim.sum()),
        "kalibrator": int(kal.sum()),
        "h0": h0,
        "sigma_h0_omega_m_sabit": float(donusum * np.sqrt(ters_kov[1, 1])),
        "sigma_h0": float(donusum * np.sqrt(var_x)),
        "omega_m": float(agirlik @ izgara),
    }


def main() -> None:
    ham = pantheon.yukle(KOK / "veri" / "ham" / "pantheon")
    s = ham.sutunlar
    yalniz_istatistik = pantheon.yukle(KOK / "veri" / "ham" / "pantheon", yalniz_istatistik=True).kovaryans
    sonuc = {
        "aciklama": "kapalı biçimli doğrusal Gauss çözümü; örnekleyici yok",
        "yayimlanan": {"h0": 73.6, "sigma": 1.1, "kaynak": "Brout2022 TeX makrosu \\fLCDMHPanSH"},
        "zincir": json.loads((BURASI / "sonuc.json").read_text(encoding="utf-8"))["analiz_2_sn_ve_sefeid"]["h0"],
        "secenekler": {
            "stat+sys, z > 0.01 (zincirle aynı seçim)": cozum(s, ham.kovaryans, 0.01),
            "stat+sys, z > 0.0233": cozum(s, ham.kovaryans, 0.0233),
            "stat+sys, 0.0233 < z < 0.15": cozum(s, ham.kovaryans, 0.0233, 0.15),
            "yalnız istatistik, z > 0.01": cozum(s, yalniz_istatistik, 0.01),
        },
    }
    (BURASI / "h0_belirsizlik_denetimi.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    for ad, c in sonuc["secenekler"].items():
        print(
            f"{ad:42s} N={c['sn_sayisi']:5d} H0={c['h0']:.2f} ± {c['sigma_h0']:.3f} "
            f"(Ωm sabit: {c['sigma_h0_omega_m_sabit']:.3f})  Ωm={c['omega_m']:.3f}"
        )
    print("zincir:", sonuc["zincir"]["ortalama"], sonuc["zincir"]["standart_sapma"])


if __name__ == "__main__":
    main()
