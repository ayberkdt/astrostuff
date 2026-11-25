"""H0 belirsizliğinin zincirden bağımsız, kapalı biçimli denetimi.

Soru: bu depodaki zincir H0 = 73,53 ± 1,00 verirken yayımlanan değer neden
73,6 ± 1,1 [Brout2022]?

Yöntem: Ωm sabitken model, M ve x = 5 log10(H0 / 70) parametrelerinde
doğrusaldır. Hubble akışındaki süpernovalarda artık m − M − μ(z; 70, Ωm) + x,
kalibratörlerde m − M − μ_Sefeid olur. Doğrusal Gauss modelinde parametre
kovaryansı (Aᵀ C⁻¹ A)⁻¹'dir; örnekleyiciye, zincir uzunluğuna ve yakınsamaya
bağlı değildir. Ωm'in katkısı, Ωm ızgarasındaki en iyi x'in saçılımından
toplam varyans yasasıyla eklenir.

İkinci denetim yayımlanan zincirin kendisidir: Pantheon+ veri sürümündeki
dört ``Pantheon+SH0ES_*.txt.gz`` zinciri (emcee) indirilir ve H0 saçılımı
zincirin baş kısmı atılarak ve atılmadan ölçülür.

Kullanım:
    python projeler/hubble/h0_belirsizlik_denetimi.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from scipy import linalg

from kozmos.arsiv import onbellek, pantheon
from kozmos.kozmoloji.uzaklik import Kozmoloji, enine_esyuruyen_uzaklik_mpc, uzaklik_modulu

KOK = Path(__file__).resolve().parents[2]
BURASI = Path(__file__).resolve().parent
H0_NOMINAL = 70.0


def cozum(
    s: dict, kovaryans: np.ndarray, z_alt: float, z_ust: float = np.inf, zhel_ile: bool = False
) -> dict[str, float]:
    """Kapalı biçimli çözüm. ``zhel_ile``: veri sürümündeki CosmoSIS olabilirliğinin uzaklık tanımı,
    d_L = (1 + z_HEL) D_M(z_HD); yanlışsa d_L = (1 + z_HD) D_M(z_HD)."""
    z = s["zHD"]
    kal = s["IS_CALIBRATOR"] == 1
    secim = kal | ((z > z_alt) & (z < z_ust))
    z, kal = z[secim], kal[secim]
    m = s["m_b_corr"][secim]
    carpan = linalg.cho_factor(kovaryans[np.ix_(secim, secim)], lower=True)
    tasarim = np.column_stack([np.ones(z.size), np.where(kal, 0.0, -1.0)])  # sütunlar: M, x
    ters_kov = np.linalg.inv(tasarim.T @ linalg.cho_solve(carpan, tasarim))

    def uyum(omega_m: float) -> tuple[float, float]:
        kozmoloji = Kozmoloji.duz(H0_NOMINAL, omega_m)
        if zhel_ile:
            model = 5.0 * np.log10((1.0 + s["zHEL"][secim]) * enine_esyuruyen_uzaklik_mpc(z, kozmoloji)) + 25.0
        else:
            model = uzaklik_modulu(z, kozmoloji)
        mu = np.where(kal, s["CEPH_DIST"][secim], model)
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


ZINCIR_KOKU = (
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/5_COSMOLOGY/chains/"
    "Pantheon%2BSH0ES/Pantheon%2BSH0ES_"
)
# Makaledeki değerler: Brout ve ark. (2022) TeX makroları \fLCDMHPanSH, \oLCDMHPanSH, \HPanSH, \wwaHPanSH.
YAYIMLANAN = {
    "FlatLambdaCDM": (73.6, 1.1),
    "LambdaCDM": (73.4, 1.1),
    "FlatwCDM": (73.5, 1.1),
    "Flatw0waCDM": (73.3, 1.1),
}


def yayimlanan_zincirler() -> dict[str, object]:
    """Veri sürümündeki dört zincirin H0 ortalaması ve saçılımı; baş kısım atılarak ve atılmadan."""
    sonuc = {}
    for model, (h0_makale, sigma_makale) in YAYIMLANAN.items():
        yol = onbellek.indir(
            f"{ZINCIR_KOKU}{model}.txt.gz",
            KOK / "veri" / "ham" / "pantheon" / f"Pantheon+SH0ES_{model}.txt.gz",
            atif="Pantheon+SH0ES veri sürümü, 5_COSMOLOGY/chains",
        )
        with gzip.open(yol, "rt") as dosya:
            basliklar = dosya.readline().lstrip("#").split()
            ornekleyici = dosya.readline().strip()
        zincir = np.loadtxt(yol, comments="#")
        h0 = 100.0 * zincir[:, basliklar.index("cosmological_parameters--h0")]
        n = h0.size
        kesitler = {}
        for ad, bas in (
            ("tamami", 0.0),
            ("ilk_yuzde_10_atilmis", 0.10),
            ("ilk_yuzde_25_atilmis", 0.25),
            ("son_yari", 0.5),
        ):
            s = h0[int(bas * n) :]
            kesitler[ad] = {"ornek": int(s.size), "h0": float(s.mean()), "sigma_h0": float(s.std())}
        sonuc[model] = {
            "makale": {"h0": h0_makale, "sigma_h0": sigma_makale},
            "ornekleyici_satiri": ornekleyici,
            "ornek_sayisi": int(n),
            "kesitler": kesitler,
        }
    return sonuc


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
            "stat+sys, z > 0.01, yayımlanan olabilirliğin uzaklık tanımı (zHEL)": cozum(
                s, ham.kovaryans, 0.01, zhel_ile=True
            ),
            "stat+sys, kesim yok (bütün süpernovalar)": cozum(s, ham.kovaryans, -1.0),
            "stat+sys, z > 0.0233": cozum(s, ham.kovaryans, 0.0233),
            "stat+sys, 0.0233 < z < 0.15": cozum(s, ham.kovaryans, 0.0233, 0.15),
            "yalnız istatistik, z > 0.01": cozum(s, yalniz_istatistik, 0.01),
        },
    }
    sonuc["yayimlanan_zincirler"] = yayimlanan_zincirler()
    (BURASI / "h0_belirsizlik_denetimi.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    for ad, c in sonuc["secenekler"].items():
        print(
            f"{ad:68s} N={c['sn_sayisi']:5d} H0={c['h0']:.2f} ± {c['sigma_h0']:.3f} "
            f"(Ωm sabit: {c['sigma_h0_omega_m_sabit']:.3f})  Ωm={c['omega_m']:.3f}"
        )
    print("zincir:", sonuc["zincir"]["ortalama"], sonuc["zincir"]["standart_sapma"])
    for model, z in sonuc["yayimlanan_zincirler"].items():
        tam, kirpik = z["kesitler"]["tamami"], z["kesitler"]["ilk_yuzde_10_atilmis"]
        print(
            f"{model:14s} {z['ornekleyici_satiri']} makale {z['makale']['h0']} ± {z['makale']['sigma_h0']} | zincir tamamı "
            f"{tam['h0']:.2f} ± {tam['sigma_h0']:.3f} | ilk %10 atılmış {kirpik['h0']:.2f} ± {kirpik['sigma_h0']:.3f}"
        )


if __name__ == "__main__":
    main()
