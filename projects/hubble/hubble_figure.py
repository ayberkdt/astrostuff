"""README için Hubble diyagramı: Pantheon+ uzaklık modülleri ve en iyi uyum eğrisi.

Girdi: ``sonuc.json`` (hubble_analizi.py çıktısı) ve yerel önbellekteki veri.
Çıktı: ``figures/hubble_diagram.png`` (+ PDF ve denetim raporu).

Şekildeki hata çubukları kovaryans köşegeninden gelir ve yalnızca görsel
amaçlıdır; uyum tam kovaryansla yapılmıştır (veri deposu README uyarısı).

Kullanım:
    python projects/hubble/hubble_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from kozmos.archives import pantheon
from kozmos.cosmology.distances import Cosmology, distance_modulus

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
import _plotting

SONUC = json.loads((Path(__file__).resolve().parent / "sonuc.json").read_text(encoding="utf-8"))


def main() -> None:
    raw = pantheon.load(ROOT / "data" / "raw" / "pantheon")
    s = raw.columns
    akis = (s["zHD"] > 0.01) & (s["IS_CALIBRATOR"] == 0)
    z = s["zHD"][akis]
    a2 = SONUC["analiz_2_sn_ve_sefeid"]
    h0, om, m = a2["h0"]["ortalama"], a2["omega_m"]["ortalama"], a2["mutlak_kadir"]["ortalama"]
    cosmology = Cosmology.flat(h0, om)
    mu = s["m_b_corr"][akis] - m  # bu analizin bulduğu mutlak kadirle uzaklık modülü
    residual = mu - distance_modulus(z, cosmology)

    fig, eksenler = _plotting.figure(
        "web", target="web-640", ratio=0.82, decimal=",", mosaic=[["ust"], ["ust"], ["alt"]], sharex=True
    )
    high, low = eksenler["ust"], eksenler["alt"]
    nokta = dict(s=5, alpha=0.35, linewidths=0, color="#6b7280", rasterized=True)

    high.scatter(z, mu, **nokta)
    zz = np.logspace(-2, np.log10(2.4), 300)
    (egri,) = high.plot(zz, distance_modulus(zz, cosmology), color="#c2410c", lw=1.8)
    high.set_xscale("log")  # datawiz: izin Hubble diyagramının alışılmış ölçeği; tikler düz sayı yazılır
    high.set_ylabel("Uzaklık modülü (mag)")
    high.annotate(
        f"Düz ΛCDM uyumu\n$H_0$ = {h0:.1f}, $Ω_m$ = {om:.2f}".replace(".", ","),
        xy=(0.3, float(distance_modulus(0.3, cosmology))),
        xytext=(0.035, 43.3),
        color="#c2410c",
        arrowprops=dict(arrowstyle="-", color="#c2410c", lw=0.8, shrinkB=3),
    )
    high.text(0.25, 35.4, f"{akis.sum()} Tür Ia süpernova\n(Pantheon+, z > 0,01)", color="#4b5563", va="top")

    low.scatter(z, residual, **nokta)
    low.axhline(0.0, color="#c2410c", lw=1.4)
    # Kutulanmış ortanca artık: eğilim var mı?
    kenar = np.logspace(-2, np.log10(2.3), 13)
    orta = np.sqrt(kenar[:-1] * kenar[1:])
    kutu = np.digitize(z, kenar) - 1
    ortanca = np.array([np.median(residual[kutu == i]) if np.sum(kutu == i) >= 5 else np.nan for i in range(orta.size)])
    low.plot(orta, ortanca, "o", ms=4.5, color="#111827", zorder=5)
    low.set_ylim(-0.75, 0.75)
    low.set_ylabel("Artık (mag)")
    low.set_xlabel("Kırmızıya kayma z")
    low.text(0.012, 0.66, "siyah: kutu ortancası", color="#111827", va="top")
    low.set_xlim(0.0095, 2.6)
    tikler = [0.01, 0.03, 0.1, 0.3, 1.0, 2.0]
    low.set_xticks(tikler)
    low.set_xticklabels(["0,01", "0,03", "0,1", "0,3", "1", "2"])
    low.minorticks_off()

    cikti = ROOT / "figures"
    cikti.mkdir(exist_ok=True)
    rapor = _plotting.save(fig, str(cikti / "hubble_diyagrami"), target="web-640", cvd=True)
    if rapor is not None:
        print({k: v for k, v in rapor.items() if k in ("hata", "uyari", "denetim", "dosyalar")})
    print(
        f"artık: ortanca {np.median(residual):+.4f} mag, sağlam saçılım {1.4826 * np.median(np.abs(residual - np.median(residual))):.3f} mag"
    )


if __name__ == "__main__":
    main()
