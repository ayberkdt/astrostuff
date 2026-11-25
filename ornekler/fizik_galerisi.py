"""Fizik modüllerinden dört şekil: kara delik, politroplar, kütle fonksiyonları, Jeans kütlesi.

Her şekil ``kozmos`` fonksiyonlarının doğrudan çıktısıdır; ek veri kullanılmaz.

Kullanım:
    python ornekler/fizik_galerisi.py
"""

from __future__ import annotations

from pathlib import Path

import _cizim as dw
import numpy as np

from kozmos.karadelik import temel as kd
from kozmos.temel.sabitler import CEKIM, GM_GUNES, PROTON_KUTLESI
from kozmos.yildiz import imf, yapi
from kozmos.yildizolusumu import bulut

KOK = Path(__file__).resolve().parents[1]
CIKTI = KOK / "belgeler"
GUNES_KUTLESI_KG = GM_GUNES.deger / CEKIM.deger  # nominal GM / G
VURGU, GRI = "#c2410c", "#6b7280"


def kara_delik() -> None:
    spin = np.linspace(0.0, 0.998, 400)
    isco = kd.isco_yaricapi_kerr_rg(spin)
    ufuk = kd.ufuk_yaricapi_kerr_rg(spin)
    verim = 100.0 * kd.isco_isinim_verimliligi(spin)
    fig, eksenler = dw.figur("web", hedef="web-640", oran=0.48, ondalik=",", mozaik=[["yaricap", "verim"]])
    ax = eksenler["yaricap"]
    ax.plot(spin, isco, color=VURGU, lw=2.0, label="En içteki kararlı yörünge")
    ax.plot(spin, ufuk, color=GRI, lw=1.6, ls="--", label="Olay ufku")
    ax.set_xlabel("Dönme parametresi a*")
    ax.set_ylabel("Yarıçap (GM/c²)")
    ax.set_ylim(0, 6.6)
    ax.text(0.42, 5.6, "En içteki kararlı yörünge", color=VURGU)
    ax.text(0.04, 1.05, "Olay ufku", color="#4b5563")
    ax = eksenler["verim"]
    ax.plot(spin, verim, color="#111827", lw=2.0)
    ax.set_xlabel("Dönme parametresi a*")
    ax.set_ylabel("Yığılma verimi (%)")
    ax.set_ylim(0, 35)
    ax.annotate(
        f"a* = 0: %{verim[0]:.1f}".replace(".", ","),
        (0.0, verim[0]),
        xytext=(0.06, 11.5),
        color="#4b5563",
        arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=0.8),
    )
    ax.annotate(
        f"a* = 0,998: %{verim[-1]:.0f}",
        (0.998, verim[-1]),
        xytext=(0.36, 29.0),
        color="#4b5563",
        arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=0.8),
    )
    dw.kaydet(fig, str(CIKTI / "kara_delik_donme"), hedef="web-640", bicimler=("png",))
    print(f"kara delik: ISCO {isco[0]:.2f} → {isco[-1]:.2f} r_g; verim %{verim[0]:.2f} → %{verim[-1]:.1f}")


def lane_emden() -> None:
    fig, ax = dw.figur("web", hedef="web-640", oran=0.62, ondalik=",")
    for n in (0.0, 1.0, 1.5, 3.0):
        xi, teta, _ = yapi.lane_emden_coz(n)
        (c,) = ax.plot(xi, teta, lw=1.9, label=f"n = {n:g}".replace(".", ","))
        ax.plot(xi[-1], 0.0, "o", ms=3.5, color=c.get_color())
    ax.set_xlabel("Boyutsuz yarıçap ξ")
    ax.set_ylabel("Boyutsuz sıcaklık θ (yoğunluk = merkez yoğunluğu × θⁿ)")
    ax.set_xlim(0, 7.6)
    ax.set_ylim(0, 1.03)
    dw.lejant.yerlestir(ax)
    dw.kaydet(fig, str(CIKTI / "lane_emden"), hedef="web-640", bicimler=("png",))
    print(
        "Lane-Emden ilk sıfırlar:",
        {n: round(yapi.lane_emden_ilk_sifir(n), 4) for n in (0, 1, 1.5, 3)},
        "merkez/ortalama yoğunluk n=3:",
        round(yapi.lane_emden_merkez_ortalama_yogunluk_orani(3.0), 2),
    )


def kutle_fonksiyonlari() -> None:
    m = np.logspace(-1, np.log10(50.0), 300)
    fig, ax = dw.figur("web", hedef="web-640", oran=0.62, ondalik=",")
    seriler = [
        ("Salpeter (1955)", imf.salpeter_dn_dm(m, 0.1, 50.0)),
        ("Kroupa (2001)", imf.kroupa_dn_dm(m, 0.1, 50.0)),
        ("Chabrier (2003)", imf.chabrier_dn_dm(m, 0.1, 50.0)),
    ]
    for ad, dn in seriler:
        ax.plot(m, m * dn, lw=1.9, label=ad)  # m dN/dm: logaritmik kütle aralığı başına sayı
    ax.set_xscale("log")  # datawiz: izin kütle fonksiyonunun alışılmış ölçeği; tikler düz sayı
    ax.set_yscale("log")
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_xticklabels(["0,1", "0,3", "1", "3", "10", "30"])
    ax.minorticks_off()
    ax.set_xlabel("Yıldız kütlesi (Güneş kütlesi)")
    ax.set_ylabel("Logaritmik kütle aralığı başına yıldız sayısı, m dN/dm")
    dw.lejant.yerlestir(ax)
    dw.kaydet(fig, str(CIKTI / "kutle_fonksiyonlari"), hedef="web-640", bicimler=("png",))
    print(
        "ortalama kütleler (0,1-50 M☉):",
        {ad: round(float(np.trapezoid(m * dn, m) / np.trapezoid(dn, m)), 3) for ad, dn in seriler},
    )


def jeans() -> None:
    n_cm3 = np.logspace(1, 7, 200)  # hidrojen molekülü sayı yoğunluğu [cm^-3]
    rho = bulut.kutle_yogunlugu_kg_m3(n_cm3 * 1e6)
    fig, ax = dw.figur("web", hedef="web-640", oran=0.62, ondalik=",")
    for sicaklik in (10.0, 20.0, 50.0):
        kutle = bulut.jeans_kutlesi_kg(bulut.ses_hizi_m_s(sicaklik), rho) / GUNES_KUTLESI_KG
        ax.plot(n_cm3, kutle, lw=1.9, label=f"{sicaklik:g} K")
    ax.set_xscale("log")  # datawiz: izin yoğunluk altı mertebe tarıyor
    ax.set_yscale("log")
    ax.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ax.set_xticklabels(["10", "10²", "10³", "10⁴", "10⁵", "10⁶", "10⁷"])
    ax.set_yticks([0.1, 1, 10, 100, 1000])
    ax.set_yticklabels(["0,1", "1", "10", "100", "1000"])
    ax.minorticks_off()
    ax.set_xlabel("Gaz parçacık yoğunluğu (cm⁻³)")
    ax.set_ylabel("Jeans kütlesi (Güneş kütlesi)")
    dw.etiket.cizgi_sonu(ax)
    dw.kaydet(fig, str(CIKTI / "jeans_kutlesi"), hedef="web-640", bicimler=("png",))
    ornek = bulut.jeans_kutlesi_kg(bulut.ses_hizi_m_s(10.0), bulut.kutle_yogunlugu_kg_m3(1e4 * 1e6)) / GUNES_KUTLESI_KG
    print(f"Jeans kütlesi, 10 K ve 10^4 cm^-3: {float(ornek):.2f} M☉; proton kütlesi {PROTON_KUTLESI.deger:.3e} kg")


if __name__ == "__main__":
    CIKTI.mkdir(exist_ok=True)
    kara_delik()
    lane_emden()
    kutle_fonksiyonlari()
    jeans()
