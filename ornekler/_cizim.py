"""Şekil betiklerinin ortak çizim katmanı.

Depodaki şekiller ``datawiz`` ile üretilmiştir; bu, depoda bulunmayan yerel bir
figür denetim aracıdır (punto, kontrast, çakışma ölçer). Araç kuruluysa çağrılar
ona gider; kurulu değilse aynı adlarla düz matplotlib karşılıkları çalışır.
Düz matplotlib çıktısı aynı veriyi gösterir ama denetlenmemiştir ve görünümü
``belgeler/`` altındaki şekillerden farklıdır.

Kullanım (betiklerde): ``import _cizim as dw``.
"""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import datawiz as _dw
except ImportError:
    _dw = None

DATAWIZ_VAR = _dw is not None
_GENISLIK_INC = 6.4  # "web-640" hedefinin düz matplotlib karşılığı


def figur(ortam: str = "web", *, hedef: str = "web-640", oran: float = 0.62, mozaik=None, **secenekler):
    """Figür ve eksen(ler)i döndürür: ``mozaik`` verilirse (fig, sözlük), yoksa (fig, eksen)."""
    if DATAWIZ_VAR:
        if mozaik is not None:
            secenekler["mozaik"] = mozaik
        return _dw.figur(ortam, hedef=hedef, oran=oran, **secenekler)
    secenekler.pop("ondalik", None)
    boyut = (_GENISLIK_INC, _GENISLIK_INC * oran)
    if mozaik is not None:
        return plt.subplot_mosaic(mozaik, figsize=boyut, constrained_layout=True, **secenekler)
    return plt.subplots(figsize=boyut, constrained_layout=True, **secenekler)


def kaydet(fig, yol: str, *, hedef: str = "web-640", bicimler=("pdf", "png"), **secenekler):
    """Figürü kaydeder; datawiz varsa denetim raporunu, yoksa ``None`` döndürür."""
    if DATAWIZ_VAR:
        return _dw.kaydet(fig, yol, hedef=hedef, bicimler=bicimler, **secenekler)
    for bicim in bicimler:
        fig.savefig(f"{yol}.{bicim}", dpi=200)
    print(f"datawiz kurulu değil: {yol} denetimsiz, düz matplotlib ile kaydedildi")
    return None


def _lejant_yerlestir(ax, **secenekler):
    return _dw.lejant.yerlestir(ax, **secenekler) if DATAWIZ_VAR else ax.legend(frameon=False)


def _cizgi_sonu(ax, **secenekler):
    return _dw.etiket.cizgi_sonu(ax, **secenekler) if DATAWIZ_VAR else ax.legend(frameon=False)


lejant = SimpleNamespace(yerlestir=_lejant_yerlestir)
etiket = SimpleNamespace(cizgi_sonu=_cizgi_sonu)
