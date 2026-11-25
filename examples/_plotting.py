"""Shared plotting layer of the figure scripts.

The figures in this repository were produced with ``datawiz``, a local figure
auditing tool that is not part of the repository (it measures font sizes,
contrast and overlaps). If the tool is installed the calls go to it; if not,
plain matplotlib equivalents with the same names are used. The plain
matplotlib output shows the same data but is not audited and looks different
from the figures under ``figures/``.

Usage in the scripts: ``import _plotting``.
"""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import datawiz as _datawiz
except ImportError:
    _datawiz = None

HAS_DATAWIZ = _datawiz is not None
_WIDTH_IN = 6.4  # plain matplotlib counterpart of the "web-640" target


def figure(medium: str = "web", *, target: str = "web-640", ratio: float = 0.62, mosaic=None, decimal=None, **options):
    """Return a figure and its axes: (fig, dict) when ``mosaic`` is given, else (fig, axes).

    ``ratio`` is height over width. ``decimal`` sets the decimal mark of tick labels (datawiz only).
    """
    if HAS_DATAWIZ:
        # datawiz has a Turkish keyword interface: hedef = target, oran = ratio, mozaik = mosaic, ondalik = decimal.
        if mosaic is not None:
            options["mozaik"] = mosaic
        if decimal is not None:
            options["ondalik"] = decimal
        return _datawiz.figur(medium, hedef=target, oran=ratio, **options)
    size = (_WIDTH_IN, _WIDTH_IN * ratio)
    if mosaic is not None:
        return plt.subplot_mosaic(mosaic, figsize=size, constrained_layout=True, **options)
    return plt.subplots(figsize=size, constrained_layout=True, **options)


def save(fig, path: str, *, target: str = "web-640", formats=("pdf", "png"), **options):
    """Save the figure; return the datawiz audit report, or ``None`` without datawiz."""
    if HAS_DATAWIZ:
        return _datawiz.kaydet(fig, path, hedef=target, bicimler=formats, **options)
    for fmt in formats:
        fig.savefig(f"{path}.{fmt}", dpi=200)
    print(f"datawiz is not installed: {path} saved with plain matplotlib, unaudited")
    return None


def _place_legend(ax, **options):
    return _datawiz.lejant.yerlestir(ax, **options) if HAS_DATAWIZ else ax.legend(frameon=False)


def _line_end_labels(ax, **options):
    return _datawiz.etiket.cizgi_sonu(ax, **options) if HAS_DATAWIZ else ax.legend(frameon=False)


legend = SimpleNamespace(place=_place_legend)
labels = SimpleNamespace(line_end=_line_end_labels)
