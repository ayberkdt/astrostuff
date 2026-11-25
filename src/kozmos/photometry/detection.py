"""Source detection by thresholding and connected pixels, with optional deblending.

Detection (groups of connected pixels above a threshold) follows the definitions in the
photutils documentation [Photutils2026]. Overlapping sources are separated with a
multi-threshold tree, the saddle criterion and a priority-queue flood. Tested with synthetic
images whose truth is known.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from kozmos._sources import source

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SourceList:
    """Measurements of the detected sources.

    Attributes
    ----------
    x, y : ndarray
        Flux-weighted centers [pixel]; ``image[y, x]``.
    flux : ndarray
        Background-subtracted total flux in the segment.
    peak : ndarray
        Value of the brightest pixel.
    n_pixels : ndarray
    """

    x: Array
    y: Array
    flux: Array
    peak: Array
    n_pixels: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.x.size)


def _markers(segment_image: Array, mask: NDArray[np.bool_], levels: Array, flux_threshold: float) -> list:
    """Return the (y, x) positions of the peaks of a segment to be counted as separate sources (multi-threshold tree, saddle criterion)."""
    pending = [(mask, 0)]
    peaks = []
    while pending:
        m, i0 = pending.pop()
        split = False
        for i in range(i0, levels.size):
            parts, n = ndimage.label(m & (segment_image > levels[i]), structure=np.ones((3, 3)))
            if n == 0:
                break
            if n == 1:
                continue
            ids = np.arange(1, n + 1)
            # Flux remaining above the saddle level: significance of the branch's own peak.
            flux_above = ndimage.sum_labels(segment_image - levels[i], parts, ids)
            significant = ids[flux_above >= flux_threshold]
            if significant.size >= 2:
                pending.extend((parts == k, i + 1) for k in significant)
                split = True
                break
            if significant.size == 1:
                m = parts == significant[0]
            else:
                break
        if not split:
            peaks.append(np.unravel_index(np.argmax(np.where(m, segment_image, -np.inf)), segment_image.shape))
    return peaks


def _flood(segment_image: Array, mask: NDArray[np.bool_], peaks: list) -> NDArray[np.int32]:
    """Priority-queue flood starting from the markers: each pixel joins the basin that reaches it at the highest level.

    ``scipy.ndimage.watershed_ift`` counts the difference between neighboring pixels as the cost (it expects a
    gradient image) and crosses the saddle in a brightness image; it was therefore not used.
    """
    ny, nx = segment_image.shape
    basin = np.zeros(segment_image.shape, dtype=np.int32)
    queue = []
    for j, (y, x) in enumerate(peaks, start=1):
        basin[y, x] = j
        heapq.heappush(queue, (-float(segment_image[y, x]), j, int(y), int(x)))
    while queue:
        level, j, y, x = heapq.heappop(queue)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < ny and 0 <= xx < nx and mask[yy, xx] and basin[yy, xx] == 0:
                    basin[yy, xx] = j
                    heapq.heappush(queue, (max(level, -float(segment_image[yy, xx])), j, yy, xx))
    return basin


def _deblend(detection_image: Array, labels: NDArray[np.int32], n_levels: int, contrast: float, min_pixels: int):
    """Split segments with more than one significant peak into watersheds; return the new label map."""
    new_labels = labels.copy()
    next_label = int(labels.max()) + 1
    for label_id, slc in enumerate(ndimage.find_objects(labels), start=1):
        if slc is None:
            continue
        mask = labels[slc] == label_id
        segment_image = np.where(mask, detection_image[slc], 0.0)
        min_value, max_value = float(segment_image[mask].min()), float(segment_image[mask].max())
        if mask.sum() < 2 * min_pixels or min_value <= 0 or max_value <= min_value:
            continue
        levels = np.geomspace(min_value, max_value, n_levels + 2)[1:-1]
        peaks = _markers(segment_image, mask, levels, contrast * float(segment_image[mask].sum()))
        peaks.sort(key=lambda yx: -segment_image[yx])  # the brightest peak keeps the old label
        while len(peaks) >= 2:
            basin = _flood(segment_image, mask, peaks)
            size = np.bincount(basin[mask], minlength=len(peaks) + 1)[1:]
            if size.min() >= min_pixels:
                break
            del peaks[int(np.argmin(size))]  # a basin that fails the minimum-pixel condition joins its neighbor
        if len(peaks) < 2:
            continue
        for j in range(2, len(peaks) + 1):
            new_labels[slc][mask & (basin == j)] = next_label
            next_label += 1
    return new_labels


@source(
    "Photutils2026",
    where="User guide: Image Segmentation and Source Deblending",
    status="read",
    note="source: at least a given number of connected pixels, each above the threshold; the threshold is a "
    "multiple of the background noise; filtering before thresholding; the default connectivity is 8 neighbors. "
    "Deblending: multi-thresholding + watershed; the 'saddle' criterion is the ratio of the flux a peak holds "
    "above the saddle level where it separates from its neighbors to the total flux; the levels lie between the "
    "smallest and largest values of the source; a saddle point between sources is needed for them to separate. "
    "The geometric spacing of the levels and building the basins by priority-queue flooding are this project's "
    "choice. The paper describing the method [BertinArnouts1996] was not read",
)
def detect_sources(
    image: ArrayLike,
    background: ArrayLike | float,
    rms: ArrayLike | float,
    *,
    threshold_sigma: float = 3.0,
    min_pixels: int = 5,
    smoothing_fwhm: float | None = None,
    deblend_levels: int = 0,
    contrast: float = 0.005,
) -> SourceList:
    """Detect groups of connected pixels above the background as sources.

    Parameters
    ----------
    image : ndarray (Y, X)
    background, rms : ndarray or float
        Outputs of ``background_map``.
    threshold_sigma : float
        Detection threshold (multiple of the rms).
    min_pixels : int
        Minimum number of connected pixels of a source; rejects single-pixel noise peaks and cosmic rays.
    smoothing_fwhm : float, optional
        If given, the detection image is smoothed with a Gaussian of this width (matched-filter approximation);
        the threshold is scaled to the smoothed noise.
    deblend_levels : int
        If greater than zero, overlapping sources are deblended with this number of threshold levels (0: no deblending).
    contrast : float
        Smallest ratio of the flux of a peak above the saddle level to the total flux of the segment for the peak
        to count as a separate source.

    Returns
    -------
    SourceList
        In decreasing order of flux.

    References
    ----------
    [Photutils2026] Image Segmentation, Source Deblending.

    Validity
    --------
    With deblending off, overlapping sources count as one. With it on, pairs with no saddle
    point between them (single-peaked) are still not separated, and neither are pairs with a
    shallow saddle: of synthetic pairs separated by 2 FWHM, with 32 levels and a contrast of 0.005,
    92-98% separate for flux ratios up to 2, 14% at ratio 3 and 1% at ratio 4; with 128 levels and a
    contrast of 0.001, 99% separate at ratio 3 and 59% at ratio 4 (200 realizations, random
    orientation). In a noisy detection image a small ``contrast`` splits noise peaks into
    sources; smoothing reduces this. Bright regions of diffuse structure (nebulae) also come out as "sources".

    Verification
    ------------
    tests/unit/test_photometry.py: completeness, position and number of false detections in a synthetic star field; separation of close pairs.
    """
    g = np.asarray(image, dtype=float) - background
    detection_image, threshold = g, threshold_sigma * np.asarray(rms, dtype=float)
    if smoothing_fwhm:
        s = smoothing_fwhm / 2.3548200450309493
        detection_image = ndimage.gaussian_filter(g, s, mode="nearest")
        threshold = threshold / (2.0 * np.sqrt(np.pi) * s)  # rms of white noise after Gaussian smoothing
    labels, n = ndimage.label(detection_image > threshold, structure=np.ones((3, 3)))
    if n:
        label_sizes = np.bincount(labels.ravel(), minlength=n + 1)
        labels[label_sizes[labels] < min_pixels] = 0
        if deblend_levels > 0:
            labels = _deblend(detection_image, labels, deblend_levels, contrast, min_pixels)
    valid = np.unique(labels[labels > 0])
    if valid.size == 0:
        empty = np.empty(0)
        return SourceList(empty, empty, empty, empty, np.empty(0, dtype=np.int64))
    pixel_counts = np.asarray(ndimage.sum_labels(np.ones_like(g), labels, valid)).astype(np.int64)
    positive = np.clip(g, 0.0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        centers = np.array(ndimage.center_of_mass(positive, labels, valid)).reshape(-1, 2)
    # A segment detected in the smoothed image but with no positive pixel in the original image: the weight sum is zero.
    # Seen in real data (SOFI stack); for these segments the center is the geometric middle of the segment.
    missing = ~np.isfinite(centers).all(axis=1)
    if missing.any():
        centers[missing] = np.array(ndimage.center_of_mass(np.ones_like(g), labels, valid[missing])).reshape(-1, 2)
    flux = np.asarray(ndimage.sum_labels(g, labels, valid))
    peak = np.asarray(ndimage.maximum(g, labels, valid))
    order = np.argsort(-flux)
    return SourceList(centers[order, 1], centers[order, 0], flux[order], peak[order], pixel_counts[order])
