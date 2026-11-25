"""Building and subtracting a sky model from dithered frames.

In the near infrared the sky is much brighter than the sources and changes within
minutes. The solution is to dither the telescope between exposures and model the sky itself
from the frames of the same night: because stars fall on a different pixel in each frame,
the pixel-by-pixel median rejects them.

The distinction between static and dynamic models and their rules were read from the arXiv TeX
source of the THELI paper (1308.4989, ``theli.tex``, "Static and dynamic background models")
[Schirmer2013]: the static model is a single median image of a set of exposures and is
rescaled for changes in the overall level; the dynamic model for the k-th exposure is built
from the m exposures closest in time; in both cases objects are masked first and the highest
pixel in the stack is rejected.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kozmos._sources import source

Array = NDArray[np.float64]


def _median_model(stack: Array, reject_highest: int) -> Array:
    if reject_highest > 0 and stack.shape[0] > reject_highest + 1:
        stack = np.sort(stack, axis=0)  # NaNs go to the end
        number = np.sum(np.isfinite(stack), axis=0)
        rank = np.arange(stack.shape[0])[:, None, None]
        stack = np.where(rank >= number - reject_highest, np.nan, stack)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # pixels masked in all values: filled below
        return np.nanmedian(stack, axis=0)


def _model(frames: Array, masks: NDArray[np.bool_] | None, reject_highest: int) -> Array:
    """Pixel-by-pixel median of the stack with each frame divided by its own median (unit-level model).

    If a pixel is masked in all frames (or no value is left after rejection), the unmasked
    median is used at that pixel; the model is never left with ``nan`` at any pixel.
    """
    level = np.nanmedian(np.where(masks, np.nan, frames) if masks is not None else frames, axis=(1, 2))
    stack = frames / level[:, None, None]
    if masks is None:
        return _median_model(stack, reject_highest)
    model = _median_model(np.where(masks, np.nan, stack), reject_highest)
    empty = ~np.isfinite(model)
    if empty.any():
        model[empty] = _median_model(stack, reject_highest)[empty]
    return model


@source(
    "Schirmer2013",
    where="'Static and dynamic background models' section: static model",
    status="read",
    note="a single median image; rescaled to the frame it is applied to; objects are masked and the highest "
    "pixel in the stack is rejected. Taking the scale from the (unmasked) median of the frame is this project's "
    "choice",
)
def subtract_static_sky(
    frames: ArrayLike, *, masks: ArrayLike | None = None, reject_highest: int = 1
) -> tuple[Array, Array]:
    """Build a single sky model from all frames and subtract it from each frame with scaling.

    Parameters
    ----------
    frames : ndarray (K, Y, X)
        Frames in the same filter, dithered between exposures (flat-field corrected).
    masks : ndarray[bool] (K, Y, X), optional
        ``True`` = object pixel; excluded from the model.
    reject_highest : int
        How many of the highest values of the stack to reject at each pixel.

    Returns
    -------
    (sky-subtracted frames, model) : the model is unit-level (median around 1).

    References
    ----------
    [Schirmer2013] 'Static and dynamic background models'.

    Validity
    --------
    Sufficient if the shape of the sky does not change over the sequence (source: near-infrared
    sequences shorter than 10-30 minutes). Objects larger than the dither step leak into the
    model and subtract themselves. In a crowded field a mask is essential: in a synthetic
    sequence with 12 stars the aperture flux was measured 6.8% low with the unmasked model and
    0.6% low with the object mask.
    Rejecting the highest value reduces the star residue but pulls the median
    down on blank sky: in a synthetic sequence of 12 frames the residual shifts to positive
    by about one tenth of the frame noise (tests/unit/test_sky.py). If an object mask is given,
    ``reject_highest=0`` removes this bias. In a real SOFI Ks sequence (16 frames, 28 minutes)
    the static model reduced the large-scale residual from 156 ADU to 0.8 ADU; in the second
    sequence, where the galaxy fills the field, the residual stayed at 4.6 ADU
    (``figures/infrared_reduction.json``).

    Verification
    ------------
    tests/unit/test_sky.py: residual falls to the noise level in a synthetic sequence with a varying level, and the star flux is preserved.
    """
    k = np.asarray(frames, dtype=float)
    if k.ndim != 3 or k.shape[0] < 3:
        raise ValueError("at least three frames of shape (K, Y, X) are required")
    m = None if masks is None else np.asarray(masks, dtype=bool)
    model = _model(k, m, reject_highest)
    level = np.nanmedian(np.where(m, np.nan, k) if m is not None else k, axis=(1, 2))
    return k - level[:, None, None] * model[None, :, :], model


@source(
    "Schirmer2013",
    where="'Static and dynamic background models' section: dynamic model",
    status="read",
    note="the model of the k-th exposure is built from the m exposures closest in time; the window need not "
    "surround the k-th exposure symmetrically. The frame itself does not enter the model (this project's "
    "choice; the source does not specify)",
)
def subtract_dynamic_sky(
    frames: ArrayLike,
    times: ArrayLike,
    *,
    n_neighbors: int = 6,
    masks: ArrayLike | None = None,
    reject_highest: int = 1,
) -> Array:
    """Build a sky model for each frame from the frames closest in time and subtract it.

    Parameters
    ----------
    frames : ndarray (K, Y, X)
    times : ndarray (K,)
        Mid-exposure times (any unit).
    n_neighbors : int
        Number m of nearest frames used for each model (excluding the frame itself).
    masks, reject_highest
        Same as in ``subtract_static_sky``.

    Returns
    -------
    ndarray (K, Y, X)
        Sky-subtracted frames.

    References
    ----------
    [Schirmer2013] 'Static and dynamic background models'.

    Validity
    --------
    Needed if the shape of the sky changes within the sequence (source: near-infrared sequences
    longer than 10-30 minutes). Few neighbors increase the noise: the noise of the model built
    from m frames is added to the subtracted frame at a ratio of about 1/√m. In the real SOFI
    sequence (28 minutes) the dynamic model with 8 neighbors came out noisier than the static
    model (8.2 and 7.1 ADU); the static model is preferred for a short sequence.

    Verification
    ------------
    tests/unit/test_sky.py: smaller residual than the static model for a sky whose shape changes with time.
    """
    k = np.asarray(frames, dtype=float)
    t = np.asarray(times, dtype=float)
    if k.ndim != 3 or t.shape != (k.shape[0],):
        raise ValueError("frames must have shape (K, Y, X) and times shape (K,)")
    if n_neighbors < 3 or n_neighbors >= k.shape[0]:
        raise ValueError("number of neighbors must be at least 3 and smaller than the number of frames")
    m = None if masks is None else np.asarray(masks, dtype=bool)
    output = np.empty_like(k)
    for i in range(k.shape[0]):
        order = np.argsort(np.abs(t - t[i]), kind="stable")
        neighbor_idx = order[order != i][:n_neighbors]
        model = _model(k[neighbor_idx], None if m is None else m[neighbor_idx], reject_highest)
        level = np.nanmedian(np.where(m[i], np.nan, k[i]) if m is not None else k[i])
        output[i] = k[i] - level * model
    return output
