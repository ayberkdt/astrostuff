"""End-to-end reduction of real raw near-infrared frames.

Data: ESO NTT/SOFI, Ks band, program 184.D-1152(F), March 2012. Two science
sequences and the dome flats of the same night:

- SN 2011bm field: 16 frames (15 × 6 s), high galactic latitude, sparse field.
- SN 2005at field (NGC 6744): 32 frames (10 × 7 s), rich in stars; the galaxy fills the field.
- Ks dome flat: 8 frames (lamp on/off, masked/unmasked).

The frames come from the ESO archive as ``.fits.Z``; the uncompressed ``.fits``
files are expected under ``data/raw/eso/`` (``gzip -d`` unpacks the ``.Z`` files;
the flat field files carry the ``flat_`` prefix and the standard star files
the ``standard_`` prefix). The file identifiers are in
``examples/eso_sofi_frames.txt``; each one is downloaded from
``https://dataportal.eso.org/dataportal_new/file/<identifier>``.

Steps:

1. Flat field: the special dome flat recipe of the SOFI User Manual (shade correction with masked frames).
2. Static sky model without masks, bad pixel map, per-frame detection.
3. Offsets between frames: start from the offset commands in the header, refine with stars.
4. Object mask from the first stack; masked static and dynamic sky models; division by the flat field.
5. Stacking, source detection, astrometric solution against the 2MASS catalog (with and without the header) and zero point.

In this test the "truth" is not known; the criteria are internal consistency
(flatness of the residual background, repeatability of the flux of the same
star across frames) and the independent catalog (2MASS positions and K
magnitudes).

Usage:
    python examples/infrared_reduction.py
"""

from __future__ import annotations

import json
from pathlib import Path

import _plotting
import numpy as np
from scipy import ndimage

from kozmos import photometry
from kozmos.archives import gaia, twomass
from kozmos.astrometry import matching, solve, triangles, wcs
from kozmos.io import fits
from kozmos.reduction import sky
from kozmos.statistics import robust
from kozmos.visualization import stretch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "eso"
APERTURE, ANNULUS = 5.0, (10.0, 15.0)
LARGE_APERTURE, LARGE_ANNULUS = 10.0, (15.0, 20.0)
STANDARDS = {
    "FS14 (night of March 13/14)": "standard_SOFI.2012-03-13T23*.fits",
    "FS24 (night of March 14/15)": "standard_SOFI.2012-03-15T10*.fits",
}
SEQUENCES = {"SN2011bm": "SOFI.2012-03-15T07*.fits", "SN2005at": "SOFI.2012-03-14T09*.fits"}


def detect(image: np.ndarray, threshold: float) -> photometry.SourceList:
    background, rms = photometry.background_map(image, cell=64)
    return photometry.detect_sources(
        image, background, rms, threshold_sigma=threshold, min_pixels=8, smoothing_fwhm=3.0
    )


def matched_offsets(list0, list1, guess, radius: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """Return the (dx, dy) differences of pairs matched within ``radius``, assuming list1 ≈ list0 + guess."""
    dx = list1.x[:, None] - list0.x[None, :]
    dy = list1.y[:, None] - list0.y[None, :]
    near = np.hypot(dx - guess[0], dy - guess[1]) < radius
    return dx[near], dy[near]


def stack(frames: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place the frames on a common canvas with whole-pixel offsets and average them; (stack, coverage, canvas origin)."""
    whole = np.round(offsets).astype(int)
    origin = whole.max(axis=0)  # canvas pixel = frame pixel − offset + origin
    ny, nx = frames.shape[1:]
    total = np.zeros((ny + int(origin[1] - whole[:, 1].min()), nx + int(origin[0] - whole[:, 0].min())))
    count = np.zeros_like(total)
    for frame, (dx, dy) in zip(frames, whole, strict=True):
        x0, y0 = int(origin[0] - dx), int(origin[1] - dy)
        good = np.isfinite(frame)
        total[y0 : y0 + ny, x0 : x0 + nx] += np.where(good, frame, 0.0)
        count[y0 : y0 + ny, x0 : x0 + nx] += good
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan), count, origin


def large_scale_structure(frames: np.ndarray, masks: np.ndarray, box: int = 64) -> float:
    """Return the within-frame scatter of the medians of 64-pixel boxes of unmasked pixels; median over frames [ADU]."""
    results = []
    for frame, mask in zip(frames, masks, strict=True):
        d = np.where(mask, np.nan, frame)
        n = d.shape[0] // box
        blocks = d[: n * box, : n * box].reshape(n, box, n, box).swapaxes(1, 2).reshape(n, n, -1)
        with np.errstate(all="ignore"):
            results.append(np.nanstd(np.nanmedian(blocks, axis=2)))
    return float(np.median(results))


def build_flat_field() -> tuple[np.ndarray, np.ndarray, dict]:
    """Build the dome flats: (custom, simple, info).

    The frame order was read from the SOFI User Manual (Fig. 3.3) and is verified against the frame levels:
    lamp off, off+mask, on+mask, on; the second half in reverse order (on, on+mask, off+mask, off).

    Simple flat field: unmasked (on − off). The manual says this does not fully remove the illumination-dependent
    shading pattern of the detector (shade) and leaves a discontinuity of a few percent in the zero point across the
    middle of the array. Custom flat field (Section 5.2.5): for each lamp state, A = columns 50-150 of the masked
    frame, B = columns 500-600 of the masked frame, C = columns 500-600 of the unmasked frame, averaged along the
    rows; the shading in the unmasked frame is A + (C − B) and is subtracted from the frame row by row; then
    on − off is taken.
    """
    paths = sorted(DATA_DIR.glob("flat_SOFI*.fits"))
    if len(paths) != 8:
        raise SystemExit(f"8 flat field frames expected under {DATA_DIR}, found {len(paths)}")
    frames = np.array([fits.read_fits(path)[0] for path in paths])  # in time order
    level = np.median(frames, axis=(1, 2))
    off, off_masked, on_masked, on = (0, 7), (1, 6), (2, 5), (3, 4)
    # Check of the order against the data: lamp-on frames are brighter than off frames, and unmasked frames are
    # brighter than masked frames in the same state.
    assert level[list(on + on_masked)].min() > level[list(off + off_masked)].max()
    assert (
        level[list(on)].min() > level[list(on_masked)].max() and level[list(off)].min() > level[list(off_masked)].max()
    )

    def without_shade(unmasked: np.ndarray, masked: np.ndarray) -> np.ndarray:
        a = masked[:, 49:150].mean(axis=1)
        b = masked[:, 499:600].mean(axis=1)
        c = unmasked[:, 499:600].mean(axis=1)
        return unmasked - (a + (c - b))[:, None]

    simple_difference = frames[list(on)].mean(axis=0) - frames[list(off)].mean(axis=0)
    custom_difference = np.mean(
        [
            without_shade(frames[on[i]], frames[on_masked[i]]) - without_shade(frames[off[i]], frames[off_masked[i]])
            for i in (0, 1)
        ],
        axis=0,
    )
    simple, custom = simple_difference / np.median(simple_difference), custom_difference / np.median(custom_difference)
    ratio = custom / simple
    return (
        custom,
        simple,
        {
            "frame_levels_adu": [round(float(x), 1) for x in level],
            "lamp_on_minus_off_adu": float(np.median(simple_difference)),
            "flat_field_scatter": float(robust.mad_scale(custom)),
            # The discontinuity mentioned in the manual: medians of the custom/simple ratio in the lower and upper half of the array.
            "custom_over_simple_lower_half": float(np.median(ratio[100:500, 100:900])),
            "custom_over_simple_upper_half": float(np.median(ratio[524:924, 100:900])),
        },
    )


def reduce_sequence(name: str, pattern: str, flat: np.ndarray, simple_flat: np.ndarray) -> tuple[dict, dict]:
    paths = sorted(DATA_DIR.glob(pattern))
    if len(paths) < 8:
        raise SystemExit(f"no {name} frames under {DATA_DIR}; identifiers are in examples/eso_sofi_frames.txt")
    frames, times, headers = [], [], []
    for path in paths:
        data, header = fits.read_fits(path)
        frames.append(data)
        times.append(float(header["MJD-OBS"]) * 1440.0)  # minutes
        headers.append(header)
    frames, times = np.array(frames), np.array(times)
    times -= times[0]
    level = np.median(frames, axis=(1, 2))
    n_frames, (ny, nx) = len(frames), frames.shape[1:]
    b0 = headers[0]

    # Static model without masks. Bad pixels: those that deviate from the local median in the sky model and those whose flat field is an outlier.
    first, first_model = sky.subtract_static_sky(frames)
    local = first_model - ndimage.median_filter(first_model, size=5, mode="nearest")
    bad = np.abs(local) > 6.0 * robust.mad_scale(local)
    bad |= (flat < 0.5) | (flat > 1.5)
    bad = ndimage.binary_dilation(bad, structure=np.ones((3, 3)))
    first_clean = first.copy()
    first_clean[:, bad] = 0.0
    lists = [detect(frame, threshold=6.0) for frame in first_clean]

    # Offsets: the offset commands in the header are the initial guess; how the command maps onto the detector axes and
    # its sign (eight possibilities) is found from the data as the option that matches the most stars, then refined with stars.
    command = np.array([[b["ESO SEQ CUMOFFSETX"], b["ESO SEQ CUMOFFSETY"]] for b in headers])
    command -= command[0]
    candidates = []
    for order in ((0, 1), (1, 0)):
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                guess = np.column_stack([sx * command[:, order[0]], sy * command[:, order[1]]])
                score = sum(matched_offsets(lists[0], lists[i], guess[i])[0].size for i in range(1, n_frames))
                candidates.append((score, order, (sx, sy), guess))
    candidates.sort(key=lambda candidate: -candidate[0])
    _, command_order, command_signs, guess = candidates[0]
    offsets, matches = [(0.0, 0.0)], []
    for i in range(1, n_frames):
        dx, dy = matched_offsets(lists[0], lists[i], guess[i])
        offsets.append((float(np.median(dx)), float(np.median(dy))) if dx.size else tuple(guess[i]))
        matches.append(int(dx.size))
    offsets = np.array(offsets)
    whole = np.round(offsets).astype(int)

    # Object mask from the first stack (in stack coordinates), carried back to the coordinates of each frame.
    first_stack, coverage, origin = stack(np.where(bad[None], np.nan, first), offsets)
    full_coverage = coverage >= 0.5 * n_frames
    filled = np.where(full_coverage, first_stack, np.nanmedian(first_stack[full_coverage]))
    stack_background, _ = photometry.background_map(filled, cell=64)
    smooth = ndimage.gaussian_filter(filled - stack_background, 1.5)
    objects = (smooth > 3.0 * robust.mad_scale(smooth[full_coverage])) & full_coverage
    objects = ndimage.binary_dilation(objects, structure=np.ones((3, 3)), iterations=3)
    masks = np.array(
        [
            objects[int(origin[1] - dy) : int(origin[1] - dy) + ny, int(origin[0] - dx) : int(origin[0] - dx) + nx]
            for dx, dy in whole
        ]
    )

    static, _ = sky.subtract_static_sky(frames, masks=masks)
    dynamic = sky.subtract_dynamic_sky(frames, times, n_neighbors=8, masks=masks)
    level_only = frames - level[:, None, None]  # only the level subtracted: comparison baseline
    excluded = masks | bad[None]
    safe_flat = np.where(bad, 1.0, flat)
    static_flat = static / safe_flat  # the sky-subtracted frame is divided by the flat field
    static_simple = static / np.where(bad, 1.0, simple_flat)

    def noise(series: np.ndarray) -> float:
        return float(np.median([robust.mad_scale(k[~m]) for k, m in zip(series, excluded, strict=True)]))

    # Stack (static model with flat field; bad pixels are left out), detection.
    final_stack, coverage, origin = stack(np.where(bad[None], np.nan, static_flat), offsets)
    full_coverage = coverage >= 0.5 * n_frames
    stack_filled = np.where(full_coverage, final_stack, np.nanmedian(final_stack[full_coverage]))
    detections = detect(stack_filled, threshold=5.0)
    interior = ndimage.binary_erosion(full_coverage, iterations=15)
    inside = interior[np.round(detections.y).astype(int), np.round(detections.x).astype(int)]
    kx, ky = detections.x[inside], detections.y[inside]
    # In the blind solution the brightness criterion is the peak value: segment flux would put diffuse galaxy knots ahead of stars.
    k_flux = detections.peak[inside]

    # Astrometry: (a) starting from the WCS in the header, (b) with triangle matching without using the header.
    header_wcs = wcs.wcs_from_header(b0)
    header_wcs = wcs.Wcs(crpix=header_wcs.crpix + origin, crval=header_wcs.crval, cd=header_wcs.cd)
    catalog = twomass.point_sources(float(b0["CRVAL1"]), float(b0["CRVAL2"]), 0.09, ROOT / "data" / "raw" / "2mass")
    shifted, votes = solve.find_shift(
        header_wcs, kx, ky, catalog["ra"], catalog["dec"], max_shift_pix=320.0, bin_pix=4.0
    )
    solved = solve.refine_wcs(
        shifted, kx, ky, catalog["ra"], catalog["dec"], radius_arcsec=4.0, final_radius_arcsec=1.5
    )
    px, py = wcs.sky_to_pixel(solved.wcs, catalog["ra"], catalog["dec"])
    in_field = (
        np.isfinite(px) & (px > 0) & (px < stack_filled.shape[1] - 1) & (py > 0) & (py < stack_filled.shape[0] - 1)
    )
    in_field[in_field] = interior[np.round(py[in_field]).astype(int), np.round(px[in_field]).astype(int)]
    ra, dec = wcs.pixel_to_sky(solved.wcs, kx, ky)
    to_detections = matching.cross_match(catalog["ra"][in_field], catalog["dec"][in_field], ra, dec, 1.5)
    try:
        # Blind solution: the WCS in the header is not used. Only the approximate sky position and field width are
        # used: the catalog is limited to a cone with a radius of half the short side of the stack.
        radius = 0.5 * min(stack_filled.shape) * float(b0["ESO INS PIXSCALE"]) / 3600.0
        distance = np.hypot(
            (catalog["ra"] - float(b0["CRVAL1"])) * np.cos(np.radians(float(b0["CRVAL2"]))),
            catalog["dec"] - float(b0["CRVAL2"]),
        )
        in_cone = distance < radius
        blind, pairs = triangles.blind_solve(
            kx,
            ky,
            k_flux,
            catalog["ra"][in_cone],
            catalog["dec"][in_cone],
            catalog["k_m"][in_cone],
            n_brightest=40,
            tolerance=0.015,
        )
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, stack_filled.shape[1] - 1, 4), np.linspace(0, stack_filled.shape[0] - 1, 4)
        )
        r1, d1 = wcs.pixel_to_sky(blind.wcs, grid_x, grid_y)
        r2, d2 = wcs.pixel_to_sky(solved.wcs, grid_x, grid_y)
        blind_result = {
            "pairs_assigned_in_triangle_match": pairs,
            "stars_in_solution": blind.n_stars,
            "max_difference_from_header_started_solution_arcsec": float(
                np.max(np.hypot((r1 - r2) * np.cos(np.radians(d2)), d1 - d2)) * 3600.0
            ),
        }
    except ValueError as error:
        # Diagnosis: how many stars did the two bright lists entering the triangle match share, using the solution started from the header?
        g = np.argsort(-k_flux)[:40]
        kk = np.argsort(catalog["k_m"][in_cone])[:40]
        common = matching.cross_match(ra[g], dec[g], catalog["ra"][in_cone][kk], catalog["dec"][in_cone][kk], 1.5)
        blind_result = {
            "failed": str(error),
            "catalog_stars_in_cone": int(in_cone.sum()),
            "common_stars_of_bright_lists": int(common.matched.sum()),
        }

    # Astrometry with Gaia DR3: it is much deeper than 2MASS, so it gives enough stars even in the sparse field.
    # Positions are moved to the observing epoch with proper motion. In the blind solution the brightness order comes from the G magnitude.
    epoch = 2000.0 + (float(b0["MJD-OBS"]) - 51544.5) / 365.25
    gaia_catalog = gaia.cone_search(
        float(b0["CRVAL1"]), float(b0["CRVAL2"]), 0.09, ROOT / "data" / "raw" / "gaia", epoch_yr=epoch
    )
    gaia_shifted, gaia_votes = solve.find_shift(
        header_wcs, kx, ky, gaia_catalog["ra"], gaia_catalog["dec"], max_shift_pix=320.0, bin_pix=4.0
    )
    gaia_solution = solve.refine_wcs(
        gaia_shifted, kx, ky, gaia_catalog["ra"], gaia_catalog["dec"], radius_arcsec=4.0, final_radius_arcsec=1.0
    )
    corner_x, corner_y = np.meshgrid(
        np.linspace(0, stack_filled.shape[1] - 1, 4), np.linspace(0, stack_filled.shape[0] - 1, 4)
    )

    def solution_difference(w1: wcs.Wcs, w2: wcs.Wcs) -> float:
        a1, d1 = wcs.pixel_to_sky(w1, corner_x, corner_y)
        a2, d2 = wcs.pixel_to_sky(w2, corner_x, corner_y)
        return float(np.max(np.hypot((a1 - a2) * np.cos(np.radians(d2)), d1 - d2)) * 3600.0)

    gaia_result = {
        "epoch_yr": epoch,
        "gaia_sources_in_cone": int(len(gaia_catalog["ra"])),
        "shift_votes": gaia_votes,
        "stars_in_solution": gaia_solution.n_stars,
        "residual_rms_arcsec": gaia_solution.residual_rms_arcsec,
        "max_difference_from_2mass_solution_arcsec": solution_difference(gaia_solution.wcs, solved.wcs),
    }
    try:
        gaia_in_cone = (
            np.hypot(
                (gaia_catalog["ra"] - float(b0["CRVAL1"])) * np.cos(np.radians(float(b0["CRVAL2"]))),
                gaia_catalog["dec"] - float(b0["CRVAL2"]),
            )
            < 0.5 * min(stack_filled.shape) * float(b0["ESO INS PIXSCALE"]) / 3600.0
        )
        gaia_blind, gaia_pairs = triangles.blind_solve(
            kx,
            ky,
            k_flux,
            gaia_catalog["ra"][gaia_in_cone],
            gaia_catalog["dec"][gaia_in_cone],
            gaia_catalog["phot_g_mean_mag"][gaia_in_cone],
            n_brightest=40,
            tolerance=0.015,
            final_radius_arcsec=1.0,
            min_validation_fraction=0.2,  # in a sparse field the bright lists share few stars; the absolute lower limit (8) is kept
        )
        gaia_result["blind_solution"] = {
            "pairs_assigned_in_triangle_match": gaia_pairs,
            "stars_in_solution": gaia_blind.n_stars,
            "residual_rms_arcsec": gaia_blind.residual_rms_arcsec,
            "max_difference_from_header_started_gaia_solution_arcsec": solution_difference(
                gaia_blind.wcs, gaia_solution.wcs
            ),
        }
    except ValueError as error:
        gaia_result["blind_solution"] = {"failed": str(error)}

    # Zero point: unsaturated, matched stars with a 2MASS K uncertainty below 0.15 mag.
    matches_2mass = matching.cross_match(ra, dec, catalog["ra"], catalog["dec"], 1.5)
    flux, _ = photometry.aperture_photometry(stack_filled, kx, ky, APERTURE, annulus=ANNULUS)
    peak = detections.peak[inside]
    candidate = matches_2mass.matched & (catalog["k_cmsig"][matches_2mass.index] < 0.15) & (flux > 0) & (peak < 15000.0)

    # Diagnosis of the zero point stars: point source or not (narrow/wide aperture ratio), present in Gaia or not, 2MASS quality.
    narrow, _ = photometry.aperture_photometry(stack_filled, kx, ky, 3.0, annulus=ANNULUS)
    wide, _ = photometry.aperture_photometry(stack_filled, kx, ky, 8.0, annulus=ANNULUS)
    g_ra, g_dec = wcs.pixel_to_sky(gaia_solution.wcs, kx, ky)
    gaia_match = matching.cross_match(g_ra, g_dec, gaia_catalog["ra"], gaia_catalog["dec"], 1.0)
    concentration = np.where(wide > 0, narrow / np.where(wide > 0, wide, 1.0), np.nan)
    concentration_all = np.nanmedian(concentration[gaia_match.matched & (flux > 0)])
    # Point source filter: the 2MASS Point Source Catalog also contains compact galaxies; a fixed aperture misses most
    # of their flux and lowers the zero point (a 0.77 mag deviation in the sparse field was found this way).
    point_source = np.abs(concentration - concentration_all) < 0.15
    k_good = candidate & point_source
    zero_points = catalog["k_m"][matches_2mass.index][k_good] + 2.5 * np.log10(flux[k_good])
    diagnostics = [
        {
            "zero_point": round(
                float(catalog["k_m"][matches_2mass.index[i]] + 2.5 * np.log10(flux[i])),
                3,
            ),
            "k_2mass": float(catalog["k_m"][matches_2mass.index[i]]),
            "k_uncertainty": float(catalog["k_cmsig"][matches_2mass.index[i]]),
            "ph_qual": str(catalog["ph_qual"][matches_2mass.index[i]]),
            "narrow_over_wide_aperture": round(float(narrow[i] / wide[i]), 3),
            "median_ratio_in_gaia_stars": round(float(concentration_all), 3),
            "point_source": bool(point_source[i]),
            "gaia_counterpart": bool(gaia_match.matched[i]),
            "gaia_g": float(gaia_catalog["phot_g_mean_mag"][gaia_match.index[i]]) if gaia_match.matched[i] else None,
            "stack_position_pix": [round(float(kx[i]), 1), round(float(ky[i]), 1)],
            "nearest_other_source_pix": round(float(np.sort(np.hypot(kx - kx[i], ky - ky[i]))[1]), 1),
        }
        for i in np.flatnonzero(candidate)
    ]

    # Aperture correction: flux ratio of the 10-pixel aperture to the 5-pixel aperture for bright point sources [mag].
    large, _ = photometry.aperture_photometry(stack_filled, kx, ky, LARGE_APERTURE, annulus=LARGE_ANNULUS)
    bright_points = np.flatnonzero(gaia_match.matched & point_source & (flux > 0) & (large > 0) & (peak < 15000.0))
    bright_points = bright_points[np.argsort(-flux[bright_points])][:8]
    corrections = 2.5 * np.log10(large[bright_points] / flux[bright_points])
    aperture_correction = float(np.median(corrections)) if bright_points.size else None

    def repeatability(series: np.ndarray) -> dict:
        """Relative robust scatter of the aperture flux of bright stars across frames, and its ratio to the expected noise."""
        measured, expected = [], []
        pixel_sigma = noise(series)
        n_pix, n_b = np.pi * APERTURE**2, np.pi * (ANNULUS[1] ** 2 - ANNULUS[0] ** 2)
        for i in np.argsort(-flux):
            x_k = kx[i] - origin[0] + offsets[:, 0]
            y_k = ky[i] - origin[1] + offsets[:, 1]
            if np.any((x_k < 20) | (x_k > nx - 21) | (y_k < 20) | (y_k > ny - 21)):
                continue
            if max(frames[k][int(round(y_k[k])), int(round(x_k[k]))] for k in range(n_frames)) > 20000.0:
                continue  # stay away from the saturation region
            f = np.array(
                [
                    float(
                        photometry.aperture_photometry(
                            np.where(bad, 0.0, series[k]), x_k[k], y_k[k], APERTURE, annulus=ANNULUS
                        )[0]
                    )
                    for k in range(n_frames)
                ]
            )
            if np.median(f) > 30.0 * pixel_sigma * np.sqrt(
                n_pix
            ):  # S/N > 30: measure the correction error, not the noise
                measured.append(robust.mad_scale(f) / np.median(f))
                expected.append(pixel_sigma * np.sqrt(n_pix * (1.0 + n_pix / n_b)) / np.median(f))
            if len(measured) >= 15:
                break
        if not measured:
            return {"star_count": 0}
        return {
            "star_count": len(measured),
            "relative_scatter": float(np.median(measured)),
            "expected_background_noise": float(np.median(expected)),
        }

    def extinction_slope(series: np.ndarray) -> dict:
        """Slope of the relative magnitude of bright stars in the sequence against airmass [mag / airmass].

        The frame fluxes of each star are divided by its own median; in each frame the median over stars is taken and
        the relative magnitude is fitted linearly against airmass. The slope cannot tell extinction from a drift in transparency.
        """
        airmass = np.array([float(b["ESO TEL AIRM START"]) for b in headers])
        relative = []
        for i in np.argsort(-flux):
            x_k = kx[i] - origin[0] + offsets[:, 0]
            y_k = ky[i] - origin[1] + offsets[:, 1]
            if np.any((x_k < 20) | (x_k > nx - 21) | (y_k < 20) | (y_k > ny - 21)):
                continue
            if max(frames[k][int(round(y_k[k])), int(round(x_k[k]))] for k in range(n_frames)) > 20000.0:
                continue
            f = np.array(
                [
                    float(
                        photometry.aperture_photometry(
                            np.where(bad, 0.0, series[k]), x_k[k], y_k[k], APERTURE, annulus=ANNULUS
                        )[0]
                    )
                    for k in range(n_frames)
                ]
            )
            if np.median(f) > 30.0 * noise(series) * np.sqrt(np.pi * APERTURE**2):
                relative.append(f / np.median(f))
            if len(relative) >= 15:
                break
        if len(relative) < 5:
            return {"star_count": len(relative), "airmass_range": [float(airmass.min()), float(airmass.max())]}
        magnitude = -2.5 * np.log10(np.median(np.array(relative), axis=0))
        design = np.column_stack([np.ones(n_frames), airmass - airmass.mean()])
        coefficient, *_ = np.linalg.lstsq(design, magnitude, rcond=None)
        residual = magnitude - design @ coefficient
        covariance = np.linalg.inv(design.T @ design) * residual.var(ddof=2)
        return {
            "star_count": len(relative),
            "airmass_range": [float(airmass.min()), float(airmass.max())],
            "slope_mag_per_airmass": float(coefficient[1]),
            "slope_uncertainty": float(np.sqrt(covariance[1, 1])),
            "residual_rms_per_frame_mag": float(residual.std(ddof=2)),
        }

    result = {
        "data": {
            "target": str(b0["OBJECT"]).strip(),
            "filter": str(b0["ESO INS FILT1 NAME"]).strip(),
            "n_frames": n_frames,
            "dit_s": float(b0["ESO DET DIT"]),
            "ndit": int(b0["ESO DET NDIT"]),
            "sequence_duration_min": float(times[-1]),
            "airmass": float(b0["ESO TEL AIRM START"]),
        },
        "sky_level_adu": [float(level.min()), float(level.max())],
        "bad_pixel_fraction": float(bad.mean()),
        "object_mask_fraction": float(masks.mean()),
        "offsets": {
            "max_pixel": float(np.abs(offsets).max()),
            "min_matches_per_frame": int(min(matches)),
            "scores_of_best_and_second_option": [int(candidates[0][0]), int(candidates[1][0])],
            "header_command_axis_order": list(command_order),
            "header_command_signs": list(command_signs),
            "rms_difference_from_header_command_pix": float(np.sqrt(np.mean((offsets - guess) ** 2))),
        },
        "large_scale_residual_adu": {
            "level_only_subtracted": large_scale_structure(level_only, excluded),
            "static_unmasked": large_scale_structure(first, excluded),
            "static_masked": large_scale_structure(static, excluded),
            "dynamic_masked": large_scale_structure(dynamic, excluded),
        },
        "pixel_noise_adu": {
            "level_only_subtracted": noise(level_only),
            "static_masked": noise(static),
            "dynamic_masked": noise(dynamic),
        },
        "repeatability": {
            "without_flat": repeatability(static),
            "simple_flat": repeatability(static_simple),
            "custom_flat": repeatability(static_flat),
        },
        "intra_sequence_extinction": extinction_slope(static_flat),
        "stack": {"size": list(stack_filled.shape), "detected_sources": int(inside.sum())},
        "astrometry": {
            "catalog_stars": int(len(catalog["ra"])),
            "shift_votes": votes,
            "header_wcs_shift_pix": (shifted.crpix - header_wcs.crpix).tolist(),
            "stars_in_solution": solved.n_stars,
            "catalog_stars_in_field": int(in_field.sum()),
            "detected_of_catalog_stars_in_field": int(to_detections.matched.sum()),
            "residual_rms_arcsec": solved.residual_rms_arcsec,
            "scale_arcsec_pix": float(np.sqrt(abs(np.linalg.det(solved.wcs.cd))) * 3600.0),
            "header_scale_arcsec_pix": float(b0["ESO INS PIXSCALE"]),
            "blind_solution": blind_result,
            "gaia": gaia_result,
        },
        "photometry": {
            "candidate_count": int(candidate.sum()),
            "discarded_as_not_point_source": int((candidate & ~point_source).sum()),
            "star_count": int(k_good.sum()),
            "zero_point_mag": float(np.median(zero_points)) if zero_points.size else None,
            "scatter_mag": float(robust.mad_scale(zero_points)) if zero_points.size > 2 else None,
            "individual_zero_points": np.sort(zero_points).round(3).tolist() if zero_points.size <= 6 else None,
            "star_diagnostics": diagnostics if candidate.sum() <= 6 else None,
            "aperture_correction_mag": aperture_correction,
            "aperture_correction_star_count": int(bright_points.size),
            "aperture_correction_scatter": float(robust.mad_scale(corrections)) if bright_points.size > 2 else None,
            "median_catalog_uncertainty_mag": float(np.median(catalog["k_cmsig"][matches_2mass.index][k_good]))
            if zero_points.size
            else None,
        },
    }
    ys, xs = np.nonzero(coverage >= n_frames - 1)
    images = {
        "raw": (frames[0], noise(level_only), float(level[0])),
        "subtracted": (np.where(bad, 0.0, static_flat[0]), noise(static_flat), 0.0),
        "stack": (stack_filled[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], None, 0.0),
    }
    return result, images


def measure_standard(pattern: str, flat: np.ndarray) -> dict:
    """Zero point from field stars in a standard star sequence (4 frames, four corners).

    The magnitude of the standard star itself is not used (the magnitudes of the UKIRT faint standards were not read
    in this work); the calibrators are the 2MASS stars in the same frames (those with a K uncertainty below 0.05 mag).
    """
    paths = sorted(DATA_DIR.glob(pattern))
    if len(paths) < 3:
        raise SystemExit(f"no {pattern} frames under {DATA_DIR}; identifiers are in examples/eso_sofi_frames.txt")
    frames, headers = [], []
    for path in paths:
        data, header = fits.read_fits(path)
        frames.append(data)
        headers.append(header)
    frames = np.array(frames)
    b0 = headers[0]
    bad = ndimage.binary_dilation((flat < 0.5) | (flat > 1.5), structure=np.ones((3, 3)))
    subtracted, _ = sky.subtract_static_sky(frames, reject_highest=0)
    subtracted = np.where(bad[None], np.nan, subtracted / np.where(bad, 1.0, flat))
    # Offsets: the rule found from the data in the science sequences (axis order X, Y; signs +, +), refined with stars.
    command = np.array([[b["ESO SEQ CUMOFFSETX"], b["ESO SEQ CUMOFFSETY"]] for b in headers])
    command -= command[0]
    lists = [detect(np.nan_to_num(frame), threshold=8.0) for frame in subtracted]
    offsets = [(0.0, 0.0)]
    for i in range(1, len(frames)):
        dx, dy = matched_offsets(lists[0], lists[i], command[i])
        offsets.append((float(np.median(dx)), float(np.median(dy))) if dx.size else tuple(command[i]))
    offsets = np.array(offsets)
    stack_image, coverage, origin = stack(subtracted, offsets)
    enough = coverage >= 2
    stack_image = np.where(enough, stack_image, np.nanmedian(stack_image[enough]))
    detections = detect(stack_image, threshold=6.0)
    interior = ndimage.binary_erosion(enough, iterations=20)
    inside = interior[np.round(detections.y).astype(int), np.round(detections.x).astype(int)]
    kx, ky, peak = detections.x[inside], detections.y[inside], detections.peak[inside]
    header_wcs = wcs.wcs_from_header(b0)
    header_wcs = wcs.Wcs(crpix=header_wcs.crpix + origin, crval=header_wcs.crval, cd=header_wcs.cd)
    catalog = twomass.point_sources(float(b0["CRVAL1"]), float(b0["CRVAL2"]), 0.09, ROOT / "data" / "raw" / "2mass")
    shifted, _ = solve.find_shift(header_wcs, kx, ky, catalog["ra"], catalog["dec"], max_shift_pix=320.0, bin_pix=4.0)
    solved = solve.refine_wcs(
        shifted, kx, ky, catalog["ra"], catalog["dec"], radius_arcsec=4.0, final_radius_arcsec=1.5
    )
    ra, dec = wcs.pixel_to_sky(solved.wcs, kx, ky)
    match = matching.cross_match(ra, dec, catalog["ra"], catalog["dec"], 1.5)
    small, _ = photometry.aperture_photometry(stack_image, kx, ky, APERTURE, annulus=ANNULUS)
    large, _ = photometry.aperture_photometry(stack_image, kx, ky, LARGE_APERTURE, annulus=LARGE_ANNULUS)
    good = match.matched & (catalog["k_cmsig"][match.index] < 0.05) & (small > 0) & (large > 0) & (peak < 15000.0)
    k_m = catalog["k_m"][match.index][good]
    dit = float(b0["ESO DET DIT"])
    zero_large = k_m + 2.5 * np.log10(large[good] / dit)  # per second, 10-pixel aperture
    # Loose selection: stars with a K uncertainty up to 0.10 mag are included too (for fields with few calibrators).
    loose = match.matched & (catalog["k_cmsig"][match.index] < 0.10) & (small > 0) & (large > 0) & (peak < 15000.0)
    zero_loose = catalog["k_m"][match.index][loose] + 2.5 * np.log10(large[loose] / dit)
    return {
        "target": str(b0.get("ESO OBS TARG NAME", b0["OBJECT"])).strip(),
        "n_frames": len(frames),
        "dit_s": dit,
        "airmass": float(b0["ESO TEL AIRM START"]),
        "astrometry_star_count": solved.n_stars,
        "astrometry_residual_rms_arcsec": solved.residual_rms_arcsec,
        "calibrator_star_count": int(good.sum()),
        "calibrator_k_magnitudes": np.sort(k_m).round(3).tolist(),
        "zero_point_1s_10pix": float(np.median(zero_large)) if good.any() else None,
        "zero_point_scatter": float(robust.mad_scale(zero_large)) if good.sum() > 2 else None,
        "individual_zero_points": np.sort(zero_large).round(3).tolist(),
        "loose_selection": {
            "star_count": int(loose.sum()),
            "zero_point_1s_10pix": float(np.median(zero_loose)) if loose.any() else None,
            "scatter": float(robust.mad_scale(zero_loose)) if loose.sum() > 2 else None,
            "individual": np.sort(zero_loose).round(3).tolist(),
            "matched_2mass_stars_in_field": int(match.matched.sum()),
        },
        "aperture_correction_mag": float(np.median(2.5 * np.log10(large[good] / small[good]))) if good.any() else None,
    }


def main() -> None:
    flat, simple_flat, flat_info = build_flat_field()
    result = {
        "data": {"instrument": "ESO NTT / SOFI", "program": "184.D-1152(F)", "source": "ESO Science Archive"},
        "flat_field": flat_info,
        "sequences": {},
    }
    images = {}
    for name, pattern in SEQUENCES.items():
        result["sequences"][name], images[name] = reduce_sequence(name, pattern, flat, simple_flat)
    # Zero point transfer: the sparse field has too few 2MASS stars. The zero point of the rich field is corrected for
    # the exposure time difference and carried to the sparse field, where it is compared with the few stars there. Pixel
    # values are per single DIT: SOFI User Manual (Section 2), "counts in a raw data file always correspond to DIT
    # seconds ... the average of NDIT sub-integrations". There is no airmass correction.
    rich, sparse = result["sequences"]["SN2005at"], result["sequences"]["SN2011bm"]
    transferred = rich["photometry"]["zero_point_mag"] - 2.5 * np.log10(rich["data"]["dit_s"] / sparse["data"]["dit_s"])
    result["zero_point_transfer"] = {
        "rich_field_zero_point": rich["photometry"]["zero_point_mag"],
        "dit_correction_mag": float(-2.5 * np.log10(rich["data"]["dit_s"] / sparse["data"]["dit_s"])),
        "transferred_to_sparse_field": float(transferred),
        "zero_points_of_point_sources_in_sparse_field": sparse["photometry"]["individual_zero_points"],
        "difference_mag": float(sparse["photometry"]["zero_point_mag"] - transferred),
        "airmasses": [rich["data"]["airmass"], sparse["data"]["airmass"]],
    }
    # Zero point from the standard star fields and comparison per night. All zero points are reduced to per second and
    # the 10-pixel aperture: the aperture correction of the science field is added to the zero point of the 5-pixel
    # aperture. The difference between standard and science field in the same night, divided by the airmass
    # difference, is converted to an extinction coefficient (from two points; only a consistency indicator).
    result["standard_stars"] = {name: measure_standard(pattern, flat) for name, pattern in STANDARDS.items()}
    night = {"SN2005at": "FS14 (night of March 13/14)", "SN2011bm": "FS24 (night of March 14/15)"}
    result["per_night_comparison"] = {}
    for field, standard in night.items():
        science, std = result["sequences"][field], result["standard_stars"][standard]
        f = science["photometry"]
        if None in (f["zero_point_mag"], f["aperture_correction_mag"], std["zero_point_1s_10pix"]):
            continue
        science_1s = f["zero_point_mag"] - 2.5 * np.log10(science["data"]["dit_s"]) + f["aperture_correction_mag"]
        difference = std["zero_point_1s_10pix"] - science_1s
        airmass_difference = science["data"]["airmass"] - std["airmass"]
        result["per_night_comparison"][field] = {
            "standard": standard,
            "science_zero_point_1s_10pix": float(science_1s),
            "science_star_count": f["star_count"],
            "standard_zero_point_1s_10pix": std["zero_point_1s_10pix"],
            "standard_star_count": std["calibrator_star_count"],
            "difference_mag": float(difference),
            "airmass_difference": float(airmass_difference),
            "extinction_mag_per_airmass": float(difference / airmass_difference),
        }
    out_dir = ROOT / "figures"
    (out_dir / "infrared_reduction.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))

    # Figure: raw frame, sky-subtracted frame, central region of the stack where all frames overlap (SN 2011bm sequence).
    def stretch_image(image: np.ndarray, scatter: float | None, floor: float) -> np.ndarray:
        scatter = robust.mad_scale(image) if scatter is None else scatter
        g = stretch.asinh_stretch(image, black=floor - 2.0 * scatter, alpha=1.0 / (10.0 * scatter), q=6.0)
        return stretch.percentile_scale(g, 0.5, 99.8)

    g = images["SN2011bm"]
    panels = [
        ("Raw frame (15 × 6 s)", stretch_image(*g["raw"])),
        ("Sky-subtracted frame", stretch_image(*g["subtracted"])),
        ("Stack of 16 frames (central region)", stretch_image(*g["stack"])),
    ]
    fig, axes = _plotting.figure(
        "web", target="web-640", ratio=0.40, mosaic=[["a", "b", "c"]], gridspec_kw={"wspace": 0.04}
    )
    for ax, (title, image) in zip(axes.values(), panels, strict=True):
        # datawiz: allowed as a relative brightness image; a color bar of asinh-stretched values gives the reader no quantity
        ax.imshow(image, origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for edge in ax.spines.values():
            edge.set_visible(False)
        ax.set_title(title, loc="left")
    _plotting.save(fig, str(out_dir / "infrared_reduction"), target="web-640", formats=("png",))


if __name__ == "__main__":
    main()
