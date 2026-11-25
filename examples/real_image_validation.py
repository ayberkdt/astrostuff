"""Test of source detection and astrometry on a real image against an independent catalog.

Image: 2MASS K-band cutout of the Flame Nebula (NGC 2024), 1 arcsec/pixel
(NASA SkyView). Reference: the 2MASS Point Source Catalog of the same field (IRSA).

Measured quantities:

1. Purity: the fraction of detected sources that have a counterpart in the catalog.
2. Catalog completeness: the fraction of catalog stars detected, by K magnitude.
3. Astrometry: the difference between positions computed with the WCS and catalog positions.
4. Photometric zero point: the difference between the aperture magnitude and the catalog K magnitude.
5. Artificial star completeness: the recovery fraction of stars added to the image.

The catalog was produced from the data of the same survey; this test
measures the correctness of the code, not that of 2MASS itself. SkyView
resamples the image to the requested grid; the position residuals include
this resampling.

Usage:
    python examples/real_image_validation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import _plotting
import numpy as np

from kozmos import photometry
from kozmos.archives import skyview, twomass
from kozmos.astrometry import matching, solve, triangles, wcs
from kozmos.observing import detector
from kozmos.statistics import robust

ROOT = Path(__file__).resolve().parents[1]
RA, DEC, FIELD_DEG, PIXELS = 85.42, -1.90, 0.15, 540
MATCH_RADIUS = 2.0  # arcsec
EDGE = 12  # pixels; catalog stars closer to the edge than this do not enter the completeness computation
APERTURE, ANNULUS = 4.0, (8.0, 12.0)
SEED = 20261125


def detect(image: np.ndarray, deblend: bool = True) -> photometry.SourceList:
    background, rms = photometry.background_map(image, cell=60)
    return photometry.detect_sources(
        image, background, rms, threshold_sigma=4.0, min_pixels=5, deblend_levels=32 if deblend else 0
    )


def main() -> None:
    image, header = skyview.download_cutout(
        RA, DEC, "2MASS-K", ROOT / "data" / "raw" / "skyview", size_deg=FIELD_DEG, pixels=PIXELS
    )
    catalog = twomass.point_sources(RA, DEC, FIELD_DEG * 0.75, ROOT / "data" / "raw" / "2mass")
    w = wcs.wcs_from_header(header)
    scale = abs(float(header["CDELT1"])) * 3600.0  # arcsec / pixel

    # Catalog: stars actually detected in K (k_cmsig filled) and falling inside the image.
    kx, ky = wcs.sky_to_pixel(w, catalog["ra"], catalog["dec"])
    inside = (kx > EDGE) & (kx < PIXELS - 1 - EDGE) & (ky > EDGE) & (ky < PIXELS - 1 - EDGE)
    valid = inside & np.isfinite(catalog["k_cmsig"])
    k_ra, k_dec, k_m, kx, ky = (d[valid] for d in (catalog["ra"], catalog["dec"], catalog["k_m"], kx, ky))

    detected = detect(image)
    s_ra, s_dec = wcs.pixel_to_sky(w, detected.x, detected.y)
    inside_source = (
        (detected.x > EDGE) & (detected.x < PIXELS - 1 - EDGE) & (detected.y > EDGE) & (detected.y < PIXELS - 1 - EDGE)
    )

    # 1. Purity: from the detections to the whole catalog (including sources with K upper limits).
    to_catalog = matching.cross_match(s_ra, s_dec, catalog["ra"], catalog["dec"], MATCH_RADIUS)
    purity = float(to_catalog.matched[inside_source].mean())

    # 2. Catalog completeness: from the catalog to the detections.
    to_detections = matching.cross_match(k_ra, k_dec, s_ra, s_dec, MATCH_RADIUS)
    bin_edges = np.arange(6.0, 16.01, 1.0)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    catalog_count = np.array(
        [np.sum((k_m >= a) & (k_m < b)) for a, b in zip(bin_edges[:-1], bin_edges[1:], strict=True)]
    )
    found = np.array(
        [
            np.sum(to_detections.matched[(k_m >= a) & (k_m < b)])
            for a, b in zip(bin_edges[:-1], bin_edges[1:], strict=True)
        ]
    )

    # 3. Astrometry: position difference in matched pairs [arcsec].
    pair = to_detections.matched
    d_ra = (s_ra[to_detections.index[pair]] - k_ra[pair]) * np.cos(np.radians(k_dec[pair])) * 3600.0
    d_dec = (s_dec[to_detections.index[pair]] - k_dec[pair]) * 3600.0

    # 4. Zero point and PSF width: from bright, isolated, unsaturated stars.
    isolated = np.array([np.sum(np.hypot(kx - x, ky - y) < 15.0) == 1 for x, y in zip(kx, ky, strict=True)])
    selection = pair & isolated & (k_m > 9.0) & (k_m < 12.5)
    flux, _ = photometry.aperture_photometry(image, kx[selection], ky[selection], APERTURE, annulus=ANNULUS)
    narrow, _ = photometry.aperture_photometry(image, kx[selection], ky[selection], 1.5, annulus=ANNULUS)
    wide, _ = photometry.aperture_photometry(image, kx[selection], ky[selection], 6.0, annulus=ANNULUS)
    good = (flux > 0) & (narrow > 0) & (wide > narrow)
    zero_points = k_m[selection][good] + 2.5 * np.log10(flux[good])
    zero_point = float(np.median(zero_points))
    zero_point_scatter = float(robust.mad_scale(zero_points))
    # In a Gaussian PSF the enclosed flux fraction at radius r is 1 − exp(−r²/2σ²): sigma from the ratio of the two apertures.
    ratio = np.median(narrow[good] / wide[good])
    sigmas = np.linspace(0.5, 4.0, 3501)
    model = (1 - np.exp(-(1.5**2) / (2 * sigmas**2))) / (1 - np.exp(-(6.0**2) / (2 * sigmas**2)))
    sigma_psf = float(sigmas[np.argmin(np.abs(model - ratio))])
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_psf

    # 5. Artificial star experiment: Gaussian stars of known magnitude are added to the same image and detected again.
    rng = np.random.default_rng(SEED)
    aperture_fraction = 1.0 - np.exp(-(APERTURE**2) / (2 * sigma_psf**2))
    artificial_mag = np.arange(10.5, 16.01, 0.5)
    artificial_fraction = []
    for m in artificial_mag:
        recovered = total = 0
        for _ in range(4):
            x = rng.uniform(EDGE, PIXELS - 1 - EDGE, 40)
            y = rng.uniform(EDGE, PIXELS - 1 - EDGE, 40)
            total_flux = 10 ** (-0.4 * (m - zero_point)) / aperture_fraction
            new = detect(image + detector.star_field(image.shape, x, y, np.full(40, total_flux), fwhm))
            distance = np.hypot(new.x[None, :] - x[:, None], new.y[None, :] - y[:, None]).min(axis=1)
            recovered += int(np.sum(distance < 1.5))
            total += 40
        artificial_fraction.append(recovered / total)
    artificial_fraction = np.array(artificial_fraction)

    # 6. Astrometric solution from the image: the WCS in the header is deliberately distorted (0.4° rotation, 0.4% scale,
    # 18 and −14 pixel shift) and re-solved from the catalog using only the detected sources; the result is compared with the header WCS.
    angle = np.radians(0.4)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    distorted = wcs.Wcs(crpix=w.crpix + np.array([18.0, -14.0]), crval=w.crval, cd=1.004 * rotation @ w.cd)
    grid_x, grid_y = np.meshgrid(np.linspace(0, PIXELS - 1, 5), np.linspace(0, PIXELS - 1, 5))

    def difference_from_header(candidate: wcs.Wcs) -> float:
        r1, d1 = wcs.pixel_to_sky(candidate, grid_x, grid_y)
        r2, d2 = wcs.pixel_to_sky(w, grid_x, grid_y)
        return float(np.max(np.hypot((r1 - r2) * np.cos(np.radians(d2)), d1 - d2)) * 3600.0)

    shifted, votes = solve.find_shift(
        distorted, detected.x, detected.y, catalog["ra"], catalog["dec"], max_shift_pix=40.0
    )
    solved = solve.refine_wcs(shifted, detected.x, detected.y, catalog["ra"], catalog["dec"], radius_arcsec=6.0)
    astrometric_solution = {
        "distortion": {"rotation_deg": 0.4, "scale_factor": 1.004, "shift_pix": [18.0, -14.0]},
        "distorted_wcs_max_error_arcsec": difference_from_header(distorted),
        "shift_votes": votes,
        "post_shift_max_error_arcsec": difference_from_header(shifted),
        "solution_max_error_arcsec": difference_from_header(solved.wcs),
        "stars_in_solution": solved.n_stars,
        "catalog_residual_rms_arcsec": solved.residual_rms_arcsec,
    }

    # 7. Blind solution: the WCS in the header is not used at all; only the pixel positions and fluxes of the detected
    # sources and the catalog of the same region are given (scale, orientation, parity and tangent point are unknown).
    blind, pairs = triangles.blind_solve(
        detected.x, detected.y, detected.flux, catalog["ra"], catalog["dec"], catalog["k_m"], n_brightest=30
    )
    astrometric_solution["blind_solution"] = {
        "pairs_assigned_in_triangle_match": pairs,
        "stars_in_solution": blind.n_stars,
        "max_difference_from_header_arcsec": difference_from_header(blind.wcs),
        "catalog_residual_rms_arcsec": blind.residual_rms_arcsec,
        "scale_arcsec_pix": float(np.sqrt(abs(np.linalg.det(blind.wcs.cd))) * 3600.0),
    }

    result = {
        "image": {
            "survey": "2MASS-K (SkyView)",
            "center_deg": [RA, DEC],
            "scale_arcsec_pix": scale,
            "pixels": PIXELS,
        },
        "catalog": {"total": int(len(catalog["ra"])), "k_detected_in_image": int(valid.sum())},
        "detected_sources": int(inside_source.sum()),
        "detected_sources_without_deblending": len(detect(image, deblend=False)),
        "match_radius_arcsec": MATCH_RADIUS,
        "purity": purity,
        "catalog_completeness": {
            "k_center": bin_centers.tolist(),
            "catalog_count": catalog_count.tolist(),
            "found": found.tolist(),
        },
        "astrometry_arcsec": {
            "pair_count": int(pair.sum()),
            "median_d_ra_cos_dec": float(np.median(d_ra)),
            "median_d_dec": float(np.median(d_dec)),
            "scatter_d_ra": float(robust.mad_scale(d_ra)),
            "scatter_d_dec": float(robust.mad_scale(d_dec)),
            "median_separation": float(np.median(to_detections.separation_arcsec[pair])),
        },
        "photometry": {
            "star_count": int(good.sum()),
            "zero_point_mag": zero_point,
            "scatter_mag": zero_point_scatter,
            "psf_fwhm_pix": fwhm,
        },
        "astrometric_solution": astrometric_solution,
        "artificial_stars": {
            "k": artificial_mag.tolist(),
            "recovery_fraction": artificial_fraction.tolist(),
            "seed": SEED,
        },
    }
    out_dir = ROOT / "figures"
    (out_dir / "real_image_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Figure: (a) completeness curves, (b) position residuals.
    fig, axes = _plotting.figure("web", target="web-640", ratio=0.48, mosaic=[["a", "b"]])
    a, b = axes["a"], axes["b"]
    populated = catalog_count >= 5
    a.plot(bin_centers[populated], 100.0 * found[populated] / catalog_count[populated], "o-", label="Catalog stars")
    a.plot(artificial_mag, 100.0 * artificial_fraction, "s--", label="Artificial stars")
    a.set_xlabel("K magnitude (Vega)")
    a.set_ylabel("Fraction detected (%)")
    a.set_ylim(0, 105)
    b.scatter(d_ra, d_dec, s=8, alpha=0.6, linewidths=0)
    b.axhline(0.0, color="0.6", lw=0.6, zorder=0)
    b.axvline(0.0, color="0.6", lw=0.6, zorder=0)
    b.set_xlim(-2.1, 2.1)
    b.set_ylim(-2.1, 2.1)
    b.set_aspect("equal")
    b.set_xlabel("Δα cos δ (arcsec)")
    b.set_ylabel("Δδ (arcsec)")
    _plotting.legend.place(a)
    _plotting.save(fig, str(out_dir / "real_image_validation"), target="web-640", formats=("png",))


if __name__ == "__main__":
    main()
