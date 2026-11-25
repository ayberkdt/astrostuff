"""Test of triangle matching and blind astrometric solution."""

import numpy as np
import pytest

from kozmos.astrometry import triangles
from kozmos.astrometry.wcs import Wcs, pixel_to_sky


def test_invariants_of_known_triangle():
    # 3-4-5 right triangle: shortest side 3, longest 5; R = 5/3. Vertex 1 is the common end of the shortest and longest sides;
    # the cosine of the angle there is adjacent side / hypotenuse = 3/5.
    u = triangles.build_triangles([0.0, 3.0, 3.0], [0.0, 0.0, 4.0])
    assert u.ratio[0] == pytest.approx(5.0 / 3.0) and u.cosine[0] == pytest.approx(0.6)
    assert u.log_perimeter[0] == pytest.approx(np.log(12.0))
    assert list(u.vertices[0]) == [0, 1, 2]  # vertex 1 = (0,0), vertex 2 = (3,0), vertex 3 = (3,4)
    # A thin triangle exceeding the ratio limit is discarded.
    assert triangles.build_triangles([0.0, 1.0, 30.0], [0.0, 0.0, 0.1]).ratio.size == 0


def test_invariants_preserved_under_transformation():
    rs = np.random.default_rng(211)
    x, y = rs.uniform(0, 100, 12), rs.uniform(0, 100, 12)
    a = np.radians(73.0)
    x2 = 2.7 * (np.cos(a) * x - np.sin(a) * y) + 40.0
    y2 = 2.7 * (np.sin(a) * x + np.cos(a) * y) - 15.0
    u, v = triangles.build_triangles(x, y), triangles.build_triangles(x2, y2)
    assert np.array_equal(u.vertices, v.vertices) and np.allclose(u.ratio, v.ratio) and np.allclose(u.cosine, v.cosine)
    assert np.array_equal(u.orientation, v.orientation) and np.allclose(v.log_perimeter - u.log_perimeter, np.log(2.7))
    # Mirror: R and C stay the same, the orientation flips.
    w = triangles.build_triangles(-x2, y2)
    assert (
        np.allclose(u.ratio, w.ratio)
        and np.allclose(u.cosine, w.cosine)
        and np.array_equal(u.orientation, -w.orientation)
    )


@pytest.mark.parametrize("mirror", [False, True])
def test_triangle_matching_finds_correspondences(mirror):
    rs = np.random.default_rng(212)
    x, y = rs.uniform(0, 1000, 22), rs.uniform(0, 1000, 22)
    a = np.radians(211.0)
    x2 = 0.37 * (np.cos(a) * x - np.sin(a) * y) + 5.0 + rs.normal(0, 0.02, 22)
    y2 = 0.37 * (np.sin(a) * x + np.cos(a) * y) - 9.0 + rs.normal(0, 0.02, 22)
    if mirror:
        x2 = -x2
    # The second list is shuffled, 4 points are removed, 4 false points are added.
    remaining = rs.permutation(22)[:18]
    x2 = np.concatenate([x2[remaining], rs.uniform(x2.min(), x2.max(), 4)])
    y2 = np.concatenate([y2[remaining], rs.uniform(y2.min(), y2.max(), 4)])
    i1, i2, votes, orientation = triangles.match_triangles(x, y, x2, y2)
    assert orientation == (-1 if mirror else 1)
    assert i1.size >= 12
    assert all(i2[k] < 18 and remaining[i2[k]] == i1[k] for k in range(i1.size))  # every assigned pair is correct
    assert np.all(np.diff(votes) <= 0)


def test_unrelated_lists_do_not_match():
    rs = np.random.default_rng(213)
    i1, _, _, _ = triangles.match_triangles(*rs.uniform(0, 1, (2, 15)), *rs.uniform(0, 1, (2, 15)))
    assert i1.size <= 3  # chance votes are cut off by the rule of votes falling by half


def test_blind_solution_without_prior_information():
    rs = np.random.default_rng(214)
    a = np.radians(128.0)
    scale = 0.6 / 3600.0
    cd = scale * np.array([[np.cos(a), np.sin(a)], [np.sin(a), -np.cos(a)]])  # rotated and mirrored
    truth = Wcs(crpix=np.array([400.0, 700.0]), crval=np.array([311.2, -37.4]), cd=cd)
    x, y = rs.uniform(0, 1500, 120), rs.uniform(0, 1500, 120)
    flux = 10 ** rs.uniform(2, 5, 120)
    ra, dec = pixel_to_sky(truth, x, y)
    magnitude = -2.5 * np.log10(flux) + rs.normal(0, 0.05, 120)
    # Detection: 85% of the stars + 25 false; catalog: 85% of the stars; position noise 0.1 pixel.
    s = rs.random(120) < 0.85
    xs = np.concatenate([x[s] + rs.normal(0, 0.1, s.sum()), rs.uniform(0, 1500, 25)])
    ys = np.concatenate([y[s] + rs.normal(0, 0.1, s.sum()), rs.uniform(0, 1500, 25)])
    detected_flux = np.concatenate([flux[s], 10 ** rs.uniform(2, 3, 25)])
    k = rs.random(120) < 0.85
    solution, pairs = triangles.blind_solve(xs, ys, detected_flux, ra[k], dec[k], magnitude[k])
    kx, ky = np.meshgrid([0.0, 750.0, 1499.0], [0.0, 750.0, 1499.0])
    r1, d1 = pixel_to_sky(solution.wcs, kx, ky)
    r2, d2 = pixel_to_sky(truth, kx, ky)
    assert np.max(np.hypot((r1 - r2) * np.cos(np.radians(d2)), d1 - d2)) * 3600.0 < 0.1
    assert pairs >= 8 and solution.n_stars > 60 and solution.residual_rms_arcsec < 0.2
    with pytest.raises(ValueError):
        triangles.blind_solve(
            xs, ys, detected_flux, ra[k] + 20.0 * rs.random(k.sum()), dec[k], rs.permutation(magnitude[k])
        )


def test_blind_solution_rejects_wrong_solution():
    # If the catalog does not cover the field of the image (here: a dense catalog of another region) a solution
    # can be built from chance triangle matches; the validation step must reject it.
    rs = np.random.default_rng(215)
    for _ in range(5):
        x, y = rs.uniform(0, 1000, 60), rs.uniform(0, 1000, 60)
        ra, dec = 150.0 + rs.uniform(-0.1, 0.1, 400), 20.0 + rs.uniform(-0.1, 0.1, 400)
        with pytest.raises(ValueError):
            triangles.blind_solve(x, y, rs.uniform(1, 100, 60), ra, dec, rs.uniform(8, 16, 400))
