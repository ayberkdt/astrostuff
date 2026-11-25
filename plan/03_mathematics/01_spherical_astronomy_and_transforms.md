# Spherical astronomy and coordinate transforms

Target modules: `core/angles.py`, `core/vectors.py`, `coordinates/`.

The equations are written at planning level, in the form in which they are remembered; before each one is coded, it is read from the relevant section of the cited source and entered into the `@source` record with its equation number.

## 1. Angle arithmetic

- Units: radian (internal representation), degree, hour-minute-second, arcminute, arcsecond, milliarcsecond. `1 h = 15°`, `1″ = π/648000 rad`.
- Wrapping: reduction to the ranges `[0, 2π)` and `[-π, π)`; the sign behavior of `fmod` goes into the boundary tests.
- Sexagesimal string parsing and formatting: preserving the minus sign in the example `-00° 30′` is a known source of bugs [Meeus1998, Section 1].
- Small angle differences: `2 sin²(θ/2)` instead of `1 - cos θ`; the rationale is floating point cancellation [Goldberg1991].

## 2. Spherical trigonometry

In a spherical triangle with sides `a, b, c` and opposite angles `A, B, C` [Smart1977, Section 1]:

- Cosine: `cos a = cos b cos c + sin b sin c cos A`
- Sine: `sin a / sin A = sin b / sin B = sin c / sin C`
- Five-part: `sin a cos B = cos b sin c − sin b cos c cos A`

**Angular separation.** The cosine formula loses precision at small angles. In practice the Vincenty form is used:

```
Δσ = atan2( sqrt( (cos δ2 sin Δα)² + (cos δ1 sin δ2 − sin δ1 cos δ2 cos Δα)² ),
            sin δ1 sin δ2 + cos δ1 cos δ2 cos Δα )
```

The haversine form breaks down near antipodal points; all three forms are written and tested against each other in a property test. Position angle, and finding a target point from a position angle and a given separation, are in the same module.

## 3. Vector and rotation matrix method

Instead of chains of trigonometric formulas, unit vectors and rotation matrices are used; this is SOFA's approach [SOFA2023].

- Spherical → Cartesian: `r = (cos δ cos α, cos δ sin α, sin δ)`; `atan2` for the inverse transform.
- Basic rotations `R1(θ), R2(θ), R3(θ)`: the convention of rotating the axes (passive). The sign convention is defined in one place and pinned by a test.
- Rotation matrix properties: `Rᵀ R = I`, `det R = +1`. A property test checks this for every composite matrix.
- Small rotation vector and Rodrigues' formula.

## 4. Coordinate systems

Each system is defined by a fundamental plane and a reference direction.

- **Horizon (alt-az):** `sin h = sin φ sin δ + cos φ cos δ cos H`; the `atan2` form for azimuth. The convention that azimuth is measured from north through east is fixed; Meeus measures from south [Meeus1998, Section 13], and the difference is noted in the text.
- **Equatorial (α, δ):** hour angle `H = LST − α`.
- **Ecliptic (λ, β):** rotation `R1(ε)` about the obliquity `ε`.
- **Galactic (l, b):** a fixed rotation matrix from ICRS. The matrix is built from the position of the north galactic pole and the position angle of the galactic center; the source of the constants is the Hipparcos catalog [Perryman1997]; the full-precision values of the matrix were read from the ERFA file `icrs2g.c` and implemented [ERFA2023].
- **Supergalactic:** a fixed rotation from galactic.

## 5. Reduction chain to the observer

The chain from a catalog position (ICRS, epoch J2000, or J2016 for Gaia) to the observed direction [Kaplan2005; UrbanSeidelmann2013, Section 7; SOFA2023]:

1. **Space motion:** propagating the epoch with proper motion, parallax and radial velocity. The linear approximation and the full vector form are separate functions.
2. **Parallax:** displacement relative to the observer's barycentric position; its size is `π″ = 1/d_pc`.
3. **Light deflection:** deflection in the Sun's gravitational field; about 1.75″ at the solar limb, about 4 mas at 90° elongation.
4. **Annual aberration:** of order `v/c`, at most about 20.5″; the full relativistic form.
5. **Frame bias, precession, nutation:** from GCRS to the true equator and equinox of date. Models: IAU 2006 precession [Capitaine2003] and IAU 2000A/2000B nutation [IERS2010, Section 5]. The equinox-based and the CIO-based paths must give the same result.
6. **Earth rotation:** Earth Rotation Angle (ERA) or Greenwich apparent sidereal time.
7. **Polar motion:** `xp, yp` (IERS Bulletin A/B).
8. **Diurnal aberration and diurnal parallax:** the observer's geocentric position and velocity.
9. **Atmospheric refraction:** to the observed altitude.

Each step is a separate, pure function; the chain is a thin layer that combines them. Accuracy targets are defined at three levels:

- Level 1 (about 1′): low-precision formulas [Meeus1998]; sufficient for telescope pointing and planning.
- Level 2 (about 1 mas): IAU 2000B nutation, full aberration.
- Level 3 (microarcsecond): IAU 2006/2000A, agreement with ERFA at the `1e-10` rad level.

The order starts from Level 1; each level uses the previous one as a test oracle.

## 6. Precession and nutation

- **Precession:** the classical form (IAU 1976) with the angles `ζ, z, θ`, and the IAU 2006 form with the Fukushima-Williams angles [Capitaine2003]. The time argument is in Julian centuries in TT: `T = (JD_TT − 2451545.0) / 36525`.
- **Mean obliquity:** the IAU 2006 polynomial for `ε0`; the initial value is 84381.406″ [IERS2010, Section 5].
- **Nutation:** Moon-Sun and planetary terms, a trigonometric series in linear combinations of the Delaunay arguments. The 2000A model contains 1365 terms, the 2000B truncation 77 terms. The coefficient table is not typed by hand; it is generated from the electronic table published by the IERS, and a digest of the file is recorded.
- **Equation of the equinoxes:** `GAST − GMST = Δψ cos ε` plus the complementary terms.

## 7. Atmospheric refraction

- Plane-parallel approximation: `R ≈ (n − 1) tan z`.
- Bennett's formula: for apparent altitude `h` (degrees), `R′ = cot( h + 7.31/(h + 4.4) )` [Bennett1982]. The pressure and temperature correction is the factor `(P/1010 hPa) · (283 K/T)`.
- Near the horizon the accuracy of the formulas drops; the range of validity is written in the docstring [Young1994].
- Refraction depends on wavelength; atmospheric dispersion is a separate function for spectroscopy and astrometry.

## 8. Projection onto the image plane

The projections of the FITS WCS standard [GreisenCalabretta2002; CalabrettaGreisen2002]:

- Gnomonic (TAN): `ξ = cos δ sin Δα / cos c`, `η = (cos δ0 sin δ − sin δ0 cos δ cos Δα) / cos c`, `cos c = sin δ0 sin δ + cos δ0 cos δ cos Δα`. This is the projection of most CCD images.
- SIN, STG, ARC, ZEA (zenithal family); CAR, MER, CEA (cylindrical); AIT, MOL (all-sky).
- Linear transform from pixel to intermediate coordinates: `CRPIX`, the `CD` or `PC` + `CDELT` matrix.
- Distortion polynomial: the SIP convention [Shupe2005].

A round-trip test checks, for each projection, that the pixel → sky → pixel error stays below `1e-9` pixel.

## 9. Discretization on the sphere

- HEALPix: equal-area, hierarchical pixelization; ring and nested numbering; `nside`, pixel area `4π/(12 nside²)` [Gorski2005].
- Cross-matching: neighbor search within a radius using a spherical k-d tree or HEALPix bucketing.

## Implementation order

`angles` → `vectors` → spherical transforms → Level 1 precession/nutation → horizon transform and refraction → TAN projection → Level 2 → Level 3 → other projections → HEALPix.
