# Time scales

Target module: `time/`.

In astronomy there is no single "time"; each scale rests on a different physical definition, and using the wrong scale produces an error of more than a minute [Eastman2010]. The equations are read from the cited source before they are coded.

## 1. Calendar and Julian date

- Julian Date (JD): the number of days since noon on 1 January 4713 BC. Modified JD: `MJD = JD − 2400000.5`.
- Gregorian ↔ JD conversion: an algorithm with integer arithmetic [Meeus1998, Section 7]. The Julian-Gregorian transition (15 October follows 4 October 1582) and negative years are test cases.
- Epochs: `J2000.0 = JD 2451545.0 (TT)`; the Julian year is 365.25 days; the Besselian epoch is for old catalogs.
- **Two-part representation.** JD is held as two `float64` values (`jd1 + jd2`). A single `float64` gives a resolution of about 40 µs near J2000; two parts bring this below a nanosecond [SOFA2023].

## 2. Scales

- **TAI:** atomic time; the continuous count of the SI second.
- **UTC:** civil time that differs from TAI by whole seconds; kept within 0.9 s of UT1 by leap seconds. `TAI − UTC` is read from the leap second table; the table is updated with IERS Bulletin C, and the copy embedded in the code has a last validity date. The behavior at the leap second instant (`23:59:60`) is tested separately.
- **UT1:** time tied to the rotation angle of the Earth. `UT1 − UTC` (DUT1) is read from IERS Bulletin A/B; it cannot be predicted, it is measured.
- **TT:** geocentric ephemeris time. `TT = TAI + 32.184 s`.
- **TDB:** barycentric dynamical time; the argument of ephemerides. `TDB − TT` is a periodic difference with an amplitude of about 1.7 ms; the low-precision single-term form and the full series are separate functions [Kaplan2005; SOFA2023].
- **TCG, TCB:** relativistic coordinate times; linearly related to TT and TDB [IERS2010, Section 10].
- **GPS time:** `TAI − 19 s`.
- **ΔT = TT − UT1:** polynomial approximations for historical periods; a prediction for the future.

Conversion graph: `UTC ↔ TAI ↔ TT ↔ TDB`, `UTC ↔ UT1`, `TT ↔ TCG`, `TDB ↔ TCB`. Each edge is a function; the conversion between two scales is built along the shortest path in the graph.

## 3. Sidereal time and Earth rotation angle

- Earth Rotation Angle: `ERA = 2π (0.7790572732640 + 1.00273781191135448 · Tu)`, `Tu = JD_UT1 − 2451545.0` [IERS2010, Section 5]. The coefficients are to be verified against the source.
- Greenwich mean sidereal time (GMST): the sum of ERA and a precession-dependent polynomial (IAU 2006); the old IAU 1982 form is for the low-precision level.
- Greenwich apparent sidereal time: `GAST = GMST + equation of the equinoxes`.
- Local sidereal time: `LST = GAST + longitude` (east positive).

## 4. Barycentric correction

Timing (variable star, transit, pulsar) and radial velocity measurements are reduced to the barycenter of the Solar System.

- **Rømer delay:** `Δt = (r · n̂) / c`; `r` is the observer's barycentric position, `n̂` the direction of the source. Its amplitude reaches up to 8.3 minutes.
- **Shapiro delay** and **Einstein delay** (TT → TDB).
- `BJD_TDB` is the standard output; the differences from `HJD` and `BJD_UTC` are documented [Eastman2010].
- Barycentric velocity correction: a simple projection for the 1 m/s level, the full relativistic form for 1 cm/s [WrightEastman2014].

These functions need the observer's barycentric position; they depend on the `ephemeris` module.

## 5. Event times

- Rise, set, meridian transit: standard altitude (−34′ for a star, −50′ for the solar limb), iterative solution [Meeus1998, Section 15].
- Twilight limits: civil −6°, nautical −12°, astronomical −18°.
- Lunar phase and illuminated fraction.

## Test cases

- Worked examples from the sources [Meeus1998, Section 7 and 12; SOFA2023 time scales cookbook].
- Agreement with ERFA and `astropy.time` at the `1e-9` s level for every conversion (provided the UT1 and leap second tables are the same).
- Property: the inverse of every conversion is the identity operation on a round trip; the scale differences are monotonic.
