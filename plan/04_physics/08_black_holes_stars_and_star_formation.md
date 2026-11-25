# Black holes, stellar physics and star formation: implemented modules

Modules: `black_holes/`, `stars/`, `star_formation/`. All three were written on 25 November 2026. The source and verification status per function is in [../STATUS.md](../STATUS.md); this document records the scope, the sources read, and the discrepancies found while reading.

## Black holes (`black_holes`)

34 functions; all records of 29 of them were read from the source text.

**Scope.**

- Geometry: gravitational radius and Schwarzschild radius, photon sphere, critical impact parameter, angular diameter of the shadow [EHT2019]; innermost stable circular orbit (ISCO) and orbital frequency [Abramowicz2013].
- Kerr: ISCO, event horizon, ergosphere, photon orbit radii and the angular velocity of the horizon [Abramowicz2013; Page2005].
- Accretion: Eddington luminosity and mass flux, Salpeter time, Bondi radius and flux [Inayoshi2020]; thin disk temperature profile [Armitage2011]; tidal disruption radius and Hills mass [Komossa2015].
- Quantum: Hawking temperature, Bekenstein-Hawking entropy [Page2005].
- Binaries: chirp mass, frequency derivative, time to merger [Sathyaprakash2009].

**Findings made while reading.**

- The Kerr photon orbit formula in [Abramowicz2013], as printed, gives 4 gravitational radii for `a* = 1`; this is the retrograde orbit. For the prograde orbit the sign of the angle was changed and the result was tested against the source's own cubic equation.
- The ISCO formula in the same source is for the prograde orbit only. The retrograde sign comes from the paper [Bardeen1972], and that paper was not read; the record is in the "pending" state. Both signs were tested against the radial epicyclic condition.
- [Abramowicz2013] gives the Eddington luminosity as 1.2 × 10³⁸ erg/s per solar mass; the value computed with the Thomson cross section and the proton mass is 1.257 × 10³⁸. The difference is rounding; the test tolerance was set to 6%.
- With the values of [EHT2019], the shadow of M87* comes out as 37.86 microarcseconds; the value in the paper is 37.6.

**Pending records.** Gravitational redshift [Carroll2019], ISCO specific energy and the efficiency relation [Bardeen1972], evaporation time [Page1976]. These functions pass independent tests (energy minimum, integral by quadrature), but their source texts have not been opened.

**Limits.** For merging binaries, the ISCO frequency is the test-particle formula applied to the total mass; it is a rough estimate. There is no waveform, no post-Newtonian terms and no spinning binaries.

## Stellar physics (`stars`)

39 functions; all records of 34 of them were read from the source text.

**Scope.**

- Polytropes: numerical solution of the Lane-Emden equation, first zero, mass coefficient, central-to-mean density ratio, the closed-form solutions for `n = 0, 1, 5`; central density, pressure and temperature from mass and radius; gravitational energy [Liu1995; Bludman2010].
- Scaling: luminosity and effective temperature, surface gravity [Prsa2016]; free-fall, Kelvin-Helmholtz and nuclear timescales [McKee2007; Zinnecker2007]; Eddington limit and critical rotation [Paxton2013]; Eddington standard model [Bludman2010].
- Compact final stages: white dwarf equation of state coefficient and Chandrasekhar mass [Bludman2010].
- Initial mass functions: Salpeter, Kroupa, Chabrier; density, mean mass, mass fraction and sampling [Kroupa2001; Chabrier2003].

**Findings made while reading.**

- The mass coefficient column in Table I of [Bludman2010] agrees with the standard definition (2.018) only for `n = 3`; for `n = 1.5` and `n = 2` it gives a different quantity. The code uses only the first-zero and density-ratio columns.
- The derivative table in [Araujo2011] is wrong for `n = 1`; only the `n = 3` solar values were used in the test comparison.
- [Kroupa2001] gives only the proportions of the segments. The segments were joined continuously; with this assumption, the mean mass reported in the paper (0.36) and the number and mass fractions are recovered within rounding.
- The table coefficients of [Chabrier2003] leave a 0.5% jump at 1 solar mass; the published numbers were kept.

**Source values reproduced.** The `n = 3` solar model: central density 76.4 g/cm³, central pressure 1.24 × 10¹⁷ dyn/cm², central temperature 1.2 × 10⁷ K. Chandrasekhar mass 1.46 solar masses (`μ_e = 2`). The `log g` of the Sun is 4.438068.

**Pending records.** The original paper for the Salpeter slope [Salpeter1955] (the slope value was taken from the two papers that were read) and the nuclear timescale [Kippenhahn2012].

**Gaps.** The sourced coefficient of the nonrelativistic white dwarf mass-radius relation, the lower bound on the hydrostatic central pressure, the main sequence lifetime scaling. A full stellar structure solution and isochrones are later steps of the roadmap ([03_stellar_astrophysics.md](03_stellar_astrophysics.md)).

## Star formation (`star_formation`)

29 functions; the equation of every one was read from the source text.

**Scope.**

- Gas: isothermal sound speed, Mach number, number and mass density, column and surface density [Krumholz2015].
- Stability: Jeans length and mass, Bonnor-Ebert critical mass and radius, virial parameter, magnetic critical mass [McKee2007; Krumholz2015].
- Timescales and efficiency: free-fall and crossing times, star formation efficiency per free-fall time, accretion rate of an isothermal sphere collapse [McKee2007].
- Galactic scale: the Kennicutt-Schmidt law and its orbital-time form [Kennicutt1998], depletion time, Toomre Q for a gas disk, star formation rate calibrations from luminosity [Kennicutt2012].

**Findings made while reading.**

- The definition of the Jeans mass varies from source to source: [McKee2007] takes the mass of a sphere whose radius is half the Jeans length, [Krumholz2015] takes the mass of a cube whose side is half the Jeans length. The ratio is 4π/3; the function offers both.
- For the coefficient of the magnetic critical mass, [McKee2007] gives 0.17 and [Krumholz2015] gives 0.12 (from different numerical studies). The default is 0.17.
- The calibrations in Table 1 of [Kennicutt2012] give a star formation rate 0.68 times the value of [Kennicutt1998] for the same H-alpha luminosity (a difference in the initial mass function).
- The Bonnor-Ebert coefficients (1.182; density contrast 14.04) were recovered from the numerical solution of the equation.

**Gaps.** An effective velocity dispersion combining thermal and turbulent support, Toomre Q for a stellar disk, comparison of the laws applied to individual galaxies with observation.

## Testing approach

These three modules have no independent library oracle. Testing comes from four sources:

- Closed-form solutions and limits (Lane-Emden `n = 0, 1, 5`; uniform sphere; non-rotating black hole).
- Numbers printed in the sources (tables, worked values), with the tolerance tied to the number of printed digits.
- Independent calculation paths: obtaining the same quantity from another equation or from a numerical integral (disk luminosity, merger time, Salpeter growth).
- Invariants with `hypothesis`: monotonicity, scaling, round trip.

The gallery figures (`examples/physics_gallery.py`) are the direct output of these functions.
