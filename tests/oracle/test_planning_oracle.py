import math

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from kozmos.observing import planning

pytestmark = pytest.mark.oracle

LATITUDE, LONGITUDE = math.radians(36.82), math.radians(30.34)  # 36.82° N, 30.34° E


def _astropy(ra, dec, jd, **atmosphere):
    location = EarthLocation(lat=LATITUDE * u.rad, lon=LONGITUDE * u.rad, height=0 * u.m)
    frame = AltAz(obstime=Time(jd, format="jd", scale="utc"), location=location, **atmosphere)
    return SkyCoord(ra * u.rad, dec * u.rad, frame="icrs").transform_to(frame)


def _separation_arcmin(az1, alt1, az2, alt2):
    c = np.sin(alt1) * np.sin(alt2) + np.cos(alt1) * np.cos(alt2) * np.cos(az1 - az2)
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 60.0


def test_visibility_against_astropy():
    ra, dec = math.radians(85.42), math.radians(-1.90)
    jd = 2461370.5 + np.linspace(0, 1, 49)  # 2026-11-25, at half-hour intervals
    ours = planning.visibility(ra, dec, LATITUDE, LONGITUDE, jd)
    oracle = _astropy(ra, dec, jd, pressure=0 * u.hPa)
    # Bias + precession are applied; the remaining difference is nutation, aberration, UT1 − UTC and polar motion.
    difference = _separation_arcmin(ours.azimuth_rad, ours.altitude_rad, oracle.az.rad, oracle.alt.rad)
    print(f"largest direction difference from astropy: {difference.max():.3f} arcminutes")
    assert difference.max() < 1.0
    assert np.all(np.isnan(ours.airmass[ours.altitude_rad <= 0]))
    assert np.nanmin(ours.airmass) >= 1.0
    # A calculation without precession is tens of times worse by the same measure (about 0.36° in 26 years).
    raw = planning.visibility(ra, dec, LATITUDE, LONGITUDE, jd, to_date=False)
    assert _separation_arcmin(raw.azimuth_rad, raw.altitude_rad, oracle.az.rad, oracle.alt.rad).max() > 10.0


def test_visibility_with_refraction_against_astropy():
    from kozmos.observing import refraction

    ra, dec = math.radians(85.42), math.radians(-1.90)
    jd = 2461370.5 + np.linspace(0, 1, 97)
    constants = refraction.refraction_constants(1013.25, 10.0, 0.5, 0.55)
    ours = planning.visibility(ra, dec, LATITUDE, LONGITUDE, jd, refraction=constants)
    oracle = _astropy(
        ra, dec, jd, pressure=1013.25 * u.hPa, temperature=10 * u.deg_C, relative_humidity=0.5, obswl=0.55 * u.micron
    )
    high = oracle.alt.deg > 10
    difference = _separation_arcmin(
        ours.azimuth_rad[high], ours.altitude_rad[high], oracle.az.rad[high], oracle.alt.rad[high]
    )
    assert high.sum() > 20 and difference.max() < 1.0
    # Refraction raises the object: about 5 arcminutes at an altitude of 10°.
    unrefracted = planning.visibility(ra, dec, LATITUDE, LONGITUDE, jd)
    lift_arcmin = np.degrees(ours.altitude_rad - unrefracted.altitude_rad)[high] * 60.0
    assert np.all(lift_arcmin > 0) and 3.0 < lift_arcmin.max() < 6.5
    # nan near the horizon, where the model is invalid.
    assert np.all(np.isnan(ours.altitude_rad[unrefracted.altitude_rad < math.radians(3.0)]))


def test_meridian_transit():
    from kozmos.coordinates.precession import icrs_to_mean_of_date

    ra, dec = math.radians(85.42), math.radians(-1.90)
    transit = planning.transit_jd(ra, LONGITUDE, 2461370.5, dec_rad=dec)
    assert 2461370.5 <= transit < 2461371.5
    moment = planning.visibility(ra, dec, LATITUDE, LONGITUDE, transit)
    assert abs(float(moment.hour_angle_rad)) < 1e-6  # precession in one day is 0.14 arcseconds: 6.6e-7 rad
    # At the meridian the altitude is 90° − |φ − δ(date)| and lower one hour before or after.
    dec_of_date = float(icrs_to_mean_of_date(ra, dec, transit)[1])
    assert float(moment.altitude_rad) == pytest.approx(math.pi / 2 - abs(LATITUDE - dec_of_date), abs=1e-9)
    neighbors = planning.visibility(ra, dec, LATITUDE, LONGITUDE, transit + np.array([-1, 1]) / 24)
    assert np.all(neighbors.altitude_rad < moment.altitude_rad)
    # With coordinates used as given the hour angle is zero (6e-7 rad remains because the GMST rate is taken as constant).
    raw = planning.transit_jd(ra, LONGITUDE, 2461370.5)
    assert abs(float(planning.visibility(ra, dec, LATITUDE, LONGITUDE, raw, to_date=False).hour_angle_rad)) < 1e-6


def test_time_above_horizon():
    dec = np.radians([-60.0, -20.0, 0.0, 40.0])
    rotation_hr = 24.0 / 1.00273781191135448  # one rotation of the Earth in SI hours: 23.9345
    assert rotation_hr == pytest.approx(23.9345, abs=5e-5)
    assert planning.time_above_horizon_hr(dec, 0.0) == pytest.approx(rotation_hr / 2)  # half a rotation at the equator
    assert float(planning.time_above_horizon_hr(math.radians(80.0), LATITUDE)) == pytest.approx(
        rotation_hr
    )  # circumpolar
    assert float(planning.time_above_horizon_hr(math.radians(-70.0), LATITUDE)) == 0.0  # star that never rises
    assert float(planning.time_above_horizon_hr(0.0, LATITUDE)) == pytest.approx(rotation_hr / 2)  # celestial equator
    # The duration gets shorter as the limit rises.
    assert planning.time_above_horizon_hr(0.0, LATITUDE, math.radians(30)) < planning.time_above_horizon_hr(
        0.0, LATITUDE
    )
