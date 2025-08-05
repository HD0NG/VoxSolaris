import pandas as pd
import pvlib
from pvlib.location import Location
from datetime import datetime, timedelta


def find_matching_azimuths_and_times(latitude, longitude, time_start, time_end, azimuths, freq='1min', tolerance=0.5):
    """
    Find one timestamp for each azimuth (within a tolerance) and return aligned lists.

    Parameters:
        latitude (float): Latitude of the location
        longitude (float): Longitude of the location
        time_start (datetime): Start of the time range (UTC or tz-aware)
        time_end (datetime): End of the time range
        azimuths (list of float): Target azimuths (degrees)
        freq (str): Time resolution for solar position calculations (default '1min')
        tolerance (float): Azimuth matching tolerance in degrees (default ±0.5°)

    Returns:
        Tuple[List[float], List[pd.Timestamp]]: matched azimuths and their timestamps
    """
    site = Location(latitude, longitude)
    times = pd.date_range(start=time_start, end=time_end, freq=freq, tz='UTC')
    solpos = site.get_solarposition(times)

    matched_azimuths = []
    timestamps = []

    for target_az in azimuths:
        mask = solpos['azimuth'].between(target_az - tolerance, target_az + tolerance)
        match = solpos[mask]
        if not match.empty:
            matched_azimuths.append(target_az)
            timestamps.append(match.index[0])  # First matching timestamp

    return matched_azimuths, timestamps



def find_closest_times_to_azimuths_for_date(date, latitude, longitude):
    matches = find_matching_azimuths_and_times(
        latitude=latitude,
        longitude=longitude,
        time_start=date,
        time_end=date+timedelta(days =1),
        azimuths=list(range(360,0,-1)),
        freq='1min'
    )

    return matches
