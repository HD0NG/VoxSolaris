import numpy
import numpy as np

import csv_reader
import sandbox1


def get_tilt_azimuth(tilt, azimuth):
    shadow = csv_reader.get_shadowdata(dense=True)
    value = shadow.loc[tilt, 'Azimuth_'+str(azimuth)]
    return value


def get_time_tilt_shadow(date, latitude, longitude):
    shadow = csv_reader.get_shadowdata(dense=True)

    azimuths, timestamps = sandbox1.find_closest_times_to_azimuths_for_date(date, latitude, longitude)

    times = []
    tilts = []
    values = []

    #print("azimuths_ in get time tilt shadow")
    #print(azimuths)

    for i in range(len(azimuths)):
        time = timestamps[i]
        azimuth = azimuths[i]

        for tilt in range(0, 90,1):
            value = shadow.loc[tilt, 'Azimuth_' + str(azimuth)]
            times.append(time)
            tilts.append(tilt)
            values.append(value)

    return times, tilts, values


def add_box_blur_shading_to_pv_output_df(df, radius, max=False):
    """
    This function returns a list of shading values based on sun azimuth and zenith angles from given dataframe.
    Shadow is read from a box with x-y intervals of azimuth-radius to azimuth + radius, zenith -radius, zenith +radius

    By default values are box averages. Use max=true for box maximums.
    """


    # reading skyview matrix
    shadow = csv_reader.get_shadowdata(dense=True)

    # reading azimuth and zenith from df
    azimuths = df["azimuth"]
    zeniths = df["zenith"]

    # rounding to 5 degree grid
    azimuths = (1 * numpy.round(azimuths / 1)).astype(int)
    sun_elevations = (1 * numpy.round((90 - zeniths) / 1)).astype(int)

    # output shadings go here
    shadings = []

    # reading shading values for azimuths and zeniths in the DF
    for i in range(len(azimuths)):
        az = azimuths.iloc[i]
        sun_elevation = sun_elevations.iloc[i]


        shadings_for_this_point = []

        for ele_delta in range(-radius, radius+1, 1):
            for azi_delta in range(-radius, radius+1, 1):
                box_elevation = sun_elevation+ ele_delta

                if 0 < box_elevation < 90:
                    shadings_for_this_point.append(shadow.loc[box_elevation, 'Azimuth_' + str(az)])
                else:
                    shadings_for_this_point.append(1)


        if len(shadings_for_this_point) > 0:
            if max:
                shadings.append(np.max(shadings_for_this_point))
            else:
                shadings.append(np.average(shadings_for_this_point))
        else:
            shadings.append(1)



    return shadings


def add_shading_to_pv_output_df(df):

    # reading skyview matrix
    shadow = csv_reader.get_shadowdata(dense=True)

    # reading azimuth and zenith from df
    azimuths = df["azimuth"]
    zeniths = df["zenith"]

    # rounding to 5 degree grid
    azimuths = (1*numpy.round(azimuths/1)).astype(int)
    sun_elevations = (1 * numpy.round((90-zeniths)/1)).astype(int)



    # output shadings go here
    shadings = []

    # reading shading values for azimuths and zeniths in the DF
    for i in range(len(azimuths)):
        az = azimuths[i]
        sun_elevation = sun_elevations[i]

        if sun_elevation < 0:
            shadings.append(1)
        else:
            shadings.append(shadow.loc[sun_elevation, 'Azimuth_' + str(az)])

    df["shading"] = shadings


    return df
