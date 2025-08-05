import os
from cmath import isnan
from datetime import datetime, timedelta
import numpy
import numpy as np
import pandas as pd
import csv_reader
import matplotlib.pyplot as plt
import miniPVforecast
import shadowmap



def print_full(x: pd.DataFrame):
    """
    Prints a dataframe without leaving any columns or rows out. Useful for debugging.
    """

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1400)
    pd.set_option('display.float_format', '{:10,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')





def generate_comparison_plot_for_shadowmap_and_observations(time_start, dayrange, save_png=True, shadowrange = 10, boxmax = False, output_folder ="figures"):
    """

    :param time_start: datetime object for the day which is to be plotted
    :param dayrange: if multiple days are to be plotted in the same plot, use dayrange > 1. Otherwise set to be 1
    :param save_png:  if you want to see a plot, set this to be False. If you want to save a .png file, set this to True
    :param shadowrange: box range used in reading shadow values. 0 is equal to point measurement. 50 reads the broad section of the sky towards which the panels are oriented towards
    :param boxmax: set this true and the shading value will be the maximum shading value within the shadow search box. Set to false and shading value will the be box average
    :param output_folder: set folder name here for .png files. Folder will be automatically created.
    :return:
    """
    ######################################################
    # feeding parameters to pvmodel
    tilt = 12
    azimuth = 170
    longitude = 27.648656
    latitude = 62.979848

    miniPVforecast.tilt = tilt
    miniPVforecast.azimuth = azimuth
    miniPVforecast.longitude = longitude
    miniPVforecast.latitude = latitude
    miniPVforecast.rated_power = 3.960 * 0.81

    ######################################################

    # reading measured pv generation
    df = csv_reader.get_year(time_start.year)

    # slicing measured pv generation
    #time_start = datetime(2021, 8, 12)
    time_end =  time_start + timedelta(days=dayrange)#datetime(2021, 10, 5)

    df = df[(df.index >= time_start) & (df.index <= time_end)]

    ######################################################

    # adding azimuths and zeniths to measured pv dataframe
    azimuths, zeniths = miniPVforecast.get_solar_azimuth_zenit_fast(df.index)
    df["azimuth"] = azimuths
    df["zenith"] = zeniths

    # adding shading values from local boxes
    search_range = shadowrange
    df["shading_blur"] = shadowmap.add_box_blur_shading_to_pv_output_df(df, search_range, max=boxmax)
    df["shading"] = shadowmap.add_box_blur_shading_to_pv_output_df(df, 0)

    # adding aoi to measured pv generation
    df["aoi"] = miniPVforecast.get_solar_angle_of_incidence_fast(df.index)

    ######################################################

    # generating clearsky values
    df_cs = miniPVforecast.get_pvlib_cleasky_irradiance(df.index)


    ######################################################


    df_noshading = df[df['shading_blur'] <= 0.05]
    df_someshading = df[df['shading_blur'] > 0.05]

    clearsky_normalized = df_cs["output"] / numpy.nanmax(df_cs["output"])
    measured_normalized = df["Energia MPP1 | Symo 8.2-3-M (1)"] / numpy.nanmax(df["Energia MPP1 | Symo 8.2-3-M (1)"])

    # delta_shade = (df_cs["output"] - df["Energia MPP1 | Symo 8.2-3-M (1)"]) / 4000
    # print_full(delta_shade)
    delta_shade = (clearsky_normalized - measured_normalized) / clearsky_normalized

    delta_shade = np.clip(delta_shade, 0, 1)

    df_cs_noshade = df_cs[df['shading_blur'] <= 0.05]
    df_cs_shade = df_cs[df['shading_blur'] > 0.05]

    # clearsky pv output with shadow removed from DNI, using point and box
    df_cs_shade_point = miniPVforecast.get_pvlib_shaded_clearsky_irradiance(df.index, df["shading"])
    df_cs_shade_blur = miniPVforecast.get_pvlib_shaded_clearsky_irradiance(df.index, df["shading_blur"])

    # shadow scatter for shading structure plot
    date_now = time_start.date()

    times = [] # x
    tilts = [] # y
    shadows = [] # alpha

    while date_now < time_end.date():
        times_a, tilts_a, shadows_a = shadowmap.get_time_tilt_shadow(date_now, latitude, longitude)
        times.extend(times_a)
        tilts.extend(tilts_a)
        shadows.extend(shadows_a)
        date_now += timedelta(days=1)
    ######################################################


    # init plot
    fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(14, 12))

    titletext = "Shadow matrix " + str(time_start.date())+ "\nShading search distance: " + str(search_range) + " deg"
    if boxmax:
        titletext += "\nshade = max within search distance box"
    else:
        titletext += "\nshade = average within search distance box"

    fig.suptitle(titletext)

    ######################################################

    # drawing plot 0
    ax0 = axes[0]

    ax0.scatter(df_noshading.index, df_noshading["Energia MPP1 | Symo 8.2-3-M (1)"], color="orange",
                label="Measured power(no box shading)", s=5)
    ax0.scatter(df_someshading.index, df_someshading["Energia MPP1 | Symo 8.2-3-M (1)"],color="black", label="Measured power(box shading)", s=5)
    ax0.plot(df_cs["time"], df_cs["output"], label="Clearsky estimate")


    ax0.set_ylabel("Power[W]")
    ax0.legend(loc="upper right")

    ######################################################
    # drawing plot 1
    ax1 = axes[1]

    ax1.plot(df_cs["time"], df_cs["output"], label="Clearsky estimate")
    ax1.plot(df_cs_shade_point["time"], df_cs_shade_point["output"], label="Clearsky DNI shade point", c="blue")
    ax1.plot(df_cs_shade_blur["time"], df_cs_shade_blur["output"], label="Clearsky DNI shade box",c="skyblue")
    ax1.plot(df.index, df["Energia MPP1 | Symo 8.2-3-M (1)"], label="Measured output", c="black")

    ax1.set_ylabel("Power[W]")
    ax1.legend(loc="upper right")

    ######################################################
    # drawing plot 2


    ax2 = axes[2]

    # delta values between power measurements and clearsky model with shading applied
    # both for point based shade and for larger box shade
    point_delta = df_cs_shade_point["output"]-df["Energia MPP1 | Symo 8.2-3-M (1)"]
    box_delta = df_cs_shade_blur["output"]-df["Energia MPP1 | Symo 8.2-3-M (1)"]

    # averages of point and box shade deltas
    avg_point_delta = np.average(point_delta[~numpy.isnan(point_delta)])
    avg_box_delta = np.average(box_delta[~numpy.isnan(box_delta)])

    # plotting visuals for model/measure deltas
    ax2.plot(df_cs_shade_point["time"], point_delta, label="Shade point\ndelta", c="blue")
    ax2.plot(df_cs_shade_point["time"], box_delta, label="Shade box\ndelta", c="black")

    # adding text with average point and box delta values for this specific run
    ax2.text(df_cs_shade["time"].iloc[0], 1000, "Avg point delta: " + str(round(avg_point_delta,1)) + "W\n"
        "Avg box delta: " + str(round(avg_box_delta,1)) + "W")


    ax2.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax2.plot()
    ax2.legend()
    ax2.set_ylabel("Model error")

    ######################################################
    # Sun elevation and shading structure plot

    ax3 = axes[3]
    ax3.scatter(times, tilts, color="grey", alpha=shadows, label="Shading structures", s=1)

    ax3.scatter(df_cs_noshade.index, df_cs_noshade["sun_elevation"], label="Sun path", c="orange", s=5)
    ax3.scatter(df_cs_shade.index, df_cs_shade["sun_elevation"], label="Sun path shaded", c="black", s=5)
    ax3.fill_between(df_cs.index, df_cs["sun_elevation"]-search_range,df_cs["sun_elevation"]+search_range, alpha=0.2, color="grey", label="Shading window")

    ax3.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax3.set_ylabel("Sun elevation[deg]")
    ax3.legend(loc="upper right")

    ######################################################
    # drawing shading and delta shading curves
    ax4 = axes[4]

    ax4.plot(df.index, df["shading"], label="Lidar measured\nshading point", color="grey")
    ax4.plot(df.index, df["shading_blur"], label="Lidar measured\nshading box", color="black")


    ax4.set_ylabel("Shading[0-1]")
    ax4.legend(loc="upper right")



    ######################################################
    # this bit here handles file saving
    if save_png:

        # folder name
        folder = output_folder+"/"

        # folder creation if it did not exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        # filename, sr stands for search range of the shadow map
        filename = "shadowmap-sr-"

        # boxmax or whether to use the max or average of shading detected within search box
        if boxmax:
            filename += "max-"
        filename += str(search_range) + "-"+ str(time_start.date())+ "+" + str(dayrange) + ".png"

        # file saving
        fullname = folder + filename
        plt.savefig(fullname,bbox_inches='tight')
    else:
        plt.show()

    plt.close()  # this should wipe matplotlib data from ram

    # returning fitness value for iterative parameter adjustment
    return avg_box_delta




# this list contains likely cloudfree days. Not confirmed, only based on visual observations of plot smoothness
# Marked WWP if Williams Wandji's plots suggest perfect cloudfree day for savilahti
list_of_clear_looking_days = [
    datetime(2021, 4, 11),
    datetime(2021, 4, 15),
    datetime(2021, 4, 16),
    datetime(2021, 4, 17),
    datetime(2021, 4, 18), # day 5
    datetime(2021, 4, 19),

    datetime(2021, 5, 12), #WWP
    datetime(2021, 5, 23),
    datetime(2021, 5, 30), #WWP

    datetime(2021, 6, 2),
    datetime(2021, 6, 4), # day 10 #WWP
    datetime(2021, 6, 5),
    datetime(2021, 6, 7),
    datetime(2021, 6, 9), #WWP
    datetime(2021, 6, 11), #WWP
    datetime(2021, 6, 29), # day 15 #WWP

    datetime(2021, 7, 3), #WWP
    datetime(2021, 7, 16), #WWP
    datetime(2021, 7, 26), #WWP
    datetime(2021, 7, 27), #WWP

    datetime(2021, 8, 5), # day 20 #WWP
    datetime(2021, 8, 7),
    datetime(2021, 8, 29),

    datetime(2021, 9, 10),
    datetime(2021, 9, 22),
    datetime(2021, 9, 27), # day 25
    datetime(2021, 9, 28),

    datetime(2021, 10, 3)
]



def generate_shadowrange_test_plots_for_single_day_box_avg(day):
    lowest_error = np.inf
    # shadowrange loop
    for sr in range(0, 51, 1):

        # calculating per measurement average delta between model and measured values
        # this also generates a plot in folder "figures4"
        result = generate_comparison_plot_for_shadowmap_and_observations(day, 1, shadowrange=sr, boxmax=False, save_png=True, output_folder="shadowrange_box_avg_test")

        # if new best fit found, print msg
        if result < lowest_error:
            lowest_error = result
            print("New best! "+ "Shadowrange: " + str(sr) + " fitness: " + str(round(result)))
        else:
            print("Shadowrange: " + str(sr) + " fitness: " + str(round(result)))


def generate_shadowrange_test_plots_for_single_day_boxmax(day):
    lowest_error = np.inf
    # shadowrange loop
    for sr in range(0, 51, 1):

        # calculating per measurement average delta between model and measured values
        # this also generates a plot in folder "figures4"
        result = generate_comparison_plot_for_shadowmap_and_observations(day, 1, shadowrange=sr, boxmax=True, save_png=True, output_folder="shadowrange_boxmax_test")
        print(result)

        if isnan(result):
            print("fit was nan for some reason?")
            print(result)
            continue

        # if new best fit found, print that new best was found
        if numpy.abs(result) < lowest_error:
            lowest_error = result
            print("New best! "+ "Shadowrange: " + str(sr) + " fitness: " + str(round(result)))
        else:
            print("Shadowrange: " + str(sr) + " fitness: " + str(round(result)))


def generate_shadowrange_test_plots_for_multiday(start_day, day_count):
    # this function generates plots for all days within given range.
    for d in range(day_count):
        runday = start_day + timedelta(days=d)
        generate_comparison_plot_for_shadowmap_and_observations(runday, dayrange=1, shadowrange=20, boxmax=False, save_png=True, output_folder="SR20_single_day_plots")
        print("Generated plot for day " + str(runday))


def generate_shadowrange_test_plots_for_chosen_days():
    # this function generates plots only for the selected good-looking days
    for day in list_of_clear_looking_days:
        generate_comparison_plot_for_shadowmap_and_observations(day, dayrange=1, shadowrange=20, boxmax=False, save_png=True, output_folder="SR20_selected_days_single_day_plots")
        print("Generated plot for day " + str(day))

def generate_shadowrange_test_plots_for_single_day(day):
    # this function generates plots for a single day
    generate_comparison_plot_for_shadowmap_and_observations(day, dayrange=1, shadowrange=20, boxmax=False, save_png=True, output_folder="SR20_single_day_plots")
    print("Generated plot for day " + str(day))


#generate_shadowrange_test_plots_for_multiday(datetime(2021, 4, 11), 200)

# generate_shadowrange_test_plots_for_chosen_days()

# these generate new plots of shadow range tests for given input day
#generate_shadowrange_test_plots_for_single_day_boxmax(list_of_clear_looking_days[25])
#generate_shadowrange_test_plots_for_single_day_box_avg(list_of_clear_looking_days[25])

#generate_shadowrange_test_plots_for_single_day()

import random
random_day = random.choice(list_of_clear_looking_days)
print("Randomly selected day for shadow range test: " + str(random_day))
generate_shadowrange_test_plots_for_single_day(random_day)