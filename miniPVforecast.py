"""
Based on FMI open pv forecast
@tsalola
"""
import datetime
import math
import pandas as pd
import pvlib
from pvlib import location, irradiance
import numpy as np


"""
This file contains all the functions required by the FMI pv model.


Model steps
1. Calculating solar irradiance values for DNI, DHI, GHI for given times and geolocation

2. Calculating irradiance on plane of array from DNI, DHI and GHI irradiances using known planel angles.

3. Calculating reflective losses from DNI, DHI and GHI and how much is absorbed

4. Calculating sum of irradiance absorbed

5. Estimating panel temperature based on air temp and absorbed irradiance.

6. Estimating output based on absorbed irradiance and modeled panel temperature.


"""


#### SIMULATED INSTALLATION PARAMETERS BELOW:
# coordinates
latitude = 64.3800
longitude = 23.1671

# panel angles
tilt = 10 # degrees. Panel flat on the roof would have tilt of 0. Wall mounted panels have tilt of 90.
azimuth = 235 # degrees, north is 0 degrees, east 90. Clockwise rotation

# rated installation power in kW, PV output at standard testing conditions
rated_power = 1 # unit kW

# ground albedo near solar panels, 0.25 is PVlib default. Has to be in range [0,1], typical values [0.1, 0.4]
# grass is 0.25, snow 0.8, worn asphalt 0.12. Values can be found from wikipedia https://en.wikipedia.org/wiki/Albedo
albedo = 0.151

# module elevation, measured from ground
module_elevation = 4 # unit meters

# dummy wind speed(meter per second) value, this will be used if wind speed from fmi open is not used
wind_speed = 2

# air temp in Celsius, this will be used if temp from fmi open is not used
air_temp = 20


#################

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


################# ASTRONOMICAL CALCULATIONS BELOW THIS LINE, SYSTEM PARAMETERS ABOVE

def get_solar_azimuth_zenit_fast(dt: datetime)-> (float, float):
    """
    Returns apparent solar zenith and solar azimuth angles in degrees.
    :param dt: time to compute the solar position for.
    :return: azimuth, zenith
    """

    # panel location and installation parameters from config file
    panel_latitude = latitude
    panel_longitude = longitude

    # panel location object, required by pvlib
    panel_location = location.Location(panel_latitude, panel_longitude)

    # solar position object
    solar_position = panel_location.get_solarposition(dt)

    # apparent zenith and azimuth, Using apparent for zenith as the atmosphere affects sun elevation.
    # apparent_zenith = Sun zenith as seen and observed from earth surface
    # zenith = True Sun zenith, would be observed if Earth had no atmosphere
    solar_apparent_zenith = solar_position["apparent_zenith"]
    solar_azimuth = solar_position["azimuth"]

    return solar_azimuth, solar_apparent_zenith


def get_solar_angle_of_incidence_fast(dt:datetime)-> float:
    """
    Returns angle of light reaching the pv panel surface. Limited to range 0-90.
    """
    solar_azimuth, solar_apparent_zenith = get_solar_azimuth_zenit_fast(dt)

    # angle of incidence, angle between direct sunlight and solar panel normal
    angle_of_incidence = irradiance.aoi(tilt, azimuth, solar_apparent_zenith, solar_azimuth)

    # restricting AOI values as projection functions do not expect AOI higher than 90. Should never be lower than 0 but setting a limit anyways
    angle_of_incidence = angle_of_incidence.clip(lower=0, upper=90)

    return angle_of_incidence

def get_air_mass_fast(time: datetime)-> float:
    """
    Generates value for air mass using pvlib default model(kastenyoung1989).
    This value tells us the relative thickness of atmosphere between sun and the PV panels.
    :param time: python datetime
    :return: air mass value, may return nans if AOI is over 90
    """
    solar_zenith = get_solar_azimuth_zenit_fast(time)[1]
    air_mass = pvlib.atmosphere.get_relative_airmass(solar_zenith)
    return air_mass




################# TRANSPOSITIONS BELOW THIS LINE

def __project_dni_to_panel_surface_using_time_fast(dni: float, dt: datetime)-> float:
    angle_of_incidence = get_solar_angle_of_incidence_fast(dt)
    output = np.abs(dni * np.cos(np.radians(angle_of_incidence)))
    return output

def __project_dhi_to_panel_surface_perez_fast(time: datetime, dhi: float, dni: float)-> float:
    """
    Alternative dhi model,
    Calculated internally by pvlib, pvlib documentation at:
    https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.perez.html
    """

    # function parameters
    dni_extra = pvlib.irradiance.get_extra_radiation(time)

    # this should take sun-earth distance variation into account
    # empirical constant 1366.1 should work nearly as well

    # installation angles
    surface_tilt = tilt
    surface_azimuth = azimuth

    # sun angles
    solar_azimuth, solar_zenith = get_solar_azimuth_zenit_fast(time)

    # air mass
    airmass = get_air_mass_fast(time)

    dhi_perez = pvlib.irradiance.perez(surface_tilt, surface_azimuth,dhi, dni, dni_extra,  solar_zenith, solar_azimuth, airmass, return_components=False)

    return dhi_perez

def __project_ghi_to_panel_surface(ghi: float, albedo=albedo)-> float:
    """
    Equation from
    https://pvpmc.sandia.gov/modeling-guide/1-weather-design-inputs/plane-of-array-poa-irradiance/calculating-poa-irradiance/poa-ground-reflected/

    Uses ground albedo and panel angles to estimate how much of the sunlight per 1m² of ground is radiated towards solar
    panel surfaces.
    :param ghi: Ground reflected solar irradiance.
    :return: Ground reflected solar irradiance hitting the solar panel surface.
    """
    step1 = (1.0-math.cos(np.radians(tilt)))/2
    step2 = ghi*albedo * step1
    return step2

def irradiance_df_to_poa_df(irradiance_df:pd.DataFrame)-> pd.DataFrame:


    #print_full(irradiance_df)
    # handling dni and dhi
    irradiance_df["dni_poa"] = __project_dni_to_panel_surface_using_time_fast(irradiance_df["dni"], irradiance_df.index)
    irradiance_df["dhi_poa"] = __project_dhi_to_panel_surface_perez_fast(irradiance_df.index, irradiance_df["dhi"], irradiance_df["dni"])

    # and finally ghi
    if "albedo" in irradiance_df.columns:
        irradiance_df["ghi_poa"] = __project_ghi_to_panel_surface(irradiance_df["ghi"], irradiance_df["albedo"])
    else:
        irradiance_df["ghi_poa"] = __project_ghi_to_panel_surface(irradiance_df["ghi"])

    # adding the sum of projections to df as poa
    irradiance_df["poa"] = irradiance_df["dhi_poa"] + irradiance_df["dni_poa"] + irradiance_df["ghi_poa"]

    return irradiance_df

###############


def add_reflection_corrected_poa_components_to_df(df: pd.DataFrame)-> pd.DataFrame:

    # these two values depend on panel angles and total radiation AND NOT THE ANGLE OF THE SUN
    dhi_reflection_value = __dhi_reflected()
    ghi_reflection_value = __ghi_reflected()

    #df["AOI"] = astronomical_calculations.get_solar_angle_of_incidence_fast(df.index)
    df["dni_rc"] = (1-__dni_reflected(df.index))*df["dni_poa"]
    df["dhi_rc"] = (1-dhi_reflection_value)*df["dhi_poa"]
    df["ghi_rc"] = (1-ghi_reflection_value)*df["ghi_poa"]

    return df

def __dni_reflected(dt: datetime)-> float:
    """
    Computes a constant in range [0,1] which represents how much of the direct irradiance is reflected from panel
    surfaces.
    :param dt: datetime
    :return: reflected radiation in range [0,1]

    F_B_(alpha) in "Calculation of the PV modules angular losses under field conditions by means of an analytical model"
    """

    a_r = 0.159

    AOI = get_solar_angle_of_incidence_fast(dt)

    # upper section of the fraction equation
    upper_fraction = math.e ** (-np.cos(np.radians(AOI)) / a_r) - math.e ** (-1.0 / a_r)
    # lower section of the fraction equation
    lower_fraction = 1.0 - math.e ** (-1.0 / a_r)

    # fraction or alpha_BN or dni_reflected
    dni_reflected = upper_fraction / lower_fraction

    return dni_reflected


def __ghi_reflected()-> float:
    """
    Computes a constant in range [0,1] which represents how much of ground reflected irradiation is reflected away from
    solar panel surfaces. Note that this is constant for an installation.
    :return: [0,1] float, 0 no light reflected, 1 no light absorbed by panels.

    F_A(beta) in "Calculation of the PV modules angular losses under field conditions by means of an analytical model"

    """

    # constants, these are from
    c1 = 4.0 / (3.0 * math.pi)

    c2 = -0.074
    a_r = 0.159
    panel_tilt = np.radians(tilt)  # theta_T

    # equation parts, part 1 is used 2 times
    part1 = math.sin(panel_tilt) + (panel_tilt - math.sin(panel_tilt)) / (1.0 - math.cos(panel_tilt))

    part2 = c1 * part1 + c2 * (part1 ** 2.0)
    part3 = (-1.0 / a_r) * part2

    ghi_reflected = math.e ** part3

    return ghi_reflected


def __dhi_reflected()-> float:
    """
    Computes a constant in range [0,1] which represents how much of atmospheric diffuse light is reflected away from
    solar panel surfaces. Constant for an installation. Almost a 1 to 1 copy of __ghi_reflected except
    "pi -" addition to part1 and "1-cos" to "1+cos" replacement in part1 as well.
    :return: [0,1] float, 0 no light reflected, 1 no light absorbed by panels.

    F_D(beta) in "Calculation of the PV modules angular losses under field conditions by means of an analytical model"
    """
    # constants

    c1 = 4.0 / (math.pi * 3.0)
    c2 = -0.074
    a_r = 0.159
    panel_tilt = np.radians(tilt)  # theta_T
    pi = math.pi

    # equation parts, part 1 is used 2 times
    part1 = math.sin(panel_tilt) + (pi - panel_tilt - math.sin(panel_tilt)) / (1.0 + math.cos(panel_tilt))

    part2 = c1 * part1 + c2 * (part1 ** 2.0)
    part3 = (-1.0 / a_r) * part2

    dhi_reflected = math.e ** part3

    return dhi_reflected

def add_reflection_corrected_poa_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds reflection corrected POA value to dataframe with name "poa_ref_cor"
    :param df:
    :return:
    """

    df["poa_ref_cor"] = df["dni_rc"]+ df["dhi_rc"]+ df["ghi_rc"]


    return df

################ PANEL TEMPERATURE ESTIMATION BELOW

def add_estimated_panel_temperature(df:pd.DataFrame)-> pd.DataFrame:
    """
    Adds an estimate for panel temperature based on wind speed, air temperature and absorbed radiation.
    If air temperature, wind speed or absorbed radiation columns are missing, aborts.
    If columns exists but temperature function returns nan due to faulty input, uses air temperature which should always
    be present in df.
    :param df:
    :return:
    """

    if "T" not in df.columns:
        df["T"] = 20
    if "wind" not in df.columns:
        df["wind"] = 2

    def helper_add_panel_temp(df):
        estimated_temp = temperature_of_module(df["poa_ref_cor"], df["wind"], module_elevation, df["T"])
        if math.isnan(estimated_temp):
            return df["T"]
        else:
            return estimated_temp

    # applying helper function to dataset and storing result as a new column
    df["module_temp"] = df.apply(helper_add_panel_temp, axis=1)

    return df

def temperature_of_module(absorbed_radiation: float, wind: float, module_elevation: float, air_temperature: float) ->float:
    """
    :param absorbed_radiation: radiation hitting solar panel after reflections are accounted for in W
    :param wind: wind speed in meters per second
    :param module_elevation: module elevation from ground, in meters
    :param air_temperature: air temperature at 2m in Celsius
    :return: module temperature in Celsius

    King 2004 model
    D.~King, J.~Kratochvil, and W.~Boyson,
    Photovoltaic Array Performance Model Vol. 8,
    PhD thesis (Sandia Naitional Laboratories, 2004).
    """

    # two empirical constants
    constant_a = -3.47
    constant_b = -0.0594

    # wind is sometimes given as west/east components

    # wind speed at model elevation, assumes 0 speed at ground, wind speed vector len at 2m and forms a
    # curve which describes the wind speed transition from 0 to 10m wind speed to higher
    wind_speed = (module_elevation / 10) ** 0.1429 * wind

    module_temperature = absorbed_radiation * math.e ** (constant_a + constant_b * wind_speed) + air_temperature

    return module_temperature


################ OUTPUT ESTIMATION BELOW

def add_output_to_df(df: pd.DataFrame)-> pd.DataFrame:
    """
    Checker function for testing if required parameters exist in DF, if they do, add output to DF.
    :param df: Pandas dataframe with required columns for absorbed irradiance and panel temperature.
    :return: Input DF with PV system output column.
    """

    # new PV output models can be added here if needed.


    if "poa_ref_cor" not in df.columns:
        print("column poa_ref_cor not found in dataframe, output can not be simulated")
    if "module_temp" not in df.columns:
        print("module temperature variable \"module_temp\" not found in dataframe")


    # filtering negative values out
    df.loc[df['poa_ref_cor'] < 0, 'poa_ref_cor'] = 0

    # this line makes sure the output estimation is not called when per w² radiation is below 0.1W. If the radiation is
    # this low, the system would not produce any power and values of 0.0 cause issues as the output model contains
    # logarithms
    df['output'] = df.apply(lambda row: 0.0 if row['poa_ref_cor'] < 0.1 else __estimate_output(row['poa_ref_cor'], row['module_temp']),axis=1 )

    # filling nans
    df['output'] = df['output'].fillna(0.0)

    return df


def __estimate_output(absorbed_radiation: float, panel_temp: float)-> float:

    """
    Huld 2010 model
    T.~Huld, R.~Gottschalg, H.~G. Beyer, and M.~Topič,
    Mapping the performance of PV modules, effects of module type and
    data averaging, Solar Energy, 84 324--338 (2010).

    The Huld 2010 model may not perform very well at low irradiances. Solar panel efficiency tends to decrease
    when irradiance per m² of panel surface is less than 300W. This is taken into account by the Huld model, but
    depending on panel types and other factors, the Huld model may actually suggest a negative efficiency.
    During testing, this occurred at m² irradiances of less than 5W but this is temperature dependant.

    As finding reliable information on PV panel output at lower than 100W/m² irradiances proved to be challenging,
    it may very well be possible that either efficiency varies vastly depending on the panel type or efficiency
    actually goes to zero.

    A minimum efficiency of 50% has been set. Adjust [min_efficiency] if this is not to your liking. This will not
    affect the performance of the model a lot, but it will eliminate negative output estimates and thus a limit.

    :param absorbed_radiation: Solar irradiance absorbed by m² of solar panel surface.
    :param panel_temp: Estimated solar panel temperature.
    :return: Estimated system output in watts.
    """

    min_efficiency = 0.3
    max_efficiency = 1.0

    # huld 2010 constants
    k1 = -0.017162
    k2 = -0.040289
    k3 = -0.004681
    k4 = 0.000148
    k5 = 0.000169
    k6 = 0.000005

    # hud et al equation:

    # main equation:
    # output = rated_power*nrad*efficiency

    # Helpers:
    # nrad = absorbed_radiation/1000
    # Tdiff = module_temp-25C
    # efficiency = 1+ k1*ln(nrad)
    # + k2*ln(nrad)²
    # + Tdiff*(k3+k4*ln(nrad) + k5*ln(nrad)²)
    # + k6*Tdiff²

    #print(absorbed_radiation)

    nrad = absorbed_radiation / 1000.0
    Tdiff = panel_temp - 25
    rk_power = rated_power * 1000.0
    base = 1

    part_k1 = k1 * np.log(nrad)
    part_k2 = k2*(np.log(nrad)**2)
    part_k3k4k5 = Tdiff*(k3+k4*np.log(nrad) + k5*(np.log(nrad)**2))
    # T*(-0.004681+0.000148*log(x) + 0.000169*log(x)²)
    part_k6 = k6*(Tdiff**2)

    efficiency = base + part_k1+ part_k2 + part_k3k4k5 + part_k6
    efficiency = np.maximum(efficiency, min_efficiency)
    efficiency = np.minimum(efficiency, max_efficiency)
    # huld efficiencies can be negative, fixing that here

    #print(absorbed_radiation)
    #print("efficiency:" + str(efficiency))

    output = rk_power*nrad*efficiency

    return output

########## CLEARSKY ESTIMATION BELOW


def __get_irradiance_pvlib(latitude, longitude, timestamps, mod="ineichen")-> pd.DataFrame:
    """
    PVlib based clear sky irradiance modeling
    :param date: Datetime object containing a date
    :param mod: One of the 3 models supported by pvlib
    :return: Dataframe with ghi, dni, dhi. Or only GHI if using haurwitz
    """

    # creating site data required by pvlib poa
    site = location.Location(latitude, longitude, )

    # measurement frequency, for example "15min" or "60min"
    measurement_frequency =  "1min"

    # measurement count, 1440 minutes per day
    #measurement_count = 1440
    """
    times = pd.date_range(start=date_start,
                          end=date_end,  # year + day for which the irradiance is calculated
                          freq=measurement_frequency,  # take measurement every 60 minutes
                          tz=site.tz)  # timezone
    """

    # creating a clear sky and solar position entities
    clearsky = site.get_clearsky(timestamps, model=mod)

    # adds index as a separate time column, for some reason this is required as even a named index is not callable
    # with df[index_name] and df.index is not supported by function apply structures
    clearsky.insert(loc=0, column="time", value=clearsky.index)

    # returning clearsky irradiance df
    return clearsky


def get_pvlib_cleasky_irradiance(timestamps):

    clearsky = __get_irradiance_pvlib(latitude, longitude, timestamps)

    df = process_external_component_table(clearsky)

    azimuths, zeniths = get_solar_azimuth_zenit_fast(df.index)
    df["sun_elevation"] =90- zeniths
    df["azimuth"] = azimuths

    return df


def get_pvlib_shaded_clearsky_irradiance(timestamps, shades):
    clearsky = __get_irradiance_pvlib(latitude, longitude, timestamps)

    clearsky["shade"] = shades
    clearsky["dni"] = clearsky["dni"]*(1-clearsky["shade"])

    df = process_external_component_table(clearsky)

    azimuths, zeniths = get_solar_azimuth_zenit_fast(df.index)
    df["sun_elevation"] = 90 - zeniths
    df["azimuth"] = azimuths

    return df



################ EXTERNAL DATA PROCESSING BELOW


def process_external_component_table(df):

    # step 2. project irradiance components to plane of array:
    data = irradiance_df_to_poa_df(df)

    # step 3. simulate how much of irradiance components is absorbed:
    data = add_reflection_corrected_poa_components_to_df(data)

    # step 4. compute sum of reflection-corrected components:
    data = add_reflection_corrected_poa_to_df(data)

    # step 5. estimate panel temperature based on wind speed, air temperature and absorbed radiation
    data = add_estimated_panel_temperature(data)

    # step 6. estimate power output
    data = add_output_to_df(data)


    return data


#generate_output_table()