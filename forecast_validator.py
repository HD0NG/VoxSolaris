# --- PV Forecast Validator ---
#
# This module contains functions to evaluate the accuracy of a PV forecast
# against measured data. It is designed to be called by the calibration script.

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Import Timo's original modules
import miniPVforecast
import shadowmap
import csv_reader # We will modify this slightly

def calculate_model_error(shadow_matrix_path, pv_data_df):
    """
    Calculates the Root Mean Squared Error (RMSE) between a shadow-corrected
    forecast and measured PV data for a given day.

    Args:
        shadow_matrix_path (str): Path to the shadow matrix CSV file to use.
        pv_data_df (pd.DataFrame): DataFrame containing the measured PV data for a specific day.

    Returns:
        float: The calculated RMSE in Watts.
    """
    
    # --- This function replicates the core logic of test_plotting.py ---
    
    # 1. Modify the csv_reader to use the specified shadow matrix
    # This is a temporary modification to point the reader to the correct file
    original_get_shadowdata = csv_reader.get_shadowdata
    csv_reader.get_shadowdata = lambda dense=True: pd.read_csv(shadow_matrix_path, index_col=0)

    # 2. Calculate solar position and shading values
    azimuths, zeniths = miniPVforecast.get_solar_azimuth_zenit_fast(pv_data_df.index)
    pv_data_df["azimuth"] = azimuths
    pv_data_df["zenith"] = zeniths
    
    # Using a fixed shadowrange of 20 as a default, based on Timo's script
    shadowrange = 20 
    pv_data_df["shading_blur"] = shadowmap.add_box_blur_shading_to_pv_output_df(pv_data_df, shadowrange, max=False)

    # 3. Generate the shadow-corrected clearsky forecast
    df_cs_shade_blur = miniPVforecast.get_pvlib_shaded_clearsky_irradiance(pv_data_df.index, pv_data_df["shading_blur"])

    # 4. Prepare data for comparison
    # Align data and handle potential NaNs
    comparison_df = pd.DataFrame({
        'measured': pv_data_df["Energia MPP1 | Symo 8.2-3-M (1)"],
        'forecast': df_cs_shade_blur["output"]
    }).dropna()

    # Restore the original csv_reader function
    csv_reader.get_shadowdata = original_get_shadowdata

    if len(comparison_df) == 0:
        print("Warning: No valid data points for comparison. Returning infinite error.")
        return float('inf')

    # 5. Calculate and return the RMSE
    rmse = np.sqrt(mean_squared_error(comparison_df['measured'], comparison_df['forecast']))
    
    return rmse
