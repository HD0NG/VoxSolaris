# --- PV Forecast Model Comparison ---
#
# This script performs a comparative analysis of three different PV forecast models
# against the measured, real-world power output. The goal is to quantify the
# improvement provided by each stage of shadow correction.
#
# The three models being tested are:
# 1. Clearsky Model: An ideal forecast with no shadow or cloud correction.
# 2. Direct Shadow Model: The clearsky forecast corrected by a direct, single-point
#    lookup in the shadow matrix.
# 3. Final Prediction (Box Blur Model): The clearsky forecast corrected by
#    averaging a small area ("box blur") of the shadow matrix, which is the
#    method used in the final plots.
#
# The script iterates through all known clear-sky days, calculates the Root Mean
# Squared Error (RMSE) for each model on each day, and then reports the
# average error for each model across the entire dataset.

import numpy as np
import pandas as pd
from datetime import datetime
import time
from sklearn.metrics import mean_squared_error

# Import the project's modules
import miniPVforecast
import shadowmap
import csv_reader

# This list is taken from test_plotting.py and contains likely cloud-free days.
LIST_OF_CLEAR_LOOKING_DAYS = [
    datetime(2021, 4, 11), datetime(2021, 4, 15), datetime(2021, 4, 16),
    datetime(2021, 4, 17), datetime(2021, 4, 18), datetime(2021, 4, 19),
    datetime(2021, 5, 12), datetime(2021, 5, 23), datetime(2021, 5, 30),
    datetime(2021, 6, 2), datetime(2021, 6, 4), datetime(2021, 6, 5),
    datetime(2021, 6, 7), datetime(2021, 6, 9), datetime(2021, 6, 11),
    datetime(2021, 6, 29), datetime(2021, 7, 3), datetime(2021, 7, 16),
    datetime(2021, 7, 26), datetime(2021, 7, 27), datetime(2021, 8, 5),
    datetime(2021, 8, 7), datetime(2021, 8, 29), datetime(2021, 9, 10),
    datetime(2021, 9, 22), datetime(2021, 9, 27), datetime(2021, 9, 28),
    datetime(2021, 10, 3)
]

def calculate_rmse(measured, forecast):
    """Calculates the Root Mean Squared Error between two series."""
    # Ensure alignment and drop any missing values
    comparison_df = pd.DataFrame({'measured': measured, 'forecast': forecast}).dropna()
    if len(comparison_df) == 0:
        return np.nan
    return np.sqrt(mean_squared_error(comparison_df['measured'], comparison_df['forecast']))

def run_comparison():
    """
    Main function to run the three models and compare their performance.
    """
    print("--- Starting PV Forecast Model Comparison ---")
    start_time = time.time()

    # --- Use the optimal shadow matrix from the calibration ---
    # This assumes the calibration has been run and the optimal matrix was saved.
    shadow_matrix_path = 'results/shadow_matrix_results/shadow_attenuation_matrix_conecasting.csv'
    
    # Temporarily modify csv_reader to load the correct matrix and fix the index
    original_get_shadowdata = csv_reader.get_shadowdata
    def load_and_fix_shadow_matrix(dense=True):
        df = pd.read_csv(shadow_matrix_path, index_col=0)
        df.index = df.index.str.replace('Altitude_', '').astype(int)
        return df
    csv_reader.get_shadowdata = load_and_fix_shadow_matrix
    
    # --- Load PV data ---
    print("Loading historical PV data for 2021...")
    full_pv_data = csv_reader.get_year(2021)
    
    # --- Store results for each model ---
    errors_clearsky = []
    errors_direct_shadow = []
    errors_box_blur_shadow = []

    # --- Loop through each clear day ---
    for day in LIST_OF_CLEAR_LOOKING_DAYS:
        print(f"\nProcessing day: {day.date()}...")
        
        # Get the measured data for the current day
        time_end = day + pd.Timedelta(days=1)
        pv_data_for_day = full_pv_data[(full_pv_data.index >= day) & (full_pv_data.index < time_end)].copy()
        
        if pv_data_for_day.empty:
            print("  -> No data for this day. Skipping.")
            continue

        # --- Model 1: Clearsky Forecast ---
        clearsky_forecast = miniPVforecast.get_pvlib_cleasky_irradiance(pv_data_for_day.index)
        rmse_clearsky = calculate_rmse(pv_data_for_day["Energia MPP1 | Symo 8.2-3-M (1)"], clearsky_forecast['output'])
        errors_clearsky.append(rmse_clearsky)
        print(f"  - Clearsky Model RMSE: {rmse_clearsky:.2f} W")

        # --- Prepare for shadow models ---
        azimuths, zeniths = miniPVforecast.get_solar_azimuth_zenit_fast(pv_data_for_day.index)
        pv_data_for_day["azimuth"] = azimuths
        pv_data_for_day["zenith"] = zeniths

        # --- Model 2: Final Prediction (Box Blur) ---
        shadowrange = 20 # Using the default from Timo's plotting script
        pv_data_for_day["shading_blur"] = shadowmap.add_box_blur_shading_to_pv_output_df(pv_data_for_day, shadowrange, max=False)
        box_blur_forecast = miniPVforecast.get_pvlib_shaded_clearsky_irradiance(pv_data_for_day.index, pv_data_for_day["shading_blur"])
        rmse_box_blur = calculate_rmse(pv_data_for_day["Energia MPP1 | Symo 8.2-3-M (1)"], box_blur_forecast['output'])
        errors_box_blur_shadow.append(rmse_box_blur)
        print(f"  - Final (Box Blur) Model RMSE: {rmse_box_blur:.2f} W")
        
        # --- Model 3: Your Direct Prediction (Point Lookup) ---
        # We use add_box_blur_shading_to_pv_output_df with radius=0, which is equivalent to a direct point lookup
        pv_data_for_day["shading_direct"] = shadowmap.add_box_blur_shading_to_pv_output_df(pv_data_for_day, radius=0, max=False)
        direct_forecast = miniPVforecast.get_pvlib_shaded_clearsky_irradiance(pv_data_for_day.index, pv_data_for_day["shading_direct"])
        rmse_direct = calculate_rmse(pv_data_for_day["Energia MPP1 | Symo 8.2-3-M (1)"], direct_forecast['output'])
        errors_direct_shadow.append(rmse_direct)
        print(f"  - Direct Shadow Model RMSE: {rmse_direct:.2f} W")

    # --- Restore original csv_reader function ---
    csv_reader.get_shadowdata = original_get_shadowdata
        
    # --- Final Averaged Results ---
    avg_error_clearsky = np.nanmean(errors_clearsky)
    avg_error_direct = np.nanmean(errors_direct_shadow)
    avg_error_box_blur = np.nanmean(errors_box_blur_shadow)

    print("\n--- Final Averaged Model Comparison ---")
    print(f"Processed {len(errors_clearsky)} clear-sky days.")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print("-" * 40)
    print(f"1. Average Clearsky Model RMSE:      {avg_error_clearsky:.2f} W")
    print(f"2. Average Direct Shadow Model RMSE: {avg_error_direct:.2f} W")
    print(f"3. Average Final (Box Blur) RMSE:    {avg_error_box_blur:.2f} W")
    print("-" * 40)
    
    improvement_over_clearsky = avg_error_clearsky - avg_error_box_blur
    print(f"The final model provides an average improvement of {improvement_over_clearsky:.2f} W (RMSE) over the base clearsky forecast.")

if __name__ == '__main__':
    run_comparison()
