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
# average error for each model across the entire dataset. It also categorizes
# days based on whether the shadow model provided a performance improvement and
# provides a final summary of average errors per month.

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
    
    print("--- Starting PV Forecast Model Comparison ---")
    start_time = time.time()

    # --- Use the optimal shadow matrix from the calibration ---
    # This assumes the calibration has been run and the optimal matrix was saved.
    # shadow_matrix_path = 'results/shadow_matrix_results/optimal_shadow_matrix.csv'
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
    results = []

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
        print(f"  - Clearsky Model RMSE: {rmse_clearsky:.2f} W")

        # --- Prepare for shadow models ---
        azimuths, zeniths = miniPVforecast.get_solar_azimuth_zenit_fast(pv_data_for_day.index)
        pv_data_for_day["azimuth"] = azimuths
        pv_data_for_day["zenith"] = zeniths
        
        # --- Model 2: Your Direct Prediction (Point Lookup) ---
        pv_data_for_day["shading_direct"] = shadowmap.add_box_blur_shading_to_pv_output_df(pv_data_for_day, radius=0, max=False)
        direct_forecast = miniPVforecast.get_pvlib_shaded_clearsky_irradiance(pv_data_for_day.index, pv_data_for_day["shading_direct"])
        rmse_direct = calculate_rmse(pv_data_for_day["Energia MPP1 | Symo 8.2-3-M (1)"], direct_forecast['output'])
        print(f"  - Direct Shadow Model RMSE: {rmse_direct:.2f} W")

        # --- Model 3: Final Prediction (Box Blur) ---
        shadowrange = 20 # Using the default from Timo's plotting script
        pv_data_for_day["shading_blur"] = shadowmap.add_box_blur_shading_to_pv_output_df(pv_data_for_day, shadowrange, max=False)
        box_blur_forecast = miniPVforecast.get_pvlib_shaded_clearsky_irradiance(pv_data_for_day.index, pv_data_for_day["shading_blur"])
        rmse_box_blur = calculate_rmse(pv_data_for_day["Energia MPP1 | Symo 8.2-3-M (1)"], box_blur_forecast['output'])
        print(f"  - Final (Box Blur) Model RMSE: {rmse_box_blur:.2f} W")

        # Store results for this day
        results.append({
            'date': day.date(),
            'rmse_clearsky': rmse_clearsky,
            'rmse_direct_shadow': rmse_direct,
            'rmse_box_blur_shadow': rmse_box_blur
        })

    # --- Restore original csv_reader function ---
    csv_reader.get_shadowdata = original_get_shadowdata
        
    # --- Final Categorized Results ---
    results_df = pd.DataFrame(results)
    results_df['improvement'] = results_df['rmse_clearsky'] - results_df['rmse_direct_shadow']
    
    days_helped = results_df[results_df['improvement'] > 0]
    days_not_helped = results_df[results_df['improvement'] <= 0]

    print("\n--- Final Model Performance Summary ---")
    print(f"Processed {len(results_df)} clear-sky days.")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    
    print("\n" + "="*70)
    print("Days Where the Shadow Model HELPED (Lowered Error vs. Clearsky)")
    print("="*70)
    if not days_helped.empty:
        print(days_helped[['date', 'rmse_clearsky', 'rmse_direct_shadow', 'rmse_box_blur_shadow', 'improvement']].round(2).to_string(index=False))
        print(f"\nAverage RMSE Improvement on these days: {days_helped['improvement'].mean():.2f} W")
    else:
        print("No days found where the shadow model improved the forecast.")

    print("\n" + "="*70)
    print("Days Where the Shadow Model DID NOT HELP (Increased Error vs. Clearsky)")
    print("="*70)
    if not days_not_helped.empty:
        print(days_not_helped[['date', 'rmse_clearsky', 'rmse_direct_shadow', 'rmse_box_blur_shadow', 'improvement']].round(2).to_string(index=False))
        print(f"\nAverage RMSE Increase on these days: {-days_not_helped['improvement'].mean():.2f} W")
    else:
        print("No days found where the shadow model worsened the forecast.")
    print("="*70)

    # --- NEW: Monthly Average RMSE Summary ---
    results_df['month'] = pd.to_datetime(results_df['date']).dt.month
    monthly_avg_errors = results_df.groupby('month')[['rmse_clearsky', 'rmse_direct_shadow', 'rmse_box_blur_shadow']].mean()
    monthly_avg_errors.index.name = 'Month'
    
    print("\n" + "="*70)
    print("Average Model RMSE by Month")
    print("="*70)
    print(monthly_avg_errors.round(2).to_string())
    print("="*70)


if __name__ == '__main__':
    run_comparison()
