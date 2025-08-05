# --- VoxSolaris K-Value Calibration Engine (Optimized) ---
#
# This script automates finding the optimal extinction coefficients (k-values)
# for vegetation classes using the Nelder-Mead optimization algorithm. It is
# more efficient and precise than a simple grid search.
#
# Workflow:
# 1. Define a list of known clear-sky days for the analysis.
# 2. Randomly split these days into a 'calibration set' (for training the model)
#    and a 'validation set' (for testing its performance on unseen data).
# 3. Define an 'objective function' that takes k-values as input and returns
#    the average forecast error (RMSE) across all days in the calibration set.
# 4. Use `scipy.optimize.minimize` with the Nelder-Mead method to find the
#    k-values that minimize the output of the objective function.
# 5. After finding the optimal k-values, run a final validation step by
#    calculating the model's error on the validation set.
# 6. As a benchmark, calculate the error on the validation set using the original,
#    uncalibrated k-values.
# 7. Report the final k-values and compare the performance of the calibrated vs.
#    original models.

import numpy as np
import pandas as pd
from datetime import datetime
import time
import random
from scipy.optimize import minimize

# Import the refactored modules
import simulation_engine
import forecast_validator
import csv_reader

# --- 1. Define and Split the Dataset ---

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

# Global variable to hold pre-loaded PV data
PV_DATA_CACHE = {}

def objective_function(k_values_array):
    """
    This is the function the optimizer will try to minimize.
    It takes an array of k-values, runs the simulation and validation,
    and returns the average error (RMSE) for the calibration dataset.
    """
    # The optimizer passes a numpy array, so we convert it to our dict format
    k_coeffs = {
        3: k_values_array[0],  # k for Low Veg
        4: k_values_array[1],  # k for Medium Veg
        5: k_values_array[2]   # k for High Veg
    }
    
    print(f"\nTesting k-values: k3={k_coeffs[3]:.4f}, k4={k_coeffs[4]:.4f}, k5={k_coeffs[5]:.4f}")

    # a. Generate the shadow matrix for the current k-values
    shadow_matrix_path = 'results/shadow_matrix_results/temp_shadow_matrix.csv'
    simulation_engine.generate_shadow_matrix(k_coeffs, shadow_matrix_path)

    # b. Calculate the forecast error across all calibration days
    total_error = 0
    for day in CALIBRATION_DAYS:
        pv_data_for_day = PV_DATA_CACHE[day]
        error = forecast_validator.calculate_model_error(shadow_matrix_path, pv_data_for_day)
        total_error += error
    
    average_error = total_error / len(CALIBRATION_DAYS)
    print(f"--> Average RMSE on Calibration Set: {average_error:.4f} W")
    
    return average_error

def run_calibration():
    """
    Main function to perform the optimization and validation.
    """
    print("--- Starting k-value calibration process ---")
    start_time = time.time()

    # --- 2. Split data into calibration and validation sets ---
    random.shuffle(LIST_OF_CLEAR_LOOKING_DAYS)
    split_index = int(len(LIST_OF_CLEAR_LOOKING_DAYS) * 0.7) # 70/30 split
    global CALIBRATION_DAYS, VALIDATION_DAYS
    CALIBRATION_DAYS = LIST_OF_CLEAR_LOOKING_DAYS[:split_index]
    VALIDATION_DAYS = LIST_OF_CLEAR_LOOKING_DAYS[split_index:]

    print(f"Total clear days: {len(LIST_OF_CLEAR_LOOKING_DAYS)}")
    print(f"Using {len(CALIBRATION_DAYS)} days for calibration.")
    print(f"Using {len(VALIDATION_DAYS)} days for validation.")

    # --- Pre-load all necessary PV data into a cache ---
    print("Pre-loading PV data for all selected days...")
    full_pv_data = csv_reader.get_year(2021) # Assuming all days are in 2021
    for day in LIST_OF_CLEAR_LOOKING_DAYS:
        time_end = day + pd.Timedelta(days=1)
        # FIX: Use .copy() to prevent SettingWithCopyWarning downstream
        PV_DATA_CACHE[day] = full_pv_data[(full_pv_data.index >= day) & (full_pv_data.index < time_end)].copy()
    print("PV data loaded.")

    # --- 3. Run the optimization ---
    # Initial guess for the k-values [k_low, k_med, k_high]
    initial_guess = [0.5, 0.5, 0.5]
    
    # Bounds for the k-values (e.g., must be between 0 and 2)
    bounds = [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]

    print("\n--- Starting Nelder-Mead Optimization ---")
    result = minimize(
        objective_function,
        initial_guess,
        method='Nelder-Mead',
        bounds=bounds,
        options={'xatol': 1e-3, 'fatol': 1.0, 'disp': True} # xatol: tolerance on k-values, fatol: tolerance on RMSE
    )

    optimal_k_array = result.x
    optimal_k_coeffs = {3: optimal_k_array[0], 4: optimal_k_array[1], 5: optimal_k_array[2]}
    final_calibration_error = result.fun

    # --- 4. Perform final validation with OPTIMIZED k-values ---
    print("\n--- Optimization Complete. Running Final Validation ---")
    print(f"Optimal k-values found: {optimal_k_coeffs}")
    
    # Generate the final, optimal shadow matrix
    optimal_matrix_path = 'results/shadow_matrix_results/optimal_shadow_matrix.csv'
    simulation_engine.generate_shadow_matrix(optimal_k_coeffs, optimal_matrix_path)

    # Calculate error on the validation set
    validation_error_total = 0
    for day in VALIDATION_DAYS:
        pv_data_for_day = PV_DATA_CACHE[day]
        error = forecast_validator.calculate_model_error(optimal_matrix_path, pv_data_for_day)
        print(f"  - Validation RMSE for {day.date()}: {error:.4f} W")
        validation_error_total += error
    
    average_validation_error = validation_error_total / len(VALIDATION_DAYS)

    # --- 5. Benchmark against ORIGINAL k-values ---
    print("\n--- Comparing with Original K-Values ---")
    original_k_coeffs = {3: 0.7, 4: 0.5, 5: 0.3}
    print(f"Original k-values: {original_k_coeffs}")
    
    # Generate shadow matrix with original k's
    original_matrix_path = 'results/shadow_matrix_results/original_shadow_matrix.csv'
    simulation_engine.generate_shadow_matrix(original_k_coeffs, original_matrix_path)

    # Calculate error on the validation set using original k's
    original_error_total = 0
    for day in VALIDATION_DAYS:
        pv_data_for_day = PV_DATA_CACHE[day]
        error = forecast_validator.calculate_model_error(original_matrix_path, pv_data_for_day)
        print(f"  - Original Model RMSE for {day.date()}: {error:.4f} W")
        original_error_total += error
        
    average_original_error = original_error_total / len(VALIDATION_DAYS)

    # --- 6. Report the final results ---
    print("\n--- Final Results ---")
    print(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes")
    print("\nOptimal k-values:")
    for veg_class, k_val in optimal_k_coeffs.items():
        print(f"  - Vegetation Class {veg_class}: k = {k_val:.4f}")
    
    print(f"\nFinal Average RMSE on CALIBRATION set (with optimal k's): {final_calibration_error:.4f} W")
    print(f"\n--- Performance on Unseen VALIDATION Data ---")
    print(f"Average RMSE with OPTIMIZED k's: {average_validation_error:.4f} W")
    print(f"Average RMSE with ORIGINAL k's:  {average_original_error:.4f} W")
    
    # Save the optimal k-values
    with open("optimal_k_values.txt", "w") as f:
        f.write(str(optimal_k_coeffs))
    print("\nOptimal k-values saved to 'optimal_k_values.txt'")


if __name__ == '__main__':
    run_calibration()
