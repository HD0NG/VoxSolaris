# Voxel-Based Cone-Casting Shadow Attenuation Matrix Generator
#
# This is a refactored version of the VoxSolaris script, designed to be
# imported and called as a function by other scripts (e.g., for calibration).

import os
import time
import math
import laspy
import numpy as np
import pandas as pd

# --- Default Configuration (can be overridden by the main function) ---
LIDAR_FILE_PATH = 'data/houselas_re_veg_2.las'
TARGET_COORDS_2D = np.array([532886, 6983516])
RELEVANT_CLASSES = {2, 3, 4, 5, 6}
GROUND_CLASS_CODE = 2
BUILDING_CLASS_CODE = 6
VEGETATION_CLASS_CODES = {3, 4, 5}
CLASS_PRIORITY = {6: 4, 5: 3, 4: 2, 3: 1, 2: 0, 0: -1}
VOXEL_SIZE = 0.5
AZIMUTH_STEPS = 360
ELEVATION_STEPS = 91
SOLAR_ANGULAR_RADIUS_DEG = 0.265
NUM_RAYS_PER_CONE = 16

# --- Main Function to be called by other scripts ---

def generate_shadow_matrix(k_coeffs, output_path):
    """
    Generates a shadow matrix for a given set of vegetation extinction coefficients.

    Args:
        k_coeffs (dict): A dictionary mapping vegetation class codes to their
                         k-values. e.g., {3: 0.7, 4: 0.5, 5: 0.3}
        output_path (str): The full path where the output CSV should be saved.
    """
    print(f"Generating shadow matrix with k-values: {k_coeffs}")
    
    # 1. Load and Prepare Data
    points, classifications = load_and_prepare_lidar(LIDAR_FILE_PATH, None, RELEVANT_CLASSES)
    if points is None: return

    # 2. Voxelize Scene
    classification_grid, density_grid, scene_min, grid_dims = voxelize_scene(points, classifications, VOXEL_SIZE)
    if classification_grid is None: return
    scene_max = scene_min + grid_dims * VOXEL_SIZE

    # 3. Define the analysis point
    analysis_point = get_analysis_point(points, classifications, scene_max)
    print(f"Analysis point set to: {analysis_point}")

    # 4. Run Simulation Loop
    simulation_results = run_simulation_loop(analysis_point, scene_min, grid_dims, classification_grid, density_grid, k_coeffs)
    
    # 5. Format and Save Matrix
    save_matrix(simulation_results, output_path)
    print(f"Shadow matrix saved to {output_path}")


# --- Helper Functions (Modules from the original script) ---

def load_and_prepare_lidar(file_path, bounding_box, relevant_classes):
    if not os.path.exists(file_path): return None, None
    las = laspy.read(file_path)
    points_xyz = np.vstack((las.x, las.y, las.z)).transpose()
    classifications = np.array(las.classification)
    mask = np.isin(classifications, list(relevant_classes))
    return points_xyz[mask], classifications[mask]

def voxelize_scene(points, classifications, voxel_size):
    scene_min = np.min(points, axis=0)
    scene_max = np.max(points, axis=0)
    grid_dims = np.ceil((scene_max - scene_min) / voxel_size).astype(int)
    
    classification_grid = np.zeros(grid_dims, dtype=np.int8)
    density_grid = np.zeros(grid_dims, dtype=np.float32)
    
    voxel_indices = np.floor((points - scene_min) / voxel_size).astype(int)
    for i in range(3):
        voxel_indices[:, i] = np.clip(voxel_indices[:, i], 0, grid_dims[i] - 1)

    for i in range(len(points)):
        idx = tuple(voxel_indices[i])
        current_class = classifications[i]
        if CLASS_PRIORITY.get(current_class, -1) > CLASS_PRIORITY.get(classification_grid[idx], -1):
            classification_grid[idx] = current_class
        if current_class in VEGETATION_CLASS_CODES:
            density_grid[idx] += 1
            
    building_indices = np.argwhere(classification_grid == BUILDING_CLASS_CODE)
    for ix, iy, iz in building_indices:
        for z_level in range(iz - 1, -1, -1):
            if CLASS_PRIORITY.get(classification_grid[ix, iy, z_level], -1) < CLASS_PRIORITY[BUILDING_CLASS_CODE]:
                classification_grid[ix, iy, z_level] = BUILDING_CLASS_CODE
            else:
                break

    voxel_volume = voxel_size ** 3
    vegetation_voxels = np.isin(classification_grid, list(VEGETATION_CLASS_CODES))
    density_grid[vegetation_voxels] /= voxel_volume
    
    return classification_grid, density_grid, scene_min, grid_dims

def get_analysis_point(points, classifications, scene_max):
    search_radius = 5.0
    mask = (
        (points[:, 0] > TARGET_COORDS_2D[0] - search_radius) &
        (points[:, 0] < TARGET_COORDS_2D[0] + search_radius) &
        (points[:, 1] > TARGET_COORDS_2D[1] - search_radius) &
        (points[:, 1] < TARGET_COORDS_2D[1] + search_radius)
    )
    points_near_target = points[mask]

    if len(points_near_target) > 0:
        target_z = np.max(points_near_target[:, 2]) + 0.01
    else:
        target_z = scene_max[2]
        
    return np.array([TARGET_COORDS_2D[0], TARGET_COORDS_2D[1], target_z])

def generate_cone_vectors(center_direction, radius_rad, num_samples):
    if np.allclose(np.abs(center_direction), [0, 0, 1]): v_up = np.array([0, 1, 0])
    else: v_up = np.array([0, 0, 1])
    u = np.cross(v_up, center_direction); u /= np.linalg.norm(u)
    v = np.cross(center_direction, u)
    cone_vectors = []
    for i in range(num_samples):
        r = radius_rad * np.sqrt((i + 0.5) / num_samples)
        theta = 2 * np.pi * 0.61803398875 * i
        offset_vec = r * (np.cos(theta) * u + np.sin(theta) * v)
        new_vec = center_direction + offset_vec
        new_vec /= np.linalg.norm(new_vec)
        cone_vectors.append(new_vec)
    return cone_vectors

def trace_ray_fast(ray_origin, ray_direction, scene_min, voxel_size, grid_dims):
    ray_pos = (ray_origin - scene_min) / voxel_size
    ix, iy, iz = int(ray_pos[0]), int(ray_pos[1]), int(ray_pos[2])
    step_x, step_y, step_z = (1 if d >= 0 else -1 for d in ray_direction)
    next_voxel_boundary_x = (ix + (step_x > 0)) * voxel_size + scene_min[0]
    next_voxel_boundary_y = (iy + (step_y > 0)) * voxel_size + scene_min[1]
    next_voxel_boundary_z = (iz + (step_z > 0)) * voxel_size + scene_min[2]
    t_max_x = (next_voxel_boundary_x - ray_origin[0]) / ray_direction[0] if ray_direction[0] != 0 else float('inf')
    t_max_y = (next_voxel_boundary_y - ray_origin[1]) / ray_direction[1] if ray_direction[1] != 0 else float('inf')
    t_max_z = (next_voxel_boundary_z - ray_origin[2]) / ray_direction[2] if ray_direction[2] != 0 else float('inf')
    t_delta_x = voxel_size / abs(ray_direction[0]) if ray_direction[0] != 0 else float('inf')
    t_delta_y = voxel_size / abs(ray_direction[1]) if ray_direction[1] != 0 else float('inf')
    t_delta_z = voxel_size / abs(ray_direction[2]) if ray_direction[2] != 0 else float('inf')
    while (0 <= ix < grid_dims[0]) and (0 <= iy < grid_dims[1]) and (0 <= iz < grid_dims[2]):
        yield (ix, iy, iz)
        if t_max_x < t_max_y:
            if t_max_x < t_max_z: ix += step_x; t_max_x += t_delta_x
            else: iz += step_z; t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z: iy += step_y; t_max_y += t_delta_y
            else: iz += step_z; t_max_z += t_delta_z

def calculate_transmittance(voxel_path_generator, classification_grid, density_grid, voxel_size, k_coeffs):
    transmittance = 1.0
    for ix, iy, iz in voxel_path_generator:
        voxel_class = classification_grid[ix, iy, iz]
        if voxel_class == BUILDING_CLASS_CODE: return 0.0
        if voxel_class in VEGETATION_CLASS_CODES:
            k_base = k_coeffs.get(voxel_class, 0.0)
            if k_base > 0:
                density = density_grid[ix, iy, iz]
                if density > 0:
                    k = k_base * density
                    transmittance *= math.exp(-k * voxel_size)
        if transmittance < 1e-6: return 0.0
    return transmittance

def run_simulation_loop(analysis_point, scene_min, grid_dims, classification_grid, density_grid, k_coeffs):
    azimuths = np.linspace(0, 2 * np.pi, AZIMUTH_STEPS, endpoint=False)
    elevations = np.linspace(0, np.pi / 2, ELEVATION_STEPS, endpoint=True)
    solar_radius_rad = np.deg2rad(SOLAR_ANGULAR_RADIUS_DEG)
    simulation_results = []
    for el in elevations:
        if el < 0.001: continue
        for az in azimuths:
            center_ray_direction = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
            cone_ray_vectors = generate_cone_vectors(center_ray_direction, solar_radius_rad, NUM_RAYS_PER_CONE)
            cone_transmittances = []
            for ray_vec in cone_ray_vectors:
                voxel_path_gen = trace_ray_fast(analysis_point, ray_vec, scene_min, VOXEL_SIZE, grid_dims)
                transmittance = calculate_transmittance(voxel_path_gen, classification_grid, density_grid, VOXEL_SIZE, k_coeffs)
                cone_transmittances.append(transmittance)
            avg_transmittance = np.mean(cone_transmittances)
            simulation_results.append({'azimuth': az, 'elevation': el, 'transmittance': avg_transmittance})
    return simulation_results

def save_matrix(simulation_results, output_path):
    df = pd.DataFrame(simulation_results)
    df['azimuth_deg'] = np.round(np.rad2deg(df['azimuth'])).astype(int)
    df['elevation_deg'] = np.round(np.rad2deg(df['elevation'])).astype(int)
    shadow_matrix_df = df.pivot_table(index='elevation_deg', columns='azimuth_deg', values='transmittance')
    
    # Convert from transmittance to attenuation (shading)
    shadow_matrix_df = 1 - shadow_matrix_df
    
    shadow_matrix_df = shadow_matrix_df.sort_index(axis=0).sort_index(axis=1)
    shadow_matrix_df.index = [f"Altitude_{i}" for i in shadow_matrix_df.index]
    shadow_matrix_df.columns = [f"Azimuth_{c}" for c in shadow_matrix_df.columns]
    if 'Azimuth_0' in shadow_matrix_df.columns:
        shadow_matrix_df['Azimuth_360'] = shadow_matrix_df['Azimuth_0']
    
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    shadow_matrix_df.to_csv(output_path, header=True, index=True)

