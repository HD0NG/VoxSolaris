# --- VoxSolaris: Final Shadow Matrix Generator ---
#
# This script is a standalone tool for generating a high-quality shadow
# attenuation matrix using the fully developed VoxSolaris simulation engine.
# It incorporates all advanced features, including:
#   - Voxelization of LiDAR data
#   - Solid building model generation via extrusion
#   - Realistic soft shadows using a cone-casting engine
#   - Support for multiple vegetation classes with distinct k-values
#
# This script is intended for generating a final matrix after the optimal
# k-values have been determined through the calibration process.

import os
import time
import math
import laspy
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
# Set all simulation parameters in this section.

# 1. Input Data and Scene Definition
LIDAR_FILE_PATH = 'data/houselas_re_veg_2.las'
OUTPUT_DIRECTORY = 'results/shadow_matrix_results'
OUTPUT_FILENAME = 'final_shadow_matrix.csv'
BOUNDING_BOX = None

# --- Target coordinates for analysis ---
TARGET_COORDS_2D = np.array([532886, 6983516])

# ASPRS Standard Classification Codes
RELEVANT_CLASSES = {2, 3, 4, 5, 6}
GROUND_CLASS_CODE = 2
BUILDING_CLASS_CODE = 6
VEGETATION_CLASS_CODES = {3, 4, 5}

# Priority for voxel classification (higher number = higher priority)
CLASS_PRIORITY = {6: 4, 5: 3, 4: 2, 3: 1, 2: 0, 0: -1}

# 2. Voxelization Parameters
VOXEL_SIZE = 0.5

# 3. Solar Position & Cone-Casting Simulation Parameters
AZIMUTH_STEPS = 360  # 1-degree steps
ELEVATION_STEPS = 91 # 1-degree steps (0-90 inclusive)
SOLAR_ANGULAR_RADIUS_DEG = 0.265
NUM_RAYS_PER_CONE = 16

# 4. Ray-Casting and Attenuation Parameters
# --- IMPORTANT: Set your desired k-values here ---
# You can use your original estimates or the optimal values found during calibration.
VEGETATION_EXTINCTION_COEFFICIENTS = {
    3: 0.7,  # k for Low Vegetation
    4: 0.5,  # k for Medium Vegetation
    5: 0.3   # k for High Vegetation
}


# --- MODULE 1: DATA INGESTOR & PREPARER ---

def load_and_prepare_lidar(file_path, bounding_box=None, relevant_classes=None):
    print(f"Loading LiDAR data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None, None
    try:
        las = laspy.read(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None
    points_xyz = np.vstack((las.x, las.y, las.z)).transpose()
    classifications = np.array(las.classification)
    if relevant_classes:
        mask = np.isin(classifications, list(relevant_classes))
        points_xyz = points_xyz[mask]
        classifications = classifications[mask]
    if len(points_xyz) == 0:
        print("Warning: No points remaining after filtering.")
        return None, None
    print(f"Data loaded and filtered. {len(points_xyz)} points remaining.")
    return points_xyz, classifications


# --- MODULE 2: VOXELIZER ---

def voxelize_scene(points, classifications, voxel_size):
    print("Voxelizing the scene...")
    scene_min = np.min(points, axis=0)
    scene_max = np.max(points, axis=0)
    grid_dims = np.ceil((scene_max - scene_min) / voxel_size).astype(int)
    print(f"Voxel grid dimensions: {grid_dims}")

    classification_grid = np.zeros(grid_dims, dtype=np.int8)
    density_grid = np.zeros(grid_dims, dtype=np.float32)
    
    voxel_indices = np.floor((points - scene_min) / voxel_size).astype(int)
    for i in range(3):
        voxel_indices[:, i] = np.clip(voxel_indices[:, i], 0, grid_dims[i] - 1)

    print("Populating classification and density grids from points...")
    for i in range(len(points)):
        idx = tuple(voxel_indices[i])
        current_class = classifications[i]
        if CLASS_PRIORITY.get(current_class, -1) > CLASS_PRIORITY.get(classification_grid[idx], -1):
            classification_grid[idx] = current_class
        if current_class in VEGETATION_CLASS_CODES:
            density_grid[idx] += 1
            
    print("Extruding building footprints to create solid models...")
    building_indices = np.argwhere(classification_grid == BUILDING_CLASS_CODE)
    for ix, iy, iz in building_indices:
        for z_level in range(iz - 1, -1, -1):
            if CLASS_PRIORITY.get(classification_grid[ix, iy, z_level], -1) < CLASS_PRIORITY[BUILDING_CLASS_CODE]:
                classification_grid[ix, iy, z_level] = BUILDING_CLASS_CODE
            else:
                break
    print("Building extrusion complete.")

    voxel_volume = voxel_size ** 3
    vegetation_voxels = np.isin(classification_grid, list(VEGETATION_CLASS_CODES))
    density_grid[vegetation_voxels] /= voxel_volume
    
    print("Voxelization complete.")
    return classification_grid, density_grid, scene_min, grid_dims


# --- MODULE 3: RAY-CASTING & CONE-CASTING ENGINE ---

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


# --- MAIN EXECUTION SCRIPT ---

if __name__ == '__main__':
    start_time = time.time()
    
    # 1. Load and Prepare Data
    points, classifications = load_and_prepare_lidar(LIDAR_FILE_PATH, BOUNDING_BOX, RELEVANT_CLASSES)
    if points is None: exit()

    # 2. Voxelize Scene
    classification_grid, density_grid, scene_min, grid_dims = voxelize_scene(points, classifications, VOXEL_SIZE)
    if classification_grid is None: exit()
    scene_max = scene_min + grid_dims * VOXEL_SIZE

    # 3. Define the analysis point
    search_radius = 5.0
    mask = (
        (points[:, 0] > TARGET_COORDS_2D[0] - search_radius) & (points[:, 0] < TARGET_COORDS_2D[0] + search_radius) &
        (points[:, 1] > TARGET_COORDS_2D[1] - search_radius) & (points[:, 1] < TARGET_COORDS_2D[1] + search_radius)
    )
    points_near_target = points[mask]
    if len(points_near_target) > 0:
        target_z = np.max(points_near_target[:, 2]) + 0.01
    else:
        print(f"Warning: No LiDAR points found near target. Using scene max Z.")
        target_z = scene_max[2]
    analysis_point = np.array([TARGET_COORDS_2D[0], TARGET_COORDS_2D[1], target_z])
    print(f"Analysis point set to: {analysis_point}")

    # 4. Run Simulation Loop
    azimuths = np.linspace(0, 2 * np.pi, AZIMUTH_STEPS, endpoint=False)
    elevations = np.linspace(0, np.pi / 2, ELEVATION_STEPS, endpoint=True)
    solar_radius_rad = np.deg2rad(SOLAR_ANGULAR_RADIUS_DEG)
    simulation_results = []
    total_directions = len(azimuths) * (len(elevations) - 1)
    print(f"\n--- Starting Cone-Casting Simulation ---")
    print(f"Casting {NUM_RAYS_PER_CONE} rays per cone for {total_directions} positions...")
    
    current_direction = 0
    for el in elevations:
        if el < 0.001: continue
        for az in azimuths:
            current_direction += 1
            if current_direction % 1000 == 0:
                 print(f"Processing direction {current_direction}/{total_directions}...")
            
            center_ray_direction = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
            cone_ray_vectors = generate_cone_vectors(center_ray_direction, solar_radius_rad, NUM_RAYS_PER_CONE)
            cone_transmittances = []
            for ray_vec in cone_ray_vectors:
                voxel_path_gen = trace_ray_fast(analysis_point, ray_vec, scene_min, VOXEL_SIZE, grid_dims)
                transmittance = calculate_transmittance(voxel_path_gen, classification_grid, density_grid, VOXEL_SIZE, VEGETATION_EXTINCTION_COEFFICIENTS)
                cone_transmittances.append(transmittance)
            avg_transmittance = np.mean(cone_transmittances)
            simulation_results.append({'azimuth': az, 'elevation': el, 'transmittance': avg_transmittance})

    # 5. Format and Save Final Matrix
    print("\n--- Aggregating Results into CSV Matrix ---")
    df = pd.DataFrame(simulation_results)
    df['azimuth_deg'] = np.round(np.rad2deg(df['azimuth'])).astype(int)
    df['elevation_deg'] = np.round(np.rad2deg(df['elevation'])).astype(int)
    shadow_matrix_df = df.pivot_table(index='elevation_deg', columns='azimuth_deg', values='transmittance')
    shadow_matrix_df = 1 - shadow_matrix_df
    shadow_matrix_df = shadow_matrix_df.sort_index(axis=0).sort_index(axis=1)
    shadow_matrix_df.index = [f"Altitude_{i}" for i in shadow_matrix_df.index]
    shadow_matrix_df.columns = [f"Azimuth_{c}" for c in shadow_matrix_df.columns]
    if 'Azimuth_0' in shadow_matrix_df.columns:
        shadow_matrix_df['Azimuth_360'] = shadow_matrix_df['Azimuth_0']
    
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    output_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILENAME)
    shadow_matrix_df.to_csv(output_path, header=True, index=True)
    
    end_time = time.time()
    print(f"\n--- Simulation Finished ---")
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"Saved final shadow matrix to {output_path}")

# --- END OF SCRIPT ---