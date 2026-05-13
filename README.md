# VoxSolaris
A Voxel-Based Ray-Casting Framework for Simulating Solar Radiation Attenuation in Complex Environments Using LiDAR Data

## Overview
VoxSolaris is a research-focused, high-performance Python simulation engine designed to accurately model solar radiation and shadow dynamics in complex 3D environments. By converting LiDAR point clouds into a volumetric grid (voxels), it can simulate the path of sunlight through intricate urban and forested landscapes with high physical realism.

The engine employs a cone-casting methodology to account for the sun's angular diameter, producing soft, realistic shadows (penumbra). It supports distinct physical models for different environmental features, treating buildings as opaque objects and modeling vegetation as a porous medium using the Beer-Lambert law with class-specific extinction coefficients.

The final output is a detailed shadow attenuation matrix, a CSV file quantifying the solar transmittance at a specific point for every degree of the sky, making it an ideal input for a wide range of scientific applications.

## FMI PV Forecasting Model
VoxSolaris uses the FMI PV forecasting model as the baseline photovoltaic production model before applying site-specific shadow correction. The analysis modules (`pv_analysis_re.py` and `pv_analysis_b2.py`) import `fmi_pv_forecaster` as `pvfc`, configure it with the project site location, panel tilt, panel azimuth, and nominal system power, then call `pvfc.process_radiation_df(...)` to produce the unshaded FMI base forecast.

The model input is an `extra_data` DataFrame containing irradiance and weather variables such as DNI, DHI, GHI, temperature, wind speed, and albedo. In this project, those inputs are built from FMI radiation and weather CSV data, with optional CAMS irradiance support. VoxSolaris then combines the FMI base forecast with the voxel-derived shadow attenuation matrix to produce a shadow-corrected PV forecast for comparison against measured inverter output.

The FMI PV forecaster is referenced from:
<https://github.com/TimoSalola/fmi_pv_forecaster.git>
<https://github.com/fmidev/fmi-open-pv-forecast-packaged/tree/vox_solaris_shading_build>

A packaged wheel for this dependency is also included under `whl/` for local installation.

## Key Features
- Voxel-Based Representation: Accurately models true 3D environments, including vertical facades, overhangs, and complex canopy structures, overcoming the limitations of 2.5D DSM/raster methods.

- Cone-Casting for Realism: Simulates sunlight as a cone of rays, not a single point, to produce physically accurate soft shadows (penumbra).

- Multi-Class Attenuation: Supports distinct attenuation models for different object types:

- Binary Occlusion for opaque objects like buildings.

- Beer-Lambert Law for porous media like vegetation.

- Differentiated Vegetation: Allows for multiple vegetation classes (e.g., low, medium, high) with unique extinction coefficients (k-values) for more nuanced forest canopy modeling.

- High-Resolution Output: Generates a detailed shadow attenuation matrix with 1° resolution in both solar azimuth and altitude.
