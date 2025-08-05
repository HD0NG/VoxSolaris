import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_dataframe(matrix, altitude_range, azimuth_range):
    altitudes = [f"Altitude_{int(alt)}" for alt in altitude_range]
    azimuths = [f"Azimuth_{int(azi)}" for azi in azimuth_range]
    return pd.DataFrame(matrix, index=altitudes, columns=azimuths)

def plot_shadow_polar(matrix, altitude_range, azimuth_range):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    for i, altitude in enumerate(altitude_range):
        for j, azimuth in enumerate(azimuth_range):
            shadow_intensity = matrix[i, j]
            if shadow_intensity > 0:
                color = plt.cm.gray_r(shadow_intensity)
                ax.plot(np.radians(azimuth), 90 - altitude, 'o', color=color, markersize=5)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_title('Shadow Intensity Polar Plot\n(Beer–Lambert Voxel Model)\n\n')

    # Add colorbar indicating shadow intensity
    sm = plt.cm.ScalarMappable(cmap='gray_r', norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=ax, label='Shadow Intensity (1 - Transmission)')
    
    plt.show()

def plot_shadow_polar_in(matrix, altitude_range, azimuth_range):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    for i, altitude in enumerate(altitude_range):
        for j, azimuth in enumerate(azimuth_range):
            shadow_intensity = matrix[i, j]
            if shadow_intensity > 0:
                color = plt.cm.gray_r(shadow_intensity)
                ax.plot(np.radians(azimuth), 90 - altitude, 'o', color=color, markersize=5)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_title('Shadow Intensity Polar Plot\n(Beer–Lambert Voxel Model)\n\n')

    # Add colorbar indicating shadow intensity
    sm = plt.cm.ScalarMappable(cmap='gray_r', norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=ax, label='Shadow Intensity (1 - Transmission)')
    
    return fig