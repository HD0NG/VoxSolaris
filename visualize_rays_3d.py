"""
3D LiDAR Scene Visualizer with Ray Beam
=========================================
Interactive 3D scatter plot showing:
  - Classified LiDAR points (ground, vegetation, building) color-coded
  - PV array panel positions
  - Solar beam direction at a specified local time
  - Ray paths from each panel toward the sun
  - Computed transmittance per panel and mean

Usage:
    python visualize_rays_3d.py --time "2021-07-04 08:00"
    python visualize_rays_3d.py --time "2021-07-04 17:00" --radius 50

Or import:
    from visualize_rays_3d import visualize_scene
    visualize_scene("2021-07-04 08:00", lidar_path="output/recovered.laz")
"""

import argparse
import numpy as np
import pandas as pd
import pvlib
import pytz
import laspy
import plotly.graph_objects as go
from datetime import datetime, timedelta


# ============================================================================
# CONFIG — match your project
# ============================================================================

DEFAULT_LIDAR = "output/reclassified_final_v5.laz"
DEFAULT_SHADOW_CSV = "results/shadow_matrix_results_SE_pro/shadow_attenuation_matrix_conecasting_SE_v1.csv"

LAT = 62.979849
LON = 27.648656
LOCAL_TZ = "Europe/Helsinki"
INVERTER_UTC_OFFSET = 3

# PV array: left-corner anchor
ARRAY_CORNER_XY = np.array([532882.50, 6983507.00])
ROOF_SEARCH_RADIUS = 2.0
OFFSET_FROM_ROOF = 1.5
TILT_DEG = 12
AZ_DEG = 170
PANEL_W = 1.0
PANEL_H = 1.6
ROW_CONFIG = (5, 4, 3)

# Visualization
DEFAULT_RADIUS = 40  # meters around array center to show
RAY_LENGTH = 60  # meters for ray visualization
POINT_SUBSAMPLE = 3  # show every Nth point for performance

# Classification colors
CLASS_COLORS = {
    2: ("Ground", "rgb(139,119,101)"),   # brown
    3: ("Low Veg", "rgb(144,238,144)"),  # light green
    4: ("Med Veg", "rgb(34,139,34)"),    # forest green
    5: ("High Veg", "rgb(0,100,0)"),     # dark green
    6: ("Building", "rgb(220,20,60)"),   # red
}

# Priority used when collapsing several classes into one voxel
# (higher value wins — buildings dominate ground, dense veg dominates sparse).
_CLASS_PRIORITY = {6: 5, 5: 4, 4: 3, 3: 2, 2: 1}

DEFAULT_VOXEL_SIZE         = 2.0
DEFAULT_VOXEL_OPACITY      = 0.0   # 0 = no face mesh (pure wireframe). Bump to add faint faces.
DEFAULT_VOXEL_EDGE_WIDTH   = 2
DEFAULT_VOXEL_EDGE_OPACITY = 0.6

DEFAULT_RAY_CONE_ANGLE_DEG = 2.0  # half-angle of the visualized ray cone
DEFAULT_RAY_CONE_SEGMENTS  = 14
DEFAULT_RAY_COLOR          = "rgb(255,170,0)"
DEFAULT_RAY_OPACITY        = 0.30


# ============================================================================
# PV ARRAY GEOMETRY (must match shadow_matrix_simulation.py)
# ============================================================================

def generate_pv_array_points(corner_coords, tilt_deg=TILT_DEG, az_deg=AZ_DEG,
                              panel_w=PANEL_W, panel_h=PANEL_H,
                              row_config=ROW_CONFIG):
    """Generate panel center world coords from left-corner anchor."""
    tilt_rad = np.radians(tilt_deg)
    rot_z_rad = np.radians(180 - az_deg)

    num_rows = len(row_config)
    total_height = (num_rows - 1) * panel_h

    local_points = []
    y_steps = np.linspace(total_height, 0.0, num_rows)

    for i, n_panels in enumerate(row_config):
        y = y_steps[i]
        for p in range(n_panels):
            x = p * panel_w + panel_w / 2
            local_points.append([x, y, 0.0])

    local_points = np.array(local_points)

    R_tilt = np.array([
        [1, 0, 0],
        [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
        [0, np.sin(tilt_rad),  np.cos(tilt_rad)]
    ])
    R_az = np.array([
        [np.cos(rot_z_rad), -np.sin(rot_z_rad), 0],
        [np.sin(rot_z_rad),  np.cos(rot_z_rad), 0],
        [0, 0, 1]
    ])

    rotated = (R_az @ R_tilt @ local_points.T).T
    return rotated + corner_coords


# ============================================================================
# SHADOW MATRIX LOOKUP
# ============================================================================

def load_shadow_matrix(path):
    """Load shadow matrix CSV, return as numpy array."""
    df = pd.read_csv(path, index_col=0)
    return np.clip(np.nan_to_num(df.values, nan=0.0), 0.0, 1.0)


def lookup_transmittance(shadow_matrix, altitude_deg, azimuth_deg):
    """Look up transmittance for a given solar position."""
    n_alt, n_azi = shadow_matrix.shape
    if altitude_deg < 0.5:
        return 0.0  # below horizon
    alt_i = int(np.clip(np.round(altitude_deg), 0, n_alt - 1))
    azi_i = int(np.clip(np.round(azimuth_deg), 0, n_azi - 1))
    shadow = shadow_matrix[alt_i, azi_i]
    return 1.0 - shadow  # transmittance


# ============================================================================
# VOXEL REPRESENTATION
# ============================================================================

def voxelize_for_viz(points, classifications, voxel_size,
                     scene_min=None, scene_max=None):
    """Sparse voxelization for display.

    Returns
    -------
    voxel_min   : (N, 3) float — lower corner of each occupied voxel (world coords)
    voxel_class : (N,)  int   — dominant class per voxel (via _CLASS_PRIORITY)
    voxel_count : (N,)  int   — number of LiDAR points that fell into the voxel
    """
    if scene_min is None:
        scene_min = np.min(points, axis=0)
    if scene_max is None:
        scene_max = np.max(points, axis=0)
    grid_dims = np.maximum(
        np.ceil((scene_max - scene_min) / voxel_size).astype(np.int64), 1
    )
    nx, ny, nz = grid_dims

    vi = np.clip(
        np.floor((points - scene_min) / voxel_size).astype(np.int64),
        0, grid_dims - 1,
    )
    keys = vi[:, 0] * (ny * nz) + vi[:, 1] * nz + vi[:, 2]

    # Priority lookup as a dense array indexed by class id (max class id used is 6).
    pri_lut = np.zeros(int(max(_CLASS_PRIORITY) + 1) + 1, dtype=np.int32)
    for c, p in _CLASS_PRIORITY.items():
        pri_lut[c] = p
    cls_clipped = np.clip(classifications, 0, len(pri_lut) - 1).astype(np.int64)
    pri = pri_lut[cls_clipped]

    # Sort so that within each voxel, the highest-priority class comes first.
    order = np.lexsort((-pri, keys))
    keys_sorted = keys[order]
    cls_sorted = classifications[order]

    uniq_keys, first_idx = np.unique(keys_sorted, return_index=True)
    dominant_cls = cls_sorted[first_idx]

    counts = np.bincount(np.searchsorted(uniq_keys, keys),
                         minlength=len(uniq_keys)).astype(np.int32)

    ix = uniq_keys // (ny * nz)
    iy = (uniq_keys // nz) % ny
    iz = uniq_keys % nz
    voxel_min = (np.stack([ix, iy, iz], axis=1).astype(np.float64)
                 * voxel_size + scene_min)

    return voxel_min, dominant_cls, counts


# Cube template: 8 corner vertices, 12 triangle faces, 12 wireframe edges.
_CUBE_VERTS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=np.float64)
_CUBE_I = np.array([0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 1, 1])
_CUBE_J = np.array([1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6])
_CUBE_K = np.array([2, 3, 6, 7, 5, 4, 6, 7, 7, 4, 6, 5])

# Cube edges: 4 bottom + 4 top + 4 verticals (12 line segments per cube).
_CUBE_EDGES = np.array([
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
], dtype=np.int64)


def make_voxel_mesh_traces(voxel_min, voxel_classes, voxel_size,
                            class_colors=CLASS_COLORS,
                            opacity=DEFAULT_VOXEL_OPACITY,
                            visible_classes=None,
                            showlegend=True):
    """Build one go.Mesh3d per LiDAR class — each occupied voxel becomes a solid cube."""
    traces = []
    cube_v = _CUBE_VERTS * voxel_size
    visible_classes = set(class_colors) if visible_classes is None else set(visible_classes)

    for cid, (label, color) in class_colors.items():
        if cid not in visible_classes:
            continue
        mask = voxel_classes == cid
        n = int(mask.sum())
        if n == 0:
            continue

        mins = voxel_min[mask]                            # (n, 3)
        verts = (mins[:, None, :] + cube_v[None, :, :])   # (n, 8, 3)
        verts = verts.reshape(-1, 3)

        offsets = (np.arange(n) * 8).reshape(-1, 1)
        i_idx = (_CUBE_I[None, :] + offsets).ravel()
        j_idx = (_CUBE_J[None, :] + offsets).ravel()
        k_idx = (_CUBE_K[None, :] + offsets).ravel()

        traces.append(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=i_idx, j=j_idx, k=k_idx,
            color=color, opacity=opacity, flatshading=True,
            name=(f"{label} voxels ({n:,})" if showlegend else None),
            showlegend=showlegend, hoverinfo="skip",
        ))
    return traces


def make_voxel_edge_traces(voxel_min, voxel_classes, voxel_size,
                           class_colors=CLASS_COLORS,
                           line_width=DEFAULT_VOXEL_EDGE_WIDTH,
                           visible_classes=None,
                           opacity=DEFAULT_VOXEL_EDGE_OPACITY):
    """Build one go.Scatter3d (line mode) per class — wireframe-only voxels.

    Disconnected edges are encoded with NaN separators, which Plotly treats
    as line breaks within a single Scatter3d trace.
    """
    traces = []
    cube_v = _CUBE_VERTS * voxel_size
    visible_classes = set(class_colors) if visible_classes is None else set(visible_classes)

    for cid, (label, color) in class_colors.items():
        if cid not in visible_classes:
            continue
        mask = voxel_classes == cid
        n = int(mask.sum())
        if n == 0:
            continue

        # (n, 8, 3) all 8 cube vertices for every selected voxel
        all_v = voxel_min[mask][:, None, :] + cube_v[None, :, :]

        # (n, 12, 2, 3): two endpoints per edge × 12 edges × n voxels
        seg = all_v[:, _CUBE_EDGES, :]

        # Pad to 3 entries per segment (start, end, NaN-break) so Plotly
        # breaks the polyline between edges.
        padded = np.full((n, 12, 3, 3), np.nan, dtype=np.float64)
        padded[:, :, :2, :] = seg
        flat = padded.reshape(-1, 3)

        traces.append(go.Scatter3d(
            x=flat[:, 0], y=flat[:, 1], z=flat[:, 2],
            mode="lines",
            line=dict(width=line_width, color=color),
            opacity=opacity,
            name=f"{label} voxels ({n:,})",
            hoverinfo="skip",
        ))
    return traces


# ============================================================================
# SOLAR RAY CONES
# ============================================================================

def _cone_geometry(apex, direction, length, half_angle_rad, n_segments):
    """Vertices + triangle indices for one cone (apex → circular base)."""
    direction = direction / np.linalg.norm(direction)

    # Pick any vector not parallel to direction, then Gram–Schmidt for two
    # orthonormal vectors spanning the base plane.
    helper = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.95 \
             else np.array([1.0, 0.0, 0.0])
    u = np.cross(direction, helper); u /= np.linalg.norm(u)
    v = np.cross(direction, u)

    base_center = apex + direction * length
    base_radius = length * np.tan(half_angle_rad)

    thetas = np.linspace(0.0, 2 * np.pi, n_segments, endpoint=False)
    base = (base_center[None, :]
            + base_radius * (np.cos(thetas)[:, None] * u[None, :]
                             + np.sin(thetas)[:, None] * v[None, :]))

    verts = np.vstack([apex[None, :], base])                  # (n_segments + 1, 3)
    # Lateral surface: apex(0) — base_i — base_{i+1 mod n}
    i_idx = np.zeros(n_segments, dtype=np.int64)
    j_idx = 1 + np.arange(n_segments, dtype=np.int64)
    k_idx = 1 + (np.arange(n_segments, dtype=np.int64) + 1) % n_segments
    return verts, i_idx, j_idx, k_idx


def make_solar_cone_trace(apexes, direction, length,
                          half_angle_deg=DEFAULT_RAY_CONE_ANGLE_DEG,
                          n_segments=DEFAULT_RAY_CONE_SEGMENTS,
                          color=DEFAULT_RAY_COLOR,
                          opacity=DEFAULT_RAY_OPACITY,
                          name="Solar rays"):
    """One go.Mesh3d holding a cone per apex (e.g. one cone per PV panel)."""
    half_angle_rad = np.radians(half_angle_deg)
    all_v, all_i, all_j, all_k = [], [], [], []
    base = 0
    for apex in apexes:
        v, i, j, k = _cone_geometry(np.asarray(apex, dtype=np.float64),
                                    direction, length, half_angle_rad, n_segments)
        all_v.append(v)
        all_i.append(i + base)
        all_j.append(j + base)
        all_k.append(k + base)
        base += len(v)

    V = np.vstack(all_v)
    return go.Mesh3d(
        x=V[:, 0], y=V[:, 1], z=V[:, 2],
        i=np.concatenate(all_i),
        j=np.concatenate(all_j),
        k=np.concatenate(all_k),
        color=color, opacity=opacity, flatshading=True,
        name=name, showlegend=True, hoverinfo="skip",
    )


# ============================================================================
# MAIN VISUALIZATION
# ============================================================================

def visualize_scene(
    local_time_str,
    lidar_path=DEFAULT_LIDAR,
    shadow_csv=DEFAULT_SHADOW_CSV,
    radius=DEFAULT_RADIUS,
    ray_length=RAY_LENGTH,
    subsample=POINT_SUBSAMPLE,
    show_points=True,
    show_voxels=False,
    voxel_size=DEFAULT_VOXEL_SIZE,
    voxel_opacity=DEFAULT_VOXEL_OPACITY,
    voxel_edges=True,
    voxel_edge_width=DEFAULT_VOXEL_EDGE_WIDTH,
    voxel_edge_opacity=DEFAULT_VOXEL_EDGE_OPACITY,
    voxel_classes=None,
    show_sun=True,
    show_rays=True,
    ray_cone_angle_deg=DEFAULT_RAY_CONE_ANGLE_DEG,
    ray_cone_segments=DEFAULT_RAY_CONE_SEGMENTS,
    ray_color=DEFAULT_RAY_COLOR,
    ray_opacity=DEFAULT_RAY_OPACITY,
):
    """
    Create an interactive 3D visualization of the LiDAR scene with
    solar beam rays and transmittance at a specific local time.

    Parameters
    ----------
    local_time_str : str
        Local time, e.g. "2021-07-04 08:00"
    lidar_path : str
        Path to .laz/.las file
    shadow_csv : str
        Path to shadow matrix CSV
    radius : float
        Meters around array center to include
    ray_length : float
        Length of ray visualization in meters
    subsample : int
        Show every Nth point (performance, point cloud only)
    show_points, show_voxels, show_sun, show_rays : bool
        Per-element visibility toggles.
    voxel_size : float
        Edge length of each voxel in metres (matches shadow-matrix grid).
    voxel_opacity : float
        Face opacity (0–1) of the cube meshes. ``0`` (default) skips the
        face mesh entirely — pure wireframe.
    voxel_edges : bool
        Render the colored cube edges (wireframe).
    voxel_edge_width : float
        Line width of wireframe edges.
    voxel_edge_opacity : float
        Opacity (0–1) of the wireframe edges.
    voxel_classes : iterable of int or None
        Restrict which class IDs become voxel cubes. ``None`` = all classes.
    ray_cone_angle_deg : float
        Half-angle of the visualised solar-ray cone (deg). ``0`` falls back
        to single-line rays.
    ray_cone_segments : int
        Circular resolution of each cone (more = smoother, heavier).
    ray_color, ray_opacity : str, float
        Cone color and opacity.
    """
    local_time = pd.Timestamp(local_time_str)
    print(f"Visualizing scene at {local_time} (local)")

    # --- Convert local → UTC for solar position ---
    tz = pytz.timezone(LOCAL_TZ)
    local_aware = tz.localize(local_time.to_pydatetime())
    utc_time = local_aware.astimezone(pytz.UTC)
    print(f"  UTC: {utc_time}")

    # --- Solar position ---
    solpos = pvlib.solarposition.get_solarposition(
        pd.DatetimeIndex([utc_time]), LAT, LON
    )
    altitude_deg = 90.0 - float(solpos["apparent_zenith"].iloc[0])
    azimuth_deg = float(solpos["azimuth"].iloc[0])
    print(f"  Solar altitude: {altitude_deg:.1f}°")
    print(f"  Solar azimuth:  {azimuth_deg:.1f}°")

    if altitude_deg < 0.5:
        print("  Sun is below horizon — no beam to visualize.")
        return

    # Sun direction vector (in local ENU: x=East, y=North, z=Up)
    el_rad = np.radians(altitude_deg)
    az_rad = np.radians(azimuth_deg)
    sun_dir = np.array([
        np.cos(el_rad) * np.sin(az_rad),  # East component
        np.cos(el_rad) * np.cos(az_rad),  # North component
        np.sin(el_rad),                    # Up component
    ])

    # --- Load LiDAR ---
    print(f"  Loading LiDAR from {lidar_path}...")
    las = laspy.read(lidar_path)
    pts = np.vstack((las.x, las.y, las.z)).T
    cls = np.array(las.classification)

    # Filter to relevant classes
    relevant = np.isin(cls, [2, 3, 4, 5, 6])
    pts = pts[relevant]
    cls = cls[relevant]

    # --- Build PV array ---
    # Find roof height
    dists = np.linalg.norm(pts[:, :2] - ARRAY_CORNER_XY, axis=1)
    roof_mask = (dists < ROOF_SEARCH_RADIUS) & (cls == 6)
    if roof_mask.any():
        target_z = np.max(pts[roof_mask, 2]) + OFFSET_FROM_ROOF
    else:
        target_z = np.median(pts[cls == 2, 2]) + 5.0
        print(f"  WARNING: No building points near array. Using z={target_z:.1f}")

    corner_3d = np.array([ARRAY_CORNER_XY[0], ARRAY_CORNER_XY[1], target_z])
    panel_points = generate_pv_array_points(corner_3d)
    array_center = panel_points.mean(axis=0)

    print(f"  Array center: ({array_center[0]:.1f}, {array_center[1]:.1f}, {array_center[2]:.1f})")
    print(f"  {len(panel_points)} panel points")

    # --- Crop scene around array ---
    dist_to_center = np.linalg.norm(pts[:, :2] - array_center[:2], axis=1)
    crop_mask = dist_to_center < radius
    pts_crop = pts[crop_mask]
    cls_crop = cls[crop_mask]

    # Subsample for performance
    pts_show = pts_crop[::subsample]
    cls_show = cls_crop[::subsample]
    print(f"  Showing {len(pts_show):,} points (of {len(pts_crop):,} within {radius}m)")

    # --- Shadow matrix lookup ---
    shadow_matrix = load_shadow_matrix(shadow_csv)
    transmittance = lookup_transmittance(shadow_matrix, altitude_deg, azimuth_deg)
    print(f"  Matrix transmittance at ({altitude_deg:.1f}°, {azimuth_deg:.1f}°): {transmittance:.3f}")
    print(f"  Shadow factor: {1 - transmittance:.3f}")

    # --- Build Plotly traces ---
    traces = []

    if show_points:
        # LiDAR points by class
        for class_id, (label, color) in CLASS_COLORS.items():
            mask = cls_show == class_id
            if mask.sum() == 0:
                continue
            p = pts_show[mask]
            traces.append(go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=p[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=color, opacity=0.6),
                name=f"{label} ({mask.sum():,})",
                hovertemplate=f"{label}<br>x=%{{x:.1f}}<br>y=%{{y:.1f}}<br>z=%{{z:.1f}}",
            ))

    if show_voxels:
        # Voxelise the full cropped cloud (not the subsampled view) so the
        # voxel grid matches what the shadow-matrix simulation would see.
        v_min, v_cls, v_cnt = voxelize_for_viz(pts_crop, cls_crop, voxel_size)
        n_total = len(v_min)
        if voxel_classes is not None:
            keep = np.isin(v_cls, list(voxel_classes))
            v_min, v_cls, v_cnt = v_min[keep], v_cls[keep], v_cnt[keep]
        print(f"  Voxelisation: {n_total:,} occupied voxels "
              f"@ {voxel_size}m (showing {len(v_min):,})")

        # Faces: optional, semi-transparent. Edges (when shown) own the
        # legend entry so we don't get two rows per class.
        if voxel_opacity > 0:
            traces.extend(make_voxel_mesh_traces(
                v_min, v_cls, voxel_size,
                class_colors=CLASS_COLORS, opacity=voxel_opacity,
                visible_classes=voxel_classes,
                showlegend=not voxel_edges,
            ))
        if voxel_edges:
            traces.extend(make_voxel_edge_traces(
                v_min, v_cls, voxel_size,
                class_colors=CLASS_COLORS, line_width=voxel_edge_width,
                visible_classes=voxel_classes,
                opacity=voxel_edge_opacity,
            ))

    # Panel points
    traces.append(go.Scatter3d(
        x=panel_points[:, 0], y=panel_points[:, 1], z=panel_points[:, 2],
        mode="markers",
        marker=dict(size=8, color="rgb(255,215,0)", symbol="diamond",
                    line=dict(width=1, color="black")),
        name=f"PV Panels ({len(panel_points)})",
        hovertemplate="Panel<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}",
    ))

    if show_rays:
        if ray_cone_angle_deg > 0:
            traces.append(make_solar_cone_trace(
                panel_points, sun_dir, ray_length,
                half_angle_deg=ray_cone_angle_deg,
                n_segments=ray_cone_segments,
                color=ray_color, opacity=ray_opacity,
                name="Solar ray cones",
            ))
        else:
            for i, pt in enumerate(panel_points):
                end = pt + sun_dir * ray_length
                traces.append(go.Scatter3d(
                    x=[pt[0], end[0]], y=[pt[1], end[1]], z=[pt[2], end[2]],
                    mode="lines",
                    line=dict(width=4, color=ray_color),
                    opacity=ray_opacity,
                    name="Solar rays" if i == 0 else None,
                    showlegend=(i == 0),
                    hoverinfo="skip",
                ))

    if show_sun:
        sun_marker = array_center + sun_dir * ray_length * 1.2
        traces.append(go.Scatter3d(
            x=[sun_marker[0]], y=[sun_marker[1]], z=[sun_marker[2]],
            mode="markers+text",
            marker=dict(size=15, color="yellow", symbol="diamond",
                        line=dict(width=2, color="orange")),
            text=[f"☀ Alt={altitude_deg:.1f}° Az={azimuth_deg:.1f}°"],
            textposition="top center",
            textfont=dict(size=12, color="orange"),
            name="Sun direction",
        ))

    # --- Layout ---
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(f"3D Scene — {local_time_str} local<br>"
                  f"<sub>Alt={altitude_deg:.1f}° Az={azimuth_deg:.1f}° "
                  f"| Transmittance={transmittance:.3f} "
                  f"| Shadow={1-transmittance:.3f}</sub>"),
            font=dict(size=16),
        ),
        scene=dict(
            xaxis=dict(title="Easting (m)",   showbackground=False, showgrid=False, zeroline=False),
            yaxis=dict(title="Northing (m)",  showbackground=False, showgrid=False, zeroline=False),
            zaxis=dict(title="Elevation (m)", showbackground=False, showgrid=False, zeroline=False),
            aspectmode="data",
            bgcolor="white",
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.0),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        width=1400,
        height=900,
    )

    # Save and show
    html_path = f"scene_3d_{local_time.strftime('%Y%m%d_%H%M')}.html"
    fig.write_html(html_path)
    print(f"\n  Saved interactive 3D view to {html_path}")
    fig.show()

    return fig


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D LiDAR + Ray Beam Visualizer")
    parser.add_argument("--time", required=True,
                        help="Local time, e.g. '2021-07-04 08:00'")
    parser.add_argument("--lidar", default=DEFAULT_LIDAR,
                        help="Path to LAS/LAZ file")
    parser.add_argument("--shadow", default=DEFAULT_SHADOW_CSV,
                        help="Path to shadow matrix CSV")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS,
                        help="Crop radius in meters (default 40)")
    parser.add_argument("--ray-length", type=float, default=RAY_LENGTH,
                        help="Ray visualization length in meters (default 60)")
    parser.add_argument("--subsample", type=int, default=POINT_SUBSAMPLE,
                        help="Point subsampling factor (default 3)")
    parser.add_argument("--voxels", action="store_true",
                        help="Render the cropped scene as voxel cubes")
    parser.add_argument("--no-points", action="store_true",
                        help="Hide the raw LiDAR point cloud (useful with --voxels)")
    parser.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE,
                        help=f"Voxel edge length in metres (default {DEFAULT_VOXEL_SIZE})")
    parser.add_argument("--voxel-opacity", type=float, default=DEFAULT_VOXEL_OPACITY,
                        help=f"Voxel cube opacity 0–1 (default {DEFAULT_VOXEL_OPACITY})")
    parser.add_argument("--voxel-classes", type=int, nargs="+", default=None,
                        help="Restrict voxel rendering to these class IDs "
                             "(e.g. 5 6 for high-veg + buildings only)")
    parser.add_argument("--no-voxel-edges", action="store_true",
                        help="Disable wireframe edge overlay on voxels")
    parser.add_argument("--voxel-edge-width", type=float,
                        default=DEFAULT_VOXEL_EDGE_WIDTH,
                        help=f"Voxel edge line width (default {DEFAULT_VOXEL_EDGE_WIDTH})")
    parser.add_argument("--voxel-edge-opacity", type=float,
                        default=DEFAULT_VOXEL_EDGE_OPACITY,
                        help=f"Voxel edge opacity 0–1 (default {DEFAULT_VOXEL_EDGE_OPACITY})")
    parser.add_argument("--no-sun", action="store_true",
                        help="Hide the sun marker")
    parser.add_argument("--no-rays", action="store_true",
                        help="Hide the solar ray cones / lines")
    parser.add_argument("--ray-cone-angle", type=float,
                        default=DEFAULT_RAY_CONE_ANGLE_DEG,
                        help="Half-angle (deg) of the ray cone, 0 = single line "
                             f"(default {DEFAULT_RAY_CONE_ANGLE_DEG})")
    parser.add_argument("--ray-cone-segments", type=int,
                        default=DEFAULT_RAY_CONE_SEGMENTS,
                        help=f"Cone circular resolution (default {DEFAULT_RAY_CONE_SEGMENTS})")

    args = parser.parse_args()

    visualize_scene(
        args.time,
        lidar_path=args.lidar,
        shadow_csv=args.shadow,
        radius=args.radius,
        ray_length=args.ray_length,
        subsample=args.subsample,
        show_points=not args.no_points,
        show_voxels=args.voxels,
        voxel_size=args.voxel_size,
        voxel_opacity=args.voxel_opacity,
        voxel_edges=not args.no_voxel_edges,
        voxel_edge_width=args.voxel_edge_width,
        voxel_edge_opacity=args.voxel_edge_opacity,
        voxel_classes=args.voxel_classes,
        show_sun=not args.no_sun,
        show_rays=not args.no_rays,
        ray_cone_angle_deg=args.ray_cone_angle,
        ray_cone_segments=args.ray_cone_segments,
    )
