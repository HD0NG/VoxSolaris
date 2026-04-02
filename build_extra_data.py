"""
build_extra_data.py — Build extra_data from FMI radiation + temp/wind CSVs.

Reads:
  1. FMI radiation CSV (1-min): Diffuse, Direct(CAMS), Global, Reflected
  2. FMI temp/wind CSV (1-min): Air temperature, Wind speed

Produces per-day extra_data with columns:
  timestamp, dni, dhi, ghi, T, wind, albedo

DNI is computed from GHI and DHI using pvlib closure equation:
  DNI = (GHI - DHI) / cos(zenith)

Data is resampled from 1-min to 5-min (mean).

Usage:
    from build_extra_data import build_all_extra_data, load_extra_for_day

    # Build and save all days as CSVs
    build_all_extra_data(
        rad_csv="data/pvdata/Kuopio Savilahti 1.4.2021 - 1.10.2021_rad.csv",
        temp_wind_csv="data/pvdata/Kuopio Savilahti 1.4.2021 - 1.10.2021_temp_wind.csv",
        output_dir="output",
        latitude=62.8924, longitude=27.6353,
    )

    # Or load a single day (returns DataFrame, no file saved)
    df = load_extra_for_day(
        rad_csv="...", temp_wind_csv="...",
        date="2021-04-17",
        latitude=62.8924, longitude=27.6353,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# FMI CSV parsers
# ═══════════════════════════════════════════════════════════════════════════

def _parse_fmi_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Generic parser for FMI observation CSVs.
    Returns DataFrame with datetime_utc index and numeric columns.
    """
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip().str.strip('"')

    # Find date/time columns
    year_col  = [c for c in df.columns if "year"  in c.lower()][0]
    month_col = [c for c in df.columns if "month" in c.lower()][0]
    day_col   = [c for c in df.columns if "day"   in c.lower()][0]
    time_col  = [c for c in df.columns if "time"  in c.lower()][0]

    df["datetime_utc"] = pd.to_datetime(
        df[year_col].astype(str) + "-" +
        df[month_col].astype(str).str.zfill(2) + "-" +
        df[day_col].astype(str).str.zfill(2) + " " +
        df[time_col].astype(str).str.strip('"'),
        format="%Y-%m-%d %H:%M",
    )

    # Convert all non-date columns to numeric (replace "-" with NaN)
    skip = {year_col, month_col, day_col, time_col, "datetime_utc"}
    skip.update(c for c in df.columns if "station" in c.lower())

    for col in df.columns:
        if col not in skip:
            s = df[col].astype(str).str.strip('"')
            s = s.where(s != "-", other=np.nan)
            df[col] = pd.to_numeric(s, errors="coerce")

    df = df.set_index("datetime_utc").sort_index()
    return df


def load_radiation(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load FMI radiation CSV → columns: dhi, ghi, reflected."""
    df = _parse_fmi_csv(csv_path)

    # Map to standard names
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "diffuse" in cl:
            col_map[col] = "dhi"
        elif "global" in cl:
            col_map[col] = "ghi"
        elif "reflect" in cl:
            col_map[col] = "reflected"
        # Skip "direct" — we compute DNI ourselves

    df = df.rename(columns=col_map)
    keep = [c for c in ["dhi", "ghi", "reflected"] if c in df.columns]
    return df[keep]


def load_temp_wind(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load FMI temp/wind CSV → columns: T, wind."""
    df = _parse_fmi_csv(csv_path)

    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "temperature" in cl:
            col_map[col] = "T"
        elif "wind" in cl:
            col_map[col] = "wind"

    df = df.rename(columns=col_map)
    keep = [c for c in ["T", "wind"] if c in df.columns]
    return df[keep]


# ═══════════════════════════════════════════════════════════════════════════
# DNI computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_dni(ghi: pd.Series, dhi: pd.Series,
                latitude: float, longitude: float) -> pd.Series:
    """
    Compute DNI from GHI and DHI using the closure equation:
        DNI = (GHI - DHI) / cos(zenith)

    Uses pvlib for solar position. Falls back to manual calc if pvlib
    is unavailable.
    """
    try:
        import pvlib

        solpos = pvlib.solarposition.get_solarposition(
            ghi.index, latitude, longitude
        )
        zenith = solpos["zenith"]

        # Use pvlib's dni function with edge-case handling
        dni = pvlib.irradiance.dni(ghi, dhi, zenith)
        dni = dni.clip(lower=0).fillna(0)

    except ImportError:
        # Manual fallback
        print("Warning: pvlib not available, using manual DNI calculation.")
        try:
            import pvlib as _pv
            solpos = _pv.solarposition.get_solarposition(
                ghi.index, latitude, longitude
            )
            zenith = solpos["zenith"]
        except ImportError:
            raise ImportError("pvlib is required for solar position calculation.")

        cos_z = np.cos(np.radians(zenith)).clip(min=0.087)  # ~5° min
        dni = ((ghi - dhi) / cos_z).clip(lower=0).fillna(0)

    return dni


# ═══════════════════════════════════════════════════════════════════════════
# Albedo estimation
# ═══════════════════════════════════════════════════════════════════════════

def estimate_albedo(ghi: pd.Series, reflected: Optional[pd.Series],
                    temperature: Optional[pd.Series],
                    default_albedo: float = 0.2,
                    snow_albedo: float = 0.6,
                    bare_albedo: float = 0.2,
                    sigmoid_center: float = 1.0,
                    sigmoid_width: float = 3.0) -> pd.Series:
    """
    Estimate surface albedo.

    Priority:
      1. If reflected radiation available: albedo = reflected / ghi
      2. If temperature available: sigmoid snow model
      3. Default constant
    """
    albedo = pd.Series(default_albedo, index=ghi.index)

    if reflected is not None and not reflected.isna().all():
        # Compute from reflected / ghi where ghi > threshold
        valid = ghi > 50  # only compute when there's meaningful radiation
        albedo[valid] = (reflected[valid] / ghi[valid]).clip(0.05, 0.95)
        # Fill non-valid times with temperature-based or default
        if temperature is not None:
            snow_frac = 1.0 / (1.0 + np.exp((temperature - sigmoid_center) / sigmoid_width))
            albedo_temp = bare_albedo + (snow_albedo - bare_albedo) * snow_frac
            albedo[~valid] = albedo_temp[~valid]
    elif temperature is not None:
        snow_frac = 1.0 / (1.0 + np.exp((temperature - sigmoid_center) / sigmoid_width))
        albedo = bare_albedo + (snow_albedo - bare_albedo) * snow_frac

    return albedo


# ═══════════════════════════════════════════════════════════════════════════
# Build extra_data for a single day
# ═══════════════════════════════════════════════════════════════════════════

def load_extra_for_day(
    rad_csv: Union[str, Path],
    temp_wind_csv: Union[str, Path],
    date: Union[str, pd.Timestamp],
    latitude: float,
    longitude: float,
    resample_min: int = 5,
    default_albedo: float = 0.2,
) -> pd.DataFrame:
    """
    Build extra_data DataFrame for a single day.

    Returns DataFrame with columns: dni, dhi, ghi, T, wind, albedo
    Index: datetime_utc (5-min intervals)
    """
    date_obj = pd.to_datetime(date).date()

    # Load full CSVs
    rad = load_radiation(rad_csv)
    tw = load_temp_wind(temp_wind_csv)

    # Filter to target day
    rad_day = rad[rad.index.date == date_obj].copy()
    tw_day = tw[tw.index.date == date_obj].copy()

    if rad_day.empty:
        return pd.DataFrame()

    # Merge radiation + temp/wind on 1-min index
    merged = rad_day.join(tw_day, how="outer")

    # Resample to 5-min
    merged_5min = merged.resample(f"{resample_min}min").mean()

    # Compute DNI from GHI and DHI
    ghi = merged_5min["ghi"].fillna(0)
    dhi = merged_5min["dhi"].fillna(0)
    dni = compute_dni(ghi, dhi, latitude, longitude)

    # Estimate albedo
    reflected = merged_5min.get("reflected")
    temperature = merged_5min.get("T")
    albedo = estimate_albedo(ghi, reflected, temperature, default_albedo=default_albedo)

    # Build output DataFrame
    result = pd.DataFrame({
        "timestamp": merged_5min.index.strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "dni": dni.values,
        "dhi": dhi.values,
        "ghi": ghi.values,
        "T": merged_5min["T"].values if "T" in merged_5min.columns else np.nan,
        "wind": merged_5min["wind"].values if "wind" in merged_5min.columns else np.nan,
        "albedo": albedo.values,
    }, index=merged_5min.index)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Batch: build all days and save CSVs
# ═══════════════════════════════════════════════════════════════════════════

def build_all_extra_data(
    rad_csv: Union[str, Path],
    temp_wind_csv: Union[str, Path],
    latitude: float,
    longitude: float,
    output_dir: Union[str, Path] = "output",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    resample_min: int = 5,
    default_albedo: float = 0.2,
) -> list:
    """
    Build extra_data CSVs for all days in the radiation file.

    Saves one CSV per day: output/extra_data_YYYY-MM-DD.csv

    Returns list of (date, path) tuples for successfully saved days.
    """
    import tqdm

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load full datasets once
    print("Loading radiation CSV...")
    rad = load_radiation(rad_csv)
    print(f"  {len(rad)} records, {rad.index.min()} → {rad.index.max()}")

    print("Loading temp/wind CSV...")
    tw = load_temp_wind(temp_wind_csv)
    print(f"  {len(tw)} records, {tw.index.min()} → {tw.index.max()}")

    # Get all unique dates
    all_dates = sorted(rad.index.date)
    unique_dates = sorted(set(all_dates))

    if start_date:
        unique_dates = [d for d in unique_dates if d >= pd.to_datetime(start_date).date()]
    if end_date:
        unique_dates = [d for d in unique_dates if d <= pd.to_datetime(end_date).date()]

    print(f"Building extra_data for {len(unique_dates)} days...")

    saved = []
    for date_obj in tqdm.tqdm(unique_dates, desc="Building extra_data"):
        try:
            # Filter to day
            rad_day = rad[rad.index.date == date_obj]
            tw_day = tw[tw.index.date == date_obj]

            if rad_day.empty:
                continue

            merged = rad_day.join(tw_day, how="outer")
            merged_5min = merged.resample(f"{resample_min}min").mean()

            ghi = merged_5min["ghi"].fillna(0)
            dhi = merged_5min["dhi"].fillna(0)
            dni = compute_dni(ghi, dhi, latitude, longitude)

            reflected = merged_5min.get("reflected")
            temperature = merged_5min.get("T")
            albedo = estimate_albedo(ghi, reflected, temperature,
                                     default_albedo=default_albedo)

            result = pd.DataFrame({
                "timestamp": merged_5min.index.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "dni": dni.values,
                "dhi": dhi.values,
                "ghi": ghi.values,
                "T": merged_5min["T"].values if "T" in merged_5min.columns else np.nan,
                "wind": merged_5min["wind"].values if "wind" in merged_5min.columns else np.nan,
                "albedo": albedo.values,
            })

            csv_path = out / f"extra_data_{date_obj}.csv"
            result.to_csv(csv_path, index=False)
            saved.append((str(date_obj), str(csv_path)))

        except Exception as e:
            print(f"  {date_obj}: FAILED — {e}")

    print(f"\nDone: {len(saved)}/{len(unique_dates)} days saved to {out}/")
    return saved


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: extra_data_loader factory for evaluate_all_days
# ═══════════════════════════════════════════════════════════════════════════

def make_extra_data_loader(
    rad_csv: Union[str, Path],
    temp_wind_csv: Union[str, Path],
    latitude: float,
    longitude: float,
    default_albedo: float = 0.2,
):
    """
    Returns a callable(date) -> DataFrame suitable for evaluate_all_days.

    Loads from pre-built CSVs if available, otherwise builds on the fly.

    Usage:
        loader = make_extra_data_loader(rad_csv, tw_csv, lat, lon)
        results = evaluate_all_days(..., extra_data_loader=loader, ...)
    """
    from pv_analysis_re import load_extra_data_csv

    def _loader(date_obj):
        date_str = pd.to_datetime(date_obj).strftime("%Y-%m-%d")
        csv_path = Path(f"output/extra_data_{date_str}.csv")

        if csv_path.exists():
            return load_extra_data_csv(str(csv_path))
        else:
            # Build on the fly
            df = load_extra_for_day(
                rad_csv, temp_wind_csv, date_str,
                latitude, longitude,
                default_albedo=default_albedo,
            )
            if not df.empty:
                # Also save for future use
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csv_path, index=False)
            return df

    return _loader


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        rad = sys.argv[1]
        tw = sys.argv[2]
        lat = float(sys.argv[3]) if len(sys.argv) > 3 else 62.8924
        lon = float(sys.argv[4]) if len(sys.argv) > 4 else 27.6353
        build_all_extra_data(rad, tw, lat, lon)
    else:
        print("Usage: python build_extra_data.py rad.csv temp_wind.csv [lat] [lon]")
