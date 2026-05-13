"""
Regenerate manuscript figures whose typography was flagged in Anders's review.

The LaTeX manuscript is the source of truth for output filenames. This script
writes only the five confirmed open-item figures directly to the article figs
directory, preserving the existing scientific data and model logic.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pvlib
import tqdm

from plot_daily_distributions import plot_ridgeline
from pv_analysis_re import (
    SiteConfig as Bank1Config,
    _plot_day_comparison,
    compute_metrics as compute_metrics_b1,
    evaluate_performance,
    find_clear_days,
    load_and_smooth_shadow_matrix as load_shadow_b1,
    load_extra_data_csv as load_extra_b1,
    load_inverter_data as load_inverter_b1,
    pv_analysis as pv_analysis_b1,
    print_performance_summary as print_summary_b1,
)
from pv_analysis_b2 import (
    SiteConfig as Bank2Config,
    compute_metrics as compute_metrics_b2,
    load_and_smooth_shadow_matrix as load_shadow_b2,
    load_extra_data_csv as load_extra_b2,
    load_inverter_data as load_inverter_b2,
    pv_analysis as pv_analysis_b2,
    print_performance_summary as print_summary_b2,
)
from visual_utils import plot_shadow_matrix_with_sunpaths, plot_shadow_polar_b2


PROJECT_DIR = Path(__file__).resolve().parent
ARTICLE_FIG_DIR = Path(
    os.environ.get(
        "VOXSOLARIS_ARTICLE_FIG_DIR",
        "/Users/hdong/Projects/VoxSolaris_Article/figs",
    )
)

RAD_FILE = PROJECT_DIR / "data/pvdata/Kuopio Savilahti 1.4.2021 - 1.10.2021_rad.csv"
TEMP_WIND_FILE = PROJECT_DIR / "data/pvdata/Kuopio Savilahti 1.4.2021 - 1.10.2021_temp_wind.csv"
CLEAR_MINUTES = PROJECT_DIR / "data/Clear_sky_minutes_kuopio_RH16.txt"
PV_EXCEL = PROJECT_DIR / "data/pvdata/pv_21.xlsx"
EXTRA_DATA_DIR = PROJECT_DIR / "output"

BANK1_SHADOW_CSV = (
    PROJECT_DIR
    / "results/shadow_matrix_results_SE_pro/shadow_attenuation_matrix_conecasting_SE_v10.csv"
)
BANK2_SHADOW_NORTH = (
    PROJECT_DIR
    / "results/shadow_matrix_results_W_pro/shadow_attenuation_matrix_conecasting_W_north_v3.csv"
)
BANK2_SHADOW_SOUTH = (
    PROJECT_DIR
    / "results/shadow_matrix_results_W_pro/shadow_attenuation_matrix_conecasting_W_south_v3.csv"
)


def replace_dni_with_computed(df_extra: pd.DataFrame, latitude: float, longitude: float) -> pd.DataFrame:
    """Mirror the notebook DNI-closure preprocessing used for the manuscript figures."""
    df_extra = df_extra.copy()
    solpos = pvlib.solarposition.get_solarposition(df_extra.index, latitude, longitude)
    dni_computed = pvlib.irradiance.dni(
        ghi=df_extra["ghi"],
        dhi=df_extra["dhi"],
        zenith=solpos["apparent_zenith"],
    )
    df_extra["dni_cams"] = df_extra["dni"].copy()
    df_extra["dni"] = dni_computed.clip(lower=0)
    return df_extra


def build_clear_days_fraction(cfg: Bank1Config) -> pd.DataFrame:
    return find_clear_days(
        CLEAR_MINUTES,
        threshold=0.85,
        ranking_mode="daylight_fraction",
        latitude=cfg.latitude,
        longitude=cfg.longitude,
        timestamp_tz="UTC",
        daylight_altitude_deg=0.0,
    )


def make_bank1_config() -> Bank1Config:
    return Bank1Config(
        latitude=62.979849,
        longitude=27.648656,
        tilt_deg=12.0,
        azimuth_deg=170.0,
        nominal_power_kw=3.96,
        system_efficiency=0.8,
        local_tz="Europe/Helsinki",
        inverter_utc_offset_hours=3,
        forecast_shift_minutes=-30,
        window_size=(3, 3),
        interval="5min",
        interval_minutes=5.0,
    )


def make_bank2_configs() -> tuple[Bank2Config, Bank2Config, Bank2Config]:
    nominal_total_w = 4620.0
    common = dict(
        latitude=62.979849,
        longitude=27.648656,
        tilt_deg=20.0,
        azimuth_deg=260.0,
        system_efficiency=0.85,
        local_tz="Europe/Helsinki",
        inverter_utc_offset_hours=3,
        window_size=(2, 2),
        interval="5min",
        interval_minutes=5.0,
        forecast_shift_minutes=-30,
    )
    cfg_north = Bank2Config(nominal_power_kw=nominal_total_w * 8 / 14 / 1000, **common)
    cfg_south = Bank2Config(nominal_power_kw=nominal_total_w * 6 / 14 / 1000, **common)
    cfg_combined = Bank2Config(nominal_power_kw=nominal_total_w / 1000, **common)
    return cfg_north, cfg_south, cfg_combined


def load_bank1_extra(date_obj, cfg: Bank1Config) -> pd.DataFrame:
    df = load_extra_b1(EXTRA_DATA_DIR / f"extra_data_{date_obj}.csv", cfg=cfg, recompute_albedo=True)
    return replace_dni_with_computed(df, cfg.latitude, cfg.longitude)


def load_bank2_extra(date_obj, cfg: Bank2Config) -> pd.DataFrame:
    df = load_extra_b2(EXTRA_DATA_DIR / f"extra_data_{date_obj}.csv", cfg=cfg)
    return replace_dni_with_computed(df, cfg.latitude, cfg.longitude)


def pv_analysis_bank2(
    target_date,
    shadow_north: np.ndarray,
    shadow_south: np.ndarray,
    excel_df: pd.DataFrame,
    df_extra: pd.DataFrame,
    cfg_north: Bank2Config,
    cfg_south: Bank2Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    day_n, fb_n, fw_n = pv_analysis_b2(
        target_date, shadow_north, excel_df, df_extra, cfg=cfg_north, plot=False
    )
    _, fb_s, fw_s = pv_analysis_b2(
        target_date, shadow_south, excel_df, df_extra, cfg=cfg_south, plot=False
    )

    fb_combined = fb_n.copy()
    fb_combined["output"] = fb_n["output"] + fb_s["output"]

    fw_combined = fw_n.copy()
    fw_combined["output_shaded"] = fw_n["output_shaded"] + fw_s["output_shaded"]

    return day_n.copy(), fb_combined, fw_combined


def evaluate_bank2_clear_days(
    clear_days_fraction: pd.DataFrame,
    shadow_north: np.ndarray,
    shadow_south: np.ndarray,
    excel_df: pd.DataFrame,
    cfg_north: Bank2Config,
    cfg_south: Bank2Config,
    cfg_combined: Bank2Config,
) -> pd.DataFrame:
    rows = []
    all_real = []
    all_base = []
    all_shaded = []

    for date_obj in tqdm.tqdm(clear_days_fraction["Date"].tolist(), desc="Evaluating Bank 2"):
        df_extra = load_bank2_extra(date_obj, cfg_combined)
        day_data, fb, fw = pv_analysis_bank2(
            date_obj, shadow_north, shadow_south, excel_df, df_extra, cfg_north, cfg_south
        )
        metrics = compute_metrics_b2(day_data, fb, fw, cfg_combined.interval_minutes)
        metrics["Date"] = str(date_obj)
        rows.append(metrics)

        mask = day_data["Power_W"] > 10
        all_real.extend(day_data.loc[mask, "Power_W"].to_numpy())
        all_base.extend(fb.loc[mask, "output"].to_numpy())
        all_shaded.extend(fw.loc[mask, "output_shaded"].to_numpy())

    results_df = pd.DataFrame(rows)
    cols = ["Date"] + [c for c in results_df.columns if c != "Date"]
    results_df = results_df[cols]
    print_summary_b2(
        results_df,
        real_arr=np.asarray(all_real),
        base_arr=np.asarray(all_base),
        shaded_arr=np.asarray(all_shaded),
    )
    return results_df


def regenerate_figures() -> None:
    ARTICLE_FIG_DIR.mkdir(parents=True, exist_ok=True)

    cfg_b1 = make_bank1_config()
    cfg_b2_north, cfg_b2_south, cfg_b2_combined = make_bank2_configs()

    print("Writing Fig. 2:", ARTICLE_FIG_DIR / "SAM_bank1.png")
    plot_shadow_matrix_with_sunpaths(
        BANK1_SHADOW_CSV,
        lat=cfg_b1.latitude,
        lon=cfg_b1.longitude,
        fill_missing=True,
        panel_tilt_deg=cfg_b1.tilt_deg,
        panel_az_deg=cfg_b1.azimuth_deg,
        save_path=ARTICLE_FIG_DIR / "SAM_bank1.png",
        show=False,
    )

    print("Writing Fig. 3:", ARTICLE_FIG_DIR / "SAM_bank2.png")
    plot_shadow_polar_b2(
        BANK2_SHADOW_NORTH,
        BANK2_SHADOW_SOUTH,
        lat=cfg_b2_combined.latitude,
        lon=cfg_b2_combined.longitude,
        fill_missing=True,
        panel_tilt_deg=cfg_b2_combined.tilt_deg,
        panel_az_deg=cfg_b2_combined.azimuth_deg,
        save_path=ARTICLE_FIG_DIR / "SAM_bank2.png",
        show=False,
    )

    clear_days_fraction = build_clear_days_fraction(cfg_b1)

    print("Evaluating Bank 1 clear-sky subset.")
    shadow_b1 = load_shadow_b1(BANK1_SHADOW_CSV, window_size=cfg_b1.window_size)
    pv_b1 = load_inverter_b1(PV_EXCEL, expected_interval_min=cfg_b1.interval_minutes)
    b1_results, _, _, _ = evaluate_performance(
        significant_days_df=clear_days_fraction,
        shadow_matrix=shadow_b1,
        excel_df=pv_b1,
        extra_data_loader=lambda date_obj: load_bank1_extra(date_obj, cfg_b1),
        cfg=cfg_b1,
    )
    print_summary_b1(b1_results)

    target_date = "2021-07-04"
    print("Writing Fig. 4:", ARTICLE_FIG_DIR / "b1_july04_timeseries.png")
    day_data, forecast_base, forecast_windowed = pv_analysis_b1(
        target_date,
        shadow_b1,
        pv_b1,
        load_bank1_extra(target_date, cfg_b1),
        cfg=cfg_b1,
        plot=False,
    )
    _plot_day_comparison(
        day_data,
        forecast_base,
        forecast_windowed,
        target_date,
        day_data.index,
        save_path=ARTICLE_FIG_DIR / "b1_july04_timeseries.png",
    )

    print("Writing Fig. 7:", ARTICLE_FIG_DIR / "b1_clearsky_ridgeline.png")
    plot_ridgeline(
        b1_results,
        save_path=ARTICLE_FIG_DIR / "b1_clearsky_ridgeline.png",
    )

    print("Evaluating Bank 2 clear-sky subset.")
    shadow_b2_north = load_shadow_b2(BANK2_SHADOW_NORTH, window_size=cfg_b2_north.window_size)
    shadow_b2_south = load_shadow_b2(BANK2_SHADOW_SOUTH, window_size=cfg_b2_south.window_size)
    pv_b2 = load_inverter_b2(
        PV_EXCEL,
        expected_interval_min=cfg_b2_combined.interval_minutes,
        energy_column="MPP2",
    )
    b2_results = evaluate_bank2_clear_days(
        clear_days_fraction,
        shadow_b2_north,
        shadow_b2_south,
        pv_b2,
        cfg_b2_north,
        cfg_b2_south,
        cfg_b2_combined,
    )

    print("Writing Fig. 11:", ARTICLE_FIG_DIR / "b2_clearsky_ridgeline.png")
    plot_ridgeline(
        b2_results,
        save_path=ARTICLE_FIG_DIR / "b2_clearsky_ridgeline.png",
    )


if __name__ == "__main__":
    regenerate_figures()
