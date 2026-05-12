"""
_nb_modifier_revision.py
Add revision-required cells to VoxSolaris notebooks.

Changes per notebook:
  main_analysis_b1_f.ipynb / main_analysis_b1_re.ipynb:
    A) After cell 43e1bc26 (find_clear_days): daylight-fraction ranking comparison
    B) After cell b3ac1388 (load extra data):  DNI QC candidate report
    C) After cell d1ced851 (single-day metrics): residual diagnostic for 4 Jul 10:15

  main_analysis_b2_f.ipynb:
    A) After cell 8c7bdade (find_clear_days): daylight-fraction ranking comparison
"""
import json
from pathlib import Path


def make_code_cell(source_lines, cell_id):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


def insert_after(cells, after_id, new_cell):
    for i, c in enumerate(cells):
        if c.get("id") == after_id:
            cells.insert(i + 1, new_cell)
            return True
    return False


def load_nb(path):
    with open(path) as f:
        return json.load(f)


def save_nb(nb, path):
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Saved: {path}")


# ─── Cell A: daylight-fraction comparison (Bank 1 notebooks) ─────────────────

CELL_A_B1_SRC = [
    "# %% --- Reviewer TODO: 03_Site_Data.tex §Clear-Sky Day Selection ---\n",
    "# Re-rank clear days using fraction of clear minutes within daylight hours\n",
    "# rather than raw clear-minute counts.\n",
    "# At high latitudes, summer days have up to 19 h of daylight, so a day\n",
    "# with 800 raw clear minutes may be less 'clear' than a shorter day with 700.\n",
    "#\n",
    "# find_clear_days() already supports ranking_mode='daylight_fraction'.\n",
    "# Run this cell and compare the two date sets.\n",
    "#\n",
    "# TODO[author]: If the set changes, update the selected days, Table text,\n",
    "#   and derived metrics. If unchanged, add a note confirming robustness.\n",
    "\n",
    "clear_days_fraction = find_clear_days(\n",
    "    CLEAR_MINUTES,\n",
    "    threshold=1.05,\n",
    "    ranking_mode=\"daylight_fraction\",\n",
    "    latitude=cfg.latitude,\n",
    "    longitude=cfg.longitude,\n",
    "    timestamp_tz=\"UTC\",\n",
    "    daylight_altitude_deg=0.0,\n",
    ")\n",
    "\n",
    "lc_dates = set(str(d) for d in clear_days[\"Date\"])\n",
    "fr_dates = set(str(d) for d in clear_days_fraction[\"Date\"])\n",
    "\n",
    "print(\"\\n\" + \"=\" * 55)\n",
    "print(\"  CLEAR-SKY DAY SELECTION — COMPARISON\")\n",
    "print(\"=\" * 55)\n",
    "print(f\"  Line-count mode  ({len(lc_dates):2d} days): {sorted(lc_dates)}\")\n",
    "print(f\"  Fraction mode    ({len(fr_dates):2d} days): {sorted(fr_dates)}\")\n",
    "\n",
    "added = fr_dates - lc_dates\n",
    "removed = lc_dates - fr_dates\n",
    "if not added and not removed:\n",
    "    print(\"\\n  >>> SAME SET: daylight-fraction ranking selects identical days.\")\n",
    "    print(\"      TODO[author]: Confirm this result in the manuscript.\")\n",
    "else:\n",
    "    print(f\"\\n  >>> SET CHANGED: +{len(added)} new | -{len(removed)} removed\")\n",
    "    if added:\n",
    "        print(f\"      Added:   {sorted(added)}\")\n",
    "    if removed:\n",
    "        print(f\"      Removed: {sorted(removed)}\")\n",
    "    print(\"  >>> TODO[author]: Decide whether to use 'fraction' ranking\")\n",
    "    print(\"      and update the manuscript accordingly.\")\n",
    "\n",
    "frac_cols = [c for c in [\"Date\", \"ClearMinutes\", \"DaylightMinutes\", \"ClearFraction\"]\n",
    "             if c in clear_days_fraction.columns]\n",
    "if frac_cols:\n",
    "    print(\"\\n  Daylight-fraction ranking (top 12 rows):\")\n",
    "    print(clear_days_fraction[frac_cols].head(12).to_string(index=False))\n",
]

# ─── Cell A: daylight-fraction comparison (Bank 2 notebook) ──────────────────

CELL_A_B2_SRC = [
    "# %% --- Reviewer TODO: 03_Site_Data.tex §Clear-Sky Day Selection (Bank 2) ---\n",
    "# Re-rank clear days using fraction of clear minutes within daylight hours.\n",
    "# Uses cfg_combined coordinates (same lat/lon as Bank 1).\n",
    "#\n",
    "# TODO[author]: Compare result with the Bank 1 comparison. If the set changes,\n",
    "#   update selected days and metrics in the manuscript.\n",
    "\n",
    "clear_days_fraction = find_clear_days(\n",
    "    CLEAR_MINUTES,\n",
    "    threshold=1.05,\n",
    "    ranking_mode=\"daylight_fraction\",\n",
    "    latitude=cfg_combined.latitude,\n",
    "    longitude=cfg_combined.longitude,\n",
    "    timestamp_tz=\"UTC\",\n",
    "    daylight_altitude_deg=0.0,\n",
    ")\n",
    "\n",
    "lc_dates = set(str(d) for d in clear_days[\"Date\"])\n",
    "fr_dates = set(str(d) for d in clear_days_fraction[\"Date\"])\n",
    "\n",
    "print(\"\\n\" + \"=\" * 55)\n",
    "print(\"  CLEAR-SKY DAY SELECTION — COMPARISON (Bank 2)\")\n",
    "print(\"=\" * 55)\n",
    "print(f\"  Line-count mode  ({len(lc_dates):2d} days): {sorted(lc_dates)}\")\n",
    "print(f\"  Fraction mode    ({len(fr_dates):2d} days): {sorted(fr_dates)}\")\n",
    "\n",
    "added = fr_dates - lc_dates\n",
    "removed = lc_dates - fr_dates\n",
    "if not added and not removed:\n",
    "    print(\"\\n  >>> SAME SET: both modes select identical days.\")\n",
    "else:\n",
    "    print(f\"\\n  >>> SET CHANGED: +{len(added)} new | -{len(removed)} removed\")\n",
    "    if added:\n",
    "        print(f\"      Added:   {sorted(added)}\")\n",
    "    if removed:\n",
    "        print(f\"      Removed: {sorted(removed)}\")\n",
    "    print(\"  >>> TODO[author]: Decide whether to use 'fraction' ranking.\")\n",
    "\n",
    "frac_cols = [c for c in [\"Date\", \"ClearMinutes\", \"DaylightMinutes\", \"ClearFraction\"]\n",
    "             if c in clear_days_fraction.columns]\n",
    "if frac_cols:\n",
    "    print(\"\\n  Daylight-fraction ranking (top 12 rows):\")\n",
    "    print(clear_days_fraction[frac_cols].head(12).to_string(index=False))\n",
]

# ─── Cell B: DNI QC candidate report (Bank 1 notebooks only) ─────────────────

CELL_B_SRC = [
    "# %% --- Reviewer TODO: 03_Site_Data.tex §DNI Overshoot Screening ---\n",
    "# Generate a candidate DNI quality-control flag report for the full season.\n",
    "# Flags produced (all configurable):\n",
    "#   low_solar_elevation      — elevation < min_solar_elevation_deg (default 5°)\n",
    "#   negative_ghi / dhi       — unphysical negative irradiance\n",
    "#   dhi_gt_ghi               — diffuse exceeds global irradiance\n",
    "#   dni_above_static_limit   — only if max_dni_wm2 is set\n",
    "#   dni_above_clearsky_ratio — only if max_clearsky_ratio is set\n",
    "#\n",
    "# TODO[author]: After reviewing output/dni_qc_report_revised.csv, confirm:\n",
    "#   1. Was any low-elevation or overshoot screening applied in the analysis?\n",
    "#   2. If yes: document the criterion (e.g., elevation < 5°) and flagged-sample\n",
    "#      counts in Section 3.4 of the manuscript.\n",
    "#   3. If no: discuss as a data-quality limitation.\n",
    "# Thresholds must be confirmed from the reference paper before any filtering.\n",
    "\n",
    "from build_extra_data import build_dni_qc_report, summarize_dni_qc\n",
    "\n",
    "# FMI Kuopio Savilahti station coordinates (source of irradiance measurements)\n",
    "FMI_LAT, FMI_LON = 62.8924, 27.6353\n",
    "\n",
    "qc_report = build_dni_qc_report(\n",
    "    rad_csv=RAD_FILE,\n",
    "    temp_wind_csv=TEMP_WIND_FILE,\n",
    "    latitude=FMI_LAT,\n",
    "    longitude=FMI_LON,\n",
    "    start_date=\"2021-04-01\",\n",
    "    end_date=\"2021-09-30\",\n",
    "    resample_min=5,\n",
    "    min_solar_elevation_deg=5.0,   # TODO[author]: confirm threshold from reference paper\n",
    "    max_dni_wm2=None,              # TODO[author]: set if a static upper bound was applied\n",
    "    max_clearsky_ratio=None,       # TODO[author]: set if a clearsky-ratio test was applied\n",
    "    output_csv=\"output/dni_qc_report_revised.csv\",\n",
    ")\n",
    "\n",
    "print(\"\\nDNI QC Summary — full season (Apr-Sep 2021, 5-min data):\")\n",
    "print(summarize_dni_qc(qc_report).to_string())\n",
    "print(\"\\nFull flag report saved to: output/dni_qc_report_revised.csv\")\n",
    "print(\"TODO[author]: Review 'low_solar_elevation' count and document in manuscript.\")\n",
]

# ─── Cell C: residual diagnostic (Bank 1 notebooks only) ─────────────────────

CELL_C_SRC = [
    "# %% --- Reviewer TODO: 07_results.tex §Case Study 4 July 2021 (~10:15) ---\n",
    "# Diagnostic for the residual mismatch around 10:15 local time on 4 July 2021.\n",
    "# Produces:\n",
    "#   results/residual_diagnostic_b1_20210704_revised.png  — 2-panel plot\n",
    "#   results/residual_diagnostic_b1_20210704_revised.csv  — data table\n",
    "#\n",
    "# PREREQUISITE: day_data, forecast_base, forecast_windowed, extra_data_df\n",
    "#   must be in scope from the single-day analysis cells above (second_day = 2021-07-04).\n",
    "#   Re-run those cells if the kernel was restarted.\n",
    "#\n",
    "# TODO[author]: After reviewing the diagnostic plot and CSV, add a sentence\n",
    "#   to Section 7 (Results §Single-Day Case Study) explaining the physical\n",
    "#   cause of the ~10:15 discrepancy. Check the irradiance (ghi/dhi/dni) and\n",
    "#   power traces for cloud edges, thin cirrus, or other transient effects.\n",
    "#   Do NOT interpret scientifically without checking the data first.\n",
    "\n",
    "from pv_analysis_re import plot_residual_diagnostics\n",
    "import os\n",
    "os.makedirs(\"results\", exist_ok=True)\n",
    "\n",
    "diagnostic_df = plot_residual_diagnostics(\n",
    "    day_data=day_data,\n",
    "    forecast_base=forecast_base,\n",
    "    forecast_windowed=forecast_windowed,\n",
    "    df_extra=extra_data_df,      # UTC-indexed; shifted to local time inside the function\n",
    "    cfg=cfg,\n",
    "    target_time=\"2021-07-04 10:15\",  # Local Helsinki time (EEST = UTC+3 in summer)\n",
    "    window=\"90min\",\n",
    "    save_path=\"results/residual_diagnostic_b1_20210704_revised.png\",\n",
    "    table_path=\"results/residual_diagnostic_b1_20210704_revised.csv\",\n",
    ")\n",
    "\n",
    "display_cols = [c for c in [\"Measured_W\", \"Baseline_W\", \"ShadowCorrected_W\",\n",
    "                             \"Residual_ShadowCorrected_W\", \"ghi\", \"dhi\", \"dni\"]\n",
    "                if c in diagnostic_df.columns]\n",
    "print(\"\\nDiagnostic window (+-90 min around 10:15 local time):\")\n",
    "print(diagnostic_df[display_cols].to_string())\n",
    "print(\"\\nOutput files:\")\n",
    "print(\"  results/residual_diagnostic_b1_20210704_revised.png\")\n",
    "print(\"  results/residual_diagnostic_b1_20210704_revised.csv\")\n",
    "print(\"\\nTODO[author]: Interpret the irradiance and power data at ~10:15.\")\n",
]


# ─── Apply to notebooks ───────────────────────────────────────────────────────

base = Path("/Users/hdong/Projects/VoxSolaris2")


def modify_b1(filename):
    path = base / filename
    nb = load_nb(path)
    cells = nb["cells"]

    ok_a = insert_after(cells, "43e1bc26",
                         make_code_cell(CELL_A_B1_SRC, "rev_daylight_frac_b1"))
    ok_b = insert_after(cells, "b3ac1388",
                         make_code_cell(CELL_B_SRC, "rev_dni_qc_b1"))
    ok_c = insert_after(cells, "d1ced851",
                         make_code_cell(CELL_C_SRC, "rev_residual_diag_b1"))

    if not ok_a:
        print(f"  WARNING: cell 43e1bc26 not found in {filename}")
    if not ok_b:
        print(f"  WARNING: cell b3ac1388 not found in {filename}")
    if not ok_c:
        print(f"  WARNING: cell d1ced851 not found in {filename}")

    save_nb(nb, path)
    return (ok_a, ok_b, ok_c)


def modify_b2(filename):
    path = base / filename
    nb = load_nb(path)
    cells = nb["cells"]

    ok_a = insert_after(cells, "8c7bdade",
                         make_code_cell(CELL_A_B2_SRC, "rev_daylight_frac_b2"))
    if not ok_a:
        print(f"  WARNING: cell 8c7bdade not found in {filename}")

    save_nb(nb, path)
    return ok_a


print("=== VoxSolaris notebook revision modifier ===\n")

print("Processing main_analysis_b1_f.ipynb ...")
r = modify_b1("main_analysis_b1_f.ipynb")
print(f"  Cells inserted: A={r[0]}, B={r[1]}, C={r[2]}")

print("\nProcessing main_analysis_b1_re.ipynb ...")
r = modify_b1("main_analysis_b1_re.ipynb")
print(f"  Cells inserted: A={r[0]}, B={r[1]}, C={r[2]}")

print("\nProcessing main_analysis_b2_f.ipynb ...")
r = modify_b2("main_analysis_b2_f.ipynb")
print(f"  Cell A inserted: {r}")

print("\nDone.")
