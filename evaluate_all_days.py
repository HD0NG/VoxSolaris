"""
evaluate_all_days.py — Compute overall RMSE, MAE, MBE, R², and energy bias
across ALL days in a date range (e.g. May–September), not just clear days.

Usage in main_analysis.py:
    from evaluate_all_days import evaluate_all_days

    overall_results, all_real, all_base, all_shad = evaluate_all_days(
        start_date="2021-05-01",
        end_date="2021-09-30",
        shadow_matrix=shadow_matrix,
        excel_df=excel_df,
        extra_data_loader=_load_cached_extra,
        cfg=cfg,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_all_days(
    start_date: str,
    end_date: str,
    shadow_matrix: np.ndarray,
    excel_df: pd.DataFrame,
    extra_data_loader: Callable,
    cfg,
    min_power_threshold: float = 10.0,
    output_csv: Optional[str] = None,
    analysis_func: Optional[Callable] = None,
) -> Tuple[pd.DataFrame, list, list, list]:
    """
    Run pv_analysis for every day in [start_date, end_date] and compute
    per-day + overall aggregate metrics.

    Parameters
    ----------
    start_date, end_date : str
        "YYYY-MM-DD" range (inclusive).
    shadow_matrix : np.ndarray
    excel_df : DataFrame — full inverter data.
    extra_data_loader : callable(date) -> DataFrame
    cfg : SiteConfig
    min_power_threshold : float
        Exclude timesteps where real power < threshold (nighttime filter).
    output_csv : str, optional
        If given, save per-day results to this CSV.
    analysis_func : callable, optional
        Defaults to pv_analysis_re.pv_analysis. Bank 2 notebooks can pass their
        dual-matrix wrapper directly instead of monkey-patching imports.

    Returns
    -------
    results_df : DataFrame — per-day metrics
    all_real, all_base, all_shad : lists — concatenated 5-min values (daytime only)
    """
    import tqdm

    if analysis_func is None:
        from pv_analysis_re import pv_analysis as analysis_func

    dates = pd.date_range(start_date, end_date, freq="D")
    print(f"Evaluating {len(dates)} days: {start_date} → {end_date}")

    daily_stats = []
    all_real = []
    all_base = []
    all_shad = []
    skipped = 0

    for date_ts in tqdm.tqdm(dates, desc="All days"):
        date_obj = date_ts.date()

        try:
            df_extra = extra_data_loader(date_obj)
            if df_extra is None or df_extra.empty:
                skipped += 1
                continue

            day_data, fb_n, fw_n = analysis_func(
                date_obj, shadow_matrix, excel_df, df_extra, cfg=cfg, plot=False
            )

            # Align indices
            real   = day_data["Power_W"].fillna(0.0)
            base   = fb_n["output"].reindex(real.index, fill_value=0.0).fillna(0.0)
            shaded = fw_n["output_shaded"].reindex(real.index, fill_value=0.0).fillna(0.0)

            # Daytime filter
            daytime = real > min_power_threshold
            if daytime.sum() < 10:
                skipped += 1
                continue

            r = real[daytime].values
            b = base[daytime].values
            s = shaded[daytime].values

            # Per-day metrics
            hours_per_step = cfg.interval_minutes / 60.0
            metrics = {
                "Date": str(date_obj),
                "RMSE_Base":    np.sqrt(mean_squared_error(r, b)),
                "RMSE_Shaded":  np.sqrt(mean_squared_error(r, s)),
                "MAE_Base":     mean_absolute_error(r, b),
                "MAE_Shaded":   mean_absolute_error(r, s),
                "MBE_Base":     float(np.mean(b - r)),
                "MBE_Shaded":   float(np.mean(s - r)),
                "R2_Base":      r2_score(r, b) if len(r) > 2 else np.nan,
                "R2_Shaded":    r2_score(r, s) if len(r) > 2 else np.nan,
                "Real_Wh":      float(real.clip(lower=0).sum() * hours_per_step),
                "Base_Wh":      float(base.clip(lower=0).sum() * hours_per_step),
                "Shaded_Wh":    float(shaded.clip(lower=0).sum() * hours_per_step),
                "N_daytime":    int(daytime.sum()),
            }
            daily_stats.append(metrics)

            all_real.extend(r.tolist())
            all_base.extend(b.tolist())
            all_shad.extend(s.tolist())

        except Exception as e:
            skipped += 1
            continue

    results_df = pd.DataFrame(daily_stats)

    if results_df.empty:
        print("No days were successfully evaluated.")
        return results_df, all_real, all_base, all_shad

    # ── Save per-day CSV ────────────────────────────────────────────────────
    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"Per-day results saved to: {output_csv}")

    # ── Print overall summary ───────────────────────────────────────────────
    n_days = len(results_df)
    r_all = np.array(all_real)
    b_all = np.array(all_base)
    s_all = np.array(all_shad)

    print(f"\n{'='*65}")
    print(f"  OVERALL METRICS — {start_date} to {end_date}")
    print(f"  {n_days} days evaluated, {skipped} skipped, {len(r_all)} daytime samples")
    print(f"{'='*65}")

    # Overall from concatenated samples
    overall_rmse_b = np.sqrt(mean_squared_error(r_all, b_all))
    overall_rmse_s = np.sqrt(mean_squared_error(r_all, s_all))
    overall_mae_b  = mean_absolute_error(r_all, b_all)
    overall_mae_s  = mean_absolute_error(r_all, s_all)
    overall_mbe_b  = float(np.mean(b_all - r_all))
    overall_mbe_s  = float(np.mean(s_all - r_all))
    overall_r2_b   = r2_score(r_all, b_all)
    overall_r2_s   = r2_score(r_all, s_all)

    # Energy totals
    total_real = results_df["Real_Wh"].sum() / 1000
    total_base = results_df["Base_Wh"].sum() / 1000
    total_shad = results_df["Shaded_Wh"].sum() / 1000
    energy_err_b = (total_base - total_real) / total_real * 100
    energy_err_s = (total_shad - total_real) / total_real * 100

    print(f"\n  {'Metric':<20s} {'Baseline':>12s} {'Shadow-Corr':>12s} {'Improvement':>12s}")
    print(f"  {'-'*56}")
    print(f"  {'RMSE (W)':<20s} {overall_rmse_b:>12.1f} {overall_rmse_s:>12.1f} {(overall_rmse_b - overall_rmse_s) / overall_rmse_b * 100:>11.1f}%")
    print(f"  {'MAE (W)':<20s} {overall_mae_b:>12.1f} {overall_mae_s:>12.1f} {(overall_mae_b - overall_mae_s) / overall_mae_b * 100:>11.1f}%")
    print(f"  {'MBE (W)':<20s} {overall_mbe_b:>12.1f} {overall_mbe_s:>12.1f} {'closer' if abs(overall_mbe_s) < abs(overall_mbe_b) else 'further':>12s}")
    print(f"  {'R²':<20s} {overall_r2_b:>12.3f} {overall_r2_s:>12.3f} {(overall_r2_s - overall_r2_b):>+12.3f}")
    print(f"\n  {'Energy (kWh)':<20s} {'Real':>8s} {'Base':>10s} {'Corrected':>10s}")
    print(f"  {'-'*48}")
    print(f"  {'Total':<20s} {total_real:>8.1f} {total_base:>10.1f} {total_shad:>10.1f}")
    print(f"  {'Error':<20s} {'':>8s} {energy_err_b:>+9.1f}% {energy_err_s:>+9.1f}%")
    print(f"{'='*65}\n")

    # ── Mean of per-day metrics (for comparison) ────────────────────────────
    print(f"  Per-day averages (mean of {n_days} daily metrics):")
    print(f"  {'RMSE':<8s}  Base: {results_df['RMSE_Base'].mean():.1f} ± {results_df['RMSE_Base'].std():.1f}"
          f"   Corr: {results_df['RMSE_Shaded'].mean():.1f} ± {results_df['RMSE_Shaded'].std():.1f}")
    print(f"  {'MAE':<8s}  Base: {results_df['MAE_Base'].mean():.1f} ± {results_df['MAE_Base'].std():.1f}"
          f"   Corr: {results_df['MAE_Shaded'].mean():.1f} ± {results_df['MAE_Shaded'].std():.1f}")
    print(f"  {'MBE':<8s}  Base: {results_df['MBE_Base'].mean():.1f} ± {results_df['MBE_Base'].std():.1f}"
          f"   Corr: {results_df['MBE_Shaded'].mean():.1f} ± {results_df['MBE_Shaded'].std():.1f}")
    if "R2_Shaded" in results_df.columns:
        print(f"  {'R²':<8s}  Base: {results_df['R2_Base'].mean():.3f} ± {results_df['R2_Base'].std():.3f}"
              f"   Corr: {results_df['R2_Shaded'].mean():.3f} ± {results_df['R2_Shaded'].std():.3f}")
    print()

    return results_df, all_real, all_base, all_shad


# ═══════════════════════════════════════════════════════════════════════════
# Plotting — matching existing main_analysis.py bar chart style
# ═══════════════════════════════════════════════════════════════════════════

import matplotlib
import matplotlib.pyplot as plt


def plot_energy_bar_chart(results_df: pd.DataFrame, title_suffix: str = "",
                          save_path: Optional[str] = None):
    """Comparative bar chart: clearsky vs corrected vs real energy yield."""
    totals = results_df[["Real_Wh", "Base_Wh", "Shaded_Wh"]].sum() / 1000

    labels = ["FMI Base Forecast", "Shadow-Corrected", "Measured Output"]
    values = [totals["Base_Wh"], totals["Shaded_Wh"], totals["Real_Wh"]]
    target = values[2]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#5B9BD5", "#ED7D31", "#A5A5A5"]
    edge_colors = ["#4178A4", "#C46420", "#808080"]
    bars = ax.bar(labels, values, color=colors, edgecolor=edge_colors,
                  width=0.55, linewidth=1.2, zorder=3)

    ax.axhline(y=target, color="#404040", ls="--", lw=1.5, alpha=0.7, zorder=2)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h - max(values) * 0.015,
                f"{h:,.1f}", ha="center", va="top", fontsize=11, fontweight="bold",
                color="#333333")

    for i in range(2):
        err = values[i] - target
        cx = bars[i].get_x() + bars[i].get_width() / 2
        h = bars[i].get_height()
        sign = "+" if err >= 0 else ""
        err_pct = err / target * 100

        ax.annotate(
            f"{sign}{err:.1f} kWh\n({sign}{err_pct:.1f}%)",
            xy=(cx + 0.3, (h + target) / 2),
            fontsize=9, color="#444444", ha="left", va="center",
            fontweight="bold",
        )

    n_days = len(results_df)
    title = "Energy Yield: Models vs. Measured"
    if title_suffix:
        title += f" ({title_suffix})"

    ax.set_ylabel("Total Energy (kWh)", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.tick_params(axis="x", labelsize=10)

    ax.text(0.02, 0.97, f"{n_days} days", transform=ax.transAxes,
            fontsize=9, va="top", color="#888888")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps"):
        plt.show()
    return fig


def _plot_metric_comparison(base_val, shaded_val, metric_name, unit="W",
                            n_days=None, title_suffix="",
                            save_path=None):
    """Generic paired bar chart for a single metric."""
    imp = base_val - shaded_val
    imp_pct = (imp / base_val) * 100 if base_val != 0 else 0

    labels = ["FMI Base Forecast", "Shadow-Corrected"]
    values = [base_val, shaded_val]

    fig, ax = plt.subplots(figsize=(6, 5.5))

    colors = ["#5B9BD5", "#ED7D31"]
    edge_colors = ["#4178A4", "#C46420"]
    bars = ax.bar(labels, values, color=colors, edgecolor=edge_colors,
                  width=0.45, linewidth=1.2, zorder=3)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h - max(values) * 0.015,
                f"{h:,.1f}", ha="center", va="top", fontsize=12,
                fontweight="bold", color="#333333")

    cx0 = bars[0].get_x() + bars[0].get_width() / 2
    cx1 = bars[1].get_x() + bars[1].get_width() / 2
    mid_x = (cx0 + cx1) / 2
    mid_y = (values[0] + values[1]) / 2

    ax.annotate(
        "", xy=(cx1, values[1]), xytext=(cx0, values[0]),
        arrowprops=dict(arrowstyle="->", color="#676767", lw=1.5,
                        connectionstyle="arc3,rad=-0.2"),
        zorder=4,
    )
    ax.text(mid_x + 0.18, mid_y,
            f"-{imp:.1f} {unit}\n({imp_pct:.1f}%)",
            fontsize=11, color="#444444", ha="right", va="top",
            fontweight="bold")

    title = f"{metric_name}: Base vs. Shadow-Corrected"
    if title_suffix:
        title += f" ({title_suffix})"

    ax.set_ylabel(f"{metric_name} ({unit})", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylim(0, max(values) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.tick_params(axis="x", labelsize=10)

    if n_days is not None:
        ax.text(0.02, 0.97, f"{n_days} days", transform=ax.transAxes,
                fontsize=9, va="top", color="#888888")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps"):
        plt.show()
    return fig


def plot_mbe_bar_chart(mbe_base, mbe_shaded, n_days=None,
                       title_suffix="", save_path=None):
    """MBE bar chart — shows bias direction, closer to 0 is better."""
    labels = ["FMI Base Forecast", "Shadow-Corrected", "Perfect (0)"]
    values = [mbe_base, mbe_shaded, 0]

    fig, ax = plt.subplots(figsize=(7, 5.5))

    colors = ["#5B9BD5", "#ED7D31", "#A5A5A5"]
    edge_colors = ["#4178A4", "#C46420", "#808080"]
    bars = ax.bar(labels, values, color=colors, edgecolor=edge_colors,
                  width=0.45, linewidth=1.2, zorder=3)

    ax.axhline(0, color="#404040", ls="--", lw=1.5, alpha=0.7, zorder=2)

    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = max(abs(mbe_base), abs(mbe_shaded)) * 0.02
        y = h + offset if h >= 0 else h - offset
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f"{h:+.1f}", ha="center", va=va, fontsize=12,
                fontweight="bold", color="#333333")

    # Improvement annotation
    reduction = abs(mbe_base) - abs(mbe_shaded)
    reduction_pct = reduction / abs(mbe_base) * 100 if mbe_base != 0 else 0

    cx0 = bars[0].get_x() + bars[0].get_width() / 2
    cx1 = bars[1].get_x() + bars[1].get_width() / 2

    ax.annotate(
        "", xy=(cx1, mbe_shaded), xytext=(cx0, mbe_base),
        arrowprops=dict(arrowstyle="->", color="#676767", lw=1.5,
                        connectionstyle="arc3,rad=-0.2"),
        zorder=4,
    )
    ax.text((cx0 + cx1) / 2 + 0.18, (mbe_base + mbe_shaded) / 2,
            f"-{reduction:.1f} W\n({reduction_pct:.0f}% closer to 0)",
            fontsize=10, color="#444444", ha="right", va="top",
            fontweight="bold")

    title = "MBE: Base vs. Shadow-Corrected"
    if title_suffix:
        title += f" ({title_suffix})"

    ax.set_ylabel("MBE (W)", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)
    ymax = max(abs(mbe_base), abs(mbe_shaded)) * 1.4
    ax.set_ylim(-ymax * 0.2, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.tick_params(axis="x", labelsize=10)

    if n_days is not None:
        ax.text(0.02, 0.97, f"{n_days} days", transform=ax.transAxes,
                fontsize=9, va="top", color="#888888")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps"):
        plt.show()
    return fig


def plot_all_metrics(results_df, title_suffix="", save_dir=None):
    """
    Plot all four charts: Energy, RMSE, MAE, MBE.
    Call after evaluate_all_days().
    """
    if results_df.empty:
        print("No data to plot.")
        return

    n_days = len(results_df)
    prefix = f"{save_dir}/" if save_dir else ""
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    plot_energy_bar_chart(
        results_df, title_suffix=title_suffix,
        save_path=f"{prefix}energy_all_days.png" if save_dir else None,
    )

    _plot_metric_comparison(
        results_df["RMSE_Base"].mean(),
        results_df["RMSE_Shaded"].mean(),
        "RMSE", "W", n_days=n_days, title_suffix=title_suffix,
        save_path=f"{prefix}rmse_all_days.png" if save_dir else None,
    )

    _plot_metric_comparison(
        results_df["MAE_Base"].mean(),
        results_df["MAE_Shaded"].mean(),
        "MAE", "W", n_days=n_days, title_suffix=title_suffix,
        save_path=f"{prefix}mae_all_days.png" if save_dir else None,
    )

    plot_mbe_bar_chart(
        results_df["MBE_Base"].mean(),
        results_df["MBE_Shaded"].mean(),
        n_days=n_days, title_suffix=title_suffix,
        save_path=f"{prefix}mbe_all_days.png" if save_dir else None,
    )
