from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .helper import safe_file_name

sns.set_theme(style="whitegrid")

_PALETTE = "tab10"


def plot_lsoa(lsoa_code, lsoa_table, metrics, cv_forecasts, output_dir):
    plot_dir = Path(output_dir) / "plots" / lsoa_code
    plot_dir.mkdir(parents=True, exist_ok=True)

    monthly_total = (
        lsoa_table.groupby("Month", as_index=False)
        .agg(crime_count=("crime_count", "sum"))
        .sort_values("Month")
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(monthly_total, x="Month", y="crime_count", marker="o", ax=ax)
    ax.set_title(f"Monthly crime count — {lsoa_code}")
    fig.tight_layout()
    fig.savefig(plot_dir / "monthly_crime_counts.png", dpi=160)
    plt.close(fig)

    # temperature overlay (only if the column exists)
    if (
        "avg_temperature_c" in lsoa_table.columns
        and not lsoa_table["avg_temperature_c"].isna().all()
    ):
        temp_total = (
            lsoa_table.groupby("Month", as_index=False)
            .agg(
                crime_count=("crime_count", "sum"),
                avg_temperature_c=("avg_temperature_c", "first"),
            )
            .sort_values("Month")
        )
        fig, ax1 = plt.subplots(figsize=(12, 4))
        sns.lineplot(
            temp_total, x="Month", y="crime_count", marker="o", ax=ax1, label="Crime"
        )
        ax2 = ax1.twinx()
        sns.lineplot(
            temp_total,
            x="Month",
            y="avg_temperature_c",
            marker="s",
            color="#d95f02",
            ax=ax2,
            label="Temperature",
        )
        ax1.set_title(f"Crime and temperature — {lsoa_code}")
        fig.tight_layout()
        fig.savefig(plot_dir / "monthly_crime_and_temperature.png", dpi=160)
        plt.close(fig)

    ok_metrics = metrics[metrics["status"].eq("ok")]
    if not ok_metrics.empty:
        summary = ok_metrics.groupby("model", as_index=False).agg(
            mae=("mae", "mean"), rmse=("rmse", "mean"), smape=("smape", "mean")
        )
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for axis, metric in zip(axes, ["mae", "rmse", "smape"]):
            sns.barplot(summary, x="model", y=metric, ax=axis)
            axis.set_title(metric.upper())
        fig.tight_layout()
        fig.savefig(plot_dir / "model_metrics.png", dpi=160)
        plt.close(fig)

    for crime_type, history in lsoa_table.groupby("Crime type", sort=True):
        ok_preds = cv_forecasts[
            cv_forecasts["Crime type"].eq(crime_type) & cv_forecasts["status"].eq("ok")
        ]
        if not ok_preds.empty:
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.lineplot(
                history,
                x="Month",
                y="crime_count",
                color="black",
                marker="o",
                label="Actual",
                ax=ax,
            )
            sns.lineplot(
                ok_preds,
                x="Month",
                y="predicted",
                hue="model",
                marker="o",
                estimator="mean",
                errorbar=None,
                ax=ax,
            )
            ax.set_title(f"CV forecast: {crime_type}")
            fig.tight_layout()
            fig.savefig(plot_dir / f"cv_{safe_file_name(crime_type)}.png", dpi=160)
            plt.close(fig)


def plot_aggregate(mean_metrics, output_dir, name):
    if mean_metrics.empty:
        return None

    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"{name}_aggregate_metrics.png"

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for axis, metric in zip(axes, ["mae", "rmse", "smape"]):
        sns.barplot(mean_metrics, x="model", y=metric, ax=axis)
        axis.set_title(metric.upper())
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ── report figure ─────────────────────────────────────────────────────────────

_TOP_N_CRIMES = 8


def plot_report(output_dir: Path, run_name: str) -> Path | None:
    """
    Generate a single report-quality figure summarising all key results.

    Panels
    ------
    Top (full width) : 12-month crime forecast aggregated across all LSOAs,
                       one line per crime type (top N by volume).
    Bottom-left      : Mean MAE per model across all CV folds.
    Bottom-centre    : Best-model win rate — how often each model won.
    Bottom-right     : Total 12-month predicted crimes by crime type.

    Saved as ``report_figure.png`` directly in ``output_dir``.
    """
    output_dir = Path(output_dir)

    forecast_path = output_dir / "12month_forecast.csv"
    metrics_path = output_dir / f"{run_name}_combined_model_comparison.csv"
    best_path = output_dir / f"{run_name}_best_models.csv"

    if not forecast_path.exists():
        return None

    forecast_df = pd.read_csv(forecast_path, parse_dates=["Month"])

    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    best_df = pd.read_csv(best_path) if best_path.exists() else pd.DataFrame()

    # ── figure skeleton ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.6, 1],
        hspace=0.45,
        wspace=0.38,
    )
    ax_forecast = fig.add_subplot(gs[0, :])
    ax_mae = fig.add_subplot(gs[1, 0])
    ax_wins = fig.add_subplot(gs[1, 1])
    ax_crimes = fig.add_subplot(gs[1, 2])

    # ── Panel 1: 12-month forecast lines ─────────────────────────────────────
    monthly = forecast_df.groupby(["Month", "Crime type"], as_index=False)[
        "predicted"
    ].sum()
    # Keep top-N crime types by total volume; merge the rest into "Other"
    top_crimes = (
        monthly.groupby("Crime type")["predicted"]
        .sum()
        .nlargest(_TOP_N_CRIMES)
        .index.tolist()
    )
    monthly["crime_label"] = monthly["Crime type"].where(
        monthly["Crime type"].isin(top_crimes), other="Other"
    )
    monthly = monthly.groupby(["Month", "crime_label"], as_index=False)[
        "predicted"
    ].sum()

    order = (
        monthly.groupby("crime_label")["predicted"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    palette = dict(zip(order, sns.color_palette(_PALETTE, len(order))))

    for label in order:
        sub = monthly[monthly["crime_label"] == label].sort_values("Month")
        ax_forecast.plot(
            sub["Month"],
            sub["predicted"],
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=label,
            color=palette[label],
        )

    ax_forecast.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_forecast.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(
        ax_forecast.xaxis.get_majorticklabels(), rotation=40, ha="right", fontsize=8
    )
    ax_forecast.set_title(
        "12-Month Crime Forecast — Total Across All LSOAs",
        fontsize=13,
        fontweight="bold",
    )
    ax_forecast.set_xlabel("")
    ax_forecast.set_ylabel("Predicted Crime Count")
    ax_forecast.legend(
        title="Crime Type",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=8,
        title_fontsize=9,
        frameon=True,
    )

    # ── Panel 2: Median MAE by model (IQR whiskers) ───────────────────────────
    if not metrics_df.empty and "status" in metrics_df.columns:
        ok = metrics_df[metrics_df["status"].eq("ok")]
        if not ok.empty:
            model_stats = (
                ok.groupby("model")["mae"]
                .agg(
                    median="median",
                    q25=lambda x: x.quantile(0.25),
                    q75=lambda x: x.quantile(0.75),
                )
                .reset_index()
                .sort_values("median")
            )
            colors = sns.color_palette("Blues_d", len(model_stats))
            bars = ax_mae.bar(
                model_stats["model"],
                model_stats["median"],
                color=colors,
                edgecolor="white",
                linewidth=0.8,
            )
            ax_mae.errorbar(
                model_stats["model"],
                model_stats["median"],
                yerr=[
                    (model_stats["median"] - model_stats["q25"]).clip(lower=0),
                    (model_stats["q75"] - model_stats["median"]).clip(lower=0),
                ],
                fmt="none",
                color="#333333",
                capsize=5,
                linewidth=1.2,
            )
            for bar, val in zip(bars, model_stats["median"]):
                ax_mae.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ax_mae.get_ylim()[1] * 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                )
    ax_mae.set_ylim(bottom=0)
    ax_mae.set_title(
        "Median MAE by Model\n(IQR whiskers, all LSOAs & folds)", fontsize=11
    )
    ax_mae.set_xlabel("")
    ax_mae.set_ylabel("MAE")

    # ── Panel 3: Best-model win rate ──────────────────────────────────────────
    if not best_df.empty and "model" in best_df.columns:
        counts = best_df["model"].value_counts()
        pct = (counts / counts.sum() * 100).reset_index()
        pct.columns = ["model", "pct"]
        bars = ax_wins.bar(
            pct["model"],
            pct["pct"],
            color=sns.color_palette("Greens_d", len(pct)),
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, val in zip(bars, pct["pct"]):
            ax_wins.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax_wins.set_title(
        "Best-Model Win Rate\n(% of LSOA × crime-type pairs)", fontsize=11
    )
    ax_wins.set_xlabel("")
    ax_wins.set_ylabel("% of pairs")
    ax_wins.set_ylim(0, 105)

    # ── Panel 4: Total forecast by crime type ─────────────────────────────────
    crime_totals = (
        forecast_df.groupby("Crime type")["predicted"].sum().sort_values(ascending=True)
    )
    colors = sns.color_palette(_PALETTE, len(crime_totals))
    ax_crimes.barh(
        crime_totals.index,
        crime_totals.values,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    for i, (val, label) in enumerate(zip(crime_totals.values, crime_totals.index)):
        ax_crimes.text(
            val + crime_totals.max() * 0.01, i, f"{val:,.0f}", va="center", fontsize=7.5
        )
    ax_crimes.set_title("Total 12-Month Predicted\nCrimes by Type", fontsize=11)
    ax_crimes.set_xlabel("Predicted Crime Count")
    ax_crimes.set_ylabel("")
    ax_crimes.tick_params(axis="y", labelsize=8)

    # ── save ──────────────────────────────────────────────────────────────────
    fig.suptitle(
        "Crime Forecasting — Summary Report",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    out_path = output_dir / "report_figure.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
