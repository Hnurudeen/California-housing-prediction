"""
Model evaluation: metrics, residual analysis, and visualisation.

All plots are saved to disk (reports/figures/) rather than displayed
inline, so this module works in non-interactive / script contexts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for script execution
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from california_housing.config import FIGURES_DIR


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred, model_name: str = "Model") -> dict:
    """
    Compute the full regression metric suite for one model.

    Returns a dict with: model, R2, MAE ($), RMSE ($), MSE, MAPE (%)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    # Guard against zero-valued actuals
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

    return {
        "model"   : model_name,
        "R2"      : round(r2, 4),
        "MAE ($)" : round(mae, 0),
        "RMSE ($)": round(rmse, 0),
        "MSE"     : round(mse, 0),
        "MAPE (%)": round(mape, 2),
    }


def print_metrics(metrics: dict) -> None:
    print(
        f"\n  R²   : {metrics['R2']:.4f}  ({metrics['R2']*100:.1f}% variance explained)"
        f"\n  MAE  : ${metrics['MAE ($)']:>10,.0f}"
        f"\n  RMSE : ${metrics['RMSE ($)']:>10,.0f}"
        f"\n  MAPE : {metrics['MAPE (%)']:.2f}%"
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _ensure_dir(output_dir) -> None:
    os.makedirs(output_dir, exist_ok=True)


def plot_actual_vs_predicted(
    y_test, y_pred, model_name: str, output_dir=None
) -> None:
    """Scatter of actual vs predicted values with a perfect-prediction line."""
    output_dir = output_dir or str(FIGURES_DIR)
    _ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.25, s=8, color="steelblue", label="Predictions")
    lims = [
        min(float(np.min(y_test)), float(np.min(y_pred))),
        max(float(np.max(y_test)), float(np.max(y_pred))),
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual House Value ($)", fontsize=11)
    ax.set_ylabel("Predicted House Value ($)", fontsize=11)
    ax.set_title(f"{model_name}\nActual vs Predicted", fontsize=13, fontweight="bold")
    ax.legend()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()

    slug = model_name.lower().replace(" ", "_")
    path = os.path.join(output_dir, f"{slug}_actual_vs_pred.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_residuals(y_test, y_pred, model_name: str, output_dir=None) -> None:
    """
    Two-panel residual analysis:
    - Left : residuals vs predicted (checks heteroscedasticity)
    - Right: residual distribution (checks normality assumption)
    """
    output_dir = output_dir or str(FIGURES_DIR)
    _ensure_dir(output_dir)
    residuals = np.asarray(y_test) - np.asarray(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.2, s=8, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted Value ($)", fontsize=10)
    axes[0].set_ylabel("Residual ($)", fontsize=10)
    axes[0].set_title("Residuals vs Predicted", fontsize=11)
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    axes[1].hist(residuals, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].axvline(np.mean(residuals), color="orange", linestyle="-", linewidth=1.5,
                    label=f"Mean: ${np.mean(residuals):,.0f}")
    axes[1].set_xlabel("Residual ($)", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].set_title("Residual Distribution", fontsize=11)
    axes[1].legend()
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.suptitle(f"{model_name} — Residual Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()

    slug = model_name.lower().replace(" ", "_")
    path = os.path.join(output_dir, f"{slug}_residuals.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(
    model, feature_names: list, model_name: str,
    output_dir=None, top_n: int = 12
) -> pd.Series:
    """
    Horizontal bar chart of top N feature importances (Mean Decrease in Impurity).
    Returns a Series ranked from most to least important.
    """
    output_dir = output_dir or str(FIGURES_DIR)
    _ensure_dir(output_dir)
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top.index, top.values, color="steelblue", alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, top.values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=10)
    ax.set_title(f"{model_name}\nTop {top_n} Feature Importances",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, top.max() * 1.18)
    plt.tight_layout()

    slug = model_name.lower().replace(" ", "_")
    path = os.path.join(output_dir, f"{slug}_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    return importances.sort_values(ascending=False)


def plot_model_comparison(results_list: list, output_dir=None) -> None:
    """
    Side-by-side bar charts comparing all models across R², MAE, and RMSE.
    Each model uses its own predictions — not a shared y_pred variable.
    """
    output_dir = output_dir or str(FIGURES_DIR)
    _ensure_dir(output_dir)
    df      = pd.DataFrame(results_list)
    metrics = ["R2", "MAE ($)", "RMSE ($)"]
    titles  = ["R² (higher is better)", "MAE in $ (lower is better)", "RMSE in $ (lower is better)"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        bars = axes[i].bar(df["model"], df[metric], color=colors[: len(df)],
                           alpha=0.88, edgecolor="white", width=0.55)
        axes[i].set_title(title, fontsize=11, fontweight="bold")
        axes[i].tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, df[metric]):
            label = f"${val:,.0f}" if metric != "R2" else f"{val:.4f}"
            axes[i].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() * 1.02, label,
                         ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_error_by_price_range(
    y_test, y_pred, model_name: str, output_dir=None
) -> None:
    """
    Absolute error broken down by actual price range.
    Shows whether the model performs consistently across the price spectrum.
    """
    output_dir = output_dir or str(FIGURES_DIR)
    _ensure_dir(output_dir)
    y_test     = np.asarray(y_test)
    y_pred     = np.asarray(y_pred)
    abs_errors = np.abs(y_test - y_pred)

    bins   = [0, 100_000, 200_000, 300_000, 400_000, 500_001]
    labels = ["<$100k", "$100k-200k", "$200k-300k", "$300k-400k", "$400k+"]
    bucket = pd.cut(y_test, bins=bins, labels=labels)
    df     = pd.DataFrame({"bucket": bucket, "abs_error": abs_errors})
    summary = df.groupby("bucket", observed=True)["abs_error"].agg(["mean", "median", "count"])

    fig, ax = plt.subplots(figsize=(9, 5))
    x    = np.arange(len(summary))
    bars = ax.bar(x, summary["mean"], color="steelblue", alpha=0.8,
                  label="Mean Abs Error", edgecolor="white")
    ax.plot(x, summary["median"], "o--", color="darkorange", linewidth=1.5,
            markersize=7, label="Median Abs Error")
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f"n={int(row['count'])}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=15)
    ax.set_ylabel("Absolute Error ($)", fontsize=10)
    ax.set_title(f"{model_name}\nPrediction Error by Price Range",
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    plt.tight_layout()

    slug = model_name.lower().replace(" ", "_")
    path = os.path.join(output_dir, f"{slug}_error_by_price_range.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
