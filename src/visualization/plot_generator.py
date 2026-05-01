import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any


def generate_plots(
    df: pd.DataFrame,
    eda: dict[str, Any],
    task_info: dict[str, Any],
    model_results: dict[str, Any],
    plots_dir: str,
) -> list[str]:
    os.makedirs(plots_dir, exist_ok=True)

    saved = [p for p in (
        _plot_missing(eda, plots_dir),
        _plot_correlation(eda, plots_dir),
    ) if p]

    target_col = task_info.get("target_column")
    if target_col and target_col in df.columns:
        if p := _plot_target_distribution(df, target_col, plots_dir):
            saved.append(p)

    importance = model_results.get("feature_importance", {})
    if importance:
        best_model = next(iter(importance))
        if p := _plot_feature_importance(importance[best_model], best_model, plots_dir):
            saved.append(p)

    return saved


def _plot_missing(eda: dict, plots_dir: str) -> str | None:
    counts = {k: v for k, v in eda.get("missing", {}).get("counts", {}).items() if v > 0}
    if not counts:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(counts) * 0.6), 4))
    ax.barh(list(counts.keys()), list(counts.values()), color="#e07b54")
    ax.set_xlabel("Missing Count")
    ax.set_title("Missing Values per Column")
    plt.tight_layout()
    path = os.path.join(plots_dir, "missing_values.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _plot_correlation(eda: dict, plots_dir: str) -> str | None:
    corr = eda.get("correlations", {})
    if not corr:
        return None
    corr_df = pd.DataFrame(corr)
    if corr_df.shape[0] < 2:
        return None

    size = max(6, corr_df.shape[0] * 0.7)
    fig, ax = plt.subplots(figsize=(size, size * 0.8))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    path = os.path.join(plots_dir, "correlation_heatmap.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _plot_target_distribution(df: pd.DataFrame, target_col: str, plots_dir: str) -> str | None:
    series = df[target_col].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))

    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
        ax.hist(series, bins=30, color="#5b8dd9", edgecolor="white")
        ax.set_ylabel("Frequency")
    else:
        vc = series.value_counts()
        ax.bar(vc.index.astype(str), vc.values, color="#5b8dd9")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")

    ax.set_xlabel(target_col)
    ax.set_title(f"Target Distribution: {target_col}")
    plt.tight_layout()
    path = os.path.join(plots_dir, "target_distribution.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _plot_feature_importance(importance: list[dict], model_name: str, plots_dir: str) -> str | None:
    if not importance:
        return None
    top = importance[:15]
    features = [d["feature"] for d in top]
    values = [d["importance"] for d in top]

    fig, ax = plt.subplots(figsize=(7, max(4, len(features) * 0.4)))
    ax.barh(features[::-1], values[::-1], color="#6dbf8b")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name}")
    plt.tight_layout()
    path = os.path.join(plots_dir, "feature_importance.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path
