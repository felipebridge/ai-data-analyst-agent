import os
from datetime import datetime
from typing import Any


def generate_report(
    dataset_path: str,
    eda: dict[str, Any],
    patterns: dict[str, Any],
    task_info: dict[str, Any],
    model_results: dict[str, Any],
    plots: list[str],
    reports_dir: str,
) -> str:
    os.makedirs(reports_dir, exist_ok=True)
    now = datetime.now()
    lines = _build_report(dataset_path, eda, patterns, task_info, model_results, plots, now)
    path = os.path.join(reports_dir, f"report_{now.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def _build_report(
    dataset_path: str,
    eda: dict,
    patterns: dict,
    task_info: dict,
    model_results: dict,
    plots: list[str],
    now: datetime,
) -> list[str]:
    lines: list[str] = []
    sep = "=" * 70

    def section(title: str) -> None:
        lines.extend(["", sep, f"  {title.upper()}", sep])

    lines.append("AUTONOMOUS DATA ANALYST — ANALYSIS REPORT")
    lines.append(f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Dataset: {dataset_path}")

    section("1. Dataset Summary")
    shape = eda.get("shape", {})
    col_types = eda.get("column_types", {})
    num_cols = sum(1 for t in col_types.values() if "float" in t or "int" in t)
    lines.append(f"  Rows        : {shape.get('rows', 'N/A')}")
    lines.append(f"  Columns     : {shape.get('columns', 'N/A')}")
    lines.append(f"  Numeric cols: {num_cols}")
    lines.append(f"  Other cols  : {len(col_types) - num_cols}")
    lines.append(f"  Duplicates  : {eda.get('duplicates', {}).get('duplicate_rows', 0)}")

    section("2. Data Quality")
    missing = eda.get("missing", {})
    lines.append(f"  Total missing values : {missing.get('total_missing', 0)}")
    lines.append(f"  Columns with missing : {missing.get('columns_with_missing', 0)}")
    high_miss = patterns.get("highly_missing_columns", [])
    if high_miss:
        lines.append(f"  High-missing columns (>40%): {', '.join(high_miss)}")
    else:
        lines.append("  No columns exceed the 40% missing threshold.")
    outliers = patterns.get("outliers", {})
    if outliers:
        lines.append("  Columns with outliers (IQR method):")
        for col, cnt in outliers.items():
            lines.append(f"    - {col}: {cnt} outlier(s)")
    else:
        lines.append("  No significant outliers detected.")

    section("3. Key Patterns Detected")
    strong_corr = patterns.get("strong_correlations", [])
    if strong_corr:
        lines.append("  Strong correlations (|r| >= threshold):")
        for pair in strong_corr[:5]:
            lines.append(f"    - {pair['col1']} <-> {pair['col2']} : {pair['correlation']}")
    else:
        lines.append("  No strong correlations found.")
    for item in patterns.get("categorical_imbalance", []):
        lines.append(f"  Class imbalance — {item['column']}: minority '{item['minority_class']}' at {item['minority_ratio']*100:.1f}%")
    target_info = patterns.get("possible_target", {})
    if target_info:
        lines.append(f"  Suggested target: '{target_info.get('column')}' ({target_info.get('reason')})")

    section("4. Inferred ML Task")
    lines.append(f"  Task   : {task_info.get('task', 'N/A').upper()}")
    lines.append(f"  Reason : {task_info.get('reason', '')}")
    if task_info.get("target_column"):
        lines.append(f"  Target : {task_info['target_column']}")
    if task_info.get("models"):
        lines.append(f"  Models : {', '.join(task_info['models'])}")

    section("5. Model Results")
    if not model_results or model_results.get("skipped"):
        lines.append(f"  Modeling skipped. {(model_results or {}).get('reason', '')}")
    elif model_results.get("error"):
        lines.append(f"  Error: {model_results['error']}")
    else:
        for model_name, metrics in model_results.get("results", {}).items():
            lines.append(f"\n  [{model_name}]")
            for metric, value in metrics.items():
                lines.append(f"    {metric}: {value}")
        fi = model_results.get("feature_importance", {})
        if fi:
            best = next(iter(fi))
            lines.append(f"\n  Top features ({best}):")
            for feat in fi[best][:5]:
                lines.append(f"    - {feat['feature']}: {feat['importance']}")

    section("6. Business Insights")
    for i, insight in enumerate(_generate_insights(eda, patterns, task_info, model_results), 1):
        lines.append(f"  {i}. {insight}")

    section("7. Recommended Next Steps")
    for i, step in enumerate(_generate_next_steps(eda, patterns, task_info), 1):
        lines.append(f"  {i}. {step}")

    section("8. Generated Visualizations")
    if plots:
        for p in plots:
            lines.append(f"  - {p}")
    else:
        lines.append("  No plots generated.")

    lines.extend(["", sep, "  END OF REPORT", sep])
    return lines


def _generate_insights(eda: dict, patterns: dict, task_info: dict, model_results: dict) -> list[str]:
    insights = []
    rows = eda.get("shape", {}).get("rows", 0)

    if rows < 500:
        insights.append(f"Small dataset ({rows} rows). Results may not generalize — consider collecting more data.")
    elif rows > 100_000:
        insights.append(f"Large dataset ({rows} rows). Prioritize scalable, memory-efficient models.")

    strong_corr = patterns.get("strong_correlations", [])
    if strong_corr:
        top = strong_corr[0]
        insights.append(
            f"Strong correlation ({top['correlation']}) between '{top['col1']}' and '{top['col2']}'. "
            "Evaluate whether both features add independent predictive value."
        )

    if patterns.get("highly_missing_columns"):
        cols = patterns["highly_missing_columns"]
        insights.append(f"Columns {cols} exceed 40% missing. Impute or drop before production use.")

    task = task_info.get("task")
    if task in ("regression", "classification") and model_results and not model_results.get("skipped"):
        results = model_results.get("results", {})
        if task == "classification":
            best_f1 = max((v.get("F1", 0) for v in results.values()), default=0)
            if best_f1 >= 0.85:
                insights.append(f"Best model F1={best_f1:.2f} — strong baseline, ready for validation on held-out data.")
            elif best_f1 >= 0.65:
                insights.append(f"Best model F1={best_f1:.2f} — moderate; feature engineering or tuning may help.")
            else:
                insights.append(f"Best model F1={best_f1:.2f} — weak; review features, imbalance, or problem framing.")
        else:
            best_r2 = max((v.get("R2", -999) for v in results.values()), default=-999)
            if best_r2 >= 0.8:
                insights.append(f"Best model R²={best_r2:.2f} — strong predictive power for this target.")
            elif best_r2 >= 0.5:
                insights.append(f"Best model R²={best_r2:.2f} — moderate fit; more features or non-linear models may help.")
            else:
                insights.append(f"Best model R²={best_r2:.2f} — weak fit; revisit target definition or acquire more data.")

    if task == "clustering":
        sil = (model_results or {}).get("results", {}).get("KMeans", {}).get("Silhouette Score", 0)
        k = (model_results or {}).get("best_k", "?")
        if sil > 0.5:
            insights.append(f"Found {k} well-separated clusters (Silhouette={sil:.2f}) — viable for business segmentation.")
        else:
            insights.append(f"Found {k} clusters but low cohesion (Silhouette={sil:.2f}) — segments need further refinement.")

    if not insights:
        insights.append("No major data quality issues found. Dataset appears ready for deeper analysis.")

    return insights


def _generate_next_steps(eda: dict, patterns: dict, task_info: dict) -> list[str]:
    steps = []
    high_miss = patterns.get("highly_missing_columns", [])
    if high_miss:
        steps.append(f"Address missing data in: {', '.join(high_miss)} (impute or drop per domain knowledge).")

    dups = eda.get("duplicates", {}).get("duplicate_rows", 0)
    if dups > 0:
        steps.append(f"Remove or investigate {dups} duplicate row(s).")

    if patterns.get("outliers"):
        steps.append("Review outliers — cap, transform, or remove based on domain context.")

    if patterns.get("categorical_imbalance"):
        steps.append("Handle class imbalance with SMOTE or class weights before training a classifier.")

    task = task_info.get("task")
    if task in ("regression", "classification"):
        steps.append("Tune the best baseline model (GridSearchCV or Optuna).")
        steps.append("Engineer domain-specific features and apply feature selection.")
        steps.append("Evaluate the final model on a fully held-out test set before any deployment.")
    elif task == "clustering":
        steps.append("Profile each cluster to assign business meaning to the segments.")
        steps.append("Compare with DBSCAN or Agglomerative clustering for robustness.")

    rows = eda.get("shape", {}).get("rows", 0)
    if 0 < rows < 1000:
        steps.append(f"Dataset has only {rows} rows — collect more data to improve ML reliability.")

    return steps
