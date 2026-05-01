import pandas as pd
from typing import Any


def detect_patterns(df: pd.DataFrame, eda: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "strong_correlations": _strong_correlations(eda, cfg["correlation_threshold"]),
        "highly_missing_columns": _highly_missing(eda, cfg["missing_threshold"]),
        "possible_target": _infer_target(df),
        "categorical_imbalance": _categorical_imbalance(df, cfg["imbalance_ratio"]),
        "outliers": _detect_outliers(df, cfg["outlier_iqr_factor"]),
    }


def _strong_correlations(eda: dict, threshold: float) -> list[dict]:
    corr = eda.get("correlations", {})
    if not corr:
        return []
    pairs = []
    cols = list(corr.keys())
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            val = corr[c1].get(c2, 0)
            if abs(val) >= threshold:
                pairs.append({"col1": c1, "col2": c2, "correlation": round(val, 4)})
    return sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)


def _highly_missing(eda: dict, threshold: float) -> list[str]:
    pcts = eda.get("missing", {}).get("percentages", {})
    return [col for col, pct in pcts.items() if pct / 100 >= threshold]


def _infer_target(df: pd.DataFrame) -> dict[str, Any]:
    hints = {
        "target", "label", "price", "sales", "churn", "survived", "outcome",
        "result", "class", "y", "output", "revenue", "score", "default",
    }
    for col in df.columns:
        if col.lower() in hints:
            return {"column": col, "reason": "name matches common target keywords"}

    last = df.columns[-1]
    if pd.api.types.is_numeric_dtype(df[last]):
        nunique = df[last].nunique()
        task_hint = "classification" if nunique <= 10 else "regression"
        return {"column": last, "reason": f"last column ({task_hint} likely, {nunique} unique values)"}
    return {"column": last, "reason": "last column (default heuristic)"}


def _categorical_imbalance(df: pd.DataFrame, ratio: float) -> list[dict]:
    imbalanced = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        vc = df[col].value_counts(normalize=True)
        if len(vc) >= 2 and vc.iloc[-1] < ratio:
            imbalanced.append({
                "column": col,
                "minority_class": str(vc.index[-1]),
                "minority_ratio": round(float(vc.iloc[-1]), 4),
            })
    return imbalanced


def _detect_outliers(df: pd.DataFrame, factor: float) -> dict[str, int]:
    result = {}
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        n_out = int(((series < q1 - factor * iqr) | (series > q3 + factor * iqr)).sum())
        if n_out > 0:
            result[col] = n_out
    return result
