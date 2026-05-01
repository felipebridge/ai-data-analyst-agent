import pandas as pd
from typing import Any

CLASSIFICATION_MAX_UNIQUE = 20
CLUSTERING_MIN_ROWS = 50


def infer_task(df: pd.DataFrame, patterns: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    target_info = patterns.get("possible_target", {})
    target_col = target_info.get("column")

    if not target_col or target_col not in df.columns:
        return _clustering_result("No suitable target column found.")

    target = df[target_col]
    n_rows = len(df)
    n_unique = target.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(target)

    if not is_numeric:
        if n_unique <= CLASSIFICATION_MAX_UNIQUE:
            return _classification_result(target_col, n_unique)
        return _eda_only_result(f"Target '{target_col}' has too many categories ({n_unique}).")

    if n_unique <= 2:
        return _classification_result(target_col, n_unique)

    if n_unique <= CLASSIFICATION_MAX_UNIQUE and (n_unique / n_rows) < 0.05:
        return _classification_result(target_col, n_unique)

    feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]
    if not feature_cols:
        if n_rows >= CLUSTERING_MIN_ROWS:
            return _clustering_result("No usable features for supervised learning.")
        return _eda_only_result("Dataset has no numeric features for modeling.")

    return _regression_result(target_col)


def _regression_result(target_col: str) -> dict[str, Any]:
    return {
        "task": "regression",
        "target_column": target_col,
        "models": ["LinearRegression", "RandomForestRegressor"],
        "metrics": ["MAE", "RMSE", "R2"],
        "reason": f"Numeric target '{target_col}' with continuous values.",
    }


def _classification_result(target_col: str, n_unique: int) -> dict[str, Any]:
    return {
        "task": "classification",
        "target_column": target_col,
        "models": ["LogisticRegression", "RandomForestClassifier"],
        "metrics": ["Accuracy", "Precision", "Recall", "F1"],
        "reason": f"Target '{target_col}' has {n_unique} discrete classes.",
    }


def _clustering_result(reason: str = "") -> dict[str, Any]:
    return {
        "task": "clustering",
        "target_column": None,
        "models": ["KMeans"],
        "metrics": ["Silhouette Score"],
        "reason": reason or "No clear supervised target; applying unsupervised clustering.",
    }


def _eda_only_result(reason: str) -> dict[str, Any]:
    return {
        "task": "eda_only",
        "target_column": None,
        "models": [],
        "metrics": [],
        "reason": reason,
    }
