import numpy as np
import pandas as pd
from typing import Any

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def train_models(df: pd.DataFrame, task_info: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    task = task_info.get("task")
    if task == "regression":
        return _train_regression(df, task_info, cfg)
    if task == "classification":
        return _train_classification(df, task_info, cfg)
    if task == "clustering":
        return _train_clustering(df, cfg)
    return {"skipped": True, "reason": task_info.get("reason", "EDA only.")}


def _prepare_features(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col]).select_dtypes(include="number")
    X = X.fillna(X.median())
    y = df[target_col]
    return X, y


def _sorted_importances(features: list, importances: list) -> list[dict]:
    return [
        {"feature": f, "importance": round(float(i), 4)}
        for f, i in sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    ]


def _train_regression(df: pd.DataFrame, task_info: dict, cfg: dict) -> dict[str, Any]:
    target_col = task_info["target_column"]
    X, y = _prepare_features(df, target_col)
    if X.empty:
        return {"error": "No numeric features available for regression."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["random_state"]
    )

    results: dict[str, Any] = {}
    feature_importance: dict[str, list] = {}

    for name, model in [
        ("LinearRegression", LinearRegression()),
        ("RandomForestRegressor", RandomForestRegressor(n_estimators=100, random_state=cfg["random_state"])),
    ]:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "MAE": round(float(mean_absolute_error(y_test, preds)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
            "R2": round(float(r2_score(y_test, preds)), 4),
        }
        if hasattr(model, "feature_importances_"):
            feature_importance[name] = _sorted_importances(X.columns, model.feature_importances_)

    return {
        "task": "regression",
        "target": target_col,
        "features_used": list(X.columns),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "results": results,
        "feature_importance": feature_importance,
    }


def _train_classification(df: pd.DataFrame, task_info: dict, cfg: dict) -> dict[str, Any]:
    target_col = task_info["target_column"]
    X, y = _prepare_features(df, target_col)
    if X.empty:
        return {"error": "No numeric features available for classification."}

    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    avg = "binary" if len(le.classes_) == 2 else "weighted"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=y_enc
    )

    results: dict[str, Any] = {}
    feature_importance: dict[str, list] = {}

    for name, model in [
        ("LogisticRegression", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=cfg["random_state"])),
        ])),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=100, random_state=cfg["random_state"])),
    ]:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "Accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "Precision": round(float(precision_score(y_test, preds, average=avg, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_test, preds, average=avg, zero_division=0)), 4),
            "F1": round(float(f1_score(y_test, preds, average=avg, zero_division=0)), 4),
        }
        clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        if hasattr(clf, "feature_importances_"):
            feature_importance[name] = _sorted_importances(X.columns, clf.feature_importances_)

    return {
        "task": "classification",
        "target": target_col,
        "classes": list(le.classes_),
        "features_used": list(X.columns),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "results": results,
        "feature_importance": feature_importance,
    }


def _train_clustering(df: pd.DataFrame, cfg: dict) -> dict[str, Any]:
    num = df.select_dtypes(include="number")
    X = num.fillna(num.median())
    if X.shape[1] < 2:
        return {"error": "Not enough numeric features for clustering."}

    X_scaled = StandardScaler().fit_transform(X)
    k_range = cfg.get("n_clusters_range", [2, 7])
    best_k, best_score = 2, -1.0

    for k in range(k_range[0], k_range[1] + 1):
        if k >= len(X_scaled):
            break
        labels = KMeans(n_clusters=k, random_state=cfg["random_state"], n_init=10).fit_predict(X_scaled)
        try:
            score = float(silhouette_score(X_scaled, labels))
        except Exception:
            score = -1.0
        if score > best_score:
            best_k, best_score = k, score

    return {
        "task": "clustering",
        "features_used": list(X.columns),
        "best_k": best_k,
        "results": {"KMeans": {"n_clusters": best_k, "Silhouette Score": round(best_score, 4)}},
        "feature_importance": {},
    }
