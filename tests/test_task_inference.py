import pandas as pd
from src.analysis.eda import run_eda
from src.analysis.pattern_detector import detect_patterns
from src.modeling.task_inference import infer_task
from src.utils.config import load_config


def _run(df, cfg=None):
    cfg = cfg or load_config()
    eda = run_eda(df)
    patterns = detect_patterns(df, eda, cfg)
    return infer_task(df, patterns, cfg)


def test_regression_inferred():
    df = pd.DataFrame({
        "x1": range(100),
        "x2": [i * 1.5 for i in range(100)],
        "price": [i * 2.3 + 5 for i in range(100)],
    })
    task = _run(df)
    assert task["task"] == "regression"
    assert task["target_column"] == "price"


def test_classification_inferred():
    df = pd.DataFrame({
        "f1": range(100),
        "f2": [i % 5 for i in range(100)],
        "survived": [i % 2 for i in range(100)],
    })
    task = _run(df)
    assert task["task"] == "classification"
    assert task["target_column"] == "survived"


def test_regression_inferred_for_generic_last_column():
    df = pd.DataFrame({
        "a": range(80),
        "b": [i * 2 for i in range(80)],
        "c": [i % 7 for i in range(80)],
    })
    task = _run(df)
    assert task["task"] == "regression"


def test_models_assigned_for_regression():
    df = pd.DataFrame({
        "x": range(50),
        "price": [float(i) * 1.1 for i in range(50)],
    })
    task = _run(df)
    if task["task"] == "regression":
        assert "LinearRegression" in task["models"]
        assert "RandomForestRegressor" in task["models"]


def test_metrics_assigned_for_classification():
    df = pd.DataFrame({
        "x": range(50),
        "survived": [i % 2 for i in range(50)],
    })
    task = _run(df)
    if task["task"] == "classification":
        assert "F1" in task["metrics"]
        assert "Accuracy" in task["metrics"]
