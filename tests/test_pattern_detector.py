import pytest
import pandas as pd
import numpy as np
from src.analysis.eda import run_eda
from src.analysis.pattern_detector import detect_patterns
from src.utils.config import load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "y": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "z": [5, 3, 8, 1, 9, 2, 7, 4, 6, 10],
        "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


def test_detect_strong_correlations(sample_df, cfg):
    eda = run_eda(sample_df)
    patterns = detect_patterns(sample_df, eda, cfg)
    strong = patterns["strong_correlations"]
    cols = {(p["col1"], p["col2"]) for p in strong}
    found = ("x", "y") in cols or ("y", "x") in cols
    assert found


def test_no_highly_missing_columns(sample_df, cfg):
    eda = run_eda(sample_df)
    patterns = detect_patterns(sample_df, eda, cfg)
    assert patterns["highly_missing_columns"] == []


def test_highly_missing_detection(cfg):
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [np.nan, np.nan, np.nan, np.nan, 1],
    })
    eda = run_eda(df)
    patterns = detect_patterns(df, eda, cfg)
    assert "b" in patterns["highly_missing_columns"]


def test_outlier_detection(cfg):
    df = pd.DataFrame({"val": [1, 2, 2, 3, 3, 3, 2, 1, 1000]})
    eda = run_eda(df)
    patterns = detect_patterns(df, eda, cfg)
    assert "val" in patterns["outliers"]
    assert patterns["outliers"]["val"] >= 1


def test_possible_target_uses_named_column(cfg):
    df = pd.DataFrame({"feature": [1, 2, 3], "price": [10, 20, 30]})
    eda = run_eda(df)
    patterns = detect_patterns(df, eda, cfg)
    assert patterns["possible_target"]["column"] == "price"


def test_possible_target_fallback_to_last(cfg):
    df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})
    eda = run_eda(df)
    patterns = detect_patterns(df, eda, cfg)
    assert patterns["possible_target"]["column"] == "col_b"
