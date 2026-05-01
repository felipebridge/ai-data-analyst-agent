import pytest
import pandas as pd
import numpy as np
from src.analysis.eda import run_eda


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, np.nan, 45, 22, 30],
        "salary": [50000, 60000, 55000, 80000, 45000, 60000],
        "city": ["NYC", "LA", "NYC", "LA", "NYC", "LA"],
    })


def test_eda_returns_all_keys(sample_df):
    result = run_eda(sample_df)
    assert set(result.keys()) == {"shape", "column_types", "missing", "duplicates", "descriptive_stats", "correlations"}


def test_shape(sample_df):
    result = run_eda(sample_df)
    assert result["shape"]["rows"] == 6
    assert result["shape"]["columns"] == 3


def test_missing_detection(sample_df):
    result = run_eda(sample_df)
    assert result["missing"]["counts"]["age"] == 1
    assert result["missing"]["counts"]["salary"] == 0
    assert result["missing"]["columns_with_missing"] == 1
    assert result["missing"]["total_missing"] == 1


def test_duplicates(sample_df):
    result = run_eda(sample_df)
    assert result["duplicates"]["duplicate_rows"] == 1


def test_column_types(sample_df):
    result = run_eda(sample_df)
    assert "float" in result["column_types"]["age"]
    assert any(t in result["column_types"]["city"] for t in ("object", "str"))


def test_descriptive_stats_numeric_only(sample_df):
    result = run_eda(sample_df)
    assert "salary" in result["descriptive_stats"]
    assert "city" not in result["descriptive_stats"]


def test_correlations_requires_two_numeric_columns():
    df_one_col = pd.DataFrame({"x": [1, 2, 3]})
    result = run_eda(df_one_col)
    assert result["correlations"] == {}


def test_correlations_present():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    result = run_eda(df)
    assert "x" in result["correlations"]
    assert abs(result["correlations"]["x"]["y"] - 1.0) < 1e-4
