import pandas as pd
from typing import Any


def run_eda(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": _missing(df),
        "duplicates": {"duplicate_rows": int(df.duplicated().sum())},
        "descriptive_stats": _descriptive_stats(df),
        "correlations": _correlations(df),
    }


def _missing(df: pd.DataFrame) -> dict[str, Any]:
    counts = df.isnull().sum()
    pcts = (counts / len(df) * 100).round(2)
    return {
        "counts": counts.to_dict(),
        "percentages": pcts.to_dict(),
        "total_missing": int(counts.sum()),
        "columns_with_missing": int((counts > 0).sum()),
    }


def _descriptive_stats(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return {}
    return numeric.describe().round(4).to_dict()


def _correlations(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return {}
    return numeric.corr().round(4).to_dict()
