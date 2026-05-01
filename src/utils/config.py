import yaml
from pathlib import Path
from typing import Any


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return _defaults()
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return {**_defaults(), **(data or {})}


def _defaults() -> dict[str, Any]:
    return {
        "missing_threshold": 0.4,
        "correlation_threshold": 0.8,
        "outlier_iqr_factor": 1.5,
        "imbalance_ratio": 0.1,
        "test_size": 0.2,
        "random_state": 42,
        "plots_dir": "plots",
        "reports_dir": "reports",
        "n_clusters_range": [2, 7],
    }
