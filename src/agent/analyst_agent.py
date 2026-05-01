import pandas as pd
from pathlib import Path
from typing import Any

from src.analysis.eda import run_eda
from src.analysis.pattern_detector import detect_patterns
from src.modeling.task_inference import infer_task
from src.modeling.model_trainer import train_models
from src.visualization.plot_generator import generate_plots
from src.reporting.report_generator import generate_report
from src.utils.logger import get_logger

logger = get_logger("analyst_agent")


class AnalystAgent:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def run(self, dataset_path: str) -> str:
        logger.info(f"Loading dataset: {dataset_path}")
        df = self._load_dataset(dataset_path)
        logger.info(f"Dataset loaded — {df.shape[0]} rows x {df.shape[1]} columns")

        logger.info("Running exploratory data analysis...")
        eda = run_eda(df)

        logger.info("Detecting patterns...")
        patterns = detect_patterns(df, eda, self.cfg)

        logger.info("Inferring ML task...")
        task_info = infer_task(df, patterns, self.cfg)
        logger.info(f"Inferred task: {task_info['task'].upper()} — {task_info['reason']}")

        model_results: dict[str, Any] = {}
        if task_info["task"] != "eda_only":
            logger.info(f"Training baseline models for {task_info['task']}...")
            model_results = train_models(df, task_info, self.cfg)
            if model_results.get("results"):
                for name, metrics in model_results["results"].items():
                    logger.info(f"  {name}: {metrics}")

        logger.info("Generating visualizations...")
        plots = generate_plots(df, eda, task_info, model_results, self.cfg["plots_dir"])
        logger.info(f"  {len(plots)} plot(s) saved.")

        logger.info("Generating report...")
        report_path = generate_report(
            dataset_path, eda, patterns, task_info, model_results, plots, self.cfg["reports_dir"]
        )
        logger.info(f"Report saved: {report_path}")
        return report_path

    def _load_dataset(self, path: str) -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        if p.suffix.lower() != ".csv":
            raise ValueError(f"Only CSV files are supported. Got: {p.suffix}")
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("The provided CSV file is empty.")
        return df
