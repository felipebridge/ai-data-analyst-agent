
"""CLI entry point for the Autonomous Data Analyst Agent."""
import argparse
import sys

from src.agent.analyst_agent import AnalystAgent
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("main")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Data Analyst Agent — AI-powered EDA, modeling, and reporting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --dataset data/sample/sales.csv
  python -m src.main --dataset data/sample/titanic.csv --config config.yaml
  python -m src.main --dataset data/sample/iris.csv --plots-dir my_plots --reports-dir my_reports
""",
    )
    parser.add_argument(
        "--dataset", "-d", required=True, help="Path to the CSV dataset."
    )
    parser.add_argument(
        "--config", "-c", default="config.yaml", help="Path to config.yaml (optional)."
    )
    parser.add_argument(
        "--plots-dir", default=None, help="Directory to save plots (overrides config)."
    )
    parser.add_argument(
        "--reports-dir", default=None, help="Directory to save reports (overrides config)."
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.plots_dir:
        cfg["plots_dir"] = args.plots_dir
    if args.reports_dir:
        cfg["reports_dir"] = args.reports_dir

    try:
        agent = AnalystAgent(cfg)
        report_path = agent.run(args.dataset)
        print(f"\nAnalysis complete. Report saved to: {report_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
