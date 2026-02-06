"""Package CLI entrypoint for anomaly detection pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from credit_anomaly.pipeline import run_pipeline


def main() -> None:
    default_data_path = Path("data") / "application_train.csv"

    parser = argparse.ArgumentParser(description="Run anomaly detection pipeline")
    parser.add_argument(
        "--data",
        default=str(default_data_path),
        help=f"Path to application_train.csv (default: {default_data_path})",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Directory to store plots and artifacts",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile for anomaly threshold",
    )
    args = parser.parse_args()

    artifacts = run_pipeline(
        data_path=Path(args.data),
        output_dir=Path(args.output),
        percentile_threshold=args.percentile,
    )

    print("Anomaly detection completed.")
    print("Metrics:")
    for key, value in artifacts.metrics.items():
        print(f"- {key}: {value:.4f}")
    print("Plots:")
    for key, path in artifacts.plots.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
