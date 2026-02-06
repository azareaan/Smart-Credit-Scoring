"""CLI entrypoint for the anomaly detection pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_import_path() -> None:
    """Ensure local `src` package directory is importable in script mode."""
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"Expected source directory was not found: {src_dir}")

    src_path = str(src_dir)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run anomaly detection pipeline")
    parser.add_argument("--data", required=True, help="Path to application_train.csv")
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

    _bootstrap_import_path()
    from credit_anomaly.pipeline import run_pipeline

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
