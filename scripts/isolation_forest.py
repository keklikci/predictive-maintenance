"""Run the isolation forest experiment on a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from predictive_maintenance import detect_anomalies, split_features_and_labels


def main() -> None:
    """Run anomaly detection and write predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/run"))
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    features, labels = split_features_and_labels(frame)
    result = detect_anomalies(features)
    output = pd.DataFrame(
        {"y_true": labels, "y_pred": result.predictions, "anomaly_score": result.scores}
    )
    args.output.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output / "predictions.csv", index=False)
    print(output["y_pred"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
