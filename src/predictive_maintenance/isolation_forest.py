"""Isolation forest helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass(frozen=True)
class DetectionResult:
    """Predictions and anomaly scores for a dataset."""

    predictions: pd.Series
    scores: pd.Series


def split_features_and_labels(
    frame: pd.DataFrame, label_column: str = "Label"
) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric features and binary labels."""
    if label_column not in frame:
        raise ValueError(f"missing label column: {label_column}")
    labels = frame[label_column].astype("int64")
    features = frame.drop(columns=[label_column])
    features = features.select_dtypes(include="number")
    if features.empty:
        raise ValueError("the input has no numeric feature columns")
    return features, labels


def detect_anomalies(
    features: pd.DataFrame,
    contamination: float = "auto",
    random_state: int = 42,
    n_estimators: int = 100,
) -> DetectionResult:
    """Fit isolation forest and return predictions encoded as zero or one."""
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    raw_predictions = model.fit_predict(features)
    predictions = pd.Series((raw_predictions == -1).astype("int64"), index=features.index)
    scores = pd.Series(-model.score_samples(features), index=features.index)
    return DetectionResult(predictions=predictions, scores=scores)
