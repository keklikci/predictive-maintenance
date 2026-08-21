import pandas as pd
import pytest

from predictive_maintenance import detect_anomalies, split_features_and_labels


def test_split_features_and_labels_drops_non_numeric_columns() -> None:
    frame = pd.DataFrame(
        {
            "TimeStamp": ["2024-01-01", "2024-01-02"],
            "sensor": [1.0, 2.0],
            "Label": [0, 1],
        }
    )

    features, labels = split_features_and_labels(frame)

    assert list(features.columns) == ["sensor"]
    assert labels.tolist() == [0, 1]


def test_split_features_and_labels_requires_label() -> None:
    with pytest.raises(ValueError, match="missing label column"):
        split_features_and_labels(pd.DataFrame({"sensor": [1.0]}))


def test_detect_anomalies_returns_binary_predictions() -> None:
    features = pd.DataFrame({"sensor": [0.0, 0.1, 0.2, 10.0]})

    result = detect_anomalies(features, contamination=0.25, n_estimators=32)

    assert set(result.predictions.unique()) <= {0, 1}
    assert len(result.predictions) == len(features)
    assert result.scores.index.equals(features.index)
