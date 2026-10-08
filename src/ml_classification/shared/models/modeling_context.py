"""Load the feature schema and label encoder from check/prepare artifacts."""

from __future__ import annotations

from typing import Any

from sklearn.preprocessing import LabelEncoder

from ml_classification.shared.config.config import ClassificationPipelineConfig
from ml_classification.shared.persistence import load_json
from shared.io import read_data


def load_modeling_context(config: ClassificationPipelineConfig) -> dict[str, Any]:
    """Load persisted check/prepare artifacts and rebuild model context objects."""
    # Recreate feature groups and label encoder from saved preparation artifacts.
    check_summary = load_json(config.check_dir / "check_summary.json")
    prepare_summary = load_json(config.prepare_dir / "prepare_summary.json")
    label_mapping = read_data(str(config.prepare_dir / "label_mapping.csv"), watch_curly_brackets=False)
    active_features = prepare_summary["active_features"]
    numeric_features = [column for column in check_summary["numeric_features"] if column in active_features]
    categorical_features = [column for column in check_summary["categorical_features"] if column in active_features]
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_mapping["target_label"].astype(str).to_numpy()
    return {
        "check_summary": check_summary,
        "prepare_summary": prepare_summary,
        "label_mapping": label_mapping,
        "active_features": active_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "label_encoder": label_encoder,
        "labels": prepare_summary["target_labels"],
    }
