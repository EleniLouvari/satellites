"""Shared model implementations used by training and final refitting."""

# Preserve serialized class and transformer paths from the previous flat module.
from satellites.ml_classification.shared.models.models import (
    ContiguousLabelClassifier as ContiguousLabelClassifier,
    ModelCandidate as ModelCandidate,
    _to_dense_matrix as _to_dense_matrix,
)
