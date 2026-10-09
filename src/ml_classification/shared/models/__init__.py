"""Shared model implementations used by training and final refitting."""

# Preserve serialized class and transformer paths from the previous flat module.
from ml_classification.shared.models.models import (
    ContiguousLabelClassifier as ContiguousLabelClassifier,
)
from ml_classification.shared.models.models import (
    ModelCandidate as ModelCandidate,
)
from ml_classification.shared.models.models import (
    _to_dense_matrix as _to_dense_matrix,
)
