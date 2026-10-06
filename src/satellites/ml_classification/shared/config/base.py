"""Shared base class helpers for pipeline step implementations.

This module centralizes common utilities used by concrete pipeline step
classes such as standardized schema attachments and manifest persistence.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from satellites.ml_classification.shared.config.config import ClassificationPipelineConfig
from satellites.ml_classification.shared.persistence import save_json
from satellites.shared.io import read_data


# Centralize behavior shared by every pipeline stage to keep step implementations focused.
class PipelineStepBase:
    """Provide shared schema and artifact helpers for step classes.

    Concrete pipeline steps inherit from this class to gain access to the
    validated `ClassificationPipelineConfig` and small helpers that attach
    stable schema metadata to produced artifacts.
    """

    def __init__(self, config: ClassificationPipelineConfig):
        """Initialize the step base with validated pipeline configuration.

        The constructor ensures required output directories exist so child
        steps can write artifacts without additional filesystem checks.
        """
        # Keep a shared config reference for all inherited step helpers.
        self.config = config
        # Ensure pipeline output layout is present before any write operations.
        self.config.ensure_directories()

    def _with_schema(self, payload: dict[str, Any], artifact_name: str) -> dict[str, Any]:
        """Attach schema metadata to an artifact payload dictionary.

        Returns a shallow copy of `payload` augmented with a `_schema`
        entry describing the artifact name and declared schema version so
        downstream consumers can validate outputs.
        """
        # Copy the payload to avoid mutating caller-owned objects.
        enriched = dict(payload)
        enriched["_schema"] = {"artifact": artifact_name, "schema_version": self.config.output_schema_version}
        return enriched

    def _save_schema_manifest(
        self,
        step_dir,
        artifact_name: str,
        json_contracts: dict[str, Any] | None = None,
        csv_contracts: dict[str, list[str]] | None = None,
    ) -> None:
        """Persist a schema manifest describing JSON and CSV artifact contracts.

        The manifest is saved as `schema_manifest.json` inside the given
        `step_dir` and contains both the declared JSON and CSV contracts so
        automated validators or report generators can assert artifact
        conformance.
        """
        # Save manifest metadata so downstream tools can validate outputs.
        manifest = self._with_schema(
            {"json_contracts": json_contracts or {}, "csv_contracts": csv_contracts or {}}, artifact_name
        )
        save_json(manifest, step_dir / "schema_manifest.json")

    @staticmethod
    def _read_optional_csv(path) -> pd.DataFrame | None:
        """Read a CSV artifact when present, otherwise return None."""
        if not path.exists():
            return None
        return read_data(str(path), watch_curly_brackets=False)

    @staticmethod
    def _ensure_artifact(path, message: str) -> None:
        """Raise a clear error when a required artifact is missing."""
        if not path.exists():
            raise FileNotFoundError(f"{message} Missing artifact: {path}")
