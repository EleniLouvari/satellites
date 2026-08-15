"""Shared base class helpers for pipeline step implementations."""

from __future__ import annotations

from typing import Any

from .persistence import save_json
from .config import ClassificationPipelineConfig


# Centralize behavior shared by every pipeline stage to keep step implementations focused.
class PipelineStepBase:
    """Provide shared schema and artifact helpers for step classes."""

    def __init__(self, config: ClassificationPipelineConfig):
        """Initialize the step base with validated pipeline configuration."""
        # Keep a shared config reference for all inherited step helpers.
        self.config = config
        self.config.ensure_directories()

    def _with_schema(self, payload: dict[str, Any], artifact_name: str) -> dict[str, Any]:
        """Attach schema metadata to an artifact payload dictionary."""
        # Copy the payload to avoid mutating caller-owned objects.
        enriched = dict(payload)
        enriched["_schema"] = {
            "artifact": artifact_name,
            "schema_version": self.config.output_schema_version,
        }
        return enriched

    def _save_schema_manifest(
        self,
        step_dir,
        artifact_name: str,
        json_contracts: dict[str, Any] | None = None,
        csv_contracts: dict[str, list[str]] | None = None,
    ) -> None:
        """Persist a schema manifest describing JSON and CSV artifact contracts."""
        # Save manifest metadata so downstream tools can validate outputs.
        manifest = self._with_schema(
            {
                "json_contracts": json_contracts or {},
                "csv_contracts": csv_contracts or {},
            },
            artifact_name,
        )
        save_json(manifest, step_dir / "schema_manifest.json")
