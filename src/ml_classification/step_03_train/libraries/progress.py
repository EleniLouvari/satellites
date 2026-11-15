"""Live per-model progress of the Train step, saved as a small JSON file next to its outputs."""

from __future__ import annotations

import contextlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

PROGRESS_FILENAME = "training_progress.json"
# pending → running → finished | failed; reused models were trained by an earlier run;
# unavailable models were selected but could not be built (for example a missing library).
STATUSES = ("pending", "running", "finished", "failed", "reused", "unavailable")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class TrainingProgress:
    """Record each candidate's training state so a notebook or web app can follow the step."""

    def __init__(self, train_dir: str | Path, models: list[str], unavailable: list[str] = ()):
        """Initialize model states and save progress in the training directory."""
        self.path = Path(train_dir) / PROGRESS_FILENAME
        self.models = [{"name": name, "status": "pending"} for name in models]
        self.models += [{"name": name, "status": "unavailable"} for name in unavailable if name not in models]
        self.started_at = _now()
        self.finished_at = None
        self._save()

    def _entry(self, name: str) -> dict:
        return next(model for model in self.models if model["name"] == name)

    def start(self, name: str) -> None:
        """Mark a model as running and save its start time."""
        self._entry(name).update(status="running", started_at=_now())
        self._save()

    def finish(self, name: str, status: str, duration_seconds: float | None = None, error: str | None = None) -> None:
        """Save a model's status and finish time with optional duration and error details."""
        if status not in STATUSES:
            raise ValueError(f"Error: Unknown training status {status!r}.")
        entry = self._entry(name)
        entry.update(status=status, finished_at=_now())
        if duration_seconds is not None:
            entry["duration_seconds"] = round(float(duration_seconds), 1)
        if error:
            entry["error"] = error[:500]
        self._save()

    def close(self) -> None:
        """Record and save the training step's finish time."""
        self.finished_at = _now()
        self._save()

    def _save(self) -> None:
        """Replace the file atomically; progress reporting must never stop training."""
        payload = {"started_at": self.started_at, "finished_at": self.finished_at, "models": self.models}
        temporary = self.path.with_suffix(".partial.json")
        with contextlib.suppress(OSError):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary, self.path)
