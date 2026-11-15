"""Train step progress file: one entry per model, updated as each model starts and ends."""

import json

import pytest

from ml_classification.step_03_train.libraries.progress import PROGRESS_FILENAME, TrainingProgress


def test_training_progress_records_each_model_state(tmp_path):
    progress = TrainingProgress(tmp_path, ["random_forest", "xgboost", "svm"], ["keras_lstm", "svm"])
    read = lambda: json.loads((tmp_path / PROGRESS_FILENAME).read_text(encoding="utf-8"))
    assert [(m["name"], m["status"]) for m in read()["models"]] == [
        ("random_forest", "pending"),
        ("xgboost", "pending"),
        ("svm", "pending"),
        ("keras_lstm", "unavailable"),
    ]
    progress.start("random_forest")
    assert read()["models"][0]["status"] == "running" and "started_at" in read()["models"][0]
    progress.finish("random_forest", "finished", 12.345)
    progress.finish("xgboost", "failed", 1.0, "boom")
    progress.finish("svm", "reused")
    progress.close()
    data = read()
    assert data["finished_at"] is not None
    assert data["models"][0]["status"] == "finished" and data["models"][0]["duration_seconds"] == 12.3
    assert data["models"][1]["status"] == "failed" and data["models"][1]["error"] == "boom"
    assert data["models"][2]["status"] == "reused"
    assert not list(tmp_path.glob("*.partial.json"))
    with pytest.raises(ValueError):
        progress.finish("svm", "done")
