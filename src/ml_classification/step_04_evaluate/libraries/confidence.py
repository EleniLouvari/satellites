"""Fit OOF confidence evidence, report holdout diagnostics, and freeze the prediction contract.

Used by step_04_evaluate/evaluate.py after model selection. Holdout results are
reported but do not fit the selected strategy or its confidence thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml_classification.step_04_evaluate.libraries.class_reliability import calculate_oof_class_reliability
from ml_classification.shared.config.config import ClassificationPipelineConfig
from ml_classification.shared.persistence import save_frame_csv
from ml_classification.shared.rank_confidence import rank_confidence_thresholds_from_config
from ml_classification.step_04_evaluate.libraries.ranking import (
    apply_class_reliability_guard,
    build_parcel_ranking_outputs,
    summarize_rank_confidence,
)


@dataclass
class EvaluationConfidence:
    """Tables and CSV contracts produced by evaluation confidence diagnostics."""

    ranking_metrics: pd.DataFrame
    train_ranking_metrics: pd.DataFrame
    metrics: pd.DataFrame
    by_class: pd.DataFrame
    oof_metrics: pd.DataFrame
    csv_contracts: dict[str, list[str]]


def evaluate_rank_confidence(
    config: ClassificationPipelineConfig, inputs: dict, selection: dict, caches: dict
) -> EvaluationConfidence:
    """Persist ranking/confidence diagnostics and attach the frozen contract to selection."""
    labels = inputs["context"]["labels"]
    y_test, id_test = inputs["y_test"], inputs["id_test"]
    probability_cache_train, probability_cache_test = caches["train"], caches["test"]
    oof_probability_cache = caches["oof"]
    # Build and persist parcel-level class suggestions based on ranking methodologies.
    parcel_ranking_df, ranking_method_summary_df = build_parcel_ranking_outputs(
        config, probability_cache=probability_cache_test, selection=selection, labels=labels, y_true=y_test, id_col=id_test
    )
    train_parcel_ranking_df, ranking_train_summary_df = build_parcel_ranking_outputs(
        config,
        probability_cache=probability_cache_train,
        selection=selection,
        labels=labels,
        y_true=inputs["y_train"],
        id_col=inputs["id_train"],
    )
    oof_parcel_ranking_df = pd.DataFrame()
    if config.rank_confidence_enabled and oof_probability_cache:
        oof_parcel_ranking_df, _ = build_parcel_ranking_outputs(
            config,
            probability_cache=oof_probability_cache,
            selection=selection,
            labels=labels,
            y_true=inputs["y_train"],
            id_col=inputs["id_train"],
        )

    confidence_enabled = bool(config.rank_confidence_enabled)
    reliability_guard_enabled = bool(confidence_enabled and config.class_reliability_enabled)
    class_reliability = None
    if confidence_enabled:
        if oof_parcel_ranking_df.empty:
            raise RuntimeError(
                "Error: Simplified confidence requires selected-model OOF probabilities. Rerun Step 3 so Step 4 can fit the OOF class-reliability contract."
            )
        # This is the only fit: the prediction labels already reproduce deployed soft voting and multipliers.
        class_reliability = calculate_oof_class_reliability(
            true_labels=oof_parcel_ranking_df["true_label"].to_numpy(),
            predicted_labels=oof_parcel_ranking_df["predicted_class"].to_numpy(),
            class_names=labels,
            minimum_support=config.class_reliability_minimum_oof_support,
            high_min_precision=config.class_reliability_high_min_precision,
            medium_min_precision=config.class_reliability_medium_min_precision,
        )
        # Attach class reliability everywhere; only combine it into final confidence when enabled.
        parcel_ranking_df = apply_class_reliability_guard(
            parcel_ranking_df, class_reliability, combine_with_rank=reliability_guard_enabled
        )
        train_parcel_ranking_df = apply_class_reliability_guard(
            train_parcel_ranking_df, class_reliability, combine_with_rank=reliability_guard_enabled
        )
        oof_parcel_ranking_df = apply_class_reliability_guard(
            oof_parcel_ranking_df, class_reliability, combine_with_rank=reliability_guard_enabled
        )

    if not parcel_ranking_df.empty:
        ranking_dir = config.evaluate_dir / "ranking"
        save_frame_csv(parcel_ranking_df, ranking_dir / "parcel_best_class_by_ranking.csv")
    if not train_parcel_ranking_df.empty:
        ranking_dir = config.evaluate_dir / "ranking"
        save_frame_csv(train_parcel_ranking_df, ranking_dir / "parcel_best_class_by_ranking_train.csv")
    if not ranking_method_summary_df.empty:
        ranking_dir = config.evaluate_dir / "ranking"
        save_frame_csv(ranking_method_summary_df, ranking_dir / "ranking_method_metrics.csv")
    if not ranking_train_summary_df.empty:
        ranking_dir = config.evaluate_dir / "ranking"
        save_frame_csv(ranking_train_summary_df, ranking_dir / "ranking_method_metrics_train.csv")

    confidence_summary_df = pd.DataFrame()
    confidence_by_class_df = pd.DataFrame()
    oof_confidence_summary_df = pd.DataFrame()
    if config.rank_confidence_enabled and not parcel_ranking_df.empty:
        # Store all confidence artifacts under one dedicated evaluation directory.
        confidence_dir = config.evaluate_dir / "confidence"
        class_reliability_df = pd.DataFrame()
        if class_reliability is not None:
            class_reliability_df = pd.DataFrame(
                [
                    {
                        "class": class_name,
                        "oof_prediction_support": record["support"],
                        "oof_correct": record["correct"],
                        "oof_precision": record["precision"],
                        "reliability_level": record["level"],
                        "reliability_valid": record["valid"],
                        "reliability_reason": record["reason"],
                    }
                    for class_name, record in class_reliability["classes"].items()
                ]
            )
            save_frame_csv(class_reliability_df, confidence_dir / "class_reliability_oof.csv")
        confidence_summary_df, confidence_by_class_df = summarize_rank_confidence(parcel_ranking_df)
        if class_reliability is not None and not confidence_by_class_df.empty:
            oof_precision = {class_name: record["precision"] for class_name, record in class_reliability["classes"].items()}
            test_precision = parcel_ranking_df.groupby("predicted_class", observed=True)["correct"].mean().to_dict()
            confidence_by_class_df["oof_class_precision"] = confidence_by_class_df["predicted_class"].map(oof_precision)
            confidence_by_class_df["test_class_precision"] = confidence_by_class_df["predicted_class"].map(test_precision)
        save_frame_csv(parcel_ranking_df, confidence_dir / "rank_confidence_test.csv")
        save_frame_csv(confidence_summary_df, confidence_dir / "confidence_level_metrics_test.csv")
        save_frame_csv(confidence_by_class_df, confidence_dir / "confidence_by_class_test.csv")
        if not oof_parcel_ranking_df.empty:
            oof_confidence_summary_df, _ = summarize_rank_confidence(oof_parcel_ranking_df)
            save_frame_csv(oof_parcel_ranking_df, confidence_dir / "rank_confidence_oof.csv")
            save_frame_csv(oof_confidence_summary_df, confidence_dir / "confidence_level_metrics_oof.csv")

    holdout_monotonic = None
    if not confidence_summary_df.empty:
        accuracy_by_level = confidence_summary_df.set_index("confidence_level")["accuracy"]
        ordered_accuracy = [accuracy_by_level.get(level, np.nan) for level in ("HIGH", "MEDIUM", "LOW")]
        if np.isfinite(ordered_accuracy).all():
            holdout_monotonic = bool(ordered_accuracy[0] > ordered_accuracy[1] > ordered_accuracy[2])

    # Freeze the complete deployed definition. Step 5 never reads live thresholds when this exists.
    confidence_contract = {
        "enabled": confidence_enabled,
        "method": ("rank_consensus_with_class_reliability_guard" if reliability_guard_enabled else "rank_consensus_only"),
        "version": "2.0",
        "prediction_basis": selection["selection_type"],
        "class_reliability_enabled": bool(config.class_reliability_enabled),
        "rank": {"minimum_models": int(config.rank_confidence_minimum_models), **rank_confidence_thresholds_from_config(config)},
        "class_reliability": class_reliability,
        "holdout_accuracy_monotonic": holdout_monotonic,
        "holdout_level_counts": (
            parcel_ranking_df[
                "prediction_confidence_level" if "prediction_confidence_level" in parcel_ranking_df else "rank_confidence_level"
            ]
            .value_counts()
            .to_dict()
            if not parcel_ranking_df.empty and confidence_enabled
            else {}
        ),
        "holdout_rank_level_counts": (
            parcel_ranking_df["rank_confidence_level"].value_counts().to_dict()
            if not parcel_ranking_df.empty and confidence_enabled
            else {}
        ),
        "holdout_class_reliability_level_counts": (
            parcel_ranking_df["class_reliability_level"].value_counts().to_dict()
            if class_reliability is not None and not parcel_ranking_df.empty
            else {}
        ),
        "oof_level_counts": (
            oof_parcel_ranking_df[
                "prediction_confidence_level"
                if "prediction_confidence_level" in oof_parcel_ranking_df
                else "rank_confidence_level"
            ]
            .value_counts()
            .to_dict()
            if not oof_parcel_ranking_df.empty and confidence_enabled
            else {}
        ),
        "oof_rank_level_counts": (
            oof_parcel_ranking_df["rank_confidence_level"].value_counts().to_dict()
            if not oof_parcel_ranking_df.empty and confidence_enabled
            else {}
        ),
        "oof_class_reliability_level_counts": (
            oof_parcel_ranking_df["class_reliability_level"].value_counts().to_dict()
            if class_reliability is not None and not oof_parcel_ranking_df.empty
            else {}
        ),
    }
    selection["confidence"] = confidence_contract
    # Retain a compact alias for report readers that predate the top-level confidence key.
    selection["rank_confidence"] = confidence_contract

    csv_contracts = {}
    if not ranking_method_summary_df.empty:
        csv_contracts["ranking_method_metrics.csv"] = ranking_method_summary_df.columns.tolist()
    if not parcel_ranking_df.empty:
        csv_contracts["parcel_best_class_by_ranking.csv"] = parcel_ranking_df.columns.tolist()
    if not confidence_summary_df.empty:
        csv_contracts["confidence/confidence_level_metrics_test.csv"] = confidence_summary_df.columns.tolist()
        csv_contracts["confidence/confidence_by_class_test.csv"] = confidence_by_class_df.columns.tolist()
        csv_contracts["confidence/rank_confidence_test.csv"] = parcel_ranking_df.columns.tolist()
    if not oof_parcel_ranking_df.empty:
        csv_contracts["confidence/rank_confidence_oof.csv"] = oof_parcel_ranking_df.columns.tolist()
        csv_contracts["confidence/confidence_level_metrics_oof.csv"] = oof_confidence_summary_df.columns.tolist()
    if class_reliability is not None:
        csv_contracts["confidence/class_reliability_oof.csv"] = class_reliability_df.columns.tolist()

    return EvaluationConfidence(
        ranking_metrics=ranking_method_summary_df,
        train_ranking_metrics=ranking_train_summary_df,
        metrics=confidence_summary_df,
        by_class=confidence_by_class_df,
        oof_metrics=oof_confidence_summary_df,
        csv_contracts=csv_contracts,
    )
