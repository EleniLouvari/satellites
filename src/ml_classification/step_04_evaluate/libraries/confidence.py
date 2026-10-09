"""Fit OOF confidence evidence, report holdout diagnostics, and freeze the prediction contract.

Used by step_04_evaluate/evaluate.py after model selection. Holdout results are
reported but do not fit the selected strategy or its confidence thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml_classification.shared.config.config import ClassificationPipelineConfig
from ml_classification.shared.persistence import save_frame_csv
from ml_classification.shared.rank_confidence import rank_confidence_thresholds_from_config
from ml_classification.step_04_evaluate.libraries.class_reliability import calculate_oof_class_reliability
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
    confidence_enabled = bool(config.rank_confidence_enabled)
    reliability_guard_enabled = bool(confidence_enabled and config.class_reliability_enabled)

    # Step 1: Build ranking outputs per split (holdout/train/oof) from already-computed probability caches.
    parcel_ranking_df, ranking_method_summary_df = _build_ranking_outputs(
        config=config,
        selection=selection,
        labels=labels,
        probability_cache=caches["test"],
        y_true=inputs["y_test"],
        id_col=inputs["id_test"],
    )
    train_parcel_ranking_df, ranking_train_summary_df = _build_ranking_outputs(
        config=config,
        selection=selection,
        labels=labels,
        probability_cache=caches["train"],
        y_true=inputs["y_train"],
        id_col=inputs["id_train"],
    )
    oof_parcel_ranking_df = _build_oof_ranking_outputs(
        config=config,
        selection=selection,
        labels=labels,
        probability_cache=caches["oof"],
        y_train=inputs["y_train"],
        id_train=inputs["id_train"],
    )

    class_reliability = None
    # Step 2: Fit class reliability once from OOF predictions, then apply it consistently to all ranking frames.
    if confidence_enabled:
        class_reliability = _fit_class_reliability_from_oof(config=config, labels=labels, oof_parcel_ranking_df=oof_parcel_ranking_df)
        parcel_ranking_df, train_parcel_ranking_df, oof_parcel_ranking_df = _apply_class_reliability_to_rankings(
            parcel_ranking_df=parcel_ranking_df,
            train_parcel_ranking_df=train_parcel_ranking_df,
            oof_parcel_ranking_df=oof_parcel_ranking_df,
            class_reliability=class_reliability,
            reliability_guard_enabled=reliability_guard_enabled,
        )

    # Step 3: Persist ranking/confidence artifacts so report generation never depends on in-memory state.
    _save_ranking_artifacts(
        config=config,
        parcel_ranking_df=parcel_ranking_df,
        train_parcel_ranking_df=train_parcel_ranking_df,
        ranking_method_summary_df=ranking_method_summary_df,
        ranking_train_summary_df=ranking_train_summary_df,
    )
    confidence_summary_df, confidence_by_class_df, oof_confidence_summary_df, class_reliability_df = _save_confidence_artifacts(
        config=config,
        confidence_enabled=confidence_enabled,
        parcel_ranking_df=parcel_ranking_df,
        oof_parcel_ranking_df=oof_parcel_ranking_df,
        class_reliability=class_reliability,
    )

    # Step 4: Freeze the deployed confidence contract for step-05 prediction (no live-threshold drift).
    holdout_monotonic = _compute_holdout_monotonic(confidence_summary_df)
    confidence_contract = _build_confidence_contract(
        config=config,
        selection=selection,
        confidence_enabled=confidence_enabled,
        reliability_guard_enabled=reliability_guard_enabled,
        class_reliability=class_reliability,
        holdout_monotonic=holdout_monotonic,
        parcel_ranking_df=parcel_ranking_df,
        oof_parcel_ranking_df=oof_parcel_ranking_df,
    )
    selection["confidence"] = confidence_contract
    # Keep backwards-compatible alias for legacy report/dashboard readers.
    selection["rank_confidence"] = confidence_contract

    csv_contracts = _build_csv_contracts(
        ranking_method_summary_df=ranking_method_summary_df,
        parcel_ranking_df=parcel_ranking_df,
        confidence_summary_df=confidence_summary_df,
        confidence_by_class_df=confidence_by_class_df,
        oof_parcel_ranking_df=oof_parcel_ranking_df,
        oof_confidence_summary_df=oof_confidence_summary_df,
        class_reliability=class_reliability,
        class_reliability_df=class_reliability_df,
    )

    return EvaluationConfidence(
        ranking_metrics=ranking_method_summary_df,
        train_ranking_metrics=ranking_train_summary_df,
        metrics=confidence_summary_df,
        by_class=confidence_by_class_df,
        oof_metrics=oof_confidence_summary_df,
        csv_contracts=csv_contracts,
    )


def _build_ranking_outputs(
    config: ClassificationPipelineConfig,
    selection: dict,
    labels: list[str],
    probability_cache: dict,
    y_true: pd.Series | np.ndarray,
    id_col: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build parcel-level ranking outputs and ranking-summary metrics for one split."""
    return build_parcel_ranking_outputs(
        config, probability_cache=probability_cache, selection=selection, labels=labels, y_true=y_true, id_col=id_col
    )


def _build_oof_ranking_outputs(
    config: ClassificationPipelineConfig,
    selection: dict,
    labels: list[str],
    probability_cache: dict | None,
    y_train: pd.Series | np.ndarray,
    id_train: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Build OOF ranking outputs used for confidence-contract fitting."""
    if not config.rank_confidence_enabled or not probability_cache:
        return pd.DataFrame()
    oof_parcel_ranking_df, _ = build_parcel_ranking_outputs(
        config=config,
        probability_cache=probability_cache,
        selection=selection,
        labels=labels,
        y_true=y_train,
        id_col=id_train,
    )
    return oof_parcel_ranking_df


def _fit_class_reliability_from_oof(
    config: ClassificationPipelineConfig, labels: list[str], oof_parcel_ranking_df: pd.DataFrame
) -> dict[str, object]:
    """Fit class reliability from OOF true/predicted labels."""
    if oof_parcel_ranking_df.empty:
        raise RuntimeError(
            "Error: Simplified confidence requires selected-model OOF probabilities. Rerun Step 3 so Step 4 can fit the OOF class-reliability contract."
        )
    return calculate_oof_class_reliability(
        true_labels=oof_parcel_ranking_df["true_label"].to_numpy(),
        predicted_labels=oof_parcel_ranking_df["predicted_class"].to_numpy(),
        class_names=labels,
        minimum_support=config.class_reliability_minimum_oof_support,
        high_min_precision=config.class_reliability_high_min_precision,
        medium_min_precision=config.class_reliability_medium_min_precision,
    )


def _apply_class_reliability_to_rankings(
    parcel_ranking_df: pd.DataFrame,
    train_parcel_ranking_df: pd.DataFrame,
    oof_parcel_ranking_df: pd.DataFrame,
    class_reliability: dict[str, object],
    reliability_guard_enabled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach class-reliability fields and optionally combine with rank confidence."""
    parcel_ranking_df = apply_class_reliability_guard(
        parcel_ranking_df, class_reliability, combine_with_rank=reliability_guard_enabled
    )
    train_parcel_ranking_df = apply_class_reliability_guard(
        train_parcel_ranking_df, class_reliability, combine_with_rank=reliability_guard_enabled
    )
    oof_parcel_ranking_df = apply_class_reliability_guard(
        oof_parcel_ranking_df, class_reliability, combine_with_rank=reliability_guard_enabled
    )
    return parcel_ranking_df, train_parcel_ranking_df, oof_parcel_ranking_df


def _save_ranking_artifacts(
    config: ClassificationPipelineConfig,
    parcel_ranking_df: pd.DataFrame,
    train_parcel_ranking_df: pd.DataFrame,
    ranking_method_summary_df: pd.DataFrame,
    ranking_train_summary_df: pd.DataFrame,
) -> None:
    """Persist ranking CSV artifacts under evaluate/ranking."""
    ranking_dir = config.evaluate_dir / "ranking"
    if not parcel_ranking_df.empty:
        save_frame_csv(parcel_ranking_df, ranking_dir / "parcel_best_class_by_ranking.csv")
    if not train_parcel_ranking_df.empty:
        save_frame_csv(train_parcel_ranking_df, ranking_dir / "parcel_best_class_by_ranking_train.csv")
    if not ranking_method_summary_df.empty:
        save_frame_csv(ranking_method_summary_df, ranking_dir / "ranking_method_metrics.csv")
    if not ranking_train_summary_df.empty:
        save_frame_csv(ranking_train_summary_df, ranking_dir / "ranking_method_metrics_train.csv")


def _build_class_reliability_frame(class_reliability: dict[str, object]) -> pd.DataFrame:
    """Convert class-reliability contract into a tabular CSV representation."""
    return pd.DataFrame(
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


def _save_confidence_artifacts(
    config: ClassificationPipelineConfig,
    confidence_enabled: bool,
    parcel_ranking_df: pd.DataFrame,
    oof_parcel_ranking_df: pd.DataFrame,
    class_reliability: dict[str, object] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Persist confidence diagnostics and return summary tables used by downstream contracts."""
    confidence_summary_df = pd.DataFrame()
    confidence_by_class_df = pd.DataFrame()
    oof_confidence_summary_df = pd.DataFrame()
    class_reliability_df = pd.DataFrame()

    if not confidence_enabled or parcel_ranking_df.empty:
        return confidence_summary_df, confidence_by_class_df, oof_confidence_summary_df, class_reliability_df

    confidence_dir = config.evaluate_dir / "confidence"
    if class_reliability is not None:
        class_reliability_df = _build_class_reliability_frame(class_reliability)
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
    return confidence_summary_df, confidence_by_class_df, oof_confidence_summary_df, class_reliability_df


def _compute_holdout_monotonic(confidence_summary_df: pd.DataFrame) -> bool | None:
    """Return whether HIGH > MEDIUM > LOW accuracy holds on holdout metrics."""
    if confidence_summary_df.empty:
        return None
    accuracy_by_level = confidence_summary_df.set_index("confidence_level")["accuracy"]
    ordered_accuracy = [accuracy_by_level.get(level, np.nan) for level in ("HIGH", "MEDIUM", "LOW")]
    if not np.isfinite(ordered_accuracy).all():
        return None
    return bool(ordered_accuracy[0] > ordered_accuracy[1] > ordered_accuracy[2])


def _confidence_level_counts(frame: pd.DataFrame, preferred_column: str, enabled: bool) -> dict:
    """Count confidence levels using preferred column when available, otherwise rank-based fallback."""
    if frame.empty or not enabled:
        return {}
    selected_column = preferred_column if preferred_column in frame else "rank_confidence_level"
    return frame[selected_column].value_counts().to_dict()


def _build_confidence_contract(
    config: ClassificationPipelineConfig,
    selection: dict,
    confidence_enabled: bool,
    reliability_guard_enabled: bool,
    class_reliability: dict[str, object] | None,
    holdout_monotonic: bool | None,
    parcel_ranking_df: pd.DataFrame,
    oof_parcel_ranking_df: pd.DataFrame,
) -> dict[str, object]:
    """Build the frozen confidence contract consumed by step-05 prediction."""
    return {
        "enabled": confidence_enabled,
        "method": ("rank_consensus_with_class_reliability_guard" if reliability_guard_enabled else "rank_consensus_only"),
        "version": "2.0",
        "prediction_basis": selection["selection_type"],
        "class_reliability_enabled": bool(config.class_reliability_enabled),
        "rank": {"minimum_models": int(config.rank_confidence_minimum_models), **rank_confidence_thresholds_from_config(config)},
        "class_reliability": class_reliability,
        "holdout_accuracy_monotonic": holdout_monotonic,
        "holdout_level_counts": _confidence_level_counts(parcel_ranking_df, "prediction_confidence_level", confidence_enabled),
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
        "oof_level_counts": _confidence_level_counts(oof_parcel_ranking_df, "prediction_confidence_level", confidence_enabled),
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


def _build_csv_contracts(
    ranking_method_summary_df: pd.DataFrame,
    parcel_ranking_df: pd.DataFrame,
    confidence_summary_df: pd.DataFrame,
    confidence_by_class_df: pd.DataFrame,
    oof_parcel_ranking_df: pd.DataFrame,
    oof_confidence_summary_df: pd.DataFrame,
    class_reliability: dict[str, object] | None,
    class_reliability_df: pd.DataFrame,
) -> dict[str, list[str]]:
    """Build CSV column contracts keyed by persisted artifact filename."""
    csv_contracts: dict[str, list[str]] = {}
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
    return csv_contracts
