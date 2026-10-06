"""Explainable inspection-priority scoring for final parcel predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

INSPECTION_OUTPUT_COLUMNS = (
    "label_prediction_status",
    "confidence_risk",
    "data_risk",
    "geometry_risk",
    "inspection_score",
    "inspection_need_base",
    "inspection_need",
    "inspection_check_type",
    "inspection_reasons",
)


def _normalize_label(value):
    """Normalize declared and predicted class codes before comparison."""
    # Preserve missing declarations so they are not mistaken for a real crop code.
    if pd.isna(value):
        return pd.NA
    # Make numerically equivalent codes such as 1.0 and "1" compare as the same label.
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _label_prediction_status(data: pd.DataFrame, config) -> pd.Series:
    """Classify the farmer declaration versus prediction when both exist."""
    # Prediction-only parcels default to NO_DECLARATION and remain scoreable.
    status = pd.Series("NO_DECLARATION", index=data.index, dtype="string")
    target_column = getattr(config, "target_column", None)
    prediction_column = getattr(config, "prediction_column", None)
    # Older/custom datasets may not carry both label columns; keep the safe default.
    if not target_column or not prediction_column or target_column not in data or prediction_column not in data:
        return status

    # Normalize both sides before comparing farmer crop codes with model outputs.
    declaration = data[target_column].map(_normalize_label)
    prediction = data[prediction_column].map(_normalize_label)
    declared = declaration.notna()
    comparable = declared & prediction.notna()
    # A known declaration without a model result cannot support a conflict decision.
    status.loc[declared & ~prediction.notna()] = "UNAVAILABLE"
    # SAME/DIFFERENT is assigned only where both values are actually available.
    status.loc[comparable] = np.where(declaration.loc[comparable].eq(prediction.loc[comparable]), "SAME", "DIFFERENT")
    return status


def _apply_label_aware_policy(
    result: pd.DataFrame, confidence: pd.Series, reliability: pd.Series, config
) -> tuple[pd.Series, pd.Series]:
    """Convert base risk into an operational need and a mutually exclusive check type."""
    # Start from the label-independent score category; declaration rules override it below.
    need = result["inspection_need_base"].copy()
    check_type = pd.Series("NONE", index=result.index, dtype="string")
    status = result["label_prediction_status"]
    geometry_risk = result["geometry_risk"]
    valid = result["inspection_score"].notna()

    # Reuse the configured quality thresholds so scoring, reasons, and policy stay aligned.
    data_high = reliability >= config.inspection_medium_data_reliability_threshold
    data_low = reliability < config.inspection_low_data_reliability_threshold
    geometry_good = geometry_risk < config.inspection_medium_geometry_risk_threshold
    geometry_complex = geometry_risk >= config.inspection_high_geometry_risk_threshold
    # Sufficient evidence requires a valid score, HIGH model confidence, and reliable EO data.
    evidence_is_sufficient = valid & confidence.eq("HIGH") & data_high & geometry_risk.notna()

    # Agreement under strong evidence needs no check; weaker agreement remains inconclusive.
    same = status.eq("SAME")
    strong_verification = same & evidence_is_sufficient & geometry_good
    need.loc[strong_verification] = "LOW"
    need.loc[same & valid & ~strong_verification] = "MEDIUM"
    need.loc[same & ~valid] = "UNKNOWN"
    check_type.loc[same & ~strong_verification] = "INSUFFICIENT_EO_EVIDENCE"

    # A reliable disagreement is a suspected declaration conflict, not an EO-quality failure.
    different = status.eq("DIFFERENT")
    declaration_conflict = different & evidence_is_sufficient
    strong_conflict = declaration_conflict & geometry_good
    need.loc[strong_conflict] = "VERY_HIGH"
    need.loc[declaration_conflict & ~geometry_good] = "HIGH"
    check_type.loc[declaration_conflict] = "DECLARATION_CONFLICT"

    # All remaining disagreements lack enough EO/model evidence to assert a declaration conflict.
    insufficient_difference = different & ~declaration_conflict
    # LOW confidence plus poor data and complex geometry is explicitly marked uncertain.
    uncertain_difference = insufficient_difference & valid & confidence.eq("LOW") & data_low & geometry_complex
    need.loc[uncertain_difference] = "MEDIUM_UNCERTAIN"
    need.loc[insufficient_difference & valid & ~uncertain_difference] = "MEDIUM_HIGH"
    need.loc[insufficient_difference & ~valid] = "UNKNOWN"
    check_type.loc[insufficient_difference] = "INSUFFICIENT_EO_EVIDENCE"

    # Without a farmer declaration, retain the base need and classify only non-LOW cases.
    no_declaration = status.eq("NO_DECLARATION")
    check_type.loc[no_declaration & need.ne("LOW")] = "INSUFFICIENT_EO_EVIDENCE"
    # A missing prediction prevents verification even when a declaration exists.
    unavailable = status.eq("UNAVAILABLE")
    need.loc[unavailable] = "UNKNOWN"
    check_type.loc[unavailable] = "INSUFFICIENT_EO_EVIDENCE"
    return need, check_type


def _inspection_reasons(row: pd.Series, config) -> str:
    """Return stable, pipe-separated explanations for one parcel score."""
    reasons: list[str] = []
    confidence = row[config.prediction_confidence_level_column]
    reliability = row[config.inspection_data_reliability_column]
    geometry_risk = row["geometry_risk"]
    label_status = row["label_prediction_status"]
    check_type = row["inspection_check_type"]

    # Put the primary operational explanation first in the pipe-separated reason string.
    if label_status == "DIFFERENT":
        if check_type == "DECLARATION_CONFLICT" and row["inspection_need"] == "VERY_HIGH":
            reasons.append("STRONG_DECLARATION_CONFLICT")
        elif check_type == "DECLARATION_CONFLICT":
            reasons.append("DECLARATION_CONFLICT")
        else:
            reasons.append("DECLARATION_PREDICTION_DIFFERENCE")
    if check_type == "INSUFFICIENT_EO_EVIDENCE":
        reasons.append("INSUFFICIENT_EO_EVIDENCE")

    # Normalize raw values again for robust row-level reason assignment.
    if not pd.isna(confidence):
        confidence = str(confidence).strip().upper()
    if not pd.isna(reliability):
        reliability = pd.to_numeric(reliability, errors="coerce")
        if pd.isna(reliability) or not 0.0 <= float(reliability) <= 1.0:
            reliability = np.nan

    if pd.isna(confidence):
        reasons.append("INVALID_MODEL_CONFIDENCE")
    elif confidence == "LOW":
        reasons.append("LOW_MODEL_CONFIDENCE")
    elif confidence == "MEDIUM":
        reasons.append("MEDIUM_MODEL_CONFIDENCE")
    elif confidence != "HIGH":
        reasons.append("INVALID_MODEL_CONFIDENCE")

    if pd.isna(reliability):
        reasons.append("MISSING_DATA_RELIABILITY")
    elif reliability < config.inspection_low_data_reliability_threshold:
        reasons.append("LOW_DATA_RELIABILITY")
    elif reliability < config.inspection_medium_data_reliability_threshold:
        reasons.append("MEDIUM_DATA_RELIABILITY")

    if pd.isna(geometry_risk):
        reasons.append("MISSING_GEOMETRY_COMPLEXITY")
    elif geometry_risk >= config.inspection_high_geometry_risk_threshold:
        reasons.append("HIGH_GEOMETRY_COMPLEXITY")
    elif geometry_risk >= config.inspection_medium_geometry_risk_threshold:
        reasons.append("MEDIUM_GEOMETRY_COMPLEXITY")

    return "|".join(reasons) if reasons else "NONE"


def calculate_inspection_metrics(data: pd.DataFrame, config) -> pd.DataFrame:
    """Calculate independent model, data, and geometry risks for every row.

    Geometry complexity is converted to a relative risk using its percentile
    rank within the prediction population. Scores remain null when any required
    risk component is unavailable, while reason codes identify the missing
    input explicitly.
    """
    required = {
        config.prediction_confidence_level_column,
        config.inspection_data_reliability_column,
        config.inspection_geometry_complexity_column,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Inspection scoring requires columns: {missing}")

    # Work on a copy so prediction inputs and existing confidence columns are never mutated.
    result = data.copy()
    confidence = result[config.prediction_confidence_level_column].astype("string").str.strip().str.upper()
    reliability = pd.to_numeric(result[config.inspection_data_reliability_column], errors="coerce")
    reliability = reliability.where(reliability.between(0.0, 1.0))
    complexity = pd.to_numeric(result[config.inspection_geometry_complexity_column], errors="coerce")
    complexity = complexity.where(complexity >= 0.0)

    # Declaration comparison is descriptive and does not enter the numeric score formula.
    result["label_prediction_status"] = _label_prediction_status(result, config)
    # Convert the three independent uncertainty signals to a common 0-1 risk direction.
    result["confidence_risk"] = confidence.map({"HIGH": 0.0, "MEDIUM": 0.5, "LOW": 1.0}).astype(float)
    result["data_risk"] = 1.0 - reliability
    result["geometry_risk"] = complexity.rank(method="average", pct=True)
    # Keep the requested 60/25/15 weighted score label-independent and reproducible.
    result["inspection_score"] = 100.0 * (
        config.inspection_model_weight * result["confidence_risk"]
        + config.inspection_data_weight * result["data_risk"]
        + config.inspection_geometry_weight * result["geometry_risk"]
    )
    result["inspection_score"] = result["inspection_score"].clip(0.0, 100.0).round(2)

    # Store the score-only category separately so the declaration policy remains auditable.
    score = result["inspection_score"]
    result["inspection_need_base"] = pd.Series(
        np.select(
            [score >= config.inspection_high_score_threshold, score >= config.inspection_medium_score_threshold],
            ["HIGH", "MEDIUM"],
            default="LOW",
        ),
        index=result.index,
        dtype="string",
    ).where(score.notna(), "UNKNOWN")
    # Apply farmer-declaration agreement only to operational need and check type.
    result["inspection_need"], result["inspection_check_type"] = _apply_label_aware_policy(
        result, confidence, reliability, config
    )
    result["inspection_reasons"] = result.apply(_inspection_reasons, axis=1, config=config).astype("string")
    return result
