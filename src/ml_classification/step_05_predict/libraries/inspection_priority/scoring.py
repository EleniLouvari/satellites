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
    "inspection_priority_label",
    "inspection_rule",
    "inspection_class_reliability_applied",
    "inspection_check_type",
    "inspection_reasons",
)

INSPECTION_PRIORITY_LABELS = {
    "VERY_HIGH": "Very High",
    "HIGH": "High",
    "MEDIUM_HIGH": "Medium–High",
    "MEDIUM": "Medium",
    "LOW_MEDIUM": "Low–Medium",
    "LOW": "Low",
    "VERY_LOW": "Very Low",
    "UNKNOWN": "Unknown",
}


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
    if not prediction_column or prediction_column not in data:
        return pd.Series("UNAVAILABLE", index=data.index, dtype="string")
    if not target_column or target_column not in data:
        status.loc[data[prediction_column].isna()] = "UNAVAILABLE"
        return status

    # Normalize both sides before comparing farmer crop codes with model outputs.
    declaration = data[target_column].map(_normalize_label)
    prediction = data[prediction_column].map(_normalize_label)
    declared = declaration.notna()
    comparable = declared & prediction.notna()
    # A known declaration without a model result cannot support a conflict decision.
    status.loc[~prediction.notna()] = "UNAVAILABLE"
    # SAME/DIFFERENT is assigned only where both values are actually available.
    status.loc[comparable] = np.where(declaration.loc[comparable].eq(prediction.loc[comparable]), "SAME", "DIFFERENT")
    return status


def _apply_label_aware_policy(
    result: pd.DataFrame, confidence: pd.Series, reliability: pd.Series, config
) -> tuple[pd.Series, pd.Series]:
    """Apply the twenty direct rules; numeric diagnostic scores never assign priority."""
    status = result["label_prediction_status"]
    geometry = result["geometry_risk"]
    valid = confidence.isin(["HIGH", "MEDIUM", "LOW"]) & reliability.notna() & geometry.notna()
    rank_valid = result.get(
        "prediction_rank_confidence_valid", result.get("prediction_confidence_valid", pd.Series(True, index=result.index))
    )
    valid &= rank_valid.fillna(False).astype(bool)
    enabled, class_level, class_valid = _policy_class_reliability(result, config)
    comparable = status.isin(["SAME", "DIFFERENT"])
    eligible = valid & comparable & (~enabled | class_valid)
    # Disabled reliability bypasses its condition, without changing the recorded class evidence.
    class_high = ~enabled | class_level.eq("HIGH")
    class_medium = enabled & class_level.eq("MEDIUM")
    high_data = reliability > config.inspection_medium_data_reliability_threshold
    good_geometry = geometry < config.inspection_medium_geometry_risk_threshold
    high_rank, medium_rank = confidence.eq("HIGH"), confidence.eq("MEDIUM")
    conditions = [
        class_high & high_rank & high_data & good_geometry,
        class_high & high_rank & high_data & ~good_geometry,
        class_medium & high_rank & high_data & good_geometry,
        class_medium & high_rank & high_data & ~good_geometry,
        class_high & medium_rank & high_data & good_geometry,
        class_high & medium_rank & high_data & ~good_geometry,
        class_medium & medium_rank & high_data,
        enabled & class_level.eq("LOW"),
        confidence.eq("LOW"),
        ~high_data,
    ]
    same_priorities = ["VERY_LOW", "LOW", "LOW", "LOW_MEDIUM", "LOW_MEDIUM"] + ["MEDIUM"] * 5
    different_priorities = ["VERY_HIGH", "HIGH", "HIGH", "MEDIUM_HIGH", "MEDIUM_HIGH"] + ["MEDIUM"] * 5
    need = pd.Series("MEDIUM", index=result.index, dtype="string")
    rule = pd.Series("CLASS_RELIABILITY_UNAVAILABLE", index=result.index, dtype="string")
    # First matching rule wins; overlapping low-evidence rules all produce MEDIUM.
    masks = [condition.fillna(False).to_numpy(dtype=bool) for condition in conditions]
    for label_status, priorities, offset in [("SAME", same_priorities, 0), ("DIFFERENT", different_priorities, 10)]:
        selected = eligible & status.eq(label_status)
        assigned = pd.Series(np.select(masks, priorities, default="MEDIUM"), index=result.index)
        identifiers = [f"R{number + offset:02d}" for number in range(1, 11)]
        matched = pd.Series(np.select(masks, identifiers, default="UNCLASSIFIED"), index=result.index)
        need.loc[selected] = assigned.loc[selected]
        rule.loc[selected] = matched.loc[selected]
    rule.loc[valid & status.eq("NO_DECLARATION")] = "NO_DECLARATION"
    unavailable = ~valid | status.eq("UNAVAILABLE")
    need.loc[unavailable] = "UNKNOWN"
    rule.loc[unavailable] = "REQUIRED_EVIDENCE_UNAVAILABLE"
    result["inspection_rule"] = rule
    result["inspection_class_reliability_applied"] = enabled & eligible
    check_type = pd.Series("INSUFFICIENT_EO_EVIDENCE", index=result.index, dtype="string")
    check_type.loc[need.isin(["VERY_LOW", "LOW"])] = "NONE"
    check_type.loc[status.eq("DIFFERENT") & need.isin(["VERY_HIGH", "HIGH", "MEDIUM_HIGH"])] = "DECLARATION_CONFLICT"
    return need, check_type


def _policy_class_reliability(result: pd.DataFrame, config) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Resolve the frozen policy switch and distinguish weak reliability from absent evidence."""
    enabled = (
        result.get(
            "inspection_class_reliability_enabled",
            pd.Series(bool(getattr(config, "class_reliability_enabled", True)), index=result.index),
        )
        .fillna(True)
        .astype(bool)
    )
    level = result.get("prediction_class_reliability_level", pd.Series(pd.NA, index=result.index, dtype="string"))
    level = level.astype("string").str.strip().str.upper()
    validity = result.get("prediction_class_reliability_valid", pd.Series(False, index=result.index))
    valid = validity.fillna(False).astype(bool) & level.isin(["HIGH", "MEDIUM", "LOW"])
    return enabled, level, valid


def _append_label_alignment_reasons(reasons: list[str], label_status: str, check_type: str, inspection_need: str) -> None:
    """Append declaration-vs-prediction reasons in stable priority order."""
    if label_status == "DIFFERENT":
        if check_type == "DECLARATION_CONFLICT" and inspection_need == "VERY_HIGH":
            reasons.append("STRONG_DECLARATION_CONFLICT")
        elif check_type == "DECLARATION_CONFLICT":
            reasons.append("DECLARATION_CONFLICT")
        else:
            reasons.append("DECLARATION_PREDICTION_DIFFERENCE")
    if check_type == "INSUFFICIENT_EO_EVIDENCE":
        reasons.append("INSUFFICIENT_EO_EVIDENCE")


def _normalize_row_confidence(value) -> str | float:
    """Normalize confidence labels for reason assignment."""
    if pd.isna(value):
        return np.nan
    return str(value).strip().upper()


def _normalize_row_reliability(value) -> float:
    """Normalize reliability values to [0, 1], returning NaN for invalid rows."""
    if pd.isna(value):
        return np.nan
    reliability = pd.to_numeric(value, errors="coerce")
    if pd.isna(reliability) or not 0.0 <= float(reliability) <= 1.0:
        return np.nan
    return float(reliability)


def _append_confidence_reasons(reasons: list[str], confidence: str | float) -> None:
    """Append confidence-level reasons in deterministic order."""
    if pd.isna(confidence):
        reasons.append("INVALID_MODEL_CONFIDENCE")
    elif confidence == "LOW":
        reasons.append("LOW_MODEL_CONFIDENCE")
    elif confidence == "MEDIUM":
        reasons.append("MEDIUM_MODEL_CONFIDENCE")
    elif confidence != "HIGH":
        reasons.append("INVALID_MODEL_CONFIDENCE")


def _append_data_reliability_reasons(reasons: list[str], reliability: float, config) -> None:
    """Append reliability reasons using configured threshold bands."""
    if pd.isna(reliability):
        reasons.append("MISSING_DATA_RELIABILITY")
    elif reliability < config.inspection_low_data_reliability_threshold:
        reasons.append("LOW_DATA_RELIABILITY")
    elif reliability <= config.inspection_medium_data_reliability_threshold:
        reasons.append("MEDIUM_DATA_RELIABILITY")


def _append_geometry_reasons(reasons: list[str], geometry_risk: float, config) -> None:
    """Append geometry complexity reasons using configured threshold bands."""
    if pd.isna(geometry_risk):
        reasons.append("MISSING_GEOMETRY_COMPLEXITY")
    elif geometry_risk >= config.inspection_high_geometry_risk_threshold:
        reasons.append("HIGH_GEOMETRY_COMPLEXITY")
    elif geometry_risk >= config.inspection_medium_geometry_risk_threshold:
        reasons.append("MEDIUM_GEOMETRY_COMPLEXITY")


def _inspection_reasons(row: pd.Series, config) -> str:
    """Return stable, pipe-separated explanations for one parcel score."""
    reasons: list[str] = []
    confidence = _normalize_row_confidence(
        row.get("prediction_rank_confidence_level", row[config.prediction_confidence_level_column])
    )
    reliability = _normalize_row_reliability(row[config.inspection_data_reliability_column])
    geometry_risk = row["geometry_risk"]
    label_status = row["label_prediction_status"]
    check_type = row["inspection_check_type"]

    _append_label_alignment_reasons(reasons, label_status, check_type, row["inspection_need"])
    _append_confidence_reasons(reasons, confidence)
    _append_data_reliability_reasons(reasons, reliability, config)
    _append_geometry_reasons(reasons, geometry_risk, config)

    if row.get("inspection_rule") == "CLASS_RELIABILITY_UNAVAILABLE":
        reasons.append("CLASS_RELIABILITY_UNAVAILABLE")
    if row.get("inspection_rule") == "NO_DECLARATION":
        reasons.append("NO_DECLARATION")
    if row.get("inspection_class_reliability_applied", False):
        level = str(row.get("prediction_class_reliability_level", "")).strip().upper()
        if level in {"LOW", "MEDIUM"}:
            reasons.append(f"{level}_CLASS_RELIABILITY")

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
        raise ValueError(f"Error: Inspection scoring requires columns: {missing}")

    # Work on a copy so prediction inputs and existing confidence columns are never mutated.
    result = data.copy()
    confidence = result.get("prediction_rank_confidence_level", result[config.prediction_confidence_level_column])
    confidence = confidence.astype("string").str.strip().str.upper()
    reliability = pd.to_numeric(result[config.inspection_data_reliability_column], errors="coerce")
    reliability = reliability.where(reliability.between(0.0, 1.0))
    complexity = pd.to_numeric(result[config.inspection_geometry_complexity_column], errors="coerce")
    complexity = complexity.where((complexity >= 0.0) & np.isfinite(complexity))

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
    result["inspection_priority_label"] = result["inspection_need"].map(INSPECTION_PRIORITY_LABELS)
    result["inspection_reasons"] = result.apply(_inspection_reasons, axis=1, config=config).astype("string")
    return result
