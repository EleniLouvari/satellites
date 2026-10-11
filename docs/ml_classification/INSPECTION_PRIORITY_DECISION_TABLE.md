# Inspection Priority Decision Table

Visual reference: [Full analytical workflow](<Full Workflow.png>), including
evidence calculations, validation branches, all 20 rules, and queue ordering.

## Policy and inputs

Step 5 applies one rule-based mechanism to five inputs: predicted-class OOF
reliability, declared-versus-predicted label agreement, parcel rank confidence,
data reliability, and geometry risk. It never caps rank confidence with class
reliability. Numeric uncertainty scores do not determine priority or queue order.

The seven ordered priorities are **Very Low, Low, Low–Medium, Medium,
Medium–High, High, Very High**. Medium means inconclusive evidence requiring
verification. Higher bands indicate increasingly convincing declaration conflicts;
lower bands indicate increasingly convincing agreement. These are operational
rules, not calibrated probabilities that a declaration is incorrect.

## Complete decision table

These rules apply with class reliability enabled and valid inputs. Conditions
within a row use AND. Any means any valid value. The first matching rule wins;
overlapping low-evidence rules always assign Medium. IDs are persisted in
`inspection_rule` so the decision can be traced to this table.

| Rule | Class reliability | Declared vs predicted | Rank confidence | Data reliability | Geometry risk | Inspection Priority Label |
|---|---|---|---|---|---|---|
| R01 | High | Same | High | > 0.8 | < 0.5 | Very Low |
| R02 | High | Same | High | > 0.8 | >= 0.5 | Low |
| R03 | Medium | Same | High | > 0.8 | < 0.5 | Low |
| R04 | Medium | Same | High | > 0.8 | >= 0.5 | Low–Medium |
| R05 | High | Same | Medium | > 0.8 | < 0.5 | Low–Medium |
| R06 | High | Same | Medium | > 0.8 | >= 0.5 | Medium |
| R07 | Medium | Same | Medium | > 0.8 | Any | Medium |
| R08 | Low | Same | Any | Any | Any | Medium |
| R09 | Any | Same | Low | Any | Any | Medium |
| R10 | Any | Same | Any | <= 0.8 | Any | Medium |
| R11 | High | Different | High | > 0.8 | < 0.5 | Very High |
| R12 | High | Different | High | > 0.8 | >= 0.5 | High |
| R13 | Medium | Different | High | > 0.8 | < 0.5 | High |
| R14 | Medium | Different | High | > 0.8 | >= 0.5 | Medium–High |
| R15 | High | Different | Medium | > 0.8 | < 0.5 | Medium–High |
| R16 | High | Different | Medium | > 0.8 | >= 0.5 | Medium |
| R17 | Medium | Different | Medium | > 0.8 | Any | Medium |
| R18 | Low | Different | Any | Any | Any | Medium |
| R19 | Any | Different | Low | Any | Any | Medium |
| R20 | Any | Different | Any | <= 0.8 | Any | Medium |

The table uses default thresholds. Data reliability must be **strictly greater
than** `inspection_medium_data_reliability_threshold` (default 0.8).
Geometry is favorable only **below** `inspection_medium_geometry_risk_threshold`
(default 0.5). Exactly 0.8 is not high data reliability; exactly 0.5 is in the
less favorable geometry band. These existing configuration settings remain
customizable; the comparison operators do not change.

## Disabled and unavailable inputs

- When class reliability is disabled, omit its condition: High rank + high data
  produces Very Low/Very High for Same/Different with favorable geometry and
  Low/High otherwise. Medium rank + high data + favorable geometry produces
  Low–Medium/Medium–High; other valid comparable cases produce Medium. The
  corresponding R01/R02/R05/R06/R09/R10 or R11/R12/R15/R16/R19/R20 identifiers
  are used with `inspection_class_reliability_applied=False`. Class evidence
  is retained unchanged, never relabeled High.
- Enabled reliability that is missing, unrecognized, or invalid due to
  insufficient OOF support produces Medium with
  `inspection_rule=CLASS_RELIABILITY_UNAVAILABLE`. It does not invalidate rank
  confidence. This differs from measured Low reliability in R08/R18.
- A missing declaration with valid prediction evidence produces Medium and
  `inspection_rule=NO_DECLARATION`. No conflict is inferred.
- Missing predictions, invalid/missing rank levels, explicitly invalid rank
  evidence, invalid data reliability (outside [0, 1]), or missing/negative/
  nonfinite geometry complexity produce `UNKNOWN` with
  `inspection_rule=REQUIRED_EVIDENCE_UNAVAILABLE`. Unknown is an unscored status,
  outside the seven priority bands.
- Entirely missing required confidence/data/geometry source columns skip the
  inspection layer with a warning and missing-column metadata.

## Confidence and class evidence

Rank confidence remains HIGH/MEDIUM/LOW from Borda agreement, mean Borda,
rank range, ties, and the minimum model count. Class reliability never changes
its level, validity, reason, or confidence-only review flag. See the
[confidence methodology](rank_based_ensemble_confidence_methodology.md).

Class reliability is learned only from training OOF predictions of the selected
strategy, using the predicted class's precision and predicted-class support:

| Class reliability | Default rule |
|---|---|
| HIGH | Support >= 100 and precision >= 0.80 |
| MEDIUM | Support >= 100 and precision >= 0.60 but < 0.80 |
| LOW, valid | Support >= 100 and precision < 0.60 |
| LOW, invalid | Support < 100; use the unavailable-evidence fallback |

Step 4 freezes this table and its enabled flag. Step 5 propagates the frozen
flag as `inspection_class_reliability_enabled`, overriding live configuration.
Direct dataframe scoring without this field uses `config.class_reliability_enabled`.

## Geometry and observation evidence

### 2) `data_reliability_score` (from parcel zonal-stat extraction)

This score measures how much of the parcel time-series comes from **original observed pixels** (not filled values).

- `expected_pixel_count = intersected_pixel_count * n_periods * n_variables`
- `observed_count = number of finite pixels in observed_mask`
- `data_reliability_score = observed_count / expected_pixel_count`

Also reported for diagnostics:
- `temporal_filled_ratio = temporal_filled_pixel_count / expected_pixel_count`
- `spatial_filled_ratio = spatial_filled_pixel_count / expected_pixel_count`

### 3) `geom_shape_complexity_score` (from parcel geometry metrics)

This score combines raster/geometry mismatch, boundary irregularity, and elongation using log penalties.

Base geometry metrics:
- `geom_interior_area_ratio = pixel_area / geom_area`
- `geom_compactness = 4πA / P²`
- `geom_perimeter_area_ratio = P / A`
- `geom_shape_index = P / (2*sqrt(πA))`
- `geom_elongation = long_side / short_side` of the minimum rotated rectangle

Penalties:
- `raster_agreement_penalty = |log(geom_interior_area_ratio)|`
- `compactness_penalty = max(0, -0.5*log(geom_compactness))`
- `normalized_perimeter_area_ratio = geom_perimeter_area_ratio / (2*sqrt(π/geom_area))`
- `perimeter_penalty = max(0, log(normalized_perimeter_area_ratio))`
- `shape_index_penalty = max(0, log(geom_shape_index))`
- `elongation_penalty = max(0, log(geom_elongation))`

Then:
- `boundary_penalty = mean(compactness_penalty, perimeter_penalty, shape_index_penalty)`
- `geom_shape_complexity_score = mean(raster_agreement_penalty, boundary_penalty, elongation_penalty)`

Higher values indicate more complex/irregular parcel geometry.

## Outputs and diagnostic score

- `inspection_need`: machine-readable priority (`VERY_LOW`, `LOW`, `LOW_MEDIUM`,
  `MEDIUM`, `MEDIUM_HIGH`, `HIGH`, `VERY_HIGH`) or unscored `UNKNOWN`.
- `inspection_priority_label`: readable label from the decision table.
- `inspection_rule`: first matching rule ID or explicit fallback identifier.
- `inspection_class_reliability_applied`: whether valid enabled class evidence
  participated in a comparable, valid policy decision.
- `inspection_check_type`: `NONE` for Very Low/Low; `DECLARATION_CONFLICT` for
  Different labels with Medium–High/High/Very High; `INSUFFICIENT_EO_EVIDENCE`
  for remaining cases. The last name is retained for compatibility and includes
  model/class uncertainty and missing evidence as well as EO quality issues.
- `inspection_reasons`: pipe-separated diagnostic reasons, including
  MEDIUM/LOW_CLASS_RELIABILITY, unavailable reliability, and missing declarations.

The former `MEDIUM_UNCERTAIN` category is replaced by Medium with explicit reasons.

For continuity, the numeric score and score-only base category remain diagnostics:

```text
confidence_risk = HIGH: 0.0, MEDIUM: 0.5, LOW: 1.0
inspection_score = 100 * (0.60*confidence_risk
                        + 0.25*(1 - data_reliability_score)
                        + 0.15*geometry_risk)
inspection_need_base = HIGH if score >= 50, MEDIUM if >= 25, otherwise LOW
```

Weights and score thresholds affect only these diagnostic fields. Neither the
score nor `inspection_need_base` enters any inspection-priority rule.

The inspection queue includes `inspection_check_type != NONE`, ordered Very
High, High, Medium–High, Medium, Low–Medium, Low, Very Low, then Unknown. Ties
preserve input order; diagnostic scores do not break ties. Exported queue rows
include the class evidence, readable label, matched rule, and reasons.

## Updating existing runs

Existing outputs remain unchanged until recalculated. With saved member
probabilities and a supported frozen OOF contract, call
`pipeline.recalculate_prediction_quality()` to apply rank-only confidence and
these rules without retraining. Legacy guard contracts are read as independent
class evidence under the new policy; they no longer cap confidence. Rerun Step 4
first to refresh evaluation confidence artifacts and write the version 3 contract,
or to change the frozen class-reliability flag. Obsolete class x Top-1 contracts
still require Step 4. Recreating HTML reports alone does not recalculate outputs.
