# Rank Consensus with Class Reliability Guard

## Purpose

The production classifier continues to choose its final class by soft voting,
including any frozen class-probability multipliers. Rank confidence is computed
after that decision and answers a narrower question: how strongly do the
selected ensemble members support the soft-voting class?

The user-facing confidence level is the lower of two components:

```text
final confidence = min(parcel rank confidence, class reliability)
LOW < MEDIUM < HIGH
```

## Parcel rank confidence

Only selected ensemble members are used. Each member's raw class probabilities
are converted to descending average ranks, so tied probabilities receive tied
average ranks. For `K` classes, rank `r` becomes a 0-100 Borda score:

```text
B(r) = 100 * (K - r) / (K - 1)
```

Scores are averaged equally across members. Rank confidence is evaluated for
the final soft-voting class, not used to replace it.

Default rules:

| Level | Required conditions |
|---|---|
| HIGH | Soft-voting class is the unique Borda winner; mean Borda >= 90; rank range <= 2; at least 3 models |
| MEDIUM | Soft-voting class is the unique Borda winner; mean Borda >= 75; rank range <= 4; at least 3 models |
| LOW | Everything else |

Winner ties, disagreement between soft voting and the Borda winner, and too few
models are explicitly retained as reasons for LOW confidence. Top-1/Top-2/Top-3
agreement, mean/median rank, standard deviation, IQR, margin, and runner-up may
remain as diagnostics but do not determine the level.

Rank calculations always use raw member probability ordering before
ensemble-level class multipliers. This keeps rank evidence independent of the
soft-voting adjustment while still explaining the actual adjusted prediction.

## Class reliability

Rank agreement cannot detect a class that all models predict consistently but
incorrectly. Step 4 therefore calculates precision for every predicted class
from selected-member out-of-fold (OOF) soft-voting predictions:

```text
precision(class c) = correct OOF predictions of c / all OOF predictions of c
```

OOF predictions use the same soft-voting logic and frozen class multipliers as
production. Training-set predictions, holdout correctness, and production rows
never fit or modify class reliability.

Default rules:

| Level | OOF predicted-class support and precision |
|---|---|
| HIGH | support >= 100 and precision >= 0.80 |
| MEDIUM | support >= 100 and precision >= 0.60 |
| LOW | support >= 100 and precision < 0.60 |
| LOW, invalid | support < 100 |

There is no class pooling, Top-1 binning, or neighboring-bin pooling.

## Frozen deployment contract

Step 4 stores the method, version, rank thresholds, minimum model count, and
complete per-class OOF reliability table under `confidence` in
`04_evaluate/selection_summary.json`. Step 5 applies that contract exactly and
does not recalculate thresholds or reliability from live configuration.

Selection summaries containing only the obsolete class x Top-1 calibration
must rerun Step 4 before the guarded method can be used in production.

## Review and outputs

When guarded confidence is active:

```text
prediction_needs_review =
    prediction_confidence_level == "LOW"
    or not prediction_confidence_valid
```

Maximum soft-voting probability is retained as a diagnostic and is used for
review only when rank confidence is disabled. Prediction artifacts retain the
final level, validity, reason, rank diagnostics, Borda winner and tie flag, and
the predicted class's frozen OOF precision, support, reliability level, and
validity.

Evaluation writes parcel-level OOF/holdout confidence, the OOF class reliability
table, confidence-level metrics for OOF and holdout, and per-class holdout
validation. The main acceptance check is whether holdout accuracy is strictly
ordered `HIGH > MEDIUM > LOW`; macro-F1 is reported but need not be monotonic.
