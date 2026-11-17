# Workflow graphic edit prompts

The workflow PNGs were updated using the built-in image generation tool in edit mode.
The scoring implementation and [decision table](INSPECTION_PRIORITY_DECISION_TABLE.md)
remain the authoritative policy; the images are visual companions.

## Full Workflow.png

Edit this existing technical workflow infographic, preserving all other content and the detailed original layout/style and readable high resolution text. Add the EDA is_outlier input and an explicit stage BETWEEN the base 20-rule matrix (section 6) and final outputs/queue (section 7). Expand the canvas vertically if necessary; do not squeeze or remove original sections. New stage title: '7. EDA outlier adjustment (optional)'. Exact policy: 'inspection_include_outlier=True AND is_outlier=True: raise one priority band'. Show 'Very Low → Low → Low–Medium → Medium → Medium–High → High → Very High'. Add 'Very High stays Very High; Unknown stays Unknown. Disabled, false, missing or invalid flag: unchanged.' Add 'Preserve inspection_rule; record inspection_need_before_outlier and inspection_outlier_adjusted. Reason: OUTLIER_PRIORITY_INCREASE only when raised.' Add 'Preserve check type; NONE raised to Low–Medium becomes INSUFFICIENT_EO_EVIDENCE. Outliers alone do not establish a declaration conflict.' Show a distinct input arrow 'EDA is_outlier: join by parcel ID; metadata, not a model feature'. Renumber old section 7 to 8 and old 8 to 9. Final queue sorts FINAL post-outlier priority with stable input-order ties. Diagnostic score is unaffected by is_outlier. Correct the section 5 false statement 'Declared labels are always available': missing declaration + valid evidence → Medium / NO_DECLARATION → outlier adjustment. Enabled unavailable class evidence → Medium → outlier adjustment. No change to rank confidence or class reliability.

Final corrections required exactly ten paired base-rule rows (twenty Same/Different rules)
matching the decision table, preserved check types in the outputs stage, and the heading
`6. Base decision table – 10 paired rows / 20 rules`.

### November 2026 patch (data-reliability calibration and matching declarations)

These four regions were repainted directly in the PNG rather than regenerated, so the rest
of the image is unchanged. Apply the same content if the image is regenerated:

- Section 3, policy box: `HIGH data for policy: d > T`, `T = min(0.80, 25th percentile of d)`,
  `(d = T is NOT high; absolute mode: T = 0.80)`.
- Section 6, data column: header `Data d vs T`; rows `> T` and the last row `<= T`.
- Section 7, check-type note: `NONE raised to Low–Medium or above becomes
  INSUFFICIENT_EO_EVIDENCE.`
- Section 8, first box: `Very Low/Low, and SAME at Low–Medium/Medium → NONE`;
  `(inspection_check_matching_declarations=True queues SAME)`;
  `DIFFERENT at Medium–High or above → DECLARATION_CONFLICT`;
  `Other cases → INSUFFICIENT_EO_EVIDENCE`;
  `Outlier raise of NONE: evidence review, never a conflict.`

## OOF and Class Reliability Workflow.png

Edit the provided OOF → CLASS RELIABILITY technical infographic. Preserve every existing section, formula, numerical example, layout, text and style except the small rightmost box in section 7 'Freeze, validate, and apply'. Expand that section slightly if required for readable text. Replace that box with: 'Step 5 inspection: base rules use class reliability (when enabled), label agreement, rank confidence, data and geometry. Then optional EDA is_outlier adjustment: inspection_include_outlier=True + is_outlier=True raises one priority band, capped at Very High. Unknown unchanged. Neuro enables this option.' Add a separate small footer note in section 7: 'EDA flag is parcel metadata, not fitted OOF class evidence. Outlier adjustment does not change rank confidence, class reliability, or diagnostic score.' Keep the OOF fitting workflow unchanged and preserve all original accuracy and support thresholds. Final output high resolution with text readable.
