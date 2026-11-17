# Inspection workflow PNG generation

Generated using the built-in image-generation tool. Content checked against
`INSPECTION_PRIORITY_DECISION_TABLE.md` and
`src/ml_classification/step_05_predict/libraries/inspection_priority/scoring.py`.
The canonical image is `INSPECTION_WORKFLOW_DIAGRAM.png`; the Help copy is
`web/src/features/help/images/inspection_priority_workflow.png`.

## Final prompt

The EDA prerequisite was corrected with a targeted built-in image edit using this prompt:

> Edit the supplied Inspection priority workflow PNG. Correct only the top-right EDA input card to distinguish the required web-app pipeline from the optional outlier adjustment. Replace "EDA (optional)" with the heading "EDA". Inside that card put these three lines of body text exactly: "Required before ML in web app" then "Outlier flag joined" then "by parcel ID". Fit all text legibly within the card, adjusting font size slightly if needed. The EDA pipeline must NOT be called optional. Preserve everything else in the entire image: all other text, numbers, stages, geometry, arrows, colors, layout, resolution and all decision logic. In particular stage 4 "Outlier option enabled AND flag true?" remains conditional and unchanged: the outlier adjustment is optional, even though EDA execution is required before ML in the web app. No other changes.

The full generation prompt, incorporating that correction, follows.

Create a crisp professional workflow PNG for Parcel Studio's Inspection pipeline, portrait 1600x2200 or similar high resolution. White background, dark navy typography, teal numbered process cards, amber conditional branches, thin clean arrows. Flat technical flowchart, no illustrations or decoration. Large readable text, generous padding. Exact title: "Inspection priority workflow". Subtitle: "From parcel evidence to a traceable review queue". Content must follow this specification accurately, simplifying only layout, never meaning:

Top: three input cards merge into main flow. Card A "ML evidence" / "Prediction, declared class, rank confidence, frozen OOF class reliability". Card B "Parcel statistics" / "Observed-data reliability and geometry complexity". Card C "EDA" / "Required before ML in web app" / "Outlier flag joined by parcel ID". EDA execution is required before ML in the web app; only the outlier-based priority adjustment is optional.

1 "Check scoring prerequisites" / "Scoring enabled and required confidence, data and geometry columns present?" Side exit labelled "No" to "Save predictions without inspection scoring". Downward Yes arrow.

2 "Prepare comparable evidence" / "Compare declared and predicted classes. Calculate risks and run-specific reliability thresholds." Small note: "Percentile thresholds by default; absolute mode available."

3 "Assign base priority" / "Apply the 20-rule decision table: first matching rule wins." Within or immediately below this card three concise lines:
"Strong agreement -> lower priority"
"Strong conflict -> higher priority"
"Weak evidence -> Medium"
Separate side note attached to this stage:
"Fallbacks: invalid required evidence -> Unknown. Missing declaration or unavailable enabled class reliability -> Medium."
Small attached note: "Save base priority, rule ID and check type."

4 amber conditional diamond "Outlier option enabled AND flag true?" Yes branch to "Raise priority by one band" with note "Very High and Unknown stay unchanged". No branch to "Keep base priority". Both merge into next card.

5 "Record final decision" / "Final priority, original rule, reasons and outlier adjustment." Note "Outliers increase review priority; they do not prove a declaration conflict."

6 "Build the review queue" / "Preserve check type. NONE raised to Low-Medium or above enters evidence review. Queue by final priority, then inspection score within a band."

Bottom horizontal legend with seven colored chips, from green to red:
"Very Low", "Low", "Low-Medium", "Medium", "Medium-High", "High", "Very High".
Separate gray chip "Unknown = unscored".
Footer note, visually distinct: "The inspection score never decides the priority band; it orders parcels within a band."
Avoid any additional factual claims or numbers. Every arrow must have an unambiguous source and destination. No duplicate stages or misspellings.
