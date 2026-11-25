"""Grouped, documented ML classification settings and strict web input validation."""

import json
import warnings
from dataclasses import MISSING, fields
from importlib.util import find_spec

from ml_classification.shared.config.config import ClassificationPipelineConfig
from ml_classification.shared.config.tune_params import _default_tune_params
from parcel_app.settings import get_settings

# Chosen by the server, never by the browser. feature_columns comes from the column picker
# and tune_params is edited as one "tune_params.<model>" field per model.
MANAGED = {
    "project_dir": "A separate directory is allocated for every ML run.",
    "reset_project_dir_on_run_check": False,
    "fail_on_cleanup_error": False,
    "force_retrain_models": False,
    "open_html_report": False,
    "output_schema_version": None,
    "log_filename": None,
    "step_names": None,
}
SEPARATE = {"feature_columns", "tune_params"}
TUNE_PREFIX = "tune_params."
# Web-only settings that shape the input table before the pipeline receives it.
WEB_DEFAULTS = {"exclude_target_values": [], "add_centroid_features": False}
CENTROID_COLUMNS = ("centroid_x", "centroid_y")
OUTLIER_COLUMN = "is_outlier"
# Fields whose library default is None, so their type cannot be read from the default.
NULLABLE = {
    "target_column": "string",
    "spatial_interpolation_method": "string",
    "spatial_interpolation_max_distance_in_meters": "number",
    "spatial_interpolation_variogram_max_distance_in_meters": "number",
    "top_voting_models": "number",
    "lstm_temporal_statistics": "array",
    "lstm_temporal_frequency": "string",
    "prediction_column": "string",
    "prediction_filled_column": "string",
}
SCORERS = ["f1_macro", "balanced_accuracy", "f1_weighted", "accuracy"]
CHOICES = {
    "float_dtype": ["float32", "float64"],
    "spatial_interpolation_method": [None, "nearest", "idw", "kriging"],
    "spatial_split_method": ["by_row", "by_group"],
    "cv_validation_method": ["disjoint", "overlapping"],
    "scoring_primary": SCORERS,
    "scoring_secondary": SCORERS,
    "cv_ranking_method": ["score_minus_std", "mean_score"],
    "label_balancing_method": ["none", "random_oversample", "smote"],
    "selection_type": ["soft_voting", "single_model"],
    "lstm_temporal_frequency": [None, "monthly"],
    "lstm_class_balancing_method": ["balanced_class_weight", "none"],
    "inspection_data_reliability_threshold_mode": ["absolute", "percentile"],
    "inspection_geometry_threshold_mode": ["absolute", "percentile"],
}
def _when(rule, *names):
    """Give several dependent fields the same parent rule."""
    return {name: rule for name in names}


# A dependent field is greyed out, with the reason shown, until its parent has one of these values.
ENABLED_WHEN = {
    **_when({"apply_iqr": [True]}, "iqr_lower_quantile", "iqr_upper_quantile", "iqr_multiplier"),
    "spatial_interpolation_method": {"fill_nulls": [True]},
    "spatial_interpolation_max_distance_in_meters": {"fill_nulls": [True], "spatial_interpolation_method": ["nearest", "idw"]},
    **_when(
        {"fill_nulls": [True], "spatial_interpolation_method": ["kriging"]},
        "spatial_interpolation_variogram_lags",
        "spatial_interpolation_variogram_max_distance_in_meters",
    ),
    **_when({"spatial_split": [True]}, "spatial_split_method", "spatial_split_grid_size"),
    "smote_k_neighbors": {"label_balancing_method": ["smote"]},
    "top_voting_models": {"selection_type": ["soft_voting"]},
    **_when(
        {"optimize_class_probabilities": [True]},
        "probability_multiplier_grid",
        "probability_optimization_iterations",
        "probability_optimization_max_accuracy_drop",
    ),
    **_when(
        {"rank_confidence_enabled": [True]},
        "rank_confidence_minimum_models",
        "rank_confidence_high_min_borda",
        "rank_confidence_high_max_range",
        "rank_confidence_medium_min_borda",
        "rank_confidence_medium_max_range",
    ),
    **_when(
        {"class_reliability_enabled": [True]},
        "class_reliability_minimum_oof_support",
        "class_reliability_high_min_precision",
        "class_reliability_medium_min_precision",
    ),
    **_when(
        {"inspection_scoring_enabled": [True]},
        "inspection_include_outlier",
        "inspection_check_matching_declarations",
        "inspection_data_reliability_column",
        "inspection_geometry_complexity_column",
        "inspection_model_weight",
        "inspection_data_weight",
        "inspection_geometry_weight",
        "inspection_data_reliability_threshold_mode",
        "inspection_low_data_reliability_threshold",
        "inspection_medium_data_reliability_threshold",
        "inspection_geometry_threshold_mode",
        "inspection_medium_geometry_complexity_threshold",
        "inspection_high_geometry_complexity_threshold",
        "inspection_medium_score_threshold",
        "inspection_high_score_threshold",
    ),
    **_when(
        {"inspection_scoring_enabled": [True], "inspection_geometry_threshold_mode": ["percentile"]},
        "inspection_medium_geometry_risk_threshold",
        "inspection_high_geometry_risk_threshold",
    ),
    **_when(
        {"inspection_scoring_enabled": [True], "inspection_data_reliability_threshold_mode": ["percentile"]},
        "inspection_low_data_reliability_percentile",
        "inspection_medium_data_reliability_percentile",
    ),
    "shap_sample_size": {"interpretability_include_shap": [True]},
}

# (group id, title, what the group is for, {field: what the field does}). Order is the form order.
GROUPS = (
    (
        "data",
        "Target and parcels",
        "What the models predict and which parcels take part. Feature columns are chosen in step 2 above.",
        {
            "target_column": "Column holding the class to predict, for example the crop category. Parcels with an empty "
            "value are not used for training; they only receive a prediction.",
            "id_column": "Column that identifies each parcel in the checks, reports and prediction table.",
            "exclude_target_values": "Classes to leave out completely, separated by commas (9999, 8888) or as a JSON "
            "list ([9999, 8888]). Parcels with these target values are neither trained on nor predicted.",
            "exclude_outliers_from_training": "Leave parcels the selected EDA analysis flagged as outliers "
            "(is_outlier = true) out of the train, validation and test sets and out of the final refit. They are still "
            "predicted by the final model and appear in the prediction table with excluded_from_training = true.",
            "add_centroid_features": "Add each parcel's centroid coordinates (centroid_x, centroid_y, in metres) as two "
            "extra features, so models can learn regional patterns. Recalculates existing coordinates; "
            "coordinates already selected as features are included only once.",
            "enforce_unique_ids": "Stop the run when two rows share the same ID. Keep enabled: the pipeline expects one "
            "row per parcel.",
            "float_dtype": "Numeric precision used for the features. float32 halves memory use; float64 keeps full precision.",
        },
    ),
    (
        "quality",
        "Data cleaning before training",
        "Optional outlier clipping and gap filling applied to the feature values during the Check step.",
        {
            "apply_iqr": "Treat feature values outside the IQR fences as missing before training. Unless they are "
            "filled from neighbours, each model fills them with the training median; the parcel stays in training.",
            "iqr_lower_quantile": "Lower quantile of the IQR range (0.25 = first quartile).",
            "iqr_upper_quantile": "Upper quantile of the IQR range (0.75 = third quartile).",
            "iqr_multiplier": "How many IQRs beyond the quantiles a value may lie before it counts as an outlier. "
            "Larger values remove fewer points.",
            "fill_nulls": "Fill missing feature values from neighbouring parcels. When disabled, parcels with values "
            "missing in the input are left out of training but are still predicted; values removed by the IQR "
            "filter are filled by the models instead.",
            "spatial_interpolation_method": "How gaps are filled: nearest parcel, inverse distance weighting (idw) or kriging.",
            "spatial_interpolation_max_distance_in_meters": "Farthest neighbour used for filling. Required for idw; "
            "blank means no limit for nearest.",
            "spatial_interpolation_variogram_lags": "Number of distance bins used to fit the kriging variogram.",
            "spatial_interpolation_variogram_max_distance_in_meters": "Largest parcel distance considered by the kriging "
            "variogram. Required for kriging.",
        },
    ),
    (
        "split",
        "Train/test split and cross-validation",
        "How labelled parcels are divided to train the models and to measure their accuracy honestly.",
        {
            "test_size": "Share of labelled parcels held out as the final test set (0.2 = 20%).",
            "cv_folds": "Number of cross-validation folds used to compare models. Every class needs at least this many parcels.",
            "cv_validation_method": "disjoint validates every parcel exactly once; overlapping draws each validation fold "
            "independently, so a parcel may be validated in several folds or in none.",
            "random_state": "Seed for splits, searches and models. The same seed and settings reproduce the same run.",
            "spatial_split": "Split by location instead of at random, so nearby parcels do not leak between train and test.",
            "spatial_split_method": "by_row spreads individual parcels across the area; by_group holds out complete grid cells.",
            "spatial_split_grid_size": "Number of grid cells per side used for the spatial split (30 = a 30 × 30 grid).",
        },
    ),
    (
        "models",
        "Models and hyperparameter search",
        "Which algorithms are trained and how their settings are searched and ranked.",
        {
            "selected_models": "Algorithms to train. More models take longer; a model whose library is not installed "
            "on the server is skipped.",
            "max_search_candidates": "Hyperparameter combinations tried per model, sampled from its grid. "
            "Higher is slower and may find better settings.",
            "n_jobs": "CPU cores used in parallel for the search, up to the server limit.",
            "scoring_primary": "Metric used to tune and rank the models. f1_macro weights every class equally.",
            "scoring_secondary": "Additional metrics calculated and reported for every model.",
            "cv_ranking_method": "score_minus_std prefers models that are stable across folds; mean_score ranks by the average only.",
        },
    ),
    (
        "balance",
        "Class imbalance",
        "How classes with few parcels are handled during training.",
        {
            "label_balancing_method": "none keeps the data as is; random_oversample repeats parcels of rare classes; "
            "smote creates synthetic ones.",
            "smote_k_neighbors": "Neighbours SMOTE uses to create a synthetic parcel. Must be smaller than the rarest class.",
            "rare_category_min_frequency": "For text (categorical) features: values seen fewer times than this are grouped "
            "as infrequent. A value below 1 is read as a share of the rows.",
        },
    ),
    (
        "ensemble",
        "Final model and probability tuning",
        "How the final predictor is built from the trained models.",
        {
            "selection_type": "soft_voting averages the class probabilities of the best models; single_model keeps only the best one.",
            "top_voting_models": "How many of the best-ranked models vote, at most the number of selected "
            "models. Blank uses every trained model.",
            "optimize_class_probabilities": "Rescale class probabilities after training to raise macro F1, which helps rare classes.",
            "probability_multiplier_grid": "Multipliers tried for each class probability, as a JSON list.",
            "probability_optimization_iterations": "Number of passes over the classes while searching the multipliers.",
            "probability_optimization_max_accuracy_drop": "Largest accuracy loss accepted in exchange for a better "
            "macro F1 (0.02 = two percentage points).",
        },
    ),
    (
        "confidence",
        "Prediction confidence",
        "How each prediction is graded HIGH, MEDIUM or LOW confidence, and how reliable each predicted class is.",
        {
            "prediction_confidence_threshold": "Review cut-off for the highest soft-voting class probability (0-1). "
            "Applied when rank confidence is disabled. With fewer voting models than the rank minimum it is also "
            "the MEDIUM limit of the probability grade.",
            "prediction_high_confidence_threshold": "With fewer voting models than the rank minimum, the mean "
            "probability of the predicted class (0-1) from which a prediction is HIGH confidence.",
            "rank_confidence_enabled": "Grade every prediction by how strongly the voting models agree on its class, "
            "using the Borda score described above.",
            "rank_confidence_minimum_models": "Fewest voting models needed for a Borda grade. With fewer models the "
            "grade comes from the predicted-class probability instead (see Maximum probability check).",
            "rank_confidence_high_min_borda": "Lowest mean Borda score (0-100) of the predicted class that still counts as HIGH.",
            "rank_confidence_high_max_range": "Largest rank range (worst minus best rank of the predicted class across "
            "the models) still allowed for HIGH. 0 means every model ranked it the same.",
            "rank_confidence_medium_min_borda": "Lowest mean Borda score (0-100) that still counts as MEDIUM. "
            "Must not exceed the HIGH value.",
            "rank_confidence_medium_max_range": "Largest rank range still allowed for MEDIUM. Must be at least the HIGH value.",
            "class_reliability_enabled": "Measure the out-of-fold precision of every predicted class and use it in the "
            "inspection priority rules.",
            "class_reliability_minimum_oof_support": "Out-of-fold predictions of a class needed before its precision is "
            "trusted. Classes with fewer are marked unavailable, not LOW.",
            "class_reliability_high_min_precision": "Out-of-fold precision (0-1) from which a class is HIGH reliability.",
            "class_reliability_medium_min_precision": "Out-of-fold precision (0-1) from which a class is MEDIUM "
            "reliability; below it the class is LOW.",
        },
    ),
    (
        "inspection",
        "Inspection priority",
        ("Ranks parcels for field inspection from five pieces of evidence: declared versus predicted class, "
        "rank confidence, class reliability, data reliability and geometry risk."),
        {
            "inspection_scoring_enabled": "Calculate an inspection priority for every predicted parcel.",
            "inspection_include_outlier": "After the rule table, raise the priority one band (for example Low to "
            "Low–Medium) for parcels the selected EDA analysis flagged as outliers. Very High stays Very High.",
            "inspection_check_matching_declarations": "Off (default): a parcel whose declaration matches the "
            "prediction is queued only when an outlier raises it. On: matching parcels with Low–Medium or Medium "
            "priority are also queued as INSUFFICIENT_EO_EVIDENCE, which usually queues most parcels.",
            "inspection_data_reliability_column": "Column with the 0-1 data reliability score: the share of the parcel's "
            "pixel values that were observed rather than filled. Inspection is skipped when the column is absent.",
            "inspection_geometry_complexity_column": "Column with the shape complexity score (0 = simple compact shape, "
            "larger = irregular or elongated). Inspection is skipped when the column is absent.",
            "inspection_model_weight": "Weight w_model of the confidence risk in the inspection score, which orders "
            "parcels within a priority band of the queue.",
            "inspection_data_weight": "Weight w_data of the data risk in the inspection score.",
            "inspection_geometry_weight": "Weight w_geometry of the geometry risk. The three weights must sum to 1.",
            "inspection_data_reliability_threshold_mode": "absolute (default) applies the two thresholds as written. "
            "percentile lowers each data reliability threshold to the matching percentile of the predicted parcels "
            "when that percentile is lower, so normal cloud gaps do not mark most parcels as weak evidence.",
            "inspection_low_data_reliability_threshold": "Reason codes only: data reliability below this value is "
            "reported as LOW_DATA_RELIABILITY. In percentile mode this is the upper limit of the applied value.",
            "inspection_medium_data_reliability_threshold": "Rule table and reason codes: data reliability must be "
            "strictly above this value to count as high. At or below it, rules R10/R20 give Medium; between the low "
            "threshold and this one the reason is MEDIUM_DATA_RELIABILITY. In percentile mode this is the upper "
            "limit of the applied value.",
            "inspection_low_data_reliability_percentile": "Percentile mode only: share of predicted parcels (0-1) "
            "allowed below the low threshold. 0.05 reports about the weakest 5% as LOW_DATA_RELIABILITY.",
            "inspection_medium_data_reliability_percentile": "Percentile mode only: share of predicted parcels (0-1) "
            "allowed at or below the medium threshold. 0.25 treats about the best 75% as high data reliability.",
            "inspection_geometry_threshold_mode": "absolute (default): only the complexity score thresholds apply. "
            "percentile: a parcel's geometry counts as medium or high complexity only when both its percentile and "
            "its complexity score pass the thresholds, so a run of simple parcels is not penalised for ranking alone.",
            "inspection_medium_geometry_risk_threshold": "Percentile mode only: geometry risk (0-1 percentile) from "
            "which a parcel can be medium complexity. Medium or high complexity drops a parcel one band in the rules "
            "and adds the reason MEDIUM_GEOMETRY_COMPLEXITY.",
            "inspection_high_geometry_risk_threshold": "Percentile mode only: geometry risk from which a parcel can be "
            "reported as HIGH_GEOMETRY_COMPLEXITY.",
            "inspection_medium_geometry_complexity_threshold": "Shape complexity score from which a parcel can be medium "
            "complexity. Compact parcels score about 0.1-0.3, 2:1 strips 0.35-0.5, 3:1 strips 0.5-0.7. Set 0 to use "
            "the percentile alone.",
            "inspection_high_geometry_complexity_threshold": "Shape complexity score from which a parcel can be high "
            "complexity (reason code only). 5:1 strips score about 0.75-0.95.",
            "inspection_medium_score_threshold": "Diagnostic score only: a score at or above this value (0-100) gives "
            "inspection_need_base MEDIUM.",
            "inspection_high_score_threshold": "Diagnostic score only: a score at or above this value (0-100) gives "
            "inspection_need_base HIGH.",
        },
    ),
    (
        "interpretability",
        "Explanations and report plots",
        "How much explanation the reports contain. These settings do not change the predictions.",
        {
            "interpretability_top_models": "Number of best models that receive feature-importance analysis. 0 skips it.",
            "interpretability_include_shap": "Also calculate SHAP values. More detailed, and considerably slower.",
            "shap_sample_size": "Parcels sampled for the SHAP calculation.",
            "feature_importance_top_n": "Number of most important features shown per model.",
            "max_distribution_features": "Features shown in the train/test distribution comparison plots.",
            "max_features_per_plot": "Largest number of features drawn in one plot.",
            "max_map_geometries": "Largest number of parcels drawn on report maps; larger sets are sampled.",
        },
    ),
    (
        "lstm",
        "LSTM time-series model",
        "Used only when keras_lstm is selected. It reads the dated feature columns as a time series.",
        {
            "lstm_temporal_statistics": "Statistics of the dated features that form the sequence, as a JSON list "
            'such as ["median"] or ["median", "sd", "range"].',
            "lstm_require_complete_timesteps": "Use only dates that have every selected statistic and index.",
            "lstm_temporal_frequency": "monthly groups the dates into months; None keeps each acquisition date.",
            "lstm_min_timesteps": "Fewest time steps required to train the LSTM.",
            "lstm_class_balancing_method": "balanced_class_weight gives rare classes more weight; none treats all parcels equally.",
        },
    ),
    (
        "output",
        "Output column names",
        "Names of the columns added to the prediction table.",
        {
            "prediction_column": "Column for the predicted class. Blank uses <target>_prediction.",
            "prediction_filled_column": "Column holding the declared class where known and the prediction elsewhere. "
            "Blank uses <target>_filled.",
            "probability_prefix": "Prefix of the per-class probability columns, for example probability_wheat.",
            "prediction_confidence_column": "Column for the highest class probability of each prediction.",
            "prediction_confidence_level_column": "Column for the HIGH, MEDIUM or LOW confidence grade.",
            "prediction_review_column": "Column marking predictions that need review.",
        },
    ),
)
# Optional sections inside a group: (id, title, explanation paragraphs, formula or None, member fields).
# Every member must belong to the enclosing group; fields of a group outside any section are shown first.
SUBGROUPS = {
    "confidence": (
        (
            "probability",
            "Maximum probability check",
            (
                ("The simplest confidence measure: the highest class probability of the final soft-voting model, "
                "saved in the prediction_confidence column for every parcel."),
                ("It flags a prediction for review only when rank confidence is disabled. With rank confidence "
                "enabled, the review flag comes from the Borda grade below and this value is kept as a diagnostic."),
                ("When fewer models vote than the rank minimum (for example a single-model run), rank agreement "
                "cannot be measured. The grade then comes from the mean probability of the predicted class: HIGH "
                "from the high threshold, MEDIUM from the review cut-off, otherwise LOW. The reason codes start "
                "with probability_fallback."),
            ),
            None,
            ("prediction_confidence_threshold", "prediction_high_confidence_threshold"),
        ),
        (
            "borda",
            "Rank confidence (Borda score)",
            (
                ("The Borda score measures how strongly the voting models agree on the predicted class. Each model "
                "ranks all K classes by its own probabilities (rank 1 = most likely; tied probabilities share an "
                "average rank) and every rank is converted into points: first place earns 100, last place 0. "
                "The points of the predicted class are averaged over the models. Raw model probabilities are used, "
                "before any class multipliers, so the score stays independent of probability tuning."),
                ("Rank range is the worst minus the best rank the models gave the predicted class; 0 means "
                "they all ranked it the same. A grade is HIGH or MEDIUM only when the predicted class is also the "
                "unique Borda winner (no tie, no other class with more points). Everything else is LOW. With fewer "
                "models than the minimum, the probability grade above is used instead."),
                ("Output: prediction_confidence_level (HIGH, MEDIUM or LOW) with the mean Borda, rank range and the "
                "reason for the grade. LOW or invalid predictions are flagged for review, and the grade is an input "
                "of the inspection priority rules and of the diagnostic inspection score."),
            ),
            ("Borda points  B(r) = 100 × (K − r) / (K − 1)\n"
            "Mean Borda    = average of B(r) for the predicted class over the voting models\n"
            "Rank range    = max(r) − min(r) of the predicted class over the voting models\n"
            "\n"
            "HIGH    unique winner · mean Borda ≥ high min · range ≤ high max · models ≥ minimum\n"
            "MEDIUM  unique winner · mean Borda ≥ medium min · range ≤ medium max · models ≥ minimum\n"
            "LOW     everything else\n"
            "\n"
            "Example  K = 10 classes, 3 models rank the class 1, 1, 2\n"
            "         B = 100, 100, 88.9 → mean 96.3, range 1 → HIGH (defaults 90 and 2)"),
            (
                "rank_confidence_enabled",
                "rank_confidence_minimum_models",
                "rank_confidence_high_min_borda",
                "rank_confidence_high_max_range",
                "rank_confidence_medium_min_borda",
                "rank_confidence_medium_max_range",
            ),
        ),
        (
            "class_reliability",
            "Class reliability",
            (
                ("Rank agreement cannot detect a class that every model predicts consistently but wrongly. Class "
                "reliability measures how often each predicted class was actually right. During the Evaluate step "
                "the selected models predict the training parcels out-of-fold (each parcel by models that did not "
                "train on it), and the precision of every predicted class is calculated. The table is frozen in "
                "Evaluate and reused by Predict."),
                ("It never changes the confidence grade or the review flag. It is used only in the inspection "
                "priority rules: HIGH allows the strongest bands (Very Low or Very High), MEDIUM gives one band "
                "less, LOW gives Medium (inconclusive), and a class below the support minimum gives Medium with "
                "rule CLASS_RELIABILITY_UNAVAILABLE. When disabled, the rules ignore this condition."),
                ("Output: prediction_class_reliability_level, precision, support and validity for the predicted "
                "class of every parcel."),
            ),
            ("precision(c) = correct out-of-fold predictions of c / all out-of-fold predictions of c\n"
            "support(c)   = all out-of-fold predictions of c\n"
            "\n"
            "HIGH    support ≥ minimum · precision ≥ high min\n"
            "MEDIUM  support ≥ minimum · precision ≥ medium min\n"
            "LOW     support ≥ minimum · precision < medium min\n"
            "n/a     support < minimum (treated as unavailable evidence)"),
            (
                "class_reliability_enabled",
                "class_reliability_minimum_oof_support",
                "class_reliability_high_min_precision",
                "class_reliability_medium_min_precision",
            ),
        ),
    ),
    "inspection": (
        (
            "inputs",
            "Switch and input columns",
            (
                ("The priority comes from a table of 20 rules (R01–R20). It compares the declared class with the "
                "prediction and checks the rank confidence and class reliability (Prediction confidence group), the "
                "data reliability and the geometry risk. Agreement backed by strong evidence gives Very Low "
                "or Low; disagreement backed by strong evidence gives High or Very High; weak evidence gives Medium."),
                ("Parcels without a declared class get Medium (rule NO_DECLARATION). Parcels with missing or "
                "invalid evidence get Unknown. The optional EDA outlier flag is applied last."),
                ("The inspection queue holds parcels whose inspection_check_type is not NONE: declaration conflicts "
                "(prediction differs, strong evidence) and insufficient evidence (prediction differs with weak "
                "evidence, no declaration, missing evidence, or an outlier increase). A matching declaration is "
                "not queued unless the matching-declarations option is on."),
                ("Output: inspection_need, inspection_priority_label, the matched inspection_rule, "
                "inspection_check_type and pipe-separated inspection_reasons for every parcel."),
            ),
            None,
            (
                "inspection_scoring_enabled",
                "inspection_include_outlier",
                "inspection_check_matching_declarations",
                "inspection_data_reliability_column",
                "inspection_geometry_complexity_column",
            ),
        ),
        (
            "weights",
            "Score weights (queue order within a band)",
            (
                ("The three weights combine three risks, each scaled 0 (no concern) to 1 (strong concern), into one "
                "0-100 inspection_score. Geometry risk is the percentile of the parcel's shape complexity among all "
                "predicted parcels, so 0.9 means more complex than 90% of them."),
                ("Output: inspection_score (higher = weaker evidence) and inspection_need_base (LOW, MEDIUM or HIGH "
                "from the score thresholds below). The score does not depend on the declared class and does not set "
                "the priority band, but it orders the queue inside each band: parcels with the higher score are "
                "listed first, so the weights decide which parcels of equal priority are visited first."),
            ),
            ("confidence_risk = 0 (HIGH) · 0.5 (MEDIUM) · 1 (LOW)   from the rank confidence grade\n"
            "data_risk       = 1 − data reliability\n"
            "geometry_risk   = percentile of shape complexity (0-1)\n"
            "\n"
            "inspection_score = 100 × (w_model × confidence_risk\n"
            "                        + w_data × data_risk\n"
            "                        + w_geometry × geometry_risk)\n"
            "\n"
            "Example  MEDIUM confidence, reliability 0.9, geometry percentile 0.4, weights 0.60/0.25/0.15\n"
            "         100 × (0.60 × 0.5 + 0.25 × 0.1 + 0.15 × 0.4) = 38.5 → MEDIUM (25 ≤ score < 50)"),
            ("inspection_model_weight", "inspection_data_weight", "inspection_geometry_weight"),
        ),
        (
            "thresholds",
            "Thresholds",
            (
                ("Data reliability and geometry have separate threshold modes, explained below. Their medium "
                "boundaries affect the inspection priority through the rule table; low reliability and high geometry "
                "boundaries add explanation codes to inspection_reasons."),
                "The score thresholds below only set inspection_need_base. They do not set the final priority band.",
            ),
            "Score bands: LOW below medium; MEDIUM from medium up to high; HIGH at or above high.",
            ("inspection_medium_score_threshold", "inspection_high_score_threshold"),
        ),
        (
            "data_reliability",
            "Data reliability",
            (
                ("Data reliability is the share of observed rather than filled pixel values (0-1); higher is better. "
                "The rule table treats reliability as high only when it is strictly above the applied medium cutoff."),
                ("Absolute (default): use the low and medium score cutoffs exactly as entered, for the same standard across "
                "runs. For example, a medium cutoff of 0.80 requires more than 80% observed values for high reliability."),
                ("Percentile: adapt each cutoff to the valid reliability scores of all predicted parcels "
                "in this run. Use the smaller of the entered cutoff and the score at the chosen percentile. "
                "A percentile of 0.25 means the 25th percentile (the lower quarter), not a reliability score of 0.25. "
                "This can lower a cutoff in a cloudy season, but never raises it above the entered value."),
                ("Example: with a medium cutoff of 0.80 and a 25th-percentile score of 0.72, percentile mode uses "
                "0.72; absolute mode uses 0.80. A parcel scoring 0.75 therefore has high reliability only in "
                "percentile mode. Percentiles do not guarantee an exact share in each band because of ties and "
                "the fixed cutoff cap. Applied cutoffs are saved with predictions and in predict_summary.json."),
            ),
            ("Absolute: applied cutoff = entered cutoff\n"
            "Percentile: applied cutoff = min(entered cutoff, score at chosen percentile)\n"
            "High data reliability: score > applied medium cutoff"),
            (
                "inspection_data_reliability_threshold_mode",
                "inspection_low_data_reliability_threshold",
                "inspection_medium_data_reliability_threshold",
                "inspection_low_data_reliability_percentile",
                "inspection_medium_data_reliability_percentile",
            ),
        ),
        (
            "geometry",
            "Geometry",
            (
                ("Geometry complexity measures shape irregularity or elongation; higher means more complex. "
                "Geometry risk is its percentile rank among all predicted parcels in this run (0-1). "
                "A rank of 0.75 is around the 75th percentile; it is not a complexity score of 0.75."),
                ("Absolute (default): compare only the complexity score with the medium and high complexity cutoffs. "
                "Percentile rank cutoffs are ignored, so the same shape receives the same level across runs."),
                ("Percentile: require BOTH the complexity cutoff and the percentile rank cutoff for "
                "each level. This keeps simple shapes from being marked complex just because they rank highly "
                "in a run of simple parcels. Unlike data reliability, this mode does not lower a score cutoff."),
                ("Example: with medium rank 0.50 and medium complexity 0.60, a parcel with complexity 0.65 and "
                "rank 0.40 is LOW in percentile mode but MEDIUM in absolute mode, assuming it is below the high "
                "complexity cutoff. A parcel with complexity 0.40 remains LOW even at rank 0.90. "
                "HIGH is checked first, then MEDIUM; otherwise the level is LOW. The result is saved as "
                "geometry_complexity_level; LOW is favourable geometry in the rule table. "
                "The weighted inspection score still uses geometry percentile rank in either mode."),
            ),
            ("Absolute: complexity >= complexity cutoff\n"
            "Percentile: complexity >= complexity cutoff AND rank >= rank cutoff\n"
            "Apply separately to HIGH and MEDIUM; otherwise LOW."),
            (
                "inspection_geometry_threshold_mode",
                "inspection_medium_geometry_risk_threshold",
                "inspection_high_geometry_risk_threshold",
                "inspection_medium_geometry_complexity_threshold",
                "inspection_high_geometry_complexity_threshold",
            ),
        ),
    ),
}
# Groups most scenarios change are shown open; the rest start collapsed to keep the form short.
OPEN_GROUPS = {"data", "split", "models"}
TUNING_GROUP = {
    "id": "tuning",
    "label": "Hyperparameter grids (advanced)",
    "description": "Values searched for each model, as JSON: {\"parameter\": [values]}. Only the grids of selected models are used.",
    "collapsed": True,
}


def available_models():
    """List models whose optional libraries are installed, without importing those libraries."""
    optional = {
        "xgboost": ("xgboost",),
        "lightgbm": ("lightgbm",),
        "tensorflow_neural_network": ("tensorflow", "scikeras"),
        "keras_lstm": ("tensorflow", "scikeras"),
    }
    return [
        name
        for name in _default_tune_params()
        if all(find_spec(module) is not None for module in optional.get(name, ()))
    ]


def ml_defaults():
    """Return JSON-compatible defaults for every editable setting, keyed as the form submits them."""
    # Read the declared defaults; building a config would clip limits to placeholder columns.
    defaults = {
        option.name: option.default
        for option in fields(ClassificationPipelineConfig)
        if option.default is not MISSING and option.name not in MANAGED
    }
    defaults |= WEB_DEFAULTS
    # The browser must choose the target; the output names are derived from it when left blank.
    defaults |= {"target_column": None, "prediction_column": None, "prediction_filled_column": None}
    defaults["selected_models"] = available_models()
    defaults["n_jobs"] = get_settings().max_batch_workers
    # The EDA outlier flag is always attached in the web app, so use it unless the user opts out.
    defaults["inspection_include_outlier"] = True
    # Web scenarios start with a row-wise spatial split, a five-model vote and probability optimisation.
    defaults |= {
        "spatial_split": True,
        "spatial_split_method": "by_row",
        "top_voting_models": 5,
        "optimize_class_probabilities": True,
        "probability_optimization_iterations": 5,
    }
    defaults |= {f"{TUNE_PREFIX}{model}": grid for model, grid in _default_tune_params().items()}
    return json.loads(json.dumps(defaults))


def _field_type(name, value):
    """Map a default value to the form control type."""
    if name in NULLABLE:
        return NULLABLE[name]
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    return "object" if isinstance(value, dict) else "string"


def ml_schema(columns):
    """Describe every setting with its group, default, choices, dependencies and help text."""
    defaults = ml_defaults()
    # A spatial split needs parcel geometry, so results without it start with a random split.
    defaults["spatial_split"] = any(column.get("geometry") for column in columns)
    names = [column["name"] for column in columns if not column.get("geometry")]
    choices = CHOICES | {
        "target_column": [None, *names],
        "id_column": names,
        "selected_models": list(_default_tune_params()),
    }
    sections = {
        name: section[0] for group_sections in SUBGROUPS.values() for section in group_sections for name in section[4]
    }
    metadata = [
        {
            "name": name,
            "type": _field_type(name, defaults[name]),
            "default": defaults[name],
            "choices": choices.get(name),
            "group": group,
            "subgroup": sections.get(name),
            "description": description,
            "enabled_when": ENABLED_WHEN.get(name),
        }
        for group, _, _, members in GROUPS
        for name, description in members.items()
    ]
    metadata += [
        {
            "name": f"{TUNE_PREFIX}{model}",
            "label": model.replace("_", " "),
            "type": "object",
            "default": defaults[f"{TUNE_PREFIX}{model}"],
            "choices": None,
            "group": TUNING_GROUP["id"],
            "description": "",
            "enabled_when": None,
        }
        for model in _default_tune_params()
    ]
    return {
        "defaults": defaults,
        "fields": metadata,
        "groups": [
            {
                "id": group,
                "label": title,
                "description": text,
                "collapsed": group not in OPEN_GROUPS,
                "subgroups": [
                    {
                        "id": section, "label": label, "description": list(paragraphs), "formula": formula,
                        "parent": "thresholds" if group == "inspection" and section in {"data_reliability", "geometry"} else None,
                    }
                    for section, label, paragraphs, formula, _ in SUBGROUPS.get(group, ())
                ],
            }
            for group, title, text, _ in GROUPS
        ]
        + [TUNING_GROUP],
        "managed": MANAGED,
        "limits": {"upload_limit_bytes": 0, "max_batch_workers": get_settings().max_batch_workers},
    }


def _valid_grid(grid):
    """Accept only a bounded {"model__parameter": [JSON scalars or lists]} search grid."""
    return (
        isinstance(grid, dict)
        and 0 < len(grid) <= 40
        and all(
            isinstance(key, str) and key.startswith("model__") and isinstance(values, list) and 0 < len(values) <= 50
            for key, values in grid.items()
        )
    )


def _valid_value(name, value, original, choices):
    """Check one setting against the type of its default and its allowed choices."""
    if name.startswith(TUNE_PREFIX):
        return _valid_grid(value)
    if value is None:
        return name in NULLABLE
    kind = _field_type(name, original)
    if kind == "boolean":
        return type(value) is bool
    if kind == "number":
        return type(value) is int if type(original) is int else type(value) in {int, float}
    if kind == "array":
        if name == "probability_multiplier_grid":
            return isinstance(value, list) and 0 < len(value) <= 50 and all(type(item) in {int, float} for item in value)
        scalar = (str, int, float) if name == "exclude_target_values" else str
        valid = isinstance(value, list) and len(value) <= 200 and all(isinstance(item, scalar) for item in value)
        return valid and (name not in choices or set(value) <= set(choices[name]))
    return isinstance(value, str) and 0 < len(value) <= 200 and (name not in choices or value in choices[name])


def validate_ml(settings, features, available, *, geometry=True):
    """Return complete validated settings, rejecting unknown keys, bad types and unusable columns.

    ``available`` lists the selectable columns (parcel result plus EDA-calculated); ``geometry``
    says whether the parcel result has a geometry column for spatial splitting and centroids.
    """
    defaults = ml_defaults()
    if not isinstance(settings, dict) or set(settings) - set(defaults):
        raise ValueError("Error: Unsupported or server-managed ML setting.")
    options = defaults | settings
    choices = CHOICES | {"selected_models": list(_default_tune_params())}
    for name, value in options.items():
        if not _valid_value(name, value, defaults[name], choices):
            raise ValueError(f"Error: Invalid value for {name}.")
    try:
        json.dumps(options, allow_nan=False)
    except ValueError:
        raise ValueError("Error: Settings must contain only finite numbers.") from None
    target, identifier = options["target_column"], options["id_column"]
    if target is None or target not in available:
        raise ValueError("Error: Choose the target column to predict.")
    if identifier not in available or identifier == target:
        raise ValueError("Error: Choose an ID column that differs from the target.")
    reserved = {target, identifier}
    if (
        not isinstance(features, list)
        or not features
        or len(features) != len(set(features))
        or set(features) - set(available)
        or reserved & set(features)
    ):
        raise ValueError("Error: Select unique feature columns from the list, excluding the target and ID.")
    if not options["selected_models"]:
        raise ValueError("Error: Select at least one model.")
    if not 1 <= options["n_jobs"] <= get_settings().max_batch_workers:
        raise ValueError(f"Error: n_jobs must be between 1 and {get_settings().max_batch_workers}.")
    if not 0 <= options["random_state"] <= 2**32 - 1:
        raise ValueError("Error: random_state must be between 0 and 4294967295.")
    if not geometry and (options["spatial_split"] or options["add_centroid_features"] or options["fill_nulls"]):
        raise ValueError("Error: This parcel result has no geometry; disable spatial split, gap filling and centroid features.")
    try:
        with warnings.catch_warnings():
            # The library reports clipped limits as warnings; they are not submission errors.
            warnings.simplefilter("ignore")
            ClassificationPipelineConfig(project_dir="managed", **pipeline_arguments(options, features))
    except (TypeError, ValueError) as error:
        message = str(error)
        raise ValueError(message if message.startswith("Error:") else f"Error: {message}") from None
    return options


# A re-run keeps the checked data and the split, so these groups and fields stay as they were.
CHECK_PREPARE_GROUPS = {"data", "quality", "split"}
# Listed in later groups but read by Check or Prepare: the input columns and the prepare-report limits.
CHECK_PREPARE_FIELDS = {
    "inspection_data_reliability_column",
    "inspection_geometry_complexity_column",
    "max_distribution_features",
    "max_features_per_plot",
    "max_map_geometries",
}
# Predict alone reads these groups; every other setting is used by Train or Evaluate.
PREDICT_GROUPS = {"inspection", "output"}
# Evaluate and Predict alone read these groups, so changing them keeps every trained model.
EVALUATE_GROUPS = {"ensemble", "confidence", "interpretability", *PREDICT_GROUPS}
# Train saves the out-of-fold probabilities Evaluate needs only while one of these is on.
OOF_SETTINGS = ("optimize_class_probabilities", "rank_confidence_enabled")
# A change to one of these makes every saved model stale.
RETRAIN_ALL_WHEN = (
    "max_search_candidates",
    "scoring_primary",
    "scoring_secondary",
    "cv_ranking_method",
    "label_balancing_method",
    "smote_k_neighbors",
    "rare_category_min_frequency",
)
LSTM_MODEL = "keras_lstm"


def rerun_rules():
    """Describe which settings a Train, Evaluate or Predict re-run may change and which changes force retraining."""
    group_of = {name: group for group, _, _, members in GROUPS for name in members}
    group_of |= {name: TUNING_GROUP["id"] for name in ml_defaults() if name.startswith(TUNE_PREFIX)}
    later = [name for name, group in group_of.items() if group not in CHECK_PREPARE_GROUPS and name not in CHECK_PREPARE_FIELDS]
    return {
        "train": later,
        "evaluate": [name for name in later if group_of[name] in EVALUATE_GROUPS],
        "predict": [name for name in later if group_of[name] in PREDICT_GROUPS],
        "retrain_all_when": list(RETRAIN_ALL_WHEN),
        "retrain_model_when": {LSTM_MODEL: [name for name, group in group_of.items() if group == "lstm"]},
    }


def required_retrain(previous, current, rules=None):
    """Return the selected models whose saved training no longer matches the new settings."""
    rules = rules or rerun_rules()
    models = current["selected_models"]
    if any(previous.get(name) != current.get(name) for name in rules["retrain_all_when"]):
        return set(models)
    changed = {model for model in models if previous.get(f"{TUNE_PREFIX}{model}") != current.get(f"{TUNE_PREFIX}{model}")}
    for model, names in rules["retrain_model_when"].items():
        if model in models and any(previous.get(name) != current.get(name) for name in names):
            changed.add(model)
    return changed


def _tuples(value):
    """Restore the tuples that JSON stored as lists, such as hidden layer sizes."""
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def pipeline_arguments(options, features):
    """Translate validated web settings into ``ClassificationPipelineConfig`` arguments."""
    arguments = {k: v for k, v in options.items() if k not in WEB_DEFAULTS and not k.startswith(TUNE_PREFIX)}
    # Existing coordinates are selectable; generating them must not duplicate selected columns.
    arguments["feature_columns"] = list(dict.fromkeys([*features, *(CENTROID_COLUMNS if options["add_centroid_features"] else ())]))
    # The vote cannot use more models than were selected; one model falls back to a single-model run.
    if arguments["top_voting_models"] and len(arguments["selected_models"]) >= 2:
        arguments["top_voting_models"] = min(arguments["top_voting_models"], len(arguments["selected_models"]))
    arguments["tune_params"] = {
        name.removeprefix(TUNE_PREFIX): {parameter: [_tuples(item) for item in values] for parameter, values in grid.items()}
        for name, grid in options.items()
        if name.startswith(TUNE_PREFIX)
    }
    return arguments | {k: v for k, v in MANAGED.items() if isinstance(v, bool)}
