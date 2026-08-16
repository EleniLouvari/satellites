"""Pipeline step for model evaluation, comparison, and selection reporting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from common_libraries.io_library import read_data
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score

# Evaluation operates only on held-out data to keep reported model comparisons unbiased.

from ..core.metrics import (
    extract_feature_importance_frame,
    load_modeling_context,
    score_predictions,
    select_top_models_for_interpretability,
)
from ..core.persistence import load_joblib, load_json, print_formatted_txt, save_frame_csv, save_json, time_decorator
from ..core.selection import apply_class_probability_multipliers, evaluate_voting_candidate, sort_metrics_without_voting
from ..reporting import write_evaluate_report, write_index_report
from ..core import PipelineStepBase
from ..visuals import (
    save_binary_curve_plots,
    save_binary_evaluation_panel,
    save_confusion_matrix_plot,
    save_feature_importance_plot,
    save_model_comparison_plot,
    save_multiclass_evaluation_panel,
    save_shap_summary_plot,
)


class EvaluateStep(PipelineStepBase):
    """Evaluate trained models on train/test sets and choose prediction strategy."""

    def _to_geo_classifier_result_row(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        probabilities: np.ndarray | None,
        labels: list[str],
        model_name: str,
    ) -> dict[str, Any]:
        """Build legacy GeoDataFrameClassifier-style metric rows for reporting."""
        # Branch metric formatting for binary versus multiclass reporting outputs.
        binary_problem = len(labels) == 2
        if binary_problem:
            conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
            if conf_matrix.shape == (2, 2):
                tn, fp, fn, tp = conf_matrix.ravel()
            else:
                tn = fp = fn = tp = 0
            # Compute confusion-derived metrics safely (avoid division-by-zero).
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            # Overall accuracy across examples.
            accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
            # Compute binary AUC when probabilities available; handle invalid cases.
            auc = np.nan
            if probabilities is not None and probabilities.shape[1] >= 2:
                try:
                    auc = float(roc_auc_score(y_true, probabilities[:, 1]))
                except ValueError:
                    auc = np.nan
            return {
                "model": model_name,
                "roc_auc": (100 * auc) if not np.isnan(auc) else "Undefined",
                "accuracy": 100 * accuracy,
                "precision": 100 * precision,
                "recall": 100 * recall,
                "f1_score": 100 * f1,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            }

        precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=labels, zero_division=0
        )
        # Per-class precision/recall/f1 and overall accuracy for multiclass problems.
        accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
        roc_auc_str = "Undefined"
        if probabilities is not None:
            try:
                # Compute per-class ROC AUC with one-vs-rest approach when possible.
                auc_per_class = roc_auc_score(y_true, probabilities, labels=labels, multi_class="ovr", average=None)
                roc_auc_str = ", ".join([f"class {idx} ({100 * score:.2f})" for idx, score in enumerate(auc_per_class)])
            except ValueError:
                roc_auc_str = "Undefined"
        precision_str = ", ".join([f"class {idx} ({100 * score:.2f})" for idx, score in enumerate(precision_arr)])
        recall_str = ", ".join([f"class {idx} ({100 * score:.2f})" for idx, score in enumerate(recall_arr)])
        f1_str = ", ".join([f"class {idx} ({100 * score:.2f})" for idx, score in enumerate(f1_arr)])
        return {
            "model": model_name,
            "accuracy": 100 * accuracy,
            "f1_score": 100 * float(np.mean(f1_arr)),
            "precision": 100 * float(np.mean(precision_arr)),
            "recall": 100 * float(np.mean(recall_arr)),
            "f1_score_class": f1_str,
            "precision_class": precision_str,
            "recall_class": recall_str,
            "roc_auc_class": roc_auc_str,
        }

    def _load_evaluation_inputs(self):
        # Load modeling context and prepared train/test datasets used for holdout evaluation.
        context = load_modeling_context(self.config)
        train_df = load_joblib(self.config.prepare_dir / "train_dataset.joblib")
        test_df = load_joblib(self.config.prepare_dir / "test_dataset.joblib")
        features = context["active_features"]
        # Return inputs packaged as a dict for downstream evaluation routines.
        return {
            "context": context,
            "model_specs": load_json(self.config.train_dir / "model_specs.json"),
            "cv_metrics": read_data(str(self.config.train_dir / "training_summary.csv"), watch_curly_brackets=False),
            "X_train": train_df[features].copy(),
            "y_train": train_df[self.config.target_column].astype(str).copy(),
            "X_test": test_df[features].copy(),
            "y_test": test_df[self.config.target_column].astype(str).copy(),
        }

    def _model_artifact_path(self, model_name):
        # Prefer model artefact inside model-specific folder, fallback to top-level train dir.
        path = self.config.train_model_dir(model_name) / "best_model.joblib"
        return path if path.exists() else self.config.train_dir / f"{model_name}_best_model.joblib"

    def _save_model_evaluation_plot(self, model_name, y_test, predictions, probabilities, labels):
        # If no probabilities, only save confusion matrix for the model.
        if probabilities is None:
            save_confusion_matrix_plot(
                y_test, predictions, labels,
                self.config.evaluate_dir / "confusion_matrices" / f"{model_name}_confusion_matrix.png",
                f"Confusion Matrix: {model_name}",
            )
            return
        # When probabilities exist, save combined ROC/confusion panels for binary or multiclass.
        panel_path = self.config.evaluate_dir / "panels" / f"{model_name}_roc_confusion.png"
        if len(labels) == 2:
            # Binary-specific evaluation visuals use the positive-class probability column.
            save_binary_evaluation_panel(y_test, predictions, probabilities[:, 1], labels, panel_path, model_name)
            save_binary_curve_plots(
                y_test, probabilities[:, 1], self.config.evaluate_dir / "curves", model_name,
                pos_label=labels[1], include_roc=False,
            )
        else:
            # Multiclass panel visualizes per-class metrics and curves.
            save_multiclass_evaluation_panel(y_test, predictions, probabilities, labels, panel_path, model_name)

    def _evaluate_base_models(self, inputs):
        # Evaluate each trained model on train/test splits, collecting metrics and artifacts.
        context = inputs["context"]
        labels = context["labels"]
        encoder = context["label_encoder"]
        # Caches hold probability arrays for train/test/oof (if available) to support voting.
        caches = {"train": {}, "test": {}, "oof": {}}
        metric_rows = {"train": [], "test": []}
        geo_rows = {"train": [], "test": []}
        estimators = {}
        for model_name in inputs["model_specs"]:
            # Load the persisted best estimator for this candidate model.
            estimator = load_joblib(self._model_artifact_path(model_name))
            estimators[model_name] = estimator
            # Load out-of-fold probabilities if saved during training (used for class optimization).
            oof_path = self.config.train_model_dir(model_name) / "oof_probabilities.joblib"
            if oof_path.exists():
                caches["oof"][model_name] = load_joblib(oof_path)
            split_values = {}
            for split_name in ("train", "test"):
                X = inputs[f"X_{split_name}"]
                y = inputs[f"y_{split_name}"]
                # Predict encoded labels and convert back to original string labels.
                encoded = estimator.predict(X)
                predictions = encoder.inverse_transform(np.asarray(encoded, dtype=int))
                # Obtain class probabilities when supported by estimator.
                probabilities = estimator.predict_proba(X) if hasattr(estimator, "predict_proba") else None
                split_values[split_name] = (predictions, probabilities)
                # Score predictions and record legacy geo-classifier-style rows.
                metric_rows[split_name].append(score_predictions(y, predictions, probabilities, labels, model_name))
                geo_rows[split_name].append(self._to_geo_classifier_result_row(y, predictions, probabilities, labels, model_name))
                if probabilities is not None:
                    caches[split_name][model_name] = probabilities
            # Save a per-model classification report and evaluation plots for the test split.
            test_predictions, test_probabilities = split_values["test"]
            report = pd.DataFrame(classification_report(inputs["y_test"], test_predictions, output_dict=True)).T.reset_index()
            report.rename(columns={"index": "label"}, inplace=True)
            save_frame_csv(report, self.config.evaluate_dir / "reports" / f"{model_name}_classification_report.csv")
            self._save_model_evaluation_plot(
                model_name, inputs["y_test"], test_predictions, test_probabilities, labels
            )
        return metric_rows, geo_rows, caches, estimators

    def _save_interpretability(self, cv_metrics_df, test_metrics_df, fitted_estimators, X_train):
        # Select top models for interpretability analysis and optionally produce SHAP/importance outputs.
        available = cv_metrics_df["model"].tolist()
        top_n = min(self.config.interpretability_top_models, len(available))
        if self.config.interpretability_top_models > top_n:
            print_formatted_txt(
                f"interpretability_top_models is larger than the available selected models. "
                f"Using {top_n} instead of {self.config.interpretability_top_models}.", "WARNING"
            )
        # Define which model families support feature importance extraction.
        tree_models = {
            "random_forest", "extra_trees", "gradient_boosting", "decision_tree",
            "hist_gradient_boosting", "xgboost", "lightgbm",
        }
        rows = []
        for model_name in select_top_models_for_interpretability(cv_metrics_df, top_n):
            estimator = fitted_estimators.get(model_name)
            if estimator is None:
                # Skip interpretability if estimator not available (should be rare).
                continue
            row = {
                "model": model_name,
                "cv_ranking_metric": float(cv_metrics_df.loc[cv_metrics_df["model"] == model_name, "cv_ranking_metric"].iloc[0]),
                "test_metric": float(test_metrics_df.loc[test_metrics_df["model"] == model_name, self.config.scoring_primary].iloc[0]),
                "feature_importance_created": False, "shap_created": False, "notes": "",
            }
            # For tree-based models, extract and persist feature importance data + plot.
            if model_name in tree_models:
                importance = extract_feature_importance_frame(estimator)
                if importance is not None and not importance.empty:
                    importance_top_n = min(self.config.feature_importance_top_n, len(importance))
                    if self.config.feature_importance_top_n > importance_top_n:
                        print_formatted_txt(
                            f"feature_importance_top_n for {model_name} is larger than the available features. "
                            f"Using {importance_top_n} instead of {self.config.feature_importance_top_n}.", "WARNING"
                        )
                    base_path = self.config.evaluate_dir / "interpretability" / f"{model_name}_feature_importance"
                    save_frame_csv(importance, base_path.with_suffix(".csv"))
                    save_feature_importance_plot(importance, base_path.with_suffix(".png"), model_name, top_n=importance_top_n)
                    row["feature_importance_created"] = True
                else:
                    row["notes"] = "Feature importance was not available for the fitted estimator."
            else:
                row["notes"] = "Feature importance is only generated for tree-based top models."
            # Optionally attempt SHAP summaries; exceptions are caught and logged.
            if self.config.interpretability_include_shap:
                sample = X_train.sample(n=min(self.config.shap_sample_size, len(X_train)), random_state=self.config.random_state)
                try:
                    save_shap_summary_plot(
                        estimator, sample,
                        self.config.evaluate_dir / "interpretability" / f"{model_name}_shap_summary.png",
                        model_name, max_display=min(self.config.feature_importance_top_n, max(len(sample.columns), 1)),
                    )
                    row["shap_created"] = True
                except Exception as exc:
                    message = f"SHAP skipped for {model_name}: {exc}"
                    print_formatted_txt(message, "WARNING")
                    row["notes"] = f"{row['notes']} {message}".strip()
            rows.append(row)
        if rows:
            save_frame_csv(pd.DataFrame(rows), self.config.evaluate_dir / "interpretability" / "interpretability_summary.csv")

    @time_decorator
    def run_evaluate(self) -> dict[str, Any]:
        """Evaluate fitted models, generate reports, and select final strategy."""
        # Load prepared data and model artifacts for holdout evaluation.
        print_formatted_txt("Evaluating models...", "SUBSECTION")
        inputs = self._load_evaluation_inputs()
        context = inputs["context"]
        cv_metrics_df = inputs["cv_metrics"]
        labels = context["labels"]
        # Unpack training/test arrays and run base model evaluations to collect metrics and plots.
        X_train, y_train, y_test = inputs["X_train"], inputs["y_train"], inputs["y_test"]
        metric_rows, geo_rows, caches, fitted_estimators = self._evaluate_base_models(inputs)
        train_rows, test_rows = metric_rows["train"], metric_rows["test"]
        geo_train_rows, geo_test_rows = geo_rows["train"], geo_rows["test"]
        probability_cache_train, probability_cache_test = caches["train"], caches["test"]
        oof_probability_cache = caches["oof"]

        train_metrics_df = pd.DataFrame(train_rows).sort_values(self.config.scoring_primary, ascending=True)
        test_metrics_df = pd.DataFrame(test_rows).sort_values(self.config.scoring_primary, ascending=True)
        train_metrics_df = sort_metrics_without_voting(train_metrics_df, self.config)
        test_metrics_df = sort_metrics_without_voting(test_metrics_df, self.config)
        save_frame_csv(train_metrics_df, self.config.evaluate_dir / "model_metrics_train.csv")
        save_frame_csv(test_metrics_df, self.config.evaluate_dir / "model_metrics_test.csv")
        save_model_comparison_plot(
            test_metrics_df, self.config.evaluate_dir / "plots" / "model_comparison.png", self.config.scoring_primary
        )

        self._save_interpretability(cv_metrics_df, test_metrics_df, fitted_estimators, X_train)

        selection = evaluate_voting_candidate(
            config=self.config,
            cv_metrics_df=cv_metrics_df,
            train_metrics_df=train_metrics_df,
            test_metrics_df=test_metrics_df,
            probability_cache_train=probability_cache_train,
            probability_cache_test=probability_cache_test,
            y_train=y_train,
            y_test=y_test,
            labels=labels,
            oof_probability_cache=oof_probability_cache,
        )

        prob_models = list(selection["selected_models"]) if selection["selection_type"] == "soft_voting" else []
        if len(prob_models) >= 2:
            averaged_probs_train = np.zeros_like(probability_cache_train[prob_models[0]], dtype=np.float64)
            averaged_probs_test = np.zeros_like(probability_cache_test[prob_models[0]], dtype=np.float64)
            for model_name in prob_models:
                averaged_probs_train += probability_cache_train[model_name]
                averaged_probs_test += probability_cache_test[model_name]
            averaged_probs_train /= len(prob_models)
            averaged_probs_test /= len(prob_models)
            averaged_probs_train = apply_class_probability_multipliers(averaged_probs_train, labels,
                                                                       selection.get("class_probability_multipliers"))
            averaged_probs_test = apply_class_probability_multipliers(averaged_probs_test, labels,
                                                                      selection.get("class_probability_multipliers"))

            voting_predictions_train = np.asarray(labels)[np.argmax(averaged_probs_train, axis=1)]
            voting_predictions_test = np.asarray(labels)[np.argmax(averaged_probs_test, axis=1)]
            geo_train_rows.append(
                self._to_geo_classifier_result_row(
                    y_true=y_train,
                    y_pred=voting_predictions_train,
                    probabilities=averaged_probs_train,
                    labels=labels,
                    model_name="Voting",
                )
            )
            geo_test_rows.append(
                self._to_geo_classifier_result_row(
                    y_true=y_test,
                    y_pred=voting_predictions_test,
                    probabilities=averaged_probs_test,
                    labels=labels,
                    model_name="Voting",
                )
            )

        geo_train_report_df = pd.DataFrame(geo_train_rows)
        geo_test_report_df = pd.DataFrame(geo_test_rows)
        for group, report_df in {"train": geo_train_report_df, "test": geo_test_report_df}.items():
            voting_df = report_df[report_df["model"] == "Voting"].copy()
            base_df = report_df[report_df["model"] != "Voting"].copy()
            if "f1_score" in base_df.columns:
                base_df = base_df.sort_values(by=["f1_score"], ascending=False)
            ordered_df = pd.concat([base_df, voting_df], ignore_index=True)
            if group == "train":
                geo_train_report_df = ordered_df
            else:
                geo_test_report_df = ordered_df
            save_frame_csv(ordered_df, self.config.evaluate_dir / f"geo_classifier_style_{group}_metrics.csv")

        selection_with_schema = self._with_schema(selection, "selection_summary")
        csv_contracts = {
            "model_metrics_train.csv": train_metrics_df.columns.tolist(),
            "model_metrics_test.csv": test_metrics_df.columns.tolist(),
            "geo_classifier_style_train_metrics.csv": geo_train_report_df.columns.tolist(),
            "geo_classifier_style_test_metrics.csv": geo_test_report_df.columns.tolist(),
        }
        interpretability_summary_path = self.config.evaluate_dir / "interpretability" / "interpretability_summary.csv"
        if interpretability_summary_path.exists():
            interpretability_summary_df = read_data(str(interpretability_summary_path), watch_curly_brackets=False)
            csv_contracts["interpretability_summary.csv"] = interpretability_summary_df.columns.tolist()
        save_json(selection_with_schema, self.config.evaluate_dir / "selection_summary.json")
        self._save_schema_manifest(
            self.config.evaluate_dir,
            "evaluate_step_schema_manifest",
            json_contracts={"selection_summary.json": sorted(selection_with_schema.keys())},
            csv_contracts=csv_contracts,
        )
        train_report_df = (
            read_data(str(self.config.evaluate_dir / "model_metrics_train_with_voting.csv"), watch_curly_brackets=False)
            if (self.config.evaluate_dir / "model_metrics_train_with_voting.csv").exists()
            else train_metrics_df
        )
        test_report_df = (
            read_data(str(self.config.evaluate_dir / "model_metrics_test_with_voting.csv"), watch_curly_brackets=False)
            if (self.config.evaluate_dir / "model_metrics_test_with_voting.csv").exists()
            else test_metrics_df
        )
        write_evaluate_report(
            self.config,
            train_report_df,
            test_report_df,
            selection,
            geo_train_metrics_df=geo_train_report_df,
            geo_test_metrics_df=geo_test_report_df,
        )
        write_index_report(self.config)
        print_formatted_txt(f"Selected strategy: {selection['selection_type']} using {selection['selected_models']}", "RESULTS")
        return selection_with_schema
