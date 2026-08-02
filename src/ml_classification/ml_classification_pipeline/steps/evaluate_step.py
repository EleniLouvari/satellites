"""Pipeline step for model evaluation, comparison, and selection reporting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from ..core.metrics import (
    extract_feature_importance_frame,
    load_modeling_context,
    score_predictions,
    select_top_models_for_interpretability,
)
from ..core.persistence import (
    load_joblib,
    load_json,
    print_formatted_txt,
    save_frame_csv,
    save_json,
    time_decorator,
)
from ..core.selection import (
    evaluate_voting_candidate,
    sort_metrics_without_voting,
)
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

    @staticmethod
    def _to_geo_classifier_result_row(
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
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
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
            y_true,
            y_pred,
            average=None,
            labels=labels,
            zero_division=0,
        )
        accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
        roc_auc_str = "Undefined"
        if probabilities is not None:
            try:
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

    @time_decorator
    def run_evaluate(self) -> dict[str, Any]:
        """Evaluate fitted models, generate reports, and select final strategy."""
        # Load prepared data and model artifacts for holdout evaluation.
        print_formatted_txt("Evaluating models...", "SUBSECTION")
        test_df = load_joblib(self.config.prepare_dir / "test_dataset.joblib")
        context = load_modeling_context(self.config)
        model_specs = load_json(self.config.train_dir / "model_specs.json")
        labels = context["labels"]
        active_features = context["active_features"]
        train_df = load_joblib(self.config.prepare_dir / "train_dataset.joblib")
        label_encoder = context["label_encoder"]
        X_train = train_df[active_features].copy()
        y_train = train_df[self.config.target_column].astype(str).copy()
        X_test = test_df[active_features].copy()
        y_test = test_df[self.config.target_column].astype(str).copy()

        train_rows: list[dict[str, Any]] = []
        test_rows: list[dict[str, Any]] = []
        geo_train_rows: list[dict[str, Any]] = []
        geo_test_rows: list[dict[str, Any]] = []
        probability_cache_train: dict[str, np.ndarray] = {}
        probability_cache_test: dict[str, np.ndarray] = {}
        fitted_estimators: dict[str, Any] = {}
        binary_problem = len(labels) == 2

        for model_name in model_specs:
            model_artifact_path = self.config.train_model_dir(model_name) / "best_model.joblib"
            if not model_artifact_path.exists():
                model_artifact_path = self.config.train_dir / f"{model_name}_best_model.joblib"
            estimator = load_joblib(model_artifact_path)
            fitted_estimators[model_name] = estimator
            train_predictions_encoded = estimator.predict(X_train)
            test_predictions_encoded = estimator.predict(X_test)
            train_predictions = label_encoder.inverse_transform(np.asarray(train_predictions_encoded, dtype=int))
            test_predictions = label_encoder.inverse_transform(np.asarray(test_predictions_encoded, dtype=int))
            train_probabilities = estimator.predict_proba(X_train) if hasattr(estimator, "predict_proba") else None
            test_probabilities = estimator.predict_proba(X_test) if hasattr(estimator, "predict_proba") else None
            train_rows.append(score_predictions(y_train, train_predictions, train_probabilities, labels, model_name))
            test_rows.append(score_predictions(y_test, test_predictions, test_probabilities, labels, model_name))
            geo_train_rows.append(
                self._to_geo_classifier_result_row(
                    y_true=y_train,
                    y_pred=train_predictions,
                    probabilities=train_probabilities,
                    labels=labels,
                    model_name=model_name,
                )
            )
            geo_test_rows.append(
                self._to_geo_classifier_result_row(
                    y_true=y_test,
                    y_pred=test_predictions,
                    probabilities=test_probabilities,
                    labels=labels,
                    model_name=model_name,
                )
            )

            report_df = pd.DataFrame(classification_report(y_test, test_predictions, output_dict=True)).T.reset_index()
            report_df.rename(columns={"index": "label"}, inplace=True)
            save_frame_csv(report_df, self.config.evaluate_dir / "reports" / f"{model_name}_classification_report.csv")

            if test_probabilities is not None:
                probability_cache_train[model_name] = train_probabilities
                probability_cache_test[model_name] = test_probabilities
                if binary_problem:
                    save_binary_evaluation_panel(
                        y_test,
                        test_predictions,
                        test_probabilities[:, 1],
                        labels,
                        self.config.evaluate_dir / "panels" / f"{model_name}_roc_confusion.png",
                        model_name,
                    )
                    save_binary_curve_plots(
                        y_test,
                        test_probabilities[:, 1],
                        self.config.evaluate_dir / "curves",
                        model_name,
                        pos_label=labels[1],
                        include_roc=False,
                    )
                else:
                    save_multiclass_evaluation_panel(
                        y_test,
                        test_predictions,
                        test_probabilities,
                        labels,
                        self.config.evaluate_dir / "panels" / f"{model_name}_roc_confusion.png",
                        model_name,
                    )
            else:
                save_confusion_matrix_plot(
                    y_test,
                    test_predictions,
                    labels,
                    self.config.evaluate_dir / "confusion_matrices" / f"{model_name}_confusion_matrix.png",
                    f"Confusion Matrix: {model_name}",
                )

        train_metrics_df = pd.DataFrame(train_rows).sort_values(self.config.scoring_primary, ascending=True)
        test_metrics_df = pd.DataFrame(test_rows).sort_values(self.config.scoring_primary, ascending=True)
        train_metrics_df = sort_metrics_without_voting(train_metrics_df, self.config)
        test_metrics_df = sort_metrics_without_voting(test_metrics_df, self.config)
        save_frame_csv(train_metrics_df, self.config.evaluate_dir / "model_metrics_train.csv")
        save_frame_csv(test_metrics_df, self.config.evaluate_dir / "model_metrics_test.csv")
        save_model_comparison_plot(
            test_metrics_df,
            self.config.evaluate_dir / "plots" / "model_comparison.png",
            self.config.scoring_primary,
        )

        interpretability_rows: list[dict[str, Any]] = []
        available_interpretability_models = test_metrics_df.loc[test_metrics_df["model"] != "soft_voting", "model"].tolist()
        effective_interpretability_top_models = min(
            self.config.interpretability_top_models,
            len(available_interpretability_models),
        )
        if self.config.interpretability_top_models > effective_interpretability_top_models:
            print_formatted_txt(
                (
                    "interpretability_top_models is larger than the available selected models. "
                    f"Using {effective_interpretability_top_models} instead of "
                    f"{self.config.interpretability_top_models}."
                ),
                "WARNING",
            )
        interpretability_models = select_top_models_for_interpretability(
            test_metrics_df,
            effective_interpretability_top_models,
        )
        tree_like_models = {
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "decision_tree",
            "hist_gradient_boosting",
            "xgboost",
            "lightgbm",
        }
        for model_name in interpretability_models:
            estimator = fitted_estimators.get(model_name)
            if estimator is None:
                continue
            row = {
                "model": model_name,
                "test_metric": float(
                    test_metrics_df.loc[test_metrics_df["model"] == model_name, self.config.scoring_primary].iloc[0]
                ),
                "feature_importance_created": False,
                "shap_created": False,
                "notes": "",
            }
            if model_name in tree_like_models:
                importance_df = extract_feature_importance_frame(estimator)
                if importance_df is not None and not importance_df.empty:
                    effective_feature_importance_top_n = min(
                        self.config.feature_importance_top_n,
                        len(importance_df),
                    )
                    if self.config.feature_importance_top_n > effective_feature_importance_top_n:
                        print_formatted_txt(
                            (
                                f"feature_importance_top_n for {model_name} is larger than the available "
                                f"features. Using {effective_feature_importance_top_n} instead of "
                                f"{self.config.feature_importance_top_n}."
                            ),
                            "WARNING",
                        )
                    importance_csv_path = self.config.evaluate_dir / "interpretability" / f"{model_name}_feature_importance.csv"
                    importance_plot_path = self.config.evaluate_dir / "interpretability" / f"{model_name}_feature_importance.png"
                    save_frame_csv(importance_df, importance_csv_path)
                    save_feature_importance_plot(
                        importance_df,
                        importance_plot_path,
                        model_name,
                        top_n=effective_feature_importance_top_n,
                    )
                    row["feature_importance_created"] = True
                else:
                    row["notes"] = "Feature importance was not available for the fitted estimator."
            else:
                row["notes"] = "Feature importance is only generated for tree-based top models."

            if self.config.interpretability_include_shap:
                sample_size = min(self.config.shap_sample_size, len(X_train))
                X_sample = X_train.sample(n=sample_size, random_state=self.config.random_state)
                try:
                    save_shap_summary_plot(
                        estimator,
                        X_sample,
                        self.config.evaluate_dir / "interpretability" / f"{model_name}_shap_summary.png",
                        model_name,
                        max_display=min(self.config.feature_importance_top_n, max(len(X_sample.columns), 1)),
                    )
                    row["shap_created"] = True
                except Exception as exc:
                    message = f"SHAP skipped for {model_name}: {exc}"
                    print_formatted_txt(message, "WARNING")
                    row["notes"] = f"{row['notes']} {message}".strip()
            interpretability_rows.append(row)

        if interpretability_rows:
            interpretability_df = pd.DataFrame(interpretability_rows)
            save_frame_csv(
                interpretability_df,
                self.config.evaluate_dir / "interpretability" / "interpretability_summary.csv",
            )

        selection = evaluate_voting_candidate(
            config=self.config,
            train_metrics_df=train_metrics_df,
            test_metrics_df=test_metrics_df,
            probability_cache_train=probability_cache_train,
            probability_cache_test=probability_cache_test,
            y_train=y_train,
            y_test=y_test,
            labels=labels,
        )

        prob_models = [model for model in test_metrics_df["model"].tolist() if model in probability_cache_test]
        if self.config.top_voting_models is not None:
            prob_models = prob_models[-self.config.top_voting_models :]
        if len(prob_models) >= 2:
            averaged_probs_train = np.zeros_like(probability_cache_train[prob_models[0]], dtype=np.float64)
            averaged_probs_test = np.zeros_like(probability_cache_test[prob_models[0]], dtype=np.float64)
            for model_name in prob_models:
                averaged_probs_train += probability_cache_train[model_name]
                averaged_probs_test += probability_cache_test[model_name]
            averaged_probs_train /= len(prob_models)
            averaged_probs_test /= len(prob_models)

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
            interpretability_summary_df = pd.read_csv(interpretability_summary_path)
            csv_contracts["interpretability_summary.csv"] = interpretability_summary_df.columns.tolist()
        save_json(selection_with_schema, self.config.evaluate_dir / "selection_summary.json")
        self._save_schema_manifest(
            self.config.evaluate_dir,
            "evaluate_step_schema_manifest",
            json_contracts={"selection_summary.json": sorted(selection_with_schema.keys())},
            csv_contracts=csv_contracts,
        )
        train_report_df = (
            pd.read_csv(self.config.evaluate_dir / "model_metrics_train_with_voting.csv")
            if (self.config.evaluate_dir / "model_metrics_train_with_voting.csv").exists()
            else train_metrics_df
        )
        test_report_df = (
            pd.read_csv(self.config.evaluate_dir / "model_metrics_test_with_voting.csv")
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
        print_formatted_txt(
            f"Selected strategy: {selection['selection_type']} using {selection['selected_models']}",
            "RESULTS",
        )
        return selection_with_schema
