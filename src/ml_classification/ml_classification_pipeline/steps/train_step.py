"""Pipeline step for hyperparameter search, model fitting, and training artifacts."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV

from ..core.metrics import load_modeling_context
from ..core.persistence import (
    calculate_time_duration,
    load_joblib,
    print_formatted_txt,
    save_frame_csv,
    save_joblib,
    save_json,
    time_decorator,
)
from ..reporting import write_index_report, write_train_report
from ..core import PipelineStepBase
from ..core.models import build_model_candidates
from ..visuals import save_cv_fold_comparison_plot, save_search_results_plot


class TrainStep(PipelineStepBase):
    """Train configured model candidates and persist ranked training outputs."""

    @time_decorator
    def run_train(self) -> dict[str, Any]:
        """Train candidate models with CV search and save ranked training artifacts."""
        # Start model training across configured candidate estimators.
        print_formatted_txt("Training models...", "SUBSECTION")
        train_df = load_joblib(self.config.prepare_dir / "train_dataset.joblib")
        context = load_modeling_context(self.config)
        prepare_summary = context["prepare_summary"]
        active_features = context["active_features"]
        numeric_features = context["numeric_features"]
        categorical_features = context["categorical_features"]
        label_encoder = context["label_encoder"]

        X_train = train_df[active_features].copy()
        y_train = label_encoder.transform(train_df[self.config.target_column].astype(str))
        cv = [(np.array(fold["train_index"]), np.array(fold["valid_index"])) for fold in prepare_summary["cv_folds"]]

        num_classes = len(prepare_summary["target_labels"])
        candidates = build_model_candidates(self.config, numeric_features, categorical_features, num_classes=num_classes)
        training_rows: list[dict[str, Any]] = []
        failed_rows: list[dict[str, Any]] = []
        best_fold_rows: list[dict[str, Any]] = []
        model_specs: dict[str, dict[str, Any]] = {}

        for candidate in candidates:
            param_distributions = dict(candidate.param_distributions)
            print("_" * 100)
            print_formatted_txt(f"Training model {candidate.name}", "INFO")
            model_started = time.perf_counter()
            model_training_ok = False
            model_output_dir = self.config.train_model_dir(candidate.name)
            model_output_dir.mkdir(parents=True, exist_ok=True)
            if candidate.name == "knn":
                min_fold_train_size = min(len(train_idx) for train_idx, _ in cv)
                safe_neighbors = [
                    value for value in param_distributions["model__n_neighbors"] if value <= max(1, min_fold_train_size)
                ]
                param_distributions["model__n_neighbors"] = safe_neighbors or [1]
            search_n_jobs = self.config.n_jobs
            search = HalvingRandomSearchCV(
                estimator=candidate.builder(),
                param_distributions=param_distributions,
                factor=2,
                cv=cv,
                scoring=self.config.scoring_primary,
                n_jobs=search_n_jobs,
                random_state=self.config.random_state,
                n_candidates=self.config.max_search_candidates,
                refit=True,
                error_score="raise",
            )
            try:
                try:
                    search.fit(X_train, y_train)
                except PermissionError:
                    search = HalvingRandomSearchCV(
                        estimator=candidate.builder(),
                        param_distributions=param_distributions,
                        factor=2,
                        cv=cv,
                        scoring=self.config.scoring_primary,
                        n_jobs=1,
                        random_state=self.config.random_state,
                        n_candidates=self.config.max_search_candidates,
                        refit=True,
                        error_score="raise",
                    )
                    search.fit(X_train, y_train)

                results_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
                cv_results_path = model_output_dir / "cv_results.csv"
                best_model_path = model_output_dir / "best_model.joblib"
                search_plot_path = model_output_dir / "search_results.png"
                save_frame_csv(results_df, cv_results_path)
                save_joblib(search.best_estimator_, best_model_path)
                save_search_results_plot(results_df, search_plot_path, self.config.scoring_primary)
                print_formatted_txt(
                    (
                        f"Saved artifacts for {candidate.name}: "
                        f"model={best_model_path}, cv_results={cv_results_path}, plot={search_plot_path}"
                    ),
                    "INFO",
                )
                best_row = results_df.iloc[0]
                split_score_columns = sorted(
                    [
                        column
                        for column in results_df.columns
                        if column.startswith("split") and column.endswith("_test_score")
                    ],
                    key=lambda value: int(value.split("_")[0].replace("split", "")),
                )
                best_fold_scores = [float(best_row[score_column]) for score_column in split_score_columns]
                cv_score_mean = float(np.mean(best_fold_scores)) if best_fold_scores else float(search.best_score_)
                cv_score_std = float(np.std(best_fold_scores, ddof=0)) if best_fold_scores else 0.0
                cv_score_minus_std = float(cv_score_mean - cv_score_std)
                for fold_idx, score_column in enumerate(split_score_columns):
                    best_fold_rows.append(
                        {
                            "model": candidate.name,
                            "fold": fold_idx,
                            "score": float(best_row[score_column]),
                        }
                    )

                training_rows.append(
                    {
                        "model": candidate.name,
                        "best_cv_score": cv_score_mean,
                        "cv_score_std": cv_score_std,
                        "cv_score_minus_std": cv_score_minus_std,
                        "best_iteration": int(getattr(search, "n_iterations_", 0)),
                        "supports_predict_proba": candidate.supports_predict_proba,
                    }
                )
                model_specs[candidate.name] = {
                    "best_params": search.best_params_,
                    "supports_predict_proba": candidate.supports_predict_proba,
                    "numeric_features": numeric_features,
                    "categorical_features": categorical_features,
                    "artifact_dir": str(model_output_dir),
                }
                model_training_ok = True
            except Exception as exc:
                print_formatted_txt(
                    f"Skipping model '{candidate.name}' because training failed: {exc}",
                    "WARNING",
                )
                failed_rows.append(
                    {
                        "model": candidate.name,
                        "error_type": exc.__class__.__name__,
                        "error_message": str(exc),
                    }
                )
            finally:
                model_duration = time.perf_counter() - model_started
                model_duration_text = calculate_time_duration(0, model_duration)
                status = "finished" if model_training_ok else "failed"
                status_level = "INFO" if model_training_ok else "WARNING"
                print_formatted_txt(
                    f"Model {candidate.name} {status} in duration: {model_duration_text}",
                    status_level,
                )

        failed_models_df = pd.DataFrame(failed_rows)
        if failed_models_df.empty:
            failed_models_df = pd.DataFrame(columns=["model", "error_type", "error_message"])

        if not training_rows:
            save_frame_csv(failed_models_df, self.config.train_dir / "training_failed_models.csv")
            raise RuntimeError("All selected models failed during training. Check training_failed_models.csv for details.")

        training_summary_df = pd.DataFrame(training_rows)
        ranking_column = "cv_score_minus_std" if self.config.cv_ranking_method == "score_minus_std" else "best_cv_score"
        training_summary_df["cv_ranking_metric"] = training_summary_df[ranking_column]
        training_summary_df["cv_ranking_method"] = self.config.cv_ranking_method
        training_summary_df = training_summary_df.sort_values(
            ["cv_ranking_metric", "best_cv_score"],
            ascending=[False, False],
        )
        best_fold_scores_df = pd.DataFrame(best_fold_rows)
        save_frame_csv(training_summary_df, self.config.train_dir / "training_summary.csv")
        save_frame_csv(failed_models_df, self.config.train_dir / "training_failed_models.csv")
        save_frame_csv(best_fold_scores_df, self.config.train_dir / "best_cv_fold_scores.csv")
        save_json(model_specs, self.config.train_dir / "model_specs.json")
        self._save_schema_manifest(
            self.config.train_dir,
            "train_step_schema_manifest",
            json_contracts={
                "model_specs.json": [
                    "<model_name>",
                    "best_params",
                    "supports_predict_proba",
                    "numeric_features",
                    "categorical_features",
                    "artifact_dir",
                ],
            },
            csv_contracts={
                "training_summary.csv": training_summary_df.columns.tolist(),
                "training_failed_models.csv": failed_models_df.columns.tolist(),
                "best_cv_fold_scores.csv": best_fold_scores_df.columns.tolist(),
            },
        )
        save_cv_fold_comparison_plot(
            best_fold_scores_df,
            self.config.train_dir / "plots" / "best_cv_fold_scores.png",
            self.config.scoring_primary,
        )
        write_train_report(self.config, training_summary_df, model_specs, failed_models_df)
        write_index_report(self.config)
        print_formatted_txt(
            (
                f"Best CV model: {training_summary_df.iloc[0]['model']} "
                f"using {training_summary_df.iloc[0]['cv_ranking_method']}="
                f"{training_summary_df.iloc[0]['cv_ranking_metric']:.4f}"
            ),
            "RESULTS",
        )
        return {
            "models_trained": training_summary_df["model"].tolist(),
            "models_failed": failed_models_df["model"].tolist() if not failed_models_df.empty else [],
            "best_cv_model": training_summary_df.iloc[0]["model"],
            "cv_ranking_method": self.config.cv_ranking_method,
        }
