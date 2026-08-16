"""Pipeline step for hyperparameter search, model fitting, and training artifacts."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV, ParameterGrid, cross_val_predict

# Successive halving limits compute by pruning weak parameter candidates early.
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

    def _prepare_search_parameters(self, candidate: Any, cv: list[tuple[np.ndarray, np.ndarray]]) -> tuple[dict, int]:
        """Return safe parameter distributions and the effective search size."""
        # Copy the candidate's parameter distributions so we can safely modify them.
        param_distributions = dict(candidate.param_distributions)
        # Calculate how many unique parameter combinations are available.
        available_candidates = len(ParameterGrid(param_distributions))
        # Limit effective candidates to configured maximum to bound compute.
        effective_candidates = min(self.config.max_search_candidates, available_candidates)
        # Special-case KNN to ensure n_neighbors is not larger than any fold's train size.
        if candidate.name == "knn":
            min_fold_train_size = min(len(train_idx) for train_idx, _ in cv)
            max_neighbors = max(1, min_fold_train_size)
            safe_neighbors = [
                value for value in param_distributions["model__n_neighbors"] if value <= max_neighbors
            ]
            param_distributions["model__n_neighbors"] = safe_neighbors or [1]
        return param_distributions, effective_candidates

    def _build_search(
        self,
        candidate: Any,
        param_distributions: dict,
        cv: list[tuple[np.ndarray, np.ndarray]],
        effective_candidates: int,
        n_jobs: int,
    ) -> HalvingRandomSearchCV:
        """Construct one consistently configured successive-halving search."""
        # Construct a HalvingRandomSearchCV with consistent configuration for each candidate.
        return HalvingRandomSearchCV(
            estimator=candidate.builder(),
            param_distributions=param_distributions,
            factor=2,
            cv=cv,
            scoring=self.config.scoring_primary,
            n_jobs=n_jobs,
            random_state=self.config.random_state,
            n_candidates=effective_candidates,
            refit=True,
            error_score="raise",
        )

    def _fit_search(
        self,
        candidate: Any,
        param_distributions: dict,
        cv: list[tuple[np.ndarray, np.ndarray]],
        effective_candidates: int,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
    ) -> HalvingRandomSearchCV:
        """Fit a search, retrying sequentially when worker creation is denied."""
        # Attempt parallel search; if OS denies worker creation, retry with single job.
        search = self._build_search(
            candidate, param_distributions, cv, effective_candidates, self.config.n_jobs
        )
        try:
            search.fit(X_train, y_train)
        except PermissionError:
            # Fall back to sequential execution to increase robustness on restricted systems.
            search = self._build_search(candidate, param_distributions, cv, effective_candidates, 1)
            search.fit(X_train, y_train)
        return search

    def _save_oof_probabilities(
        self,
        candidate: Any,
        estimator: Any,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        cv: list[tuple[np.ndarray, np.ndarray]],
        model_output_dir: Any,
    ) -> None:
        """Persist OOF probabilities only when class optimization requires them."""
        # Only compute out-of-fold predicted probabilities if optimization needs them and estimator supports it.
        if not (self.config.optimize_class_probabilities and candidate.supports_predict_proba):
            return
        try:
            probabilities = cross_val_predict(
                estimator, X_train, y_train, cv=cv, method="predict_proba", n_jobs=self.config.n_jobs
            )
        except PermissionError:
            # Retry without parallelism if process spawning is restricted.
            probabilities = cross_val_predict(estimator, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)
        # Persist OOF probabilities for later class probability optimization.
        save_joblib(probabilities, model_output_dir / "oof_probabilities.joblib")

    def _persist_search_artifacts(
        self,
        candidate: Any,
        search: HalvingRandomSearchCV,
        results_df: pd.DataFrame,
        model_output_dir: Any,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        cv: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Save the fitted estimator, diagnostics, plot, and optional OOF probabilities."""
        # Persist CV results table, the best-fitted estimator, and create a search results plot.
        cv_results_path = model_output_dir / "cv_results.csv"
        best_model_path = model_output_dir / "best_model.joblib"
        search_plot_path = model_output_dir / "search_results.png"
        save_frame_csv(results_df, cv_results_path)
        save_joblib(search.best_estimator_, best_model_path)
        # Optionally compute and save out-of-fold probabilities if required for later optimization.
        self._save_oof_probabilities(candidate, search.best_estimator_, X_train, y_train, cv, model_output_dir)
        save_search_results_plot(results_df, search_plot_path, self.config.scoring_primary)
        print_formatted_txt(
            f"Saved artifacts for {candidate.name}: model={best_model_path}, "
            f"cv_results={cv_results_path}, plot={search_plot_path}",
            "INFO",
        )

    def _summarize_search(self, candidate: Any, search: HalvingRandomSearchCV, results_df: pd.DataFrame) -> tuple[dict, list[dict]]:
        """Extract the model summary and per-fold scores from the winning search row."""
        best_row = results_df.iloc[0]
        split_columns = sorted(
            [column for column in results_df if column.startswith("split") and column.endswith("_test_score")],
            key=lambda value: int(value.split("_")[0].replace("split", "")),
        )
        fold_scores = [float(best_row[column]) for column in split_columns]
        score_mean = float(np.mean(fold_scores)) if fold_scores else float(search.best_score_)
        score_std = float(np.std(fold_scores, ddof=0)) if fold_scores else 0.0
        summary = {
            "model": candidate.name,
            "best_cv_score": score_mean,
            "cv_score_std": score_std,
            "cv_score_minus_std": score_mean - score_std,
            "best_iteration": int(getattr(search, "n_iterations_", 0)),
            "supports_predict_proba": candidate.supports_predict_proba,
        }
        per_fold = [
            {"model": candidate.name, "fold": fold_idx, "score": float(best_row[column])}
            for fold_idx, column in enumerate(split_columns)
        ]
        # Return summary for ranking and a per-fold list for reporting.
        return summary, per_fold

    def _train_candidate(
        self,
        candidate: Any,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        cv: list[tuple[np.ndarray, np.ndarray]],
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> tuple[dict | None, list[dict], dict | None, dict | None]:
        """Train one candidate and return either its artifacts or a structured failure."""
        # Top-level logging and timing for a single candidate training run.
        print("_" * 100)
        print_formatted_txt(f"Training model {candidate.name}", "INFO")
        started = time.perf_counter()
        succeeded = False
        try:
            model_output_dir = self.config.train_model_dir(candidate.name)
            model_output_dir.mkdir(parents=True, exist_ok=True)
            param_distributions, effective_candidates = self._prepare_search_parameters(candidate, cv)
            search = self._fit_search(candidate, param_distributions, cv, effective_candidates, X_train, y_train)
            results_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
            self._persist_search_artifacts(candidate, search, results_df, model_output_dir, X_train, y_train, cv)
            summary, fold_rows = self._summarize_search(candidate, search, results_df)
            model_spec = {
                "best_params": search.best_params_,
                "supports_predict_proba": candidate.supports_predict_proba,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "artifact_dir": str(model_output_dir),
            }
            succeeded = True
            return summary, fold_rows, model_spec, None
        except Exception as exc:
            # Capture failure details and return so other candidates can continue training.
            print_formatted_txt(f"Skipping model '{candidate.name}' because training failed: {exc}", "WARNING")
            failure = {"model": candidate.name, "error_type": exc.__class__.__name__, "error_message": str(exc)}
            return None, [], None, failure
        finally:
            # Log the training duration and whether it succeeded for observability.
            duration = calculate_time_duration(0, time.perf_counter() - started)
            status = "finished" if succeeded else "failed"
            level = "INFO" if succeeded else "WARNING"
            print_formatted_txt(f"Model {candidate.name} {status} in duration: {duration}", level)

    def _finalize_training(self, training_rows, failed_rows, best_fold_rows, model_specs):
        """Rank successful models and persist the complete training-stage contract."""
        # Consolidate failures into a DataFrame for reporting and persistence.
        failures = pd.DataFrame(failed_rows)
        if failures.empty:
            failures = pd.DataFrame(columns=["model", "error_type", "error_message"])
        if not training_rows:
            save_frame_csv(failures, self.config.train_dir / "training_failed_models.csv")
            raise RuntimeError("All selected models failed during training. Check training_failed_models.csv for details.")
        # Build a DataFrame summarizing successful models and compute ranking metric.
        summary = pd.DataFrame(training_rows)
        ranking_column = "cv_score_minus_std" if self.config.cv_ranking_method == "score_minus_std" else "best_cv_score"
        summary["cv_ranking_metric"] = summary[ranking_column]
        summary["cv_ranking_method"] = self.config.cv_ranking_method
        summary = summary.sort_values(["cv_ranking_metric", "best_cv_score"], ascending=[False, False])
        fold_scores = pd.DataFrame(best_fold_rows)
        # Persist tabular outputs and model specs for downstream steps.
        save_frame_csv(summary, self.config.train_dir / "training_summary.csv")
        save_frame_csv(failures, self.config.train_dir / "training_failed_models.csv")
        save_frame_csv(fold_scores, self.config.train_dir / "best_cv_fold_scores.csv")
        save_json(model_specs, self.config.train_dir / "model_specs.json")
        # Save schema manifest and visual diagnostics for CV fold comparison.
        self._save_schema_manifest(
            self.config.train_dir, "train_step_schema_manifest",
            json_contracts={"model_specs.json": [
                "<model_name>", "best_params", "supports_predict_proba",
                "numeric_features", "categorical_features", "artifact_dir",
            ]},
            csv_contracts={
                "training_summary.csv": summary.columns.tolist(),
                "training_failed_models.csv": failures.columns.tolist(),
                "best_cv_fold_scores.csv": fold_scores.columns.tolist(),
            },
        )
        save_cv_fold_comparison_plot(
            fold_scores, self.config.train_dir / "plots" / "best_cv_fold_scores.png", self.config.scoring_primary
        )
        write_train_report(self.config, summary, model_specs, failures)
        write_index_report(self.config)
        best = summary.iloc[0]
        print_formatted_txt(
            f"Best CV model: {best['model']} using {best['cv_ranking_method']}={best['cv_ranking_metric']:.4f}",
            "RESULTS",
        )
        return {
            "models_trained": summary["model"].tolist(),
            "models_failed": failures["model"].tolist(),
            "best_cv_model": best["model"],
            "cv_ranking_method": self.config.cv_ranking_method,
        }

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
            summary, fold_rows, model_spec, failure = self._train_candidate(
                candidate, X_train, y_train, cv, numeric_features, categorical_features
            )
            if failure is not None:
                failed_rows.append(failure)
                continue
            training_rows.append(summary)
            best_fold_rows.extend(fold_rows)
            model_specs[candidate.name] = model_spec

        return self._finalize_training(training_rows, failed_rows, best_fold_rows, model_specs)
