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
    load_json,
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

    _TENSORFLOW_CANDIDATES = frozenset({"keras_lstm", "tensorflow_neural_network"})

    def _effective_n_jobs(self, candidate: Any) -> int:
        """Use sequential outer CV for TensorFlow estimators to avoid process/GPU contention."""
        if candidate.name in self._TENSORFLOW_CANDIDATES:
            return 1
        return int(self.config.n_jobs)

    def _model_artifact_paths(self, model_name: str) -> tuple[Any, Any]:
        """Return expected per-model artifact paths used for incremental resume checks."""
        model_dir = self.config.train_model_dir(model_name)
        return model_dir / "best_model.joblib", model_dir / "cv_results.csv"

    def _is_model_fully_trained(self, model_name: str, supports_predict_proba: bool = True) -> bool:
        """Return True when all required artifacts for a trained model are present."""
        best_model_path, cv_results_path = self._model_artifact_paths(model_name)
        required_paths = [best_model_path, cv_results_path]
        requires_oof = bool(
            getattr(self.config, "optimize_class_probabilities", False)
            or getattr(self.config, "rank_confidence_enabled", False)
        )
        if requires_oof and supports_predict_proba:
            required_paths.append(self.config.train_model_dir(model_name) / "oof_probabilities.joblib")
        return all(path.exists() for path in required_paths)

    def _load_reused_training_metrics(
        self, candidate: Any, supports_predict_proba: bool
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Load summary/fold metrics for a reused model from persisted CV results."""
        _, cv_results_path = self._model_artifact_paths(candidate.name)
        if not cv_results_path.exists():
            return None, []
        results_df = pd.read_csv(cv_results_path)
        if results_df.empty:
            return None, []
        if "rank_test_score" in results_df.columns:
            best_row = results_df.sort_values("rank_test_score").iloc[0]
        else:
            best_row = results_df.iloc[0]

        split_columns = sorted(
            [column for column in results_df.columns if column.startswith("split") and column.endswith("_test_score")],
            key=lambda value: int(value.split("_")[0].replace("split", "")),
        )
        fold_scores = [float(best_row[column]) for column in split_columns]
        if fold_scores:
            score_mean = float(np.mean(fold_scores))
            score_std = float(np.std(fold_scores, ddof=0))
        else:
            score_mean = float(best_row.get("mean_test_score", np.nan))
            score_std = float(best_row.get("std_test_score", 0.0))

        summary = {
            "model": candidate.name,
            "best_cv_score": score_mean,
            "cv_score_std": score_std,
            "cv_score_minus_std": score_mean - score_std,
            "best_iteration": int(best_row.get("iter", 0)) if pd.notna(best_row.get("iter", 0)) else 0,
            "supports_predict_proba": supports_predict_proba,
        }
        per_fold = [
            {"model": candidate.name, "fold": fold_idx, "score": float(best_row[column])}
            for fold_idx, column in enumerate(split_columns)
        ]
        return summary, per_fold

    def _build_recovered_model_spec(
        self,
        candidate: Any,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> dict[str, Any]:
        """Reconstruct a minimal model spec for already-trained artifacts."""
        model_dir = self.config.train_model_dir(candidate.name)
        best_model_path, _ = self._model_artifact_paths(candidate.name)
        supports_predict_proba = bool(getattr(candidate, "supports_predict_proba", False))
        try:
            estimator = load_joblib(best_model_path)
            supports_predict_proba = hasattr(estimator, "predict_proba")
        except Exception:
            # Keep fallback from candidate metadata if model loading fails.
            pass
        return {
            "best_params": {},
            "supports_predict_proba": supports_predict_proba,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "artifact_dir": str(model_dir),
        }

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
            safe_neighbors = [value for value in param_distributions["model__n_neighbors"] if value <= max_neighbors]
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
        # TensorFlow estimators run sequentially at the outer CV/search level to avoid
        # concurrent TensorFlow runtimes competing for RAM/GPU resources.
        n_jobs = self._effective_n_jobs(candidate)
        if n_jobs == 1 and int(self.config.n_jobs) != 1 and candidate.name in self._TENSORFLOW_CANDIDATES:
            print_formatted_txt(
                f"Using n_jobs=1 for {candidate.name} to avoid parallel TensorFlow worker contention.",
                "INFO",
            )
        search = self._build_search(candidate, param_distributions, cv, effective_candidates, n_jobs)
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
        """Persist OOF probabilities when downstream optimization or validation requires them."""
        # OOF probabilities support both class-multiplier learning and leakage-safe
        # rank-confidence validation.
        requires_oof = bool(
            getattr(self.config, "optimize_class_probabilities", False)
            or getattr(self.config, "rank_confidence_enabled", False)
        )
        if not (requires_oof and candidate.supports_predict_proba):
            return
        n_jobs = self._effective_n_jobs(candidate)
        try:
            probabilities = cross_val_predict(
                estimator, X_train, y_train, cv=cv, method="predict_proba", n_jobs=n_jobs
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

        # Persist the exact temporal input contract used by the winning LSTM.
        if candidate.name == "keras_lstm":
            model = getattr(search.best_estimator_, "named_steps", {}).get("model")
            temporal_schema = getattr(model, "temporal_schema_", None)
            if temporal_schema is not None:
                save_json(temporal_schema, model_output_dir / "temporal_schema.json")

        # Optionally compute and save OOF probabilities for later optimization or confidence validation.
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

    def _finalize_training(self, training_rows, failed_rows, best_fold_rows, model_specs, skipped_models=None):
        """Rank successful models and persist the complete training-stage contract."""
        if skipped_models is None:
            skipped_models = []
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

        # Load existing model_specs to implement incremental training behavior.
        # This allows re-running with a different subset of models without retraining shared models.
        existing_model_specs: dict[str, dict[str, Any]] = {}
        existing_model_specs_path = self.config.train_dir / "model_specs.json"
        skipped_models = []
        if existing_model_specs_path.exists() and not self.config.force_retrain_models:
            existing_model_specs = load_json(existing_model_specs_path)
            print_formatted_txt("Incremental training enabled: checking for previously trained models", "INFO")

        # Recover reusable model specs directly from model artifact folders when
        # model_specs.json is missing or incomplete (common after interrupted runs).
        recovered_models = []
        if not self.config.force_retrain_models:
            for candidate in candidates:
                if candidate.name in existing_model_specs:
                    continue
                if not self._is_model_fully_trained(candidate.name, candidate.supports_predict_proba):
                    continue
                existing_model_specs[candidate.name] = self._build_recovered_model_spec(
                    candidate,
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                )
                recovered_models.append(candidate.name)
            if recovered_models:
                print_formatted_txt(
                    f"Recovered {len(recovered_models)} trained models from artifact folders: {recovered_models}",
                    "INFO",
                )

        for candidate in candidates:
            # Check if this model was already trained and we're not forcing retraining.
            if (
                candidate.name in existing_model_specs
                and not self.config.force_retrain_models
                and self._is_model_fully_trained(candidate.name, candidate.supports_predict_proba)
            ):
                print_formatted_txt(
                    f"Skipping already-trained model '{candidate.name}' (set force_retrain_models=True to retrain)",
                    "INFO",
                )
                skipped_models.append(candidate.name)
                reused_spec = existing_model_specs[candidate.name]
                model_specs[candidate.name] = reused_spec
                reused_summary, reused_fold_rows = self._load_reused_training_metrics(
                    candidate,
                    supports_predict_proba=bool(reused_spec.get("supports_predict_proba", candidate.supports_predict_proba)),
                )
                if reused_summary is not None:
                    training_rows.append(reused_summary)
                best_fold_rows.extend(reused_fold_rows)
                continue

            summary, fold_rows, model_spec, failure = self._train_candidate(
                candidate, X_train, y_train, cv, numeric_features, categorical_features
            )
            if failure is not None:
                failed_rows.append(failure)
                continue
            training_rows.append(summary)
            best_fold_rows.extend(fold_rows)
            model_specs[candidate.name] = model_spec

        # Log incremental training summary.
        if skipped_models:
            print_formatted_txt(
                f"Incremental training: {len(skipped_models)} models reused, "
                f"{len(training_rows)} newly trained",
                "INFO",
            )

        return self._finalize_training(training_rows, failed_rows, best_fold_rows, model_specs, skipped_models)
