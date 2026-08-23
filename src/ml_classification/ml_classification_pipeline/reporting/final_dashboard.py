"""Build a compact, source-backed final dashboard artifact.

The module deliberately separates data preparation from HTML rendering.  It
creates the canonical ``artifact.json`` payload consumed by the Data Analytics
portable dashboard builder.  This keeps the analytical transformations easy to
test while leaving layout, responsive behaviour, charts, tables, and print
fallbacks to the shared dashboard runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
import json
from pathlib import Path
import re
from typing import Any
import warnings

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

from ..core.persistence import ensure_dir, load_joblib
from .html import write_html_report


_INDEX_COLUMN_PATTERN = re.compile(r"^(NDVI|NDWI)_median__(\d{8})$")
_MODELLED_LABEL_COLUMN = "initial_label_code"
_MODELLED_LABEL_NAME_COLUMN = "initial_label"
_ID_COLUMN = "parcel_code"


@dataclass(frozen=True)
class DashboardPaths:
    """Resolved pipeline artifacts required by the final dashboard."""

    parcels: Path
    project: Path

    @property
    def prepare_summary(self) -> Path:
        return self.project / "02_prepare" / "prepare_summary.json"

    @property
    def training_summary(self) -> Path:
        return self.project / "03_train" / "training_summary.csv"

    @property
    def cv_fold_scores(self) -> Path:
        return self.project / "03_train" / "best_cv_fold_scores.csv"

    @property
    def selection_summary(self) -> Path:
        return self.project / "04_evaluate" / "selection_summary.json"

    @property
    def train_metrics(self) -> Path:
        return self.project / "04_evaluate" / "model_metrics_train_with_voting.csv"

    @property
    def test_metrics(self) -> Path:
        return self.project / "04_evaluate" / "model_metrics_test_with_voting.csv"

    def required(self) -> tuple[Path, ...]:
        """Return the complete, final-run source contract."""
        return (
            self.parcels,
            self.prepare_summary,
            self.training_summary,
            self.cv_fold_scores,
            self.selection_summary,
            self.train_metrics,
            self.test_metrics,
        )


def _clean_label_code(value: Any) -> str:
    """Normalize numeric and string class codes without introducing ``.0``."""
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _display_name(value: Any) -> str:
    """Convert source-style labels to short reader-facing names."""
    text = str(value).strip().replace("_", " ")
    return text.title() if text else "Unknown"


def _json_value(value: Any) -> Any:
    """Convert pandas/numpy scalars to JSON-safe values."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return records containing only JSON-native scalar values."""
    return [{str(key): _json_value(value) for key, value in row.items()} for row in frame.to_dict("records")]


def _require_complete_sources(paths: DashboardPaths) -> None:
    """Fail closed when a pipeline run has not produced its final artifacts."""
    missing = [path for path in paths.required() if not path.is_file()]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "The final dashboard requires a completed prepare/train/evaluate run. "
            f"Missing artifacts:\n{details}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def _index_columns(path: Path) -> list[str]:
    """Discover the twelve monthly median NDVI/NDWI columns from metadata."""
    names = pq.ParquetFile(path).schema.names
    columns = [name for name in names if _INDEX_COLUMN_PATTERN.fullmatch(name)]
    by_index = {index: [] for index in ("NDVI", "NDWI")}
    for column in columns:
        match = _INDEX_COLUMN_PATTERN.fullmatch(column)
        if match:
            by_index[match.group(1)].append(column)
    incomplete = {index: values for index, values in by_index.items() if len(values) < 8}
    if incomplete:
        raise ValueError(f"NDVI/NDWI time series are too sparse for a trend chart: {incomplete}")
    return sorted(columns)


def _read_parcels(paths: DashboardPaths, modelled_labels: set[str]) -> gpd.GeoDataFrame:
    """Read only the fields needed for the map and phenology views."""
    index_columns = _index_columns(paths.parcels)
    schema_names = set(pq.ParquetFile(paths.parcels).schema.names)
    required = {_ID_COLUMN, _MODELLED_LABEL_COLUMN, "geometry", *index_columns}
    missing = sorted(required - schema_names)
    if missing:
        raise ValueError(f"Parcel source is missing dashboard fields: {missing}")
    optional = [_MODELLED_LABEL_NAME_COLUMN] if _MODELLED_LABEL_NAME_COLUMN in schema_names else []
    frame = gpd.read_parquet(paths.parcels, columns=[*sorted(required), *optional])
    frame[_MODELLED_LABEL_COLUMN] = frame[_MODELLED_LABEL_COLUMN].map(_clean_label_code)
    return frame.loc[frame[_MODELLED_LABEL_COLUMN].isin(modelled_labels)].copy()


def _build_class_lookup(parcels: pd.DataFrame) -> dict[str, str]:
    """Create a stable class-code to class-name mapping."""
    if _MODELLED_LABEL_NAME_COLUMN not in parcels.columns:
        return {code: f"Class {code}" for code in sorted(parcels[_MODELLED_LABEL_COLUMN].unique())}
    lookup = (
        parcels.loc[:, [_MODELLED_LABEL_COLUMN, _MODELLED_LABEL_NAME_COLUMN]]
        .dropna(subset=[_MODELLED_LABEL_COLUMN])
        .drop_duplicates(_MODELLED_LABEL_COLUMN)
    )
    return {
        str(row[_MODELLED_LABEL_COLUMN]): _display_name(row[_MODELLED_LABEL_NAME_COLUMN])
        for _, row in lookup.iterrows()
    }


def _build_index_distribution_rows(parcels: pd.DataFrame, class_lookup: dict[str, str]) -> pd.DataFrame:
    """Reshape parcel-level monthly NDVI/NDWI values for distribution plots."""
    index_columns = [column for column in parcels.columns if _INDEX_COLUMN_PATTERN.fullmatch(str(column))]
    output_columns = ["class_code", "class_label", "index", "date", "parcel_value"]
    if not index_columns:
        return pd.DataFrame(columns=output_columns)
    long = parcels.melt(
        id_vars=[_MODELLED_LABEL_COLUMN],
        value_vars=index_columns,
        var_name="feature",
        value_name="parcel_value",
    )
    parsed = long["feature"].str.extract(_INDEX_COLUMN_PATTERN)
    long["index"] = parsed[0]
    long["date"] = pd.to_datetime(parsed[1], format="%Y%m%d", errors="raise")
    long["parcel_value"] = pd.to_numeric(long["parcel_value"], errors="coerce")
    long["class_code"] = long[_MODELLED_LABEL_COLUMN].astype(str)
    long["class_label"] = long["class_code"].map(class_lookup).fillna("Class " + long["class_code"])
    return long.dropna(subset=["parcel_value"]).loc[:, output_columns]


def _summarize_phenology(distribution: pd.DataFrame) -> pd.DataFrame:
    """Aggregate parcel-level index distributions into monthly class summaries."""
    if distribution.empty:
        return pd.DataFrame(
            columns=["class_code", "class_label", "index", "date", "lowest", "q25", "median", "q75", "highest", "parcels"]
        )
    grouped = (
        distribution.groupby(["class_code", "class_label", "index", "date"], observed=True)["parcel_value"]
        .agg(
            lowest="min",
            q25=lambda values: values.quantile(0.25),
            median="median",
            q75=lambda values: values.quantile(0.75),
            highest="max",
            parcels="count",
        )
        .reset_index()
    )
    grouped["date"] = grouped["date"].dt.strftime("%Y-%m-%d")
    return grouped.loc[
        :, ["class_code", "class_label", "index", "date", "lowest", "q25", "median", "q75", "highest", "parcels"]
    ]


def _build_seasonal_range_rows(phenology: pd.DataFrame) -> pd.DataFrame:
    """Return tidy monthly lowest/highest lines for each modeled class and index."""
    if phenology.empty:
        return pd.DataFrame(columns=["class_code", "class_label", "index", "date", "bound", "value", "parcels"])
    return pd.concat(
        [
            phenology.loc[:, ["class_code", "class_label", "index", "date", "parcels", bound]]
            .rename(columns={bound: "value"})
            .assign(bound=label)
            for bound, label in (("highest", "Highest"), ("lowest", "Lowest"))
        ],
        ignore_index=True,
    )


def _build_phenology_rows(parcels: pd.DataFrame, class_lookup: dict[str, str]) -> pd.DataFrame:
    """Aggregate monthly median NDVI/NDWI by modeled class."""
    return _summarize_phenology(_build_index_distribution_rows(parcels, class_lookup))


def _build_seasonal_summary(phenology: pd.DataFrame) -> pd.DataFrame:
    """Summarize each class by its peak monthly NDVI and NDWI."""
    peak_index = phenology.groupby(["class_code", "class_label", "index"], observed=True)["median"].idxmax()
    peaks = phenology.loc[peak_index, ["class_code", "class_label", "index", "date", "median", "parcels"]].copy()
    peaks["peak_month"] = pd.to_datetime(peaks["date"]).dt.strftime("%b %Y")
    values = peaks.pivot(index=["class_code", "class_label"], columns="index", values="median")
    months = peaks.pivot(index=["class_code", "class_label"], columns="index", values="peak_month")
    counts = peaks.groupby(["class_code", "class_label"], observed=True)["parcels"].max()
    result = pd.DataFrame(index=values.index)
    result["parcels"] = counts
    for index in ("NDVI", "NDWI"):
        result[f"{index.lower()}_peak"] = values.get(index)
        result[f"{index.lower()}_peak_month"] = months.get(index)
    return result.reset_index().sort_values("parcels", ascending=False)


def _build_cv_rows(training: pd.DataFrame, fold_scores: pd.DataFrame, selected_models: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the CV fold series and the exact ranked CV table."""
    required = {"model", "best_cv_score", "cv_score_std", "cv_ranking_metric"}
    if not required.issubset(training.columns):
        raise ValueError(f"training_summary.csv is missing fields: {sorted(required - set(training.columns))}")
    ranking = training.sort_values(["cv_ranking_metric", "best_cv_score"], ascending=False).reset_index(drop=True).copy()
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking["selected"] = ranking["model"].astype(str).map(lambda value: "Yes" if value in selected_models else "No")
    ranking = ranking.rename(
        columns={
            "best_cv_score": "mean_cv_score",
            "cv_score_std": "cv_std",
            "cv_ranking_metric": "selection_score",
        }
    )
    ranking_columns = ["rank", "model", "mean_cv_score", "cv_std", "selection_score", "selected"]
    ranking = ranking.loc[:, ranking_columns]

    fold_required = {"model", "fold", "score"}
    if not fold_required.issubset(fold_scores.columns):
        raise ValueError(f"best_cv_fold_scores.csv is missing fields: {sorted(fold_required - set(fold_scores.columns))}")
    folds = fold_scores.loc[:, ["model", "fold", "score"]].copy()
    folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int) + 1
    folds["selected"] = folds["model"].astype(str).map(lambda value: "Yes" if value in selected_models else "No")
    folds = folds.merge(ranking.loc[:, ["model", "rank"]], on="model", how="left", validate="many_to_one")
    return folds.sort_values(["rank", "fold"]), ranking


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    """Choose comparable evaluation metrics present in the completed run."""
    preferred = ["f1_macro", "balanced_accuracy", "accuracy", "f1_weighted"]
    return [column for column in preferred if column in frame.columns]


def _build_performance_rows(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build long-form charts and a wide train/test comparison table."""
    if "model" not in train.columns or "model" not in test.columns:
        raise ValueError("Evaluation metrics must include a model column.")
    metrics = [metric for metric in _metric_columns(train) if metric in test.columns]
    if not metrics:
        raise ValueError("No shared classification metrics were found in train/test evaluation outputs.")
    long_parts = []
    for split, frame in (("Train", train), ("Test", test)):
        part = frame.loc[:, ["model", *metrics]].melt(id_vars="model", var_name="metric", value_name="score")
        part["split"] = split
        part["metric_label"] = part["metric"].map(_display_name)
        long_parts.append(part)
    long = pd.concat(long_parts, ignore_index=True)

    train_wide = train.set_index("model")[metrics].add_prefix("train_")
    test_wide = test.set_index("model")[metrics].add_prefix("test_")
    wide = train_wide.join(test_wide, how="outer").reset_index()
    for metric in metrics:
        wide[f"{metric}_gap"] = wide[f"train_{metric}"] - wide[f"test_{metric}"]
    ordered_columns = [
        "model",
        *(column for metric in metrics for column in (f"train_{metric}", f"test_{metric}", f"{metric}_gap")),
    ]
    wide = wide.loc[:, ordered_columns]
    primary = "test_f1_macro" if "test_f1_macro" in wide.columns else f"test_{metrics[0]}"
    return long, wide.sort_values(primary, ascending=False)


def _strategy_model_name(selection: dict[str, Any], test_metrics: pd.DataFrame) -> str:
    """Resolve the final strategy row as persisted by the evaluation step."""
    model_values = test_metrics["model"].astype(str)
    if selection.get("selection_type") == "soft_voting":
        for candidate in ("soft_voting", "Voting", "voting"):
            if candidate in set(model_values):
                return candidate
    selected = [str(value) for value in selection.get("selected_models", [])]
    for candidate in selected:
        if candidate in set(model_values):
            return candidate
    if test_metrics.empty:
        raise ValueError("Test metrics are empty; a final strategy cannot be summarized.")
    metric = "f1_macro" if "f1_macro" in test_metrics.columns else _metric_columns(test_metrics)[0]
    return str(test_metrics.loc[pd.to_numeric(test_metrics[metric], errors="coerce").idxmax(), "model"])


def _filter_model_rows(frame: pd.DataFrame, allowed_models: set[str], source_name: str) -> pd.DataFrame:
    """Keep only explicitly selected or final-strategy model rows."""
    if "model" not in frame.columns:
        raise ValueError(f"{source_name} is missing the model column.")
    if not allowed_models:
        raise ValueError("selection_summary contains no selected_models for dashboard filtering.")
    filtered = frame.loc[frame["model"].astype(str).isin(allowed_models)].copy()
    if filtered.empty:
        raise ValueError(f"{source_name} contains none of the selected models: {sorted(allowed_models)}")
    return filtered


def _build_map_html(parcels: gpd.GeoDataFrame, source_id: str, max_points: int = 700) -> str:
    """Render a bounded offline study-area map with an OSM deep link."""
    if parcels.empty or parcels.crs is None:
        return "<section><h2>Study area</h2><p>Geometry or CRS is unavailable.</p></section>"
    geographic = parcels.to_crs(4326)
    points = geographic.geometry.representative_point()
    if len(points) > max_points:
        points = points.sample(max_points, random_state=42)
    min_x, min_y, max_x, max_y = geographic.total_bounds
    x_span = max(max_x - min_x, 1e-9)
    y_span = max(max_y - min_y, 1e-9)
    width, height, pad = 920.0, 390.0, 24.0

    circles = []
    for point in points:
        x = pad + ((point.x - min_x) / x_span) * (width - 2 * pad)
        y = pad + (1 - (point.y - min_y) / y_span) * (height - 2 * pad)
        circles.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='2.1' fill='#2e6f95' fill-opacity='.48'/>")
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    osm_url = f"https://www.openstreetmap.org/#map=10/{center_y:.5f}/{center_x:.5f}"
    grid = "".join(
        [f"<line x1='{x}' y1='{pad}' x2='{x}' y2='{height-pad}'/>" for x in (200, 380, 560, 740)]
        + [f"<line x1='{pad}' y1='{y}' x2='{width-pad}' y2='{y}'/>" for y in (100, 195, 290)]
    )
    return (
        "<section style='font-family:ui-sans-serif,system-ui,sans-serif'>"
        "<div style='display:flex;justify-content:space-between;gap:16px;align-items:flex-start;flex-wrap:wrap'>"
        "<div><h2 style='margin:0 0 5px'>Study area and parcel coverage</h2>"
        "<p style='margin:0 0 14px;color:#52606d'>Offline representative-point map; exact parcel geometries remain in the source GeoParquet.</p></div>"
        f"<a href='{escape(osm_url)}' target='_blank' rel='noreferrer' style='color:#1f5f86'>Open center in OpenStreetMap ↗</a>"
        "</div>"
        f"<svg viewBox='0 0 {width:.0f} {height:.0f}' role='img' aria-label='Parcel coverage map' "
        "style='display:block;width:100%;height:auto;background:#f7f8f5;border:1px solid #d9e2e8;border-radius:12px'>"
        f"<g stroke='#d9e2e8' stroke-width='1' stroke-dasharray='3 5'>{grid}</g>"
        + "".join(circles)
        + f"<text x='{pad}' y='{height-7}' font-size='12' fill='#52606d'>"
        f"{min_y:.3f}°N, {min_x:.3f}°E — {max_y:.3f}°N, {max_x:.3f}°E</text>"
        "</svg>"
        f"<p style='font-size:12px;color:#7b8794;margin:10px 0 0'>Source: {escape(source_id)}. "
        "The dashboard does not download background tiles, so it remains portable and works offline.</p>"
        "</section>"
    )


def _source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    generated_at: str,
    transformation_code: str,
) -> dict[str, Any]:
    """Create canonical file provenance shared by cards, charts, and tables."""
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": transformation_code,
            "description": description,
            "executed_at": generated_at,
            "tables_used": [path],
            "filters": ["Modeled classes from prepare_summary.target_labels", "No row-level parcel data embedded"],
        },
    }


def build_final_dashboard_artifact(
    parcels_path: str | Path,
    project_dir: str | Path,
    *,
    title: str = "Crop Classification — Final Dashboard",
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the validated-dashboard input from one completed pipeline run."""
    paths = DashboardPaths(Path(parcels_path), Path(project_dir))
    _require_complete_sources(paths)
    generated_at = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    prepare = _read_json(paths.prepare_summary)
    selection = _read_json(paths.selection_summary)
    modelled_labels = {_clean_label_code(value) for value in prepare.get("target_labels", [])}
    if not modelled_labels:
        raise ValueError("prepare_summary.json contains no modeled target_labels.")

    parcels = _read_parcels(paths, modelled_labels)
    if parcels.empty:
        raise ValueError("No parcel rows match the modeled target labels.")
    class_lookup = _build_class_lookup(parcels)
    phenology = _build_phenology_rows(parcels, class_lookup)
    seasonal = _build_seasonal_summary(phenology)
    seasonal_range = _build_seasonal_range_rows(phenology)

    training = pd.read_csv(paths.training_summary)
    fold_scores = pd.read_csv(paths.cv_fold_scores)
    train_metrics = pd.read_csv(paths.train_metrics)
    test_metrics = pd.read_csv(paths.test_metrics)
    selected_models = {str(value) for value in selection.get("selected_models", [])}
    strategy_model = _strategy_model_name(selection, test_metrics)
    training = _filter_model_rows(training, selected_models, "training_summary.csv")
    fold_scores = _filter_model_rows(fold_scores, selected_models, "best_cv_fold_scores.csv")
    performance_models = selected_models | {strategy_model}
    train_metrics = _filter_model_rows(train_metrics, performance_models, "train evaluation metrics")
    test_metrics = _filter_model_rows(test_metrics, performance_models, "test evaluation metrics")
    cv_folds, cv_ranking = _build_cv_rows(training, fold_scores, selected_models)
    performance, performance_table = _build_performance_rows(train_metrics, test_metrics)

    strategy_row = test_metrics.loc[test_metrics["model"].astype(str) == strategy_model].iloc[0]
    best_cv = cv_ranking.iloc[0]
    class_counts = (
        parcels.groupby(_MODELLED_LABEL_COLUMN, observed=True).size().rename("parcels").reset_index()
    )
    class_counts["class_code"] = class_counts[_MODELLED_LABEL_COLUMN].astype(str)
    class_counts["class_label"] = class_counts["class_code"].map(class_lookup)
    class_counts = class_counts.loc[:, ["class_code", "class_label", "parcels"]].sort_values("parcels", ascending=False)

    primary_metric = "f1_macro" if "f1_macro" in test_metrics.columns else _metric_columns(test_metrics)[0]
    train_strategy = train_metrics.loc[train_metrics["model"].astype(str) == strategy_model]
    train_strategy_score = float(train_strategy.iloc[0][primary_metric]) if not train_strategy.empty else None
    test_strategy_score = float(strategy_row[primary_metric])
    overview = [{
        "modeled_parcels": int(prepare.get("train_rows", 0)) + int(prepare.get("test_rows", 0)),
        "train_parcels": int(prepare.get("train_rows", 0)),
        "test_parcels": int(prepare.get("test_rows", 0)),
        "modeled_classes": len(modelled_labels),
        "best_cv_model": str(best_cv["model"]),
        "best_cv_score": float(best_cv["mean_cv_score"]),
        "strategy": "Soft voting" if selection.get("selection_type") == "soft_voting" else str(selection.get("selection_type")),
        "strategy_members": len(selected_models),
        "test_primary_score": test_strategy_score,
        "generalization_gap": (train_strategy_score - test_strategy_score) if train_strategy_score is not None else None,
    }]

    probability_rows = []
    optimization = selection.get("probability_optimization") or {}
    for metric, baseline_key, optimized_key in (
        ("Macro F1", "baseline_oof_macro_f1", "optimized_oof_macro_f1"),
        ("Accuracy", "baseline_oof_accuracy", "optimized_oof_accuracy"),
    ):
        if baseline_key in optimization and optimized_key in optimization:
            probability_rows.extend(
                [
                    {"metric": metric, "state": "Baseline", "score": float(optimization[baseline_key])},
                    {"metric": metric, "state": "Optimized", "score": float(optimization[optimized_key])},
                ]
            )

    logical_root = Path(paths.parcels).parent.parent.name or "study-area"
    satellite_path = f"{logical_root}/1_parcel_stats/{paths.parcels.name}"
    prepare_path = f"{logical_root}/{paths.project.name}/02_prepare/prepare_summary.json"
    training_path = f"{logical_root}/{paths.project.name}/03_train/training_summary.csv"
    folds_path = f"{logical_root}/{paths.project.name}/03_train/best_cv_fold_scores.csv"
    evaluation_path = f"{logical_root}/{paths.project.name}/04_evaluate/model_metrics_*_with_voting.csv"
    selection_path = f"{logical_root}/{paths.project.name}/04_evaluate/selection_summary.json"
    satellite_columns = ", ".join(f'"{column}"' for column in _index_columns(paths.parcels))
    modeled_label_sql = ", ".join(f"'{label}'" for label in sorted(modelled_labels))
    selected_model_sql = ", ".join(f"'{model.replace(chr(39), chr(39) * 2)}'" for model in sorted(selected_models))
    performance_model_sql = ", ".join(
        f"'{model.replace(chr(39), chr(39) * 2)}'" for model in sorted(performance_models)
    )
    satellite_sql = (
        f"WITH modeled AS (SELECT * FROM read_parquet('{satellite_path}') "
        f"WHERE CAST({_MODELLED_LABEL_COLUMN} AS VARCHAR) IN ({modeled_label_sql})), "
        f"long AS (UNPIVOT modeled ON {satellite_columns} INTO NAME feature VALUE parcel_value) "
        f"SELECT CAST({_MODELLED_LABEL_COLUMN} AS VARCHAR) AS class_code, feature, "
        "min(parcel_value) AS lowest, median(parcel_value) AS median, "
        "max(parcel_value) AS highest, quantile_cont(parcel_value, 0.25) AS q25, "
        "quantile_cont(parcel_value, 0.75) AS q75, count(parcel_value) AS parcels "
        "FROM long GROUP BY 1, 2"
    )
    full_sources = [
        _source(
            "satellite",
            "Parcel satellite statistics",
            satellite_path,
            "Monthly median NDVI/NDWI aggregated by modeled class; parcel geometry remains in the source dataset.",
            generated_at,
            satellite_sql,
        ),
        _source(
            "prepare",
            "Preparation summary",
            prepare_path,
            "Authoritative modeled label set, spatial split sizes, and fold definitions.",
            generated_at,
            f"SELECT train_rows, test_rows, target_labels, split_strategy FROM read_json_auto('{prepare_path}')",
        ),
        _source(
            "training",
            "Training and fold results",
            training_path,
            "Selected-model cross-validation mean, variation, and configured selection score.",
            generated_at,
            f"SELECT * FROM read_csv_auto('{training_path}') WHERE model IN ({selected_model_sql}) "
            "ORDER BY cv_ranking_metric DESC, best_cv_score DESC",
        ),
        _source(
            "folds",
            "Best CV score per fold",
            folds_path,
            "Fold-level score for each selected model's best hyperparameter setting.",
            generated_at,
            f"SELECT model, fold + 1 AS fold, score FROM read_csv_auto('{folds_path}') "
            f"WHERE model IN ({selected_model_sql})",
        ),
        _source(
            "evaluation",
            "Train and holdout metrics",
            evaluation_path,
            "Train/test classification metrics for selected models and the final strategy result.",
            generated_at,
            f"SELECT *, 'Train' AS split FROM read_csv_auto('{logical_root}/{paths.project.name}/04_evaluate/model_metrics_train_with_voting.csv') "
            f"WHERE model IN ({performance_model_sql}) UNION ALL "
            f"SELECT *, 'Test' AS split FROM read_csv_auto('{logical_root}/{paths.project.name}/04_evaluate/model_metrics_test_with_voting.csv') "
            f"WHERE model IN ({performance_model_sql})",
        ),
        _source(
            "selection",
            "Frozen model-selection summary",
            selection_path,
            "Final strategy, selected voting members, and out-of-fold probability optimization.",
            generated_at,
            f"SELECT * FROM read_json_auto('{selection_path}')",
        ),
    ]

    cards = [
        {"id": "modeled_parcels", "description": "Labeled parcels used by the authoritative spatial split.", "dataset": "overview", "sourceId": "prepare", "metrics": [{"label": "Modeled parcels", "field": "modeled_parcels", "format": "number"}]},
        {"id": "modeled_classes", "description": "Classes retained after rare-class checks.", "dataset": "overview", "sourceId": "prepare", "metrics": [{"label": "Modeled classes", "field": "modeled_classes", "format": "number"}]},
        {"id": "best_cv", "description": "Top model by the configured cross-validation ranking score.", "dataset": "overview", "sourceId": "training", "metrics": [{"label": "Best CV model", "field": "best_cv_model"}, {"label": "Mean CV score", "field": "best_cv_score", "format": "percent"}]},
        {"id": "strategy", "description": "Frozen final model strategy and number of component estimators.", "dataset": "overview", "sourceId": "selection", "metrics": [{"label": "Final strategy", "field": "strategy"}, {"label": "Members", "field": "strategy_members", "format": "number"}]},
        {"id": "holdout", "description": f"{_display_name(primary_metric)} on the spatial holdout set.", "dataset": "overview", "sourceId": "evaluation", "metrics": [{"label": f"Test {_display_name(primary_metric)}", "field": "test_primary_score", "format": "percent"}]},
        {"id": "gap", "description": "Train minus test score for the final strategy; smaller is more stable.", "dataset": "overview", "sourceId": "evaluation", "metrics": [{"label": "Generalization gap", "field": "generalization_gap", "format": "percent", "signed": True}]},
    ]

    charts = [
        {
            "id": "phenology",
            "title": "Monthly median vegetation index by crop class",
            "subtitle": "Median parcel value with monthly source coverage from Oct 2023 to Sep 2024; use the class and index filters.",
            "type": "line",
            "dataset": "phenology",
            "sourceId": "satellite",
            "encodings": {
                "x": {"field": "date", "type": "temporal", "label": "Month"},
                "y": {"field": "median", "type": "quantitative", "label": "Median index value", "format": "number"},
                "color": {"field": "class_label", "type": "nominal", "label": "Crop class"},
            },
            "yAxisTitle": "Median index value",
            "valueFormat": "number",
            "layout": "full",
        },
        {
            "id": "seasonal_range",
            "title": "Monthly highest and lowest index values by modeled class",
            "subtitle": "Highest and lowest parcel median in each month; choose one crop class and vegetation index.",
            "type": "line",
            "dataset": "seasonal_range",
            "sourceId": "satellite",
            "encodings": {
                "x": {"field": "date", "type": "temporal", "label": "Month"},
                "y": {"field": "value", "type": "quantitative", "label": "Index value", "format": "number"},
                "color": {"field": "bound", "type": "nominal", "label": "Monthly bound"},
            },
            "yAxisTitle": "Parcel median index value",
            "valueFormat": "number",
            "layout": "full",
        },
        {
            "id": "cv_folds",
            "title": "Selected-model cross-validation score by fold",
            "subtitle": "Best hyperparameter setting for each selected model; use the model filter to inspect fold-to-fold stability.",
            "type": "line",
            "dataset": "cv_folds",
            "sourceId": "folds",
            "encodings": {
                "x": {"field": "fold", "type": "ordinal", "label": "Fold"},
                "y": {"field": "score", "type": "quantitative", "label": "CV score", "format": "percent"},
                "color": {"field": "model", "type": "nominal", "label": "Model"},
            },
            "yAxisTitle": "CV score",
            "valueFormat": "percent",
        },
        {
            "id": "performance",
            "title": "Selected-model train and spatial holdout performance",
            "subtitle": "Selected models and the final strategy result only; lower train-test gaps indicate better generalization.",
            "type": "bar",
            "dataset": "performance",
            "sourceId": "evaluation",
            "encodings": {
                "x": {"field": "model", "type": "nominal", "label": "Model"},
                "y": {"field": "score", "type": "quantitative", "label": "Score", "format": "percent"},
                "color": {"field": "split", "type": "nominal", "label": "Split"},
            },
            "yAxisTitle": "Score",
            "valueFormat": "percent",
        },
    ]
    if probability_rows:
        charts.append(
            {
                "id": "probability_optimization",
                "title": "Out-of-fold probability optimization",
                "subtitle": "Baseline versus class-multiplier-optimized ensemble scores on out-of-fold predictions.",
                "type": "bar",
                "dataset": "probability_optimization",
                "sourceId": "selection",
                "encodings": {
                    "x": {"field": "metric", "type": "nominal", "label": "Metric"},
                    "y": {"field": "score", "type": "quantitative", "label": "Score", "format": "percent"},
                    "color": {"field": "state", "type": "nominal", "label": "State"},
                },
                "yAxisTitle": "Score",
                "valueFormat": "percent",
            }
        )

    tables = [
        {
            "id": "cv_ranking",
            "title": "Selected-model cross-validation ranking",
            "subtitle": "Only selected strategy members are shown; selection score is the configured ranking metric.",
            "dataset": "cv_ranking",
            "sourceId": "training",
            "defaultSort": {"field": "rank", "direction": "asc"},
            "density": "dense",
            "columns": [
                {"field": "rank", "label": "Rank", "format": "number"},
                {"field": "model", "label": "Model", "type": "text"},
                {"field": "mean_cv_score", "label": "Mean CV", "format": "percent"},
                {"field": "cv_std", "label": "CV std", "format": "percent"},
                {"field": "selection_score", "label": "Selection score", "format": "percent"},
                {"field": "selected", "label": "Voting member", "type": "text"},
            ],
        },
        {
            "id": "performance_table",
            "title": "Exact selected-model train and holdout metrics",
            "subtitle": "Selected models plus the final strategy result; gap fields equal train minus test.",
            "dataset": "performance_table",
            "sourceId": "evaluation",
            "defaultSort": {"field": f"test_{primary_metric}", "direction": "desc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "model", "label": "Model", "type": "text"},
                *[
                    {
                        "field": field,
                        "label": _display_name(field),
                        "format": "percent",
                        **({"movement": True} if field.startswith("test_") else {}),
                    }
                    for field in performance_table.columns
                    if field != "model"
                ],
            ],
        },
    ]

    default_class = str(class_counts.iloc[0]["class_label"])
    default_cv_model = str(best_cv["model"])
    blocks: list[dict[str, Any]] = [
        {"id": "metrics", "type": "metric-strip", "cardIds": [card["id"] for card in cards]},
        {"id": "phenology", "type": "chart", "chartId": "phenology", "layout": "full"},
        {"id": "seasonal", "type": "chart", "chartId": "seasonal_range", "layout": "full"},
        {"id": "cv_folds", "type": "chart", "chartId": "cv_folds"},
        {"id": "performance", "type": "chart", "chartId": "performance"},
    ]
    if probability_rows:
        blocks.append({"id": "probability", "type": "chart", "chartId": "probability_optimization"})
    blocks.extend(
        [
            {"id": "cv_table", "type": "table", "tableId": "cv_ranking", "layout": "full"},
            {"id": "performance_table", "type": "table", "tableId": "performance_table", "layout": "full"},
            {
                "id": "methodology",
                "type": "markdown",
                "body": (
                    "## Reading notes\n\n"
                    f"The raw parcel source contains additional rare labels, while model preparation retained **{len(modelled_labels)} supported classes**. "
                    "All class trends and counts use that modeled population. The authoritative split sizes come from `prepare_summary.json`; "
                    "the upstream parcel `set_type` field is not used. NDVI/NDWI distribution data include monthly lowest, quartiles, median, "
                    "and highest parcel medians by class. The live pipeline report uses the generated OpenStreetMap-backed map only."
                ),
            },
        ]
    )

    manifest = {
        "version": 1,
        "surface": "dashboard",
        "title": title,
        "description": "Study-area coverage, crop phenology, cross-validation stability, holdout performance, and final ensemble selection.",
        "generatedAt": generated_at,
        "filters": [
            {"id": "crop_class", "label": "Crop class", "dataset": "phenology", "field": "class_label", "defaultValue": default_class, "includeAll": False, "targets": [{"dataset": "phenology", "field": "class_label"}, {"dataset": "seasonal_range", "field": "class_label"}]},
            {"id": "vegetation_index", "label": "Vegetation index", "dataset": "phenology", "field": "index", "defaultValue": "NDVI", "includeAll": False, "targets": [{"dataset": "phenology", "field": "index"}, {"dataset": "seasonal_range", "field": "index"}]},
            {"id": "cv_model", "label": "CV model", "dataset": "cv_folds", "field": "model", "defaultValue": default_cv_model, "includeAll": True, "targets": [{"dataset": "cv_folds", "field": "model"}]},
            {"id": "performance_metric", "label": "Performance metric", "dataset": "performance", "field": "metric", "defaultValue": primary_metric, "includeAll": False, "targets": [{"dataset": "performance", "field": "metric"}]},
        ],
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": full_sources,
        "blocks": blocks,
    }
    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "overview": overview,
            "class_counts": _records(class_counts),
            "phenology": _records(phenology),
            "seasonal_summary": _records(seasonal),
            "seasonal_range": _records(seasonal_range),
            "cv_folds": _records(cv_folds),
            "cv_ranking": _records(cv_ranking),
            "performance": _records(performance),
            "performance_table": _records(performance_table),
            "probability_optimization": probability_rows,
        },
    }
    return {"surface": "dashboard", "manifest": manifest, "snapshot": snapshot, "sources": full_sources}


def _pipeline_parcels(config, prepare: dict[str, Any]) -> tuple[gpd.GeoDataFrame, dict[str, str]]:
    """Load and normalize the check-stage dataset for automatic reporting."""
    source = load_joblib(config.check_dir / "input_dataset.joblib")
    if config.target_column not in source.columns:
        raise ValueError(f"Input dataset is missing target column {config.target_column!r}.")

    parcels = source.copy()
    parcels[_MODELLED_LABEL_COLUMN] = parcels[config.target_column].map(_clean_label_code)
    modelled_labels = {_clean_label_code(value) for value in prepare.get("target_labels", [])}
    parcels = parcels.loc[parcels[_MODELLED_LABEL_COLUMN].isin(modelled_labels)].copy()

    label_name_candidates = [
        config.target_column.removesuffix("_code"),
        "initial_label",
        "label_name",
        "class_name",
    ]
    label_name_column = next(
        (
            column
            for column in label_name_candidates
            if column != config.target_column and column in parcels.columns
        ),
        None,
    )
    if label_name_column:
        parcels[_MODELLED_LABEL_NAME_COLUMN] = parcels[label_name_column]
    class_lookup = _build_class_lookup(parcels)
    return parcels, class_lookup


def _save_monthly_index_boxplots(distribution: pd.DataFrame, output_path: Path) -> None:
    """Save one 12-row figure with NDVI left, NDWI right, and class boxes on x."""
    plot = distribution.loc[distribution["index"].isin(["NDVI", "NDWI"])].copy()
    if plot.empty:
        return
    ensure_dir(output_path.parent)
    plot["date"] = pd.to_datetime(plot["date"])
    month_order = sorted(plot["date"].drop_duplicates().tolist())
    class_order = (
        plot.loc[:, ["class_code", "class_label"]]
        .drop_duplicates()
        .sort_values(["class_code", "class_label"], kind="stable")["class_label"]
        .tolist()
    )
    index_order = ("NDVI", "NDWI")
    colors = {"NDVI": "#2e6f95", "NDWI": "#d99a2b"}

    fig, axes = plt.subplots(
        len(month_order),
        2,
        figsize=(18, 2.5 * len(month_order) + 2.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    median_color = "#17212b"
    for row_index, month in enumerate(month_order):
        month_label = pd.Timestamp(month).strftime("%b %Y")
        for column_index, index in enumerate(index_order):
            ax = axes[row_index, column_index]
            panel_rows = plot.loc[(plot["date"] == month) & (plot["index"] == index)]
            if not panel_rows.empty:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="vert: bool will be deprecated.*",
                        category=PendingDeprecationWarning,
                    )
                    sns.boxplot(
                        data=panel_rows,
                        x="class_label",
                        y="parcel_value",
                        order=class_order,
                        ax=ax,
                        color=colors[index],
                        width=0.64,
                        linewidth=0.8,
                        showfliers=False,
                        medianprops={"color": median_color, "linewidth": 1.35},
                        whiskerprops={"color": "#52606d", "linewidth": 0.8},
                        capprops={"color": "#52606d", "linewidth": 0.8},
                        boxprops={"edgecolor": "#52606d"},
                    )
                ax.text(
                    0.01,
                    0.93,
                    f"n={len(panel_rows):,}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color="#52606d",
                )
                if panel_rows["parcel_value"].nunique(dropna=True) <= 1:
                    constant_value = float(panel_rows["parcel_value"].iloc[0])
                    ax.text(
                        0.99,
                        0.93,
                        f"No variation ({constant_value:.2f})",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=7,
                        color="#8a3b2f",
                    )
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="#64748b")

            if row_index == 0:
                ax.set_title(index, loc="left", fontsize=13, fontweight="bold", color=colors[index])
            else:
                ax.set_title("")
            ax.axhline(0, color="#7b8794", linewidth=0.7, linestyle="--", zorder=0)
            ax.set_xlabel("")
            ax.set_ylabel(f"{month_label}\nIndex value" if column_index == 0 else "")
            ax.set_ylim(-1, 1)
            ax.grid(axis="y", color="#d9e2e8", linewidth=0.6)
            ax.grid(axis="x", visible=False)
            is_bottom_row = row_index == len(month_order) - 1
            ax.tick_params(axis="x", labelsize=7, labelbottom=is_bottom_row, rotation=58)
            if is_bottom_row:
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment("right")
                    label.set_rotation_mode("anchor")
            ax.tick_params(axis="y", labelsize=8)

    first_month = pd.Timestamp(month_order[0]).strftime("%b %Y")
    last_month = pd.Timestamp(month_order[-1]).strftime("%b %Y")
    fig.suptitle(
        "Monthly NDVI and NDWI distributions by modeled class",
        x=0.075,
        y=0.995,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.978,
        f"Parcel-level values, {first_month}-{last_month}, fixed index scale -1 to 1. "
        "Each row is one month: NDVI is left and NDWI is right. Classes share the x-axis; "
        "boxes show IQR and median, whiskers use 1.5 x IQR, and outlier markers are hidden.",
        ha="left",
        va="top",
        fontsize=9,
        color="#52606d",
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.105, top=0.945, hspace=0.18, wspace=0.08)
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_monthly_index_range_lines(phenology: pd.DataFrame, index: str, output_path: Path) -> None:
    """Save monthly highest/lowest parcel lines faceted by modeled class."""
    plot = phenology.loc[phenology["index"] == index].copy()
    if plot.empty:
        return
    ensure_dir(output_path.parent)
    plot["date"] = pd.to_datetime(plot["date"])
    class_order = (
        plot.groupby("class_label", observed=True)["parcels"].max().sort_values(ascending=False).index.tolist()
    )
    ncols = min(4, max(1, len(class_order)))
    nrows = int(np.ceil(len(class_order) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 3.35 * nrows + 1.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for panel_index, (ax, class_label) in enumerate(zip(axes.flat, class_order)):
        rows = plot.loc[plot["class_label"] == class_label].sort_values("date")
        ax.plot(
            rows["date"],
            rows["highest"],
            color="#2e6f95",
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            label="Highest",
        )
        ax.plot(
            rows["date"],
            rows["lowest"],
            color="#d99a2b",
            marker="o",
            markerfacecolor="white",
            markersize=3.5,
            linewidth=1.5,
            linestyle="--",
            label="Lowest",
        )
        parcel_count = int(rows["parcels"].max())
        ax.set_title(f"{class_label} (n={parcel_count:,})", loc="left", fontsize=9, fontweight="bold")
        ax.axhline(0, color="#7b8794", linewidth=0.7, linestyle=":", zorder=0)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("")
        ax.set_ylabel(index if panel_index % ncols == 0 else "")
        ax.grid(axis="y", color="#d9e2e8", linewidth=0.6)
        ax.grid(axis="x", visible=False)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%y"))
        ax.tick_params(axis="x", labelsize=7, rotation=0)
        ax.tick_params(axis="y", labelsize=8)
        if rows[["lowest", "highest"]].nunique(dropna=True).max() <= 1:
            constant_value = float(rows["highest"].iloc[0])
            ax.text(
                0.98,
                0.92,
                f"No variation ({constant_value:.2f})",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="#8a3b2f",
            )
    for ax in axes.flat[len(class_order) :]:
        ax.set_visible(False)
    first_month = plot["date"].min().strftime("%b %Y")
    last_month = plot["date"].max().strftime("%b %Y")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.975), frameon=False, ncol=2)
    fig.suptitle(
        f"Monthly highest and lowest {index} by modeled class",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.968,
        f"Highest and lowest parcel median in each class and month, {first_month}–{last_month}; fixed index scale −1 to 1.",
        ha="left",
        va="top",
        fontsize=9,
        color="#52606d",
    )
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.055, top=0.925, hspace=0.43, wspace=0.18)
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_train_test_plot(performance: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Save a horizontal train-versus-holdout comparison for one metric."""
    plot = performance.loc[performance["metric"] == metric].copy()
    if plot.empty:
        return
    ensure_dir(output_path.parent)
    wide = plot.pivot(index="model", columns="split", values="score")
    sort_column = "Test" if "Test" in wide.columns else wide.columns[0]
    wide = wide.sort_values(sort_column, ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(6, len(wide) * 0.55)))
    wide.plot(kind="barh", ax=ax, color={"Train": "#2e6f95", "Test": "#d99a2b"}, width=0.72)
    ax.set_xlim(0, 1)
    ax.set_xlabel(_display_name(metric))
    ax.set_ylabel("")
    ax.set_title(f"Train versus test holdout — {_display_name(metric)}", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#d9e2e8", linewidth=0.7)
    ax.legend(title="Split", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_selected_cv_fold_plot(cv_folds: pd.DataFrame, output_path: Path) -> None:
    """Save fold stability for selected strategy members only."""
    if cv_folds.empty:
        return
    ensure_dir(output_path.parent)
    model_order = (
        cv_folds.loc[:, ["model", "rank"]]
        .drop_duplicates("model")
        .sort_values("rank")["model"]
        .astype(str)
        .tolist()
    )
    palette = ["#2e6f95", "#d99a2b", "#7a9a48", "#b279a2", "#c66b3d"]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for model_index, model in enumerate(model_order):
        rows = cv_folds.loc[cv_folds["model"].astype(str) == model].sort_values("fold")
        ax.plot(
            rows["fold"],
            rows["score"],
            color=palette[model_index % len(palette)],
            marker="o",
            markersize=5,
            linewidth=1.8,
            label=model,
        )
    scores = pd.to_numeric(cv_folds["score"], errors="coerce").dropna()
    if not scores.empty and scores.between(0, 1).all():
        ax.set_ylim(0, 1)
    ax.set_xticks(sorted(pd.to_numeric(cv_folds["fold"], errors="coerce").dropna().unique()))
    ax.set_xlabel("CV fold")
    ax.set_ylabel("Validation score")
    ax.set_title("Cross-validation stability of selected models", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#d9e2e8", linewidth=0.7)
    ax.legend(title="Selected model", frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_pipeline_final_dashboard(
    config,
    train_metrics_df: pd.DataFrame,
    test_metrics_df: pd.DataFrame,
    selection: dict[str, Any],
) -> Path:
    """Create the final HTML dashboard automatically after evaluation."""
    dashboard_dir = ensure_dir(config.final_dashboard_dir)
    plots_dir = ensure_dir(dashboard_dir / "plots")
    data_dir = ensure_dir(dashboard_dir / "data")
    prepare = _read_json(config.prepare_dir / "prepare_summary.json")
    parcels, class_lookup = _pipeline_parcels(config, prepare)

    index_columns = [column for column in parcels.columns if _INDEX_COLUMN_PATTERN.fullmatch(str(column))]
    phenology = pd.DataFrame()
    seasonal_range = pd.DataFrame()
    seasonal_images: list[dict[str, Any]] = []
    seasonal_quality_notes: list[str] = []
    if index_columns:
        distribution = _build_index_distribution_rows(parcels, class_lookup)
        phenology = _summarize_phenology(distribution)
        seasonal_range = _build_seasonal_range_rows(phenology)
        phenology.to_csv(data_dir / "monthly_index_by_class.csv", index=False)
        seasonal_range.to_csv(data_dir / "monthly_index_high_low_by_class.csv", index=False)
        _save_monthly_index_boxplots(
            distribution,
            plots_dir / "seasonal_ndvi_ndwi_boxplots_by_month_class.png",
        )
        _save_monthly_index_range_lines(
            phenology,
            "NDVI",
            plots_dir / "seasonal_ndvi_high_low_by_class.png",
        )
        _save_monthly_index_range_lines(
            phenology,
            "NDWI",
            plots_dir / "seasonal_ndwi_high_low_by_class.png",
        )
        for index in ("NDVI", "NDWI"):
            values = distribution.loc[distribution["index"] == index, "parcel_value"]
            if not values.empty and values.nunique(dropna=True) <= 1:
                seasonal_quality_notes.append(
                    f"Data-quality warning: every {index} value in this evaluation input equals {float(values.iloc[0]):.2f}, "
                    "so its monthly boxes have no spread."
                )
        for stale_plot in (
            "seasonal_peak_by_class.png",
            "seasonal_ndvi_boxplots_by_class_month.png",
            "seasonal_ndwi_boxplots_by_class_month.png",
        ):
            (plots_dir / stale_plot).unlink(missing_ok=True)
        seasonal_images = [
            {
                "title": "Monthly NDVI and NDWI Distributions by Modeled Class",
                "path": plots_dir / "seasonal_ndvi_ndwi_boxplots_by_month_class.png",
            },
            {
                "title": "Monthly Highest and Lowest NDVI by Modeled Class",
                "path": plots_dir / "seasonal_ndvi_high_low_by_class.png",
            },
            {
                "title": "Monthly Highest and Lowest NDWI by Modeled Class",
                "path": plots_dir / "seasonal_ndwi_high_low_by_class.png",
            },
        ]

    training = pd.read_csv(config.train_dir / "training_summary.csv")
    fold_scores = pd.read_csv(config.train_dir / "best_cv_fold_scores.csv")
    selected_models = {str(value) for value in selection.get("selected_models", [])}
    strategy_model = _strategy_model_name(selection, test_metrics_df)
    training = _filter_model_rows(training, selected_models, "training_summary.csv")
    fold_scores = _filter_model_rows(fold_scores, selected_models, "best_cv_fold_scores.csv")
    performance_models = selected_models | {strategy_model}
    selected_train_metrics = _filter_model_rows(train_metrics_df, performance_models, "train evaluation metrics")
    selected_test_metrics = _filter_model_rows(test_metrics_df, performance_models, "test evaluation metrics")
    cv_folds, cv_ranking = _build_cv_rows(training, fold_scores, selected_models)
    performance, performance_table = _build_performance_rows(selected_train_metrics, selected_test_metrics)
    primary_metric = config.scoring_primary if config.scoring_primary in performance["metric"].unique() else "f1_macro"
    if primary_metric not in performance["metric"].unique():
        primary_metric = str(performance["metric"].iloc[0])
    _save_selected_cv_fold_plot(cv_folds, plots_dir / "selected_model_cv_folds.png")
    _save_train_test_plot(performance, primary_metric, plots_dir / "train_test_model_comparison.png")
    cv_ranking.to_csv(data_dir / "cv_model_ranking.csv", index=False)
    performance_table.to_csv(data_dir / "train_test_model_metrics.csv", index=False)

    strategy_test = selected_test_metrics.loc[selected_test_metrics["model"].astype(str) == strategy_model].iloc[0]
    strategy_train = selected_train_metrics.loc[selected_train_metrics["model"].astype(str) == strategy_model]
    train_score = float(strategy_train.iloc[0][primary_metric]) if not strategy_train.empty else np.nan
    test_score = float(strategy_test[primary_metric])
    best_cv = cv_ranking.iloc[0]
    summary = {
        "modeled_parcels": int(prepare.get("train_rows", 0)) + int(prepare.get("test_rows", 0)),
        "train_parcels": int(prepare.get("train_rows", 0)),
        "spatial_holdout_parcels": int(prepare.get("test_rows", 0)),
        "modeled_classes": len(prepare.get("target_labels", [])),
        "best_cv_model": best_cv["model"],
        "best_mean_cv_score": float(best_cv["mean_cv_score"]),
        "final_strategy": selection.get("selection_type"),
        "selected_models": list(selection.get("selected_models", [])),
        f"test_{primary_metric}": test_score,
        f"train_test_{primary_metric}_gap": float(train_score - test_score) if not np.isnan(train_score) else np.nan,
    }

    probability_optimization = selection.get("probability_optimization") or {}
    probability_table = pd.DataFrame(
        [
            {
                "metric": "Macro F1",
                "baseline": probability_optimization.get("baseline_oof_macro_f1"),
                "optimized": probability_optimization.get("optimized_oof_macro_f1"),
            },
            {
                "metric": "Accuracy",
                "baseline": probability_optimization.get("baseline_oof_accuracy"),
                "optimized": probability_optimization.get("optimized_oof_accuracy"),
            },
        ]
    ).dropna(how="all", subset=["baseline", "optimized"])

    map_embeds = []
    osm_map = config.check_dir / "plots" / "known_labels_map_osm.html"
    if osm_map.exists():
        map_embeds.append({"title": "Known Modeled Parcels on OpenStreetMap", "path": osm_map})
    data_links = [
        {"label": "CV Model Ranking", "path": data_dir / "cv_model_ranking.csv"},
        {"label": "Train-Test Model Metrics", "path": data_dir / "train_test_model_metrics.csv"},
    ]
    if not phenology.empty:
        data_links[0:0] = [
            {"label": "Monthly NDVI/NDWI by Class", "path": data_dir / "monthly_index_by_class.csv"},
            {"label": "Monthly High/Low Index Values by Class", "path": data_dir / "monthly_index_high_low_by_class.csv"},
        ]
    test_metric_column_styles = {
        str(column): "success" for column in performance_table.columns if str(column).startswith("test_")
    }
    gap_metric_cell_styles = {
        str(column): {"positive": "danger", "negative": "success"}
        for column in performance_table.columns
        if str(column).endswith("_gap")
    }
    voting_test_cell_styles = [
        {
            "column": "model",
            "values": [strategy_model],
            "target_columns": list(test_metric_column_styles),
            "style": "warning",
        }
    ]

    sections: list[dict[str, Any]] = [
        {"title": "Evaluation Snapshot", "kv": summary},
        {
            "title": "Study Area and Parcel Coverage",
            "text": "The interactive map uses OpenStreetMap tiles when network access is available.",
            "embeds": map_embeds,
        },
        {
            "title": "Seasonal Crop Profiles",
            "text": (
                "The combined Seaborn figure has one row per month and two columns: NDVI on the left and NDWI "
                "on the right. Every panel places modeled class/label on the shared x-axis and the parcel-level "
                "index distribution on the y-axis. "
                "Boxes show the interquartile range and median; whiskers use 1.5×IQR and outlier markers are hidden for readability. "
                "The high/low line figures show each class's highest and lowest parcel median in every month. "
                + " ".join(seasonal_quality_notes)
            ),
            "images": seasonal_images,
        },
        {
            "title": "Cross-Validation Stability",
            "text": "Only selected strategy members are shown. Fold scores use each selected model's best hyperparameter setting; the table follows the configured CV ranking score.",
            "images": [
                {"title": "Selected-Model CV Score per Fold", "path": plots_dir / "selected_model_cv_folds.png"}
            ],
            "table": cv_ranking,
            "row_styles_where": _voting_member_row_styles_for_pipeline(cv_ranking, selected_models),
        },
        {
            "title": "Train versus Spatial Holdout",
            "text": "The chart contains only selected models and the final strategy result, using the primary selection metric. Test/holdout values are green for selected models and yellow for the final voting result. Each gap is train minus test: negative gaps are green because test is higher, positive gaps are red because test is lower, and zero is neutral.",
            "images": [
                {"title": "Train versus Spatial Holdout", "path": plots_dir / "train_test_model_comparison.png"}
            ],
            "table": performance_table,
            "column_styles": test_metric_column_styles,
            "numeric_cell_styles": gap_metric_cell_styles,
            "cell_styles_where": voting_test_cell_styles,
            "highlight_rows_where": {"column": "model", "values": [strategy_model]},
        },
        {
            "title": "Best Model and Voting Result",
            "kv": {
                "selection_type": selection.get("selection_type"),
                "selected_models": list(selection.get("selected_models", [])),
                "selected_metric": selection.get("selected_metric"),
                "selected_cv_score": selection.get("selected_score"),
                f"holdout_{primary_metric}": test_score,
            },
            "table": probability_table,
        },
        {
            "title": "Dashboard Data",
            "links": data_links,
        },
    ]
    output_path = dashboard_dir / "report.html"
    write_html_report(
        output_path,
        "Final Crop Classification Dashboard",
        "Study-area coverage, crop seasonality, cross-validation stability, holdout performance, and final model selection.",
        sections,
    )
    return output_path


def _voting_member_row_styles_for_pipeline(table: pd.DataFrame, selected_models: set[str]) -> list[dict[str, Any]]:
    """Return report-renderer row styles for selected and non-selected CV models."""
    models = table["model"].astype(str).tolist() if "model" in table.columns else []
    return [
        {"column": "model", "values": [model for model in models if model in selected_models], "style": "success"},
        {"column": "model", "values": [model for model in models if model not in selected_models], "style": "muted"},
    ]


def write_final_dashboard_artifact(
    output_path: str | Path,
    parcels_path: str | Path,
    project_dir: str | Path,
    *,
    title: str = "Crop Classification — Final Dashboard",
) -> Path:
    """Build and write the canonical dashboard artifact JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_final_dashboard_artifact(parcels_path, project_dir, title=title)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final crop-classification dashboard artifact.")
    parser.add_argument("--parcels", required=True, type=Path, help="Source GeoParquet with monthly parcel statistics.")
    parser.add_argument("--project-dir", required=True, type=Path, help="Completed ML classification project directory.")
    parser.add_argument("--output", required=True, type=Path, help="Destination artifact.json path.")
    parser.add_argument("--title", default="Crop Classification — Final Dashboard")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = _parse_args()
    output = write_final_dashboard_artifact(args.output, args.parcels, args.project_dir, title=args.title)
    print(f"Wrote dashboard artifact: {output}")


if __name__ == "__main__":
    main()
