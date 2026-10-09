"""TensorFlow LSTM classifier and temporal tensor builders for parcel time-series features."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

TEMPORAL_FEATURE_PATTERN = re.compile(r"^(?P<base>.+?)__(?P<date>\d{8})$")

# Keep this helper focused on a single transformation so the reporting pipeline stays easy to follow.



def _safe_import(module_name: str):
    """Import a module by name and return ``None`` when import fails."""
    try:
        module = __import__(module_name, fromlist=[module_name.rsplit(".", maxsplit=1)[-1]])
    except Exception:
        logging.getLogger(__name__).debug("Error: _safe_import failed; using its fallback.", exc_info=True)
        return None
    return module


@dataclass(slots=True)
class TemporalTensorSchema:
    """Describe deterministic tensor ordering used by the LSTM wrapper."""

    dates: list[str]
    base_features: list[str]
    temporal_columns: list[str]
    temporal_frequency: str | None
    ignored_features: list[str]


class TemporalTensorBuilder:
    """Parse flattened temporal columns and create (samples, timesteps, features) tensors."""

    def __init__(
        self,
        temporal_statistics: tuple[str, ...] | None = ("median",),
        require_complete_timesteps: bool = True,
        temporal_frequency: str | None = "monthly",
        min_timesteps: int = 6,
    ):
        """Store parsing options controlling how temporal columns are discovered."""
        self.temporal_statistics = temporal_statistics
        self.require_complete_timesteps = require_complete_timesteps
        self.temporal_frequency = temporal_frequency
        self.min_timesteps = min_timesteps

    def fit(self, frame: pd.DataFrame) -> TemporalTensorBuilder:
        """Infer temporal schema from dataframe columns."""
        entries = self._parse_temporal_entries(frame.columns)
        if not entries:
            raise ValueError(
                "Error: No temporal features were detected for keras_lstm. Expected feature names like NDVI_median__20250101."
            )

        ordered_dates = self._ordered_dates(entries)
        self._validate_temporal_dates(ordered_dates)
        ordered_bases = self._ordered_bases(entries)

        if len(ordered_dates) < int(self.min_timesteps):
            raise ValueError(f"Error: keras_lstm requires at least {self.min_timesteps} timesteps, found {len(ordered_dates)}.")

        expected_columns = self._expected_temporal_columns(ordered_dates, ordered_bases)
        missing_columns = [column for column in expected_columns if column not in frame.columns]
        if missing_columns and self.require_complete_timesteps:
            missing_preview = missing_columns[:20]
            raise ValueError(
                f"Error: Temporal feature grid is incomplete for keras_lstm. Missing {len(missing_columns)} columns, examples: {missing_preview}"
            )

        available_columns = [column for column in expected_columns if column in frame.columns]
        if not available_columns:
            raise ValueError("Error: No valid temporal columns remained for keras_lstm after schema validation.")

        ignored_features = self._ignored_features(frame.columns, available_columns)
        self.schema_ = TemporalTensorSchema(
            dates=ordered_dates,
            base_features=ordered_bases,
            temporal_columns=available_columns,
            temporal_frequency=self.temporal_frequency,
            ignored_features=ignored_features,
        )
        self.temporal_index_ = self._build_temporal_index(self.schema_)
        return self

    @staticmethod
    def _ordered_dates(entries: list[tuple[str, str]]) -> list[str]:
        """Return sorted temporal dates from parsed entries."""
        return sorted({date_key for _, date_key in entries})

    @staticmethod
    def _ordered_bases(entries: list[tuple[str, str]]) -> list[str]:
        """Keep first-seen base-feature order from parsed entries."""
        ordered_bases: list[str] = []
        seen_bases: set[str] = set()
        for base_name, _ in entries:
            if base_name in seen_bases:
                continue
            seen_bases.add(base_name)
            ordered_bases.append(base_name)
        return ordered_bases

    @staticmethod
    def _expected_temporal_columns(ordered_dates: list[str], ordered_bases: list[str]) -> list[str]:
        """Build dense expected grid of temporal columns."""
        return [f"{base_name}__{date_key}" for date_key in ordered_dates for base_name in ordered_bases]

    @staticmethod
    def _ignored_features(all_columns, available_columns: list[str]) -> list[str]:
        """List non-temporal or excluded columns from schema."""
        available = set(available_columns)
        return [str(column) for column in all_columns if str(column) not in available]

    @staticmethod
    def _build_temporal_index(schema: TemporalTensorSchema) -> dict[str, tuple[int, int]]:
        """Map temporal column names to tensor timestep/feature indices."""
        date_lookup = {date_key: index for index, date_key in enumerate(schema.dates)}
        base_lookup = {base_name: index for index, base_name in enumerate(schema.base_features)}
        return {
            column: (
                date_lookup[column.rsplit("__", maxsplit=1)[1]],
                base_lookup[column.rsplit("__", maxsplit=1)[0]],
            )
            for column in schema.temporal_columns
        }

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Convert dataframe into a temporal tensor matching the fitted schema."""
        if not hasattr(self, "schema_"):
            raise RuntimeError("Error: TemporalTensorBuilder must be fitted before calling transform().")

        missing_required = [column for column in self.schema_.temporal_columns if column not in frame.columns]
        if missing_required:
            missing_preview = missing_required[:20]
            raise ValueError(
                f"Error: Prediction input is missing temporal columns required by keras_lstm. Missing {len(missing_required)} columns, examples: {missing_preview}"
            )

        n_rows = len(frame)
        n_timesteps = len(self.schema_.dates)
        n_features = len(self.schema_.base_features)
        tensor = np.full((n_rows, n_timesteps, n_features), np.nan, dtype=np.float32)

        for column, (timestep_index, feature_index) in self.temporal_index_.items():
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32, copy=False)
            tensor[:, timestep_index, feature_index] = values
        return tensor

    def to_schema_dict(self) -> dict[str, object]:
        """Return a JSON-serializable temporal schema payload."""
        if not hasattr(self, "schema_"):
            raise RuntimeError("Error: TemporalTensorBuilder must be fitted before schema export.")
        return {
            "dates": list(self.schema_.dates),
            "features": list(self.schema_.base_features),
            "temporal_columns": list(self.schema_.temporal_columns),
            "temporal_frequency": self.schema_.temporal_frequency,
            "ignored_features": list(self.schema_.ignored_features),
            "n_timesteps": len(self.schema_.dates),
            "n_features_per_timestep": len(self.schema_.base_features),
        }

    def _validate_temporal_dates(self, ordered_dates: list[str]) -> None:
        """Validate date strings and, when requested, enforce one consecutive timestep per month."""
        try:
            parsed = pd.to_datetime(ordered_dates, format="%Y%m%d", errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("Error: Temporal feature columns contain invalid YYYYMMDD dates for keras_lstm.") from exc

        if self.temporal_frequency is None:
            return
        if str(self.temporal_frequency).strip().lower() != "monthly":
            raise ValueError("Error: Unsupported temporal_frequency for keras_lstm. Use None or 'monthly'.")

        periods = parsed.to_period("M")
        if periods.duplicated().any():
            duplicate_months = sorted({str(period) for period in periods[periods.duplicated(keep=False)]})
            raise ValueError(
                f"Error: keras_lstm monthly sequences require exactly one timestep per calendar month. Multiple dates were found for months: {duplicate_months}."
            )

        expected = pd.period_range(start=periods.min(), end=periods.max(), freq="M")
        missing = expected.difference(periods)
        if len(missing) > 0:
            raise ValueError(
                f"Error: keras_lstm monthly sequence has missing calendar months. Missing months: {[str(period) for period in missing]}."
            )

    def _parse_temporal_entries(self, columns) -> list[tuple[str, str]]:
        """Extract ``(base_feature, yyyymmdd)`` entries from temporal columns."""
        temporal_stats = None
        if self.temporal_statistics is not None:
            temporal_stats = {str(value).strip().lower().lstrip("_") for value in self.temporal_statistics}

        entries: list[tuple[str, str]] = []
        for column in columns:
            match = TEMPORAL_FEATURE_PATTERN.match(str(column))
            if match is None:
                continue
            base_name = str(match.group("base"))
            date_key = str(match.group("date"))
            suffix = base_name.rsplit("_", maxsplit=1)[-1].lower() if "_" in base_name else base_name.lower()
            if temporal_stats is not None and suffix not in temporal_stats:
                continue
            entries.append((base_name, date_key))
        return entries


class KerasLSTMClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible LSTM classifier operating on flattened temporal parcel features."""

    def __init__(
        self,
        temporal_statistics: tuple[str, ...] | None = ("median",),
        require_complete_timesteps: bool = True,
        temporal_frequency: str | None = "monthly",
        min_timesteps: int = 6,
        class_balancing_method: str = "balanced_class_weight",
        lstm_units_1: int = 64,
        lstm_units_2: int = 32,
        dense_units: int = 32,
        dropout_rate: float = 0.25,
        bidirectional: bool = False,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        epochs: int = 80,
        batch_size: int = 64,
        validation_split: float = 0.15,
        early_stopping_patience: int = 10,
        reduce_lr_patience: int = 5,
        reduce_lr_factor: float = 0.5,
        min_delta: float = 0.0,
        verbose: int = 0,
        random_state: int | None = None,
    ):
        """Store clone-compatible constructor parameters for sklearn usage."""
        self.temporal_statistics = temporal_statistics
        self.require_complete_timesteps = require_complete_timesteps
        self.temporal_frequency = temporal_frequency
        self.min_timesteps = min_timesteps
        self.class_balancing_method = class_balancing_method
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.min_delta = min_delta
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the LSTM model after tensor conversion, imputation, and scaling."""
        tf_module = _safe_import("tensorflow")
        scikeras_wrappers = _safe_import("scikeras.wrappers")
        if tf_module is None or scikeras_wrappers is None:
            raise ImportError("Error: keras_lstm requires both TensorFlow and SciKeras to be installed.")

        frame = self._ensure_frame(X)
        self.feature_names_in_ = list(frame.columns)
        y_array = np.asarray(y, dtype=np.int64)
        if y_array.ndim != 1:
            raise ValueError("Error: KerasLSTMClassifier expects a 1D target array.")

        self.tensor_builder_ = TemporalTensorBuilder(
            temporal_statistics=self.temporal_statistics,
            require_complete_timesteps=bool(self.require_complete_timesteps),
            temporal_frequency=self.temporal_frequency,
            min_timesteps=int(self.min_timesteps),
        ).fit(frame)
        tensor = self.tensor_builder_.transform(frame)

        self.n_timesteps_ = tensor.shape[1]
        self.n_features_per_timestep_ = tensor.shape[2]
        self.classes_ = np.unique(y_array)

        # Learn fallback values from this fit input only; inference reuses them without refitting.
        self.feature_medians_ = self._compute_feature_medians(tensor)
        tensor = self._impute_tensor(tensor)

        self.scaler_ = StandardScaler()
        # Pool samples and dates to fit one scaler per temporal feature, then restore the sequence axes.
        tensor_2d = tensor.reshape(-1, self.n_features_per_timestep_)
        self.scaler_.fit(tensor_2d)
        tensor_scaled = self.scaler_.transform(tensor_2d).reshape(tensor.shape).astype(np.float32)

        tf_module.keras.backend.clear_session()
        if self.random_state is not None:
            tf_module.keras.utils.set_random_seed(int(self.random_state))

        callbacks = [
            tf_module.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(self.early_stopping_patience),
                min_delta=float(self.min_delta),
                restore_best_weights=True,
            ),
            tf_module.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=int(self.reduce_lr_patience), factor=float(self.reduce_lr_factor), verbose=0
            ),
        ]

        class_weight = self._resolve_class_weight()
        self.estimator_ = scikeras_wrappers.KerasClassifier(
            model=build_lstm_classifier_model,
            model__n_timesteps=int(self.n_timesteps_),
            model__n_features=int(self.n_features_per_timestep_),
            model__n_classes=len(self.classes_),
            model__lstm_units_1=int(self.lstm_units_1),
            model__lstm_units_2=int(self.lstm_units_2),
            model__dense_units=int(self.dense_units),
            model__dropout_rate=float(self.dropout_rate),
            model__bidirectional=bool(self.bidirectional),
            model__learning_rate=float(self.learning_rate),
            model__optimizer=str(self.optimizer),
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
            callbacks=callbacks,
            epochs=int(self.epochs),
            batch_size=int(self.batch_size),
            validation_split=float(self.validation_split),
            class_weight=class_weight,
            verbose=int(self.verbose),
            random_state=self.random_state,
        )
        self.estimator_.fit(tensor_scaled, y_array)
        self.temporal_schema_ = self.tensor_builder_.to_schema_dict()
        self.temporal_schema_["class_balancing_method"] = str(self.class_balancing_method)
        return self

    def predict(self, X):
        """Predict labels from temporal tensors derived from flattened features."""
        tensor = self._prepare_inference_tensor(X)
        return np.asarray(self.estimator_.predict(tensor))

    def predict_proba(self, X):
        """Predict class probabilities from temporal tensors derived from flattened features."""
        tensor = self._prepare_inference_tensor(X)
        return np.asarray(self.estimator_.predict_proba(tensor), dtype=float)

    def _resolve_class_weight(self):
        """Map the explicit LSTM balancing mode to SciKeras class_weight semantics."""
        method = str(self.class_balancing_method).strip().lower()
        if method == "none":
            return None
        if method == "balanced_class_weight":
            return "balanced"
        raise ValueError("Error: class_balancing_method must be either 'none' or 'balanced_class_weight'.")

    def _prepare_inference_tensor(self, X) -> np.ndarray:
        """Convert inference frame to an imputed+scaled temporal tensor."""
        frame = self._ensure_frame(X)
        tensor = self.tensor_builder_.transform(frame)
        tensor = self._impute_tensor(tensor)
        tensor_2d = tensor.reshape(-1, self.n_features_per_timestep_)
        # Apply the fitted training scale to inference sequences without learning from prediction rows.
        return self.scaler_.transform(tensor_2d).reshape(tensor.shape).astype(np.float32)

    def _ensure_frame(self, X) -> pd.DataFrame:
        """Ensure input is a pandas dataframe with known feature names."""
        if isinstance(X, pd.DataFrame):
            return X
        if hasattr(self, "feature_names_in_"):
            values = np.asarray(X)
            if values.ndim != 2 or values.shape[1] != len(self.feature_names_in_):
                raise ValueError("Error: Input array shape does not match fitted feature_names_in_ for keras_lstm.")
            return pd.DataFrame(values, columns=self.feature_names_in_)
        raise ValueError("Error: keras_lstm requires a pandas DataFrame input during the first fit call.")

    def _compute_feature_medians(self, tensor: np.ndarray) -> np.ndarray:
        """Compute one fallback median per temporal feature across all samples/timesteps."""
        medians = np.nanmedian(tensor.reshape(-1, tensor.shape[2]), axis=0)
        # Features with no finite median receive a deterministic zero fallback.
        medians = np.where(np.isfinite(medians), medians, 0.0)
        return medians.astype(np.float32)

    def _impute_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """Impute NaNs by forward/backward fill over time and feature-level medians."""
        filled = np.asarray(tensor, dtype=np.float32).copy()
        n_samples, n_timesteps, n_features = filled.shape

        for sample_index in range(n_samples):
            sample = filled[sample_index]
            for feature_index in range(n_features):
                series = sample[:, feature_index]
                mask = np.isnan(series)
                if mask.all():
                    series[:] = self.feature_medians_[feature_index]
                    continue
                # Propagate previous observations first; the backward pass then fills leading gaps.
                for t_idx in range(1, n_timesteps):
                    if np.isnan(series[t_idx]) and not np.isnan(series[t_idx - 1]):
                        series[t_idx] = series[t_idx - 1]
                for t_idx in range(n_timesteps - 2, -1, -1):
                    if np.isnan(series[t_idx]) and not np.isnan(series[t_idx + 1]):
                        series[t_idx] = series[t_idx + 1]
                if np.isnan(series).any():
                    series[np.isnan(series)] = self.feature_medians_[feature_index]
            filled[sample_index] = sample
        return filled


def _build_optimizer(tf_module, optimizer: str, learning_rate: float):
    """Create a Keras optimizer instance from a short optimizer name."""
    optimizer_name = str(optimizer).strip().lower()
    optimizer_map = {
        "adam": tf_module.keras.optimizers.Adam,
        "nadam": tf_module.keras.optimizers.Nadam,
        "rmsprop": tf_module.keras.optimizers.RMSprop,
    }
    optimizer_class = optimizer_map.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError(f"Error: Unsupported optimizer: {optimizer}")
    return optimizer_class(learning_rate=float(learning_rate))


def build_lstm_classifier_model(
    n_timesteps: int,
    n_features: int,
    n_classes: int,
    lstm_units_1: int = 64,
    lstm_units_2: int = 32,
    dense_units: int = 32,
    dropout_rate: float = 0.25,
    bidirectional: bool = False,
    learning_rate: float = 0.001,
    optimizer: str = "adam",
):
    """Build and compile a compact LSTM classifier for parcel time-series tensors."""
    tf_module = _safe_import("tensorflow")
    if tf_module is None:
        raise ImportError("Error: TensorFlow is required to build keras_lstm models.")

    inputs = tf_module.keras.Input(shape=(int(n_timesteps), int(n_features)), name="parcel_time_series")
    # Keep the time axis for the second LSTM, which reduces the sequence to one parcel representation.
    first_layer = tf_module.keras.layers.LSTM(int(lstm_units_1), return_sequences=True)
    if bidirectional:
        temporal = tf_module.keras.layers.Bidirectional(first_layer, name="bilstm_1")(inputs)
    else:
        temporal = first_layer(inputs)
    temporal = tf_module.keras.layers.Dropout(float(dropout_rate), name="dropout_1")(temporal)

    second_layer = tf_module.keras.layers.LSTM(int(lstm_units_2), return_sequences=False)
    if bidirectional:
        temporal = tf_module.keras.layers.Bidirectional(second_layer, name="bilstm_2")(temporal)
    else:
        temporal = second_layer(temporal)
    temporal = tf_module.keras.layers.Dropout(float(dropout_rate), name="dropout_2")(temporal)

    hidden = tf_module.keras.layers.Dense(int(dense_units), activation="relu", name="dense_1")(temporal)
    hidden = tf_module.keras.layers.Dropout(max(float(dropout_rate) - 0.05, 0.0), name="dropout_3")(hidden)
    outputs = tf_module.keras.layers.Dense(int(n_classes), activation="softmax", name="class_probabilities")(hidden)

    model = tf_module.keras.Model(inputs=inputs, outputs=outputs, name="parcel_lstm_classifier")
    model.compile(
        optimizer=_build_optimizer(tf_module, optimizer, learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model
