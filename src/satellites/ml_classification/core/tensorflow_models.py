"""TensorFlow model builders and sklearn-compatible wrappers for parcel classification."""

from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd


def _safe_import(module_name: str):
    """Import a module by name and return ``None`` when import fails."""
    try:
        module = __import__(module_name, fromlist=[module_name.rsplit(".", maxsplit=1)[-1]])
    except Exception:
        return None
    return module


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
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    return optimizer_class(learning_rate=float(learning_rate))


def _apply_hidden_block(
    tf_module,
    hidden_input,
    *,
    units: int,
    activation: str,
    dropout_rate: float,
    use_batch_normalization: bool,
    l2_regularization: float,
    layer_index: int,
):
    """Append one configurable dense hidden block to a Keras graph."""
    regularizer = None
    if float(l2_regularization) > 0:
        regularizer = tf_module.keras.regularizers.l2(float(l2_regularization))
    hidden_output = tf_module.keras.layers.Dense(
        int(units),
        activation=None,
        kernel_regularizer=regularizer,
        name=f"dense_{layer_index}",
    )(hidden_input)
    if use_batch_normalization:
        hidden_output = tf_module.keras.layers.BatchNormalization(name=f"batch_norm_{layer_index}")(hidden_output)
    hidden_output = tf_module.keras.layers.Activation(activation, name=f"activation_{layer_index}")(hidden_output)
    if float(dropout_rate) > 0:
        hidden_output = tf_module.keras.layers.Dropout(float(dropout_rate), name=f"dropout_{layer_index}")(hidden_output)
    return hidden_output


def build_dense_classifier_model(
    input_dim: int,
    num_classes: int,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    activation: str = "relu",
    dropout_rate: float = 0.2,
    use_batch_normalization: bool = False,
    optimizer: str = "adam",
    learning_rate: float = 0.001,
    l2_regularization: float = 0.0,
):
    """Build a configurable dense TensorFlow classifier for tabular parcel features."""
    tf_module = _safe_import("tensorflow")
    if tf_module is None:
        raise ImportError("TensorFlow is required to build tensorflow_neural_network models.")
    if int(input_dim) < 1:
        raise ValueError("input_dim must be >= 1.")
    if int(num_classes) < 2:
        raise ValueError("num_classes must be >= 2.")
    if not hidden_layer_sizes:
        raise ValueError("hidden_layer_sizes must contain at least one layer size.")

    inputs = tf_module.keras.Input(shape=(int(input_dim),), name="parcel_features")
    hidden_output = inputs
    for layer_index, units in enumerate(hidden_layer_sizes, start=1):
        hidden_output = _apply_hidden_block(
            tf_module,
            hidden_output,
            units=int(units),
            activation=activation,
            dropout_rate=float(dropout_rate),
            use_batch_normalization=bool(use_batch_normalization),
            l2_regularization=float(l2_regularization),
            layer_index=layer_index,
        )

    outputs = tf_module.keras.layers.Dense(int(num_classes), activation="softmax", name="class_probabilities")(hidden_output)
    model = tf_module.keras.Model(inputs=inputs, outputs=outputs, name="parcel_tensorflow_classifier")
    model.compile(
        optimizer=_build_optimizer(tf_module, optimizer, learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class TensorFlowDenseClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible wrapper around a configurable TensorFlow dense classifier."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        activation: str = "relu",
        dropout_rate: float = 0.2,
        use_batch_normalization: bool = False,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        l2_regularization: float = 0.0,
        epochs: int = 120,
        batch_size: int = 64,
        validation_split: float = 0.15,
        patience: int = 12,
        min_delta: float = 0.0,
        verbose: int = 0,
        random_state: int | None = None,
    ):
        """Store TensorFlow estimator hyperparameters in clone-friendly form."""
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_normalization = use_batch_normalization
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the TensorFlow classifier using SciKeras as the sklearn bridge."""
        tf_module = _safe_import("tensorflow")
        scikeras_wrappers = _safe_import("scikeras.wrappers")
        if tf_module is None or scikeras_wrappers is None:
            raise ImportError(
                "tensorflow_neural_network requires both TensorFlow and SciKeras to be installed in the environment."
            )

        X_array = self._to_numpy(X)
        y_array = np.asarray(y, dtype=np.int64)
        if X_array.ndim != 2:
            raise ValueError("TensorFlowDenseClassifier expects a 2D feature matrix.")
        if y_array.ndim != 1:
            raise ValueError("TensorFlowDenseClassifier expects a 1D target array.")

        self.n_features_in_ = int(X_array.shape[1])
        self.feature_names_in_ = list(X.columns) if hasattr(X, "columns") else None
        self.classes_ = np.unique(y_array)

        tf_module.keras.backend.clear_session()
        if self.random_state is not None:
            tf_module.keras.utils.set_random_seed(int(self.random_state))

        callbacks = [
            tf_module.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(self.patience),
                min_delta=float(self.min_delta),
                restore_best_weights=True,
            )
        ]
        self.estimator_ = scikeras_wrappers.KerasClassifier(
            model=build_dense_classifier_model,
            model__input_dim=int(self.n_features_in_),
            model__num_classes=int(len(self.classes_)),
            model__hidden_layer_sizes=tuple(int(units) for units in self.hidden_layer_sizes),
            model__activation=str(self.activation),
            model__dropout_rate=float(self.dropout_rate),
            model__use_batch_normalization=bool(self.use_batch_normalization),
            model__optimizer=str(self.optimizer),
            model__learning_rate=float(self.learning_rate),
            model__l2_regularization=float(self.l2_regularization),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            callbacks=callbacks,
            epochs=int(self.epochs),
            batch_size=int(self.batch_size),
            validation_split=float(self.validation_split),
            verbose=int(self.verbose),
            random_state=self.random_state,
        )
        self.estimator_.fit(X_array, y_array)
        return self

    def predict(self, X):
        """Predict class labels for input feature rows."""
        return np.asarray(self.estimator_.predict(self._to_numpy(X)))

    def predict_proba(self, X):
        """Predict class probabilities for input feature rows."""
        return np.asarray(self.estimator_.predict_proba(self._to_numpy(X)), dtype=float)

    def _to_numpy(self, X):
        """Convert feature input to a dense numpy array for Keras consumption."""
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float32, copy=False)
        return np.asarray(X, dtype=np.float32)
