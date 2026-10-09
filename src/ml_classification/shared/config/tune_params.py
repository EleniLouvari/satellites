"""Default hyperparameter tuning grids used by pipeline configuration."""

from copy import deepcopy

# Keep this helper focused on a single transformation so the reporting pipeline stays easy to follow.



def _default_tune_params() -> dict[str, dict[str, list]]:
    """Return an independent copy of the default search space for every model."""
    # Each configuration owns its nested lists so per-run overrides cannot mutate another search space.
    return deepcopy(
        {
            # Parameter paths start at the pipeline model step; nested estimators add another routing segment.
            "logistic_regression": {
                "model__C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
                "model__solver": ["liblinear"],
                "model__penalty": ["l1", "l2"],
            },
            "linear_sgd_classifier": {"model__alpha": [1e-4, 1e-3, 1e-2, 1e-1], "model__l1_ratio": [0.0, 0.15, 0.5, 0.8, 1.0]},
            "support_vector_machine": {
                "model__C": [0.1, 1.0, 10.0, 20.0],
                "model__kernel": ["linear", "rbf", "poly"],
                "model__gamma": ["scale", "auto"],
                # Soft-voting strategies require the SVM candidate to expose class probabilities.
                "model__probability": [True],
            },
            "random_forest": {
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [5, 8, 12, 16],
                "model__min_samples_split": [10, 20, 40],
                "model__min_samples_leaf": [3, 5, 10, 20],
                "model__max_features": ["sqrt", "log2", 0.3, 0.5],
                "model__max_samples": [0.6, 0.8, None],
                "model__ccp_alpha": [0.0, 0.0001, 0.001, 0.005],
                "model__class_weight": [None],
                "model__criterion": ["gini", "entropy"],
            },
            "extra_trees": {
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [5, 8, 12, 16],
                "model__min_samples_split": [10, 20, 40],
                "model__min_samples_leaf": [3, 5, 10, 20],
                "model__max_features": ["sqrt", "log2", 0.3, 0.5],
                "model__max_samples": [0.6, 0.8, None],
                "model__ccp_alpha": [0.0, 0.0001, 0.001],
            },
            "decision_tree": {
                "model__max_depth": [3, 5, 8, 12, 16],
                "model__min_samples_split": [10, 20, 40],
                "model__min_samples_leaf": [5, 10, 20, 40],
                "model__criterion": ["gini", "entropy", "log_loss"],
                "model__max_features": [None, "sqrt", 0.5],
                "model__ccp_alpha": [0.0, 0.0001, 0.001, 0.005, 0.01],
                "model__splitter": ["best", "random"],
            },
            "knn": {
                "model__n_neighbors": [5, 9, 15, 25, 40],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
                "model__leaf_size": [20, 30, 50],
                "model__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            },
            "neural_network": {
                "model__hidden_layer_sizes": [(32,), (64,), (64, 32)],
                "model__alpha": [0.0001, 0.001, 0.01, 0.1],
                "model__learning_rate_init": [0.0001, 0.0005, 0.001],
                "model__activation": ["relu", "tanh"],
            },
            "tensorflow_neural_network": {
                "model__hidden_layer_sizes": [(32,), (64,), (64, 32)],
                "model__dropout_rate": [0.0, 0.2, 0.4],
                "model__use_batch_normalization": [False, True],
                "model__learning_rate": [0.0001, 0.0005, 0.001],
                "model__batch_size": [32, 64],
                "model__l2_regularization": [0.0, 0.0001, 0.001],
                "model__activation": ["relu", "tanh"],
            },
            "bayesian": {"model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
            "keras_lstm": {
                "model__lstm_units_1": [32, 64],
                "model__lstm_units_2": [16, 32],
                "model__dense_units": [16, 32],
                "model__dropout_rate": [0.10, 0.25],
                "model__bidirectional": [False, True],
                "model__learning_rate": [0.0005, 0.001],
                "model__batch_size": [32, 64],
            },
            "bagging": {
                "model__n_estimators": [100, 200, 400],
                "model__max_samples": [0.5, 0.7, 0.9],
                "model__max_features": [0.5, 0.7, 0.9],
                # These parameters tune the base tree inside bagging, not the bagging wrapper itself.
                "model__estimator__max_depth": [5, 8, 12, 16],
                "model__estimator__min_samples_split": [10, 20, 40],
                "model__estimator__min_samples_leaf": [3, 5, 10, 20],
            },
            "gradient_boosting": {
                "model__n_estimators": [100, 200, 400],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__max_depth": [1, 2, 3],
                "model__min_samples_split": [10, 20, 40],
                "model__min_samples_leaf": [5, 10, 20],
                "model__subsample": [0.6, 0.8],
                "model__max_features": ["sqrt", 0.5, None],
            },
            "hist_gradient_boosting": {
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__max_iter": [100, 200, 400],
                "model__max_depth": [3, 5, 8],
                "model__max_leaf_nodes": [7, 15, 31],
                "model__min_samples_leaf": [20, 40, 60],
                "model__l2_regularization": [0.01, 0.1, 1.0, 10.0],
            },
            "xgboost": {
                # Reach the boosted estimator through its label-handling wrapper.
                "model__base_estimator__n_estimators": [200, 400, 600],
                "model__base_estimator__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__base_estimator__max_depth": [2, 3, 4, 5],
                "model__base_estimator__min_child_weight": [3, 5, 10],
                "model__base_estimator__subsample": [0.6, 0.8, 1.0],
                "model__base_estimator__colsample_bytree": [0.5, 0.7, 0.9],
                "model__base_estimator__reg_alpha": [0.0, 0.01, 0.1, 1.0],
                "model__base_estimator__reg_lambda": [1.0, 5.0, 10.0],
                "model__base_estimator__gamma": [0.0, 0.1, 0.5],
            },
            "lightgbm": {
                "model__base_estimator__n_estimators": [200, 400, 600],
                "model__base_estimator__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__base_estimator__num_leaves": [7, 15, 31],
                "model__base_estimator__max_depth": [3, 5, 8],
                "model__base_estimator__min_child_samples": [20, 40, 80],
                "model__base_estimator__subsample": [0.6, 0.8, 1.0],
                "model__base_estimator__colsample_bytree": [0.5, 0.7, 0.9],
                "model__base_estimator__reg_alpha": [0.0, 0.1, 1.0],
                "model__base_estimator__reg_lambda": [0.1, 1.0, 10.0],
            },
        }
    )
