from import_libraries import *
import global_variables as gb_l
import libraries.common_libraries.generic_library as cm_l
import libraries.common_libraries.geom_library as geom_l
import libraries.ml_common_libraries.ml_features_library as ml_feat


#-----------------------------------------PREPARING FEATURES---------------------------------------------------------------
    # Verification of stratification
def distribution_check(original, train, test, column, num_bins=10):
    print("Distribution Check:")
    print("Original data distribution:")
    print(pd.qcut(original[column], q=num_bins, labels=False).value_counts(normalize=True))
    print("\nTrain data distribution:")
    print(pd.qcut(train[column], q=num_bins, labels=False).value_counts(normalize=True))
    print("\nTest data distribution:")
    print(pd.qcut(test[column], q=num_bins, labels=False).value_counts(normalize=True))


def stratified_sampling_by_column(df, col_target, test_size=0.2, random_state=42, num_bins=10,
                                  stratification_method="quantile"):
    """
    Perform stratified sampling on a dataframe based on binning a numeric column.

    Parameters:
    -----------
    df : pandas.DataFrame, Input dataframe to be sampled
    col_target : str, Name of the column to use for stratification
    test_size : float, optional (default=0.2), Proportion of the dataset to include in the test split
    random_state : int, optional (default=42), Controls the shuffling applied to the data before applying the split
    num_bins : int, optional (default=10), Number of bins to create for stratification
    stratification_method : str, optional (default='quantile'), Method for creating stratification bins
                            Options: 'quantile', 'uniform', 'kmeans'
    Returns:
    --------
    tuple : (train_df, test_df), Tuple containing train and test dataframes
    """
    # Validate input column is numeric
    if not pd.api.types.is_numeric_dtype(df[col_target]):
        raise ValueError(f"Column '{col_target}' must be numeric.")

    valid_methods = ['quantile', 'uniform', 'kmeans']
    if not stratification_method in valid_methods:
        raise ValueError(f"Method '{stratification_method}' is not valid. Valid ones: {valid_methods}")

    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()

    # Advanced binning techniques
    if stratification_method == "quantile":
        # Quantile-based binning - ensures equal number of samples in each bin
        discretizer = KBinsDiscretizer(n_bins=num_bins, encode="ordinal",  strategy="quantile")
        stratification_bins = discretizer.fit_transform(df_copy[[col_target]]).flatten()

    elif stratification_method == "uniform":
        # Uniform-width binning
        discretizer = KBinsDiscretizer(n_bins=num_bins, encode="ordinal", strategy="uniform")
        stratification_bins = discretizer.fit_transform(df_copy[[col_target]]).flatten()

    elif stratification_method == "kmeans":
        # K-means based binning - clusters based on data distribution
        discretizer = KBinsDiscretizer(n_bins=num_bins, encode="ordinal", strategy="kmeans")
        stratification_bins = discretizer.fit_transform(df_copy[[col_target]]).flatten()

    # Perform stratified train-test split
    train_df, test_df = train_test_split(df_copy, test_size=test_size, stratify=stratification_bins, random_state=random_state)

    return train_df, test_df


def drop_features(df, base_columns, keep_features):
    df = df.copy()
    all_columns_to_keep = list(set(base_columns + keep_features))
    # Identify the columns to be dropped that are present in the dataframe
    cols_to_drop = [col for col in df.columns if col not in all_columns_to_keep]

    if cols_to_drop:
        # Drop all identified columns at once
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropping columns: {', '.join(cols_to_drop)}")
    else:
        print("There aren't any columns to drop!")
    return df


def select_features_for_ml(df, base_columns, features):
    """Select only the features to be used for the ml."""
    columns = [x for x in df.columns if x not in base_columns]
    features_to_keep = []

    for col in columns:
        for feature_to_keep in features:
            if col == feature_to_keep or col.startswith(feature_to_keep):
                features_to_keep.append(col)
                break  # Once the feature is added to features_to_keep, we can break the loop

    features_to_remove = [feature for feature in columns if feature not in features_to_keep]
    return features_to_keep, features_to_remove


def create_extra_feature_names(gdf, col_id, features, extra_feature_names, plot_graphs, verbose):
    """Create extra features based on USA general data."""
    state, _ = geom_l.get_geodataframe_US_state_and_center(gdf)

    if any(feature in features for feature in [extra_feature_names['house_yr_median'], extra_feature_names['house_score']]):
        gdf = ml_feat.calculate_house_year_risk(gdf, state, col_id, gb_l.gis_library_dir, extra_feature_names['house_yr_median'], extra_feature_names['house_score'], plot_graphs, verbose)
    else:
        print(f"Features '{extra_feature_names['house_yr_median']}', '{extra_feature_names['house_score']}' will not be calculated!")

    if extra_feature_names['income'] in features:
        gdf = ml_feat.calculate_income(gdf, state, col_id, gb_l.gis_library_dir, col_name=extra_feature_names['income'], plot=plot_graphs, verbose=verbose)
    else:
        print(f"Feature '{extra_feature_names['income']}' will not be calculated!")

    if extra_feature_names['dis_com'] in features:
        gdf = ml_feat.calculate_disadvantages_communities(gdf, state, col_id, gb_l.gis_library_dir, col_name=extra_feature_names['dis_com'], plot=plot_graphs, verbose=verbose)
    else:
        print(f"Feature '{extra_feature_names['dis_com']}' will not be calculated!")

    return gdf


def create_target_dependent_features(gdf, col_id, col_target, col_year, search_distance_in_ft, features, target_feature_names, ban_year, plot_graphs):
    gdf = gdf.copy()

    if any(feature in features for feature in [target_feature_names['nearby'], target_feature_names['ratio'], target_feature_names['avrg_distance']]):
        gdf = ml_feat.create_features_nearby(gdf, col_id, col_target, target_feature_names['nearby'], target_feature_names['ratio'], target_feature_names['avrg_distance'],
                                             search_distance_in_ft, features, plot=plot_graphs)
    else:
        cm_l.print_formatted_txt(f"Features {target_feature_names['nearby']}, {target_feature_names['ratio']}, {target_feature_names['avrg_distance']} are not created.", "WARNING")

    if target_feature_names['kernel'] in features:
        gdf = ml_feat.create_feature_kernel(gdf, col_target, target_feature_names['kernel'], plot=plot_graphs)
    else:
        cm_l.print_formatted_txt(f"Feature {target_feature_names['kernel']} is not created.", "WARNING")

    if target_feature_names['voronoi'] in features:
        gdf = ml_feat.create_feature_voronoi(gdf, col_id, col_target, target_feature_names['voronoi'], plot=plot_graphs)
    else:
        cm_l.print_formatted_txt(f"Feature {target_feature_names['voronoi']} is not created.", "WARNING")

    if target_feature_names['year_score'] in features:
        gdf = ml_feat.calculate_target_percentage_per_decade(gdf, col_year, col_target, target_feature_names['year_score'], ban_year, plot=plot_graphs)
    else:
        cm_l.print_formatted_txt(f"Feature {target_feature_names['year_score']} is not created.", "WARNING")

    return gdf


def plot_train_test_distributions(df, col_target, col_id, train_ids, test_ids):
    print("\nChecking target variable distribution:")

    # Plot map
    if isinstance(df, gpd.GeoDataFrame):
        fig, ax = plt.subplots(1, 1, figsize=(18, 10))
        df[df[col_id].isin(train_ids)].plot(ax=ax, color="lightblue")
        df[df[col_id].isin(test_ids)].plot(ax=ax, color="red")
        plt.title("Spatial Target Variable Distribution")
        plt.tight_layout()
        plt.show()

    # Plot Target Distribution
    target_distribution = df[col_target].value_counts(normalize=True)
    if len(target_distribution) > 2 and len(target_distribution) <= 10:
        # If there are more than two classes, visualize the distribution
        fig_width = max(10, min(len(target_distribution)*2, 18))
        plt.figure(figsize=(fig_width, 4))
        sns.barplot(x=target_distribution.index, y=target_distribution.values)
        plt.title("Target Variable Distribution")
        plt.xlabel("Target")
        plt.ylabel("Percentage")
        plt.tight_layout()
        plt.show()
    elif len(target_distribution) > 10:
        plt.figure(figsize=(18, 4))
        sns.histplot(df[col_target], bins=100)
        plt.tight_layout()
        plt.show()
    else:
        # If it's binary classification, print the distribution directly
        print("Binary classification detected.")
        print(target_distribution)

    print("\nChecking feature distributions in train and test data:")
    _, object_cols, numerical_cols = cm_l.get_column_types(df, exclude_geometry_column=True)

    # Calculate number of columns needed based on the number of numerical features
    num_cols_per_row = 3
    features = numerical_cols + object_cols
    plot_features = [x for x in features if x not in [col_id, 'geometry']]

    num_features = len(plot_features)
    num_rows = (num_features - 1) // num_cols_per_row + 1

    fig, axs = plt.subplots(num_rows, num_cols_per_row, figsize=(18, 4*num_rows))
    axs = axs.flatten()  # Flatten axs to a 1-dimensional array
    for i, col in enumerate(plot_features):
        sns.histplot(data=df[df[col_id].isin(train_ids)], x=col, bins=30, color="blue", label="Train Data", kde=True, ax=axs[i])
        sns.histplot(data=df[df[col_id].isin(test_ids)], x=col, bins=30, color="red", label="Test Data", kde=True, ax=axs[i])
        axs[i].set_title(f"{col} Distribution in Train and Test Data")
        axs[i].set_xlabel(col)
        axs[i].set_ylabel("Frequency")
        axs[i].legend()

    # Adjust layout to avoid overlapping
    plt.tight_layout()
    plt.show()


#-----------------------------------------TRAINING-------------------------------------------------------------------
def label_encoder(df, col_target, col_encoded):
    label_encoder = LabelEncoder()
    df[col_encoded] = label_encoder.fit_transform(df[col_target])
    return df


def apply_smote(X_train, y_train, seed_number):
    smote = SMOTE(random_state=seed_number)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, y_train


def get_pca_components(X_train, X_test, n_components):
    # Apply PCA to reduce the dimensionality of the dataset
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca


def encode_categorical_features_to_dummies(df, base_columns):
    """Encodes categorical features."""
    df = df.copy()
    # Separate numeric and categorical columns
    _, object_cols, numerical_cols = cm_l.get_column_types(df)
    cat_features = [x for x in object_cols if x not in base_columns]
    if len(cat_features) > 0:
        df = pd.get_dummies(df, columns=cat_features, drop_first=True)

    return df


def encode_categorical_features_to_numbers(df, base_columns):
    """Encodes categorical features."""
    df = df.copy()

    # Separate numeric and categorical columns
    object_cols, numerical_cols = cm_l.get_column_types(df)
    cat_features = [x for x in object_cols if x not in base_columns]

    if len(cat_features) > 0:
        encoders = {}  # Dictionary to store encoders for reuse
        for col in cat_features:
            # Create a LabelEncoder instance
            le = LabelEncoder()
            # Fit the encoder on the data in the column
            le.fit(df[col])
            # Encode the column using the fitted encoder
            df[col] = le.transform(df[col])
            # Store the encoder for potential future use (optional)
            encoders[col] = le
    return df


def split_into_train_test_ids(df, col_id, col_target, grid_sample, seed_number, split_ratio):
    """Split dataframe into Train: 80%, Test: 20% sets."""
    print(f"Splitting {round(split_ratio, 1), round(1.0-split_ratio, 1)}...")
    df = df[df[col_target]>=0].copy()
    def select_random_data(df, ratio, seed_number, col_id, grid_sample):
        total_data_to_select = round(ratio * len(df.index))
        return geom_l.create_uniform_spatial_random_sample(df, total_data_to_select, seed_number, col_id, grid_sample)

    # Split data into target and no target
    df_target = df[df[col_target] == 1].copy()
    df_no_target = df[df[col_target] == 0].copy()

    # Create train samples
    df_target_train = select_random_data(df_target, split_ratio, seed_number, col_id, grid_sample)
    df_no_target_train = select_random_data(df_no_target, split_ratio, seed_number, col_id, grid_sample)
    train_ids = set(df_target_train[col_id]).union(set(df_no_target_train[col_id]))

    # Create test samples from remaining
    df_target_test = df_target[~df_target[col_id].isin(train_ids)].copy()
    df_no_target_test = df_no_target[~df_no_target[col_id].isin(train_ids)].copy()
    test_ids = set(df_target_test[col_id]).union(set(df_no_target_test[col_id]))

    print(f"Data split into "
          f"\ntrain: {len(train_ids)} ({len(df_target_train)} {col_target}), "
          f"\ntest: {len(test_ids)} ({len(df_target_test)} {col_target}).")

    assert len(train_ids & test_ids) == 0, "Common ids in the training-testing data!"
    return train_ids, test_ids


def prepare_splitted_features(gdf_features, col_id, col_target, col_year, col_group, train_ids, test_ids, search_distance_in_ft, features, target_feature_names, ban_year, plot_graphs):
    """Differentiate train-test data and create target depenendant features using only the train data and populate to the test data."""
    train_test_features = gdf_features[gdf_features[col_id].isin(list(train_ids) + list(test_ids))].copy()
    train_test_features['or_target'] = train_test_features[col_target].copy()
    train_test_features.loc[train_test_features[col_id].isin(test_ids), col_target] = -1
    train_test_features = create_target_dependent_features(train_test_features, col_id, col_target, col_year, search_distance_in_ft, features, target_feature_names, ban_year, plot_graphs)
    train_test_features[col_target] = train_test_features['or_target'].copy()
    train_test_features.drop(columns=['or_target'], inplace=True)
    train_test_features.loc[train_test_features[col_id].isin(train_ids), col_group] = "train"
    train_test_features.loc[train_test_features[col_id].isin(test_ids), col_group] = "test"
    return train_test_features


def init_model_instance(model_class, params):
    if isinstance(model_class, type):
        # It's a class, instantiate it
        if "random_state" in inspect.signature(model_class.__init__).parameters:
            model_instance = model_class(**params, random_state=gb_l.SEED_NUMBER)
        else:
            model_instance = model_class(**params)
    else:
        # It's already an instance, just set its parameters
        model_instance = model_class
        if isinstance(model_instance, CalibratedClassifierCV):
            if "random_state" in inspect.signature(model_instance.estimator.__class__.__init__).parameters:
                model_instance.estimator.set_params(**params, random_state=gb_l.SEED_NUMBER)
            else:
                model_instance.estimator.set_params(**params)
        else:
            if "random_state" in inspect.signature(model_instance.__class__.__init__).parameters:
                model_instance.set_params(**params, random_state=gb_l.SEED_NUMBER)
            else:
                model_instance.set_params(**params)
    return model_instance


def train_model(params, model_class, X_train, y_train):
    """Trains the ML model using specific params."""
    model_instance = init_model_instance(model_class, params)
    conf_matrix, precision, recall, f1, accuracy = get_confusion_matrix_metrics(X_train, y_train)
    return f1, params


def train_model_cv(params, model_class, X_train, y_train, X_test, y_test):
    """Trains the ML model using specific params of cv fold."""
    model_instance = init_model_instance(model_class, params)
    model_instance.fit(X_train, y_train)
    y_pred = model_instance.predict(X_test)
    conf_matrix, precision, recall, f1, accuracy = get_confusion_matrix_metrics(y_test, y_pred)
    return f1, params

def train_model_cv_reggressor(params, model_class, X_train, y_train, X_test, y_test):
    """Trains the ML model using specific params of cv fold."""
    model_instance = init_model_instance(model_class, params)
    model_instance.fit(X_train, y_train)
    y_pred = model_instance.predict(X_test)
    val_metrics = compute_regression_metrics(y_test, y_pred)
    return val_metrics['R2'], params

def tune_model(model_class, param_dict, X_train, y_train, num_processes=None, verbose=True):
    # Create a DataFrame to store the combinations
    if num_processes is None:
        raise Exception("num_processes cannot be None")
    param_combinations = list(product(*param_dict.values()))
    param_df = pd.DataFrame(param_combinations, columns=param_dict.keys())
    param_df = param_df.drop_duplicates()
    print(f"Number of tuning tests: {len(param_df)}")

    best_score = 0
    # Use multiprocessing to parallelize the tuning process
    if num_processes is None:
        num_processes = 1
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2) as pool:
        results = pool.starmap(train_model, [(params, model_class, X_train, y_train) for params in param_df.to_dict(orient="records")], chunksize=num_processes)

    if verbose:
        results_df = pd.DataFrame(results, columns=['score', 'params'])
        results_df = results_df.sort_values(by=['score'], ascending=False)
        display(results_df[['params', 'score']].head(5))

    # Find the best model based on the highest score
    for score, params in results:
        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def tune_model_cv(model_class, model_name, param_dict, folds, num_processes=1, verbose=True, plot_graphs=True, classification=True):
    """Tunes the ML model based on search parameters for each fold."""
    if num_processes is None:
        raise Exception("num_processes cannot be None")

    # Create a DataFrame to store the combinations
    param_combinations = list(product(*param_dict.values()))
    param_df = pd.DataFrame(param_combinations, columns=param_dict.keys(), dtype=object)
    param_df = param_df.drop_duplicates()

    fold_results = []
    for i, fold in enumerate(folds):
        print(f"Running {i} fold...")
        X_train, y_train = fold['X_train'], fold['y_train']
        X_test, y_test = fold['X_test'], fold['y_test']

        # Ensure numpy arrays are contiguous
        X_train = np.ascontiguousarray(X_train)
        y_train = np.ascontiguousarray(y_train)
        X_test = np.ascontiguousarray(X_test)
        y_test = np.ascontiguousarray(y_test)

        # Use multiprocessing to parallelize the tuning process
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2) as pool:
            if classification:
                results = pool.starmap(train_model_cv, [(params, model_class, X_train, y_train, X_test, y_test) \
                                                        for params in param_df.to_dict(orient="records")], chunksize=num_processes)
            else:
                results = pool.starmap(train_model_cv_reggressor, [(params, model_class, X_train, y_train, X_test, y_test) \
                                                        for params in param_df.to_dict(orient="records")], chunksize=num_processes)
            # Create a DataFrame from results with 'score' and 'params' columns
            results_df = pd.DataFrame(results, columns=['score', 'params'])
            # Add 'fold' information to each row
            results_df['fold'] = fold['fold']
            fold_results.append(results_df)

    # Combine results from all folds
    fold_results = pd.concat(fold_results, ignore_index=True)
    fold_results['params'] = fold_results['params'].astype(str)
    fold_results = fold_results.sort_values(by=['score'], ascending=False)

    if verbose:
        display(fold_results[['fold', 'params', 'score']].head(5))

    if plot_graphs:
        fig, ax = plt.subplots(figsize=(15, 5))
        fold_results.groupby('params').plot(x="fold", y="score", marker="o", ax=ax)
        ax.legend().set_visible(False)
        plt.show()

    # Calculate mean and standard deviation of 'score' for each set of parameters
    params_avg_std = fold_results.groupby('params')['score'].agg(['mean', 'std']).reset_index()

    # Calculate a new column as mean - std
    params_avg_std['mean_minus_std'] = params_avg_std['mean'] - params_avg_std['std']

    # Find the parameters associated with the maximum (mean - std)
    best_params_row = params_avg_std.loc[params_avg_std['mean_minus_std'].idxmax()]

    # Extract best parameters
    best_params = eval(best_params_row['params'])
    best_score = best_params_row['mean_minus_std']
    if verbose:
        cm_l.print_formatted_txt(f"Best params for model {model_name}: {best_params}, score (avg-std): {best_score}", "RESULTS")
    return best_params

def shuffle_two_dfs_together(df1, df2):
    shuffled_idx = np.random.permutation(df1.index)
    df1.reindex(shuffled_idx)
    df2.reindex(shuffled_idx)


def get_data_subset(features, labels, percent):
    """Return a stratified random subsample of the data according to the given <percent>."""
    assert 0.0 < percent < 1.0, "Percent must be between [0, 1.0]"
    labels.reset_index(drop=True, inplace=True)
    features.reset_index(drop=True, inplace=True)
    events = labels[labels == 1]
    non_events = labels[labels == 0]

    # Get a random stratified subsample of events and non-events
    events = events.sample(round(len(events) * percent))
    non_events = non_events.sample(round(len(non_events) * percent))
    data_subset_index = events.index.union(non_events.index)

    new_feat, new_label = features.loc[data_subset_index], labels.loc[data_subset_index]
    shuffle_two_dfs_together(new_feat, new_label)
    print(f"Trying with randomized subset of events. Number of data points: {len(new_feat)}")
    return new_feat, new_label


# def plot_feature_importance(model, model_name, X_train, top_n=10):
#     """Plot the top N feature importance for the model."""
#     if hasattr(model, "feature_importances_"):
#         print(f"\nFeature impportance of {model_name} model...")
#         plt.figure(figsize=(10, 6))
#         if model_name == "KNN":
#             # KNN doesn't have inherent feature importance, but you can plot feature importance based on distance weights
#             feature_importance = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
#         elif model_name not in ['LogisticRegression', 'SVM']:
#             feature_importance = model.feature_importances_

#         # Reverse the order of feature importances
#         sorted_indices = np.argsort(feature_importance)[::-1]
#         sorted_feature_importance = feature_importance[sorted_indices]
#         sorted_features = X_train.columns[sorted_indices]

#         # Select top N features
#         top_n = min(top_n, len(sorted_feature_importance))
#         sorted_feature_importance = sorted_feature_importance[:top_n]
#         sorted_features = sorted_features[:top_n]

#         sns.barplot(x=sorted_feature_importance, y=sorted_features)
#         plt.title(f"{model_name} Top {top_n} Feature Importance")
#         plt.xlabel("Feature Importance")
#         plt.ylabel("Feature")
#         plt.show()


def plot_feature_importance(model, model_name, X_train, feature_names=None, top_n=10):
    """
    Plot the top N feature importances for classification or regression models.

    Parameters:
    - model: The trained model (regression or classification).
    - model_name: Name of the model (string).
    - X_train: Training data used to fit the model (DataFrame or NumPy array).
    - feature_names: List of feature names (default: None). Required if X_train is a NumPy array.
    - top_n: Number of top features to plot (default: 10).
    """
    if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
        print(f"\nFeature importance for {model_name} model...")
        plt.figure(figsize=(10, 6))

        # Retrieve feature importance or coefficients
        if hasattr(model, "feature_importances_"):
            feature_importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            if len(model.coef_.shape) == 1:  # Single-dimensional coefficients
                feature_importance = np.abs(model.coef_)
            else:  # Multi-dimensional coefficients (e.g., multi-output regression)
                feature_importance = np.mean(np.abs(model.coef_), axis=0)
        else:
            print(f"{model_name} does not support feature importance or coefficients.")
            return

        # Determine feature names
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns
        elif feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(len(feature_importance))]

        # Sort features by importance
        sorted_indices = np.argsort(feature_importance)[::-1]
        sorted_feature_importance = feature_importance[sorted_indices]
        sorted_features = np.array(feature_names)[sorted_indices]

        # Select top N features
        top_n = min(top_n, len(sorted_feature_importance))
        sorted_feature_importance = sorted_feature_importance[:top_n]
        sorted_features = sorted_features[:top_n]

        # Create bar plot
        sns.barplot(x=sorted_feature_importance, y=sorted_features, legend=False)
        plt.title(f"{model_name} Top {top_n} Feature Importance")
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()
    else:
        print(f"{model_name} does not provide feature importance or coefficients.")


def plot_shap_values(model, model_name, X_train):
    print(f"\nSHAP values for {model_name} model...")
    explainer = shap.Explainer(model)

    shap_values = explainer.shap_values(X_train)  # Returns 3D array
    print(f"SHAP value shape: {shap_values.shape}")  # (n_samples, n_features, n_classes)

    if len(shap_values.shape) == 3:  # Multiclass case
        n_classes = shap_values.shape[2]

        # Set up grid: 3 columns, enough rows to fit all classes
        n_cols = 3
        n_rows = math.ceil(n_classes / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))  # Adjust size

        for class_idx in range(n_classes):
            row, col = divmod(class_idx, n_cols)  # Calculate row and col in the grid
            ax = axes[row, col]  # Select correct subplot

            # Create a temporary figure to capture SHAP plot as an image
            shap_fig = plt.figure()
            shap.summary_plot(shap_values[:, :, class_idx], X_train, max_display=10, show=False)

            # Convert SHAP plot to an image
            shap_fig.canvas.draw()
            img = np.frombuffer(shap_fig.canvas.tostring_rgb(), dtype="uint8")
            img = img.reshape(shap_fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(shap_fig)  # Close temporary SHAP figure

            # Display the image in the subplot
            ax.imshow(img)
            ax.axis("off")  # Hide axes for cleaner visualization

            # Add a title to the subplot
            ax.set_title(f"Class {class_idx}", fontsize=10)

        # Hide any unused subplots (if n_classes isn't a multiple of 3)
        for i in range(n_classes, n_rows * n_cols):
            fig.delaxes(axes.flatten()[i])  # Remove extra axes

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    else:  # Binary classification or regression
        print("\nPlotting SHAP values...")
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # For binary classification, use positive class
        shap.summary_plot(shap_values, X_train, max_display=10, title=f"SHAP Summary Plot for {model_name}")
        plt.show()


def display_grid_search_results(cv_results: dict, verbose=True, plot_graph=True):
    """Display the results of a grid search in a readable table format.."""

    # Collecting the parameter names
    param_names = [key for key in cv_results.keys() if key.startswith('param_')]

    # Combine all parameters into a single 'Hyperparameters' column
    data = {'Hyperparameters': [', '.join(f"{param.split('param_')[1]}={cv_results[param][i]}" for param in param_names) for i in range(len(cv_results[param_names[0]]))]}
    data['Mean Test Score'] = cv_results['mean_test_score']
    data['Std Test Score'] = cv_results['std_test_score']
    data['Rank Test Score'] = cv_results['rank_test_score']

    # Creating the DataFrame
    df = pd.DataFrame(data)

    if verbose:
        # Sort the DataFrame by rank
        print("Hyperparameter tuning results:")
        df_sorted = df.sort_values(by="Rank Test Score").reset_index(drop=True)
        display(df_sorted)

    if plot_graph:
        plot_grid_search_results(df_sorted)


def plot_grid_search_results(params_df):
    """Plot the scores of all folds for each parameter combination."""

    plt.figure(figsize=(12, 6))

    sns.lineplot(x=params_df.index, y="Mean Test Score", data=params_df)
    plt.fill_between(params_df['Mean Test Score'],
                     params_df['Mean Test Score'] - params_df['Std Test Score'],
                     params_df['Mean Test Score'] + params_df['Std Test Score'],
                     alpha=0.2)

    plt.title("Grid Search Scores")
    plt.xlabel("Parameter Combination Index")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.show()


#-----------------------------------------EVALUATION-------------------------------------------------------------------
def compute_regression_metrics(y_true, y_pred, model_name="Statistics", col_model="Model"):
    """Helper function to compute regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {col_model: model_name, 'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2)}


def create_regression_results_df(df, models, features, target, verbose, sort_metric="R2", col_model="Model"):
    df = df.copy()
    X = df[features].values
    y = df[target].values

    # Get the list of esemble methods
    ensemble_methods = []
    for method in ['Voting', 'Stacking']:
        if method in list(models.keys()):
            ensemble_methods.append(method)

    # Compute metrics
    results = []
    for name, model in models.items():
        y_pred = model.predict(X)
        results.append(compute_regression_metrics(y, y_pred, name, col_model))
        df[name] = y_pred
    df_results = pd.DataFrame.from_dict(results)

    # Sort base and esemble models
    cond = df_results[col_model].isin(ensemble_methods)
    df_results = pd.concat([df_results[~cond].sort_values(by=sort_metric, ascending=False),
                            df_results[cond].sort_values(by=sort_metric, ascending=False)],
                            ignore_index=True)

    # Print the results
    metrics_styled = df_results.set_index(col_model)
    if verbose:
        metrics_styled = metrics_styled.style.apply(cm_l.highlight_df_rows(ensemble_methods, "honeydew"), axis=1)
        display(metrics_styled)

    return df_results


def plot_roc_curve_and_confusion_matrix_binary(fpr, tpr, conf_matrix):
    """Plot the ROC curve and the confusion matrx of the model.

    Keyword arguments:
    fpr -- false positive rate
    tpr -- true positive rate
    conf_matrix -- confusion matrix
    """
    fig, axis = plt.subplots(1, 2, figsize=(17, 7))
    # plot the receiver operating characteristic curve
    axis[0].set_title("ROC curve")
    axis[0].set_xlabel("False Positive Rate")
    axis[0].set_ylabel("True Positive Rate")
    axis[0].grid(True)
    axis[0].plot(fpr, tpr)

    # plot the confusion matrix
    axis[1].imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Wistia)
    axis[1].set_title("Confusion Matrix")
    axis[1].set_ylabel("Ground Truth")
    axis[1].set_xlabel("Predicted")

    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            axis[1].text(j, i, f"{labels[i][j]}  = {conf_matrix[i][j]}", ha="center")

    class_names = ["Negative", "Positive"]
    tick_marks = np.arange(len(class_names))
    axis[1].set_xticks(tick_marks, class_names, rotation=0)
    axis[1].set_yticks(tick_marks, class_names)

    plt.show()


def plot_roc_curve_and_confusion_matrix_multiclass(fpr, tpr, conf_matrix, class_names):
    fig, axis = plt.subplots(1, 2, figsize=(12, 6))

    # Plot ROC curve
    if isinstance(fpr, dict):  # Multi-class
        for i in fpr:
            RocCurveDisplay(fpr=fpr[i], tpr=tpr[i], roc_auc=None, estimator_name=f"class {i}").plot(ax=axis[0])
    axis[0].set_title("ROC Curve")
    axis[0].legend(loc="lower right")

    # Plot confusion matrix
    im = axis[1].imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    axis[1].set(xticks=np.arange(conf_matrix.shape[1]),
                yticks=np.arange(conf_matrix.shape[0]),
                xticklabels=class_names,
                yticklabels=class_names,
                title="Confusion Matrix",
                ylabel="True Label",
                xlabel="Predicted Label")

    # Rotate the tick labels and set their alignment
    plt.setp(axis[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    fmt = "d"
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            axis[1].text(j, i, format(conf_matrix[i, j], fmt),
                         ha="center", va="center",
                         color="white" if conf_matrix[i, j] > thresh else "black")

    axis[1].set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def get_confusion_matrix_metrics(y_test, y_pred):
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    if conf_matrix.shape == (2, 2):
        # For binary classification
        tn, fp, fn, tp = conf_matrix.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    else:
        # For multiclass, use sklearn's classification metrics
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        accuracy = accuracy_score(y_test, y_pred)

    return conf_matrix, precision, recall, f1, accuracy


def evaluate_model(model, name, X_test, y_test, plot_graphs, class_names=None):
    """Evaluates the model and returns the results as a dictionary."""
    if name == "NeuralNetworktf":
        y_proba = model.predict(X_test)
        c_threshold = 0.5
        y_pred = np.where(y_proba < c_threshold, 0, 1)
    else:
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            raise RuntimeError(f"Error while predicting with model {name}: {str(e)}")

    # Initialize variables
    conf_matrix = None
    accuracy = None
    precision = None
    recall = None
    f1 = None

    # Check if the classification is binary
    if len(set(y_test)) == 2:  # Binary classification
        y_proba = model.predict_proba(X_test)[:, 1]
        if np.any(y_test == 1):  # Ensure there are positive samples
            auc = roc_auc_score(y_test, y_proba)
            conf_matrix, precision, recall, f1, accuracy = get_confusion_matrix_metrics(y_test, y_pred)
        else:
            auc = None
            precision = recall = f1 = accuracy = None
        result = {
            'model': name,
            'roc_auc': 100 * auc if auc is not None else 'Undefined',
            'accuracy': 100 * accuracy if accuracy is not None else 'Undefined',
            'precision': 100 * precision if precision is not None else 'Undefined',
            'recall': 100 * recall if recall is not None else 'Undefined',
            'f1_score': 100 * f1 if f1 is not None else 'Undefined',
            'true_positives': None if conf_matrix is None else conf_matrix.ravel()[3] if len(conf_matrix.ravel()) == 4 else 'Undefined',
            'false_positives': None if conf_matrix is None else conf_matrix.ravel()[1] if len(conf_matrix.ravel()) == 4 else 'Undefined',
            'true_negatives': None if conf_matrix is None else conf_matrix.ravel()[0] if len(conf_matrix.ravel()) == 4 else 'Undefined',
            'false_negatives': None if conf_matrix is None else conf_matrix.ravel()[2] if len(conf_matrix.ravel()) == 4 else 'Undefined'
        }
    else:  # Multi-class classification
        y_proba = model.predict_proba(X_test)
        auc_per_class = roc_auc_score(y_test, y_proba, multi_class="ovr", average=None)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
        accuracy = np.mean(y_test == y_pred)

        # Create strings for each class
        roc_auc_str = ", ".join([f"class {i} ({100 * score:.2f})" for i, score in enumerate(auc_per_class)])
        precision_str = ", ".join([f"class {i} ({100 * score:.2f})" for i, score in enumerate(precision)])
        recall_str = ", ".join([f"class {i} ({100 * score:.2f})" for i, score in enumerate(recall)])
        f1_str = ", ".join([f"class {i} ({100 * score:.2f})" for i, score in enumerate(f1)])

        # Construct the result dictionary
        result = {
            'model': name,
            'accuracy': 100 * accuracy,
            'f1_score': 100 * np.mean(f1),
            'precision': 100 * np.mean(precision),
            'recall': 100 * np.mean(recall),
            'f1_score_class': f1_str,
            'precision_class': precision_str,
            'recall_class': recall_str,
            'roc_auc_class': roc_auc_str
        }

    if plot_graphs:
        if len(set(y_test)) == 2:  # Binary classification
            if auc is not None:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                plot_roc_curve_and_confusion_matrix_binary(fpr, tpr, conf_matrix)
        else:  # Multi-class classification
            fpr = {}
            tpr = {}
            for i in range(y_proba.shape[1]):
                fpr[i], tpr[i], _ = roc_curve(y_test == i, y_proba[:, i])
            conf_matrix = confusion_matrix(y_test, y_pred)
            plot_roc_curve_and_confusion_matrix_multiclass(fpr, tpr, conf_matrix, class_names)  # No confusion matrix for multi-class case

    return result


def calculate_top_accuracy(df, col_target, top_perc_dict, metrics):
    """Calculates a metric taking into the weights included in 'the top_perc_dict' dictionary."""

    total_records = len(df.index)
    total_target = len(df[df[col_target] == 1])
    total_score = 0
    for col_name, (percentage, weight) in top_perc_dict.items():
        num_records = int(percentage * total_records / 100)
        df_top = df.head(num_records).copy()
        top_target_regords = len(df_top[df_top[col_target] == 1])
        top_regords = len(df_top[df_top[col_target] >= 0])
        metrics[f"{col_name} count"] = top_target_regords
        metrics[f"{col_name} perc"] = 100 * top_target_regords / total_target
        metrics[f"{col_name} acc"] = 100 * top_target_regords / top_regords
        total_score += metrics[f"{col_name} perc"] * weight
    metrics["top weighted score"] = total_score
    return metrics


def get_ideal_target_df(df, col_target, col_rank):
    # This is the ideal dataframe where all the target records are ranked at the top
    df = df.copy()
    ideal_df = df.sort_values(by=col_target, ascending=False)
    ideal_df[col_rank] = list(range(1, len(df.index) + 1))
    return ideal_df


def calculate_median_accuracy(df, col_target, col_rank):
    # This is the target dataframe
    target_df = df[(df[col_target] == 1)].copy()
    ideal_df = get_ideal_target_df(target_df, col_target, col_rank)

    scores = []
    for df_score in [target_df, ideal_df]:
        median_pos = df_score[col_rank].median()
        median_score =  1 - (median_pos / len(df))
        scores.append(median_score)

    # Normalized median score
    norm_median_score = 100. * scores[0] / scores[1] if scores[1] > 0 else 0.0
    return norm_median_score


def calculate_power_accuracy(df, col_target, col_rank):
    df = df.copy()
    number_of_records = len(df)
    ideal_df = get_ideal_target_df(df, col_target, col_rank)

    scores = []
    for df_score in [df, ideal_df]:
        df_score['pos_weight'] = (100 - 100.0 * (df_score[col_rank] - 1) / number_of_records) ** 3 / 10000
        power_score = df_score.loc[df_score[col_target] == 1, 'pos_weight'].mean()
        scores.append(power_score)

    # Normalized power score
    norm_power_score = 100. * scores[0] / scores[1] if scores[1] > 0 else 0.0
    return norm_power_score


def calc_dcg(relevance_scores):
    """
    Calculate the Discounted Cumulative Gain (DCG)
    """
    positions = list(range(2, len(relevance_scores) + 1))
    scores = relevance_scores[1:]

    dcg = relevance_scores[0]
    for pos, reli in zip(positions, scores):
        dcg += reli / math.log2(pos)
    return dcg


def calculate_ndcg_accuracy(df, col_target, col_rank, top_rank_perc):
    """Compute the normalized Discounted Cumulative Gain (nDCG) at position k for a test set.

    Parameters:
    df (pd.DataFrame): The test set containing the target and rank columns.
    col_target (str): The name of the target column.
    col_rank (str): The name of the rank column.
    top_rank_perc (float): The percentage of the top ranks to consider for nDCG.

    Returns:
    float: The nDCG value.
    """
    # Ensure the dataframe is sorted by the rank column
    df = df.copy()
    df = df.sort_values(by=col_rank)
    number_of_records = len(df)
    k = int(top_rank_perc * number_of_records)

    # Get the actual scores (relevance scores)
    actual_scores = df[col_target].values[:k]

    # Get the ideal dataframe and corresponding scores
    ideal_df = get_ideal_target_df(df, col_target, col_rank)
    ideal_scores = ideal_df[col_target].values[:k]

    scores = []
    for dcg_scores in [actual_scores, ideal_scores]:
        scores.append(calc_dcg(dcg_scores))

    # Calculate nDCG
    norm_dcg = 100. * scores[0] / scores[1] if scores[1] > 0 else 0.0
    return norm_dcg


def calculate_rank_metrics_for_each_model(df, model_name, col_target, col_rank, top_perc_dict, top_rank_perc):
    df = df.copy()
    df = df.sort_values(by=col_rank)

    metrics = {}
    metrics = calculate_top_accuracy(df, col_target, top_perc_dict, metrics)
    metrics['median'] = calculate_median_accuracy(df, col_target, col_rank)
    metrics['power'] = calculate_power_accuracy(df, col_target, col_rank)
    metrics['nDCG'] = calculate_ndcg_accuracy(df, col_target, col_rank, top_rank_perc)
    metrics['Model'] = model_name
    return metrics


def calculate_probabilities_for_all_models(df_features, df_predictions, models):
    df_predictions = df_predictions.copy()
    # Calculate probabilities for each model
    for model_name, model in models.items():
        if hasattr(model, "predict_proba"):
            print(f"Predict probabilities for {model_name}")
            df_predictions[model_name] = model.predict_proba(df_features)[:, 1]
        else:
            if model_name == "LightGBM":
                print(f"Create probabilities for {model_name}")
                df_predictions[model_name] = model.predict(df_features, pred_contrib=True)
                df_predictions[model_name] = df_predictions[model_name].apply(lambda x:  1 / (1 + np.exp(-x)))
            else:
                print(f"Predict classes for {model_name}")
                df_predictions[model_name] = model.predict(df_features)
    return df_predictions


def create_ranking(df, tuned_models, col_id, col_year, col_diameter, col_target, columns_to_exclude, top_perc_dict, col_rank, class_labels, top_rank_perc, score_sort_list, verbose):
    df = df.copy()
    # Get predictions for the test dataset using all tuned models and the ensemble
    df_features = df.copy()
    cols_to_keep = [col for col in [col_id, col_target, col_year, col_diameter] if col in df.columns]
    df_predictions = df[cols_to_keep].copy()
    # Create the dataframe used in models
    drop_columns = [x for x in columns_to_exclude if x in list(df_features.columns)]
    df_features.drop(columns=drop_columns, inplace=True)
    # Create a list with the ranking numbers
    rank_list = list(range(1, len(df.index) + 1))
    # Create a list with the ranking methods
    result_rows_indexes = ['Voting', 'Average', 'Median']

    # Calculate probabilities for all models and add them as column in the df_predictions
    df_predictions = calculate_probabilities_for_all_models(df_features, df_predictions, tuned_models)

    model_names = list(tuned_models.keys())
    # move 'Voting' to to end of the list
    if "Voting" in model_names:
        model_names.remove("Voting")
        model_names.append('Voting')
    else:
        result_rows_indexes.remove("Voting")

    metrics_results = []
    # Rank predictions for each model
    for model_name in model_names:
        col_model_rank = f"{model_name}_rank"
        sort_cols = [model_name] + [col for col in [col_year, col_diameter] if col in df_predictions.columns]
        sort_order = [False] + [True] * (len(sort_cols) - 1)
        df_predictions = df_predictions.sort_values(by=sort_cols, ascending=sort_order)
        df_predictions[col_model_rank] = rank_list
        metrics_results.append(calculate_rank_metrics_for_each_model(df_predictions, model_name, col_target, col_model_rank, top_perc_dict, top_rank_perc))

    # Calculate average rank for each record after removing highest and lowest rank number
    model_ranks = [f"{model_name}_rank" for model_name in model_names if model_name!="Voting"]
    if len(model_ranks) < 3:
        df_predictions['Average_rank'] = df_predictions[model_ranks].mean(axis=1)
    else:
        df_predictions['min_rank'] = df_predictions[model_ranks].min(axis=1)
        df_predictions['max_rank'] = df_predictions[model_ranks].max(axis=1)
        df_predictions['sum_rank'] = df_predictions[model_ranks].sum(axis=1)
        df_predictions['Average_rank'] = df_predictions['sum_rank'] - df_predictions['min_rank'] - df_predictions['max_rank']
    df_predictions = df_predictions.sort_values(by="Average_rank")
    df_predictions['Average_rank'] = rank_list
    metrics_results.append(calculate_rank_metrics_for_each_model(df_predictions, "Average", col_target, "Average_rank", top_perc_dict, top_rank_perc))

    # Calculate the median rank
    df_predictions['Median_rank'] = df_predictions[model_ranks].median(axis=1)
    df_predictions = df_predictions.sort_values(by="Median_rank")
    df_predictions['Median_rank'] = rank_list
    metrics_results.append(calculate_rank_metrics_for_each_model(df_predictions, "Median", col_target, "Median_rank", top_perc_dict, top_rank_perc))

    # Print the results
    metrics_results = pd.DataFrame.from_dict(metrics_results)
    metrics_results = metrics_results.set_index("Model")
    if verbose:
        print("\nStatistics of the ranked records:")
        metrics_styled = metrics_results.style.apply(cm_l.highlight_df_rows(result_rows_indexes, "honeydew"), axis=1)
        display(metrics_styled)

    print(f"\nNumber of ranked rencords: {len(df)}")
    print(f"Number of {class_labels[1]} records: {len(df[df[col_target]==1])}")

    # Selecting the rows with index names 'Voting', 'Average', and 'Median' to find the maximum score
    selected_rows = metrics_results.loc[result_rows_indexes].copy()
    # Sort based on the score list and select the best ones
    max_best_method_row = selected_rows.sort_values(by=score_sort_list, ascending=False).head(1)
    max_best_method_row = max_best_method_row.reset_index().rename(columns={'Model':'Method'}).to_dict(orient="records")[0]
    best_method = score_sort_list[0]

    # Create final rank based on the best power score
    score_value = max_best_method_row[best_method]
    df_predictions.rename(columns={f"{max_best_method_row['Method']}_rank": col_rank}, inplace=True)
    fin_columns = [col_id, col_rank]
    if "Voting" in model_names:
        df_predictions.rename(columns={'Voting':'Voting_prob'}, inplace=True)
        df_predictions['Voting_prob'] = 100 * df_predictions['Voting_prob']
        fin_columns.append("Voting_prob")
    print(f"\n Best ranking method by {max_best_method_row['Method']}, Metric: {best_method}, Score={round(score_value, 2)}.")
    return df_predictions[fin_columns], max_best_method_row


def plot_ranked_records(df_ranked, col_rank, max_bins, col_target, class_labels, col_likelihood):
    # Define bins
    df = df_ranked.copy()
    df = df.sort_values(by=col_rank)
    max_rank = df[col_rank].max()
    bins = np.linspace(0, max_rank, max_bins+1)

    # Assigning bins
    df['bin'] = pd.cut(df[col_rank], bins=bins, labels=False, include_lowest=True)

    # Calculating the average rank for each bin
    bin_averages = df.groupby('bin')[col_rank].max().reset_index(name="bin_max")

    # Calculating counts in each bin
    bin_data = df.groupby(['bin', col_target]).size().reset_index(name="count")

    # Create a MultiIndex with all possible combinations of 'bin' and 'col_target'
    index = pd.MultiIndex.from_product([bin_data['bin'].unique(), df[col_target].unique()], names=['bin', col_target])

    # Reindex bin_data to include all possible combinations
    bin_data = bin_data.set_index(['bin', col_target]).reindex(index, fill_value=0).reset_index()

    # Merge with bin_averages
    bin_data = pd.merge(bin_data, bin_averages, on="bin", how="left")
    bin_data['bin_max'] = bin_data['bin_max'].fillna(0).astype(int)
    bin_data = bin_data.sort_values(by=['bin_max', col_target])

    # Splitting data for separate plotting
    df_class_1 = bin_data[bin_data[col_target] == 1].copy()
    df_class_0 = bin_data[bin_data[col_target] == 0].copy()
    df_class_m1 = bin_data[bin_data[col_target] == -1].copy()

    # Plotting
    plt.figure(figsize=(20, 16))

    g1 = sns.barplot(data=df_class_m1, y="bin_max", x="count", color="lightgrey", label=class_labels[-1].capitalize(), orient="h", alpha=0.8, width=1)
    g2 = sns.barplot(data=df_class_0, y="bin_max", x="count", color="lightgreen", label=class_labels[0].capitalize(), orient="h", alpha=0.8, width=0.7)
    g3 = sns.barplot(data=df_class_1, y="bin_max", x="count", color="red", label=class_labels[1].capitalize(), orient="h", alpha=0.8, width=0.4)

    # Annotate counts on bars in g3 only
    for bar in g3.containers[-1]:
        height = bar.get_height()
        width = bar.get_width()
        y = bar.get_y()
        if int(width) > 0:  # Exclude values equal to 0
            g3.annotate(format(width, ".0f"),
                        xy=(width, y + height / 2),
                        xytext=(9, 0),
                        textcoords="offset points",
                        ha="left", va="center",
                        fontsize=10,
                        color="black")

    # Plot High-Medium likelihood lines
    if col_likelihood in df.columns:
        for likelihood_class, color in {'very high': 'red', 'high': 'orange', 'medium': 'yellow', 'low': 'green'}.items():
            df_tmp = df[df[col_likelihood].str.lower() == likelihood_class]
            rank_pos = df_tmp[col_rank].max()
            # Find the closest bin for rank
            closest_pos = min(bin_data['bin_max'], key=lambda x: abs(x - rank_pos))
            y_pos = bin_data[bin_data['bin_max']==closest_pos].iloc[0]['bin']
            plt.axhline(y=y_pos, color=color, linestyle="--", label=f"{likelihood_class.capitalize()} Risk")

    plt.title(f"Distribution of {col_target.capitalize()}")
    plt.ylabel("Rank position")
    plt.xlabel("Count")
    plt.legend(title="", loc="lower right")
    plt.show()


def compare_age_ml_methods(df_age, df_ml, col_target, col_rank, top_stat_dict, top_rank_perc, plot_graphs):
    compare_results = []
    for test, df in {'by_age': df_age, 'by_ml': df_ml}.items():
        result = calculate_rank_metrics_for_each_model(df, test, col_target, col_rank, top_stat_dict, top_rank_perc)
        compare_results.append(result)
    # Print the results
    compare_results = pd.DataFrame.from_dict(compare_results)
    display(compare_results)

    # Plotting a bar comparing the results
    if plot_graphs:
        stat_cols = [col for col in compare_results.columns if 'count' in col]
        df_melted = compare_results[['Model'] + stat_cols].copy()
        df_melted = df_melted.melt(id_vars="Model", value_vars=stat_cols, var_name="Stats", value_name="Count", ignore_index=True)
        plt.figure(figsize=(15, 10))
        ax = sns.barplot(data=df_melted, x="Stats", y="Count", hue="Model", palette={'by_age': 'lightsalmon', 'by_ml': 'lightblue'})
        # Annotate the bars with the count numbers
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(format(p.get_height(), ".0f"),
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha = "center", va = "center",
                                xytext = (0, 9),
                                textcoords = "offset points")
        plt.title("")
        plt.show()


#------------------------------------------PREDICTIONS----------------------------------------------
def create_likelihood_classification(df_ranked, col_target, col_rank, col_likelihood):
    """Classifies the ranked dataframe into three classes: High-Medium-Low based on the ranking positions of the target.
    0 - 25% Very high
    25% - 75% High
    75% - IQR Medium
    IQR - 100% low
    rest - Very low
    """
    df_ranked = df_ranked.copy()

    df_target = df_ranked[df_ranked[col_target]==1].copy()
    num_target = len(df_target)
    # Get the rank position based on the quantiles
    rank_vhigh = round(df_target[col_rank].quantile(0.25))
    rank_high = round(df_target[col_rank].quantile(0.75))
    # Get the IQR for high and medium class
    _, rank_medium = cm_l.get_IQR_outlier_limits(df_target, field_name=col_rank, quantiles=[0.25, 0.75], max_range=1.5, verbose=False, plot=False)
    rank_medium = round(rank_medium)
    # Get the rank position of the 100%
    rank_100 = df_target.iloc[-1][col_rank]

    num_outliers = min(max(round(num_target * 0.01), 2), 5)
    if num_outliers:
        df_target_outliers = df_target.tail(num_outliers).copy()
        df_target_outliers['rank_dif'] = df_target_outliers[col_rank].shift(periods=-1, fill_value=0) - df_target_outliers[col_rank]
        # Check if the last ranked target records have at least 30% difference in the ranking position from the previous ranked records
        df_target_outliers = df_target_outliers[(df_target_outliers['rank_dif'] >= round(0.3 * len(df_ranked)))]
        if len(df_target_outliers):
            rank_100 = df_target_outliers.sort_values(by="rank_dif", ascending=False).iloc[0][col_rank]
            if rank_100 <= rank_medium:
                rank_medium = round(df_target[col_rank].quantile(0.85))

    df_ranked.loc[(df_ranked[col_rank] <= rank_vhigh), col_likelihood] = "Very High"
    df_ranked.loc[(df_ranked[col_rank] > rank_vhigh) & (df_ranked[col_rank] <= rank_high), col_likelihood] = "High"
    df_ranked.loc[(df_ranked[col_rank] > rank_high) & (df_ranked[col_rank] <= rank_medium), col_likelihood] = "Medium"
    df_ranked.loc[(df_ranked[col_rank] > rank_medium) & (df_ranked[col_rank] <= rank_100), col_likelihood] = "Low"
    df_ranked.loc[(df_ranked[col_rank] > rank_100), col_likelihood] = "Very Low"
    return df_ranked


def display_statistics_of_predictions(df_ranked, col_target, col_likelihood, class_labels):
    """Statistics on all ranked rectords based on the likelihood classification."""
    df = df_ranked.copy()
    classes = ['Very High', 'High', 'Medium', 'Low', 'Very Low']

    no_target_name = class_labels[0]
    target_name = class_labels[1]
    other_name = class_labels[-1]

    df_target = df[df[col_target] == 1]
    df_notarget = df[df[col_target] == 0]
    df_other = df[df[col_target] == -1]

    c = df_target[col_likelihood].value_counts(dropna=False)
    p = df_target[col_likelihood].value_counts(dropna=False, normalize=True).mul(100).round(1).astype(str) + "%"
    f0 = pd.DataFrame(pd.concat([c, p], axis=1, keys=['counts', '%']))
    f0[target_name] = f0.apply(lambda row: f"{row['counts']} ({row['%']})", axis=1)

    c = df_notarget[col_likelihood].value_counts(dropna=False)
    p = df_notarget[col_likelihood].value_counts(dropna=False, normalize=True).mul(100).round(1).astype(str) + "%"
    f1 = pd.DataFrame(pd.concat([c, p], axis=1, keys=['counts', '%']))
    f1[no_target_name] = f1.apply(lambda row: f"{row['counts']} ({row['%']})", axis=1)

    c = df_other[col_likelihood].value_counts(dropna=False)
    p = df_other[col_likelihood].value_counts(dropna=False, normalize=True).mul(100).round(1).astype(str) + "%"
    f2 = pd.DataFrame(pd.concat([c, p], axis=1, keys=['counts', '%']))
    f2[other_name] = f2.apply(lambda row: f"{row['counts']} ({row['%']})", axis=1)

    df_stat = pd.DataFrame(index=classes)
    df_stat = pd.concat([df_stat, f0[[target_name]], f1[[no_target_name]], f2[[other_name]]], axis=1)
    df_stat = df_stat.fillna(0)

    df_stat['Accuracy'] = None
    for index, row in df_stat.iterrows():
        # Convert the target and no_target counts to strings before splitting
        num_class_target = str(row[target_name]).split(" ")[0] if " " in str(row[target_name]) else str(row[target_name])
        num_class_notarget = str(row[no_target_name]).split(" ")[0] if " " in str(row[no_target_name]) else str(row[no_target_name])

        num_class_target = int(num_class_target.strip())
        num_class_notarget = int(num_class_notarget.strip())

        num_class_all = num_class_target + num_class_notarget
        df_stat.loc[index, 'Accuracy'] = f"{round(100 * int(num_class_target) / int(num_class_all), 2)}%"

    print(f"\nStatistics of Likelihood of {target_name}:")
    display(df_stat)


def calculate_likelihood(t, n, p1, p2, s, min_s, decay_rate):
    """Returns a likelihood value that corresponds to the given rank position <t>. This value is calculated
    from curve fitting a function that starts with exponential decay and ends up linear.
    The calculated likelihood values range from 1.0 for the first pipe in the ranking and 0 for the last pipe.

    Keyword arguments:
    t -- ranking of the pipe
    n -- total number of pipes
    p1, p2, s -- parameters that determine the curve of likelihood PDF
    min_s -- minimum % point in the ranking where function changes from exponential to linear
    decay_rate -- the decay rate of the exponential function
    """
    # p1 has to be positive to make sure when t == N, the last point is above 0
    # p2 has to be positive to make sure when t == N * min_s, the start point of linear function is above 0
    l_curve = abs(p1) * np.exp(decay_rate * t / (n * (min_s + abs(s)))) + \
              abs(p2) - abs(p2) * (t - n * (min_s + abs(s))) / (n * (1 - (min_s + abs(s))))
    return l_curve


def calculate_prob_distribution(df, col_target, col_rank, class_labels, plot_graphs=True):
    """Calculates and returns the parameters of the PDF (Probability Distribution Function) that represents the
    probability of event based on the given <events_df> that occurred in the given <ranked_pipes_df>.
    """
    df = df.copy()
    max_bins = 100

    df = df.sort_values(by=col_rank)
    max_rank = df[col_rank].max()
    bins = np.linspace(0, max_rank, max_bins+1)

    # assigning bins
    df['bin'] = pd.cut(df[col_rank], bins=bins, labels=False, include_lowest=True)

    # Calculating the count rank for each bin
    bin_mid = df.groupby('bin')[col_rank].median().reset_index(name="bin_mid")

    # Calculating counts in each bin
    bin_data = df[df[col_target]==1].groupby(['bin', col_target]).size().reset_index(name="bin_count")

    bin_data = pd.merge(bin_data, bin_mid, on="bin", how="right")
    bin_data = bin_data.fillna(0)

    bin_data['bin_count'] = bin_data['bin_count'].astype(int)
    bin_data['bin_mid'] = pd.to_numeric(bin_data['bin_mid'])
    bin_data['prob'] = 100. * bin_data['bin_count'] / (len(df) / max_bins)
    x = bin_data['bin_mid'].astype("float64")
    y = bin_data['prob']
    # build a function that starts as exponential, and then at some point becomes linear
    N = len(df)
    print(f"N (number of ranked pipes) is {N}")
    # decay rate
    decay_rate_range = np.linspace(-20, -5, 16)
    # minimum % point in the ranking where function changes from exponential to linear
    min_s_range = np.linspace(0.05, 0.3, 20)
    best_fit_params = None
    selected_p_cov = None
    min_squared_error = np.inf
    curve_fit_params = []
    predicted_vals = []

    # loop through all the possible combinations of decay_rate and min_s
    for decay_rate in decay_rate_range:
        for min_s in min_s_range:
            try:
                def target_fit(t, p1, p2, s):
                    return calculate_likelihood(t, N, p1, p2, s, min_s, decay_rate)

                bounds = ([1, 0, 0.01], [np.inf, np.inf, 0.7])
                p_opt, p_cov = curve_fit(target_fit, x, y, p0=(1, 0, 0.01), bounds=bounds)
                p1, p2, s = p_opt[0], p_opt[1], p_opt[2]
                if (s + min_s) < 1:
                    y_vals = target_fit(x, p1, p2, s)
                    # calculate the mean squared error
                    mse = mean_squared_error(y, y_vals)
                    # find the best model with the minimum mean squared error
                    if mse < min_squared_error:
                        min_squared_error = mse
                        best_fit_params = (p1, p2, s, min_s, decay_rate)
                        predicted_vals = y_vals
                        selected_p_cov = p_cov
            except Exception as e:
                cm_l.print_formatted_txt(f"Curve fit params could not be calculated: {str(e)} for curve fit "
                                         f"parameters: p1={p1}, p2={p2},s={s}, {min_s=}, {decay_rate=}. "
                                         f"Continue with other parameter values.", "WARNING")

    if best_fit_params is not None:
        p1, p2, s, min_s, decay_rate = best_fit_params
        # calculate residuals and chi-square
        residuals = y - predicted_vals
        chi_square = np.sum((residuals ** 2) / predicted_vals)
        # calculate reduced chi-square
        n, m = len(y), len(selected_p_cov)  # number of data points, number of parameters
        degrees_of_freedom = n - m
        reduced_chi_square = chi_square / degrees_of_freedom
        std_devs = np.sqrt(np.diag(selected_p_cov))  # standard deviations of the parameters
        # print results
        print(f"Curve fit parameters: {p1=}, {p2=}, {s=}, {min_s=}, {decay_rate=}")
        print(f"Function switch point: {int((min_s + s) * 100)}% of ranking")
        print(f"Min squared error: {min_squared_error}")
        print(f"Standard deviations of the parameters: {std_devs}")
        print(f"Chi-square: {chi_square}")
        if reduced_chi_square > 5:
            cm_l.print_formatted_txt(f"Reduced chi-square: {reduced_chi_square} is more than 5; "
                                     f"check the curve fit", "WARNING")
        else:
            print(f"Reduced chi-square: {reduced_chi_square}")

        # save the curve fit parameters
        curve_fit_params = pd.DataFrame({'N': N, 'p1': p1, 'p2': p2, 's': s, 'min_s': min_s, 'decay_rate': decay_rate},
                                        index=[0])
        t = 1
        print(f"Value at #first: {target_fit(t, p1, p2, s)}")
        t = N
        print(f"Value at #last: {target_fit(t, p1, p2, s)}")
    else:
        cm_l.print_formatted_txt(f"Curve fit params could not be calculated.", "ERROR")

    if plot_graphs:
        fig, ax = plt.subplots(figsize=(20, 10))
        if best_fit_params is not None:
            ax.plot(x, predicted_vals, color="red", label="Fitted Curve")
        ax.scatter(x, y, marker="s", label="Probabilities", color="blue")

        plt.xlabel("rank position")
        plt.ylabel("Likelihood")
        ax.xaxis.set_major_locator(plt.MaxNLocator(20))
        ax.yaxis.set_major_locator(plt.MaxNLocator(20))
        plt.grid(True)
        plt.legend(loc=1)
        plt.title(f"Likelihood of {class_labels[1]}")
        plt.show()

    return curve_fit_params


def calculate_probabilities(curve_fit_params, df_rankings, col_prob, col_rank):
    """Calculates the LoLead values based on the ranking position of the pipes and the PDF
    (Probability Distribution Function).
    """
    df_rankings = df_rankings.copy()
    if len(curve_fit_params) > 0:
        n = curve_fit_params.iloc[0]['N']
        p1 = curve_fit_params.iloc[0]['p1']
        p2 = curve_fit_params.iloc[0]['p2']
        s = curve_fit_params.iloc[0]['s']
        min_s = curve_fit_params.iloc[0]['min_s']
        decay_rate = curve_fit_params.iloc[0]['decay_rate']
        df_rankings[col_prob] = calculate_likelihood(df_rankings[col_rank], n, p1, p2, s, min_s, decay_rate)
        df_rankings.loc[df_rankings[col_prob] < 0, col_prob] = 0
        df_rankings.loc[df_rankings[col_prob] > 100, col_prob] = 100
    return df_rankings


#---------------------------------------TUNING--------------------------------------------
def read_results_files(folder_path, sort_list, col_features, col_models):
    """
    Reads all CSV files in the specified folder and returns a concatenated DataFrame.

    :param folder_path: The path to the folder containing CSV files
    :param sort_list: The list of columns to sort the DataFrame by
    :return: A DataFrame with all results concatenated along columns
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if len(csv_files) == 0:
        return None

    data_frames = []
    col_value = "value"
    col_parameter = "parameter"
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        for col in df.columns:
            if "value" in col:
                col_value = col
            elif "parameter" in col:
                col_parameter = col

        df = df.rename(columns={col_value: file, col_parameter: 'file'})
        df = df.set_index("file")
        data_frames.append(df)

    # Concatenate DataFrames along columns, ensuring correct alignment
    df_results = pd.concat(data_frames, axis=1)
    df_results = df_results.T

    # Drop duplicates based on specified columns
    dupl_cols = sort_list + [col_features, col_models]
    df_results = df_results.drop_duplicates(subset=dupl_cols)

    # Ensure sort_list columns exist in the DataFrame
    missing_columns = [col for col in sort_list if col not in df_results.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} are not in the DataFrame")

    # Sort the DataFrame by the given columns
    for col in sort_list:
        if df_results[col].dtype != float:
            df_results[col] = df_results[col].astype(float)
    df_results = df_results.sort_values(by=sort_list, ascending=False)

    return df_results


def check_finished_tune_runs(df_tune_results, col_features, features_to_check, score_method, seed=None):
    if df_tune_results is not None:
        to_check_list = sorted([x.strip() for x in features_to_check])
        for index, row in df_tune_results.iterrows():
            checked_list = sorted([x.strip() for x in row[col_features].split(",")])
            if seed is None:
                if to_check_list == checked_list:
                    return row[score_method]
            else:
                if to_check_list == checked_list and int(row['seed'])==int(seed):
                    return row[score_method]
    return 0


def display_sensitivity_results(df_results, metric):
    df_results = df_results.copy()
    df_data = df_results.copy()
    _, _, numerical_cols = cm_l.get_column_types(df_results)

    for col_name in numerical_cols:
        data = list(df_data[col_name])
        conf_level, conf_level_text = cm_l.calculate_confidence_level_from_list(data, verbose=False)
        conf_interval, conf_interval_text = cm_l.calculate_confidence_interval_from_list(data, conf_level=95, test=col_name, verbose=False)

        mean_data = np.mean(data)
        stdev_data = np.std(data)
        min_data, max_data = min(data), max(data)
        df_results[col_name] = df_results[col_name].astype(str)
        df_results.loc['min', col_name] = min_data
        df_results.loc['max', col_name] = max_data
        df_results.loc['var', col_name] = statistics.variance(data)
        df_results.loc['sdev', col_name] = stdev_data
        df_results.loc['average', col_name] = mean_data
        df_results.loc['median', col_name] = np.median(data)
        df_results.loc['max-min', col_name] = max_data - min_data
        df_results.loc['conf_level', col_name] = conf_level_text
        df_results.loc['conf_interval', col_name] = conf_interval_text

    # Create the point plot
    plt.figure(figsize=(18, 8))
    sns.pointplot(x=df_data.index, y=df_data[metric], linestyle="none", marker="D", color="blue")
    plt.axhline(y=mean_data, color="red", linestyle="-", label="Mean")
    plt.axhline(y=mean_data+stdev_data, color="orange", linestyle=":", label=f"+- 1 St.Dev")
    plt.axhline(y=mean_data-stdev_data, color="orange", linestyle=":")
    plt.axhline(y=mean_data+2*stdev_data, color="green", linestyle="--", label=f"+- 2 St.Dev")
    plt.axhline(y=mean_data-2*stdev_data, color="green", linestyle="--")
    plt.title("Sensitivity Analysis")
    plt.xlabel("Seed")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    # Highlight the statistics rows
    def highlight_rows(s):
        if s.name in ['min', 'max', 'var', 'sdev', 'average', 'median', 'max-min', 'conf_level', 'conf_interval']:
            return ['background-color: yellow']*len(s)
        else:
            return ['']*len(s)
    display(df_results.style.apply(highlight_rows, axis=1))


def scale_features(X):
    X = X.copy()

    encoders = {}
    scalers = {}
    # Loop through each feature
    for column in X.columns:
        if X[column].dtype in ['object', 'string', 'boolean', 'str']:
            unique_values = X[column].nunique()
            if unique_values == 2:
                print(f"Selecting LabelEncode for column: {column}.")
                encoder = LabelEncoder()
            elif 3 <= unique_values <= 5:
                print(f"Selecting OneHotEncoder for column: {column}.")
                encoder = OneHotEncoder(sparse=False, drop="first")
            else:
                print(f"Selecting TargetEncoder for column: {column}.")
                encoder = TargetEncoder()
            encoders[column] = encoder
        else:
            skewness = skew(X[column])
            _, p_value = normaltest(X[column])
            if p_value > 0.05 and abs(skewness) < 0.5:  # Normal distribution
                print(f"Selecting StandardScaler for column: {column}.")
                scaler = StandardScaler()
            elif abs(skewness) < 1:
                print(f"Selecting MinMaxScaler for column: {column}.")
                scaler = MinMaxScaler()
            else:
                print(f"Selecting StandardScaler for log1p of column: {column}.")
                scaler = "log1p"  #StandardScaler()
            scalers[column] = scaler
    return encoders, scalers

def apply_encoding(train_data, test_data, encoders, target):
    encoded_train = train_data.copy()
    encoded_test = test_data.copy() if isinstance(test_data, pd.DataFrame) else None

    for column, encoder in encoders.items():
        if column in encoded_train.columns:
            encoder.fit(encoded_train[column].values.ravel())
            if isinstance(encoder, LabelEncoder):
                encoded_train[column] = encoder.transform(encoded_train[column].values.ravel())
                if encoded_test is not None:
                    encoded_test[column] = encoder.transform(encoded_test[column].values.ravel())

            elif isinstance(encoder, OneHotEncoder):
                train_encoded_cols = pd.DataFrame(encoder.transform(encoded_train[[column]]),
                                                  columns=encoder.get_feature_names_out([column]), index=encoded_train.index)
                encoded_train = pd.concat([encoded_train.drop(columns=[column]), train_encoded_cols], axis=1)
                if encoded_test is not None:
                    test_encoded_cols = pd.DataFrame(encoder.transform(encoded_test[[column]]),
                                                     columns=encoder.get_feature_names_out([column]), index=encoded_test.index)
                    encoded_test = pd.concat([encoded_test.drop(columns=[column]), test_encoded_cols], axis=1)

            elif isinstance(encoder, TargetEncoder):
                encoded_train[column] = encoder.transform(encoded_train[column].values.ravel(), encoded_train[target].values.ravel(),
                                                          random_state=gb_l.SEED_NUMBER)
                if encoded_test is not None:
                    encoded_test[column] = encoder.transform(encoded_train[column].values.ravel())

    return encoded_train, encoded_test


def apply_scaling(train_data, test_data, scalers):
    scaled_train = train_data.copy()
    scaled_test = test_data.copy() if isinstance(test_data, pd.DataFrame) else None

    for column, scaler in scalers.items():
        if column in scaled_train.columns:
            if scaler == "log1p":
                scaled_train[column] = np.log1p(scaled_train[column])
                if scaled_test is not None:
                    scaled_test[column] = np.log1p(scaled_test[column])
                current_scaler = StandardScaler()
            else:
                current_scaler = scaler

            current_scaler.fit(scaled_train[[column]])
            scaled_train[column] = current_scaler.transform(scaled_train[[column]])
            if scaled_test is not None:
                scaled_test[column] = current_scaler.transform(scaled_test[[column]])

    return scaled_train, scaled_test

# def scale_features(gdf_to_scale, col_id, train_ids, base_columns, scaler="standard"):
#     """Scales the features based on the train data distributions, using the StandardScaler or RobustScaler."""
#     gdf_to_scale = gdf_to_scale.copy()

#     # check scaler
#     valid_scalers = ['minmax', 'standard', 'robust']
#     if scaler.lower() not in valid_scalers:
#         raise Exception(f"Unrecognized scaler {scaler}. Acceptable values: {valid_scalers}")
#     # Get column types
#     object_cols, numerical_cols = cm_l.get_column_types(gdf_to_scale)
#     # Filter columns for scaling
#     numerical_cols = [x for x in numerical_cols if x not in base_columns]

#     # Initialize StandardScaler and fit to training features
#     train_features = gdf_to_scale[gdf_to_scale[col_id].isin(train_ids)].copy()
#     if scaler.lower() == "robust":
#         feature_scaler = RobustScaler()
#     elif scaler.lower() == "standard":
#         feature_scaler = StandardScaler()
#     elif scaler.lower() == "minmax":
#         feature_scaler = MinMaxScaler()
#     feature_scaler.fit(train_features[numerical_cols])

#     gdf_scaled = pd.DataFrame(data=feature_scaler.transform(gdf_to_scale[numerical_cols]), columns=numerical_cols, index=gdf_to_scale.index)
#     gdf_scaled = pd.concat([gdf_to_scale.drop(columns=numerical_cols), gdf_scaled], axis=1)

#     return gdf_scaled
