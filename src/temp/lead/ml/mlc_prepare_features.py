from import_libraries import *
import libraries.common_libraries.generic_library as cm_l
import libraries.ml_common_libraries.ml_library as ml_l

class PrepareFeatures:
    def __init__(self, project):
        self.project = project

        self.gdf = None
        self.customer = self.project.customer
        self.prediction_year = self.project.prediction_year
        self.ban_year = self.project.ban_year
        self.max_diameter = self.project.max_diameter
        self.work_dir = self.project.work_dir
        self.project_dir = self.project.project_dir
        self.clean_dir = self.project.clean_dir
        self.prepare_dir = self.project.prepare_dir
        self.col_target = self.project.col_target
        self.col_id = self.project.col_id
        self.col_year = self.project.col_year
        self.col_diameter = self.project.col_diameter
        self.plot_graphs = self.project.plot_graphs
        self.verbose = self.project.verbose
        self.search_distance_in_ft = self.project.search_distance_in_ft
        self.data_random_seed = self.project.data_random_seed
        self.features = self.project.features
        self.base_columns = self.project.base_columns
        self.col_group = self.project.col_group
        self.grid_sample = self.project.grid_sample
        self.split_ratio = self.project.split_ratio
        self.cv_folds = self.project.cv_folds
        self.target_feature_names = self.project.target_feature_names
        self.extra_feature_names = self.project.extra_feature_names
        self.field_dtypes = self.project.field_dtypes
        self.feature_scaler = self.project.feature_scaler


    def prepare_process(self):
        """All feature preparation processes are here"""
        cm_l.print_formatted_txt("Preparing features process...", "SECTION")

        cm_l.print_formatted_txt("Reading cleaned data...", "SUBSECTION")
        self.gdf = cm_l.read_data(self.clean_dir, "clean_data")

        cm_l.print_formatted_txt("Keep only needed columns...", "SUBSECTION")
        self.gdf = ml_l.drop_features(self.gdf, self.base_columns, self.features)

        cm_l.print_formatted_txt(f"Drop records with {self.col_year}>{self.prediction_year}...", "SUBSECTION")
        filter_drop = (self.gdf[self.col_year]>self.prediction_year) & (self.gdf[self.col_target]!=-1)
        print(f"Drop {len(self.gdf[filter_drop])} records with {self.col_year}>{self.prediction_year}")
        self.gdf = self.gdf[~filter_drop].copy()

        cm_l.print_formatted_txt(f"Drop records with {self.col_diameter}>{self.max_diameter}...", "SUBSECTION")
        filter_drop = (self.gdf[self.col_diameter]>self.max_diameter) & (self.gdf[self.col_target]!=-1)
        print(f"Drop {len(self.gdf[filter_drop])} records with {self.col_diameter}>{self.max_diameter}")
        self.gdf = self.gdf[~filter_drop].copy()

        cm_l.print_formatted_txt(f"Drop duplicate geometries...", "SUBSECTION")
        self.gdf = self.gdf.sort_values(by=self.col_year, ascending=True)
        to_keep = self.gdf.drop_duplicates(subset="geometry", keep="first")
        filter_drop = (~self.gdf[self.col_id].isin(list(to_keep[self.col_id]))) & (self.gdf[self.col_target]!=-1)
        print(f"Drop {len(self.gdf[filter_drop])} records with duplicate geometries!")
        self.gdf = self.gdf[~filter_drop].copy()

        cm_l.print_formatted_txt("Creating extra features for all data...", "SUBSECTION")
        self.gdf = ml_l.create_extra_feature_names(self.gdf, self.col_id, self.features, self.extra_feature_names, self.plot_graphs, self.verbose)
        if self.verbose:
            display(self.gdf.head(5))
            display(skim(self.gdf))

        cm_l.print_formatted_txt("Encoding object features..", "SUBSECTION")
        gdf_encoded = ml_l.encode_categorical_features_to_dummies(self.gdf, self.base_columns)
        if self.verbose:
            display(gdf_encoded.head(5))
            display(skim(gdf_encoded))

        cm_l.print_formatted_txt("Creating target dependent features for all data...", "SUBSECTION")
        all_features = gdf_encoded.copy()
        # Create features that depend on the target variable=1
        all_features = ml_l.create_target_dependent_features(all_features, self.col_id, self.col_target, self.col_year, self.search_distance_in_ft, self.features,
                                                             self.target_feature_names, self.ban_year, self.plot_graphs)
        if self.verbose:
            display(all_features.head(5))
            display(skim(all_features))

        cm_l.print_formatted_txt("Splitting training-testing data...", "SUBSECTION")
        train_ids, test_ids = ml_l.split_into_train_test_ids(self.gdf, self.col_id, self.col_target, self.grid_sample, self.data_random_seed, self.split_ratio)
        cm_l.print_formatted_txt("Creating target dependent features for train-test data...", "SUBSECTION")
        train_test_features = ml_l.prepare_splitted_features(gdf_encoded, self.col_id, self.col_target, self.col_year, self.col_group, train_ids, test_ids,
                                                             self.search_distance_in_ft, self.features, self.target_feature_names, self.ban_year, plot_graphs=False)

        if self.plot_graphs:
            cm_l.print_formatted_txt("Train-Test distribution of all features...", "SUBSECTION")
            ml_l.plot_train_test_distributions(all_features, self.col_target, self.col_id, train_ids, test_ids)
            list_features = [self.col_target] + [x for x in list(all_features.columns) if x not in self.base_columns]
            fig_width = max(10, min(len(list_features), 20))
            cm_l.plot_correlation_matrix(all_features[list_features], figsize=(fig_width, fig_width))

        cm_l.print_formatted_txt("Creating cross validation folds...", "SUBSECTION")
        fold_train_test_features = train_test_features[train_test_features[self.col_group]=="train"].copy()
        folds = {}
        for i in range(1, self.cv_folds + 1):
            print(f"Splitting train-validate data for fold: {i}")
            train_ids, test_ids = ml_l.split_into_train_test_ids(fold_train_test_features, self.col_id, self.col_target, self.grid_sample, self.data_random_seed+100*i, self.split_ratio)
            print(f"Creating target dependent features for fold: {i}")
            folds[i] = ml_l.prepare_splitted_features(fold_train_test_features, self.col_id, self.col_target, self.col_year, self.col_group, train_ids, test_ids, self.search_distance_in_ft,
                                                      self.features, self.target_feature_names, self.ban_year, plot_graphs=False)

        cm_l.print_formatted_txt("\nScaling of numerical features...", "SUBSECTION")
        all_features = ml_l.scale_features(gdf_to_scale=all_features, col_id=self.col_id, train_ids=list(all_features[all_features[self.col_target]>=0][self.col_id]), base_columns=self.base_columns,
                                           scaler=self.feature_scaler)
        train_test_features = ml_l.scale_features(gdf_to_scale=train_test_features, col_id=self.col_id, train_ids=list(train_test_features[train_test_features[self.col_group]=="train"][self.col_id]),
                                                  base_columns=self.base_columns, scaler=self.feature_scaler)
        for i, df in folds.items():
            folds[i] = ml_l.scale_features(gdf_to_scale=df, col_id=self.col_id, train_ids=list(df[df[self.col_group]=="train"][self.col_id]), base_columns=self.base_columns, scaler=self.feature_scaler)

        cm_l.print_formatted_txt("Formating data type...", "SUBSECTION")
        print("Formating all features:")
        all_features = cm_l.format_df_dtypes(all_features, self.col_id, self.field_dtypes, verbose=self.verbose)
        if self.verbose:
            display(all_features.head())
            display(skim(all_features))

        cm_l.print_formatted_txt("Format the field types...", "SUBSECTION")
        train_test_features = cm_l.format_df_dtypes(train_test_features, self.col_id, self.field_dtypes, verbose=self.verbose)
        for i, df in folds.items():
            print(f"Formating fold {i} features:")
            folds[i] = cm_l.format_df_dtypes(df, self.col_id, self.field_dtypes, verbose=self.verbose)

        cm_l.print_formatted_txt("Exporting data...", "SUBSECTION")
        cm_l.write_data(all_features, os.path.join(self.prepare_dir, "all_features"))
        cm_l.write_data(train_test_features, os.path.join(self.prepare_dir, "train_test_features"))
        for i, df in folds.items():
            cm_l.write_data(df, os.path.join(self.prepare_dir, f"fold_{i}"))
