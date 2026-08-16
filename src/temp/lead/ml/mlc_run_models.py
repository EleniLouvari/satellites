from import_libraries import *
import global_variables as gb_l
import libraries.common_libraries.generic_library as cm_l
import libraries.ml_common_libraries.ml_library as ml_l

class RunModels:
    def __init__(self, project):
        self.project = project

    def ml_process(self):
        """All ML processes are here"""
        cm_l.print_formatted_txt("ML process...", "SECTION")

        self.gdf = self.project.gdf.copy()
        self.customer = self.project.customer
        self.work_dir = self.project.work_dir
        self.project_dir = self.project.project_dir
        self.prepare_dir = self.project.prepare_dir
        self.models_dir = self.project.models_dir
        self.results_dir = self.project.results_dir
        self.col_target = self.project.col_target
        self.col_id = self.project.col_id
        self.col_year = self.project.col_year
        self.plot_graphs = self.project.plot_graphs
        self.verbose = self.project.verbose
        self.models = self.project.models
        self.num_processes = self.project.num_processes
        self.base_columns = self.project.base_columns
        self.features = self.project.features
        self.col_group = self.project.col_group
        self.cv_folds = self.project.cv_folds
        self.apply_smote = self.project.apply_smote

        self.run_models = {}
        self.models_grid = self.project.models_grid
        self.models_lib = {
            'KNN': KNeighborsClassifier,
            'LogisticRegression': LogisticRegression,
            'DecisionTree': DecisionTreeClassifier,
            'RandomForest': RandomForestClassifier,
            'SVM': SVC,
            'LightGBM': LGBMClassifier,
            'Bagging': BaggingClassifier,
            'Bayesian': GaussianNB,
            'GradientBoost': GradientBoostingClassifier,
            #'CatBoost': CatBoostClassifier,
            'NeuralNetwork': MLPClassifier
        }

        cm_l.print_formatted_txt("Reading prepared data...", "SUBSECTION")
        df_all_features = cm_l.read_data(self.prepare_dir, "all_features")
        train_test_features = cm_l.read_data(self.prepare_dir, "train_test_features")
        folds_dfs = {}
        for i in range(1, self.cv_folds + 1):
            folds_dfs[i] = cm_l.read_data(self.prepare_dir, f"fold_{i}")

        cm_l.print_formatted_txt("Preparing train-test data...", "SUBSECTION")
        features_to_keep, features_to_remove = ml_l.select_features_for_ml(df_all_features, self.base_columns, self.features)
        print(f"Features to be used: {features_to_keep}")
        print(f"Features removed: {features_to_remove}")

        # Add hidden layers to NeuralNetwork based on the number of features
        self.models_grid['NeuralNetwork']['hidden_layer_sizes'] = [(len(features_to_keep),), (len(features_to_keep), len(features_to_keep)), (len(features_to_keep), len(features_to_keep) // 2)]

        # Append in the KNN neighbors the square root of the total records
        self.models_grid['KNN']['n_neighbors'].append(int(math.sqrt(len(df_all_features))))

        # All features
        cond = (df_all_features[self.col_target]>=0)
        X_train_all = df_all_features[cond][features_to_keep].copy()
        y_train_all = df_all_features[cond][self.col_target].copy()

        # Train features form the train-test sets
        cond = (train_test_features[self.col_group]=="train")
        X_train = train_test_features[cond][features_to_keep].copy()
        y_train = train_test_features[cond][self.col_target].copy()

        # Validation features
        folds = []
        for i, df in folds_dfs.items():
            current_fold = {}
            current_fold['fold'] = f"fold_{i}"
            cond = (df[self.col_group]=="train")
            current_fold['X_train'] = df[cond][features_to_keep].copy()
            current_fold['y_train'] = df[cond][self.col_target].copy()

            cond = (df[self.col_group]=="test")
            current_fold['X_test'] = df[cond][features_to_keep].copy()
            current_fold['y_test'] = df[cond][self.col_target].copy()
            folds.append(current_fold)

        # Clean memory
        del df_all_features
        del train_test_features
        del folds_dfs

        if self.apply_smote:
            print("\nApplying SMOTE to train data")
            X_train, y_train = ml_l.apply_smote(X_train, y_train, gb_l.SEED_NUMBER)
            for i, fold in enumerate(folds):
                folds[i]['X_train'], folds[i]['y_train'] = ml_l.apply_smote(fold['X_train'], fold['y_train'], gb_l.SEED_NUMBER)

        cm_l.print_formatted_txt("Initializing models...", "SUBSECTION")
        for name, model_class in self.models_lib.items():
            try:
                if name in self.models:
                    params = {}
                    signature = inspect.signature(model_class.__init__).parameters
                    if "random_state" in signature:
                        params["random_state"] = gb_l.SEED_NUMBER
                    if "verbosity" in signature:
                        params["verbosity"] = -1
                    elif "verbose" in signature:
                        params["verbose"] = 0
                    if hasattr(model_class(), "predict_proba"):
                        self.run_models[name] = model_class(**params)
                    elif name == "SVM":
                        base_estimator = SVC(probability=True, random_state=gb_l.SEED_NUMBER)
                        self.run_models[name] = CalibratedClassifierCV(base_estimator, method="sigmoid", cv=5)
                    else:
                        base_estimator = model_class(**params)
                        self.run_models[name] = CalibratedClassifierCV(base_estimator, method="sigmoid", cv=5)
            except Exception as e:
                raise RuntimeError(f"Error initializing model {name}: {str(e)}")

        cm_l.print_formatted_txt("Training models...", "SUBSECTION")
        for name, model in self.run_models.items():
            print(f"\nTraining {name} model...")

            if self.cv_folds > 0:
                tuned_params = ml_l.tune_model_cv(model_class=model, model_name=name,
                                                  param_dict=self.models_grid[name], folds=folds, num_processes=self.num_processes,
                                                  verbose=self.verbose, plot_graphs=self.plot_graphs)
            else:
                tuned_params = ml_l.tune_model(model_class=model, param_dict=self.models_grid[name],
                                               num_processes=self.num_processes, verbose=self.verbose)

            # Update the model parameters
            if isinstance(model, CalibratedClassifierCV):
                model.estimator.set_params(**tuned_params)
            else:
                model.set_params(**tuned_params)

            # Create a copy for the all_model
            all_model = deepcopy(model)

            # Fit the models
            model.fit(X_train, y_train)
            all_model.fit(X_train_all, y_train_all)

            # Save the models
            cm_l.save_ml_model(model, os.path.join(self.models_dir, f"{name}_train"))
            cm_l.save_ml_model(all_model, os.path.join(self.models_dir, f"{name}_all"))
