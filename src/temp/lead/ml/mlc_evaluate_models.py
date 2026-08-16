from import_libraries import *
import global_variables as gb_l
import libraries.common_libraries.generic_library as cm_l
import libraries.ml_common_libraries.ml_library as ml_l

class EvalModels:
    def __init__(self, project):
        self.project = project

    def eval_process(self):
        """All ML processes are here"""
        cm_l.print_formatted_txt("Evaluation process...", "SECTION")

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
        self.col_diameter = self.project.col_diameter
        self.plot_graphs = self.project.plot_graphs
        self.verbose = self.project.verbose
        self.models = self.project.models
        self.rank_bin_count = self.project.rank_bin_count
        self.base_columns = self.project.base_columns
        self.features = self.project.features
        self.top_stat_dict = self.project.top_stat_dict
        self.class_labels = self.project.class_labels
        self.col_rank = self.project.col_rank
        self.col_group = self.project.col_group
        self.col_likelihood = self.project.col_likelihood
        self.nDCG_top_rank_perc = self.project.nDCG_top_rank_perc
        self.score_sort_list = self.project.score_sort_list
        self.record_eval_results = self.project.record_eval_results
        self.multi_run_folder = self.project.multi_run_folder
        self.eval_col_features = self.project.eval_col_features
        self.eval_col_models = self.project.eval_col_models
        self.tuned_models = {}

        cm_l.print_formatted_txt("Reading prepared data...", "SUBSECTION")
        train_test_features = cm_l.read_data(self.prepare_dir, "train_test_features")

        cm_l.print_formatted_txt("Preparing train-test data...", "SUBSECTION")
        features_to_keep, features_to_remove = ml_l.select_features_for_ml(train_test_features, self.base_columns, self.features)
        print(f"Features to be used: {features_to_keep}")
        print(f"Features dropped: {features_to_remove}")
        all_cols_to_exclude = self.base_columns + features_to_remove

        cond = (train_test_features[self.col_group]=="train")
        X_train = train_test_features[cond][features_to_keep].copy()
        y_train = train_test_features[cond][self.col_target].copy()

        cond = (train_test_features[self.col_group]=="test")
        test_features = train_test_features[cond]
        X_test = test_features[features_to_keep].copy()
        y_test = test_features[self.col_target].copy()

        # Clean memory
        del train_test_features

        cm_l.print_formatted_txt("Reading models...", "SUBSECTION")
        for model in self.models:
            self.tuned_models[model] = cm_l.read_ml_model(os.path.join(self.models_dir, f"{model}_train"))

        cm_l.print_formatted_txt("Evaluating models...", "SUBSECTION")
        train_results, test_results = [], []
        for name, model in self.tuned_models.items():
            print(f"Evaluating {name} model ...")
            train_results.append(ml_l.evaluate_model(model, name, X_train, y_train, False))
            test_results.append(ml_l.evaluate_model(model, name, X_test, y_test, self.plot_graphs))

        # Ensemble voting
        or_estimators = [(name, model) for name, model in self.tuned_models.items()] # if hasattr(model, "predict_proba")]
        if len(or_estimators) > 0:
            cm_l.print_formatted_txt("Adding VotingClassifier Model...", "SUBSECTION")
            estimators = []
            for name, model in or_estimators:
                if hasattr(model, "random_state"):
                    model.set_params(random_state=gb_l.SEED_NUMBER)
                estimators.append((name, model))

            ensemble_model = VotingClassifier(estimators=estimators, voting="soft")
            ensemble_model.fit(X_train, y_train)
            # Append ensemble model to tuned_models dictionary
            self.tuned_models['Voting'] = ensemble_model
            # Evaluate the ensemble model
            train_results.append(ml_l.evaluate_model(ensemble_model, "Voting", X_train, y_train, False))
            test_results.append(ml_l.evaluate_model(ensemble_model, "Voting", X_test, y_test, self.plot_graphs))

        if self.plot_graphs:
            cm_l.print_formatted_txt("Plotting Feature Importance...", "SUBSECTION")
            for name, model in self.tuned_models.items():
                if name in ['LightGBM', 'CatBoost', 'XGBoost', 'GradientBoost']:
                    ml_l.plot_shap_values(model, name, X_train)
                else:
                    ml_l.plot_feature_importance(model, name, X_train)

        # Create and print the model results dataframe
        for group, results in {'train': train_results, 'test': test_results}.items():
            cm_l.print_formatted_txt(f"Model Results on {group} set", "RESULTS")
            df = pd.DataFrame.from_dict(results)
            df.sort_values(by=['f1_score'], ascending=False, inplace=True)
            for col in ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']:
                df[col] = df[col].astype(int)
            if self.verbose:
                display(df)
            if group == "test":
                self.project.model_results = df

        cm_l.print_formatted_txt(f"Ranking of test data by {self.col_year}...", "SUBSECTION")
        df_age = test_features.sort_values(by=[self.col_year, self.col_diameter]).reset_index(drop=True)
        df_age[self.col_rank] = df_age.index + 1
        if self.plot_graphs:
            ml_l.plot_ranked_records(df_age, col_rank=self.col_rank, max_bins=self.rank_bin_count, col_target=self.col_target,
                                     class_labels=self.class_labels, col_likelihood=self.col_likelihood)

        cm_l.print_formatted_txt("Ranking of test data by ML...", "SUBSECTION")
        df_ml, eval_results = ml_l.create_ranking(test_features, self.tuned_models, self.col_id, self.col_year, self.col_diameter, self.col_target,
                                                  all_cols_to_exclude, self.top_stat_dict, self.col_rank, self.class_labels,
                                                  self.nDCG_top_rank_perc, self.score_sort_list, self.verbose)
        df_ml = pd.merge(test_features, df_ml[[self.col_id, self.col_rank]], on=self.col_id)
        df_ml = df_ml.sort_values(by=self.col_rank)
        if self.plot_graphs:
            ml_l.plot_ranked_records(df_ml, col_rank=self.col_rank, max_bins=self.rank_bin_count, col_target=self.col_target,
                                     class_labels=self.class_labels, col_likelihood=self.col_likelihood)

        cm_l.print_formatted_txt("Compare by_age vs by_ml...", "SUBSECTION")
        if self.verbose:
            ml_l.compare_age_ml_methods(df_age, df_ml, self.col_target, self.col_rank, self.top_stat_dict, self.nDCG_top_rank_perc, self.plot_graphs)

        cm_l.print_formatted_txt("Saving evaluation results...", "SUBSECTION")
        eval_results[self.eval_col_features] = cm_l.list_to_str(self.features)
        eval_results[self.eval_col_models] = cm_l.list_to_str(self.models)
        eval_results['seed'] = self.project.data_random_seed
        for key, value in eval_results.items():
            eval_results[key] = [value]
        eval_results = pd.DataFrame.from_dict(eval_results)
        eval_results = eval_results.T.reset_index()
        eval_results.columns = ['parameter', 'value']
        self.project.eval_results = eval_results
        if self.verbose:
            display(self.project.eval_results)
        cm_l.write_data(self.project.eval_results, os.path.join(self.results_dir, f"{self.customer}_eval_results.csv"))

        if self.record_eval_results:
            # save the same file under customer's folder for comparing results during tunining
            str_now = datetime.now().strftime("%Y_%m_%d %H_%M_%S").replace(" ", "_")
            cm_l.write_data(self.project.eval_results, os.path.join(self.multi_run_folder, f"{self.customer}_eval_results_{str_now}.csv"))
