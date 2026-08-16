from import_libraries import *
import global_variables as gb_l
import libraries.common_libraries.generic_library as cm_l
import libraries.ml_common_libraries.ml_library as ml_l

class CalculatePredictions:
    def __init__(self, project):
        self.project = project

    def predict_process(self):
        """All ML processes are here"""
        cm_l.print_formatted_txt("Creating predictions process...", "SECTION")

        self.verbose = self.project.verbose
        self.customer = self.project.customer
        self.work_dir = self.project.work_dir
        self.project_dir = self.project.project_dir
        self.clean_dir = self.project.clean_dir
        self.prepare_dir = self.project.prepare_dir
        self.models_dir = self.project.models_dir
        self.results_dir = self.project.results_dir
        self.col_target = self.project.col_target
        self.col_id = self.project.col_id
        self.col_year = self.project.col_year
        self.col_diameter = self.project.col_diameter
        self.plot_graphs = self.project.plot_graphs
        self.models = self.project.models
        self.rank_bin_count = self.project.rank_bin_count
        self.base_columns = self.project.base_columns
        self.features = self.project.features
        self.top_stat_dict = self.project.top_stat_dict
        self.class_labels = self.project.class_labels
        self.col_rank = self.project.col_rank
        self.col_prob = self.project.col_prob
        self.col_likelihood = self.project.col_likelihood
        self.nDCG_top_rank_perc = self.project.nDCG_top_rank_perc
        self.score_sort_list = self.project.score_sort_list
        self.tuned_models = {}

        cm_l.print_formatted_txt("Reading initial data...", "SUBSECTION")
        gdf = cm_l.read_data(self.clean_dir, "clean_data")
        to_predict = gdf[gdf[self.col_target]==-1].copy()

        cm_l.print_formatted_txt("Reading prepared data...", "SUBSECTION")
        all_features = cm_l.read_data(self.prepare_dir, "all_features")

        cm_l.print_formatted_txt("Reading models...", "SUBSECTION")
        for model in self.models:
            self.tuned_models[model] = cm_l.read_ml_model(os.path.join(self.models_dir, f"{model}_all"))

        cm_l.print_formatted_txt("Preparing train-test data...", "SUBSECTION")
        features_to_keep, features_to_remove = ml_l.select_features_for_ml(all_features, self.base_columns, self.features)
        print(f"Features to be used: {features_to_keep}")
        print(f"Features dropped: {features_to_remove}")
        all_cols_to_exclude = self.base_columns + features_to_remove

        cond = all_features[self.col_target]>=0
        X_train_all = all_features[cond][features_to_keep].copy()
        y_train_all = all_features[cond][self.col_target].copy()

        or_estimators = [(name, model) for name, model in self.tuned_models.items()] # if hasattr(model, "predict_proba")]
        if len(or_estimators) > 0:
            cm_l.print_formatted_txt("Adding VotingClassifier Model...", "SUBSECTION")
            estimators = []
            for name, model in or_estimators:
                if hasattr(model, "random_state"):
                    model.set_params(random_state=gb_l.SEED_NUMBER)
                estimators.append((name, model))

            ensemble_model = VotingClassifier(estimators=estimators, voting="soft")
            ensemble_model.fit(X_train_all, y_train_all)
            # Append ensemble model to tuned_models dictionary
            self.tuned_models['Voting'] = ensemble_model

        cm_l.print_formatted_txt("Ranking of all data...", "SUBSECTION")
        all_features, _ = ml_l.create_ranking(all_features, self.tuned_models, self.col_id, self.col_year, self.col_diameter, self.col_target,
                                              all_cols_to_exclude, self.top_stat_dict, self.col_rank, self.class_labels,
                                              self.nDCG_top_rank_perc, self.score_sort_list, self.verbose)
        gdf = pd.merge(gdf, all_features, on=self.col_id)
        gdf = gdf.sort_values(by=self.col_rank).reset_index(drop=True)

        cm_l.print_formatted_txt("Create likelihood classification...", "SUBSECTION")
        gdf = ml_l.create_likelihood_classification(gdf, col_target=self.col_target, col_rank=self.col_rank, col_likelihood=self.col_likelihood)

        num_ranked_unknowns = len(gdf[gdf[self.col_target] == -1])
        num_to_predict = len(to_predict)
        assert num_ranked_unknowns == num_to_predict, f"Unknowns in the ranking list: {num_ranked_unknowns} do not match with the total number of Unknowns: {num_to_predict}!"
        assert len(gdf[gdf[self.col_target]==-1]) == len(to_predict), "Unknowns in the ranking list do not match with the total number of Unknowns!"

        if self.plot_graphs:
            cm_l.print_formatted_txt("Plot all ranked records...", "SUBSECTION")
            ml_l.plot_ranked_records(gdf, col_rank=self.col_rank, max_bins=self.rank_bin_count, col_target=self.col_target,
                                     class_labels=self.class_labels, col_likelihood=self.col_likelihood)

        if self.verbose:
            cm_l.print_formatted_txt("Statistics on all ranked records...", "SUBSECTION")
            ml_l.display_statistics_of_predictions(gdf, col_target=self.col_target, col_likelihood=self.col_likelihood, class_labels=self.class_labels)

        cm_l.print_formatted_txt("Calculate probabilities...", "SUBSECTION")
        curve_fit_params = ml_l.calculate_prob_distribution(gdf, self.col_target, self.col_rank, self.class_labels, self.plot_graphs)
        gdf = ml_l.calculate_probabilities(curve_fit_params, gdf, self.col_prob, self.col_rank)

        cm_l.print_formatted_txt("Last ranked records...", "SUBSECTION")
        if self.verbose:
            display(gdf[gdf[self.col_target]==1].tail(10))

        cm_l.print_formatted_txt("Saving predictions of all system...", "SUBSECTION")
        cm_l.write_data(gdf, os.path.join(self.results_dir, f"{self.customer}_predictions_system"))
        if self.verbose:
            display(gdf.head(5))
            display(skim(gdf))

        cm_l.print_formatted_txt("Saving predictions of unknowns...", "SUBSECTION")
        gdf_unknowns = gdf[gdf[self.col_target]==-1].copy()
        gdf_unknowns = gdf_unknowns.sort_values(by=self.col_rank).reset_index(drop=True)
        gdf_unknowns[self.col_rank] = gdf_unknowns.index + 1
        cm_l.write_data(gdf_unknowns, os.path.join(self.results_dir, f"{self.customer}_predictions_unknowns"))
        if self.verbose:
            display(gdf_unknowns.head(5))
            display(skim(gdf_unknowns))

        self.project.gdf = gdf.copy()
