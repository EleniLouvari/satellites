from import_libraries import *
import libraries.common_libraries.generic_library as cm_l
import libraries.ml_common_libraries.ml_library as ml_l


class ML_Sensitivity:
    def __init__(self, ml_class):
        """
        Initialize the ML_Tune class with the provided machine learning class instance.
        """
        self.ml_class = deepcopy(ml_class)
        self.best_score_method = ml_class.score_sort_list[0]
        self.results = None
        self.best_results = None
        self.seeds = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

        self.df_tune_results = ml_l.read_results_files(
            self.ml_class.multi_run_folder,
            self.ml_class.score_sort_list,
            self.ml_class.eval_col_features,
            self.ml_class.eval_col_models
        )
        self.ml_class.record_eval_results = True

    def calculate_num_features(self, df):
        df['num_features'] = df[self.ml_class.eval_col_features].apply(
            lambda x: len(x) if isinstance(x, list) else len([item.strip() for item in x.split(",")]) if isinstance(x, str) else 0
        )
        return df

    def run_sensitivity_analysis(self):
        cm_l.print_formatted_txt("Starting sensitivity analysis...", "SUBSECTION")

        self.ml_class.run_data_ckecks()

        features = self.ml_class.features
        for seed in self.seeds:
            check_run = ml_l.check_finished_tune_runs(self.df_tune_results, self.ml_class.eval_col_features, features, self.best_score_method, seed)
            if check_run == 0:
                print(f"\n--------------------Testing seed: {seed}-----------------------\n")
                self.ml_class.data_random_seed = seed
                self.ml_class.run_prepare_features()
                self.ml_class.run_ml()
                self.ml_class.run_eval()
                eval_results = self.ml_class.eval_results.set_index("parameter")
                check_run = eval_results.loc[self.best_score_method].value

        df_results = ml_l.read_results_files(self.ml_class.multi_run_folder, self.ml_class.score_sort_list, self.ml_class.eval_col_features, self.ml_class.eval_col_models)

        # Ensure indices and columns are unique
        df_results = df_results.drop_duplicates(subset=['seed', self.ml_class.eval_col_features])
        double_seeds = df_results[df_results.duplicated(subset="seed", keep=False)]
        if not double_seeds.empty:
            raise Exception(f"Duplicate seed runs found: \n{display(double_seeds.sort_values(by="seed"))}")

        df_results = df_results.set_index("seed")
        _, numerical_cols = cm_l.get_column_types(df_results)
        df_results = df_results[numerical_cols].copy()

        # Calculate the statistics
        ml_l.display_sensitivity_results(df_results, self.best_score_method)
