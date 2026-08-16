from import_libraries import *
import libraries.common_libraries.generic_library as cm_l
import libraries.ml_common_libraries.ml_library as ml_l
import libraries.common_libraries.spatial_stat_library as s_stat_l


class ML_Tune:
    def __init__(self, ml_class):
        """
        Initialize the ML_Tune class with the provided machine learning class instance.
        """
        self._test_features = None
        self._multi_run_folder = None
        self.ml_class = deepcopy(ml_class)
        self.best_score_method = ml_class.score_sort_list[0]
        self.base_features: List[str] = []
        self.custom_features: Dict[str, List[str]] = {}
        self.results = None
        self.best_results = None
        self.ml_class.record_eval_results = True

    @property
    def test_features(self) -> List[str]:
        """
        Get the test features, excluding the base features.
        """
        if self._test_features is None:
            return sorted([x for x in set(self.ml_class.features) if x not in set(self.base_features)])
        return self._test_features

    @test_features.setter
    def test_features(self, test_features: List[str]):
        """
        Set the test features, ensuring base features are excluded.
        """
        self._test_features = sorted([x for x in set(test_features) if x not in set(self.base_features)])

    @property
    def multi_run_folder(self) -> List[str]:
        if self._multi_run_folder is None:
            return self.ml_class.multi_run_folder
        return self._multi_run_folder

    @multi_run_folder.setter
    def multi_run_folder(self, multi_run_folder: List[str]):
        self.ml_class.multi_run_folder = multi_run_folder
        self._multi_run_folder = multi_run_folder

    def calculate_num_features(self, df):
        df['num_features'] = df[self.ml_class.eval_col_features].apply(
            lambda x: len(x) if isinstance(x, list) else len([item.strip() for item in x.split(",")]) if isinstance(x, str) else 0
        )
        return df

    def run_multitests(self, tests_to_run: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Run multiple tests with specified features and return results in a DataFrame.
        """
        results = []
        for test_name, test_features in tests_to_run.items():
            print(f"\n----------------Testing: {test_name}------------------------------")
            print(f"Features: {test_features}")
            self.ml_class.features = deepcopy(test_features)
            self.ml_class.run_ml()
            self.ml_class.run_eval()

            eval_results = self.ml_class.eval_results.set_index("parameter")
            results.append({
                'name': test_name,
                self.ml_class.eval_col_features: deepcopy(test_features),
                self.best_score_method: eval_results.loc[self.best_score_method].value
            })
        return pd.DataFrame(results)

    def process_features(self, tests, finished_runs, description, features):
        check_run = ml_l.check_finished_tune_runs(self.df_tune_results, self.ml_class.eval_col_features, deepcopy(features), self.best_score_method)
        if check_run == 0:
            tests[f"{description}"] = deepcopy(features)
        else:
            finished_runs.append({'name': f"{description}", self.ml_class.eval_col_features: deepcopy(features), self.best_score_method: check_run})

    def run_single_features(self):
        """
        Test each feature individually and display the best features.
        """
        cm_l.print_formatted_txt("Starting checking one by one all test features...", "SUBSECTION")
        self.df_tune_results = ml_l.read_results_files(self.ml_class.multi_run_folder, self.ml_class.score_sort_list,
                                                       self.ml_class.eval_col_features, self.ml_class.eval_col_models)
        tests = {}
        finished_runs = []
        self.process_features(tests, finished_runs, "test base", list(self.base_features))
        self.process_features(tests, finished_runs, "test all", list(self.base_features + self.test_features))
        for feature in self.test_features:
            self.process_features(tests, finished_runs, f"test {feature}", self.base_features + [feature])

        df_results = self.run_multitests(tests)
        if finished_runs:
            df_results = pd.concat([df_results, pd.DataFrame(finished_runs)], ignore_index=True)

        self.highlight_best_features(df_results, highlight_min_features=True, show_single_features=True)
        cm_l.print_formatted_txt("Checking testing one by one features finished!", "SUBSECTION")

    def run_custom_features(self):
        """
        Test custom feature combinations and display the best features.
        """
        cm_l.print_formatted_txt("Starting testing custom feature combinations...", "SUBSECTION")
        self.df_tune_results = ml_l.read_results_files(self.ml_class.multi_run_folder, self.ml_class.score_sort_list,
                                                       self.ml_class.eval_col_features, self.ml_class.eval_col_models)

        if not self.custom_features or not isinstance(self.custom_features, dict):
            raise ValueError("Custom features variable is invalid; it should be a dictionary with values lists of features!")

        tests = {}
        finished_runs = []
        for test_name, test_features in self.custom_features.items():
            self.process_features(tests, finished_runs, f"custom {test_name}", test_features)

        df_results = self.run_multitests(tests)
        if finished_runs:
            df_results = pd.concat([df_results, pd.DataFrame(finished_runs)], ignore_index=True)

        self.highlight_best_features(df_results, highlight_min_features=True)
        cm_l.print_formatted_txt("Checking testing custom feature combinations finished!", "SUBSECTION")

    def run_adding_features(self):
        """
        Test adding features incrementally and display the results.
        """
        cm_l.print_formatted_txt("Starting adding features...", "SUBSECTION")
        self.df_tune_results = ml_l.read_results_files(self.ml_class.multi_run_folder, self.ml_class.score_sort_list,
                                                       self.ml_class.eval_col_features, self.ml_class.eval_col_models)

        test_features = deepcopy(self.base_features)
        finished_runs = []

        def evaluate_features(feature_set, description):
            self.ml_class.features = deepcopy(feature_set)
            self.ml_class.run_ml()
            self.ml_class.run_eval()
            eval_results = self.ml_class.eval_results.set_index("parameter")
            score = eval_results.loc[self.best_score_method].value
            finished_runs.append({'name': description, self.ml_class.eval_col_features: deepcopy(feature_set), self.best_score_method: score})
            return score

        previous_result = ml_l.check_finished_tune_runs(self.df_tune_results, self.ml_class.eval_col_features, deepcopy(test_features), self.best_score_method)
        if previous_result > 0:
            finished_runs.append({'name': "base features", self.ml_class.eval_col_features: deepcopy(test_features), self.best_score_method: previous_result})
        else:
            print(f"\n{'*' * 50}\n-----------test base features {test_features}--------------------------------------------------")
            previous_result = evaluate_features(test_features, "base features")

        for feature in self.test_features:
            test_features.append(feature)
            check_run = ml_l.check_finished_tune_runs(self.df_tune_results, self.ml_class.eval_col_features, deepcopy(test_features), self.best_score_method)
            if check_run > 0:
                finished_runs.append({'name': f"add {feature}", self.ml_class.eval_col_features: deepcopy(test_features), self.best_score_method: check_run})
                if check_run <= previous_result:
                    test_features.remove(feature)
                else:
                    previous_result = check_run
            else:
                print(f"\n{'*' * 50}\n-----------test adding {feature}--------------------------------------------------")
                print(f"Features used: {test_features}")
                current_result = evaluate_features(test_features, f"add {feature}")
                if current_result <= previous_result:
                    test_features.remove(feature)
                else:
                    previous_result = current_result

        df_results = pd.DataFrame(finished_runs)
        self.highlight_best_features(df_results, highlight_min_features=True)

    def run_removing_features(self):
        """
        Test removing features incrementally and display the results.
        """
        cm_l.print_formatted_txt("Starting removing features...", "SUBSECTION")
        self.df_tune_results = ml_l.read_results_files(self.ml_class.multi_run_folder, self.ml_class.score_sort_list,
                                                       self.ml_class.eval_col_features, self.ml_class.eval_col_models)
        # Sort the features based on the p-values
        test_features = self.calc_and_sort_p_values()
        all_features = self.base_features + test_features

        tests = {}
        finished_runs = []
        self.process_features(tests, finished_runs, "remove none", all_features)
        for feature in test_features:
            self.process_features(tests, finished_runs, f"remove {feature}", [f for f in all_features if f != feature])

        df_results = self.run_multitests(tests)
        if finished_runs:
            df_results = pd.concat([df_results, pd.DataFrame(finished_runs)], ignore_index=True)

        self.highlight_best_features(df_results, highlight_min_features=False)
        cm_l.print_formatted_txt("Removing features finished!", "SUBSECTION")

    def calc_and_sort_p_values(self):
        cm_l.print_formatted_txt("Calculating z-score and p-values for all features...", "SUBSECTION")

        X_train = cm_l.read_data(self.ml_class.prepare_dir, "train_test_features")
        X_train = X_train[(X_train[self.ml_class.col_group]=="train")].copy()

        object_cols = []
        for col in self.test_features:
            for x in X_train.columns:
                if x.startswith(col) and x!=col:
                    object_cols.append(col)
        object_cols = list(set(object_cols))
        results = []
        numerical_cols = [x for x in self.test_features if x not in object_cols]
        for feature in numerical_cols:
            col_mean = f"mean_{feature}"
            col_stdev = f"std_{feature}"
            col_zscore = f"zscore_{feature}"
            col_pvalue = f"pvalue_{feature}"
            gdf = s_stat_l.calc_zscore_and_pvalue(X_train, feature, col_mean, col_stdev, col_zscore, col_pvalue)
            results.append({'feature': feature, 'max_zscore': gdf[col_zscore].max(), 'p_value': gdf[col_pvalue].max()})

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="p_value", ascending=False)
        display(df_results)
        return(object_cols + list(df_results['feature']))

    def plot_score_results(self, df_results, base_score):
        """
        Plot score results with a horizontal line at the score using base features.
        """
        plt.figure(figsize=(18, 8))
        sns.pointplot(x=df_results.index, y=df_results[self.best_score_method])
        plt.axhline(y=base_score, color="red", linestyle="--", label=f"Base Features: {self.base_features}")

        plt.title("Feature Scores")
        plt.xlabel("Feature")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

    def highlight_rows(self, row, base_score):
        """
        Highlight rows based on the score values.
        """
        if row[self.best_score_method] > base_score:
            return [f'background-color: yellow' for _ in row]
        elif row[self.best_score_method] == base_score:
            return [f'background-color: honeydew' for _ in row]
        else:
            return ""

    def highlight_best_features(self, df_results, highlight_min_features=False, show_single_features=False):
        """
        Highlight and display the best features and results.
        """
        df_results = self.calculate_num_features(df_results)
        df_results = df_results.sort_values(by=self.best_score_method, ascending=False).reset_index(drop=True)

        if highlight_min_features:
            idx_base = df_results['num_features'].idxmin()
            if show_single_features:
                df_results = df_results[df_results['num_features'].isin([len(self.base_features), len(self.base_features)+1, len(self.base_features)+len(self.test_features)])]
        else:
            idx_base = df_results['num_features'].idxmax()

        base_score = df_results.loc[idx_base][self.best_score_method]
        display(df_results.style.apply(lambda row: self.highlight_rows(row, base_score), axis=1))
        self.plot_score_results(df_results, base_score)
