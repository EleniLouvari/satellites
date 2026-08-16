from import_libraries import *
import global_variables as gb_l
import libraries.common_libraries.generic_library as cm_l
from libraries.lead.ml.mlc_data_ckecks import CkeckData
from libraries.lead.ml.mlc_prepare_features import PrepareFeatures
from libraries.lead.ml.mlc_run_models import RunModels
from libraries.lead.ml.mlc_evaluate_models import EvalModels
from libraries.lead.ml.mlc_predictions import CalculatePredictions

class GeoMLClassifier:
    def __init__(self, gdf, customer, col_target, col_id, col_year, col_diameter, prediction_year, max_diameter, ban_year):
        random.seed(gb_l.SEED_NUMBER)
        np.random.seed(gb_l.SEED_NUMBER)
        tf.random.set_seed(gb_l.SEED_NUMBER)
        tf.compat.v1.get_default_graph()

        self.__work_dir = None
        self.__clean_dir = None
        self.__prepare_dir = None
        self.__models_dir = None
        self.__results_dir = None
        self.__multi_run_folder = None

        self.verbose = True
        self.gdf = gdf.copy()
        self.customer = customer
        self.prediction_year = prediction_year
        self.max_diameter = max_diameter
        self.base_dir = gb_l.global_work_dir
        self.project_dir = os.path.join(self.base_dir, self.customer)

        self.data_random_seed = 200
        self.col_target = col_target
        self.col_id = col_id
        self.col_year = col_year
        self.col_diameter = col_diameter
        self.ban_year = ban_year
        self.default_nulls = {'numeric': np.nan, 'object': None}
        self.search_distance_in_ft = 300
        self.plot_graphs = True

        # For feature preparation
        self.num_processes = 2
        self.apply_smote = False
        self.col_group = "group"
        self.split_ratio = 0.8
        self.grid_sample = 10
        self.cv_folds = 5
        self.model_results = {}
        self.eval_results = {}
        self.feature_scaler = "standard"  # valid values: 'minmax', 'standard', 'robust'

        # New extra features
        self.base_columns = [self.col_id, self.col_target, self.col_group, 'geometry']
        self.target_feature_names = {'kernel': f"{self.col_target}_kernel",
                                     'voronoi': f"{self.col_target}_voronoi",
                                     'nearby': f"{self.col_target}_nearby",
                                     'ratio': f"{self.col_target}_ratio",
                                     'avrg_distance': f"{self.col_target}_avrg_distance",
                                     'year_score': f"{self.col_target}_year_score"}
        self.extra_feature_names = {'house_yr_median': 'house_yr_median',
                                    'house_score': 'house_score',
                                    'income': 'income',
                                    'dis_com': 'dis_com'}

        # Features to be used for ML
        self.features = list(set(set(self.gdf.columns) | set(self.extra_feature_names.values()) | set(self.target_feature_names.values())).difference(set(self.base_columns)))
        self.field_dtypes = {'int16': [self.col_target, self.extra_feature_names['dis_com']],
                             'float64': list(self.target_feature_names.values()) + [self.col_year, self.col_diameter, self.extra_feature_names['house_yr_median'],
                                                                                    self.extra_feature_names['house_score'], self.extra_feature_names['income']],
                             'str': [self.col_id]}
        self.needed_cols = [self.col_id, self.col_target, self.col_year, self.col_diameter, 'geometry']

        self.models = ['KNN',
                       'NeuralNetwork',
                       'LogisticRegression',
                       'DecisionTree',
                       'RandomForest',
                       'SVM',
                       'LightGBM',
                       'Bayesian',
                       'Bagging',
                       #'CatBoost',
                       'XGBoost']

        self.models_grid = {
            'KNN': {
                'n_neighbors': [5, 10],
                'metric': ['euclidean'],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [10, 20, 30],  # Adjust according to the data size
                'p': [1, 2],  # for Minkowski metric
                },
            'LogisticRegression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear'],
                'penalty': ['l1', 'l2'],
                },
            'DecisionTree': {
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                },
            'RandomForest': {
                'n_estimators': [10, 50, 100],
                'max_depth': [10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                #'bootstrap': [True, False],
                'class_weight': [None],
                'criterion': ['gini', 'entropy'],
                'n_jobs': [self.num_processes]  # -1 uses all available cores
            },
            'SVM': {
                'C': [0.1, 1, 10, 20],
                'gamma': [0.1, 1, 5, 10],
                'kernel': ['rbf'],
                'probability': [True]
                },
            'LightGBM': {
                'objective': ['binary'],
                'n_estimators': [10, 50, 100],
                'learning_rate': [0.05, 0.01, 0.1],
                'max_depth': [3, 6, 10],
                'num_leaves': [5, 10, 20],
                #'max_bin': [256, 512, 1024],
                'metric': ['auc'],
                'n_jobs': [self.num_processes],
                'objective': ['binary'],
                'colsample_bytree': [0.65],
                'reg_alpha': [0.8, 1],
                'reg_lambda': [0.8, 1],
                'force_row_wise': ['true'],
                'verbose': [-1]
                },
            'Bayesian': {
                'var_smoothing': [1e-9, 1e-8, 1e-7]
                },
            'Bagging': {
                # 'base_estimator': [DecisionTreeClassifier(), RandomForestClassifier()],  # You can add more base estimators here
                'oob_score': [True, False],
                'n_estimators': [10, 50, 100],  # Number of base estimators in the ensemble
                'max_samples': [0.25, 0.5, 0.8, 1.0],  # Proportion of samples to draw from the training set for each base estimator
                'max_features': [0.5, 0.8, 1.0],  # Proportion of features to draw from the feature set for each base estimator
                # 'bootstrap': [True, False],  # Whether samples are drawn with replacement
                # 'bootstrap_features': [True, False],  # Whether features are drawn with replacement
            },
            'CatBoost': {
                #'iterations': [500, 1000],
                'learning_rate': [0.05, 0.1, 0.2, 1],
                'depth': [4, 6, 8],
                'l2_leaf_reg': [1, 3, 5],
                'bagging_temperature': [0.5, 0.8, 1.0],
                'verbose': [False]
                },
            'GradientBoost': {
                'learning_rate': [0.05, 0.1, 0.2, 1],
                'n_estimators': [10, 50, 100],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                #'min_samples_leaf': [1, 2, 4],
                #'subsample': [0.8, 1.0],
                #'max_features': ['sqrt', 'log2', None],
                'loss': ['log_loss']
            },
            'NeuralNetwork': {
                'hidden_layer_sizes': [],  # this is updated in the code
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['adam','sgd'],
                'max_iter': [500, 1000],
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]
                },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5]
            }
        }

        # For evaluation
        self.top_stat_dict = {'Top 1%': [1, 0.5], 'Top 2%': [2, 0.4], 'Top 3%': [3, 0.3], 'Top 5%': [5, 0.2], 'Top 10%': [10, 0.1]}
        self.class_labels = {1: self.col_target, 0: f"Non-{self.col_target}",  -1: "Unknown"}
        self.col_rank = "rank"
        self.col_prob = "prob"
        self.col_likelihood = "Likelihood"
        self.nDCG_top_rank_perc = 0.01  # percentage of top ranked records for selecting the best nDCG score
        self.score_sort_list = ['nDCG', 'power', 'top weighted score', 'median']
        self.rank_bin_count = 30

        # For tuning process
        self.record_eval_results = False
        self.eval_col_features = "features_used"
        self.eval_col_models = "models_used"

        self.time_dict = {}  # keep track of the processes to calculate their duration

    @property
    def work_dir(self):
        # if a user has set a name manually, then return that
        if self.__work_dir is not None:
            return str(self.__work_dir)

        # if Jupyter notebook name is available, return that
        notebook_path = cm_l.get_notebook_name()
        if notebook_path is not None:
            _, notebook_name, _ = cm_l.separate_path_filename_extension(notebook_path)
        else:
            notebook_name = None

        if notebook_name is None:
            # if we are working with Python file, return that
            if len(sys.argv) > 0 and sys.argv[0]:
                path_list = sys.argv[0].split(".")  # remove extension
                path_list = path_list[0].split(os.path.sep)  # handle os-specific path separator
                notebook_name = path_list[-1]
            else:
                notebook_name = None

        if notebook_name is not None:
            return os.path.join(self.project_dir, notebook_name)
        else:
            return self.project_dir

    @work_dir.setter
    def work_dir(self, work_dir):
        self.__work_dir = work_dir

    @property
    def clean_dir(self):
        """Returns the path to the working clean directory"""
        if self.__clean_dir is None:
            clean_dir = os.path.join(self.work_dir, "1_clean")
            return clean_dir + "/"
        else:
            return str(self.__clean_dir)

    @clean_dir.setter
    def clean_dir(self, clean_dir):
        self.__clean_dir = clean_dir

    @property
    def prepare_dir(self):
        """Returns the path to the working prepare directory"""
        if self.__prepare_dir is None:
            prepare_dir = os.path.join(self.work_dir, "2_prepare")
            return prepare_dir + "/"
        else:
            return str(self.__prepare_dir)

    @prepare_dir.setter
    def prepare_dir(self, prepare_dir):
        self.__prepare_dir = prepare_dir

    @property
    def models_dir(self):
        """Returns the path to the working models directory"""
        if self.__models_dir is None:
            models_dir = os.path.join(self.work_dir, "3_models")
            return models_dir + "/"
        else:
            return str(self.__models_dir)

    @models_dir.setter
    def models_dir(self, models_dir):
        self.__models_dir = models_dir

    @property
    def results_dir(self):
        """Returns the path to the working models directory"""
        if self.__results_dir is None:
            results_dir = os.path.join(self.work_dir, "4_results")
            return results_dir + "/"
        else:
            return str(self.__results_dir)

    @results_dir.setter
    def results_dir(self, results_dir):
        self.__results_dir = results_dir

    @property
    def multi_run_folder(self):
        """Returns the path to the tuning directory"""
        if self.__multi_run_folder is None:
            multi_run_folder = os.path.join(self.project_dir, "tuning")
            cm_l.create_folder(multi_run_folder)
            return multi_run_folder + "/"
        else:
            return str(self.__multi_run_folder)

    @multi_run_folder.setter
    def multi_run_folder(self, multi_run_folder):
        if not self.record_eval_results:
            cm_l.print_formatted_txt("The <record_eval_results> variable is set to True!!", "WARNING")
            self.record_eval_results = True
        cm_l.create_folder(multi_run_folder)
        self.__multi_run_folder = multi_run_folder

    @cm_l.time_decorator
    def run_data_ckecks(self):
        cleanup = CkeckData(self)
        cleanup.ckeck_process()

    @cm_l.time_decorator
    def run_prepare_features(self):
        prepare = PrepareFeatures(self)
        prepare.prepare_process()

    @cm_l.time_decorator
    def run_ml(self):
        ml = RunModels(self)
        ml.ml_process()

    @cm_l.time_decorator
    def run_eval(self):
        eval = EvalModels(self)
        eval.eval_process()

    @cm_l.time_decorator
    def run_predict(self):
        predict = CalculatePredictions(self)
        predict.predict_process()

    @cm_l.time_decorator
    def run_ml_process(self):
        self.run_data_ckecks()
        self.run_prepare_features()
        self.run_ml()
        self.run_eval()
        self.run_predict()
