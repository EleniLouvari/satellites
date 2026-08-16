from import_libraries import *
import global_variables as gb_l
import libraries.common_libraries.generic_library as cm_l
import libraries.ml_common_libraries.ml_library as ml_l

class CkeckData:
    def __init__(self, project):
        self.project = project

    def ckeck_process(self):
        """All checks and cleanup processes are here."""
        cm_l.print_formatted_txt("Checking data process...", "SECTION")

        self.gdf = self.project.gdf.copy()
        self.customer = self.project.customer
        self.project_dir = self.project.project_dir
        self.work_dir = self.project.work_dir
        self.clean_dir = self.project.clean_dir
        self.prepare_dir = self.project.prepare_dir
        self.models_dir = self.project.models_dir
        self.results_dir = self.project.results_dir
        self.multi_run_folder = self.project.multi_run_folder
        self.col_target = self.project.col_target
        self.col_id = self.project.col_id
        self.col_year = self.project.col_year
        self.plot_graphs = self.project.plot_graphs
        self.verbose = self.project.verbose
        self.default_nulls = self.project.default_nulls
        self.needed_cols = self.project.needed_cols
        self.base_columns = self.project.base_columns
        self.col_year = self.project.col_year
        self.col_diameter = self.project.col_diameter
        self.field_dtypes = self.project.field_dtypes
        self.features = self.project.features
        self.target_feature_names = self.project.target_feature_names
        self.extra_feature_names = self.project.extra_feature_names
        self.record_eval_results = self.project.record_eval_results

        cm_l.print_formatted_txt("Checking input geodataframe...", "SUBSECTION")
        if not isinstance(self.gdf, gpd.GeoDataFrame):
            raise ValueError("Input data must be a GeoDataFrame")

        cm_l.print_formatted_txt("Creating folders...", "SUBSECTION")
        cm_l.create_folder(self.project_dir)
        cm_l.delete_folder(self.work_dir)
        for fld in [self.project_dir, self.work_dir, self.clean_dir, self.prepare_dir, self.models_dir, self.results_dir]:
            cm_l.create_folder(fld)
        if self.record_eval_results:
            cm_l.create_folder(self.multi_run_folder)

        cm_l.print_formatted_txt("Input data...", "SUBSECTION")
        if self.verbose:
            display(skim(self.gdf))

        cm_l.print_formatted_txt("Checking needed columns...", "SUBSECTION")
        cm_l.check_needed_df_columns(self.gdf, self.needed_cols)
        print(f"The geodataframe has all needed columns: {self.needed_cols}.")

        cm_l.print_formatted_txt("Checking wrong input features...", "SUBSECTION")
        missing_features = [feature for feature in self.features if feature not in self.gdf.columns]
        missing_features = [feature for feature in missing_features if feature not in list(self.target_feature_names.values())+list(self.extra_feature_names.values())]
        if missing_features:
            raise Exception(f"The {missing_features} are included in the features list but they are missing from the input geodataframe!")
        else:
            print("All input features are included in the input geodataframe.")
            print("")

        cm_l.print_formatted_txt("Standardize input data...", "SUBSECTION")
        self.gdf = cm_l.standardize_dataframe(self.gdf, gb_l.string_null_values_list, gb_l.numeric_null_values_list,
                                              string_null_value=self.default_nulls['object'], numeric_null_value=self.default_nulls['numeric'])
        if self.verbose:
            display(skim(self.gdf))

        if any(self.gdf.isnull().sum()):
            raise Exception("Null values in the input geodataframe!")
        else:
            print("No null values found.")

        cm_l.print_formatted_txt(f"Checking duplicate {self.col_id}...", "SUBSECTION")
        df_douplicates = self.gdf[self.gdf.duplicated(subset=[self.col_id])].copy()
        if not df_douplicates.empty:
            raise Exception(f"There are {len(df_douplicates)}! of {self.col_id}! These must be unique!")
        else:
            print(f"The geodataframe has unique values of: {self.col_id}.")

        cm_l.print_formatted_txt("Checking projection...", "SUBSECTION")
        if not self.gdf.crs.is_projected:
            raise Exception("The input geodataframe is not in a projected system!")
        else:
            print("The geodataframe is projected:")
            display(self.gdf.crs)

        cm_l.print_formatted_txt("Grouping categorical columns...", "SUBSECTION")
        self.gdf = cm_l.group_all_categorical_in_df(self.gdf, self.col_id)

        cm_l.print_formatted_txt("Format the field types...", "SUBSECTION")
        self.gdf = cm_l.format_df_dtypes(self.gdf, self.col_id, self.field_dtypes, verbose=self.verbose)

        cm_l.print_formatted_txt("Fix JSON characters in column names")
        self.gdf = cm_l.fix_json_chars_in_column_names(self.gdf)

        cm_l.print_formatted_txt("Fix JSON characters in column values")
        self.gdf = cm_l.fix_json_chars_in_column_values(self.gdf, self.base_columns)

        cm_l.print_formatted_txt(f"Checking target column {self.col_target}...", "SUBSECTION")
        self.gdf[self.col_target] = self.gdf[self.col_target].fillna(-1)
        if not all(self.gdf[self.col_target].isin([-1, 0, 1])):
            raise Exception(f"The {self.col_target} has invalid values! Valid values are: -1 (for unknowns), 0, 1")
        else:
            print(f"Total records with {self.col_target} to predict: {len(self.gdf[self.gdf[self.col_target]==-1])}")
            print(f"Total records with {self.col_target} equals 0: {len(self.gdf[self.gdf[self.col_target]==0])}")
            print(f"Total records with {self.col_target} equals 1: {len(self.gdf[self.gdf[self.col_target]==1])}")

        cm_l.print_formatted_txt("Writting cleaned data...", "SUBSECTION")
        if self.verbose:
            display(self.gdf.head())
        cm_l.write_data(self.gdf, os.path.join(self.clean_dir, "clean_data"))
