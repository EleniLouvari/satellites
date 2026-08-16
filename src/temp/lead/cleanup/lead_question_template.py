from import_libraries import *
import global_variables as gb_l
import libraries.common_libraries.generic_library as cm_l
import libraries.common_libraries.geom_library as geom_l
import libraries.common_libraries.question_template_library as cm_tm_l
import libraries.lead.cleanup.lead_question_template_library as ld_tm_l
import libraries.lead.cleanup.lead_cleanup_library as ld_cl_l

class LeadTemplate:
    """Lead analysis: template for questions."""

    def __init__(self, project):
        self.project = project
        self.verbose = self.project.verbose
        self.pub_pipes = self.project.pub_pipes
        self.pr_pipes = self.project.pr_pipes
        self.abd_pub_pipes = self.project.abd_pub_pipes
        self.abd_pr_pipes = self.project.abd_pr_pipes
        self.joined_ids = self.project.joined_ids
        self.field_mapping = self.project.field_mapping
        self.pipe_id_field = self.field_mapping['pipe_id_field']
        self.join_public_field = self.field_mapping['join_public_field']
        self.join_private_field = self.field_mapping['join_private_field']
        self.install_yr_field = self.field_mapping['install_yr_field']
        self.abandoned_field = self.field_mapping['abandoned_field']
        self.abandon_yr_field = self.field_mapping['abandon_yr_field']
        self.abandon_mat_field = self.field_mapping['abandon_mat_field']
        self.diameter_field = self.field_mapping['diameter_field']
        self.material_field = self.field_mapping['material_field']
        self.active_field = self.field_mapping['active_field']
        self.verified_field = self.field_mapping['verified_field']
        self.vrsource_field = self.field_mapping['vrsource_field']
        self.hsmaterial_field = self.field_mapping['hsmaterial_field']
        self.address_field = self.field_mapping['address_field']
        self.geometry_field = self.field_mapping['geometry_field']
        self.pub_prefix = self.field_mapping['pb_prefix']
        self.pr_prefix = self.field_mapping['pr_prefix']
        self.customer = self.project.customer
        self.work_dir = self.project.work_dir
        self.prediction_year = self.project.prediction_year
        self.material_dict = None
        self.diameter_dict = None
        self.null_values_list = self.project.null_values_list
        self.string_null_values_list = self.project.string_null_values_list
        self.min_inst_yr = self.project.min_inst_yr
        self.max_inst_yr = date.today().year
        self.min_diameter = 0
        self.max_diameter = self.project.max_diameter
        self.min_length = self.project.min_length
        self.max_length = self.project.max_length
        self.plot_graphs = self.project.plot_graphs
        self.diameter_unit = self.project.diameter_unit
        self.min_cluster_distance = 5000  # ft
        self.max_geocode_distance = 500  # ft
        self.max_nearby_geometry_distance = 5  # ft
        self.max_connected_distance = 0.5 # ft, max distance between the public and the connected private pipe
        self.output_folder = os.path.join("/cloud/shared_folder/", self.customer)
        self.pipes_columns = [self.pipe_id_field, self.material_field, self.diameter_field, self.verified_field,
                              self.active_field, self.geometry_field]
        self.street_name_field = "street_name"
        self.street_num_field = "street_num"
        self.city_field = "city"
        self.zip_field = "zip"
        self.state_field = "state"
        self.country_field = "country"
        self.address_columns = [self.city_field, self.zip_field, self.state_field, self.country_field]

        self.status_act_key = "Active pipes"
        self.status_abd_key = "Abandoned pipes"
        self.act_pub_key = "Active public pipes"
        self.abd_pub_key = "Abandoned public pipes"
        self.act_pr_key = "Active private pipes"
        self.abd_pr_key = "Abandoned private pipes"
        # create a dictionary with the dataframes {status: {key: df}} where status is active or abandoned
        self.dict_data = {self.status_act_key: {self.act_pub_key: self.pub_pipes, self.act_pr_key: self.pr_pipes},
                          self.status_abd_key:{self.abd_pub_key: self.abd_pub_pipes, self.abd_pr_key: self.abd_pr_pipes}}
        self.dict_issues = {self.status_act_key: {self.act_pub_key: {}, self.act_pr_key: {}},
                            self.status_abd_key:{self.abd_pub_key: {}, self.abd_pr_key: {}}}

        self.all_pipes = self.create_empty_geodataframe(self.pipes_columns)
        self.act_pub_pipes_xls_flnm = f"{self.customer}_act_pub_pipes_issues.xlsx"
        self.abd_pub_pipes_xls_flnm = f"{self.customer}_abd_pub_pipes_issues.xlsx"
        self.act_pr_pipes_xls_flnm = f"{self.customer}_act_pr_pipes_issues.xlsx"
        self.abd_pr_pipes_xls_flnm = f"{self.customer}_abd_pr_pipes_issues.xlsx"
        self.pipes_crs = None
        self.all_pipes = None

    @staticmethod
    def create_empty_geodataframe(list_columns):
        return gpd.GeoDataFrame(columns=list_columns, geometry="geometry", crs=4326)

    def initial_field_checks(self):
        """Checks for needed columns and adds an ID column."""

        # check if both public and private side dataframes are empty
        dict_act_pipes = self.dict_data[self.status_act_key].copy()
        all_empty = all(df.empty for df in dict_act_pipes.values())
        if all_empty:
            raise Exception("Both active public side and private side dataframes are empty.")

        cm_l.print_formatted_txt("Checks for needed columns and empty geodataframes")
        dict_data = self.dict_data.copy()
        for status, status_dict in dict_data.items():
            for key, df in status_dict.items():
                if df is None or len(df) == 0:
                    print(f"Checking {key}: (Number of records: 0)")
                    print(f"Creating empty dataframe for {status} - {key}")
                    df = self.create_empty_geodataframe(self.pipes_columns)
                else:
                    print(f"Checking {key}: (Number of records: {len(df)})")
                    cm_l.check_needed_df_columns(df, self.pipes_columns)
                    # if address is found then check if the appropriate fields are found
                    if self.address_field in list(df.columns):
                        cm_l.check_needed_df_columns(df, self.address_columns)
                    self.pipes_crs = df.crs
                cm_l.reset_index(df)
                df['auto_id'] = df.index.to_series().apply(lambda x: f"id_{str(x)}")
                dict_data[status][key] = df.copy()
        self.dict_data = dict_data.copy()

    def check_nulls_and_preprocess_pipe_id(self):
        """Checks for null pipe_ids."""
        cm_l.print_formatted_txt("Checks for null pipe ids")
        dict_data = self.dict_data.copy()
        for status, status_dict in dict_data.items():
            for key, df in status_dict.items():
                if not df.empty:
                    cm_l.print_formatted_txt(f"Checking {key}")
                    df, df_issues = cm_tm_l.get_null_col_id(df, self.string_null_values_list, self.pipe_id_field)
                    dict_data[status][key] = df.copy()
                    if not df_issues.empty:
                        self.dict_issues[status][key]['null_pipe_id'] = df_issues.copy()
        self.dict_data = dict_data.copy()

    def check_and_preprocess_material(self):
        """Checks material values and standardize them based on a material dictionary."""
        cm_l.print_formatted_txt("Checks the material values")
        dict_data = self.dict_data.copy()
        for status, status_dict in dict_data.items():
            for key, df in status_dict.items():
                if not df.empty:
                    cm_l.print_formatted_txt(f"Checking {key}")
                    print("Materials before formatting:")
                    display(df[self.material_field].value_counts(dropna=False))
                    df = cm_tm_l.preprocess_material(df, self.material_field, self.material_dict, self.null_values_list,
                                                     show_stats=False)
                    df = ld_cl_l.fill_and_standardize_materials(df, self.string_null_values_list, self.material_field)
                    print("Materials after formatting:")
                    display(df[self.material_field].value_counts(dropna=False))
                    dict_data[status][key] = df.copy()
        self.dict_data = dict_data.copy()

    def check_and_preprocess_diameter(self):
        """Checks material values and standardize them based on a material dictionary."""
        cm_l.print_formatted_txt("Checks for valid diameter values")
        dict_data = self.dict_data.copy()
        for status, status_dict in dict_data.items():
            for key, df in status_dict.items():
                if not df.empty:
                    cm_l.print_formatted_txt(f"Checking {key}")
                    df, df_issues = cm_tm_l.preprocess_diameter(df, self.diameter_field, self.diameter_dict,
                                                                self.min_diameter, self.max_diameter,
                                                                self.null_values_list, plot_graphs=self.plot_graphs)
                    dict_data[status][key] = df.copy()
                    if not df_issues.empty:
                        self.dict_issues[status][key]['invalid_diameter'] = df_issues.copy()
        self.dict_data = dict_data.copy()

    def check_diameter_conflicts(self):
        """Checks for conflicts in the diameters between the public and private pipes"""
        cm_l.print_formatted_txt("Checks for conflicts in the diameters between the public and private pipes")
        pipe_id_field, diameter_field = self.pipe_id_field, self.diameter_field
        join_public_field, join_private_field = self.join_public_field, self.join_private_field
        joined_df = self.joined_ids.copy()
        dict_data = self.dict_data.copy()
        public = dict_data[self.status_act_key][self.act_pub_key].copy()
        private = dict_data[self.status_act_key][self.act_pr_key].copy()

        if not public.empty and not private.empty:
            public = public.rename(columns={pipe_id_field: 'pbid', diameter_field: 'pbdiam'})
            private = private.rename(columns={pipe_id_field: 'prid', diameter_field: 'prdiam'})
            joined_df = joined_df.rename(columns={join_public_field: 'pbid', join_private_field: 'prid'})
            df_merge = pd.merge(private[['prid', 'prdiam']], joined_df, on="prid")
            df_merge = pd.merge(df_merge, public[['pbid', 'pbdiam']], on="pbid")
            df_merge = df_merge[(df_merge['prdiam'] > 0) & (df_merge['pbdiam'] > 0)]
            df_merge['diam_dif'] = abs(df_merge['pbdiam'] - df_merge['prdiam'])
            df_issues = df_merge[df_merge['diam_dif'] > 2]
            df_issues = df_issues.sort_values(by=['diam_dif'])
            if not df_issues.empty:
                cm_l.print_formatted_txt(
                    f"There are {len(df_issues)} active public-private connections with incompatible "
                    f"diameters:", "WARNING")
                display(df_issues.head())
                self.dict_issues[self.status_act_key][self.act_pub_key]['diameter_conflicts'] = df_issues.copy()
            else:
                cm_l.print_formatted_txt("There aren't any active public-private connections with incompatible "
                                         "diameters")

    def check_and_preprocess_install_yr(self):
        """Checks the pipes for null and invalid installation year values."""
        cm_l.print_formatted_txt("Checks for valid install year values")
        dict_data = self.dict_data.copy()
        for status, status_dict in dict_data.items():
            for key, df in status_dict.items():
                if not df.empty:
                    cm_l.print_formatted_txt(f"Checking {key}")
                    df, df_issues = cm_tm_l.preprocess_year(df, self.install_yr_field, self.null_values_list,
                                                            min_year=self.min_inst_yr, max_year=self.max_inst_yr,
                                                            plot_graphs=self.plot_graphs)
                    dict_data[status][key] = df.copy()
                    if not df_issues.empty:
                        self.dict_issues[status][key]['invalid_install_yr'] = df_issues.copy()
        self.dict_data = dict_data.copy()

    def check_and_preprocess_abandon_yr(self):
        """Checks the abandoned pipes for null and invalid abandon year values."""
        cm_l.print_formatted_txt("Checks for valid abandon year values")
        dict_data = self.dict_data.copy()
        for status, status_dict in dict_data.items():
            for key, df in status_dict.items():
                if status == self.status_abd_key and not df.empty:
                    cm_l.print_formatted_txt(f"Checking {key}")
                    if self.abandon_yr_field in df.columns:
                        df, df_issues = cm_tm_l.preprocess_year(df, self.abandon_yr_field, self.null_values_list,
                                                                min_year=self.min_inst_yr, max_year=self.max_inst_yr,
                                                                plot_graphs=self.plot_graphs)
                        dict_data[status][key] = df.copy()
                        if not df_issues.empty:
                            self.dict_issues[status][key]['invalid_abandon_yr'] = df_issues.copy()
                    else:
                        print(f"{key} does not have an abandon year field.")
        self.dict_data = dict_data.copy()

    def check_pipes_duplicate_multiple_columns(self):
        """Checks the pipes having duplicates based on multiple columns."""
        cm_l.print_formatted_txt("Checks for duplicates based on multiple columns")
        dict_data = self.dict_data.copy()
        for status, status_dict in dict_data.items():
            for key, df in status_dict.items():
                if not df.empty:
                    cm_l.print_formatted_txt(f"Checking {key}")
                    df, df_issues = ld_tm_l.drop_duplicate_pipes_on_multiple_columns(df, key, self.field_mapping)
                    dict_data[status][key] = df.copy()
                    if not df_issues.empty:
                        self.dict_issues[status][key]['double_records'] = df_issues.copy()
        self.dict_data = dict_data.copy()

    def check_pipes_duplicate_pipe_id(self):
        """Checks for duplicate pipe_ids in pipes."""
        cm_l.print_formatted_txt("Checks for duplicates based on pipe ids")
        dict_data = self.dict_data.copy()
        for status, status_dict in dict_data.items():
            for key, df in status_dict.items():
                if not df.empty:
                    cm_l.print_formatted_txt(f"Checking {key}")
                    df_nulls = pd.DataFrame()
                    df_doubles = pd.DataFrame()
                    if "null_pipes_id" in self.dict_issues[status].keys():
                        df_nulls = self.dict_issues[status][key]['null_pipes_id'].copy()
                    if "double_records" in self.dict_issues[status].keys():
                        df_doubles = self.dict_issues[status][key]['double_records'].copy()

                    df_issues = ld_tm_l.get_duplicate_pipe_ids(df, df_nulls, df_doubles, self.pipe_id_field)
                    if not df_issues.empty:
                        self.dict_issues[status][key]['double_pipe_id'] = df_issues.copy()

    def check_conflicts_active_state(self):
        """Checks for conflicts between the connected active public and private pipes."""
        cm_l.print_formatted_txt("Checks for conflicts in the active/inactive state between the public and private pipes")
        df_public = self.dict_data[self.status_act_key][self.act_pub_key].copy()
        df_private = self.dict_data[self.status_act_key][self.act_pr_key].copy()
        joined_df = self.joined_ids.copy()
        if not df_public.empty and not df_private.empty:
            df_issues = ld_tm_l.get_active_state_conflicts(df_public, df_private, joined_df, self.pipe_id_field,
                                                           self.active_field, self.join_public_field,
                                                           self.join_private_field)
            if not df_issues.empty:
                self.dict_issues[self.status_act_key][self.act_pub_key]['active_state_conflicts'] = df_issues.copy()

    def check_conflicts_abandoned_year(self):
        """Checks for conflicts in the installation/abandon year between the inplace and abandoned pipes."""
        dict_data = self.dict_data.copy()
        for system in ['public', 'private']:
            cm_l.print_formatted_txt(
                f"Checks for installation year conflicts between the in-place and abandoned {system} pipes")
            if system == "public":
                system_key_for_issues = self.act_pub_key
                df_active = dict_data[self.status_act_key][self.act_pub_key].copy()
                df_abandoned = dict_data[self.status_abd_key][self.abd_pub_key].copy()
            else:
                system_key_for_issues = self.act_pr_key
                df_active = dict_data[self.status_act_key][self.act_pr_key].copy()
                df_abandoned = dict_data[self.status_abd_key][self.abd_pr_key].copy()

            if not df_active.empty and not df_abandoned.empty:
                df_issues = ld_tm_l.get_install_yr_conflicts(df_active, df_abandoned, self.pipe_id_field,
                                                             self.install_yr_field, self.abandon_yr_field,
                                                             self.material_field)
                if not df_issues.empty:
                    self.dict_issues[self.status_act_key][system_key_for_issues]['abd_year_conflicts'] = df_issues.copy()

    def check_for_unknowns_in_abandoned(self):
        """Checks if there are unknowns in the abandoned pipes."""
        dict_data = self.dict_data.copy()
        for system in ['public', 'private']:
            cm_l.print_formatted_txt(
                f"Checks for records with Unknowns material in the abandoned {system} pipes")
            if system == "public":
                system_key_for_issues = self.abd_pub_key
                df_abandoned = dict_data[self.status_abd_key][self.abd_pub_key].copy()
            else:
                system_key_for_issues = self.abd_pr_key
                df_abandoned = dict_data[self.status_abd_key][self.abd_pr_key].copy()

            if not df_abandoned.empty:
                df_issues = df_abandoned[df_abandoned[self.material_field].isin(self.null_values_list)].copy()
                if not df_issues.empty:
                    self.dict_issues[self.status_abd_key][system_key_for_issues]['unknowns_in_abandoned'] = df_issues.copy()
                else:
                    print(f"There aren't any abandoned pipes with unknown material")

    def check_for_unknowns_in_replaced(self):
        """Checks if there are unknowns in the in-place pipes that seem to be replaced, based on the abandoned pipes."""
        dict_data = self.dict_data.copy()
        for system in ['public', 'private']:
            cm_l.print_formatted_txt(
                f"Checks for records with Unknowns material in the {system} pipes, which seem to be replaced")
            if system == "public":
                system_key_for_issues = self.abd_pub_key
                df_active = dict_data[self.status_act_key][self.act_pub_key].copy()
                df_abandoned = dict_data[self.status_abd_key][self.abd_pub_key].copy()
            else:
                system_key_for_issues = self.abd_pr_key
                df_active = dict_data[self.status_act_key][self.act_pr_key].copy()
                df_abandoned = dict_data[self.status_abd_key][self.abd_pr_key].copy()

            if not df_active.empty and not df_abandoned.empty:
                abd_ids = list(df_abandoned[self.pipe_id_field])
                df_issues = df_active[(df_active[self.material_field].isin(self.null_values_list)) & \
                                      (df_active[self.pipe_id_field]).isin(abd_ids)].copy()
                if not df_issues.empty:
                    self.dict_issues[self.status_abd_key][system_key_for_issues]['unknowns_replaced'] = df_issues.copy()
                else:
                    print(f"There aren't any in-place pipes with unknown material that seem to be replaced")

    def check_and_preprocess_geometries(self):
        """Simplifies and re-projects all geometries. It also finds invalid geometries, which are geocoded,
        and duplicate geometries."""
        cm_l.print_formatted_txt("Checks and preprocess of geometries")
        dict_data = self.dict_data.copy()
        for status, status_dict in self.dict_data.items():
            for key, df in status_dict.items():
                if not df.empty:
                    cm_l.print_formatted_txt(f"Checking {key}")
                    df_doubles = pd.DataFrame()
                    if "double_pipe_id" in self.dict_issues[status][key].keys():
                        df_doubles = self.dict_issues[status][key]['double_pipe_id'].copy()
                    df, invalid_geoms, df_doubles = ld_tm_l.preprocess_geometries(df, df_doubles, self.pipe_id_field,
                                                                                  key, self.pipes_crs,
                                                                                  self.null_values_list,
                                                                                  self.field_mapping, self.project.public_dir)
                    dict_data[status][key] = df.copy()
                    if not invalid_geoms.empty:
                        self.dict_issues[status][key]['invalid_geometry'] = invalid_geoms.copy()
                    if not df_doubles.empty:
                        self.dict_issues[status][key]['double_pipe_id_after_explode'] = df_doubles.copy()
        self.dict_data = dict_data.copy()

    def check_and_remove_invalid_coordinates(self):
        """Finds outliers based on the geometry coordinates."""
        cm_l.print_formatted_txt("Checks for coordinate outliers")
        for status, status_dict in self.dict_data.items():
            for key, df in status_dict.items():
                if not df.empty:
                    cm_l.print_formatted_txt(f"Checking {key}")
                    df, df_issues = geom_l.get_coordinate_outliers(df, "auto_id", self.plot_graphs)
                    self.dict_data[status][key] = df.copy()
                    if not df_issues.empty:
                        self.dict_issues[status][key]['invalid_coordinates'] = df_issues.copy()

    def check_pipe_length(self):
        """Checks pipes length for short pipes."""
        cm_l.print_formatted_txt("Checks the pipes length")
        dict_data = self.dict_data.copy()
        for status, status_dict in dict_data.items():
            for key, df in status_dict.items():
                if not df.empty:
                    df['geom_type'] = df.geometry.apply(lambda x: x.geom_type.lower())
                    df_lines = df[df['geom_type'].str.contains("line")].copy()
                    if not df_lines.empty:
                        cm_l.print_formatted_txt(f"Checking {key}")
                        df = cm_l.calculate_length_and_convert_to_ft(df)
                        df_issues_min_len = df[(df['pipe_len'] < self.min_length)].copy()
                        if not df_issues_min_len.empty:
                            print(f"Number of pipes with length < {self.min_length}: {len(df_issues_min_len)}")
                            self.dict_issues[status][key]['short_pipes'] = df_issues_min_len.copy()
                        else:
                            print(f"There aren't any pipes with length < {self.min_length}")

                        df_issues_max_len = df[(df['pipe_len'] > self.max_length)].copy()
                        if not df_issues_max_len.empty:
                            print(f"Number of pipes with length > {self.max_length}: {len(df_issues_max_len)}")
                            self.dict_issues[status][key]['long_pipes'] = df_issues_max_len.copy()
                        else:
                            print(f"There aren't any pipes with length > {self.max_length}")
                        cond_issue_len = df[self.pipe_id_field].isin(df_issues_min_len[self.pipe_id_field]) | \
                                         df[self.pipe_id_field].isin(df_issues_max_len[self.pipe_id_field])
                        df.drop(columns=['geom_type'], inplace=True)
                        dict_data[status][key] = df[~cond_issue_len].copy()
                    else:
                        print(f"{key} do not contain line geometries")
        self.dict_data = dict_data.copy()

    def check_and_preprocess_address(self):
        """Checks the pipes address."""
        cm_l.print_formatted_txt("Checks the pipes address")
        dict_data = self.dict_data.copy()
        for status, status_dict in dict_data.items():
            for key, df in status_dict.items():
                if not df.empty:
                    cm_l.print_formatted_txt(f"Checking {key}")
                    if self.address_field in df.columns:
                        df, df_issues = ld_tm_l.preprocess_address_and_check_invalid(df, self.address_field,
                                                                                     self.street_name_field,
                                                                                     self.street_num_field,
                                                                                     self.string_null_values_list)
                        if not df_issues.empty:
                            self.dict_issues[status][key]['invalid_address'] = df_issues.copy()
                            dict_data[status][key] = df.copy()
                    else:
                        print(f"{key} do not have an address field")
        self.dict_data = dict_data.copy()

    def check_address_geometry_conflicts(self):
        """Checks for conflicts between geometries derived from geocoding, using the addresses,
        and the actual geometries."""
        cm_l.print_formatted_txt("Checks for conflicts between addresses and geometries")
        dict_data = self.dict_data.copy()
        for status, status_dict in dict_data.items():
            for key, df in status_dict.items():
                if not df.empty:
                    if self.address_field in df.columns:
                        cm_l.print_formatted_txt(f"Checking {key}")
                        df_issues = ld_tm_l.get_conflicts_address_geometries(df, self.project.public_dir,
                                                                             self.pipe_id_field, self.street_name_field,
                                                                             self.street_num_field, self.city_field,
                                                                             self.zip_field, self.max_geocode_distance)
                        if not df_issues.empty:
                            self.dict_issues[status][key]['conflict_geocode_geometries'] = df_issues.copy()
                    else:
                        print(f"{key} do not have an address field")

    def check_for_mismatched_addresses(self):
        """Checks for mismatched addresses between the actual address values and the ones derived from the geometry."""
        cm_l.print_formatted_txt("Checks for mismatched addresses")
        dict_data = self.dict_data.copy()
        for status, status_dict in dict_data.items():
            for key, df in status_dict.items():
                if not df.empty:
                    if self.address_field in df.columns:
                        cm_l.print_formatted_txt(f"Checking {key}")
                        df_issues = ld_tm_l.get_mismatched_addresses(df, self.pipe_id_field, self.street_name_field,
                                                                     self.street_num_field, self.city_field,
                                                                     self.zip_field, self.state_field,
                                                                     self.country_field, self.string_null_values_list)
                        if not df_issues.empty:
                            self.dict_issues[status][key]['mismatched_addresses'] = df_issues.copy()
                    else:
                        print(f"{key} do not have an address field")

    def check_connection_issues(self):
        """Checks for connection issues between the active public and private pipes; more specifically:
        # 1. finds the public pipes that are not connected with any private pipe,
        # 2. finds the public pipes that are connected with multiple private pipes
        # 3. finds the private pipes that are not connected with any public pipe,
        # 4. finds the private pipes that are connected with multiple public pipes
        """
        cm_l.print_formatted_txt("Checks for connection issues between the active public and private pipes")
        df_public = self.dict_data[self.status_act_key][self.act_pub_key].copy()
        df_private = self.dict_data[self.status_act_key][self.act_pr_key].copy()

        df_issues = ld_tm_l.get_public_private_connections(df_public, df_private, self.pipe_id_field,
                                                           self.max_connected_distance)
        # issues in the public side
        if not df_issues['public']['multiple_connections'].empty:
            self.dict_issues[self.status_act_key][self.act_pub_key]['multiple_connections'] = df_issues['public'][
                'multiple_connections'].copy()
        if not df_issues['public']['unknown_connections'].empty:
            self.dict_issues[self.status_act_key][self.act_pub_key]['unknown_connections'] = df_issues['public'][
                'unknown_connections'].copy()
        # issues in the private side
        if not df_issues['private']['multiple_connections'].empty:
            self.dict_issues[self.status_act_key][self.act_pr_key]['multiple_connections'] = df_issues['private'][
                'multiple_connections'].copy()
        if not df_issues['private']['unknown_connections'].empty:
            self.dict_issues[self.status_act_key][self.act_pr_key]['unknown_connections'] = df_issues['private'][
                'unknown_connections'].copy()

    def check_duplicate_geometries(self):
        """Checks for duplicate geometries."""
        cm_l.print_formatted_txt("Checks for duplicate geometries")
        dict_data = self.dict_data.copy()
        for status, status_dict in dict_data.items():
            for key, df in status_dict.items():
                if not df.empty:
                    cm_l.print_formatted_txt(f"Checking {key}")
                    df_issues = ld_tm_l.get_duplicate_geometries(df, self.pipe_id_field, self.geometry_field)
                    if not df_issues.empty:
                        self.dict_issues[status][key]['duplicate_geometries'] = df_issues.copy()

    def check_nearby_geometries(self):
        """Checks for nearby geometries with different address."""
        cm_l.print_formatted_txt("Checks for nearby geometries geometries with different address")
        dict_data = self.dict_data.copy()
        for status, status_dict in dict_data.items():
            for key, df in status_dict.items():
                if not df.empty:
                    if self.address_field in df.columns:
                        cm_l.print_formatted_txt(f"Checking {key}")
                        df_issues = ld_tm_l.get_nearby_geometries(df, self.pipe_id_field, self.address_field,
                                                                  self.geometry_field,
                                                                  self.max_nearby_geometry_distance)
                        if not df_issues.empty:
                            self.dict_issues[status][key]['nearby_geometries'] = df_issues.copy()
                    else:
                        print(f"{key} do not have an address field")

    def check_verified_unknowns(self):
        """Checks for verified with unknown material."""
        cm_l.print_formatted_txt("Checks for verified pipes with unknown material")
        dict_data = self.dict_data.copy()
        for status, status_dict in dict_data.items():
            for key, df in status_dict.items():
                if not df.empty:
                    cm_l.print_formatted_txt(f"Checking {key}")
                    df_issues = df[
                        (df[self.verified_field]) & (df[self.material_field].isin(self.null_values_list))].copy()
                    if not df_issues.empty:
                        print(f"Number of verified pipes with unknown material: {len(df_issues)}")
                        self.dict_issues[status][key]['verified_unknowns'] = df_issues.copy()
                    else:
                        print(f"There aren't any verified pipes with unknown material")

    def check_lead_with_invalid_install_yr(self):
        """Checks the pipes having lead material and invalid installation year based on the ban year."""
        cm_l.print_formatted_txt(
            f"Checking erroneous lead diameter values based on the ban year: {self.project.ban_year}")
        dict_data = self.dict_data.copy()
        for status, status_dict in dict_data.items():
            for key, df in status_dict.items():
                if not df.empty:
                    cm_l.print_formatted_txt(f"Checking {key}")
                    cond_errors = (df[self.material_field] == "LEAD") & \
                                  (df[self.install_yr_field] > self.project.ban_year)
                    df_issues = df[cond_errors].copy()
                    if not df_issues.empty:
                        self.dict_issues[status][key]['lead_invalid_install_yr'] = df_issues.copy()
                    else:
                        print(f"There aren't any lead pipes with invalid installation year")
                    dict_data[status][key] = df[~cond_errors].copy()
        self.dict_data = dict_data.copy()

    def check_lead_with_invalid_diameter(self):
        """Checks the pipes having lead material and invalid diameter based on the thresshold."""
        cm_l.print_formatted_txt(f"Checking erroneous lead installation year values based on the "
                                 f"threshold: {self.project.diam_threshold_for_knowns}")
        dict_data = self.dict_data.copy()
        for status, status_dict in dict_data.items():
            for key, df in status_dict.items():
                if not df.empty:
                    cm_l.print_formatted_txt(f"Checking {key}")
                    cond_errors = (df[self.material_field] == "LEAD") & \
                                  (df[self.diameter_field] >= self.project.diam_threshold_for_knowns)
                    df_issues = df[cond_errors].copy()
                    if not df_issues.empty:
                        self.dict_issues[status][key]['lead_invalid_diameter'] = df_issues.copy()
                    else:
                        print(f"There aren't any lead pipes with invalid diameter")
                    dict_data[status][key] = df[~cond_errors].copy()
        self.dict_data = dict_data.copy()

    def concatenate_act_and_abd_dfs(self):
        """Concatenate all pipes."""
        cm_l.print_formatted_txt("Concatenate all pipes")
        dict_data = self.dict_data.copy()
        all_pipes = []
        for status, status_dict in dict_data.items():
            for key, df in status_dict.items():
                if not df.empty:
                    if status == self.status_act_key:
                        df[self.abandoned_field] = 0
                        df[self.abandon_yr_field] = 0
                    else:
                        df[self.abandoned_field] = 1
                    if key == self.act_pub_key:
                        df[self.pipe_id_field] = df[self.pipe_id_field].apply(lambda x: f"{self.pub_prefix}{x}")
                    elif key == self.act_pr_key:
                        df[self.pipe_id_field] = df[self.pipe_id_field].apply(lambda x: f"{self.pr_prefix}{x}")
                    elif key == self.abd_pub_key:
                        df[self.pipe_id_field] = df[self.pipe_id_field].apply(lambda x: f"{self.pub_prefix}_abd{x}")
                    elif key == self.abd_pr_key:
                        df[self.pipe_id_field] = df[self.pipe_id_field].apply(lambda x: f"{self.pr_prefix}_abd{x}")
                    all_pipes.append(df)
        self.all_pipes = pd.concat(all_pipes, ignore_index=True)

    def pipes_statistics_per_material(self):
        """Prints the pipes statistics per material and per diameter."""
        cm_l.print_formatted_txt("Installation year statistics per material")
        cm_l.get_statistics_per_group(self.all_pipes, self.material_field, field_to_check=self.install_yr_field,
                                         exclude_zeros=True, digits=0)
        cm_l.print_formatted_txt("Diameter statistics per material")
        cm_l.get_statistics_per_group(self.all_pipes, self.material_field, field_to_check=self.diameter_field,
                                         exclude_zeros=True, digits=2)

    def check_for_isolated_pipes(self):
        """Finds the isolated pipes."""
        cm_l.print_formatted_txt("Find isolated pipes")
        clusters, not_main_cluster = cm_tm_l.get_isolated_pipes(self.all_pipes, self.project.isolated_pipes_col_nm,
                                                                self.min_cluster_distance)
        if not not_main_cluster.empty:
            self.all_pipes = clusters.copy()
            act_isolated = not_main_cluster[not_main_cluster[self.abandoned_field] == 0].copy()
            if not act_isolated.empty:
                self.dict_issues[self.status_act_key][self.act_pub_key]['act_isolated_pipes'] = act_isolated.copy()

            abd_isolated = not_main_cluster[not_main_cluster[self.abandoned_field] == 1].copy()
            if not abd_isolated.empty:
                self.dict_issues[self.status_abd_key][self.abd_pub_key]['abd_isolated_pipes'] = abd_isolated.copy()

    def save_issues_in_xlsx(self):
        """Saves all the dataframes with the issues in xlsx files."""
        if not os.path.exists(self.output_folder) and not os.path.isdir(self.output_folder):
            cm_l.mkdir(self.output_folder)

        cm_l.print_formatted_txt("Exporting the issues in the active public pipes")
        dict_act_pub_issues = self.dict_issues[self.status_act_key][self.act_pub_key].copy()
        if not all(df.empty for df in dict_act_pub_issues.values()):
            cm_tm_l.export_issue_df_to_xls(dict_act_pub_issues, self.act_pub_pipes_xls_flnm, self.output_folder)
        cm_l.print_formatted_txt("Exporting the issues in the active private pipes")
        dict_act_pr_issues = self.dict_issues[self.status_act_key][self.act_pr_key].copy()
        if not all(df.empty for df in dict_act_pr_issues.values()):
            cm_tm_l.export_issue_df_to_xls(dict_act_pr_issues, self.act_pr_pipes_xls_flnm, self.output_folder)

        cm_l.print_formatted_txt("Exporting the issues in the abandoned public pipes")
        dict_abd_pub_issues = self.dict_issues[self.status_abd_key][self.abd_pub_key].copy()
        if not all(df.empty for df in dict_abd_pub_issues.values()):
            cm_tm_l.export_issue_df_to_xls(dict_abd_pub_issues, self.abd_pub_pipes_xls_flnm, self.output_folder)
        cm_l.print_formatted_txt("Exporting the issues in the abandoned private pipes")
        dict_abd_pr_issues = self.dict_issues[self.status_abd_key][self.abd_pr_key].copy()
        if not all(df.empty for df in dict_abd_pr_issues.values()):
            cm_tm_l.export_issue_df_to_xls(dict_abd_pr_issues, self.abd_pr_pipes_xls_flnm, self.output_folder)
