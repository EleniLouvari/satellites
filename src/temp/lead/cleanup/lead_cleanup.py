from import_libraries import *
import global_variables as gb_l
import libraries.common_libraries.generic_library as cm_l
import libraries.common_libraries.geom_library as geom_l
import libraries.lead.cleanup.lead_cleanup_library as ld_cl_l
import libraries.common_libraries.cleanup_library as ld_cm_l

class LeadGeneric:
    """Automation of cleaning data"""

    def __init__(self, project):
        self.project = project

    def cleanup_process(self):
        """All checks and cleanup process are here"""
        self.project.print_started_message(self.__class__.__name__)

        customer = self.project.customer
        lead_run_type = self.project.lead_run_type
        pub_pipes = self.project.pub_pipes
        abd_pub_pipes = self.project.abd_pub_pipes
        pr_pipes = self.project.pr_pipes
        abd_pr_pipes = self.project.abd_pr_pipes
        extra_dfs = self.project.extra_dfs
        extra_features = self.project.extra_features
        extra_dir = self.project.extra_dir
        verified_samples = self.project.verified_samples
        joined_ids = self.project.joined_ids
        field_mapping = self.project.field_mapping
        pipe_id_field = field_mapping['pipe_id_field']
        public_id_field = field_mapping['join_public_field']
        private_id_field = field_mapping['join_private_field']
        install_yr_field = field_mapping['install_yr_field']
        abandoned_field = field_mapping['abandoned_field']
        abandon_yr_field = field_mapping['abandon_yr_field']
        abandon_mat_field = field_mapping['abandon_mat_field']
        diameter_field = field_mapping['diameter_field']
        material_field = field_mapping['material_field']
        verified_field = field_mapping['verified_field']
        vrsource_field = field_mapping['vrsource_field']
        hsmaterial_field = field_mapping['hsmaterial_field']
        address_field = field_mapping['address_field']
        active_fields = self.project.active_fields
        abd_fields = self.project.abd_fields
        plot_graphs = self.project.plot_graphs
        diameter_unit = self.project.diameter_unit
        min_inst_yr = self.project.min_inst_yr
        max_inst_yr = datetime.now().year
        min_diameter = self.project.min_diameter
        max_diameter = self.project.max_diameter
        max_diam_fill_value = self.project.max_diam_fill_value
        string_null_values_list = self.project.string_null_values_list
        null_values_list = self.project.null_values_list
        fill_missing_txt_method = self.project.fill_missing_txt_method
        ban_year = self.project.ban_year
        diam_threshold_for_knowns = self.project.diam_threshold_for_knowns
        prediction_year = self.project.prediction_year
        verbose = self.project.verbose
        data_resample_rate = self.project.data_resample_rate
        field_dtypes = self.project.field_dtypes
        sample_unknown_only = self.project.sample_unknown_only

        act_list = [field_mapping[key] for key in active_fields]
        abd_list = [field_mapping[key] for key in abd_fields]

        cm_l.print_formatted_txt(f"Check the existence of public/private pipes")
        pub_pipes, pr_pipes = ld_cl_l.check_for_public_private_pipes(pub_pipes, pr_pipes, act_list)

        # TODO: check if we need to create variables for the joined_ids columns: ['public_id', 'private_id']
        if not pub_pipes.empty and not pr_pipes.empty:
            cm_l.print_formatted_txt(f"Check the existence of joined public/private dataframe")
            if not joined_ids.empty:
                print("Joined public/private dataframe found")
                cm_l.check_needed_df_columns(joined_ids, ['public_id', 'private_id'])
            else:
                error_msg = "You need to provide the dataframe <joined_ids> for public/private ids."
                raise Exception(error_msg)
        else:
            joined_ids = pd.DataFrame(columns=['public_id', 'private_id'])

        cm_l.print_formatted_txt("Uppercase all material values")
        pub_pipes = ld_cl_l.uppercase_field_values(pub_pipes, material_field)
        pr_pipes = ld_cl_l.uppercase_field_values(pr_pipes, material_field)

        cm_l.print_formatted_txt("Check for abandoned pipes")
        abd_pub_pipes = ld_cl_l.check_for_abandoned_pipes(pub_pipes, abd_pub_pipes, abd_list,
                                                          "public")
        abd_pr_pipes = ld_cl_l.check_for_abandoned_pipes(pr_pipes, abd_pr_pipes, abd_list,
                                                         "private")

        cm_l.print_formatted_txt("Be sure that the pipe_id is object type")
        ld_cm_l.convert_pipe_id_to_str_helper(pub_pipes, abd_pub_pipes)
        ld_cm_l.convert_pipe_id_to_str_helper(pr_pipes, abd_pr_pipes)
        ld_cm_l.convert_pipe_id_to_str(joined_ids, field_name=public_id_field)
        ld_cm_l.convert_pipe_id_to_str(joined_ids, field_name=private_id_field)

        cm_l.print_formatted_txt("Cleanup pipe_id fields from spaces")
        ld_cm_l.clean_pipe_id_helper(pub_pipes, abd_pub_pipes)
        ld_cm_l.clean_pipe_id_helper(pr_pipes, abd_pr_pipes)
        joined_ids[public_id_field] = joined_ids[public_id_field].apply(
            lambda x: str(x).strip())
        joined_ids[private_id_field] = joined_ids[private_id_field].apply(
            lambda x: str(x).strip())

        cm_l.print_formatted_txt("Create df to record drops")
        dropped_pipes_df = ld_cm_l.create_dropped_df(pipe_id_field)

        cm_l.print_formatted_txt("Drop any customer requested record")
        pub_pipes, dropped_pipes_df = ld_cm_l.drop_customer_requested_records(pub_pipes, dropped_pipes_df, pipe_id_field)
        pr_pipes, dropped_pipes_df = ld_cm_l.drop_customer_requested_records(pr_pipes, dropped_pipes_df,
                                                                             pipe_id_field)
        abd_pub_pipes, dropped_pipes_df = ld_cm_l.drop_customer_requested_records(abd_pub_pipes, dropped_pipes_df,
                                                                                  pipe_id_field)
        abd_pr_pipes, dropped_pipes_df = ld_cm_l.drop_customer_requested_records(abd_pr_pipes, dropped_pipes_df,
                                                                                 pipe_id_field)

        cm_l.print_formatted_txt("Reset index")
        cm_l.reset_index(pub_pipes, abd_pub_pipes, pr_pipes, abd_pr_pipes)

        cm_l.print_formatted_txt("Check if we have all needed fields for public/private active and abandoned pipes")
        cm_l.check_needed_df_columns(pub_pipes, act_list)
        cm_l.check_needed_df_columns(abd_pub_pipes, abd_list)
        cm_l.check_needed_df_columns(pr_pipes, act_list)
        cm_l.check_needed_df_columns(abd_pr_pipes, abd_list)

        if len(extra_dfs):
            cm_l.print_formatted_txt(f"Check if we have all needed fields for extra dataframes")
            for key in extra_dfs:
                cm_l.check_needed_df_columns(extra_dfs[key],
                                             [install_yr_field, abandoned_field,
                                              field_mapping['geometry_field']])

        # TODO: this must not be in the generic process
        if len(verified_samples):
            cm_l.check_needed_df_columns(verified_samples, [pipe_id_field, material_field, diameter_field,
                                                            install_yr_field, abandon_mat_field])

        cm_l.print_formatted_txt("Create abandoned and abandon_yr fields")
        pub_pipes, abd_pub_pipes = ld_cl_l.create_abd_pipes_and_check_field_columns(pub_pipes, abd_pub_pipes,
                                                                                    field_mapping, abd_list)
        pr_pipes, abd_pr_pipes = ld_cl_l.create_abd_pipes_and_check_field_columns(pr_pipes, abd_pr_pipes,
                                                                                  field_mapping, abd_list)

        if plot_graphs:
            cm_l.print_formatted_txt("Plot missing values")
            if not pub_pipes.empty:
                display(pub_pipes.info())
                #ld_cm_l.plot_missing_values(pub_pipes, name="Public pipes")
            if not pr_pipes.empty:
                display(pr_pipes.info())
                #ld_cm_l.plot_missing_values(pr_pipes, name="Private pipes")

        cm_l.print_formatted_txt("Extract installation year for active and abandoned pipes")
        for df in [pub_pipes, pr_pipes,abd_pub_pipes, abd_pr_pipes]:
            message, cases = ld_cm_l.guess_and_set_date_format(df, install_yr_field, "%Y")
            if len(message) != 0:
                print(message, cases)

        cm_l.print_formatted_txt("Extract abandon year for abandoned pipes")
        for df in [abd_pub_pipes, abd_pr_pipes]:
            message, cases = ld_cm_l.guess_and_set_date_format(df, abandon_yr_field, "%Y")
            if len(message) != 0:
                print(message, cases)

        if not joined_ids.empty:
            cm_l.print_formatted_txt("Check that all ids in SL matched with joined ones")
            issue = pub_pipes[
                ~pub_pipes[pipe_id_field].isin(list(joined_ids[public_id_field]))]
            if not issue.empty:
                display(issue.head())
                raise Exception(
                    f"Found {len(issue)} public {pipe_id_field} not included in the joined dataframe!")
            # check all pipe_id in private service lines matches with join_ids
            issue = pr_pipes[
                ~pr_pipes[pipe_id_field].isin(list(joined_ids[private_id_field]))].copy()
            if not issue.empty:
                display(issue.head())
                raise Exception(
                    f"Found {len(issue)} private {pipe_id_field} not included in the joined dataframe!")

        cm_l.print_formatted_txt("Check values in the fields")
        print("Public pipes:")
        pub_pipes, abd_pub_pipes = ld_cl_l.check_field_values(pub_pipes, abd_pub_pipes, field_mapping, null_values_list)
        print("Private pipes:")
        pr_pipes, abd_pr_pipes = ld_cl_l.check_field_values(pr_pipes, abd_pr_pipes, field_mapping, null_values_list)

        cm_l.print_formatted_txt("Remove records without pipe_id")
        print("Public active")
        public, dropped_pipes_df = ld_cm_l.remove_rows_with_nan_value_in_field(pub_pipes,
                                                                               pipe_id_field,
                                                                               pipe_id_field,
                                                                               dropped_pipes_df)
        print("Public abandoned")
        abd_pub_pipes, dropped_pipes_df = ld_cm_l.remove_rows_with_nan_value_in_field(abd_pub_pipes,
                                                                                      pipe_id_field,
                                                                                      pipe_id_field,
                                                                                      dropped_pipes_df)
        print("Private active")
        private, dropped_pipes_df = ld_cm_l.remove_rows_with_nan_value_in_field(pr_pipes,
                                                                                pipe_id_field,
                                                                                pipe_id_field,
                                                                                dropped_pipes_df)
        print("Private abandoned")
        abd_pr_pipes, dropped_pipes_df = ld_cm_l.remove_rows_with_nan_value_in_field(abd_pr_pipes,
                                                                                     pipe_id_field,
                                                                                     pipe_id_field,
                                                                                     dropped_pipes_df)

        cm_l.print_formatted_txt("Format ids to avoid duplications")
        pub_pipes, pr_pipes, abd_pub_pipes, abd_pr_pipes, joined_ids = ld_cl_l.format_pipe_id(pub_pipes, pr_pipes,
                                                                                              abd_pub_pipes,
                                                                                              abd_pr_pipes, joined_ids,
                                                                                              field_mapping)

        cm_l.print_formatted_txt("Make sure pipes and abd_pipes have the same crs")
        if not pub_pipes.empty:
            pub_pipes = ld_cm_l.check_and_reproject(pub_pipes, pub_pipes, "pub_pipes", "pub_pipes", False,
                                                    self.project.lat_lon_crs)
            abd_pub_pipes = ld_cm_l.check_and_reproject(abd_pub_pipes, pub_pipes, "abd_pub_pipes", "pub_pipes", True,
                                                        self.project.lat_lon_crs)
            if not pr_pipes.empty:
                pr_pipes = ld_cm_l.check_and_reproject(pr_pipes, pub_pipes, "pr_pipes", "pub_pipes",
                                                       True, self.project.lat_lon_crs)
                abd_pr_pipes = ld_cm_l.check_and_reproject(abd_pr_pipes, pub_pipes, "abd_pr_pipes", "pub_pipes",
                                                           True, self.project.lat_lon_crs)
        elif not pr_pipes.empty:
            pr_pipes = ld_cm_l.check_and_reproject(pr_pipes, pr_pipes, "pr_pipes", "pr_pipes", False,
                                                   self.project.lat_lon_crs)
            abd_pr_pipes = ld_cm_l.check_and_reproject(abd_pr_pipes, pr_pipes, "abd_pr_pipes", "pr_pipes", True,
                                                       self.project.lat_lon_crs)

        cm_l.print_formatted_txt("Merge public & private pipes")
        pipes = ld_cl_l.concatenate_public_private_pipes(pub_pipes, pr_pipes)
        abd_pipes = ld_cl_l.concatenate_public_private_pipes(abd_pub_pipes, abd_pr_pipes, raise_exception=False)

        cm_l.print_formatted_txt("Remove pipe records without geometry")
        pipes, dropped_pipes_df = ld_cm_l.remove_rows_with_no_geometry(pipes, pipe_id_field, dropped_pipes_df)
        abd_pipes, dropped_pipes_df = ld_cm_l.remove_rows_with_no_geometry(abd_pipes, pipe_id_field,
                                                                           dropped_pipes_df, raise_exception=False)

        cm_l.print_formatted_txt("Convert pipe_diam to inches")
        pipes = ld_cm_l.convert_to_inches(pipes, diameter_field, diameter_unit)
        abd_pipes = ld_cm_l.convert_to_inches(abd_pipes, diameter_field, diameter_unit)

        pipes.name = "Pipes"
        abd_pipes.name = "Abandoned pipes"
        ld_cm_l.display_value_counts_distribution_of_field_values(pipes, install_yr_field)
        ld_cm_l.display_value_counts_distribution_of_field_values(abd_pipes, install_yr_field)

        if ban_year is None:
            ban_year = ld_cl_l.get_ban_year(pipes)
            self.project.ban_year = ban_year

        if len(extra_dfs):
            cm_l.print_formatted_txt("Check extra dataframes to be used for filling the installation year")
            extra_dfs = ld_cl_l.check_and_format_extra_dataframes(pipes, extra_dfs, min_inst_yr, max_inst_yr,
                                                                  install_yr_field, abandoned_field)

        # TODO: this must not be here
        if len(verified_samples):
            cm_l.print_formatted_txt("Check the verified samples provided by the customer")
            # TODO: do not fill with 0 the pipes that are not in min and max limits
            verified_samples = ld_cl_l.check_and_format_verified_data(verified_samples, field_mapping,
                                                                      min_diameter, max_diameter,
                                                                      diam_threshold_for_knowns,
                                                                      min_inst_yr, max_inst_yr,
                                                                      ban_year, null_values_list)

        if not abd_pipes.empty:
            cm_l.print_formatted_txt("Check abandoned pipes geometries")
            print("Drop Z dimension of abandoned pipes")
            abd_pipes = geom_l.convert_geometry_three_to_two_d(abd_pipes)
            print("\nAbandoned pipes: convert geometries to lines")
            abd_pipes = ld_cl_l.handle_no_line_geometries(abd_pipes, pipe_id_field, field_mapping['pb_prefix'],
                                                          field_mapping['pr_prefix'])
            print("\nAbandoned pipes: calculate and append pipe_len field to the pipes data based on geometry")
            abd_pipes = ld_cm_l.calculate_length_and_convert_to_ft(abd_pipes)
            print("\nHandle duplicate abandoned pipes")
            abd_pipes, dropped_pipes_df = ld_cm_l.handle_duplicate_pipe_ids(abd_pipes, dropped_pipes_df)
            print("\nExplode multi geometries of abandoned pipes")
            abd_pipes, _ = geom_l.pipes_simplify_multigeometries(abd_pipes, None, pipe_id_field)

        cm_l.print_formatted_txt("Check pipes geometries")
        print("Drop Z dimension of pipes")
        pipes = geom_l.convert_geometry_three_to_two_d(pipes)
        print("\nActive pipes: convert geometries to lines")
        pipes = ld_cl_l.handle_no_line_geometries(pipes, pipe_id_field, field_mapping['pb_prefix'],
                                                  field_mapping['pr_prefix'])
        print("\nActive pipes: calculate and append pipe_len field to the pipes data based on geometry")
        pipes = ld_cm_l.calculate_length_and_convert_to_ft(pipes)
        print("\nHandle duplicate active pipes")
        pipes, dropped_pipes_df = ld_cm_l.handle_duplicate_pipe_ids(pipes, dropped_pipes_df)
        print("\nExplode multi geometries of active pipes")
        pipes, _ = geom_l.pipes_simplify_multigeometries(pipes, None, pipe_id_field)
        print("\nDrop duplicate geometries of active pipes")
        pb_pipes = pipes[pipes[pipe_id_field].str.startswith(field_mapping['pb_prefix'])].copy()
        pr_pipes = pipes[pipes[pipe_id_field].str.startswith(field_mapping['pr_prefix'])].copy()
        pb_pipes, dropped_pipes_df = geom_l.drop_duplicate_geometries(pb_pipes, None, dropped_pipes_df, pipe_id_field,
                                                                      install_yr_field)
        pr_pipes, dropped_pipes_df = geom_l.drop_duplicate_geometries(pr_pipes, None, dropped_pipes_df, pipe_id_field,
                                                                      install_yr_field)
        pipes = pd.concat([pb_pipes, pr_pipes], ignore_index=True)
        pipes = pipes.reset_index(drop=True)

        # ACT + ABD PIPES - GENERIC CLEANUP
        if not abd_pipes.empty:
            cm_l.print_formatted_txt("Concatenate active and abandoned pipes")
            pipes = ld_cm_l.concatenate_act_and_abd_dfs(pipes, abd_pipes)

        cm_l.print_formatted_txt("Fill missing material values with UNKNOWN and standardize lead, galvanized, copper")
        print("Materials before formatting:")
        display(pipes[material_field].value_counts(dropna=False))
        pipes = ld_cl_l.fill_and_standardize_materials(pipes, string_null_values_list, material_field)
        print("Materials after formatting:")
        display(pipes[material_field].value_counts(dropna=False))

        # address must be in the form: 'Number Street, City, State'
        cm_l.print_formatted_txt("Fill missing address values with UNKNOWN")
        pipes = ld_cl_l.fill_null_address(pipes, field_mapping, null_values_list)

        # TODO: how to handle the common records between public-abd_public and private-abd_private?
        cm_l.print_formatted_txt("Check common active-abandoned pipes with same install year")
        pipes, _ = ld_cl_l.find_common_install_yr_act_abd(pipes, field_mapping, min_inst_yr, ban_year)

        cm_l.print_formatted_txt("Add zero in missing or <0 diameter values")
        pipes[diameter_field].fillna(0, inplace=True)
        pipes.loc[pipes[diameter_field] < 0, diameter_field] = 0

        # TODO: replace all hardcoded 'LEAD' material with a material variable
        cm_l.print_formatted_txt(f"Checking erroneous lead diameter values based on threshold: {diam_threshold_for_knowns}")
        cond = (pipes[material_field] == "LEAD") & (pipes[diameter_field] >= diam_threshold_for_knowns)
        issues = pipes[cond].copy()
        if not issues.empty:
            issues = issues.sort_values(by=diameter_field, ascending=False)
            cm_l.print_formatted_txt(f"There are {len(issues)} lead pipes with diameter>={diam_threshold_for_knowns} "
                                     f"inches. They will be set to 0.", "WARNING")
            display(issues.head())
            pipes.loc[cond, diameter_field] = 0

        cm_l.print_formatted_txt(f"Define as knowns the pipes with pipe_diam >= {diam_threshold_for_knowns} inches")
        pipes = ld_cl_l.set_unknown_material_based_on_diameter(pipes, material_field, vrsource_field, diameter_field,
                                                               diam_threshold_for_knowns)
        display(pipes[material_field].value_counts(dropna=False))

        cm_l.print_formatted_txt("Fill missing diameter values")
        ld_cl_l.print_percentage_of_null_diameter(pipes, field_mapping, lead_run_type)
        pipes = ld_cl_l.fill_null_diameters(pipes, joined_ids, lead_run_type, field_mapping,
                                            self.project.remove_features, max_diam_fill_value)

        cm_l.print_formatted_txt("Checking erroneous installation year values of pipes")
        cond = (pipes[material_field] == "LEAD") & (pipes[install_yr_field] >= ban_year)
        issued = pipes[cond].copy()
        if not issued.empty:
            issued = issued.sort_values(by=install_yr_field, ascending=False)
            cm_l.print_formatted_txt(f"There are {len(issued)} lead pipes with install_yr>={ban_year}."
                                     f"They will be set to 0.", "WARNING")
            display(issued.head())
            pipes.loc[cond, install_yr_field] = 0

        cm_l.print_formatted_txt(f"{customer} ban year: {ban_year}")
        pipes = ld_cl_l.set_unknown_material_based_on_install_yr(pipes, install_yr_field, material_field,
                                                                 vrsource_field, ban_year)
        display(pipes[material_field].value_counts(dropna=False))

        cm_l.print_formatted_txt("Replace missing or erroneous installation year values")
        pipes = ld_cl_l.fill_install_yr_for_service_lines(pipes, extra_dfs, field_mapping, min_inst_yr, max_inst_yr,
                                                          ban_year)
        pipes[install_yr_field] = pipes[install_yr_field].astype("int16")

        cm_l.print_formatted_txt("Fix install year >= prediction year")
        pipes = ld_cl_l.set_unknown_material_based_on_install_yr(pipes, install_yr_field, material_field,
                                                                 vrsource_field, prediction_year,
                                                                 extra_msg="after filling nulls")

        cm_l.print_formatted_txt(f"Sanity minimum length check")
        min_len = pipes['pipe_len'].min()
        cm_l.print_formatted_txt(f"Minimum length detected: {min_len}")
        display(pipes[pipes['pipe_len'] == min_len])

        # TODO: check if this needs to be moved before the filling of null diameter and install year values
        cm_l.print_formatted_txt(f"Fill {hsmaterial_field} values with historical material values")
        pipes[hsmaterial_field] = "UNKNOWN"
        cond = (~pipes[verified_field]) & (~pipes[material_field].str.upper().str.contains("UNKNOWN"))
        cond = cond & (pipes[diameter_field] < diam_threshold_for_knowns) & (pipes[install_yr_field] <= ban_year)
        pipes.loc[cond, hsmaterial_field] = pipes[cond][material_field]
        pipes.loc[cond, vrsource_field] = VerificationSource.HISTORICAL.name

        if len(verified_samples) > 0:
            self.project.verified_samples = verified_samples
            # TODO: move to a <sample evaluation process> the calculation of EPA statistics and the inventory accuracy
            # TODO: insights must be created for the UI including the samples evaluation
            cm_l.print_formatted_txt("Sampling: get unknown material pipes according to EPA")
            # we don't need here pb_epa_unknown and pr_epa_unknown dataframes, but these contained in the return
            # statement of the function
            df_epa_unknown, pb_epa_unknown, pr_epa_unknown = sampl_l.filter_pipes_by_epa(pipes, ban_year, field_mapping,
                                                                                         diam_threshold_for_knowns,
                                                                                         print_msg=True,
                                                                                         sample_unknown_only=sample_unknown_only)
            extra_verified_samples = verified_samples[
                ~verified_samples[pipe_id_field].isin(list(df_epa_unknown[pipe_id_field]))].copy()
            if not extra_verified_samples.empty:
                cm_l.print_formatted_txt(f"There are {len(extra_verified_samples)} verified samples not included "
                                         f"in the sampling pool", "WARNING")
                display(extra_verified_samples)

            cm_l.print_formatted_txt(f"Sampling: calculate system statistics according to EPA")
            self.project.lead_expected = ld_cl_l.calculate_epa_sample_statistics(pipes, df_epa_unknown,
                                                                                 verified_samples, field_mapping)

            cm_l.print_formatted_txt(f"Sampling: calculate inventory accuracy")
            ld_cl_l.calculate_inventory_accuracy(pipes, verified_samples, field_mapping)

            # TODO: we need to fix this asap - cleanup sampling process from generic
            cm_l.print_formatted_txt(f"Sampling: Update pipes with verified sample data")
            pipes = ld_cl_l.update_pipes_from_samples(pipes, verified_samples, field_mapping)

        cm_l.print_formatted_txt(f"Material value counts for public and private side")
        ld_cl_l.display_public_private_value_counts(pipes, field_mapping, field_to_count=material_field)
        cm_l.print_formatted_txt(f"Verification source value counts for public and private side")
        ld_cl_l.display_public_private_value_counts(pipes, field_mapping, field_to_count=vrsource_field)
        cm_l.print_formatted_txt(f"Unverified historical pipes info")
        ld_cl_l.print_unverified_historical_material_info(pipes, field_mapping)

        if data_resample_rate < 1.0:
            # we will use this for sampling Regressions test's customers
            cm_l.print_formatted_txt(f"Resample pipes data. Before resampling, number of pipes: {len(pipes)}")
            pipes = pipes.sample(frac=data_resample_rate, random_state=np.random.seed(self.project.data_random_state))
            cm_l.print_formatted_txt(f"After resampling, number of pipes: {len(pipes)}")

        if extra_features:
            cm_l.print_formatted_txt(f"Add extra features: {extra_features}")
            # after adding extra features, we don't want to fill missing values in these fields
            cols_to_skip = [pipe_id_field, diameter_field, address_field, material_field, hsmaterial_field,
                            vrsource_field]
            pipes, self.project.extra_features_col_nms = ld_cl_l.add_extra_features_to_pipes(pipes, pipe_id_field,
                                                                                             extra_dir, extra_features,
                                                                                             string_null_values_list,
                                                                                             fill_missing_txt_method,
                                                                                             cols_to_skip)
            cm_l.print_formatted_txt("Format extra feature column names")
            max_len = max(len(lst) for lst in self.project.extra_features_col_nms.values())
            # make all lists the same length by appending None to shorter lists
            test = {k: v + [None] * (max_len - len(v)) for k, v in self.project.extra_features_col_nms.items()}
            self.project.extra_features_col_nms = df_extra_ftr_col_nms = pd.DataFrame(test)
            display(self.project.extra_features_col_nms.head())

            # make sure we didn't accidentally create duplicate pipes
            cm_l.print_formatted_txt("Drop duplicate pipes if there are any after adding extra features")
            pipes, dropped_pipes_df = geom_l.drop_duplicate_pipe_ids_with_the_same_location(pipes, dropped_pipes_df)

        cm_l.print_formatted_txt(f"Total dropped pipes: {len(dropped_pipes_df)}")
        temp_df = (dropped_pipes_df['DropReason'].value_counts(dropna=False).rename_axis("DropReason")
                   .reset_index(name="count"))
        if not temp_df.empty:
            display(temp_df)

        cm_l.print_formatted_txt("Keep fields according to specs")
        ld_cm_l.move_col_to_last(pipes, "geometry", inplace=True)
        final_pipes = pipes.copy()

        cm_l.print_formatted_txt("Perform some final sanity checks")
        display(final_pipes.head())
        print(final_pipes[final_pipes[install_yr_field] == 0])
        assert np.sum(final_pipes.install_yr == 0) == 0, "We have 0 value as installation year."
        assert len(final_pipes.pipe_id.unique()) == len(final_pipes), "No unique pipe ids."

        print("Format the field types")
        final_pipes = ld_cm_l.define_field_data_types(final_pipes, field_dtypes)

        cm_l.print_formatted_txt("Fix JSON characters in column names")
        final_pipes = ld_cm_l.fix_json_chars_in_column_names(final_pipes)

        cm_l.print_formatted_txt("Fix JSON characters in column values")
        exclude_cols = [pipe_id_field, 'pipe_len', diameter_field, install_yr_field, abandon_yr_field, material_field,
                        hsmaterial_field, address_field, verified_field, 'geometry']
        final_pipes = ld_cm_l.fix_json_chars_in_column_values(final_pipes, exclude_cols)

        if verbose:
            print(final_pipes.info())
            print(HTML(final_pipes.describe().T.to_html()))
            print(joined_ids.info())
            print(HTML(joined_ids.describe().T.to_html()))

        self.project.final_pipes = final_pipes
        self.project.joined_ids = joined_ids

        cm_l.print_formatted_txt("Write final data")
        cm_l.write_data(final_pipes, self.project.work_dir, self.project.all_pipes_flnm)
        cm_l.write_data(joined_ids, self.project.work_dir, self.project.joined_flnm)
        cm_l.write_data(dropped_pipes_df, self.project.work_dir, self.project.dropped_pipes_flnm + ".csv")
        if extra_features:
            cm_l.write_data(df_extra_ftr_col_nms, self.project.work_dir, self.project.extra_feature_flnm + ".csv")

        self.project.print_ended_message(self.__class__.__name__)
