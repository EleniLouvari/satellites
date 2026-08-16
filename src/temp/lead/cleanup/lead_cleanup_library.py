from import_libraries import *
import global_variables as gb_l
import libraries.common_libraries.generic_library as cm_l
import libraries.common_libraries.geom_library as geom_l
import libraries.common_libraries.cleanup_library as cm_cl_l


@staticmethod
def get_cols_from_df(pipe_id_field, file_path):
    return list(cm_l.read_data(file_path).drop(pipe_id_field, axis=1).columns)


def fill_and_standardize_materials(pipes, string_null_values_list, material_field="material"):
    """Fill missing material values with the value = UNKNOWN and standardize lead, galvanized, copper values.

    Keyword arguments:
    pipes -- the dataframe with the pipes
    string_null_values_list -- a list with the string values that are considered as missing values
    material_field -- the column name holding the material
    """
    df = pipes.copy()
    df[material_field] = df[material_field].replace(string_null_values_list, None)
    df.loc[df[material_field].isna(), material_field] = "UNKNOWN"
    # safety
    df[material_field].fillna("UNKNOWN", inplace=True)
    df[material_field] = df[material_field].astype(str)
    df[material_field] = df[material_field].str.strip().str.upper()

    # PB is the chemical abbreviation of lead
    lead_values = ['L', 'LD', 'LEAD', 'PB']
    galvanized_values = ['G', 'GAL', 'C/G', 'G/C', 'GAVANIZED', 'GI', 'GALV']
    copper_values = ['C', 'COP', 'COPER']
    df.loc[(df[material_field].isin(lead_values)), material_field] = "LEAD"
    df.loc[(df[material_field].isin(galvanized_values)), material_field] = "GALVANIZED"
    df.loc[(df[material_field].str.contains("GALVAN", na=False)), material_field] = "GALVANIZED"
    df.loc[(df[material_field].isin(copper_values)), material_field] = "COPPER"
    assert df[material_field].isna().sum() == 0, "We have null values in the material field!"
    return df


def fix_install_yr_issues(df_active, df_abandoned, ban_year, pb_prefix="pb", pr_prefix="pr"):
    df_active = df_active.copy()
    df_abandoned = df_abandoned.copy()

    df_active['pipe_id'] = df_active['pipe_id'].str.replace(pb_prefix, "", regex=True)
    df_active['pipe_id'] = df_active['pipe_id'].str.replace(pr_prefix, "", regex=True)
    df_abandoned['pipe_id'] = df_abandoned['pipe_id'].str.replace(pb_prefix, "", regex=True)
    df_abandoned['pipe_id'] = df_abandoned['pipe_id'].str.replace(pr_prefix, "", regex=True)

    df_abandoned['pipe_id'] = df_abandoned['pipe_id'].str.replace("abd_", "", regex=True)
    df_abandoned['pipe_id'] = df_abandoned['pipe_id'].str.replace("_abd", "", regex=True)
    df_abandoned['pipe_id'] = df_abandoned['pipe_id'].str.replace("abd", "", regex=True)

    df_abandoned.loc[df_abandoned['material'].str.upper() == "UNKNOWN", 'material'] = "OTHER"
    cond = (df_active['pipe_id'].isin(list(df_abandoned['pipe_id']))) & (df_active['material'].str.upper() == "UNKNOWN")
    df_active.loc[cond, 'material'] = "OTHER"

    cond = (df_abandoned['install_yr'] == df_abandoned['abandon_yr']) & (df_abandoned['install_yr'] < ban_year)
    df_abandoned.loc[cond, 'abandon_yr'] = 0
    cond = (df_abandoned['install_yr'] == df_abandoned['abandon_yr']) & (df_abandoned['install_yr'] >= ban_year)
    df_abandoned.loc[cond, 'install_yr'] = 0

    tmp_abd = df_abandoned[['pipe_id', 'install_yr', 'material', 'abandon_yr']].copy()
    tmp_abd.rename(columns={'install_yr': 'abd_install_yr', 'abandon_yr': 'abd_abandon_yr', 'material': 'abd_material'},
                   inplace=True)

    df_active = pd.merge(df_active, tmp_abd, how="left", on="pipe_id")
    df_active['abd_install_yr'].fillna(0, inplace=True)
    df_active['abd_abandon_yr'].fillna(0, inplace=True)
    df_active['abd_install_yr'] = df_active['abd_install_yr'].astype(int)
    df_active['abd_abandon_yr'] = df_active['abd_abandon_yr'].astype(int)

    cond = (df_active['install_yr'] <= df_active['abd_install_yr']) & (df_active['abd_abandon_yr'] > 0)
    df_active.loc[cond, 'install_yr'] = df_active.loc[cond]['abd_abandon_yr']

    cond = (df_active['install_yr'] <= df_active['abd_install_yr']) & (df_active['abd_abandon_yr'] == 0)
    cond = cond & (df_active['abd_material'].str.upper() == "LEAD") & (df_active['abd_install_yr'] < ban_year) & (
            df_active['abd_install_yr'] > 0)
    df_active.loc[cond, 'install_yr'] = df_active.loc[cond]['install_yr'] + 20

    cond = (df_active['install_yr'] <= df_active['abd_install_yr']) & (df_active['abd_abandon_yr'] == 0)
    cond = cond & (df_active['abd_material'].str.upper() == "LEAD") & (df_active['abd_install_yr'] >= ban_year)
    issues_ids = list(df_active[cond]['pipe_id'])
    if len(issues_ids) > 0:
        df_abandoned.loc[df_abandoned['pipe_id'].isin(issues_ids), 'install_yr'] = 0

    df_active.drop(columns=['abd_install_yr', 'abd_abandon_yr', 'abd_material'])

    return df_active, df_abandoned


def simplify_pipe_id(df, col_id="pipe_id", pb_prefix="pb", pr_prefix="pr", replace_pbpr=False):
    """The main purpose of this function is to remove the 'abd' keyword from the pipe_id. It is used when we want to
    match the active with the abandoned pipes using the pipe_id.

    Keyword arguments:
    df -- the dataframe with the pipes
    col_id -- the column name holding the pipe_id
    pb_prefix -- the prefix used for the public side pipes
    pr_prefix -- the prefix used for the private side pipes
    replace_pbpr -- a boolean variable: if True then it also replaces the <pb_prefix> and <pr_prefix>
    :return:
    the <df> with updated the column <col_id>
    """
    df = df.copy()
    df[col_id] = df[col_id].str.strip()
    if replace_pbpr:
        df[col_id] = df[col_id].str.replace(pb_prefix, "", regex=True)
        df[col_id] = df[col_id].str.replace(pr_prefix, "", regex=True)
    df[col_id] = df[col_id].str.replace("abd_", "", regex=True)
    df[col_id] = df[col_id].str.replace("_abd", "", regex=True)
    df[col_id] = df[col_id].str.replace("abd", "", regex=True)
    df[col_id] = df[col_id].apply(lambda x: x[1:] if x[0] == "_" else x)
    return df


def join_private_public_by_dist(df_public, df_private, field_mapping, max_distance=1):
    pipe_id_field = field_mapping['pipe_id_field']
    abandoned_field = field_mapping['abandoned_field']
    ins_yr_field = field_mapping['install_yr_field']
    pb_prefix = field_mapping['pb_prefix']
    pr_prefix = field_mapping['pr_prefix']

    needed_fields = [pipe_id_field, ins_yr_field, abandoned_field, 'geometry']
    cm_l.check_needed_df_columns(df_public, needed_fields)
    cm_l.check_needed_df_columns(df_private, needed_fields)

    df_public = df_public.copy()
    df_private = df_private.copy()
    df_public.rename(columns={pipe_id_field: 'pbid', abandoned_field: 'pbabandoned', ins_yr_field: 'pbinsyr'},
                     inplace=True)
    df_private.rename(columns={pipe_id_field: 'prid', abandoned_field: 'prabandoned', ins_yr_field: 'prinsyr'},
                      inplace=True)

    df_public['pb_cid'] = df_public['pbid']
    df_private['pr_cid'] = df_private['prid']
    df_public = simplify_pipe_id(df_public, "pb_cid", pb_prefix, pr_prefix, replace_pbpr=True)
    df_private = simplify_pipe_id(df_private, "pr_cid", pb_prefix, pr_prefix, replace_pbpr=True)

    tmp_pr = gpd.sjoin_nearest(df_private[['prid', 'pr_cid', 'prabandoned', 'geometry']],
                               df_public[['pbid', 'pb_cid', 'pbabandoned', 'pbinsyr', 'geometry']], how="left",
                               distance_col="pipe_dist", max_distance=max_distance)
    tmp_pr.sort_values(by=['pr_cid', 'prabandoned', 'pipe_dist', 'pbabandoned', 'pbinsyr'],
                       ascending=[True, True, True, True, False], inplace=True)
    tmp_pr.drop_duplicates(subset=['pr_cid'], inplace=True)
    tmp_pb = pd.merge(df_public, tmp_pr, how="left", on="pbid")
    tmp_pb = tmp_pb[tmp_pb['prid'].isna()]
    df_merge = pd.concat([tmp_pr[['pbid', 'prid']], tmp_pb[['pbid', 'prid']]], ignore_index=True)

    df_merge['pbid'].fillna("UNKNOWN", inplace=True)
    df_merge['prid'].fillna("UNKNOWN", inplace=True)
    return df_merge


def check_for_public_private_pipes(pub_pipes, pr_pipes, act_list):
    missing_dfs = []
    system_type_map = {"public": pub_pipes, "private": pr_pipes}

    for pipe_type, pipe_df in system_type_map.items():
        if not isinstance(pipe_df, pd.DataFrame) or pipe_df.empty:
            missing_dfs.append(pipe_type)
            system_type_map[pipe_type] = gpd.GeoDataFrame(columns=act_list, geometry="geometry", crs=4326)
            print(f"{pipe_type.capitalize()} pipes are missing")
        else:
            print(f"{pipe_type.capitalize()} pipes exist")

    if len(missing_dfs) == 2:
        raise Exception("Both public and private pipes are missing. Cannot continue.")

    return system_type_map['public'], system_type_map['private']


def check_for_abandoned_pipes(df, abd_df, abd_list, df_name):
    try:
        if len(abd_df) == 0:
            print(f"Abandoned {df_name} pipes dataframe exist but it's empty. Creating a fake entry.")
            abd_df = gpd.GeoDataFrame(columns=abd_list, geometry="geometry", crs=df.crs)
        else:
            print(f"Abandoned {df_name} pipes dataframe exists")
    except:
        print(f"Abandoned {df_name} pipes dataframe is not defined. Creating a fake entry.")
        abd_df = gpd.GeoDataFrame(columns=abd_list, geometry="geometry", crs=df.crs)
    return abd_df


def check_field_type_is_boolean(df, fillna_val=False, field_name="active"):
    df[field_name] = df[field_name].fillna(fillna_val)

    if is_bool_dtype(df[field_name]):
        return df
    elif is_numeric_dtype(df[field_name]):
        if "int" in str(df[field_name].dtype):
            cm_l.print_formatted_txt(
                f"Field type {str(df[field_name].dtype)} is converted to boolean for {field_name}!", "WARNING")
            df.loc[df[field_name] == 0, field_name] = False
            df.loc[df[field_name] == 1, field_name] = True
        else:
            raise Exception(f"Column type {str(df[field_name].dtype)} is not accepted for {field_name}!")
    elif is_string_dtype(df[field_name]):
        valid_values = ['0', '1', 'FALSE', 'TRUE']
        assert len(
            df[~(df[field_name].str.upper().isin(valid_values))]) == 0, f"Invalid values in the column {field_name}!"
        cm_l.print_formatted_txt(f"Field type string {field_name} is converted to boolean!", "WARNING")
        df.loc[df[field_name].str.upper() == "FALSE", field_name] = False
        df.loc[df[field_name].str.upper() == "TRUE", field_name] = True
        df.loc[df[field_name] == "0", field_name] = False
        df.loc[df[field_name] == "1", field_name] = True

    df[field_name] = df[field_name].astype(bool)
    return df


def initialize_abandoned_fields(df_samples, field_mapping):
    """Checks the following fields in the <df_samples> dataframe:
    1. abandoned: if it does not exist, then add it to the dataframe and set the value to 0
                  if it exists, then fills it with 0 and convert it to integer
    2. abandon_yr: if it does not exist, then add it to the dataframe and set the value to 0
    3. abandon_mat: if it does not exist, then add it to the dataframe and set the value to "UNKNOWN"

    Keyword arguments:
    df_samples -- the dataframe with the sample inspection results
    field_mapping -- the dictionary with the mapping of the field names
    :return:
    the <df_samples> dataframe with checked the abandoned related fields
    """

    df_samples = df_samples.copy()
    abandoned_field = field_mapping['abandoned_field']
    abandon_yr_field = field_mapping['abandon_yr_field']
    abandon_mat_field = field_mapping['abandon_mat_field']
    if abandoned_field not in list(df_samples.columns):
        df_samples[abandoned_field] = 0
    else:
        df_samples = cm_cl_l.numeric_abandoned_field(df_samples, 0, abandoned_field)

    if abandon_yr_field not in list(df_samples.columns):
        df_samples[abandon_yr_field] = 0

    if abandon_mat_field not in list(df_samples.columns):
        df_samples[abandon_mat_field] = "UNKNOWN"
    return df_samples


def create_abd_pipes_and_check_field_columns(pipes, abd_pipes, field_mapping, abd_list):
    """Create abandoned and abandon_yr fields in pipes and abd_pipes dataframes.
    pipes -- the pipes dataframe
    abd_pipes -- the abandon pipes dataframe
    """
    pipes = pipes.copy()
    abd_pipes = abd_pipes.copy()

    if field_mapping['abandoned_field'] not in pipes.columns:
        pipes.insert(len(pipes.columns), field_mapping['abandoned_field'], 0)
    if field_mapping['abandon_yr_field'] not in pipes.columns:
        pipes.insert(len(pipes.columns), field_mapping['abandon_yr_field'], 0)

    pipes = cm_cl_l.numeric_abandoned_field(pipes, 0, field_mapping['abandoned_field'])
    pipes = check_field_type_is_boolean(pipes, fillna_val=True, field_name=field_mapping['active_field'])
    pipes = check_field_type_is_boolean(pipes, field_name=field_mapping['verified_field'])

    if len(abd_pipes) > 0:
        if field_mapping['abandoned_field'] not in abd_pipes.columns:
            abd_pipes.insert(len(abd_pipes.columns), field_mapping['abandoned_field'], 1)
        if field_mapping['active_field'] not in abd_pipes.columns:
            abd_pipes.insert(len(abd_pipes.columns), field_mapping['active_field'], False)
        if field_mapping['vrsource_field'] not in abd_pipes.columns:
            abd_pipes.insert(len(abd_pipes.columns), field_mapping['vrsource_field'], VerificationSource.ABANDONED.name)
        if field_mapping['verified_field'] not in abd_pipes.columns:
            abd_pipes.insert(len(abd_pipes.columns), field_mapping['verified_field'], True)

        abd_pipes = cm_cl_l.numeric_abandoned_field(abd_pipes, 1, field_mapping['abandoned_field'])
        abd_pipes = check_field_type_is_boolean(abd_pipes, field_name=field_mapping['active_field'])
        abd_pipes = check_field_type_is_boolean(abd_pipes, fillna_val=True,
                                                field_name=field_mapping['verified_field'])
    else:
        new_field_list = abd_list + [field_mapping['abandoned_field'], field_mapping['active_field'],
                                     field_mapping['vrsource_field'], field_mapping['verified_field']]
        abd_pipes = gpd.GeoDataFrame(columns=new_field_list, geometry="geometry", crs=pipes.crs)
    return pipes, abd_pipes


def check_field_values_active(pipes, field_mapping, null_values_list):
    pipes = pipes.copy()

    verified_field = field_mapping['verified_field']
    abandoned_field = field_mapping['abandoned_field']
    active_field = field_mapping['active_field']
    vrsource_field = field_mapping['vrsource_field']
    material_field = field_mapping['material_field']

    tmp = pipes[(pipes[abandoned_field] != 0)]
    if len(tmp):
        print(f"We have {len(tmp)} not accepted values in the {abandoned_field} of the active pipes.")
        assert len(tmp) == 0, "We cannot handle these field values from active pipes."

    pipes[active_field].fillna(True, inplace=True)

    pipes[vrsource_field].fillna(VerificationSource.UNKNOWN.name, inplace=True)
    pipes[vrsource_field] = pipes[vrsource_field].astype(str).str.upper()
    cond = (pipes[vrsource_field] == "") | (pipes[vrsource_field].isin(null_values_list))
    tmp = pipes[cond]
    if len(tmp):
        print(f"We have {len(tmp)} null {vrsource_field} values in the active pipes. They will be set to UNKNOWN.")
        pipes.loc[cond, vrsource_field] = VerificationSource.UNKNOWN.name

    cond = (pipes[verified_field]) & (pipes[vrsource_field].isin(null_values_list))
    tmp = pipes[cond]
    if len(tmp):
        print(
            f"We have {len(tmp)} verified active pipes with null values in the column {vrsource_field}. They will be "
            f"set to CUSTOMER.")
        pipes.loc[cond, vrsource_field] = VerificationSource.CUSTOMER_REQUEST.name

    cond = (~pipes[verified_field]) & (~pipes[vrsource_field].isin(null_values_list))
    tmp = pipes[cond]
    if len(tmp):
        print(f"We have {len(tmp)} non verified active pipes with values in the column {vrsource_field}.")
        assert len(tmp) == 0, "We cannot handle these field values from active pipes."

    cond = (pipes[material_field].str.upper().str.contains("UNKNOWN")) & (pipes[verified_field])
    tmp = pipes[cond]
    if len(tmp):
        print(f"We have {len(tmp)} verified with UNKNOWN {material_field} in the active pipes.")
        assert len(tmp) == 0, "We cannot handle these field values from active pipes."

    return pipes


def check_field_values_abandoned(abd_pipes, field_mapping, null_values_list):
    abd_pipes = abd_pipes.copy()

    verified_field = field_mapping['verified_field']
    abandoned_field = field_mapping['abandoned_field']
    active_field = field_mapping['active_field']
    vrsource_field = field_mapping['vrsource_field']

    tmp = abd_pipes[abd_pipes[abandoned_field] != 1]
    if len(tmp):
        print(f"We have {len(tmp)} not accepted values in the {abandoned_field} of the abandoned pipes.")
        assert len(tmp) == 0, "We cannot handle these field values from abandoned pipes."

    abd_pipes[active_field].fillna(False, inplace=True)
    tmp = abd_pipes[abd_pipes[active_field]]
    if len(tmp):
        print(f"We have {len(tmp)} active in abandoned pipes.")
        assert len(tmp) == 0, "We cannot handle these field values from abandoned pipes."

    abd_pipes[verified_field].fillna(True, inplace=True)
    tmp = abd_pipes[~(abd_pipes[verified_field])]
    if len(tmp):
        print(f"We have {len(tmp)} unverified in abandoned pipes.")
        assert len(tmp) == 0, "We cannot handle these field values from abandoned pipes."

    abd_pipes[vrsource_field].fillna(VerificationSource.ABANDONED.name, inplace=True)
    abd_pipes[vrsource_field] = abd_pipes[vrsource_field].astype(str).str.upper()
    cond = (abd_pipes[vrsource_field] == "") | (abd_pipes[vrsource_field].isin(null_values_list))
    tmp = abd_pipes[cond]
    if len(tmp):
        print(f"We have {len(tmp)} null {vrsource_field} values in the abandoned pipes. They will be set to ABANDONED.")
        abd_pipes.loc[cond, vrsource_field] = VerificationSource.ABANDONED.name

    tmp = abd_pipes[abd_pipes[vrsource_field] != VerificationSource.ABANDONED.name]
    if len(tmp):
        print(f"We have {len(tmp)} invalid {vrsource_field} values.")
        assert len(tmp) == 0, "We cannot handle these field values from abandoned pipes."
    return abd_pipes


def check_common_active_abandoned(pipes, abd_pipes, field_mapping):
    """Checks the common active and abandoned pipes and sets the active pipes to verified=True,
    if the material is known.
    The logic behind this, following also EPA recommendations on handling maintenance data,
    is that if we have a pipe that is active and was replaced some time ago then it is field verified.
    If the material is unknown then we cannot set it as verified.

    Keyword arguments:
    pipes -- the dataframe with the active pipes
    abd_pipes -- the dataframe with the abandoned pipes
    field_mapping -- the dictionary with the mapping of the field names
    :return: the <pipes> dataframe with updated the verified field
    """

    pipes = pipes.copy()
    abd_pipes = abd_pipes.copy()
    pipe_id_field = field_mapping['pipe_id_field']
    verified_field = field_mapping['verified_field']
    material_field = field_mapping['material_field']
    vrsource_field = field_mapping['vrsource_field']
    pb_prefix = field_mapping['pb_prefix']
    pr_prefix = field_mapping['pr_prefix']

    pipes['tmp_id'] = pipes[pipe_id_field]
    abd_pipes['tmp_id'] = abd_pipes[pipe_id_field]
    pipes = simplify_pipe_id(pipes, "tmp_id", pb_prefix, pr_prefix, replace_pbpr=True)
    abd_pipes = simplify_pipe_id(abd_pipes, "tmp_id", pb_prefix, pr_prefix, replace_pbpr=True)
    common_pipes = pipes[
        (pipes['tmp_id'].isin(list(abd_pipes['tmp_id']))) & (pipes[material_field] != "UNKNOWN")].copy()
    if not common_pipes.empty:
        common_unverified_pipes = common_pipes[~common_pipes[verified_field]].copy()
        if not common_unverified_pipes.empty:
            print(f"We have {len(common_unverified_pipes)} active pipes existing in the abandoned. "
                  f"They will be set to verified")
            pipes.loc[(pipes['tmp_id'].isin(list(common_unverified_pipes['tmp_id']))), verified_field] = True
            pipes.loc[(pipes['tmp_id'].isin(list(common_unverified_pipes['tmp_id']))), vrsource_field] = \
                VerificationSource.ASSUMED_VERIFIED_FROM_ABD.name
    pipes.drop(columns=['tmp_id'], inplace=True)
    return pipes


def check_field_values(pipes, abd_pipes, field_mapping, null_values_list):
    pipes = check_field_values_active(pipes, field_mapping, null_values_list)
    if len(abd_pipes):
        abd_pipes = check_field_values_abandoned(abd_pipes, field_mapping, null_values_list)
        pipes = check_common_active_abandoned(pipes, abd_pipes, field_mapping)
    return pipes, abd_pipes


def format_pipe_id(public, private, abd_public, abd_private, joined_df, field_mapping):
    pipe_id_field = field_mapping['pipe_id_field']
    join_public_field = field_mapping['join_public_field']
    join_private_field = field_mapping['join_private_field']
    pb_prefix = field_mapping['pb_prefix']
    pr_prefix = field_mapping['pr_prefix']

    if len(public):
        public = public.copy()
        public = simplify_pipe_id(public, pipe_id_field, pb_prefix, pr_prefix, replace_pbpr=True)
        cond = (public[pipe_id_field].isna()) | (public[pipe_id_field] == "")
        public.loc[~cond, pipe_id_field] = pb_prefix + public[~cond][pipe_id_field]
    if len(abd_public):
        abd_public = abd_public.copy()
        abd_public = simplify_pipe_id(abd_public, pipe_id_field, pb_prefix, pr_prefix, replace_pbpr=True)
        cond = (abd_public[pipe_id_field].isna()) | (abd_public[pipe_id_field] == "")
        abd_public.loc[~cond, pipe_id_field] = pb_prefix + "_abd" + abd_public[~cond][pipe_id_field]

    if len(private):
        private = private.copy()
        private = simplify_pipe_id(private, pipe_id_field, pb_prefix, pr_prefix, replace_pbpr=True)
        cond = (private[pipe_id_field].isna()) | (private[pipe_id_field] == "")
        private.loc[~cond, pipe_id_field] = pr_prefix + private[~cond][pipe_id_field]
    if len(abd_private):
        abd_private = abd_private.copy()
        abd_private = simplify_pipe_id(abd_private, pipe_id_field, pb_prefix, pr_prefix, replace_pbpr=True)
        cond = (abd_private[pipe_id_field].isna()) | (abd_private[pipe_id_field] == "")
        abd_private.loc[~cond, pipe_id_field] = pr_prefix + "_abd" + abd_private[~cond][pipe_id_field]

    if len(joined_df):
        joined_df = joined_df.copy()
        joined_df = simplify_pipe_id(joined_df, join_public_field, pb_prefix, pr_prefix, replace_pbpr=True)
        cond = (joined_df[join_public_field].isna()) | (joined_df[join_public_field] == "")
        joined_df.loc[~cond, join_public_field] = pb_prefix + joined_df[~cond][join_public_field]

        joined_df = simplify_pipe_id(joined_df, join_private_field, pb_prefix, pr_prefix, replace_pbpr=True)
        cond = (joined_df[join_private_field].isna()) | (joined_df[join_private_field] == "")
        joined_df.loc[~cond, join_private_field] = pr_prefix + joined_df[~cond][join_private_field]

    return public, private, abd_public, abd_private, joined_df


def concatenate_public_private_pipes(df1, df2, raise_exception=True):
    """
    Concatenates two given dataframes. Raises exception if the raise_exception is True
    otherwise returns an empty dataframe.

    Keyword arguments:
    df1 -- first dataframe
    df2 -- second dataframe
    raise_exception -- whether to raise exception or not
    :return: the concatenated dataframes
    """
    if df1.empty and df2.empty:
        if raise_exception:
            raise Exception("\nBoth dataframes are empty!")
        else:
            return pd.DataFrame()
    dfs = []
    if len(df1) > 0:
        dfs.append(df1)
    if len(df2) > 0:
        dfs.append(df2)
    concatenated_df = pd.concat(dfs, ignore_index=True)

    return concatenated_df


def check_and_format_extra_dataframes(pipes, extra_dfs, min_install_yr, max_install_yr,
                                      ins_yr_field="install_yr", abandoned_field="abandoned"):
    formatted_dfs = {}
    for key in extra_dfs:
        print(f"\nExtracting installation year for {key}")
        df = extra_dfs[key].copy()
        message, cases = cm_cl_l.guess_and_set_date_format(df, ins_yr_field, "%Y")
        if len(message) != 0:
            print(message, cases)

        df[ins_yr_field] = df[ins_yr_field].apply(
            lambda x: np.NaN if x < min_install_yr or x > max_install_yr else x)
        assert len(df[df[ins_yr_field].isna()]) != len(df), f"All field values are missing in {key}. " \
                                                            f"We don't have any value to search. Please check again " \
                                                            f"the field: {ins_yr_field}"
        df[ins_yr_field] = df[ins_yr_field].fillna(0).astype(int)
        df = df[df[ins_yr_field] > 0]

        if len(df) == 0:
            print(f"All records in {key} have null installation year. It will be removed!")
        else:
            if abandoned_field not in list(df.columns):
                df[abandoned_field] = 0
            df[abandoned_field] = df[abandoned_field].fillna(0).astype(int)

            df = geom_l.return_valid_geometries(df)
            df = geom_l.explode_multigeometries(df)
            df = geom_l.convert_geometry_three_to_two_d(df)
            if df.crs != pipes.crs:
                df = df.to_crs(pipes.crs)
            formatted_dfs[key] = df

    return formatted_dfs


def check_verified_values(df, material_field, diameter_field, ins_yr_field, min_diameter, max_diameter,
                          diameter_threshold, min_install_year, ban_year):
    df = df.copy()
    df[ins_yr_field] = df[ins_yr_field].fillna(0).astype(int)
    df[diameter_field] = df[diameter_field].fillna(0).astype(float)
    cond_issue = (df[material_field] == "UNKNOWN") & (df[diameter_field] == 0) & (df[ins_yr_field] == 0)
    issues = df[cond_issue]
    if len(issues):
        print(100 * "-")
        print(f"There are {len(issues)} non verified pipes. They will be removed.")
        display(issues.head())
        df = df[~cond_issue].copy()

    currentYear = datetime.now().year
    cond_issue = (df[ins_yr_field] < min_install_year) | (df[ins_yr_field] > currentYear)
    issues = df[cond_issue]
    if len(issues) > 0:
        print(100 * "-")
        print(f"There are {len(issues)} pipes with invalid installation year. They will be set to 0.")
        display(issues.head())
        df.loc[cond_issue, ins_yr_field] = 0

    cond_issue = (df[diameter_field] < float(min_diameter)) | (df[diameter_field] > float(max_diameter))
    issues = df[cond_issue]
    if len(issues) > 0:
        print(100 * "-")
        print(f"There are {len(issues)} pipes with invalid diameter. They will be set to 0.")
        display(issues.head())
        df.loc[cond_issue, diameter_field] = 0

    cond_issue = (df[material_field] == "LEAD") & (df[diameter_field] >= diameter_threshold)
    issues = df[cond_issue]
    if len(issues):
        issues = issues.sort_values(by=diameter_field, ascending=False)
        cm_l.print_formatted_txt(f"There are {len(issues)} lead pipes in verified data with "
                                 f"diameter>={diameter_threshold} inches. They will be set to 0.", "ERROR")
        display(issues.head())
        df.loc[cond_issue, diameter_field] = 0

    cond_issue = (df[material_field] == "LEAD") & (df[ins_yr_field] >= ban_year)
    issues = df[cond_issue]
    if len(issues):
        issues = issues.sort_values(by=ins_yr_field, ascending=False)
        cm_l.print_formatted_txt(
            f"Warning: There are {len(issues)} lead pipes in verified data with install_yr>={ban_year}. "
            f"They will be set to 0.", "ERROR")
        display(issues.head())
        df.loc[cond_issue, ins_yr_field] = 0

    return df


def check_and_format_verified_data(df_verified, field_mapping, min_diameter, max_diameter, diameter_threshold,
                                   min_install_yr, max_install_yr, ban_year, null_values_list):
    """Checks the verified data after their field inspection, regarding the installation year, the material and
    the diameter values.

    Keyword arguments:
    df_verified -- the dataframe with the field inspected locations
    field_mapping -- the dictionary with the mapping of the field mappings
    min_diameter -- the min diameter threshold for diameter value to be considered as valid
    max_diameter -- the max diameter threshold for diameter value to be considered as valid
    diameter_threshold -- this is the EPA threshold value, above which the service line is considered as Unlikely lead
    min_install_yr -- the min installation year threshold for diameter value to be considered as valid
    max_install_yr -- the max installation year threshold for diameter value to be considered as valid
    ban_year -- the lead ban year
    null_values_list -- a list with all possible null values
    :return:
    the <verified_samples> dataframe after the formatting
    """
    material_field = field_mapping['material_field']
    diameter_field = field_mapping['diameter_field']
    ins_yr_field = field_mapping['install_yr_field']
    abandon_yr_field = field_mapping['abandon_yr_field']
    abandon_mat_field = field_mapping['abandon_mat_field']

    df_verified = df_verified.copy()
    df_verified = initialize_abandoned_fields(df_verified, field_mapping)

    message, cases = cm_cl_l.guess_and_set_date_format(df_verified, ins_yr_field, "%Y")
    if len(message) != 0:
        print(message, cases)

    message, cases = cm_cl_l.guess_and_set_date_format(df_verified, abandon_yr_field, "%Y")
    if len(message) != 0:
        print(message, cases)

    df_verified[ins_yr_field] = df_verified[ins_yr_field].apply(
        lambda x: np.NaN if x < min_install_yr or x > max_install_yr else x)
    df_verified[ins_yr_field] = df_verified[ins_yr_field].fillna(0).astype(int)
    df_verified[abandon_yr_field] = df_verified[abandon_yr_field].fillna(0).astype(int)
    df_verified[diameter_field] = df_verified[diameter_field].fillna(0).astype(float)

    df_verified = fill_and_standardize_materials(df_verified, null_values_list, material_field)
    df_verified = fill_and_standardize_materials(df_verified, null_values_list, abandon_mat_field)
    df_verified = check_verified_values(df_verified, material_field, diameter_field, ins_yr_field,
                                        min_diameter, max_diameter, diameter_threshold, min_install_yr, ban_year)
    return df_verified


def handle_no_line_geometries(pipes, col_pipe_id, pb_prefix, pr_prefix):
    pipes = pipes.copy()
    pb_pipes = pipes[pipes[col_pipe_id].str.startswith(pb_prefix)].copy()
    pr_pipes = pipes[pipes[col_pipe_id].str.startswith(pr_prefix)].copy()
    print("Public pipes")
    pb_pipes = geom_l.convert_geometries_to_lines(pb_pipes, max_length=-10)
    print("Private pipes")
    pr_pipes = geom_l.convert_geometries_to_lines(pr_pipes, max_length=5)
    pipes = pd.concat([pb_pipes, pr_pipes], ignore_index=True)
    pipes = pipes.reset_index(drop=True)
    return pipes


def fill_null_address(pipes, field_mapping, null_values_list):
    """Fills the address column using the geometry and reverse geocoding.
    :return: the pipes dataframe with filled the address_field
    """
    pipe_id_field = field_mapping['pipe_id_field']
    address_field = field_mapping['address_field']
    pb_prefix = field_mapping['pb_prefix']
    pr_prefix = field_mapping['pr_prefix']

    pipes = pipes.copy()
    pipes.reset_index(inplace=True, drop=True)
    pipes[field_mapping['address_field']] = pipes[field_mapping['address_field']].str.strip().str.upper()
    pipes.loc[(pipes[field_mapping['address_field']] == ""), field_mapping['address_field']] = None
    pipes.loc[(pipes[field_mapping['address_field']].isin(null_values_list)), field_mapping['address_field']] = None
    pipes[field_mapping['address_field']].fillna("UNKNOWN", inplace=True)
    public = pipes[pipes[pipe_id_field].str.lower().str.startswith(pb_prefix)].copy()
    private = pipes[pipes[pipe_id_field].str.lower().str.startswith(pr_prefix)].copy()

    # first the private and update the connected public for speed purposes
    tmp = private[(private[address_field] == "UNKNOWN")].copy()
    if len(tmp):
        del tmp[address_field]
        print(f"Found {len(tmp)} private pipes with Null or UNKNOWN address."
              f" They will be updated using reverse geocoding.")
        tmp = geom_l.reverse_geocode_address_arcgis(tmp, "tmp_address")
        private = pd.merge(private, tmp[[pipe_id_field, 'tmp_address']], how="left", on=pipe_id_field)
        cond = (private[address_field] == "UNKNOWN") & (~private['tmp_address'].isna()) & (
                private['tmp_address'] != "UNKNOWN")
        private.loc[cond, address_field] = private[cond]['tmp_address']

        private['pr_id'] = private[pipe_id_field]
        private['pr_address'] = private[address_field]
        public = geom_l.find_nearest_item(public, "pr_id", private, "pr_id", max_distance_in_feet=5,
                                           distance_field=None, item_to_point=False, drop_duplicate=True)
        public = pd.merge(public, private[['pr_id', 'pr_address']], how="left", on="pr_id")

        cond = (public[address_field] == "UNKNOWN") & (~public['pr_address'].isna()) & (
                public['pr_address'] != "UNKNOWN")
        public.loc[cond, address_field] = public[cond]['pr_address']
        public.drop(columns=['pr_id', 'pr_address'], inplace=True)
        private.drop(columns=['pr_id', 'pr_address', 'tmp_address'], inplace=True)
        pipes = pd.concat([public, private], ignore_index=True)
        pipes.reset_index(inplace=True, drop=True)

    # fill the rest
    tmp = pipes[(pipes[address_field] == "UNKNOWN")].copy()
    if len(tmp) > 0:
        del tmp[address_field]
        print(f"Found {len(tmp)} pipes with Null or UNKNOWN address. They will be updated using reverse geocoding.")
        tmp = geom_l.reverse_geocode_address_arcgis(tmp, "tmp_address")
        pipes = pd.merge(pipes, tmp[[pipe_id_field, 'tmp_address']], how="left", on=pipe_id_field)
        cond = (pipes[address_field] == "UNKNOWN")
        pipes.loc[cond, address_field] = pipes[cond]['tmp_address']
        del pipes['tmp_address']

    pipes[address_field] = pipes[address_field].str.strip().str.upper()
    return pipes


def find_common_install_yr_act_abd(pipes, field_mapping, min_install_year, ban_year):
    pipe_id_field, material_field, diameter_field = field_mapping['pipe_id_field'], field_mapping['material_field'], \
        field_mapping['diameter_field']
    install_yr_field, abandoned_field, abandon_yr_field = field_mapping['install_yr_field'], \
        field_mapping['abandoned_field'], field_mapping['abandon_yr_field']

    pb_prefix = field_mapping['pb_prefix']
    pr_prefix = field_mapping['pr_prefix']

    needed_fields = [pipe_id_field, material_field, diameter_field, install_yr_field, abandoned_field, abandon_yr_field,
                     'geometry']
    cm_l.check_needed_df_columns(pipes, needed_fields)

    pipes = pipes.copy()
    pipes.reset_index(inplace=True, drop=True)
    pipes['idx'] = pipes.index

    for analysis in [pb_prefix, pr_prefix]:
        system = "Public" if analysis == pb_prefix else "Private"
        tmp_active = pipes[(pipes[pipe_id_field].str.startswith(analysis)) & (pipes[abandoned_field] == 0)].copy()
        tmp_abandoned = pipes[(pipes[pipe_id_field].str.startswith(analysis)) & (pipes[abandoned_field] == 1)].copy()
        tmp_active['tmpid'], tmp_abandoned['tmpid'] = tmp_active[pipe_id_field], tmp_abandoned[pipe_id_field]

        tmp_active = simplify_pipe_id(tmp_active, "tmpid", pb_prefix, pr_prefix, replace_pbpr=True)
        tmp_abandoned = simplify_pipe_id(tmp_abandoned, "tmpid", pb_prefix, pr_prefix, replace_pbpr=True)

        tmp_active.rename(columns={'idx': 'act_idx', pipe_id_field: 'act_pipe_id', install_yr_field: 'act_install_yr',
                                   material_field: 'act_material', diameter_field: 'act_diameter'}, inplace=True)
        tmp_abandoned.rename(
            columns={'idx': 'abd_idx', pipe_id_field: 'abd_pipe_id', install_yr_field: 'abd_install_yr',
                     material_field: 'abd_material', diameter_field: 'abd_diameter'}, inplace=True)
        df_merge = pd.merge(
            tmp_active[['act_idx', 'tmpid', 'act_pipe_id', 'act_install_yr', 'act_material', 'act_diameter']],
            tmp_abandoned[['abd_idx', 'tmpid', 'abd_pipe_id', 'abd_install_yr', 'abd_material', 'abd_diameter',
                           abandon_yr_field]],
            how="inner", on="tmpid")

        df_merge['act_install_yr'].fillna(0, inplace=True)
        df_merge['act_install_yr'] = df_merge['act_install_yr'].astype(int)
        df_merge['abd_install_yr'].fillna(0, inplace=True)
        df_merge['abd_install_yr'] = df_merge['abd_install_yr'].astype(int)

        issues = df_merge[(df_merge['act_material'] == "UNKNOWN") & (df_merge['abd_material'] == "UNKNOWN")]
        if len(issues):
            cm_l.print_formatted_txt(f"There are {len(issues)} abandoned pipes, stated as UNKNOWN in both the active "
                                     f"and abandoned dataframes", "WARNING")
            display(issues)

        issues = df_merge[
            (df_merge['act_install_yr'] <= df_merge['abd_install_yr']) & (df_merge['abd_material'] == "LEAD") & (
                    df_merge['act_install_yr'] > 0)]
        if len(issues):
            display(issues)
            raise Exception("Issue with the install year value of abandoned lead pipes!")

        issues = df_merge[df_merge['act_install_yr'] <= df_merge['abd_install_yr']]
        if len(issues):
            cm_l.print_formatted_txt(f"There are {len(issues)} common {system} pipes between active and abandoned "
                                     f"having install year issues", "WARNING")
            display(issues.head(10))
            # TODO: check and discuss
            for idx in list(issues.index):
                row = issues.loc[idx]

                if row[abandon_yr_field] > 0:
                    if row['act_install_yr'] <= row['abd_install_yr']:
                        pipes.loc[row['act_idx'], install_yr_field] = row[abandon_yr_field]
                    if (row['abd_install_yr'] == 0) and (row['abd_material'] != "LEAD"):
                        pipes.loc[row['abd_idx'], install_yr_field] = max(min_install_year, row[abandon_yr_field] - 20)
                else:
                    if 0 < row['act_install_yr'] < row['abd_install_yr']:
                        pipes.loc[row['act_idx'], install_yr_field] = row['abd_install_yr']
                        pipes.loc[row['abd_idx'], install_yr_field] = row['act_install_yr']
                    elif (row['act_install_yr'] == row['abd_install_yr']) and (row['act_install_yr'] > 0) and (
                            row['abd_material'] != "LEAD"):
                        pipes.loc[row['abd_idx'], install_yr_field] = max(min_install_year, row['abd_install_yr'] - 20)
        else:
            cm_l.print_formatted_txt(f"There aren't any common {system} pipes between active and abandoned having "
                                     f"install year issues!", "RESULTS")

    del pipes['idx']
    return pipes, issues


def get_ban_year(pipes):
    """Find the lead ban year for the state where the pipes are (taken from the EPA document).
    Ask customer for the actual lead ban year.
    :return: the lead ban year
    """
    state, study_center = cm_l.get_geodataframe_US_state_and_center(pipes)

    lead_ban_dict = {'CT': '12/31/1988',
                     'ME': '08/01/1987',
                     'MA': '01/01/1986',
                     'NH': '08/01/1987',
                     'RI': '01/01/1987',
                     'VT': '12/28/1988',  # or 09/10/1982
                     'NJ': '02/02/1987',
                     'NY': '01/01/1986',
                     'PR': '06/19/1988',
                     'VI': '02/06/1989',
                     'DE': '06/17/1988',
                     'DC': '08/19/1988',
                     'MD': '08/16/1988',
                     'PA': '01/06/1991',
                     'VA': '04/01/1986',
                     'WV': '06/19/1988',
                     'AL': '05/10/1988',
                     'FL': '06/10/1988',
                     'GA': '07/01/1987',  # or 01/89
                     'KY': '01/04/1988',
                     'MS': '04/13/1988',  # or 05/88
                     'NC': '03/01/1987',
                     'SC': '11/01/1988',
                     'TN': '03/18/1988',
                     'IL': '04/01/1986',
                     'IN': '03/01/1987',
                     'MI': '06/07/1988',
                     'MN': '06/01/1985',
                     'OH': '09/12/1988',
                     'WI': '01/01/1986',
                     'AR': '06/01/1988',
                     'LA': '09/20/1988',
                     'NM': '04/01/1987',
                     'OK': '05/06/1987',
                     'TX': '07/01/1988',
                     'IA': '09/14/1988',
                     'KS': '04/19/1988',
                     'MO': '08/31/1988',
                     'NE': '05/01/1988',
                     'CO': '01/31/1988',
                     'MT': '12/31/1987',
                     'ND': '01/01/1988',
                     'SD': '09/03/1987',
                     'UT': '04/24/1989',
                     'AZ': '08/18/1987',
                     'CA': '07/01/1986',
                     'HI': '06/05/1987',
                     'NV': '01/01/1989',
                     'AS': '01/01/1989',
                     'MP': '03/13/1989',
                     'GU': '03/01/1988',
                     'AK': '06/05/1988',
                     'ID': '08/11/1988',
                     'OR': '09/04/1984',
                     'WA': '04/27/1987'}  # 03/17/88 , 03/30/89

    if state in lead_ban_dict:
        ban_date = lead_ban_dict[state]
        ban_year = pd.to_datetime(ban_date, format="%m/%d/%Y").year
        print(f"\nGeneral Lead ban date: {ban_date} for state {state}, according to EPA.")
        return ban_year
    else:
        print("\nNo lead ban date found for state: {state}")
        return None


def print_percentage_of_null_diameter(pipes, field_mapping, lead_run_type):
    """Prints the percentage of null diameter values."""
    pipe_id_field = field_mapping['pipe_id_field']
    diameter_field, material_field = field_mapping['diameter_field'], field_mapping['material_field']
    pb_prefix, pr_prefix = field_mapping['pb_prefix'], field_mapping['pr_prefix']

    if lead_run_type == LeadRunType.BOTH.name:
        perc_null = round(100 * len(pipes[pipes[diameter_field] == 0]) / len(pipes), 2)
        if perc_null > 10:
            cm_l.print_formatted_txt(f"The {perc_null}% of the pipes having null diameter values.", "WARNING")
    else:
        for prefix in [pb_prefix, pr_prefix]:
            system = "public" if prefix == pb_prefix else "private"
            cond = pipes[pipe_id_field].str.startswith(prefix)
            perc_null = round(100 * len(pipes[cond][pipes[diameter_field] == 0]) / len(pipes[cond]), 2)
            if perc_null > 10:
                cm_l.print_formatted_txt(f"The {perc_null}% of the {system} pipes having null diameter values.",
                                         "WARNING")


def fill_null_diameters(pipes, joined_ids, lead_run_type, field_mapping, remove_features, max_fill_value):
    """Fill the null diameter values based on the analysis type: BOTH or SEPARATE, taking into account the
    remove features.

    Keyword arguments:
    pipes -- the dataframe with the pipes
    joined_ids -- the dataframe containing the adjacent public_id - private_id
    lead_run_type -- analysis type: BOTH or SEPARATE
    field_mapping -- the dictionary with the field name mappings
    remove_features -- the list with the remove feature
    max_fill_value -- the maximum diameter threshold value to be used when filling the null values
    :return: the <pipes> dataframe with updated diameter values
    """
    if len(remove_features) == 0:
        return pipes
    pipes = pipes.copy()
    pipe_id_field = field_mapping['pipe_id_field']
    diameter_field, material_field = field_mapping['diameter_field'], field_mapping['material_field']
    pb_prefix, pr_prefix = field_mapping['pb_prefix'], field_mapping['pr_prefix']

    if (lead_run_type == LeadRunType.BOTH.name and diameter_field in remove_features[0]['both']) or \
            (lead_run_type == LeadRunType.SEPARATE.name and
             diameter_field in remove_features[0]['public'] and
             diameter_field in remove_features[0]['private']):
        cm_l.print_formatted_txt(f"{diameter_field} is included in remove_features. Filling is skipped.",
                                 "WARNING")
    else:
        pipes_filled = fill_diameter_from_adjacent(pipes, joined_ids, field_mapping)
        if (lead_run_type == LeadRunType.SEPARATE.name) and \
                (diameter_field in remove_features[0]['public'] or diameter_field in remove_features[0]['private']):
            if diameter_field in remove_features[0]['public']:
                cm_l.print_formatted_txt(f"{diameter_field} is included in public remove_features. "
                                         "Filling diameter only in private side.", "WARNING")
                pipes_filled = pipes_filled[pipes_filled[pipe_id_field].str.startswith(pr_prefix)].copy()
                pipes_filled = fill_nulls_per_material(pipes_filled, material_field, diameter_field, max_fill_value)
            elif diameter_field in remove_features[0]['private']:
                cm_l.print_formatted_txt(f"{diameter_field} is included in private remove_features. "
                                         "Filling diameter only in public side.", "WARNING")
                pipes_filled = pipes_filled[pipes_filled[pipe_id_field].str.startswith(pb_prefix)].copy()
                pipes_filled = fill_nulls_per_material(pipes_filled, material_field, diameter_field, max_fill_value)
            pipes_filled = pipes_filled[[pipe_id_field, diameter_field]].copy()
            pipes_filled[diameter_field].fillna(0, inplace=True)
            pipes_filled.rename(columns={diameter_field: 'filled_diameter'}, inplace=True)
            pipes = pd.merge(pipes, pipes_filled, how="left", on=pipe_id_field)
            cond = (pipes[diameter_field] == 0) & (pipes['filled_diameter'] > 0)
            pipes.loc[cond, diameter_field] = pipes[cond]['filled_diameter']
            pipes.drop(columns=['filled_diameter'], inplace=True)
        else:
            pipes = fill_nulls_per_material(pipes_filled, material_field, diameter_field, max_fill_value)
    return pipes


def fill_diameter_from_adjacent(pipes, joined_df, field_mapping):
    pipes = pipes.copy()
    count_pipes = len(pipes)
    pipe_id_field = field_mapping['pipe_id_field']
    diameter_field = field_mapping['diameter_field']
    material_field = field_mapping['material_field']
    join_public_field = field_mapping['join_public_field']
    join_private_field = field_mapping['join_private_field']
    pb_prefix = field_mapping['pb_prefix']
    pr_prefix = field_mapping['pr_prefix']

    cond = (pipes[diameter_field] == 0) & (pipes[material_field] != "LEAD")
    null_diams = pipes[cond][[pipe_id_field, diameter_field]].copy()
    col_diam_x = f"{diameter_field}_x"
    col_diam_y = f"{diameter_field}_y"
    col_id_x = f"{pipe_id_field}_x"
    col_diam_fill = f"{diameter_field}_fill"

    df_no_nulls = []
    public = null_diams[null_diams[pipe_id_field].str.startswith(pb_prefix)].copy()
    public = pd.merge(public, joined_df, left_on=pipe_id_field, right_on=join_public_field)
    public = pd.merge(public, pipes[[pipe_id_field, diameter_field]], left_on=join_private_field,
                      right_on=pipe_id_field)
    public[[col_diam_x, col_diam_y]].fillna(0, inplace=True)
    cond = (public[col_diam_x] == 0) & (public[col_diam_y] > 0)
    public.loc[cond, col_diam_x] = public[cond][col_diam_y]
    public = public[public[col_diam_x] > 0].copy()
    public = public[[col_id_x, col_diam_x]].rename(columns={col_id_x: pipe_id_field, col_diam_x: col_diam_fill})
    public = public.sort_values(by=[pipe_id_field, col_diam_fill]).drop_duplicates(subset=[pipe_id_field])
    df_no_nulls.append(public)

    private = null_diams[null_diams[pipe_id_field].str.startswith(pr_prefix)].copy()
    private = pd.merge(private, joined_df, left_on=pipe_id_field, right_on=join_private_field)
    private = pd.merge(private, pipes[[pipe_id_field, diameter_field]], left_on=join_public_field,
                       right_on=pipe_id_field)
    private[[col_diam_x, col_diam_y]].fillna(0, inplace=True)
    cond = (private[col_diam_x] == 0) & (private[col_diam_y] > 0)
    private.loc[cond, col_diam_x] = private[cond][col_diam_y]
    private = private[private[col_diam_x] > 0].copy()
    private = private[[col_id_x, col_diam_x]].rename(columns={col_id_x: pipe_id_field, col_diam_x: col_diam_fill})
    private = private.sort_values(by=[pipe_id_field, col_diam_fill]).drop_duplicates(subset=[pipe_id_field])
    df_no_nulls.append(private)

    df_no_nulls = pd.concat(df_no_nulls, ignore_index=True)
    if len(df_no_nulls):
        pipes = pd.merge(pipes, df_no_nulls, how="left", on=pipe_id_field)
        pipes[col_diam_fill].fillna(0, inplace=True)
        cond = (pipes[diameter_field] == 0) & (pipes[col_diam_fill] > 0)
        print(f"We will fill {len(pipes[cond])} null diameter values from the adjacent pipe!")
        display(pipes[cond][[pipe_id_field, material_field, diameter_field, col_diam_fill]].head())
        pipes.loc[cond, diameter_field] = pipes[cond][col_diam_fill]
        pipes.drop(columns=[col_diam_fill], inplace=True)

    assert len(pipes) == count_pipes, f"Something went wrong filling the column: {diameter_field}"
    return pipes


def fill_nulls_per_material(pipes, material_field="material", col_to_fill="pipe_diam", exclude_zeros=True,
                            fill_method="median", max_fill_value=2):
    needed_fields = [material_field, col_to_fill]
    cm_l.check_needed_df_columns(pipes, needed_fields)

    pipes = pipes.copy()
    count_pipes = len(pipes)
    if col_to_fill == "geometry":
        raise Exception("Geometry column cannot be filled!")

    if fill_method not in ['median', 'mode']:
        raise Exception(f"fill method: {fill_method} for column: {col_to_fill} is not accepted. "
                        f"Use only mode or median!")

    if exclude_zeros and str(pipes[col_to_fill].dtype) == "object":
        print(f"Found variable <exclude_zeros>=True but the column type is {pipes[col_to_fill].dtype}. "
              f"Set <exclude_zeros> to False.")
        exclude_zeros = False

    def find_nearest_value(df, col_to_search, col_value):
        pipes_col = df[col_to_search].to_numpy()
        min_index = np.argmin(np.abs(pipes_col - col_value))
        return pipes_col[min_index]

    cond = (~pipes[col_to_fill].isna())
    if exclude_zeros:
        cond = cond & (pipes[col_to_fill] > 0)
    if fill_method == "median":
        total_fill_value = pipes[cond][col_to_fill].median()
        # get the nearest existing value
        total_fill_value = find_nearest_value(pipes[cond], col_to_fill, total_fill_value)
    else:
        total_fill_value = pipes[cond][col_to_fill].mode()[0]

    # set a maximum value in order to avoid filling with large values
    if total_fill_value > max_fill_value:
        cm_l.print_formatted_txt(f"Maximum diameter {fill_method} value found: {total_fill_value}. "
                                 f"It will be set to {max_fill_value}.", "WARNING")
        total_fill_value = max_fill_value

    lst_materials = [material for material in list(pipes[material_field].unique()) if "UNKNOWN" not in material.upper()]
    for material in lst_materials:
        cond_material = (pipes[material_field] == material)
        cond_null = (pipes[col_to_fill].isna())
        if exclude_zeros:
            cond_null = (cond_null | (pipes[col_to_fill] == 0))

        df_null = pipes[cond_material & cond_null]
        if len(df_null) > 0:
            df_not_null = pipes[cond_material & ~cond_null]
            if len(df_not_null) > 10:
                if fill_method == "median":
                    fill_value = df_not_null[col_to_fill].median()
                    fill_value = find_nearest_value(df_not_null, col_to_fill, fill_value)
                else:
                    fill_value = df_not_null[col_to_fill].mode()[0]
                cond = (cond_material & cond_null)
                pipes.loc[cond, col_to_fill] = fill_value
                print(f"Update {len(pipes[cond])} null {col_to_fill} with {fill_value}, having material: {material}.")
            else:
                print(f"All {col_to_fill} values in {material} are null.")

    cond = (pipes[col_to_fill].isna())
    if exclude_zeros:
        cond = cond | (pipes[col_to_fill] == 0)
    if len(pipes[cond]) > 0:
        print(f"Left {len(pipes[cond])} total null values in {col_to_fill}. They will be filled with {fill_method}: "
              f"{total_fill_value}.")
        pipes.loc[cond, col_to_fill] = total_fill_value

    assert len(pipes) == count_pipes, f"Something went wrong filling the column: {col_to_fill}"
    return pipes


def fill_install_yr_of_service_lines_from_nearest_item(pipes, df, ban_year, install_yr_field="install_yr",
                                                       material_field="material", max_distance=500):
    """Helper function for <fill_install_yr_of_service_lines_from_extra_source>."""
    pipes = pipes.copy()
    pipes.reset_index(inplace=True, drop=True)
    pipes['pid'] = "p" + pipes.index.astype(str)

    tmp_df = df[df[install_yr_field] > 0][[install_yr_field, 'geometry']].copy()
    tmp_df.reset_index(inplace=True, drop=True)
    tmp_df.rename(columns={install_yr_field: 'tmp_install_yr'}, inplace=True)
    tmp_df['tmp_id'] = "tmp" + tmp_df.index.astype(str)

    install_null = 9999
    pipes = geom_l.find_nearest_item(pipes, "tmp_id", tmp_df, "tmp_id",
                                      max_distance_in_feet=max_distance, drop_duplicate=False)
    pipes = pd.merge(pipes, tmp_df[['tmp_id', 'tmp_install_yr']], how="left", on="tmp_id")
    pipes['tmp_install_yr'].fillna(install_null, inplace=True)
    pipes['tmp_install_yr'] = pipes['tmp_install_yr'].astype(int)
    pipes = pipes.sort_values(by="tmp_install_yr", ascending=True).drop_duplicates(subset=['pid'], keep="first")

    cond = (pipes[material_field] != "LEAD") & (pipes[install_yr_field] == 0)
    cond = cond & (pipes['tmp_install_yr'] > 0) & (pipes['tmp_install_yr'] != install_null)
    pipes.loc[cond, install_yr_field] = pipes[cond]['tmp_install_yr']

    cond = (pipes[material_field] == "LEAD") & (pipes[install_yr_field] == 0)
    cond = cond & (pipes['tmp_install_yr'] > 0) & (pipes['tmp_install_yr'] < ban_year) & (
            pipes['tmp_install_yr'] != install_null)
    pipes.loc[cond, install_yr_field] = pipes[cond]['tmp_install_yr']

    pipes.drop(columns=['pid', 'tmp_id', 'tmp_install_yr'], inplace=True)
    return pipes


def fill_install_yr_of_service_lines_from_extra_source(pipes, extra_dfs, field_mapping, ban_year,
                                                       max_distance=500):
    """ Fills the installation year of pipes using the extra_dfs."""
    pipe_id_field = field_mapping['pipe_id_field']
    install_yr_field = field_mapping['install_yr_field']
    material_field = field_mapping['material_field']
    abandoned_field = field_mapping['abandoned_field']
    pb_prefix = field_mapping['pb_prefix']
    pr_prefix = field_mapping['pr_prefix']

    public = pipes[pipes[pipe_id_field].str.startswith(pb_prefix)][
        [pipe_id_field, material_field, install_yr_field, abandoned_field, 'geometry']].copy()
    private = pipes[pipes[pipe_id_field].str.startswith(pr_prefix)][
        [pipe_id_field, material_field, install_yr_field, abandoned_field, 'geometry']].copy()
    filled_pipes = []
    for key in extra_dfs:
        df = extra_dfs[key].copy()
        if len(private) > 0:
            if "parcel" in key.lower():
                print(f"\nFilling {install_yr_field} for Private pipes from {key}")
                private = fill_install_yr_of_service_lines_from_nearest_item(private, df, ban_year, install_yr_field,
                                                                             material_field, max_distance)
                private = private[private[install_yr_field] > 0][[pipe_id_field, install_yr_field]].copy()
                filled_pipes.append(private)

        if len(public) > 0:
            if "main" in key.lower():
                for pipe_type in [1, 0]:
                    pipe_str = "Abandoned" if pipe_type == 1 else "Active"
                    print(f"\nFilling {install_yr_field} for {pipe_str} Public pipes from {key}")
                    tmp_main = df[df[abandoned_field] == pipe_type].copy()
                    if len(tmp_main) > 0:
                        tmp_pb = public[public[abandoned_field] == pipe_type].copy()
                        tmp_pb = fill_install_yr_of_service_lines_from_nearest_item(tmp_pb, tmp_main, ban_year,
                                                                                    install_yr_field, material_field,
                                                                                    max_distance)
                        tmp_pb = tmp_pb[tmp_pb[install_yr_field] > 0][[pipe_id_field, install_yr_field]].copy()
                        filled_pipes.append(tmp_pb)

    if len(filled_pipes) > 0:
        filled_pipes = pd.concat(filled_pipes, ignore_index=True)
        filled_pipes = filled_pipes.sort_values(by=[pipe_id_field, install_yr_field], ascending=True).drop_duplicates(
            subset=pipe_id_field)
        filled_pipes.rename(columns={install_yr_field: 'tmp_install_yr'}, inplace=True)

        pipes = pd.merge(pipes, filled_pipes[[pipe_id_field, 'tmp_install_yr']], how="left", on=pipe_id_field)
        pipes['tmp_install_yr'] = pipes['tmp_install_yr'].fillna(0)
        pipes['tmp_install_yr'] = pipes['tmp_install_yr'].astype(int)
        cond = (pipes[install_yr_field] == 0) & (pipes['tmp_install_yr'] > 0)
        print(f"\nFilled {install_yr_field} of {len(pipes[cond])} pipes from {list(extra_dfs.keys())}")
        pipes.loc[cond, install_yr_field] = pipes[cond]['tmp_install_yr']
        del pipes['tmp_install_yr']
    return pipes


def fill_null_install_yr_of_lead_service_lines_with_median(pipes, install_yr_field="install_yr",
                                                           material_field="material"):
    """Helper function for <lead_fill_install_yr_for_service_lines>."""
    pipes = pipes.copy()
    cond_null = (pipes[material_field] == "LEAD") & (pipes[install_yr_field] == 0)
    left_null_lead = len(pipes[cond_null])
    if left_null_lead > 0:
        no_null_lead = pipes[(pipes[material_field] == "LEAD") & (pipes[install_yr_field] > 0)]
        if len(no_null_lead) > 0:
            median_value = round(no_null_lead[install_yr_field].median())
            print(f"\nThere are {left_null_lead} remaining null {install_yr_field} Lead pipes.")
            print(f"They will be filled with median value: {median_value}.")
            pipes.loc[cond_null, install_yr_field] = median_value
        else:
            raise Exception("All lead pipes have null installation year. Please check.")
    return pipes


def fill_null_install_yr_of_service_lines_from_neighborhood(pipes, min_value, max_value, install_yr_field="install_yr",
                                                            abandoned_field="abandoned"):
    """Helper function for <lead_fill_install_yr_for_service_lines>."""
    pipes = pipes.copy()

    left_null = len(pipes[(pipes[install_yr_field] == 0)])
    if left_null > 0:
        print(f"\nFilling the remaining {left_null} null values of {install_yr_field} from neighborhood.")
        pipes.reset_index(inplace=True, drop=True)
        pipes['pid'] = "p" + pipes.index.astype(str)
        keep_cols = ['pid', install_yr_field, 'geometry']
        active_pipes = pipes[pipes[abandoned_field] == 0][keep_cols].copy()
        abd_pipes = pipes[pipes[abandoned_field] == 1][keep_cols].copy()
        print("Active pipes:")
        active_pipes = cm_gn_l.fill_out_of_bound_or_null_column_val_with_median(active_pipes, install_yr_field,
                                                                                min_value, max_value)
        cm_gn_l.fill_null_values_with_median(active_pipes, install_yr_field)

        if len(abd_pipes) > 0:
            print("Abandoned pipes:")
            abd_pipes = cm_gn_l.fill_out_of_bound_or_null_column_val_with_median(abd_pipes, install_yr_field, min_value,
                                                                                 max_value)
            cm_gn_l.fill_null_values_with_median(abd_pipes, install_yr_field)
            tmp_pipes = pd.concat([active_pipes, abd_pipes], ignore_index=True)
        else:
            tmp_pipes = active_pipes.copy()

        tmp_pipes.rename(columns={install_yr_field: 'tmp_install_yr'}, inplace=True)
        pipes = pd.merge(pipes, tmp_pipes[['pid', 'tmp_install_yr']], how="left", on="pid")
        pipes['tmp_install_yr'].fillna(0, inplace=True)
        pipes['tmp_install_yr'] = pipes['tmp_install_yr'].astype(int)
        cond = (pipes[install_yr_field] == 0) & (pipes['tmp_install_yr'] > 0)
        pipes.loc[cond, install_yr_field] = pipes[cond]['tmp_install_yr']
        pipes.drop(columns=['pid', 'tmp_install_yr'], inplace=True)

    return pipes


def fill_install_yr_for_service_lines(pipes, extra_dfs, field_mapping, min_value, max_value, ban_year,
                                      max_distance=500):
    pipe_id_field = field_mapping['pipe_id_field']
    install_yr_field = field_mapping['install_yr_field']
    material_field = field_mapping['material_field']
    abandoned_field = field_mapping['abandoned_field']
    cm_l.check_needed_df_columns(pipes, [pipe_id_field, install_yr_field, material_field, abandoned_field])

    pipes = pipes.copy()
    count_pipes = len(pipes)
    pipes[install_yr_field] = pipes[install_yr_field].apply(lambda x: np.NaN if x < min_value or x > max_value else x)
    assert len(pipes[pipes[
        install_yr_field].isna()]) != count_pipes, f"All field values are missing. We don't have any value " \
                                                   f"to search. Please check again the field: {install_yr_field}"
    pipes[install_yr_field] = pipes[install_yr_field].fillna(0).astype(int)
    if extra_dfs:
        pipes = fill_install_yr_of_service_lines_from_extra_source(pipes, extra_dfs, field_mapping, ban_year,
                                                                   max_distance)
    # fill the rest of lead pipes with null install year with median value
    pipes = fill_null_install_yr_of_lead_service_lines_with_median(pipes, install_yr_field, material_field)
    # TODO: how neighborhood function can be called as median will always return values?
    # fill rest null values from neighborhood
    pipes = fill_null_install_yr_of_service_lines_from_neighborhood(pipes, min_value, max_value, install_yr_field,
                                                                    abandoned_field)
    pipes.reset_index(inplace=True, drop=True)
    assert len(pipes) == count_pipes, f"Something went wrong filling the column: {install_yr_field}"
    return pipes


def set_unknown_material_based_on_diameter(df, material_field, vrsource_field, diameter_field,
                                           diam_threshold_for_knowns):
    """Sets the pipes having Unknown material and diameter > 4 to Unknown-Unlikely Lead, according to EPA.

    Keyword arguments:
    df -- the dataframe with the pipes
    material_field -- the column name holding the material values
    vrsource_field -- the column name holding the vrsource values, taken from the enumerator:VerificationSource
    diameter_field -- the column name holding the diameter values
    diam_threshold_for_knowns -- the thresshold diameter value above which the pipe's is set to 'Unknown-Unlikely Lead'
    :return:
    the updated dataframe <df>
    """
    df = df.copy()
    cond = (df[diameter_field] >= diam_threshold_for_knowns) & (df[material_field] == "UNKNOWN")
    cm_l.print_formatted_txt(f"We will assign material='UNKNOWN-UNLIKELY LEAD' for {len(df[cond])} records "
                             f"having material = Unknown and pipe_diam >= {diam_threshold_for_knowns}")
    df.loc[cond, material_field] = f"UNKNOWN-UNLIKELY LEAD (DIAMETER>={diam_threshold_for_knowns})"
    df.loc[cond, vrsource_field] = VerificationSource.UNKNOWN_UNLIKELY_LEAD.name
    return df


def set_unknown_material_based_on_install_yr(df, inst_yr_field, material_field, vrsource_field, min_year, extra_msg=""):
    """Sets the pipes having Unknown material and installation year > min_year to Unknown-Unlikely Lead,
       according to EPA.

    Keyword arguments:
    df -- the dataframe with the pipes
    inst_yr_field -- the column name holding the installation year values
    material_field -- the column name holding the material values
    vrsource_field -- the column name holding the vrsource values, taken from the enumerator:VerificationSource
    min_year -- the minimum year above which the pipe's is set to 'Unknown-Unlikely Lead'
    extra_msg -- a text string to add to the updated material value
    :return:
    the updated dataframe <df>
    """
    df = df.copy()
    df[inst_yr_field].fillna(0, inplace=True)
    df[inst_yr_field] = df[inst_yr_field].astype(int)

    # set all unknown material > min_year to Assumed Non Lead
    cond = ((df[material_field] == "UNKNOWN") & (df[inst_yr_field] > min_year))
    msg = f"UNKNOWN-UNLIKELY LEAD (INSTALL>{min_year})"
    if extra_msg != "":
        msg += f" ({extra_msg})"
    print(f"We will assign material=Unknown-Unlikely Lead for {len(df[cond])} records having material = Unknown and "
          f"install_yr > {min_year}", extra_msg)
    df.loc[cond, material_field] = msg
    df.loc[cond, vrsource_field] = VerificationSource.UNKNOWN_UNLIKELY_LEAD.name
    return df


def set_pipe_id_of_abd_pipes(abd_pipes, field_mapping):
    """Adds the prefix <_abd> to the pipe_id of the <abd_pipes>."""
    abd_pipes = abd_pipes.copy()
    pipe_id_field = field_mapping['pipe_id_field']
    pb_prefix = field_mapping['pb_prefix']
    pr_prefix = field_mapping['pr_prefix']

    abd_pipes = simplify_pipe_id(abd_pipes, pipe_id_field, pb_prefix, pr_prefix)
    for prefix in [pb_prefix, pr_prefix]:
        cond = (abd_pipes[pipe_id_field].str.startswith(prefix, na=False))
        abd_pipes.loc[cond, pipe_id_field] = abd_pipes[cond].apply(lambda x: x.replace(prefix, f"{prefix}_abd",
                                                                                       regex=True), axis=1)
    return abd_pipes


def handle_the_replaced_from_verified(pipes, verified, field_mapping):
    """Handles the replaced/abandoned pipes in verified data."""
    pipes = pipes.copy()
    verified = verified.copy()

    pipe_id_field = field_mapping['pipe_id_field']
    install_yr_field = field_mapping['install_yr_field']
    diameter_field = field_mapping['diameter_field']
    abandoned_field = field_mapping['abandoned_field']
    abandon_yr_field = field_mapping['abandon_yr_field']
    vrsource_field = field_mapping['vrsource_field']
    verified_field = field_mapping['verified_field']
    material_field = field_mapping['material_field']
    hsmaterial_field = field_mapping['hsmaterial_field']
    active_field = field_mapping['active_field']
    abandon_mat_field = field_mapping['abandon_mat_field']

    # TODO: temp solution until it is handled from the UI like the inspections
    # we need to make an assumption for the inspected lead pipes, verified as no-lead
    all_lead_ids = list(pipes[(pipes[material_field] == "LEAD")][pipe_id_field])
    # these are the lead pipes in the verified that are assumed to be replaced
    cond_replaced = (verified[pipe_id_field].isin(all_lead_ids)) & (verified[abandoned_field] == 0) & \
                    (verified[material_field] != "LEAD")
    assumed_lead_replaced_ids = list(verified[cond_replaced][pipe_id_field])
    verified.loc[cond_replaced, abandoned_field] = 1
    verified.loc[cond_replaced, abandon_mat_field] = "LEAD"
    verified.loc[cond_replaced, abandon_yr_field] = 0
    verified['verified_diameter'] = verified[diameter_field]

    # TODO: handle this also in the verified data coming from the UI (samples and auto Lead)
    verified_replaced = verified[(verified[abandoned_field] == 1)].copy()
    replaced_ids = list(verified_replaced[pipe_id_field])
    if len(replaced_ids) > 0:
        # 1. handle the abandoned/replaced
        cm_l.print_formatted_txt(f"There are {len(replaced_ids)} total replaced pipes in the verified data "
                                 f"(assumed lead replaced: {len(assumed_lead_replaced_ids)}).", "WARNING")

        # create new dataframe with the replaced pipes
        pipes_replaced = pipes[(pipes[pipe_id_field].isin(replaced_ids))].copy()
        pipes_replaced[abandoned_field] = 1
        pipes_replaced[verified_field] = True
        pipes_replaced[vrsource_field] = VerificationSource.REPLACED_FIELD_VERIFIED.name
        pipes_replaced[active_field] = False
        pipes_replaced.loc[(pipes_replaced[pipe_id_field].isin(
            assumed_lead_replaced_ids)), vrsource_field] = VerificationSource.REPLACED_ASSUMED.name
        # get the abandon year, the diameter and the material
        if abandon_yr_field in list(pipes_replaced.columns):
            pipes_replaced.drop(columns=abandon_yr_field, inplace=True)
        pipes_replaced = pd.merge(pipes_replaced,
                                  verified_replaced[[pipe_id_field, abandon_yr_field, abandon_mat_field,
                                                     'verified_diameter']],
                                  on=pipe_id_field)
        # get the replaced material from the verified replaced material, if it isn't Unknown
        cond_known_replaced_material = (pipes_replaced[abandon_mat_field] != "UNKNOWN")
        pipes_replaced.loc[cond_known_replaced_material, material_field] = pipes_replaced[cond_known_replaced_material][
            abandon_mat_field]
        # if the replaced material is Unknown, then update with the historical material
        cond_unknown_replaced_material = (pipes_replaced[abandon_mat_field] == "UNKNOWN") & \
                                         (pipes_replaced[hsmaterial_field] != "UNKNOWN")
        pipes_replaced.loc[cond_unknown_replaced_material, material_field] = \
            pipes_replaced[cond_unknown_replaced_material][hsmaterial_field]
        # update the diameter of the replaced pipes based on the verified
        cond_update_diameter = (pipes_replaced['verified_diameter'] > 0)
        pipes_replaced.loc[cond_update_diameter, diameter_field] = \
            pipes_replaced[cond_update_diameter]['verified_diameter']
        # update the pipe_id, so that it will contain 'abd' abbreviation
        pipes_replaced = set_pipe_id_of_abd_pipes(pipes_replaced, field_mapping)
        pipes_replaced.drop(columns=[abandon_mat_field, 'verified_diameter'], inplace=True)
        display(pipes_replaced.head())

        # 2. handle the new pipes
        # update the installation year of the new pipes
        current_year = datetime.now().year
        verified_replaced.rename(columns={abandon_yr_field: 'tmp_abd_yr', install_yr_field: 'tmp_ins_yr'}, inplace=True)
        pipes = pd.merge(pipes, verified_replaced[[pipe_id_field, 'tmp_ins_yr', 'tmp_abd_yr']], how="left",
                         on=pipe_id_field)
        pipes.fillna({'tmp_ins_yr': 0, 'tmp_abd_yr': 0}, inplace=True)
        cond_replaced = (pipes[pipe_id_field].isin(replaced_ids))
        cond_inst = ((pipes['tmp_ins_yr'] == 0) | (pipes['tmp_ins_yr'] == pipes[install_yr_field]))
        # if the replaced year is greater than 0 then use this as the installation year of the replaced pipes
        cond = cond_replaced & cond_inst & (pipes['tmp_abd_yr'] > 0)
        pipes.loc[cond, install_yr_field] = pipes[cond]['tmp_abd_yr']
        # if the replaced year equals 0, then assume that pipe was replaced after 20 years from the installation year
        cond = cond_replaced & cond_inst & (pipes['tmp_abd_yr'] == 0)
        pipes.loc[cond, install_yr_field] = pipes[cond][install_yr_field].apply(lambda x: min(x + 20, current_year))
        pipes.drop(columns=['tmp_abd_yr', 'tmp_ins_yr'], inplace=True)
        # update the hsmaterial to UNKNOWN for the replaced pipes, because it is already handled
        pipes.loc[(pipes[pipe_id_field].isin(replaced_ids)), hsmaterial_field] = "UNKNOWN"
        # set the installation year of the replaced to 0, because it is already handled
        verified.loc[(verified[pipe_id_field].isin(replaced_ids)), install_yr_field] = 0
        verified.drop(columns=['verified_diameter'], inplace=True)

        # 3. merge the abandoned/replaced pipes with all the pipes
        pipes = pd.concat([pipes, pipes_replaced], ignore_index=True)

    return pipes, verified


# TODO: this function needs a refactor asap - duplication is evil + we should not have one function for handling
# both samples and inspections!
def update_pipes_from_samples(pipes, inspections_df, field_mapping):
    # TODO: when UI team is ready add abandoned_field, abandon_yr_field, abandon_mat_field in needed_fields
    pipe_id_field = field_mapping['pipe_id_field']
    material_field = field_mapping['material_field']
    diameter_field = field_mapping['diameter_field']
    install_yr_field = field_mapping['install_yr_field']
    verified_field = field_mapping['verified_field']
    vrsource_field = field_mapping['vrsource_field']

    needed_fields = [pipe_id_field, material_field, diameter_field, install_yr_field]
    cm_l.check_needed_df_columns(pipes, needed_fields + [verified_field, vrsource_field])
    cm_l.check_needed_df_columns(inspections_df, needed_fields)

    inspections_df = inspections_df.copy()
    pipes = pipes.copy()

    pipes, inspections_df = handle_the_replaced_from_verified(pipes, inspections_df, field_mapping)

    inspections_df.rename(columns={material_field: 'ver_material',
                                   diameter_field: 'ver_diam',
                                   install_yr_field: 'ver_install'}, inplace=True)

    pipes = pd.merge(pipes, inspections_df[[pipe_id_field, 'ver_material', 'ver_diam', 'ver_install']], how="left",
                     on=pipe_id_field)
    pipes['ver_material'].fillna("UNKNOWN", inplace=True)
    pipes['ver_diam'].fillna(0, inplace=True)
    pipes['ver_install'].fillna(0, inplace=True)

    # mark the new verified records as verified
    cond = (pipes['ver_material'] != "UNKNOWN")
    pipes.loc[cond, verified_field] = True
    pipes.loc[cond, vrsource_field] = VerificationSource.FIELD_VERIFIED.name

    # update the material
    cond = (pipes['ver_material'] != "UNKNOWN") & (pipes[material_field] != pipes['ver_material'])
    pipes.loc[cond, material_field] = pipes[cond]['ver_material']
    print(f"There are {len(pipes[cond])} materials updated.")

    # update the diameter
    cond = (pipes['ver_diam'] != 0) & (pipes[diameter_field] != pipes['ver_diam'])
    pipes.loc[cond, diameter_field] = pipes[cond]['ver_diam']
    print(f"There are {len(pipes[cond])} diameter updated.")

    # update the installation year
    pipes[install_yr_field] = pipes[install_yr_field].astype(int)
    pipes['ver_install'] = pipes['ver_install'].astype(int)
    cond = (pipes['ver_install'] != 0) & (pipes[install_yr_field] != pipes['ver_install'])
    pipes.loc[cond, install_yr_field] = pipes[cond]['ver_install']
    print(f"There are {len(pipes[cond])} install year updated.")

    pipes.drop(columns=['ver_material', 'ver_diam', 'ver_install'], inplace=True)
    return pipes


def update_pipes_from_inspections(pipes_df, inspections_df, field_mapping):
    """Updates the pipes dataframe with the inspection data and prints a report with the changes.
    The inspection data is coming from the UI side: expected changes in: diameter, material, hsmaterial, verified,
    vrsource, abandoned, abandoned year.
    This function can be used only from auto lead analysis through inspection class.
    ASSUMPTION: pipes with lead material -> found non-lead after field inspection -> they were indeed lead,
    but customer replaced them
    ATTENTION: verified values are stored in a database as boolean values, so we need to keep them as boolean values
    UNIT TEST: experimental/kyriaki/auto_lead/handle_verified_data.ipynb

    Keyword arguments:
    pipes_df -- the pipes dataframe
    inspections_df -- the inspections dataframe with all the updated pipes values
    field_mapping -- the field mapping dictionary
    return: updated pipes dataframe with inspection data
    """
    pipe_id_field = field_mapping['pipe_id_field']
    material_field = field_mapping['material_field']
    verified_field = field_mapping['verified_field']
    vrsource_field = field_mapping['vrsource_field']
    abandoned_field = field_mapping['abandoned_field']
    abandon_yr_field = field_mapping['abandon_yr_field']
    diameter_field = field_mapping['diameter_field']
    hsmaterial_field = field_mapping['hsmaterial_field']
    pipes_df = pipes_df.copy()
    needed_fields = [pipe_id_field, material_field, verified_field, vrsource_field, abandoned_field, abandon_yr_field,
                     diameter_field, hsmaterial_field]
    cm_l.check_needed_df_columns(pipes_df, needed_fields)
    cm_l.check_needed_df_columns(inspections_df, needed_fields)

    # in the below code, we change 4 column values in if statement that has to do with different fields
    # we have a conflict as with the for loop we can overwrite them
    # that's why we need tmp fields - to keep the final changes
    pipes_df['tmp_abandoned'] = 0
    pipes_df['tmp_verified'] = False
    pipes_df['tmp_vrsource'] = pipes_df[vrsource_field]
    pipes_df['tmp_hsmaterial'] = "UNKNOWN"
    # initialize a dictionary to store the counts of differences
    diff_counts = {col: 0 for col in inspections_df.columns if col != pipe_id_field}
    # iterate over each row in inspections_df
    for index, inspection in inspections_df.iterrows():
        # find the corresponding row in pipes_df
        pipe = pipes_df.loc[pipes_df[pipe_id_field] == inspection[pipe_id_field]]
        # iterate over each column in inspections_df (except pipe id)
        for col in inspections_df.columns:
            if col != pipe_id_field:
                # compare the values and increment the counter
                if pipe[col].values[0] != inspection[col]:
                    diff_counts[col] += 1

                if col == material_field and pipe[col].values[0] == "LEAD" and inspection[hsmaterial_field] != "LEAD":
                    pipes_df.loc[pipes_df[pipe_id_field] == inspection[pipe_id_field], 'tmp_hsmaterial'] = "LEAD"

                if col == material_field and pipe[col].values[0] == "LEAD" and inspection[col] != "LEAD" and inspection[
                    abandoned_field] == 0:
                    cm_l.print_formatted_txt(
                        f"We will assume that the pipe {inspection[pipe_id_field]} has been replaced as the historical material was LEAD",
                        "WARNING")
                    pipes_df.loc[pipes_df[pipe_id_field] == inspection[pipe_id_field], 'tmp_abandoned'] = 1
                    # EPA: if we have a pipe that is active and was replaced some time ago, then it is verified
                    pipes_df.loc[pipes_df[pipe_id_field] == inspection[pipe_id_field], 'tmp_verified'] = True
                    pipes_df.loc[pipes_df[pipe_id_field] == inspection[
                        pipe_id_field], 'tmp_vrsource'] = VerificationSource.REPLACED_ASSUMED.name

                # keep the hsmaterial that UI sends - if it will not replace a LEAD historical value
                if col == hsmaterial_field and pipe[col].values[0] != "LEAD":
                    pipes_df.loc[pipes_df[pipe_id_field] == inspection[pipe_id_field], hsmaterial_field] = inspection[
                        hsmaterial_field]
                    continue

                # update the value in pipes_df
                pipes_df.loc[pipes_df[pipe_id_field] == inspection[pipe_id_field], col] = inspection[col]
                # if the previous material value was LEAD - the new material is not LEAD - the pipe is not abandoned
    pipes_df.loc[pipes_df['tmp_abandoned'] == 1, abandoned_field] = 1
    pipes_df.loc[pipes_df['tmp_verified'], verified_field] = True
    pipes_df.loc[pipes_df[
                     'tmp_vrsource'] == VerificationSource.REPLACED_ASSUMED.name, vrsource_field] = VerificationSource.REPLACED_ASSUMED.name
    pipes_df.loc[pipes_df['tmp_hsmaterial'] != "UNKNOWN", hsmaterial_field] = "LEAD"
    pipes_df.drop(columns=['tmp_abandoned', 'tmp_verified', 'tmp_vrsource', 'tmp_hsmaterial'], inplace=True)

    # print the report
    sorted_counts = sorted(diff_counts.items(), key=lambda item: item[1], reverse=True)
    if sum(diff_counts.values()) == 0:
        print("No changes detected based on inspections data.")
    else:
        for col, count in diff_counts.items():
            if count > 0:
                print(f"Detected {count} inspections with different {col} values")
    pipes_df[verified_field] = pipes_df[verified_field].astype(bool)

    return pipes_df


def create_abandoned_pipes_based_on_inspections(pipes_df, field_mapping):
    """Creates abandoned pipes based on the inspection data and adjusts the abandoned pipes values.
    This function can be used after calling <update_pipes_from_inspections> function.
    These two functions are part of the auto lead analysis.
    UNIT TEST: experimental/kyriaki/auto_lead/handle_verified_data.ipynb

    Keyword arguments:
    pipes -- the pipes dataframe (active, abandoned)
    field_mapping -- the field mapping dictionary
    """
    pipe_id_field = field_mapping['pipe_id_field']
    material_field = field_mapping['material_field']
    hsmaterial_field = field_mapping['hsmaterial_field']
    verified_field = field_mapping['verified_field']
    vrsource_field = field_mapping['vrsource_field']
    abandoned_field = field_mapping['abandoned_field']
    abandon_yr_field = field_mapping['abandon_yr_field']
    install_yr_field = field_mapping['install_yr_field']
    active_field = field_mapping['active_field']
    pipes_df = pipes_df.copy()

    abandon_sources = [VerificationSource.REPLACED_ASSUMED.name, VerificationSource.REPLACED_FIELD_VERIFIED.name,
                       VerificationSource.ABANDONED.name]
    # find pipes that already replaced through the sampling process
    already_replaced = pipes_df[pipes_df[vrsource_field].isin(abandon_sources)].copy()
    already_replaced = already_replaced[already_replaced[pipe_id_field].str.contains("abd")].copy()
    # the pipes we need to create abandoned ones marked already with these 2 sources in <update_pipes_from_inspections>
    abd_pipes = pipes_df[(pipes_df[vrsource_field] == VerificationSource.REPLACED_ASSUMED.name) | (
            pipes_df[vrsource_field] == VerificationSource.REPLACED_FIELD_VERIFIED.name)].copy()
    # exclude the pipes that already replaced through the sampling process
    abd_pipes = abd_pipes[~abd_pipes[pipe_id_field].isin(already_replaced[pipe_id_field])].copy()
    # when we have an unknown active pipe that was replaced, there is a verified abandoned pipe
    # and through auto-lead customer sends us the inspection data: material to replace unknown and abandoned 1 to
    # indicate that the pipe was replaced - in this case we should not create a new abandoned pipe
    # we should keep the previous abandon values
    abd_copy = abd_pipes.copy()
    abd_pipes_to_check = set_pipe_id_of_abd_pipes(abd_copy, field_mapping)
    abd_pipes_to_check = abd_pipes_to_check[(abd_pipes_to_check[pipe_id_field].isin(pipes_df[pipe_id_field]))].copy()
    if len(abd_pipes_to_check) > 0:
        abd_pipes_to_check['id_without_prefix'] = abd_pipes_to_check[pipe_id_field].str.replace("_abd", "")
        pipes_df.loc[pipes_df[pipe_id_field].isin(abd_pipes_to_check['id_without_prefix']), abandoned_field] = 0
        pipes_df.loc[pipes_df[pipe_id_field].isin(abd_pipes_to_check['id_without_prefix']), abandon_yr_field] = 0
        abd_pipes = abd_pipes[~abd_pipes[pipe_id_field].isin(abd_pipes_to_check['id_without_prefix'])].copy()
    abd_ids = list(abd_pipes[pipe_id_field])

    if len(abd_pipes) > 0:
        cm_l.print_formatted_txt(f"There are {len(abd_pipes)} abandoned pipes we need to create", "WARNING")
        # handle the abandoned pipes
        abd_pipes[verified_field] = True  # we cannot handle abandoned pipes with verified False
        # vrsource here can be REPLACED_ASSUMED or REPLACED_FIELD_VERIFIED -> valid for abandoned pipes
        abd_pipes[active_field] = False  # all abd pipes are inactive
        abd_pipes[abandoned_field] = 1  # safety reasons
        # TODO: replace 20 with a value that comes from RUL (per material)
        # if abandon_yr_field is 0, then we need to update the abandon_yr_field with the installation value + 20
        abd_pipes.loc[abd_pipes[abandon_yr_field] == 0, abandon_yr_field] = abd_pipes[install_yr_field] + 20
        # use the hsmaterial of active pipes as the abandoned material value
        abd_pipes[material_field] = abd_pipes[hsmaterial_field]
        assert len(abd_pipes[abd_pipes[pipe_id_field].str.contains(
            "abd")]) == 0, "There are active pipes with <abd> prefix in pipe ids."

        # update the pipe_id, so that it will contain 'abd' abbreviation
        abd_pipes = set_pipe_id_of_abd_pipes(abd_pipes, field_mapping)
        display(abd_pipes)

        # handle the active pipes that have a correlated replaced pipe
        pipes_df.loc[pipes_df[pipe_id_field].isin(abd_ids), verified_field] = True
        pipes_df.loc[pipes_df[pipe_id_field].isin(abd_ids), abandoned_field] = 0
        # pipes with this source can only be verified: user cannot click replaced if verified is not enabled
        pipes_df.loc[pipes_df[
                         vrsource_field] == VerificationSource.REPLACED_ASSUMED.name, vrsource_field] = VerificationSource.FIELD_VERIFIED.name
        pipes_to_change_vrsource = (pipes_df[pipe_id_field].isin(abd_ids)) & (
                pipes_df[vrsource_field] != VerificationSource.FIELD_VERIFIED.name)
        pipes_df.loc[pipes_to_change_vrsource, vrsource_field] = VerificationSource.ASSUMED_VERIFIED_FROM_ABD.name
        # for pipes that abandon_yr_field is not 0, we need to update the install_yr_field
        pipes_df.loc[(pipes_df[pipe_id_field].isin(abd_ids)) & (pipes_df[abandon_yr_field] > 0), install_yr_field] = \
            pipes_df[abandon_yr_field]
        # TODO: replace 20 with a value that comes from RUL (per material)
        # if abandon_yr_field is 0, then we need to update the install_yr_field with the existing value + 20
        pipes_df.loc[
            (pipes_df[pipe_id_field].isin(abd_ids)) & (pipes_df[abandon_yr_field] == 0), install_yr_field] += 20
        # after that set abandon_yr to 0
        pipes_df.loc[(pipes_df[pipe_id_field].isin(abd_ids)) & (pipes_df[abandon_yr_field] > 0), abandon_yr_field] = 0

        pipes_df = pd.concat([pipes_df, abd_pipes], ignore_index=True)
    else:
        print("No need detected for creating abandoned pipes.")

    return pipes_df


def merge_pipes_with_extra_features(pipes, pipe_id_field_nm, extra_feat_df, extra_feat_nm):
    """Merge pipes data with extra df data.

    Keyword arguments:
    pipes -- the pipes dataframe
    pipe_id_field_nm -- pipe id field name
    extra_feat_df -- the extra feature dataframe
    extra_feat_nm -- the name of extra feature we want to add
    """
    # we accept strings
    print(f"We are adding {extra_feat_nm} data\n")

    # check if there are duplicate columns in the extra features
    pipe_columns = list(pipes.columns)
    extra_feat_columns = list(extra_feat_df.columns)
    if pipe_id_field_nm in extra_feat_columns:
        extra_feat_columns.remove("pipe_id")
    assert not any(
        x in pipe_columns for x in
        extra_feat_columns), f"Multiple fields found while adding extra features: {extra_feat_nm}"

    if pipe_id_field_nm in extra_feat_df.columns:
        # drop geometry because we'll use the geometry of the pipes
        if "geometry" in extra_feat_df.columns:
            extra_feat_df.drop(columns=['geometry'], inplace=True)
        origin_len = len(extra_feat_df)
        # drop any duplicates - shouldn't be any, but if there are, we need to get rid of them
        extra_feat_df.drop_duplicates(subset=[pipe_id_field_nm], keep="first", inplace=True)
        if len(extra_feat_df) != origin_len:
            print(
                f"We dropped {origin_len - len(extra_feat_df)} duplicate pipes that created while adding extra features")
        # check for how many of the pipes we have new data
        df_id = set(extra_feat_df[pipe_id_field_nm].values)
        pipe_ids = set(pipes[pipe_id_field_nm].values)
        common_ids = pipe_ids.intersection(df_id)
        pct_match = len(common_ids) * 100.0 / len(pipe_ids)
        assert pct_match > 80, f"Only {pct_match}% of pipes matched with {extra_feat_nm} data."
        pipes = pd.merge(pipes, extra_feat_df, how="left", on="pipe_id")
    else:
        cm_gn_l.create_index_col(extra_feat_df, "df_id")
        pipes = cm_gn_l.find_nearest_item(pipes, "df_id", extra_feat_df, "df_id", 500)
        # we don't need anymore geometry field
        extra_feat_df.drop(columns=['geometry'], inplace=True)
        pipes = pipes.merge(extra_feat_df, how="left", on="df_id")
        pipes.drop("df_id", axis=1, inplace=True)
    assert "geometry_x" not in pipes.columns, "Multiple geometry fields found"

    return pipes


def add_extra_features_to_pipes_helper(key, item, pipe_id_field, cols_to_exclude, extra_features_col_nms, extra_dir):
    """Helper function that: reads the extra features, does specific checks and adds the columns in
    <extra_features_col_nms>.

    Keyword arguments:
    key -- the name of analysis type (both, public and private)
    item -- the name of the extra feature
    pipe_id_field -- the pipe id field name
    cols_to_exclude -- columns we want to exclude from tracking
    extra_features_col_nms -- columns we want to track
    extra_dir -- directory of extra data
    :return: a dataframe with extra features"""
    df = cm_l.read_data(os.path.join(extra_dir, item))
    if pipe_id_field in df.columns:
        cm_gn_l.convert_pipe_id_to_str(df)
        cm_gn_l.clean_pipe_id(df)
        # we don't have to record these drops if existing
        dropped_pipes = None
        df, dropped_pipes = cm_gn_l.remove_rows_with_nan_value_in_field(df, pipe_id_field, pipe_id_field, dropped_pipes)
    else:
        cm_l.print_formatted_txt(f"No pipe id field found: {item}", "WARNING")
    columns_names = [col for col in df.columns if col not in cols_to_exclude]
    if key in extra_features_col_nms:
        extra_features_col_nms[key].extend(columns_names)
    else:
        extra_features_col_nms[key] = columns_names
    # In Python, when you pass a mutable object e.g., list/dictionary to a function gets a reference to that object
    # so id the function modifies the object, the changes are reflected in the original object as well
    # we don't need to return extra_features_col_nms
    return df


def add_extra_features_to_pipes(pipes, pipe_id_field_nm, extra_dir, extra_features, string_null_values_list,
                                fill_missing_txt_method, cols_to_skip):
    """This function does sanity checks while adding extra features in the pipes dataframe.
    Also, we track column names per analysis type.

    Keyword arguments:
    pipes -- the pipes dataframe
    pipe_id_field_nm -- pipe id field name
    extra_dir -- the directory of extra data
    extra_features -- the extra features we want to add
    string_null_values_list -- the list containing all possible strings for nan values
    fill_missing_txt_method -- the method we want to use to fill missing text values
    cols_to_skip -- the columns we want to skip from filling missing text values
    :return: the pipes dataframe with new columns based on extra features and the extra features per analysis type
    """
    pipes_df = pipes.copy()
    pipes_crs = pipes.crs
    cols_to_exclude = [pipe_id_field_nm, 'geometry']
    extra_features_col_nms = {}

    if isinstance(extra_features, list) and all(isinstance(i, str) for i in extra_features):
        print(f"{LeadRunType.BOTH.name} analysis detected")
        for item in extra_features:
            df = add_extra_features_to_pipes_helper(LeadRunType.BOTH.name, item, pipe_id_field_nm, cols_to_exclude,
                                                    extra_features_col_nms, extra_dir)
            pipes_df = merge_pipes_with_extra_features(pipes_df, pipe_id_field_nm, df, item)
    else:
        print(f"{LeadRunType.SEPARATE.name} analysis detected")
        for key, value in extra_features[0].items():
            for item in value:
                df = add_extra_features_to_pipes_helper(key, item, pipe_id_field_nm, cols_to_exclude,
                                                        extra_features_col_nms, extra_dir)
                # we need to not add twice the extra features that are common between public and private
                if not df.columns.isin(pipes_df.columns).all():
                    pipes_df = merge_pipes_with_extra_features(pipes_df, pipe_id_field_nm, df, item)
    assert pipes_crs == pipes_df.crs, "we lost pipe's crs after adding extra features"

    cm_gn_l.fill_null_values_with_median(pipes_df)
    # TODO: test if connected is working better than neighborhood
    pipes_df = cm_gn_l.fill_missing_values_in_text_fields(pipes_df, string_null_values_list, cols_to_skip,
                                                          fill_missing_txt_method)
    pipes_df.reset_index(drop=True, inplace=True)

    print("\nPipe fields after adding extra features:")
    display(pipes_df.info())
    return pipes_df, extra_features_col_nms


def calc_confidence_level(zeta):
    return norm.cdf(zeta, loc=0, scale=1) - norm.cdf(-zeta, loc=0, scale=1)


def calc_confidence_level_for_sample(n_population, n_sample):
    if n_population > n_sample:
        e = 0.05
        p = 0.5
        a = n_sample * (n_population - 1) / (n_population - n_sample)
        z = math.sqrt((a * e * e) / (p * (1 - p)))
        return round(calc_confidence_level(z) * 100, 1)
    elif n_population == n_sample:
        return 100
    else:
        raise Exception("The sample size exceeds the population.")


def zeta_score(conf_level):
    z = abs(norm.interval(conf_level, loc=0, scale=1)[1])
    return z


def get_population_mean_and_margins(n_population, n_sample, lead_in_sample, conf_level=95):
    p = lead_in_sample / n_sample
    z = zeta_score(conf_level / 100)
    total_mean = round(n_population * p)
    total_variance = n_population * p * (1 - p)
    total_sdev = math.sqrt(total_variance)
    margin = round(z * total_sdev)
    p = round(100 * p, 2)
    return p, total_mean, margin


def calculate_epa_sample_statistics(pipes, df_unknown, df_verified, field_mapping):
    """Calculates the statistics according to epa: calculates the percentage of active and replaced lead found in
    the sample and projects these percentages to the total number of unknowns.

    Keyword arguments:
    pipes -- the dataframe with the pipes
    df_unknown -- the dataframe with the unknowns according to EPA, based on which the samples were created
    field_mapping -- the dictionary with the field mapping
    :return:
    a list with the total expected lead pipes and the margin of error
    """
    # important: select only the verified existing in the Unknown for the statistics
    pipe_id_field, material_field = field_mapping['pipe_id_field'], field_mapping['material_field']
    abandoned_field, hsmaterial_field = field_mapping['abandoned_field'], field_mapping['hsmaterial_field']
    verified_field, abandon_mat_field = field_mapping['verified_field'], field_mapping['abandon_mat_field']
    pb_prefix, pr_prefix = field_mapping['pb_prefix'], field_mapping['pr_prefix']

    unknown_ids = list(df_unknown[pipe_id_field])
    df_verified = df_verified.copy()
    df_verified = df_verified[df_verified[pipe_id_field].isin(unknown_ids)]
    df_verified = df_verified[df_verified[material_field] != "UNKNOWN"]

    system_details = {}
    system_makeup = {}
    high_medium_expected = {}
    for analysis in [pb_prefix, pr_prefix]:
        system = "public" if analysis == pb_prefix else "private"
        system_pipes = pipes[pipes[pipe_id_field].str.startswith(analysis)].copy()
        sample_verified = df_verified[df_verified[pipe_id_field].str.startswith(analysis)].copy()
        n_population = len(df_unknown[df_unknown[pipe_id_field].str.startswith(analysis)])
        n_sample = len(sample_verified)

        # get all the lead pipes in the system
        df_system_lead = system_pipes[(system_pipes[material_field] == "LEAD")].copy()
        system_lead_ids = list(df_system_lead[pipe_id_field])
        # these are the verified lead pipes
        df_ver_lead_in_system = df_system_lead[df_system_lead[verified_field]].copy()
        # these are the unverified lead pipes; they can be only active because all abandoned/replaced are verified
        df_unver_lead_in_system = df_system_lead[~df_system_lead[verified_field]].copy()

        # the verified can be either active or abandoned/replaced
        active_ver_lead_in_system_ids = list(
            df_ver_lead_in_system[(df_ver_lead_in_system[abandoned_field] == 0)][pipe_id_field])
        replaced_lead_in_system_ids = list(
            df_ver_lead_in_system[(df_ver_lead_in_system[abandoned_field] == 1)][pipe_id_field])
        # these are the unverified active lead pipes
        active_unver_lead_in_system_ids = list(df_unver_lead_in_system[(df_unver_lead_in_system[abandoned_field] == 0)]
                                               [pipe_id_field])

        # check if we have verified samples in the <system> (we may have samples only in one side)
        if n_sample > 0:
            active_lead_in_sample_ids = list(sample_verified[(sample_verified[material_field] == "LEAD") &
                                                             (sample_verified[abandoned_field] == 0)][pipe_id_field])
            replaced_lead_in_sample_ids = list(sample_verified[(sample_verified[abandon_mat_field] == "LEAD") &
                                                               (sample_verified[abandoned_field] == 1)][pipe_id_field])

            confid_level = calc_confidence_level_for_sample(n_population, n_sample)
            system_details[system] = {'Total pipes': len(system_pipes), 'EPA unknowns': n_population,
                                      'Verified samples': n_sample, 'Confidence level': f"{confid_level}%"}

            # calculate total expected ACTIVE lead
            active_lead_percentage, active_expected_lead, active_expected_margin = \
                get_population_mean_and_margins(n_population, n_sample, len(active_lead_in_sample_ids))
            # calculate total expected REPLACED lead
            replaced_lead_percentage, replaced_expected_lead, replaced_expected_margin = \
                get_population_mean_and_margins(n_population, n_sample, len(replaced_lead_in_sample_ids))
            # show a table with the system make-up of the verified and unverified lead pipes
            system_makeup[f"{system}:Active Lead"] = {'In the System':
                                                          f"Verified: {len(active_ver_lead_in_system_ids)}, "
                                                          f"Unverified: {len(active_unver_lead_in_system_ids)}",
                                                      'In the Sample': f"{len(active_lead_in_sample_ids)} "
                                                                       f"({active_lead_percentage}%)",
                                                      'Expected': active_expected_lead,
                                                      'Margin of Error': active_expected_margin}
            system_makeup[f"{system}:Replaced Lead"] = {'In the System': len(replaced_lead_in_system_ids),
                                                        'In the Sample': f"{len(replaced_lead_in_sample_ids)} "
                                                                         f"({replaced_lead_percentage}%)",
                                                        'Expected': replaced_expected_lead,
                                                        'Margin of Error': replaced_expected_margin}
            # create a list with all unique lead pipes from the system and the samples
            all_lead_ids = system_lead_ids + active_lead_in_sample_ids + replaced_lead_in_sample_ids
            # remove the '_abd' prefix from the pipe_id
            all_lead_ids = [x.replace("_abd", "") for x in all_lead_ids]
            total_lead = len(set(all_lead_ids))
            # these are the total expected lead pipes in the whole system
            total_expected_lead = total_lead + active_expected_lead + active_expected_margin
            high_medium_expected[system] = [total_lead, total_expected_lead]
        else:
            system_details[system] = {'Total pipes': len(system_pipes), 'EPA unknowns': n_population,
                                      'Verified samples': 0, 'Confidence level': '0%'}
            high_medium_expected[system] = [len(system_lead_ids), len(system_lead_ids)]
            cm_l.print_formatted_txt(f"There aren't any verified data for the {system} side", "WARNING")

    cm_l.print_formatted_txt(f"System details:", "RESULTS")
    system_details = pd.DataFrame.from_dict(system_details).T
    display(system_details[['Total pipes', 'EPA unknowns', 'Verified samples', 'Confidence level']])

    cm_l.print_formatted_txt(f"System make-up of lead pipes:", "RESULTS")
    display(pd.DataFrame.from_dict(system_makeup).T)

    cm_l.print_formatted_txt(f"\nExpected Lead Pipes: {high_medium_expected}\n", "WARNING")
    return high_medium_expected


def calculate_inventory_accuracy(pipes, verified_samples, field_mapping):
    """To calculate the current accuracy of the inventory and print the results in a cross-validation table.
    The term ‘inventory accuracy’ answers to the question: what is the accuracy of the materials referenced in the
    inventory as published in VODA’s UI. To answer this, we create a cross-validation table, checking the historical
    material versus the field verified material.

    Keyword arguments:
    pipes -- a dataframe with all the service lines
    verified_samples -- the dataframe with the field verified samples
    field_mapping -- the dictionary with the column names
    """
    pipe_id_field = field_mapping['pipe_id_field']
    material_field, hsmaterial_field = field_mapping['material_field'], field_mapping['hsmaterial_field']
    for col in [material_field, hsmaterial_field]:
        cond = (pipes[col].str.upper().str.contains("GALV")) | (pipes[col].str.upper().str.startswith("GRR "))
        pipes.loc[cond, col] = "GALVANIZED"

    if len(verified_samples) == 0:
        print("There aren't any valid verified samples to check the accuracy of the inventory.")
    else:
        # TODO: it is not clear from EPA if this check should be done separately for the public and private side
        # for now it is done for the whole system, because of the high cost of the field inspections
        df_to_check = pd.merge(pipes[[pipe_id_field, hsmaterial_field]],
                               verified_samples[[pipe_id_field, material_field]],
                               on=pipe_id_field)
        all_historical = pipes[~(pipes[hsmaterial_field].str.upper().str.contains("UNKNOWN"))].copy()
        # TODO: these insights must be used from the UI when both teams are ready
        # we don't need here inventory_accuracy, but is contained in the return statement of the function
        cm_l.print_formatted_txt(f"Inventory material accuracy results:", "RESULTS")
        inventory_accuracy = calculate_and_print_accuracy_results(all_historical, df_to_check, material_field,
                                                                  hsmaterial_field, combine_all_no_lead=False)


def calculate_and_print_accuracy_results(all_historical, df_to_check, col_ver_material, col_his_material,
                                         combine_all_no_lead):
    """Prints a confusion matrix with verified materials vs the historical material values and calculates the
    accuracy of all the historical, the lead and the non-lead material values.

    Keyword arguments:
    all_historical -- a dataframe with all the pipes having historical material
    df_to_check -- a dataframe which is used to check and compare the historical material with the verified
                   material
    col_ver_material -- the column name having the verified material
    col_his_material -- the column name having the historical material
    combine_all_no_lead -- if all non-lead will be handled as one group in the accuracy check
    :return:
    a dictionary with the accuracy results:
        1. for all the historical values,
        2. only for the lead and
        3. only for all the non-lead.
    """
    df_to_check = df_to_check.copy()
    df_to_check = df_to_check.rename(columns={col_ver_material: 'ver_material', col_his_material: 'his_material'})

    # create the pivot table
    stat_table = pd.pivot_table(df_to_check[['ver_material', 'his_material']], index="his_material",
                                columns="ver_material", aggfunc=len, fill_value=0)
    # compute the row totals
    row_totals = stat_table.sum(axis=1)
    # create a dataFrame with proportions
    stat_proportions = (stat_table.div(row_totals, axis=0) * 100).round(1)
    # combine counts and proportions in the same cell
    stat_table = stat_table.astype(str) + " (" + stat_proportions.astype(str) + "%)"

    # calculate the percentage of the accuracy and their confidence level
    df_to_check = df_to_check[df_to_check['his_material'].str.upper() != "UNKNOWN"]
    df_lead = df_to_check[df_to_check['his_material'].str.upper() == "LEAD"].copy()
    df_no_lead = df_to_check[~df_to_check['his_material'].str.upper().isin(["LEAD", "UNKNOWN"])].copy()
    if combine_all_no_lead:
        df_no_lead['his_material'] = "no_lead"
        no_lead_cond = (~df_no_lead['ver_material'].str.upper().isin(["LEAD", "UNKNOWN"]))
        df_no_lead.loc[no_lead_cond, 'ver_material'] = "no_lead"

    accuracy_result = {}
    checks = {'all': df_to_check, 'lead': df_lead, 'no_lead': df_no_lead}
    for key, df in checks.items():
        df_accurate = df[(df['his_material'] == df['ver_material'])].copy()
        if not df.empty:
            perc_accurate = round(100 * len(df_accurate) / len(df), 1)
        else:
            perc_accurate = 0
        len_sample = len(df)
        len_historical = len(all_historical)
        if key == "lead":
            len_historical = len(all_historical[all_historical[col_his_material].str.upper() == "LEAD"])
        elif key == "no_lead":
            len_historical = len(
                all_historical[~all_historical[col_his_material].str.upper().isin(["LEAD", "UNKNOWN"])])
        confid_level = calc_confidence_level_for_sample(len_historical, len_sample)
        accuracy_result[key] = {'sampling pool': len_historical, 'samples': len_sample,
                                'accuracy': perc_accurate, 'conf_level': confid_level}

    df_results = pd.DataFrame.from_dict(accuracy_result).T
    for col in ['accuracy', 'conf_level']:
        df_results[col] = df_results[col].apply(lambda x: f"{round(x, 1)}%")
    df_results[['sampling pool', 'samples']] = df_results[['sampling pool', 'samples']].astype(int).astype(str)
    display(df_results)

    display(stat_table.style.apply(
        lambda x: ['background: lightyellow' if "(0.0%)" not in v else "" for v in x],
        axis=1))
    return accuracy_result


def set_galvanized_pipes(pipes, joined_df, field_mapping):
    """Finds the galvanized private pipes that are connected with public lead/unknown and characterizes it as
    'galvanized requiring replacement'.
    :return: the dataframe <df> with the updated material
    """
    pipes = pipes.copy()
    pipe_id_field = field_mapping['pipe_id_field']
    material_field = field_mapping['material_field']
    join_public_field = field_mapping['join_public_field']
    join_private_field = field_mapping['join_private_field']
    pb_prefix = field_mapping['pb_prefix']
    pr_prefix = field_mapping['pr_prefix']

    needed_fields = [pipe_id_field, material_field]
    cm_l.check_needed_df_columns(pipes, needed_fields)
    cm_l.check_needed_df_columns(joined_df, [join_public_field, join_private_field])

    public = pipes[(pipes[pipe_id_field].str.lower().str.startswith(pb_prefix))][needed_fields].copy()
    private = pipes[(pipes[pipe_id_field].str.lower().str.startswith(pr_prefix))][needed_fields].copy()
    if len(public) > 0 and len(private) > 0:
        joined_df = joined_df.copy()
        joined_df = joined_df.rename(columns={join_public_field: 'pbid', join_private_field: 'prid'})

        public = public.rename(columns={pipe_id_field: 'pbid', material_field: 'pbmaterial'})
        private = private.rename(columns={pipe_id_field: 'prid', material_field: 'prmaterial'})
        public = simplify_pipe_id(public, "pbid", pb_prefix, pr_prefix, False)

        df_galv = private[private['prmaterial'] == "GALVANIZED"].copy()
        df_merge = pd.merge(df_galv, joined_df, on="prid")
        df_merge = pd.merge(df_merge, public, on="pbid")

        lead_ids = set(list(df_merge[(df_merge['pbmaterial'] == "LEAD")]['prid']))
        # public: is Lead
        if lead_ids:
            new_material = "GRR (downstream of LSL)".upper()
            print(f"Found {len(lead_ids)} private galvanized pipes next to Lead public. They will be updated to: "
                  f"{new_material}")
            pipes.loc[pipes[pipe_id_field].isin(lead_ids), material_field] = new_material

        # all the rest are GRR
        new_material = "GRR (downstream of Unknown)".upper()
        unknown_ids = set(df_galv[(~df_galv['prid'].isin(lead_ids))]['prid'])
        if len(unknown_ids) > 0:
            print(f"Found {len(unknown_ids)} private galvanized pipes next to Lead Unknown. They will be updated to: "
                  f"{new_material}")
            pipes.loc[(pipes[pipe_id_field].isin(unknown_ids)), material_field] = new_material

        if len(pipes[pipes[material_field].str.upper().str.startswith("GRR ", na=False)]) == 0:
            cm_l.print_formatted_txt("\nNo Galvanized Requiring Replacement pipes were found!", "RESULTS")

        display(pipes[material_field].value_counts(dropna=False))
    else:
        cm_l.print_formatted_txt("\nPublic or Private pipes are missing: GRR categorization is skipped!", "RESULTS")
    return pipes


def check_adjacent_public_private_pipes_by_diameter(pipes, joined_df, field_mapping):
    pipe_id_field = field_mapping['pipe_id_field']
    diameter_field = field_mapping['diameter_field']
    abandoned_field = field_mapping['abandoned_field']
    join_public_field = field_mapping['join_public_field']
    join_private_field = field_mapping['join_private_field']
    pb_prefix = field_mapping['pb_prefix']
    pr_prefix = field_mapping['pr_prefix']

    needed_fields = [pipe_id_field, diameter_field, abandoned_field, 'geometry']
    cm_l.check_needed_df_columns(pipes, needed_fields)
    cm_l.check_needed_df_columns(joined_df, [join_public_field, join_private_field])

    public = pipes[(pipes[pipe_id_field].str.lower().str.startswith(pb_prefix)) & (pipes[abandoned_field] == 0)].copy()
    private = pipes[(pipes[pipe_id_field].str.lower().str.startswith(pr_prefix)) & (pipes[abandoned_field] == 0)].copy()
    if len(public) and len(private):
        joined_df = joined_df.copy()
        public = public.rename(columns={pipe_id_field: 'pbid', diameter_field: 'pbdiam'})
        private = private.rename(columns={pipe_id_field: 'prid', diameter_field: 'prdiam'})
        joined_df = joined_df.rename(columns={join_public_field: 'pbid', join_private_field: 'prid'})
        df_merge = pd.merge(private[['prid', 'prdiam']], joined_df, on="prid")
        df_merge = pd.merge(df_merge, public[['pbid', 'pbdiam']], on="pbid")
        df_merge = df_merge[(df_merge['prdiam'] > 0) & (df_merge['pbdiam'] > 0)]
        df_merge['diam_dif'] = abs(df_merge['pbdiam'] - df_merge['prdiam'])
        issues_df = df_merge[df_merge['diam_dif'] > 2]
        issues_df = issues_df.sort_values(by=['diam_dif'])
        if len(issues_df):
            cm_l.print_formatted_txt(f"There are {len(issues_df)} active public-private connections with incompatible "
                                     f"diameters:", "WARNING")
            display(issues_df.head())
        return issues_df
    else:
        return []


def create_customer_year_bins(df, customer_year_bins, prediction_year, field_mapping):
    """Create a column <Year_Range> in the df based on customer's year bins in the <df>."""
    install_yr_field = field_mapping['install_yr_field']
    year_range_field = field_mapping['year_range_field']

    cm_l.check_needed_df_columns(df, [install_yr_field])

    df = df.copy()
    df[year_range_field] = None
    current_year = date.today().year
    max_year = prediction_year
    if prediction_year > current_year:
        max_year = current_year

    if not customer_year_bins:
        customer_year_bins = [0, 1929, 1950, 1979, 1986, max_year]

    customer_year_bins.sort()
    if customer_year_bins[0] != 0:
        customer_year_bins.insert(0, 0)
    if customer_year_bins[-1] < max_year:
        customer_year_bins.append(max_year)
    else:
        customer_year_bins[-1] = max_year

    customer_year_bins = list(set(customer_year_bins))
    customer_year_bins.sort()

    if customer_year_bins[-1] < current_year:
        customer_year_bins.append(current_year)

    for i in range(0, len(customer_year_bins) - 1):
        cond = (df[install_yr_field] > customer_year_bins[i]) & (df[install_yr_field] <= customer_year_bins[i + 1])
        df.loc[cond, year_range_field] = str(customer_year_bins[i]) + "<year<=" + str(customer_year_bins[i + 1])

    return df, customer_year_bins


# ----------------------------------------------------------------
# END OF GENERIC CLEAN UP - FOR ADDITIONAL FUNCTIONS ADD THE CODE BELOW
def find_spatial_outlier(gdf_to_update, col_to_search, buffer_zone, threshold, exclude_zeros=True):
    """It finds if a geometry contains an outlier in the column <col_to_search> by comparing the values
    of the surrounding geometries. It uses the z_score.

    gdf_to_update -- a geodataframe containing the column <col_to_search>
    col_to_search -- a numeric column which will be checked if it has outliers
    buffer_zone -- a distance in feet which will be used to define the search area for each geometry
    threshold -- the z threshold value, above which the geometry will be considered an outlier
    exclude_zeros - it will exclude zeros in the calculations if True
    :return: the original <gdf_to_update> with additional columns:
             [z_score, neigbors, neigbors_mean, neigbors_stddev, is_outlier]
    """

    cm_l.check_needed_df_columns(gdf_to_update, [col_to_search])

    if not is_numeric_dtype(gdf_to_update[col_to_search]):
        raise ValueError(f"The {col_to_search} column is not numeric.")

    gdf_to_update = gdf_to_update.copy()
    col_id = "tmp_id"
    gdf_to_update[col_id] = gdf_to_update.index
    df = gdf_to_update.copy()
    if exclude_zeros:
        df = df[df[col_to_search] > 0].copy()

    df_buffer = df[[col_id, 'geometry']].copy()
    max_buffer = cm_gn_l.convert_value_in_ft_to_pipe_units(df, buffer_zone)
    df_buffer['geometry'] = df_buffer['geometry'].buffer(max_buffer)

    col_left, col_right = col_id + "_left", col_id + "_right"
    df_merge = gpd.sjoin(df_buffer[[col_id, 'geometry']], df[[col_id, col_to_search, 'geometry']])
    df_merge = df_merge[df_merge[col_left] != df_merge[col_right]][[col_left, col_right, col_to_search]]

    df_grouped = df_merge.groupby(col_left).agg(
        neigbors=pd.NamedAgg(col_right, "size"),
        neigbors_mean=pd.NamedAgg(col_to_search, "mean"),
        neigbors_stddev=pd.NamedAgg(col_to_search, "std")).reset_index()

    gdf = pd.merge(df, df_grouped, how="left", left_on=col_id, right_on=col_left)
    gdf['z_score'] = np.where(gdf['neigbors_stddev'] > 0,
                              (gdf[col_to_search] - gdf['neigbors_mean']) / gdf['neigbors_stddev'], 0)
    gdf['z_score'] = gdf['z_score'].fillna(0).astype(float).round(1)
    gdf['is_outlier'] = abs(gdf['z_score']) > threshold
    gdf['is_outlier'].fillna(False, inplace=True)

    gdf_to_update = pd.merge(gdf_to_update, gdf[[col_id, 'neigbors', 'neigbors_mean', 'neigbors_stddev',
                                                 'z_score', 'is_outlier']], how="left", on=col_id)
    gdf_to_update = gdf_to_update.fillna({'z_score': 0, 'neigbors': 0, 'neigbors_mean': 0, 'neigbors_stddev': 0,
                                          'is_outlier': False})
    if exclude_zeros:
        gdf_to_update.loc[gdf_to_update[col_to_search] == 0, 'is_outlier'] = True
    gdf_to_update.drop(columns=[col_id], inplace=True)
    print(f"Total outliers based on column {col_to_search}: {len(gdf_to_update[gdf_to_update['is_outlier']])}")
    return gdf_to_update


def check_and_update_act_abd_install_year(active_df, abandoned_df, min_install_year, max_install_year, col_id,
                                          ban_year, check_material=True):
    """This function checks for conflicts between the installation years of active and abandoned pipes and updates
    them with respect to the abandon year of abandoned service lines and lead ban year.
    It merges the active and abandoned dataframes based on the <col_id>. It iterates through the merged data and checks
    each row's abandon year. Based on different conditions involving the abandon year, installation year of active and
    abandoned pipes, it updates the installation or abandon year and adds an "updated" column with a description of the
    update applied.

    Keyword arguments:
    active_df -- the dataframe with the active pipes
    abandoned_df -- the dataframe with the abandoned pipes
    min_install_year -- the minimum valid installation year of the pipes
    max_install_year -- the maximum valid installation year of the pipes
    col_id -- the column with the ID of the pipes
    ban_year -- the ban year of the lead pipes
    check_material -- if True, it checks the lead pipes with respect to the ban_year, otherwise it does not check the
    material
    :return:
    the <active_df> and <abandoned_df> with updated the installation year values and a new column <updated> holding
    information about the update of the installation year
    """
    active_df = active_df.copy()
    abandoned_df = abandoned_df.copy()
    cols = [col_id, 'install_yr']
    if check_material:
        cols += ['material']

    cm_l.check_needed_df_columns(active_df, cols + ['geometry'])
    cm_l.check_needed_df_columns(abandoned_df, cols + ['abandon_yr', 'geometry'])

    assert len(active_df[active_df.duplicated(subset=[col_id])]) == 0 and \
           len(abandoned_df[abandoned_df.duplicated(subset=[col_id])]) == 0, f"Duplicates {col_id} found!"

    active_df['updated'] = None
    abandoned_df['updated'] = None
    cols.append("updated")
    tmp_merge = pd.merge(active_df[cols], abandoned_df[cols + ['abandon_yr']], on=col_id, suffixes=("_act", "_abd"))

    active_df.set_index(col_id, inplace=True)
    abandoned_df.set_index(col_id, inplace=True)
    for idx, row in tqdm(tmp_merge.iterrows(), total=len(tmp_merge)):
        install_yr_act, install_yr_abd, abandon_yr = row['install_yr_act'], row['install_yr_abd'], row['abandon_yr']
        material_abd = row['material_abd'] if check_material else ""
        txt_update_act = []
        txt_update_abd = []

        # if abandon year is not missing
        if abandon_yr > 0:
            # if the installation year of the active pipe is wrong then update from the abandon year
            if install_yr_act < abandon_yr:
                active_df.loc[row[col_id], 'install_yr'] = abandon_yr
                txt_update_act.append("update the active pipe install_yr from the abandoned pipe abandon_yr")
            # if the installation year of the abandoned is missing:
            if install_yr_abd == 0:
                # if the installation year of the active is less than the abandon year then update the installation
                # year of the abandoned from the active
                if 0 < install_yr_act < abandon_yr:
                    abandoned_df.loc[row[col_id], 'install_yr'] = install_yr_act
                    txt_update_abd.append("update the abandoned pipe install_yr from the active pipe install_yr")
                # if the installation year of the abandoned is missing and the active has no issue, then assume
                # that the installation year of the abandoned is 20 years before the abandon year
                # except when abandoned material is lead, where we need to check the ban year
                else:
                    assumed_year = min(ban_year, abandon_yr - 20) if material_abd == "LEAD" else max(min_install_year,
                                                                                                     abandon_yr - 20)
                    abandoned_df.loc[row[col_id], 'install_yr'] = assumed_year
                    txt_update_abd.append("assume the abandoned pipe install_yr from it's abandon_yr-20 years")
        # if abandon year is missing
        else:
            # if the installation year of the active and abandoned are valid then update the abandon year from
            # the installation year of the active pipe
            if 0 < install_yr_abd < install_yr_act:
                abandoned_df.loc[row[col_id], 'abandon_yr'] = install_yr_act
                txt_update_abd.append("update the abandoned pipe abandon_yr from the active pipe install_yr")
            # if the installation years of the active and abandoned are same then check the ban year for lead pipes:
            elif 0 < install_yr_abd == install_yr_act:
                # TODO: replace hard coded material 'LEAD' with the actual rule
                # 1. if the active is lead and has installation year around 20 years before the ban year then assume
                #    that the installation year of the active is wrong and add 20 years
                if material_abd == "LEAD" and install_yr_act < (ban_year - 20):
                    assumed_year = min(max_install_year, install_yr_act + 20)
                    active_df.loc[row[col_id], 'install_yr'] = assumed_year
                    txt_update_act.append("assume the active pipe install_yr from the abandoned pipe install_yr+20 "
                                          "years")

                    abandoned_df.loc[row[col_id], 'abandon_yr'] = assumed_year
                    txt_update_abd.append("assume the abandoned pipe abandon_yr from the abandoned pipe install_yr+20 "
                                          "years")
                # 2. else assume that the installation year of the abandoned is wrong and subtract 20 years
                else:
                    assumed_year = max(min_install_year, install_yr_abd - 20)
                    abandoned_df.loc[row[col_id], 'install_yr'] = assumed_year
                    txt_update_abd.append("assume the abandoned pipe install_yr from it's install_yr-20 years")

                    abandoned_df.loc[row[col_id], 'abandon_yr'] = install_yr_abd
                    txt_update_abd.append("update the abandoned pipe abandon_yr from it's install_yr")
            # if the installation year of the active and abandoned are flipped
            elif 0 < install_yr_act < install_yr_abd:
                active_df.loc[row[col_id], 'install_yr'] = install_yr_abd
                txt_update_act.append("update the active pipe install_yr from the abandoned pipe install_yr")

                abandoned_df.loc[row[col_id], 'install_yr'] = install_yr_act
                txt_update_abd.append("update the abandoned pipe install_yr from the active pipe install_yr")

                abandoned_df.loc[row[col_id], 'abandon_yr'] = install_yr_abd
                txt_update_abd.append("update the abandoned pipe abandon_yr from it's install_yr")

        active_df.loc[row[col_id], 'updated'] = ",".join(txt_update_act) if len(txt_update_act) > 0 else None
        abandoned_df.loc[row[col_id], 'updated'] = ",".join(txt_update_abd) if len(txt_update_abd) > 0 else None

    for df, label in zip([active_df, abandoned_df], ['Active', 'Abandoned']):
        df['updated'] = df['updated'].str.strip()
        df.reset_index(inplace=True)
        print(label)
        display(df['updated'].value_counts(dropna=False))

    return active_df, abandoned_df


def find_laterals_intersections_with_points(laterals, df_points, col_id, check_reverse, col_prefix,
                                            intersect_tolerance):
    """This functions finds the intersection of the geodataframe <laterals> with other geometries included in the
        geodataframe <df_other>.

    Keyword arguments:
    laterals -- the geodataframe with the line geometries that we want to find its intersections with another
                geodataframe
    df_points -- the geodataframe containing the geometries to check if intersected with the <laterals>
    col_id -- the column name containing the id value of the <laterals>
    check_reverse -- boolean: if True then the line of the <laterals> dataframe is reversed in such a way that its
                     start point intersects the geometry of the <df_other>
    col_prefix -- a prefix that is used to create three more columns in the <laterals>:
                  inters_{col_prefix}, where the <laterals> geometry intersects a geometry in the <df_other>
                  inters_{col_prefix}_start, where the start point of the <laterals> geometry intersects a geometry
                                             in the <df_other>
                  inters_{col_prefix}_end, where the end point of the <laterals> geometry intersects a geometry
                                           in the <df_other>
    intersect_tolerance -- the maximum distance between the <df> geometry and <df_other> geometry to be considered
                  as intersected
    :return:
    the updated geodataframe with the laterals, where the new columns inters_{col_prefix} are appended
    """
    laterals = laterals.copy()
    points_sindex = df_points.sindex

    buffer_value = cm_gn_l.convert_value_in_ft_to_pipe_units(laterals, intersect_tolerance)
    if not {'st_buffer', 'en_buffer', 'polygon'}.issubset(list(laterals.columns)):
        laterals['st_buffer'] = laterals['geometry'].apply(lambda g: geom_l.get_begin(g).buffer(buffer_value))
        laterals['en_buffer'] = laterals['geometry'].apply(lambda g: geom_l.get_end(g).buffer(buffer_value))
        laterals['polygon'] = laterals['geometry'].apply(lambda g: g.buffer(buffer_value))

    field_start = f"inters_{col_prefix}_start"
    field_end = f"inters_{col_prefix}_end"
    field_all = f"inters_{col_prefix}"
    for col in [field_start, field_end, field_all]:
        laterals[col] = False

    if check_reverse and "reversed" not in list(laterals.columns):
        laterals['reversed'] = False

    df_dict = laterals.to_dict("records")
    laterals.set_index(col_id, inplace=True)
    for row in tqdm(df_dict, desc=f"Finding geometry intersections"):
        polygon, st_buffer, en_buffer = row['polygon'], row['st_buffer'], row['en_buffer']
        possible_matches_index = list(points_sindex.intersection(polygon.bounds))
        possible_matches = df_points.iloc[possible_matches_index].copy()
        if len(possible_matches):
            precise_matches = possible_matches[(possible_matches.intersects(polygon))]
            if len(precise_matches):
                laterals.at[row[col_id], field_all] = True

            precise_matches = possible_matches[(possible_matches.intersects(st_buffer))]
            if len(precise_matches):
                laterals.at[row[col_id], field_start] = True
            else:
                precise_matches = possible_matches[(possible_matches.intersects(en_buffer))]
                if len(precise_matches):
                    laterals.at[row[col_id], field_end] = True
                    if check_reverse:
                        laterals.at[row[col_id], 'reversed'] = True
                        laterals.at[row[col_id], 'geometry'] = LineString(
                            geom_l.get_line_coordinates(row['geometry'], reverse=True))
    laterals.reset_index(inplace=True)
    return laterals


def find_laterals_intersections_with_nearby_laterals(laterals, col_id, col_prefix, intersect_tolerance,
                                                     network_tolerance):
    """This function finds the intersection of each geometry included in the <laterals> geodataframe with other
        geometries of the same geodataframe.

    Keyword arguments:
    laterals -- the geodataframe with the line geometries that we want to find its intersections with other geometries
                of the same geodataframe
    col_id -- the column name containing the id value of the <laterals>
    col_prefix -- a prefix that is used to create four more columns in the <laterals>:
                  inters_{col_prefix}_all, where the line geometry intersects any other line geometry
                  inters_{col_prefix}_other, where either the start point or the end point of a line geometry
                                             intersects any other geometry
                  inters_{col_prefix}_end, where the end point of a geometry intersects either the start point or
                                           the end point of another geometry
                  inters_{col_prefix}, where a geometry intersects the start or end point of another geometry, but not
                                       near its start point
    intersect_tolerance -- the maximum distance between the <df> geometry and <df_other> geometry to be considered
                           as intersected
    network_tolerance -- the maximum distance between the continues laterals to form a network
    :return:
    the updated geodataframe with the laterals, where the new columns inters_{col_prefix} are appended
    """
    buffer_value = cm_gn_l.convert_value_in_ft_to_pipe_units(laterals, intersect_tolerance)
    net_value = cm_gn_l.convert_value_in_ft_to_pipe_units(laterals, network_tolerance)
    # create a directed network in order to find the downstream connected lines of each geometry
    # the col_id of the downstream laterals are saved in the column: <downstream_id>
    g, _ = geom_l.create_pipe_network(laterals, laterals, lines_to_update_colid=col_id, lines_to_search_colid=col_id,
                                      network_tolerance=net_value, directed=True)

    laterals = laterals.copy()
    laterals_pt = geom_l.convert_line_start_end_to_points(laterals)
    lateralspt_sindex = laterals_pt.sindex
    laterals_sindex = laterals.sindex

    field_all = f"inters_{col_prefix}_all"
    field_other = f"inters_{col_prefix}_other"
    field_point_end = f"inters_{col_prefix}_end"
    field_point = f"inters_{col_prefix}"
    for col in [field_all, field_other, field_point_end, field_point]:
        laterals[col] = False
    laterals['downstream_id'] = None

    df_dict = laterals.to_dict("records")
    laterals.set_index(col_id, inplace=True)
    for row in tqdm(df_dict, desc=f"Finding geometry intersections with nearby geometries"):
        st, en = geom_l.get_line_startend(row['geometry'])

        polygon = row['geometry'].buffer(buffer_value)
        st_buffer = st.buffer(buffer_value)
        en_buffer = en.buffer(buffer_value)

        downstream = list(g.out_edges(row[col_id]))
        if len(downstream) > 0:
            downstream = [x[1] for x in downstream]
        else:
            downstream = []
        # check lateral lines
        possible_matches_index = list(laterals_sindex.intersection(polygon.bounds))
        possible_matches = laterals.iloc[possible_matches_index].copy()
        possible_matches = possible_matches[possible_matches.index != row[col_id]]
        if len(possible_matches):
            precise_matches = possible_matches[(possible_matches.intersects(polygon))]
            if len(precise_matches):
                laterals.at[row[col_id], field_all] = True

            precise_matches = possible_matches[
                (possible_matches.intersects(st_buffer)) | (possible_matches.intersects(en_buffer))]
            if len(precise_matches):
                laterals.at[row[col_id], field_other] = True
                downstream = list(set(downstream + list(precise_matches.index)))

        # check lateral points
        possible_matches_index = list(lateralspt_sindex.intersection(polygon.bounds))
        possible_matches = laterals_pt.iloc[possible_matches_index].copy()
        possible_matches = possible_matches[possible_matches[col_id] != row[col_id]]
        if len(possible_matches):
            precise_matches = possible_matches[(possible_matches.intersects(en_buffer))]
            if len(precise_matches):
                laterals.at[row[col_id], field_point_end] = True
                downstream = list(set(downstream + list(precise_matches[col_id])))
            else:
                precise_matches = possible_matches[
                    (possible_matches.intersects(polygon)) & (~possible_matches.intersects(st_buffer))]
                if len(precise_matches):
                    laterals.at[row[col_id], field_point] = True
                    downstream = list(set(downstream + list(precise_matches[col_id])))

        laterals.at[row[col_id], 'downstream_id'] = downstream

    laterals.reset_index(inplace=True)

    return laterals


def display_public_private_value_counts(pipes, field_mapping, field_to_count):
    """Calculates the value counts of the field <field_to_count> separately for the public and the private side,
    combines them in a single dataframe and finally prints it.

    Keyword arguments:
    pipes -- the dataframe with the pipes
    field_mapping -- the dictionary with the field name mappings
    field_to_count -- the field name to use for the value_counts statistics
    :return:
    it just prints the dataframe with the value_counts of the <field_to_count> for the public and private side.
    """

    # filter the dataFrame for 'public' and 'private' cases
    df_public = pipes[pipes[field_mapping['pipe_id_field']].str.startswith(field_mapping['pb_prefix'])].copy()
    df_private = pipes[pipes[field_mapping['pipe_id_field']].str.startswith(field_mapping['pr_prefix'])].copy()

    # calculate value counts for both 'public' and 'private' dataFrames
    public_value_counts = df_public[field_to_count].value_counts(dropna=False)
    private_value_counts = df_private[field_to_count].value_counts(dropna=False)

    # merge the two dataFrames on the common index
    result_df = pd.concat([public_value_counts, private_value_counts], axis=1, sort=True)
    columns = ['public', 'private']
    result_df.columns = columns

    print(f"Value counts of {field_to_count}")
    for col in columns:
        result_df[col] = result_df[col].fillna(0).astype(int)

    display(result_df)


def print_unverified_historical_material_info(pipes, field_mapping):
    """Prints the number of unverified historical pipes for the public and private side. It also prints the number
    of the historical LEAD pipes.

    Keyword arguments:
    pipes -- the dataframe with the pipes
    field_mapping -- the dictionary with the field name mappings
    """
    pb_prefix, pr_prefix = field_mapping['pb_prefix'], field_mapping['pr_prefix']
    pipe_id_field, material_field = field_mapping['pipe_id_field'], field_mapping['material_field']
    verified_field, vrsource_field = field_mapping['verified_field'], field_mapping['vrsource_field']
    for system, prefix in {'Public': pb_prefix, 'Private': pr_prefix}.items():
        df = pipes[pipes[pipe_id_field].str.startswith(prefix)].copy()
        df_historical = df[(~df[verified_field]) & (df[vrsource_field] == VerificationSource.HISTORICAL.name)]
        if not df_historical.empty:
            df_lead_historical = df_historical[df_historical[material_field] == "LEAD"].copy()
            print(f"{system} system: there are {len(df_historical)} unverified historical material values to be "
                  f"used in ML. (Total LEAD historical: {len(df_lead_historical)}).")
        else:
            print(f"{system} system: all service lines have verified material.")


def uppercase_field_values(df, field_to_change):
    """It converts the values of the <field_to_change> field to uppercase. Strips the values as well.

    Keyword arguments:
    df -- the dataframe with the pipes
    field_to_change -- the field name with the values we want to change
    """
    df = df.copy()
    df[field_to_change] = df[field_to_change].str.strip().str.upper()
    return df
