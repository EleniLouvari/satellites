"""Shared display constants and defaults; importing this module has no plotting or TensorFlow side effects."""
import os

import numpy as np

SEED_NUMBER = 42
gis_library_dir = os.environ.get("GIS_LIBRARY")
global_work_dir = os.environ.get("GLOBAL_WORK_DIR")

numeric_null_values_list = [0, -1, '-1', '0', -np.inf, np.inf]
string_null_values_list = ['NONE', 'NULL', 'UNKNOWN', 'NAN', 'NON', 'UNK', 'UKN', 'NA', 'N/A', '<NULL>','']
null_values_list = numeric_null_values_list + string_null_values_list

z_scores_dict = {'Cold Spot with 99% Confidence': [np.inf, -2.58, '#4575B5'],
                 'Cold Spot with 95% Confidence': [-2.58, -1.96, '#849EBA'],
                 'Cold Spot with 90% Confidence': [-1.96, -1.65,  '#C0CCBE'],
                 'Not Significant': [-1.65, 1.65, '#9C9C9C'],
                 'Hot Spot with 90% Confidence': [1.65, 1.96, '#FAB984'],
                 'Hot Spot with 95% Confidence': [1.96, 2.58, '#ED7551'],
                 'Hot Spot with 99% Confidence': [2.58, np.inf, '#D62F27']}

outliers_dict = {'yes': 'red', 'no': 'blue', 'null': '#9C9C9C'}


ndvi_colors = ['darkblue', 'blue', 'lightskyblue', 'lightblue'] + ['white'] + ['orange', 'yellow', 'lightgreen', 'darkgreen'] # negative to postitive values
ndwi_colors = ['white', 'lightsteelblue', 'blue']
evi_colors = ['red', 'yellow', 'green']
savi_colors = ['blue', 'orange', 'green']
ndci_colors = ['blue', 'yellow', 'red']
wiw_colors = ['white', 'blue']
moisture_colors = ['darkred', 'red', 'orange', 'yellow', 'lime', 'cyan', 'blue', 'darkblue']
sdt_colors = ['whitesmoke', 'lightsteelblue', 'darkblue']
ndti_colors = ['red', 'white', 'green']
wiw_colors = ['white', 'blue']
gemi_colors = ['green', 'yellow', 'red']

dict_sat_index_params = {'NDWI': {'colors':ndwi_colors, 'clip_values':[-1, 1]},
                         'NDVI': {'colors':ndvi_colors, 'clip_values':[-1, 1]},
                         'NDTI': {'colors':ndti_colors, 'clip_values':[-1, 1]},
                         'NDCI': {'colors':ndci_colors, 'clip_values':[-1, 1]},
                         'EVI': {'colors':evi_colors, 'clip_values':[-1, 1]},
                         'SAVI': {'colors':savi_colors, 'clip_values':[-1, 1]},
                         'MSAVI': {'colors':savi_colors, 'clip_values':[-1, 1]},
                         'BAIS2': {'colors':savi_colors, 'clip_values':[-2, 2]},
                         'NBR1': {'colors':ndvi_colors, 'clip_values':[-1, 1]},
                         'NBR2': {'colors':ndvi_colors, 'clip_values':[-1, 1]},
                         'BURN_BOUNDARY': {'colors':savi_colors, 'clip_values':[-15, 15]},
                         'SDT': {'colors':sdt_colors, 'clip_values':[0, 5.3]}}

# ==============================================================================
# LOG Separator for consistent log formatting across the pipeline
LOG_SEPARATOR_STAGE = 100 * "="
LOG_SEPARATOR_PRODUCT = 100 * "-"
LOG_SEPARATOR_STEP = 100 * "_"
LOG_SEPARATOR_HEADER = 100 * "*"
LOG_SEPARATOR_DURATION = 100 * "."
