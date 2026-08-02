import os
from import_libraries import *

# Set global parameters for some packages
plt.rcParams.update({'font.size': 12})  # set the font size for all the plots
pd.options.display.float_format = "{:.10f}".format  # set the precision of floating numbers
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', 20)  # Set maximum column width

display(HTML("<style>.container { width:100% !important; }</style>"))
shap.initjs()
sns.set_theme(font_scale=0.8)

SEED_NUMBER = 42

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
os.environ['TF_CUDNN_USE_AUTOTUNE'] = "0"
os.environ['OMP_NUM_THREADS'] = "1"
os.environ['PYTHONHASHSEED'] = str(SEED_NUMBER)

gis_library_dir = os.environ.get("GIS_LIBRARY")
global_work_dir = os.environ.get("GLOBAL_WORK_DIR")

random.seed(SEED_NUMBER)
np.random.seed(SEED_NUMBER)
tf.random.set_seed(SEED_NUMBER)

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=1, allow_soft_placement=True, device_count={'CPU': 1})
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)

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
