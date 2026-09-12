"""Legacy notebook initialization; new code should use explicit settings."""
import os
from satellites.shared.legacy.imports import *

# Set global parameters for some packages
plt.rcParams.update({'font.size': 12})  # set the font size for all the plots
pd.options.display.float_format = "{:.10f}".format  # set the precision of floating numbers
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', 20)  # Set maximum column width

display(HTML("<style>.container { width:100% !important; }</style>"))
shap.initjs()
sns.set_theme(font_scale=0.8)

SEED_NUMBER = 42

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = str(SEED_NUMBER)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

gis_library_dir = os.environ.get("GIS_LIBRARY")
global_work_dir = os.environ.get("GLOBAL_WORK_DIR")

random.seed(SEED_NUMBER)
np.random.seed(SEED_NUMBER)
tf.random.set_seed(SEED_NUMBER)

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=1, allow_soft_placement=True, device_count={'CPU': 1})
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)


from satellites.shared.constants import *  # noqa: F403
