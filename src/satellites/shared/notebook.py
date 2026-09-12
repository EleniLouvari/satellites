"""Opt-in notebook display and reproducibility setup."""
import os
import random


def configure_notebook(seed: int = 42, *, tensorflow: bool = False) -> None:
    """Apply historical notebook defaults explicitly, without affecting library imports.

    Enable TensorFlow only in kernels that need the historical CPU-only setup.
    Environment variables should be set before importing TensorFlow in the kernel.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from IPython.display import HTML, display

    plt.rcParams.update({'font.size': 12})
    pd.options.display.float_format = "{:.10f}".format
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 20)
    display(HTML("<style>.container { width:100% !important; }</style>"))
    sns.set_theme(font_scale=0.8)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if tensorflow:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf

        tf.get_logger().setLevel('ERROR')
        tf.random.set_seed(seed)
