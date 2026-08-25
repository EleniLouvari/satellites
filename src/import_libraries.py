"""Import all required libraries for the pipeline."""


import sys
import os
import pathlib
import warnings
import time
import shutil
import inspect
import base64
import gc
import psutil
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from operator import eq
from cftime import DatetimeGregorian
from glob import glob
from os.path import basename
from IPython.display import display, HTML
from IPython import get_ipython
import pyproj
from pyproj import CRS, Transformer
import geopandas as gpd
from geopandas.tools import geocode
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_string_dtype, is_object_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime_dtype
import pandas.io.sql as psql
from geopy.geocoders import Nominatim, Photon
from datetime import date, datetime, timedelta
from dateutil.parser import parse
from tqdm.auto import tqdm as tq_auto
from tqdm.notebook import tqdm
from numbers import Number
import numpy as np
from numpy import array, random, arange, datetime_as_string
from collections import deque, defaultdict
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.dates import YearLocator, DateFormatter, MonthLocator
import mapclassify
import math
import statistics

import sklearn
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, classification_report, roc_auc_score, precision_recall_fscore_support, roc_curve, RocCurveDisplay, make_scorer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, VotingClassifier, IsolationForest, VotingRegressor, StackingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder, KBinsDiscretizer, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import KNNImputer
from sklearn.manifold import TSNE

from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
# from catboost import CatBoostClassifier, CatBoostRegressor

from scikeras.wrappers import KerasClassifier, KerasRegressor
from keras import backend as k

import tensorflow as tf
from  tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Layer, LSTM, Conv2D, Conv3D, Dense, Flatten, Lambda, Activation, BatchNormalization, GlobalAveragePooling3D, MaxPooling3D, Dropout, Reshape, SpatialDropout3D, TimeDistributed, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

import optuna
from prophet import Prophet

#from imblearn.over_sampling import SMOTE
import seaborn as sns
import traceback
import re
from functools import reduce
import shapely
from shapely import geometry, ops
from shapely.ops import split, transform, voronoi_diagram, snap
from shapely.validation import make_valid
from shapely.strtree import STRtree
from shapely.geometry import (
    shape,
    box,
    mapping,
    Polygon,
    MultiPolygon,
    LineString,
    MultiLineString,
    Point,
    MultiPoint,
    GeometryCollection
)
import fiona
import libpysal.weights
from libpysal.weights import Queen, DistanceBand
from libpysal.weights.util import fill_diagonal
import esda
from esda.getisord import G_Local
from esda.moran import Moran, Moran_Local
from splot.libpysal import plot_spatial_weights
from splot.esda import moran_scatterplot, plot_local_autocorrelation, lisa_cluster
import scipy
from scipy import stats
from scipy.spatial import distance_matrix, cKDTree, distance
from scipy.spatial.distance import pdist, squareform, minkowski
from scipy.stats import mode, norm, zscore, gaussian_kde, shapiro, ks_2samp, mannwhitneyu, pearsonr, chi2, skew, normaltest, chisquare, wilcoxon, anderson, skew, kurtosis, jarque_bera, boxcox
from scipy.optimize import curve_fit
from scipy.interpolate import NearestNDInterpolator
from scipy.fft import fft, fftfreq
from scipy.signal import welch, spectrogram
from scipy.ndimage import generic_filter
from phik import phik_matrix
from xicor import xicor
from typing import Tuple, List, Optional, Literal

import osmnx as ox
import networkx as nx
import folium
import psycopg2 as pg

from sqlalchemy import create_engine
import geoalchemy2
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from tqdm.contrib.concurrent import process_map
import random
from pykrige.ok import OrdinaryKriging
from pykrige.variogram_models import linear_variogram_model, power_variogram_model, gaussian_variogram_model, exponential_variogram_model, spherical_variogram_model
from typing import Any, Callable, Dict, Final, Generator, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import obspy
from obspy import Stream, Trace
from obspy.signal.tf_misfit import plot_tfr
from affine import Affine
from itertools import product
import dill
import csv
import shap
import geocoder
from copy import deepcopy
import ipykernel
import urllib
import json
from jupyter_server import serverapp
import logging
from typing import List, Dict, Any

import openeo
from openeo import processes as eop
import xarray as xr
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio import windows
from rasterio import features
from rasterio import warp
from rasterio.windows import from_bounds
from PIL import Image
import rioxarray
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from numba import jit, prange
# from chrs_persiann import CHRS
import zipfile
from skimpy import skim
import cdsapi

from osgeo import gdal, osr
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.extras import RealDictCursor
