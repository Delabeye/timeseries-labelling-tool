"""

General purpose classes, functions, global variables and imports

"""

### Local
from mtslab.utils._type_utils import *

### External
import os, sys, shutil
import inspect
from pathlib import Path
import contextlib

import cloudpickle as pickle
import json
import hashlib
import copy
import pprint

import time
import datetime
from operator import itemgetter, attrgetter
import itertools
import functools

import matplotlib

matplotlib.use("Qt5Agg") # backend

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import pandas as pd
import dask.dataframe as dd
import numpy as np
import scipy
import scipy.signal
import scipy.interpolate


# # Dev only
# from icecream import ic, install

# install()  # icecream


"""

Toolbox

"""

def get_neighbours_in_list(x: Any, x_list: Sequence):
    """Get the nearest values in a list (preceding/following `x` in `x_list`)."""
    x_list_sorted = sorted(x_list)
    idx = min(range(len(x_list_sorted)), key=lambda i: abs(x_list_sorted[i] - x))
    if x - x_list_sorted[idx] > 0:
        return (x_list_sorted[idx], x_list_sorted[idx + 1])
    return (x_list_sorted[idx - 1], x_list_sorted[idx])
