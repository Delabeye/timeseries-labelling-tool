"""

    Convert data to .mat format for use in MATLAB.

"""

from __future__ import absolute_import

import os, sys
from pathlib import Path
import scipy.io
import numpy as np

dir_path = Path(__file__).parent
sys.path.append(str(dir_path))

from utils import *

path2data = dir_path / "sample_dataset/data/"
path2mat = dir_path / "sample_dataset/data_as_mat/"

for trial_dir in os.listdir(path2data):
    
    path = path2data / trial_dir

    data, meta = itemgetter("data", "meta")(rebuild(path))

    clustering_labels = decompress_binary_matrix(
        meta["labels"]["clustering"], index=data.index, dtype=bool
    )
    decomposition_labels = decompress_binary_matrix(
        meta["labels"]["decomposition"], index=data.index, dtype=bool
    )

    matfile_variables = {
        "time_index": data.index.values,
        "data_columns": np.array(data.columns),
        "data": data.values,
        "clustering_labels_columns": np.array(clustering_labels.columns),
        "clustering_labels": clustering_labels.values.T,
        "decomposition_labels_columns": np.array(decomposition_labels.columns),
        "decomposition_labels": decomposition_labels.values.T,
    }

    path2mat.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(path2mat / f"{trial_dir}.mat", matfile_variables)
