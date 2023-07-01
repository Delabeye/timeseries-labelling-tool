"""

    Read data, metadata and labels from parquet files.

"""

import os, sys
from pathlib import Path
import pprint

dir_path = Path(__file__).parent
sys.path.append(str(dir_path))

from utils import *


def rebuild_trial_data(path_to_trial_dir: str) -> dict:
    """Rebuild dataframe (data and labels) from parquet file"""

    data, meta = itemgetter("data", "meta")(rebuild(path_to_trial_dir))

    clustering_labels = decompress_binary_matrix(
        meta["labels"]["clustering"], index=data.index, dtype=bool
    )
    decomposition_labels = decompress_binary_matrix(
        meta["labels"]["decomposition"], index=data.index, dtype=bool
    )

    return {
        "data": data,
        "meta": meta,
        "clustering_labels": clustering_labels,
        "decomposition_labels": decomposition_labels,
    }


### Gather all trials (data, metadata and labels) in a dictionary

path2data = dir_path / "sample_dataset/data/"

trials = {
    trial_dir: rebuild_trial_data(path2data / trial_dir)
    for trial_dir in os.listdir(path2data)
}

pprint.pprint(trials)