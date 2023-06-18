from __future__ import annotations

from pathlib import Path
from operator import itemgetter
import matplotlib.pyplot as plt

### Local
from mtslab import Labeller, MultiHotEncoder
from mtslab import rebuild, decompress_binary_matrix, compress_binary_matrix, update_meta


### Load raw data
load_data_from = Path(__file__).parent / f"dataset/data/trial42"
save_to = Path(__file__).parent / f"dataset/labels/label_trial42.pkl"

df_raw, meta = itemgetter("data", "meta")(rebuild(load_data_from))
print(f"\n{df_raw=}\n{meta=}\n")

### Data to be used by the Labeller
dfs = [df_raw[["I1", "A1"]]]

### (optional) Spectrogram views
fs, win, ovl = 6250, 0.2, 0.5
kw_specgram = {
    col: {
        "xextent": (df_raw.index[0], df_raw.index[-1]),
        "NFFT": int(fs * win),
        "Fs": fs,
        "noverlap": int(fs * win * ovl),
    }
    for col in df_raw.columns
}

### Label names
label_names = ["heater", "grinder", "pump", "infuser", "electronics"]

### Start labelling data (save as pickle)
if label_these_time_series := True:
    labeller = Labeller(dfs, save_to, label_names=label_names, kw_specgram=kw_specgram)
    labeller.run()

### Transfer labels to metadata (save within parquet directory)
if transfer_labels_to_metadata := True:

    # (optional) Load labels into the Labeller
    labeller = Labeller(dfs, save_to, label_names=label_names)
    plt.close("all")
    labeller.load(save_to, erase=True)

    # Fetch decomposition labels (compressed)
    compressed_labels = labeller.labels_compressed_
    print(f"\n{compressed_labels=}")
    
    # Compute clustering labels (compressed multihot > onehot > compressed)
    decompressed = decompress_binary_matrix(
        compressed_labels, index=labeller.labels_.index
    )
    print(f"\n{decompressed=}")
    multilabel = MultiHotEncoder(decompressed)
    clustering_labels_as_onehot = multilabel.to_onehot()
    print(f"\n{clustering_labels_as_onehot=}")
    clustering_labels = compress_binary_matrix(clustering_labels_as_onehot)
    print(f"\n{clustering_labels=}")


    # Map clustering to decomposition labels
    mapping = {
        cluster_name: MultiHotEncoder.decode_multilabel_name(cluster_name)[0]
        for cluster_name in clustering_labels_as_onehot.columns
    }

    # Update metadata (parquet _metadata file)
    add2meta = dict(
        labels={
            "decomposition": compressed_labels,
            "clustering": clustering_labels,
            "mapping": mapping,
        }
    )
    print(f"\n{add2meta=}")
    update_meta(load_data_from, add2meta)
