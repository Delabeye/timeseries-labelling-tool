"""

    Utility functions for storing and rebuilding the dataframes from parquet files.

"""

import logging
from pathlib import Path
import json
import pyarrow.parquet as pq
import pandas as pd
import dask.dataframe as dd
import numpy as np
from operator import itemgetter


def store(
    df: pd.DataFrame | dd.DataFrame,
    path: str | Path,
    meta: dict = {},
    npartitions: int = None,
) -> None:
    """Store uniformly sampled dataframe to parquet directory.

    Parameters
    ----------
    df : pd.DataFrame | dd.DataFrame
        Dataframe to store.
    path : str | Path
        Path to store the dataframe to.
    meta : dict, optional
        Metadata, by default {}
    npartitions : int, optional
        Number of partitions (default is either 1 if `df` is a pandas DataFrame,
        or `df.npartitions` if `df` is a dask DataFrame),
        by default None

    Raises
    ------
    ValueError
        When `df` is neither a pandas nor a dask DataFrame.
    """

    if isinstance(df, pd.DataFrame):
        npartitions = npartitions or 1
        data = dd.from_pandas(df, npartitions=npartitions)
    elif isinstance(df, dd.DataFrame):
        npartitions = npartitions or df.npartitions
        data = df
    else:
        raise ValueError("df must be either a pandas or a dask dataframe.")

    Path(path).mkdir(exist_ok=True, parents=True)
    meta = {"n_samples": len(df), "time_range": (df.index[0], df.index[-1])} | meta
    dd.to_parquet(
        data,
        Path(path),
        write_index=False,
        overwrite=True,
        custom_metadata={"meta": json.dumps(meta)},
        write_metadata_file=True,
    )


def update_metadata(path: str | Path, add: dict = {}, erase: bool = False) -> dict:
    """Store uniformly sampled dataframe to parquet directory.

    Parameters
    ----------
    path : str | Path
        Path to store the dataframe to.
    add : dict, optional
        Additional metadata (metadata will be updated with `add`), by default {}
    erase : bool, optional
        Whether to erase previous metadata, by default False

    Returns
    -------
    dict
        Up-to-date metadata which was written to the metadata file.
    """

    meta, data = itemgetter("meta", "data")(rebuild(path))
    npartitions = dd.read_parquet(Path(path), index=False).npartitions

    meta = add if erase else meta | add

    store(data, path, meta, npartitions=npartitions)

    return meta


def rebuild(path: str | Path, labels: bool = True) -> dict:
    """Load and rebuild data from parquet directory.

    Parameters
    ----------
    path : str | Path
        Path to parquet directory.

    Returns
    -------
    dict
        Dictionary with keys "meta", "data".
    """

    # Load metadata
    meta = fetch_metadata(path)

    # Rebuild dataframe
    n_samples = meta["n_samples"]
    index = pd.Index(np.linspace(*meta["time_range"], n_samples), name="time")
    df = dd.read_parquet(Path(path), index=False).compute().set_index(index)

    return {"meta": meta, "data": df}


def fetch_metadata(path: str | Path) -> dict:
    """Load metadata from parquet directory."""
    return json.loads(pq.read_metadata(Path(path) / "_metadata").metadata[b"meta"])


def compress_binary_matrix(
    x: pd.DataFrame | np.ndarray,
    include_window_size: bool = False,
    squeeze: bool = False,
    use_index_scale: bool = False,
) -> dict[str, list[tuple[int, int]]] | list[tuple[int, int]]:
    """Compress a binary matrix to dictionary of indices (start/stop).

    Parameters
    ----------
    x : pd.DataFrame | np.ndarray
        Input data as a time-indexed dataframe or ndarray (n_components, n_samples).
    include_window_size : bool, optional
        Include window size in the output (start, stop, stop - start),
        [start, stop) otherwise,
        by default False
    squeeze : bool, optional
        Squeeze output data to a list of tuples (start, stop, [stop - start]),
        only if input is at most 1d (i.e., 1 column or array),
        by default False
    use_index : bool, optional
        Use index scale (e.g., time) instead of index (e.g., sample number),
        by default False

    Returns
    -------
    dict[str, list[tuple[int, int]]]
        Compressed data, e.g., {"a":[(0, 10), (20,30)]}.
    """

    if isinstance(x, pd.DataFrame | pd.Series):
        x = pd.DataFrame(x)
    elif isinstance(x, np.ndarray):
        x = pd.DataFrame(x.T)
    else:
        raise TypeError("Input data must be a dataframe or ndarray.")

    x_index = np.arange(len(x)).tolist()
    out = {}
    for col in x.columns:
        ### Find discontinuities
        discontinuities = list([np.where(np.abs(np.diff(x[col])) > 0)[0] + 1].pop())
        discontinuities = np.array([0] + discontinuities + [len(x)], dtype=int)

        ### Retrieve the indices where these discontinuities occur
        ### (and where the section is non-zero)
        wrap = (
            lambda start, stop: (start, stop, stop - start)
            if include_window_size
            else (start, stop)
        )
        indices_starts_and_stops = [
            wrap(x_index[i], x_index[min(nxt, len(x) - 1)])
            for i, nxt in zip(discontinuities[:-1], discontinuities[1:])
            if x[col].values[i]
        ]

        if use_index_scale:
            out[col] = [
                tuple(x.index[idx] for idx in rrange)
                for rrange in indices_starts_and_stops
            ]
        else:
            out[col] = indices_starts_and_stops

    ### Squeeze output or return as dict
    if squeeze and len(out) == 1:
        return list(out.values()).pop()
    else:
        return out


def decompress_binary_matrix(
    compressed_x: dict[str, list[tuple[int, int]]],
    index: pd.Index | np.ndarray = None,
    dtype=bool,
) -> pd.DataFrame:
    """Decompress a binary matrix from a dictionary of indices [start, stop).

    Parameters
    ----------
    compressed_x : dict[str, list[tuple[int, int]]]
        Compressed input data, e.g., {"a":[(0, 10), (20,30)]}.
    dtype : bool, optional
        Cast output data to `dtype`, by default bool

    Returns
    -------
    pd.DataFrame
        Decompressed data.
    """

    if index is None:
        n_samples = max(
            *[stop for _, rrange in compressed_x.items() for _, stop in rrange]
        )
        logging.warning(
            "`decompress_binary_matrix()`: No index provided, using range (0, n_samples) instead,"
            "where n_samples is the maximum stop index in the compressed data."
        )
    else:
        n_samples = len(index)

    x_dict2df = {col: np.zeros(n_samples, dtype=bool) for col in compressed_x.keys()}
    for col, index_ranges in compressed_x.items():
        for index_range in index_ranges:
            start, stop = index_range[0], index_range[1]
            # include last index if it is the last index in the index range,
            # [start, stop) otherwise
            if stop == n_samples - 1:
                stop += 1  # add small offset to include last index
            x_dict2df[col][start:stop] = True

    return pd.DataFrame(x_dict2df, index=index, dtype=dtype)
