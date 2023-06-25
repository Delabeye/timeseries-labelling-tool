# timeseries-labelling-tool

A rudimentary matplotlib-based time series labeller, compatible with multi-label disaggregation.

## Installation

```bash
conda env create -f environment.yml
pip install -e .
```

## Requirements
* matplotlib
* mplcursors
* seaborn
* pandas
* dask
* pyarrow
* scikit_learn
* scipy
* numpy

## Run the demo

```bash
cd examples
python label.py
```
