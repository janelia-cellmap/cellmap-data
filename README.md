# cellmap.data

<!-- [![License](https://img.shields.io/pypi/l/cellmap.data.svg?color=green)](https://github.com/janelia-cellmap/cellmap.data/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cellmap.data.svg?color=green)](https://pypi.org/project/cellmap.data)
[![Python Version](https://img.shields.io/pypi/pyversions/cellmap.data.svg?color=green)](https://python.org) -->
[![CI](https://github.com/janelia-cellmap/cellmap.data/actions/workflows/ci.yml/badge.svg)](https://github.com/janelia-cellmap/cellmap.data/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/janelia-cellmap/cellmap.data/branch/main/graph/badge.svg)](https://codecov.io/gh/janelia-cellmap/cellmap.data)

Utility for loading CellMap data for machine learning training, utilizing PyTorch, Xarray, TensorStore, and PyDantic.

You can select classes to load to construct targets separately from the labels you want to predict. This allows you to train a model to predict a subset of labels, while still using all labels to construct the target from true negatives as well as true positives.

## Installation

```bash
micromamba create -n cellmap -y -c conda-forge -c pytorch python=3.11
micromamba activate cellmap
git clone https://github.com/janelia-cellmap/cellmap.data.git
cd cellmap-data
pip install -e .
```