<img src="https://raw.githubusercontent.com/janelia-cellmap/dacapo/main/docs/source/_static/CellMapLogo.png" alt="CellMap logo" width="85%">

# [CellMap-Data](https://janelia-cellmap.github.io/cellmap-data/)

[![PyPI](https://img.shields.io/pypi/v/cellmap-data.svg?color=green)](https://pypi.org/project/cellmap-data)
![Build](https://github.com/janelia-cellmap/cellmap-data/actions/workflows/ci.yml/badge.svg?branch=main)
![GitHub License](https://img.shields.io/github/license/janelia-cellmap/cellmap-data)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fjanelia-cellmap%2Fcellmap-data%2Fmain%2Fpyproject.toml)
[![codecov](https://codecov.io/gh/janelia-cellmap/cellmap-data/branch/main/graph/badge.svg)](https://codecov.io/gh/janelia-cellmap/cellmap-data)



Utility for loading CellMap data for machine learning training, utilizing PyTorch, Xarray, TensorStore, and PyDantic.

You can select classes to load to construct targets separately from the labels you want to predict. This allows you to train a model to predict a subset of labels, while still using all labels to construct the target from true negatives as well as true positives.

## Installation

```bash
micromamba create -n cellmap -y -c conda-forge -c pytorch python=3.11
micromamba activate cellmap
git clone https://github.com/janelia-cellmap/cellmap-data.git
cd cellmap-data
pip install -e .
```
