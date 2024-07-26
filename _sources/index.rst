.. cellmap-data documentation master file, created by
   sphinx-quickstart on Tue Jul 23 10:14:38 2024.

.. .. image:: https://raw.githubusercontent.com/janelia-cellmap/cellmap-data/main/docs/source/_static/CellMapLogo.png
..    :alt: CellMap logo
..    :width: 85%

CellMap-Data: the Docs
==========================

This library provides a collection of classes and functions for working with cellmap data, specifically for the CellMap project team training machine learning models. The capabilities include loading data from the CellMap ZarrDataset, transforming data, and splitting data into training and validation sets. Functionality is not provided for writing data.


Contents
==============
.. autosummary::
   
   :recursive:
   :toctree:

   cellmap_data
   cellmap_data.CellMapImage
   cellmap_data.CellMapDataset
   cellmap_data.CellMapMultiDataset
   cellmap_data.CellMapSubset
   cellmap_data.CellMapDataSplit
   cellmap_data.CellMapDataLoader
   cellmap_data.transforms
   cellmap_data.utils
   

.. include:: ../../README.md
   :parser: recommonmark


Links
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`