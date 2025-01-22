CellMap Loader
==============

CellMap Loader is a utility for loading and managing datasets for the `CellMap Segmentation Challenge <#>`_ using PyTorch. It facilitates the preparation of training and validation data loaders, enabling efficient data handling and preprocessing for cell segmentation tasks.

Table of Contents
-----------------

- `Features`_
- `Prerequisites`_
- `Installation`_
- `Usage`_
  - `Data Preparation`_
  - `Loading the Data`_
  - `Accessing the DataLoader`_
  - `Inspecting the Dataset`_
  - `Visualizing Samples`_
  - `Iterating Over the Dataset`_
- `Example`_
- `License`_

Features
--------

- **Flexible Data Splitting**: Easily specify training and validation splits.
- **Customizable Input and Target Configuration**: Define input and target array shapes and scales.
- **Integration with PyTorch**: Utilize PyTorch's DataLoader for efficient batching and parallel data loading.
- **Dataset Inspection**: Access detailed information about datasets, including file paths and annotated pixels.
- **Deterministic Data Access**: Retrieve specific data samples deterministically for debugging and visualization.




Usage
-----

Data Preparation
^^^^^^^^^^^^^^^

Prepare a CSV file (`datasplit.csv`) that defines the data splits for training and validation. The CSV should include paths to raw data and ground truth labels. Example structure:

.. code-block:: csv

    split,raw_path,gt_path,class
    train,/path/to/raw1.zarr,dataset_raw,/path/to/gt1.zarr,dataset_gt[class1,class2]
    validate,/path/to/raw1.zarr,dataset_raw,/path/to/gt1.zarr,dataset_gt[class1,class2]

Loading the Data
^^^^^^^^^^^^^^^^

Import necessary modules and define data loader parameters:

.. code-block:: python

    from cellmap_segmentation_challenge.utils import get_dataloader
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

Define the path to the data splits and specify the classes:

.. code-block:: python

    datasplit_path = "/home/user/Desktop/cellmap/datasplits/datasplit.csv"
    classes = ["merged_classes_11"]

Configure input and target array information:

.. code-block:: python

    input_array_info = {
        "shape": (1, 128, 128),
        "scale": (8, 8, 8),
    } 
    target_array_info = {
        "shape": (1, 128, 128),
        "scale": (8, 8, 8),
    } 

Accessing the DataLoader
^^^^^^^^^^^^^^^^^^^^^^^

Create training and validation data loaders:

.. code-block:: python

    train_cellmap_loader, val_cellmap_loader = get_dataloader(
        datasplit_path,
        classes,
        batch_size=1,
        input_array_info=input_array_info,
        target_array_info=target_array_info,
        iterations_per_epoch=1000,
        device="cpu",
        target_value_transforms=None,
    )

Inspecting the Dataset
^^^^^^^^^^^^^^^^^^^^^^

The `CellMapLoader` contains a `torch` loader attribute which is the actual PyTorch `DataLoader`.

.. code-block:: python

    loader = train_cellmap_loader.loader
    dataiter = iter(loader)

Access the dataset and its details:

.. code-block:: python

    # Access the entire dataset
    print(train_cellmap_loader.dataset)
    
    # Access individual datasets within the loader
    print(train_cellmap_loader.dataset.datasets)
    
    # Access a specific dataset
    specific_dataset = train_cellmap_loader.dataset.datasets[0]
    print(specific_dataset)

Each `CellMapDataset` includes:

- **Raw Path**: Path to the raw EM data.
- **GT Path(s)**: Path(s) to the ground truth labels.
- **Classes**: List of classes included in the dataset.

Visualizing Samples
^^^^^^^^^^^^^^^^^^^

Retrieve and visualize a sample from the dataset:

.. code-block:: python

    import matplotlib.pyplot as plt
    
    batch = train_cellmap_loader.dataset.datasets[0][0]
    input_tensor = batch["input"][None,...]
    output_tensor = batch["output"][None,...]
    print(input_tensor.shape, output_tensor.shape)
    
    inputs = input_tensor.numpy()
    targets = output_tensor.numpy()
    print(inputs.shape, targets.shape)
    
    fig, axes = plt.subplots(1, 1 + len(classes), figsize=(15, 5))
    axes[0].imshow(inputs[0, 0], cmap='gray')
    axes[0].set_title('Input')
    
    for j in range(len(classes)):
        axes[j + 1].imshow(targets[0, j], cmap='tab20')
        axes[j + 1].set_title(f'{classes[j]}')
        print(f'{classes[j]}: {np.unique(targets[0, j], return_counts=True)}')
    
    plt.show()

Iterating Over the Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

Iterate through the dataset and access specific attributes:

.. code-block:: python

    for dataset in loader.dataset.datasets:
        sources = dataset.target_sources["output"]
        if sources[list(sources)[0]]._current_coords is not None:
            coords = sources[list(sources)[0]]._current_coords
            print(sources[list(sources)[0]].interpolation)
            assert isinstance(list(coords.values())[0][0], float)
            assert sources["merged_11"].pad

Example
-------

Here's a complete example demonstrating how to set up and use the `CellMap Loader`:

.. code-block:: python

    # Import necessary modules
    from cellmap_segmentation_challenge.utils import get_dataloader
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    # Define data split path and classes
    datasplit_path = "/home/user/Desktop/cellmap/datasplits/datasplit_merged.csv"
    classes = ["merged_11"]
    
    # Configure input and target array information
    input_array_info = {
        "shape": (1, 128, 128),
        "scale": (8, 8, 8),
    } 
    target_array_info = {
        "shape": (1, 128, 128),
        "scale": (8, 8, 8),
    } 
    
    # Create data loaders
    train_cellmap_loader, val_cellmap_loader = get_dataloader(
        datasplit_path,
        classes,
        batch_size=1,
        input_array_info=input_array_info,
        target_array_info=target_array_info,
        iterations_per_epoch=1000,
        device="cpu",
        target_value_transforms=None,
    )
    
    # Access the PyTorch DataLoader
    loader = train_cellmap_loader.loader
    dataiter = iter(loader)
    
    # Inspect the dataset
    print(train_cellmap_loader.dataset)
    
    # Access and visualize a specific sample
    batch = train_cellmap_loader.dataset.datasets[0][0]
    input_tensor = batch["input"][None,...]
    output_tensor = batch["output"][None,...]
    print(input_tensor.shape, output_tensor.shape)
    
    inputs = input_tensor.numpy()
    targets = output_tensor.numpy()
    print(inputs.shape, targets.shape)
    
    fig, axes = plt.subplots(1, 1 + len(classes), figsize=(15, 5))
    axes[0].imshow(inputs[0, 0], cmap='gray')
    axes[0].set_title('Input')
    
    for j in range(len(classes)):
        axes[j + 1].imshow(targets[0, j], cmap='tab20')
        axes[j + 1].set_title(f'{classes[j]}')
        print(f'{classes[j]}: {np.unique(targets[0, j], return_counts=True)}')
    
    plt.show()
    
    # Iterate through the dataset
    for dataset in loader.dataset.datasets:
        sources = dataset.target_sources["output"]
        if sources[list(sources)[0]]._current_coords is not None:
            coords = sources[list(sources)[0]]._current_coords
            print(sources[list(sources)[0]].interpolation)
            assert isinstance(list(coords.values())[0][0], float)
            assert sources["merged_11"].pad

