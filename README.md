<img src="https://raw.githubusercontent.com/janelia-cellmap/dacapo/main/docs/source/_static/CellMapLogo.png" alt="CellMap logo" width="85%">

# [CellMap-Data](https://janelia-cellmap.github.io/cellmap-data/)

[![PyPI](https://img.shields.io/pypi/v/cellmap-data.svg?color=green)](https://pypi.org/project/cellmap-data)
![Build](https://github.com/janelia-cellmap/cellmap-data/actions/workflows/ci.yml/badge.svg?branch=main)
![GitHub License](https://img.shields.io/github/license/janelia-cellmap/cellmap-data)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fjanelia-cellmap%2Fcellmap-data%2Fmain%2Fpyproject.toml)
[![codecov](https://codecov.io/gh/janelia-cellmap/cellmap-data/branch/main/graph/badge.svg)](https://codecov.io/gh/janelia-cellmap/cellmap-data)

A comprehensive PyTorch-based data loading and preprocessing library for CellMap biological imaging datasets, designed for efficient machine learning training on large-scale 2D/3D volumetric data.

## Overview

CellMap-Data is a specialized data loading utility that bridges the gap between large biological imaging datasets and machine learning frameworks. It provides efficient, memory-optimized data loading for training deep learning models on cell microscopy data, with support for multi-class segmentation, spatial transformations, and advanced augmentation techniques.

### Key Features

- **🔬 Biological Data Optimized**: Native support for multiscale biological imaging formats (OME-NGFF/Zarr)
- **⚡ High-Performance Loading**: Efficient data streaming with TensorStore backend and optimized PyTorch integration
- **🎯 Flexible Target Construction**: Support for multi-class segmentation with mutually exclusive class relationships
- **🔄 Advanced Augmentations**: Comprehensive spatial and value transformations for robust model training
- **📊 Smart Sampling**: Weighted sampling strategies and validation set management
- **🚀 Scalable Architecture**: Memory-efficient handling of datasets larger than available RAM
- **🔧 Production Ready**: Thread-safe, multiprocess-compatible with extensive test coverage

## Installation

```bash
pip install cellmap-data
```

### Dependencies

CellMap-Data leverages several powerful libraries:

- **PyTorch**: Neural network training and tensor operations
- **TensorStore**: High-performance array storage and retrieval
- **Xarray**: Labeled multi-dimensional arrays with metadata
- **PyDantic**: Data validation and settings management
- **Zarr**: Chunked, compressed array storage

## Quick Start

### Basic Dataset Setup

```python
from cellmap_data import CellMapDataset

# Define input and target array specifications
input_arrays = {
    "raw": {
        "shape": (64, 64, 64),  # Training patch size
        "scale": (8, 8, 8),     # Voxel resolution in nm
    }
}

target_arrays = {
    "segmentation": {
        "shape": (64, 64, 64),
        "scale": (8, 8, 8),
    }
}

# Create dataset
dataset = CellMapDataset(
    raw_path="/path/to/raw/data.zarr",
    target_path="/path/to/labels/data.zarr",
    classes=["mitochondria", "endoplasmic_reticulum", "nucleus"],
    input_arrays=input_arrays,
    target_arrays=target_arrays,
    is_train=True
)
```

### Data Loading with Augmentations

```python
from cellmap_data import CellMapDataLoader
from cellmap_data.transforms import Normalize, RandomContrast, GaussianNoise, Binarize
import torchvision.transforms.v2 as T

# Define spatial transformations
spatial_transforms = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5, "z": 0.2}},
    "rotate": {"axes": {"z": [-30, 30]}},
    "transpose": {"axes": ["x", "y"]}
}

# Define value transformations
raw_value_transforms = T.Compose([
    Normalize(scale=1/255),           # Normalize to [0,1]
    GaussianNoise(std=0.05),          # Add noise for augmentation
    RandomContrast((0.8, 1.2)),       # Vary contrast
])

target_value_transforms = T.Compose([
    Binarize(threshold=0.5),          # Convert to binary masks
    T.ToDtype(torch.float32)          # Ensure correct dtype
])

# Create dataset with transforms
dataset = CellMapDataset(
    raw_path="/path/to/raw/data.zarr",
    target_path="/path/to/labels/data.zarr",
    classes=["mitochondria", "endoplasmic_reticulum", "nucleus"],
    input_arrays=input_arrays,
    target_arrays=target_arrays,
    spatial_transforms=spatial_transforms,
    raw_value_transforms=raw_value_transforms,
    target_value_transforms=target_value_transforms,
    is_train=True
)

# Configure data loader
loader = CellMapDataLoader(
    dataset,
    batch_size=4,
    num_workers=8,
    weighted_sampler=True,  # Balance classes automatically
    is_train=True
)

# Training loop
for batch in loader:
    inputs = batch["raw"]      # Shape: [batch, channels, z, y, x]
    targets = batch["segmentation"]  # Multi-class targets
    
    # Your training code here
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
```

### Multi-Dataset Training

```python
from cellmap_data import CellMapDataSplit

# Define datasets from CSV or dictionary
datasplit = CellMapDataSplit(
    csv_path="path/to/datasplit.csv",
    classes=["mitochondria", "er", "nucleus"],
    input_arrays=input_arrays,
    target_arrays=target_arrays,
    spatial_transforms={
        "mirror": {"axes": {"x": 0.5, "y": 0.5}},
        "rotate": {"axes": {"z": [-180, 180]}},
        "transpose": {"axes": ["x", "y"]}
    }
)

# Access combined datasets
train_loader = CellMapDataLoader(
    datasplit.train_datasets_combined,
    batch_size=8,
    weighted_sampler=True
)

val_loader = CellMapDataLoader(
    datasplit.validation_datasets_combined,
    batch_size=16,
    is_train=False
)
```

## Core Components

### CellMapDataset

The foundational dataset class that handles individual image volumes:

```python
dataset = CellMapDataset(
    raw_path="path/to/raw.zarr",
    target_path="path/to/gt.zarr", 
    classes=["class1", "class2"],
    input_arrays=input_arrays,
    target_arrays=target_arrays,
    is_train=True,
    pad=True,  # Pad arrays to requested size if needed
    device="cuda"
)
```

**Key Features**:

- Automatic 2D/3D handling and slicing
- Multiscale data support
- Memory-efficient random cropping
- Class balancing and weighting
- Spatial transformation pipeline

### CellMapMultiDataset  

Combines multiple datasets for training across different samples:

```python
from cellmap_data import CellMapMultiDataset

multi_dataset = CellMapMultiDataset(
    classes=classes,
    input_arrays=input_arrays,
    target_arrays=target_arrays,
    datasets=[dataset1, dataset2, dataset3]
)

# Weighted sampling across datasets
sampler = multi_dataset.get_weighted_sampler(batch_size=4)
```

### CellMapDataLoader

High-performance data loader with optimization features:

```python
loader = CellMapDataLoader(
    dataset,
    batch_size=16,
    num_workers=12,
    weighted_sampler=True,
    device="cuda",
    iterations_per_epoch=1000  # For large datasets
)

# Optimized GPU memory transfer
loader.to("cuda", non_blocking=True)
```

**Optimizations**:

- CUDA streams for parallel GPU transfer
- Persistent workers for reduced overhead  
- Automatic memory estimation and optimization
- Thread-safe multiprocessing

### CellMapDataSplit

Manages train/validation splits with configuration:

```python
datasplit = CellMapDataSplit(
    dataset_dict={
        "train": [
            {"raw": "path1/raw.zarr", "gt": "path1/gt.zarr"},
            {"raw": "path2/raw.zarr", "gt": "path2/gt.zarr"}
        ],
        "validate": [
            {"raw": "path3/raw.zarr", "gt": "path3/gt.zarr"}
        ]
    },
    classes=classes,
    input_arrays=input_arrays,
    target_arrays=target_arrays
)
```

## Advanced Features

### Spatial Transformations

Comprehensive augmentation pipeline for robust training:

```python
spatial_transforms = {
    "mirror": {
        "axes": {"x": 0.5, "y": 0.5, "z": 0.1}  # Probability per axis
    },
    "rotate": {
        "axes": {"z": [-45, 45], "y": [-15, 15]}  # Angle ranges
    },
    "transpose": {
        "axes": ["x", "y"]  # Axes to randomly reorder
    }
}
```

### Value Transformations

Built-in preprocessing and augmentation transforms:

```python
from cellmap_data.transforms import (
    Normalize, GaussianNoise, RandomContrast, 
    RandomGamma, Binarize, NaNtoNum, GaussianBlur
)

# Input preprocessing
raw_transforms = T.Compose([
    Normalize(scale=1/255),      # Normalize to [0,1]
    GaussianNoise(std=0.1),      # Add noise
    RandomContrast((0.8, 1.2)),  # Vary contrast
    NaNtoNum({"nan": 0})         # Handle NaN values
])

# Target preprocessing
target_transforms = T.Compose([
    Binarize(threshold=0.5),     # Convert to binary
    T.ToDtype(torch.float32)     # Ensure float32
])
```

### Class Relationship Handling

Support for mutually exclusive classes and true negative inference:

```python
# Define class relationships
class_relation_dict = {
    "mitochondria": ["cytoplasm", "nucleus"],     # Mutually exclusive
    "endoplasmic_reticulum": ["mitochondria"],    # Cannot overlap
}

dataset = CellMapDataset(
    # ... other parameters ...
    classes=["mitochondria", "er", "nucleus", "cytoplasm"],
    class_relation_dict=class_relation_dict,
    # True negatives automatically inferred from relationships
)
```

### Memory-Efficient Large Dataset Handling

For datasets larger than available memory:

```python
# Use subset sampling for large datasets
loader = CellMapDataLoader(
    large_dataset,
    batch_size=8,
    iterations_per_epoch=5000,  # Subsample each epoch
    weighted_sampler=True
)

# Refresh sampler between epochs
for epoch in range(num_epochs):
    loader.refresh()  # New random subset
    for batch in loader:
        # Training code
        ...
```

### Writing Predictions

Generate predictions and write to disk efficiently:

```python
from cellmap_data import CellMapDatasetWriter

writer = CellMapDatasetWriter(
    raw_path="input.zarr",
    target_path="predictions.zarr", 
    classes=["class1", "class2"],
    input_arrays=input_arrays,
    target_arrays=target_arrays,
    target_bounds={"array": {"x": [0, 1000], "y": [0, 1000], "z": [0, 100]}}
)

# Write predictions tile by tile
for idx in range(len(writer)):
    inputs = writer[idx]
    predictions = model(inputs)
    writer[idx] = {"segmentation": predictions}
```

## Data Format Support

### Input Formats

- **OME-NGFF/Zarr**: Primary format with multiscale support and full read/write capabilities
- **Local/S3/GCS**: Various storage backends via TensorStore

### Multiscale Support

Automatic handling of multiscale datasets:

```python
# Automatically selects appropriate scale level
dataset = CellMapDataset(
    raw_path="data.zarr",  # Contains s0, s1, s2, ... scale levels
    target_path="labels.zarr",
    # ... other parameters ...
)

# Multiscale input arrays can be specified
input_arrays = {
    "raw_4nm": {
        "shape": (128, 128, 128),
        "scale": (4, 4, 4),
    },
    "raw_8nm": {
        "shape": (64, 64, 64),
        "scale": (8, 8, 8),
    }
}
```

## Performance Optimization

### Memory Management

- Efficient tensor operations with minimal copying
- Automatic GPU memory management
- Streaming data loading for large volumes

### Parallel Processing  

- Multi-threaded data loading
- CUDA streams for GPU optimization
- Process-safe dataset pickling

### Caching Strategy

- Persistent ThreadPoolExecutor for reduced overhead
- Optimized coordinate transformations
- Minimal redundant computations

## Use Cases

### 1. Cell Segmentation Training

```python
# Multi-class cell segmentation
classes = ["cell_boundary", "mitochondria", "nucleus", "er"]
spatial_transforms = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "rotate": {"axes": {"z": [-180, 180]}}
}

dataset = CellMapDataset(
    raw_path="em_data.zarr",
    target_path="segmentation_labels.zarr",
    classes=classes,
    input_arrays={"em": {"shape": (128, 128, 128), "scale": (4, 4, 4)}},
    target_arrays={"labels": {"shape": (128, 128, 128), "scale": (4, 4, 4)}},
    spatial_transforms=spatial_transforms,
    is_train=True
)
```

### 2. Large-Scale Multi-Dataset Training

```python
# Training across multiple biological samples
datasplit = CellMapDataSplit(
    csv_path="multi_sample_split.csv",
    classes=organelle_classes,
    input_arrays=input_config,
    target_arrays=target_config,
    spatial_transforms=augmentation_config
)

# Balanced sampling across datasets
train_loader = CellMapDataLoader(
    datasplit.train_datasets_combined,
    batch_size=16,
    weighted_sampler=True,
    num_workers=16
)
```

### 3. Inference and Prediction Writing  

```python
# Generate predictions on new data
writer = CellMapDatasetWriter(
    raw_path="new_sample.zarr",
    target_path="predictions.zarr",
    classes=trained_classes,
    input_arrays=inference_config,
    target_arrays=output_config,
    target_bounds=volume_bounds
)

# Process in tiles
for idx in writer.writer_indices:  # Non-overlapping tiles
    batch = writer[idx]
    with torch.no_grad():
        predictions = model(batch["input"])
    writer[idx] = {"segmentation": predictions}
```

## Best Practices

### Dataset Configuration

- Choose patch sizes that fit comfortably in GPU memory
- Enable padding for datasets smaller than patch size

### Training Optimization

- Use weighted sampling for imbalanced datasets
- Configure appropriate number of workers (typically 2x CPU cores)
- Enable CUDA streams for multi-GPU setups

### Memory Optimization

- Monitor memory usage with large datasets
- Use iterations_per_epoch for very large datasets
- Refresh samplers between epochs for dataset variety

### Debugging

- Start with small patch sizes and single workers
- Use force_has_data=True for testing with empty datasets
- Check dataset.verify() before training

## API Reference

For complete API documentation, visit: [https://janelia-cellmap.github.io/cellmap-data/](https://janelia-cellmap.github.io/cellmap-data/)

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements  
- Documentation expectations
- Pull request process

## Citation

If you use CellMap-Data in your research, please cite:

```bibtex
@software{cellmap_data,
  title={CellMap-Data: PyTorch Data Loading for Biological Imaging},
  author={Rhoades, Jeff and the CellMap Team},
  url={https://github.com/janelia-cellmap/cellmap-data},
  year={2024}
}
```

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 [Documentation](https://janelia-cellmap.github.io/cellmap-data/)
- 🐛 [Issue Tracker](https://github.com/janelia-cellmap/cellmap-data/issues)
- 💬 [Discussions](https://github.com/janelia-cellmap/cellmap-data/discussions)
- 📧 Contact: [rhoadesj@hhmi.org](mailto:rhoadesj@hhmi.org)
