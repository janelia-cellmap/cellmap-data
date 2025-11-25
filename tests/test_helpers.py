"""
Test helpers for creating real test data without mocks.

This module provides utilities to create real Zarr/OME-NGFF datasets
for testing purposes.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import zarr
from pydantic_ome_ngff.v04.axis import Axis
from pydantic_ome_ngff.v04.multiscale import (
    MultiscaleMetadata,
)
from pydantic_ome_ngff.v04.multiscale import (
    Dataset as MultiscaleDataset,
)
from pydantic_ome_ngff.v04.transform import VectorScale


def create_test_zarr_array(
    path: Path,
    data: np.ndarray,
    axes: Sequence[str] = ("z", "y", "x"),
    scale: Sequence[float] = (1.0, 1.0, 1.0),
    chunks: Optional[Sequence[int]] = None,
    multiscale: bool = True,
    absent: int = 0,
) -> zarr.Array:
    """
    Create a test Zarr array with OME-NGFF metadata.

    Args:
        path: Path to create the Zarr array
        data: Numpy array data
        axes: Axis names
        scale: Scale for each axis in physical units
        chunks: Chunk size for Zarr array
        multiscale: Whether to create multiscale metadata

    Returns:
        Created zarr.Array
    """
    path.mkdir(parents=True, exist_ok=True)

    if chunks is None:
        chunks = tuple(min(32, s) for s in data.shape)

    # Create zarr group
    store = zarr.DirectoryStore(str(path))
    root = zarr.group(store=store, overwrite=True)

    if multiscale:
        # Create multiscale group with s0 level
        s0 = root.create_dataset(
            "s0",
            data=data,
            chunks=chunks,
            dtype=data.dtype,
            overwrite=True,
        )

        # Create OME-NGFF multiscale metadata
        axis_list = tuple(
            Axis(
                name=name,
                type="space" if name in ["x", "y", "z"] else "channel",
                unit="nanometer" if name in ["x", "y", "z"] else None,
            )
            for name in axes
        )

        datasets = (
            MultiscaleDataset(
                path="s0",
                coordinateTransformations=(
                    VectorScale(type="scale", scale=tuple(scale)),
                ),
            ),
        )

        multiscale_metadata = MultiscaleMetadata(
            version="0.4",
            name="test_data",
            axes=axis_list,
            datasets=datasets,
        )

        root.attrs["multiscales"] = [
            multiscale_metadata.model_dump(mode="json", exclude_none=True)
        ]

        s0.attrs["cellmap"] = {"annotation": {"complement_counts": {"absent": absent}}}

        return s0
    else:
        # Create simple array without multiscale
        arr = root.create_dataset(
            name="data",
            data=data,
            chunks=chunks,
            dtype=data.dtype,
            overwrite=True,
        )
        return arr


def create_test_image_data(
    shape: Sequence[int],
    dtype: np.dtype = np.float32,
    pattern: str = "gradient",
    seed: int = 42,
) -> np.ndarray:
    """
    Create test image data with various patterns.

    Args:
        shape: Shape of the array
        dtype: Data type
        pattern: Type of pattern ("gradient", "checkerboard", "random", "constant", "sphere")
        seed: Random seed

    Returns:
        Generated numpy array
    """
    rng = np.random.default_rng(seed)

    if pattern == "gradient":
        # Create a gradient along the last axis
        data = np.zeros(shape, dtype=dtype)
        for i in range(shape[-1]):
            data[..., i] = i / shape[-1]
    elif pattern == "checkerboard":
        # Create checkerboard pattern
        indices = np.indices(shape)
        data = np.sum(indices, axis=0) % 2
        data = data.astype(dtype)
    elif pattern == "random":
        # Random values between 0 and 1
        data = rng.random(shape, dtype=np.float32).astype(dtype)
    elif pattern == "constant":
        # Constant value
        data = np.ones(shape, dtype=dtype)
    elif pattern == "sphere":
        # Create a sphere in the center
        data = np.zeros(shape, dtype=dtype)
        center = tuple(s // 2 for s in shape)
        radius = min(shape) // 4

        indices = np.indices(shape)
        distances = np.sqrt(
            sum((indices[i] - center[i]) ** 2 for i in range(len(shape)))
        )
        data[distances <= radius] = 1.0
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return data


def create_test_label_data(
    shape: Sequence[int],
    num_classes: int = 3,
    pattern: str = "regions",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Create test label data for multiple classes.

    Args:
        shape: Shape of the arrays
        num_classes: Number of classes to generate
        pattern: Type of pattern ("regions", "random", "stripes")
        seed: Random seed

    Returns:
        Dictionary mapping class names to label arrays
    """
    rng = np.random.default_rng(seed)
    labels = {}

    if pattern == "regions":
        # Divide the volume into regions for different classes
        for i in range(num_classes):
            class_label = np.zeros(shape, dtype=np.uint8)
            # Create regions along first axis
            start = (i * shape[0]) // num_classes
            end = ((i + 1) * shape[0]) // num_classes
            class_label[start:end] = 1
            labels[f"class_{i}"] = class_label
    elif pattern == "random":
        # Random labels
        for i in range(num_classes):
            labels[f"class_{i}"] = (rng.random(shape) > 0.5).astype(np.uint8)
    elif pattern == "stripes":
        # Create stripes along last axis
        for i in range(num_classes):
            class_label = np.zeros(shape, dtype=np.uint8)
            # Create stripes
            for j in range(shape[-1]):
                if j % num_classes == i:
                    class_label[..., j] = 1
            if np.sum(class_label) == 0 and shape[-1] > 0:
                class_label[..., 0] = 1  # Ensure at least one pixel
            labels[f"class_{i}"] = class_label
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return labels


def create_test_dataset(
    tmp_path: Path,
    raw_shape: Sequence[int] = (64, 64, 64),
    gt_shape: Optional[Sequence[int]] = None,
    num_classes: int = 3,
    raw_scale: Sequence[float] = (4.0, 4.0, 4.0),
    gt_scale: Optional[Sequence[float]] = None,
    seed: int = 0,
    raw_pattern: str = "random",
    label_pattern: str = "regions",
) -> Dict[str, Any]:
    """
    Create a test dataset with raw and ground truth Zarr arrays.

    Args:
        tmp_path: Path to create the dataset
        raw_shape: Shape of the raw data
        gt_shape: Shape of the ground truth data
        num_classes: Number of classes in ground truth
        raw_scale: Scale of the raw data
        gt_scale: Scale of the ground truth data
        seed: Random seed for data generation
        raw_pattern: Pattern for raw data
        label_pattern: Pattern for label data

    Returns:
        Dictionary with paths and parameters of the created dataset
    """
    dataset_path = tmp_path / "dataset.zarr"
    raw_data = create_test_image_data(
        raw_shape, dtype=np.dtype(np.uint8), pattern=raw_pattern, seed=seed
    )
    create_test_zarr_array(dataset_path / "raw", raw_data, scale=raw_scale)

    classes = [f"class_{i}" for i in range(num_classes)]
    if gt_shape is None:
        gt_shape = raw_shape
    if gt_scale is None:
        gt_scale = raw_scale

    label_data = create_test_label_data(
        gt_shape, num_classes, pattern=label_pattern, seed=seed
    )

    for class_name, gt_data in label_data.items():
        class_path = dataset_path / class_name
        create_test_zarr_array(
            class_path,
            gt_data,
            scale=gt_scale,
            absent=np.count_nonzero(gt_data == 0),
        )

    return {
        "raw_path": str(dataset_path / "raw"),
        "gt_path": str(dataset_path / f"[{','.join(classes)}]"),
        "classes": classes,
        "raw_shape": raw_shape,
        "gt_shape": gt_shape,
        "raw_scale": raw_scale,
        "gt_scale": gt_scale,
    }


def create_minimal_test_dataset(tmp_path: Path) -> Dict[str, Any]:
    """
    Create a minimal test dataset for quick tests.

    Args:
        tmp_path: Temporary directory path

    Returns:
        Dictionary with paths and metadata
    """
    return create_test_dataset(
        tmp_path,
        raw_shape=(16, 16, 16),
        num_classes=2,
        raw_scale=(4.0, 4.0, 4.0),
    )


def check_device_transfer(loader, device):
    """
    Check if data transfer between CPU and GPU works as expected.

    Args:
        loader: Data loader providing the data
        device: Device to transfer the data to (e.g., "cuda" or "cpu")

    Returns:
        None
    """
    # Iterate through the data loader
    for batch in loader:
        # Transfer the batch to the specified device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Check if the transfer was successful
        for k, v in batch.items():
            assert v.device == device

        # Break after the first batch to avoid transferring all data
        break
