import dask
import torch
import numpy as np
from cellmap_data.image import CellMapImage
from cellmap_data.empty_image import EmptyImage
from cellmap_data.image_writer import ImageWriter
import pytest


def test_empty_image_basic():
    img = EmptyImage("test", [1.0, 1.0, 1.0], [4, 4, 4])
    assert img.store.shape == (4, 4, 4)
    assert img.class_counts == 0.0
    assert img.bg_count == 0.0
    assert img.bounding_box is None
    assert img.sampling_box is None
    arr = img[{"x": 0.0, "y": 0.0, "z": 0.0}]
    assert torch.all(arr == img.empty_value)
    img.to("cpu")
    img.set_spatial_transforms(None)


def test_image_writer_shape_and_coords(tmp_path):
    # Minimal test for ImageWriter shape/coords
    bbox = {"x": [0.0, 4.0], "y": [0.0, 4.0], "z": [0.0, 4.0]}
    writer = ImageWriter(
        path=tmp_path / "test.zarr",
        label_class="test",
        scale={"x": 1.0, "y": 1.0, "z": 1.0},
        bounding_box=bbox,
        write_voxel_shape={"x": 4, "y": 4, "z": 4},
    )
    shape = writer.shape
    assert shape == {"x": 4, "y": 4, "z": 4}
    center = writer.center
    assert all(isinstance(v, float) for v in center.values())
    offset = writer.offset
    assert all(isinstance(v, float) for v in offset.values())
    coords = writer.full_coords
    assert isinstance(coords, tuple)
    assert hasattr(writer, "array")
    assert "ImageWriter" in repr(writer)


@pytest.mark.timeout(5)  # Fail if test takes longer than 5 seconds
def test_cellmap_image_write_and_read(tmp_path):
    # Create a large, but empty zarr dataset using ImageWriter
    bbox = {"x": [0.0, 4000.0], "y": [0.0, 4000.0], "z": [0.0, 400.0]}
    write_shape = {"x": 4, "y": 4, "z": 4}
    write_shape_list = list(write_shape.values())
    dtype = np.float32
    # Only write a small chunk at the center
    arr = torch.arange(np.prod(write_shape_list), dtype=torch.float32).reshape(
        *write_shape_list
    )
    writer = ImageWriter(
        path=tmp_path / "test.zarr",
        label_class="test",
        scale={"x": 1.0, "y": 1.0, "z": 1.0},
        bounding_box=bbox,
        write_voxel_shape=write_shape,
        dtype=dtype,
        overwrite=True,
    )
    # Write a small block at the center
    writer[writer.center] = arr

    # Now read back only the small chunk with CellMapImage
    img = CellMapImage(
        path=str(tmp_path / "test.zarr"),
        target_class="test",
        target_scale=[1.0, 1.0, 1.0],
        target_voxel_shape=write_shape_list,
    )
    assert img.path == writer.base_path, "Paths should match"
    assert writer.center == img.center, "Center coordinates should match"
    assert writer.scale == img.scale, "Scale should match"
    assert all(
        [all(i == w) for i, w in zip(img.full_coords, writer.full_coords)]
    ), "Coordinates should match"
    img.to("cpu")
    # Test __getitem__ with a center in the middle of the bounding box
    arr_out = img[img.center]
    assert isinstance(arr_out, torch.Tensor)
    assert arr_out.shape == tuple(
        write_shape_list
    ), "Output shape should match write shape"
    assert all(
        [
            all([float(_w) == float(_i) for _w, _i in zip(w, i)])
            for w, i in zip(
                writer.aligned_coords_from_center(writer.center).values(),
                img._current_coords.values(),
            )
        ]
    ), "Aligned writer coords should match image current coords"
    # The values should match the original arr (modulo possible dtype/casting)
    np.testing.assert_allclose(
        arr_out.cpu().numpy(), arr.cpu().numpy(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.timeout(5)  # Fail if test takes longer than 5 seconds
def test_cellmap_image_read_with_zarr_backend(tmp_path, monkeypatch):
    # Set the CELLMAP_DATA_BACKEND environment variable to 'zarr'
    monkeypatch.setenv("CELLMAP_DATA_BACKEND", "zarr")
    monkeypatch.setenv("PYDEVD_UNBLOCK_THREADS_TIMEOUT", "0.01")
    dask.config.set(scheduler="synchronous")
    test_cellmap_image_write_and_read(tmp_path)


def test_image_writer_repr_and_array(tmp_path):
    bbox = {"x": [0.0, 2.0], "y": [0.0, 2.0], "z": [0.0, 2.0]}
    writer = ImageWriter(
        path=tmp_path / "repr_test.zarr",
        label_class="test",
        scale={"x": 1.0, "y": 1.0, "z": 1.0},
        bounding_box=bbox,
        write_voxel_shape={"x": 2, "y": 2, "z": 2},
    )
    # Check __repr__ contains useful info
    r = repr(writer)
    assert "ImageWriter" in r
    assert "test" in r
    # Check array property
    arr = writer.array
    assert arr.shape == (2, 2, 2)


def test_empty_image_slice_and_device():
    img = EmptyImage("test", [1.0, 1.0, 1.0], [2, 2, 2])
    # Test __getitem__ with a dict
    arr = img[{"x": 0.0, "y": 0.0, "z": 0.0}]
    assert arr.shape == (2, 2, 2)
    # Test to() method
    img.to("cpu")
