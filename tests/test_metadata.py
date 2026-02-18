"""
Tests for utils/metadata.py.

Tests OME-NGFF metadata generation, writing, and scale-level lookup.
"""

import json
import os

import numpy as np
import pytest
import zarr

from cellmap_data.utils.metadata import (
    add_multiscale_metadata_levels,
    create_multiscale_metadata,
    find_level,
    generate_base_multiscales_metadata,
    write_metadata,
)


class TestGenerateBaseMultiscalesMetadata:
    """Tests for generate_base_multiscales_metadata."""

    def test_basic_structure(self):
        z_attrs = generate_base_multiscales_metadata(
            ds_name="my_dataset",
            scale_level=0,
            voxel_size=[4.0, 4.0, 4.0],
            translation=[0.0, 0.0, 0.0],
            units="nanometer",
            axes=["z", "y", "x"],
        )
        assert "multiscales" in z_attrs
        assert len(z_attrs["multiscales"]) == 1
        ms = z_attrs["multiscales"][0]
        assert ms["version"] == "0.4"

    def test_axes_populated(self):
        z_attrs = generate_base_multiscales_metadata(
            ds_name="test",
            scale_level=0,
            voxel_size=[8.0, 8.0, 8.0],
            translation=[0.0, 0.0, 0.0],
            units="nanometer",
            axes=["z", "y", "x"],
        )
        axes = z_attrs["multiscales"][0]["axes"]
        assert len(axes) == 3
        axis_names = [a["name"] for a in axes]
        assert axis_names == ["z", "y", "x"]
        for a in axes:
            assert a["type"] == "space"
            assert a["unit"] == "nanometer"

    def test_dataset_path_uses_scale_level(self):
        z_attrs = generate_base_multiscales_metadata(
            ds_name="test",
            scale_level=2,
            voxel_size=[16.0, 16.0, 16.0],
            translation=[0.0, 0.0, 0.0],
            units="nanometer",
            axes=["z", "y", "x"],
        )
        datasets = z_attrs["multiscales"][0]["datasets"]
        assert datasets[0]["path"] == "s2"

    def test_voxel_size_stored(self):
        voxel_size = [4.0, 8.0, 16.0]
        z_attrs = generate_base_multiscales_metadata(
            ds_name="test",
            scale_level=0,
            voxel_size=voxel_size,
            translation=[0.0, 0.0, 0.0],
            units="nanometer",
            axes=["z", "y", "x"],
        )
        datasets = z_attrs["multiscales"][0]["datasets"]
        transforms = datasets[0]["coordinateTransformations"]
        scale_transform = next(t for t in transforms if t.get("type") == "scale")
        assert scale_transform["scale"] == voxel_size

    def test_zarr_suffix_stripped_from_name(self):
        z_attrs = generate_base_multiscales_metadata(
            ds_name="some_path/dataset.zarr/subgroup",
            scale_level=0,
            voxel_size=[4.0, 4.0, 4.0],
            translation=[0.0, 0.0, 0.0],
            units="nanometer",
            axes=["z", "y", "x"],
        )
        name = z_attrs["multiscales"][0]["name"]
        assert ".zarr" not in name

    def test_name_stored(self):
        z_attrs = generate_base_multiscales_metadata(
            ds_name="my_group",
            scale_level=0,
            voxel_size=[4.0, 4.0, 4.0],
            translation=[0.0, 0.0, 0.0],
            units="nanometer",
            axes=["z", "y", "x"],
        )
        assert z_attrs["multiscales"][0]["name"] == "my_group"

    def test_2d_axes(self):
        z_attrs = generate_base_multiscales_metadata(
            ds_name="2d_test",
            scale_level=0,
            voxel_size=[4.0, 4.0],
            translation=[0.0, 0.0],
            units="nanometer",
            axes=["y", "x"],
        )
        axes = z_attrs["multiscales"][0]["axes"]
        assert len(axes) == 2


class TestAddMultiscaleMetadataLevels:
    """Tests for add_multiscale_metadata_levels."""

    @pytest.fixture
    def base_metadata(self):
        return generate_base_multiscales_metadata(
            ds_name="test",
            scale_level=0,
            voxel_size=[4.0, 4.0, 4.0],
            translation=[0.0, 0.0, 0.0],
            units="nanometer",
            axes=["z", "y", "x"],
        )

    def test_adds_correct_number_of_levels(self, base_metadata):
        result = add_multiscale_metadata_levels(base_metadata, 0, 3)
        datasets = result["multiscales"][0]["datasets"]
        # Started with 1 level (s0), added 3 more (s1, s2, s3)
        assert len(datasets) == 4

    def test_added_paths_sequential(self, base_metadata):
        result = add_multiscale_metadata_levels(base_metadata, 0, 2)
        datasets = result["multiscales"][0]["datasets"]
        paths = [d["path"] for d in datasets]
        assert "s1" in paths
        assert "s2" in paths

    def test_scale_formula(self, base_metadata):
        # With base_scale_level=1, the added level uses pow(2, 1)=2, so scale doubles
        result = add_multiscale_metadata_levels(base_metadata, 1, 1)
        datasets = result["multiscales"][0]["datasets"]
        s0_scale = datasets[0]["coordinateTransformations"][0]["scale"]
        s1_scale = datasets[1]["coordinateTransformations"][0]["scale"]
        # Formula: sn = dim * pow(2, level) where level=1
        for i in range(len(s0_scale)):
            assert s1_scale[i] == pytest.approx(s0_scale[i] * 2, rel=1e-5)

    def test_zero_levels_adds_nothing(self, base_metadata):
        original_count = len(base_metadata["multiscales"][0]["datasets"])
        result = add_multiscale_metadata_levels(base_metadata, 0, 0)
        assert len(result["multiscales"][0]["datasets"]) == original_count


class TestCreateMultiscaleMetadata:
    """Tests for create_multiscale_metadata."""

    def test_returns_metadata_without_outpath(self):
        result = create_multiscale_metadata(
            ds_name="test",
            voxel_size=[4.0, 4.0, 4.0],
            translation=[0.0, 0.0, 0.0],
            units="nanometer",
            axes=["z", "y", "x"],
        )
        assert result is not None
        assert "multiscales" in result

    def test_with_extra_levels(self):
        result = create_multiscale_metadata(
            ds_name="test",
            voxel_size=[4.0, 4.0, 4.0],
            translation=[0.0, 0.0, 0.0],
            units="nanometer",
            axes=["z", "y", "x"],
            levels_to_add=2,
        )
        datasets = result["multiscales"][0]["datasets"]
        assert len(datasets) == 3

    def test_writes_to_file(self, tmp_path):
        out_path = str(tmp_path / "zattrs.json")
        result = create_multiscale_metadata(
            ds_name="test",
            voxel_size=[4.0, 4.0, 4.0],
            translation=[0.0, 0.0, 0.0],
            units="nanometer",
            axes=["z", "y", "x"],
            out_path=out_path,
        )
        # When out_path given, should return None and write file
        assert result is None
        assert os.path.exists(out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert "multiscales" in data


class TestWriteMetadata:
    """Tests for write_metadata."""

    def test_writes_valid_json(self, tmp_path):
        z_attrs = {"multiscales": [{"version": "0.4", "name": "test"}]}
        out_path = str(tmp_path / "metadata.json")
        write_metadata(z_attrs, out_path)
        assert os.path.exists(out_path)
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded == z_attrs

    def test_overwrites_existing_file(self, tmp_path):
        out_path = str(tmp_path / "metadata.json")
        write_metadata({"version": "old"}, out_path)
        write_metadata({"version": "new"}, out_path)
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["version"] == "new"

    def test_indented_output(self, tmp_path):
        z_attrs = {"multiscales": [{"version": "0.4"}]}
        out_path = str(tmp_path / "indented.json")
        write_metadata(z_attrs, out_path)
        with open(out_path) as f:
            content = f.read()
        # Should be pretty-printed (indented)
        assert "\n" in content


class TestFindLevel:
    """Tests for find_level."""

    @pytest.fixture
    def multiscale_zarr(self, tmp_path):
        """Create a Zarr group with multiple scale levels."""
        store = zarr.DirectoryStore(str(tmp_path / "test.zarr"))
        root = zarr.group(store=store, overwrite=True)

        # Create two scale levels
        root.create_dataset("s0", data=np.zeros((64, 64, 64), dtype=np.float32))
        root.create_dataset("s1", data=np.zeros((32, 32, 32), dtype=np.float32))

        root.attrs["multiscales"] = [
            {
                "version": "0.4",
                "axes": [
                    {"name": "z", "type": "space", "unit": "nanometer"},
                    {"name": "y", "type": "space", "unit": "nanometer"},
                    {"name": "x", "type": "space", "unit": "nanometer"},
                ],
                "datasets": [
                    {
                        "path": "s0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [4.0, 4.0, 4.0]},
                            {"type": "translation", "translation": [0.0, 0.0, 0.0]},
                        ],
                    },
                    {
                        "path": "s1",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [8.0, 8.0, 8.0]},
                            {"type": "translation", "translation": [2.0, 2.0, 2.0]},
                        ],
                    },
                ],
            }
        ]
        return str(tmp_path / "test.zarr")

    def test_find_fine_level(self, multiscale_zarr):
        # Target scale smaller than s0 -> should return s0
        level = find_level(multiscale_zarr, {"z": 2.0, "y": 2.0, "x": 2.0})
        assert level == "s0"

    def test_find_coarse_level(self, multiscale_zarr):
        # Target scale between s0 and s1 -> should return s0 (last level not exceeding target)
        level = find_level(multiscale_zarr, {"z": 6.0, "y": 6.0, "x": 6.0})
        assert level == "s0"

    def test_find_last_level(self, multiscale_zarr):
        # Target scale larger than all levels -> should return last level
        level = find_level(multiscale_zarr, {"z": 100.0, "y": 100.0, "x": 100.0})
        assert level == "s1"
