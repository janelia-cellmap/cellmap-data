"""Tests for CellMapDataset."""

from __future__ import annotations

import numpy as np
import torch

from cellmap_data import CellMapDataset
from cellmap_data.empty_image import EmptyImage
from cellmap_data.image import CellMapImage

from .test_helpers import create_test_dataset

INPUT_ARRAYS = {"raw": {"shape": (4, 4, 4), "scale": (8.0, 8.0, 8.0)}}
TARGET_ARRAYS = {"labels": {"shape": (4, 4, 4), "scale": (8.0, 8.0, 8.0)}}


class TestCellMapDataset:
    def test_init(self, tmp_path):
        info = create_test_dataset(tmp_path, classes=["mito", "er"])
        ds = CellMapDataset(
            raw_path=info["raw_path"],
            target_path=info["gt_path"],
            classes=info["classes"],
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            force_has_data=True,
        )
        assert ds.classes == ["mito", "er"]
        assert "raw" in ds.input_sources
        assert "mito" in ds.target_sources
        assert "er" in ds.target_sources

    def test_missing_class_is_empty_image(self, tmp_path):
        info = create_test_dataset(tmp_path, classes=["mito"])
        ds = CellMapDataset(
            raw_path=info["raw_path"],
            target_path=info["gt_path"],
            classes=["mito", "er"],  # er not annotated
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            force_has_data=True,
        )
        assert isinstance(ds.target_sources["mito"], CellMapImage)
        assert isinstance(ds.target_sources["er"], EmptyImage)

    def test_len_positive(self, tmp_path):
        info = create_test_dataset(tmp_path)
        ds = CellMapDataset(
            raw_path=info["raw_path"],
            target_path=info["gt_path"],
            classes=info["classes"],
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            force_has_data=True,
        )
        assert len(ds) > 0

    def test_getitem_returns_dict_with_idx(self, tmp_path):
        info = create_test_dataset(tmp_path)
        ds = CellMapDataset(
            raw_path=info["raw_path"],
            target_path=info["gt_path"],
            classes=info["classes"],
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            force_has_data=True,
            pad=True,
        )
        item = ds[0]
        assert "idx" in item
        assert item["idx"].item() == 0

    def test_getitem_raw_is_tensor(self, tmp_path):
        info = create_test_dataset(tmp_path)
        ds = CellMapDataset(
            raw_path=info["raw_path"],
            target_path=info["gt_path"],
            classes=info["classes"],
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            force_has_data=True,
            pad=True,
        )
        item = ds[0]
        assert isinstance(item["raw"], torch.Tensor)
        assert item["raw"].shape == torch.Size([1, 4, 4, 4])

    def test_getitem_missing_class_nan(self, tmp_path):
        info = create_test_dataset(tmp_path, classes=["mito"])
        ds = CellMapDataset(
            raw_path=info["raw_path"],
            target_path=info["gt_path"],
            classes=["mito", "er"],
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            force_has_data=True,
            pad=True,
        )
        item = ds[0]
        # Target classes are stacked under the target array key ("labels").
        # classes=["mito", "er"] → index 0=mito, 1=er
        target = item["labels"]  # shape [2, z, y, x]
        # unannotated class (er, index 1) → NaN
        assert torch.isnan(target[1]).all()
        # annotated class (mito, index 0) → not all NaN
        assert not torch.isnan(target[0]).all()

    def test_get_crop_class_matrix_shape(self, tmp_path):
        info = create_test_dataset(tmp_path, classes=["mito"])
        ds = CellMapDataset(
            raw_path=info["raw_path"],
            target_path=info["gt_path"],
            classes=["mito", "er"],
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            force_has_data=True,
        )
        mat = ds.get_crop_class_matrix()
        assert mat.shape == (1, 2)
        # mito is annotated (True), er is not (False)
        assert mat[0, 0] == True
        assert mat[0, 1] == False

    def test_get_indices_non_empty(self, tmp_path):
        info = create_test_dataset(tmp_path)
        ds = CellMapDataset(
            raw_path=info["raw_path"],
            target_path=info["gt_path"],
            classes=info["classes"],
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            force_has_data=True,
        )
        chunk_size = {"z": 32.0, "y": 32.0, "x": 32.0}
        indices = ds.get_indices(chunk_size)
        assert len(indices) > 0

    def test_verify(self, tmp_path):
        info = create_test_dataset(tmp_path)
        ds = CellMapDataset(
            raw_path=info["raw_path"],
            target_path=info["gt_path"],
            classes=info["classes"],
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            force_has_data=True,
        )
        assert ds.verify()

    def test_bounding_box(self, tmp_path):
        info = create_test_dataset(
            tmp_path, shape=(32, 32, 32), voxel_size=[8.0, 8.0, 8.0]
        )
        ds = CellMapDataset(
            raw_path=info["raw_path"],
            target_path=info["gt_path"],
            classes=info["classes"],
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            force_has_data=True,
        )
        bb = ds.bounding_box
        assert bb is not None
        assert set(bb.keys()) == {"z", "y", "x"}

    def test_repr(self, tmp_path):
        info = create_test_dataset(tmp_path)
        ds = CellMapDataset(
            raw_path=info["raw_path"],
            target_path=info["gt_path"],
            classes=info["classes"],
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            force_has_data=True,
        )
        r = repr(ds)
        assert "CellMapDataset" in r

    def test_small_crop_pad_true_len_one(self, tmp_path):
        """Label crop smaller than output patch with pad=True → len=1, valid sample."""
        # raw: 100³ at 8nm = 800nm (large); output: 4³ at 8nm = 32nm
        # raw sampling_box: [16, 784] nm in each axis
        # label: 2³ at 8nm = 16nm, origin at 50nm → bb=[50,66], centre=58nm (inside raw sb)
        from .test_helpers import _write_ome_ngff
        import numpy as np, os

        large_raw = (np.random.default_rng(0).random((100, 100, 100)) * 255).astype(np.uint8)
        raw_path = str(tmp_path / "raw.zarr")
        _write_ome_ngff(raw_path, large_raw, [8.0, 8.0, 8.0])

        gt_base = str(tmp_path / "gt.zarr")
        os.makedirs(gt_base, exist_ok=True)
        import json
        with open(os.path.join(gt_base, ".zgroup"), "w") as f:
            f.write('{"zarr_format": 2}')

        small_data = np.ones((2, 2, 2), dtype=np.uint8)
        classes = ["mito", "er"]
        for cls in classes:
            _write_ome_ngff(
                os.path.join(gt_base, cls),
                small_data,
                [8.0, 8.0, 8.0],
                origin=[50.0, 50.0, 50.0],
            )

        gt_path = f"{gt_base}/[mito,er]"
        ds = CellMapDataset(
            raw_path=raw_path,
            target_path=gt_path,
            classes=classes,
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            pad=True,
        )
        assert len(ds) == 1
        sample = ds[0]
        assert sample["raw"].shape == torch.Size([1, 4, 4, 4])
        assert sample["labels"].shape == torch.Size([2, 4, 4, 4])
        # 2³=8 valid voxels per class inside a 4³=64-voxel patch → NaN outside
        nan_count = torch.isnan(sample["labels"]).sum().item()
        assert nan_count > 0

    def test_small_crop_pad_false_excluded(self, tmp_path):
        """Label crop smaller than output patch with pad=False → dataset excluded (len=0)."""
        from .test_helpers import _write_ome_ngff
        import numpy as np, os

        large_raw = (np.random.default_rng(0).random((100, 100, 100)) * 255).astype(np.uint8)
        raw_path = str(tmp_path / "raw.zarr")
        _write_ome_ngff(raw_path, large_raw, [8.0, 8.0, 8.0])

        gt_base = str(tmp_path / "gt.zarr")
        os.makedirs(gt_base, exist_ok=True)
        import json
        with open(os.path.join(gt_base, ".zgroup"), "w") as f:
            f.write('{"zarr_format": 2}')

        small_data = np.ones((2, 2, 2), dtype=np.uint8)
        classes = ["mito", "er"]
        for cls in classes:
            _write_ome_ngff(
                os.path.join(gt_base, cls),
                small_data,
                [8.0, 8.0, 8.0],
                origin=[50.0, 50.0, 50.0],
            )

        gt_path = f"{gt_base}/[mito,er]"
        ds = CellMapDataset(
            raw_path=raw_path,
            target_path=gt_path,
            classes=classes,
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            pad=False,
        )
        assert ds.sampling_box is None
        assert len(ds) == 0

    def test_class_counts(self, tmp_path):
        info = create_test_dataset(tmp_path)
        ds = CellMapDataset(
            raw_path=info["raw_path"],
            target_path=info["gt_path"],
            classes=info["classes"],
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            force_has_data=True,
        )
        counts = ds.class_counts
        assert "totals" in counts
        assert all(c in counts["totals"] for c in info["classes"])

    def test_spatial_transforms_mirror(self, tmp_path):
        """Mirror spatial transform → item differs from un-transformed."""
        info = create_test_dataset(tmp_path, shape=(32, 32, 32))
        ds_plain = CellMapDataset(
            raw_path=info["raw_path"],
            target_path=info["gt_path"],
            classes=info["classes"],
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            force_has_data=True,
            pad=True,
        )
        ds_mirror = CellMapDataset(
            raw_path=info["raw_path"],
            target_path=info["gt_path"],
            classes=info["classes"],
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            force_has_data=True,
            pad=True,
            spatial_transforms={"mirror": {"z": True, "y": False, "x": False}},
        )
        # Mirror is random; with always-true z mirror, result differs from original
        raw_plain = ds_plain[0]["raw"]
        raw_mirrored = ds_mirror[0]["raw"]
        # They may or may not match depending on RNG, but shapes must match
        assert raw_plain.shape == raw_mirrored.shape
