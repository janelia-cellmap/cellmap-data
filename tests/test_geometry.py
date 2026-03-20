"""Tests for geometry utilities."""

from __future__ import annotations

from cellmap_data.utils.geometry import box_intersection, box_shape, box_union


class TestBoxIntersection:
    def test_overlap(self):
        a = {"z": (0.0, 100.0), "y": (0.0, 100.0)}
        b = {"z": (50.0, 150.0), "y": (50.0, 150.0)}
        result = box_intersection(a, b)
        assert result == {"z": (50.0, 100.0), "y": (50.0, 100.0)}

    def test_no_overlap_returns_none(self):
        a = {"z": (0.0, 50.0)}
        b = {"z": (60.0, 100.0)}
        assert box_intersection(a, b) is None

    def test_touching_returns_none(self):
        a = {"z": (0.0, 50.0)}
        b = {"z": (50.0, 100.0)}
        assert box_intersection(a, b) is None  # lo >= hi

    def test_one_contains_other(self):
        a = {"z": (0.0, 200.0)}
        b = {"z": (50.0, 150.0)}
        result = box_intersection(a, b)
        assert result == {"z": (50.0, 150.0)}

    def test_missing_axis_skipped(self):
        a = {"z": (0.0, 100.0), "y": (0.0, 100.0)}
        b = {"z": (10.0, 90.0)}  # no y key
        result = box_intersection(a, b)
        assert result == {"z": (10.0, 90.0)}

    def test_empty_result_returns_none(self):
        # No shared axes at all
        a = {"z": (0.0, 100.0)}
        b = {"y": (0.0, 100.0)}
        assert box_intersection(a, b) is None


class TestBoxUnion:
    def test_same_boxes(self):
        a = {"z": (0.0, 100.0)}
        result = box_union(a, a)
        assert result == a

    def test_disjoint(self):
        a = {"z": (0.0, 50.0)}
        b = {"z": (70.0, 120.0)}
        result = box_union(a, b)
        assert result == {"z": (0.0, 120.0)}

    def test_missing_axis_in_one(self):
        a = {"z": (0.0, 100.0)}
        b = {"y": (5.0, 50.0)}
        result = box_union(a, b)
        assert result["z"] == (0.0, 100.0)
        assert result["y"] == (5.0, 50.0)


class TestBoxShape:
    def test_basic(self):
        box = {"z": (0.0, 160.0), "y": (0.0, 160.0), "x": (0.0, 160.0)}
        scale = {"z": 8.0, "y": 8.0, "x": 8.0}
        result = box_shape(box, scale)
        assert result == {"z": 20, "y": 20, "x": 20}

    def test_min_one(self):
        box = {"z": (0.0, 4.0)}
        scale = {"z": 8.0}
        # 4/8 = 0.5 → rounds to 1 (at least 1)
        assert box_shape(box, scale)["z"] == 1

    def test_non_integer_rounds(self):
        box = {"z": (0.0, 12.0)}
        scale = {"z": 8.0}
        # 12/8 = 1.5 → rounds to 2
        assert box_shape(box, scale)["z"] == 2
