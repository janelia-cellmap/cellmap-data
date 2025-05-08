# Modified from funlib.geometry to allow float coordinates
# @rhoadesScholar, HHMI Janelia Research Campus 2025

import copy
import logging
import numbers
from typing import Iterable, Optional, Tuple, Union

import numpy as np

from .coordinate import Coordinate

logger = logging.getLogger(__file__)


class Roi:
    """A rectangular region of interest, defined by an offset and a shape.
    Special Cases:
        An infinite/unbounded ROI:
            offset = (None, None, ...)
            shape = (None, None, ...)

        An empty ROI (e.g. output of intersecting two non overlapping Rois):
            offset = (None, None, ...)
            shape = (0, 0, ...)

        A ROI that only specifies a shape is not supported (just use
        Coordinate).

        There is no guessing size of offset or shape (expanding to number of
        dims of the other).

    Basic Operations:
        Addition/subtraction (Coordinate, int, or float) - shifts the offset
        elementwise (alias for shift)

        Multiplication/division (Coordinate, int, or float) - multiplies/divides the
        offset and the shape, elementwise

    Roi Operations:
        Intersect, union

    Similar to :class:`Coordinate`, supports simple arithmetics, e.g.::

        roi = Roi((1., 1., 1), (10., 10., 10))
        voxel_size = Coordinate((10, 5, 1))
        roi * voxel_size = Roi((10, 5, 1), (100, 50, 10))
        scale_shift = roi*voxel_size + 1 # == Roi((11, 6, 2), (101, 51, 11))

    Args:

        offset (array-like of ``int`` or ``float``):

            The offset of the ROI. Entries can be ``None`` to indicate
            there is no offset (either unbounded or empty).

        shape (array-like of ``int`` or ``float``):

            The shape of the ROI. Entries can be ``None`` to indicate
            unboundedness.
    """

    def __init__(
        self,
        offset: Iterable[Optional[int | float]],
        shape: Iterable[Optional[int | float]],
    ):
        self.__offset = Coordinate(offset)
        self.__shape = Coordinate(shape)

        self.__consolidate_offset()

    def squeeze(self, dim: int = 0) -> "Roi":
        return Roi(self.offset.squeeze(dim), self.shape.squeeze(dim))

    @property
    def offset(self) -> Coordinate:
        return self.__offset

    @offset.setter
    def offset(self, offset: Iterable[Optional[int | float]]) -> None:
        """Set the offset of this Roi.

        Args:
            offset (array-like):

                The new offset.  Entries can be ``None``` to indicate
                unboundedness or empty ROI.
        """

        self.__offset = Coordinate(offset)
        self.__consolidate_offset()

    def get_offset(self) -> Coordinate:
        return self.offset

    def set_offset(self, new_offset: Iterable[Optional[int | float]]) -> None:
        self.offset = Coordinate(new_offset)

    @property
    def shape(self) -> Coordinate:
        return self.__shape

    @shape.setter
    def shape(self, shape: Iterable[Optional[int | float]]) -> None:
        """Set the shape of this ROI.

        Args:

            shape (array-like or ``None``):

                The new shape. Entries can be ``None`` to indicate
                unboundedness.
        """

        self.__shape = Coordinate(shape)
        self.__consolidate_offset()

    def get_shape(self) -> Coordinate:
        return self.shape

    def set_shape(self, new_shape: Iterable[Optional[int | float]]) -> None:
        self.shape = Coordinate(new_shape)

    def __consolidate_offset(self) -> None:
        """Ensure that offset and shape have same number of dimensions and
        offsets for unbounded or empty dimensions are None."""

        assert (
            self.__offset.dims == self.__shape.dims
        ), "offset dimension %d != shape dimension %d" % (
            self.__offset.dims,
            self.__shape.dims,
        )

        self.__offset = Coordinate(
            (o if s is not None else None for o, s in zip(self.__offset, self.__shape))
        )

    @property
    def begin(self) -> Coordinate:
        """Smallest coordinate inside ROI."""
        return self.__offset

    def get_begin(self) -> Coordinate:
        return self.begin

    @property
    def end(self) -> Coordinate:
        """Smallest coordinate which is component-wise larger than any
        inside ROI."""
        return self.__offset + self.__shape

    def get_end(self) -> Coordinate:
        return self.end

    @property
    def center(self) -> Coordinate:
        """Get the center of this ROI."""
        return self.__offset + self.__shape / 2

    def get_center(self) -> Coordinate:
        return self.center

    def to_slices(
        self, grid_size: Union[Coordinate, int, float, Iterable[int | float]] = 1
    ) -> Tuple[slice, ...]:
        """
        Get a ``tuple`` of ``slice`` that represent this ROI and can be used
        to index arrays.

        Args:

            grid_size (:class:`Coordinate` or ``int`` or ``float`` or ``None``, optional):

            The size of the grid to divide coordinates by. If ``None``,
            coordinates are simply rounded outward to ensure full ROI
            coverage. If a number, it is used for all dimensions.
        """
        if grid_size is None:
            grid_size = 1
        if isinstance(grid_size, numbers.Number):
            grid_size = Coordinate((grid_size,) * self.dims)
        elif not isinstance(grid_size, Coordinate) and isinstance(grid_size, Iterable):
            grid_size = Coordinate(grid_size)
        assert isinstance(
            grid_size, (Coordinate, int, float)
        ), "grid_size must be a Coordinate or a number"

        slices = []
        for d in range(self.dims):
            if self.__shape[d] is None:
                s = slice(None, None)
            elif self.__shape[d] == 0:
                s = slice(None, 0)
            else:
                start = int(np.floor(self.__offset[d] / grid_size[d]))  # type: ignore
                end = int(np.ceil((self.__offset[d] + self.__shape[d]) / grid_size[d]))  # type: ignore
                s = slice(start, end)
            slices.append(s)

        return tuple(slices)

    def get_bounding_box(self) -> Tuple[slice, ...]:
        """Alias for ``to_slices()``."""

        return self.to_slices()

    @property
    def dims(self) -> int:
        """The the number of dimensions of this ROI."""

        return self.__shape.dims

    @property
    def size(self) -> Optional[float]:
        """Get the volume of this ROI. Returns ``None`` if the ROI is
        unbounded."""

        if self.unbounded:
            return None

        size = 1
        for d in self.__shape:
            size *= d
        return size

    def get_size(self) -> Optional[float]:
        return self.size

    @property
    def empty(self) -> bool:
        """Test if this ROI is empty."""

        return any([x is not None and x <= 0 for x in self.shape])

    @property
    def unbounded(self) -> bool:
        """Test if this ROI is unbounded."""

        return None in self.__shape

    def contains(self, other: Union["Roi", Iterable[Optional[int | float]]]) -> bool:
        """Test if this ROI contains ``other``, which can be another
        :class:`Roi`, :class:`Coordinate`, or ``tuple``."""

        if isinstance(other, Roi):
            if other.empty:
                # gunpowder expects empty rois to contain empty
                return self.empty or self.contains(other.begin)

            return self.contains(other.begin) and self.contains(other.end - 1)

        elif isinstance(other, Iterable):
            axis_containment = [
                (b is None or (p is not None and p >= b))
                and (e is None or (p is not None and p < e))
                for p, b, e in zip(other, self.begin, self.end)
            ]
            return len(axis_containment) == self.dims and all(axis_containment)
        else:
            raise Exception(
                f"cannot compute containment on object of type: {type(other)}"
            )

    def intersects(self, other: "Roi") -> bool:
        """Test if this ROI intersects with another :class:`Roi`."""

        assert self.dims == other.dims

        if self.empty or other.empty:
            return False

        # separated if at least one dimension is separated
        separated = any(
            [
                # a dimension is separated if:
                # none of the shapes is unbounded
                (None not in [b1, b2, e1, e2])
                and (
                    # either b1 starts after e2
                    (b1 >= e2)
                    or
                    # or b2 starts after e1
                    (b2 >= e1)
                )
                for b1, b2, e1, e2 in zip(self.begin, other.begin, self.end, other.end)
            ]
        )

        return not separated

    def intersect(self, other: "Roi") -> "Roi":
        """Get the intersection of this ROI with another :class:`Roi`."""

        if not self.intersects(other):
            return Roi((None,) * self.dims, (0,) * self.dims)  # empty ROI

        begin = Coordinate(
            (self.__left_max(b1, b2) for b1, b2 in zip(self.begin, other.begin))
        )
        end = Coordinate(
            (self.__right_min(e1, e2) for e1, e2 in zip(self.end, other.end))
        )

        return Roi(begin, end - begin)

    def union(self, other: "Roi") -> "Roi":
        """Get the union of this ROI with another :class:`Roi`."""

        if self.empty:
            return other

        if other.empty:
            return self

        begin = Coordinate(
            (self.__left_min(b1, b2) for b1, b2 in zip(self.begin, other.begin))
        )
        end = Coordinate(
            (self.__right_max(e1, e2) for e1, e2 in zip(self.end, other.end))
        )

        return Roi(begin, end - begin)

    def shift(self, by: Union["Coordinate", int, float]) -> "Roi":
        """Shift this ROI."""

        return Roi(self.__offset + by, self.__shape)

    def snap_to_grid(
        self, voxel_size: Iterable[Optional[int | float]], mode: str = "grow"
    ) -> "Roi":
        """Align a ROI with a given voxel size.

        Args:

            voxel_size (:class:`Coordinate` or ``tuple``):

                The voxel size of the grid to snap to.

            mode (string, optional):

                How to align the ROI if it is not a multiple of the voxel size.
                Available modes are 'grow', 'shrink', and 'closest'. Defaults
                to 'grow'.
        """
        if not isinstance(voxel_size, Coordinate):
            voxel_size = Coordinate(voxel_size)

        assert (
            voxel_size.dims == self.dims
        ), "dimension of voxel size does not match ROI"

        assert 0 not in voxel_size, "Voxel size cannot contain zero"

        if mode == "closest":
            begin_in_voxel = self.begin.round_division(voxel_size)
            end_in_voxel = self.end.round_division(voxel_size)
        elif mode == "grow":
            begin_in_voxel = self.begin.floor_division(voxel_size)
            end_in_voxel = self.end.ceil_division(voxel_size)
        elif mode == "shrink":
            begin_in_voxel = self.begin.ceil_division(voxel_size)
            end_in_voxel = self.end.floor_division(voxel_size)
        else:
            raise RuntimeError("Unknown mode %s for snap_to_grid" % mode)

        return Roi(
            begin_in_voxel * voxel_size, (end_in_voxel - begin_in_voxel) * voxel_size
        )

    def grow(
        self,
        amount_neg: Union[Iterable[Optional[int | float]], int | float] = 0,
        amount_pos: Union[Iterable[Optional[int | float]], int | float] = 0,
    ) -> "Roi":
        """Grow a ROI by the given amounts in each direction:

        Args:

            amount_neg (:class:`Coordinate` or ``int`` or ``float``):

                Amount (per dimension) to grow into the negative direction.
                Passing in a single integer grows that amount in all
                dimensions. Defaults to zero.

            amount_pos (:class:`Coordinate` or ``int`` or ``float``):

                Amount (per dimension) to grow into the positive direction.
                Passing in a single integer grows that amount in all
                dimensions. Defaults to zero.
        """
        if isinstance(amount_neg, Iterable):
            _amount_neg: Union[Coordinate, int, float] = Coordinate(amount_neg)
        else:
            _amount_neg = amount_neg
        if isinstance(amount_pos, Iterable):
            _amount_pos: Union[Coordinate, int, float] = Coordinate(amount_pos)
        else:
            _amount_pos = amount_pos

        offset = self.__offset - _amount_neg
        shape = self.__shape + _amount_neg + _amount_pos

        return Roi(offset, shape)

    def copy(self) -> "Roi":
        """Create a copy of this ROI."""
        return copy.deepcopy(self)

    def __left_min(self, x, y):
        # None is considered -inf

        if x is None or y is None:
            return None
        return min(x, y)

    def __left_max(self, x, y):
        # None is considered -inf

        if x is None:
            return y
        if y is None:
            return x
        return max(x, y)

    def __right_min(self, x, y):
        # None is considered +inf

        if x is None:
            return y
        if y is None:
            return x
        return min(x, y)

    def __right_max(self, x, y):
        # None is considered +inf

        if x is None or y is None:
            return None
        return max(x, y)

    def __add__(self, other: Union["Coordinate", int, float]) -> "Roi":
        if not isinstance(other, numbers.Number):
            other = Coordinate(other)
        assert isinstance(
            other, (Coordinate, int, float)
        ), "can only add number or Coordinate-like to Roi"
        return self.shift(other)

    def __sub__(self, other: Union["Coordinate", int, float]) -> "Roi":
        if not isinstance(other, numbers.Number):
            other = Coordinate(other)
        assert isinstance(
            other, (Coordinate, int, float)
        ), "can only subtract number or Coordinate-like from Roi"
        return self.shift(-other)

    def __mul__(self, other: Union["Coordinate", int, float]) -> "Roi":
        if not isinstance(other, numbers.Number):
            other = Coordinate(other)
        assert isinstance(
            other, (Coordinate, int, float)
        ), "can only multiply with a number or Coordinate-like"
        return Roi(self.__offset * other, self.__shape * other)

    def __div__(self, other: Union["Coordinate", int, float]) -> "Roi":
        if not isinstance(other, numbers.Number):
            other = Coordinate(other)
        assert isinstance(
            other, (Coordinate, int, float)
        ), "can only divide by a number or Coordinate-like"
        return Roi(self.__offset / other, self.__shape / other)

    def __truediv__(self, other: Union["Coordinate", int, float]) -> "Roi":
        if not isinstance(other, numbers.Number):
            other = Coordinate(other)
        assert isinstance(
            other, (Coordinate, int, float)
        ), "can only divide by a number or Coordinate-like"
        return Roi(self.__offset / other, self.__shape / other)

    def __floordiv__(self, other: Union["Coordinate", int, float]) -> "Roi":
        if not isinstance(other, numbers.Number):
            other = Coordinate(other)
        assert isinstance(
            other, (Coordinate, int, float)
        ), "can only divide by a number or Coordinate-like"
        return Roi(self.__offset // other, self.__shape // other)

    def __mod__(
        self, other: Union["Coordinate", int, float]
    ) -> "Roi":  # pragma: py3 no cover
        if not isinstance(other, numbers.Number):
            other = Coordinate(other)
        assert isinstance(
            other, (Coordinate, int, float)
        ), "can only mod by a number or Coordinate-like"
        return Roi(self.__offset % other, self.__shape % other)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Roi):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        if isinstance(other, Roi):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f"Roi({self.offset}, {self.shape})"

    def __str__(self) -> str:
        if self.empty:
            return "[empty ROI]"
        slices = ", ".join(
            [
                (str(b) if b is not None else "")
                + ":"
                + (str(e) if e is not None else "")
                for b, e in zip(self.begin, self.end)
            ]
        )
        dims = ", ".join(str(a) if a is not None else "inf" for a in self.__shape)
        return "[" + slices + "] (" + dims + ")"
