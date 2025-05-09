# Modified from funlib.geometry to allow float coordinates
# @rhoadesScholar, HHMI Janelia Research Campus 2025

import numbers
from typing import Any, Iterable, Union

import numpy as np


class Coordinate(tuple):
    """A ``tuple`` of integers.

    Allows the following element-wise operators: addition, subtraction,
    multiplication, division, absolute value, and negation. All operations are
    applied element wise and support both Coordinates and Numbers. This allows to
    perform simple arithmetics with coordinates, e.g.::

        shape = Coordinate(2, 3, 4)
        voxel_size = Coordinate(10, 5, 1)
        size = shape*voxel_size # == Coordinate(20, 15, 4)
        size * 2 + 1 # == Coordinate(41, 31, 9)

    Coordinates can be initialized with any iterable of ints or floats, e.g.::

        Coordinate((1,2,3))
        Coordinate([1.,2.5,3.])
        Coordinate(np.array([1.,2.5,3]))

    Coordinates can also pack multiple args into an iterable, e.g.::

        Coordinate(1,2,3)
    """

    def __new__(cls, *array_like):
        if len(array_like) == 1 and isinstance(array_like[0], Iterable):
            array_like = array_like[0]
        return super(Coordinate, cls).__new__(
            cls, [float(x) if x is not None else None for x in array_like]
        )

    @property
    def dims(self) -> int:
        return len(self)

    def squeeze(self, dim: int = 0) -> "Coordinate":
        return Coordinate(c for i, c in enumerate(self) if i != dim)

    def is_multiple_of(self, coordinate: "Coordinate") -> bool:
        """Test if this coordinate is a multiple of the given coordinate."""

        return all([np.isclose(a % b, 0) for a, b in zip(self, coordinate)])

    def round_division(self, other: "Coordinate") -> "Coordinate":
        """
        Will always round down if self % other == other / 2.
        """
        return (self + (other - 1) // 2) // other

    def floor_division(self, other: "Coordinate") -> "Coordinate":
        return self // other

    def ceil_division(self, other: "Coordinate") -> "Coordinate":
        return (self + other - 1) // other

    def __neg__(self) -> "Coordinate":
        return Coordinate(-a if a is not None else None for a in self)

    def __abs__(self) -> "Coordinate":
        return Coordinate(abs(a) if a is not None else None for a in self)

    def __add__(self, other: Union[Any, "Coordinate", int, float]) -> "Coordinate":
        if isinstance(other, Coordinate):
            assert (
                self.dims == other.dims
            ), "can only add Coordinate of equal dimensions"
            return Coordinate(
                a + b if a is not None and b is not None else None
                for a, b in zip(self, other)
            )

        elif isinstance(other, numbers.Number):
            return Coordinate(a + other if a is not None else None for a in self)

        else:
            raise TypeError(
                "addition of Coordinate with type %s not supported" % type(other)
            )

    def __sub__(self, other: Union["Coordinate", int, float]) -> "Coordinate":
        if isinstance(other, Coordinate):
            assert (
                self.dims == other.dims
            ), "can only subtract Coordinate of equal dimensions"
            return Coordinate(
                a - b if a is not None and b is not None else None
                for a, b in zip(self, other)
            )

        elif isinstance(other, numbers.Number):
            return Coordinate(a - other if a is not None else None for a in self)

        else:
            raise TypeError(
                "subtraction of Coordinate with type %s not supported" % type(other)
            )

    def __mul__(self, other: Union[Any, "Coordinate", int, float]) -> "Coordinate":
        if isinstance(other, Coordinate):
            assert (
                self.dims == other.dims
            ), "can only multiply Coordinate of equal dimensions"

            return Coordinate(
                a * b if a is not None and b is not None else None
                for a, b in zip(self, other)
            )

        elif isinstance(other, numbers.Number):
            return Coordinate(a * other if a is not None else None for a in self)

        else:
            raise TypeError(
                "multiplication of Coordinate with type %s not supported" % type(other)
            )

    def __div__(self, other: Union["Coordinate", int, float]) -> "Coordinate":
        if isinstance(other, Coordinate):
            assert (
                self.dims == other.dims
            ), "can only divide Coordinate of equal dimensions"

            return Coordinate(
                a / b if a is not None and b is not None else None
                for a, b in zip(self, other)
            )

        elif isinstance(other, numbers.Number):
            return Coordinate(a / other if a is not None else None for a in self)

        else:
            raise TypeError(
                "division of Coordinate with type %s not supported" % type(other)
            )

    def __truediv__(self, other: Union["Coordinate", int, float]) -> "Coordinate":
        if isinstance(other, Coordinate):
            assert (
                self.dims == other.dims
            ), "can only divide Coordinate of equal dimensions"

            return Coordinate(
                a / b if a is not None and b is not None else None
                for a, b in zip(self, other)
            )

        elif isinstance(other, numbers.Number):
            return Coordinate(a / other if a is not None else None for a in self)

        else:
            raise TypeError(
                "division of Coordinate with type %s not supported" % type(other)
            )

    def __floordiv__(self, other: Union["Coordinate", int, float]) -> "Coordinate":
        if isinstance(other, Coordinate):
            assert (
                self.dims == other.dims
            ), "can only divide Coordinate of equal dimensions"

            return Coordinate(
                a // b if a is not None and b is not None else None
                for a, b in zip(self, other)
            )

        elif isinstance(other, numbers.Number):
            return Coordinate(a // other if a is not None else None for a in self)

        else:
            raise TypeError(
                "division of Coordinate with type %s not supported" % type(other)
            )

    def __mod__(self, other: Union["Coordinate", int, float]) -> "Coordinate":
        if isinstance(other, Coordinate):
            assert (
                self.dims == other.dims
            ), "can only mod Coordinate of equal dimensions"

            return Coordinate(
                a % b if a is not None and b is not None else None
                for a, b in zip(self, other)
            )

        elif isinstance(other, numbers.Number):
            return Coordinate(a % other if a is not None else None for a in self)

        else:
            raise TypeError(
                "mod of Coordinate with type %s not supported" % type(other)
            )

    def __pow__(self, other: Union["Coordinate", int, float]) -> "Coordinate":
        if isinstance(other, Coordinate):
            assert (
                self.dims == other.dims
            ), "can only raise to Coordinate of equal dimensions"

            return Coordinate(
                a**b if a is not None and b is not None else None
                for a, b in zip(self, other)
            )

        elif isinstance(other, numbers.Number):
            return Coordinate(a**other if a is not None else None for a in self)

        else:
            raise TypeError(
                "raising Coordinate with type %s not supported" % type(other)
            )
