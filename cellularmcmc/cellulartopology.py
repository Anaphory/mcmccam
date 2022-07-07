"""Topology helper methods.

This module contains helper functions to generate grid topology matrices.

"""
import abc
import itertools
import typing

import numpy


class Neighborhood(abc.ABC):
    """Base class for neighborhoods."""

    @classmethod
    @abc.abstractmethod
    def neighbors(
        c, multiindex: typing.Sequence[int]
    ) -> typing.Iterable[typing.Sequence[int]]:
        """Return the neighbor indices."""
        raise NotImplementedError()


class Moore(Neighborhood):
    """A Moore (4) neighborhood on a grid.

    This neighborhood actually works for higher-dimensional grids, too.
    """

    @classmethod
    def neighbors(
        c, multiindex: typing.Sequence[int]
    ) -> typing.Iterable[typing.Sequence[int]]:
        """Iterate over the neighbors.

        >>> sorted(Moore.neighbors((0, 0)))
        [(-1, 0), (0, -1), (0, 1), (1, 0)]
        """
        multiindex = tuple(multiindex)
        for i in range(len(multiindex)):
            for dxi in [-1, +1]:
                yield multiindex[:i] + (multiindex[i] + dxi,) + (multiindex[i + 1 :])


class VonNeumann(Neighborhood):
    """A von Neumann (8) neighborhood on a grid.

    This neighborhood actually works for higher-dimensional grids, too.
    """

    @classmethod
    def neighbors(
        c, multiindex: typing.Sequence[int]
    ) -> typing.Iterable[typing.Sequence[int]]:
        """Iterate over the neighbors.

        >>> sorted(VonNeumann.neighbors((0, 0)))
        [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        """
        multiindex = numpy.asarray(multiindex)
        for dx in itertools.product([-1, 0, 1], repeat=len(multiindex)):
            if dx.any():
                yield tuple(multiindex + dx)


class PseudoHex(Neighborhood):
    """A hexagon-like neighborhood on a grid.

    The grid generated is one like this:

    0 0 0 0 0 0
     0 0 0 0 0 0
    0 0 0 0 0 0
     0 0 0 0 0 0
    0 0 0 0 0 0
     0 0 0 0 0 0

    As opposed to other neighborhoods, this is implemented only for
    2-dimensional grids – dense lattice hypersphere packings in arbitrary
    dimensions are just impossible to generalize.

    """

    @classmethod
    def neighbors(
        c, multiindex: typing.Sequence[int]
    ) -> typing.Iterable[typing.Sequence[int]]:
        """Iterate over the neighbors.

        Raises
        ======

        TypeError when multiindex has not exactly 2 entries.

        The internals may be a bit confusing, but you can see that neighbors-of-neighbors are exactly 1+6+12:

             2 2 2
            2 1 1 2
           2 1 0 1 2
            2 1 1 2
             2 2 2

        >>> len({c2 for c1 in PseudoHex.neighbors((0, 0)) for c2 in PseudoHex.neighbors((c1))})
        19
        >>> len({c2 for c1 in PseudoHex.neighbors((0, 1)) for c2 in PseudoHex.neighbors((c1))})
        19


        """
        x, y = multiindex
        if y % 2:
            return [
                (x - 1, y),
                (x, y - 1),
                (x, y + 1),
                (x + 1, y - 1),
                (x + 1, y),
                (x + 1, y + 1),
            ]
        else:
            return [
                (x - 1, y - 1),
                (x - 1, y),
                (x - 1, y + 1),
                (x, y - 1),
                (x, y + 1),
                (x + 1, y),
            ]


def make_block(rows, columns, neighborhood: Neighborhood = VonNeumann()):
    """Return the rate matrix for a square grid.

    >>> make_block(3, 3, Moore())
    array([[0., 1., 0., 1., 0., 0., 0., 0., 0.],
           [1., 0., 1., 0., 1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 1., 0., 0., 0.],
           [1., 0., 0., 0., 1., 0., 1., 0., 0.],
           [0., 1., 0., 1., 0., 1., 0., 1., 0.],
           [0., 0., 1., 0., 1., 0., 0., 0., 1.],
           [0., 0., 0., 1., 0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1., 0., 1., 0., 1.],
           [0., 0., 0., 0., 0., 1., 0., 1., 0.]])

    """
    array = numpy.zeros((rows * columns, rows * columns))
    for i in range(rows * columns):
        effective_i = numpy.unravel_index(i, (rows, columns))
        neighbors = neighborhood.neighbors(effective_i)
        for n in neighbors:
            try:
                j = numpy.ravel_multi_index(n, (rows, columns))
                array[i, j] = 1
            except ValueError:
                # Invalid index
                continue
    return array
