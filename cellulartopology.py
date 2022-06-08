import numpy
import typing
import itertools


class Neighborhood:
    ...


class Moore(Neighborhood):
    """A Moore (4) neighborhood on a grid"""

    @classmethod
    def neighbors(c, multiindex: typing.Sequence[int]):
        """Iterate over the neighbors.

        >>> sorted(Moore.neighbors((0, 0)))
        [(-1, 0), (0, -1), (0, 1), (1, 0)]
        """
        multiindex = tuple(multiindex)
        for i in range(len(multiindex)):
            for dxi in [-1, +1]:
                yield multiindex[:i] + (multiindex[i] + dxi,) + (multiindex[i + 1 :])


class VonNeumann(Neighborhood):
    """A von Neumann (8) neighborhood on a grid"""

    @classmethod
    def neighbors(c, multiindex: typing.Sequence[int]):
        """Iterate over the neighbors.

        >>> sorted(VonNeumann.neighbors((0, 0)))
        [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        """
        multiindex = numpy.asarray(multiindex)
        for dx in itertools.product([-1, 0, 1], repeat=len(multiindex)):
            yield tuple(multiindex + dx)


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
