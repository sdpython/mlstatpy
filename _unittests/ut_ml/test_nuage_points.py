# -*- coding: utf-8 -*-
"""
@brief      test log(time=1s)
"""

import sys
import os
import unittest
import numpy
from numpy.testing import assert_array_equal
from sklearn.neighbors import NearestNeighbors
from pyquickhelper.loghelper import fLOG


try:
    import src
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..")))
    if path not in sys.path:
        sys.path.append(path)
    import src

from src.mlstatpy.ml.kppv import NuagePoints
from src.mlstatpy.ml.kppv_laesa import NuagePointsLaesa


class TestNuagePoints(unittest.TestCase):

    def test_nuage_points_1d(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        X = numpy.array([[0], [3], [1]])
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(X)
        dist, y = neigh.kneighbors(X)

        nuage = NuagePoints()
        nuage.fit(X)
        dist2, y2 = nuage.kneighbors(X)

        assert_array_equal(y.ravel(), y2.ravel())
        assert_array_equal(dist.ravel(), dist2.ravel())

    def test_nuage_points_1d_leasa(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        X = numpy.array([[0], [3], [1]])
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(X)
        dist, y = neigh.kneighbors(X)

        nuage = NuagePointsLaesa(2)
        nuage.fit(X)
        dist2, y2 = nuage.kneighbors(X)

        assert_array_equal(y.ravel(), y2.ravel())
        assert_array_equal(dist.ravel(), dist2.ravel())

    def test_nuage_points_2d(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        X = numpy.array([[0, 0], [3, 3], [1, 1]])
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(X)
        dist, y = neigh.kneighbors(X)

        nuage = NuagePoints()
        nuage.fit(X)
        dist2, y2 = nuage.kneighbors(X)

        assert_array_equal(y.ravel(), y2.ravel())
        assert_array_equal(dist.ravel(), dist2.ravel())

    def test_nuage_points_2d_leasa(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        X = numpy.array([[0, 0], [3, 3], [1, 1]])
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(X)
        dist, y = neigh.kneighbors(X)

        for k in [1, 2, 3]:
            nuage = NuagePointsLaesa(k)
            nuage.fit(X)
            dist2, y2 = nuage.kneighbors(X)

            assert_array_equal(y.ravel(), y2.ravel())
            assert_array_equal(dist.ravel(), dist2.ravel())


if __name__ == "__main__":
    unittest.main()
