import unittest
import numpy
from numpy.testing import assert_array_equal
from mlstatpy.ext_test_case import ExtTestCase, ignore_warnings
from sklearn.neighbors import NearestNeighbors
from mlstatpy.ml.kppv import NuagePoints
from mlstatpy.ml.kppv_laesa import NuagePointsLaesa


class TestNuagePoints(ExtTestCase):
    @ignore_warnings(DeprecationWarning)
    def test_nuage_points_1d(self):
        X = numpy.array([[0], [3], [1]])
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(X)
        dist, y = neigh.kneighbors(X)

        nuage = NuagePoints()
        nuage.fit(X)
        dist2, y2 = nuage.kneighbors(X)

        assert_array_equal(y.ravel(), y2.ravel())
        assert_array_equal(dist.ravel(), dist2.ravel())

    @ignore_warnings(DeprecationWarning)
    def test_nuage_points_1d_leasa(self):
        X = numpy.array([[0], [3], [1]])
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(X)
        dist, y = neigh.kneighbors(X)

        nuage = NuagePointsLaesa(2)
        nuage.fit(X)
        dist2, y2 = nuage.kneighbors(X)

        assert_array_equal(y.ravel(), y2.ravel())
        assert_array_equal(dist.ravel(), dist2.ravel())

    @ignore_warnings(DeprecationWarning)
    def test_nuage_points_2d(self):
        X = numpy.array([[0, 0], [3, 3], [1, 1]])
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(X)
        dist, y = neigh.kneighbors(X)

        nuage = NuagePoints()
        nuage.fit(X)
        dist2, y2 = nuage.kneighbors(X)

        assert_array_equal(y.ravel(), y2.ravel())
        assert_array_equal(dist.ravel(), dist2.ravel())

    @ignore_warnings(DeprecationWarning)
    def test_nuage_points_2d_leasa(self):
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
