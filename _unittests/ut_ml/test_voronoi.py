# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""

import sys
import os
import unittest
import random
import numpy
from sklearn.datasets import load_iris


try:
    import pyquickhelper as skip_
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..",
                "..",
                "pyquickhelper",
                "src")))
    if path not in sys.path:
        sys.path.append(path)
    import pyquickhelper as skip_


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

from pyquickhelper.pycode import ExtTestCase, add_missing_development_version
from sklearn.linear_model import LogisticRegression


class TestVoronoi(ExtTestCase):

    def setUp(self):
        add_missing_development_version(["mlinsights"], __file__, hide=True)

    def test_iris(self):
        from src.mlstatpy.ml.voronoi import voronoi_estimation
        data = load_iris()
        X, y = data.data[:, :2], data.target
        clr = LogisticRegression()
        clr.fit(X, y)
        C = [1., 0.]
        D = 3.
        self.assertRaise(lambda: voronoi_estimation(
            clr.coef_, clr.intercept_, C, None), ValueError)
        self.assertRaise(lambda: voronoi_estimation(
            clr.coef_, clr.intercept_, C, [D]), TypeError)

        points = voronoi_estimation(clr.coef_, clr.intercept_, C, D, qr=False)
        self.assertEqual(points.shape, (3, 2))
        expected_values = numpy.array(
            [[3., 4.12377262], [5.03684606, 0.2827372], [5.48745959, 0.18503334]])
        self.assertEqualArray(expected_values, points, decimal=5)

        # not yet available
        points = voronoi_estimation(clr.coef_, clr.intercept_, C, D, qr=True)
        self.assertEqual(points.shape, (3, 2))
        expected_values = numpy.array(
            [[3., 4.12377262], [5.03684606, 0.2827372], [5.48745959, 0.18503334]])
        self.assertEqualArray(expected_values, points, decimal=5)


if __name__ == "__main__":
    unittest.main()
