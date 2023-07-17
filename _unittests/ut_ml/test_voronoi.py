# -*- coding: utf-8 -*-
"""
@brief      test log(time=6s)
"""
import math
import unittest
from io import StringIO
from contextlib import redirect_stdout
import numpy
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from pyquickhelper.pycode import ExtTestCase, add_missing_development_version


class TestVoronoi(ExtTestCase):
    def setUp(self):
        add_missing_development_version(["mlinsights"], __file__, hide=True)

    def test_iris(self):
        from mlstatpy.ml import voronoi_estimation_from_lr

        data = load_iris()
        X, y = data.data[:, :2], data.target
        clr = LogisticRegression(solver="liblinear")
        clr.fit(X, y)
        C = [1.0, 0.0]
        D = 3.0
        self.assertRaise(
            lambda: voronoi_estimation_from_lr(clr.coef_, clr.intercept_, C, None),
            ValueError,
        )
        self.assertRaise(
            lambda: voronoi_estimation_from_lr(clr.coef_, clr.intercept_, C, [D]),
            TypeError,
        )

        std = StringIO()
        with redirect_stdout(std):
            points = voronoi_estimation_from_lr(
                clr.coef_, clr.intercept_, C, D, qr=False, verbose=True
            )
            self.assertEqual(points.shape, (3, 2))
            expected_values = numpy.array(
                [[3.0, 4.137], [5.044, 0.281], [5.497, 0.184]]
            )
            self.assertEqualArray(expected_values, points, decimal=2)

            points = voronoi_estimation_from_lr(
                clr.coef_, clr.intercept_, C, D, qr=True, verbose=True
            )
            self.assertEqual(points.shape, (3, 2))
            expected_values = numpy.array(
                [[3.0, 4.137], [5.044, 0.281], [5.497, 0.184]]
            )
            self.assertEqualArray(expected_values, points, decimal=2)
        std = std.getvalue()
        self.assertIn("[voronoi_estimation_from_lr] iter=", std)

    def test_iris_dim4(self):
        from mlstatpy.ml.voronoi import voronoi_estimation_from_lr

        data = load_iris()
        X, y = data.data[:, :4], data.target
        clr = LogisticRegression(solver="liblinear")
        clr.fit(X, y)
        C = [1.0, 0.0]
        D = 3.0
        self.assertRaise(
            lambda: voronoi_estimation_from_lr(clr.coef_, clr.intercept_, C, None),
            ValueError,
        )
        self.assertRaise(
            lambda: voronoi_estimation_from_lr(clr.coef_, clr.intercept_, C, [D]),
            ValueError,
        )

        C = [1.0, 0.0, 0.0, 0.0]
        points = voronoi_estimation_from_lr(clr.coef_, clr.intercept_, C, D, qr=False)
        self.assertEqual(points.shape, (3, 4))
        points2 = voronoi_estimation_from_lr(clr.coef_, clr.intercept_, C, D, qr=True)
        self.assertEqual(points2.shape, (3, 4))
        self.assertEqualArray(points2, points2, decimal=5)

    def test_square(self):
        from mlstatpy.ml.voronoi import voronoi_estimation_from_lr

        Xs = []
        Ys = []
        n = 20
        for i in range(0, 4):
            for j in range(0, 3):
                x1 = numpy.random.rand(n) + i * 1.1
                x2 = numpy.random.rand(n) + j * 1.1
                Xs.append(numpy.vstack([x1, x2]).T)
                Ys.extend([i * 3 + j] * n)
        X = numpy.vstack(Xs)
        Y = numpy.array(Ys)

        clr = LogisticRegression(solver="liblinear")
        clr.fit(X, Y)

        points = voronoi_estimation_from_lr(
            clr.coef_, clr.intercept_, qr=True, verbose=False
        )
        self.assertEqual(points.shape, (12, 2))
        self.assertGreater(points.ravel().min(), -35)
        self.assertLesser(points.ravel().max(), 10.5)

    def test_hexa_scale(self):
        from mlstatpy.ml.voronoi import voronoi_estimation_from_lr

        n = 4
        a = math.pi * 2 / 3
        points = []
        Ys = []
        for i in range(n):
            for j in range(n):
                dil = ((i + 1) ** 2 + (j + 1) ** 2) ** 0.6
                for _ in range(0, 20):
                    x = i + j * math.cos(a)
                    y = j * math.sin(a)
                    points.append([x * dil, y * dil])
                    Ys.append(i * n + j)
                    mi = 0.5
                    for r in [0.1, 0.3, mi]:
                        nb = 6 if r == mi else 12
                        for k2 in range(0, nb):
                            ang = math.pi * 2 / nb * k2 + math.pi / 6
                            x = i + j * math.cos(a) + r * math.cos(ang)
                            y = j * math.sin(a) + r * math.sin(ang)
                            points.append([x * dil, y * dil])
                            Ys.append(i * n + j)
        X = numpy.array(points)
        Y = numpy.array(Ys)

        clr = LogisticRegression(solver="liblinear")
        clr.fit(X, Y)

        std = StringIO()
        with redirect_stdout(std):
            points = voronoi_estimation_from_lr(
                clr.coef_, clr.intercept_, qr=True, verbose=True, max_iter=20
            )
        self.assertEqual(points.shape, (16, 2))
        self.assertGreater(points.ravel().min(), -15)
        self.assertLesser(points.ravel().max(), 16)
        std = std.getvalue()
        self.assertIn("del P", std)
        self.assertIn("[voronoi_estimation_from_lr] iter", std)


if __name__ == "__main__":
    unittest.main()
