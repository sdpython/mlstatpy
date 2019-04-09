# -*- coding: utf-8 -*-
"""
@brief      test log(time=6s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlstatpy.ml.matrices import gram_schmidt, linear_regression


class TestMatrices(ExtTestCase):

    def test_gram_schmidt(self):
        mat1 = numpy.array([[1, 0], [0, 1]], dtype=float)
        res = gram_schmidt(mat1)
        self.assertEqual(res, mat1)

        mat = numpy.array([[1, 0.5], [0.5, 1]], dtype=float)
        res = gram_schmidt(mat)
        self.assertEqualArray(mat1, res @ res.T)
        res2 = gram_schmidt(mat.T).T

        res2, change2 = gram_schmidt(mat, change=True)
        self.assertEqual(res, res2)
        res3 = change2 @ mat
        self.assertEqual(res3, res2)

        mat1 = numpy.array([[1, 0, 0], [0, 0, 1]], dtype=float)
        res = gram_schmidt(mat1)
        self.assertEqual(res, mat1)

        mat1 = numpy.array([[1, 0.5, 0], [0, 0.5, 1]], dtype=float)
        res = gram_schmidt(mat1)
        self.assertEqual(res[0, 2], 0)

        mat1 = numpy.array(
            [[1, 0.5, 0], [0, 0.5, 1], [1, 0.5, 1]], dtype=float)
        res = gram_schmidt(mat1)
        self.assertEqualArray(numpy.identity(3), res @ res.T)

    def test_gram_schmidt_xx(self):
        X = numpy.array([[1, 0.5, 0], [0, 0.4, 2]], dtype=float).T
        U, P = gram_schmidt(X.T, change=True)
        P = P.T
        U = U.T
        m = P.T @ X.T
        z = m @ m.T
        self.assertEqual(z, numpy.identity(2))
        m = X @ P
        self.assertEqual(m, U)
        z2 = m.T @ m
        self.assertEqual(z2, numpy.identity(2))

    def test_lineear_regression(self):
        X = numpy.array([[1, 0.5, 0], [0, 0.4, 2]], dtype=float).T
        y = numpy.array([1, 1.3, 3.9])
        b1 = linear_regression(X, y)
        b2 = linear_regression(X, y, algo="gram")
        self.assertEqualArray(b1, b2)


if __name__ == "__main__":
    unittest.main()
