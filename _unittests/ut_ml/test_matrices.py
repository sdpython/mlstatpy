# -*- coding: utf-8 -*-
"""
@brief      test log(time=6s)
"""
import unittest
import numpy
import numpy.random as rnd
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
        T, P = gram_schmidt(X.T, change=True)
        P = P.T
        T = T.T
        m = P.T @ X.T
        z = m @ m.T
        self.assertEqual(z, numpy.identity(2))
        m = X @ P
        self.assertEqual(m, T)
        z2 = m.T @ m
        self.assertEqual(z2, numpy.identity(2))

    def test_linear_regression(self):
        X = numpy.array([[1, 0.5, 0], [0, 0.4, 2]], dtype=float).T
        y = numpy.array([1, 1.3, 3.9])
        b1 = linear_regression(X, y)
        b2 = linear_regression(X, y, algo="gram")
        self.assertEqualArray(b1, b2)

    def test_linear_regression_qr(self):
        X = numpy.array([[1, 0.5, 0], [0, 0.4, 2]], dtype=float).T
        y = numpy.array([1, 1.3, 3.9])
        b1 = linear_regression(X, y)
        b3 = linear_regression(X, y, algo="gram")
        b2 = linear_regression(X, y, algo="qr")
        self.assertEqualArray(b1, b3)
        self.assertEqualArray(b1, b2)

    def test_linear_regression_qr3(self):
        X = numpy.array([[1, 0.5, 0], [0, 0.4, 2], [
                        0, 0.4, 2.1]], dtype=float).T
        y = numpy.array([1, 1.3, 3.9])
        b1 = linear_regression(X, y)
        b3 = linear_regression(X, y, algo="gram")
        b2 = linear_regression(X, y, algo="qr")
        self.assertEqualArray(b1, b3)
        self.assertEqualArray(b1, b2)

    def test_dim_lin_reg(self):
        X = rnd.randn(100, 7)
        eps = rnd.randn(100, 1) / 3
        y = X.sum(axis=1).reshape((X.shape[0], 1)) + eps
        b1 = linear_regression(X, y)
        b3 = linear_regression(X, y, algo="gram")
        b2 = linear_regression(X, y, algo="qr")
        self.assertEqualArray(b1.ravel(), b3.ravel())
        self.assertEqualArray(b1.ravel(), b2.ravel())

    def test_inner_code(self):

        X = numpy.array([[1., 2., 3., 4.],
                         [5., 6., 6., 6.],
                         [5., 6., 7., 8.]]).T
        y = numpy.array([0.1, 0.2, 0.19, 0.29])
        Xt = X.T
        Tt = numpy.empty(Xt.shape)
        Pt = numpy.identity(X.shape[1])
        for i in range(0, Xt.shape[0]):
            Tt[i, :] = Xt[i, :]
            for j in range(0, i):
                d = numpy.dot(Tt[j, :], Xt[i, :])
                Tt[i, :] -= Tt[j, :] * d
                Pt[i, :] -= Pt[j, :] * d
            d = numpy.dot(Tt[i, :], Tt[i, :])
            if d > 0:
                d **= 0.5
                Tt[i, :] /= d
                Pt[i, :] /= d

        self.assertEqual(Tt.shape, Xt.shape)
        self.assertEqual(Pt.shape, (X.shape[1], X.shape[1]))
        _Tt = Pt @ Xt
        self.assertEqualArray(_Tt, Tt)
        self.assertEqualArray(Tt @ Tt.T, numpy.identity(Tt.shape[0]))

        beta1 = numpy.linalg.inv(Xt @ X) @ Xt @ y
        beta2 = Tt @ y @ Pt
        self.assertEqualArray(beta1, beta2)


if __name__ == "__main__":
    unittest.main()
