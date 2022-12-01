# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import numpy
import numpy.random as rnd
from pyquickhelper.pycode import ExtTestCase
from mlstatpy.ml.matrices import (
    gram_schmidt, linear_regression, streaming_gram_schmidt,
    streaming_linear_regression, streaming_linear_regression_gram_schmidt,
    norm2)


class TestMatrices(ExtTestCase):

    def test_gram_schmidt(self):
        mat1 = numpy.array([[1, 0], [0, 1]], dtype=float)
        res = gram_schmidt(mat1)
        self.assertEqual(res, mat1)

        mat = numpy.array([[1, 0.5], [0.5, 1]], dtype=float)
        res = gram_schmidt(mat)
        self.assertEqualArray(mat1, res @ res.T, atol=1e-10)
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
        self.assertEqualArray(numpy.identity(3), res @ res.T, atol=1e-10)

    def test_gram_schmidt_xx(self):
        X = numpy.array([[1, 0.5, 0], [0, 0.4, 2]], dtype=float).T
        T, P = gram_schmidt(X.T, change=True)
        P = P.T
        T = T.T
        m = P.T @ X.T
        z = m @ m.T
        self.assertEqualArray(z, numpy.identity(2), atol=1e-10)
        m = X @ P
        self.assertEqualArray(m, T, atol=1e-10)
        z2 = m.T @ m
        self.assertEqualArray(z2, numpy.identity(2), atol=1e-10)

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
        y = X.sum(axis=1).reshape(  # pylint: disable=E1101
            (X.shape[0], 1)) + eps  # pylint: disable=E1101
        y = y.ravel()
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
        self.assertEqualArray(Tt @ Tt.T, numpy.identity(Tt.shape[0]), atol=1e-10)

        beta1 = numpy.linalg.inv(Xt @ X) @ Xt @ y
        beta2 = Tt @ y @ Pt
        self.assertEqualArray(beta1, beta2)

    def test_streaming_gram_schmidt(self):
        X0 = numpy.array([[1, 0.5, 10., 5., -2.],
                          [0, 0.4, 20, 4., 2.],
                          [0, 0.4, 19, 4.2, 2.],
                          [0, 0.7, 18, 4.1, 1.4]], dtype=float).T
        for dim in (3, 2, 4):
            X = X0[:, :dim]
            Xt = X.T
            algo1 = []
            algo1_t = []
            idd = numpy.identity(dim)
            for i in range(dim, Xt.shape[1] + 1):
                t, p = gram_schmidt(Xt[:, :i], change=True)
                algo1_t.append(t)
                algo1.append(p)
                t_ = X[:i] @ p.T
                self.assertEqualArray(t @ t.T, idd, atol=1e-10)
                self.assertEqualArray(t_.T @ t_, idd, atol=1e-10)
            algo2 = []
            self.assertRaise(lambda: list(streaming_gram_schmidt(X)),  # pylint: disable=W0640
                             RuntimeError)
            for i, p in enumerate(streaming_gram_schmidt(Xt)):
                p2 = p.copy()
                t2 = X[:i + dim] @ p2.T
                algo2.append(p2)
                self.assertNotEmpty(t2)
                self.assertNotEmpty(p2)
                self.assertEqualArray(t2.T @ t2, idd, atol=1e-10)
            self.assertEqual(len(algo1), len(algo2))

    def test_streaming_linear_regression(self):
        X0 = numpy.array([[1, 0.5, 10., 5., -2.],
                          [0, 0.4, 20, 4., 2.],
                          [0, 0.4, 19, 4.2, 2.],
                          [0, 0.7, 18, 4.1, 1.4]], dtype=float).T
        y0 = numpy.array([1., 2.1, 3.1, 3.9, 5.])

        for dim in (2, 3, 4):
            X = X0[:, :dim]
            y = y0
            algo1 = []
            for i in range(dim, X.shape[0] + 1):
                bk = linear_regression(X[:i], y[:i])
                algo1.append(bk)
            algo2 = []
            self.assertRaise(lambda: list(streaming_linear_regression(X.T, y)),  # pylint: disable=W0640
                             RuntimeError)
            for i, bk in enumerate(streaming_linear_regression(X, y)):
                algo2.append(bk.copy())
                self.assertNotEmpty(bk)
                self.assertEqualArray(algo1[i], algo2[i])
            self.assertEqual(len(algo1), len(algo2))

    def test_streaming_linear_regression_graph_schmidt(self):
        X0 = numpy.array([[1, 0.5, 10., 5., -2.],
                          [0, 0.4, 20, 4., 2.],
                          [0, 0.4, 19, 4.2, 2.],
                          [0, 0.7, 18, 4.1, 1.4]], dtype=float).T
        y0 = numpy.array([1., 2.1, 3.1, 3.9, 5.])

        for dim in (3, 2, 4):
            X = X0[:, :dim]
            y = y0
            algo1 = []
            for i in range(dim, X.shape[0] + 1):
                bk = linear_regression(X[:i], y[:i])
                algo1.append(bk)
            algo2 = []
            self.assertRaise(lambda: list(streaming_linear_regression_gram_schmidt(X.T, y)),  # pylint: disable=W0640
                             RuntimeError)
            for i, bk in enumerate(streaming_linear_regression_gram_schmidt(X, y)):
                algo2.append(bk.copy())
                self.assertNotEmpty(bk)
                self.assertEqualArray(algo1[i], algo2[i])
            self.assertEqual(len(algo1), len(algo2))

    def test_profile(self):
        N = 1000
        X = rnd.randn(N, 10)
        eps = rnd.randn(N, 1) / 3
        y = X.sum(axis=1).reshape(  # pylint: disable=E1101
            (X.shape[0], 1)) + eps  # pylint: disable=E1101
        y = y.ravel()
        res = self.profile(lambda: list(
            streaming_linear_regression_gram_schmidt(X, y)))
        if __name__ == "__main__":
            print("***", res[1])
        self.assertIn("streaming", res[1])
        res = self.profile(lambda: list(streaming_linear_regression(X, y)))
        if __name__ == "__main__":
            print("***", res[1])
        self.assertIn("streaming", res[1])

    def test_norm2(self):
        X = numpy.array([[1, 0.5, 0], [0, 0.4, 2]], dtype=float).T
        n2 = norm2(X)
        self.assertNotEmpty(n2)


if __name__ == "__main__":
    unittest.main()
